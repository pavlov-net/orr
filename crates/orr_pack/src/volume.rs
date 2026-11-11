//! Volume class: scalar grids on one voxel lattice, stored as sparse 8^3-voxel bricks.
//!
//! - Voxel `i` (integer index coordinates) has its center at `origin + i * voxel_size`;
//!   values between centers are trilinear in the eight voxels around them.
//! - Brick `b` covers voxels `brick_min + 8 b` to `brick_min + 8 b + 7` and holds 10^3
//!   texels, x fastest: texel `t` per axis is voxel `brick_min + 8 b + t - 1`. The one-voxel
//!   apron on each side keeps the taps of any trilinear sample in the brick inside it.
//! - A grid stores only bricks with a texel other than its background. Absent bricks and
//!   voxels outside the brick grid read the background.
//! - A value is the texel as a shader reads it (unorm 0-1, or the half) times `scale` plus
//!   `offset`.
//!
//! Blob 0 holds a [`VolumeHeader`] and its tables ([`VolumeInfo`]). Blob 1 holds the
//! majorants: per grid, one f32 per cell of 4^3 bricks, x fastest, at least every value a
//! sample in the cell can return. The blobs after them are pages of at most 64 KiB, grid by
//! grid: page `p` of a grid holds its bricks `p n` to `p n + n - 1` in table order, brick
//! `p n + s` at byte `s * brick_bytes`, with `n = 65536 / brick_bytes`.
//!
//! Resident on the GPU ([`ResidentLayout`]), a volume is one [`VolumeGpu`] record, its
//! [`VolumeGridGpu`] records, per grid a slot table (the position of every lattice brick
//! among the grid's bricks, x fastest, [`NO_BRICK`] where absent), the majorants, and per
//! grid its decoded pages back to back, so brick `s` sits at byte `s * brick_bytes`.
//! `orr_pack.volume` samples it as [`GridSampler`] does.

use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::blob::{put_table, read, table};
use crate::instance::morton;
use crate::texture::vk;
use crate::{Aabb, Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Voxels per brick and axis, apron excluded.
pub const BRICK: u32 = 8;

/// Texels per brick and axis: the brick and a one-voxel apron on each side.
pub const BRICK_TEXELS: u32 = BRICK + 2;

/// Texels per brick.
pub const BRICK_TEXEL_COUNT: usize = (BRICK_TEXELS * BRICK_TEXELS * BRICK_TEXELS) as usize;

/// Bricks per majorant cell and axis.
pub const MAJORANT_CELL: u32 = 4;

/// Largest page.
const PAGE_BYTES: usize = 64 << 10;

/// A slot table's entry for an absent brick.
pub const NO_BRICK: u32 = u32::MAX;

/// Alignment of each part of a resident volume.
const RESIDENT_ALIGN: u64 = 256;

/// Blob 0 header. Spans name tables in blob 0: `first` is a byte offset, `count` a record
/// count.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct VolumeHeader {
    /// World position of voxel `(0, 0, 0)`'s center, in m.
    pub origin: [f32; 3],
    /// Voxel spacing per axis, in m.
    pub voxel_size: [f32; 3],
    /// Index of brick `(0, 0, 0)`'s first voxel; multiples of 8.
    pub brick_min: [i32; 3],
    /// Bricks per axis.
    pub bricks: [u32; 3],
    /// Majorant cells per axis: `bricks` / 4 rounded up.
    pub majorant_cells: [u32; 3],
    /// [`VolumeGrid`] table.
    pub grids: Span,
    /// Grid names, UTF-8, each grid's `name` a range of these bytes.
    pub names: Span,
    /// [`VolumeBrick`] table of every grid, each grid's `bricks` a range of it.
    pub brick_table: Span,
}

impl VolumeHeader {
    /// Every sample outside reads the background of each grid, in m: from one voxel before the
    /// brick grid's first voxel to one past its last, the reach of their trilinear taps.
    pub fn bounds(&self) -> Aabb {
        let corner =
            |idx: [i64; 3]| [0, 1, 2].map(|a| self.origin[a] + idx[a] as f32 * self.voxel_size[a]);
        let min = self.brick_min.map(|m| i64::from(m) - 1);
        let max =
            [0, 1, 2].map(|a| i64::from(self.brick_min[a]) + i64::from(self.bricks[a] * BRICK));
        Aabb {
            min: corner(min),
            max: corner(max),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct VolumeGrid {
    /// Byte range within the header's `names`.
    pub name: Span,
    /// [`vk::R8_UNORM`] or [`vk::R16_SFLOAT`].
    pub vk_format: u32,
    pub scale: f32,
    pub offset: f32,
    /// Value of absent bricks and of voxels outside the brick grid.
    pub background: f32,
    /// Smallest and largest input value over the stored texels and the background.
    pub value_range: [f32; 2],
    /// Range of the brick table, sorted by [`morton`] of the brick coordinates.
    pub bricks: Span,
    /// Entry blob of the first page, and the page count.
    pub pages: Span,
    /// Index of the grid's first f32 in blob 1.
    pub majorant: u32,
    pub _pad: u32,
}

/// A stored brick.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct VolumeBrick {
    pub coord: [u16; 3],
    pub _pad: u16,
}

/// A resident volume's record: its lattice and its grids.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct VolumeGpu {
    pub origin: [f32; 3],
    /// Records at `grids`. `orr_pack.volume`'s samplers read a grid at or past it as 0
    /// everywhere.
    pub grid_count: u32,
    pub inv_voxel_size: [f32; 3],
    pub _pad0: u32,
    pub brick_min: [i32; 3],
    pub _pad1: u32,
    pub bricks: [u32; 3],
    pub _pad2: u32,
    pub majorant_cells: [u32; 3],
    pub _pad3: u32,
    /// Address of `grid_count` [`VolumeGridGpu`] records.
    pub grids: u64,
}

/// One grid of a resident volume.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct VolumeGridGpu {
    /// Address of the grid's bricks in table order, back to back.
    pub bricks: u64,
    /// Address of the slot table: per lattice brick, x fastest, its position among the
    /// grid's bricks or [`NO_BRICK`].
    pub slots: u64,
    /// Address of the grid's majorants, one f32 per cell, x fastest.
    pub majorants: u64,
    pub vk_format: u32,
    pub scale: f32,
    pub offset: f32,
    pub background: f32,
}

/// Byte offsets of the parts of a resident volume from its start, each aligned to 256
/// bytes. The [`VolumeGpu`] record is at 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentLayout {
    pub size: u64,
    /// The [`VolumeGridGpu`] records.
    pub grids: u64,
    /// Per grid, its slot table.
    pub slots: Vec<u64>,
    /// Blob 1 as stored.
    pub majorants: u64,
    /// Per grid, its first brick.
    pub bricks: Vec<u64>,
}

impl ResidentLayout {
    /// Offset of decoded page `page` of grid `grid`.
    pub fn page(&self, info: &VolumeInfo, grid: usize, page: usize) -> u64 {
        let g = &info.grids[grid];
        self.bricks[grid] + (page * g.bricks_per_page() * g.brick_bytes()) as u64
    }
}

impl VolumeGrid {
    /// Quantization of a grid with values in `range` (background included). R8 maps `[min,
    /// max]` onto 0-1. R16F has offset 0 and a power-of-two scale that brings the largest
    /// magnitude into [2^14, 2^15), clear of the half's overflow.
    pub fn fit(vk_format: u32, range: [f32; 2], background: f32) -> Result<Self> {
        if !range[0].is_finite() || !range[1].is_finite() || range[0] > range[1] {
            return Err(Error::Encode(format!("volume value range {range:?}")));
        }
        let (scale, offset) = match vk_format {
            vk::R8_UNORM => (range[1] - range[0], range[0]),
            vk::R16_SFLOAT => {
                let m = range[0].abs().max(range[1].abs());
                if m == 0.0 {
                    (1.0, 0.0)
                } else {
                    (2f64.powi((binade(m) - 14).max(-149)) as f32, 0.0)
                }
            }
            _ => return Err(Error::Encode(format!("volume format {vk_format}"))),
        };
        Ok(Self {
            name: Span::default(),
            vk_format,
            scale,
            offset,
            background,
            value_range: range,
            bricks: Span::default(),
            pages: Span::default(),
            majorant: 0,
            _pad: 0,
        })
    }

    fn texel_bytes(&self) -> usize {
        if self.vk_format == vk::R8_UNORM { 1 } else { 2 }
    }

    pub fn brick_bytes(&self) -> usize {
        BRICK_TEXEL_COUNT * self.texel_bytes()
    }

    /// Bricks per page.
    pub fn bricks_per_page(&self) -> usize {
        PAGE_BYTES / self.brick_bytes()
    }

    /// The texel bits of `v`: the unorm byte, or the half's bits.
    fn code(&self, v: f32) -> u16 {
        if self.vk_format == vk::R8_UNORM {
            if self.scale == 0.0 {
                return 0;
            }
            ((v - self.offset) / self.scale * 255.0)
                .round()
                .clamp(0.0, 255.0) as u16
        } else {
            f16::from_f32(v / self.scale).to_bits()
        }
    }

    /// Appends the texels of `values`.
    pub fn encode(&self, values: &[f32], out: &mut Vec<u8>) {
        for &v in values {
            let c = self.code(v);
            if self.vk_format == vk::R8_UNORM {
                out.push(c as u8);
            } else {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
    }

    /// The value of texel bits `code`.
    pub fn decode(&self, code: u16) -> f32 {
        let texel = if self.vk_format == vk::R8_UNORM {
            f32::from(code) / 255.0
        } else {
            f16::from_bits(code).to_f32()
        };
        texel * self.scale + self.offset
    }

    /// Largest difference between a value in the range and its decoded texel, plus the
    /// float rounding of the decode: one quantization step.
    pub fn step(&self) -> f32 {
        if self.vk_format == vk::R8_UNORM {
            self.scale / 255.0
        } else {
            16.0 * self.scale
        }
    }

    /// Texel bits at `index` of a brick's bytes.
    pub fn texel(&self, brick: &[u8], index: usize) -> u16 {
        if self.vk_format == vk::R8_UNORM {
            u16::from(brick[index])
        } else {
            u16::from_le_bytes([brick[2 * index], brick[2 * index + 1]])
        }
    }
}

/// Exponent of the binade holding the positive finite `m`: `floor(log2 m)`.
fn binade(m: f32) -> i32 {
    let bits = m.to_bits();
    let e = ((bits >> 23) & 0xff) as i32;
    if e == 0 {
        -127 - ((bits & 0x7f_ffff).leading_zeros() as i32 - 9)
    } else {
        e - 127
    }
}

/// Index of the majorant cell holding `brick` among `cells` per axis, x fastest.
pub fn majorant_cell(cells: [u32; 3], brick: [u32; 3]) -> usize {
    let c = brick.map(|b| (b / MAJORANT_CELL) as usize);
    let n = cells.map(|n| n as usize);
    c[0] + n[0] * (c[1] + n[1] * c[2])
}

/// A volume blob 0.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeInfo {
    pub header: VolumeHeader,
    pub grids: Vec<VolumeGrid>,
    pub names: Vec<String>,
    pub bricks: Vec<VolumeBrick>,
}

impl VolumeInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let malformed = Error::Malformed;
        let h: VolumeHeader = read(blob, 0).ok_or_else(|| malformed("volume header"))?;
        let grids: Vec<VolumeGrid> = table(blob, h.grids)?;
        let name_bytes: Vec<u8> = table(blob, h.names)?;
        let bricks: Vec<VolumeBrick> = table(blob, h.brick_table)?;
        if h.brick_min.iter().any(|m| m % BRICK as i32 != 0)
            || h.bricks.iter().any(|&b| b == 0 || b > u32::from(u16::MAX))
            || h.majorant_cells != h.bricks.map(|b| b.div_ceil(MAJORANT_CELL))
            || h.voxel_size.iter().any(|&s| s.is_nan() || s <= 0.0)
        {
            return Err(malformed("volume lattice"));
        }
        let cells = h.majorant_cells.iter().product::<u32>();
        let mut names = Vec::with_capacity(grids.len());
        let mut next_blob = 2u32;
        for (i, g) in grids.iter().enumerate() {
            if !matches!(g.vk_format, vk::R8_UNORM | vk::R16_SFLOAT) {
                return Err(malformed("volume grid format"));
            }
            let at = g.name.first as usize;
            let name = at
                .checked_add(g.name.count as usize)
                .and_then(|end| name_bytes.get(at..end))
                .and_then(|b| std::str::from_utf8(b).ok())
                .ok_or_else(|| malformed("volume grid name"))?;
            names.push(name.to_owned());
            let first = g.bricks.first as usize;
            let own = first
                .checked_add(g.bricks.count as usize)
                .and_then(|end| bricks.get(first..end))
                .ok_or_else(|| malformed("volume grid bricks"))?;
            let mut last = None;
            for b in own {
                if (0..3).any(|a| u32::from(b.coord[a]) >= h.bricks[a]) {
                    return Err(malformed("volume brick outside the lattice"));
                }
                let m = morton(b.coord);
                if last.is_some_and(|l| l >= m) {
                    return Err(malformed("volume bricks out of order"));
                }
                last = Some(m);
            }
            let pages = (g.bricks.count as usize).div_ceil(g.bricks_per_page()) as u32;
            if g.pages.first != next_blob
                || g.pages.count != pages
                || g.majorant != i as u32 * cells
            {
                return Err(malformed("volume grid pages"));
            }
            next_blob += pages;
        }
        Ok(Self {
            header: h,
            grids,
            names,
            bricks,
        })
    }

    /// Blob bytes; the header's spans and each grid's name span are set from the tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut grids = self.grids.clone();
        let mut names = Vec::new();
        for (g, n) in grids.iter_mut().zip(&self.names) {
            g.name = Span {
                first: names.len() as u32,
                count: n.len() as u32,
            };
            names.extend_from_slice(n.as_bytes());
        }
        let mut out = vec![0; size_of::<VolumeHeader>()];
        h.grids = put_table(&mut out, &grids);
        h.names = put_table(&mut out, &names);
        h.brick_table = put_table(&mut out, &self.bricks);
        out[..size_of::<VolumeHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    /// Index of the grid named `name`.
    pub fn grid(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// The bricks of `grid`, in table order.
    fn grid_bricks(&self, grid: usize) -> &[VolumeBrick] {
        let s = self.grids[grid].bricks;
        &self.bricks[s.first as usize..(s.first + s.count) as usize]
    }

    /// Position of the brick at `coord` among the bricks of `grid`, if stored.
    pub fn brick(&self, grid: usize, coord: [u16; 3]) -> Option<usize> {
        let key = morton(coord);
        self.grid_bricks(grid)
            .binary_search_by(|b| morton(b.coord).cmp(&key))
            .ok()
    }

    /// The majorants of `grid` from blob 1.
    pub fn majorants(&self, blob: &[u8], grid: usize) -> Result<Vec<f32>> {
        let cells = self.header.majorant_cells.iter().product::<u32>();
        table(
            blob,
            Span {
                first: self.grids[grid].majorant * 4,
                count: cells,
            },
        )
    }

    /// Index of the majorant cell holding `brick` within a grid's majorants.
    pub fn majorant_index(&self, brick: [u32; 3]) -> usize {
        majorant_cell(self.header.majorant_cells, brick)
    }

    /// Index coordinates of world point `p`.
    pub fn to_index(&self, p: [f32; 3]) -> [f32; 3] {
        let h = &self.header;
        [0, 1, 2].map(|a| (p[a] - h.origin[a]) / h.voxel_size[a])
    }

    /// The slot table of `grid`: per lattice brick, x fastest, its position among the
    /// grid's bricks or [`NO_BRICK`].
    fn brick_slots(&self, grid: usize) -> Vec<u32> {
        let n = self.header.bricks.map(|b| b as usize);
        let mut slots = vec![NO_BRICK; n[0] * n[1] * n[2]];
        for (s, b) in self.grid_bricks(grid).iter().enumerate() {
            let c = b.coord.map(usize::from);
            slots[c[0] + n[0] * (c[1] + n[1] * c[2])] = s as u32;
        }
        slots
    }

    /// Where a resident copy keeps each part.
    pub fn resident_layout(&self) -> ResidentLayout {
        let mut size = 0u64;
        let mut place = |bytes: u64| {
            let at = size;
            size += bytes.max(1).next_multiple_of(RESIDENT_ALIGN);
            at
        };
        place(size_of::<VolumeGpu>() as u64);
        let grids = place((self.grids.len() * size_of::<VolumeGridGpu>()) as u64);
        let lattice = u64::from(self.header.bricks.iter().product::<u32>());
        let slots = self.grids.iter().map(|_| place(4 * lattice)).collect();
        let cells = u64::from(self.header.majorant_cells.iter().product::<u32>());
        let majorants = place(4 * cells * self.grids.len() as u64);
        let bricks = self
            .grids
            .iter()
            .map(|g| place(u64::from(g.bricks.count) * g.brick_bytes() as u64))
            .collect();
        ResidentLayout {
            size,
            grids,
            slots,
            majorants,
            bricks,
        }
    }

    /// The first `layout.majorants` bytes of a resident copy at address `base`: its records
    /// and every slot table.
    pub fn resident_head(&self, layout: &ResidentLayout, base: u64) -> Vec<u8> {
        let h = &self.header;
        let mut out = vec![0; layout.majorants as usize];
        let record = VolumeGpu {
            origin: h.origin,
            grid_count: self.grids.len() as u32,
            inv_voxel_size: h.voxel_size.map(|s| 1.0 / s),
            brick_min: h.brick_min,
            bricks: h.bricks,
            majorant_cells: h.majorant_cells,
            grids: base + layout.grids,
            ..Zeroable::zeroed()
        };
        out[..size_of::<VolumeGpu>()].copy_from_slice(bytemuck::bytes_of(&record));
        for (i, g) in self.grids.iter().enumerate() {
            let r = VolumeGridGpu {
                bricks: base + layout.bricks[i],
                slots: base + layout.slots[i],
                majorants: base + layout.majorants + 4 * u64::from(g.majorant),
                vk_format: g.vk_format,
                scale: g.scale,
                offset: g.offset,
                background: g.background,
            };
            let at = layout.grids as usize + i * size_of::<VolumeGridGpu>();
            out[at..at + size_of::<VolumeGridGpu>()].copy_from_slice(bytemuck::bytes_of(&r));
            let slots = self.brick_slots(i);
            let at = layout.slots[i] as usize;
            out[at..at + 4 * slots.len()].copy_from_slice(bytemuck::cast_slice(&slots));
        }
        out
    }
}

/// Reads one grid's voxels and samples from its decoded pages, as a shader would: the
/// CPU reference of the layout.
pub struct GridSampler<'a> {
    info: &'a VolumeInfo,
    grid: &'a VolumeGrid,
    pages: &'a [Vec<u8>],
    /// [`VolumeInfo::brick_slots`].
    slots: Vec<u32>,
}

impl<'a> GridSampler<'a> {
    /// `pages` are the grid's decoded pages, in order.
    pub fn new(info: &'a VolumeInfo, grid: usize, pages: &'a [Vec<u8>]) -> Result<Self> {
        let g = &info.grids[grid];
        let per_page = g.bricks_per_page();
        let count = g.bricks.count as usize;
        let short = pages.len() != g.pages.count as usize
            || pages
                .iter()
                .enumerate()
                .any(|(p, b)| b.len() != (count - p * per_page).min(per_page) * g.brick_bytes());
        if short {
            return Err(Error::Malformed("volume pages"));
        }
        Ok(Self {
            info,
            grid: g,
            pages,
            slots: info.brick_slots(grid),
        })
    }

    /// Bytes of the brick at `b`, if stored.
    fn brick(&self, b: [i64; 3]) -> Option<&'a [u8]> {
        let n = self.info.header.bricks.map(i64::from);
        if (0..3).any(|a| b[a] < 0 || b[a] >= n[a]) {
            return None;
        }
        let slot = self.slots[(b[0] + n[0] * (b[1] + n[1] * b[2])) as usize];
        if slot == NO_BRICK {
            return None;
        }
        let per_page = self.grid.bricks_per_page();
        let (page, at) = (slot as usize / per_page, slot as usize % per_page);
        let bytes = self.grid.brick_bytes();
        Some(&self.pages[page][at * bytes..(at + 1) * bytes])
    }

    fn texel(&self, brick: &[u8], t: [i64; 3]) -> f32 {
        let n = i64::from(BRICK_TEXELS);
        let index = (t[0] + n * (t[1] + n * t[2])) as usize;
        self.grid.decode(self.grid.texel(brick, index))
    }

    /// The value of voxel `i`, read from the brick that holds it.
    pub fn voxel(&self, i: [i32; 3]) -> f32 {
        let rel = [0, 1, 2].map(|a| i64::from(i[a]) - i64::from(self.info.header.brick_min[a]));
        let b = rel.map(|r| r.div_euclid(i64::from(BRICK)));
        match self.brick(b) {
            Some(brick) => self.texel(brick, rel.map(|r| r.rem_euclid(i64::from(BRICK)) + 1)),
            None => self.grid.background,
        }
    }

    /// The trilinear value at world point `p`, with all eight taps from one brick: the brick
    /// holding the lower corner voxel, or brick 0 of an axis whose lower corner is the voxel
    /// just below the brick grid.
    pub fn sample(&self, p: [f32; 3]) -> f32 {
        let u = self.info.to_index(p);
        let h = &self.info.header;
        let mut brick = [0i64; 3];
        let mut t0 = [0i64; 3];
        let mut w = [0f32; 3];
        for a in 0..3 {
            let f = u[a].floor();
            w[a] = u[a] - f;
            let rel = f as i64 - i64::from(h.brick_min[a]);
            if rel < -1 || rel >= i64::from(h.bricks[a] * BRICK) {
                return self.grid.background;
            }
            brick[a] = rel.max(0) / i64::from(BRICK);
            t0[a] = rel - brick[a] * i64::from(BRICK) + 1;
        }
        let Some(bytes) = self.brick(brick) else {
            return self.grid.background;
        };
        let mut v = 0.0;
        for corner in 0..8 {
            let d = [corner & 1, (corner >> 1) & 1, corner >> 2];
            let weight: f32 = (0..3)
                .map(|a| if d[a] == 1 { w[a] } else { 1.0 - w[a] })
                .product();
            v += weight * self.texel(bytes, [0, 1, 2].map(|a| t0[a] + i64::from(d[a])));
        }
        v
    }
}

const _: () = {
    assert!(size_of::<VolumeHeader>() == 84);
    assert!(size_of::<VolumeGrid>() == 56);
    assert!(size_of::<VolumeBrick>() == 8);
    assert!(size_of::<VolumeGpu>() == 88);
    assert!(size_of::<VolumeGridGpu>() == 40);
};
