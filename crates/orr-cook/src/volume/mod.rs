//! Volume cook: scalar grids from OpenVDB files or memory into one volume entry
//! (`orr_pack::volume`): sparse bricks with aprons, quantized per grid, in pages, with a
//! majorant per 4^3 bricks.

mod vdb;

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use orr_pack::instance::morton;
use orr_pack::texture::vk;
use orr_pack::volume::{
    self, BRICK, BRICK_TEXEL_COUNT, BRICK_TEXELS, MAJORANT_CELL, VolumeBrick, VolumeGrid,
    VolumeHeader, VolumeInfo,
};
use orr_pack::{Aabb, ArchiveWriter, BlobIn, Class, Hash32, Span};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::key::{Key, ZSTD_LEVEL};
use crate::state::{Memo, read_checked};
use crate::{Error, Result, mount};

/// Voxels of one 8^3 block, x fastest.
pub type Block = [f32; 512];

/// A volume source.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeInput {
    /// An OpenVDB file. Float grids (full or half precision) with a scale or
    /// scale-translate transform are read; all selected grids must share one transform.
    /// Values follow OpenVDB's `getValue` with a background of 0, so fog volumes are exact
    /// and level sets are not supported.
    Vdb { path: PathBuf },
    /// Grids built in memory; library callers only.
    #[serde(skip)]
    Grids(VolumeGrids),
}

/// Grids on one lattice: voxel `i` has its center at `origin + i * voxel_size`, in m.
#[derive(Clone, Debug)]
pub struct VolumeGrids {
    pub origin: [f32; 3],
    pub voxel_size: [f32; 3],
    pub grids: Vec<GridInput>,
}

#[derive(Clone, Debug)]
pub struct GridInput {
    pub name: String,
    /// Value of every voxel the input does not give.
    pub background: f32,
    pub voxels: GridVoxels,
}

#[derive(Clone, Debug)]
pub enum GridVoxels {
    /// `size` voxels per axis from voxel `min`, x fastest.
    Dense {
        min: [i32; 3],
        size: [u32; 3],
        values: Vec<f32>,
    },
    /// 8^3 blocks by block coordinate: block `b` holds voxels `8 b` to `8 b + 7`.
    Blocks(Vec<([i32; 3], Box<Block>)>),
}

/// Which grids a volume keeps, in order, and how each is stored.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VolumeSettings {
    /// Empty keeps every float grid of a VDB file sorted by name, or every input grid.
    pub grids: Vec<VolumeGridSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumeGridSpec {
    pub name: String,
    /// `None` stores a grid named `density` as R8 and any other as R16F.
    #[serde(default)]
    pub format: Option<VolumeFormat>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumeFormat {
    /// 8-bit unorm over the grid's value range.
    R8,
    /// Half float with a power-of-two scale.
    R16f,
}

impl VolumeFormat {
    fn vk_format(self) -> u32 {
        match self {
            Self::R8 => vk::R8_UNORM,
            Self::R16f => vk::R16_SFLOAT,
        }
    }

    fn default_for(name: &str) -> Self {
        if name == "density" {
            Self::R8
        } else {
            Self::R16f
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct VolumeReport {
    pub voxel_size: [f32; 3],
    /// Bricks per axis of the lattice.
    pub bricks: [u32; 3],
    pub majorant_bytes: u64,
    pub grids: Vec<VolumeGridReport>,
}

#[derive(Clone, Debug, Serialize)]
pub struct VolumeGridReport {
    pub name: String,
    pub vk_format: u32,
    /// Input blocks of 8^3 voxels with a value other than the background.
    pub blocks: usize,
    pub bricks: usize,
    pub pages: usize,
    /// Decoded and stored bytes of its pages.
    pub raw: u64,
    pub stored: u64,
    pub value_range: [f32; 2],
}

/// A planned volume: its input and settings, and the content hash of a VDB file.
pub(crate) struct Plan<'m> {
    input: &'m VolumeInput,
    settings: &'m VolumeSettings,
    hash: Option<Hash32>,
}

pub(crate) struct Planned<'m> {
    pub plan: Plan<'m>,
    pub key: Hash32,
    pub working_set: u64,
    pub file: Option<PathBuf>,
}

pub(crate) fn volume_error(asset: &str, message: impl Into<String>) -> Error {
    Error::Volume {
        asset: asset.into(),
        message: message.into(),
    }
}

/// Hashes the input of a volume and computes its key.
pub(crate) fn plan<'m>(
    asset: &str,
    input: &'m VolumeInput,
    settings: &'m VolumeSettings,
    memo: &Memo,
) -> Result<Planned<'m>> {
    let mut key = Key::new(Class::Volume, asset).debug(settings);
    let (hash, working_set, file) = match input {
        VolumeInput::Vdb { path } => {
            let hash = memo.hash(path)?;
            key = key.str("vdb").hash(hash);
            let len = std::fs::metadata(path).map_or(0, |m| m.len());
            (Some(hash), 24 * len + (64 << 20), Some(path.clone()))
        }
        VolumeInput::Grids(g) => {
            key = key.str("grids").debug(&(g.origin, g.voxel_size));
            let mut values = 0;
            for grid in &g.grids {
                key = key.str(&grid.name).bytes(&grid.background.to_le_bytes());
                match &grid.voxels {
                    GridVoxels::Dense {
                        min,
                        size,
                        values: v,
                    } => {
                        key = key
                            .str("dense")
                            .debug(&(min, size))
                            .bytes(bytemuck::cast_slice(v));
                        values += v.len();
                    }
                    GridVoxels::Blocks(blocks) => {
                        key = key.str("blocks");
                        for (c, b) in blocks {
                            key = key
                                .bytes(bytemuck::cast_slice(c))
                                .bytes(bytemuck::cast_slice(&b[..]));
                        }
                        values += 512 * blocks.len();
                    }
                }
            }
            (None, 16 * values as u64 + (64 << 20), None)
        }
    };
    Ok(Planned {
        plan: Plan {
            input,
            settings,
            hash,
        },
        key: key.finish(),
        working_set,
        file,
    })
}

/// One input grid as 8^3 blocks with a value other than its background.
pub(crate) struct SourceGrid<'a> {
    pub name: String,
    pub background: f32,
    pub blocks: HashMap<[i32; 3], Cow<'a, Block>>,
}

impl SourceGrid<'_> {
    fn voxel(&self, v: [i64; 3]) -> f32 {
        let b = v.map(|x| x.div_euclid(8) as i32);
        match self.blocks.get(&b) {
            Some(block) => {
                let l = v.map(|x| x.rem_euclid(8) as usize);
                block[l[0] + 8 * l[1] + 64 * l[2]]
            }
            None => self.background,
        }
    }
}

/// Grids sharing one lattice.
pub(crate) struct SourceVolume<'a> {
    pub origin: [f32; 3],
    pub voxel_size: [f32; 3],
    pub grids: Vec<SourceGrid<'a>>,
}

fn from_memory<'a>(asset: &str, g: &'a VolumeGrids) -> Result<SourceVolume<'a>> {
    let mut grids = Vec::with_capacity(g.grids.len());
    for (i, input) in g.grids.iter().enumerate() {
        let fail = |m: String| volume_error(asset, format!("grid {}: {m}", input.name));
        if g.grids[..i].iter().any(|o| o.name == input.name) {
            return Err(fail("two grids have this name".into()));
        }
        let bg = input.background;
        if !bg.is_finite() {
            return Err(fail("the background is not finite".into()));
        }
        let mut blocks = HashMap::new();
        match &input.voxels {
            GridVoxels::Dense { min, size, values } => {
                let n = size.map(|s| s as usize);
                if values.len() != n[0] * n[1] * n[2] {
                    return Err(fail(format!("{} values for {size:?} voxels", values.len())));
                }
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(fail("a value is not finite".into()));
                }
                let lo = [0, 1, 2].map(|a| i64::from(min[a]).div_euclid(8));
                let hi =
                    [0, 1, 2].map(|a| (i64::from(min[a]) + i64::from(size[a]) - 1).div_euclid(8));
                for bz in lo[2]..=hi[2] {
                    for by in lo[1]..=hi[1] {
                        for bx in lo[0]..=hi[0] {
                            let mut block = [bg; 512];
                            let mut any = false;
                            for (i, v) in block.iter_mut().enumerate() {
                                let p = [
                                    bx * 8 + (i % 8) as i64,
                                    by * 8 + (i / 8 % 8) as i64,
                                    bz * 8 + (i / 64) as i64,
                                ];
                                let d = [0, 1, 2].map(|a| p[a] - i64::from(min[a]));
                                if (0..3).all(|a| d[a] >= 0 && d[a] < i64::from(size[a])) {
                                    let d = d.map(|x| x as usize);
                                    *v = values[d[0] + n[0] * (d[1] + n[1] * d[2])];
                                    any |= *v != bg;
                                }
                            }
                            if any {
                                blocks.insert([bx as i32, by as i32, bz as i32], Cow::Owned(block));
                            }
                        }
                    }
                }
            }
            GridVoxels::Blocks(list) => {
                for (c, b) in list {
                    if b.iter().any(|v| !v.is_finite()) {
                        return Err(fail(format!(
                            "block {c:?} holds a value that is not finite"
                        )));
                    }
                    if b.iter().any(|&v| v != bg)
                        && blocks.insert(*c, Cow::Borrowed(&**b)).is_some()
                    {
                        return Err(fail(format!("block {c:?} is given twice")));
                    }
                }
            }
        }
        grids.push(SourceGrid {
            name: input.name.clone(),
            background: bg,
            blocks,
        });
    }
    Ok(SourceVolume {
        origin: g.origin,
        voxel_size: g.voxel_size,
        grids,
    })
}

/// A grid's stored bricks: lattice block coordinate and its 10^3 texel values.
struct Bricks {
    coords: Vec<[i32; 3]>,
    texels: Vec<Vec<f32>>,
}

/// The bricks of `g` with a texel other than its background: candidates are its blocks
/// and their neighbors, whose aprons may reach them.
fn gather(g: &SourceGrid) -> Bricks {
    let mut candidates = HashSet::new();
    for c in g.blocks.keys() {
        for d in 0..27 {
            let o = [d % 3 - 1, d / 3 % 3 - 1, d / 9 - 1];
            candidates.insert([c[0] + o[0], c[1] + o[1], c[2] + o[2]]);
        }
    }
    let mut candidates: Vec<[i32; 3]> = candidates.into_iter().collect();
    candidates.sort_unstable();
    let n = BRICK_TEXELS as usize;
    let found: Vec<([i32; 3], Vec<f32>)> = candidates
        .par_iter()
        .filter_map(|&c| {
            let first = c.map(|x| i64::from(x) * i64::from(BRICK) - 1);
            let texels: Vec<f32> = (0..BRICK_TEXEL_COUNT)
                .map(|i| {
                    let t = [i % n, i / n % n, i / (n * n)];
                    g.voxel([0, 1, 2].map(|a| first[a] + t[a] as i64))
                })
                .collect();
            texels
                .iter()
                .any(|&v| v != g.background)
                .then_some((c, texels))
        })
        .collect();
    let (coords, texels) = found.into_iter().unzip();
    Bricks { coords, texels }
}

/// Cooks a planned volume into `w` as one entry.
pub(crate) fn cook(
    w: &mut ArchiveWriter,
    asset: &str,
    key: Hash32,
    plan: &Plan,
) -> Result<VolumeReport> {
    let specs = &plan.settings.grids;
    for (i, s) in specs.iter().enumerate() {
        if specs[..i].iter().any(|o| o.name == s.name) {
            return Err(volume_error(
                asset,
                format!("grid {} is selected twice", s.name),
            ));
        }
    }
    let src = match plan.input {
        VolumeInput::Vdb { path } => {
            let bytes = read_checked(path, plan.hash.expect("planned hash"))?;
            let names: Vec<&str> = specs.iter().map(|g| g.name.as_str()).collect();
            vdb::read(asset, path, &bytes, &names)?
        }
        VolumeInput::Grids(g) => select(asset, from_memory(asset, g)?, specs)?,
    };
    if src.voxel_size.iter().any(|&s| !(s.is_finite() && s > 0.0))
        || src.origin.iter().any(|o| !o.is_finite())
    {
        return Err(volume_error(
            asset,
            "the voxel size must be positive and the origin finite",
        ));
    }
    if src.grids.is_empty() {
        return Err(volume_error(asset, "a volume needs a grid"));
    }
    let formats: Vec<u32> = src
        .grids
        .iter()
        .map(|g| {
            specs
                .iter()
                .find(|s| s.name == g.name)
                .and_then(|s| s.format)
                .unwrap_or_else(|| VolumeFormat::default_for(&g.name))
                .vk_format()
        })
        .collect();
    let gathered: Vec<Bricks> = src.grids.iter().map(gather).collect();

    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    for c in gathered.iter().flat_map(|b| &b.coords) {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    if lo[0] > hi[0] {
        (lo, hi) = ([0; 3], [0; 3]);
    }
    let bricks = [0, 1, 2].map(|a| (i64::from(hi[a]) - i64::from(lo[a]) + 1) as u64);
    if bricks.iter().any(|&b| b > u64::from(u16::MAX)) {
        return Err(volume_error(
            asset,
            format!("{bricks:?} bricks per axis; at most 65535 fit"),
        ));
    }
    let bricks = bricks.map(|b| b as u32);
    let cells = bricks.map(|b| b.div_ceil(MAJORANT_CELL));
    let cell_count = cells.iter().product::<u32>() as usize;
    let brick_min = lo.map(|l| l * BRICK as i32);
    let corner =
        |idx: [i64; 3]| [0, 1, 2].map(|a| src.origin[a] + idx[a] as f32 * src.voxel_size[a]);
    let bounds = Aabb {
        min: corner(brick_min.map(|m| i64::from(m) - 1)),
        max: corner([0, 1, 2].map(|a| i64::from(brick_min[a]) + i64::from(bricks[a] * BRICK))),
    };

    let mut grids = Vec::new();
    let mut table = Vec::new();
    let mut majorants = Vec::with_capacity(cell_count * src.grids.len());
    let mut grid_bricks: Vec<Vec<u8>> = Vec::new();
    let mut next_blob = 2;
    for (gi, (g, b)) in src.grids.iter().zip(&gathered).enumerate() {
        let mut range = [g.background; 2];
        for v in b.texels.iter().flatten() {
            range = [range[0].min(*v), range[1].max(*v)];
        }
        let mut record = VolumeGrid::fit(formats[gi], range, g.background)
            .map_err(|e| volume_error(asset, format!("grid {}: {e}", g.name)))?;
        let rel: Vec<[u16; 3]> = b
            .coords
            .iter()
            .map(|c| [0, 1, 2].map(|a| (c[a] - lo[a]) as u16))
            .collect();
        let mut order: Vec<usize> = (0..rel.len()).collect();
        order.sort_unstable_by_key(|&i| morton(rel[i]));
        let encoded: Vec<(Vec<u8>, f32)> = order
            .par_iter()
            .map(|&i| {
                let mut out = Vec::with_capacity(record.brick_bytes());
                record.encode(&b.texels[i], &mut out);
                let max = (0..BRICK_TEXEL_COUNT)
                    .map(|t| record.decode(record.texel(&out, t)))
                    .fold(f32::NEG_INFINITY, f32::max);
                (out, max)
            })
            .collect();
        let mut cell_max = vec![f32::NEG_INFINITY; cell_count];
        let mut cell_stored = vec![0u32; cell_count];
        for (&i, (_, m)) in order.iter().zip(&encoded) {
            let cell = volume::majorant_cell(cells, rel[i].map(u32::from));
            cell_stored[cell] += 1;
            cell_max[cell] = cell_max[cell].max(*m);
        }
        let full = MAJORANT_CELL.pow(3);
        majorants.extend(cell_max.iter().zip(&cell_stored).map(|(&m, &n)| {
            let m = if n < full { m.max(g.background) } else { m };
            m.next_up()
        }));
        let per_page = record.bricks_per_page();
        let pages = encoded.len().div_ceil(per_page);
        record.bricks = Span {
            first: table.len() as u32,
            count: encoded.len() as u32,
        };
        record.pages = Span {
            first: next_blob,
            count: pages as u32,
        };
        record.majorant = (gi * cell_count) as u32;
        next_blob += pages as u32;
        table.extend(order.iter().map(|&i| VolumeBrick {
            coord: rel[i],
            _pad: 0,
        }));
        grids.push(record);
        grid_bricks.push(encoded.into_iter().flat_map(|(bytes, _)| bytes).collect());
    }
    let info = VolumeInfo {
        header: VolumeHeader {
            origin: src.origin,
            voxel_size: src.voxel_size,
            brick_min,
            bricks,
            majorant_cells: cells,
            _pad: 0,
            bounds,
            grids: Span::default(),
            names: Span::default(),
            brick_table: Span::default(),
        },
        grids,
        names: src.grids.iter().map(|g| g.name.clone()).collect(),
        bricks: table,
    };
    let head = info.to_bytes();
    let majorant_bytes: &[u8] = bytemuck::cast_slice(&majorants);
    let gdeflate = w.gdeflate();
    let mut report = VolumeReport {
        voxel_size: src.voxel_size,
        bricks,
        majorant_bytes: majorant_bytes.len() as u64,
        grids: Vec::new(),
    };
    for ((g, record), b) in src.grids.iter().zip(&info.grids).zip(&gathered) {
        report.grids.push(VolumeGridReport {
            name: g.name.clone(),
            vk_format: record.vk_format,
            blocks: g.blocks.len(),
            bricks: b.coords.len(),
            pages: record.pages.count as usize,
            raw: 0,
            stored: 0,
            value_range: record.value_range,
        });
    }
    w.add_streamed(
        asset,
        Class::Volume as u16,
        volume::SCHEMA,
        key,
        |e| -> Result<()> {
            e.push(&mount(&head))?;
            e.push(&mount(majorant_bytes))?;
            for ((bytes, record), r) in grid_bricks.iter().zip(&info.grids).zip(&mut report.grids) {
                let page_bytes = record.bricks_per_page() * record.brick_bytes();
                let pages: Vec<&[u8]> = bytes.chunks(page_bytes).collect();
                let encoded = pages
                    .par_iter()
                    .map(|p| BlobIn::new(p).encode(ZSTD_LEVEL, gdeflate.as_deref()))
                    .collect::<orr_pack::Result<Vec<_>>>()?;
                for (p, enc) in pages.iter().zip(&encoded) {
                    r.raw += p.len() as u64;
                    r.stored += enc.stored_len() as u64;
                    e.push_encoded(enc)?;
                }
            }
            Ok(())
        },
    )?;
    Ok(report)
}

/// Keeps and orders the grids `specs` name; empty keeps every grid as given.
fn select<'a>(
    asset: &str,
    mut src: SourceVolume<'a>,
    specs: &[VolumeGridSpec],
) -> Result<SourceVolume<'a>> {
    if specs.is_empty() {
        return Ok(src);
    }
    let mut grids = Vec::with_capacity(specs.len());
    for s in specs {
        let i = src
            .grids
            .iter()
            .position(|g| g.name == s.name)
            .ok_or_else(|| volume_error(asset, format!("no grid named {}", s.name)))?;
        grids.push(src.grids.swap_remove(i));
    }
    src.grids = grids;
    Ok(src)
}
