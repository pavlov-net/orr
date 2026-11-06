//! Opacity micromaps of alpha-tested clusters, rasterized from the cooked opacity mip 0 over
//! the triangles and corners as the page stores them.

use meshopt::{AlphaTexture, OpacityStates};
use orr_pack::mesh::{
    MicromapEntry, OMM_OPAQUE, OMM_STATE_OPAQUE, OMM_STATE_TRANSPARENT, OMM_STATE_UNKNOWN_OPAQUE,
    OMM_STATE_UNKNOWN_TRANSPARENT, OMM_TRANSPARENT,
};
use rayon::prelude::*;
use serde::Deserialize;

use super::maps::{self, FAIL, Map, Opacity, OpacityRead, PASS, Ranges, Raster};
use crate::texture::Wrap;

const STATES: OpacityStates = OpacityStates::Four;

/// How alpha-tested clusters get opacity micromaps.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OmmSettings {
    /// Highest subdivision level, at most 12.
    pub max_level: u8,
    /// Micro-triangle edge to aim for, in texels of the opacity mip 0; 0 subdivides every
    /// triangle to `max_level`.
    pub target_edge: f32,
    /// Largest micromap block of one cluster (its indices, entries and states), at most
    /// 64 KiB. A cluster takes the highest level whose block fits it before compaction.
    pub cluster_bytes_max: u32,
}

impl Default for OmmSettings {
    fn default() -> Self {
        Self {
            max_level: 6,
            target_edge: 2.0,
            cluster_bytes_max: 16 << 10,
        }
    }
}

impl OmmSettings {
    pub(super) fn check(&self) -> Result<(), String> {
        if self.max_level > 12 {
            return Err(format!("omm max_level {} is over 12", self.max_level));
        }
        if !(self.target_edge.is_finite() && self.target_edge >= 0.0) {
            return Err(format!(
                "omm target_edge {} is not a texel count",
                self.target_edge
            ));
        }
        if self.cluster_bytes_max > 64 << 10 {
            return Err(format!(
                "omm cluster_bytes_max {} is over 64 KiB",
                self.cluster_bytes_max
            ));
        }
        Ok(())
    }
}

/// The opacity mip 0 as UNORM8, remapped so the material cutoff lands on the fixed 0.5
/// cutoff of meshoptimizer's rasterizer: a texel passes the cutoff exactly when its
/// remapped value is at least 128.
pub(super) struct OmmAlpha {
    alpha: Vec<u8>,
    w: u32,
    h: u32,
    sides: Sides,
}

/// Tiles of `TILE`, `TILE`², ... texels a side, finest first: per tile, which sides of the
/// cutoff its texels are on (`PASS`, `FAIL` or both).
struct Sides {
    levels: Vec<SideLevel>,
}

struct SideLevel {
    tile: usize,
    cols: usize,
    flags: Vec<u8>,
}

impl Sides {
    const TILE: usize = 8;
    /// Most tiles one query reads at a level before it walks the texels instead.
    const QUERY_MAX: usize = 4096;

    fn new(alpha: &[u8], w: usize, h: usize) -> Self {
        let (mut cols, mut rows) = (w.div_ceil(Self::TILE), h.div_ceil(Self::TILE));
        let mut flags = vec![0u8; cols * rows];
        flags
            .par_chunks_mut(cols)
            .zip(alpha.par_chunks(w * Self::TILE))
            .for_each(|(out, band)| {
                for row in band.chunks_exact(w) {
                    for (x, &a) in row.iter().enumerate() {
                        out[x / Self::TILE] |= if a >= 128 { PASS } else { FAIL };
                    }
                }
            });
        let mut levels = vec![SideLevel {
            tile: Self::TILE,
            cols,
            flags,
        }];
        while cols > 1 || rows > 1 {
            let fine = levels.last().expect("a level");
            let (c, r) = (cols.div_ceil(Self::TILE), rows.div_ceil(Self::TILE));
            let mut flags = vec![0u8; c * r];
            for (y, row) in fine.flags.chunks_exact(cols).enumerate() {
                let out = &mut flags[y / Self::TILE * c..][..c];
                for (x, &f) in row.iter().enumerate() {
                    out[x / Self::TILE] |= f;
                }
            }
            levels.push(SideLevel {
                tile: fine.tile * Self::TILE,
                cols: c,
                flags,
            });
            (cols, rows) = (c, r);
        }
        Self { levels }
    }

    /// `PASS` or `FAIL` when every texel in `xs` by `ys` is on that side; `None` when
    /// tiles small enough to read hold both.
    fn uniform(&self, [xs, ys]: &[Ranges; 2]) -> Option<u8> {
        let ranges = |r: &Ranges| r.0.into_iter().filter(|r| r[0] <= r[1]);
        for l in self.levels.iter().rev() {
            let span = |r: &Ranges| -> usize {
                ranges(r).map(|r| r[1] / l.tile - r[0] / l.tile + 1).sum()
            };
            if span(xs) * span(ys) > Self::QUERY_MAX {
                return None;
            }
            let mut sides = 0;
            'tiles: for y in ranges(ys) {
                for x in ranges(xs) {
                    for ty in y[0] / l.tile..=y[1] / l.tile {
                        let row = &l.flags[ty * l.cols..];
                        for &f in &row[x[0] / l.tile..=x[1] / l.tile] {
                            sides |= f;
                        }
                        if sides == PASS | FAIL {
                            break 'tiles;
                        }
                    }
                }
            }
            if sides != PASS | FAIL {
                return Some(sides);
            }
        }
        None
    }
}

impl OmmAlpha {
    /// Largest texture side the rasterizer takes.
    pub(super) const MAX_SIZE: u32 = 16384;

    /// `None` for a texture wider or taller than [`Self::MAX_SIZE`].
    pub(super) fn new(mip0: &Raster, cutoff: f32) -> Option<Self> {
        let _z = crate::profile::zone!("mesh.omm_alpha");
        if mip0.w > Self::MAX_SIZE || mip0.h > Self::MAX_SIZE {
            return None;
        }
        let alpha: Vec<u8> = mip0.texels.par_iter().map(|&t| remap(t, cutoff)).collect();
        let sides = Sides::new(&alpha, mip0.w as usize, mip0.h as usize);
        Some(Self {
            alpha,
            w: mip0.w,
            h: mip0.h,
            sides,
        })
    }

    fn texture(&self) -> AlphaTexture<'_> {
        AlphaTexture {
            data: &self.alpha,
            stride: 1,
            pitch: self.w as usize,
            width: self.w,
            height: self.h,
        }
    }
}

/// One texture as micromaps read it.
pub(super) struct OmmRead<'a> {
    read: &'a OpacityRead,
    alpha: OmmAlpha,
}

impl<'a> OmmRead<'a> {
    /// `None` for a texture wider or taller than [`OmmAlpha::MAX_SIZE`].
    pub(super) fn new(read: &'a OpacityRead, cutoff: f32) -> Option<Self> {
        let alpha = OmmAlpha::new(&read.mip0, cutoff)?;
        Some(Self { read, alpha })
    }

    /// Whether mapped uvs `p` read the texture as the rasterizer's sampler does, which repeats.
    fn repeats_over(&self, p: &[[f64; 2]; 3]) -> bool {
        let within = |a: usize| p.iter().all(|c| (0.0..=1.0).contains(&c[a]));
        (0..2).all(|a| self.read.mip0.wrap[a] == Wrap::Repeat || within(a))
    }

    /// Which sides of the cutoff a bilinear sample on uv triangle `tri` can read.
    fn sides(&self, tri: [[f32; 2]; 3]) -> u8 {
        let mip0 = &self.read.mip0;
        // Tiles on one side settle the footprint they hold without walking its texels.
        // The tiles rarely settle a footprint narrower than 16 of them; the runs are cheaper.
        let ranges = mip0.footprint_ranges(tri);
        let uniform = || ranges.and_then(|r| self.alpha.sides.uniform(&r));
        let wide = ranges.is_some_and(|r| r.iter().any(|r| r.len() > 16 * Sides::TILE));
        let sides = wide
            .then(uniform)
            .flatten()
            .unwrap_or_else(|| self.read.passes.footprint(mip0, tri, PASS | FAIL));
        // An empty footprint keeps the side its tiles are on.
        if sides == 0 {
            uniform().unwrap_or(0)
        } else {
            sides
        }
    }
}

fn remap(t: f32, cutoff: f32) -> u8 {
    if t >= cutoff {
        let y = if cutoff < 1.0 {
            0.5 + 0.5 * (t - cutoff) / (1.0 - cutoff)
        } else {
            1.0
        };
        ((y.min(1.0) * 255.0).round() as u8).max(128)
    } else {
        ((0.5 * t.max(0.0) / cutoff * 255.0).round() as u8).min(127)
    }
}

/// Micromaps of one cluster's alpha-tested triangles.
#[derive(Clone, Debug)]
pub(super) struct ClusterOmm {
    /// Per alpha-tested triangle: an entry, or a special index.
    indices: Vec<i32>,
    /// Subdivision level per entry.
    levels: Vec<u8>,
    /// Entry states back to back, in entry order.
    pub(crate) data: Vec<u8>,
    /// The `max_level` its entries were measured under.
    pub(super) max_level: u8,
}

impl ClusterOmm {
    /// The most bytes the cluster adds to its page's micromap region.
    pub(super) fn bytes(&self) -> usize {
        block_bytes(self.indices.len(), self.levels.len(), self.data.len())
    }
}

/// Bytes of a micromap block of `triangles` indices and `entries` entries of `data` bytes.
fn block_bytes(triangles: usize, entries: usize, data: usize) -> usize {
    (2 * triangles).next_multiple_of(4) + size_of::<MicromapEntry>() * entries + data
}

impl Map<OmmRead<'_>> {
    /// Mesh uv `p` as the slot maps it.
    fn mapped(&self, p: [f32; 2]) -> [f64; 2] {
        match self {
            Self::Image(t) => t.read.mip0.mapped(p),
            Self::Tiles(t) => maps::mapped(t.uv, p),
        }
    }
}

/// Triangles of a cluster rasterized alike, measured together.
struct Batch<'a> {
    /// The texture each lies whole in, which the rasterizer can read; `None` for triangles that
    /// reach past one or past an edge it reads otherwise.
    read: Option<&'a OmmRead<'a>>,
    /// Texels a side of the largest texture they read, which their levels aim at.
    size: [u32; 2],
    /// Their places among the cluster's triangles.
    tris: Vec<usize>,
    /// Their corners, in `read`'s mapped uvs or the slot's.
    uvs: Vec<[f32; 2]>,
}

/// The cluster's triangles `tris` over its `uvs` by what they read, in order of first use.
fn batches<'a>(alpha: &'a Map<OmmRead<'a>>, uvs: &[[f32; 2]], tris: &[[u8; 3]]) -> Vec<Batch<'a>> {
    let narrow = |p: [f64; 2]| p.map(|x| x as f32);
    let mut out: Vec<Batch<'a>> = Vec::new();
    for (t, tri) in tris.iter().enumerate() {
        let uv = tri.map(|v| uvs[usize::from(v)]);
        let (mut parts, mut size, mut last) = (0, [1, 1], None);
        alpha.parts(uv, |p| {
            parts += 1;
            size = [size[0].max(p.read.alpha.w), size[1].max(p.read.alpha.h)];
            last = Some((p.read, p.uv));
            true
        });
        let whole = last.filter(|_| parts == 1);
        let whole = whole.map(|(r, p)| (r, p.map(|c| r.read.mip0.mapped(c))));
        let (read, corners) = match whole.filter(|(r, p)| r.repeats_over(p)) {
            Some((r, p)) => (Some(r), p.map(narrow)),
            None => (None, uv.map(|c| narrow(alpha.mapped(c)))),
        };
        let alike = |b: &Batch<'a>| {
            b.size == size
                && match (b.read, read) {
                    (Some(a), Some(b)) => std::ptr::eq(a, b),
                    (a, b) => a.is_none() && b.is_none(),
                }
        };
        let at = out.iter().position(alike).unwrap_or_else(|| {
            out.push(Batch {
                read,
                size,
                tris: Vec::new(),
                uvs: Vec::new(),
            });
            out.len() - 1
        });
        out[at].tris.push(t);
        out[at].uvs.extend(corners);
    }
    out
}

/// Micromaps of a cluster's alpha-tested triangles: `tris` in stored corner order over the
/// cluster's stored vertex uvs. Known states never contradict the alpha test, a bilinear
/// sample of mip 0 against the cutoff, anywhere on their micro-triangle. `None` when the
/// opacity is unknown to the cook.
pub(super) fn cluster_omm(
    opacity: &Opacity,
    alpha: Option<&Map<OmmRead<'_>>>,
    uvs: &[[f32; 2]],
    tris: &[[u8; 3]],
    s: &OmmSettings,
) -> Option<ClusterOmm> {
    if tris.is_empty() {
        return None;
    }
    let cutoff = match opacity {
        Opacity::Constant(pass) => {
            return Some(ClusterOmm {
                indices: vec![
                    i32::from(if *pass { OMM_OPAQUE } else { OMM_TRANSPARENT });
                    tris.len()
                ],
                levels: Vec::new(),
                data: Vec::new(),
                max_level: s.max_level,
            });
        }
        Opacity::Texture { cutoff, .. } => *cutoff,
        Opacity::Unknown => return None,
    };
    let alpha = alpha?;
    let _z = crate::profile::zone!("omm.cluster");
    // The rasterizer reads the texture at the uvs the material samples it at.
    let batches = batches(alpha, uvs, tris);
    // The highest level whose block fits the cap before compaction, which only shrinks it.
    let mut level = s.max_level;
    let (measured, mut levels, mut offsets, size) = loop {
        let measured: Vec<_> = batches
            .iter()
            .map(|b| {
                let corners: Vec<u32> = (0..b.uvs.len() as u32).collect();
                let [w, h] = b.size;
                meshopt::opacity_map_measure(&corners, &b.uvs, w, h, level, s.target_edge)
            })
            .collect();
        let levels: Vec<u8> = measured
            .iter()
            .flat_map(|m| m.levels.iter().copied())
            .collect();
        let (offsets, size) = meshopt::opacity_map_layout(&levels, STATES);
        let bytes = block_bytes(tris.len(), levels.len(), size);
        if level == 0 || bytes <= s.cluster_bytes_max as usize {
            break (measured, levels, offsets, size);
        }
        level -= 1;
    };
    let mut data = vec![0; size];
    let mut indices = vec![0; tris.len()];
    let mut entries = Vec::with_capacity(levels.len());
    for (b, m) in batches.iter().zip(&measured) {
        let first = entries.len() as i32;
        for (&t, &i) in b.tris.iter().zip(&m.indices) {
            indices[t] = first + i;
        }
        entries.extend(m.sources.iter().map(|&t| (b, t as usize)));
    }
    // Rasterizing and checking run as two passes so a profile shows each.
    let entry = |levels: &[u8], i: usize| {
        let at = offsets[i] as usize;
        at..at + meshopt::opacity_map_entry_size(levels[i], STATES)
    };
    let mesh_uv = |b: &Batch<'_>, t: usize| tris[b.tris[t]].map(|v| uvs[usize::from(v)]);
    {
        let _z = crate::profile::zone!("omm.rasterize");
        for (i, &(b, t)) in entries.iter().enumerate() {
            let entry = &mut data[entry(&levels, i)];
            match b.read {
                Some(r) => {
                    let at = [0, 1, 2].map(|k| b.uvs[3 * t + k]);
                    let texture = r.alpha.texture();
                    meshopt::opacity_map_rasterize(entry, levels[i], STATES, &at, &texture);
                }
                None => rasterize_across(entry, levels[i], mesh_uv(b, t), alpha, cutoff),
            }
        }
    }
    {
        let _z = crate::profile::zone!("omm.check");
        for (i, &(b, t)) in entries.iter().enumerate() {
            let entry = &mut data[entry(&levels, i)];
            known_states_hold(entry, levels[i], mesh_uv(b, t), alpha);
        }
    }
    let _z = crate::profile::zone!("omm.compact");
    meshopt::opacity_map_compact(&mut data, &mut levels, &mut offsets, &mut indices, STATES);
    Some(ClusterOmm {
        indices,
        levels,
        data,
        max_level: level,
    })
}

/// What `opacity_map_rasterize` writes, for a triangle no one texture holds where it reads only
/// one: four-state micro-triangles from the alpha at their corners and centers, without the edge
/// samples it adds to micro-triangles a few texels wide. Unknown states pick their side as it does
/// from those points, which rays forced to two states see.
fn rasterize_across(
    data: &mut [u8],
    level: u8,
    uv: [[f32; 2]; 3],
    alpha: &Map<OmmRead<'_>>,
    cutoff: f32,
) {
    fn visit(
        data: &mut [u8],
        index: usize,
        level: u8,
        tri: [[f32; 2]; 3],
        alpha: &Map<OmmRead<'_>>,
        cutoff: f32,
    ) {
        let [c0, c1, c2] = tri;
        if level > 0 {
            let mid = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let (c01, c12, c20) = (mid(c0, c1), mid(c1, c2), mid(c2, c0));
            let l = level - 1;
            visit(data, index * 4, l, [c0, c01, c20], alpha, cutoff);
            visit(data, index * 4 + 1, l, [c20, c12, c01], alpha, cutoff);
            visit(data, index * 4 + 2, l, [c01, c1, c12], alpha, cutoff);
            visit(data, index * 4 + 3, l, [c12, c20, c2], alpha, cutoff);
            return;
        }
        let value = |r: &OmmRead<'_>, p| f32::from(remap(r.read.mip0.sample(p), cutoff)) / 255.0;
        let corner = |k: usize| {
            let c = alpha.corner(tri, k);
            value(c.read, c.uv)
        };
        let (r, center) = alpha.at([0, 1].map(|a| (c0[a] + c1[a] + c2[a]) / 3.0));
        let a = [corner(0), corner(1), corner(2), value(r, center)];
        let state = if a.iter().all(|&a| a < 0.5) {
            OMM_STATE_TRANSPARENT
        } else if a.iter().all(|&a| a > 0.5) {
            OMM_STATE_OPAQUE
        } else if (a[0] + a[1] + a[2]) * 0.12 + a[3] * 0.64 >= 0.5 {
            OMM_STATE_UNKNOWN_OPAQUE
        } else {
            OMM_STATE_UNKNOWN_TRANSPARENT
        };
        data[index / 4] |= state << (2 * (index % 4));
    }
    visit(data, 0, level, uv, alpha, cutoff);
}

/// Whether micro-triangles `first..first + count` of a four-state entry hold a known state.
fn any_known(data: &[u8], first: usize, count: usize) -> bool {
    if count >= 4 {
        data[first / 4..(first + count) / 4]
            .iter()
            .any(|&b| !b & 0xAA != 0)
    } else {
        (first..first + count).any(|m| data[m / 4] >> (2 * (m % 4)) & 2 == 0)
    }
}

/// Turns each known state into the unknown state of the same side unless every texel a
/// bilinear sample on its micro-triangle reads is on that side of the cutoff.
fn known_states_hold(data: &mut [u8], level: u8, uv: [[f32; 2]; 3], alpha: &Map<OmmRead<'_>>) {
    // Micro-triangles in the order meshoptimizer rasterizes them: children 0 and 2 keep
    // the winding, 1 and 3 flip it. A triangle whose footprint stays on one side of the
    // cutoff settles every micro-triangle under it.
    fn visit(
        data: &mut [u8],
        index: usize,
        level: u8,
        tri: [[f32; 2]; 3],
        alpha: &Map<OmmRead<'_>>,
    ) {
        // Only known states change, so a run of unknown ones needs no footprint.
        if !any_known(data, index << (2 * level), 1 << (2 * level)) {
            return;
        }
        let mut sides = 0;
        alpha.parts(tri, |p| {
            sides |= p.read.sides(p.uv);
            sides != PASS | FAIL
        });
        if sides == PASS | FAIL && level > 0 {
            let [c0, c1, c2] = tri;
            let mid = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let (c01, c12, c20) = (mid(c0, c1), mid(c1, c2), mid(c2, c0));
            let l = level - 1;
            visit(data, index * 4, l, [c0, c01, c20], alpha);
            visit(data, index * 4 + 1, l, [c20, c12, c01], alpha);
            visit(data, index * 4 + 2, l, [c01, c1, c12], alpha);
            visit(data, index * 4 + 3, l, [c12, c20, c2], alpha);
            return;
        }
        let first = index << (2 * level);
        for m in first..first + (1 << (2 * level)) {
            let (byte, shift) = (m / 4, 2 * (m % 4));
            let state = data[byte] >> shift & 3;
            let side = if state == OMM_STATE_OPAQUE {
                PASS
            } else {
                FAIL
            };
            if state < OMM_STATE_UNKNOWN_TRANSPARENT && sides != side {
                data[byte] |= OMM_STATE_UNKNOWN_TRANSPARENT << shift;
            }
        }
    }
    visit(data, 0, level, uv, alpha);
}

/// A page's micromap region and where its clusters' indices start in it.
pub(crate) struct Region {
    pub(super) bytes: Vec<u8>,
    /// Per cluster: its indices' offset in the region, or `None`.
    pub(super) omm_first: Vec<Option<u32>>,
    pub(super) usage: [u32; 13],
}

/// Lays out a page's micromap region: the entries of all its clusters, deduplicated, then
/// each cluster's indices into them, then the entries' states.
pub(crate) fn page_region(clusters: &[Option<&ClusterOmm>]) -> Region {
    let (mut levels, mut data, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for c in clusters.iter().flatten() {
        let base = levels.len() as i32;
        indices.extend(c.indices.iter().map(|&i| if i < 0 { i } else { i + base }));
        levels.extend_from_slice(&c.levels);
        data.extend_from_slice(&c.data);
    }
    let mut omm_first = vec![None; clusters.len()];
    if indices.is_empty() {
        return Region {
            bytes: Vec::new(),
            omm_first,
            usage: [0; 13],
        };
    }
    let (mut offsets, _) = meshopt::opacity_map_layout(&levels, STATES);
    meshopt::opacity_map_compact(&mut data, &mut levels, &mut offsets, &mut indices, STATES);

    let table = size_of::<MicromapEntry>() * levels.len();
    let runs = clusters
        .iter()
        .flatten()
        .map(|c| (2 * c.indices.len()).next_multiple_of(4));
    let data_start = table + runs.sum::<usize>();
    let mut bytes = vec![0; data_start + data.len()];
    let mut usage = [0; 13];
    let entries: Vec<MicromapEntry> = levels
        .iter()
        .zip(&offsets)
        .map(|(&level, &offset)| {
            usage[usize::from(level)] += 1;
            MicromapEntry {
                data_offset: (data_start + offset as usize) as u32,
                level: level.into(),
                format: MicromapEntry::FOUR_STATE,
            }
        })
        .collect();
    bytes[..table].copy_from_slice(bytemuck::cast_slice(&entries));
    let (mut at, mut next) = (table, indices.iter());
    for (first, c) in omm_first.iter_mut().zip(clusters) {
        let Some(c) = c else { continue };
        *first = Some(at as u32);
        for (k, &i) in next.by_ref().take(c.indices.len()).enumerate() {
            let i = i16::try_from(i).expect("micromap index fits i16");
            bytes[at + 2 * k..at + 2 * k + 2].copy_from_slice(&i.to_le_bytes());
        }
        at += (2 * c.indices.len()).next_multiple_of(4);
    }
    bytes[data_start..].copy_from_slice(&data);
    Region {
        bytes,
        omm_first,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        ClusterOmm, FAIL, Map, OMM_OPAQUE, OMM_STATE_OPAQUE, OMM_STATE_TRANSPARENT, OmmAlpha,
        OmmRead, OmmSettings, Opacity, OpacityRead, PASS, cluster_omm, remap,
    };
    use crate::mesh::maps::{Raster, Tiles};
    use crate::texture::Wrap;

    #[test]
    fn uniform_tiles_agree_with_the_texel_walk() {
        // Blobs over a 157 x 119 texture, so tiles straddle its edges.
        let (w, h) = (157u32, 119u32);
        let texels = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                ((x * 0.05).sin() * (y * 0.04).cos() + 0.2).clamp(0.0, 1.0)
            })
            .collect();
        let mut seed = 1u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed as f32 / u32::MAX as f32 * 3.0 - 1.0
        };
        let mut raster = Raster {
            w,
            h,
            texels: Arc::new(texels),
            wrap: [Wrap::Repeat; 2],
            uv: [0.0; 6],
            outside: 0.0,
        };
        for wrap in [Wrap::Repeat, Wrap::Clamp, Wrap::Mirror] {
            raster.wrap = [wrap, Wrap::Repeat];
            let alpha = OmmAlpha::new(&raster, 0.5).unwrap();
            let mut settled = 0;
            for k in 0..4000 {
                let size = [0.5, 0.05, 0.005][k % 3];
                let o = [next(), next()];
                let tri = [0, 1, 2].map(|_| [o[0] + next() * size, o[1] + next() * size]);
                let Some(r) = raster.footprint_ranges(tri) else {
                    continue;
                };
                if let Some(side) = alpha.sides.uniform(&r) {
                    settled += 1;
                    let mut walked = 0;
                    raster.footprint(tri, |t| {
                        walked |= if t >= 0.5 { PASS } else { FAIL };
                        true
                    });
                    assert_eq!(walked, side, "{wrap:?} {tri:?}");
                }
            }
            assert!(
                settled > 500,
                "{wrap:?}: tiles settled {settled} footprints"
            );
        }
    }

    /// `n` x `n` texels past the cutoff 0.5 on the right half of a tile, or `all` of them.
    fn opacity(n: u32, all: bool) -> OpacityRead {
        let mip0 = Raster {
            w: n,
            h: n,
            texels: Arc::new(
                (0..n * n)
                    .map(|i| f32::from(u8::from(all || i % n >= n / 2)))
                    .collect(),
            ),
            wrap: [Wrap::Clamp; 2],
            uv: [0.0; 6],
            outside: 0.0,
        };
        OpacityRead::new(mip0, 0.5)
    }

    fn set(tiles: Vec<((u16, u16), OpacityRead)>) -> Map<OpacityRead> {
        Map::Tiles(Tiles {
            uv: [0.0; 6],
            tiles: tiles.into_iter().collect(),
            missing: OpacityRead::solid(0.0, 0.5),
        })
    }

    fn settings(level: u8) -> OmmSettings {
        OmmSettings {
            max_level: level,
            target_edge: 0.0,
            cluster_bytes_max: 64 << 10,
        }
    }

    fn cluster(
        map: Map<OpacityRead>,
        uvs: &[[f32; 2]],
        tris: &[[u8; 3]],
        s: &OmmSettings,
    ) -> ClusterOmm {
        let opacity = Opacity::Texture {
            map,
            cutoff: 0.5,
            uv_set: 0,
        };
        let Opacity::Texture { map, .. } = &opacity else {
            unreachable!()
        };
        let alpha = map.try_map(|t| OmmRead::new(t, 0.5)).unwrap();
        cluster_omm(&opacity, Some(&alpha), uvs, tris, s).unwrap()
    }

    /// Triangles inside a tile rasterize as the same image's do, wherever the tile lies.
    #[test]
    fn micromaps_of_a_tile_match_the_same_image() {
        let uvs: Vec<[f32; 2]> = (0..64)
            .map(|i| [(i % 8) as f32 / 8.0, (i / 8) as f32 / 8.0])
            .collect();
        let tris: Vec<[u8; 3]> = (0..56u8)
            .filter(|i| i % 8 != 7)
            .flat_map(|i| [[i, i + 1, i + 8], [i + 1, i + 9, i + 8]])
            .collect();
        let moved: Vec<[f32; 2]> = uvs.iter().map(|c| [c[0] + 3.0, c[1] - 2.0]).collect();
        let s = OmmSettings::default();
        let image = cluster(Map::Image(opacity(64, false)), &uvs, &tris, &s);
        let tile = cluster(set(vec![((2, 3), opacity(64, false))]), &moved, &tris, &s);
        assert!(!image.data.is_empty());
        assert_eq!(image.indices, tile.indices);
        assert_eq!(image.levels, tile.levels);
        assert_eq!(image.data, tile.data);
    }

    /// Whole in one cell, a micro-triangle states what that cell holds, whichever cell it is.
    #[test]
    fn a_triangle_across_tiles_states_what_each_part_reads() {
        // One triangle over an opaque tile, a tile half opaque and a cell with no tile.
        let map = set(vec![
            ((0, 0), opacity(16, true)),
            ((0, 1), opacity(32, false)),
        ]);
        let uvs = [[0.25, 0.25], [2.75, 0.25], [0.25, 0.75]];
        let level = 5;
        let m = cluster(map, &uvs, &[[0, 1, 2]], &settings(level));
        assert_eq!(m.levels, [level]);
        fn walk(
            m: &ClusterOmm,
            index: usize,
            level: u8,
            [c0, c1, c2]: [[f32; 2]; 3],
            seen: &mut [u32; 3],
        ) {
            if level > 0 {
                let mid = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
                let (c01, c12, c20) = (mid(c0, c1), mid(c1, c2), mid(c2, c0));
                walk(m, index * 4, level - 1, [c0, c01, c20], seen);
                walk(m, index * 4 + 1, level - 1, [c20, c12, c01], seen);
                walk(m, index * 4 + 2, level - 1, [c01, c1, c12], seen);
                walk(m, index * 4 + 3, level - 1, [c12, c20, c2], seen);
                return;
            }
            let state = m.data[index / 4] >> (2 * (index % 4)) & 3;
            let (lo, hi) = [c0, c1, c2]
                .iter()
                .fold((9.0f32, -9.0f32), |(l, h), c| (l.min(c[0]), h.max(c[0])));
            // Texels a bilinear tap reads reach a texel past the half of the second tile.
            if hi < 0.99 {
                assert_eq!(state, OMM_STATE_OPAQUE, "opaque tile at {lo}..{hi}");
                seen[0] += 1;
            } else if lo > 1.01 && hi < 1.45 {
                assert_eq!(state, OMM_STATE_TRANSPARENT, "left half at {lo}..{hi}");
                seen[1] += 1;
            } else if lo > 1.55 && hi < 1.99 {
                assert_eq!(state, OMM_STATE_OPAQUE, "right half at {lo}..{hi}");
                seen[2] += 1;
            } else if lo > 2.01 {
                assert_eq!(state, OMM_STATE_TRANSPARENT, "no tile at {lo}..{hi}");
                seen[1] += 1;
            }
        }
        let mut seen = [0; 3];
        walk(&m, 0, level, uvs, &mut seen);
        assert!(seen.iter().all(|&n| n > 20), "{seen:?}");
    }

    /// Past an edge, a triangle reads what the wrap says and not what repeating would: over the
    /// opaque half of a tile and on past it, clamping and mirroring stay opaque all the way.
    #[test]
    fn a_triangle_past_an_edge_reads_through_the_images_wrap() {
        let uvs = [[0.75, 0.25], [1.45, 0.25], [0.75, 0.75]];
        let wrapped = |wrap: Wrap| {
            let mut read = opacity(32, false);
            read.mip0.wrap = [wrap; 2];
            cluster(Map::Image(read), &uvs, &[[0, 1, 2]], &settings(4))
        };
        for wrap in [Wrap::Clamp, Wrap::Mirror] {
            assert_eq!(wrapped(wrap).indices, [i32::from(OMM_OPAQUE)], "{wrap:?}");
        }
        assert!(
            wrapped(Wrap::Repeat).indices[0] >= 0,
            "the left half repeats in"
        );
    }

    #[test]
    fn remapped_alpha_passes_at_128_exactly_when_it_passes_the_cutoff() {
        for cutoff in [0.0, 0.1, 0.3, 0.5, 0.5019608, 0.73, 0.999, 1.0, 1.5] {
            for a in 0..=255u8 {
                let t = f32::from(a) / 255.0;
                assert_eq!(remap(t, cutoff) >= 128, t >= cutoff, "{a} at {cutoff}");
            }
        }
    }
}
