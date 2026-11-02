//! Opacity micromaps of alpha-tested clusters, rasterized from the cooked opacity mip 0 over
//! the triangles and corners as the page stores them.

use meshopt::{AlphaTexture, OpacityStates};
use orr_pack::mesh::{
    MicromapEntry, OMM_OPAQUE, OMM_STATE_OPAQUE, OMM_STATE_UNKNOWN_TRANSPARENT, OMM_TRANSPARENT,
};
use serde::Deserialize;

use super::maps::{FAIL, Opacity, PASS, Passes, Ranges, Raster};

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
    pub(crate) fn check(&self) -> Result<(), String> {
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
pub(crate) struct OmmAlpha {
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
        for (y, row) in alpha.chunks_exact(w).enumerate() {
            let out = &mut flags[y / Self::TILE * cols..][..cols];
            for (x, &a) in row.iter().enumerate() {
                out[x / Self::TILE] |= if a >= 128 { PASS } else { FAIL };
            }
        }
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
    pub(crate) const MAX_SIZE: u32 = 16384;

    /// `None` for a texture wider or taller than [`Self::MAX_SIZE`].
    pub(crate) fn new(mip0: &Raster, cutoff: f32) -> Option<Self> {
        let _z = crate::profile::zone!("mesh.omm_alpha");
        if mip0.w > Self::MAX_SIZE || mip0.h > Self::MAX_SIZE {
            return None;
        }
        let alpha: Vec<u8> = mip0.texels.iter().map(|&t| remap(t, cutoff)).collect();
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
pub(crate) struct ClusterOmm {
    /// Per alpha-tested triangle: an entry, or a special index.
    pub indices: Vec<i32>,
    /// Subdivision level per entry.
    pub levels: Vec<u8>,
    /// Entry states back to back, in entry order.
    pub data: Vec<u8>,
    /// The `max_level` its entries were measured under.
    pub max_level: u8,
}

impl ClusterOmm {
    /// The most bytes the cluster adds to its page's micromap region.
    pub(crate) fn bytes(&self) -> usize {
        block_bytes(self.indices.len(), self.levels.len(), self.data.len())
    }
}

/// Bytes of a micromap block of `triangles` indices and `entries` entries of `data` bytes.
fn block_bytes(triangles: usize, entries: usize, data: usize) -> usize {
    (2 * triangles).next_multiple_of(4) + size_of::<MicromapEntry>() * entries + data
}

/// Micromaps of a cluster's alpha-tested triangles: `tris` in stored corner order over the
/// cluster's stored vertex uvs. Known states never contradict the alpha test, a bilinear
/// sample of `mip0` against the cutoff, anywhere on their micro-triangle. `None` when the
/// opacity is unknown to the cook.
pub(crate) fn cluster_omm(
    opacity: &Opacity,
    alpha: Option<&OmmAlpha>,
    uvs: &[[f32; 2]],
    tris: &[[u8; 3]],
    s: &OmmSettings,
) -> Option<ClusterOmm> {
    if tris.is_empty() {
        return None;
    }
    let (mip0, passes) = match opacity {
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
        Opacity::Texture { mip0, passes, .. } => (mip0, passes),
        Opacity::Unknown => return None,
    };
    let alpha = alpha?;
    let _z = crate::profile::zone!("omm.cluster");
    let corners: Vec<u32> = tris.iter().flatten().map(|&v| u32::from(v)).collect();
    // The rasterizer reads the texture at the uvs the material samples it at.
    let mapped: Vec<[f32; 2]> = uvs
        .iter()
        .map(|&p| mip0.mapped(p).map(|x| x as f32))
        .collect();
    // The highest level whose block fits the cap before compaction, which only shrinks it.
    let mut level = s.max_level;
    let (m, mut offsets, size) = loop {
        let m =
            meshopt::opacity_map_measure(&corners, &mapped, alpha.w, alpha.h, level, s.target_edge);
        let (offsets, size) = meshopt::opacity_map_layout(&m.levels, STATES);
        let bytes = block_bytes(tris.len(), m.levels.len(), size);
        if level == 0 || bytes <= s.cluster_bytes_max as usize {
            break (m, offsets, size);
        }
        level -= 1;
    };
    let mut levels = m.levels;
    let mut data = vec![0; size];
    // Rasterizing and checking run as two passes so a profile shows each.
    let entry = |i: usize| {
        let at = offsets[i] as usize;
        at..at + meshopt::opacity_map_entry_size(levels[i], STATES)
    };
    {
        let _z = crate::profile::zone!("omm.rasterize");
        for (i, &source) in m.sources.iter().enumerate() {
            let at = tris[source as usize].map(|v| mapped[usize::from(v)]);
            let entry = &mut data[entry(i)];
            meshopt::opacity_map_rasterize(entry, levels[i], STATES, &at, &alpha.texture());
        }
    }
    {
        let _z = crate::profile::zone!("omm.check");
        for (i, &source) in m.sources.iter().enumerate() {
            let uv = tris[source as usize].map(|v| uvs[usize::from(v)]);
            let entry = &mut data[entry(i)];
            known_states_hold(entry, levels[i], uv, mip0, passes, &alpha.sides);
        }
    }
    let _z = crate::profile::zone!("omm.compact");
    let mut indices = m.indices;
    meshopt::opacity_map_compact(&mut data, &mut levels, &mut offsets, &mut indices, STATES);
    Some(ClusterOmm {
        indices,
        levels,
        data,
        max_level: level,
    })
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
fn known_states_hold(
    data: &mut [u8],
    level: u8,
    uv: [[f32; 2]; 3],
    mip0: &Raster,
    passes: &Passes,
    tiles: &Sides,
) {
    // Micro-triangles in the order meshoptimizer rasterizes them: children 0 and 2 keep
    // the winding, 1 and 3 flip it. A triangle whose footprint stays on one side of the
    // cutoff settles every micro-triangle under it.
    fn visit(
        data: &mut [u8],
        index: usize,
        level: u8,
        tri: [[f32; 2]; 3],
        mip0: &Raster,
        passes: &Passes,
        tiles: &Sides,
    ) {
        // Only known states change, so a run of unknown ones needs no footprint.
        if !any_known(data, index << (2 * level), 1 << (2 * level)) {
            return;
        }
        // Tiles on one side settle the footprint they hold without walking its texels.
        // The tiles rarely settle a footprint narrower than 16 of them; the runs are cheaper.
        let ranges = mip0.footprint_ranges(tri);
        let uniform = || ranges.and_then(|r| tiles.uniform(&r));
        let wide = ranges.is_some_and(|r| r.iter().any(|r| r.len() > 16 * Sides::TILE));
        let mut sides = wide
            .then(uniform)
            .flatten()
            .unwrap_or_else(|| passes.footprint(mip0, tri, PASS | FAIL));
        // An empty footprint keeps the side its tiles are on.
        if sides == 0 {
            sides = uniform().unwrap_or(0);
        }
        if sides == PASS | FAIL && level > 0 {
            let [c0, c1, c2] = tri;
            let mid = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let (c01, c12, c20) = (mid(c0, c1), mid(c1, c2), mid(c2, c0));
            let l = level - 1;
            visit(data, index * 4, l, [c0, c01, c20], mip0, passes, tiles);
            visit(data, index * 4 + 1, l, [c20, c12, c01], mip0, passes, tiles);
            visit(data, index * 4 + 2, l, [c01, c1, c12], mip0, passes, tiles);
            visit(data, index * 4 + 3, l, [c12, c20, c2], mip0, passes, tiles);
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
    visit(data, 0, level, uv, mip0, passes, tiles);
}

/// A page's micromap region and where its clusters' indices start in it.
pub(crate) struct Region {
    pub bytes: Vec<u8>,
    /// Per cluster: its indices' offset in the region, or `None`.
    pub omm_first: Vec<Option<u32>>,
    pub usage: [u32; 13],
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
    use super::{FAIL, OmmAlpha, PASS, remap};
    use crate::mesh::maps::Raster;
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
            texels,
            wrap: [Wrap::Repeat; 2],
            uv: [0.0; 6],
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
