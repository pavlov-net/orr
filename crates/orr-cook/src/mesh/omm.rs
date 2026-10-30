//! Opacity micromaps of alpha-tested clusters, rasterised from the cooked opacity mip 0 over
//! the triangles and corners as the page stores them.

use meshopt::{AlphaTexture, OpacityStates};
use orr_pack::mesh::{
    MicromapEntry, OMM_OPAQUE, OMM_STATE_OPAQUE, OMM_STATE_UNKNOWN_TRANSPARENT, OMM_TRANSPARENT,
};
use serde::Deserialize;

use super::maps::{Opacity, Raster};

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
    /// 64 KiB. A cluster over it is measured again one level lower until it fits.
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
/// cutoff of meshoptimizer's rasteriser: a texel passes the cutoff exactly when its
/// remapped value is at least 128.
pub(crate) struct OmmAlpha {
    alpha: Vec<u8>,
    w: u32,
    h: u32,
}

impl OmmAlpha {
    /// Largest texture side the rasteriser takes.
    pub(crate) const MAX_SIZE: u32 = 16384;

    /// `None` for a texture wider or taller than [`Self::MAX_SIZE`].
    pub(crate) fn new(mip0: &Raster, cutoff: f32) -> Option<Self> {
        if mip0.w > Self::MAX_SIZE || mip0.h > Self::MAX_SIZE {
            return None;
        }
        let alpha = mip0.texels.iter().map(|&t| remap(t, cutoff)).collect();
        Some(Self {
            alpha,
            w: mip0.w,
            h: mip0.h,
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
}

impl ClusterOmm {
    /// The most bytes the cluster adds to its page's micromap region.
    pub(crate) fn bytes(&self) -> usize {
        (2 * self.indices.len()).next_multiple_of(4)
            + size_of::<MicromapEntry>() * self.levels.len()
            + self.data.len()
    }
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
    let (mip0, cutoff) = match opacity {
        Opacity::Constant(pass) => {
            return Some(ClusterOmm {
                indices: vec![
                    i32::from(if *pass { OMM_OPAQUE } else { OMM_TRANSPARENT });
                    tris.len()
                ],
                levels: Vec::new(),
                data: Vec::new(),
            });
        }
        Opacity::Texture { mip0, cutoff } => (mip0, *cutoff),
        Opacity::Unknown => return None,
    };
    let alpha = alpha?;
    let corners: Vec<u32> = tris.iter().flatten().map(|&v| u32::from(v)).collect();
    let mut level = s.max_level;
    loop {
        let m = meshopt::opacity_map_measure(&corners, uvs, alpha.w, alpha.h, level, s.target_edge);
        let mut levels = m.levels;
        let (mut offsets, size) = meshopt::opacity_map_layout(&levels, STATES);
        let mut data = vec![0; size];
        for (i, &source) in m.sources.iter().enumerate() {
            let uv = tris[source as usize].map(|v| uvs[usize::from(v)]);
            let len = meshopt::opacity_map_entry_size(levels[i], STATES);
            let entry = &mut data[offsets[i] as usize..][..len];
            meshopt::opacity_map_rasterize(entry, levels[i], STATES, &uv, &alpha.texture());
            known_states_hold(entry, levels[i], uv, mip0, cutoff);
        }
        let mut indices = m.indices;
        meshopt::opacity_map_compact(&mut data, &mut levels, &mut offsets, &mut indices, STATES);
        let omm = ClusterOmm {
            indices,
            levels,
            data,
        };
        if level == 0 || omm.bytes() <= s.cluster_bytes_max as usize {
            return Some(omm);
        }
        level -= 1;
    }
}

/// Turns each known state into the unknown state of the same side unless every texel a
/// bilinear sample on its micro-triangle reads is on that side of the cutoff.
fn known_states_hold(data: &mut [u8], level: u8, uv: [[f32; 2]; 3], mip0: &Raster, cutoff: f32) {
    const PASS: u8 = 1;
    const FAIL: u8 = 2;
    // Micro-triangles in the order meshoptimizer rasterises them: children 0 and 2 keep
    // the winding, 1 and 3 flip it. A triangle whose footprint stays on one side of the
    // cutoff settles every micro-triangle under it.
    fn visit(
        data: &mut [u8],
        index: usize,
        level: u8,
        tri: [[f32; 2]; 3],
        mip0: &Raster,
        cutoff: f32,
    ) {
        let mut sides = 0;
        mip0.footprint(tri, |t| {
            sides |= if t >= cutoff { PASS } else { FAIL };
            sides != PASS | FAIL
        });
        if sides == PASS | FAIL && level > 0 {
            let [c0, c1, c2] = tri;
            let mid = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
            let (c01, c12, c20) = (mid(c0, c1), mid(c1, c2), mid(c2, c0));
            let l = level - 1;
            visit(data, index * 4, l, [c0, c01, c20], mip0, cutoff);
            visit(data, index * 4 + 1, l, [c20, c12, c01], mip0, cutoff);
            visit(data, index * 4 + 2, l, [c01, c1, c12], mip0, cutoff);
            visit(data, index * 4 + 3, l, [c12, c20, c2], mip0, cutoff);
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
    visit(data, 0, level, uv, mip0, cutoff);
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
    use super::remap;

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
