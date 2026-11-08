//! Tangents: meshoptimizer's MikkTSpace-compatible ones for a surface that states none, and
//! whether a mesh's tangents can be dropped for tangents orr_scene derives per triangle from
//! positions and uvs (`triangle_geometry` without `MESH_TANGENT`).
//!
//! Over every texel center of each normal map that a triangle covers, the check builds the
//! shading normal `material_normal` gives from the stored tangent and from the derived one,
//! and takes the 99th percentile of the angle between them. A tile set counts each tile's texels,
//! and the cells it has no tile in at the density of its largest. A mesh whose triangles cover
//! more than [`SAMPLES`] texels, as uvs that repeat a map many times do, samples a lattice of
//! texel centers at a whole-texel stride instead, offset by a hash of each triangle, so every
//! center has the same chance of a sample and the histogram is the dense one's in expectation.

use meshopt::TangentOptions;
use orr_math::{DVec2, Vec3};
use orr_pack::texture::normal;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::maps::{Map, Raster};
use super::{MeshData, NormalSpec, SectionMaps};

/// Histogram bin width, degrees.
const BIN: f32 = 0.01;
const BINS: usize = (180.0 / BIN) as usize + 1;

/// Triangles per parallel job.
const CHUNK: usize = 1024;

/// Samples a mesh's check takes, at most about: a 99th percentile from them is within a few
/// thousandths of a percentile of the dense one.
const SAMPLES: u64 = 1 << 22;

/// The tangent of a vertex nothing reads, and of one only degenerate triangles touch: xyz and
/// bitangent sign.
pub(crate) const UNREAD: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// MikkTSpace-compatible tangents, xyz and bitangent sign: one per index, or per vertex without
/// `indices`. Triangles wind counter-clockwise seen from the side of `normals`, which are unit,
/// and `uvs` are the cook's, with v running down the image.
pub(crate) fn generate(
    indices: Option<&[u32]>,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
) -> Vec<[f32; 4]> {
    let _z = crate::profile::zone!("mesh.tangents");
    let mut tangents =
        meshopt::generate_tangents(indices, positions, normals, uvs, TangentOptions::Compatible);
    // MikkTSpace's v runs up the image, and flipping v negates the sign alone.
    tangents.iter_mut().for_each(|t| t[3] = -t[3]);
    tangents
}

/// [`generate`] per entry of `indices` for a surface without normals, from flat ones.
pub(crate) fn generate_flat(
    indices: &[u32],
    positions: &[[f32; 3]],
    uvs: &[[f32; 2]],
) -> Vec<[f32; 4]> {
    let at: Vec<[f32; 3]> = indices.iter().map(|&i| positions[i as usize]).collect();
    let uv: Vec<[f32; 2]> = indices.iter().map(|&i| uvs[i as usize]).collect();
    let faces = at.as_chunks::<3>().0.iter();
    let flat: Vec<[f32; 3]> = faces
        .flat_map(|t| [face_normal(t.map(Vec3::from)).to_array(); 3])
        .collect();
    generate(None, &at, &flat, &uv)
}

/// The unit normal of triangle `p`, counter-clockwise; zero where it has no area.
pub(crate) fn face_normal(p: [Vec3; 3]) -> Vec3 {
    (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero()
}

/// `v` with the elements `copies` names appended, unless it is empty.
pub(crate) fn with_copies<T: Copy>(mut v: Vec<T>, copies: &[u32]) -> Vec<T> {
    if !v.is_empty() {
        for &i in copies {
            v.push(v[i as usize]);
        }
    }
    v
}

/// One tangent per vertex from one per entry of `indices`, for `count` vertices. Corners of one
/// vertex can differ, where uvs fold: those corners move to new vertices past `count`. Returns
/// the tangents and, per new vertex, the vertex it copies.
pub(crate) fn per_vertex(
    count: usize,
    indices: &mut [u32],
    corners: &[[f32; 4]],
) -> (Vec<[f32; 4]>, Vec<u32>) {
    let mut tangents = vec![None; count];
    let mut copies = Vec::new();
    let mut copied: FxHashMap<(u32, [u32; 4]), u32> = FxHashMap::default();
    for (i, &t) in indices.iter_mut().zip(corners) {
        match &mut tangents[*i as usize] {
            slot @ None => *slot = Some(t),
            Some(first) if *first == t => {}
            Some(_) => {
                let source = *i;
                // Adding zero turns -0.0 into 0.0, which `==` counts as equal.
                let key = t.map(|x| (x + 0.0).to_bits());
                *i = *copied.entry((source, key)).or_insert_with(|| {
                    copies.push(source);
                    (count + copies.len() - 1) as u32
                });
                if *i as usize == tangents.len() {
                    tangents.push(Some(t));
                }
            }
        }
    }
    let tangents = tangents.into_iter().map(|t| t.unwrap_or(UNREAD)).collect();
    (tangents, copies)
}

/// The outcome of [`check`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct TangentCheck {
    /// Drop `TANGENT`.
    pub(crate) derive: bool,
    /// Degrees, the upper edge of its histogram bin; `None` when not compared.
    pub(crate) angle_p99: Option<f32>,
    /// Samples compared.
    pub(crate) texels: u64,
}

impl TangentCheck {
    /// Stored tangents kept without comparing.
    pub(super) const KEPT: Self = Self {
        derive: false,
        angle_p99: None,
        texels: 0,
    };
}

/// Compares the shading of `m`'s stored tangents with derived ones on its normal maps;
/// derives when the 99th-percentile angle is within `max_deg`. A section whose material
/// reads the tangent beyond a readable normal map keeps the stored tangents.
pub(super) fn check(
    m: &MeshData,
    tangents: &[[f32; 4]],
    maps: &[SectionMaps],
    max_deg: f32,
) -> TangentCheck {
    let _z = crate::profile::zone!("mesh.tangent_check");
    let mut checked = Vec::new();
    for (sec, sm) in m.sections.iter().zip(maps) {
        let (map, set, frame) = match (&sec.maps.normal, &sm.normal) {
            (NormalSpec::None, _) => continue,
            (
                &NormalSpec::Texture {
                    uv_set, frame_set, ..
                },
                Some(map),
            ) if frame_set < m.uvs.len() => (map, uv_set, frame_set),
            _ => return TangentCheck::KEPT,
        };
        checked.push((sec, map, set, frame));
    }
    // Tiles of a set weigh by the uv area they cover, as texels of one image do, whatever
    // their sizes: each texel counts as the largest tile's do.
    let texels_of = |[w, h]: [u32; 2]| f64::from(w) * f64::from(h);
    let largest = |map: &Map<[Raster; 2]>| {
        map.each()
            .map(|t| [t[0].w, t[0].h])
            .max_by(|a, b| texels_of(*a).total_cmp(&texels_of(*b)))
            .expect("a map holds a texture")
    };
    // Whole texels per part, so neither the sum nor the stride depends on the order threads
    // add in.
    let area: u64 = checked
        .iter()
        .map(|&(sec, map, set, _)| {
            let largest = largest(map);
            m.indices[sec.indices.clone()]
                .par_chunks(3 * CHUNK)
                .map(|chunk| {
                    let mut area = 0u64;
                    for t in chunk.as_chunks::<3>().0 {
                        map.parts(t.map(|i| m.uvs[set][i as usize]), |p| {
                            let [x, ..] = p.read;
                            let size = match [x.w, x.h] {
                                [1, 1] => largest,
                                s => s,
                            };
                            area += texel_area(size, p.uv.map(|c| x.mapped(c))).ceil() as u64;
                            true
                        });
                    }
                    area
                })
                .sum::<u64>()
        })
        .sum();
    let stride = (area > SAMPLES).then(|| (area as f64 / SAMPLES as f64).sqrt().ceil());
    let mut hist = vec![0.0f64; BINS];
    let mut texels = 0u64;
    for (sec, map, set, frame) in checked {
        let largest = largest(map);
        let first = sec.indices.start / 3;
        let (h, n) = m.indices[sec.indices.clone()]
            .par_chunks(3 * CHUNK)
            .enumerate()
            .fold(
                || (vec![0.0f64; BINS], 0u64),
                |(mut h, mut n), (k, chunk)| {
                    for (j, t) in chunk.as_chunks::<3>().0.iter().enumerate() {
                        let tri = Triangle::new(m, [set, frame], tangents, t.map(|i| i as usize));
                        let lattice = stride.map(|s| (s, phase((first + k * CHUNK + j) as u64)));
                        map.parts(tri.uv, |p| {
                            let [x, ..] = p.read;
                            // One texel holds one value and no grid: it reads as finely as the set.
                            let (size, scale) = match [x.w, x.h] {
                                [1, 1] => (largest, 1.0),
                                s => (s, texels_of(largest) / texels_of(s)),
                            };
                            let uv = p.uv.map(|c| x.mapped(c));
                            texel_centers(size, uv, lattice, |w, weight| {
                                let [a, b, c] = p.weights;
                                let at = [0, 1].map(|i| {
                                    w.x * p.uv[0][i] + w.y * p.uv[1][i] + w.z * p.uv[2][i]
                                });
                                let v =
                                    normal::decode([p.read[0].sample(at), p.read[1].sample(at)]);
                                let v = Vec3::from(map.normal_frame(v));
                                let deg = tri.angle(v, a * w.x + b * w.y + c * w.z);
                                h[((deg / BIN) as usize).min(BINS - 1)] += weight * scale;
                                n += 1;
                            });
                            true
                        });
                    }
                    (h, n)
                },
            )
            .reduce(
                || (vec![0.0; BINS], 0),
                |(mut a, na), (b, nb)| {
                    a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
                    (a, na + nb)
                },
            );
        hist.iter_mut().zip(&h).for_each(|(x, y)| *x += y);
        texels += n;
    }
    let total: f64 = hist.iter().sum();
    let angle_p99 = if total == 0.0 {
        0.0
    } else {
        let mut sum = 0.0;
        let bin = hist
            .iter()
            .position(|&c| {
                sum += c;
                sum >= 0.99 * total
            })
            .unwrap_or(BINS - 1);
        (bin + 1) as f32 * BIN
    };
    TangentCheck {
        derive: angle_p99 <= max_deg,
        angle_p99: Some(angle_p99),
        texels,
    }
}

/// One source triangle in object space.
struct Triangle {
    uv: [[f32; 2]; 3],
    normals: [Vec3; 3],
    /// Stored tangents and corner 0's bitangent sign.
    tangents: [Vec3; 3],
    sign: f32,
    geometric: Vec3,
    /// The uv-derived tangent and bitangent; zero when the uvs are degenerate.
    tangent: Vec3,
    bitangent: Vec3,
}

impl Triangle {
    /// The triangle with corners `v`, its map read on uv set `read` and its derived tangent
    /// from uv set `frame`.
    fn new(m: &MeshData, [read, frame]: [usize; 2], tangents: &[[f32; 4]], v: [usize; 3]) -> Self {
        let p = v.map(|i| Vec3::from(m.positions[i]));
        let uv = v.map(|i| m.uvs[read][i]);
        let f = v.map(|i| m.uvs[frame][i]);
        let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
        let uv1 = [f[1][0] - f[0][0], f[1][1] - f[0][1]];
        let uv2 = [f[2][0] - f[0][0], f[2][1] - f[0][1]];
        let r = uv1[0] * uv2[1] - uv2[0] * uv1[1];
        let (tangent, bitangent) = if r != 0.0 {
            (
                (e1 * uv2[1] - e2 * uv1[1]) / r,
                // Uv v runs down the image and a normal map's y up it.
                (e1 * uv2[0] - e2 * uv1[0]) / r,
            )
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };
        Self {
            uv,
            normals: v.map(|i| Vec3::from(m.normals[i]).normalize_or(Vec3::Z)),
            tangents: v.map(|i| Vec3::from_slice(&tangents[i][..3]).normalize_or(Vec3::X)),
            sign: tangents[v[0]][3],
            geometric: e1.cross(e2).normalize_or(Vec3::Z),
            tangent,
            bitangent,
        }
    }

    /// Degrees between the two shading normals at barycentrics `w`, of tangent-space normal `v`.
    fn angle(&self, v: Vec3, w: Vec3) -> f32 {
        let at = |c: [Vec3; 3]| w.x * c[0] + w.y * c[1] + w.z * c[2];
        let frame = at(self.normals);
        let n = frame.normalize_or(self.geometric);

        let tangent = at(self.tangents);
        let bitangent = frame.cross(tangent) * self.sign;
        let stored = (tangent * v.x + bitangent * v.y + frame * v.z).normalize_or(n);

        let (mut t, mut b) = (self.tangent, self.bitangent);
        if t.dot(t) == 0.0 {
            (t, b) = orthonormal_basis(n);
        }
        let t = (t - n * n.dot(t)).normalize_or(Vec3::X);
        let sign = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };
        let derived = (t * v.x + n.cross(t) * sign * v.y + n * v.z).normalize_or(n);

        stored.dot(derived).clamp(-1.0, 1.0).acos().to_degrees()
    }
}

/// `orr_shaders.math`'s `orthonormal_basis`: a tangent and bitangent of unit `n`.
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * c, -s * n.x),
        Vec3::new(c, s + n.y * n.y * a, -n.y),
    )
}

/// Texels of a map of `size` texels that the mapped uv triangle `uv` covers.
fn texel_area(size: [u32; 2], uv: [[f64; 2]; 3]) -> f64 {
    let [w, h] = size.map(f64::from);
    let [a, b, c] = uv.map(|t| DVec2::new(t[0] * w, t[1] * h));
    (b - a).perp_dot(c - a).abs() * 0.5
}

/// A point of the unit square from triangle `t`'s index.
fn phase(t: u64) -> [f64; 2] {
    // splitmix64's finalizer.
    let mut z = t.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    [z >> 32, z & 0xffff_ffff].map(|h| h as f64 / 4_294_967_296.0)
}

/// Calls `f` with the barycentrics and weight of every texel center of a map of `size` texels
/// inside the mapped uv triangle `uv`. A triangle covering more texels than the map holds samples
/// a grid coarse enough to stay at the map's texel count, each sample weighted by its cell's
/// texels; a triangle covering no center samples its centroid. With `lattice`, a stride and a
/// point of the unit square, the grid is at least that coarse in whole texels and starts at the
/// texel center the point picks in its first cell. A triangle covering no center then samples its
/// centroid, at the grid's weight, where that center is the cell's first, so each sample and the
/// centroid keep their dense weight in expectation.
fn texel_centers(
    size: [u32; 2],
    uv: [[f64; 2]; 3],
    lattice: Option<(f64, [f64; 2])>,
    mut f: impl FnMut(Vec3, f64),
) {
    let cap = (texel_area(size, uv) / (f64::from(size[0]) * f64::from(size[1]))).sqrt();
    let size = size.map(f64::from);
    let p = uv.map(|t| DVec2::new(t[0] * size[0], t[1] * size[1]));
    let dense = cap.max(1.0);
    // Bilinear reads between texel centers would smooth the normals a lattice samples.
    let (step, phase, first) = match lattice {
        Some((stride, at)) => {
            let step = cap.max(stride).ceil();
            let cell = at.map(|a| (a * step).floor());
            (step, cell.map(|c| (c + 0.5) / step), cell == [0.0; 2])
        }
        None => (dense, [0.5; 2], true),
    };
    let weight = step * step;
    let any = grid_points(p, step, phase, |w| {
        f(w, weight);
        true
    });
    if !any && first && (lattice.is_none() || !grid_points(p, dense, [0.5; 2], |_| false)) {
        let centroid = if lattice.is_some() { weight } else { 1.0 };
        f(Vec3::splat(1.0 / 3.0), centroid);
    }
}

/// Calls `f` with the barycentrics of each point `(i + phase) * step` inside the triangle `p`, in
/// texels, until it returns false, and returns whether there was one. A triangle of no area has
/// none.
fn grid_points(p: [DVec2; 3], step: f64, phase: [f64; 2], mut f: impl FnMut(Vec3) -> bool) -> bool {
    let cross = |a: DVec2, b: DVec2, c: DVec2| (b - a).perp_dot(c - a);
    let area2 = cross(p[0], p[1], p[2]);
    if area2 == 0.0 {
        return false;
    }
    let [phase_x, phase_y] = phase;
    // Inside where every edge function ex x + ey y + e0, signed by the winding, is >= 0.
    let s = area2.signum();
    let edges = [0, 1, 2].map(|i| {
        let (a, b) = (p[(i + 1) % 3], p[(i + 2) % 3]);
        let (ex, ey) = (-(b[1] - a[1]) * s, (b[0] - a[0]) * s);
        (ex, ey, -(ex * a[0] + ey * a[1]))
    });
    let lo_y = p.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min);
    let hi_y = p.iter().map(|q| q[1]).fold(f64::NEG_INFINITY, f64::max);
    let mut any = false;
    let mut row = ((lo_y / step) - phase_y).ceil();
    while (row + phase_y) * step <= hi_y {
        let y = (row + phase_y) * step;
        let (mut x0, mut x1) = (f64::NEG_INFINITY, f64::INFINITY);
        for &(ex, ey, e0) in &edges {
            let c = ey * y + e0;
            if ex > 0.0 {
                x0 = x0.max(-c / ex);
            } else if ex < 0.0 {
                x1 = x1.min(-c / ex);
            } else if c < 0.0 {
                x1 = f64::NEG_INFINITY;
            }
        }
        let mut col = ((x0 / step) - phase_x).ceil();
        while (col + phase_x) * step <= x1 {
            let x = (col + phase_x) * step;
            let q = DVec2::new(x, y);
            let b1 = cross(p[0], q, p[2]) / area2;
            let b2 = cross(p[0], p[1], q) / area2;
            let w = Vec3::new((1.0 - b1 - b2) as f32, b1 as f32, b2 as f32);
            any = true;
            if !f(w.max(Vec3::ZERO) / w.max(Vec3::ZERO).element_sum()) {
                return true;
            }
            col += 1.0;
        }
        row += 1.0;
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Over phases spread evenly across the unit square, a lattice gives each triangle its dense
    /// weight: one per texel center it covers, and one for its centroid where it covers none or
    /// has no area.
    #[test]
    fn a_lattice_keeps_each_triangles_dense_weight_in_expectation() {
        let texel = |c: [f64; 2]| c.map(|v| v / 64.0);
        let triangles = [
            [[0.1, 0.1], [0.3, 0.1], [0.1, 0.3]],
            [[0.0, 0.0], [7.2, 0.0], [0.0, 7.2]],
            [[0.2, 0.2], [0.4, 0.4], [0.6, 0.6]],
        ];
        for t in triangles {
            let uv = t.map(texel);
            let mut dense = 0.0;
            texel_centers([64, 64], uv, None, |_, w| dense += w);
            let n = 64;
            let mut sum = 0.0;
            for i in 0..n * n {
                let at = [i % n, i / n].map(|k| (f64::from(k) + 0.5) / f64::from(n));
                texel_centers([64, 64], uv, Some((4.0, at)), |_, w| sum += w);
            }
            let mean = sum / f64::from(n * n);
            assert!((mean - dense).abs() < 1e-9, "{t:?}: {mean} against {dense}");
        }
    }
}
