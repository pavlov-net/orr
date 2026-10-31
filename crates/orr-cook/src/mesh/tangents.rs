//! Whether a mesh's stored tangents can be dropped for tangents orr-scene derives per
//! triangle from positions and uvs (`triangle_geometry` without `MESH_TANGENT`).
//!
//! Over every texel center of each normal map that a triangle covers, the check builds the
//! shading normal `material_normal` gives from the stored tangent and from the derived one,
//! and takes the 99th percentile of the angle between them.

use glam::Vec3;
use orr_pack::texture::normal;
use rayon::prelude::*;

use super::maps::Raster;
use super::{MeshData, NormalSpec, SectionMaps};

/// Histogram bin width, degrees.
const BIN: f32 = 0.01;
const BINS: usize = (180.0 / BIN) as usize + 1;

/// Triangles per parallel job.
const CHUNK: usize = 1024;

/// The outcome of [`check`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct TangentCheck {
    /// Drop `TANGENT`.
    pub derive: bool,
    /// Degrees, the upper edge of its histogram bin; `None` when not compared.
    pub angle_p99: Option<f32>,
    /// Samples compared.
    pub texels: u64,
}

impl TangentCheck {
    /// Stored tangents kept without comparing.
    pub(crate) const KEPT: Self = Self {
        derive: false,
        angle_p99: None,
        texels: 0,
    };
}

/// Compares the shading of `m`'s stored tangents with derived ones on its normal maps;
/// derives when the 99th-percentile angle is within `max_deg`. A section whose material
/// reads the tangent beyond a readable normal map keeps the stored tangents.
pub(crate) fn check(
    m: &MeshData,
    tangents: &[[f32; 4]],
    maps: &[SectionMaps],
    max_deg: f32,
) -> TangentCheck {
    let mut hist = vec![0.0f64; BINS];
    let mut texels = 0u64;
    for (sec, sm) in m.sections.iter().zip(maps) {
        let map = match (&sec.maps.normal, &sm.normal) {
            (NormalSpec::None, _) => continue,
            (NormalSpec::Texture { .. }, Some(map)) => map,
            _ => return TangentCheck::KEPT,
        };
        let (h, n) = m.indices[sec.indices.clone()]
            .par_chunks(3 * CHUNK)
            .map(|chunk| {
                let mut h = vec![0.0f64; BINS];
                let mut n = 0u64;
                for t in chunk.as_chunks::<3>().0 {
                    let tri = Triangle::new(m, tangents, t.map(|i| i as usize));
                    texel_centers(&map[0], tri.uv, |w, weight| {
                        let deg = tri.angle(map, w);
                        h[((deg / BIN) as usize).min(BINS - 1)] += weight;
                        n += 1;
                    });
                }
                (h, n)
            })
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
    fn new(m: &MeshData, tangents: &[[f32; 4]], v: [usize; 3]) -> Self {
        let p = v.map(|i| Vec3::from(m.positions[i]));
        let uv = v.map(|i| m.uvs[i]);
        let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
        let uv1 = [uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]];
        let uv2 = [uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]];
        let r = uv1[0] * uv2[1] - uv2[0] * uv1[1];
        let (tangent, bitangent) = if r != 0.0 {
            (
                (e1 * uv2[1] - e2 * uv1[1]) / r,
                (e2 * uv1[0] - e1 * uv2[0]) / r,
            )
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };
        Self {
            uv,
            normals: v.map(|i| Vec3::from(m.normals[i])),
            tangents: v.map(|i| Vec3::from_slice(&tangents[i][..3])),
            sign: tangents[v[0]][3],
            geometric: e1.cross(e2).normalize_or(Vec3::Z),
            tangent,
            bitangent,
        }
    }

    /// Degrees between the two shading normals at barycentrics `w`.
    fn angle(&self, map: &[Raster; 2], w: Vec3) -> f32 {
        let at = |c: [Vec3; 3]| w.x * c[0] + w.y * c[1] + w.z * c[2];
        let uv = [0, 1].map(|a| w.x * self.uv[0][a] + w.y * self.uv[1][a] + w.z * self.uv[2][a]);
        let v = Vec3::from(normal::decode([map[0].sample(uv), map[1].sample(uv)]));
        let n = at(self.normals).normalize_or(self.geometric);

        let stored = shading_normal(n, at(self.tangents).normalize_or(Vec3::X), self.sign, v);

        let (mut t, mut b) = (self.tangent, self.bitangent);
        if t.dot(t) == 0.0 {
            (t, b) = orthonormal_basis(n);
        }
        let t = (t - n * n.dot(t)).normalize_or(Vec3::X);
        let sign = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };
        let derived = shading_normal(n, t, sign, v);

        stored.dot(derived).clamp(-1.0, 1.0).acos().to_degrees()
    }
}

/// `material_normal`: tangent-space normal `v` in the frame of `n` and tangent `t`.
fn shading_normal(n: Vec3, t: Vec3, sign: f32, v: Vec3) -> Vec3 {
    let t = (t - n * n.dot(t)).normalize_or(Vec3::X);
    let b = n.cross(t) * sign;
    (t * v.x + b * v.y + n * v.z).normalize_or(n)
}

/// `orr.math`'s `orthonormal_basis`: a tangent and bitangent of unit `n`.
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let c = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * c, -s * n.x),
        Vec3::new(c, s + n.y * n.y * a, -n.y),
    )
}

/// Calls `f` with the barycentrics and weight of every texel center of `map` inside the uv
/// triangle `uv`. A triangle covering more texels than the map holds samples a grid coarse
/// enough to stay at the map's texel count, each sample weighted by its cell's texels; a
/// triangle covering no center samples its centroid.
fn texel_centers(map: &Raster, uv: [[f32; 2]; 3], mut f: impl FnMut(Vec3, f64)) {
    let size = [f64::from(map.w), f64::from(map.h)];
    let p = uv.map(|t| [f64::from(t[0]) * size[0], f64::from(t[1]) * size[1]]);
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let area2 = cross(p[0], p[1], p[2]);
    let mut any = false;
    if area2 != 0.0 {
        let step = (area2.abs() * 0.5 / (size[0] * size[1])).sqrt().max(1.0);
        let weight = step * step;
        // Inside where every edge function ex x + ey y + e0, signed by the winding, is >= 0.
        let s = area2.signum();
        let edges = [0, 1, 2].map(|i| {
            let (a, b) = (p[(i + 1) % 3], p[(i + 2) % 3]);
            let (ex, ey) = (-(b[1] - a[1]) * s, (b[0] - a[0]) * s);
            (ex, ey, -(ex * a[0] + ey * a[1]))
        });
        let lo_y = p.iter().map(|q| q[1]).fold(f64::INFINITY, f64::min);
        let hi_y = p.iter().map(|q| q[1]).fold(f64::NEG_INFINITY, f64::max);
        let mut row = ((lo_y / step) - 0.5).ceil();
        while (row + 0.5) * step <= hi_y {
            let y = (row + 0.5) * step;
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
            let mut col = ((x0 / step) - 0.5).ceil();
            while (col + 0.5) * step <= x1 {
                let x = (col + 0.5) * step;
                let b1 = cross(p[0], [x, y], p[2]) / area2;
                let b2 = cross(p[0], p[1], [x, y]) / area2;
                let w = Vec3::new((1.0 - b1 - b2) as f32, b1 as f32, b2 as f32);
                f(w.max(Vec3::ZERO) / w.max(Vec3::ZERO).element_sum(), weight);
                any = true;
                col += 1.0;
            }
            row += 1.0;
        }
    }
    if !any {
        f(Vec3::splat(1.0 / 3.0), 1.0);
    }
}
