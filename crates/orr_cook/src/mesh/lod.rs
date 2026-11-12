//! Welding, part assignment and the discrete levels of a mesh.

use std::collections::BTreeMap;

use meshopt::{SimplifyOptions, SimplifyVertex, VertexDataAdapter, VertexStream};
use orr_math::{DVec3, Vec3};
use orr_pack::Aabb;
use orr_pack::mesh::Part;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::par::prelude::*;

use super::dag::{Uvs, uv_bits};
use super::maps::Displacement;
use super::{Grid, MeshData, MeshSettings, Pivots, Skin};

/// The mesh with identical vertices merged; sections keep their index ranges.
pub(super) struct Welded {
    pub(super) positions: Vec<[f32; 3]>,
    pub(super) normals: Vec<[f32; 3]>,
    /// Per vertex, the smoothing class of its normal: vertices at one position whose normals
    /// are one class shade alike ([`normal_class`]).
    pub(super) classes: Vec<u32>,
    pub(super) tangents: Option<Vec<[f32; 4]>>,
    /// Per uv set, a uv per vertex.
    pub(super) uvs: Vec<Vec<[f32; 2]>>,
    pub(super) colors: Option<Vec<[f32; 4]>>,
    pub(super) moves: Option<Vec<[[f32; 3]; 2]>>,
    pub(super) skin: Option<Skin>,
    pub(super) values: Option<Vec<u16>>,
    pub(super) pivots: Option<Vec<[f32; 3]>>,
    pub(super) indices: Vec<u32>,
    /// Section index range per section.
    pub(super) sections: Vec<std::ops::Range<usize>>,
    /// Per vertex [`Self::wedge`] adds, the vertex it stands for.
    pub(super) stands_for: FxHashMap<u32, u32>,
}

impl Welded {
    /// Welds `m`, without its tangents unless `tangents` and its moves unless `moves`.
    pub(super) fn new(m: &MeshData, tangents: bool, moves: bool) -> Self {
        let _z = crate::profile::zone!("mesh.weld");
        let n = m.positions.len();
        let stored = m.tangents.as_ref().filter(|_| tangents);
        let moves = m.moves.as_ref().filter(|_| moves);
        let mut streams = vec![
            VertexStream::new(m.positions.as_ptr()),
            VertexStream::new(m.normals.as_ptr()),
        ];
        streams.extend(m.uvs.iter().map(|u| VertexStream::new(u.as_ptr())));
        if let Some(t) = stored {
            streams.push(VertexStream::new(t.as_ptr()));
        }
        if let Some(c) = &m.colors {
            streams.push(VertexStream::new(c.as_ptr()));
        }
        if let Some(c) = moves {
            streams.push(VertexStream::new(c.as_ptr()));
        }
        if let Some(k) = &m.skin {
            streams.push(VertexStream::new(k.joints.as_ptr()));
            streams.push(VertexStream::new(k.weights.as_ptr()));
        }
        let pivots = m.parts.as_ref().and_then(|p| match &p.pivots {
            Pivots::PerVertex(v) => Some(v.as_slice()),
            _ => None,
        });
        if let Some(p) = &m.parts {
            streams.push(VertexStream::new(p.values.as_ptr()));
        }
        if let Some(v) = pivots {
            streams.push(VertexStream::new(v.as_ptr()));
        }
        let (count, remap) = meshopt::generate_vertex_remap_multi(n, &streams, Some(&m.indices));
        let remap_all = |v: &[[f32; 3]]| meshopt::remap_vertex_buffer(v, count, &remap);
        let normals = remap_all(&m.normals);
        Self {
            positions: remap_all(&m.positions),
            classes: normals.iter().map(|&n| normal_class(n)).collect(),
            normals,
            tangents: stored.map(|t| meshopt::remap_vertex_buffer(t, count, &remap)),
            uvs: m
                .uvs
                .iter()
                .map(|u| meshopt::remap_vertex_buffer(u, count, &remap))
                .collect(),
            colors: m
                .colors
                .as_ref()
                .map(|c| meshopt::remap_vertex_buffer(c, count, &remap)),
            moves: moves.map(|c| meshopt::remap_vertex_buffer(c, count, &remap)),
            skin: m.skin.as_ref().map(|k| Skin {
                joints: meshopt::remap_vertex_buffer(&k.joints, count, &remap),
                weights: meshopt::remap_vertex_buffer(&k.weights, count, &remap),
            }),
            values: m
                .parts
                .as_ref()
                .map(|p| meshopt::remap_vertex_buffer(&p.values, count, &remap)),
            pivots: pivots.map(remap_all),
            indices: meshopt::remap_index_buffer(Some(&m.indices), count, &remap),
            sections: m.sections.iter().map(|s| s.indices.clone()).collect(),
            stands_for: FxHashMap::default(),
        }
    }

    /// The welded vertices of triangle `t`.
    pub(super) fn triangle(&self, t: usize) -> [u32; 3] {
        self.indices[3 * t..3 * t + 3]
            .try_into()
            .expect("three corners")
    }

    pub(super) fn section_triangles(&self, s: usize) -> impl Iterator<Item = [u32; 3]> + '_ {
        let r = self.sections[s].clone();
        self.indices[r].as_chunks::<3>().0.iter().copied()
    }

    /// The vertex `v` stands for: the one [`Self::wedge`] placed it as, else itself.
    pub(super) fn base(&self, v: u32) -> u32 {
        self.stands_for.get(&v).copied().unwrap_or(v)
    }

    /// Adds a vertex placed as `at` is, by its position, moves, skin, part value and pivot,
    /// with normal `normal`, uvs `uvs` and the tangent, colors and normal class of `from`;
    /// returns it. It stands for the vertex `at` does.
    pub(super) fn wedge(&mut self, at: u32, from: u32, normal: [f32; 3], uvs: Uvs) -> u32 {
        fn copy<T: Copy>(v: &mut Vec<T>, i: u32) {
            v.push(v[i as usize]);
        }
        let new = self.positions.len() as u32;
        self.stands_for.insert(new, self.base(at));
        copy(&mut self.positions, at);
        self.moves.iter_mut().for_each(|m| copy(m, at));
        if let Some(k) = &mut self.skin {
            copy(&mut k.joints, at);
            copy(&mut k.weights, at);
        }
        self.values.iter_mut().for_each(|v| copy(v, at));
        self.pivots.iter_mut().for_each(|p| copy(p, at));
        self.normals.push(normal);
        copy(&mut self.classes, from);
        self.tangents.iter_mut().for_each(|t| copy(t, from));
        self.uvs.iter_mut().zip(uvs).for_each(|(u, uv)| u.push(uv));
        self.colors.iter_mut().for_each(|c| copy(c, from));
        new
    }
}

/// What a corner of a simplified triangle takes in place of its vertex ([`fit`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum Take {
    /// The vertex the input holds at the corner.
    Input(u32),
    /// The tangent, colors and normal class of `attributes`, the normal of [`Fit::normals`] at
    /// `normal`, and `uvs`.
    Fit {
        attributes: u32,
        normal: u32,
        uvs: Uvs,
    },
}

/// Vertex `v` of a simplification: at `at(v)` with normal `normal(v)` of class `class(v)`
/// ([`Welded::classes`]) in section `section(v)`, at the position `position(v)` names, which
/// every vertex there shares, standing for `base(v)` in simplification. `locked(v)` when the simplifier leaves it where it is for other
/// pieces to share, and `moved(v)` when it moved after the simplifier ran.
pub(super) struct FitVertices<'a> {
    pub(super) at: &'a dyn Fn(u32) -> DVec3,
    pub(super) normal: &'a dyn Fn(u32) -> DVec3,
    pub(super) class: &'a dyn Fn(u32) -> u32,
    pub(super) uvs: &'a dyn Fn(u32) -> Uvs,
    pub(super) section: &'a dyn Fn(u32) -> u16,
    pub(super) position: &'a dyn Fn(u32) -> u32,
    pub(super) base: &'a dyn Fn(u32) -> u32,
    pub(super) locked: &'a dyn Fn(u32) -> bool,
    pub(super) moved: &'a dyn Fn(u32) -> bool,
}

/// What [`fit`] gives a simplified mesh's corners.
#[derive(Default)]
pub(super) struct Fit {
    /// The corners that take something in place of their vertex, by their index in the
    /// simplified triangles.
    pub(super) corners: Vec<(usize, Take)>,
    /// Unit normals.
    pub(super) normals: Vec<[f32; 3]>,
}

/// What the corners of the triangles `simplified` take in place of their vertex. `simplified`
/// comes from `input` with each vertex replaced by the one it stands for, `bases` per corner
/// of `input`: a vertex added for an earlier corner takes no part in simplification, so it
/// constrains no collapse. A triangle `input` holds as it is, by the vertices they stand for
/// and with no vertex moved, keeps the corners the input gives it.
///
/// Another triangle stands for the surface it replaces, which its vertices may not: the
/// `sloppy` simplifier joins vertices without regard to their attributes, across the two sides
/// of a thin sheet and across uv charts, and a vertex keeps the normal it has at its own point
/// though its triangles now cover all that surface. Where the attribute-aware simplifier made
/// `simplified`, a triangle whose corners face its face from vertices of one patch keeps them,
/// and so does such a corner of another triangle. Each sample of the triangle, at the
/// centroids of a regular subdivision into parts of about half the area of the triangle a
/// point of `input` lies in, takes the nearest point of `input` in the triangle's section
/// whose normal faces the triangle's side, and its normal; where there is none, the nearest
/// point of the section, and the triangle's own face for its normal.
///
/// The triangle's corners take their uvs, tangent, colors and normal class from the patch of
/// `input` most of its samples land on, the triangles joined by shared vertices, across which
/// attributes run on, so its uvs keep to one chart. A corner whose vertex lies on the patch
/// keeps its vertex's. Another takes the uvs the patch's parametrization gives its position,
/// carried on from the patch triangle nearest it ([`SurfaceTriangle::unfold`]), so the
/// triangle's uvs are as dense as the surface's: a nearby vertex's uvs would pinch them to a
/// sliver, whose footprint asks for texels far finer than the surface's. It takes the other
/// attributes of that triangle's corner nearest it, which on a seam is the vertex the patch
/// has there. A corner takes whole a vertex of `input` on the patch whose normal faces its
/// face where a kept triangle holds that vertex at the corner, so the vertex does not split,
/// and where the corner is locked and the vertex is the one it takes, standing for the
/// corner's own, so the pieces that share the position agree.
///
/// Each other corner's normal is the least squares fit of the samples' normals over the
/// corner's triangles, with each vertex's mass lumped: a weighted mean, so it stays among the
/// normals it fits. Corners at one position that take their attributes from one class at one
/// position share the fit, so the input's creases stay and its uv seams take one normal; a
/// corner whose face the shared normal does not face takes one of its own, so no corner's
/// normal opposes its face.
pub(super) fn fit(
    simplified: &[u32],
    input: &[u32],
    bases: &[u32],
    sloppy: bool,
    vx: &FitVertices<'_>,
) -> Fit {
    assert_eq!(bases.len(), input.len(), "a base per input corner");
    // The corner a triangle starts at so that it starts at its least vertex, which keeps its
    // winding.
    let first = |t: &[u32; 3]| (0..3).min_by_key(|&k| t[k]).expect("three corners");
    let rotated = |t: &[u32; 3], k: usize| [t[k], t[(k + 1) % 3], t[(k + 2) % 3]];
    // The input's triangles by the vertices they stand for, each in the order that starts at
    // the least of those.
    let mut kept: FxHashMap<[u32; 3], [u32; 3]> = FxHashMap::default();
    for (b, i) in bases
        .as_chunks::<3>()
        .0
        .iter()
        .zip(input.as_chunks::<3>().0)
    {
        let k = first(b);
        kept.entry(rotated(b, k)).or_insert_with(|| rotated(i, k));
    }
    let tris = simplified.as_chunks::<3>().0;
    // Per triangle, the input's corners where the input holds it.
    let kept_corners: Vec<Option<[u32; 3]>> = tris
        .iter()
        .map(|tri| {
            let k0 = first(tri);
            let held = kept.get(&rotated(tri, k0))?;
            (!tri.iter().any(|&v| (vx.moved)(v))).then(|| [0, 1, 2].map(|k| held[(k + 3 - k0) % 3]))
        })
        .collect();
    // Per vertex, the input vertices the kept triangles hold there.
    let mut holds: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    let mut out = Fit::default();
    for (t, (tri, held)) in tris.iter().zip(&kept_corners).enumerate() {
        let Some(held) = *held else {
            continue;
        };
        for (k, (&v, u)) in tri.iter().zip(held).enumerate() {
            if u != v {
                out.corners.push((3 * t + k, Take::Input(u)));
            }
            let h = holds.entry(v).or_default();
            if !h.contains(&u) {
                h.push(u);
            }
        }
    }
    // The area the fitted triangles cover.
    let fitted: f64 = tris
        .iter()
        .zip(&kept_corners)
        .filter(|(_, held)| held.is_none())
        .map(|(t, _)| {
            let p = t.map(|v| (vx.at)(v));
            0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
        })
        .sum();
    let mut surface: Option<Surface> = None;
    // Per fitted corner: its index, its face, its normal sum, the vertex whose other attributes
    // it takes and its uvs.
    let mut sums: Vec<(usize, DVec3, DVec3, u32, Uvs)> = Vec::new();
    for (t, (tri, held)) in tris.iter().zip(&kept_corners).enumerate() {
        if held.is_some() {
            continue;
        }
        let surface = surface.get_or_insert_with(|| Surface::new(input, vx));
        let p = tri.map(|v| (vx.at)(v));
        let cross = (p[1] - p[0]).cross(p[2] - p[0]);
        let twice = cross.length();
        // A triangle with no area has no side and no samples.
        if twice == 0.0 {
            continue;
        }
        let face = cross / twice;
        let faces = |u: u32| (vx.normal)(u).normalize_or_zero().dot(face) > 0.0;
        // Where the attribute-aware simplifier made the triangle, a corner that faces its face
        // on its triangle's patch keeps its vertex, whose normal the simplifier weighed; the
        // sloppy simplifier weighs none.
        let keeps =
            |v: u32, patch: u32| !sloppy && surface.patch.get(&v) == Some(&patch) && faces(v);
        if let Some(&patch) = surface.patch.get(&tri[0])
            && tri.iter().all(|&v| keeps(v, patch))
        {
            continue;
        }
        let section = (vx.section)(tri[0]);
        let in_section = |s: &SurfaceTriangle| s.section == section;
        // Subdivisions per edge, so each part holds about half an input triangle, but for
        // fitted triangles far larger than the input, as the sloppy simplifier spans the gaps
        // between leaves with, no more parts than four per input triangle: the input holds no
        // more normals to sample.
        let part = (0.5 * surface.area).max(fitted / (4 * surface.tris.len()) as f64);
        let s = (0.5 * twice / part).sqrt().ceil().max(1.0);
        let weight = 0.5 * twice / (s * s);
        let mut sum = [DVec3::ZERO; 3];
        // Per patch the samples land on, their weight.
        let mut votes: Vec<(u32, f64)> = Vec::new();
        let mut sample = |a: f64, b: f64| {
            let l = [1.0 - a - b, a, b];
            let q = p[0] * l[0] + p[1] * l[1] + p[2] * l[2];
            // Where no point faces the triangle's side, as on a triangle the sloppy simplifier
            // turns over, the triangle's own face is the only way it has to face.
            let (st, n) = match surface.nearest(q, Some(face), in_section) {
                Some((st, _, n, _)) => (Some(st), n),
                None => (surface.nearest(q, None, in_section).map(|h| h.0), face),
            };
            for k in 0..3 {
                sum[k] += n * (l[k] * weight);
            }
            if let Some(st) = st {
                let patch = surface.tris[st].patch;
                match votes.iter_mut().find(|v| v.0 == patch) {
                    Some(v) => v.1 += weight,
                    None => votes.push((patch, weight)),
                }
            }
        };
        let n = s as u32;
        for i in 0..n {
            for j in 0..n - i {
                sample(
                    (f64::from(i) + 1.0 / 3.0) / s,
                    (f64::from(j) + 1.0 / 3.0) / s,
                );
                if i + j + 1 < n {
                    sample(
                        (f64::from(i) + 2.0 / 3.0) / s,
                        (f64::from(j) + 2.0 / 3.0) / s,
                    );
                }
            }
        }
        // The section holds no triangle with area: the corners keep their vertices.
        let Some(&(patch, _)) = votes
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        else {
            continue;
        };
        for (k, &v) in tri.iter().enumerate() {
            if keeps(v, patch) {
                continue;
            }
            // The vertex a kept triangle holds here on the patch, which a fitted vertex would
            // split from.
            let kept = holds.get(&v).and_then(|h| {
                h.iter()
                    .copied()
                    .find(|&u| surface.patch.get(&u) == Some(&patch) && faces(u))
            });
            if let Some(u) = kept {
                if u != v {
                    out.corners.push((3 * t + k, Take::Input(u)));
                }
                continue;
            }
            let (u, uvs) = if surface.patch.get(&v) == Some(&patch) {
                (v, (vx.uvs)(v))
            } else {
                let (st, l, ..) = surface
                    .nearest(p[k], None, |s| s.patch == patch)
                    .expect("the patch holds a triangle");
                let near = surface.tris[st].v[corner_of(l)];
                (near, surface.carry(st, l, p[k], vx))
            };
            if (vx.locked)(v) && (vx.base)(u) == v && faces(u) {
                if u != v {
                    out.corners.push((3 * t + k, Take::Input(u)));
                }
                continue;
            }
            sums.push((3 * t + k, face, sum[k], u, uvs));
        }
    }
    // Corners by the normal they share, in the order they first come.
    let mut index: FxHashMap<(u32, u32, u32), usize> = FxHashMap::default();
    let mut shares: Vec<Vec<usize>> = Vec::new();
    for (i, &(c, _, _, u, _)) in sums.iter().enumerate() {
        let key = (
            (vx.position)(simplified[c]),
            (vx.position)(u),
            (vx.class)(u),
        );
        let g = *index.entry(key).or_insert_with(|| {
            shares.push(Vec::new());
            shares.len() - 1
        });
        shares[g].push(i);
    }
    for members in shares {
        let mut pending = vec![members];
        while let Some(m) = pending.pop() {
            let n: DVec3 = m.iter().map(|&i| sums[i].2).sum();
            let (keep, off): (Vec<usize>, Vec<usize>) =
                m.into_iter().partition(|&i| n.dot(sums[i].1) > 0.0);
            let mut emit = |members: &[usize], n: DVec3| {
                let normal = out.normals.len() as u32;
                out.normals.push(n.normalize().as_vec3().to_array());
                out.corners.extend(members.iter().map(|&i| {
                    let (attributes, uvs) = (sums[i].3, sums[i].4);
                    let take = Take::Fit {
                        attributes,
                        normal,
                        uvs,
                    };
                    (sums[i].0, take)
                }));
            };
            match (keep.is_empty(), off.as_slice()) {
                (_, []) => emit(&keep, n),
                // A corner's own sum faces its face but where its samples' normals lie within
                // rounding of the face's plane, which leaves it the face.
                (true, &[i]) => emit(&[i], sums[i].1),
                (true, _) => pending.extend(off.into_iter().map(|i| vec![i])),
                (false, _) => {
                    pending.push(keep);
                    pending.push(off);
                }
            }
        }
    }
    out
}

/// A simplification's input triangles, for nearest point queries, with the patch each lies
/// in: the triangles its vertices join, across which their attributes run on.
struct Surface {
    tris: Vec<SurfaceTriangle>,
    /// A bounding volume hierarchy: an inner node's children are the next node and the node
    /// `first` names; a leaf holds `count` triangles from `first`.
    nodes: Vec<SurfaceNode>,
    /// The area of the triangle a point of the surface lies in, on the mean: the triangles'
    /// areas weighted by area, which small triangles holding little of the surface leave be.
    area: f64,
    /// Per vertex, its patch.
    patch: FxHashMap<u32, u32>,
    /// Per vertex, the triangles at it.
    fans: FxHashMap<u32, Vec<u32>>,
}

struct SurfaceTriangle {
    p: [DVec3; 3],
    /// Unit vertex normals.
    n: [DVec3; 3],
    v: [u32; 3],
    section: u16,
    patch: u32,
}

struct SurfaceNode {
    lo: DVec3,
    hi: DVec3,
    /// A cone holding the node's vertex normals, and so every normal between them: its axis
    /// and half angle.
    axis: DVec3,
    spread: f64,
    first: u32,
    /// 0 for an inner node.
    count: u32,
}

impl SurfaceTriangle {
    /// Where the parametrization takes `q`, whose nearest point of the triangle is at
    /// barycentrics `l`: above the triangle, that point; past an edge or corner, the point as
    /// far past it in the triangle's plane as `q` is from it, so a coarse triangle laid past the
    /// patch keeps its length in uv.
    fn unfold(&self, q: DVec3, l: [f64; 3]) -> DVec3 {
        let x = self.p[0] * l[0] + self.p[1] * l[1] + self.p[2] * l[2];
        if l.iter().all(|&l| l > 0.0) {
            return x;
        }
        let n = (self.p[1] - self.p[0])
            .cross(self.p[2] - self.p[0])
            .normalize();
        let d = q - x;
        let along = d - n * d.dot(n);
        let length = along.length();
        if length == 0.0 {
            return x;
        }
        x + along * (d.length() / length)
    }

    /// The point at barycentrics `l` and the triangle's uvs there.
    fn at(&self, l: [f64; 3], vx: &FitVertices<'_>) -> (DVec3, [[f64; 2]; 2]) {
        let x = self.p[0] * l[0] + self.p[1] * l[1] + self.p[2] * l[2];
        let uvs = self.v.map(|v| (vx.uvs)(v));
        let uv = std::array::from_fn(|set| {
            std::array::from_fn(|axis| (0..3).map(|k| l[k] * f64::from(uvs[k][set][axis])).sum())
        });
        (x, uv)
    }

    /// Twice the triangle's area times the 3D gradient of each of its uvs, which a triangle of
    /// little area, however its uvs spread, holds little of.
    fn gradient(&self, vx: &FitVertices<'_>) -> [[DVec3; 2]; 2] {
        let [a, b, c] = self.p;
        let n = (b - a).cross(c - a).normalize();
        // The edges facing each corner, around the triangle.
        let across = [c - b, a - c, b - a].map(|e| n.cross(e));
        let uvs = self.v.map(|v| (vx.uvs)(v));
        std::array::from_fn(|set| {
            std::array::from_fn(|axis| {
                (0..3)
                    .map(|k| across[k] * f64::from(uvs[k][set][axis]))
                    .sum()
            })
        })
    }
}

impl Surface {
    fn new(input: &[u32], vx: &FitVertices<'_>) -> Self {
        fn root(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                let up = parent[parent[i as usize] as usize];
                parent[i as usize] = up;
                i = up;
            }
            i
        }
        let mut vertices = input.to_vec();
        vertices.sort_unstable();
        vertices.dedup();
        let dense = |v: u32| vertices.binary_search(&v).expect("an input vertex") as u32;
        let mut parent: Vec<u32> = (0..vertices.len() as u32).collect();
        for t in input.as_chunks::<3>().0 {
            let r = root(&mut parent, dense(t[0]));
            for &v in &t[1..] {
                let s = root(&mut parent, dense(v));
                parent[s as usize] = r;
            }
        }
        let patch: FxHashMap<u32, u32> = (0..vertices.len() as u32)
            .map(|i| (vertices[i as usize], root(&mut parent, i)))
            .collect();
        let mut tris: Vec<SurfaceTriangle> = input
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| SurfaceTriangle {
                p: t.map(|v| (vx.at)(v)),
                n: t.map(|v| (vx.normal)(v).normalize_or_zero()),
                v: *t,
                section: (vx.section)(t[0]),
                patch: patch[&t[0]],
            })
            // A triangle with no area holds no point its neighbors do not, and has no side.
            .filter(|t| (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]) != DVec3::ZERO)
            .collect();
        let (sum, squares) = tris.iter().fold((0.0, 0.0), |(s, q), t| {
            let a = 0.5 * (t.p[1] - t.p[0]).cross(t.p[2] - t.p[0]).length();
            (s + a, q + a * a)
        });
        let mut nodes = Vec::with_capacity(tris.len() / 2);
        if !tris.is_empty() {
            Self::build(&mut tris, 0, &mut nodes);
        }
        let mut fans: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        for (t, tri) in tris.iter().enumerate() {
            for v in tri.v {
                fans.entry(v).or_default().push(t as u32);
            }
        }
        Self {
            tris,
            nodes,
            area: squares / sum,
            patch,
            fans,
        }
    }

    /// The uvs of patch `patch`'s parametrization carried to `q`, whose nearest point of the
    /// patch is at barycentrics `l` of triangle `t`: above the patch, the uvs there; past its
    /// border, those at the point as far past it in the triangle's plane as `q` is
    /// ([`SurfaceTriangle::unfold`]), along the patch's uv gradient at the triangle's corner
    /// nearest, the area-weighted mean of its triangles' there, which a sliver's ill-shaped
    /// gradient barely moves.
    fn carry(&self, t: usize, l: [f64; 3], q: DVec3, vx: &FitVertices<'_>) -> Uvs {
        let tri = &self.tris[t];
        let (x, uv) = tri.at(l, vx);
        let d = tri.unfold(q, l) - x;
        let (mut sum, mut twice) = ([[DVec3::ZERO; 2]; 2], 0.0);
        if d != DVec3::ZERO {
            for &f in &self.fans[&tri.v[corner_of(l)]] {
                let f = &self.tris[f as usize];
                if f.patch == tri.patch {
                    let g = f.gradient(vx);
                    for (s, g) in sum.iter_mut().flatten().zip(g.iter().flatten()) {
                        *s += *g;
                    }
                    twice += (f.p[1] - f.p[0]).cross(f.p[2] - f.p[0]).length();
                }
            }
        }
        std::array::from_fn(|set| {
            std::array::from_fn(|axis| {
                let along = if twice > 0.0 {
                    sum[set][axis].dot(d) / twice
                } else {
                    0.0
                };
                (uv[set][axis] + along) as f32
            })
        })
    }

    fn build(tris: &mut [SurfaceTriangle], first: u32, nodes: &mut Vec<SurfaceNode>) {
        let (mut lo, mut hi) = (DVec3::INFINITY, DVec3::NEG_INFINITY);
        let (mut clo, mut chi) = (DVec3::INFINITY, DVec3::NEG_INFINITY);
        let mut sum = DVec3::ZERO;
        for t in tris.iter() {
            for p in t.p {
                (lo, hi) = (lo.min(p), hi.max(p));
            }
            let c = t.p[0] + t.p[1] + t.p[2];
            (clo, chi) = (clo.min(c), chi.max(c));
            sum += t.n[0] + t.n[1] + t.n[2];
        }
        let axis = sum.normalize_or_zero();
        let spread = if axis == DVec3::ZERO {
            std::f64::consts::PI
        } else {
            tris.iter()
                .flat_map(|t| t.n)
                .map(|n| axis.dot(n).clamp(-1.0, 1.0).acos())
                .fold(0.0, f64::max)
        };
        let at = nodes.len();
        nodes.push(SurfaceNode {
            lo,
            hi,
            axis,
            spread,
            first,
            count: tris.len() as u32,
        });
        if tris.len() <= 4 {
            return;
        }
        let size = chi - clo;
        let axis = (0..3)
            .max_by(|&a, &b| size[a].total_cmp(&size[b]))
            .expect("three axes");
        let centroid = |t: &SurfaceTriangle| t.p[0][axis] + t.p[1][axis] + t.p[2][axis];
        let mid = tris.len() / 2;
        tris.select_nth_unstable_by(mid, |a, b| centroid(a).total_cmp(&centroid(b)));
        let (left, right) = tris.split_at_mut(mid);
        nodes[at].count = 0;
        Self::build(left, first, nodes);
        nodes[at].first = nodes.len() as u32;
        Self::build(right, first + mid as u32, nodes);
    }

    /// The nearest point to `q` of a triangle `accept` takes, and with `face`, whose
    /// interpolated normal there faces `face`'s side, of those as near the one whose normal is
    /// nearest `face`: the triangle, the point's barycentrics, the unit normal there and the
    /// squared distance.
    fn nearest(
        &self,
        q: DVec3,
        face: Option<DVec3>,
        accept: impl Fn(&SurfaceTriangle) -> bool,
    ) -> Option<(usize, [f64; 3], DVec3, f64)> {
        type Best = Option<(f64, f64, usize, [f64; 3], DVec3)>;
        let mut best: Best = None;
        if self.nodes.is_empty() {
            return None;
        }
        let past = |d: f64, best: &Best| best.is_some_and(|b| d > b.0);
        let gap = |node: &SurfaceNode| {
            let d = (node.lo - q).max(q - node.hi).max(DVec3::ZERO);
            d.length_squared()
        };
        // Whether no normal of the node's cone faces `face`'s side.
        let away = |node: &SurfaceNode| {
            face.is_some_and(|f| {
                let apart = node.axis.dot(f).clamp(-1.0, 1.0).acos();
                apart - node.spread >= std::f64::consts::FRAC_PI_2
            })
        };
        // Nodes to visit with their gap, the nearer child last.
        let mut stack = vec![(0usize, gap(&self.nodes[0]))];
        while let Some((i, g)) = stack.pop() {
            let node = &self.nodes[i];
            if past(g, &best) || away(node) {
                continue;
            }
            if node.count == 0 {
                let (a, b) = (i + 1, node.first as usize);
                let (ga, gb) = (gap(&self.nodes[a]), gap(&self.nodes[b]));
                if ga <= gb {
                    stack.extend([(b, gb), (a, ga)]);
                } else {
                    stack.extend([(a, ga), (b, gb)]);
                }
                continue;
            }
            let first = node.first as usize;
            for (t, tri) in self.tris[first..first + node.count as usize]
                .iter()
                .enumerate()
            {
                if !accept(tri) {
                    continue;
                }
                let l = closest_on_triangle(q, tri.p);
                let x = tri.p[0] * l[0] + tri.p[1] * l[1] + tri.p[2] * l[2];
                let d = x.distance_squared(q);
                if past(d, &best) {
                    continue;
                }
                let n = (tri.n[0] * l[0] + tri.n[1] * l[1] + tri.n[2] * l[2]).normalize_or_zero();
                let dot = face.map_or(0.0, |f| n.dot(f));
                if face.is_some() && dot <= 0.0 {
                    continue;
                }
                if best.is_none_or(|b| d < b.0 || dot > b.1) {
                    best = Some((d, dot, first + t, l, n));
                }
            }
        }
        best.map(|(d, _, t, l, n)| (t, l, n, d))
    }
}

/// The corner of a triangle nearest the point at barycentrics `l`.
fn corner_of(l: [f64; 3]) -> usize {
    (0..3)
        .max_by(|&a, &b| l[a].total_cmp(&l[b]))
        .expect("three corners")
}

/// Welded vertex `v`'s uvs, in each set the mesh has.
pub(super) fn uvs_of(w: &Welded, v: u32) -> Uvs {
    let v = v as usize;
    [w.uvs[0][v], w.uvs.get(1).map_or([0.0; 2], |u| u[v])]
}

/// The smoothing class of a source normal: the normal a cluster stores for it, so normals
/// that store alike are one. A fitted normal keeps the class of the vertex it is shaded as.
pub(super) fn normal_class(n: [f32; 3]) -> u32 {
    orr_pack::mesh::oct_encode(Vec3::from(n).normalize_or(Vec3::Z).to_array())
}

/// Barycentrics of the point of triangle `p` nearest `q` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn closest_on_triangle(q: DVec3, p: [DVec3; 3]) -> [f64; 3] {
    let [a, b, c] = p;
    let (ab, ac, aq) = (b - a, c - a, q - a);
    let (d1, d2) = (ab.dot(aq), ac.dot(aq));
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bq = q - b;
    let (d3, d4) = (ab.dot(bq), ac.dot(bq));
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return [1.0 - v, v, 0.0];
    }
    let cq = q - c;
    let (d5, d6) = (ab.dot(cq), ac.dot(cq));
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return [1.0 - w, 0.0, w];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - w, w];
    }
    let (v, w) = (vb / (va + vb + vc), vc / (va + vb + vc));
    [1.0 - v - w, v, w]
}

/// Parts of a mesh: [`Cluster::part`](orr_pack::mesh::Cluster::part) names them in a u16.
const MAX_PARTS: usize = 1 << 16;

/// The part table and each triangle's part.
pub(super) struct Parts {
    /// Pivot and joint per part; bounds and sections are filled from the clusters.
    pub(super) table: Vec<Part>,
    /// Part value is not 0.
    pub(super) animated: Vec<bool>,
    /// Part of each triangle, in [`Welded::indices`] order.
    pub(super) of_triangle: Vec<u16>,
}

impl Parts {
    pub(super) fn new(w: &Welded, m: &MeshData, grid: &Grid) -> Result<Self, String> {
        let _z = crate::profile::zone!("mesh.parts");
        let triangles = w.indices.len() / 3;
        let part = |pivot, joint| Part {
            bounds: Aabb::EMPTY,
            pivot,
            joint,
            _pad: 0,
            sections: 0,
        };
        let (Some(p), Some(values)) = (&m.parts, &w.values) else {
            return Ok(Self {
                table: vec![part([0.0; 3], 0)],
                animated: vec![false],
                of_triangle: vec![0; triangles],
            });
        };
        let sections = w.sections.len();
        let per_value = if p.by_section { sections } else { 1 };
        let total = usize::from(p.count) * per_value;
        if total == 0 || total > MAX_PARTS {
            return Err(format!(
                "{} parts ({} values{}); a mesh holds 1 to {MAX_PARTS}",
                total,
                p.count,
                if p.by_section { " x sections" } else { "" }
            ));
        }
        let mut of_triangle = vec![0u16; triangles];
        for (s, r) in w.sections.iter().enumerate() {
            for t in r.start / 3..r.end / 3 {
                let v = [0, 1, 2].map(|k| values[w.indices[3 * t + k] as usize]);
                let value = if v[0] == v[1] || v[0] == v[2] {
                    v[0]
                } else if v[1] == v[2] {
                    v[1]
                } else {
                    v[0].min(v[1]).min(v[2])
                };
                if let Some(&bad) = v.iter().find(|&&x| x >= p.count) {
                    return Err(format!(
                        "part value {bad} is not below the declared {} parts",
                        p.count
                    ));
                }
                let index = usize::from(value) * per_value + if p.by_section { s } else { 0 };
                of_triangle[t] = index as u16;
            }
        }
        // Per part, the pivot its vertices carry: a vertex belongs to the part of its own
        // value (and its triangle's section).
        let mut vertex_pivots: Vec<Option<[f32; 3]>> = vec![None; total];
        if let Some(piv) = &w.pivots {
            for (s, r) in w.sections.iter().enumerate() {
                for &i in &w.indices[r.clone()] {
                    let index = usize::from(values[i as usize]) * per_value
                        + if p.by_section { s } else { 0 };
                    let q = piv[i as usize];
                    let f = *vertex_pivots[index].get_or_insert(q);
                    if (0..3).any(|a| f64::from((q[a] - f[a]).abs()) > grid.step()) {
                        return Err(format!(
                            "part {index}: pivots {f:?} and {q:?} m differ by more than a grid step"
                        ));
                    }
                }
            }
        }
        let mut table = Vec::with_capacity(total);
        let mut animated = Vec::with_capacity(total);
        for (index, own) in vertex_pivots.iter().enumerate() {
            let value = (index / per_value) as u16;
            let pivot = match &p.pivots {
                Pivots::PerValue(v) => v[usize::from(value)],
                Pivots::Origin => [0.0; 3],
                Pivots::PerVertex(_) => own.unwrap_or([0.0; 3]),
            };
            let joint = p.joints.as_ref().map_or(value, |j| j[usize::from(value)]);
            table.push(part(pivot, joint));
            animated.push(value != 0);
        }
        Ok(Self {
            table,
            animated,
            of_triangle,
        })
    }
}

/// Triangles of one part and section at one level, or of a section's merged stage.
#[derive(Clone)]
pub(crate) struct Piece {
    pub(super) part: u16,
    pub(super) section: u16,
    pub(super) merged: bool,
    pub(super) indices: Vec<u32>,
}

pub(crate) struct LevelGeometry {
    /// Error from the source, meters; 0 at level 0.
    pub(super) error: f32,
    /// Sorted by part, then section.
    pub(super) pieces: Vec<Piece>,
}

/// The levels of the mesh at `errors`, past level 0; `keep_borders` per section says whether its
/// open borders stay ([`simplify`]).
pub(crate) fn levels(
    w: &mut Welded,
    grid: &mut Grid,
    parts: &Parts,
    s: &MeshSettings,
    errors: &[f32],
    keep_borders: &[bool],
) -> Vec<LevelGeometry> {
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&grid.floats), 12, 0)
        .expect("positions are 12-byte vertices");
    let pieces = pieces(w, parts)
        .into_iter()
        .map(|p| Piece {
            indices: meshopt::spatial_sort_triangles(&p.indices, &adapter),
            ..p
        })
        .collect();
    let mut out = vec![LevelGeometry { error: 0.0, pieces }];
    let n = grid.reps.len();
    let seams = seams(w, n, w.indices.iter().map(|&v| (v, grid.reps[v as usize])));
    let mergeable = parts.table.len() > 1;
    for (i, &target) in errors.iter().enumerate() {
        let level = i + 1;
        let prev = out.last().expect("level 0");
        let merged = mergeable && s.merge_level.is_some_and(|l| level >= usize::from(l));
        let input: Vec<Piece> = if merged && !prev.pieces.iter().any(|p| p.merged) {
            let mut by_section: BTreeMap<u16, Vec<u32>> = BTreeMap::new();
            for p in prev.pieces.iter().filter(|p| !p.indices.is_empty()) {
                by_section.entry(p.section).or_default().extend(&p.indices);
            }
            by_section
                .into_iter()
                .map(|(section, indices)| Piece {
                    part: 0,
                    section,
                    merged: true,
                    indices,
                })
                .collect()
        } else {
            prev.pieces.clone()
        };
        let shared = shared(
            n,
            input
                .iter()
                .map(|p| p.indices.iter().map(|&v| grid.reps[v as usize])),
        );
        let mut step_error = 0f32;
        let pieces = input
            .into_iter()
            .map(|p| {
                let how = Simplify {
                    target,
                    sloppy: true,
                    keep_borders: keep_borders[usize::from(p.section)],
                };
                let (indices, e) = simplify(&p.indices, w, grid, &shared, &seams, how);
                step_error = step_error.max(e);
                Piece { indices, ..p }
            })
            .collect();
        let error = prev.error + step_error;
        out.push(LevelGeometry { error, pieces });
    }
    out
}

/// The triangles of each (part, section), sorted by part, then section.
pub(crate) fn pieces(w: &Welded, parts: &Parts) -> Vec<Piece> {
    piece_triangles(w, parts)
        .into_iter()
        .map(|p| Piece {
            part: p.part,
            section: p.section,
            merged: false,
            indices: p
                .triangles
                .iter()
                .flat_map(|&t| w.triangle(t as usize))
                .collect(),
        })
        .collect()
}

/// The triangles of one part and section, by index.
pub(crate) struct PieceTriangles {
    pub(super) part: u16,
    pub(super) section: u16,
    pub(super) triangles: Vec<u32>,
}

/// The triangles of each (part, section), ascending, sorted by part, then section.
pub(crate) fn piece_triangles(w: &Welded, parts: &Parts) -> Vec<PieceTriangles> {
    let _z = crate::profile::zone!("mesh.pieces");
    let mut index: FxHashMap<(u16, u16), usize> = FxHashMap::default();
    let mut out: Vec<PieceTriangles> = Vec::new();
    for (sec, r) in w.sections.iter().enumerate() {
        let section = sec as u16;
        let mut last = None;
        let first = r.start / 3;
        for (t, &part) in (first..).zip(&parts.of_triangle[first..r.end / 3]) {
            let i = match last {
                Some((p, i)) if p == part => i,
                _ => *index.entry((part, section)).or_insert_with(|| {
                    out.push(PieceTriangles {
                        part,
                        section,
                        triangles: Vec::new(),
                    });
                    out.len() - 1
                }),
            };
            last = Some((part, i));
            out[i].triangles.push(t as u32);
        }
    }
    out.sort_unstable_by_key(|p| (p.part, p.section));
    out
}

/// Per position id below `positions`: whether corners (welded vertex, position id) with
/// different uvs, in either set, or colors meet there.
pub(super) fn seams(
    w: &Welded,
    positions: usize,
    corners: impl IntoIterator<Item = (u32, u32)>,
) -> Vec<bool> {
    let _z = crate::profile::zone!("mesh.seams");
    let mut first = vec![u32::MAX; positions];
    let mut seams = vec![false; positions];
    for (v, id) in corners {
        let f = &mut first[id as usize];
        if *f == u32::MAX {
            *f = v;
            continue;
        }
        if attributes_differ(w, *f, v) {
            seams[id as usize] = true;
        }
    }
    seams
}

/// Whether welded vertices `a` and `b` differ in uvs, in either set, or colors.
pub(super) fn attributes_differ(w: &Welded, a: u32, b: u32) -> bool {
    let (a, b) = (a as usize, b as usize);
    w.colors.as_ref().is_some_and(|c| c[a] != c[b]) || w.uvs.iter().any(|u| u[a] != u[b])
}

/// Per position id below `positions`: whether more than one piece uses it, as on part and
/// section borders.
pub(super) fn shared<P: IntoIterator<Item = u32>>(
    positions: usize,
    pieces: impl IntoIterator<Item = P>,
) -> Vec<bool> {
    let _z = crate::profile::zone!("mesh.shared");
    let mut owner = vec![u32::MAX; positions];
    let mut shared = vec![false; positions];
    for (pi, ids) in pieces.into_iter().enumerate() {
        let pi = pi as u32;
        for id in ids {
            let o = &mut owner[id as usize];
            if *o == u32::MAX {
                *o = pi;
            } else if *o != pi {
                shared[id as usize] = true;
            }
        }
    }
    shared
}

/// [`open_borders`]' mark of a position on an open edge.
pub(super) const BORDER: u8 = 1;
/// [`open_borders`]' mark of a border position the simplifier keeps.
pub(super) const KEEP: u8 = 2;

/// Marks of the positions of the triangles `indices` on open edges: an open border may
/// meet geometry the cook does not see, another mesh or another copy of this one, which a
/// receding border would open a crack to, however small the simplifier's error, and that
/// error does not bound how far a border recedes. Every border position is kept except the
/// inner vertices of straight runs, within a grid cell of the segment between the run's kept
/// ends: a border vertex collapses only along its border loop, so the border there stays on
/// the run. Positions on more than two open edges, with more than one vertex, or on a
/// triangle with two corners at one position are kept, as the simplifier may move those off
/// the loop. Such a triangle has no area, so its edges close none: a source can join two
/// borders with them across a slit the simplifier would otherwise widen. Vertex `v` is at
/// position `pos[v]`, itself a vertex, whose grid cell is `cell(pos[v])`.
pub(super) fn open_borders(
    indices: &[u32],
    pos: &[u32],
    cell: impl Fn(u32) -> [i32; 3],
) -> Vec<u8> {
    let _z = crate::profile::zone!("mesh.open_borders");
    const SEEN: u8 = 4;
    let n = pos.len();
    let at = |t: &[u32; 3]| t.map(|v| pos[v as usize]);
    let flat = |p: [u32; 3]| p[0] == p[1] || p[1] == p[2] || p[2] == p[0];
    let mut degenerate = vec![false; n];
    for t in indices.as_chunks::<3>().0 {
        let p = at(t);
        if flat(p) {
            for q in p {
                degenerate[q as usize] = true;
            }
        }
    }
    let tris: Vec<[u32; 3]> = indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(at)
        .filter(|&p| !flat(p))
        .collect();
    let edge = |p: &[u32; 3], e: usize| (p[e], p[(e + 1) % 3]);
    let mut edges: FxHashSet<(u32, u32)> = Default::default();
    edges.reserve(3 * tris.len());
    for t in &tris {
        for e in 0..3 {
            edges.insert(edge(t, e));
        }
    }
    // Per position: its open edges out and in, and an open edge out.
    let (mut outs, mut ins) = (vec![0u8; n], vec![0u8; n]);
    let mut next = vec![u32::MAX; n];
    let mut open = Vec::new();
    for t in &tris {
        for e in 0..3 {
            let (a, b) = edge(t, e);
            if !edges.contains(&(b, a)) {
                outs[a as usize] = outs[a as usize].saturating_add(1);
                ins[b as usize] = ins[b as usize].saturating_add(1);
                next[a as usize] = b;
                open.push((a, b));
            }
        }
    }
    drop(edges);
    let mut vertices = vec![0u8; n];
    for &p in pos {
        vertices[p as usize] = vertices[p as usize].saturating_add(1);
    }
    let mut marks = vec![0u8; n];
    for p in 0..n {
        if outs[p] > 0 || ins[p] > 0 {
            marks[p] = BORDER;
            if outs[p] != 1 || ins[p] != 1 || vertices[p] != 1 || degenerate[p] {
                marks[p] |= KEEP;
            }
        }
    }
    let cell = |p: u32| cell(p).map(f64::from);
    // Squared distance in cells from `p` to the segment `a`..`b`.
    let off = |a: [f64; 3], b: [f64; 3], p: [f64; 3]| {
        let d = [0, 1, 2].map(|i| b[i] - a[i]);
        let q = [0, 1, 2].map(|i| p[i] - a[i]);
        let dd: f64 = (0..3).map(|i| d[i] * d[i]).sum();
        let t = if dd > 0.0 {
            ((0..3).map(|i| q[i] * d[i]).sum::<f64>() / dd).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (0..3).map(|i| (q[i] - d[i] * t).powi(2)).sum::<f64>()
    };
    // Splits the chain of free positions from `q`, after the kept `start`, into runs, keeping
    // where a run ends. A free position has one open edge out, so the chain ends at a kept one.
    let mut chain = Vec::new();
    let mut runs = |marks: &mut [u8], start: u32, mut q: u32| {
        chain.clear();
        while marks[q as usize] & KEEP == 0 {
            chain.push(q);
            marks[q as usize] |= SEEN;
            q = next[q as usize];
        }
        let (mut s, mut first) = (cell(start), 0);
        for j in 0..=chain.len() {
            let e = cell(chain.get(j).copied().unwrap_or(q));
            if !chain[first..j].iter().all(|&i| off(s, e, cell(i)) <= 1.0) {
                let k = chain[j - 1];
                marks[k as usize] |= KEEP;
                s = cell(k);
                first = j;
            }
        }
    };
    // A chain starts on each open edge from a kept position to a free one; a loop with no
    // kept position keeps one.
    for &(a, b) in &open {
        if marks[a as usize] & KEEP != 0 && marks[b as usize] & (KEEP | SEEN) == 0 {
            runs(&mut marks, a, b);
        }
    }
    for p in 0..n {
        if marks[p] & (BORDER | KEEP | SEEN) == BORDER {
            marks[p] |= KEEP;
            runs(&mut marks, p as u32, next[p]);
        }
    }
    for m in &mut marks {
        *m &= !SEEN;
    }
    marks
}

/// Simplifier flags of a vertex at position id `id`: locked on a shared position, protected
/// on a seam.
pub(super) fn vertex_flags(id: u32, shared: &[bool], seams: &[bool]) -> SimplifyVertex {
    let mut f = SimplifyVertex::None;
    if shared[id as usize] {
        f |= SimplifyVertex::Lock;
    }
    if seams[id as usize] {
        f |= SimplifyVertex::Protect;
    }
    f
}

/// How [`simplify`] simplifies a piece.
#[derive(Copy, Clone)]
pub(super) struct Simplify {
    /// Absolute error, meters.
    pub(super) target: f32,
    /// Sloppy simplification where permissive keeps more than three quarters.
    pub(super) sloppy: bool,
    /// The open borders stay where [`open_borders`] keeps them.
    pub(super) keep_borders: bool,
}

/// Simplifies one piece as `how` says; returns its indices and the error reached. A corner
/// of a triangle that stands for a surface its vertices do not takes a vertex added to `w`
/// and `grid` ([`fit`]).
pub(super) fn simplify(
    indices: &[u32],
    w: &mut Welded,
    grid: &mut Grid,
    shared: &[bool],
    seams: &[bool],
    how: Simplify,
) -> (Vec<u32>, f32) {
    let Simplify {
        target,
        sloppy,
        keep_borders,
    } = how;
    // A compact vertex buffer of the piece, so the simplifier's work follows its size, over
    // the vertices the piece's stand for: those added for earlier corners constrain no
    // collapse ([`fit`]).
    let mut local_of: FxHashMap<u32, u32> = FxHashMap::default();
    let mut global = Vec::new();
    let local: Vec<u32> = indices
        .iter()
        .map(|&v| {
            let v = w.base(v);
            *local_of.entry(v).or_insert_with(|| {
                global.push(v);
                global.len() as u32 - 1
            })
        })
        .collect();
    let positions: Vec<[f32; 3]> = global.iter().map(|&v| grid.floats[v as usize]).collect();
    let normals: Vec<f32> = global.iter().flat_map(|&v| w.normals[v as usize]).collect();
    let mut locks: Vec<SimplifyVertex> = global
        .iter()
        .map(|&v| vertex_flags(grid.reps[v as usize], shared, seams))
        .collect();
    // Per local vertex, the marks of its position, which the first local vertex there names.
    let borders: Option<Vec<u8>> = keep_borders.then(|| {
        let mut first: FxHashMap<u32, u32> = FxHashMap::default();
        let pos: Vec<u32> = (0..global.len() as u32)
            .map(|i| {
                *first
                    .entry(grid.reps[global[i as usize] as usize])
                    .or_insert(i)
            })
            .collect();
        let marks = open_borders(&local, &pos, |p| grid.cells[global[p as usize] as usize]);
        pos.iter().map(|&p| marks[p as usize]).collect()
    });
    for (l, &m) in locks.iter_mut().zip(borders.iter().flatten()) {
        if m & KEEP != 0 {
            *l |= SimplifyVertex::Lock;
        }
    }
    let adapter = VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0)
        .expect("positions are 12-byte vertices");
    let mut error = 0.0;
    let mut out = meshopt::simplify_with_attributes_and_locks(
        &local,
        &adapter,
        &normals,
        &[0.5; 3],
        12,
        &locks,
        0,
        target,
        SimplifyOptions::Permissive
            | SimplifyOptions::ErrorAbsolute
            | SimplifyOptions::ErrorClamped,
        Some(&mut error),
    );
    let mut fell_back = false;
    if sloppy && out.len() * 4 > local.len() * 3 {
        let scale = meshopt::simplify_scale(&adapter);
        // Sloppy simplification moves any vertex it does not lock, along a border or not.
        let on_border = |i: usize| borders.as_ref().is_some_and(|b| b[i] & BORDER != 0);
        let lock: Vec<bool> = locks
            .iter()
            .enumerate()
            .map(|(i, l)| l.contains(SimplifyVertex::Lock) || on_border(i))
            .collect();
        let mut sloppy_error = 0.0;
        let sloppy = meshopt::simplify_sloppy_with_locks(
            &out,
            &adapter,
            &lock,
            0,
            target / scale,
            Some(&mut sloppy_error),
        );
        if sloppy.len() < out.len() {
            out = sloppy;
            error += sloppy_error * scale;
            fell_back = true;
        }
    }
    let mut out: Vec<u32> = out.into_iter().map(|v| global[v as usize]).collect();
    let bases: Vec<u32> = local.iter().map(|&l| global[l as usize]).collect();
    let fit = {
        let at = |v: u32| Vec3::from(grid.floats[v as usize]).as_dvec3();
        let normal = |v: u32| Vec3::from(w.normals[v as usize]).as_dvec3();
        let class = |v: u32| w.classes[v as usize];
        let uvs = |v: u32| uvs_of(w, v);
        let position = |v: u32| grid.reps[v as usize];
        let locked = |v: u32| locks[local_of[&v] as usize].contains(SimplifyVertex::Lock);
        let vx = FitVertices {
            at: &at,
            normal: &normal,
            class: &class,
            uvs: &uvs,
            section: &|_| 0,
            position: &position,
            base: &|v| w.base(v),
            locked: &locked,
            moved: &|_| false,
        };
        fit(&out, indices, &bases, fell_back, &vx)
    };
    // A wedge per vertex, attribute source and normal, also where a vertex at the corner's
    // position has the attributes and normal it takes: unlike that vertex, the wedge stands
    // for the corner's vertex in simplification.
    let mut made: FxHashMap<(u32, u32, u32, [u32; 4]), u32> = FxHashMap::default();
    for (k, take) in fit.corners {
        let v = out[k];
        out[k] = match take {
            Take::Input(u) => u,
            Take::Fit {
                attributes,
                normal,
                uvs,
            } => *made
                .entry((v, attributes, normal, uv_bits(uvs)))
                .or_insert_with(|| {
                    grid.wedge(v);
                    w.wedge(v, attributes, fit.normals[normal as usize], uvs)
                }),
        };
    }
    (out, error)
}

/// Warnings for vertices on a uv seam or a tile's edge of section `s` whose triangles read
/// displacements there more than 1/1024 of the declared range apart.
pub(super) fn displacement_seams(
    w: &Welded,
    grid: &Grid,
    s: usize,
    d: &Displacement,
) -> Vec<String> {
    let Some(map) = &d.map else {
        return Vec::new();
    };
    let _z = crate::profile::zone!("mesh.displacement_seams");
    let tolerance = (d.max - d.min) / 1024.0;
    // Per position: the least and most height its corners read.
    let mut heights = vec![[f32::INFINITY, f32::NEG_INFINITY]; grid.reps.len()];
    for t in w.section_triangles(s) {
        let uv = t.map(|v| w.uvs[d.uv_set][v as usize]);
        for (k, &v) in t.iter().enumerate() {
            let (height, _) = d.corner(map, uv, k);
            let e = &mut heights[grid.reps[v as usize] as usize];
            *e = [e[0].min(height), e[1].max(height)];
        }
    }
    let mut bad: Vec<[i32; 3]> = heights
        .iter()
        .enumerate()
        .filter(|(_, [lo, hi])| hi - lo > tolerance)
        .map(|(id, _)| grid.cells[id])
        .collect();
    if bad.is_empty() {
        return Vec::new();
    }
    bad.sort_unstable();
    let step = grid.step();
    let named: Vec<String> = bad
        .iter()
        .take(8)
        .map(|c| format!("{:?}", c.map(|v| (f64::from(v) * step) as f32)))
        .collect();
    vec![format!(
        "section {s}: displacement differs across a uv seam or tile edge at {} vertices, at {}{}",
        bad.len(),
        named.join(", "),
        if bad.len() > named.len() { ", .." } else { "" }
    )]
}

/// Whether section `s` is closed: on the mesh grid, each edge between two positions is used
/// as often in one direction as in the other.
pub(super) fn closed(w: &Welded, grid: &Grid, s: usize) -> bool {
    let _z = crate::profile::zone!("mesh.closed");
    assert!(
        grid.reps.len() <= 1 << 31,
        "edge keys hold two position ids of 31 bits"
    );
    // Edges between position ids, from triangles over three positions: the lower id, the
    // higher, and the direction in the lowest bit.
    let mut edges: Vec<u64> = w
        .section_triangles(s)
        .map(|t| t.map(|v| grid.reps[v as usize]))
        .filter(|p| p[0] != p[1] && p[1] != p[2] && p[0] != p[2])
        .flat_map(|p| {
            (0..3).map(move |i| {
                let (a, b) = (p[i], p[(i + 1) % 3]);
                u64::from(a.min(b)) << 33 | u64::from(a.max(b)) << 1 | u64::from(a > b)
            })
        })
        .collect();
    if edges.is_empty() {
        return false;
    }
    edges.par_sort_unstable();
    // Each edge's run holds its uses one way, then the other: as many of each.
    edges.chunk_by(|a, b| a >> 1 == b >> 1).all(|run| {
        let back = run.iter().filter(|&&e| e & 1 == 1).count();
        2 * back == run.len()
    })
}

/// Edges of section `s` used by one triangle, by grid position.
pub(super) fn border_edges(w: &Welded, grid: &Grid, s: usize) -> usize {
    let _z = crate::profile::zone!("mesh.border_edges");
    let mut edges: Vec<u64> = w
        .section_triangles(s)
        .map(|t| t.map(|v| grid.reps[v as usize]))
        .flat_map(|p| {
            (0..3).map(move |i| {
                let (a, b) = (p[i], p[(i + 1) % 3]);
                u64::from(a.min(b)) << 32 | u64::from(a.max(b))
            })
        })
        .collect();
    edges.par_sort_unstable();
    edges
        .chunk_by(|a, b| a == b)
        .filter(|run| run.len() == 1)
        .count()
}
