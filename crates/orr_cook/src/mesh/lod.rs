//! Welding, part assignment and the discrete levels of a mesh.

use std::collections::BTreeMap;

use meshopt::{SimplifyOptions, SimplifyVertex, VertexDataAdapter, VertexStream};
use orr_math::{DVec3, Vec3};
use orr_pack::Aabb;
use orr_pack::mesh::Part;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::par::prelude::*;

use super::maps::Displacement;
use super::{Grid, MeshData, MeshSettings, Pivots, Skin};

/// The mesh with identical vertices merged; sections keep their index ranges.
pub(super) struct Welded {
    pub(super) positions: Vec<[f32; 3]>,
    pub(super) normals: Vec<[f32; 3]>,
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
        Self {
            positions: remap_all(&m.positions),
            normals: remap_all(&m.normals),
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

    /// Adds a vertex placed as `at` is, by its position, moves, skin, part value and pivot,
    /// with the normal of `normal` and the tangent, uvs and colors of `from`; returns it.
    pub(super) fn wedge(&mut self, at: u32, normal: u32, from: u32) -> u32 {
        fn copy<T: Copy>(v: &mut Vec<T>, i: u32) {
            v.push(v[i as usize]);
        }
        let new = self.positions.len() as u32;
        copy(&mut self.positions, at);
        self.moves.iter_mut().for_each(|m| copy(m, at));
        if let Some(k) = &mut self.skin {
            copy(&mut k.joints, at);
            copy(&mut k.weights, at);
        }
        self.values.iter_mut().for_each(|v| copy(v, at));
        self.pivots.iter_mut().for_each(|p| copy(p, at));
        copy(&mut self.normals, normal);
        self.tangents.iter_mut().for_each(|t| copy(t, from));
        self.uvs.iter_mut().for_each(|u| copy(u, from));
        self.colors.iter_mut().for_each(|c| copy(c, from));
        new
    }
}

/// A corner of a simplified triangle whose vertex normal opposes the triangle's face
/// ([`opposed_corners`]), and the vertices it takes its shading from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) struct Opposed {
    /// The corner's index in the simplified triangles.
    pub(super) corner: usize,
    /// The vertex whose normal the corner takes.
    pub(super) normal: u32,
    /// The vertex whose tangent, uvs and colors the corner takes: its own vertex, or another.
    pub(super) attributes: u32,
}

/// The vertices of a triangle list, for searches by position, and the patch each lies in: the
/// vertices the list's triangles join, across which their attributes run on.
struct Near {
    /// The vertices by their coordinate on the axis their box is longest on, so a search stops
    /// once that coordinate alone is farther than the nearest found: a piece can hold a whole
    /// section.
    axis: usize,
    sorted: Vec<(f64, u32)>,
    patch: FxHashMap<u32, u32>,
}

impl Near {
    fn new(triangles: &[u32], at: impl Fn(u32) -> DVec3) -> Self {
        fn root(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                let up = parent[parent[i as usize] as usize];
                parent[i as usize] = up;
                i = up;
            }
            i
        }
        let mut c = triangles.to_vec();
        c.sort_unstable();
        c.dedup();
        let dense = |v: u32| c.binary_search(&v).expect("a vertex of the triangles") as u32;
        let mut parent: Vec<u32> = (0..c.len() as u32).collect();
        for t in triangles.as_chunks::<3>().0 {
            let r = root(&mut parent, dense(t[0]));
            for &v in &t[1..] {
                let s = root(&mut parent, dense(v));
                parent[s as usize] = r;
            }
        }
        let patch = (0..c.len() as u32)
            .map(|i| (c[i as usize], root(&mut parent, i)))
            .collect();
        let (lo, hi) = c
            .iter()
            .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), &u| {
                (lo.min(at(u)), hi.max(at(u)))
            });
        let size = hi - lo;
        let axis = (0..3)
            .max_by(|&a, &b| size[a].total_cmp(&size[b]))
            .expect("three axes");
        let mut sorted: Vec<(f64, u32)> = c.into_iter().map(|u| (at(u)[axis], u)).collect();
        sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        Self {
            axis,
            sorted,
            patch,
        }
    }

    fn patch(&self, v: u32) -> u32 {
        self.patch[&v]
    }

    /// The vertex `take` accepts nearest `p`, the least of those as near, when there is one.
    fn nearest(
        &self,
        p: DVec3,
        at: impl Fn(u32) -> DVec3,
        take: impl Fn(u32) -> bool,
    ) -> Option<u32> {
        let x = p[self.axis];
        let mut nearest: Option<(f64, u32)> = None;
        // Whether `u`, at `c` on the axis, and every vertex past it are farther than the
        // nearest; else takes `u` if it is nearer.
        let mut past = |c: f64, u: u32| {
            if nearest.is_some_and(|(d, _)| (c - x) * (c - x) > d) {
                return true;
            }
            if take(u) {
                let d = at(u).distance_squared(p);
                if nearest.is_none_or(|(e, w)| d < e || (d == e && u < w)) {
                    nearest = Some((d, u));
                }
            }
            false
        };
        let mid = self.sorted.partition_point(|&(c, _)| c < x);
        for &(c, u) in &self.sorted[mid..] {
            if past(c, u) {
                break;
            }
        }
        for &(c, u) in self.sorted[..mid].iter().rev() {
            if past(c, u) {
                break;
            }
        }
        nearest.map(|(_, u)| u)
    }
}

/// The corners of the triangles `simplified` whose vertex normal opposes their face, with the
/// vertices of `input` they take their shading from. Simplification joins vertices across thin
/// features, where one coarse vertex can stand for both sides of a sheet while it carries one
/// side's normal, and it can fold a curved surface past its normals. Such a corner takes the
/// normal of the vertex in its vertex's section nearest it whose normal faces the triangle's
/// way, when there is one. Its uvs, tangent and colors continue those of the surface the
/// triangle stands on, or its uvs would span unrelated parts of a texture: the patches of
/// `input`, the vertices its triangles join, that hold the triangle's corners whose normal
/// faces its way, or all its corners where none does. A corner whose vertex lies on that
/// surface keeps its vertex's; another takes those of the vertex on it nearest it. A triangle
/// `input` holds as it is, with no vertex `moved`, keeps its corners as the input has them.
/// Vertex `v` is at `at(v)` with normal `normal(v)` in section `section(v)`.
pub(super) fn opposed_corners(
    simplified: &[u32],
    input: &[u32],
    moved: impl Fn(u32) -> bool,
    at: impl Fn(u32) -> DVec3,
    normal: impl Fn(u32) -> DVec3,
    section: impl Fn(u32) -> u16,
) -> Vec<Opposed> {
    // A triangle by its corners from the least, which keeps its winding.
    let key = |t: &[u32; 3]| {
        let k = (0..3).min_by_key(|&k| t[k]).expect("three corners");
        [t[k], t[(k + 1) % 3], t[(k + 2) % 3]]
    };
    let mut kept: Option<FxHashSet<[u32; 3]>> = None;
    let mut near: Option<Near> = None;
    let mut out = Vec::new();
    for (t, tri) in simplified.as_chunks::<3>().0.iter().enumerate() {
        let p = tri.map(&at);
        let face = (p[1] - p[0]).cross(p[2] - p[0]);
        let faces = tri.map(|v| normal(v).dot(face) >= 0.0);
        if faces == [true; 3] {
            continue;
        }
        let kept = kept.get_or_insert_with(|| input.as_chunks::<3>().0.iter().map(key).collect());
        if kept.contains(&key(tri)) && !tri.iter().any(|&v| moved(v)) {
            continue;
        }
        let near = near.get_or_insert_with(|| Near::new(input, &at));
        let stands = if faces == [false; 3] {
            [true; 3]
        } else {
            faces
        };
        let surface: Vec<u32> = (0..3)
            .filter(|&k| stands[k])
            .map(|k| near.patch(tri[k]))
            .collect();
        for (k, &v) in tri.iter().enumerate() {
            if faces[k] {
                continue;
            }
            let s = section(v);
            let normal = near
                .nearest(p[k], &at, |u| section(u) == s && normal(u).dot(face) > 0.0)
                .unwrap_or(v);
            let attributes = if surface.contains(&near.patch(v)) {
                v
            } else {
                near.nearest(p[k], &at, |u| surface.contains(&near.patch(u)))
                    .expect("the triangle's corners lie on its surface")
            };
            if (normal, attributes) != (v, v) {
                out.push(Opposed {
                    corner: 3 * t + k,
                    normal,
                    attributes,
                });
            }
        }
    }
    out
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
/// whose vertex normal comes to oppose its face takes a vertex added to `w` and `grid`
/// ([`opposed_corners`]).
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
    // A compact vertex buffer of the piece, so the simplifier's work follows its size.
    let mut local_of: FxHashMap<u32, u32> = FxHashMap::default();
    let mut global = Vec::new();
    let local: Vec<u32> = indices
        .iter()
        .map(|&v| {
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
        }
    }
    let wedges = opposed_corners(
        &out,
        &local,
        |_| false,
        |v| Vec3::from(positions[v as usize]).as_dvec3(),
        |v| Vec3::from_slice(&normals[3 * v as usize..]).as_dvec3(),
        |_| 0,
    );
    let mut out: Vec<u32> = out.into_iter().map(|v| global[v as usize]).collect();
    let mut made: FxHashMap<[u32; 3], u32> = FxHashMap::default();
    for o in wedges {
        let v = out[o.corner];
        let [n, a] = [o.normal, o.attributes].map(|u| global[u as usize]);
        out[o.corner] = if n == a && grid.reps[n as usize] == grid.reps[v as usize] {
            n
        } else {
            *made.entry([v, n, a]).or_insert_with(|| {
                grid.wedge(v);
                w.wedge(v, n, a)
            })
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
