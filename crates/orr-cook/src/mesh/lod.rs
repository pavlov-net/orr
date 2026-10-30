//! Welding, part assignment and the discrete levels of a mesh.

use std::collections::{HashMap, HashSet};

use meshopt::{SimplifyOptions, SimplifyVertex, VertexDataAdapter, VertexStream};
use orr_pack::Aabb;
use orr_pack::mesh::Part;

use super::maps::Displacement;
use super::{Grid, MeshData, MeshSettings, Pivots, Skin};

/// The mesh with identical vertices merged; sections keep their index ranges.
pub(crate) struct Welded {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Option<Vec<[f32; 4]>>,
    pub uvs: Vec<[f32; 2]>,
    pub skin: Option<Skin>,
    pub values: Option<Vec<u16>>,
    pub pivots: Option<Vec<[f32; 3]>>,
    pub indices: Vec<u32>,
    /// Section index range per section.
    pub sections: Vec<std::ops::Range<usize>>,
}

impl Welded {
    pub(crate) fn new(m: &MeshData) -> Self {
        let n = m.positions.len();
        let mut streams = vec![
            VertexStream::new(m.positions.as_ptr()),
            VertexStream::new(m.normals.as_ptr()),
            VertexStream::new(m.uvs.as_ptr()),
        ];
        if let Some(t) = &m.tangents {
            streams.push(VertexStream::new(t.as_ptr()));
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
            tangents: m
                .tangents
                .as_ref()
                .map(|t| meshopt::remap_vertex_buffer(t, count, &remap)),
            uvs: meshopt::remap_vertex_buffer(&m.uvs, count, &remap),
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

    fn section_triangles(&self, s: usize) -> impl Iterator<Item = [u32; 3]> + '_ {
        let r = self.sections[s].clone();
        self.indices[r].chunks_exact(3).map(|t| [t[0], t[1], t[2]])
    }
}

/// The part table and each triangle's part.
pub(crate) struct Parts {
    /// Pivot and joint per part; bounds are filled from the clusters.
    pub table: Vec<Part>,
    /// Part value is not 0.
    pub animated: Vec<bool>,
    /// Part of each triangle, in [`Welded::indices`] order.
    pub of_triangle: Vec<u8>,
}

impl Parts {
    pub(crate) fn new(w: &Welded, m: &MeshData, grid: &Grid) -> Result<Self, String> {
        let triangles = w.indices.len() / 3;
        let part = |pivot, joint| Part {
            bounds: Aabb::EMPTY,
            pivot,
            joint,
            _pad: 0,
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
        if total == 0 || total > 256 {
            return Err(format!(
                "{} parts ({} values{}); a mesh holds 1 to 256",
                total,
                p.count,
                if p.by_section { " x sections" } else { "" }
            ));
        }
        let mut of_triangle = vec![0u8; triangles];
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
                of_triangle[t] = index as u8;
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
    pub part: u8,
    pub section: u8,
    pub merged: bool,
    pub indices: Vec<u32>,
}

pub(crate) struct LevelGeometry {
    /// Error from the source, metres; 0 at level 0.
    pub error: f32,
    /// Sorted by part, then section.
    pub pieces: Vec<Piece>,
}

pub(crate) fn levels(
    w: &Welded,
    grid: &Grid,
    parts: &Parts,
    s: &MeshSettings,
    errors: &[f32],
    sections: usize,
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
    let seams = seam_cells(w, w.indices.iter().map(|&v| (v, grid.cells[v as usize])));
    let mergeable = parts.table.len() > 1;
    for (i, &target) in errors.iter().enumerate() {
        let level = i + 1;
        let prev = out.last().expect("level 0");
        let merged = mergeable && s.merge_level.is_some_and(|l| level >= usize::from(l));
        let input: Vec<Piece> = if merged && !prev.pieces.iter().any(|p| p.merged) {
            (0..sections as u8)
                .filter_map(|section| {
                    let indices: Vec<u32> = prev
                        .pieces
                        .iter()
                        .filter(|p| p.section == section)
                        .flat_map(|p| p.indices.iter().copied())
                        .collect();
                    (!indices.is_empty()).then_some(Piece {
                        part: 0,
                        section,
                        merged: true,
                        indices,
                    })
                })
                .collect()
        } else {
            prev.pieces.clone()
        };
        let shared = shared_cells(
            input
                .iter()
                .map(|p| p.indices.iter().map(|&v| grid.cells[v as usize])),
        );
        let mut step_error = 0f32;
        let pieces = input
            .into_iter()
            .map(|p| {
                let (indices, e) = simplify(&p.indices, w, grid, &shared, &seams, target, true);
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
    let mut by_key: std::collections::BTreeMap<(u8, u8), Vec<u32>> = Default::default();
    for (sec, r) in w.sections.iter().enumerate() {
        for t in r.start / 3..r.end / 3 {
            by_key
                .entry((parts.of_triangle[t], sec as u8))
                .or_default()
                .extend_from_slice(&w.indices[3 * t..3 * t + 3]);
        }
    }
    by_key
        .into_iter()
        .map(|((part, section), indices)| Piece {
            part,
            section,
            merged: false,
            indices,
        })
        .collect()
}

/// Grid positions where corners (welded vertex, grid position) with different uvs meet.
pub(crate) fn seam_cells(
    w: &Welded,
    corners: impl IntoIterator<Item = (u32, [i32; 3])>,
) -> HashSet<[i32; 3]> {
    let mut first: HashMap<[i32; 3], u32> = HashMap::new();
    let mut seams = HashSet::new();
    for (v, cell) in corners {
        let f = *first.entry(cell).or_insert(v);
        if w.uvs[f as usize] != w.uvs[v as usize] {
            seams.insert(cell);
        }
    }
    seams
}

/// Grid positions used by more than one piece: part and section borders.
pub(crate) fn shared_cells<P: IntoIterator<Item = [i32; 3]>>(
    pieces: impl IntoIterator<Item = P>,
) -> HashSet<[i32; 3]> {
    let mut owner: HashMap<[i32; 3], usize> = HashMap::new();
    let mut shared = HashSet::new();
    for (pi, cells) in pieces.into_iter().enumerate() {
        for c in cells {
            if *owner.entry(c).or_insert(pi) != pi {
                shared.insert(c);
            }
        }
    }
    shared
}

/// Simplifier flags of a vertex at `cell`: locked on a shared position, protected on a
/// seam.
pub(crate) fn vertex_flags(
    cell: [i32; 3],
    shared: &HashSet<[i32; 3]>,
    seams: &HashSet<[i32; 3]>,
) -> SimplifyVertex {
    let mut f = SimplifyVertex::None;
    if shared.contains(&cell) {
        f |= SimplifyVertex::Lock;
    }
    if seams.contains(&cell) {
        f |= SimplifyVertex::Protect;
    }
    f
}

/// Simplifies one piece to an absolute error, with sloppy simplification where permissive
/// keeps more than three quarters when `sloppy`; returns its indices and the error reached.
pub(crate) fn simplify(
    indices: &[u32],
    w: &Welded,
    grid: &Grid,
    shared: &HashSet<[i32; 3]>,
    seams: &HashSet<[i32; 3]>,
    target: f32,
    sloppy: bool,
) -> (Vec<u32>, f32) {
    // A compact vertex buffer of the piece, so the simplifier's work follows its size.
    let mut local_of: HashMap<u32, u32> = HashMap::new();
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
    let locks: Vec<SimplifyVertex> = global
        .iter()
        .map(|&v| vertex_flags(grid.cells[v as usize], shared, seams))
        .collect();
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
        let lock: Vec<bool> = locks
            .iter()
            .map(|l| l.contains(SimplifyVertex::Lock))
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
    (out.into_iter().map(|v| global[v as usize]).collect(), error)
}

/// Warnings for vertices on a uv seam of section `s` whose displacement differs between
/// its uvs by more than 1/1024 of the declared range.
pub(crate) fn displacement_seams(
    w: &Welded,
    grid: &Grid,
    s: usize,
    d: &Displacement,
) -> Vec<String> {
    let Some(mip0) = d.mips.first() else {
        return Vec::new();
    };
    let tolerance = (d.max - d.min) / 1024.0;
    let mut uvs: HashMap<[i32; 3], Vec<[f32; 2]>> = HashMap::new();
    for t in w.section_triangles(s) {
        for v in t {
            let e = uvs.entry(grid.cells[v as usize]).or_default();
            let uv = w.uvs[v as usize];
            if !e.contains(&uv) {
                e.push(uv);
            }
        }
    }
    let mut bad: Vec<[i32; 3]> = uvs
        .iter()
        .filter(|(_, uv)| {
            let disp: Vec<f32> = uv
                .iter()
                .map(|&u| d.scale * mip0.sample(u) + d.offset)
                .collect();
            let lo = disp.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = disp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            hi - lo > tolerance
        })
        .map(|(c, _)| *c)
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
        "section {s}: displacement differs across the uv seam at {} vertices, at {}{}",
        bad.len(),
        named.join(", "),
        if bad.len() > named.len() { ", .." } else { "" }
    )]
}

/// Edges of section `s` used by one triangle, by grid position.
pub(crate) fn border_edges(w: &Welded, grid: &Grid, s: usize) -> usize {
    let mut uses: HashMap<([i32; 3], [i32; 3]), u32> = HashMap::new();
    for t in w.section_triangles(s) {
        let c = t.map(|v| grid.cells[v as usize]);
        for i in 0..3 {
            let (a, b) = (c[i], c[(i + 1) % 3]);
            *uses.entry(if a < b { (a, b) } else { (b, a) }).or_default() += 1;
        }
    }
    uses.values().filter(|&&n| n == 1).count()
}
