//! meshoptimizer's `clodBuild` (demo/clusterlod.h), ported so each level's groups
//! simplify in parallel. Groups of one level share only locked vertices, so each reads the
//! positions it needs and moves only its own vertices when it dilates borders; groups come
//! out in the order the original emits them, with the same numbering. Unlike the original,
//! level 0 clusters each section apart, unless the triangles make one cluster, the sloppy
//! fallback can keep open borders ([`Mesh::keep_borders`]), and a corner of a simplified
//! triangle takes a wedge with the attributes of the surface the triangle stands for and a
//! normal fitted to it ([`lod::fit`]).

use std::borrow::Cow;

use meshopt::{SimplifyOptions, SimplifyVertex, ffi};

use super::dag::{Uvs, uv_bits};
use super::lod;
use crate::par::prelude::*;
use orr_math::Vec3;
use rustc_hash::{FxHashMap, FxHashSet};

/// A sphere and the simplification error of the geometry it bounds.
#[derive(Copy, Clone)]
pub(crate) struct Bounds {
    pub(super) center: [f32; 3],
    pub(super) radius: f32,
    pub(super) error: f32,
}

/// One cluster: its corners in `indices[index_offset..][..index_count]`.
#[derive(Copy, Clone)]
pub(crate) struct Cluster {
    index_offset: usize,
    index_count: usize,
    /// The group whose simplification made it, or -1.
    pub(super) refined: i32,
    bounds: Bounds,
}

impl Cluster {
    pub(super) fn corners<'a>(&self, indices: &'a [u32]) -> &'a [u32] {
        &indices[self.index_offset..self.index_offset + self.index_count]
    }
}

/// The mesh `build` reads besides its positions: per vertex its normal, 3 floats the
/// simplifier weighs by `weights`, its lock bits, the first vertex at its position, as
/// `meshopt_generatePositionRemap` gives it, and its section, which a triangle takes from its
/// first corner.
pub(super) struct Mesh<'a> {
    pub(super) indices: &'a [u32],
    pub(super) normals: Vec<f32>,
    /// Per vertex, its normal's class ([`lod::Welded::classes`]) and its uvs.
    pub(super) classes: Vec<u32>,
    pub(super) uvs: Vec<Uvs>,
    pub(crate) weights: &'a [f32],
    pub(super) lock: Vec<u8>,
    pub(super) remap: &'a [u32],
    pub(super) sections: &'a [u16],
    /// Per vertex, the vertex it stands for in simplification, where some stand for others:
    /// a corner an earlier build gave another normal or attributes ([`lod::fit`]).
    pub(super) base: Option<Vec<u32>>,
    /// The open borders stay where `lock` leaves them: the sloppy fallback, which does not
    /// keep to border loops, locks every vertex on one.
    pub(super) keep_borders: bool,
}

/// The mesh vertices each vertex of `build` takes after: a mesh vertex itself, and a wedge
/// the vertex that places it, by position and skin, and the one that shades it, by its
/// attributes but its normal, which is its own.
pub(super) struct Sources {
    mesh: u32,
    /// Per wedge, its shading and placing mesh vertices.
    wedges: Vec<[u32; 2]>,
}

impl Sources {
    fn of(&self, v: u32, i: usize) -> u32 {
        v.checked_sub(self.mesh)
            .map_or(v, |w| self.wedges[w as usize][i])
    }

    pub(super) fn shading(&self, v: u32) -> u32 {
        self.of(v, 0)
    }

    pub(super) fn placing(&self, v: u32) -> u32 {
        self.of(v, 1)
    }
}

/// What `build` keeps per vertex: the mesh's vertices, then the wedges it adds, each placed as
/// one vertex is and shaded as another, with a normal and uvs of its own ([`lod::fit`]).
struct Vertices {
    positions: Vec<f32>,
    normals: Vec<f32>,
    /// Per vertex, its normal's class and its uvs.
    classes: Vec<u32>,
    uvs: Vec<Uvs>,
    /// The mesh's lock bits.
    lock: Vec<u8>,
    remap: Vec<u32>,
    sources: Sources,
    /// The mesh vertex each vertex stands for in simplification: a wedge, the one the vertex
    /// it is placed as does.
    base: Vec<u32>,
}

impl Vertices {
    /// Adds a vertex placed as `at` is, by its position and lock bits, shaded as `from` is,
    /// with its normal's class, and with normal `normal` and uvs `uvs`; returns it.
    fn wedge(&mut self, at: u32, from: u32, normal: [f32; 3], uvs: Uvs) -> u32 {
        let new = self.remap.len() as u32;
        let s = &mut self.sources;
        s.wedges.push([s.shading(from), s.placing(at)]);
        let at = at as usize;
        self.positions.extend_from_within(3 * at..3 * at + 3);
        self.normals.extend_from_slice(&normal);
        self.classes.push(self.classes[from as usize]);
        self.uvs.push(uvs);
        self.lock.push(self.lock[at]);
        self.remap.push(self.remap[at]);
        self.base.push(self.base[at]);
        new
    }
}

/// Positions that groups of one level write at disjoint unlocked vertices.
struct Positions(*mut f32, usize);

// SAFETY: groups of one level write only vertices no other group of the level reads (their
// unlocked vertices); every other access is a read.
unsafe impl Sync for Positions {}

impl Positions {
    fn read(&self) -> &[f32] {
        // SAFETY: see the Sync impl.
        unsafe { std::slice::from_raw_parts(self.0, self.1) }
    }

    /// Vertex `v`, read through the pointer so it sees the calling group's writes.
    fn get(&self, v: usize) -> [f32; 3] {
        assert!(3 * v + 3 <= self.1, "vertex in range");
        // SAFETY: in range; see the Sync impl.
        unsafe { std::ptr::read(self.0.add(3 * v).cast::<[f32; 3]>()) }
    }

    /// # Safety
    /// `v` is an unlocked vertex of the calling group.
    unsafe fn write(&self, v: usize, p: [f32; 3]) {
        // SAFETY: the caller's vertex is its alone this level.
        unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), self.0.add(3 * v), 3) };
    }
}

const LOCK: u8 = SimplifyVertex::Lock.bits();
const PROTECT: u8 = SimplifyVertex::Protect.bits();

fn bounds_compute(positions: &[f32], indices: &[u32], error: f32) -> Bounds {
    // SAFETY: `indices` refer to vertices of `positions`.
    let b = unsafe {
        ffi::meshopt_computeClusterBounds(
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr(),
            positions.len() / 3,
            12,
        )
    };
    Bounds {
        center: b.center,
        radius: b.radius,
        error,
    }
}

fn bounds_merge(clusters: &[Cluster]) -> Bounds {
    let stride = size_of::<Cluster>();
    // SAFETY: each cluster's center and radius, `stride` bytes apart.
    let m = unsafe {
        ffi::meshopt_computeSphereBounds(
            clusters[0].bounds.center.as_ptr(),
            clusters.len(),
            stride,
            &raw const clusters[0].bounds.radius,
            stride,
        )
    };
    let error = clusters.iter().fold(0.0f32, |e, c| e.max(c.bounds.error));
    Bounds {
        center: m.center,
        radius: m.radius,
        error,
    }
}

/// Splits `indices` into clusters and their corners, in the order meshoptimizer emits them.
/// With `sections`, each section's triangles cluster apart and in parallel, sections
/// ascending.
fn clusterize(
    c: &ffi::clodConfig,
    positions: &[f32],
    indices: Cow<'_, [u32]>,
    sections: Option<&[u16]>,
    refined: i32,
) -> (Vec<Cluster>, Vec<u32>) {
    let _z = crate::profile::zone!("clod.clusterize");
    let tris = indices.as_chunks::<3>().0;
    let section = |t: &[u32; 3]| sections.map_or(0, |s| s[t[0] as usize]);
    let (lo, hi) = match sections {
        Some(_) => tris.iter().fold((u16::MAX, 0), |(lo, hi), t| {
            let s = section(t);
            (lo.min(s), hi.max(s))
        }),
        None => (0, 0),
    };
    // Each section's corners, sections ascending, which become its clusters' corners in
    // place: the clusters of a section hold every one of its triangles once.
    let (mut corners, ends) = if lo >= hi {
        let n = indices.len();
        (indices.into_owned(), vec![n])
    } else {
        let mut ends = vec![0usize; usize::from(hi - lo) + 1];
        for t in tris {
            ends[usize::from(section(t) - lo)] += 3;
        }
        let mut at = 0;
        for e in &mut ends {
            at += *e;
            *e = at - *e;
        }
        let mut corners = vec![0u32; indices.len()];
        for t in tris {
            let e = &mut ends[usize::from(section(t) - lo)];
            corners[*e..*e + 3].copy_from_slice(t);
            *e += 3;
        }
        ends.dedup();
        (corners, ends)
    };
    let mut rest = corners.as_mut_slice();
    let mut parts = Vec::with_capacity(ends.len());
    let mut start = 0;
    for &end in &ends {
        let (part, tail) = rest.split_at_mut(end - start);
        parts.push((start, part));
        rest = tail;
        start = end;
    }
    let cluster = |(start, part): (usize, &mut [u32])| {
        let mut clusters = meshlets(c, positions, part, refined);
        for k in &mut clusters {
            k.index_offset += start;
        }
        clusters
    };
    let clusters = if parts.len() > 1 {
        let each: Vec<Vec<Cluster>> = parts.into_par_iter().map(cluster).collect();
        each.concat()
    } else {
        parts.into_iter().flat_map(cluster).collect()
    };
    (clusters, corners)
}

/// Clusters `corners`, over vertices of `positions`, and writes their corners over it in
/// cluster order; returns the clusters, their offsets into `corners`.
fn meshlets(
    c: &ffi::clodConfig,
    positions: &[f32],
    corners: &mut [u32],
    refined: i32,
) -> Vec<Cluster> {
    // meshoptimizer sizes scratch by the vertex count, so it sees only the vertices from
    // the first `corners` uses to the last, or, where those span more vertices than there are
    // corners, as wedges past the mesh's vertices make them, only the vertices it uses, in
    // order.
    let (Some(&first), Some(&last)) = (corners.iter().min(), corners.iter().max()) else {
        return Vec::new();
    };
    let used: Option<Vec<u32>> = ((last - first) as usize >= corners.len()).then(|| {
        let mut used = corners.to_vec();
        used.sort_unstable();
        used.dedup();
        used
    });
    let compact: Vec<f32>;
    let positions = match &used {
        Some(used) => {
            for v in corners.iter_mut() {
                *v = used.binary_search(v).expect("a used vertex") as u32;
            }
            compact = used
                .iter()
                .flat_map(|&v| {
                    positions[3 * v as usize..3 * v as usize + 3]
                        .iter()
                        .copied()
                })
                .collect();
            &compact[..]
        }
        None => {
            for v in corners.iter_mut() {
                *v -= first;
            }
            &positions[first as usize * 3..last as usize * 3 + 3]
        }
    };
    let vertex = |v: u32| used.as_ref().map_or(first + v, |used| used[v as usize]);
    let indices = &*corners;
    // SAFETY: plain bound query.
    let max =
        unsafe { ffi::meshopt_buildMeshletsBound(indices.len(), c.max_vertices, c.min_triangles) };
    let mut meshlets = vec![
        ffi::meshopt_Meshlet {
            vertex_offset: 0,
            triangle_offset: 0,
            vertex_count: 0,
            triangle_count: 0
        };
        max
    ];
    let mut vertices = vec![0u32; indices.len()];
    let mut triangles = vec![0u8; indices.len()];
    // SAFETY: the buffers hold the worst case the bound and index count give.
    let count = unsafe {
        if c.cluster_spatial {
            ffi::meshopt_buildMeshletsSpatial(
                meshlets.as_mut_ptr(),
                vertices.as_mut_ptr(),
                triangles.as_mut_ptr(),
                indices.as_ptr(),
                indices.len(),
                positions.as_ptr(),
                positions.len() / 3,
                12,
                c.max_vertices,
                c.min_triangles,
                c.max_triangles,
                c.cluster_fill_weight,
            )
        } else {
            ffi::meshopt_buildMeshletsFlex(
                meshlets.as_mut_ptr(),
                vertices.as_mut_ptr(),
                triangles.as_mut_ptr(),
                indices.as_ptr(),
                indices.len(),
                positions.as_ptr(),
                positions.len() / 3,
                12,
                c.max_vertices,
                c.min_triangles,
                c.max_triangles,
                0.0,
                c.cluster_split_factor,
            )
        }
    };
    let mut clusters = Vec::with_capacity(count);
    let mut at = 0;
    for m in &meshlets[..count] {
        let (vo, to) = (m.vertex_offset as usize, m.triangle_offset as usize);
        let n = m.triangle_count as usize * 3;
        clusters.push(Cluster {
            index_offset: at,
            index_count: n,
            refined,
            bounds: Bounds {
                center: [0.0; 3],
                radius: 0.0,
                error: 0.0,
            },
        });
        for (o, &t) in corners[at..at + n].iter_mut().zip(&triangles[to..to + n]) {
            *o = vertex(vertices[vo + usize::from(t)]);
        }
        at += n;
    }
    assert_eq!(at, corners.len(), "clusters hold every triangle once");
    clusters
}

/// Partitions `clusters` into groups in place; returns the group offsets.
fn partition(
    clusters: &mut Vec<Cluster>,
    c: &ffi::clodConfig,
    positions: &[f32],
    indices: &[u32],
    remap: &[u32],
) -> Vec<u32> {
    let _z = crate::profile::zone!("clod.partition");
    if clusters.len() <= c.partition_size {
        return vec![0, clusters.len() as u32];
    }
    let counts: Vec<u32> = clusters.iter().map(|k| k.index_count as u32).collect();
    let flat: Vec<u32> = clusters
        .iter()
        .flat_map(|k| {
            indices[k.index_offset..k.index_offset + k.index_count]
                .iter()
                .map(|&v| remap[v as usize])
        })
        .collect();
    let mut part = vec![0u32; clusters.len()];
    // SAFETY: `flat` holds `counts` corners per cluster, all below `remap.len()`.
    let parts = unsafe {
        ffi::meshopt_partitionClusters(
            part.as_mut_ptr(),
            flat.as_ptr(),
            flat.len(),
            counts.as_ptr(),
            counts.len(),
            if c.partition_spatial {
                positions.as_ptr()
            } else {
                std::ptr::null()
            },
            remap.len(),
            12,
            c.partition_size,
        )
    };
    if c.partition_sort {
        let mut points = vec![0.0f32; parts * 3];
        for (k, &p) in clusters.iter().zip(&part) {
            points[p as usize * 3..p as usize * 3 + 3].copy_from_slice(&k.bounds.center);
        }
        let mut order = vec![0u32; parts];
        // SAFETY: one point per partition.
        unsafe { ffi::meshopt_spatialSortRemap(order.as_mut_ptr(), points.as_ptr(), parts, 12) };
        for p in &mut part {
            *p = order[*p as usize];
        }
    }
    let mut offsets = vec![0u32; parts + 1];
    for &p in &part {
        offsets[p as usize + 1] += 1;
    }
    for i in 1..=parts {
        offsets[i] += offsets[i - 1];
    }
    let mut write = offsets.clone();
    let mut sorted = clusters.clone();
    for (k, &p) in clusters.iter().zip(&part) {
        sorted[write[p as usize] as usize] = *k;
        write[p as usize] += 1;
    }
    *clusters = sorted;
    offsets
}

/// [`lock_boundary`]'s mark on a position some group of the level has used.
const USED: u8 = 0x80;

/// Locks the positions groups share, and keeps protect bits and the mesh's own locks: each
/// vertex takes its position's lock, and the lock the mesh puts on the position's first
/// vertex, which `remap` gives. `marks` is scratch of a byte per vertex: a byte, so the
/// random access of a level's corners stays in cache.
fn lock_boundary(
    locks: &mut [u8],
    marks: &mut [u8],
    clusters: &[Cluster],
    offsets: &[u32],
    indices: &[u32],
    remap: &[u32],
    mesh_lock: &[u8],
) {
    const { assert!(USED >> 7 == LOCK) };
    marks.fill(0);
    for g in offsets.windows(2) {
        let group = &clusters[g[0] as usize..g[1] as usize];
        let corners = || group.iter().flat_map(|k| k.corners(indices));
        // A position an earlier group used is shared.
        for &v in corners() {
            let m = &mut marks[remap[v as usize] as usize];
            *m |= *m >> 7;
        }
        for &v in corners() {
            marks[remap[v as usize] as usize] |= USED;
        }
    }
    for (i, l) in locks.iter_mut().enumerate() {
        let r = remap[i] as usize;
        *l = (marks[r] & LOCK) | (mesh_lock[r] & LOCK) | (*l & PROTECT) | mesh_lock[i];
    }
}

fn simplify_fallback(
    positions: &[f32],
    indices: &[u32],
    locks: &[u8],
    keep_borders: Option<&[u32]>,
    target: usize,
    error: &mut f32,
) -> Vec<u32> {
    let subset: Vec<f32> = indices
        .iter()
        .flat_map(|&v| {
            let v = v as usize * 3;
            [positions[v], positions[v + 1], positions[v + 2]]
        })
        .collect();
    let mut subset_locks: Vec<u8> = indices.iter().map(|&v| locks[v as usize]).collect();
    // `keep_borders` is the position remap when open borders stay: every corner at a position
    // on an open edge of the group locks, as sloppy simplification moves any vertex it does
    // not lock. A triangle with two corners at one position has no area and closes no edge.
    if let Some(remap) = keep_borders {
        let tris: Vec<[u32; 3]> = indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| t.map(|v| remap[v as usize]))
            .filter(|p| p[0] != p[1] && p[1] != p[2] && p[2] != p[0])
            .collect();
        let edge = |p: &[u32; 3], e: usize| (p[e], p[(e + 1) % 3]);
        let edges: FxHashSet<(u32, u32)> = tris
            .iter()
            .flat_map(|t| (0..3).map(move |e| edge(t, e)))
            .collect();
        let mut border = FxHashSet::default();
        for t in &tris {
            for e in 0..3 {
                let (a, b) = edge(t, e);
                if !edges.contains(&(b, a)) {
                    border.extend([a, b]);
                }
            }
        }
        for (l, &v) in subset_locks.iter_mut().zip(indices) {
            if border.contains(&remap[v as usize]) {
                *l |= LOCK;
            }
        }
    }
    let mut lod: Vec<u32> = (0..indices.len() as u32).collect();
    // SAFETY: the subset holds one vertex per corner and `lod` indexes it; sloppy
    // simplification may run in place.
    let n = unsafe {
        ffi::meshopt_simplifySloppy(
            lod.as_mut_ptr(),
            lod.as_ptr(),
            lod.len(),
            subset.as_ptr(),
            indices.len(),
            12,
            subset_locks.as_ptr(),
            target,
            f32::MAX,
            error,
        )
    };
    lod.truncate(n);
    // SAFETY: as above.
    *error *= unsafe { ffi::meshopt_simplifyScale(subset.as_ptr(), indices.len(), 12) };
    lod.iter().map(|&i| indices[i as usize]).collect()
}

/// `indices` simplified toward `target` indices, with the error reached in `error` and
/// whether the sloppy fallback made them in `sloppy`.
#[allow(clippy::too_many_arguments)]
fn simplify(
    c: &ffi::clodConfig,
    weights: &[f32],
    keep_borders: bool,
    normals: &[f32],
    remap: &[u32],
    positions: &[f32],
    indices: &[u32],
    locks: &[u8],
    target: usize,
    error: &mut f32,
    sloppy: &mut bool,
) -> Vec<u32> {
    let _z = crate::profile::zone!("clod.simplify");
    let mut options = SimplifyOptions::Sparse | SimplifyOptions::ErrorAbsolute;
    options.set(SimplifyOptions::ErrorClamped, c.simplify_error_clamped);
    options.set(SimplifyOptions::Permissive, c.simplify_permissive);
    options.set(SimplifyOptions::Regularize, c.simplify_regularize);
    options.set(SimplifyOptions::PreserveFolds, c.simplify_preserve_folds);
    let run = |options: SimplifyOptions, error: &mut f32| {
        let mut lod = vec![0u32; indices.len()];
        // SAFETY: positions, attributes and locks cover every vertex `indices` refer to.
        let n = unsafe {
            ffi::meshopt_simplifyWithAttributes(
                lod.as_mut_ptr(),
                indices.as_ptr(),
                indices.len(),
                positions.as_ptr(),
                positions.len() / 3,
                12,
                normals.as_ptr(),
                12,
                weights.as_ptr(),
                weights.len(),
                locks.as_ptr(),
                target,
                f32::MAX,
                options.bits(),
                error,
            )
        };
        lod.truncate(n);
        lod
    };
    let mut lod = run(options, error);
    if lod.len() > target && c.simplify_fallback_permissive && !c.simplify_permissive {
        lod = run(options | SimplifyOptions::Permissive, error);
    }
    if lod.len() > target && c.simplify_fallback_sloppy {
        let keep = keep_borders.then_some(remap);
        lod = simplify_fallback(positions, indices, locks, keep, target, error);
        *error *= c.simplify_error_factor_sloppy;
        *sloppy = true;
    }
    if lod.is_empty() && !indices.is_empty() {
        lod.extend_from_slice(&indices[..3]);
    }
    lod
}

/// Area of the triangles of `indices` with an unlocked open edge.
fn boundary_area(
    positions: &[f32],
    indices: &[u32],
    locks: &[u8],
    remap: &[u32],
    edges: &mut FxHashSet<u64>,
) -> f32 {
    edges.clear();
    let edge = |t: &[u32], e: usize| {
        (
            remap[t[e] as usize],
            remap[t[if e == 2 { 0 } else { e + 1 }] as usize],
        )
    };
    for t in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let (a, b) = edge(t, e);
            edges.insert(u64::from(a) << 32 | u64::from(b));
        }
    }
    let mut area = 0.0f32;
    for t in indices.as_chunks::<3>().0 {
        let border = (0..3).any(|e| {
            let (a, b) = edge(t, e);
            locks[a as usize] & locks[b as usize] & LOCK == 0
                && !edges.contains(&(u64::from(b) << 32 | u64::from(a)))
        });
        if !border {
            continue;
        }
        let p = |k: usize| Vec3::from_slice(&positions[t[k] as usize * 3..]);
        let (va, vb, vc) = (p(0), p(1), p(2));
        area += (vb - va).cross(vc - va).length() * 0.5;
    }
    area
}

/// Pushes the open, unlocked borders of `simplified` out so it keeps `merged`'s border
/// area, writing the group's own vertices; returns the positions it moves.
fn dilate_borders(
    positions: &Positions,
    merged: &[u32],
    simplified: &[u32],
    locks: &[u8],
    remap: &[u32],
    bounds: &mut Bounds,
) -> FxHashSet<u32> {
    let _z = crate::profile::zone!("clod.dilate");
    let mut moved = FxHashSet::default();
    let mut edges = FxHashSet::default();
    let old_area = boundary_area(positions.read(), merged, locks, remap, &mut edges);
    if old_area == 0.0 {
        return moved;
    }
    let new_area = boundary_area(positions.read(), simplified, locks, remap, &mut edges);
    if new_area >= old_area || new_area < old_area / 4.0 {
        return moved;
    }
    let mut offsets: FxHashMap<u32, [f32; 4]> = FxHashMap::default();
    let mut perimeter = 0.0f32;
    let at = |v: u32| positions.get(v as usize);
    for t in simplified.as_chunks::<3>().0 {
        for e in 0..3 {
            let (ia, ib, ic) = (t[e], t[if e == 2 { 0 } else { e + 1 }], t[(e + 2) % 3]);
            let (a, b) = (remap[ia as usize], remap[ib as usize]);
            if locks[a as usize] & locks[b as usize] & LOCK != 0
                || edges.contains(&(u64::from(b) << 32 | u64::from(a)))
            {
                continue;
            }
            let (va, vb, vc) = (Vec3::from(at(ia)), Vec3::from(at(ib)), Vec3::from(at(ic)));
            let ev = vb - va;
            let el = ev.length();
            let cv = va - vc;
            let cp = cv.dot(ev) / if el > 0.0 { el * el } else { 1.0 };
            let nv = cv - ev * cp;
            let nl = nv.length();
            let ns = if nl > 0.0 { el / nl } else { 0.0 };
            for v in [a, b] {
                if locks[v as usize] & LOCK == 0 {
                    let o = offsets.entry(v).or_insert([0.0; 4]);
                    o[0] += nv[0] * ns;
                    o[1] += nv[1] * ns;
                    o[2] += nv[2] * ns;
                    o[3] += el;
                }
            }
            perimeter += el;
        }
    }
    let distance = if perimeter > 0.0 {
        (old_area - new_area) / perimeter
    } else {
        0.0
    };
    for &i in simplified {
        let r = remap[i as usize];
        if locks[r as usize] & LOCK != 0 {
            continue;
        }
        let mut vr = at(r);
        if let Some(n) = offsets.get_mut(&r)
            && n[3] > 0.0
        {
            let (nx, ny, nz) = (n[0] / n[3], n[1] / n[3], n[2] / n[3]);
            let nn = nx * nx + ny * ny + nz * nz;
            let ns = distance / if nn > 0.15 { nn } else { 0.15 };
            vr = [vr[0] + nx * ns, vr[1] + ny * ns, vr[2] + nz * ns];
            *n = [0.0; 4];
            // SAFETY: `r` is unlocked, so the group's own.
            unsafe { positions.write(r as usize, vr) };
            moved.insert(r);
            let d = Vec3::from(vr).distance(Vec3::from(bounds.center));
            bounds.radius = bounds.radius.max(d);
        }
        // SAFETY: `i` shares `r`'s position, so it is unlocked and the group's own too.
        unsafe { positions.write(i as usize, vr) };
    }
    moved
}

/// What a group's emission gets: its depth and bounds, its clusters, their corners, the
/// positions as they are when it is emitted, the mesh vertices each vertex takes after, and
/// the one each stands for in simplification.
pub(super) struct Emitted<'a> {
    pub(super) depth: u32,
    pub(crate) bounds: Bounds,
    pub(crate) clusters: &'a [Cluster],
    pub(super) indices: &'a [u32],
    pub(super) positions: &'a [f32],
    pub(super) normals: &'a [f32],
    pub(super) classes: &'a [u32],
    pub(super) uvs: &'a [Uvs],
    pub(super) sources: &'a Sources,
    pub(super) base: &'a [u32],
}

/// A group simplified and emitted: its triangles, their bounds, and what its corners take in
/// place of their vertex ([`lod::fit`]).
type Simplified = (Vec<u32>, Bounds, lod::Fit);

/// Builds the DAG of `mesh` at `positions` under `c`, calling `emit` once per group with its
/// index fixed by emission order, and returns the emitted values in that order.
pub(super) fn build<G: Send>(
    c: &ffi::clodConfig,
    mesh: Mesh<'_>,
    positions: Vec<f32>,
    emit: impl Fn(Emitted<'_>) -> G + Sync,
) -> Vec<G> {
    debug_assert!(
        !c.optimize_clusters && !c.optimize_bounds && c.simplify_error_edge_limit == 0.0,
        "clusterlod options the port leaves out"
    );
    let Mesh {
        indices,
        normals,
        classes,
        uvs,
        weights,
        lock,
        remap,
        sections,
        base,
        keep_borders,
    } = mesh;
    let vertex_count = positions.len() / 3;
    assert_eq!(
        remap.len(),
        vertex_count,
        "a position remap entry per vertex"
    );
    let mut vx = Vertices {
        positions,
        normals,
        classes,
        uvs,
        lock,
        remap: remap.to_vec(),
        sources: Sources {
            mesh: vertex_count as u32,
            wedges: Vec::new(),
        },
        base: base.unwrap_or_else(|| (0..vertex_count as u32).collect()),
    };
    assert_eq!(vx.base.len(), vertex_count, "a base vertex per vertex");
    let (mut locks, mut marks) = (Vec::new(), Vec::new());
    let mut out = Vec::new();

    // Level 0 clusters each section apart, unless the triangles make one cluster: that one is
    // the root, as in clodBuild.
    let (mut clusters, mut cluster_indices) =
        clusterize(c, &vx.positions, indices.into(), Some(sections), -1);
    if clusters.len() > 1 && indices.len() <= c.max_triangles * 3 {
        let one = clusterize(c, &vx.positions, indices.into(), None, -1);
        if one.0.len() == 1 {
            (clusters, cluster_indices) = one;
        }
    }
    {
        let _z = crate::profile::zone!("clod.bounds");
        clusters.par_iter_mut().for_each(|k| {
            k.bounds = bounds_compute(&vx.positions, k.corners(&cluster_indices), 0.0);
        });
    }

    let mut depth = 0u32;
    while clusters.len() > 1 {
        let _z = crate::profile::zone!("clod.level");
        let offsets = partition(&mut clusters, c, &vx.positions, &cluster_indices, &vx.remap);
        let lock = crate::profile::zone!("clod.lock");
        locks.resize(vx.remap.len(), 0);
        marks.resize(vx.remap.len(), 0);
        lock_boundary(
            &mut locks,
            &mut marks,
            &clusters,
            &offsets,
            &cluster_indices,
            &vx.remap,
            &vx.lock,
        );
        drop(lock);
        let base = out.len() as i32;
        let positions = Positions(vx.positions.as_mut_ptr(), vx.positions.len());
        let (normals, classes, uvs) = (&vx.normals, &vx.classes, &vx.uvs);
        let (remap, sources, bases) = (&vx.remap, &vx.sources, &vx.base);
        let groups: Vec<(G, Option<Simplified>)> = offsets
            .par_windows(2)
            .map(|w| {
                let group = &clusters[w[0] as usize..w[1] as usize];
                let merged: Vec<u32> = group
                    .iter()
                    .flat_map(|k| k.corners(&cluster_indices))
                    .copied()
                    .collect();
                // Simplification sees the vertices the wedges stand for, so a wedge constrains
                // no collapse.
                let merged_bases: Vec<u32> = merged.iter().map(|&v| bases[v as usize]).collect();
                let target = ((merged.len() / 3) as f32 * c.simplify_ratio) as usize * 3;
                let mut bounds = bounds_merge(group);
                let mut error = 0.0f32;
                let mut sloppy = false;
                let simplified = simplify(
                    c,
                    weights,
                    keep_borders,
                    normals,
                    remap,
                    positions.read(),
                    &merged_bases,
                    &locks,
                    target,
                    &mut error,
                    &mut sloppy,
                );
                let emitted = |bounds: Bounds| {
                    let _z = crate::profile::zone!("clod.emit");
                    emit(Emitted {
                        depth,
                        bounds,
                        clusters: group,
                        indices: &cluster_indices,
                        positions: positions.read(),
                        normals,
                        classes,
                        uvs,
                        sources,
                        base: bases,
                    })
                };
                if simplified.len() as f32 > merged.len() as f32 * c.simplify_threshold {
                    bounds.error = f32::MAX;
                    return (emitted(bounds), None);
                }
                bounds.error = (bounds.error * c.simplify_error_merge_previous).max(error)
                    + error * c.simplify_error_merge_additive;
                let g = emitted(bounds);
                let moved = if c.simplify_dilate_borders {
                    dilate_borders(
                        &positions,
                        &merged_bases,
                        &simplified,
                        &locks,
                        remap,
                        &mut bounds,
                    )
                } else {
                    FxHashSet::default()
                };
                let _z = crate::profile::zone!("clod.fit");
                let at = |v: u32| Vec3::from(positions.get(v as usize)).as_dvec3();
                let normal = |v: u32| Vec3::from_slice(&normals[3 * v as usize..]).as_dvec3();
                let class = |v: u32| classes[v as usize];
                let uv = |v: u32| uvs[v as usize];
                let section = |v: u32| sections[sources.shading(v) as usize];
                let position = |v: u32| remap[v as usize];
                let locked = |v: u32| locks[v as usize] & LOCK != 0;
                let moved = |v: u32| moved.contains(&remap[v as usize]);
                let vertices = lod::FitVertices {
                    at: &at,
                    normal: &normal,
                    class: &class,
                    uvs: &uv,
                    section: &section,
                    position: &position,
                    base: &|v| bases[v as usize],
                    locked: &locked,
                    moved: &moved,
                };
                let fit = lod::fit(&simplified, &merged, &merged_bases, sloppy, &vertices);
                (g, Some((simplified, bounds, fit)))
            })
            .collect();
        let wedges = crate::profile::zone!("clod.wedges");
        // A wedge per group, vertex, attribute source and normal, also where a vertex at the
        // corner's position has the attributes and normal it takes: unlike that vertex, the
        // wedge stands for the corner's vertex in simplification.
        let mut next = Vec::new();
        for (gi, (g, simplified)) in groups.into_iter().enumerate() {
            out.push(g);
            let Some((mut simplified, bounds, fit)) = simplified else {
                continue;
            };
            let mut made: FxHashMap<(u32, u32, u32, [u32; 4]), u32> = FxHashMap::default();
            for (k, take) in fit.corners {
                let v = simplified[k];
                simplified[k] = match take {
                    lod::Take::Input(u) => u,
                    lod::Take::Fit {
                        attributes,
                        normal,
                        uvs,
                    } => *made
                        .entry((v, attributes, normal, uv_bits(uvs)))
                        .or_insert_with(|| {
                            vx.wedge(v, attributes, fit.normals[normal as usize], uvs)
                        }),
                };
            }
            next.push((base + gi as i32, simplified, bounds));
        }
        drop(wedges);
        let next: Vec<(Vec<Cluster>, Vec<u32>)> = next
            .into_par_iter()
            .map(|(refined, simplified, bounds)| {
                let (mut next, next_indices) =
                    clusterize(c, &vx.positions, simplified.into(), None, refined);
                for k in &mut next {
                    k.bounds = bounds;
                }
                (next, next_indices)
            })
            .collect();
        let _gather = crate::profile::zone!("clod.gather");
        let (mut pending, mut pending_indices) = (Vec::new(), Vec::new());
        for (next, next_indices) in next {
            let shift = pending_indices.len();
            pending.extend(next.into_iter().map(|mut k| {
                k.index_offset += shift;
                k
            }));
            pending_indices.extend(next_indices);
        }
        clusters = pending;
        cluster_indices = pending_indices;
        depth += 1;
    }
    if let [k] = clusters.as_slice() {
        let mut bounds = k.bounds;
        bounds.error = f32::MAX;
        out.push(emit(Emitted {
            depth,
            bounds,
            clusters: &clusters,
            indices: &cluster_indices,
            positions: &vx.positions,
            normals: &vx.normals,
            classes: &vx.classes,
            uvs: &vx.uvs,
            sources: &vx.sources,
            base: &vx.base,
        }));
    }
    out
}
