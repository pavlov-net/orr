//! meshoptimizer's `clodBuild` (demo/clusterlod.h), ported so each level's groups
//! simplify in parallel. Groups of one level share only locked vertices, so each reads the
//! positions it needs and moves only its own vertices when it dilates borders; groups come
//! out in the order the original emits them, with the same numbering.

use meshopt::{SimplifyOptions, SimplifyVertex, ffi};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// A sphere and the simplification error of the geometry it bounds.
#[derive(Copy, Clone)]
pub(crate) struct Bounds {
    pub center: [f32; 3],
    pub radius: f32,
    pub error: f32,
}

/// One cluster: its corners in `indices[index_offset..][..index_count]`.
#[derive(Copy, Clone)]
pub(crate) struct Cluster {
    index_offset: usize,
    index_count: usize,
    /// The group whose simplification made it, or -1.
    pub refined: i32,
    bounds: Bounds,
}

impl Cluster {
    pub(crate) fn corners<'a>(&self, indices: &'a [u32]) -> &'a [u32] {
        &indices[self.index_offset..self.index_offset + self.index_count]
    }
}

/// The mesh `build` reads besides its positions: 3 floats of attributes per vertex.
pub(crate) struct Mesh<'a> {
    pub indices: &'a [u32],
    pub attributes: &'a [f32],
    pub weights: &'a [f32],
    pub lock: &'a [u8],
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

/// Splits `indices` into clusters appended to `clusters`, their corners to `out`.
fn clusterize(
    clusters: &mut Vec<Cluster>,
    out: &mut Vec<u32>,
    c: &ffi::clodConfig,
    positions: &[f32],
    indices: &[u32],
    refined: i32,
) {
    let _z = crate::profile::zone!("clod.clusterize");
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
    for m in &meshlets[..count] {
        let (vo, to) = (m.vertex_offset as usize, m.triangle_offset as usize);
        let n = m.triangle_count as usize * 3;
        clusters.push(Cluster {
            index_offset: out.len(),
            index_count: n,
            refined,
            bounds: Bounds {
                center: [0.0; 3],
                radius: 0.0,
                error: 0.0,
            },
        });
        out.extend(
            triangles[to..to + n]
                .iter()
                .map(|&t| vertices[vo + usize::from(t)]),
        );
    }
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

/// Locks the positions groups share, and keeps protect bits and the mesh's own locks.
fn lock_boundary(
    locks: &mut [u8],
    clusters: &[Cluster],
    offsets: &[u32],
    indices: &[u32],
    remap: &[u32],
    mesh_lock: &[u8],
) {
    for l in locks.iter_mut() {
        *l &= !(LOCK | 0x80);
    }
    for g in offsets.windows(2) {
        let group = &clusters[g[0] as usize..g[1] as usize];
        let corners = || {
            group
                .iter()
                .flat_map(|k| &indices[k.index_offset..k.index_offset + k.index_count])
        };
        for &v in corners() {
            let r = remap[v as usize] as usize;
            locks[r] |= locks[r] >> 7;
        }
        for &v in corners() {
            locks[remap[v as usize] as usize] |= 0x80;
        }
    }
    for i in 0..locks.len() {
        let r = remap[i] as usize;
        locks[i] = (locks[r] & LOCK) | (locks[i] & PROTECT) | mesh_lock[i];
    }
}

fn simplify_fallback(
    positions: &[f32],
    indices: &[u32],
    locks: &[u8],
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
    let subset_locks: Vec<u8> = indices.iter().map(|&v| locks[v as usize]).collect();
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

fn simplify(
    c: &ffi::clodConfig,
    mesh: &Mesh<'_>,
    positions: &[f32],
    indices: &[u32],
    locks: &[u8],
    target: usize,
    error: &mut f32,
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
                mesh.attributes.as_ptr(),
                12,
                mesh.weights.as_ptr(),
                mesh.weights.len(),
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
        lod = simplify_fallback(positions, indices, locks, target, error);
        *error *= c.simplify_error_factor_sloppy;
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
    for t in indices.chunks_exact(3) {
        for e in 0..3 {
            let (a, b) = edge(t, e);
            edges.insert(u64::from(a) << 32 | u64::from(b));
        }
    }
    let mut area = 0.0f32;
    for t in indices.chunks_exact(3) {
        let border = (0..3).any(|e| {
            let (a, b) = edge(t, e);
            locks[a as usize] & locks[b as usize] & LOCK == 0
                && !edges.contains(&(u64::from(b) << 32 | u64::from(a)))
        });
        if !border {
            continue;
        }
        let p = |k: usize| {
            let v = t[k] as usize * 3;
            [positions[v], positions[v + 1], positions[v + 2]]
        };
        let (va, vb, vc) = (p(0), p(1), p(2));
        let ab = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
        let ac = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
        let nt = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        area += (nt[0] * nt[0] + nt[1] * nt[1] + nt[2] * nt[2]).sqrt() * 0.5;
    }
    area
}

/// Pushes the open, unlocked borders of `simplified` out so it keeps `merged`'s border
/// area, writing the group's own vertices.
fn dilate_borders(
    positions: &Positions,
    merged: &[u32],
    simplified: &[u32],
    locks: &[u8],
    remap: &[u32],
    bounds: &mut Bounds,
) {
    let _z = crate::profile::zone!("clod.dilate");
    let mut edges = FxHashSet::default();
    let old_area = boundary_area(positions.read(), merged, locks, remap, &mut edges);
    if old_area == 0.0 {
        return;
    }
    let new_area = boundary_area(positions.read(), simplified, locks, remap, &mut edges);
    if new_area >= old_area || new_area < old_area / 4.0 {
        return;
    }
    let mut offsets: FxHashMap<u32, [f32; 4]> = FxHashMap::default();
    let mut perimeter = 0.0f32;
    let at = |v: u32| positions.get(v as usize);
    for t in simplified.chunks_exact(3) {
        for e in 0..3 {
            let (ia, ib, ic) = (t[e], t[if e == 2 { 0 } else { e + 1 }], t[(e + 2) % 3]);
            let (a, b) = (remap[ia as usize], remap[ib as usize]);
            if locks[a as usize] & locks[b as usize] & LOCK != 0
                || edges.contains(&(u64::from(b) << 32 | u64::from(a)))
            {
                continue;
            }
            let (va, vb, vc) = (at(ia), at(ib), at(ic));
            let ev = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
            let el = (ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]).sqrt();
            let cv = [va[0] - vc[0], va[1] - vc[1], va[2] - vc[2]];
            let cp = (cv[0] * ev[0] + cv[1] * ev[1] + cv[2] * ev[2])
                / if el > 0.0 { el * el } else { 1.0 };
            let nv = [cv[0] - ev[0] * cp, cv[1] - ev[1] * cp, cv[2] - ev[2] * cp];
            let nl = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
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
            let d = [
                vr[0] - bounds.center[0],
                vr[1] - bounds.center[1],
                vr[2] - bounds.center[2],
            ];
            bounds.radius = bounds
                .radius
                .max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        }
        // SAFETY: `i` shares `r`'s position, so it is unlocked and the group's own too.
        unsafe { positions.write(i as usize, vr) };
    }
}

/// What a group's emission gets: its depth and bounds, its clusters, their corners and
/// the positions as they are when it is emitted.
pub(crate) struct Emitted<'a> {
    pub depth: u32,
    pub bounds: Bounds,
    pub clusters: &'a [Cluster],
    pub indices: &'a [u32],
    pub positions: &'a [f32],
}

/// Builds the DAG of `mesh` under `c`, calling `emit` once per group with its index fixed
/// by emission order, and returns the emitted values in that order.
pub(crate) fn build<G: Send>(
    c: &ffi::clodConfig,
    mesh: Mesh<'_>,
    positions: &mut [f32],
    emit: impl Fn(Emitted<'_>) -> G + Sync,
) -> Vec<G> {
    debug_assert!(
        !c.optimize_clusters && !c.optimize_bounds && c.simplify_error_edge_limit == 0.0,
        "clusterlod options the port leaves out"
    );
    let vertex_count = positions.len() / 3;
    let mut remap = vec![0u32; vertex_count];
    // SAFETY: one position per vertex.
    unsafe {
        ffi::meshopt_generatePositionRemap(
            remap.as_mut_ptr(),
            positions.as_ptr(),
            vertex_count,
            12,
        );
    }
    let mut locks = vec![0u8; vertex_count];
    let positions = Positions(positions.as_mut_ptr(), positions.len());
    let mut out = Vec::new();

    let (mut clusters, mut cluster_indices) = (Vec::new(), Vec::new());
    clusterize(
        &mut clusters,
        &mut cluster_indices,
        c,
        positions.read(),
        mesh.indices,
        -1,
    );
    clusters.par_iter_mut().for_each(|k| {
        k.bounds = bounds_compute(positions.read(), k.corners(&cluster_indices), 0.0);
    });

    let mut depth = 0u32;
    while clusters.len() > 1 {
        let _z = crate::profile::zone!("clod.level");
        let offsets = partition(&mut clusters, c, positions.read(), &cluster_indices, &remap);
        lock_boundary(
            &mut locks,
            &clusters,
            &offsets,
            &cluster_indices,
            &remap,
            mesh.lock,
        );
        let base = out.len() as i32;
        let groups: Vec<(G, Vec<Cluster>, Vec<u32>)> = offsets
            .par_windows(2)
            .enumerate()
            .map(|(gi, w)| {
                let group = &clusters[w[0] as usize..w[1] as usize];
                let merged: Vec<u32> = group
                    .iter()
                    .flat_map(|k| k.corners(&cluster_indices))
                    .copied()
                    .collect();
                let target = ((merged.len() / 3) as f32 * c.simplify_ratio) as usize * 3;
                let mut bounds = bounds_merge(group);
                let mut error = 0.0f32;
                let simplified = simplify(
                    c,
                    &mesh,
                    positions.read(),
                    &merged,
                    &locks,
                    target,
                    &mut error,
                );
                let emitted = |bounds: Bounds| {
                    let _z = crate::profile::zone!("clod.emit");
                    emit(Emitted {
                        depth,
                        bounds,
                        clusters: group,
                        indices: &cluster_indices,
                        positions: positions.read(),
                    })
                };
                if simplified.len() as f32 > merged.len() as f32 * c.simplify_threshold {
                    bounds.error = f32::MAX;
                    return (emitted(bounds), Vec::new(), Vec::new());
                }
                bounds.error = (bounds.error * c.simplify_error_merge_previous).max(error)
                    + error * c.simplify_error_merge_additive;
                let g = emitted(bounds);
                if c.simplify_dilate_borders {
                    dilate_borders(
                        &positions,
                        &merged,
                        &simplified,
                        &locks,
                        &remap,
                        &mut bounds,
                    );
                }
                let (mut next, mut next_indices) = (Vec::new(), Vec::new());
                clusterize(
                    &mut next,
                    &mut next_indices,
                    c,
                    positions.read(),
                    &simplified,
                    base + gi as i32,
                );
                for k in &mut next {
                    k.bounds = bounds;
                }
                (g, next, next_indices)
            })
            .collect();
        let (mut pending, mut pending_indices) = (Vec::new(), Vec::new());
        for (g, next, next_indices) in groups {
            out.push(g);
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
            positions: positions.read(),
        }));
    }
    out
}
