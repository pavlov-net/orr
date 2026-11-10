//! Cluster LOD DAGs through meshoptimizer's clusterlod.
//!
//! The part stage is one DAG per (part, class), a class being the sections that cluster
//! alike (the same vertex limit, and no border dilation, which moves each side of a seam on
//! its own), with every grid position another stage shares locked. Within a stage the
//! sections stay apart as attribute seams: a vertex belongs to one section, and positions
//! where sections meet are protected, so the (permissive) simplifier moves them only along
//! the border and never onto another section's vertex; the border coarsens with the rest.
//! The finest clusters each lie in one section, unless the stage is one cluster; other
//! clusters are cut by section as they are emitted, and the pieces of one cluster share its
//! group and bounds, so a cut takes all of them or none. With parts and a merge error the
//! part stage is cut at that error: groups at or over it become terminal and the clusters
//! simplified from them go. Each class then starts a merged stage from the terminal
//! clusters of all its parts at rest pose, its errors raised by the merge error. A copy
//! draws one stage or the other as a whole, so the stages need no crack-free join.
//!
//! Each DAG is `clodBuild` as ported in [`super::clod`], one level's groups in parallel.
//! Border dilation moves positions in place, so each group's clusters take their positions
//! as the group is emitted and snap them to the mesh grid.

use std::collections::BTreeMap;
use std::ops::Range;
use std::os::raw::c_int;

use crate::par::prelude::*;
use meshopt::ffi;
use orr_pack::mesh::{
    ClusterSets, Group, MeshHeader, MeshInfo, NODE_PAGE_MAX, Node, NodePage, OMM_ALIGN, PAGE_MAX,
    PageHeader, PageInfo, Section, SurfaceClass, cluster_resident_len,
};
use orr_pack::{EncodedBlob, Span, Sphere};
use rustc_hash::FxHashMap;

use super::lod::{self, Parts, Welded};
use super::{
    Built, ClusterData, Ctx, Grid, MAX_CLUSTER_CORNERS, MAX_CLUSTER_VERTICES, MeshSettings,
    page_triangles, store_page, write_page,
};

/// Sections that share a stage: the cluster vertex limit, and the section itself when its
/// borders dilate.
type Class = (u16, Option<u16>);

/// A triangle corner: its welded vertex, grid position and section.
type Corner = (u32, [i32; 3], u16);

/// One input to `clodBuild`: vertices referring to welded vertices, at grid positions, in
/// sections.
struct Input {
    welded: Vec<u32>,
    cells: Vec<[i32; 3]>,
    sections: Vec<u16>,
    lock: Vec<u8>,
    /// Per vertex, the first vertex at its position.
    remap: Vec<u32>,
    indices: Vec<u32>,
}

impl Input {
    fn with_capacity(corners: usize) -> Self {
        Self {
            welded: Vec::new(),
            cells: Vec::new(),
            sections: Vec::new(),
            lock: Vec::new(),
            remap: Vec::new(),
            indices: Vec::with_capacity(corners),
        }
    }

    /// Adds a vertex of welded vertex `v` at `cell` in `section` whose position the vertex
    /// `first` stands for, the new one when it is `None`; returns it.
    fn push(&mut self, v: u32, cell: [i32; 3], section: u16, lock: u8, first: Option<u32>) -> u32 {
        let l = self.welded.len() as u32;
        self.welded.push(v);
        self.cells.push(cell);
        self.sections.push(section);
        self.lock.push(lock);
        self.remap.push(first.unwrap_or(l));
        l
    }
}

struct DagCluster {
    /// Its vertices in its group's `verts` and `cells`, and its cluster-local corners in the
    /// group's `tris`.
    verts: Range<usize>,
    tris: Range<usize>,
    /// The stage group whose simplification made the cluster; -1 for the stage's source.
    refined: i32,
    section: u16,
}

/// A group's clusters, and what they hold.
#[derive(Default)]
struct Clusters {
    list: Vec<DagCluster>,
    /// Welded vertex per cluster vertex, for the attributes.
    verts: Vec<u32>,
    /// Grid position per cluster vertex, as emitted.
    cells: Vec<[i32; 3]>,
    tris: Vec<u8>,
}

impl Clusters {
    /// Keeps the clusters `keep` takes, and only what they hold.
    fn retain(&mut self, mut keep: impl FnMut(&DagCluster) -> bool) {
        let (mut v, mut t) = (0, 0);
        self.list.retain_mut(|c| {
            if !keep(c) {
                return false;
            }
            let (vn, tn) = (c.verts.len(), c.tris.len());
            self.verts.copy_within(c.verts.clone(), v);
            self.cells.copy_within(c.verts.clone(), v);
            self.tris.copy_within(c.tris.clone(), t);
            (c.verts, c.tris) = (v..v + vn, t..t + tn);
            (v, t) = (v + vn, t + tn);
            true
        });
        self.verts.truncate(v);
        self.cells.truncate(v);
        self.tris.truncate(t);
    }
}

struct DagGroup {
    depth: u32,
    /// Bounds and error of the geometry the group simplifies into; infinity when terminal.
    sphere: Sphere,
    error: f32,
    clusters: Clusters,
}

struct Stage {
    part: u16,
    /// The sections of the stage's class that have triangles, ascending.
    sections: Vec<u16>,
    merged: bool,
    /// Error of the stage's source geometry: 0, or the merge error on a merged stage.
    base: f32,
    groups: Vec<DagGroup>,
}

/// Cluster limits and options of one DAG.
#[derive(Copy, Clone)]
struct Limits {
    max_vertices: u16,
    max_triangles: u16,
    /// Most clusters one group may hold.
    group_max: usize,
}

/// Grid snapping of emitted positions, clamped to the mesh's grid box.
struct Snap {
    step: f64,
    min: [i32; 3],
    max: [i32; 3],
}

impl Snap {
    fn cell(&self, p: [f32; 3]) -> [i32; 3] {
        [0, 1, 2].map(|a| {
            let g = (f64::from(p[a]) / self.step + 0.5).floor();
            (g.clamp(f64::from(self.min[a]), f64::from(self.max[a]))) as i32
        })
    }
}

/// Every stage of the mesh: the part stage per (part, class) with triangles, part-major,
/// then the merged stage per class.
fn stages(
    w: &Welded,
    grid: &Grid,
    parts: &Parts,
    s: &MeshSettings,
    flags: &[u8],
    limits: impl Fn(u16) -> Limits + Sync,
    skinned: bool,
) -> Vec<Stage> {
    let _z = crate::profile::zone!("dag.stages");
    let snap = Snap {
        step: grid.step(),
        min: [0, 1, 2].map(|a| grid.cells.iter().map(|c| c[a]).min().unwrap_or(0)),
        max: [0, 1, 2].map(|a| grid.cells.iter().map(|c| c[a]).max().unwrap_or(0)),
    };
    let options = |section: u16| Options {
        limits: limits(section),
        dilate: dilates(s, flags[usize::from(section)]),
        regularize: skinned,
    };
    // Sections that cluster alike share a stage. Border dilation moves each side of a
    // seam on its own, so a dilated section keeps a stage of its own.
    let class = |section: u16| -> Class {
        let o = options(section);
        (o.limits.max_vertices, o.dilate.then_some(section))
    };
    // Each part stage's pieces, in (part, section) order.
    let mut by_stage: BTreeMap<(u16, Class), Vec<lod::PieceTriangles>> = BTreeMap::new();
    for p in lod::piece_triangles(w, parts) {
        by_stage
            .entry((p.part, class(p.section)))
            .or_default()
            .push(p);
    }
    let pieces: Vec<(u16, Vec<lod::PieceTriangles>)> = by_stage
        .into_iter()
        .map(|((part, _), pieces)| (part, pieces))
        .collect();
    let corners = |pieces| part_corners(w, grid, pieces);
    let (shared, seams) = borders(
        w,
        grid.reps.len(),
        pieces.iter().map(|p| corners(p.1.as_slice())),
    );
    let inputs = part_inputs(w, grid, &pieces, &shared, &seams);
    drop((shared, seams));
    let mut stages: Vec<Stage> = pieces
        .into_par_iter()
        .zip(inputs)
        .map(|((part, pieces), input)| {
            let _z = crate::profile::zone!("dag.build_stage");
            let sections: Vec<u16> = pieces.iter().map(|p| p.section).collect();
            drop(pieces);
            Stage {
                part,
                groups: build(&input, w, &snap, options(sections[0])),
                sections,
                merged: false,
                base: 0.0,
            }
        })
        .collect();

    let Some(m) = merge_error(s, parts) else {
        return stages;
    };
    for st in &mut stages {
        truncate(&mut st.groups, m);
    }
    let _sources = crate::profile::zone!("dag.merged_sources");
    // Each class's merged source: its parts' terminal clusters at their emitted positions.
    let mut sources: BTreeMap<Class, (Vec<u16>, Vec<Corner>)> = BTreeMap::new();
    for st in &stages {
        let (sections, corners) = sources.entry(class(st.sections[0])).or_default();
        sections.extend(&st.sections);
        for g in st.groups.iter().filter(|g| g.error == f32::INFINITY) {
            let gc = &g.clusters;
            for c in &gc.list {
                for &k in &gc.tris[c.tris.clone()] {
                    let k = c.verts.start + usize::from(k);
                    corners.push((gc.verts[k], gc.cells[k], c.section));
                }
            }
        }
    }
    let sources: Vec<(Vec<u16>, Vec<Corner>)> = sources
        .into_values()
        .filter(|s| !s.1.is_empty())
        .map(|(mut sections, corners)| {
            sections.sort_unstable();
            sections.dedup();
            (sections, corners)
        })
        .collect();
    // Dense ids of the positions the merged sources use.
    let mut id_of: FxHashMap<[i32; 3], u32> = FxHashMap::default();
    let ids: Vec<Vec<u32>> = sources
        .iter()
        .map(|(_, corners)| {
            let next = |id_of: &FxHashMap<_, _>| id_of.len() as u32;
            corners
                .iter()
                .map(|&(_, cell, _)| {
                    let n = next(&id_of);
                    *id_of.entry(cell).or_insert(n)
                })
                .collect()
        })
        .collect();
    let positions = id_of.len();
    drop(id_of);
    drop(_sources);
    let (shared, seams) = borders(
        w,
        positions,
        sources.iter().zip(&ids).map(|((_, corners), ids)| {
            corners
                .iter()
                .zip(ids)
                .map(|(&(v, _, section), &id)| (v, id, section))
        }),
    );
    // Vertices of different parts join where they share a position, attributes and section.
    let attrs = |v: u32| {
        let v = v as usize;
        let mut k = [0u32; 17];
        k[..3].copy_from_slice(&w.normals[v].map(f32::to_bits));
        k[3..5].copy_from_slice(&w.uvs[0][v].map(f32::to_bits));
        if let Some(t) = &w.tangents {
            k[5..9].copy_from_slice(&t[v].map(f32::to_bits));
        }
        if let Some(sk) = &w.skin {
            k[9] = u32::from_le_bytes(sk.joints[v]);
            k[10] = u32::from_le_bytes(sk.weights[v]);
        }
        if let Some(u) = w.uvs.get(1) {
            k[11..13].copy_from_slice(&u[v].map(f32::to_bits));
        }
        if let Some(c) = &w.colors {
            k[13..17].copy_from_slice(&c[v].map(f32::to_bits));
        }
        k
    };
    let merged: Vec<Stage> = sources
        .par_iter()
        .zip(&ids)
        .map(|((sections, corners), ids)| {
            let _z = crate::profile::zone!("dag.build_merged_stage");
            let welded = w.positions.len();
            let input = merged_input(corners, ids, positions, welded, attrs, &shared, &seams);
            let mut groups = build(&input, w, &snap, options(sections[0]));
            for g in &mut groups {
                g.error += m;
            }
            Stage {
                part: 0,
                sections: sections.clone(),
                merged: true,
                base: m,
                groups,
            }
        })
        .collect();
    stages.extend(merged);
    stages
}

/// Whether a section with `flags` dilates its open borders. Glass has no coverage to keep.
fn dilates(s: &MeshSettings, flags: u8) -> bool {
    s.dilate_double_sided
        && flags & Section::DOUBLE_SIDED != 0
        && SurfaceClass::from_flags(flags) != Some(SurfaceClass::Dielectric)
}

/// The corners of a part stage's pieces: welded vertex, the id of its grid position (its
/// [`Grid::reps`] entry) and section.
fn part_corners<'a>(
    w: &'a Welded,
    grid: &'a Grid,
    pieces: &'a [lod::PieceTriangles],
) -> impl Iterator<Item = (u32, u32, u16)> + Clone + 'a {
    pieces.iter().flat_map(move |p| {
        p.triangles.iter().flat_map(move |&t| {
            w.triangle(t as usize)
                .map(|v| (v, grid.reps[v as usize], p.section))
        })
    })
}

/// Positions the stages share, which stay locked, and uv and section seams, which stay
/// protected, by the ids below `positions` that the stages' corners (welded vertex, position
/// id, section) name.
fn borders<C: Iterator<Item = (u32, u32, u16)>>(
    w: &Welded,
    positions: usize,
    stages: impl Iterator<Item = C>,
) -> (Vec<bool>, Vec<bool>) {
    let _z = crate::profile::zone!("dag.borders");
    /// What the corners so far say of a position.
    #[derive(Copy, Clone)]
    struct Seen {
        /// The first stage and welded vertex to use it.
        owner: u32,
        first: u32,
        /// The last stage to use it and its section there.
        stage: u32,
        section: u16,
    }
    let mut seen = vec![
        Seen {
            owner: u32::MAX,
            first: 0,
            stage: u32::MAX,
            section: 0,
        };
        positions
    ];
    let (mut shared, mut seams) = (vec![false; positions], vec![false; positions]);
    for (si, corners) in stages.enumerate() {
        let si = si as u32;
        for (v, id, section) in corners {
            let (s, i) = (&mut seen[id as usize], id as usize);
            if s.owner == u32::MAX {
                *s = Seen {
                    owner: si,
                    first: v,
                    stage: si,
                    section,
                };
                continue;
            }
            shared[i] |= s.owner != si;
            seams[i] |= lod::attributes_differ(w, s.first, v);
            if s.stage != si {
                (s.stage, s.section) = (si, section);
            } else if s.section != section {
                seams[i] = true;
            }
        }
    }
    (shared, seams)
}

/// The `clodBuild` input of each part stage: one vertex per welded vertex and section, in
/// the order the stage's corners first use them.
fn part_inputs(
    w: &Welded,
    grid: &Grid,
    stages: &[(u16, Vec<lod::PieceTriangles>)],
    shared: &[bool],
    seams: &[bool],
) -> Vec<Input> {
    let _z = crate::profile::zone!("dag.input");
    let n = grid.reps.len();
    // Per welded vertex: the last stage that numbered it, its section and vertex there.
    let mut numbered = vec![(u32::MAX, 0u16, 0u32); n];
    // Per position: the last stage that used it and its first vertex there.
    let mut first = vec![(u32::MAX, 0u32); n];
    let mut out = Vec::with_capacity(stages.len());
    for (si, (_, pieces)) in stages.iter().enumerate() {
        let si = si as u32;
        let corners = 3 * pieces.iter().map(|p| p.triangles.len()).sum::<usize>();
        let mut input = Input::with_capacity(corners);
        // A welded vertex in a second section of the stage, which seams protect.
        let mut more: FxHashMap<(u32, u16), u32> = FxHashMap::default();
        for (v, id, section) in part_corners(w, grid, pieces) {
            let mut add = |input: &mut Input| {
                let f = &mut first[id as usize];
                let stands = (f.0 == si).then_some(f.1);
                let lock = lod::vertex_flags(id, shared, seams).bits();
                let l = input.push(v, grid.cells[v as usize], section, lock, stands);
                if stands.is_none() {
                    *f = (si, l);
                }
                l
            };
            let at = &mut numbered[v as usize];
            let l = if at.0 != si {
                let l = add(&mut input);
                *at = (si, section, l);
                l
            } else if at.1 == section {
                at.2
            } else {
                *more.entry((v, section)).or_insert_with(|| add(&mut input))
            };
            input.indices.push(l);
        }
        out.push(input);
    }
    out
}

/// The `clodBuild` input of a merged stage over `corners`, whose positions have the ids
/// `ids` below `positions`: one vertex per position, attributes (`attrs` of its welded
/// vertex, one of `welded`) and section, in the order the corners first use them.
fn merged_input<K: std::hash::Hash + Eq>(
    corners: &[Corner],
    ids: &[u32],
    positions: usize,
    welded: usize,
    attrs: impl Fn(u32) -> K,
    shared: &[bool],
    seams: &[bool],
) -> Input {
    let _z = crate::profile::zone!("dag.input");
    // Per welded vertex, the id of its attributes among those the corners use.
    let mut class_of = vec![u32::MAX; welded];
    let mut classes: FxHashMap<K, u32> = FxHashMap::default();
    let mut local_of: FxHashMap<(u32, u32, u16), u32> = FxHashMap::default();
    let mut first = vec![u32::MAX; positions];
    let mut input = Input::with_capacity(corners.len());
    for (&(v, cell, section), &id) in corners.iter().zip(ids) {
        let class = &mut class_of[v as usize];
        if *class == u32::MAX {
            let next = classes.len() as u32;
            *class = *classes.entry(attrs(v)).or_insert(next);
        }
        let l = *local_of.entry((id, *class, section)).or_insert_with(|| {
            let f = &mut first[id as usize];
            let stands = (*f != u32::MAX).then_some(*f);
            let lock = lod::vertex_flags(id, shared, seams).bits();
            let l = input.push(v, cell, section, lock, stands);
            if stands.is_none() {
                *f = l;
            }
            l
        });
        input.indices.push(l);
    }
    input
}

/// Groups at or over `m` become terminal; clusters made from them go, and so do groups
/// left without clusters.
fn truncate(groups: &mut Vec<DagGroup>, m: f32) {
    let _z = crate::profile::zone!("dag.truncate");
    for g in groups.iter_mut() {
        if g.error >= m {
            g.error = f32::INFINITY;
        }
    }
    let terminal: Vec<bool> = groups.iter().map(|g| g.error == f32::INFINITY).collect();
    for g in groups.iter_mut() {
        g.clusters
            .retain(|c| c.refined < 0 || !terminal[c.refined as usize]);
    }
    let mut new_index = vec![-1i32; groups.len()];
    let mut next = 0;
    for (i, g) in groups.iter().enumerate() {
        if !g.clusters.list.is_empty() {
            new_index[i] = next;
            next += 1;
        }
    }
    groups.retain(|g| !g.clusters.list.is_empty());
    for c in groups.iter_mut().flat_map(|g| &mut g.clusters.list) {
        if c.refined >= 0 {
            c.refined = new_index[c.refined as usize];
            debug_assert!(c.refined >= 0, "a kept cluster comes from a kept group");
        }
    }
}

#[derive(Copy, Clone)]
struct Options {
    limits: Limits,
    dilate: bool,
    regularize: bool,
}

/// The DAG groups of one stage, in emission order.
fn build(input: &Input, w: &Welded, snap: &Snap, o: Options) -> Vec<DagGroup> {
    let attributes = crate::profile::zone!("dag.attributes");
    let mut positions: Vec<f32> = input
        .cells
        .iter()
        .flat_map(|c| c.map(|v| (f64::from(v) * snap.step) as f32))
        .collect();
    let normals: Vec<f32> = input
        .welded
        .iter()
        .flat_map(|&v| w.normals[v as usize])
        .collect();
    drop(attributes);
    let weights = [0.5f32; 3];
    let mut config = meshopt::clod_default_config_rt(usize::from(o.limits.max_triangles));
    config.max_vertices = usize::from(o.limits.max_vertices);
    config.min_triangles = usize::from(o.limits.max_triangles) / 4;
    config.partition_sort = true;
    config.partition_size = partition_size(o.limits.group_max);
    config.simplify_error_clamped = true;
    config.simplify_dilate_borders = o.dilate;
    config.simplify_regularize = o.regularize;
    config.optimize_clusters = false;
    let max_indices = usize::from(o.limits.max_triangles) * 3;
    let mesh = super::clod::Mesh {
        indices: &input.indices,
        attributes: &normals,
        weights: &weights,
        lock: &input.lock,
        remap: &input.remap,
        sections: &input.sections,
    };
    let mut groups = super::clod::build(&config, mesh, &mut positions, |e| {
        group_of(&e, &input.welded, &input.sections, snap, max_indices)
    });
    bound_groups(&mut groups, snap.step);
    groups
}

/// Grows each group's sphere about its center to hold the stored positions of its clusters
/// and of the clusters simplified from it, and the spheres of the groups its clusters come
/// from. `clodBuild` gives a group the bounds of its clusters before its borders dilate and
/// its simplification snaps to the grid, so the clusters made from it can reach past them,
/// and the traversal's tests of a group's sphere stand for both sides of its replacement.
/// Groups come before the groups their simplification feeds, so each sees the spheres it
/// holds final.
fn bound_groups(groups: &mut [DagGroup], step: f64) {
    let at = |c: [i32; 3]| c.map(|v| f64::from(v) * step);
    let distance = |a: [f64; 3], b: [f32; 3]| {
        (0..3)
            .map(|i| (a[i] - f64::from(b[i])).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    // Per group, the farthest stored position of the clusters simplified from it.
    let mut reach = vec![0.0f64; groups.len()];
    for g in groups.iter() {
        for c in &g.clusters.list {
            let Ok(from) = usize::try_from(c.refined) else {
                continue;
            };
            let center = groups[from].sphere.center;
            for &cell in &g.clusters.cells[c.verts.clone()] {
                reach[from] = reach[from].max(distance(at(cell), center));
            }
        }
    }
    for i in 0..groups.len() {
        let center = groups[i].sphere.center;
        let mut radius = f64::from(groups[i].sphere.radius).max(reach[i]);
        for &cell in &groups[i].clusters.cells {
            radius = radius.max(distance(at(cell), center));
        }
        for c in &groups[i].clusters.list {
            if let Ok(from) = usize::try_from(c.refined) {
                let s = groups[from].sphere;
                let center64 = s.center.map(f64::from);
                radius = radius.max(distance(center64, center) + f64::from(s.radius));
            }
        }
        let mut r = radius as f32;
        if f64::from(r) < radius {
            r = r.next_up();
        }
        groups[i].sphere.radius = r;
    }
}

/// The partition target whose largest partitions (a third over the target) hold at most
/// `group_max` clusters.
fn partition_size(group_max: usize) -> usize {
    let mut p = 16;
    while p > 1 && p + p / 3 > group_max {
        p -= 1;
    }
    p
}

/// A group as the DAG keeps it: each cluster cut by section, its vertices as welded
/// vertices at grid positions snapped from `positions`.
fn group_of(
    e: &super::clod::Emitted<'_>,
    welded: &[u32],
    sections: &[u16],
    snap: &Snap,
    max_indices: usize,
) -> DagGroup {
    let positions = e.positions;
    let mut local_vertices = [0u32; MAX_CLUSTER_CORNERS];
    let mut local_triangles = [0u8; MAX_CLUSTER_CORNERS];
    let mut out = Clusters::default();
    let push = |out: &mut Clusters, section: u16, refined: i32, used: &[u32], tris: &[u8]| {
        let (v, t) = (out.verts.len(), out.tris.len());
        out.verts.extend(used.iter().map(|&v| welded[v as usize]));
        out.cells.extend(used.iter().map(|&v| {
            let p = &positions[v as usize * 3..v as usize * 3 + 3];
            snap.cell([p[0], p[1], p[2]])
        }));
        out.tris.extend_from_slice(tris);
        out.list.push(DagCluster {
            verts: v..out.verts.len(),
            tris: t..out.tris.len(),
            refined,
            section,
        });
    };
    for k in e.clusters {
        let (refined, indices) = (k.refined, k.corners(e.indices));
        assert!(
            indices.len() <= max_indices && max_indices <= local_triangles.len(),
            "cluster over its limits"
        );
        // SAFETY: the local buffers hold `indices.len()` entries.
        let n = unsafe {
            ffi::meshopt_extractMeshletIndices(
                local_vertices.as_mut_ptr(),
                local_triangles.as_mut_ptr(),
                indices.as_ptr(),
                indices.len(),
            )
        };
        let local = &local_vertices[..n];
        let tris = &local_triangles[..indices.len()];
        let section_of = |t: &[u8]| sections[local[usize::from(t[0])] as usize];
        // One cluster per section: a triangle's corners all lie in one section.
        let triangles = tris.as_chunks::<3>().0;
        if triangles.iter().all(|t| section_of(t) == section_of(tris)) {
            push(&mut out, section_of(tris), refined, local, tris);
            continue;
        }
        // The triangles by section, ascending, each section's in order.
        let mut by_section = [(0u16, 0u8); MAX_CLUSTER_CORNERS / 3];
        let by_section = &mut by_section[..triangles.len()];
        for (b, (i, t)) in by_section.iter_mut().zip(triangles.iter().enumerate()) {
            *b = (section_of(t), i as u8);
        }
        by_section.sort_unstable();
        for run in by_section.chunk_by(|a, b| a.0 == b.0) {
            let (mut new_of, mut used, mut corners) = (
                [u16::MAX; MAX_CLUSTER_VERTICES],
                [0u32; MAX_CLUSTER_VERTICES],
                [0u8; MAX_CLUSTER_CORNERS],
            );
            let (mut nv, mut nc) = (0, 0);
            for &(_, i) in run {
                for &k in &triangles[usize::from(i)] {
                    let k = usize::from(k);
                    if new_of[k] == u16::MAX {
                        new_of[k] = nv as u16;
                        used[nv] = local[k];
                        nv += 1;
                    }
                    corners[nc] = new_of[k] as u8;
                    nc += 1;
                }
            }
            push(&mut out, run[0].0, refined, &used[..nv], &corners[..nc]);
        }
    }
    let b = e.bounds;
    DagGroup {
        depth: e.depth,
        sphere: Sphere {
            center: b.center,
            radius: b.radius,
        },
        error: if b.error == f32::MAX {
            f32::INFINITY
        } else {
            b.error
        },
        clusters: out,
    }
}

/// Hierarchy nodes of one stage over its groups, by stage group index, and the root: a
/// forest per depth, joined under one root when there are several.
fn hierarchy(groups: &[DagGroup]) -> (Vec<ffi::clodNode>, usize) {
    let mut depths: Vec<u32> = groups.iter().map(|g| g.depth).collect();
    depths.sort_unstable();
    depths.dedup();
    let clod: Vec<meshopt::ClodGroup> = groups
        .iter()
        .map(|g| meshopt::ClodGroup {
            depth: depths.binary_search(&g.depth).expect("depth listed") as c_int,
            simplified: meshopt::ClodBounds {
                center: g.sphere.center,
                radius: g.sphere.radius,
                error: g.error,
            },
        })
        .collect();
    let levels = depths.len();
    let mut nodes = meshopt::clod_build_hierarchy(&clod, 8, levels);
    if levels > 1 {
        let roots = &nodes[..levels];
        let centers: Vec<[f32; 3]> = roots.iter().map(|n| n.bounds.center).collect();
        let radii: Vec<f32> = roots.iter().map(|n| n.bounds.radius).collect();
        let sphere = meshopt::compute_sphere_bounds(
            meshopt::PositionDataAdapter {
                data: bytemuck::cast_slice(&centers),
                position_count: centers.len(),
                position_stride: 12,
                position_offset: 0,
            },
            Some(meshopt::RadiusDataAdapter {
                data: bytemuck::cast_slice(&radii),
                radius_count: radii.len(),
                radius_stride: 4,
                radius_offset: 0,
            }),
        );
        let error = roots.iter().map(|n| n.bounds.error).fold(0.0, f32::max);
        nodes.push(ffi::clodNode {
            bounds: meshopt::ClodBounds {
                center: sphere.center,
                radius: sphere.radius,
                error,
            },
            group: -1,
            child_offset: 0,
            child_count: levels as u32,
        });
    }
    let root = if levels > 1 { nodes.len() - 1 } else { 0 };
    (nodes, root)
}

/// The merge error a mesh's parts merge from, if it has parts.
fn merge_error(s: &MeshSettings, parts: &Parts) -> Option<f32> {
    s.merge_error.filter(|_| parts.table.len() > 1)
}

/// Triangles of the groups one batch of [`cook`] cooks at once, unless the mesh cooks whole.
const BATCH_TRIANGLES: usize = 1 << 22;

/// A group with its cooked clusters.
struct Cooked {
    depth: u32,
    sphere: Sphere,
    error: f32,
    clusters: Vec<ClusterData>,
}

/// The DAG mode mesh: stages, clusters, pages of whole groups coarse to fine with the
/// terminal groups' pages pinned, page dependencies and the hierarchy.
pub(super) fn cook(cx: &Ctx<'_>) -> Result<Built, String> {
    let s = cx.s;
    let flags: Vec<u8> = cx.sections.iter().map(|sec| sec.flags).collect();
    let header = size_of::<PageHeader>();
    let limits = |section: u16| {
        let maxv = cx.max_vertices(flags[usize::from(section)]);
        let cluster_max =
            cluster_resident_len(usize::from(maxv), usize::from(s.max_triangles), cx.stride);
        Limits {
            max_vertices: maxv,
            max_triangles: s.max_triangles,
            group_max: (PAGE_MAX - header) / cluster_max,
        }
    };
    let mut stages = stages(
        cx.w,
        cx.grid,
        cx.parts,
        s,
        &flags,
        limits,
        cx.w.skin.is_some(),
    );
    // Global group index: stages in order, each stage's groups in emission order.
    let mut first = Vec::with_capacity(stages.len());
    let mut at: Vec<(usize, usize)> = Vec::new();
    for (si, st) in stages.iter().enumerate() {
        first.push(at.len());
        at.extend((0..st.groups.len()).map(|gi| (si, gi)));
    }
    let group = |stages: &[Stage], g: usize| -> (bool, u32) {
        let (si, gi) = at[g];
        (stages[si].merged, stages[si].groups[gi].depth)
    };
    let terminal = |stages: &[Stage], g: usize| {
        let (si, gi) = at[g];
        stages[si].groups[gi].error == f32::INFINITY
    };
    // Terminal groups first and on pages of their own, then merged stages, then coarse to
    // fine.
    let mut order: Vec<usize> = (0..at.len()).collect();
    order.sort_by_key(|&g| {
        let (merged, depth) = group(&stages, g);
        (!terminal(&stages, g), !merged, std::cmp::Reverse(depth), g)
    });
    let mut table_index = vec![0u32; at.len()];
    for (t, &g) in order.iter().enumerate() {
        table_index[g] = t as u32;
    }
    let refined_of = |si: usize, r: i32| usize::try_from(r).map_or(-1, |r| (first[si] + r) as i32);

    // Groups cook in page order a batch at a time, and each page is written and its clusters
    // freed once it is full. Thickness reads every cluster, so a mesh with subsurface
    // sections cooks in one batch.
    let whole = cx.sections.iter().any(|sec| sec.subsurface);
    let mut cooked: Vec<Option<Cooked>> = (0..at.len()).map(|_| None).collect();
    let mut refined: Vec<Vec<i32>> = vec![Vec::new(); at.len()];
    let mut page_groups: Vec<Vec<usize>> = Vec::new();
    let mut page_of = vec![0u32; at.len()];
    let mut pinned_pages = 0;
    let (mut size, mut aligned) = (header, false);
    let mut pages = Vec::new();
    let mut page_info: Vec<PageInfo> = Vec::new();
    let mut group_table: Vec<Group> = vec![bytemuck::Zeroable::zeroed(); at.len()];
    let mut set_clusters = Vec::new();
    let mut parts = cx.parts.table.clone();
    let (mut next, mut written) = (0, 0);
    while next < order.len() {
        let mut end = next;
        let mut triangles = 0;
        while end < order.len() && (whole || triangles < BATCH_TRIANGLES) {
            let (si, gi) = at[order[end]];
            let g = &stages[si].groups[gi];
            triangles += g.clusters.tris.len() / 3;
            end += 1;
        }
        let batch = &order[next..end];
        let work: Vec<(usize, Clusters)> = batch
            .iter()
            .map(|&g| {
                let (si, gi) = at[g];
                (g, std::mem::take(&mut stages[si].groups[gi].clusters))
            })
            .collect();
        let stages = &stages;
        // Each group, cooked, with per cluster the global group it was simplified from, or -1.
        let mut made: Vec<(usize, Cooked, Vec<i32>)> = work
            .into_par_iter()
            .map(|(gl, clusters)| {
                let _z = crate::profile::zone!("dag.group_clusters");
                let (si, gi) = at[gl];
                let st = &stages[si];
                let g = &st.groups[gi];
                let refined = clusters
                    .list
                    .iter()
                    .map(|c| refined_of(si, c.refined))
                    .collect();
                let clusters = clusters
                    .list
                    .iter()
                    .map(|c| {
                        let (verts, cells) = (
                            &clusters.verts[c.verts.clone()],
                            &clusters.cells[c.verts.clone()],
                        );
                        let tris = clusters.tris[c.tris.clone()].to_vec();
                        let mut d = cx.cluster(c.section, st.part, st.merged, verts, cells, tris);
                        let r = &mut d.record;
                        (r.self_sphere, r.self_error) = match usize::try_from(c.refined) {
                            Ok(from) => {
                                r.from_group = table_index[first[si] + from];
                                (st.groups[from].sphere, st.groups[from].error)
                            }
                            Err(_) => (r.cull_sphere, st.base),
                        };
                        (r.parent_sphere, r.parent_error) = (g.sphere, g.error);
                        d
                    })
                    .collect();
                (
                    gl,
                    Cooked {
                        depth: g.depth,
                        sphere: g.sphere,
                        error: g.error,
                        clusters,
                    },
                    refined,
                )
            })
            .collect();
        cx.omm_all(
            made.iter_mut()
                .flat_map(|(_, g, _)| &mut g.clusters)
                .collect(),
        );
        {
            let _z = crate::profile::zone!("dag.fit");
            made.par_iter_mut()
                .try_for_each(|(_, g, _)| fit(cx, &mut g.clusters))?;
        }
        super::thickness::assign(
            made.iter_mut()
                .flat_map(|(_, g, _)| &mut g.clusters)
                .collect(),
            cx.grid,
            cx.sections,
        );
        for (g, c, r) in made {
            cooked[g] = Some(c);
            refined[g] = r;
        }

        let _z = crate::profile::zone!("dag.pages");
        for (k, &gi) in order.iter().enumerate().take(end).skip(next) {
            let g = cooked[gi].as_ref().expect("cooked in this batch");
            let omm = g.clusters.iter().any(|c| c.omm.is_some());
            let bytes = |aligned: bool| {
                g.clusters.iter().map(|c| c.cost(cx.stride)).sum::<usize>()
                    + if omm && !aligned {
                        OMM_ALIGN as usize - 1
                    } else {
                        0
                    }
            };
            let block_end = k > 0 && terminal(stages, order[k - 1]) && !terminal(stages, gi);
            if block_end {
                pinned_pages = page_groups.len() as u32;
            }
            if page_groups.is_empty() || block_end || size + bytes(aligned) > PAGE_MAX {
                page_groups.push(Vec::new());
                (size, aligned) = (header, false);
            }
            size += bytes(aligned);
            aligned |= omm;
            page_of[gi] = page_groups.len() as u32 - 1;
            page_groups.last_mut().expect("a page").push(gi);
        }
        next = end;
        // Every page but the last is full; the last is too once every group has cooked.
        let full = if next == order.len() {
            page_groups.len()
        } else {
            page_groups.len() - 1
        };
        let ready = &page_groups[written..full];
        written = full;
        // Per page: its resident length, its cluster and triangle counts and its stored
        // bytes.
        let built: Vec<(usize, usize, u32, EncodedBlob<'static>)> = ready
            .par_iter()
            .map(|members| {
                let clusters: Vec<&ClusterData> = members
                    .iter()
                    .flat_map(|&g| &cooked[g].as_ref().expect("cooked").clusters)
                    .collect();
                let bytes = write_page(&clusters, cx.stride);
                // Terminal groups' pages are the pinned ones.
                let pinned = terminal(stages, members[0]);
                let stored = store_page(&bytes, pinned, cx.compress).map_err(|e| e.to_string())?;
                Ok((
                    bytes.len(),
                    clusters.len(),
                    page_triangles(&clusters),
                    stored,
                ))
            })
            .collect::<Result<_, String>>()?;
        for (members, (resident_len, cluster_count, tri_count, stored)) in ready.iter().zip(built) {
            let p = pages.len() as u32;
            page_info.push(PageInfo {
                blob: p + 1,
                resident_len: resident_len as u32,
                tri_count,
                cluster_count: cluster_count as u16,
                _pad: 0,
                deps: Span::default(),
            });
            pages.push(stored);
            let mut first_cluster = 0u16;
            for &gi in members {
                let g = cooked[gi].take().expect("cooked");
                group_table[table_index[gi] as usize] = Group {
                    simplified: g.sphere,
                    error: g.error,
                    page: p,
                    first_cluster,
                    cluster_count: g.clusters.len() as u16,
                    depth: u8::try_from(g.depth).map_err(|_| format!("DAG depth {}", g.depth))?,
                    _pad: [0; 3],
                    node_page: 0,
                };
                first_cluster += g.clusters.len() as u16;
                for (c, &r) in g.clusters.iter().zip(&refined[gi]) {
                    let refined = usize::try_from(r).map_or(-1, |r| table_index[r] as i32);
                    set_clusters.push(c.set_cluster(table_index[gi], refined));
                    c.add_to(&mut parts, cx.grid);
                }
            }
        }
    }
    if order.last().is_some_and(|&g| terminal(&stages, g)) {
        pinned_pages = page_groups.len() as u32;
    }

    // Each page depends on the pages holding the coarser clusters its groups simplify into.
    let mut into: Vec<Vec<u32>> = vec![Vec::new(); at.len()];
    for (gi, r) in refined.iter().enumerate() {
        for &r in r {
            if let Ok(r) = usize::try_from(r) {
                into[r].push(page_of[gi]);
            }
        }
    }
    let mut page_deps = Vec::new();
    for (p, members) in page_groups.iter().enumerate() {
        let mut deps: Vec<u32> = members
            .iter()
            .flat_map(|&g| into[g].iter().copied())
            .filter(|&q| q as usize != p)
            .collect();
        deps.sort_unstable();
        deps.dedup();
        assert!(
            deps.iter().all(|&q| (q as usize) < p),
            "pages depend on earlier pages"
        );
        page_info[p].deps = Span {
            first: page_deps.len() as u32,
            count: deps.len() as u32,
        };
        page_deps.extend(deps);
    }
    let _hierarchy = crate::profile::zone!("dag.hierarchy");
    // Hierarchy: each stage's nodes, then breadth first from the roots.
    let mut nodes = Vec::new();
    let mut stage_root = Vec::with_capacity(stages.len());
    for (si, st) in stages.iter().enumerate() {
        let (stage_nodes, root) = hierarchy(&st.groups);
        let base = nodes.len() as u32;
        stage_root.push(base + root as u32);
        for n in stage_nodes {
            let leaf = usize::try_from(n.group).ok();
            nodes.push(Node {
                sphere: Sphere {
                    center: n.bounds.center,
                    radius: n.bounds.radius,
                },
                max_error: n.bounds.error,
                child_first: if leaf.is_some() {
                    0
                } else {
                    base + n.child_offset
                },
                child_count: if leaf.is_some() {
                    0
                } else {
                    n.child_count as u16
                },
                _pad: 0,
                group: leaf.map_or(-1, |g| table_index[first[si] + g] as i32),
            });
        }
    }
    let sections = cx.sections.len();
    // A stage's root sits at its first section's slot.
    let root_of: FxHashMap<(usize, usize, bool), u32> = stages
        .iter()
        .zip(stage_root)
        .map(|(st, root)| {
            let slot = (usize::from(st.part), usize::from(st.sections[0]), st.merged);
            (slot, root)
        })
        .collect();
    let stage_of = |part: usize, section: usize, merged: bool| {
        root_of
            .get(&(part, section, merged))
            .copied()
            .unwrap_or(MeshHeader::NO_ROOT)
    };
    let mut roots: Vec<u32> = (0..cx.parts.table.len() * sections)
        .map(|i| stage_of(i / sections, i % sections, false))
        .collect();
    let merge = merge_error(s, cx.parts);
    if merge.is_some() {
        roots.extend((0..sections).map(|sec| stage_of(0, sec, true)));
    }
    let pinned = |g: i32| group_table[g as usize].page < pinned_pages;
    let (nodes, roots, floor) = breadth_first(&nodes, &roots, pinned);
    drop(_hierarchy);
    let _node_pages = crate::profile::zone!("dag.node_pages");
    let (nodes, node_pages, node_blobs) = if nodes.len() * size_of::<Node>()
        <= s.hierarchy_resident_max as usize
    {
        (nodes, Vec::new(), Vec::new())
    } else {
        let per_page = NODE_PAGE_MAX / size_of::<Node>();
        let (table, blobs, page_of) = node_pages(&nodes, floor, pages.len() as u32 + 1, per_page);
        for (n, &page) in nodes.iter().zip(&page_of) {
            if let Ok(g) = usize::try_from(n.group) {
                group_table[g].node_page = page;
            }
        }
        (Vec::new(), table, blobs)
    };

    let info = MeshInfo {
        header: MeshHeader {
            merge_level: MeshHeader::NO_MERGE_LEVEL,
            merge_error: merge.unwrap_or(f32::INFINITY),
            pinned_pages,
            ..bytemuck::Zeroable::zeroed()
        },
        sections: Vec::new(),
        parts,
        part_levels: Vec::new(),
        levels: Vec::new(),
        nodes,
        groups: group_table,
        node_pages,
        roots,
        pages: page_info,
        page_deps,
        cut_rows: Vec::new(),
        emissive: ClusterSets::default(),
        casters: ClusterSets::default(),
    };
    let _free = crate::profile::zone!("dag.free");
    drop(stages);
    Ok(Built {
        info,
        pages,
        node_pages: node_blobs,
        clusters: set_clusters,
    })
}

/// Makes a group fit one page: its clusters' micromaps are measured again under halving
/// caps, and dropped below 64 bytes.
fn fit(cx: &Ctx<'_>, clusters: &mut [ClusterData]) -> Result<(), String> {
    let mut cap = cx.s.omm.as_ref().map_or(0, |o| o.cluster_bytes_max);
    loop {
        let omm = clusters.iter().any(|c| c.omm.is_some());
        let bytes = size_of::<PageHeader>()
            + clusters.iter().map(|c| c.cost(cx.stride)).sum::<usize>()
            + if omm { OMM_ALIGN as usize - 1 } else { 0 };
        if bytes <= PAGE_MAX {
            break;
        }
        if !omm {
            return Err(format!(
                "a group of {} clusters takes {bytes} bytes, over a {PAGE_MAX}-byte page",
                clusters.len()
            ));
        }
        cap /= 2;
        clusters
            .par_iter_mut()
            .filter(|c| c.omm.is_some())
            .for_each(|c| {
                c.omm = if cap >= 64 { cx.omm(c, cap) } else { None };
            });
    }
    clusters.iter_mut().for_each(|c| c.uvs = Vec::new());
    Ok(())
}

/// The nodes renumbered breadth first from the roots, so each node's children are
/// contiguous and the roots come first; the roots renumbered alike; and the floor, the first
/// nodes: the roots and the children of every node above a leaf whose group `pinned` takes,
/// which the top node page holds. The rest follow breadth first.
fn breadth_first(
    nodes: &[Node],
    roots: &[u32],
    pinned: impl Fn(i32) -> bool,
) -> (Vec<Node>, Vec<u32>, usize) {
    let tops = || roots.iter().copied().filter(|&r| r != MeshHeader::NO_ROOT);
    let children = |n: &Node| n.child_first..n.child_first + u32::from(n.child_count);
    let mut order: Vec<u32> = tops().collect();
    let mut i = 0;
    while i < order.len() {
        let n = &nodes[order[i] as usize];
        if n.group < 0 {
            order.extend(children(n));
        }
        i += 1;
    }
    // Whether a pinned group's leaf lies below each node: children follow their parent.
    let mut above = vec![false; nodes.len()];
    for &k in order.iter().rev() {
        let n = &nodes[k as usize];
        above[k as usize] = if n.group < 0 {
            children(n).any(|c| above[c as usize])
        } else {
            pinned(n.group)
        };
    }
    let expand = |order: &mut Vec<u32>, floor: bool| {
        let mut i = 0;
        while i < order.len() {
            let k = order[i] as usize;
            let n = &nodes[k];
            if n.group < 0 && above[k] == floor {
                order.extend(children(n));
            }
            i += 1;
        }
    };
    let mut order: Vec<u32> = tops().collect();
    expand(&mut order, true);
    let floor = order.len();
    expand(&mut order, false);
    let mut new_of = vec![u32::MAX; nodes.len()];
    for (i, &old) in order.iter().enumerate() {
        new_of[old as usize] = i as u32;
    }
    let out = order
        .iter()
        .map(|&old| {
            let mut n = nodes[old as usize];
            if n.group < 0 {
                n.child_first = new_of[n.child_first as usize];
            }
            n
        })
        .collect();
    let roots = roots
        .iter()
        .map(|&r| {
            if r == MeshHeader::NO_ROOT {
                r
            } else {
                new_of[r as usize]
            }
        })
        .collect();
    (out, roots, floor)
}

/// Cuts the nodes of [`breadth_first`] into pages: the top page holds the `floor` nodes,
/// whatever their count, and each other node's children join the current page when they fit
/// `per_page` nodes and the page holds their parent or has their parent's page as its parent.
/// Returns the page table, with blob indices from `first_blob`, the page blobs and each
/// node's page.
fn node_pages(
    nodes: &[Node],
    floor: usize,
    first_blob: u32,
    per_page: usize,
) -> (Vec<NodePage>, Vec<Vec<u8>>, Vec<u32>) {
    let mut table = vec![NodePage {
        first_node: 0,
        node_count: floor as u32,
        blob: first_blob,
        parent_page: NodePage::TOP,
    }];
    let mut page_of = vec![0u32; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        if n.group >= 0 || (n.child_first as usize) < floor {
            continue;
        }
        let (first, count) = (n.child_first, u32::from(n.child_count));
        let parent = page_of[i];
        let current = table.len() - 1;
        let page = &table[current];
        let joins = (parent as usize == current || page.parent_page == parent)
            && page.first_node + page.node_count == first
            && (page.node_count + count) as usize <= per_page;
        if joins {
            table[current].node_count += count;
        } else {
            table.push(NodePage {
                first_node: first,
                node_count: count,
                blob: first_blob + table.len() as u32,
                parent_page: parent,
            });
        }
        let p = table.len() as u32 - 1;
        page_of[first as usize..(first + count) as usize].fill(p);
    }
    let blobs = table
        .iter()
        .map(|p| {
            let r = p.first_node as usize..(p.first_node + p.node_count) as usize;
            bytemuck::cast_slice(&nodes[r]).to_vec()
        })
        .collect();
    (table, blobs, page_of)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(child_first: u32, child_count: u16, group: i32) -> Node {
        Node {
            sphere: Sphere {
                center: [0.0; 3],
                radius: 1.0,
            },
            max_error: 1.0,
            child_first,
            child_count,
            _pad: 0,
            group,
        }
    }

    /// A tree of internal nodes with `width` children per level, `depth` levels deep, then
    /// a leaf under each.
    fn tree(width: u32, depth: u32) -> Vec<Node> {
        let mut nodes = vec![node(0, 0, -1)];
        let mut level = vec![0usize];
        let mut group = 0;
        for d in 0..=depth {
            let mut next = Vec::new();
            for &n in &level {
                let first = nodes.len() as u32;
                let (count, leaf) = if d == depth {
                    (1, true)
                } else {
                    (width, false)
                };
                nodes[n].child_first = first;
                nodes[n].child_count = count as u16;
                for _ in 0..count {
                    next.push(nodes.len());
                    nodes.push(node(0, 0, if leaf { group } else { -1 }));
                    group += i32::from(leaf);
                }
            }
            level = next;
        }
        nodes
    }

    /// Pages hold contiguous nodes with their siblings, under their parent's page, and the
    /// top page holds every leaf of a pinned group with each node above it, however many.
    #[test]
    fn node_pages_keep_siblings_and_parents_together_and_pinned_leaves_on_top() {
        let cases: [fn(i32) -> bool; 3] = [|_| false, |g| g % 37 == 5, |g| g < 40];
        for pinned in cases {
            let (nodes, roots, floor) = breadth_first(&tree(4, 4), &[0], pinned);
            let mut parent = vec![None; nodes.len()];
            for (i, n) in nodes.iter().enumerate().filter(|(_, n)| n.group < 0) {
                for c in n.child_first..n.child_first + u32::from(n.child_count) {
                    assert!(c as usize > i, "children follow their parent");
                    parent[c as usize] = Some(i);
                }
            }
            for per_page in [5, 7, 16, 64] {
                let (table, blobs, page_of) = node_pages(&nodes, floor, 10, per_page);
                assert!(table.len() > 1);
                assert_eq!(table[0].parent_page, NodePage::TOP);
                assert!(table[0].node_count as usize >= floor.max(roots.len()));
                let mut next = 0;
                for (i, p) in table.iter().enumerate() {
                    assert_eq!(p.first_node, next, "pages are contiguous");
                    assert!(i == 0 || p.node_count as usize <= per_page);
                    assert_eq!(p.blob, 10 + i as u32);
                    assert_eq!(blobs[i].len(), p.node_count as usize * size_of::<Node>());
                    let r = p.first_node as usize..(p.first_node + p.node_count) as usize;
                    assert!(page_of[r].iter().all(|&q| q as usize == i));
                    next += p.node_count;
                }
                assert_eq!(next as usize, nodes.len());
                for (i, n) in nodes.iter().enumerate() {
                    if n.group >= 0 {
                        let mut at = Some(i);
                        while let Some(k) = at.filter(|_| pinned(n.group)) {
                            assert_eq!(page_of[k], 0, "node {k} above pinned group {}", n.group);
                            at = parent[k];
                        }
                        continue;
                    }
                    let first = page_of[n.child_first as usize] as usize;
                    let last = n.child_first + u32::from(n.child_count) - 1;
                    assert_eq!(first, page_of[last as usize] as usize);
                    let parent = page_of[i] as usize;
                    assert!(parent == first || table[first].parent_page as usize == parent);
                }
            }
        }
    }
}
