//! Cluster LOD DAGs through meshoptimizer's clusterlod.
//!
//! The part stage is one DAG per (part, class), a class being the sections that cluster
//! alike (the same vertex limit, and no border dilation, which moves each side of a seam on
//! its own), with every grid position another stage shares locked. Within a stage the
//! sections stay apart as attribute seams: a vertex belongs to one section, and positions
//! where sections meet are protected, so the (permissive) simplifier moves them only along
//! the border and never onto another section's vertex; the border coarsens with the rest.
//! Clusters are cut by section as they are emitted; the pieces of one cluster share its
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
use std::os::raw::c_int;

use meshopt::ffi;
use orr_pack::mesh::{
    Cluster, Group, MeshHeader, MeshInfo, NODE_PAGE_MAX, Node, NodePage, OMM_ALIGN, PAGE_MAX,
    PageHeader, PageInfo, Section, cluster_resident_len,
};
use orr_pack::{Aabb, Span, Sphere};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use super::lod::{self, Parts, Welded};
use super::{Built, ClusterData, Ctx, Grid, MeshSettings, store_page, write_page};

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
    indices: Vec<u32>,
}

struct DagCluster {
    /// Welded vertex per cluster vertex, for the attributes.
    verts: Vec<u32>,
    /// Grid position per cluster vertex, as emitted.
    cells: Vec<[i32; 3]>,
    /// Cluster-local corners.
    tris: Vec<u8>,
    /// The stage group whose simplification made the cluster; -1 for the stage's source.
    refined: i32,
    section: u16,
}

struct DagGroup {
    depth: u32,
    /// Bounds and error of the geometry the group simplifies into; infinity when terminal.
    sphere: Sphere,
    error: f32,
    clusters: Vec<DagCluster>,
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
        dilate: s.dilate_double_sided && flags[usize::from(section)] & Section::DOUBLE_SIDED != 0,
        regularize: skinned,
    };
    // Sections that cluster alike share a stage. Border dilation moves each side of a
    // seam on its own, so a dilated section keeps a stage of its own.
    let class = |section: u16| -> Class {
        let o = options(section);
        (o.limits.max_vertices, o.dilate.then_some(section))
    };
    let mut by_stage: BTreeMap<(u16, Class), (Vec<u16>, Vec<Corner>)> = BTreeMap::new();
    for p in lod::pieces(w, parts) {
        let (sections, corners) = by_stage.entry((p.part, class(p.section))).or_default();
        sections.push(p.section);
        corners.extend(
            p.indices
                .iter()
                .map(|&v| (v, grid.cells[v as usize], p.section)),
        );
    }
    let pieces: Vec<(u16, Vec<u16>, Vec<Corner>)> = by_stage
        .into_iter()
        .map(|((part, _), (sections, corners))| (part, sections, corners))
        .collect();
    let (shared, seams) = borders(w, pieces.iter().map(|p| p.2.as_slice()));
    let mut stages: Vec<Stage> = pieces
        .par_iter()
        .map(|(part, sections, corners)| {
            let _z = crate::profile::zone!("dag.build_stage");
            let input = input(corners, |v, _, section| (v, section), &shared, &seams);
            Stage {
                part: *part,
                sections: sections.clone(),
                merged: false,
                base: 0.0,
                groups: build(&input, w, &snap, options(sections[0])),
            }
        })
        .collect();

    let Some(m) = merge_error(s, parts) else {
        return stages;
    };
    for st in &mut stages {
        truncate(&mut st.groups, m);
    }
    // Each class's merged source: its parts' terminal clusters at their emitted positions.
    let mut sources: BTreeMap<Class, (Vec<u16>, Vec<Corner>)> = BTreeMap::new();
    for st in &stages {
        let (sections, corners) = sources.entry(class(st.sections[0])).or_default();
        sections.extend(&st.sections);
        for g in st.groups.iter().filter(|g| g.error == f32::INFINITY) {
            for c in &g.clusters {
                for &k in &c.tris {
                    let k = usize::from(k);
                    corners.push((c.verts[k], c.cells[k], c.section));
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
    let (shared, seams) = borders(w, sources.iter().map(|s| s.1.as_slice()));
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
        .map(|(sections, corners)| {
            let _z = crate::profile::zone!("dag.build_merged_stage");
            let input = input(
                corners,
                |v, cell, section| (cell, attrs(v), section),
                &shared,
                &seams,
            );
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

/// Positions the stages share, which stay locked, and uv and section seams, which stay
/// protected.
fn borders<'a>(
    w: &Welded,
    stages: impl Iterator<Item = &'a [Corner]> + Clone,
) -> (FxHashSet<[i32; 3]>, FxHashSet<[i32; 3]>) {
    let shared = lod::shared_cells(stages.clone().map(|c| c.iter().map(|&(_, cell, _)| cell)));
    let mut seams = lod::seam_cells(w, stages.clone().flatten().map(|&(v, cell, _)| (v, cell)));
    for corners in stages {
        let mut section_of: FxHashMap<[i32; 3], u16> = FxHashMap::default();
        for &(_, cell, section) in corners {
            if *section_of.entry(cell).or_insert(section) != section {
                seams.insert(cell);
            }
        }
    }
    (shared, seams)
}

/// A compact `clodBuild` input over `corners`, one vertex per distinct `key`; corners with
/// different keys never share a vertex.
fn input<K: std::hash::Hash + Eq>(
    corners: &[Corner],
    key: impl Fn(u32, [i32; 3], u16) -> K,
    shared: &FxHashSet<[i32; 3]>,
    seams: &FxHashSet<[i32; 3]>,
) -> Input {
    let mut local_of: FxHashMap<K, u32> = FxHashMap::default();
    let mut input = Input {
        welded: Vec::new(),
        cells: Vec::new(),
        sections: Vec::new(),
        lock: Vec::new(),
        indices: Vec::with_capacity(corners.len()),
    };
    for &(v, cell, section) in corners {
        let l = *local_of.entry(key(v, cell, section)).or_insert_with(|| {
            input.welded.push(v);
            input.cells.push(cell);
            input.sections.push(section);
            input
                .lock
                .push(lod::vertex_flags(cell, shared, seams).bits());
            input.welded.len() as u32 - 1
        });
        input.indices.push(l);
    }
    input
}

/// Groups at or over `m` become terminal; clusters made from them go, and so do groups
/// left without clusters.
fn truncate(groups: &mut Vec<DagGroup>, m: f32) {
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
        if !g.clusters.is_empty() {
            new_index[i] = next;
            next += 1;
        }
    }
    groups.retain(|g| !g.clusters.is_empty());
    for c in groups.iter_mut().flat_map(|g| &mut g.clusters) {
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
    };
    super::clod::build(&config, mesh, &mut positions, |e| {
        group_of(&e, &input.welded, &input.sections, snap, max_indices)
    })
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
    let mut local_vertices = vec![0u32; max_indices];
    let mut local_triangles = vec![0u8; max_indices];
    let mut out = Vec::new();
    for k in e.clusters {
        let (refined, indices) = (k.refined, k.corners(e.indices));
        assert!(indices.len() <= max_indices, "cluster over its limits");
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
        let mut pieces: Vec<(u16, Vec<u32>, Vec<u8>)> = Vec::new();
        if tris
            .chunks_exact(3)
            .all(|t| section_of(t) == section_of(tris))
        {
            pieces.push((section_of(tris), local.to_vec(), tris.to_vec()));
        } else {
            let mut by_section: BTreeMap<u16, Vec<u8>> = BTreeMap::new();
            for t in tris.chunks_exact(3) {
                by_section
                    .entry(section_of(t))
                    .or_default()
                    .extend_from_slice(t);
            }
            for (section, corners) in by_section {
                let mut new_of = [u16::MAX; 256];
                let mut used = Vec::new();
                let tris = corners
                    .iter()
                    .map(|&k| {
                        let k = usize::from(k);
                        if new_of[k] == u16::MAX {
                            new_of[k] = used.len() as u16;
                            used.push(local[k]);
                        }
                        new_of[k] as u8
                    })
                    .collect();
                pieces.push((section, used, tris));
            }
        }
        for (section, used, tris) in pieces {
            let cells = used
                .iter()
                .map(|&v| {
                    let p = &positions[v as usize * 3..v as usize * 3 + 3];
                    snap.cell([p[0], p[1], p[2]])
                })
                .collect();
            out.push(DagCluster {
                verts: used.iter().map(|&v| welded[v as usize]).collect(),
                cells,
                tris,
                refined,
                section,
            });
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

/// A group with its cooked clusters, by global group index (stages in order, each stage's
/// groups in emission order).
struct Cooked {
    stage: usize,
    depth: u32,
    sphere: Sphere,
    error: f32,
    clusters: Vec<ClusterData>,
    /// Per cluster: the global group it was simplified from, or -1.
    refined: Vec<i32>,
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
    let stages = stages(
        cx.w,
        cx.grid,
        cx.parts,
        s,
        &flags,
        limits,
        cx.w.skin.is_some(),
    );
    let mut first = Vec::with_capacity(stages.len());
    let mut flat = Vec::new();
    for (si, st) in stages.iter().enumerate() {
        first.push(flat.len());
        flat.extend(st.groups.iter().map(|g| (si, g)));
    }
    let mut groups: Vec<Cooked> = flat
        .par_iter()
        .map(|&(si, g)| {
            let _z = crate::profile::zone!("dag.group_clusters");
            let st = &stages[si];
            let clusters = g
                .clusters
                .iter()
                .map(|c| {
                    let tris = c.tris.clone();
                    let mut d = cx.cluster(c.section, st.part, st.merged, &c.verts, &c.cells, tris);
                    let r = &mut d.record;
                    (r.self_sphere, r.self_error) = match usize::try_from(c.refined) {
                        Ok(from) => (st.groups[from].sphere, st.groups[from].error),
                        Err(_) => (r.cull_sphere, st.base),
                    };
                    (r.parent_sphere, r.parent_error) = (g.sphere, g.error);
                    d
                })
                .collect();
            Cooked {
                stage: si,
                depth: g.depth,
                sphere: g.sphere,
                error: g.error,
                clusters,
                refined: g
                    .clusters
                    .iter()
                    .map(|c| usize::try_from(c.refined).map_or(-1, |r| (first[si] + r) as i32))
                    .collect(),
            }
        })
        .collect();
    drop(flat);
    cx.omm_all(groups.iter_mut().flat_map(|g| &mut g.clusters).collect());
    {
        let _z = crate::profile::zone!("dag.fit");
        groups
            .par_iter_mut()
            .try_for_each(|g| fit(cx, &mut g.clusters))?;
    }
    super::thickness::assign(
        groups.iter_mut().flat_map(|g| &mut g.clusters).collect(),
        cx.grid,
        cx.sections,
    );

    let _z = crate::profile::zone!("dag.pages");
    // Terminal groups first and on pages of their own, then merged stages, then coarse to
    // fine.
    let terminal = |g: &Cooked| g.error == f32::INFINITY;
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&i| {
        let g = &groups[i];
        (
            !terminal(g),
            !stages[g.stage].merged,
            std::cmp::Reverse(g.depth),
            i,
        )
    });
    let mut table_index = vec![0u32; groups.len()];
    for (t, &g) in order.iter().enumerate() {
        table_index[g] = t as u32;
    }
    let mut page_groups: Vec<Vec<usize>> = Vec::new();
    let mut page_of = vec![0u32; groups.len()];
    let mut pinned_pages = 0;
    let (mut size, mut aligned) = (header, false);
    for (k, &gi) in order.iter().enumerate() {
        let g = &groups[gi];
        let omm = g.clusters.iter().any(|c| c.omm.is_some());
        let bytes = |aligned: bool| {
            g.clusters.iter().map(|c| c.cost(cx.stride)).sum::<usize>()
                + if omm && !aligned {
                    OMM_ALIGN as usize - 1
                } else {
                    0
                }
        };
        let block_end = k > 0 && terminal(&groups[order[k - 1]]) && !terminal(g);
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
    if order.last().is_some_and(|&g| terminal(&groups[g])) {
        pinned_pages = page_groups.len() as u32;
    }

    // Each page depends on the pages holding the coarser clusters its groups simplify into.
    let mut into: Vec<Vec<u32>> = vec![Vec::new(); groups.len()];
    for (gi, g) in groups.iter().enumerate() {
        for &r in &g.refined {
            if let Ok(r) = usize::try_from(r) {
                into[r].push(page_of[gi]);
            }
        }
    }
    let mut pages = Vec::with_capacity(page_groups.len());
    let mut page_info = Vec::with_capacity(page_groups.len());
    let mut page_deps = Vec::new();
    let mut group_table: Vec<Group> = vec![bytemuck::Zeroable::zeroed(); groups.len()];
    // Per page: its resident length, its cluster count and its stored bytes.
    let built: Vec<(usize, usize, Vec<u8>)> = page_groups
        .par_iter()
        .map(|members| {
            let clusters: Vec<&ClusterData> =
                members.iter().flat_map(|&g| &groups[g].clusters).collect();
            let bytes = write_page(&clusters, cx.stride);
            let stored = store_page(&bytes).map_err(|e| e.to_string())?;
            Ok((bytes.len(), clusters.len(), stored))
        })
        .collect::<Result<_, String>>()?;
    for (p, (members, (resident_len, cluster_count, stored))) in
        page_groups.iter().zip(built).enumerate()
    {
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
        page_info.push(PageInfo {
            blob: p as u32 + 1,
            resident_len: resident_len as u32,
            cluster_count: cluster_count as u16,
            _pad: 0,
            deps: Span {
                first: page_deps.len() as u32,
                count: deps.len() as u32,
            },
        });
        page_deps.extend(deps);
        pages.push(stored);
        let mut first_cluster = 0u16;
        for &gi in members {
            let g = &groups[gi];
            group_table[table_index[gi] as usize] = Group {
                simplified: g.sphere,
                error: g.error,
                page: p as u32,
                first_cluster,
                cluster_count: g.clusters.len() as u16,
                depth: u8::try_from(g.depth).map_err(|_| format!("DAG depth {}", g.depth))?,
                _pad: [0; 3],
            };
            first_cluster += g.clusters.len() as u16;
        }
    }

    let mut part_bounds = vec![Aabb::EMPTY; cx.parts.table.len()];
    for c in groups.iter().flat_map(|g| &g.clusters) {
        if c.record.flags & Cluster::MERGED == 0 {
            c.extend(&mut part_bounds[usize::from(c.record.part)], cx.grid);
        }
    }

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
    let stage_of = |part: usize, section: usize, merged: bool| {
        stages
            .iter()
            .position(|st| {
                usize::from(st.part) == part
                    && usize::from(st.sections[0]) == section
                    && st.merged == merged
            })
            .map_or(MeshHeader::NO_ROOT, |si| stage_root[si])
    };
    let mut roots: Vec<u32> = (0..cx.parts.table.len() * sections)
        .map(|i| stage_of(i / sections, i % sections, false))
        .collect();
    let merge = merge_error(s, cx.parts);
    if merge.is_some() {
        roots.extend((0..sections).map(|sec| stage_of(0, sec, true)));
    }
    let (nodes, roots) = breadth_first(&nodes, &roots);
    let (nodes, node_pages, node_blobs) =
        if nodes.len() * size_of::<Node>() <= s.hierarchy_resident_max as usize {
            (nodes, Vec::new(), Vec::new())
        } else {
            let per_page = NODE_PAGE_MAX / size_of::<Node>();
            let (table, blobs) = node_pages(&nodes, &roots, pages.len() as u32 + 1, per_page);
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
        parts: Vec::new(),
        part_levels: Vec::new(),
        levels: Vec::new(),
        nodes,
        groups: group_table,
        node_pages,
        roots,
        pages: page_info,
        page_deps,
    };
    Ok(Built {
        info,
        pages,
        node_pages: node_blobs,
        part_bounds,
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
/// contiguous and the roots come first; the roots renumbered alike.
fn breadth_first(nodes: &[Node], roots: &[u32]) -> (Vec<Node>, Vec<u32>) {
    let mut order: Vec<u32> = roots
        .iter()
        .copied()
        .filter(|&r| r != MeshHeader::NO_ROOT)
        .collect();
    let mut new_of = vec![u32::MAX; nodes.len()];
    let mut i = 0;
    while i < order.len() {
        let n = &nodes[order[i] as usize];
        new_of[order[i] as usize] = i as u32;
        if n.group < 0 {
            order.extend(n.child_first..n.child_first + u32::from(n.child_count));
        }
        i += 1;
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
    (out, roots)
}

/// Cuts breadth-first nodes into pages of at most `per_page` nodes: the roots open the top
/// page, and each node's children join the current page when they fit and the page holds
/// their parent or has their parent's page as its parent. Returns the page table, with
/// blob indices from `first_blob`, and the page blobs.
fn node_pages(
    nodes: &[Node],
    roots: &[u32],
    first_blob: u32,
    per_page: usize,
) -> (Vec<NodePage>, Vec<Vec<u8>>) {
    let top = roots.iter().filter(|&&r| r != MeshHeader::NO_ROOT).count();
    assert!(top <= per_page, "the roots fit the top page");
    let mut table = vec![NodePage {
        first_node: 0,
        node_count: top as u32,
        blob: first_blob,
        parent_page: NodePage::TOP,
    }];
    let mut page_of = vec![0u32; nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        if n.group >= 0 {
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
    (table, blobs)
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

    #[test]
    fn node_pages_keep_siblings_and_parents_together() {
        let (nodes, roots) = breadth_first(&tree(4, 4), &[0]);
        for per_page in [5, 7, 16, 64] {
            let (table, blobs) = node_pages(&nodes, &roots, 10, per_page);
            assert!(table.len() > 1);
            assert_eq!(table[0].parent_page, NodePage::TOP);
            let mut next = 0;
            for (i, p) in table.iter().enumerate() {
                assert_eq!(p.first_node, next, "pages are contiguous");
                assert!(p.node_count as usize <= per_page);
                assert_eq!(p.blob, 10 + i as u32);
                assert_eq!(blobs[i].len(), p.node_count as usize * size_of::<Node>());
                next += p.node_count;
            }
            assert_eq!(next as usize, nodes.len());
            let page_of = |n: u32| {
                table
                    .iter()
                    .position(|p| (p.first_node..p.first_node + p.node_count).contains(&n))
                    .unwrap()
            };
            for (i, n) in nodes.iter().enumerate() {
                if n.group >= 0 {
                    continue;
                }
                let first = page_of(n.child_first);
                assert_eq!(first, page_of(n.child_first + u32::from(n.child_count) - 1));
                let parent = page_of(i as u32);
                assert!(parent == first || table[first].parent_page as usize == parent);
            }
        }
    }
}
