//! Cluster LOD DAGs: cuts, the merged stage, dilation, pages and the hierarchy, on meshes
//! handed over in memory, and static displacement through glTF.

#[allow(dead_code)]
#[path = "common/dag.rs"]
mod dag_read;
#[allow(dead_code)]
#[path = "common/glb.rs"]
mod glb;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use dag_read::{Cut, Dag, coverage, covers_once, load, read};
use glb::Glb;
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MeshInput, MeshParts, MeshSettings,
    Source, StaticDisplacement,
};
use orr_math::{Vec2, Vec3};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{Cluster, MeshHeader, NodePage};
use orr_pack::model::ModelInfo;
use orr_pack::{Archive, AssetId, Span, Target};

fn cook_archive(name: &str, source: Source) -> Archive {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: name.into(),
                source,
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let a = Archive::from_memory(bytes).unwrap();
    for e in a.entries() {
        a.verify(e).unwrap();
    }
    a
}

fn cook(mesh: MeshInput, flags: &[u8], settings: MeshSettings) -> Archive {
    let materials = flags
        .iter()
        .map(|&double_sided| Material {
            double_sided,
            ..Material::default()
        })
        .collect();
    cook_archive(
        "m",
        Source::Mesh {
            mesh,
            materials,
            settings,
        },
    )
}

fn dag() -> MeshSettings {
    MeshSettings {
        lod: LodMode::Dag,
        ..MeshSettings::default()
    }
}

/// Edges of the selected triangles, by position, that are not used once in each direction;
/// edges with both ends on the `rim` may be used once.
fn unmatched_edges(d: &Dag, selected: &[usize], rim: impl Fn([f32; 3]) -> bool) -> usize {
    // Per edge: uses, and forward minus backward uses.
    let mut uses: HashMap<([u32; 3], [u32; 3]), (u32, i32)> = HashMap::new();
    for &c in selected {
        for t in &d.clusters[c].tris {
            for k in 0..3 {
                let (a, b) = (t[k].map(f32::to_bits), t[(k + 1) % 3].map(f32::to_bits));
                let (key, dir) = if a < b { ((a, b), 1) } else { ((b, a), -1) };
                let e = uses.entry(key).or_default();
                e.0 += 1;
                e.1 += dir;
            }
        }
    }
    uses.iter()
        .filter(|&(e, &(n, balance))| {
            let on_rim = rim(e.0.map(f32::from_bits)) && rim(e.1.map(f32::from_bits));
            !(n == 2 && balance == 0 || n == 1 && on_rim)
        })
        .count()
}

/// A wavy grid of `n` x `n` quads over 4 m in xz; two sections split at z = 0 when `split`.
fn wavy(n: u32, split: bool) -> MeshInput {
    let mut m = MeshInput::default();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            let x = (u - 0.5) * 4.0;
            let z = (v - 0.5) * 4.0;
            m.positions
                .push(Vec3::new(x, 0.15 * (x * 2.3).sin() * (z * 1.7).cos(), z));
            m.normals.push(Vec3::Y);
        }
    }
    m.uvs = Some(
        (0..=n)
            .flat_map(|j| (0..=n).map(move |i| Vec2::new(i as f32 / n as f32, j as f32 / n as f32)))
            .collect(),
    );
    let w = n + 1;
    let (mut lower, mut upper) = (Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            let quad = [c, c + w, c + 1, c + 1, c + w, c + w + 1];
            if split && j >= n / 2 {
                &mut upper
            } else {
                &mut lower
            }
            .extend(quad);
        }
    }
    m.sections = vec![Span {
        first: 0,
        count: lower.len() as u32,
    }];
    if split {
        m.sections.push(Span {
            first: lower.len() as u32,
            count: upper.len() as u32,
        });
    }
    m.indices = [lower, upper].concat();
    m
}

/// Checks every structural rule of a DAG mesh; returns the part stage's triangles per depth.
fn check_structure(d: &Dag) -> Vec<u64> {
    let info = &d.info;
    assert!(info.levels.is_empty() && info.part_levels.is_empty());
    assert_eq!(info.header.merge_level, MeshHeader::NO_MERGE_LEVEL);
    // Groups partition each page's clusters, in order; a group's clusters share its error.
    let mut seen = vec![false; d.clusters.len()];
    for g in &info.groups {
        for c in d.group_of(g) {
            assert!(!seen[c]);
            seen[c] = true;
            let r = &d.clusters[c].record;
            assert_eq!(r.parent_error.to_bits(), g.error.to_bits());
            assert_eq!(r.parent_sphere, g.simplified);
            assert!(r.self_error <= r.parent_error);
            assert_eq!(d.clusters[c].page, g.page as usize);
        }
    }
    assert!(seen.iter().all(|&s| s), "every cluster is in a group");
    // Pinned pages hold the terminal groups; dependencies point at earlier pages.
    let pinned = info.header.pinned_pages as usize;
    assert!(pinned > 0);
    for g in &info.groups {
        assert_eq!(
            g.error == f32::INFINITY,
            (g.page as usize) < pinned,
            "{g:?}"
        );
    }
    for (p, page) in info.pages.iter().enumerate() {
        let deps = &info.page_deps[page.deps.first as usize..][..page.deps.count as usize];
        assert!(
            deps.iter().all(|&q| (q as usize) < p),
            "page {p} deps {deps:?}"
        );
        assert!(deps.is_sorted());
        if p < pinned {
            assert!(deps.is_empty());
        }
    }
    // Every node's children are in range; every group has one leaf.
    let mut leaves = vec![0; info.groups.len()];
    for n in &d.nodes {
        if n.group >= 0 {
            leaves[n.group as usize] += 1;
        } else {
            assert!(n.child_count > 0);
            assert!(n.child_first as usize + usize::from(n.child_count) <= d.nodes.len());
        }
    }
    assert!(leaves.iter().all(|&l| l == 1));
    // A cluster names the group it was simplified from, one step finer, whose bounds and error
    // are its own and whose page depends on its page; every group a cut can leave has some.
    let mut simplified = vec![0; info.groups.len()];
    for g in &info.groups {
        for c in d.group_of(g) {
            let r = &d.clusters[c].record;
            if r.from_group == Cluster::SOURCE {
                assert_eq!(g.depth, 0, "a source cluster");
                continue;
            }
            let from = &info.groups[r.from_group as usize];
            simplified[r.from_group as usize] += 1;
            assert_eq!(from.depth + 1, g.depth);
            assert_eq!(from.simplified, r.self_sphere);
            assert_eq!(from.error.to_bits(), r.self_error.to_bits());
            let deps = &info.pages[from.page as usize].deps;
            let deps = &info.page_deps[deps.first as usize..][..deps.count as usize];
            assert!(from.page == g.page || deps.contains(&g.page));
        }
    }
    for (g, &n) in info.groups.iter().zip(&simplified) {
        assert_eq!(n > 0, g.error.is_finite(), "{g:?}");
    }
    // A group's sphere holds the stored positions of its clusters and of the clusters
    // simplified from it, and the spheres of the groups its clusters come from: the traversal
    // tests it for both sides of the group's replacement.
    let center = |s: &orr_pack::Sphere| s.center.map(f64::from);
    let apart =
        |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt();
    let holds = |s: &orr_pack::Sphere, c: usize| {
        d.clusters[c]
            .tris
            .iter()
            .flatten()
            .all(|p| apart(p.map(f64::from), center(s)) <= f64::from(s.radius))
    };
    for g in &info.groups {
        for c in d.group_of(g) {
            assert!(holds(&g.simplified, c), "{g:?} holds cluster {c}");
            let from = d.clusters[c].record.from_group;
            if from == Cluster::SOURCE {
                continue;
            }
            let f = &info.groups[from as usize];
            assert!(
                holds(&f.simplified, c),
                "{f:?} holds cluster {c} simplified from it"
            );
            let reach = apart(center(&f.simplified), center(&g.simplified))
                + f64::from(f.simplified.radius);
            assert!(reach <= f64::from(g.simplified.radius), "{g:?} holds {f:?}");
        }
    }
    // A leaf's bounds are its group's, bit for bit, which the traversal's leaf and cluster
    // tests both read.
    for n in &d.nodes {
        if let Ok(g) = usize::try_from(n.group) {
            let g = &info.groups[g];
            assert_eq!(n.sphere, g.simplified);
            assert_eq!(n.max_error.to_bits(), g.error.to_bits());
        }
    }
    // An internal node's sphere holds its children's and its error is at least theirs, so
    // the traversal's test of a node stands for every group under it.
    for n in d.nodes.iter().filter(|n| n.group < 0) {
        let children = &d.nodes[n.child_first as usize..][..usize::from(n.child_count)];
        for c in children {
            let reach = apart(center(&c.sphere), center(&n.sphere)) + f64::from(c.sphere.radius);
            assert!(reach <= f64::from(n.sphere.radius), "{n:?} holds {c:?}");
            assert!(c.max_error <= n.max_error, "{n:?} over {c:?}");
        }
    }
    // A group names its leaf's node page, the top one for a group in a pinned page.
    for (i, n) in d.nodes.iter().enumerate() {
        let Ok(g) = usize::try_from(n.group) else {
            continue;
        };
        let page = info
            .node_pages
            .iter()
            .position(|p| (p.first_node..p.first_node + p.node_count).contains(&(i as u32)))
            .unwrap_or(0);
        let g = &info.groups[g];
        assert_eq!(g.node_page as usize, page);
        assert!(g.page as usize >= pinned || page == 0, "{g:?}");
    }
    d.part_triangles_per_depth()
}

fn eyes() -> [Vec3; 3] {
    [
        Vec3::new(-2.5, 0.6, -2.5),
        Vec3::new(0.0, 0.4, 0.0),
        Vec3::new(3.0, 1.5, 1.0),
    ]
}

#[test]
fn cuts_cover_the_surface_once() {
    let input = wavy(96, true);
    let source = input.indices.len() as u64 / 3;
    let a = cook(input, &[0, 0], dag());
    let d = load(&a, "m/mesh");
    let per_depth = check_structure(&d);
    assert_eq!(per_depth[0], source);
    assert!(per_depth.len() >= 4, "{per_depth:?}");
    for w in per_depth.windows(2) {
        assert!(w[1] as f64 <= 0.86 * w[0] as f64, "{per_depth:?}");
    }
    assert_eq!(d.info.roots.len(), 2);
    assert_eq!(d.info.header.merge_error, f32::INFINITY);
    for c in &d.clusters {
        if c.record.self_error == 0.0 {
            assert_eq!(c.record.self_sphere, c.record.cull_sphere);
        }
    }

    // Cuts up to 1% of the size: coarser ones lose area to simplification, not to cracks.
    let inside = |_: Vec2| true;
    let mut cuts: Vec<Cut> = d.thresholds(0.04).into_iter().map(Cut::Uniform).collect();
    for eye in eyes() {
        for t in [0.0005, 0.002, 0.01] {
            cuts.push(Cut::View(eye, t));
        }
    }
    let mut levels = HashSet::new();
    for cut in cuts {
        let flat = d.select(cut, |_| true);
        assert_eq!(d.traverse(cut, &d.info.roots), flat, "{cut:?}");
        let wrong = covers_once(&d, &flat, Vec2::splat(-1.98), 3.96, inside);
        assert_eq!(wrong, 0, "{cut:?}");
        let rim = |p: [f32; 3]| p[0].abs() == 2.0 || p[2].abs() == 2.0;
        assert_eq!(unmatched_edges(&d, &flat, rim), 0, "{cut:?}");
        if let Cut::View(..) = cut {
            let depths: HashSet<u32> = flat
                .iter()
                .map(|&c| d.clusters[c].record.self_error.to_bits())
                .collect();
            levels.insert(depths.len() > 1);
        }
    }
    assert!(levels.contains(&true), "a view cut mixes levels");

    // The two sections are one stage: the row they share coarsens with the rest.
    assert_eq!(d.info.roots[1], MeshHeader::NO_ROOT);
    let seam = |cut: Cut| {
        let flat = d.select(cut, |_| true);
        let side = |s: u16| -> HashSet<[u32; 3]> {
            flat.iter()
                .filter(|&&c| d.clusters[c].record.section == s)
                .flat_map(|&c| d.clusters[c].tris.iter().flatten())
                .map(|p| p.map(f32::to_bits))
                .collect()
        };
        side(0).intersection(&side(1)).count()
    };
    let (finest, coarsest) = (seam(Cut::Uniform(0.0)), seam(Cut::Uniform(f32::MAX)));
    assert!(
        coarsest * 4 < finest,
        "{coarsest} of {finest} seam positions"
    );
}

/// A `size` m square of `n` x `n` quads over hills, every point moved off the coarser grids
/// by under a millimeter, the rim kept on its square.
fn hills(n: u32, size: f32) -> MeshInput {
    let mut m = wavy(n, false);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut jitter = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 * 0.0008 - 0.0004
    };
    let half = size / 2.0;
    for p in &mut m.positions {
        let (x, z) = (p.x * size / 4.0, p.z * size / 4.0);
        let rim = x.abs() == half || z.abs() == half;
        let y = 0.05 * size * (x / size * 7.0).sin() * (z / size * 5.0).cos();
        *p = if rim {
            Vec3::new(x, y + jitter(), z)
        } else {
            Vec3::new(x + jitter(), y + jitter(), z + jitter())
        };
    }
    m
}

/// A 200 m DAG mesh stays on the 2^-11 m grid: its finest clusters are the source triangles
/// on it, every point of every level is a source point on it, the coarse clusters span more
/// than 16 bits of it, and its cuts cover the surface once with no open edge inside.
#[test]
fn a_large_mesh_keeps_the_finest_step_at_every_level() {
    let input = hills(96, 200.0);
    let a = cook(input.clone(), &[0], dag());
    let d = load(&a, "m/mesh");
    assert_eq!(d.info.header.position_exponent, -11);
    let per_depth = check_structure(&d);
    assert_eq!(per_depth[0], input.indices.len() as u64 / 3);
    let snap = |p: Vec3| {
        p.to_array()
            .map(|x| ((f64::from(x) * 2048.0 + 0.5).floor() / 2048.0) as f32)
            .map(f32::to_bits)
    };
    let source: HashSet<[u32; 3]> = input.positions.iter().map(|&p| snap(p)).collect();
    for c in &d.clusters {
        let points = c.tris.iter().flatten();
        assert!(
            points
                .map(|p| p.map(f32::to_bits))
                .all(|p| source.contains(&p))
        );
    }
    assert!(d.clusters.iter().any(|c| c.record.wide()));
    let rotate = |t: [[u32; 3]; 3]| {
        let r = (0..3).min_by_key(|&i| t[i]).unwrap();
        [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
    };
    let mut expected: Vec<[[u32; 3]; 3]> = input
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| rotate(t.map(|i| snap(input.positions[i as usize]))))
        .collect();
    let mut finest: Vec<[[u32; 3]; 3]> = d
        .select(Cut::Uniform(0.0), |_| true)
        .iter()
        .flat_map(|&c| &d.clusters[c].tris)
        .map(|t| rotate(t.map(|p| p.map(f32::to_bits))))
        .collect();
    expected.sort_unstable();
    finest.sort_unstable();
    assert_eq!(finest, expected);

    let mut cuts: Vec<Cut> = d.thresholds(2.0).into_iter().map(Cut::Uniform).collect();
    for eye in eyes() {
        for t in [0.0005, 0.002, 0.01] {
            cuts.push(Cut::View(eye * 50.0, t));
        }
    }
    for cut in cuts {
        let flat = d.select(cut, |_| true);
        assert_eq!(d.traverse(cut, &d.info.roots), flat, "{cut:?}");
        let wrong = covers_once(&d, &flat, Vec2::splat(-99.0), 198.0, |_| true);
        assert_eq!(wrong, 0, "{cut:?}");
        let rim = |p: [f32; 3]| p[0].abs() == 100.0 || p[2].abs() == 100.0;
        assert_eq!(unmatched_edges(&d, &flat, rim), 0, "{cut:?}");
    }
}

#[test]
fn parts_merge_past_the_merge_error() {
    let mut input = wavy(64, true);
    x_parts(&mut input, 5);
    let merge = 0.004;
    let settings = MeshSettings {
        merge_error: Some(merge),
        ..dag()
    };
    let a = cook(input, &[0, 0], settings);
    let d = load(&a, "m/mesh");
    check_structure(&d);
    assert_eq!(d.info.header.merge_error, merge);
    assert_eq!(d.info.parts.len(), 5);
    // The two sections cluster alike, so each stage covers both from its first section's
    // slot.
    assert_eq!(d.info.roots.len(), 5 * 2 + 2);
    for (i, &r) in d.info.roots.iter().enumerate() {
        assert_eq!(r != MeshHeader::NO_ROOT, i % 2 == 0, "root {i}");
    }
    for c in &d.clusters {
        let r = &c.record;
        if r.flags & Cluster::MERGED != 0 {
            assert_eq!(r.part, 0);
            assert_eq!(r.flags & Cluster::ANIMATED_PART, 0);
            assert!(r.self_error >= merge);
        } else {
            assert!(r.self_error < merge, "{}", r.self_error);
            assert_eq!(r.flags & Cluster::ANIMATED_PART != 0, r.part != 0);
        }
    }
    let part_roots = &d.info.roots[..10];
    let merged_roots = &d.info.roots[10..];
    let part_stage = |c: &Cluster| c.flags & Cluster::MERGED == 0;
    let merged_stage = |c: &Cluster| c.flags & Cluster::MERGED != 0;
    let inside = |_: Vec2| true;
    // The part stage at any threshold, its coarsest cut included, and the merged stage
    // from the merge error, each cover the surface once.
    for t in [0.0, 0.001, merge * 0.999, 1.0] {
        let cut = Cut::Uniform(t);
        let flat = d.select(cut, part_stage);
        assert_eq!(d.traverse(cut, part_roots), flat);
        assert_eq!(
            covers_once(&d, &flat, Vec2::splat(-1.98), 3.96, inside),
            0,
            "{t}"
        );
    }
    let coarsest = d.select(Cut::Uniform(f32::MAX), part_stage);
    assert!(
        coarsest
            .iter()
            .all(|&c| d.clusters[c].record.parent_error == f32::INFINITY)
    );
    // View cuts that ask no finer than the merged stage's source anywhere.
    let mut merged_cuts: Vec<Cut> = [merge, merge * 2.0, 0.05, f32::MAX]
        .into_iter()
        .map(Cut::Uniform)
        .collect();
    for eye in [
        Vec3::new(-2.5, 0.6, -2.5),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(3.0, 1.5, 1.0),
    ] {
        merged_cuts.push(Cut::View(eye, 0.01));
    }
    for cut in merged_cuts {
        let flat = d.select(cut, merged_stage);
        assert_eq!(d.traverse(cut, merged_roots), flat);
        assert_eq!(
            covers_once(&d, &flat, Vec2::splat(-1.98), 3.96, inside),
            0,
            "{cut:?}"
        );
    }
    let tris = |sel: &[usize]| -> u64 {
        sel.iter()
            .map(|&c| u64::from(d.clusters[c].record.tri_count))
            .sum()
    };
    let merged_coarsest = d.select(Cut::Uniform(f32::MAX), merged_stage);
    assert!(tris(&merged_coarsest) < tris(&coarsest));
}

/// A part's sections come from the clusters of every page: single quads of a second section
/// simplify away before the terminal groups, so only streamed pages hold them.
#[test]
fn part_sections_cover_streamed_clusters() {
    let n = 64;
    let mut input = wavy(n, false);
    x_parts(&mut input, 3);
    let (mut plain, mut islands) = (Vec::new(), Vec::new());
    for (k, quad) in input.indices.chunks(6).enumerate() {
        let (i, j) = (k as u32 % n, k as u32 / n);
        if i % 13 == 5 && j % 17 == 7 {
            islands.extend(quad);
        } else {
            plain.extend(quad);
        }
    }
    input.sections = vec![
        Span {
            first: 0,
            count: plain.len() as u32,
        },
        Span {
            first: plain.len() as u32,
            count: islands.len() as u32,
        },
    ];
    input.indices = [plain, islands].concat();
    let a = cook(input, &[0, 0], dag());
    let d = load(&a, "m/mesh");
    let pinned = d.info.header.pinned_pages as usize;
    let mut sections = [0u32; 3];
    let mut pinned_sections = [0u32; 3];
    for c in &d.clusters {
        let r = &c.record;
        if r.flags & Cluster::ANIMATED_PART != 0 {
            sections[usize::from(r.part)] |= 1 << r.section;
            if c.page < pinned {
                pinned_sections[usize::from(r.part)] |= 1 << r.section;
            }
        }
    }
    assert_eq!(sections, [0, 3, 3]);
    assert_ne!(
        pinned_sections, sections,
        "some islands are on streamed pages alone"
    );
    let cooked: Vec<u32> = d.info.parts.iter().map(|p| p.sections).collect();
    assert_eq!(cooked, sections);
}

/// Splits a [`wavy`] mesh into `count` parts, in equal bands along x.
fn x_parts(input: &mut MeshInput, count: u16) {
    let values = input
        .positions
        .iter()
        .map(|p| ((p.x + 2.0) / 4.0 * (f32::from(count) - 0.001)).floor() as u16)
        .collect();
    input.parts = Some(MeshParts {
        values,
        count,
        by_section: false,
        pivots: None,
    });
}

/// The cluster vertex positions of a cooked DAG, as bits.
fn positions(d: &Dag, keep: impl Fn(&Cluster) -> bool) -> HashSet<[u32; 3]> {
    d.clusters
        .iter()
        .filter(|c| keep(&c.record))
        .flat_map(|c| c.tris.iter().flatten())
        .map(|p| p.map(f32::to_bits))
        .collect()
}

#[test]
fn dilation_moves_only_open_unlocked_borders() {
    // A closed double-sided sphere has no open border to dilate.
    let mut sphere = MeshInput::default();
    let (rings, spokes) = (40u32, 64u32);
    for r in 0..=rings {
        let th = r as f32 / rings as f32 * std::f32::consts::PI;
        for s in 0..spokes {
            let ph = s as f32 / spokes as f32 * std::f32::consts::TAU;
            let p = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
            sphere.positions.push(p);
            sphere.normals.push(p);
        }
    }
    let at = |r: u32, s: u32| r * spokes + s % spokes;
    for r in 0..rings {
        for s in 0..spokes {
            sphere
                .indices
                .extend([at(r, s), at(r, s + 1), at(r + 1, s)]);
            sphere
                .indices
                .extend([at(r, s + 1), at(r + 1, s + 1), at(r + 1, s)]);
        }
    }
    sphere.sections = vec![Span {
        first: 0,
        count: sphere.indices.len() as u32,
    }];
    // The poles' rows are one point each; weld them by position.
    let source: HashSet<[u32; 3]> = {
        let a = cook(sphere.clone(), &[0], MeshSettings::default());
        let d = load(&a, "m/mesh");
        positions(&d, |_| true)
    };
    let d = load(&cook(sphere, &[1], dag()), "m/mesh");
    let moved = positions(&d, |_| true).difference(&source).count();
    assert_eq!(moved, 0, "a closed mesh keeps its positions");

    // Leaves split across two double-sided sections: their open rims dilate, while the
    // row the sections share is locked and stays at every cut.
    let input = crown(400, 2);
    let source = load(
        &cook(input.clone(), &[0, 0], MeshSettings::default()),
        "m/mesh",
    );
    let shared: HashSet<[u32; 3]> = positions(&source, |c| c.section == 0)
        .intersection(&positions(&source, |c| c.section == 1))
        .copied()
        .collect();
    assert!(shared.len() >= 400 * 5);
    let source = positions(&source, |_| true);
    let d = load(&cook(input, &[1, 1], dag()), "m/mesh");
    check_structure(&d);
    let all = positions(&d, |_| true);
    assert!(all.difference(&source).count() > 0, "open rims dilate");
    for t in d.thresholds(f32::MAX) {
        let cut: HashSet<[u32; 3]> = d
            .select(Cut::Uniform(t), |_| true)
            .into_iter()
            .flat_map(|c| d.clusters[c].tris.iter().flatten())
            .map(|p| p.map(f32::to_bits))
            .collect();
        assert!(shared.is_subset(&cut), "locked rows stay at {t}");
    }
}

#[test]
fn paged_hierarchy_keeps_children_together() {
    let resident = load(&cook(wavy(128, true), &[0, 0], dag()), "m/mesh");
    let settings = MeshSettings {
        hierarchy_resident_max: 0,
        ..dag()
    };
    let paged = load(&cook(wavy(128, true), &[0, 0], settings), "m/mesh");
    check_structure(&paged);
    assert!(resident.info.node_pages.is_empty());
    assert_eq!(resident.nodes, paged.nodes, "paging keeps the node order");
    let info = &paged.info;
    assert!(!info.node_pages.is_empty());
    let top = &info.node_pages[0];
    assert_eq!(top.parent_page, NodePage::TOP);
    assert!(
        info.roots
            .iter()
            .all(|&r| r == MeshHeader::NO_ROOT || r < top.node_count)
    );
    let page_of = |n: u32| {
        info.node_pages
            .iter()
            .position(|p| (p.first_node..p.first_node + p.node_count).contains(&n))
            .unwrap()
    };
    for (i, n) in paged.nodes.iter().enumerate() {
        if n.group >= 0 {
            continue;
        }
        let first = page_of(n.child_first);
        assert_eq!(first, page_of(n.child_first + u32::from(n.child_count) - 1));
        let parent = page_of(i as u32);
        let page = &info.node_pages[first];
        assert!(parent == first || page.parent_page as usize == parent);
    }
    for cut in [
        Cut::Uniform(0.001),
        Cut::View(Vec3::new(0.0, 0.5, -3.0), 0.002),
    ] {
        assert_eq!(
            paged.traverse(cut, &info.roots),
            paged.select(cut, |_| true)
        );
    }
}

/// Leaves of a crown: `count` 4 x 4 quad leaves, curved, in a pseudo-random spread over a
/// 2 m ball; with two sections each leaf's rows from row 2 are the second.
fn crown(count: u32, sections: u32) -> MeshInput {
    let mut m = MeshInput::default();
    let mut upper = Vec::new();
    let mut seed = 0x9E37_79B9u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    for _ in 0..count {
        let c = Vec3::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0) * 0.9;
        let a = Vec3::new(next() - 0.5, next() - 0.5, next() - 0.5).normalize_or(Vec3::X);
        let b = a.any_orthonormal_vector();
        let n = a.cross(b);
        let base = m.positions.len() as u32;
        for j in 0..=4 {
            for i in 0..=4 {
                let (u, v) = (i as f32 / 4.0 - 0.5, j as f32 / 4.0);
                let bend = 0.05 * v * v;
                m.positions
                    .push(c + a * (u * 0.08) + b * (v * 0.16) + n * bend);
                m.normals.push(n);
            }
        }
        for j in 0..4 {
            for i in 0..4 {
                let k = base + j * 5 + i;
                let quad = [k, k + 5, k + 1, k + 1, k + 5, k + 6];
                if sections == 2 && j >= 2 {
                    &mut upper
                } else {
                    &mut m.indices
                }
                .extend(quad);
            }
        }
    }
    m.uvs = None;
    m.sections = vec![Span {
        first: 0,
        count: m.indices.len() as u32,
    }];
    if sections == 2 {
        m.sections.push(Span {
            first: m.indices.len() as u32,
            count: upper.len() as u32,
        });
        m.indices.extend(upper);
    }
    m
}

#[test]
fn double_sided_foliage_keeps_its_coverage() {
    let input = crown(1500, 1);
    let a = cook(input, &[1], dag());
    let d = load(&a, "m/mesh");
    check_structure(&d);
    let tris = |sel: &[usize]| -> Vec<[[f32; 3]; 3]> {
        sel.iter()
            .flat_map(|&c| d.clusters[c].tris.iter().copied())
            .collect()
    };
    let source_tris = tris(&d.select(Cut::Uniform(0.0), |_| true));
    let view = (Vec3::ZERO, 2.4);
    let source = coverage(&source_tris, view, 256);
    let mut levels = 0;
    for t in d.thresholds(f32::MAX) {
        let sel = tris(&d.select(Cut::Uniform(t), |_| true));
        // Coarser levels erode the leaves past what dilation restores.
        if sel.len() * 32 < source_tris.len() {
            break;
        }
        let cov = coverage(&sel, view, 256);
        for (v, (&c, &s)) in cov.iter().zip(&source).enumerate() {
            let ratio = c as f64 / s as f64;
            assert!((0.9..=1.1).contains(&ratio), "view {v} at {t}: {ratio}");
        }
        levels += 1;
    }
    assert!(levels > 2);
}

/// Overlapping thin plates, as a tiled roof lays them: `rows` x `cols` plates of 6 x 4 quads,
/// 30 cm long and 4 mm thick at the middle, each a top and an underside that meet at its rim,
/// where each side has its own normal and uvs. Rows step up the slope by half a plate and lie
/// on the row below.
fn plates(rows: u32, cols: u32) -> MeshInput {
    let (nu, nv) = (6u32, 4u32);
    let mut m = MeshInput::default();
    let mut uvs = Vec::new();
    let slope = Vec3::new(0.0, 0.3, 1.0).normalize();
    let across = Vec3::X;
    let up = across.cross(slope).normalize();
    for r in 0..rows {
        for c in 0..cols {
            let origin = across * (0.22 * c as f32 + 0.11 * (r % 2) as f32)
                + slope * (0.15 * r as f32)
                + up * (0.004 * r as f32);
            for side in [1.0f32, -1.0] {
                let base = m.positions.len() as u32;
                for j in 0..=nv {
                    for i in 0..=nu {
                        let (u, v) = (i as f32 / nu as f32, j as f32 / nv as f32);
                        let swell = 4.0 * u * (1.0 - u) * 4.0 * v * (1.0 - v);
                        let p = origin
                            + across * (0.2 * u)
                            + slope * (0.3 * v)
                            + up * side * swell * 0.002;
                        m.positions.push(p);
                        m.normals.push(up * side);
                        uvs.push(Vec2::new(if side > 0.0 { u } else { 1.0 - u }, v));
                    }
                }
                let w = nu + 1;
                for j in 0..nv {
                    for i in 0..nu {
                        let k = base + j * w + i;
                        let quad = [k, k + 1, k + w, k + 1, k + w + 1, k + w];
                        if side > 0.0 {
                            m.indices.extend(quad);
                        } else {
                            m.indices
                                .extend(quad.chunks(3).flat_map(|t| [t[0], t[2], t[1]]));
                        }
                    }
                }
            }
        }
    }
    m.uvs = Some(uvs);
    m.sections = vec![Span {
        first: 0,
        count: m.indices.len() as u32,
    }];
    m
}

#[test]
fn thin_plates_keep_their_normals_on_their_faces() {
    let discrete = MeshSettings {
        lod: LodMode::Discrete {
            errors: vec![0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
        },
        ..MeshSettings::default()
    };
    for settings in [dag(), discrete] {
        let lod = settings.lod.clone();
        let d = load(&cook(plates(16, 16), &[0], settings), "m/mesh");
        if lod == LodMode::Dag {
            let per_depth = check_structure(&d);
            assert!(per_depth.len() >= 5, "{per_depth:?}");
        }
        // Every cluster stands in some cut. A normal opposes its face past what the
        // octahedral encoding moves one.
        let mut opposed = 0;
        for c in &d.clusters {
            for (t, n) in c.tris.iter().zip(&c.normals) {
                let p = t.map(Vec3::from);
                let face = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                opposed += n
                    .iter()
                    .filter(|&&n| Vec3::from(n).dot(face) < -1e-3)
                    .count();
            }
        }
        assert_eq!(opposed, 0, "{lod:?}");
    }
}

/// A rough log: a cylinder 0.6 m long of `around` x `along` quads whose 0.1 m radius varies by
/// `amplitude` of it from vertex to vertex, with smooth normals. Its side is `charts` strips
/// around, each on a cell of its own of an 8 x 8 atlas, with the vertices where they meet apart.
fn log(around: u32, along: u32, charts: u32, amplitude: f32) -> MeshInput {
    let mut seed = 0x2545_F491u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    let bumps: Vec<f32> = (0..around * (along + 1)).map(|_| next()).collect();
    let at = |i: u32, j: u32| {
        let i = i % around;
        let a = i as f32 / around as f32 * std::f32::consts::TAU;
        let r = 0.1 * (1.0 + amplitude * (bumps[(j * around + i) as usize] - 0.5));
        Vec3::new(r * a.cos(), r * a.sin(), 0.6 * j as f32 / along as f32)
    };
    let normal = |i: u32, j: u32| {
        let (l, r) = (at(i + around - 1, j), at(i + 1, j));
        let (d, u) = (at(i, j.saturating_sub(1)), at(i, (j + 1).min(along)));
        (r - l).cross(u - d).normalize()
    };
    let mut m = MeshInput::default();
    let mut uvs = Vec::new();
    let w = around / charts;
    for c in 0..charts {
        let base = m.positions.len() as u32;
        for j in 0..=along {
            for k in 0..=w {
                let i = c * w + k;
                m.positions.push(at(i, j));
                m.normals.push(normal(i, j));
                let cell = Vec2::new((c % 8) as f32, (c / 8) as f32);
                let uv = Vec2::new(k as f32 / w as f32, j as f32 / along as f32);
                uvs.push((cell + Vec2::splat(0.125) + uv * 0.75) / 8.0);
            }
        }
        for j in 0..along {
            for k in 0..w {
                let q = base + j * (w + 1) + k;
                m.indices
                    .extend([q, q + 1, q + w + 1, q + 1, q + w + 2, q + w + 1]);
            }
        }
    }
    m.uvs = Some(uvs);
    m.sections = vec![Span {
        first: 0,
        count: m.indices.len() as u32,
    }];
    m
}

#[test]
fn simplified_triangles_keep_to_one_uv_chart() {
    // Simplification folds the bumps past their normals, and a corner that comes to oppose its
    // face can lie nearest a vertex of the next strip.
    let d = load(&cook(log(96, 144, 6, 0.5), &[0], dag()), "m/mesh");
    let cell = |uv: [f32; 2]| uv.map(|x| (x * 8.0).floor() as i32);
    let mut spanning = 0;
    for c in &d.clusters {
        for uv in &c.uvs {
            let cells = uv.map(cell);
            spanning += usize::from(cells[0] != cells[1] || cells[1] != cells[2]);
        }
    }
    assert_eq!(spanning, 0);
}

#[test]
fn skinned_thin_plates_keep_each_vertex_skin_with_its_position() {
    // Two joints whose weights change through the plates' thickness as well as across them,
    // so a vertex skinned as another of a different position poses apart from its neighbors.
    let m = plates(16, 16);
    let up = Vec3::X.cross(Vec3::new(0.0, 0.3, 1.0)).normalize();
    let weights: Vec<[f32; 4]> = m
        .positions
        .iter()
        .map(|&p| {
            let w = 0.5 + 0.4 * (400.0 * p.dot(up) + 2.0 * p.x).sin();
            [w, 1.0 - w, 0.0, 0.0]
        })
        .collect();
    let mut g = Glb::default();
    let pa = g.floats(
        &m.positions.iter().map(Vec3::to_array).collect::<Vec<_>>(),
        "VEC3",
    );
    let na = g.floats(
        &m.normals.iter().map(Vec3::to_array).collect::<Vec<_>>(),
        "VEC3",
    );
    let uvs = m.uvs.as_ref().unwrap();
    let ta = g.floats(&uvs.iter().map(Vec2::to_array).collect::<Vec<_>>(), "VEC2");
    let joints: Vec<[u16; 4]> = vec![[0, 1, 0, 0]; m.positions.len()];
    let jv = g.view(bytemuck::cast_slice(&joints), None);
    let ja = g.accessor(jv, glb::U16, joints.len(), "VEC4", "");
    let wa = g.floats(&weights, "VEC4");
    let ia = g.indices(&m.indices);
    let path = tmp("dag-skinned-plates").join("plates.glb");
    g.write(
        &path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0,1]}}],
"nodes":[{{"mesh":0,"skin":0}},{{"name":"a","children":[2]}},{{"name":"b"}}],
"skins":[{{"joints":[1,2]}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TEXCOORD_0":{ta},"JOINTS_0":{ja},"WEIGHTS_0":{wa}}},"indices":{ia}}}]}}]"#
        ),
    );
    let settings = GltfSettings {
        mesh: dag(),
        ..Default::default()
    };
    let d = load(
        &cook_archive("g", Source::Gltf { path, settings }),
        "g/mesh",
    );
    let corners = |c: &dag_read::C| {
        c.tris
            .iter()
            .zip(&c.skin)
            .flat_map(|(t, s)| t.iter().zip(s))
            .map(|(p, s)| {
                let (j, w) = s.expect("a skinned mesh");
                (p.map(f32::to_bits), (j, w.map(f32::to_bits)))
            })
            .collect::<Vec<_>>()
    };
    // The skins the source clusters hold at each position.
    let mut source: HashMap<_, HashSet<_>> = HashMap::new();
    for c in d.clusters.iter().filter(|c| c.record.self_error == 0.0) {
        for (p, s) in corners(c) {
            source.entry(p).or_default().insert(s);
        }
    }
    let mut apart = 0;
    for c in &d.clusters {
        for (p, s) in corners(c) {
            apart += usize::from(!source.get(&p).is_some_and(|k| k.contains(&s)));
        }
    }
    assert_eq!(apart, 0);
}

/// Half of a curved 4 m sheet: x from 0 to 2 m on the `side` of x = 0, the cut both halves
/// share and that bends in y, so neither half keeps it straight. With `bridged`, a triangle
/// with two corners at one end of each edge of the cut closes it, as exports have them.
fn half_sheet(side: f32, bridged: bool) -> MeshInput {
    let (nx, nz) = (48u32, 96u32);
    let mut m = MeshInput::default();
    for j in 0..=nz {
        for i in 0..=nx {
            let x = side * 2.0 * i as f32 / nx as f32;
            let z = 4.0 * j as f32 / nz as f32 - 2.0;
            let y = 0.3 * (1.7 * z).cos() * (1.1 * x).cos() + 0.1 * (3.0 * z + x).sin();
            m.positions.push(Vec3::new(x, y, z));
            m.normals.push(Vec3::Y);
        }
    }
    let w = nx + 1;
    for j in 0..nz {
        for i in 0..nx {
            let c = j * w + i;
            let quad = [c, c + w, c + 1, c + 1, c + w, c + w + 1];
            if side < 0.0 {
                m.indices
                    .extend(quad.chunks(3).flat_map(|t| [t[0], t[2], t[1]]));
            } else {
                m.indices.extend(quad);
            }
        }
        if bridged {
            // A second vertex at the edge's start, which a normal of its own keeps apart.
            let twin = m.positions.len() as u32;
            m.positions.push(m.positions[(j * w) as usize]);
            m.normals.push(Vec3::X);
            m.indices.extend([(j + 1) * w, j * w, twin]);
        }
    }
    m.uvs = None;
    m.sections = vec![Span {
        first: 0,
        count: m.indices.len() as u32,
    }];
    m
}

#[test]
fn abutting_meshes_stay_closed_at_every_cut() {
    // Each half cooks on its own, as two meshes that meet in a scene do: neither sees the
    // other, so each keeps its open border where it was, at every cut.
    for bridged in [false, true] {
        let halves =
            [-1.0, 1.0].map(|side| load(&cook(half_sheet(side, bridged), &[0], dag()), "m/mesh"));
        halves_stay_closed(&halves);
    }
}

fn halves_stay_closed(halves: &[Dag; 2]) {
    let step = 2f32.powi(-11);
    let rim = |p: [f32; 3]| p[0].abs() > 1.99 || p[2].abs() > 1.99;
    let cut = |d: &Dag, t: f32| -> Vec<[[f32; 3]; 3]> {
        d.select(Cut::Uniform(t), |_| true)
            .into_iter()
            .flat_map(|c| d.clusters[c].tris.iter().copied())
            .collect()
    };
    let mut levels = 0;
    for (a, b) in [(0, 1), (1, 0)] {
        let (da, db) = (&halves[a], &halves[b]);
        check_structure(da);
        let ta = da.thresholds(f32::MAX);
        assert!(ta.len() >= 4, "{ta:?}");
        levels = levels.max(ta.len());
        for &t in &ta {
            let tris = cut(da, t);
            // Open edges off the rim: the border the halves share. A triangle with two corners
            // at one position has no area and closes no edge.
            let mut edges = HashSet::new();
            for tri in tris
                .iter()
                .filter(|t| t[0] != t[1] && t[1] != t[2] && t[2] != t[0])
            {
                for k in 0..3 {
                    edges.insert((tri[k].map(f32::to_bits), tri[(k + 1) % 3].map(f32::to_bits)));
                }
            }
            let shared: Vec<([f32; 3], [f32; 3])> = edges
                .iter()
                .filter(|&&(p, q)| !edges.contains(&(q, p)))
                .map(|&(p, q)| (p.map(f32::from_bits), q.map(f32::from_bits)))
                .filter(|&(p, q)| !(rim(p) && rim(q)))
                .collect();
            assert!(!shared.is_empty());
            for &u in &db.thresholds(f32::MAX) {
                let other: Vec<[[f32; 3]; 3]> = cut(db, u)
                    .into_iter()
                    .filter(|tri| tri.iter().any(|p| p[0].abs() < 0.5))
                    .collect();
                for &(p, q) in &shared {
                    for s in 0..=4 {
                        let x = Vec3::from(p).lerp(Vec3::from(q), s as f32 / 4.0);
                        let d = other
                            .iter()
                            .map(|tri| point_triangle(x, tri.map(Vec3::from)))
                            .fold(f32::INFINITY, f32::min);
                        assert!(
                            d <= 3.0 * step,
                            "half {a} at {t} is {d} m from half {b} at {u} at {x}"
                        );
                    }
                }
            }
        }
    }
    assert!(levels >= 4);
}

/// Distance from `p` to the triangle `t`.
fn point_triangle(p: Vec3, t: [Vec3; 3]) -> f32 {
    let [a, b, c] = t;
    let n = (b - a).cross(c - a);
    if n.length_squared() > 0.0 {
        let n = n.normalize();
        let q = p - n * n.dot(p - a);
        let inside = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(u, v)| (v - u).cross(q - u).dot(n) >= 0.0);
        if inside {
            return (p - q).length();
        }
    }
    [(a, b), (b, c), (c, a)]
        .iter()
        .map(|&(u, v)| {
            let d = v - u;
            let s = ((p - u).dot(d) / d.length_squared()).clamp(0.0, 1.0);
            (p - (u + d * s)).length()
        })
        .fold(f32::INFINITY, f32::min)
}

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// A displaced plane in two uv islands that meet at x = 0.5 with different uvs there.
fn displaced_plane(path: &Path, tex: &[u8]) {
    let (mut pos, mut uv, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for island in 0..2u32 {
        let base = pos.len() as u32;
        let n = 6;
        for j in 0..=n {
            for i in 0..=n {
                let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
                pos.push([0.5 * (island as f32 + u), 0.0, v]);
                uv.push([0.45 * u + 0.55 * island as f32, v]);
            }
        }
        let w = n + 1;
        for j in 0..n {
            for i in 0..n {
                let c = base + j * w + i;
                idx.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
            }
        }
    }
    let mut g = Glb::default();
    let pa = g.floats(&pos, "VEC3");
    let na = g.floats(&vec![[0.0f32, 1.0, 0.0]; pos.len()], "VEC3");
    let ta = g.floats(&uv, "VEC2");
    let ia = g.indices(&idx);
    let img = g.image(tex);
    g.write(
        path,
        &format!(
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":{pa},"NORMAL":{na},"TEXCOORD_0":{ta}}},"indices":{ia},"material":0}}]}}],
"extensionsUsed":["KHR_materials_displacement"],
"materials":[{{"extensions":{{"KHR_materials_displacement":{{"displacementGeometryFactor":0.04,"displacementGeometryOffset":-0.01,"displacementGeometryTexture":{{"index":0}}}}}}}}],
"textures":[{{"source":0}}],"images":[{{"bufferView":{img},"mimeType":"image/png"}}]"#
        ),
    );
}

#[test]
fn static_displacement_bakes_a_crack_free_surface() {
    let d = tmp("dag-displaced");
    let tex = png(64, 64, |x, y| {
        let v = (128.0 + 100.0 * ((x as f32 * 0.15).sin() * (y as f32 * 0.1).cos())) as u8;
        [v, v, v, 255]
    });
    let path = d.join("plane.glb");
    displaced_plane(&path, &tex);
    let settings = GltfSettings {
        mesh: MeshSettings {
            displace_static: Some(StaticDisplacement {
                max_edge: 0.01,
                target_error: 0.0005,
            }),
            ..dag()
        },
        ..Default::default()
    };
    let a = cook_archive("g", Source::Gltf { path, settings });
    // The model binds a material without the displacement its geometry holds.
    let e = a.find(AssetId::from_name("g")).unwrap();
    let model = ModelInfo::parse(&read(&a, &a.blobs(e)[0])).unwrap();
    assert_eq!(
        model.materials,
        vec![AssetId::from_name("g/material/0/static")]
    );
    let e = a.find(model.materials[0]).unwrap();
    let m: Material = bytemuck::pod_read_unaligned(&read(&a, &a.blobs(e)[0]));
    assert_eq!(m.textures[slot::DISPLACEMENT].texture, AssetId::NONE);
    assert_eq!(m.displacement_scale, 0.0);
    assert!(a.find(AssetId::from_name("g/material/0")).is_some());

    let dag = load(&a, "g/mesh");
    check_structure(&dag);
    let source = dag.select(Cut::Uniform(0.0), |_| true);
    let source_tris: usize = source.iter().map(|&c| dag.clusters[c].tris.len()).sum();
    assert!(
        source_tris > 1000,
        "split to the edge length: {source_tris}"
    );
    let heights: Vec<f32> = source
        .iter()
        .flat_map(|&c| dag.clusters[c].tris.iter().flatten().map(|p| p[1]))
        .collect();
    let (lo, hi) = heights
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &y| (lo.min(y), hi.max(y)));
    assert!(
        lo < 0.0 && hi > 0.015 && lo >= -0.0102 && hi <= 0.0302,
        "{lo}..{hi}"
    );
    for c in &dag.clusters {
        assert_eq!(
            (c.record.displacement_min, c.record.displacement_max),
            (0, 0)
        );
    }
    let mut cuts: Vec<Cut> = dag.thresholds(0.01).into_iter().map(Cut::Uniform).collect();
    for eye in [Vec3::new(0.0, 0.3, 0.0), Vec3::new(0.5, 0.2, 0.5)] {
        cuts.push(Cut::View(eye, 0.002));
    }
    // Simplification may fold a sliver over in projection, so the check is on edges.
    assert!(cuts.len() > 3);
    let rim = |p: [f32; 3]| p[0] == 0.0 || p[0] == 1.0 || p[2] == 0.0 || p[2] == 1.0;
    for cut in cuts {
        let flat = dag.select(cut, |_| true);
        assert_eq!(unmatched_edges(&dag, &flat, rim), 0, "{cut:?}");
    }
}
