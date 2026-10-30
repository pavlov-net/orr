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
use glam::{Vec2, Vec3};
use glb::Glb;
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MeshInput, MeshParts, MeshSettings,
    Source, StaticDisplacement,
};
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
}

#[test]
fn parts_merge_past_the_merge_error() {
    let mut input = wavy(64, true);
    let n = input.positions.len();
    let values: Vec<u16> = input
        .positions
        .iter()
        .map(|p| ((p.x + 2.0) / 4.0 * 4.999).floor() as u16)
        .collect();
    assert_eq!(values.len(), n);
    input.parts = Some(MeshParts {
        values,
        count: 5,
        by_section: false,
        pivots: None,
    });
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
    assert_eq!(d.info.roots.len(), 5 * 2 + 2);
    assert!(d.info.roots.iter().all(|&r| r != MeshHeader::NO_ROOT));
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
    assert!(resident.info.node_pages.is_empty());
    assert_eq!(resident.nodes, paged.nodes, "paging keeps the node order");
    let info = &paged.info;
    assert!(!info.node_pages.is_empty());
    let top = &info.node_pages[0];
    assert_eq!(top.parent_page, NodePage::TOP);
    assert!(info.roots.iter().all(|&r| r < top.node_count));
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
