//! Cooked cluster sets, class flags and section bits, on meshes handed over in memory.

#[allow(dead_code)]
#[path = "common/dag.rs"]
mod dag_read;

use dag_read::{C, Dag, load};
use orr_cook::{
    ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshSettings, Source, TextureSettings,
};
use orr_math::{Vec2, Vec3};
use orr_pack::material::{Material, slot};
use orr_pack::mesh::{
    ClusterSet, ClusterSets, MeshHeader, NO_RANK, SET_LEAF, Section, SurfaceClass, in_cut,
    oct_decode,
};
use orr_pack::{Archive, AssetId, Span, Target};

fn cook(assets: Vec<AssetSpec>) -> Archive {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    Archive::from_memory(bytes).unwrap()
}

fn mesh(name: &str, mesh: MeshInput, materials: Vec<Material>, lod: LodMode) -> AssetSpec {
    AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials,
            settings: MeshSettings {
                lod,
                ..MeshSettings::default()
            },
        },
    }
}

fn glow() -> Material {
    Material {
        emission_color: [1.0, 0.5, 0.25],
        emission_luminance: 10.0,
        ..Material::default()
    }
}

fn glass() -> Material {
    Material {
        transmission_weight: 1.0,
        specular_roughness: 0.02,
        ..Material::default()
    }
}

/// An `n` x `n` wavy grid in `sections` bands along z, facing +y.
fn grid(n: u32, sections: u32) -> MeshInput {
    let mut m = MeshInput::default();
    let mut uvs = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            let (x, z) = ((u - 0.5) * 4.0, (v - 0.5) * 4.0);
            let y = 0.3 * (x * 1.9).sin() * (z * 1.3).cos();
            m.positions.push(Vec3::new(x, y, z));
            m.normals.push(Vec3::Y);
            uvs.push(Vec2::new(u, v));
        }
    }
    m.uvs = Some(uvs);
    let w = n + 1;
    for s in 0..sections {
        let first = m.indices.len() as u32;
        for j in n * s / sections..n * (s + 1) / sections {
            for i in 0..n {
                let c = j * w + i;
                m.indices.extend([c, c + w, c + 1, c + 1, c + w, c + w + 1]);
            }
        }
        m.sections.push(Span {
            first,
            count: m.indices.len() as u32 - first,
        });
    }
    m
}

/// The clusters of each row of a mesh.
fn rows(d: &Dag) -> Vec<Vec<usize>> {
    let h = &d.info.header;
    if !d.info.levels.is_empty() {
        let level = |l: &orr_pack::mesh::Level| {
            let s = l.clusters;
            (s.first as usize..(s.first + s.count) as usize).collect()
        };
        return d.info.levels.iter().map(level).collect();
    }
    let row = |&t: &f32| {
        let cut = |c: &&C| {
            let r = &c.record;
            in_cut(r.self_error, r.parent_error, r.flags, t, h.merge_error)
        };
        let first = d.clusters.as_ptr();
        d.clusters
            .iter()
            .filter(cut)
            // SAFETY: both point into `d.clusters`.
            .map(|c| unsafe { std::ptr::from_ref(c).offset_from(first) } as usize)
            .collect()
    };
    d.info.cut_rows.iter().map(row).collect()
}

fn area(t: &[[f32; 3]; 3]) -> f64 {
    let p = t.map(|v| Vec3::from(v).as_dvec3());
    0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
}

/// The set of section `section` at row `row`.
fn set_of(sets: &ClusterSets, row: usize, section: usize) -> ClusterSet {
    let rank = sets.ranks[section];
    assert_ne!(rank, NO_RANK);
    sets.sets[row * sets.rank_count() as usize + rank as usize]
}

/// Every set against the clusters of its row: `weight` of each one in order, one list, a
/// CDF over the weights and an index, and nodes bounding what lies below them.
fn check_sets(d: &Dag, sets: &ClusterSets, two_sided: bool, weight: impl Fn(&C) -> f64) -> usize {
    let mut checked = 0;
    for (r, row) in rows(d).iter().enumerate() {
        for (section, _) in sets
            .ranks
            .iter()
            .enumerate()
            .filter(|&(_, &k)| k != NO_RANK)
        {
            let set = set_of(sets, r, section);
            assert_eq!(set.section, section as u32);
            let mut expected: Vec<u32> = row
                .iter()
                .filter(|&&c| {
                    usize::from(d.clusters[c].record.section) == section
                        && weight(&d.clusters[c]) > 0.0
                })
                .map(|&c| c as u32)
                .collect();
            let list = set.first as usize..(set.first + set.count) as usize;
            let mut listed = sets.clusters[list.clone()].to_vec();
            listed.sort_unstable();
            expected.sort_unstable();
            assert_eq!(listed, expected, "row {r} section {section}");
            if expected.is_empty() {
                continue;
            }
            let total: f64 = expected
                .iter()
                .map(|&c| weight(&d.clusters[c as usize]))
                .sum();
            assert!((f64::from(set.total) - total).abs() <= 1e-3 * total);

            // Each step is its cluster's share of the total, and every position is indexed.
            let cdf = &sets.cdf[list.clone()];
            assert_eq!(*cdf.last().unwrap(), u32::MAX);
            let index = &sets.index[list.clone()];
            assert!(index.windows(2).all(|p| p[0][0] < p[1][0]));
            for (p, &c) in sets.clusters[list.clone()].iter().enumerate() {
                let step = cdf[p] - if p > 0 { cdf[p - 1] } else { 0 };
                assert!(step >= 1);
                let share = weight(&d.clusters[c as usize]) / total;
                assert!((f64::from(step) / f64::from(u32::MAX) - share).abs() < 1e-3);
                let at = index.binary_search_by_key(&c, |e| e[0]).unwrap();
                assert_eq!(index[at][1] as usize, p);
            }

            if set.count <= SET_LEAF {
                assert_eq!(set.node_count, 0);
                continue;
            }
            let nodes = &sets.nodes[set.first_node as usize..][..set.node_count as usize];
            assert_eq!((nodes[0].first, nodes[0].count), (0, set.count));
            for (i, n) in nodes.iter().enumerate() {
                let (left, right) = (2 * i + 1, 2 * i + 2);
                if right < nodes.len() {
                    assert_eq!(nodes[left].first, n.first);
                    assert_eq!(nodes[right].first, n.first + nodes[left].count);
                    assert_eq!(nodes[left].count + nodes[right].count, n.count);
                } else {
                    assert!(n.count <= SET_LEAF, "a leaf holds at most {SET_LEAF}");
                }
                let center = Vec3::from(n.center);
                let axis = Vec3::from(oct_decode(n.axis));
                if two_sided {
                    assert_eq!(n.cone_cos, -1.0);
                }
                let below = &sets.clusters[set.first as usize + n.first as usize..];
                for &c in &below[..n.count as usize] {
                    for t in &d.clusters[c as usize].tris {
                        for &p in t {
                            assert!((Vec3::from(p) - center).length() <= n.radius * 1.0001);
                        }
                        let [a, b, c] = t.map(Vec3::from);
                        let face = (b - a).cross(c - a).normalize_or_zero();
                        if face != Vec3::ZERO {
                            assert!(face.dot(axis) >= n.cone_cos - 1e-4);
                        }
                    }
                }
            }
            checked += 1;
        }
    }
    checked
}

#[test]
fn sets_hold_each_cut_of_their_sections_in_a_tree() {
    let a = cook(vec![
        mesh(
            "levels",
            grid(96, 2),
            vec![Material::default(), glow()],
            LodMode::Discrete {
                errors: vec![0.01, 0.1],
            },
        ),
        mesh("dag", grid(96, 2), vec![glow(), glass()], LodMode::Dag),
    ]);
    let area_of = |c: &C| c.tris.iter().map(area).sum::<f64>();

    let levels = load(&a, "levels/mesh");
    assert_eq!(levels.info.emissive.ranks, [NO_RANK, 0]);
    assert!(levels.info.casters.ranks.iter().all(|&r| r == NO_RANK));
    assert!(levels.info.cut_rows.is_empty());
    assert_eq!(levels.info.emissive.sets.len(), levels.info.levels.len());
    assert!(check_sets(&levels, &levels.info.emissive, false, area_of) > 0);

    let dag = load(&a, "dag/mesh");
    assert_eq!(dag.info.emissive.ranks, [0, NO_RANK]);
    assert_eq!(dag.info.casters.ranks, [NO_RANK, 0]);
    let rows = dag.info.cut_rows.len();
    assert!(rows > 2);
    assert_eq!(dag.info.cut_rows[1], 0.0);
    // Row 1 is level 0, which an edit traces: every cluster of the depth 0 groups.
    let mut level0: Vec<usize> = Vec::new();
    for g in dag.info.groups.iter().filter(|g| g.depth == 0) {
        let first = usize::from(g.first_cluster);
        let held = &dag.by_page[g.page as usize][first..][..usize::from(g.cluster_count)];
        level0.extend(held);
    }
    level0.sort_unstable();
    assert_eq!(self::rows(&dag)[1], level0);
    assert_eq!(dag.info.emissive.sets.len(), rows);
    assert_eq!(dag.info.casters.sets.len(), rows);
    assert!(check_sets(&dag, &dag.info.emissive, false, area_of) > 0);
    assert!(check_sets(&dag, &dag.info.casters, true, area_of) > 0);
    // Coarser cuts hold fewer clusters, and a cut that keeps them shares their range.
    let finest = set_of(&dag.info.emissive, 1, 0);
    let coarsest = set_of(&dag.info.emissive, rows - 1, 0);
    assert!(coarsest.count < finest.count);
}

#[test]
fn a_dag_with_streamed_pages_has_a_row_for_its_pinned_cut() {
    let a = cook(vec![mesh("big", grid(320, 1), vec![glow()], LodMode::Dag)]);
    let d = load(&a, "big/mesh");
    let h = &d.info.header;
    assert!((h.pinned_pages as usize) < d.info.pages.len());
    assert!(d.info.cut_rows[0] > 0.0);
    // Its clusters are the pinned pages' alone.
    let set = set_of(&d.info.emissive, 0, 0);
    let list = &d.info.emissive.clusters[set.first as usize..][..set.count as usize];
    assert!(!list.is_empty());
    for &c in list {
        assert!((d.clusters[c as usize].page as u32) < h.pinned_pages);
    }
    assert!(
        check_sets(&d, &d.info.emissive, false, |c| c
            .tris
            .iter()
            .map(area)
            .sum())
            > 0
    );
}

/// `half` lights the part of the texture below u = 0.5, `other` the rest.
fn emitter(path: &std::path::Path, half: u8, other: u8) -> Vec<AssetSpec> {
    let w = 64;
    let img = image::RgbaImage::from_fn(w, w, |x, _| {
        let v = if x < w / 2 { half } else { other };
        image::Rgba([v, v, v, 255])
    });
    img.save(path).unwrap();
    let mut m = glow();
    m.textures[slot::EMISSION].texture = AssetId::from_name("glow");
    vec![
        AssetSpec {
            name: "glow".into(),
            source: Source::Texture {
                path: path.into(),
                settings: TextureSettings::default(),
            },
        },
        mesh(
            "lamp",
            grid(96, 1),
            vec![m],
            LodMode::Discrete { errors: Vec::new() },
        ),
    ]
}

#[test]
fn emission_textures_weigh_clusters_and_key_the_mesh() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join("cluster_sets_emission.png");
    let a = cook(emitter(&path, 255, 0));
    let d = load(&a, "lamp/mesh");
    let e = &d.info.emissive;
    let set = set_of(e, 0, 0);
    // Clusters wholly over the black half leave the set; the edge texels wrap around.
    assert!(set.count > 0 && (set.count as usize) < d.clusters.len());
    let total: f64 = d.clusters.iter().flat_map(|c| &c.tris).map(area).sum();
    let share = f64::from(set.total) / total;
    assert!((0.4..0.6).contains(&share), "lit share {share}");
    let lit = |c: usize| {
        let t = &d.clusters[c].tris;
        t.iter().map(|t| t[0][0]).sum::<f32>() / t.len() as f32
    };
    let list = &e.clusters[set.first as usize..][..set.count as usize];
    assert!(!list.iter().any(|&c| (0.5..1.5).contains(&lit(c as usize))));

    // A brighter image re-cooks the mesh: the weights live in it.
    let mesh_hash = |a: &Archive| {
        let id = AssetId::from_name("lamp/mesh");
        a.entries()
            .iter()
            .find(|e| e.id == id)
            .unwrap()
            .content_hash
    };
    let brighter = cook(emitter(&path, 255, 255));
    assert_ne!(mesh_hash(&a), mesh_hash(&brighter));
    let d = load(&brighter, "lamp/mesh");
    assert_eq!(
        set_of(&d.info.emissive, 0, 0).count as usize,
        d.clusters.len()
    );
}

/// A unit cube: a closed section of flat faces, or five of them with `open`.
fn cube(open: bool) -> MeshInput {
    let mut m = MeshInput::default();
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (-Vec3::X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (-Vec3::Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (-Vec3::Z, Vec3::Y, Vec3::X),
    ];
    let count = if open { 5 } else { 6 };
    for (n, u, v) in faces.into_iter().take(count) {
        let base = m.positions.len() as u32;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            m.positions.push(n + a * u + b * v);
            m.normals.push(n);
        }
        m.indices
            .extend([0, 1, 2, 0, 2, 3].into_iter().map(|i| base + i));
    }
    m.uvs = Some(vec![Vec2::ZERO; m.positions.len()]);
    m.sections = vec![Span {
        first: 0,
        count: m.indices.len() as u32,
    }];
    m
}

/// A sphere of smooth normals, closed.
fn sphere() -> MeshInput {
    let (nu, nv) = (24u32, 12u32);
    let mut m = MeshInput::default();
    for j in 0..=nv {
        for i in 0..nu {
            let (a, b) = (
                i as f32 / nu as f32 * std::f32::consts::TAU,
                j as f32 / nv as f32 * std::f32::consts::PI,
            );
            let n = Vec3::new(a.cos() * b.sin(), b.cos(), a.sin() * b.sin());
            m.positions.push(n);
            m.normals.push(n);
        }
    }
    for j in 0..nv {
        for i in 0..nu {
            let (c, d) = (j * nu + i, j * nu + (i + 1) % nu);
            m.indices.extend([c, d, c + nu, d, d + nu, c + nu]);
        }
    }
    m.uvs = Some(vec![Vec2::ZERO; m.positions.len()]);
    m.sections = vec![Span {
        first: 0,
        count: m.indices.len() as u32,
    }];
    m
}

#[test]
fn sections_carry_their_class_and_shape_and_clusters_their_class() {
    let presence = Material {
        alpha_mode: Material::BLEND,
        geometry_opacity: 0.5,
        ..Material::default()
    };
    let none = || LodMode::Discrete { errors: Vec::new() };
    let mut many_sections = vec![Material::default(); 17];
    many_sections.push(glass());
    let a = cook(vec![
        mesh("glass_cube", cube(false), vec![glass()], none()),
        mesh("open_box", cube(true), vec![glass()], none()),
        mesh("wall", cube(false), vec![Material::default()], none()),
        mesh("ball", sphere(), vec![glass()], none()),
        mesh("veil", grid(8, 1), vec![presence], none()),
        mesh(
            "mixed",
            grid(16, 3),
            vec![glass(), Material::default(), presence],
            none(),
        ),
        mesh("many", grid(18, 18), many_sections, none()),
    ]);
    let section = |name: &str| load(&a, &format!("{name}/mesh")).info.sections[0];

    let flags = section("glass_cube").flags;
    assert_eq!(
        section("glass_cube").class(),
        Some(SurfaceClass::Dielectric)
    );
    assert_ne!(flags & Section::CLOSED, 0);
    assert_ne!(flags & Section::CASTER, 0);
    assert_eq!(section("open_box").flags & Section::CLOSED, 0);
    assert_eq!(section("wall").class(), Some(SurfaceClass::Opaque));
    assert_eq!(section("wall").flags & Section::CASTER, 0);
    assert_eq!(section("veil").class(), Some(SurfaceClass::Presence));

    // Every triangle of a cube of a caster section is flat, none of a sphere; other sections
    // keep no count. Flat triangles come before alpha-tested ones.
    let flat = |name: &str| -> (u32, u32) {
        let d = load(&a, &format!("{name}/mesh"));
        d.clusters.iter().fold((0, 0), |(f, t), c| {
            let r = &c.record;
            assert!(r.flat_triangles <= r.alpha_first);
            (f + u32::from(r.flat_triangles), t + u32::from(r.tri_count))
        })
    };
    let (f, t) = flat("glass_cube");
    assert_eq!(f, t);
    assert_eq!(flat("ball").0, 0);
    assert_eq!(flat("wall").0, 0);
    assert_ne!(section("ball").flags & Section::CLOSED, 0);

    let d = load(&a, "mixed/mesh");
    let h: &MeshHeader = &d.info.header;
    assert_eq!((h.opaque_sections, h.layer_sections), (0b010, 0b101));
    // Sections an instance can't hide share a bit, whichever of them holds the class.
    let h = load(&a, "many/mesh").info.header;
    assert_eq!((h.opaque_sections, h.layer_sections), (0x1ffff, 1 << 16));
    let classes = [
        SurfaceClass::Dielectric,
        SurfaceClass::Opaque,
        SurfaceClass::Presence,
    ];
    for c in &d.clusters {
        assert_eq!(
            c.record.class(),
            Some(classes[usize::from(c.record.section)])
        );
    }
}
