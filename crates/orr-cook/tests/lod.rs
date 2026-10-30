//! Discrete levels, rigid parts and the merged stage, on meshes handed over in memory.

use std::collections::HashSet;

use glam::{Vec2, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, LodMode, Manifest, MeshInput, MeshParts, MeshSettings, Source,
};
use orr_pack::material::Material;
use orr_pack::mesh::{self, Cluster, MeshInfo, PageView};
use orr_pack::{Aabb, Archive, AssetId, Span, Target};

/// A wavy grid of `n` x `n` quads over 4 m, two sections (lower and upper half of the rows),
/// part value by column band.
fn wavy(n: u32, bands: u16) -> MeshInput {
    let mut m = MeshInput::default();
    let mut values = Vec::new();
    let mut pivots = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            let x = (u - 0.5) * 4.0;
            let z = (v - 0.5) * 4.0;
            m.positions
                .push(Vec3::new(x, 0.15 * (x * 2.3).sin() * (z * 1.7).cos(), z));
            m.normals.push(Vec3::Y);
            let band = (i * u32::from(bands) / (n + 1)) as u16;
            values.push(band);
            pivots.push(Vec3::new(f32::from(band), 1.0, 0.0));
        }
    }
    m.uvs = Some(
        (0..=n)
            .flat_map(|j| (0..=n).map(move |i| Vec2::new(i as f32 / n as f32, j as f32 / n as f32)))
            .collect(),
    );
    let w = n + 1;
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let c = j * w + i;
            let quad = [c, c + w, c + 1, c + 1, c + w, c + w + 1];
            if j < n / 2 { &mut lower } else { &mut upper }.extend(quad);
        }
    }
    m.sections = vec![
        Span {
            first: 0,
            count: lower.len() as u32,
        },
        Span {
            first: lower.len() as u32,
            count: upper.len() as u32,
        },
    ];
    m.indices = [lower, upper].concat();
    m.parts = Some(MeshParts {
        values,
        count: bands + 1,
        by_section: false,
        pivots: Some(pivots),
    });
    m
}

fn cook(mesh: MeshInput, settings: MeshSettings) -> orr_cook::Result<Archive> {
    let materials = vec![Material::default(); mesh.sections.len()];
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "m".into(),
                source: Source::Mesh {
                    mesh,
                    materials,
                    settings,
                },
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())?
        .memory
        .remove(0);
    let a = Archive::from_memory(bytes).unwrap();
    for e in a.entries() {
        a.verify(e).unwrap();
    }
    Ok(a)
}

fn read(a: &Archive, b: &orr_pack::Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

/// A resident page and its clusters.
type Page = (Vec<u8>, Vec<Cluster>);

/// Every cluster of the mesh with its resident page.
fn clusters(a: &Archive) -> (MeshInfo, Vec<Page>) {
    let e = a.find(AssetId::from_name("m/mesh")).unwrap();
    let blobs = a.blobs(e);
    let info = MeshInfo::parse(&read(a, &blobs[0])).unwrap();
    let pages = info
        .pages
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let b = &blobs[p.blob as usize];
            assert_eq!(b.is_mount(), (i as u32) < info.header.pinned_pages);
            let mut page = vec![0; p.resident_len as usize];
            mesh::decode_page(&read(a, b), &mut page).unwrap();
            let view = PageView::new(&page, &info.header).unwrap();
            let c = (0..view.cluster_count()).map(|i| view.cluster(i)).collect();
            (page, c)
        })
        .collect();
    (info, pages)
}

fn snap(p: Vec3, k: i8) -> [u32; 3] {
    let step = 2f64.powi(i32::from(k));
    p.to_array()
        .map(|x| (((f64::from(x) / step + 0.5).floor()) * step) as f32)
        .map(f32::to_bits)
}

#[test]
fn levels_simplify_and_parts_merge() {
    let input = wavy(48, 5);
    let errors = vec![0.002, 0.02];
    let settings = MeshSettings {
        lod: LodMode::Discrete {
            errors: errors.clone(),
        },
        merge_level: Some(2),
        ..Default::default()
    };
    let a = cook(input.clone(), settings).unwrap();
    let (info, pages) = clusters(&a);
    let h = info.header;
    assert_eq!(info.levels.len(), 3);
    assert_eq!(h.merge_level, 2);
    let parts = info.parts.len();
    assert_eq!(parts, 6);
    let src_parts = input.parts.as_ref().unwrap();
    let src: HashSet<[u32; 3]> = input
        .positions
        .iter()
        .map(|&p| snap(p, h.position_exponent))
        .collect();

    // Clusters by mesh index, in page order.
    let mut all = Vec::new();
    for (page, cs) in &pages {
        let view = PageView::new(page, &h).unwrap();
        for c in cs {
            let positions: Vec<[u32; 3]> = (0..u32::from(c.vertex_count))
                .map(|v| view.position(c, v).map(f32::to_bits))
                .collect();
            let tris: Vec<[[u32; 3]; 3]> = (0..u32::from(c.tri_count))
                .map(|t| view.triangle(c, t).map(|i| positions[usize::from(i)]))
                .collect();
            all.push((*c, positions, tris));
        }
    }
    let mut triangles = Vec::new();
    for (li, level) in info.levels.iter().enumerate() {
        let range =
            level.clusters.first as usize..(level.clusters.first + level.clusters.count) as usize;
        let mut n = 0;
        for (c, positions, tris) in &all[range.clone()] {
            n += tris.len();
            assert_eq!(c.self_error, level.error);
            let parent = info.levels.get(li + 1).map_or(f32::INFINITY, |l| l.error);
            assert_eq!(c.parent_error, parent);
            // Simplification keeps source vertices, so every position is a source one.
            assert!(positions.iter().all(|p| src.contains(p)));
            let merged = li >= 2;
            assert_eq!(c.flags & Cluster::MERGED != 0, merged);
            assert_eq!(
                c.flags & Cluster::ANIMATED_PART != 0,
                !merged && c.part != 0
            );
            if merged {
                assert_eq!(c.part, 0);
            }
            let b = info.parts[usize::from(c.part)].bounds;
            if !merged {
                for p in positions {
                    let p = p.map(f32::from_bits);
                    assert!((0..3).all(|a| b.min[a] <= p[a] && p[a] <= b.max[a]));
                }
            }
        }
        triangles.push(n);
        // Part spans partition the level, every part's clusters contiguous.
        let mut covered = 0;
        for p in 0..parts {
            let s = info.part_levels[p * info.levels.len() + li].clusters;
            covered += s.count;
            for (c, ..) in &all[s.first as usize..(s.first + s.count) as usize] {
                assert_eq!(usize::from(c.part), p);
                assert!(range.contains(&(s.first as usize)));
            }
            if li >= 2 && p != 0 {
                assert_eq!(s.count, 0);
            }
        }
        assert_eq!(covered, level.clusters.count);
    }
    assert!(
        triangles[0] > triangles[1] && triangles[1] > triangles[2],
        "{triangles:?}"
    );
    assert_eq!(triangles[0] * 3, input.indices.len());
    assert!(info.levels[1].error <= errors[0] * 1.001);
    assert!(info.levels[2].error <= (errors[0] + errors[1]) * 1.001);
    assert!(info.levels[2].error >= info.levels[1].error);

    // Level 0 holds the source triangles, each in its majority part.
    let mut expected: Vec<(u8, [[u32; 3]; 3])> = input
        .indices
        .chunks(3)
        .map(|t| {
            let v = [0, 1, 2].map(|k| src_parts.values[t[k] as usize]);
            let value = if v[0] == v[1] || v[0] == v[2] {
                v[0]
            } else if v[1] == v[2] {
                v[1]
            } else {
                v[0].min(v[1]).min(v[2])
            };
            (
                value as u8,
                rotate(
                    [0, 1, 2].map(|k| snap(input.positions[t[k] as usize], h.position_exponent)),
                ),
            )
        })
        .collect();
    let l0 = info.levels[0].clusters;
    let mut cooked: Vec<(u8, [[u32; 3]; 3])> = all
        [l0.first as usize..(l0.first + l0.count) as usize]
        .iter()
        .flat_map(|(c, _, tris)| tris.iter().map(|t| (c.part, rotate(*t))))
        .collect();
    expected.sort_unstable();
    cooked.sort_unstable();
    assert_eq!(cooked, expected);

    // Pivots from the attribute; the declared but unused part stays empty.
    for (p, part) in info.parts.iter().enumerate().take(5) {
        assert_eq!(part.pivot, [p as f32, 1.0, 0.0]);
        assert_eq!(part.joint, p as u16);
    }
    assert_eq!(info.parts[5].bounds, Aabb::EMPTY);
    // The coarsest level's pages are the pinned ones.
    let coarsest = info.levels[2].clusters;
    let pinned: u32 = info.pages[..h.pinned_pages as usize]
        .iter()
        .map(|p| u32::from(p.cluster_count))
        .sum();
    assert_eq!((coarsest.first, coarsest.count), (0, pinned));
}

/// Triangle corners rotated so the smallest comes first, winding kept.
fn rotate(t: [[u32; 3]; 3]) -> [[u32; 3]; 3] {
    let r = (0..3).min_by_key(|&i| t[i]).unwrap();
    [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
}

/// One triangle per corner-value triple, each its own section when `sections` > 1.
fn triangles(values: &[[u16; 3]], sections: usize) -> MeshInput {
    let mut m = MeshInput::default();
    let mut per_section = vec![Vec::new(); sections];
    let mut vals = Vec::new();
    for (i, t) in values.iter().enumerate() {
        let base = m.positions.len() as u32;
        let o = Vec3::new(i as f32 * 2.0, 0.0, 0.0);
        for (k, &v) in t.iter().enumerate() {
            m.positions.push(o + [Vec3::ZERO, Vec3::X, Vec3::Z][k]);
            m.normals.push(Vec3::Y);
            vals.push(v);
        }
        per_section[i % sections].extend([base, base + 2, base + 1]);
    }
    let mut first = 0;
    for s in per_section {
        m.sections.push(Span {
            first,
            count: s.len() as u32,
        });
        first += s.len() as u32;
        m.indices.extend(s);
    }
    m.parts = Some(MeshParts {
        values: vals,
        count: 4,
        by_section: false,
        pivots: None,
    });
    m
}

fn part_triangles(a: &Archive) -> Vec<(u8, u32)> {
    let (_, pages) = clusters(a);
    let mut out: Vec<(u8, u32)> = pages
        .iter()
        .flat_map(|(_, cs)| cs.iter().map(|c| (c.part, u32::from(c.tri_count))))
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn parts_follow_the_majority_then_the_lowest_value() {
    // Two corners agree; all three differ (lowest wins); all agree.
    let input = triangles(&[[1, 1, 2], [3, 1, 2], [2, 3, 2], [3, 3, 3]], 1);
    let a = cook(input, MeshSettings::default()).unwrap();
    let mut counts = [0; 4];
    for (p, n) in part_triangles(&a) {
        counts[usize::from(p)] += n;
    }
    assert_eq!(counts, [0, 2, 1, 1]);

    // By section: part = value x sections + section.
    let mut input = triangles(&[[1, 1, 1], [1, 1, 1], [2, 2, 2]], 2);
    input.parts.as_mut().unwrap().by_section = true;
    let a = cook(input, MeshSettings::default()).unwrap();
    let (info, _) = clusters(&a);
    assert_eq!(info.parts.len(), 8);
    let parts: Vec<u8> = part_triangles(&a).iter().map(|p| p.0).collect();
    assert_eq!(parts, [2, 3, 4]);
    assert_eq!(info.parts[3].joint, 1);

    // A value at the declared count fails, and so do pivots that differ within a part.
    let mut input = triangles(&[[4, 4, 4]], 1);
    assert!(matches!(
        cook(input.clone(), MeshSettings::default()),
        Err(orr_cook::Error::Mesh { .. })
    ));
    input.parts.as_mut().unwrap().values = vec![1; 3];
    input.parts.as_mut().unwrap().pivots = Some(vec![Vec3::ZERO, Vec3::ZERO, Vec3::X]);
    assert!(matches!(
        cook(input, MeshSettings::default()),
        Err(orr_cook::Error::Mesh { .. })
    ));
}

#[test]
fn grid_steps_follow_the_extent() {
    for (extent, k) in [(1.0f32, -11), (32.0, -10), (31.0, -11), (1000.0, -6)] {
        let mut input = triangles(&[[0, 0, 0]], 1);
        input.parts = None;
        input.positions[1] = Vec3::new(extent, 0.0, 0.0);
        let a = cook(input, MeshSettings::default()).unwrap();
        let (info, _) = clusters(&a);
        let expected = (-11).max((f64::from(extent) / 65535.0).log2().ceil() as i32);
        assert_eq!(i32::from(info.header.position_exponent), expected);
        assert_eq!(expected, k);
    }
}
