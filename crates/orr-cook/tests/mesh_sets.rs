#![allow(clippy::print_stdout)] // the test reports measured figures

//! A content set of glTF files cooked with three discrete levels (ignored by default).
//!
//! - `ORR_MESH_SET=<dir>`: every `.glb` and `.gltf` under `dir`, with level errors from
//!   `ORR_MESH_ERRORS` (metres, default `0.005,0.05`).
//! - `ORR_MESH_PARTS=<attribute>:<count>`: also cooks each file with rigid parts from that
//!   attribute and reports its coarsest level against the cook without parts.
//!
//! Every level's positions lie on level 0's and levels never gain triangles; per-level
//! triangles, errors and bytes per triangle are printed.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, LodMode, Manifest, MeshSettings, PartAttribute, Source,
};
use orr_pack::mesh::{self, MeshInfo, PageView};
use orr_pack::{Archive, Class, Target};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else if p
            .extension()
            .is_some_and(|x| x.eq_ignore_ascii_case("glb") || x.eq_ignore_ascii_case("gltf"))
        {
            out.push(p);
        }
    }
}

fn cook(path: &Path, settings: GltfSettings) -> Archive {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "set".into(),
                source: Source::Gltf {
                    path: path.into(),
                    settings,
                },
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .memory
        .remove(0);
    Archive::from_memory(bytes).unwrap()
}

/// A level's triangle count and grid positions.
type LevelFacts = (u64, HashSet<[i32; 3]>);

fn levels(a: &Archive, e: &orr_pack::Entry) -> (MeshInfo, Vec<LevelFacts>) {
    let blobs = a.blobs(e);
    let read = |b: &orr_pack::Blob| {
        let mut v = vec![0; b.raw_len as usize];
        a.read(b, &mut v).unwrap();
        v
    };
    let info = MeshInfo::parse(&read(&blobs[0])).unwrap();
    let mut out = vec![(0, HashSet::new()); info.levels.len()];
    let mut first = 0;
    for p in &info.pages {
        let level = info
            .levels
            .iter()
            .position(|l| (l.clusters.first..l.clusters.first + l.clusters.count).contains(&first))
            .unwrap();
        let mut page = vec![0; p.resident_len as usize];
        mesh::decode_page(&read(&blobs[p.blob as usize]), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            out[level].0 += u64::from(c.tri_count);
            for v in 0..u32::from(c.vertex_count) {
                out[level].1.insert(view.grid(&c, v));
            }
        }
        first += u32::from(p.cluster_count);
    }
    (info, out)
}

#[test]
#[ignore = "needs ORR_MESH_SET"]
fn mesh_set_cooks_with_three_levels() {
    let dir = PathBuf::from(std::env::var("ORR_MESH_SET").expect("ORR_MESH_SET"));
    let errors: Vec<f32> = std::env::var("ORR_MESH_ERRORS")
        .unwrap_or_else(|_| "0.005,0.05".into())
        .split(',')
        .map(|e| e.trim().parse().unwrap())
        .collect();
    assert_eq!(errors.len(), 2);
    let parts = std::env::var("ORR_MESH_PARTS").ok().map(|p| {
        let (name, count) = p.split_once(':').expect("<attribute>:<count>");
        PartAttribute {
            name: name.into(),
            parts: count.parse().unwrap(),
            ..Default::default()
        }
    });
    let mut list = Vec::new();
    files(&dir, &mut list);
    list.sort();
    assert!(!list.is_empty());
    let settings = GltfSettings {
        mesh: MeshSettings {
            lod: LodMode::Discrete { errors },
            ..Default::default()
        },
        ..Default::default()
    };
    for path in list {
        let a = cook(&path, settings.clone());
        let mesh = a
            .entries()
            .iter()
            .find(|e| e.class == Class::Mesh as u16)
            .unwrap();
        let (info, per_level) = levels(&a, mesh);
        assert_eq!(per_level.len(), 3);
        for l in 1..3 {
            assert!(per_level[l].0 <= per_level[l - 1].0);
            assert!(per_level[l].1.is_subset(&per_level[0].1));
        }
        let blobs = a.blobs(mesh);
        let mut line = format!("{}:", path.display());
        for (l, (tris, _)) in per_level.iter().enumerate() {
            let (mut stored, mut resident) = (0u64, 0u64);
            let mut c = 0;
            for p in &info.pages {
                let span = info.levels[l].clusters;
                if (span.first..span.first + span.count).contains(&c) {
                    stored += u64::from(blobs[p.blob as usize].stored_len);
                    resident += u64::from(p.resident_len);
                }
                c += u32::from(p.cluster_count);
            }
            line += &format!(
                " L{l} {tris} tris {:.1}/{:.1} B per tri (disk/resident), error {:.4};",
                stored as f64 / *tris as f64,
                resident as f64 / *tris as f64,
                info.levels[l].error
            );
        }
        if let Some(pa) = &parts {
            let with = cook(
                &path,
                GltfSettings {
                    parts_from_attribute: Some(pa.clone()),
                    ..settings.clone()
                },
            );
            let mesh = with
                .entries()
                .iter()
                .find(|e| e.class == Class::Mesh as u16)
                .unwrap();
            let (_, parted) = levels(&with, mesh);
            line += &format!(
                " with parts: coarsest {} tris ({:.2}x)",
                parted[2].0,
                parted[2].0 as f64 / per_level[2].0.max(1) as f64
            );
        }
        println!("{line}");
    }
}
