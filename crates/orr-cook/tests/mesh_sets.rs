#![allow(clippy::print_stdout)] // the test reports measured figures

//! A content set of glTF files cooked with three discrete levels and as DAGs (ignored by
//! default).
//!
//! - `ORR_MESH_SET=<dir>`: every `.glb` and `.gltf` under `dir`, with level errors from
//!   `ORR_MESH_ERRORS` (metres, default `0.005,0.05`).
//! - `ORR_MESH_PARTS=<attribute>:<count>`: also cooks each file with rigid parts from that
//!   attribute and reports its coarsest level against the cook without parts; the DAG cook
//!   uses the parts, merged from `ORR_MESH_MERGE_ERROR` metres when it is set.
//!
//! Every level's positions lie on level 0's and levels never gain triangles; per-level
//! triangles, errors and bytes per triangle are printed. DAG cooks keep every page within
//! its size and the part stage's triangles shrink with depth; double-sided sections keep
//! their projected coverage within 10% of the source at every uniform cut, along the three
//! axes (six axis views, opposite ones alike). Per-depth triangles, cook time and the worst coverage ratio are printed.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "common/dag.rs"]
mod dag_read;

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

#[test]
#[ignore = "needs ORR_MESH_SET"]
fn mesh_set_cooks_as_dags() {
    let dir = PathBuf::from(std::env::var("ORR_MESH_SET").expect("ORR_MESH_SET"));
    let parts = std::env::var("ORR_MESH_PARTS").ok().map(|p| {
        let (name, count) = p.split_once(':').expect("<attribute>:<count>");
        PartAttribute {
            name: name.into(),
            parts: count.parse().unwrap(),
            ..Default::default()
        }
    });
    let merge_error = std::env::var("ORR_MESH_MERGE_ERROR")
        .ok()
        .map(|e| e.parse().unwrap());
    let mut list = Vec::new();
    files(&dir, &mut list);
    list.sort();
    assert!(!list.is_empty());
    let settings = GltfSettings {
        mesh: MeshSettings {
            lod: LodMode::Dag,
            merge_error,
            ..Default::default()
        },
        parts_from_attribute: parts,
        ..Default::default()
    };
    for path in list {
        let start = std::time::Instant::now();
        let a = cook(&path, settings.clone());
        let seconds = start.elapsed().as_secs_f64();
        let e = a
            .entries()
            .iter()
            .find(|e| e.class == Class::Mesh as u16)
            .unwrap();
        let d = dag_read::load(&a, a.name(e));
        let per_depth = d.part_triangles_per_depth();
        assert!(per_depth.windows(2).all(|w| w[1] <= w[0]), "{per_depth:?}");
        let mut line = format!(
            "{}: {seconds:.2} s, {} groups, triangles per depth {per_depth:?}",
            path.display(),
            d.info.groups.len()
        );
        let foliage = |c: &mesh::Cluster| {
            d.info.sections[usize::from(c.section)].flags & mesh::Section::DOUBLE_SIDED != 0
                && c.flags & mesh::Cluster::MERGED == 0
        };
        let tris = |cut: dag_read::Cut| -> Vec<[[f32; 3]; 3]> {
            d.select(cut, foliage)
                .into_iter()
                .flat_map(|c| d.clusters[c].tris.iter().copied())
                .collect()
        };
        let source = tris(dag_read::Cut::Uniform(0.0));
        if !source.is_empty() {
            let b = d.info.header.bounds;
            let centre = glam::Vec3::from(b.min).lerp(glam::Vec3::from(b.max), 0.5);
            let size = (0..3).map(|k| b.max[k] - b.min[k]).fold(0.0, f32::max) * 1.05;
            let view = (centre, size);
            let base = dag_read::coverage(&source, view, 512);
            let (mut worst, mut at) = (1.0f64, 0.0);
            for t in d.thresholds(f32::MAX) {
                let cov = dag_read::coverage(&tris(dag_read::Cut::Uniform(t)), view, 512);
                for (&c, &s) in cov.iter().zip(&base) {
                    let r = c as f64 / s.max(1) as f64;
                    if (r - 1.0).abs() > (worst - 1.0).abs() {
                        (worst, at) = (r, t);
                    }
                }
            }
            line += &format!(", double-sided coverage worst {worst:.3} of the source at {at} m");
            assert!((0.9..=1.1).contains(&worst), "{line}");
        }
        println!("{line}");
    }
}
