#![allow(clippy::print_stdout)] // the test reports measured figures

//! Cooks an external placement set; ignored unless asked for, since the set lives outside
//! the repository.
//!
//! `ORR_SCATTER_SET=<dir>`: every `chunk_*.bin` in `dir`, each a varint record count, then
//! per record an id, a prototype and a layer as varints and ten f32 (position, rotation
//! x y z w, scale). Prototypes all place one fixture model. Cooks one archive at cell_log2 8
//! and checks that every compact record puts the fixture within half its grid step, at the
//! copy's scale, of where its placement does, and counts the records that take exact rows.
//!
//! `cargo test --release -p orr_cook --test instance_sets -- --ignored --nocapture`

mod common;

use std::collections::BTreeMap;

use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, Instances, Linear, Manifest, Placement, PrototypeSpec,
    Source,
};
use orr_math::{DAffine3, DQuat, DVec3, Quat, Vec3};
use orr_pack::instance::{self, InstancePageInfo};
use orr_pack::{Aabb, Archive, Class, Target};

fn varint(d: &[u8], p: &mut usize) -> u64 {
    let mut v = 0;
    let mut shift = 0;
    loop {
        let b = d[*p];
        *p += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}

struct Record {
    prototype: u64,
    f: [f32; 10],
}

fn read_chunk(d: &[u8]) -> Vec<Record> {
    let mut p = 0;
    let count = varint(d, &mut p);
    let records = (0..count)
        .map(|_| {
            varint(d, &mut p);
            let prototype = varint(d, &mut p);
            varint(d, &mut p);
            let f = std::array::from_fn(|i| f32::from_le_bytes(d[p..].as_chunks().0[i]));
            p += 40;
            Record { prototype, f }
        })
        .collect();
    assert_eq!(p, d.len(), "trailing bytes");
    records
}

#[test]
#[ignore = "needs ORR_SCATTER_SET"]
fn scatter_set_cooks_within_its_error_bounds() {
    let dir = std::env::var("ORR_SCATTER_SET").expect("ORR_SCATTER_SET");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy();
            n.starts_with("chunk_") && n.ends_with(".bin")
        })
        .collect();
    files.sort();
    let chunks: Vec<Vec<Record>> = files
        .iter()
        .map(|f| read_chunk(&std::fs::read(f).unwrap()))
        .collect();
    let ids: BTreeMap<u64, u16> = {
        let mut ids: Vec<u64> = chunks.iter().flatten().map(|r| r.prototype).collect();
        ids.sort_unstable();
        ids.dedup();
        ids.into_iter()
            .enumerate()
            .map(|(i, id)| (id, i as u16))
            .collect()
    };
    let prototypes: Vec<PrototypeSpec> = Vec::from_fn(ids.len(), |i| PrototypeSpec {
        model: "block".into(),
        flags: i as u32,
        ..Default::default()
    });

    let mut assets = vec![AssetSpec {
        name: "block".into(),
        source: Source::Gltf {
            path: common::block_glb(),
            settings: GltfSettings::default(),
        },
    }];
    let total: usize = chunks.iter().map(Vec::len).sum();
    for (file, recs) in files.iter().zip(&chunks) {
        assets.push(AssetSpec {
            name: file.file_stem().unwrap().to_string_lossy().into_owned(),
            source: Source::Instances(Instances {
                frame: None,
                motion: None,
                records: recs
                    .iter()
                    .map(|r| Placement {
                        position: DVec3::new(r.f[0].into(), r.f[1].into(), r.f[2].into())
                            .to_array(),
                        linear: Linear::Scaled {
                            rotation: Quat::from_xyzw(r.f[3], r.f[4], r.f[5], r.f[6]).to_array(),
                            scale: Vec3::new(r.f[7], r.f[8], r.f[9]).to_array(),
                        },
                        prototype: ids[&r.prototype],
                    })
                    .collect(),
                prototypes: prototypes.clone(),
                user: (0..recs.len() as u32).flat_map(u32::to_le_bytes).collect(),
                user_stride: 4,
                joined: false,
            }),
        });
    }
    let out = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("scatter.orr");
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::File(out.clone()), assets)],
    };
    orr_cook::cook(&manifest, &Default::default()).unwrap();

    let a = Archive::open(&out).unwrap();
    let file_index: BTreeMap<String, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.file_stem().unwrap().to_string_lossy().into_owned(), i))
        .collect();
    let (mut pages, mut records, mut split, mut rows) = (0, 0, 0, 0);
    let (mut worst, mut worst_row) = (0f64, 0f64);
    let block = Aabb {
        min: [0.0; 3],
        max: [1.0; 3],
    };
    for e in a
        .entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
    {
        let name = a.name(e);
        let (source, page) = name.split_once('/').unwrap();
        split += usize::from(!page.ends_with("/0"));
        let recs_src = &chunks[file_index[source]];
        let read = |b: usize| {
            let blob = &a.blobs(e)[b];
            let mut v = vec![0; blob.raw_len as usize];
            a.read(blob, &mut v).unwrap();
            v
        };
        let (head, recs, user) = (read(0), read(1), read(2));
        let h = InstancePageInfo::parse(&head).unwrap().header;
        let (raw, _) = instance::records(&h, &recs).unwrap();
        let corner = DVec3::from(h.cell.map(f64::from)) * 256.0;
        let (_, decoded) = instance::decode(&head, &recs).unwrap();
        for (k, (record, (_, m))) in raw.iter().zip(decoded).enumerate() {
            let r = &recs_src[u32::from_le_bytes(user.as_chunks().0[k]) as usize];
            let q = DQuat::from_xyzw(r.f[3].into(), r.f[4].into(), r.f[5].into(), r.f[6].into())
                .normalize();
            let s = Vec3::new(r.f[7], r.f[8], r.f[9]).as_dvec3();
            let at = DVec3::new(r.f[0].into(), r.f[1].into(), r.f[2].into()) - corner;
            let want = DAffine3::from_scale_rotation_translation(s, q, at);
            let placed =
                DAffine3::from_mat3_translation(m.matrix3.as_dmat3(), m.translation.as_dvec3());
            let error = instance::displacement(&want, &placed, &block);
            if record.exact_row().is_some() {
                rows += 1;
                worst_row = worst_row.max(error);
            } else {
                // Half the block's grid step at the copy's scale.
                let bound = 0.5 / 2048.0 * s.abs().min_element();
                assert!(error <= bound, "{name}: a compact record moves {error} m");
                worst = worst.max(error / bound);
            }
        }
        pages += 1;
        records += usize::from(h.record_count);
    }
    assert_eq!(records, total);
    println!(
        "{} files, {total} records, {} prototypes -> {pages} pages ({split} split pages), {} B",
        files.len(),
        ids.len(),
        std::fs::metadata(&out).unwrap().len()
    );
    println!(
        "{rows} records in exact rows; compact records at {worst:.4} of their tolerance, rows moving points up to {worst_row:.2e} m"
    );
    assert_eq!(split, 0, "a page split");
}
