#![allow(clippy::print_stdout)] // the test reports measured figures

//! Cooks an external placement set; ignored unless asked for, since the set lives outside
//! the repository.
//!
//! `ORR_SCATTER_SET=<dir>`: every `chunk_*.bin` in `dir`, each a varint record count, then
//! per record an id, a prototype and a layer as varints and ten f32 (position, rotation
//! x y z w, scale). Prototypes all place one fixture model. Cooks one archive at cell_log2 8
//! and checks every record's position, rotation and scale error against the format's bounds:
//! half a step plus the f32 spacing, 0.17 degrees, and half a step of the page's log2 scale
//! range.
//!
//! `cargo test --release -p orr-cook --test instance_sets -- --ignored --nocapture`

mod common;

use std::collections::BTreeMap;

use glam::{DQuat, DVec3, Quat, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, Instances, Manifest, Placement, PrototypeSpec, Source,
};
use orr_pack::instance::{self, InstancePageInfo};
use orr_pack::{Archive, Class, Target};

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
            let f = std::array::from_fn(|i| {
                f32::from_le_bytes(d[p + 4 * i..p + 4 * i + 4].try_into().unwrap())
            });
            p += 40;
            Record { prototype, f }
        })
        .collect();
    assert_eq!(p, d.len(), "trailing bytes");
    records
}

/// Spacing of f32 values at `v`: the decode's own rounding adds up to this to the half step.
fn ulp(v: f32) -> f32 {
    v.abs().next_up() - v.abs()
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
    let prototypes: Vec<PrototypeSpec> = (0..ids.len())
        .map(|i| PrototypeSpec {
            model: "block".into(),
            flags: i as u32,
            ..Default::default()
        })
        .collect();

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
                records: recs
                    .iter()
                    .map(|r| Placement {
                        position: DVec3::new(r.f[0].into(), r.f[1].into(), r.f[2].into()),
                        rotation: Quat::from_xyzw(r.f[3], r.f[4], r.f[5], r.f[6]),
                        scale: Vec3::new(r.f[7], r.f[8], r.f[9]),
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
    orr_cook::cook(&manifest).unwrap();

    let a = Archive::open(&out).unwrap();
    let file_index: BTreeMap<String, usize> = files
        .iter()
        .enumerate()
        .map(|(i, f)| (f.file_stem().unwrap().to_string_lossy().into_owned(), i))
        .collect();
    let (mut pages, mut records, mut split) = (0, 0, 0);
    let (mut pos_err, mut pos_abs, mut rot_err, mut scale_err) = (0f64, 0f64, 0f64, 0f64);
    let (mut wide, mut widest) = (0, 0f32);
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
        let range = h.log2_scale_max - h.log2_scale_min;
        widest = widest.max(range);
        wide += usize::from(range > 2046.0 * 1.002f32.log2());
        // Half a 10-bit step of the page's log2 range, plus f32 rounding.
        let scale_bound = (f64::from(range) / 2046.0).exp2() - 1.0 + 1e-5;
        let corner = DVec3::from(h.cell.map(f64::from)) * 256.0;
        let half = h.extent.map(|e| f64::from(e) / 65535.0 / 2.0);
        let (_, decoded) = instance::decode(&head, &recs).unwrap();
        for (k, (_, m)) in decoded.enumerate() {
            let r =
                &recs_src[u32::from_le_bytes(user[4 * k..4 * k + 4].try_into().unwrap()) as usize];
            for ax in 0..3 {
                let want = f64::from(r.f[ax]) - corner[ax];
                let err = (f64::from(m.rows[ax][3]) - want).abs();
                assert!(
                    err <= half[ax] + f64::from(ulp(m.rows[ax][3])),
                    "{name}: axis {ax} error {err}, want {want} got {} origin {:?} extent {:?}",
                    m.rows[ax][3],
                    h.origin,
                    h.extent
                );
                pos_err = pos_err.max(err / (half[ax] + f64::from(ulp(m.rows[ax][3]))));
                pos_abs = pos_abs.max(err);
            }
            let col = |c: usize| {
                DVec3::new(
                    m.rows[0][c].into(),
                    m.rows[1][c].into(),
                    m.rows[2][c].into(),
                )
            };
            let mut cols = [col(0), col(1), col(2)];
            for (c, v) in cols.iter_mut().enumerate() {
                let s = v.length();
                let err = (s / f64::from(r.f[7 + c]) - 1.0).abs();
                assert!(err <= scale_bound, "{name}: scale error {err}");
                scale_err = scale_err.max(err);
                *v /= s;
            }
            let q = DQuat::from_xyzw(r.f[3].into(), r.f[4].into(), r.f[5].into(), r.f[6].into())
                .normalize();
            let d = DQuat::from_mat3(&glam::DMat3::from_cols(cols[0], cols[1], cols[2]));
            rot_err = rot_err.max(q.angle_between(d).to_degrees());
        }
        pages += 1;
        records += recs.len() / 16;
    }
    assert_eq!(records, total);
    println!(
        "{} files, {total} records, {} prototypes -> {pages} pages ({split} split pages), {} B",
        files.len(),
        ids.len(),
        std::fs::metadata(&out).unwrap().len()
    );
    println!(
        "max position error {pos_abs:.6} m, {pos_err:.4} of (half step + f32 spacing); rotation {rot_err:.4} deg; scale {:.4}%",
        scale_err * 100.0
    );
    println!(
        "pages whose scale range puts a half step over 0.2%: {wide}; widest range {:.3}x",
        widest.exp2()
    );
    assert_eq!(split, 0, "a page split");
    assert!(rot_err <= 0.17);
}
