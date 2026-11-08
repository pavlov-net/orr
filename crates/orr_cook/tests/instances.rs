mod common;
#[path = "common/assembly.rs"]
mod common_assembly;

use std::path::{Path, PathBuf};

use orr_cook::{
    ArchiveSpec, AssemblySettings, AssetSpec, Error, GltfSettings, Instances, Linear, Manifest,
    Placement, PrototypeSpec, SectionSpec, Source,
};
use orr_math::{DVec3, Mat3, Quat, Vec3};
use orr_pack::instance::{self, InstancePageInfo, InstanceRecord};
use orr_pack::{AffineRows, Archive, AssetId, Class, Entry, Target};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

fn model(path: &Path) -> AssetSpec {
    AssetSpec {
        name: "kit/block".into(),
        source: Source::Gltf {
            path: path.into(),
            settings: GltfSettings::default(),
        },
    }
}

fn proto(pose: u16) -> PrototypeSpec {
    PrototypeSpec {
        model: "kit/block".into(),
        flags: 0x5,
        pose,
        sections: vec![],
    }
}

fn cook(assets: Vec<AssetSpec>) -> orr_cook::Result<Archive> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())?
        .memory
        .remove(0);
    Ok(Archive::from_memory(bytes).unwrap())
}

fn cook_instances(src: Instances) -> orr_cook::Result<Archive> {
    cook(vec![
        model(&common::block_glb()),
        AssetSpec {
            name: "scatter".into(),
            source: Source::Instances(src),
        },
    ])
}

fn read(a: &Archive, e: &Entry, blob: usize) -> Vec<u8> {
    let b = &a.blobs(e)[blob];
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

fn pages(a: &Archive) -> Vec<&Entry> {
    a.entries()
        .iter()
        .filter(|e| e.class == Class::InstancePage as u16)
        .collect()
}

/// Random rotation, either sign, some exactly about the axes.
fn rotation(rng: &mut Rng) -> Quat {
    if rng.next().is_multiple_of(8) {
        let turns = [0.0, 90.0, 180.0, 270.0].map(f32::to_radians);
        let t = |r: &mut Rng| turns[(r.next() % 4) as usize];
        return Quat::from_rotation_y(t(rng)) * Quat::from_rotation_x(t(rng));
    }
    let q = Quat::from_xyzw(
        rng.range(-1.0, 1.0) as f32,
        rng.range(-1.0, 1.0) as f32,
        rng.range(-1.0, 1.0) as f32,
        rng.range(-1.0, 1.0) as f32,
    )
    .normalize();
    if rng.next().is_multiple_of(2) { -q } else { q }
}

/// Angle between a rotation and the rotation of a decoded linear part, its columns
/// normalized to take out the scale.
fn angle_deg(a: Quat, linear: Mat3) -> f64 {
    let col = |c: usize| linear.col(c).as_dvec3().normalize();
    let m = orr_math::DMat3::from_cols(col(0), col(1), col(2));
    a.as_dquat()
        .normalize()
        .angle_between(orr_math::DQuat::from_mat3(&m))
        .to_degrees()
}

/// Spacing of f32 values at `v`: the decode's own rounding adds up to this to the half step.
fn ulp(v: f32) -> f32 {
    v.abs().next_up() - v.abs()
}

#[test]
fn fixture_page_decodes_within_its_error_bounds() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let cell = [-3, 0, 7];
    let corner = DVec3::new(-768.0, 0.0, 1792.0);
    let n = 6000;
    let records: Vec<Placement> = (0..n)
        .map(|i| Placement {
            position: (corner
                + DVec3::new(
                    rng.range(3.0, 250.0),
                    rng.range(10.0, 40.0),
                    rng.range(0.0, 256.0),
                ))
            .to_array(),
            linear: Linear::Scaled {
                rotation: rotation(&mut rng).to_array(),
                scale: (if i % 3 == 0 {
                    Vec3::splat(rng.range(0.5, 2.0) as f32)
                } else {
                    Vec3::new(
                        rng.range(0.6, 1.8) as f32,
                        rng.range(0.6, 1.8) as f32,
                        rng.range(0.6, 1.8) as f32,
                    )
                })
                .to_array(),
            },
            prototype: (i % 3) as u16,
        })
        .collect();
    let stride = 4;
    let user: Vec<u8> = (0..n as u32).flat_map(u32::to_le_bytes).collect();
    let prototypes = vec![
        proto(0),
        proto(3),
        PrototypeSpec {
            sections: vec![
                SectionSpec {
                    material: Some("kit/block/material/0".into()),
                    hidden: false,
                },
                SectionSpec {
                    material: None,
                    hidden: true,
                },
            ],
            ..proto(255)
        },
    ];
    let a = cook_instances(Instances {
        frame: None,
        motion: None,
        records: records.clone(),
        prototypes,
        user,
        user_stride: stride,
        joined: false,
    })
    .unwrap();
    for e in a.entries() {
        a.verify(e).unwrap();
    }
    let pages = pages(&a);
    assert_eq!(pages.len(), 1);
    let e = pages[0];
    assert_eq!(a.name(e), "scatter/-3_0_7/0");

    let head = read(&a, e, 0);
    let recs = read(&a, e, 1);
    let user = read(&a, e, 2);
    let info = InstancePageInfo::parse(&head).unwrap();
    assert_eq!(usize::from(info.header.record_count), n);
    let h = info.header;
    assert_eq!(info.prototypes.len(), 3);
    assert_eq!(info.prototypes[1].pose, 3);
    assert!(
        info.prototypes
            .iter()
            .all(|p| p.model == AssetId::from_name("kit/block"))
    );
    assert!(info.sections_of(&info.prototypes[0]).is_empty());
    let over = info.sections_of(&info.prototypes[2]);
    assert_eq!(over.len(), 2);
    assert_eq!(over[0].material, AssetId::from_name("kit/block/material/0"));
    assert_eq!((over[0].hidden, over[1].hidden), (0, 1));
    assert_eq!(over[1].material, AssetId::NONE);

    let (got_cell, decoded) = instance::decode(&head, &recs).unwrap();
    assert_eq!(got_cell, cell);
    let raw: Vec<InstanceRecord> = bytemuck::pod_collect_to_vec(&recs);
    let codes: Vec<u64> = raw.iter().map(|r| instance::morton(r.pos)).collect();
    assert!(codes.is_sorted(), "records are not in Morton order");

    let step = h.extent.map(|e| f64::from(e) / 65535.0);
    let scale_half_step = f64::from(h.log2_scale_max - h.log2_scale_min) / 1023.0 / 2.0;
    let mut seen = vec![false; n];
    for (k, (prototype, m)) in decoded.enumerate() {
        let src_index = u32::from_le_bytes(user[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
        assert!(!std::mem::replace(&mut seen[src_index], true));
        let p = &records[src_index];
        assert_eq!(prototype, p.prototype);
        for a in 0..3 {
            let want = p.position[a] - corner[a];
            let got = m.translation[a];
            let err = (f64::from(got) - want).abs();
            assert!(
                err <= step[a] * 0.5 + f64::from(ulp(got)),
                "axis {a}: {err} m"
            );
        }
        let Linear::Scaled { rotation, scale } = p.linear else {
            panic!("scaled");
        };
        let decoded_scale = Vec3::from_array([0, 1, 2].map(|c| m.matrix3.col(c).length()));
        for a in 0..3 {
            let rel = (f64::from(decoded_scale[a]) / f64::from(scale[a]) - 1.0).abs();
            assert!(rel <= scale_half_step.exp2() - 1.0 + 1e-5, "scale {rel}");
            assert!(rel <= 0.002, "scale {rel}");
        }
        let err = angle_deg(Quat::from_array(rotation), m.matrix3);
        assert!(err <= 0.17, "rotation {err} deg");
        if raw[k].exact_axis() {
            assert!(err < 1e-4);
        }
    }
    assert!(raw.iter().any(InstanceRecord::exact_axis));
    // One part per model, so a record costs one PTLAS instance.
    let starts: Vec<u32> = info.partitions.iter().map(|&s| u32::from(s)).collect();
    assert!(starts.len() >= n / 1000);
    assert_eq!(
        Some(starts),
        instance::partitions(&codes, &vec![1; n], instance::PARTITION_INSTANCES)
    );
}

#[test]
fn cells_split_placements_and_large_cells_split_into_pages() {
    let mut rng = Rng(99);
    let mut records = Vec::new();
    // 40,000 in cell (0, 0, 0), a few in negative cells.
    for _ in 0..40_000 {
        records.push(Placement {
            position: DVec3::new(
                rng.range(0.0, 256.0),
                rng.range(0.0, 50.0),
                rng.range(0.0, 256.0),
            )
            .to_array(),
            linear: Linear::Scaled {
                rotation: Quat::IDENTITY.to_array(),
                scale: [1.0; 3],
            },
            prototype: 0,
        });
    }
    for x in [-0.5, -256.0, -256.5] {
        records.push(Placement {
            position: DVec3::new(x, -1.0, 3.0).to_array(),
            linear: Linear::Scaled {
                rotation: Quat::from_rotation_y(0.3).to_array(),
                scale: Vec3::splat(2.0).to_array(),
            },
            prototype: 0,
        });
    }
    let a = cook_instances(Instances {
        records,
        prototypes: vec![proto(0)],
        ..Default::default()
    })
    .unwrap();
    let mut names: Vec<&str> = pages(&a).iter().map(|e| a.name(e)).collect();
    names.sort_unstable();
    assert_eq!(names[..2], ["scatter/-1_-1_0/0", "scatter/-2_-1_0/0"]);
    let split = names.len() - 2;
    assert!(split >= 3);
    assert!((0..split).all(|i| names.contains(&format!("scatter/0_0_0/{i}").as_str())));
    let mut total = 0;
    for e in pages(&a) {
        let head = read(&a, e, 0);
        let recs = read(&a, e, 1);
        let count = usize::from(InstancePageInfo::parse(&head).unwrap().header.record_count);
        assert!(count <= instance::MAX_RECORDS);
        total += count;
        let (cell, decoded) = instance::decode(&head, &recs).unwrap();
        let corner = DVec3::from(cell.map(f64::from)) * 256.0;
        for (_, m) in decoded {
            let world = corner + m.translation.as_dvec3();
            assert_eq!((world / 256.0).floor().as_ivec3().to_array(), cell);
        }
    }
    assert_eq!(total, 40_003);
}

#[test]
fn joined_pages_decode_exactly() {
    let mut records = Vec::new();
    let step = 2f64.powi(-11);
    for i in 0..500 {
        let turn = Quat::from_rotation_y((90.0 * (i % 4) as f32).to_radians())
            * Quat::from_rotation_z((90.0 * (i % 3) as f32).to_radians());
        records.push(Placement {
            position: DVec3::new(
                100.0 + f64::from(i % 20) * 1.5 + 37.0 * step,
                f64::from(i / 20) * 0.75,
                -20.0 - f64::from(i % 7) * 2.0,
            )
            .to_array(),
            linear: Linear::Scaled {
                rotation: turn.to_array(),
                scale: [1.0; 3],
            },
            prototype: 0,
        });
    }
    let src = Instances {
        records: records.clone(),
        prototypes: vec![proto(0)],
        joined: true,
        ..Default::default()
    };
    let a = cook_instances(src.clone()).unwrap();
    let e = pages(&a)[0];
    let recs = read(&a, e, 1);
    let (cell, decoded) = instance::decode(&read(&a, e, 0), &recs).unwrap();
    assert_eq!(cell, [0, 0, -1]);
    let corner = DVec3::new(0.0, 0.0, -256.0);
    let mut got: Vec<[u32; 12]> = decoded
        .map(|(_, m)| bytemuck::cast::<[[f32; 4]; 3], [u32; 12]>(AffineRows::from(m).rows))
        .collect();
    let mut want: Vec<[u32; 12]> = records
        .iter()
        .map(|p| {
            let Linear::Scaled { rotation, .. } = p.linear else {
                panic!("scaled");
            };
            let r = Mat3::from_quat(Quat::from_array(rotation));
            let t = DVec3::from(p.position) - corner;
            let rows = [0, 1, 2].map(|i| {
                // + 0.0 turns the rounded -0.0 entries into the 0.0 the decode writes.
                let row = r.row(i).to_array().map(|v| v.round() + 0.0);
                [row[0], row[1], row[2], t[i] as f32]
            });
            bytemuck::cast(rows)
        })
        .collect();
    got.sort_unstable();
    want.sort_unstable();
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.map(f32::from_bits), w.map(f32::from_bits));
    }

    let fails = |f: &dyn Fn(&mut Placement)| {
        let mut bad = src.clone();
        f(&mut bad.records[7]);
        matches!(cook_instances(bad), Err(Error::Instances { .. }))
    };
    assert!(fails(&|p| p.position[0] += step * 0.25));
    let scaled = |rotation: Quat, scale| Linear::Scaled {
        rotation: rotation.to_array(),
        scale,
    };
    assert!(fails(
        &|p| p.linear = scaled(Quat::IDENTITY, [1.0, 1.01, 1.0])
    ));
    assert!(fails(
        &|p| p.linear = scaled(Quat::from_rotation_y(0.2), [1.0; 3])
    ));
    // Small enough that a compact record would drop it and decode exactly.
    assert!(fails(&|p| p.linear = Linear::Matrix([
        [1.0, 5e-4, 0.0],
        [5e-4, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])));
    assert!(fails(&|p| p.linear = Linear::Matrix([
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])));
}

/// A sheared copy keeps its matrix bit for bit in the page's linear table, a row per distinct
/// matrix, and leaves the scale range to the compact records; a transform a rotation and scale
/// hold stays compact.
#[test]
fn sheared_placements_take_table_rows_and_the_rest_stay_compact() {
    let shear = Mat3::from_cols(
        Vec3::new(1.0, 0.2, 0.0),
        Vec3::new(0.5, 1.0, 0.3),
        Vec3::new(0.0, -0.4, 1.5),
    );
    let mirrored = shear * Mat3::from_diagonal(Vec3::new(-1.0, 1.0, 1.0));
    // Its short columns shear by 30 degrees, a small error beside the long column.
    let wide = Mat3::from_cols(Vec3::X * 1000.0, Vec3::Y, Vec3::new(0.0, 0.5, 0.866));
    // Columns that enclose little more volume than a flat transform.
    let thin = Mat3::from_cols(Vec3::X, Vec3::new(1.0, 2e-5, 0.0), Vec3::Z);
    // Finite entries, but column lengths past the range of a float.
    let huge = 1.5e38
        * Mat3::from_cols(
            Vec3::ONE,
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, -2.0),
        );
    let rotated = Mat3::from_quat(Quat::from_rotation_y(0.7));
    let nearly = rotated * Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0))
        + Mat3::from_cols(Vec3::ZERO, rotated.col(0) * 1e-4, Vec3::ZERO);
    let tables = [shear, mirrored, wide, thin, huge, shear];
    let matrix = |x: usize, m: Mat3| Placement {
        position: [1.0 + x as f64, 1.0, 2.0],
        linear: Linear::Matrix(m.to_cols_array_2d()),
        prototype: 0,
    };
    let mut records: Vec<Placement> = tables
        .iter()
        .enumerate()
        .map(|(i, &m)| matrix(i, m))
        .collect();
    records.push(matrix(6, nearly));
    records.push(Placement {
        position: [8.0, 1.0, 2.0],
        linear: Linear::Scaled {
            rotation: Quat::from_rotation_x(0.3).to_array(),
            scale: [2.0, 4.0, 4.0],
        },
        prototype: 0,
    });
    let a = cook_instances(Instances {
        records,
        prototypes: vec![proto(0)],
        ..Default::default()
    })
    .unwrap();
    let e = pages(&a)[0];
    let (head, blob) = (read(&a, e, 0), read(&a, e, 1));
    let h = InstancePageInfo::parse(&head).unwrap().header;
    let (records, table) = instance::records(&h, &blob).unwrap();
    assert_eq!(table.len(), 5);
    assert_eq!(
        records.iter().filter(|r| r.linear_row().is_some()).count(),
        6
    );
    assert_eq!((h.log2_scale_min, h.log2_scale_max), (1.0, 2.0));
    let (_, decoded) = instance::decode(&head, &blob).unwrap();
    for (_, m) in decoded {
        let x = m.translation.x.round() as usize - 1;
        match tables.get(x) {
            Some(want) => assert_eq!(m.matrix3, *want),
            None if x == 6 => {
                assert!(m.matrix3.abs_diff_eq(nearly, 0.02), "{m:?}");
            }
            None => assert_eq!(x, 7),
        }
    }
}

/// A scale far from the rest takes a table row, so that it costs the rest no precision: a range
/// of 80 octaves would leave every compact scale 2.7% out.
#[test]
fn an_outlying_scale_leaves_the_range_to_the_rest() {
    let turned = Quat::from_rotation_y(0.7);
    let scaled = |x: usize, scale: [f32; 3]| Placement {
        position: [1.0 + x as f64, 1.0, 2.0],
        linear: Linear::Scaled {
            rotation: turned.to_array(),
            scale,
        },
        prototype: 0,
    };
    let scales = [
        [1.0, 1.1, 1.3],
        [1e-12, 1e-12, 1e-12],
        [0.7, 0.7, 0.9],
        [1.0, 1000.0, 1.0],
        [1.7, 1.2, 1.0],
        [1e12, 1e12, 1e12],
    ];
    let records: Vec<Placement> = scales
        .iter()
        .enumerate()
        .map(|(i, &s)| scaled(i, s))
        .collect();
    let a = cook_instances(Instances {
        records,
        prototypes: vec![proto(0)],
        ..Default::default()
    })
    .unwrap();
    let e = pages(&a)[0];
    let (head, blob) = (read(&a, e, 0), read(&a, e, 1));
    let h = InstancePageInfo::parse(&head).unwrap().header;
    assert!(h.log2_scale_max - h.log2_scale_min <= 6.0, "{h:?}");
    let (records, table) = instance::records(&h, &blob).unwrap();
    assert_eq!(table.len(), 3);
    let (_, decoded) = instance::decode(&head, &blob).unwrap();
    for (r, m) in records.iter().zip(decoded.map(|d| d.1)) {
        let i = m.translation.x.round() as usize - 1;
        for (axis, want) in scales[i].into_iter().enumerate() {
            let error = m.matrix3.col(axis).length() / want - 1.0;
            let bound = if r.linear_row().is_some() {
                1e-6
            } else {
                0.002
            };
            assert!(error.abs() <= bound, "copy {i} axis {axis}: {error}");
        }
    }
}

/// One rule decides what may be an instance, whatever form the transform arrives in: it inverts
/// in f32. Scale on either side does not stop that; a determinant lost to rounding does.
#[test]
fn a_transform_that_does_not_invert_is_no_placement() {
    let flat = |linear: Linear| {
        let r = cook_instances(Instances {
            records: vec![Placement {
                position: [1.0; 3],
                linear,
                prototype: 0,
            }],
            prototypes: vec![proto(0)],
            ..Default::default()
        });
        match r {
            Ok(_) => false,
            Err(Error::Instances { .. }) => true,
            Err(e) => panic!("{e}"),
        }
    };
    let matrix = |m: Mat3| Linear::Matrix(m.to_cols_array_2d());
    let scaled = |scale| Linear::Scaled {
        rotation: Quat::IDENTITY.to_array(),
        scale,
    };
    let turn = |x, y, z, w| Mat3::from_quat(Quat::from_xyzw(x, y, z, w).normalize());
    let (a, b) = (turn(1.0, 2.0, 3.0, 4.0), turn(4.0, -1.0, 2.0, 3.0));
    let z = Mat3::from_rotation_z(0.6);
    let squash = |y| Mat3::from_diagonal(Vec3::new(1.0, y, 1.0));

    assert!(flat(matrix(Mat3::from_cols(
        Vec3::X,
        Vec3::Y,
        Vec3::new(1.0, 1.0, 0.0)
    ))));
    assert!(flat(matrix(a * squash(0.0) * b)));
    assert!(flat(matrix(a * squash(1e-7) * b)));
    assert!(flat(matrix(Mat3::IDENTITY * f32::NAN)));
    assert!(flat(matrix(Mat3::IDENTITY * 1e-39)));
    assert!(flat(matrix(Mat3::from_diagonal(Vec3::new(
        1.0, 1e-25, 1e-25
    )))));
    // Its determinant is sound at unit size, but its inverse is past the range of a float.
    assert!(flat(matrix(
        1e-35
            * Mat3::from_cols(
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0001, 0.0),
                Vec3::Z
            )
    )));
    assert!(flat(scaled([1.0, 0.0, 1.0])));
    assert!(flat(Linear::Scaled {
        rotation: [0.0; 4],
        scale: [1.0; 3]
    }));

    assert!(!flat(scaled([1.0, 1e-12, 1.0])));
    assert!(!flat(matrix(z * squash(1e-6))));
    assert!(!flat(matrix(squash(1e-6) * z)));
    assert!(!flat(matrix(a * squash(1e-3) * b)));
    assert!(!flat(matrix(Mat3::IDENTITY * 1e-30)));
    assert!(!flat(matrix(Mat3::from_cols(
        Vec3::X,
        Vec3::new(1.0, 2e-5, 0.0),
        Vec3::Z
    ))));
}

#[test]
fn prototypes_outside_the_format_fail_with_named_errors() {
    let one = |p: PrototypeSpec| Instances {
        records: vec![Placement {
            position: DVec3::splat(1.0).to_array(),
            linear: Linear::Scaled {
                rotation: Quat::IDENTITY.to_array(),
                scale: [1.0; 3],
            },
            prototype: 0,
        }],
        prototypes: vec![p],
        ..Default::default()
    };
    let r = cook_instances(one(PrototypeSpec {
        model: "kit/missing".into(),
        ..proto(0)
    }));
    assert!(matches!(r, Err(Error::UnknownModel { model, .. }) if model == "kit/missing"));
    let r = cook_instances(one(proto(256)));
    assert!(matches!(r, Err(Error::PoseVariant { pose: 256, .. })));
    let r = cook_instances(one(PrototypeSpec {
        sections: vec![SectionSpec {
            material: Some("kit/other/material/0".into()),
            hidden: false,
        }],
        ..proto(0)
    }));
    assert!(matches!(r, Err(Error::UnknownMaterial { .. })));

    // Checked before the model's section count.
    let mut sections = vec![SectionSpec::default(); 17];
    sections[16].hidden = true;
    let r = cook_instances(one(PrototypeSpec {
        sections,
        ..proto(0)
    }));
    assert!(matches!(r, Err(Error::HiddenSection { section: 16, .. })));
}

#[test]
fn user_classes_pass_through() {
    let bytes: Vec<u8> = (0..700_000u32).map(|i| (i * 7 % 251) as u8).collect();
    let a = cook(vec![AssetSpec {
        name: "layers/ids".into(),
        source: Source::User {
            class: 0x8003,
            schema: 9,
            version: 2,
            bytes: bytes.clone(),
        },
    }])
    .unwrap();
    let e = a
        .expect(AssetId::from_name("layers/ids"), 0x8003, 9)
        .unwrap();
    assert_eq!(read(&a, e, 0), bytes);
    a.verify(e).unwrap();
    let r = cook(vec![AssetSpec {
        name: "bad".into(),
        source: Source::User {
            class: 7,
            schema: 1,
            version: 1,
            bytes: vec![1],
        },
    }]);
    assert!(matches!(r, Err(Error::UserClass { class: 7, .. })));
}

fn assembly_model(path: &Path) -> AssetSpec {
    let mut settings = GltfSettings::default();
    settings.mesh.assemblies = Some(AssemblySettings::default());
    AssetSpec {
        name: "kit/rack".into(),
        source: Source::Gltf {
            path: path.into(),
            settings,
        },
    }
}

fn cook_with_report(assets: Vec<AssetSpec>) -> orr_cook::Result<(Archive, orr_cook::CookReport)> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default())?;
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    Ok((a, cooked.report))
}

#[test]
fn assemblies_place_their_units_through_instance_sources() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let copies = 40;
    let placements = [
        (
            DVec3::new(10.0, 2.0, 12.0),
            Quat::from_rotation_y(0.4),
            Vec3::ONE,
        ),
        (
            DVec3::new(40.0, 1.0, 30.0),
            Quat::from_rotation_x(-1.2),
            Vec3::splat(1.5),
        ),
        (
            DVec3::new(70.0, 0.0, 80.0),
            Quat::IDENTITY,
            Vec3::splat(0.8),
        ),
    ];
    let src = Instances {
        records: placements
            .iter()
            .map(|&(position, rotation, scale)| Placement {
                position: position.to_array(),
                linear: Linear::Scaled {
                    rotation: rotation.to_array(),
                    scale: scale.to_array(),
                },
                prototype: 0,
            })
            .collect(),
        prototypes: vec![PrototypeSpec {
            model: "kit/rack".into(),
            flags: 0x5,
            pose: 0,
            sections: vec![
                SectionSpec {
                    material: Some("paint".into()),
                    hidden: false,
                },
                SectionSpec {
                    material: None,
                    hidden: true,
                },
            ],
        }],
        ..Default::default()
    };
    let paint = AssetSpec {
        name: "paint".into(),
        source: Source::Material(Default::default()),
    };
    let scatter = |src: Instances| AssetSpec {
        name: "scatter".into(),
        source: Source::Instances(src),
    };

    for sheet in [true, false] {
        let path = dir.join(format!("assembly-{sheet}.glb"));
        let centroids = common_assembly::write(&path, copies, sheet);
        let assets = vec![assembly_model(&path), paint.clone(), scatter(src.clone())];
        let (a, report) = cook_with_report(assets).unwrap();

        let unit = a
            .expect(
                AssetId::from_name("kit/rack/unit/0"),
                Class::Model as u16,
                orr_pack::model::SCHEMA,
            )
            .unwrap();
        let info = orr_pack::model::ModelInfo::parse(&read(&a, unit, 0)).unwrap();
        assert_eq!(info.header.mesh, AssetId::from_name("kit/rack/unit/0/mesh"));
        assert_eq!(info.materials, [AssetId::from_name("kit/rack/material/0")]);
        assert!(a.find(AssetId::from_name("kit/rack/unit/1")).is_none());
        assert!(a.find(AssetId::from_name("kit/rack/mesh")).is_some());

        let mut body = 0;
        let mut found: Vec<Vec3> = Vec::new();
        for e in pages(&a) {
            let head = read(&a, e, 0);
            let page = InstancePageInfo::parse(&head).unwrap();
            let edge = f64::from(ArchiveSpec::DEFAULT_CELL_LOG2).exp2();
            let corner = DVec3::from(page.header.cell.map(f64::from)) * edge;
            let recs = read(&a, e, 1);
            let (_, decoded) = instance::decode(&head, &recs).unwrap();
            for (proto, m) in decoded {
                let p = &page.prototypes[usize::from(proto)];
                let sections = &page.sections
                    [p.sections.first as usize..(p.sections.first + p.sections.count) as usize];
                assert_eq!(p.flags, 0x5);
                if p.model == AssetId::from_name("kit/rack") {
                    body += 1;
                    assert_eq!(sections.len(), 2);
                    assert_eq!(sections[1].hidden, 1);
                    continue;
                }
                assert_eq!(p.model, AssetId::from_name("kit/rack/unit/0"));
                assert_eq!(sections.len(), 1);
                assert_eq!(sections[0].material, AssetId::from_name("paint"));
                assert_eq!(sections[0].hidden, 0);
                found.push((m.translation.as_dvec3() + corner).as_vec3());
            }
        }
        assert_eq!(body, if sheet { placements.len() } else { 0 });
        assert_eq!(found.len(), placements.len() * copies);
        for &(position, rotation, scale) in &placements {
            for c in &centroids {
                let want = position.as_vec3() + rotation * (scale * *c);
                let near = found.iter().any(|f| f.distance(want) < 2e-3);
                assert!(near, "no unit copy at {want}");
            }
        }

        let asset = report.assets.iter().find(|r| r.name == "kit/rack").unwrap();
        let [ar] = &asset.assemblies[..] else {
            panic!("{:?}", asset.assemblies);
        };
        assert_eq!(ar.units.len(), 1);
        assert_eq!(ar.units[0].copies, copies as u32);
        assert_eq!(ar.units[0].triangles, common_assembly::UNIT_TRIANGLES);
        assert_eq!(ar.kept_triangles, if sheet { 128 } else { 0 });
        assert_eq!(ar.triangles, ar.kept_triangles + copies as u64 * 240);
        assert_eq!(ar.repeated_triangles, (copies as u64 - 1) * 240);
        assert!(ar.units[0].stored > 0);
        assert_eq!(ar.repeated_bytes, (copies as u64 - 1) * ar.units[0].stored);
    }

    let path = dir.join("assembly-true.glb");
    let joined = Instances {
        joined: true,
        ..src.clone()
    };
    let r = cook_with_report(vec![assembly_model(&path), paint.clone(), scatter(joined)]);
    assert!(matches!(r, Err(Error::Instances { .. })), "{:?}", r.err());
    // Uneven scale over a unit copy's rotation shears it, which a table row holds.
    let mut uneven = src;
    uneven.records[0].linear = Linear::Scaled {
        rotation: Quat::from_rotation_y(0.4).to_array(),
        scale: [1.0, 2.0, 1.0],
    };
    let (a, _) = cook_with_report(vec![assembly_model(&path), paint, scatter(uneven)]).unwrap();
    assert!(pages(&a).iter().any(|e| {
        let h = InstancePageInfo::parse(&read(&a, e, 0)).unwrap().header;
        h.linear_rows(a.blobs(e)[1].raw_len as usize).unwrap() > 0
    }));
}
