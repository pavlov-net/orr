mod common;
#[path = "common/assembly.rs"]
mod common_assembly;

use std::path::{Path, PathBuf};

use orr_cook::{
    ArchiveSpec, AssemblySettings, AssetSpec, Error, GltfSettings, Instances, Linear, Manifest,
    MeshInput, Placement, PrototypeSpec, SectionSpec, Source,
};
use orr_math::{Affine3, DAffine3, DMat3, DQuat, DVec3, Mat3, Quat, Vec2, Vec3};
use orr_pack::instance::{self, InstancePageInfo, InstanceRecord};
use orr_pack::material::Material;
use orr_pack::{Aabb, AffineRows, Archive, AssetId, Class, Entry, Span, Target};

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

/// The grid step of the fixture models' meshes, 2^-11 m.
const GRID_STEP: f64 = 1.0 / 2048.0;

/// The bounds of the block fixture.
const BLOCK: Aabb = Aabb {
    min: [0.0; 3],
    max: [1.0; 3],
};

/// Where a placement puts its model, relative to the corner of cell `cell`.
fn source(p: &Placement, cell: [i32; 3]) -> DAffine3 {
    let corner = DVec3::from(cell.map(f64::from)) * 256.0;
    let linear = match p.linear {
        Linear::Scaled { rotation, scale } => {
            DMat3::from_quat(DQuat::from_array(rotation.map(f64::from)).normalize())
                * DMat3::from_diagonal(Vec3::from(scale).as_dvec3())
        }
        Linear::Matrix(columns) => Mat3::from_cols_array_2d(&columns).as_dmat3(),
    };
    DAffine3::from_mat3_translation(linear, DVec3::from(p.position) - corner)
}

fn dvec(m: Affine3) -> DAffine3 {
    DAffine3::from_mat3_translation(m.matrix3.as_dmat3(), m.translation.as_dvec3())
}

/// A copy's compact record keeps every point of the model within half a grid step of the
/// copy's scale; a row holds the f32 transform, whose rounding moves points far less.
fn assert_placed(r: &InstanceRecord, p: &Placement, cell: [i32; 3], m: Affine3, bounds: &Aabb) {
    let want = source(p, cell);
    let error = instance::displacement(&want, &dvec(m), bounds);
    let l = want.matrix3;
    let s = [l.x_axis.length(), l.y_axis.length(), l.z_axis.length()];
    let bound = if r.exact_row().is_some() {
        let reach = Vec3::from(bounds.min)
            .abs()
            .max(Vec3::from(bounds.max).abs())
            .length();
        1e-6 * (1.0 + want.translation.length() + s[0].max(s[1]).max(s[2]) * f64::from(reach))
    } else {
        0.5 * GRID_STEP * s[0].min(s[1]).min(s[2])
    };
    assert!(
        error <= bound,
        "{r:?} moves the model {error} m, past {bound}"
    );
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
                scale: (if i % 4 == 0 {
                    Vec3::ONE
                } else if i % 3 == 0 {
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
    let (raw, table) = instance::records(&h, &recs).unwrap();
    let codes: Vec<u64> = raw
        .iter()
        .map(|r| instance::morton(instance::grid_position(r)))
        .collect();
    assert!(codes.is_sorted(), "records are not in Morton order");
    // Each prototype's records rank from 0 in record order, up to its count.
    let mut ranks = vec![0u16; info.prototypes.len()];
    for r in &raw {
        let next = &mut ranks[usize::from(r.prototype)];
        assert_eq!(
            r.rank, *next,
            "rank of a record of prototype {}",
            r.prototype
        );
        *next += 1;
    }
    let counts: Vec<u16> = info.prototypes.iter().map(|p| p.records).collect();
    assert_eq!(counts, ranks);

    let mut seen = vec![false; n];
    let mut turned = 0;
    for (k, (prototype, m)) in decoded.enumerate() {
        let src_index = u32::from_le_bytes(user[k * 4..k * 4 + 4].try_into().unwrap()) as usize;
        assert!(!std::mem::replace(&mut seen[src_index], true));
        let p = &records[src_index];
        assert_eq!(prototype, p.prototype);
        assert_placed(&raw[k], p, cell, m, &BLOCK);
        if let Some(row) = raw[k].exact_row() {
            assert_eq!(table[row], AffineRows::from(m));
        }
        // Axis turns at scale 1 hold exactly but for the translation's grid.
        if let Linear::Scaled { rotation, scale } = p.linear
            && instance::exact_axis_index(rotation).is_some()
            && scale == [1.0; 3]
        {
            assert!(raw[k].exact_axis(), "{p:?}");
            turned += 1;
        }
    }
    assert!(turned > n / 100);
    // One part per model, so a record costs one PTLAS instance.
    let starts: Vec<u32> = info.partitions.iter().map(|&s| u32::from(s)).collect();
    assert!(starts.len() >= n / 1000);
    assert_eq!(
        starts,
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

/// A sheared copy keeps its transform bit for bit in the page's exact table, a row per distinct
/// transform, and leaves the scale grid to the compact records; a transform a compact record
/// holds exactly stays compact.
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
    let scaled = |x: f64, rotation: Quat, scale| Placement {
        position: [x, 1.0, 2.0],
        linear: Linear::Scaled {
            rotation: rotation.to_array(),
            scale,
        },
        prototype: 0,
    };
    records.push(scaled(8.0, Quat::from_rotation_x(0.31), [2.0, 4.0, 4.0]));
    records.push(scaled(9.0, Quat::IDENTITY, [2.0; 3]));
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
    assert_eq!(table.len(), 8);
    assert_eq!(
        records.iter().filter(|r| r.exact_row().is_some()).count(),
        8
    );
    assert_eq!(h.log2_scale_min, 1.0);
    let (_, decoded) = instance::decode(&head, &blob).unwrap();
    for (_, m) in decoded {
        let x = m.translation.x.round() as usize - 1;
        let t = Vec3::new(x as f32 + 1.0, 1.0, 2.0);
        match tables.get(x) {
            Some(want) => assert_eq!((m.matrix3, m.translation), (*want, t)),
            None if x == 6 => assert_eq!(m.matrix3, nearly),
            None if x == 8 => {
                let want =
                    Affine3::from_scale_rotation_translation(Vec3::splat(2.0), Quat::IDENTITY, t);
                assert_eq!(m, want);
            }
            None => assert_eq!(x, 7),
        }
    }
}

/// A scale far from the rest takes a table row, so that it costs the rest no precision: a range
/// of 80 octaves would leave no compact scale within the copies' tolerance. Scales on the grid,
/// 1 and other powers of two, decode exactly.
#[test]
fn an_outlying_scale_leaves_the_range_to_the_rest() {
    let scaled = |x: usize, scale: [f32; 3]| Placement {
        position: [1.0 + x as f64, 1.0, 2.0],
        linear: Linear::Scaled {
            rotation: Quat::from_rotation_y(90f32.to_radians()).to_array(),
            scale,
        },
        prototype: 0,
    };
    let scales = [
        [1.0, 2.0, 0.5],
        [1e-12, 1e-12, 1e-12],
        [0.5, 0.5, 4.0],
        [1.0, 1000.0, 1.0],
        [1.0, 1.1, 1.3],
        [1e12, 1e12, 1e12],
        [1.0; 3],
    ];
    let records: Vec<Placement> = scales
        .iter()
        .enumerate()
        .map(|(i, &s)| scaled(i, s))
        .collect();
    let a = cook_instances(Instances {
        records: records.clone(),
        prototypes: vec![proto(0)],
        ..Default::default()
    })
    .unwrap();
    let e = pages(&a)[0];
    let (head, blob) = (read(&a, e, 0), read(&a, e, 1));
    let h = InstancePageInfo::parse(&head).unwrap().header;
    // The finest power-of-two step over the rest's three octaves.
    assert_eq!((h.log2_scale_min, h.log2_scale_step), (-1.0, 2f32.powi(-8)));
    let (raw, table) = instance::records(&h, &blob).unwrap();
    assert_eq!(table.len(), 4);
    let (cell, decoded) = instance::decode(&head, &blob).unwrap();
    for (r, m) in raw.iter().zip(decoded.map(|d| d.1)) {
        let i = m.translation.x.round() as usize - 1;
        assert_placed(r, &records[i], cell, m, &BLOCK);
        let on_grid = [0, 2, 6].contains(&i);
        assert_eq!(r.exact_row().is_none(), on_grid, "copy {i}");
        if on_grid {
            let col = |c: usize| m.matrix3.col(c).length();
            assert_eq!([col(0), col(1), col(2)], scales[i], "copy {i}");
        }
    }
}

/// The page's box holds every decoded translation: the highest copy here is two thirds of the
/// 2^-13 m step past a grid point, so its compact record rounds up past where it was placed.
#[test]
fn the_page_box_holds_a_copy_its_record_rounds_up() {
    let at = |x: f64| Placement {
        position: [x, 1.0, 2.0],
        linear: Linear::Scaled {
            rotation: Quat::IDENTITY.to_array(),
            scale: [1.0; 3],
        },
        prototype: 0,
    };
    let a = cook_instances(Instances {
        records: vec![at(0.0), at(250.00008)],
        prototypes: vec![proto(0)],
        ..Default::default()
    })
    .unwrap();
    let e = pages(&a)[0];
    let (head, blob) = (read(&a, e, 0), read(&a, e, 1));
    let h = InstancePageInfo::parse(&head).unwrap().header;
    let (records, _) = instance::records(&h, &blob).unwrap();
    assert!(records.iter().all(|r| r.exact_row().is_none()));
    let (_, decoded) = instance::decode(&head, &blob).unwrap();
    let top = decoded.map(|d| d.1.translation.x).fold(0.0, f32::max);
    assert!(f64::from(top) > 250.00008);
    assert!(top <= h.origin[0] + h.extent[0], "{top} past {h:?}");
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
        // A mesh whose triangles all went to units holds nothing, and neither does its model.
        let rack = a
            .expect(
                AssetId::from_name("kit/rack"),
                Class::Model as u16,
                orr_pack::model::SCHEMA,
            )
            .unwrap();
        let rack = orr_pack::model::ModelInfo::parse(&read(&a, rack, 0)).unwrap();
        assert_eq!(rack.header.bounds == Aabb::EMPTY, !sheet);

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
    // Uneven scale over a unit copy's rotation shears it, which an exact row holds.
    let mut uneven = src;
    uneven.records[0].linear = Linear::Scaled {
        rotation: Quat::from_rotation_y(0.4).to_array(),
        scale: [1.0, 2.0, 1.0],
    };
    let (a, _) = cook_with_report(vec![assembly_model(&path), paint, scatter(uneven)]).unwrap();
    assert!(pages(&a).iter().any(|e| {
        let h = InstancePageInfo::parse(&read(&a, e, 0)).unwrap().header;
        h.exact_rows(a.blobs(e)[1].raw_len as usize).unwrap() > 0
    }));
}

/// A flat annular wedge from 0 to `degrees` about y, radii 19.5 to 20.5 m, its points less
/// `center`. Turning it by `degrees` about y puts the edge at 0 where the edge at `degrees` was.
fn wedge(degrees: f32, center: Vec3) -> (MeshInput, [Vec3; 4]) {
    let at = |deg: f32, r: f32, h: f32| {
        let a = deg.to_radians();
        Vec3::new(r * a.cos(), h, -r * a.sin()) - center
    };
    let positions = vec![
        at(0.0, 19.5, 0.0),
        at(0.0, 20.5, 0.0),
        at(degrees, 19.5, 0.0),
        at(degrees, 20.5, 0.0),
        at(0.0, 19.5, 1.0),
        at(0.0, 20.5, 1.0),
        at(degrees, 19.5, 1.0),
        at(degrees, 20.5, 1.0),
    ];
    let mesh = MeshInput {
        positions: positions.clone(),
        normals: vec![Vec3::Y; 8],
        uvs: Some(vec![Vec2::ZERO; 8]),
        indices: vec![0, 1, 2, 2, 1, 3, 4, 6, 5, 6, 7, 5],
        sections: vec![Span {
            first: 0,
            count: 12,
        }],
        ..Default::default()
    };
    let start = [positions[0], positions[1], positions[4], positions[5]];
    (mesh, start)
}

fn mesh_asset(name: &str, mesh: MeshInput) -> AssetSpec {
    AssetSpec {
        name: name.into(),
        source: Source::Mesh {
            mesh,
            materials: vec![Material::default()],
            settings: Default::default(),
        },
    }
}

/// Every record of the archive's pages: its prototype's model, its record and its world
/// transform.
fn world_records(a: &Archive) -> Vec<(AssetId, InstanceRecord, DAffine3)> {
    let mut out = Vec::new();
    for e in pages(a) {
        let head = read(a, e, 0);
        let info = InstancePageInfo::parse(&head).unwrap();
        let blob = read(a, e, 1);
        let (raw, _) = instance::records(&info.header, &blob).unwrap();
        let (cell, decoded) = instance::decode(&head, &blob).unwrap();
        let corner = DVec3::from(cell.map(f64::from)) * 256.0;
        for (r, (p, m)) in raw.into_iter().zip(decoded) {
            let mut m = dvec(m);
            m.translation += corner;
            out.push((info.prototypes[usize::from(p)].model, r, m));
        }
    }
    out
}

/// A ring of 16 wedges at 22.5 degree steps, 20 m from their pivots: half turned about the
/// ring's center as one model, half as a model centered on its own wedge, so their pivots fall in
/// two cells and pages. Every wedge's end meets the next one's start within a grid step.
#[test]
fn a_ring_of_wedges_meets_across_pages() {
    let step = 22.5f32;
    let (ring, start) = wedge(step, Vec3::ZERO);
    let center = Vec3::from_array(
        start
            .iter()
            .fold([0.0; 3], |c, p| [0, 1, 2].map(|a| c[a] + p[a] * 0.25)),
    );
    let (centered, _) = wedge(step, center);
    let pivot = DVec3::new(259.0, 3.0, 100.0);
    let turn = |k: usize| Quat::from_rotation_y((step * k as f32).to_radians());
    let records: Vec<Placement> = (0..16)
        .map(|k| {
            let (position, prototype) = if k < 8 {
                (pivot, 0)
            } else {
                (pivot + (turn(k) * center).as_dvec3(), 1)
            };
            Placement {
                position: position.to_array(),
                linear: Linear::Scaled {
                    rotation: turn(k).to_array(),
                    scale: [1.0; 3],
                },
                prototype,
            }
        })
        .collect();
    let a = cook(vec![
        mesh_asset("kit/wedge", ring),
        mesh_asset("kit/wedge-centered", centered),
        AssetSpec {
            name: "ring".into(),
            source: Source::Instances(Instances {
                records,
                prototypes: ["kit/wedge", "kit/wedge-centered"]
                    .map(|model| PrototypeSpec {
                        model: model.into(),
                        ..Default::default()
                    })
                    .to_vec(),
                ..Default::default()
            }),
        },
    ])
    .unwrap();
    assert_eq!(pages(&a).len(), 2);
    // Each copy as placed, sorted by its turn: the x axis of turn k points at -22.5 k degrees.
    let mut placed = world_records(&a);
    placed.sort_by_key(|(_, _, m)| {
        let x = m.matrix3.x_axis;
        (((-x.z).atan2(x.x).to_degrees() / 22.5).round() as i64).rem_euclid(16)
    });
    assert_eq!(placed.len(), 16);
    let end = |k: usize, i: usize| {
        let (model, _, m) = &placed[k];
        let lift = if *model == AssetId::from_name("kit/wedge") {
            Vec3::ZERO
        } else {
            center
        };
        // The end edge is the start edge turned by one step.
        m.transform_point3((turn(1) * start[i] - lift).as_dvec3())
    };
    let begin = |k: usize, i: usize| {
        let (model, _, m) = &placed[k];
        let lift = if *model == AssetId::from_name("kit/wedge") {
            Vec3::ZERO
        } else {
            center
        };
        m.transform_point3((start[i] - lift).as_dvec3())
    };
    for k in 0..16 {
        for i in 0..4 {
            let gap = end(k, i).distance(begin((k + 1) % 16, i));
            assert!(
                gap <= GRID_STEP,
                "wedges {k} and {}: {gap} m apart",
                (k + 1) % 16
            );
        }
    }
    // Turns of 90 degrees hold exactly in a compact record; the others move points 20 m out
    // too far and take rows.
    let compact = placed.iter().filter(|p| p.1.exact_row().is_none()).count();
    assert_eq!(compact, 4);
}

/// Small copies turned any way and scaled a little keep compact records over a whole cell.
#[test]
fn a_field_of_small_copies_stays_compact() {
    // On the mesh grid, so the cooked bounds are these.
    const EDGE: f32 = 31.0 / 2048.0;
    const HEIGHT: f32 = 8.0 / 2048.0;
    let mut rng = Rng(0xa076_1d64_78bd_642f);
    let pebble = MeshInput {
        positions: vec![
            Vec3::ZERO,
            Vec3::new(EDGE, 0.0, 0.0),
            Vec3::new(0.0, 0.0, EDGE),
            Vec3::new(EDGE, HEIGHT, EDGE),
        ],
        normals: vec![Vec3::Y; 4],
        uvs: Some(vec![Vec2::ZERO; 4]),
        indices: vec![0, 2, 1, 1, 2, 3],
        sections: vec![Span { first: 0, count: 6 }],
        ..Default::default()
    };
    let records: Vec<Placement> = (0..5000)
        .map(|_| Placement {
            position: [
                rng.range(0.0, 256.0),
                rng.range(0.0, 30.0),
                rng.range(0.0, 256.0),
            ],
            linear: Linear::Scaled {
                rotation: rotation(&mut rng).to_array(),
                scale: [rng.range(0.9, 1.1) as f32; 3],
            },
            prototype: 0,
        })
        .collect();
    let a = cook(vec![
        mesh_asset("kit/pebble", pebble),
        AssetSpec {
            name: "field".into(),
            source: Source::Instances(Instances {
                records: records.clone(),
                prototypes: vec![PrototypeSpec {
                    model: "kit/pebble".into(),
                    ..Default::default()
                }],
                user: (0..5000u32).flat_map(u32::to_le_bytes).collect(),
                user_stride: 4,
                ..Default::default()
            }),
        },
    ])
    .unwrap();
    let e = pages(&a)[0];
    let (head, blob, user) = (read(&a, e, 0), read(&a, e, 1), read(&a, e, 2));
    let h = InstancePageInfo::parse(&head).unwrap().header;
    let (raw, table) = instance::records(&h, &blob).unwrap();
    assert!(table.is_empty());
    let bounds = Aabb {
        min: [0.0; 3],
        max: [EDGE, HEIGHT, EDGE],
    };
    let (cell, decoded) = instance::decode(&head, &blob).unwrap();
    for (k, (r, (_, m))) in raw.iter().zip(decoded).enumerate() {
        let i = u32::from_le_bytes(user[4 * k..4 * k + 4].try_into().unwrap()) as usize;
        assert_placed(r, &records[i], cell, m, &bounds);
    }
}
