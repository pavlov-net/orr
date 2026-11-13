//! `orr_pack.instance`'s `decode_instance` on the GPU against `orr_pack::instance::
//! decode_record`: bit for bit on joined pages, exact rows and compact translations, within the
//! GPU's rounding for compact linear parts.

use orr_cook::{
    ArchiveSpec, AssetSpec, Instances, Linear, Manifest, MeshInput, Placement, PrototypeSpec,
    Source,
};
use orr_gpu::MemoryKind;
use orr_math::{DVec3, Mat3, Quat, Vec2, Vec3};
use orr_pack::instance::{
    self, AXIS_ROTATIONS, InstancePageHeader, InstancePageInfo, InstanceRecord,
};
use orr_pack::material::Material;
use orr_pack::{AffineRows, Archive, Class, Span, Target};
use orr_scene::testing::{assert_clean, device};
use pack_suite::shaders::instances;
use pack_suite::{dispatch, read};

/// The GPU decode of every record.
fn decode_on_gpu(
    header: &InstancePageHeader,
    records: &[InstanceRecord],
    table: &[AffineRows],
) -> Vec<AffineRows> {
    let d = device();
    let mut h = d
        .gpu_malloc::<InstancePageHeader>(MemoryKind::Upload, 1, "header")
        .unwrap();
    h.as_mut_slice().unwrap()[0] = *header;
    let mut r = d
        .gpu_malloc::<InstanceRecord>(MemoryKind::Upload, records.len() as u64, "records")
        .unwrap();
    r.as_mut_slice().unwrap().copy_from_slice(records);
    let mut t = d
        .gpu_malloc::<AffineRows>(MemoryKind::Upload, table.len().max(1) as u64, "table")
        .unwrap();
    t.as_mut_slice().unwrap()[..table.len()].copy_from_slice(table);
    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 3 * records.len() as u64, "out")
        .unwrap();
    let mut root = d
        .gpu_malloc::<instances::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = instances::Root {
        header: h.gpu(),
        records: r.gpu(),
        table: t.gpu(),
        out: out.gpu(),
        count: records.len() as u32,
        _pad0: 0,
        _pad1: 0,
    };
    dispatch(
        &instances::MODULE,
        root.gpu(),
        records.len().div_ceil(64) as u32,
    );
    assert_clean();
    out.as_slice()
        .unwrap()
        .as_chunks::<3>()
        .0
        .iter()
        .map(|m| AffineRows { rows: *m })
        .collect()
}

/// Largest difference between GPU and CPU decodes of compact linear parts, relative to the
/// record's largest scale.
#[derive(Debug, Default)]
struct Spread(f32);

impl Spread {
    /// The GPU's `exp2`, `sqrt` and the quaternion's products are not correctly rounded (Vulkan
    /// allows `exp2` 3 + 2|x| ulp) and it may fuse multiply-adds; on 621.01 the linear part
    /// differs by up to 3 ulp at 1.0.
    fn assert_within(&self) {
        let bound = 2f32.powi(-20);
        assert!(self.0 <= bound, "{} against {bound}", self.0);
    }
}

/// Exact rows decode as stored and compact translations exactly, the same on both; compact
/// linear parts add to `spread`.
fn check_page(
    header: &InstancePageHeader,
    records: &[InstanceRecord],
    table: &[AffineRows],
    spread: &mut Spread,
) {
    let gpu = decode_on_gpu(header, records, table);
    let bits = |m: &AffineRows| bytemuck::cast::<_, [u32; 12]>(m.rows);
    for (g, r) in gpu.iter().zip(records) {
        let cpu = AffineRows::from(instance::decode_record(header, table, r));
        if r.exact_row().is_some() {
            assert_eq!(bits(g), bits(&cpu), "an exact row decodes as stored");
            continue;
        }
        let scale = (0..3)
            .flat_map(|r| (0..3).map(move |c| cpu.rows[r][c].abs()))
            .fold(f32::MIN_POSITIVE, f32::max);
        for (g, c) in g.rows.iter().zip(&cpu.rows) {
            assert_eq!(
                g[3].to_bits(),
                c[3].to_bits(),
                "{r:?}: GPU {g:?}, CPU {c:?}"
            );
            for k in 0..3 {
                spread.0 = spread.0.max((g[k] - c[k]).abs() / scale);
            }
        }
    }
}

/// A 1 m quad, as the prototype every placement names.
fn piece() -> MeshInput {
    MeshInput {
        positions: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ],
        normals: vec![Vec3::Y; 4],
        uvs: Some(vec![Vec2::ZERO, Vec2::X, Vec2::Y, Vec2::ONE]),
        indices: vec![0, 2, 1, 1, 2, 3],
        sections: vec![Span { first: 0, count: 6 }],
        ..Default::default()
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }
}

/// Placements over three cells (one negative), random rotations of either sign, some about
/// the axes only, and uniform and non-uniform scales, every fifth sheared.
fn scattered(rng: &mut Rng) -> Vec<Placement> {
    Vec::from_fn(5000, |i| {
        let rotation = if i.is_multiple_of(7) {
            let turn = |r: &mut Rng| (90.0 * (r.next() % 4) as f32).to_radians();
            Quat::from_rotation_y(turn(rng)) * Quat::from_rotation_x(turn(rng))
        } else {
            let q = Quat::from_xyzw(
                rng.range(-1.0, 1.0) as f32,
                rng.range(-1.0, 1.0) as f32,
                rng.range(-1.0, 1.0) as f32,
                rng.range(-1.0, 1.0) as f32,
            )
            .normalize();
            if i.is_multiple_of(2) { -q } else { q }
        };
        let scale = if i % 4 == 1 {
            Vec3::ONE
        } else if i.is_multiple_of(3) {
            Vec3::splat(rng.range(0.25, 4.0) as f32)
        } else {
            Vec3::new(
                rng.range(0.5, 2.0) as f32,
                rng.range(0.5, 2.0) as f32,
                rng.range(0.5, 2.0) as f32,
            )
        };
        let linear = if i.is_multiple_of(5) {
            let shear = Mat3::from_cols(
                Vec3::X,
                Vec3::new(rng.range(0.1, 1.0) as f32, 1.0, 0.0),
                Vec3::Z,
            );
            let m = Mat3::from_quat(rotation) * shear * Mat3::from_diagonal(scale);
            Linear::Matrix(m.to_cols_array_2d())
        } else {
            Linear::Scaled {
                rotation: rotation.to_array(),
                scale: scale.to_array(),
            }
        };
        Placement {
            position: DVec3::new(
                rng.range(-300.0, 500.0),
                rng.range(-20.0, 60.0),
                rng.range(-100.0, 100.0),
            )
            .to_array(),
            linear,
            prototype: 0,
        }
    })
}

/// Placements on the piece's 2^-11 m grid, turned by multiples of 90 degrees.
fn joined() -> Vec<Placement> {
    let step = 2f64.powi(-11);
    (0..500)
        .map(|i| Placement {
            position: DVec3::new(
                100.0 + f64::from(i % 20) * 1.25 + 37.0 * step,
                f64::from(i / 20) * 0.75 - 3.0,
                -20.0 - f64::from(i % 7) * 2.0 + 5.0 * step,
            )
            .to_array(),
            linear: Linear::Scaled {
                rotation: (Quat::from_rotation_y((90.0 * (i % 4) as f32).to_radians())
                    * Quat::from_rotation_z((90.0 * (i % 3) as f32).to_radians()))
                .to_array(),
                scale: [1.0; 3],
            },
            prototype: 0,
        })
        .collect()
}

#[test]
fn cooked_pages_decode_alike() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let instances = |records, joined| {
        Source::Instances(Instances {
            records,
            prototypes: vec![PrototypeSpec {
                model: "kit/piece".into(),
                ..Default::default()
            }],
            joined,
            ..Default::default()
        })
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                AssetSpec {
                    name: "kit/piece".into(),
                    source: Source::Mesh {
                        mesh: piece(),
                        materials: vec![Material::default()],
                        settings: Default::default(),
                    },
                },
                AssetSpec {
                    name: "scatter".into(),
                    source: instances(scattered(&mut rng), false),
                },
                AssetSpec {
                    name: "joined".into(),
                    source: instances(joined(), true),
                },
            ],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let a = Archive::from_memory(bytes).unwrap();
    let mut spread = Spread::default();
    let (mut scattered_pages, mut joined_pages, mut rows, mut compact) = (0, 0, 0, 0);
    for e in a.entries() {
        if e.class != Class::InstancePage as u16 {
            continue;
        }
        let blobs = a.blobs(e);
        let header = InstancePageInfo::parse(&read(&a, &blobs[0]))
            .unwrap()
            .header;
        let (records, table) = instance::records(&header, &read(&a, &blobs[1])).unwrap();
        if a.name(e).starts_with("joined/") {
            joined_pages += 1;
            let gpu = decode_on_gpu(&header, &records, &table);
            for (i, (g, r)) in gpu.iter().zip(&records).enumerate() {
                let cpu = AffineRows::from(instance::decode_record(&header, &table, r));
                assert_eq!(
                    bytemuck::cast::<_, [u32; 12]>(g.rows),
                    bytemuck::cast::<_, [u32; 12]>(cpu.rows),
                    "joined record {i}: GPU {g:?}, CPU {cpu:?}"
                );
            }
        } else {
            scattered_pages += 1;
            rows += table.len();
            compact += records.iter().filter(|r| r.exact_row().is_none()).count();
            check_page(&header, &records, &table, &mut spread);
        }
    }
    assert!(scattered_pages >= 3 && joined_pages >= 1 && rows >= 1000 && compact >= 50);
    spread.assert_within();
}

/// Rotations at the encoding's edges, either sign: the axes, two or four components tied
/// for largest, and near-ties that pick the other component.
fn edge_rotations() -> Vec<[f32; 4]> {
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let mut qs = Vec::new();
    for a in 0..4 {
        let mut q = [0.0; 4];
        q[a] = 1.0;
        qs.push(q);
        for b in a + 1..4 {
            for (sa, sb) in [(1.0, 1.0), (1.0, -1.0)] {
                let mut q = [0.0; 4];
                (q[a], q[b]) = (h * sa, h * sb);
                qs.push(q);
                (q[a], q[b]) = (h * sa * 1.0001, h * sb * 0.9999);
                qs.push(q);
            }
        }
    }
    for signs in 0..16 {
        qs.push([0, 1, 2, 3].map(|i| if signs >> i & 1 != 0 { -0.5 } else { 0.5 }));
    }
    let negated: Vec<[f32; 4]> = qs.iter().map(|q| q.map(|c| -c)).collect();
    qs.extend(negated);
    qs
}

#[test]
fn edge_codes_decode_alike() {
    let header = InstancePageHeader {
        cell: [-2, 0, 5],
        origin: [-3.25, 17.0, 0.125],
        extent: [250.5, 40.0, 255.875],
        step: [2f32.powi(-13), 2f32.powi(-15), 2f32.powi(-13)],
        log2_scale_min: -2.0,
        log2_scale_step: 2f32.powi(-8),
        ..bytemuck::Zeroable::zeroed()
    };
    let rotations = edge_rotations()
        .into_iter()
        .map(|q| (instance::encode_rotation(q), 0))
        .chain((0..AXIS_ROTATIONS + 2).map(|i| (i, InstanceRecord::INDEXED)));
    let records: Vec<InstanceRecord> = rotations
        .enumerate()
        .map(|(n, (rot, flags))| {
            let s = [n % 4, n / 4 % 4, n / 16 % 4].map(|i| [0, 1, 511, 1023][i]);
            let p = (n * 7919) as u16;
            InstanceRecord {
                pos: [p, p.wrapping_mul(3), 65535 - p],
                prototype: 0,
                rot,
                scale: s[0] | s[1] << 10 | s[2] << 20 | flags,
                rank: n as u16,
                fine: (n * 4099) as u16 & 0x7fff,
            }
        })
        .collect();
    let table = [
        AffineRows {
            rows: [
                [1.0, 0.5, -0.25, 3.0e-3],
                [0.0, 2.0, 0.75, 17.5],
                [-0.5, 0.0, 3.0, -1.0e-7],
            ],
        },
        AffineRows {
            rows: [
                [-1.0, 0.125, 0.0, 255.9],
                [0.375, 1.0, 0.0, 0.0],
                [0.0, 0.625, 0.5, 64.25],
            ],
        },
    ];
    let mut spread = Spread::default();
    check_page(&header, &records, &table, &mut spread);
    spread.assert_within();
}
