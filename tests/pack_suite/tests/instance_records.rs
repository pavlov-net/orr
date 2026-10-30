//! `orr_pack.instance`'s `decode_instance` on the GPU against `orr_pack::instance::
//! decode_record`: bit for bit on joined pages, within the GPU's rounding elsewhere.

use glam::{DVec3, Quat, Vec2, Vec3};
use orr_cook::{
    ArchiveSpec, AssetSpec, Instances, Manifest, MeshInput, Placement, PrototypeSpec, Source,
};
use orr_gpu::MemoryKind;
use orr_pack::instance::{self, InstancePageHeader, InstancePageInfo, InstanceRecord};
use orr_pack::material::Material;
use orr_pack::{Affine3, Archive, Class, Span, Target};
use orr_scene::testing::{assert_clean, device};
use pack_suite::shaders::instances;
use pack_suite::{dispatch, read};

/// The GPU decode of every record.
fn decode_on_gpu(header: &InstancePageHeader, records: &[InstanceRecord]) -> Vec<Affine3> {
    let d = device();
    let mut h = d
        .gpu_malloc::<InstancePageHeader>(MemoryKind::Upload, 1, "header")
        .unwrap();
    h.as_mut_slice().unwrap()[0] = *header;
    let mut r = d
        .gpu_malloc::<InstanceRecord>(MemoryKind::Upload, records.len() as u64, "records")
        .unwrap();
    r.as_mut_slice().unwrap().copy_from_slice(records);
    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 3 * records.len() as u64, "out")
        .unwrap();
    let mut root = d
        .gpu_malloc::<instances::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = instances::Root {
        header: h.gpu,
        records: r.gpu,
        out: out.gpu,
        count: records.len() as u32,
        _pad: 0,
    };
    dispatch(
        &instances::MODULE,
        root.gpu,
        records.len().div_ceil(64) as u32,
    );
    assert_clean();
    out.as_slice()
        .unwrap()
        .chunks_exact(3)
        .map(|m| Affine3 {
            rows: [m[0], m[1], m[2]],
        })
        .collect()
}

/// Largest differences between GPU and CPU decodes over a set of records.
#[derive(Debug, Default)]
struct Spread {
    /// Linear part, relative to the record's largest scale.
    linear: f32,
    /// Translation, relative to the largest `|origin| + extent` axis of the page.
    translation: f32,
}

impl Spread {
    fn add(&mut self, header: &InstancePageHeader, gpu: &Affine3, cpu: &Affine3) {
        let scale = (0..3)
            .flat_map(|r| (0..3).map(move |c| cpu.rows[r][c].abs()))
            .fold(f32::MIN_POSITIVE, f32::max);
        let reach = (0..3)
            .map(|a| header.origin[a].abs() + header.extent[a])
            .fold(f32::MIN_POSITIVE, f32::max);
        for (g, c) in gpu.rows.iter().zip(&cpu.rows) {
            for k in 0..3 {
                self.linear = self.linear.max((g[k] - c[k]).abs() / scale);
            }
            self.translation = self.translation.max((g[3] - c[3]).abs() / reach);
        }
    }

    /// The GPU's `exp2` is not correctly rounded (Vulkan allows 3 + 2|x| ulp) and it may
    /// fuse the decode's multiply-adds; on 621.01 the linear part differs by up to
    /// 3 ulp at 1.0 and translations by 1 ulp of the page's reach.
    fn assert_within(&self) {
        let bounds = Spread {
            linear: 2f32.powi(-20),
            translation: 2f32.powi(-22),
        };
        assert!(
            self.linear <= bounds.linear && self.translation <= bounds.translation,
            "{self:?} against {bounds:?}"
        );
    }
}

fn check_page(header: &InstancePageHeader, records: &[InstanceRecord], spread: &mut Spread) {
    let gpu = decode_on_gpu(header, records);
    for (g, r) in gpu.iter().zip(records) {
        spread.add(header, g, &instance::decode_record(header, r));
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
/// the axes only, and uniform and non-uniform scales.
fn scattered(rng: &mut Rng) -> Vec<Placement> {
    (0..5000)
        .map(|i| {
            let rotation = if i % 7 == 0 {
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
                if i % 2 == 0 { -q } else { q }
            };
            let scale = if i % 3 == 0 {
                Vec3::splat(rng.range(0.25, 4.0) as f32)
            } else {
                Vec3::new(
                    rng.range(0.5, 2.0) as f32,
                    rng.range(0.5, 2.0) as f32,
                    rng.range(0.5, 2.0) as f32,
                )
            };
            Placement {
                position: DVec3::new(
                    rng.range(-300.0, 500.0),
                    rng.range(-20.0, 60.0),
                    rng.range(-100.0, 100.0),
                ),
                rotation,
                scale,
                prototype: 0,
            }
        })
        .collect()
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
            ),
            rotation: Quat::from_rotation_y((90.0 * (i % 4) as f32).to_radians())
                * Quat::from_rotation_z((90.0 * (i % 3) as f32).to_radians()),
            scale: Vec3::ONE,
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
    let (mut scattered_pages, mut joined_pages) = (0, 0);
    for e in a.entries() {
        if e.class != Class::InstancePage as u16 {
            continue;
        }
        let blobs = a.blobs(e);
        let header = InstancePageInfo::parse(&read(&a, &blobs[0]))
            .unwrap()
            .header;
        let records: Vec<InstanceRecord> = bytemuck::pod_collect_to_vec(&read(&a, &blobs[1]));
        if a.name(e).starts_with("joined/") {
            joined_pages += 1;
            let gpu = decode_on_gpu(&header, &records);
            for (i, (g, r)) in gpu.iter().zip(&records).enumerate() {
                let cpu = instance::decode_record(&header, r);
                assert_eq!(
                    bytemuck::cast::<_, [u32; 12]>(g.rows),
                    bytemuck::cast::<_, [u32; 12]>(cpu.rows),
                    "joined record {i}: GPU {g:?}, CPU {cpu:?}"
                );
            }
        } else {
            scattered_pages += 1;
            check_page(&header, &records, &mut spread);
        }
    }
    assert!(scattered_pages >= 3 && joined_pages >= 1);
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
        log2_scale_min: -2.0,
        log2_scale_max: 3.5,
        ..bytemuck::Zeroable::zeroed()
    };
    let rotations = edge_rotations()
        .into_iter()
        .map(|q| (instance::encode_rotation(q), 0))
        .chain((0..24).map(|i| (i, InstanceRecord::EXACT_AXIS)));
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
            }
        })
        .collect();
    let mut spread = Spread::default();
    check_page(&header, &records, &mut spread);
    spread.assert_within();
}
