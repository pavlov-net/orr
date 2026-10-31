//! The light set: selection probabilities, the directional share, presampled light tiles and
//! emissive producer power.

use std::f64::consts::PI;

use bytemuck::Zeroable;
use glam::{DVec3, Mat3, Vec3};
use lighting_suite::shaders::{emit, hotspot_bake, nee_probe, procedural_bake, tile_probe};
use lighting_suite::{
    TestScene, affine, assert_clean, compute_barrier, finish, kernels, pipeline, read, readback,
    render, upload,
};
use orr_gpu::{ComputePipeline, GpuAddress, GpuPtr, KernelSet};
use orr_lighting::{
    DirectionalLight, DiscProfile, EmissiveSelection, LIGHT_TILE_SIZE, LIGHT_TILES, Light, LightId,
    LightSample, LightSetGpu, SceneLighting, SceneLightingSettings, ScenePass,
};
use orr_pack::texture::octahedral;
use orr_scene::{
    Affine3, ClusterTopology, Cone, EmissiveGpu, GroupWriter, InstanceFlags, MaterialGpu,
    MaterialId, MaterialWriter, ProducerDesc, ProducerId, ProducerUpdate, SceneCompute, SceneDesc,
    SceneWriter, TransformGroupDesc, material_flags,
};
use shader_maths::stats::chi_square_p;

const BAKE: &str = "environment/bake";
const SAMPLES: u32 = LIGHT_TILES * LIGHT_TILE_SIZE;

/// Luminance of linear Rec.2020 (`orr.color`).
const Y2020: [f64; 3] = [0.262700212, 0.677998072, 0.0593017165];

fn y2020(c: [f32; 3]) -> f64 {
    (0..3).map(|i| Y2020[i] * f64::from(c[i])).sum()
}

/// Records the scene pass alone, as frame `frame`.
fn scene_pass(scene: &TestScene, lighting: &mut SceneLighting, frame: u64) {
    let mut list = render().command_list().unwrap();
    lighting.record(
        &mut list,
        &ScenePass {
            frame,
            ..scene.pass([0.0; 3])
        },
    );
    finish(list);
}

/// Resolved tile samples seen from one receiver over several presampled frames.
struct Tiles {
    /// Per bin, then outside the cone.
    observed: Vec<f64>,
    expected: Vec<f64>,
    empty: u32,
    mismatches: u32,
    /// Mean and standard error of luminance x cos to the axis / density.
    irradiance: (f64, f64),
}

impl Tiles {
    fn assert_match_density(&self, what: &str) {
        let n: f64 = self.observed.iter().sum();
        assert_eq!(self.empty, 0, "{what}: empty samples");
        assert!(
            f64::from(self.mismatches) <= 1e-4 * n,
            "{what}: {} samples whose density differs from the tiles' density",
            self.mismatches
        );
        for (o, e) in self.observed.iter().zip(&self.expected) {
            assert!(
                *e > 0.0 || *o == 0.0,
                "{what}: samples in a bin of zero density"
            );
        }
        let p = chi_square_p(&self.observed, &self.expected);
        assert!(p > 0.01, "{what}: chi-square p = {p}");
    }

    fn assert_irradiance(&self, expected: f64, what: &str) {
        let (mean, error) = self.irradiance;
        let allowed = 4.0 * error + 3e-3 * expected;
        assert!(
            (mean - expected).abs() <= allowed,
            "{what}: irradiance {mean} against {expected} (allowed {allowed})"
        );
    }
}

/// Equal-solid-angle bins over the cone of half angle `half_angle` (rad) around `axis`, seen
/// from `receiver`.
#[derive(Clone, Copy)]
struct Bins {
    receiver: Vec3,
    axis: Vec3,
    half_angle: f64,
    rings: u32,
    sectors: u32,
}

impl Bins {
    fn around_y(half_angle: f64, rings: u32, sectors: u32) -> Self {
        Self {
            receiver: Vec3::ZERO,
            axis: Vec3::Y,
            half_angle,
            rings,
            sectors,
        }
    }
}

/// Eight presampled frames of `lighting`'s tiles, binned.
fn tiles(scene: &TestScene, lighting: &mut SceneLighting, bins: &Bins) -> Tiles {
    let Bins {
        receiver,
        axis,
        half_angle,
        rings,
        sectors,
    } = *bins;
    let frames = 8u32;
    let bins = rings * sectors;
    let mut histogram = readback::<u32>(u64::from(bins + 2), "histogram");
    histogram.as_mut_slice().unwrap().fill(0);
    let mut mismatches = readback::<u32>(1, "mismatches");
    mismatches.as_mut_slice().unwrap().fill(0);
    let probability = readback::<f32>(u64::from(bins), "bin probability");
    let values = readback::<f32>(u64::from(SAMPLES), "values");
    let omc = 2.0 * (0.5 * half_angle).sin().powi(2);
    let probe = [
        pipeline(&tile_probe::MODULE, "histogram"),
        pipeline(&tile_probe::MODULE, "integrate"),
    ];
    let (mut sum, mut sum2) = (0.0f64, 0.0f64);
    for f in 0..frames {
        let frame = 1000 + u64::from(f);
        let root = upload(
            &[tile_probe::Root {
                set: lighting.light_set(),
                histogram: histogram.gpu,
                mismatches: mismatches.gpu,
                bin_probability: probability.gpu,
                values: values.gpu,
                _pad: 0,
                receiver: receiver.extend(0.0).to_array(),
                axis: axis.normalize().extend(omc as f32).to_array(),
                rings,
                sectors,
                subdivisions: 96,
                frame: frame as u32,
            }],
            "tile probe root",
        );
        let mut list = render().command_list().unwrap();
        lighting.record(
            &mut list,
            &ScenePass {
                frame,
                ..scene.pass([0.0; 3])
            },
        );
        compute_barrier(&mut list);
        list.dispatch(&probe[0], root.gpu, [SAMPLES / 64, 1, 1]);
        if f == 0 {
            list.dispatch(&probe[1], root.gpu, [bins, 1, 1]);
        }
        finish(list);
        for &v in values.as_slice().unwrap() {
            let v = f64::from(v);
            sum += v;
            sum2 += v * v;
        }
    }
    let n = f64::from(SAMPLES) * f64::from(frames);
    let h = histogram.as_slice().unwrap();
    let p: Vec<f64> = probability
        .as_slice()
        .unwrap()
        .iter()
        .map(|&p| f64::from(p))
        .collect();
    let inside: f64 = p.iter().sum();
    let mut expected: Vec<f64> = p.iter().map(|p| p * n).collect();
    expected.push((1.0 - inside).max(0.0) * n);
    let mean = sum / n;
    Tiles {
        observed: h[..=bins as usize].iter().map(|&c| f64::from(c)).collect(),
        expected,
        empty: h[bins as usize + 1],
        mismatches: mismatches.as_slice().unwrap()[0],
        irradiance: (mean, ((sum2 / n - mean * mean) / n).sqrt()),
    }
}

/// Irradiance at the origin, on a surface facing `n`, from a Lambertian polygon of luminance
/// `luminance` whose corners `v` all lie above the surface (Lambert's formula).
fn polygon_irradiance(v: &[DVec3], n: DVec3, luminance: f64) -> f64 {
    let mut s = 0.0;
    for i in 0..v.len() {
        let (a, b) = (v[i].normalize(), v[(i + 1) % v.len()].normalize());
        s += a.angle_between(b) * n.dot(a.cross(b).normalize());
    }
    0.5 * luminance * s.abs()
}

#[test]
fn environment_tiles_match_the_density() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 4,
        ..SceneDesc::default()
    });
    scene.write(|_, _| {});
    let mut lighting = scene.lighting(&kernels(&[(&procedural_bake::MODULE, BAKE)]));
    let t = tiles(&scene, &mut lighting, &Bins::around_y(PI, 16, 32));
    t.assert_match_density("procedural sky");
    assert_clean();
}

#[test]
fn a_half_degree_hot_spot_is_sampled_at_its_own_density() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 4,
        ..SceneDesc::default()
    });
    scene.write(|_, _| {});
    let mut lighting = scene.lighting(&kernels(&[(&hotspot_bake::MODULE, BAKE)]));
    let spot = Vec3::from_array(octahedral::to_dir([100.25 / 256.0, 60.25 / 256.0]));
    let bins = Bins {
        axis: spot,
        ..Bins::around_y(2f64.to_radians(), 8, 8)
    };
    let t = tiles(&scene, &mut lighting, &bins);
    t.assert_match_density("hot spot");
    let n: f64 = t.observed.iter().sum();
    let near = 1.0 - t.observed[t.observed.len() - 1] / n;
    assert!(
        near > 0.005,
        "{near} of the samples within 2 deg of the hot spot"
    );
    assert_clean();
}

#[test]
fn sphere_tiles_match_the_density_and_irradiance() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 4,
        ..SceneDesc::default()
    });
    let (a, b) = (DVec3::new(0.0, 3.0, 0.0), DVec3::new(5.0, 1.0, -4.0));
    scene.write(|w, _| {
        w.set_light(
            LightId(1),
            Some(&Light::Sphere {
                position: a,
                radius: 0.5,
                intensity: Vec3::new(100.0, 80.0, 60.0),
                cone: Some(Cone {
                    direction: -Vec3::Y,
                    inner: 30f32.to_radians(),
                    outer: 40f32.to_radians(),
                }),
            }),
        )
        .unwrap();
        w.set_light(
            LightId(3),
            Some(&Light::Sphere {
                position: b,
                radius: 0.25,
                intensity: Vec3::splat(500.0),
                cone: None,
            }),
        )
        .unwrap();
    });
    let mut lighting = scene.lighting(&KernelSet::default());
    // The cap of sphere a ends on a ring boundary.
    let rings = 8;
    let cap = 2.0 * (0.5 * (0.5f64 / 3.0).asin()).sin().powi(2);
    let half_angle = 2.0
        * (0.5 * cap * f64::from(rings) / f64::from(rings - 2))
            .sqrt()
            .asin();
    let t = tiles(
        &scene,
        &mut lighting,
        &Bins::around_y(half_angle, rings, 16),
    );
    t.assert_match_density("spheres");
    let lights = scene.lights();
    // A sphere wholly above the horizon gives I cos / d^2.
    let e = |l: usize, p: DVec3| y2020(lights[l].color) * p.normalize().y / p.length_squared();
    t.assert_irradiance(e(1, a) + e(3, b), "spheres");
    assert_clean();
}

#[test]
fn rect_tiles_match_the_density_and_irradiance() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 4,
        ..SceneDesc::default()
    });
    let center = DVec3::new(0.5, 2.0, 0.3);
    let (ex, ey) = (Vec3::new(1.2, 0.0, 0.1), Vec3::new(-0.8 / 12.0, 0.3, 0.8));
    scene.write(|w, _| {
        w.set_light(
            LightId(0),
            Some(&Light::Rect {
                center,
                edge_x: ex,
                edge_y: ey,
                luminance: Vec3::new(3000.0, 2500.0, 2000.0),
            }),
        )
        .unwrap();
        // Faces away from the receiver: sampled, but it adds no irradiance.
        w.set_light(
            LightId(2),
            Some(&Light::Rect {
                center: DVec3::new(-1.0, 1.5, -1.0),
                edge_x: Vec3::new(0.0, 0.0, 0.6),
                edge_y: Vec3::new(0.6, 0.0, 0.0),
                luminance: Vec3::splat(500.0),
            }),
        )
        .unwrap();
    });
    let mut lighting = scene.lighting(&KernelSet::default());
    let t = tiles(
        &scene,
        &mut lighting,
        &Bins::around_y(70f64.to_radians(), 8, 16),
    );
    t.assert_match_density("rects");
    let (ex, ey) = (ex.as_dvec3(), ey.as_dvec3());
    let corners = [
        center - 0.5 * ex - 0.5 * ey,
        center + 0.5 * ex - 0.5 * ey,
        center + 0.5 * ex + 0.5 * ey,
        center - 0.5 * ex + 0.5 * ey,
    ];
    let y = y2020(scene.lights()[0].color);
    t.assert_irradiance(polygon_irradiance(&corners, DVec3::Y, y), "rects");
    assert_clean();
}

fn emitting(color: [f32; 3], luminance: f32, flags: u32) -> MaterialGpu {
    let mut m = MaterialGpu::zeroed();
    m.emission_color = color;
    m.emission_luminance = luminance;
    m.flags = material_flags::EMISSIVE | flags;
    m
}

/// What `emit.slang` emits for one producer entry.
#[derive(Clone, Copy)]
struct Quads {
    producer: u32,
    count: u32,
    size: f32,
    origin: Vec3,
    step: Vec3,
}

impl Quads {
    /// Corners of quad `c`, relative to the entry's transform.
    fn corners(&self, c: u32) -> [Vec3; 4] {
        let o = self.origin + c as f32 * self.step;
        let s = self.size;
        [
            o,
            o + Vec3::new(s, 0.0, 0.0),
            o + Vec3::new(s, 0.0, s),
            o + Vec3::new(0.0, 0.0, s),
        ]
    }

    fn dispatch(&self, c: &mut SceneCompute<'_, '_>, p: &ComputePipeline) {
        let root = c
            .put_root(&emit::Root {
                scene: c.root,
                producer: self.producer,
                count: self.count,
                size: self.size,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                origin: self.origin.extend(0.0).to_array(),
                step: self.step.extend(0.0).to_array(),
            })
            .unwrap();
        c.cmd.dispatch(p, root, [self.count.div_ceil(64), 1, 1]);
    }
}

/// Materials 0-2 of the emitters: warm and single-sided, dark, cool and double-sided.
fn producer_materials(m: &mut MaterialWriter<'_>) -> [MaterialId; 3] {
    [
        m.create(&emitting([1.0, 0.6, 0.3], 400.0, 0)).unwrap(),
        m.create(&MaterialGpu::zeroed()).unwrap(),
        m.create(&emitting(
            [0.3, 0.6, 1.0],
            150.0,
            material_flags::DOUBLE_SIDED,
        ))
        .unwrap(),
    ]
}

/// Producer entries `0..transforms.len()`, each with its transform in one group anchored at
/// `anchor`.
fn set_producers(
    w: &mut SceneWriter<'_>,
    materials: &[MaterialId; 3],
    anchor: DVec3,
    transforms: &[Affine3],
    update: ProducerUpdate,
) {
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: transforms.len() as u32,
            writer: GroupWriter::Cpu,
            anchor,
        })
        .unwrap();
    w.transforms(g, 0, transforms.len() as u32)
        .unwrap()
        .copy_from_slice(transforms);
    let topology = [ClusterTopology {
        vertex_count: 4,
        triangles: &[[0, 1, 2], [0, 2, 3]],
    }];
    for i in 0..transforms.len() as u32 {
        w.set_producer(
            ProducerId(i),
            Some(&ProducerDesc {
                max_clusters: 32,
                max_vertices: 128,
                update,
                topologies: &topology,
                materials,
                transform: (g, i),
                previous_positions: false,
                vertex_user: false,
                flags: InstanceFlags::RASTER | InstanceFlags::RAY,
                user_tag: 0,
                ..ProducerDesc::default()
            }),
        )
        .unwrap();
    }
}

#[test]
fn emissive_producer_tiles_match_the_density_and_irradiance() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 4,
        producer_capacity: 4,
        emissive_capacity: 0,
        ..SceneDesc::default()
    });
    // Entry 0 faces the receiver from y = 2; entry 1 is turned over at y = 2.5 beside it, so
    // only its double-sided quads light the receiver.
    let flip = Mat3::from_rotation_x(PI as f32);
    let transforms = [
        affine(Mat3::IDENTITY, Vec3::new(0.0, 2.0, 0.0)),
        affine(flip, Vec3::new(0.0, 2.5, 0.0)),
    ];
    let quads = [
        Quads {
            producer: 0,
            count: 6,
            size: 0.4,
            origin: Vec3::new(-1.6, 0.0, -0.2),
            step: Vec3::new(0.55, 0.0, 0.05),
        },
        Quads {
            producer: 1,
            count: 6,
            size: 0.3,
            origin: Vec3::new(-1.0, 0.0, -2.0),
            step: Vec3::new(0.4, 0.0, 0.1),
        },
    ];
    let p = pipeline(&emit::MODULE, "main");
    let mut materials = [MaterialId(0); 3];
    scene.frame(
        |w, m| {
            materials = producer_materials(m);
            set_producers(
                w,
                &materials,
                DVec3::ZERO,
                &transforms,
                ProducerUpdate::Static,
            );
        },
        |c| quads.iter().for_each(|q| q.dispatch(c, &p)),
    );
    let mut lighting = scene.lighting(&KernelSet::default());
    let t = tiles(
        &scene,
        &mut lighting,
        &Bins::around_y(70f64.to_radians(), 8, 16),
    );
    t.assert_match_density("emitters");

    let luminance = [
        y2020([1.0, 0.6, 0.3]) * 400.0,
        0.0,
        y2020([0.3, 0.6, 1.0]) * 150.0,
    ];
    let mut e = 0.0;
    for (q, m) in quads.iter().zip([Mat3::IDENTITY, flip]) {
        let t = Vec3::new(0.0, if q.producer == 0 { 2.0 } else { 2.5 }, 0.0);
        for c in 0..q.count {
            let visible = q.producer == 0 || c % 3 == 2;
            if visible {
                let v = q.corners(c).map(|p| (m * p + t).as_dvec3());
                e += polygon_irradiance(&v, DVec3::Y, luminance[(c % 3) as usize]);
            }
        }
    }
    t.assert_irradiance(e, "emitters");
    assert_clean();
}

/// Runs the scene pass, then the NEE probe's `threads` candidate threads and its totals pass.
fn run_nee_probe(
    scene: &TestScene,
    lighting: &mut SceneLighting,
    root: GpuPtr<nee_probe::Root>,
    threads: u32,
) {
    let probe = [
        pipeline(&nee_probe::MODULE, "candidates"),
        pipeline(&nee_probe::MODULE, "totals"),
    ];
    let mut list = render().command_list().unwrap();
    lighting.record(&mut list, &scene.pass([0.0; 3]));
    compute_barrier(&mut list);
    list.dispatch(&probe[0], root, [threads / 64, 1, 1]);
    list.dispatch(&probe[1], root, [1, 1, 1]);
    finish(list);
}

#[test]
fn directional_share_matches_the_lights_weights() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 8,
        ..SceneDesc::default()
    });
    let sun = |d: Vec3, e: f32, r: f32| DirectionalLight {
        direction_to_light: d.normalize(),
        illuminance: Vec3::splat(e),
        angular_radius: r,
        disc: DiscProfile::Uniform,
    };
    let lights = [
        sun(Vec3::new(0.3, 0.8, 0.2), 1.0e5, 0.05),
        DirectionalLight {
            disc: DiscProfile::LimbDarkened,
            ..sun(Vec3::new(-0.5, 0.6, 0.1), 5.0e4, 0.02)
        },
        DirectionalLight {
            disc: DiscProfile::Textured {
                albedo: orr_gpu::TextureIndex::NONE,
                up: Vec3::Y,
                lit_by: [LightId(0), LightId::NONE, LightId::NONE, LightId::NONE],
            },
            ..sun(Vec3::new(0.1, 0.9, -0.3), 0.0, 0.03)
        },
        sun(Vec3::new(0.2, -0.5, 0.7), 2.0e4, 0.01),
        sun(Vec3::new(0.9, 0.05, 0.1), 3.0e4, 0.04),
    ];
    // LightIds 0, 2, 3, 5, 6 with a sphere at 1, so the directional list skips a slot.
    let ids = [0u32, 2, 3, 5, 6];
    scene.write(|w, _| {
        for (&id, l) in ids.iter().zip(&lights) {
            w.set_light(LightId(id), Some(&Light::Directional(*l)))
                .unwrap();
        }
        w.set_light(
            LightId(1),
            Some(&Light::Sphere {
                position: DVec3::new(0.0, 4.0, 0.0),
                radius: 0.1,
                intensity: Vec3::splat(50.0),
                cone: None,
            }),
        )
        .unwrap();
    });
    let mut lighting = scene.lighting(&kernels(&[(&procedural_bake::MODULE, BAKE)]));
    let (rings, sectors, frames, threads) = (4u32, 8u32, 16u32, 1u32 << 16);
    let per_light = rings * sectors;
    let bins = ids.len() as u32 * per_light + 2;
    let mut histogram = readback::<u32>(u64::from(bins), "histogram");
    histogram.as_mut_slice().unwrap().fill(0);
    let mut checks = readback::<u32>(3, "checks");
    checks.as_mut_slice().unwrap().fill(0);
    let normals = [
        Vec3::Y,
        -Vec3::Y,
        Vec3::X,
        Vec3::ZERO,
        Vec3::new(-0.3, -0.2, -0.9).normalize(),
    ];
    let normals_gpu = upload(&normals.map(|n| n.extend(0.0).to_array()), "normals");
    let totals = readback::<[f32; 4]>(normals.len() as u64, "totals");
    let root = upload(
        &[nee_probe::Root {
            set: lighting.light_set(),
            histogram: histogram.gpu,
            checks: checks.gpu,
            totals: totals.gpu,
            normals: normals_gpu.gpu,
            _pad: 0,
            position: [0.0; 4],
            normal: [0.0, 1.0, 0.0, 0.0],
            rings,
            sectors,
            frames,
            normal_count: normals.len() as u32,
        }],
        "nee probe root",
    );
    run_nee_probe(&scene, &mut lighting, root.gpu, threads);

    let table = scene.lights();
    let states = read(lighting.directional_states(), 8);
    let weight = |id: u32, n: Vec3| {
        let w = f64::from(states[id as usize].weight);
        let l = Vec3::from_array(table[id as usize].direction);
        if n == Vec3::ZERO {
            w
        } else {
            w * f64::from(n.dot(l).max(0.0))
        }
    };
    for (k, &id) in ids.iter().enumerate() {
        let expected = y2020(states[id as usize].focus_illuminance);
        assert!(
            (f64::from(states[id as usize].weight) / expected - 1.0).abs() < 1e-5,
            "light {k}: weight {} against {expected}",
            states[id as usize].weight
        );
    }

    let n = f64::from(threads * frames);
    let total: f64 = ids.iter().map(|&id| weight(id, Vec3::Y)).sum();
    let mut expected = Vec::new();
    for &id in &ids {
        let p = 0.5 * weight(id, Vec3::Y) / total;
        expected.extend((0..per_light).map(|_| p / f64::from(per_light) * n));
    }
    expected.push(0.5 * n);
    expected.push(0.0);
    let observed: Vec<f64> = histogram
        .as_slice()
        .unwrap()
        .iter()
        .map(|&c| f64::from(c))
        .collect();
    for (o, e) in observed.iter().zip(&expected) {
        assert!(*e > 0.0 || *o == 0.0, "candidates where the weight is zero");
    }
    let p = chi_square_p(&observed, &expected);
    assert!(p > 0.01, "chi-square p = {p}");
    let c = checks.as_slice().unwrap();
    assert_eq!(
        c[0], 0,
        "directional densities differ from nee_directional_pdf"
    );
    assert_eq!(c[1], 0, "candidates with non-finite values");

    for (i, &nrm) in normals.iter().enumerate() {
        let [sum, share, total, _] = totals.as_slice().unwrap()[i];
        let w: f64 = ids.iter().map(|&id| weight(id, nrm)).sum();
        let expected = if w > 0.0 { 0.5 } else { 0.0 };
        assert_eq!(f64::from(share), expected, "normal {nrm}: share");
        assert!(
            (f64::from(total) - w).abs() <= 1e-5 * w,
            "normal {nrm}: weights {total} against {w}"
        );
        assert!(
            (sum - share).abs() < 1e-6,
            "normal {nrm}: directional probabilities sum to {sum}, the share is {share}"
        );
    }
    assert_clean();
}

#[test]
fn selection_probabilities_follow_power_and_sum_to_one() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 8,
        producer_capacity: 2,
        emissive_capacity: 4,
        ..SceneDesc::default()
    });
    let quads = Quads {
        producer: 1,
        count: 5,
        size: 0.5,
        origin: Vec3::ZERO,
        step: Vec3::new(1.0, 0.0, 0.0),
    };
    let p = pipeline(&emit::MODULE, "main");
    scene.frame(
        |w, m| {
            let materials = producer_materials(m);
            set_producers(
                w,
                &materials,
                DVec3::new(10.0, 5.0, 0.0),
                &[Affine3::IDENTITY, Affine3::IDENTITY],
                ProducerUpdate::Static,
            );
            for (id, i) in [(0u32, 100.0f32), (4, 900.0)] {
                w.set_light(
                    LightId(id),
                    Some(&Light::Sphere {
                        position: DVec3::new(f64::from(id), 3.0, 0.0),
                        radius: 0.2,
                        intensity: Vec3::splat(i),
                        cone: None,
                    }),
                )
                .unwrap();
            }
            w.set_light(
                LightId(2),
                Some(&Light::Rect {
                    center: DVec3::new(0.0, 3.0, 2.0),
                    edge_x: Vec3::new(2.0, 0.0, 0.0),
                    edge_y: Vec3::new(0.0, 0.0, 0.5),
                    luminance: Vec3::splat(2000.0),
                }),
            )
            .unwrap();
            for id in [3u32, 6] {
                w.set_light(
                    LightId(id),
                    Some(&Light::Directional(DirectionalLight {
                        direction_to_light: Vec3::new(0.2, 1.0, 0.1 * id as f32).normalize(),
                        illuminance: Vec3::splat(1e4 * id as f32),
                        angular_radius: 0.01,
                        disc: DiscProfile::Uniform,
                    })),
                )
                .unwrap();
            }
        },
        |c| quads.dispatch(c, &p),
    );
    let mut lighting = scene.lighting(&kernels(&[(&procedural_bake::MODULE, BAKE)]));
    scene_pass(&scene, &mut lighting, scene.frame);
    let stats = lighting.stats(scene.frame).unwrap();
    assert_eq!(
        (
            stats.directional_lights,
            stats.local_lights,
            stats.emissive_entries,
            stats.emissive_clusters,
            stats.dropped_entries
        ),
        (2, 3, 1, 5, 0)
    );

    let set: LightSetGpu = read(lighting.light_set(), 1)[0];
    let light_p = read(set.light_probability, 8);
    let emissive_p: Vec<EmissiveSelection> = read(set.emissive, 6);
    let power: Vec<EmissiveGpu> = read(scene.root_value().emissive, 6);
    let env = f64::from(set.environment_probability);
    assert_eq!(env, 0.25);
    let lights = scene.lights();
    let mut weights = [0.0f64; 8];
    for id in [0usize, 4] {
        weights[id] = 4.0 * PI * y2020(lights[id].color);
    }
    weights[2] = PI * y2020(lights[2].color) * 2.0 * 0.5;
    let emissive_w = PI * y2020(power[1].power);
    let total: f64 = weights.iter().sum::<f64>() + emissive_w;
    let close = |got: f32, w: f64, what: &str| {
        let expected = (1.0 - env) * w / total;
        assert!(
            (f64::from(got) - expected).abs() <= 1e-5 * expected.max(1e-3),
            "{what}: probability {got} against {expected}"
        );
    };
    for id in 0..8 {
        close(light_p[id], weights[id], &format!("light {id}"));
    }
    close(emissive_p[1].probability, emissive_w, "emissive entry");
    for (i, e) in emissive_p.iter().enumerate().filter(|&(i, _)| i != 1) {
        assert_eq!(e.probability, 0.0, "entry {i}");
    }
    let sum: f64 = env
        + light_p.iter().map(|&p| f64::from(p)).sum::<f64>()
        + emissive_p
            .iter()
            .map(|e| f64::from(e.probability))
            .sum::<f64>();
    assert!((sum - 1.0).abs() < 1e-5, "tile probabilities sum to {sum}");
    assert_clean();
}

/// Producer entry 0 emits `quads` under `linear`; the scene's and lighting's summed power
/// against the analytic area x luminance.
fn check_power(
    scene: &TestScene,
    lighting: &mut SceneLighting,
    quads: Quads,
    linear: Mat3,
    what: &str,
) {
    scene_pass(scene, lighting, scene.frame);
    let set: LightSetGpu = read(lighting.light_set(), 1)[0];
    let selection: EmissiveSelection = read(set.emissive, 1)[0];
    let entry: EmissiveGpu = read(scene.root_value().emissive, 1)[0];
    let luminance = [
        y2020([1.0, 0.6, 0.3]) * 400.0,
        0.0,
        y2020([0.3, 0.6, 1.0]) * 150.0,
    ];
    let scale = f64::from(linear.determinant().abs()).powf(2.0 / 3.0);
    let expected: f64 = (0..quads.count)
        .map(|c| f64::from(quads.size).powi(2) * scale * luminance[(c % 3) as usize])
        .sum();
    let scene_power = y2020(entry.power);
    for (got, source) in [
        (scene_power, "scene power"),
        (f64::from(selection.total), "cluster CDF total"),
    ] {
        assert!(
            (got / expected - 1.0).abs() < 1e-3,
            "{what}: {source} {got} against {expected}"
        );
    }
    assert_eq!(selection.count, quads.count, "{what}: clusters in the CDF");
}

#[test]
fn producer_power_equals_area_times_luminance_after_it_moves() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 2,
        producer_capacity: 1,
        emissive_capacity: 0,
        ..SceneDesc::default()
    });
    let linear = Mat3::from_rotation_z(0.4) * Mat3::from_diagonal(Vec3::splat(1.5));
    let first = Quads {
        producer: 0,
        count: 7,
        size: 0.5,
        origin: Vec3::ZERO,
        step: Vec3::new(1.0, 0.0, 0.0),
    };
    let p = pipeline(&emit::MODULE, "main");
    scene.frame(
        |w, m| {
            let materials = producer_materials(m);
            set_producers(
                w,
                &materials,
                DVec3::new(3000.0, 10.0, 7000.0),
                &[affine(linear, Vec3::new(4.0, 0.0, -2.0))],
                ProducerUpdate::EveryFrame,
            );
        },
        |c| first.dispatch(c, &p),
    );
    let mut lighting = scene.lighting(&KernelSet::default());
    check_power(&scene, &mut lighting, first, linear, "first frame");

    let moved = Quads {
        count: 11,
        size: 0.8,
        origin: Vec3::new(-3.0, 1.0, 2.0),
        step: Vec3::new(0.5, 0.2, 1.0),
        ..first
    };
    scene.frame(|_, _| {}, |c| moved.dispatch(c, &p));
    check_power(&scene, &mut lighting, moved, linear, "after moving");

    // A cluster pool too small for the entry leaves it out of the selection.
    let mut small = SceneLighting::new(
        lighting_suite::device(),
        &KernelSet::default(),
        &scene.desc,
        &SceneLightingSettings {
            emissive_clusters: 4,
            ..SceneLightingSettings::default()
        },
    )
    .unwrap();
    scene_pass(&scene, &mut small, scene.frame);
    let stats = small.stats(scene.frame).unwrap();
    assert_eq!((stats.emissive_entries, stats.dropped_entries), (0, 1));
    assert_clean();
}

#[test]
fn a_scene_without_lights_gives_empty_candidates_and_no_nan() {
    let mut scene = TestScene::new(SceneDesc {
        light_capacity: 4,
        producer_capacity: 2,
        emissive_capacity: 4,
        ..SceneDesc::default()
    });
    scene.write(|w, _| w.set_environment(GpuAddress(0)));
    let mut lighting = scene.lighting(&KernelSet::default());
    let threads = 1u32 << 14;
    let mut histogram = readback::<u32>(2, "histogram");
    histogram.as_mut_slice().unwrap().fill(0);
    let mut checks = readback::<u32>(3, "checks");
    checks.as_mut_slice().unwrap().fill(0);
    let totals = readback::<[f32; 4]>(1, "totals");
    let normals = upload(&[[0.0, 1.0, 0.0, 0.0f32]], "normals");
    let root = upload(
        &[nee_probe::Root {
            set: lighting.light_set(),
            histogram: histogram.gpu,
            checks: checks.gpu,
            totals: totals.gpu,
            normals: normals.gpu,
            _pad: 0,
            position: [1.0, 2.0, 3.0, 0.0],
            normal: [0.0, 1.0, 0.0, 0.0],
            rings: 1,
            sectors: 1,
            frames: 4,
            normal_count: 1,
        }],
        "nee probe root",
    );
    run_nee_probe(&scene, &mut lighting, root.gpu, threads);

    let c = checks.as_slice().unwrap();
    assert_eq!((c[1], c[2]), (0, 0), "non-finite or non-empty candidates");
    assert_eq!(histogram.as_slice().unwrap(), &[0, threads * 4]);
    assert_eq!(totals.as_slice().unwrap()[0], [0.0; 4]);
    let set: LightSetGpu = read(lighting.light_set(), 1)[0];
    assert_eq!(
        (set.candidate_total, set.environment_probability),
        (0.0, 0.0)
    );
    let tiles: Vec<LightSample> = read(set.tiles, u64::from(SAMPLES));
    assert!(
        tiles.iter().all(|t| t.pdf == 0.0),
        "a tile sample with a density"
    );
    let sky = lighting.stats(scene.frame).unwrap().sky;
    assert_eq!(
        (sky.dome_luminance, sky.directional),
        (0.0, [0.0; 4]),
        "sky statistics {sky:?}"
    );
    assert_clean();
}
