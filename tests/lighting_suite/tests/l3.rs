//! The light set: selection probabilities, the directional share, presampled light tiles and
//! emissive producer power.

use std::f64::consts::PI;
use std::time::Duration;

use bytemuck::Zeroable;
use lighting_suite::shaders::{emit, hotspot_bake, local_probe, nee_probe, tile_probe};
use lighting_suite::{
    TestScene, assert_clean, compute_barrier, copy_back, finish, kernels, linear_clamp, pipeline,
    read, readback, render, texture, upload,
};
use orr_gpu::{Allocation, ComputePipeline, DEFAULT_MEDIUM, GpuPtr, Hazards, Stages};
use orr_lighting::types::{LocalLight, LocalLights};
use orr_lighting::{
    DirectionalLight, DiscProfile, EmissiveSelection, LIGHT_TILE_SIZE, LIGHT_TILES, Light, LightId,
    LightSample, LightSetGpu, SceneLighting, SceneLightingSettings, ScenePass,
};
use orr_math::{Affine3, DVec3, Mat3, Vec3};
use orr_pack::texture::octahedral;
use orr_scene::{
    ClusterTopology, Cone, EmissiveGpu, EnvironmentData, FrameDesc, FrameTime, GroupWriter,
    InstanceFlags, LightTables, MaterialGpu, MaterialId, MaterialWriter, Opacity, ProducerDesc,
    ProducerId, ProducerUpdate, SceneCompute, SceneDesc, SceneWriter, TransformGroupDesc,
    material_flags,
};
use shader_math::stats::chi_square_p;

const BAKE: &str = "environment/bake";
const SAMPLES: u32 = LIGHT_TILES * LIGHT_TILE_SIZE;

/// `hotspot_bake.slang`'s `HotSpot`, which reads no record.
struct HotSpotData;

impl EnvironmentData for HotSpotData {
    const TYPE: &'static str = "HotSpot";
    const NULL_SAFE: bool = true;
}

fn y2020(c: [f32; 3]) -> f64 {
    orr_color::luminance(DVec3::new(c[0].into(), c[1].into(), c[2].into()))
}

/// Records the scene pass alone, as frame `frame`.
fn scene_pass(scene: &TestScene, lighting: &mut SceneLighting, frame: u64) {
    let mut list = render().command_list().unwrap();
    lighting
        .record(
            &mut list,
            &ScenePass {
                frame,
                ..scene.pass([0.0; 3])
            },
        )
        .unwrap();
    finish(list);
}

/// Light samples seen from one receiver, binned by direction, against the bins' probabilities.
struct Binned {
    /// Per bin, then outside the cone.
    observed: Vec<f64>,
    expected: Vec<f64>,
    empty: u32,
    mismatches: u32,
}

impl Binned {
    /// `histogram`'s bins and the samples outside the cone and empty, against the probability
    /// in each bin of `n` samples, from the 64 lane `partials` of each.
    fn new(histogram: &[u32], partials: &[f32], mismatches: u32, n: f64) -> Self {
        let bins = partials.len() / 64;
        let p: Vec<f64> = partials
            .chunks(64)
            .map(|lanes| lanes.iter().map(|&p| f64::from(p)).sum())
            .collect();
        let inside: f64 = p.iter().sum();
        let mut expected: Vec<f64> = p.iter().map(|p| p * n).collect();
        expected.push((1.0 - inside).max(0.0) * n);
        Self {
            observed: histogram[..=bins].iter().map(|&c| f64::from(c)).collect(),
            expected,
            empty: histogram[bins + 1],
            mismatches,
        }
    }

    fn assert_match_density(&self, what: &str) {
        let n: f64 = self.observed.iter().sum();
        assert_eq!(self.empty, 0, "{what}: empty samples");
        assert!(
            f64::from(self.mismatches) <= 1e-4 * n,
            "{what}: {} samples whose density differs from the bins' density",
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
}

/// Resolved tile samples seen from one receiver over several presampled frames.
struct Tiles {
    binned: Binned,
    /// Mean and standard error of luminance x cos to the axis / density.
    irradiance: (f64, f64),
}

impl Tiles {
    fn assert_match_density(&self, what: &str) {
        self.binned.assert_match_density(what);
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
    let partials = readback::<f32>(u64::from(bins) * 64, "bin partials");
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
                histogram: histogram.gpu(),
                mismatches: mismatches.gpu(),
                bin_partials: partials.gpu(),
                values: values.gpu(),
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
        lighting
            .record(
                &mut list,
                &ScenePass {
                    frame,
                    ..scene.pass([0.0; 3])
                },
            )
            .unwrap();
        compute_barrier(&mut list);
        list.dispatch(&probe[0], root.gpu(), [SAMPLES / 64, 1, 1]);
        if f == 0 {
            list.dispatch(&probe[1], root.gpu(), [bins, 1, 1]);
        }
        finish(list);
        for &v in values.as_slice().unwrap() {
            let v = f64::from(v);
            sum += v;
            sum2 += v * v;
        }
    }
    let n = f64::from(SAMPLES) * f64::from(frames);
    let mean = sum / n;
    Tiles {
        binned: Binned::new(
            histogram.as_slice().unwrap(),
            partials.as_slice().unwrap(),
            mismatches.as_slice().unwrap()[0],
            n,
        ),
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
    let mut scene = TestScene::procedural(SceneDesc::default());
    scene.write(|_, _| {});
    let mut lighting = scene.lighting();
    let t = tiles(&scene, &mut lighting, &Bins::around_y(PI, 16, 32));
    t.assert_match_density("procedural sky");
    assert_clean();
}

#[test]
fn a_half_degree_hot_spot_is_sampled_at_its_own_density() {
    let linked = kernels(
        HotSpotData::TYPE,
        DEFAULT_MEDIUM,
        &[(&hotspot_bake::MODULE, BAKE)],
    );
    let mut scene = TestScene::linked(SceneDesc::default(), linked, |w| {
        w.set_environment(GpuPtr::<HotSpotData>::NULL);
    });
    scene.write(|_, _| {});
    let mut lighting = scene.lighting();
    let spot = Vec3::from_array(octahedral::to_dir([100.25 / 256.0, 60.25 / 256.0]));
    let bins = Bins {
        axis: spot,
        ..Bins::around_y(2f64.to_radians(), 8, 8)
    };
    let t = tiles(&scene, &mut lighting, &bins);
    t.assert_match_density("hot spot");
    let observed = &t.binned.observed;
    let n: f64 = observed.iter().sum();
    let near = 1.0 - observed[observed.len() - 1] / n;
    assert!(
        near > 0.005,
        "{near} of the samples within 2 deg of the hot spot"
    );
    assert_clean();
}

#[test]
fn sphere_tiles_match_the_density_and_irradiance() {
    let mut scene = TestScene::new(SceneDesc::default());
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
    let mut lighting = scene.lighting();
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
    let mut scene = TestScene::new(SceneDesc::default());
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
    let mut lighting = scene.lighting();
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
                producer: c.producer_entry(ProducerId(self.producer)).unwrap(),
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

/// Opaque producer entries `0..transforms.len()`, each with its transform in one group anchored
/// at `anchor`.
fn set_producers(
    w: &mut SceneWriter<'_>,
    materials: &[MaterialId; 3],
    anchor: DVec3,
    transforms: &[Affine3],
    update: ProducerUpdate,
) {
    set_producers_of(w, materials, anchor, transforms, update, Opacity::Opaque);
}

/// [`set_producers`], the entries of `opacity`.
fn set_producers_of(
    w: &mut SceneWriter<'_>,
    materials: &[MaterialId; 3],
    anchor: DVec3,
    transforms: &[Affine3],
    update: ProducerUpdate,
    opacity: Opacity,
) {
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: transforms.len() as u32,
            writer: GroupWriter::Cpu,
            anchor,
        })
        .unwrap();
    let out = w.transforms(g, 0, transforms.len() as u32).unwrap();
    for (o, &t) in out.iter_mut().zip(transforms) {
        *o = t.into();
    }
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
                opacity,
                ..ProducerDesc::default()
            }),
        )
        .unwrap();
    }
}

#[test]
fn emissive_producer_tiles_match_the_density_and_irradiance() {
    let mut scene = TestScene::new(SceneDesc::default());
    // Entry 0 faces the receiver from y = 2; entry 1 is turned over at y = 2.5 beside it, so
    // only its double-sided quads light the receiver.
    let flip = Mat3::from_rotation_x(PI as f32);
    let transforms = [
        Affine3::from_translation(Vec3::new(0.0, 2.0, 0.0)),
        Affine3::from_mat3_translation(flip, Vec3::new(0.0, 2.5, 0.0)),
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
    let mut lighting = scene.lighting();
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

/// A sample carries what a ray sees at its point, past 1 included, while selection stays by the
/// factor: a texel of 5 lights the receiver five times as much as the factor alone, a presence
/// layer's opacity texel of 0.25 a quarter as much, and one under an alpha cutoff not at all.
#[test]
fn textured_emissive_producers_light_with_their_texels() {
    use orr_pack::material::slot;
    let solid = Opacity::Opaque;
    lit_through_a_texture(slot::EMISSION, 0, solid, 5.0, 5.0, "textured emitters");
    let presence = material_flags::PRESENCE;
    let what = "textured presence emitters";
    lit_through_a_texture(slot::OPACITY, presence, solid, 0.25, 0.25, what);
    let tested = material_flags::ALPHA_TESTED;
    let cut = Opacity::AlphaTested;
    lit_through_a_texture(slot::OPACITY, tested, cut, 0.25, 0.0, "cut away emitters");
    let what = "alpha tested emitters traced whole";
    lit_through_a_texture(slot::OPACITY, tested, solid, 0.25, 1.0, what);
}

/// Producers of `opacity` with a material of `flags`, whose slot `s` reads `texel`, light a
/// receiver `shown` times as much as their factor.
fn lit_through_a_texture(
    s: usize,
    flags: u32,
    opacity: Opacity,
    texel: f32,
    shown: f32,
    what: &str,
) {
    let mut scene = TestScene::new(SceneDesc::default());
    let transforms = [Affine3::from_translation(Vec3::new(0.0, 2.0, 0.0))];
    let quads = Quads {
        producer: 0,
        count: 6,
        size: 0.4,
        origin: Vec3::new(-1.6, 0.0, -0.2),
        step: Vec3::new(0.55, 0.0, 0.05),
    };
    let image = texture(1, 1, &[[texel, texel, texel, 1.0]], "texel");
    let mut textured = emitting([1.0, 0.6, 0.3], 400.0, flags);
    textured.geometry_opacity = 1.0;
    textured.alpha_cutoff = 0.5;
    textured.textures[s] = image.index;
    textured.samplers[s].sampler = linear_clamp();
    let p = pipeline(&emit::MODULE, "main");
    scene.frame(
        |w, m| {
            let id = m.create(&textured).unwrap();
            set_producers_of(
                w,
                &[id; 3],
                DVec3::ZERO,
                &transforms,
                ProducerUpdate::Static,
                opacity,
            );
        },
        |c| quads.dispatch(c, &p),
    );
    let mut lighting = scene.lighting();
    let t = tiles(
        &scene,
        &mut lighting,
        &Bins::around_y(70f64.to_radians(), 8, 16),
    );
    let luminance = y2020([1.0, 0.6, 0.3]) * 400.0 * f64::from(shown);
    let lit = Vec3::new(0.0, 2.0, 0.0);
    let e: f64 = (0..quads.count)
        .map(|c| {
            let v = quads.corners(c).map(|p| (p + lit).as_dvec3());
            polygon_irradiance(&v, DVec3::Y, luminance)
        })
        .sum();
    t.assert_irradiance(e, what);
    assert_clean();
}

/// Next-event candidates of a receiver whose tile reads `local`, binned, over `frames` frames
/// of the scene pass.
fn local_candidates(
    scene: &TestScene,
    lighting: &mut SceneLighting,
    local: &LocalLights,
    bins: &Bins,
    frames: u32,
) -> Binned {
    let Bins {
        receiver,
        axis,
        half_angle,
        rings,
        sectors,
    } = *bins;
    let bins = rings * sectors;
    let mut histogram = readback::<u32>(u64::from(bins + 2), "histogram");
    histogram.as_mut_slice().unwrap().fill(0);
    let mut mismatches = readback::<u32>(1, "mismatches");
    mismatches.as_mut_slice().unwrap().fill(0);
    let partials = readback::<f32>(u64::from(bins) * 64, "bin partials");
    let local = upload(std::slice::from_ref(local), "forced local sampler");
    let omc = 2.0 * (0.5 * half_angle).sin().powi(2);
    let probe = [
        pipeline(&local_probe::MODULE, "candidates"),
        pipeline(&local_probe::MODULE, "integrate"),
    ];
    for f in 0..frames {
        let frame = 1000 + u64::from(f);
        let root = upload(
            &[local_probe::Root {
                set: lighting.light_set(),
                local: local.gpu(),
                histogram: histogram.gpu(),
                mismatches: mismatches.gpu(),
                bin_partials: partials.gpu(),
                _pad: 0,
                receiver: receiver.extend(0.0).to_array(),
                axis: axis.normalize().extend(omc as f32).to_array(),
                rings,
                sectors,
                subdivisions: 96,
                frame: frame as u32,
            }],
            "local probe root",
        );
        let mut list = render().command_list().unwrap();
        lighting
            .record(
                &mut list,
                &ScenePass {
                    frame,
                    ..scene.pass([0.0; 3])
                },
            )
            .unwrap();
        compute_barrier(&mut list);
        list.dispatch(&probe[0], root.gpu(), [SAMPLES / 64, 1, 1]);
        if f == 0 {
            list.dispatch(&probe[1], root.gpu(), [bins, 1, 1]);
        }
        finish(list);
    }
    Binned::new(
        histogram.as_slice().unwrap(),
        partials.as_slice().unwrap(),
        mismatches.as_slice().unwrap()[0],
        f64::from(SAMPLES) * f64::from(frames),
    )
}

/// A local sampler of slots `(light, first, count, probability)`, sorted by light and first
/// position, and `share`; an emissive entry's slot ranges over positions of equal mass among
/// `positions`.
fn forced_local(
    slots: &[(u32, u32, u32, f64)],
    positions: u32,
    lights: u32,
    share: f32,
) -> LocalLights {
    let mut local = LocalLights::zeroed();
    let mut sum = 0.0;
    for (slot, &(id, first, count, p)) in local.lights.iter_mut().zip(slots) {
        sum += p;
        let range = if id >= lights {
            f64::from(count) / f64::from(positions)
        } else {
            1.0
        };
        *slot = LocalLight {
            id,
            first,
            count,
            cdf: (sum * f64::from(u32::MAX)).round() as u32,
            gain: (p / range) as f32,
        };
    }
    local.lights[slots.len() - 1].cdf = u32::MAX;
    local.count = slots.len() as u32;
    local.share = share;
    local.receiver = [0.0, 0.0, 0.0, -1.0];
    local
}

/// Next-event candidates of a tile whose local sampler holds ranges of an emissive producer's
/// list beside a sphere light come at the density BSDF rays read on what they hit, through the
/// slot each falls in: drawn from the sampler alone they follow it, a position no slot holds
/// included, and mixed with the tile samples they keep the density of the mixture.
#[test]
fn local_sampler_candidates_match_the_hit_side_density() {
    let mut scene = TestScene::new(SceneDesc::default());
    let quads = Quads {
        producer: 0,
        count: 8,
        size: 0.3,
        origin: Vec3::new(-1.6, 0.0, -0.2),
        step: Vec3::new(0.4, 0.0, 0.05),
    };
    let p = pipeline(&emit::MODULE, "main");
    scene.frame(
        |w, m| {
            let id = m.create(&emitting([1.0, 0.6, 0.3], 400.0, 0)).unwrap();
            set_producers(
                w,
                &[id; 3],
                DVec3::ZERO,
                &[Affine3::from_translation(Vec3::new(0.0, 2.0, 0.0))],
                ProducerUpdate::Static,
            );
            w.set_light(
                LightId(1),
                Some(&Light::Sphere {
                    position: DVec3::new(0.0, 2.5, 1.5),
                    radius: 0.2,
                    intensity: Vec3::splat(300.0),
                    cone: None,
                }),
            )
            .unwrap();
        },
        |c| quads.dispatch(c, &p),
    );
    let mut lighting = scene.lighting();
    // Producer entry 0's feedback id follows the lights; its position 3 is in no slot.
    let lights = scene.tables.lights;
    let entry = lights;
    let slots = [
        (1, 0, 1, 0.2),
        (entry, 0, 3, 0.4),
        (entry, 4, 2, 0.15),
        (entry, 6, 2, 0.25),
    ];
    let bins = Bins::around_y(70f64.to_radians(), 8, 16);
    let alone = forced_local(&slots, quads.count, lights, 1.0);
    local_candidates(&scene, &mut lighting, &alone, &bins, 4).assert_match_density("local sampler");
    let mixed = forced_local(&slots, quads.count, lights, 0.5);
    let m = local_candidates(&scene, &mut lighting, &mixed, &bins, 1);
    assert_eq!(m.empty, 0, "mixed: empty candidates");
    assert!(
        f64::from(m.mismatches) <= 1e-4 * f64::from(SAMPLES),
        "mixed: {} candidates off the hit side's density",
        m.mismatches
    );
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
    lighting.record(&mut list, &scene.pass([0.0; 3])).unwrap();
    compute_barrier(&mut list);
    list.dispatch(&probe[0], root, [threads / 64, 1, 1]);
    list.dispatch(&probe[1], root, [1, 1, 1]);
    finish(list);
}

#[test]
fn directional_share_matches_the_lights_weights() {
    let mut scene = TestScene::procedural(SceneDesc::default());
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
    let mut lighting = scene.lighting();
    let (rings, sectors, frames, threads) = (4u32, 8u32, 64u32, 1u32 << 16);
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
            histogram: histogram.gpu(),
            checks: checks.gpu(),
            totals: totals.gpu(),
            normals: normals_gpu.gpu(),
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
    run_nee_probe(&scene, &mut lighting, root.gpu(), threads);

    let table = scene.lights();
    let states = read(
        lighting.directional_states(),
        u64::from(scene.tables.lights),
    );
    let weight = |id: u32, n: Vec3| {
        let w = f64::from(states[id as usize].weight);
        let l = Vec3::from_array(table[id as usize].direction);
        if n == Vec3::ZERO {
            w
        } else {
            w * f64::from(n.dot(l).max(0.0))
        }
    };
    // The weight is the focus illuminance's luminance: the light's own, or the textured disc's
    // integral where it has none, through the medium.
    for (k, &id) in ids.iter().enumerate() {
        let s = &states[id as usize];
        let color = table[id as usize].color;
        let e = if color == [0.0; 3] { s.integral } else { color };
        let expected = y2020([0, 1, 2].map(|c| e[c] * s.transmittance[c]));
        assert!(
            (f64::from(s.weight) / expected - 1.0).abs() < 1e-5,
            "light {k}: weight {} against {expected}",
            s.weight
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
    let mut scene = TestScene::procedural(SceneDesc::default());
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
    let mut lighting = scene.lighting();
    scene_pass(&scene, &mut lighting, scene.frame);
    let stats = lighting.stats(scene.frame).unwrap();
    assert_eq!(
        (
            stats.directional_lights,
            stats.local_lights,
            stats.emissive_entries,
            stats.emissive_clusters
        ),
        (2, 3, 1, 5)
    );

    let set: LightSetGpu = read(lighting.light_set(), 1)[0];
    let light_p = read(set.light_probability, u64::from(scene.tables.lights));
    let entries = u64::from(scene.tables.entries);
    let emissive_p: Vec<EmissiveSelection> = read(set.emissive, entries);
    let power: Vec<EmissiveGpu> = read(scene.root_value().emissive, entries);
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
    for (id, &p) in light_p.iter().enumerate() {
        let w = weights.get(id).copied().unwrap_or(0.0);
        close(p, w, &format!("light {id}"));
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
    assert!(
        (scene_power / expected - 1.0).abs() < 1e-3,
        "{what}: scene power {scene_power} against {expected}"
    );
    assert_eq!(selection.count, quads.count, "{what}: clusters in the CDF");
    // Each cluster's fixed-point step follows its share of the power, a cluster that emits
    // nothing keeps a step of 0, and the last value is the top.
    let cdf: Vec<u32> = read(selection.cdf, u64::from(quads.count));
    assert_eq!(*cdf.last().unwrap(), u32::MAX, "{what}: last CDF value");
    let mut below = 0u32;
    for (c, &v) in (0..).zip(&cdf) {
        let share = f64::from(quads.size).powi(2) * scale * luminance[(c % 3) as usize] / expected;
        let step = f64::from(v - below) / f64::from(u32::MAX);
        assert!(
            (step - share).abs() < 1e-4,
            "{what}: cluster {c} step {step} against {share}"
        );
        if share == 0.0 {
            assert_eq!(v, below, "{what}: cluster {c} emits nothing");
        }
        below = v;
    }
}

#[test]
fn producer_power_equals_area_times_luminance_after_it_moves() {
    let mut scene = TestScene::new(SceneDesc::default());
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
                &[Affine3::from_mat3_translation(
                    linear,
                    Vec3::new(4.0, 0.0, -2.0),
                )],
                ProducerUpdate::EveryFrame,
            );
        },
        |c| first.dispatch(c, &p),
    );
    let mut lighting = scene.lighting();
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
    assert_clean();
}

/// The producer entries' cluster CDFs are runs of one list that holds every cluster the scene's
/// producers can emit, which follows the producers set from pass to pass. A producer that
/// emits every cluster its batch holds fills the list; once two more are set the next pass
/// makes a list for all three, each entry's CDF its own run of it.
#[test]
fn producer_cluster_cdfs_follow_the_producers_set() {
    let mut scene = TestScene::new(SceneDesc::default());
    let quads = |producer: u32| Quads {
        producer,
        count: 32,
        size: 0.25,
        origin: Vec3::new(0.0, 0.0, producer as f32),
        step: Vec3::new(0.5, 0.0, 0.0),
    };
    let p = pipeline(&emit::MODULE, "main");
    let mut materials = None;
    scene.frame(
        |w, m| {
            let made = producer_materials(m);
            let one = [Affine3::IDENTITY];
            set_producers(w, &made, DVec3::ZERO, &one, ProducerUpdate::EveryFrame);
            materials = Some(made);
        },
        |c| quads(0).dispatch(c, &p),
    );
    assert_eq!(scene.tables.producer_clusters, 32);
    let mut lighting = scene.lighting();
    scene_pass(&scene, &mut lighting, scene.frame);
    let stats = lighting.stats(scene.frame).unwrap();
    assert_eq!((stats.emissive_entries, stats.emissive_clusters), (1, 32));

    let materials = materials.unwrap();
    scene.frame(
        |w, _| {
            let three = [Affine3::IDENTITY; 3];
            set_producers(
                w,
                &materials,
                DVec3::ZERO,
                &three,
                ProducerUpdate::EveryFrame,
            );
        },
        |c| {
            for producer in 0..3 {
                quads(producer).dispatch(c, &p);
            }
        },
    );
    // The clusters the three batches hold, as a table grows to hold them.
    assert_eq!(scene.tables.producer_clusters, 128);
    scene_pass(&scene, &mut lighting, scene.frame);
    let stats = lighting.stats(scene.frame).unwrap();
    assert_eq!((stats.emissive_entries, stats.emissive_clusters), (3, 96));
    let set: LightSetGpu = read(lighting.light_set(), 1)[0];
    let entries = u64::from(scene.tables.entries);
    let selections: Vec<EmissiveSelection> = read(set.emissive, entries);
    let mut runs: Vec<u64> = Vec::new();
    for s in selections.iter().filter(|s| s.count > 0) {
        assert_eq!(s.count, 32);
        let cdf: Vec<u32> = read(s.cdf, 32);
        assert_eq!(*cdf.last().unwrap(), u32::MAX, "a run's last CDF value");
        assert!(cdf.is_sorted(), "a run's CDF falls");
        runs.push(s.cdf.address().0);
    }
    runs.sort_unstable();
    assert_eq!(runs.len(), 3);
    assert!(
        runs.windows(2).all(|w| w[1] - w[0] >= 32 * 4),
        "two entries' runs share clusters of the list: {runs:?}"
    );
    assert_clean();
}

#[test]
fn a_scene_without_lights_gives_empty_candidates_and_no_nan() {
    let mut scene = TestScene::new(SceneDesc::default());
    scene.write(|_, _| {});
    let mut lighting = scene.lighting();
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
            histogram: histogram.gpu(),
            checks: checks.gpu(),
            totals: totals.gpu(),
            normals: normals.gpu(),
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
    run_nee_probe(&scene, &mut lighting, root.gpu(), threads);

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

/// Copies `counts` into the scene's light feedback counts, as the views of a frame would add
/// them.
fn feed_back(set: &LightSetGpu, counts: &[u32]) {
    let staging = upload(counts, "feedback counts");
    let mut list = render().command_list().unwrap();
    list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
    list.copy(
        set.feedback_counts.cast(),
        staging.gpu().cast(),
        size_of_val(counts) as u64,
    );
    finish(list);
}

#[test]
fn light_feedback_blends_into_the_selection_and_clears_on_a_reset() {
    let mut scene = TestScene::procedural(SceneDesc::default());
    scene.write(|w, _| {
        // Light 3 is set and cleared, so the table holds an empty entry for it.
        for (id, i) in [(0u32, 100.0f32), (1, 900.0), (3, 1.0)] {
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
        w.set_light(LightId(3), None).unwrap();
    });
    assert!(scene.tables.lights >= 4);
    let mut lighting = scene.lighting();
    let share = f64::from(SceneLightingSettings::default().feedback_share);
    let mut frame = scene.frame;
    scene_pass(&scene, &mut lighting, frame);
    let set: LightSetGpu = read(lighting.light_set(), 1)[0];
    let probabilities = |set: &LightSetGpu| {
        let p = read(set.light_probability, 4);
        [p[0], p[1], set.environment_probability].map(f64::from)
    };
    let power = probabilities(&set);
    assert_eq!(power[2], 0.25);

    // Light 3 is empty: what fed back for it takes no share.
    let n = set.candidate_count as usize;
    let mut counts = vec![0u32; n + 1];
    counts[0] = 3000;
    counts[3] = 500;
    counts[n] = 1000;
    feed_back(&set, &counts);
    frame += 1;
    scene_pass(&scene, &mut lighting, frame);
    assert_eq!(lighting.stats(frame).unwrap().feedback_pixels, 4500);
    let fed = [3000.0, 0.0, 1000.0].map(|c| c / 4000.0);
    let expected: [f64; 3] = std::array::from_fn(|i| (1.0 - share) * power[i] + share * fed[i]);
    let check = |got: [f64; 3], what: &str| {
        for i in 0..3 {
            assert!(
                (got[i] - expected[i]).abs() <= 1e-5,
                "{what}: probabilities {got:?} against {expected:?}"
            );
        }
        let sum: f64 = got.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "{what}: probabilities sum to {sum}"
        );
    };
    check(probabilities(&set), "after feedback");
    assert_eq!(read(set.light_probability, 4)[3], 0.0);

    // Without new feedback the weights decay together, so the blend holds.
    frame += 1;
    scene_pass(&scene, &mut lighting, frame);
    check(probabilities(&set), "a frame later");

    lighting.reset_history();
    frame += 1;
    scene_pass(&scene, &mut lighting, frame);
    let reset = probabilities(&set);
    assert!(
        (0..3).all(|i| (reset[i] - power[i]).abs() <= 1e-6),
        "after a reset: {reset:?} against {power:?}"
    );
    assert_clean();
}

/// What one frame's scene pass left: its tile samples and its light set record.
struct Passed {
    frame: u64,
    tiles: Allocation<LightSample>,
    set: Allocation<LightSetGpu>,
}

/// A scene with sphere light 0, an emitting producer entry and the procedural sky takes
/// feedback for the three, then lights 1, 9 and 40 in three frames, each past its light table
/// and each with a producer more, which grow the producer table and the emissive list, and two
/// frames more. Returns each of the five frames' scene pass, the scene's light tables after
/// them, and the probabilities of light 0, the entry and the environment after them and once a
/// reset has dropped the feedback. With `in_flight` the three frames that grow the tables run
/// one behind another, none waiting for the one before.
fn grow_tables(in_flight: bool) -> (Vec<Passed>, LightTables, [[f64; 3]; 2]) {
    let mut scene = TestScene::procedural(SceneDesc::default());
    let quads = Quads {
        producer: 0,
        count: 4,
        size: 0.5,
        origin: Vec3::new(-1.0, 0.0, 0.0),
        step: Vec3::new(0.6, 0.0, 0.0),
    };
    let sphere = |id: u32| Light::Sphere {
        position: DVec3::new(f64::from(id), 3.0, 0.0),
        radius: 0.2,
        intensity: Vec3::splat(100.0 + id as f32),
        cone: None,
    };
    let p = pipeline(&emit::MODULE, "main");
    scene.frame(
        |w, m| {
            let id = m.create(&emitting([1.0, 0.6, 0.3], 400.0, 0)).unwrap();
            let above = Affine3::from_translation(Vec3::new(0.0, 2.0, 0.0));
            set_producers(w, &[id; 3], DVec3::ZERO, &[above], ProducerUpdate::Static);
            w.set_light(LightId(0), Some(&sphere(0))).unwrap();
        },
        |c| quads.dispatch(c, &p),
    );
    let mut lighting = scene.lighting();
    scene_pass(&scene, &mut lighting, scene.frame);
    let set: LightSetGpu = read(lighting.light_set(), 1)[0];
    let probabilities = |lighting: &SceneLighting, tables: LightTables| {
        let set: LightSetGpu = read(lighting.light_set(), 1)[0];
        assert_eq!(set.candidate_count, tables.lights + tables.entries);
        [
            read(set.light_probability, 1)[0],
            read(set.emissive, 1)[0].probability,
            set.environment_probability,
        ]
        .map(f64::from)
    };
    // Light 0, entry 0 after the table's one light, and the environment after every candidate.
    let mut counts = vec![0u32; set.candidate_count as usize + 1];
    counts[0] = 3000;
    counts[scene.tables.lights as usize] = 2000;
    counts[set.candidate_count as usize] = 1000;
    feed_back(&set, &counts);

    let list = [scene.id];
    let mut passed = Vec::new();
    let mut tables = LightTables::default();
    for (k, id) in [Some(1u32), Some(9), Some(40), None, None]
        .into_iter()
        .enumerate()
    {
        let mut f = scene
            .scenes
            .begin_frame(&FrameDesc {
                time: FrameTime {
                    absolute: 0.0,
                    delta: 1.0 / 60.0,
                },
                scenes: &list,
                low_latency: None,
            })
            .unwrap();
        let frame = f.id();
        if let Some(id) = id {
            let mut w = f.scene(scene.id).unwrap();
            w.set_light(LightId(id), Some(&sphere(id))).unwrap();
            // A producer that emits nothing takes an entry of each table all the same.
            let idle = ProducerDesc {
                flags: InstanceFlags::RASTER,
                ..ProducerDesc::default()
            };
            w.set_producer(ProducerId(100 + id), Some(&idle)).unwrap();
        }
        let mut pass = None;
        f.update_scenes(&mut |c| {
            pass = Some(ScenePass {
                frame,
                scene: c.root,
                focus: [0.0; 3],
                tables: c.light_tables,
                features: c.features,
            });
        })
        .unwrap();
        let pass = pass.unwrap();
        tables = pass.tables;
        let list = f.list().unwrap();
        lighting.record(list, &pass).unwrap();
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        let tiles = copy_back(list, set.tiles, u64::from(SAMPLES), "tiles");
        let record = copy_back(list, lighting.light_set(), 1, "light set");
        list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
        let point = f.end(&[]).unwrap();
        passed.push(Passed {
            frame,
            tiles,
            set: record,
        });
        // The frames that grow the table are in flight together; the third is waited for.
        if !in_flight || k >= 2 {
            render()
                .wait_while_completing(point, Duration::from_secs(10))
                .unwrap();
        }
        // The first frame takes the feedback counted before the table grew, and no frame
        // after it finds any.
        if k >= 2 {
            let pixels = |k: usize| lighting.stats(passed[k].frame).unwrap().feedback_pixels;
            let counted: Vec<u32> = (k - 2..=k).map(pixels).collect();
            assert_eq!(counted, [if k == 2 { 6000 } else { 0 }, 0, 0]);
        }
    }
    let fed = probabilities(&lighting, tables);
    lighting.reset_history();
    let mut list = render().command_list().unwrap();
    let pass = ScenePass {
        frame: passed[4].frame + 1,
        scene: scene.root,
        focus: [0.0; 3],
        tables,
        features: scene.features,
    };
    lighting.record(&mut list, &pass).unwrap();
    finish(list);
    let power = probabilities(&lighting, tables);
    (passed, tables, [fed, power])
}

/// Lights set past the light table and producers past the producer table and the emissive list
/// in three frames running, none waiting for the one before, leave every frame's light samples
/// and selection as frames that each waited do, and the feedback counted before the tables
/// grew stays with its light, its emissive entry and the environment: an entry's feedback id
/// moves with the light table and the environment's with that and the emissive list, and each
/// pass takes the feedback so far by the ids of the tables before.
#[test]
fn feedback_follows_its_lights_as_the_light_tables_grow_under_frames_in_flight() {
    let (flight, tables, [fed, power]) = grow_tables(true);
    assert!(
        tables.lights > 40 && tables.entries >= 4 && tables.producers >= 4,
        "the light tables hold {tables:?}"
    );
    let share = f64::from(SceneLightingSettings::default().feedback_share);
    let counted = [3000.0, 2000.0, 1000.0].map(|c| c / 6000.0);
    for i in 0..3 {
        let expected = (1.0 - share) * power[i] + share * counted[i];
        assert!(
            (fed[i] - expected).abs() <= 1e-5,
            "probabilities {fed:?} against the power's {power:?} with the feedback {counted:?}"
        );
    }
    let (settled, _, probabilities) = grow_tables(false);
    assert_eq!(probabilities, [fed, power]);
    let selection = |p: &Passed| {
        let s = p.set.as_slice().unwrap()[0];
        (
            s.candidate_count,
            s.candidate_total,
            s.environment_probability,
            s.local_lights,
        )
    };
    for (k, (a, b)) in flight.iter().zip(&settled).enumerate() {
        assert_eq!(a.frame, b.frame);
        assert_eq!(selection(a), selection(b), "frame {k}: the selection");
        let (a, b) = (a.tiles.as_slice().unwrap(), b.tiles.as_slice().unwrap());
        let differing = a
            .iter()
            .zip(b)
            .filter(|(a, b)| bytemuck::bytes_of(*a) != bytemuck::bytes_of(*b))
            .count();
        assert_eq!(differing, 0, "frame {k}: tile samples that differ");
    }
    assert_clean();
}

/// A 2 m square grid of `n` x `n` cells facing -y around the origin, its two halves along z
/// in two sections.
fn emitter_grid(n: u32) -> orr_cook::MeshInput {
    let mut m = orr_cook::MeshInput::default();
    for j in 0..=n {
        for i in 0..=n {
            let (x, z) = (
                i as f32 / n as f32 * 2.0 - 1.0,
                j as f32 / n as f32 * 2.0 - 1.0,
            );
            m.positions.push(Vec3::new(x, 0.0, z));
            m.normals.push(-Vec3::Y);
        }
    }
    m.uvs = Some(vec![orr_math::Vec2::ZERO; m.positions.len()]);
    let w = n + 1;
    for half in 0..2 {
        let first = m.indices.len() as u32;
        for j in n * half / 2..n * (half + 1) / 2 {
            for i in 0..n {
                let c = j * w + i;
                m.indices.extend([c, c + 1, c + w, c + 1, c + w + 1, c + w]);
            }
        }
        m.sections.push(orr_pack::Span {
            first,
            count: m.indices.len() as u32 - first,
        });
    }
    m
}

/// A cooked emitter of two sections of unequal emission, stretched unevenly and tilted over the
/// receiver: its tiles' samples follow the density that hits on it read, since the probe takes
/// each bin's density from what a ray toward it meets.
#[test]
fn cooked_emitter_tiles_match_the_hit_side_density() {
    let d = lighting_suite::device();
    if d.caps().cluster.max_tris == 0 || d.caps().cluster.max_partitions == 0 {
        return;
    }
    let emits = |color: [f32; 3], luminance: f32| orr_pack::material::Material {
        emission_color: color,
        emission_luminance: luminance,
        ..orr_pack::material::Material::default()
    };
    let manifest = orr_cook::Manifest {
        archives: vec![orr_cook::ArchiveSpec::new(
            orr_pack::Target::Memory,
            vec![orr_cook::AssetSpec {
                name: "lamp".into(),
                source: orr_cook::Source::Mesh {
                    mesh: emitter_grid(24),
                    materials: vec![emits([1.0, 0.8, 0.6], 200.0), emits([0.4, 0.6, 1.0], 60.0)],
                    settings: orr_cook::MeshSettings::default(),
                },
            }],
        )],
    };
    let bytes = orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0);
    let mut scene = TestScene::new(SceneDesc::default());
    let m = scene.mount(bytes);
    let model = scene
        .scenes
        .model(m, orr_pack::AssetId::from_name("lamp"))
        .unwrap();
    let linear = Mat3::from_rotation_x(0.3) * Mat3::from_diagonal(Vec3::new(1.5, 1.0, 0.8));
    scene.write(|w, _| {
        w.set_focus(DVec3::ZERO);
        let g = w
            .create_transform_group(&TransformGroupDesc {
                count: 1,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        w.transforms(g, 0, 1).unwrap()[0] =
            Affine3::from_mat3_translation(linear, Vec3::new(0.2, 1.0, -0.1)).into();
        w.add_instances(&orr_scene::InstancesDesc {
            model,
            copies: 1,
            transforms: g,
            first: 0,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    });
    for _ in 0..4 {
        scene.write(|_, _| {});
    }
    let mut lighting = scene.lighting();
    let t = tiles(&scene, &mut lighting, &Bins::around_y(1.3, 6, 12));
    t.assert_match_density("cooked emitter");
    assert_clean();
}
