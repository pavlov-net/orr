//! The baked environment, its sampling and statistics, directional discs, and media.

use std::f64::consts::PI;

use glam::{DVec3, Mat3, Vec3};
use lighting_suite::shaders::{
    disc_probe, env_probe, fog_probe, glint, octahedral as mapping, procedural_bake,
    slab_directional,
};
use lighting_suite::{
    TestTexture, assert_clean, copy_back, device, kernels, linear_clamp, pipeline, readback,
    render, run, texture, upload,
};
use orr_gpu::{GpuAddress, GpuPtr, Hazards, KernelSet, Stages, TextureIndex};
use orr_lighting::{
    DirectionalLight, DirectionalLightGpu, DirectionalState, DiscProfile, EnvironmentMapGpu,
    SceneLighting, ScenePass,
};
use orr_pack::texture::octahedral;
use shader_maths::stats::{Rng, chi_square_p};

const BAKE: &str = "environment/bake";
const DIRECTIONAL: &str = "light_sampling/directional";

/// The slab's extinction and height (`media_types.slang`).
const SLAB_SIGMA_T: [f64; 3] = [2e-4, 4e-4, 8e-4];
const SLAB_HEIGHT: f64 = 1000.0;

/// `REC2020_TO_XYZ` of `orr.color`.
const REC2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.636958048, 0.144616904, 0.168880975],
    [0.262700212, 0.677998072, 0.0593017165],
    [0.0, 0.028072693, 1.06098506],
];

const SUN_RADIUS: f32 = 4.65e-3;

fn scene(kernels: &KernelSet, capacity: u32) -> SceneLighting {
    SceneLighting::new(device(), kernels, &Default::default(), capacity).unwrap()
}

fn pass(
    focus: [f32; 3],
    environment: GpuAddress,
    lights: GpuPtr<DirectionalLightGpu>,
    count: u32,
) -> ScenePass {
    ScenePass {
        frame: 0,
        focus,
        environment,
        medium: GpuAddress(0),
        lights,
        light_count: count,
    }
}

fn after_scene_pass(list: &mut orr_gpu::CommandList<'_>) {
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
}

fn finish(mut list: orr_gpu::CommandList<'_>) {
    list.barrier(
        Stages::COMPUTE | Stages::TRANSFER,
        Stages::HOST,
        Hazards::empty(),
    );
    run(list);
}

fn v3(a: [f32; 3]) -> DVec3 {
    Vec3::from_array(a).as_dvec3()
}

/// Transmittance through the slab from altitude `y0` toward a directional light.
fn slab_transmittance(y0: f64, dir: DVec3) -> DVec3 {
    let exit = if dir.y > 0.0 { SLAB_HEIGHT } else { 0.0 };
    let a = ((exit - y0.clamp(0.0, SLAB_HEIGHT)) / dir.y).abs();
    DVec3::from_array(SLAB_SIGMA_T.map(|s| (-s * a).exp()))
}

fn rel(a: DVec3, b: DVec3) -> f64 {
    ((a - b).abs() / b.abs().max(DVec3::splat(1e-30))).max_element()
}

fn uniform_sun(direction: Vec3, illuminance: Vec3) -> DirectionalLight {
    DirectionalLight {
        direction_to_light: direction.normalize(),
        illuminance,
        angular_radius: SUN_RADIUS,
        disc: DiscProfile::Uniform,
    }
}

fn textured(
    direction: Vec3,
    illuminance: Vec3,
    radius: f32,
    albedo: TextureIndex,
) -> DirectionalLight {
    DirectionalLight {
        direction_to_light: direction.normalize(),
        illuminance,
        angular_radius: radius,
        disc: DiscProfile::Textured {
            albedo,
            up: Vec3::Y,
            lit_by: [Some(0), None, None, None],
        },
    }
}

/// A 16 x 16 albedo varying smoothly from 0.2 to 0.8.
fn gradient_albedo() -> TestTexture {
    let texels: Vec<[f32; 4]> = (0..256)
        .map(|i| {
            let (x, y) = ((i % 16) as f32 / 15.0, (i / 16) as f32 / 15.0);
            [0.2 + 0.6 * x, 0.2 + 0.6 * y, 0.5, 1.0]
        })
        .collect();
    texture(16, 16, &texels, "albedo gradient")
}

#[test]
fn equal_area_mapping_matches_the_cook() {
    let n = 4096u32;
    let points: Vec<[f32; 2]> = (0..n)
        .map(|i| {
            [
                ((i % 64) as f32 + 0.37) / 64.0,
                ((i / 64) as f32 + 0.71) / 64.0,
            ]
        })
        .collect();
    let mut rng = Rng::new(7);
    let dirs: Vec<[f32; 4]> = (0..n)
        .map(|_| {
            let [x, y, z] = rng.unit();
            [x, y, z, 0.0]
        })
        .collect();
    let points_in = upload(&points, "points");
    let dirs_in = upload(&dirs, "dirs");
    let dirs_out = readback::<[f32; 4]>(u64::from(n), "dirs out");
    let points_out = readback::<[f32; 2]>(u64::from(n), "points out");
    let root = upload(
        &[mapping::Root {
            points: points_in.gpu,
            dirs_out: dirs_out.gpu,
            dirs_in: dirs_in.gpu,
            points_out: points_out.gpu,
            count: n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }],
        "octahedral root",
    );
    let mut list = render().command_list().unwrap();
    list.dispatch(
        &pipeline(&mapping::MODULE, "main"),
        root.gpu,
        [n / 64, 1, 1],
    );
    finish(list);

    let mut worst = 0.0f32;
    for i in 0..n as usize {
        let d = octahedral::to_dir(points[i]);
        let g = dirs_out.as_slice().unwrap()[i];
        worst = worst.max(
            (Vec3::from_array(d) - Vec3::new(g[0], g[1], g[2]))
                .abs()
                .max_element(),
        );
        let p = octahedral::from_dir([dirs[i][0], dirs[i][1], dirs[i][2]]);
        let q = points_out.as_slice().unwrap()[i];
        worst = worst.max((p[0] - q[0]).abs().max((p[1] - q[1]).abs()));
    }
    assert!(
        worst < 2e-6,
        "largest difference from the cook's mapping {worst}"
    );
    assert_clean();
}

/// Samples the scene's baked environment 2^20 times into equal-area latitude-longitude bins
/// and compares the counts with the bins' probabilities integrated from its density.
fn sampling_matches_density(lighting: &mut SceneLighting, environment: GpuAddress) {
    let bins = 32 * 16u64;
    let threads = 1u32 << 14;
    let per_thread = 64u32;
    let mut histogram = readback::<u32>(bins, "histogram");
    histogram.as_mut_slice().unwrap().fill(0);
    let mut mismatches = readback::<u32>(1, "mismatches");
    mismatches.as_mut_slice().unwrap().fill(0);
    let probability = readback::<f32>(bins, "bin probability");
    let root = upload(
        &[env_probe::Root {
            environment: lighting.environment(),
            histogram: histogram.gpu,
            mismatches: mismatches.gpu,
            bin_probability: probability.gpu,
            samples_per_thread: per_thread,
            seed: 3,
            subdivisions: 96,
            _pad: 0,
        }],
        "env probe root",
    );
    let mut list = render().command_list().unwrap();
    lighting.record(&mut list, &pass([0.0; 3], environment, GpuPtr::NULL, 0));
    after_scene_pass(&mut list);
    list.dispatch(
        &pipeline(&env_probe::MODULE, "sample"),
        root.gpu,
        [threads / 64, 1, 1],
    );
    list.dispatch(
        &pipeline(&env_probe::MODULE, "integrate"),
        root.gpu,
        [bins as u32, 1, 1],
    );
    finish(list);

    let samples = f64::from(threads * per_thread);
    let p: Vec<f64> = probability
        .as_slice()
        .unwrap()
        .iter()
        .map(|&p| f64::from(p))
        .collect();
    let total: f64 = p.iter().sum();
    assert!((total - 1.0).abs() < 2e-3, "density integrates to {total}");
    let expected: Vec<f64> = p.iter().map(|p| p * samples).collect();
    let observed: Vec<f64> = histogram
        .as_slice()
        .unwrap()
        .iter()
        .map(|&c| f64::from(c))
        .collect();
    for (o, e) in observed.iter().zip(&expected) {
        assert!(*e > 0.0 || *o == 0.0, "samples in a bin of zero density");
    }
    let pvalue = chi_square_p(&observed, &expected);
    assert!(pvalue > 0.01, "chi-square p = {pvalue}");
    let wrong = mismatches.as_slice().unwrap()[0];
    assert!(
        f64::from(wrong) < samples * 1e-4,
        "{wrong} samples whose density differs from environment_pdf"
    );
}

#[test]
fn hdr_map_sampling_matches_its_density() {
    // An equal-area HDR map in Rec.709: a dim floor, a broad warm lobe, a sky gradient and a
    // 2 deg hot spot.
    let n = 512u32;
    let lobe = DVec3::new(0.3, 0.4, 0.8).normalize();
    let spot = DVec3::new(-0.6, 0.3, 0.2).normalize();
    let texels: Vec<[f32; 4]> = (0..n * n)
        .map(|i| {
            let st = [
                ((i % n) as f32 + 0.5) / n as f32,
                ((i / n) as f32 + 0.5) / n as f32,
            ];
            let d = v3(octahedral::to_dir(st));
            let mut c = DVec3::splat(40.0)
                + DVec3::new(1.0, 0.8, 0.6) * 3000.0 * (-30.0 * (1.0 - d.dot(lobe))).exp()
                + DVec3::new(0.5, 0.7, 1.0) * 500.0 * d.y.max(0.0);
            if d.dot(spot) > 2.0f64.to_radians().cos() {
                c += DVec3::splat(20000.0);
            }
            [c.x as f32, c.y as f32, c.z as f32, 1.0]
        })
        .collect();
    let map = texture(n, n, &texels, "hdr map");
    let r = Mat3::from_rotation_y(0.5);
    let data = upload(
        &[EnvironmentMapGpu {
            map: map.index,
            sampler: linear_clamp(),
            intensity: 2.0,
            primaries: 0,
            from_world: [
                [r.x_axis.x, r.y_axis.x, r.z_axis.x, 0.0],
                [r.x_axis.y, r.y_axis.y, r.z_axis.y, 0.0],
                [r.x_axis.z, r.y_axis.z, r.z_axis.z, 0.0],
            ],
        }],
        "hdr map data",
    );
    let mut lighting = scene(&KernelSet::default(), 1);
    sampling_matches_density(&mut lighting, data.gpu.address());
    assert_clean();
}

#[test]
fn procedural_sky_sampling_matches_its_density() {
    let set = kernels(&[(&procedural_bake::MODULE, BAKE)]);
    let mut lighting = scene(&set, 1);
    sampling_matches_density(&mut lighting, GpuAddress(0));
    assert_clean();
}

#[test]
fn sky_stats_match_a_uniform_environment_and_a_slab() {
    let c = [1024.0f32, 512.0, 256.0];
    let map = texture(2, 2, &[[c[0], c[1], c[2], 1.0]; 4], "uniform map");
    let data = upload(
        &[EnvironmentMapGpu {
            map: map.index,
            sampler: linear_clamp(),
            intensity: 1.0,
            primaries: 2,
            from_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }],
        "uniform map data",
    );
    let lights = [
        uniform_sun(Vec3::new(0.3, 0.8, 0.2), Vec3::new(1.0e5, 0.9e5, 0.8e5)),
        DirectionalLight {
            disc: DiscProfile::LimbDarkened,
            ..uniform_sun(Vec3::new(-0.5, 0.6, 0.1), Vec3::new(5.0e4, 5.0e4, 4.0e4))
        },
        textured(
            Vec3::new(0.1, 0.9, -0.3),
            Vec3::splat(0.3),
            0.01,
            TextureIndex::NONE,
        ),
    ]
    .map(|l| l.gpu());
    let table = upload(&lights, "lights");
    let set = kernels(&[(&slab_directional::MODULE, DIRECTIONAL)]);
    let mut lighting = scene(&set, 4);
    let focus = [10.0, 100.0, -5.0];
    let mut list = render().command_list().unwrap();
    lighting.record(&mut list, &pass(focus, data.gpu.address(), table.gpu, 3));
    finish(list);
    let stats = lighting.stats(0).expect("frame 0 statistics").sky;
    assert!(lighting.stats(1).is_none());

    let c = v3(c);
    let xyz = DVec3::from_array(REC2020_TO_XYZ.map(|row| DVec3::from_array(row).dot(c)));
    let xy = [xyz.x / xyz.element_sum(), xyz.y / xyz.element_sum()];
    assert!(
        (f64::from(stats.dome_luminance) / xyz.y - 1.0).abs() < 1e-5,
        "dome luminance {} against {}",
        stats.dome_luminance,
        xyz.y
    );
    for k in 0..2 {
        assert!(
            (f64::from(stats.dome_xy[k]) - xy[k]).abs() < 1e-5,
            "dome xy {:?} against {xy:?}",
            stats.dome_xy
        );
    }
    let expected: DVec3 = lights
        .iter()
        .map(|l| {
            v3(l.illuminance) * slab_transmittance(f64::from(focus[1]), v3(l.direction_to_light))
        })
        .sum();
    let got = DVec3::new(
        f64::from(stats.directional[0]),
        f64::from(stats.directional[1]),
        f64::from(stats.directional[2]),
    );
    assert!(
        rel(got, expected) < 1e-4,
        "directional {got} against {expected}"
    );
    assert_clean();
}

/// Lights of `table` with their states from a default scene pass at `focus`, and per light
/// the cosine-weighted integral of `directional_disc` through the slab over its cone, and
/// `textured_disc_integral` from a scene compute kernel.
struct Discs {
    states: Vec<DirectionalState>,
    integrals: Vec<DVec3>,
    textured: Vec<[f32; 4]>,
}

fn discs(table: &[DirectionalLightGpu], focus: [f32; 3]) -> Discs {
    let count = table.len() as u32;
    let lights = upload(table, "lights");
    let mut lighting = scene(&KernelSet::default(), count);
    let integrals = readback::<[f32; 4]>(u64::from(count), "disc integrals");
    let textured = readback::<[f32; 4]>(u64::from(count), "textured integrals");
    let root = upload(
        &[disc_probe::Root {
            lights: lights.gpu,
            states: lighting.directional_states(),
            integrals: integrals.gpu,
            textured: textured.gpu,
            eye: [focus[0], focus[1], focus[2], 0.0],
            count,
            resolution: 1024,
            _pad0: 0,
            _pad1: 0,
        }],
        "disc probe root",
    );
    let mut list = render().command_list().unwrap();
    lighting.record(&mut list, &pass(focus, GpuAddress(0), lights.gpu, count));
    after_scene_pass(&mut list);
    list.dispatch(
        &pipeline(&disc_probe::MODULE, "integrate"),
        root.gpu,
        [count, 1, 1],
    );
    list.dispatch(
        &pipeline(&disc_probe::MODULE, "textured"),
        root.gpu,
        [1, 1, 1],
    );
    let states = copy_back(
        &mut list,
        lighting.directional_states(),
        u64::from(count),
        "states",
    );
    finish(list);
    Discs {
        states: states.as_slice().unwrap().to_vec(),
        integrals: integrals
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| DVec3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2])))
            .collect(),
        textured: textured.as_slice().unwrap().to_vec(),
    }
}

#[test]
fn disc_integrals_equal_illuminance_times_transmittance() {
    let albedo = gradient_albedo();
    let sun = Vec3::new(0.3, 0.8, 0.2).normalize();
    // The textured disc is lit at a 45 deg phase.
    let side = sun.cross(Vec3::Z).normalize();
    let moon = -(Mat3::from_axis_angle(side, 45f32.to_radians()) * sun);
    let table = [
        uniform_sun(sun, Vec3::new(1.0e5, 0.9e5, 0.8e5)),
        DirectionalLight {
            disc: DiscProfile::LimbDarkened,
            ..uniform_sun(Vec3::new(-0.5, 0.6, 0.1), Vec3::new(5.0e4, 5.0e4, 4.0e4))
        },
        textured(moon, Vec3::new(0.3, 0.25, 0.2), 0.02, albedo.index),
    ]
    .map(|l| l.gpu());
    let focus = [0.0, 100.0, 0.0];
    let d = discs(&table, focus);
    for (i, l) in table.iter().enumerate() {
        let expected = v3(l.illuminance) * slab_transmittance(100.0, v3(l.direction_to_light));
        let e = rel(d.integrals[i], expected);
        assert!(
            e < 5e-3,
            "light {i}: disc integral {} against {expected} ({e})",
            d.integrals[i]
        );
    }
    assert_clean();
}

/// Within 1% of the full disc: a fixed 64-point pattern cannot hold 1% of a thin crescent.
#[test]
fn textured_disc_follows_the_lambert_phase_curve() {
    let sun = Vec3::new(0.2, 0.7, 0.3).normalize();
    let side = sun.cross(Vec3::Z).normalize();
    let phases = [0.0f32, 45.0, 90.0, 135.0];
    let mut table = vec![uniform_sun(sun, Vec3::ONE).gpu()];
    for a in phases {
        // The angle between the sunlight's travel and the view toward the disc is `a`.
        let toward_disc = -(Mat3::from_axis_angle(side, a.to_radians()) * sun);
        table.push(textured(toward_disc, Vec3::ZERO, SUN_RADIUS, TextureIndex::NONE).gpu());
    }
    let d = discs(&table, [0.0; 3]);
    let integral = |i: usize| f64::from(d.states[i].integral[1]);
    let full = integral(1);
    let sin2 = f64::from(SUN_RADIUS).sin().powi(2);
    let lambert_full = sin2 * 2.0 / 3.0 * f64::from(table[0].illuminance[1]);
    assert!(
        (full / lambert_full - 1.0).abs() < 5e-3,
        "full disc {full} against {lambert_full}"
    );
    for (k, a) in phases.iter().enumerate() {
        let a = f64::from(*a).to_radians();
        let curve = (a.sin() + (PI - a) * a.cos()) / PI;
        let got = integral(k + 1) / full;
        assert!(
            (got - curve).abs() < 0.01,
            "phase {} deg: {got} of full against {curve}",
            a.to_degrees()
        );
    }
    for (i, s) in d.states.iter().enumerate() {
        assert_eq!(
            s.integral.map(f32::to_bits),
            [0, 1, 2].map(|c| d.textured[i][c].to_bits()),
            "light {i}: the scene compute kernel's integral differs from the scene pass's"
        );
    }
    assert_clean();
}

#[test]
fn glint_converges_to_the_same_value_from_nee_and_bsdf_sampling() {
    let albedo = gradient_albedo();
    let table = [
        DirectionalLight {
            angular_radius: 0.05,
            ..uniform_sun(Vec3::new(0.3, 0.8, 0.2), Vec3::splat(1.0e5))
        },
        DirectionalLight {
            disc: DiscProfile::LimbDarkened,
            ..uniform_sun(Vec3::new(-0.5, 0.6, 0.1), Vec3::new(5.0e4, 5.0e4, 4.0e4))
        },
        textured(
            Vec3::new(0.1, 0.9, -0.3),
            Vec3::splat(0.3),
            0.03,
            albedo.index,
        ),
    ]
    .map(|l| l.gpu());
    let lights = upload(&table, "lights");
    let mut lighting = scene(&KernelSet::default(), 3);
    let threads = 1u32 << 16;
    let per_thread = 1024;
    let sums: Vec<_> = (0..table.len())
        .map(|i| readback::<[f32; 4]>(u64::from(threads), &format!("glint sums {i}")))
        .collect();
    let roots: Vec<_> = table
        .iter()
        .zip(&sums)
        .enumerate()
        .map(|(i, (l, s))| {
            let a = l.direction_to_light;
            upload(
                &[glint::Root {
                    lights: lights.gpu,
                    states: lighting.directional_states(),
                    sums: s.gpu,
                    _pad: 0,
                    wo: [-a[0], a[1], -a[2], 0.0],
                    light: i as u32,
                    samples_per_thread: per_thread,
                    alpha: 0.1,
                    seed: 5,
                }],
                "glint root",
            )
        })
        .collect();
    let mut list = render().command_list().unwrap();
    lighting.record(&mut list, &pass([0.0; 3], GpuAddress(0), lights.gpu, 3));
    after_scene_pass(&mut list);
    let p = pipeline(&glint::MODULE, "main");
    for root in &roots {
        list.dispatch(&p, root.gpu, [threads / 64, 1, 1]);
    }
    finish(list);

    let n = f64::from(threads) * f64::from(per_thread);
    for (i, s) in sums.iter().enumerate() {
        let mut t = [0.0f64; 4];
        for v in s.as_slice().unwrap() {
            for k in 0..4 {
                t[k] += f64::from(v[k]);
            }
        }
        let mean = |k: usize| t[k] / n;
        let error = |k: usize| ((t[k + 1] / n - mean(k).powi(2)) / n).sqrt();
        let (nee, bsdf) = (mean(0), mean(2));
        let sigma = (error(0).powi(2) + error(2).powi(2)).sqrt();
        assert!(nee > 0.0, "light {i}: no glint");
        assert!(
            (nee - bsdf).abs() < 4.0 * sigma,
            "light {i}: NEE {nee} against BSDF {bsdf} (sigma {sigma})"
        );
    }
    assert_clean();
}

#[test]
fn stock_height_fog_matches_a_numerical_integral() {
    let (density, base, falloff) = (2e-3f64, 50.0, 200.0);
    let fog = upload(
        &[[density as f32, base as f32, falloff as f32, 0.0]],
        "fog data",
    );
    let mut rays: Vec<[f32; 8]> = Vec::new();
    for y0 in [-100.0f32, 0.0, 50.0, 300.0, 2000.0] {
        for dy in [1.0f32, 0.5, 0.05, 1e-4, 0.0, -1e-4, -0.3, -1.0] {
            let d = Vec3::new((1.0 - dy * dy).sqrt(), dy, 0.0);
            for t in [1.0f32, 300.0, 20000.0, f32::INFINITY] {
                if t.is_infinite() && dy <= 0.0 {
                    continue;
                }
                rays.push([0.0, y0, 0.0, t, d.x, d.y, d.z, 0.0]);
            }
        }
    }
    let n = 8192usize;
    let cosines: Vec<f32> = (0..=n)
        .map(|k| 1.0 - 2.0 * (k as f64 / n as f64).powi(4))
        .map(|c| c as f32)
        .collect();
    let rays_gpu = upload(&rays, "rays");
    let cos_gpu = upload(&cosines, "cosines");
    let transmittance = readback::<f32>(rays.len() as u64, "transmittance");
    let diameters = [5.0f32, 20.0, 50.0];
    let phases: Vec<_> = diameters
        .iter()
        .map(|_| readback::<f32>(cosines.len() as u64, "phase"))
        .collect();
    let roots: Vec<_> = diameters
        .iter()
        .zip(&phases)
        .map(|(&d, phase)| {
            upload(
                &[fog_probe::Root {
                    fog: fog.gpu.address().0,
                    rays: rays_gpu.gpu.cast(),
                    transmittance: transmittance.gpu,
                    cosines: cos_gpu.gpu,
                    phase: phase.gpu,
                    ray_count: rays.len() as u32,
                    cosine_count: cosines.len() as u32,
                    diameter_um: d,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                }],
                "fog root",
            )
        })
        .collect();
    let p = pipeline(&fog_probe::MODULE, "main");
    let mut list = render().command_list().unwrap();
    for root in &roots {
        list.dispatch(&p, root.gpu, [(cosines.len() as u32).div_ceil(64), 1, 1]);
    }
    finish(list);

    let extinction = |h: f64| density * (-(h - base) / falloff).exp();
    for (ray, &got) in rays.iter().zip(transmittance.as_slice().unwrap()) {
        let (y0, dy) = (f64::from(ray[1]), f64::from(ray[5]));
        let t = if ray[3].is_infinite() {
            40.0 * falloff / dy
        } else {
            f64::from(ray[3])
        };
        // Simpson's rule along the ray.
        let steps = 20_000;
        let h = t / f64::from(steps);
        let depth: f64 = (0..=steps)
            .map(|k| {
                let w = if k == 0 || k == steps {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                w * extinction(y0 + dy * h * f64::from(k))
            })
            .sum::<f64>()
            * h
            / 3.0;
        let expected = (-depth).exp();
        let got = f64::from(got);
        if expected > 1e-20 {
            assert!(
                (got / expected - 1.0).abs() < 1e-3,
                "ray {ray:?}: {got} against {expected}"
            );
        } else {
            assert!(got < 1e-15, "ray {ray:?}: {got} against {expected}");
        }
    }
    for (d, phase) in diameters.iter().zip(&phases) {
        let p = phase.as_slice().unwrap();
        let integral: f64 = (0..n)
            .map(|k| {
                let dc = f64::from(cosines[k]) - f64::from(cosines[k + 1]);
                0.5 * dc * (f64::from(p[k]) + f64::from(p[k + 1]))
            })
            .sum::<f64>()
            * 2.0
            * PI;
        assert!(
            (integral - 1.0).abs() < 0.01,
            "fog phase at {d} um integrates to {integral}"
        );
    }
    assert_clean();
}
