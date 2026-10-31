//! The dielectric dispatch on the bed: the interior list, Beer transmittance through a slab,
//! rough refraction, the glass furnace, single scattering against a CPU integral, an albedo-1
//! medium, presence layers, a camera inside water, and layer guides under camera rotation.

use lighting_suite::bed::{
    Accumulated, Analytic, Bed, CLOUD_SPHERE, EMITTER_LUMINANCE, EXTENT, FLOOR, FURNACE_RADIANCE,
    GLASS_SPHERE, Item, LIGHT_FLOOR, MILK_SIGMA_S, MILK_SIGMA_T, MILKY_SLAB, PRESENCE_VEIL,
    PRESENCE_VEIL_OPACITY, Paths, ROUGH_SLAB, SUN, Spec, TINTED_SLAB, WATER_BOX, WATER_SIGMA,
};
use lighting_suite::shaders::interior_probe;
use lighting_suite::{assert_clean, pipeline, readback, render, run, upload};
use orr_color::RgbPrimaries;
use orr_gpu::{Hazards, KernelSet, Stages};
use orr_lighting::{
    DirectionalLight, DiscProfile, Light, PathEntry, ReferenceSettings, ViewLighting,
};
use orr_math::{Mat3, Vec3};

/// `MATERIAL_NONE`.
const NONE: u32 = u32::MAX;

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), PathEntry::Fastest, settings)
}

fn reference(max_bounces: u32) -> ViewLighting {
    ViewLighting {
        max_bounces,
        reference: Some(ReferenceSettings::default()),
        ..ViewLighting::default()
    }
}

fn render_paths(bed: &mut Bed, paths: &Paths, frames: u32) -> Accumulated {
    bed.accumulate(frames, 1, &mut |list, f| paths.record(list, f))
}

fn at(model: &'static str, at: Vec3) -> Item {
    Item {
        model,
        linear: Mat3::IDENTITY,
        at,
    }
}

/// A camera 3-4 m above `copies`, looking straight down at the origin.
fn from_above(copies: Vec<Item>, lights: Vec<Light>, height: f32) -> Spec {
    Spec {
        copies,
        lights,
        environment: None,
        eye: Vec3::new(0.0, height, 0.01),
        target: Vec3::ZERO,
        fov_y: 0.3,
        focus: None,
    }
}

/// Mean rgb of the `side` x `side` pixels at the image's center.
fn center(a: &Accumulated, side: u32) -> [f64; 3] {
    let [w, h] = EXTENT;
    let mut sum = [0.0f64; 3];
    for y in (h - side) / 2..(h + side) / 2 {
        for x in (w - side) / 2..(w + side) / 2 {
            let p = a.mean[(y * w + x) as usize];
            for c in 0..3 {
                sum[c] += f64::from(p[c]);
            }
        }
    }
    sum.map(|s| s / f64::from(side * side))
}

/// Mean rgb over the pixels a surface or layer was drawn on.
fn drawn_mean(a: &Accumulated) -> [f64; 3] {
    let drawn: Vec<_> = a.mean.iter().filter(|p| p[3] > 0.5).collect();
    assert!(!drawn.is_empty(), "nothing drawn");
    let mut sum = [0.0f64; 3];
    for p in &drawn {
        for c in 0..3 {
            sum[c] += f64::from(p[c]);
        }
    }
    sum.map(|s| s / drawn.len() as f64)
}

fn assert_close(what: &str, got: [f64; 3], wanted: [f64; 3], tolerance: f64) {
    for c in 0..3 {
        let e = (got[c] - wanted[c]).abs() / wanted[c].abs().max(1e-12);
        assert!(
            e <= tolerance,
            "{what}: channel {c} {got:?} against {wanted:?}"
        );
    }
}

/// Unpolarized Fresnel reflectance into a medium `eta` times denser at incidence cosine `c`.
fn fresnel(eta: f64, c: f64) -> f64 {
    let s2 = (1.0 - c * c) / (eta * eta);
    if s2 >= 1.0 {
        return 1.0;
    }
    let t = (1.0 - s2).sqrt();
    let rs = (c - eta * t) / (c + eta * t);
    let rp = (t - eta * c) / (t + eta * c);
    0.5 * (rs * rs + rp * rp)
}

#[test]
fn the_interior_list_keeps_nested_media_innermost_last() {
    let script: [(u32, u32); 10] = [
        (1, 1),
        (2, 1),
        (2, 0),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (4, 0),
        (9, 0),
        (5, 1),
    ];
    let steps: Vec<_> = script
        .iter()
        .map(|&(material, front)| interior_probe::Crossing { material, front })
        .collect();
    let steps = upload(&steps, "interior steps");
    let lists = readback::<[u32; 4]>(script.len() as u64, "interior lists");
    let outside = readback::<u32>(script.len() as u64, "interior outside");
    let root = upload(
        &[interior_probe::Root {
            steps: steps.gpu,
            lists: lists.gpu,
            outside: outside.gpu,
            count: script.len() as u32,
            _pad0: 0,
        }],
        "interior root",
    );
    let probe = pipeline(&interior_probe::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.dispatch(&probe, root.gpu, [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);
    let n = NONE;
    let wanted: [([u32; 4], u32); 10] = [
        ([1, n, n, n], n),
        ([1, 2, n, n], 1),
        ([1, n, n, n], 1),
        ([1, 3, n, n], 1),
        ([1, 3, 4, n], 3),
        ([1, 3, 4, 5], 4),
        // A full list drops its outermost entry.
        ([3, 4, 5, 6], 5),
        ([3, 5, 6, n], 6),
        // Leaving a medium the list does not hold, or entering one it holds, changes nothing.
        ([3, 5, 6, n], 6),
        ([3, 5, 6, n], 6),
    ];
    let got: Vec<_> = lists
        .as_slice()
        .unwrap()
        .iter()
        .copied()
        .zip(outside.as_slice().unwrap().iter().copied())
        .collect();
    assert_eq!(got, wanted);
    assert_clean();
}

/// A tinted slab over an emitter, seen straight down: Fresnel at both faces and Beer's law
/// over the thickness, its internal reflections included.
#[test]
fn a_slab_transmits_its_beer_law_share_between_two_fresnel_faces() {
    let spec = from_above(
        vec![
            at(LIGHT_FLOOR, Vec3::ZERO),
            at(TINTED_SLAB, Vec3::new(0.0, 0.5, 0.0)),
        ],
        Vec::new(),
        3.0,
    );
    let image = render_paths(&mut Bed::with(spec), &paths(reference(2)), 32);
    let (f, t) = (fresnel(1.5, 1.0), 0.5);
    let through = (1.0 - f) * (1.0 - f) * t / (1.0 - f * f * t * t);
    let l = f64::from(EMITTER_LUMINANCE) * through;
    assert_close("slab", center(&image, 8), [l; 3], 0.015);
    assert_clean();
}

/// A clear glass sphere in a uniform environment keeps at most, and here nearly all, the
/// environment's radiance.
#[test]
fn a_glass_sphere_in_a_furnace_keeps_its_energy() {
    let spec = Spec {
        copies: vec![at(GLASS_SPHERE, Vec3::ZERO)],
        lights: Vec::new(),
        environment: Some(FURNACE_RADIANCE),
        eye: Vec3::new(0.0, 0.0, 4.0),
        target: Vec3::ZERO,
        fov_y: 0.6,
        focus: None,
    };
    let settings = ViewLighting {
        dielectric_depth: 8,
        ..reference(4)
    };
    let image = render_paths(&mut Bed::with(spec), &paths(settings), 16);
    let mean = drawn_mean(&image);
    for c in 0..3 {
        let ratio = mean[c] / f64::from(FURNACE_RADIANCE[c]);
        assert!(
            (0.97..=1.005).contains(&ratio),
            "channel {c}: {ratio} of the environment"
        );
    }
    assert_clean();
}

/// A scattering slab under a sun 10 degrees from the zenith, seen straight down, against
/// the single-scattering integral of the same model: light enters the medium through the
/// surface without bending, scatters isotropically once, and leaves through the camera's
/// refraction.
#[test]
fn a_scattering_slab_matches_single_scattering_within_5_percent() {
    let tilt = 10f32.to_radians();
    let sun = DirectionalLight {
        direction_to_light: Vec3::new(tilt.sin(), tilt.cos(), 0.0),
        illuminance: Vec3::splat(1000.0),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    };
    let spec = from_above(
        vec![at(MILKY_SLAB, Vec3::new(0.0, 0.5, 0.0))],
        vec![Light::Directional(sun)],
        4.0,
    );
    let image = render_paths(&mut Bed::with(spec), &paths(reference(0)), 128);
    let (sigma_t, sigma_s) = (f64::from(MILK_SIGMA_T), f64::from(MILK_SIGMA_S));
    let c = f64::from(tilt.cos());
    let k = sigma_t * (1.0 + 1.0 / c);
    let depth = 1.0;
    let enter = 1.0 - fresnel(1.5, 1.0);
    let leave = 1.0 - fresnel(1.0 / 1.5, c);
    let phase = 1.0 / (4.0 * std::f64::consts::PI);
    let l = 1000.0 * enter * leave * phase * sigma_s * (1.0 - (-k * depth).exp()) / k;
    assert_close("single scattering", center(&image, 16), [l; 3], 0.05);
    assert_clean();
}

/// A sphere of non-absorbing scattering medium in a uniform environment keeps at most its
/// energy.
#[test]
fn an_albedo_one_medium_keeps_at_most_its_energy() {
    let spec = Spec {
        copies: vec![at(CLOUD_SPHERE, Vec3::ZERO)],
        lights: Vec::new(),
        environment: Some(FURNACE_RADIANCE),
        eye: Vec3::new(0.0, 0.0, 4.0),
        target: Vec3::ZERO,
        fov_y: 0.6,
        focus: None,
    };
    let settings = ViewLighting {
        dielectric_depth: 8,
        ..reference(32)
    };
    let image = render_paths(&mut Bed::with(spec), &paths(settings), 16);
    let mean = drawn_mean(&image);
    for c in 0..3 {
        let ratio = mean[c] / f64::from(FURNACE_RADIANCE[c]);
        assert!(
            (0.3..=1.02).contains(&ratio),
            "channel {c}: {ratio} of the environment"
        );
    }
    assert_clean();
}

/// A black veil of opacity 1/4 over an emitter passes three quarters of its light.
#[test]
fn presence_layers_pass_light_by_their_opacity() {
    let spec = from_above(
        vec![
            at(LIGHT_FLOOR, Vec3::ZERO),
            at(PRESENCE_VEIL, Vec3::new(0.0, 0.5, 0.0)),
        ],
        Vec::new(),
        3.0,
    );
    let image = render_paths(&mut Bed::with(spec), &paths(reference(1)), 64);
    let l = f64::from(EMITTER_LUMINANCE) * f64::from(1.0 - PRESENCE_VEIL_OPACITY);
    assert_close("veil", center(&image, 16), [l; 3], 0.03);
    assert_clean();
}

/// The floor seen from inside a water box lit by the sun: against the same floor without
/// water, the water's absorption along the camera ray and along the sun's path to the
/// surface, and Fresnel where sunlight enters.
#[test]
fn a_camera_inside_water_sees_its_absorption() {
    let eye = Vec3::new(0.0, 1.5, 2.0);
    let spec = |copies| Spec {
        copies,
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye,
        target: Vec3::ZERO,
        fov_y: 0.3,
        focus: None,
    };
    let top = 3.99;
    let mut wet = Bed::with(spec(vec![
        at(FLOOR, Vec3::ZERO),
        at(WATER_BOX, Vec3::new(0.0, top - 2.0, 0.0)),
    ]));
    let water = wet.material(WATER_BOX);
    let inside = ViewLighting {
        camera_medium: Some(water),
        ..reference(0)
    };
    let under = center(&render_paths(&mut wet, &paths(inside), 64), 4);
    let mut dry = Bed::with(spec(vec![at(FLOOR, Vec3::ZERO)]));
    let open = center(&render_paths(&mut dry, &paths(reference(0)), 64), 4);
    let sun = SUN.direction_to_light.normalize();
    let path = f64::from(eye.length() + top / sun.y);
    let leave = 1.0 - fresnel(1.0 / 1.33, f64::from(sun.y));
    // The material's transmission color converts to Rec.2020 as every authored color does.
    let color = WATER_SIGMA.map(|s| f64::from(-s).exp());
    let m = RgbPrimaries::REC709.matrix_to(RgbPrimaries::REC2020);
    let transmitted: [f64; 3] = std::array::from_fn(|r| (0..3).map(|k| m[r][k] * color[k]).sum());
    let wanted: [f64; 3] = std::array::from_fn(|c| open[c] * leave * transmitted[c].powf(path));
    assert_close("under water", under, wanted, 0.05);
    assert_clean();
}

/// A camera turning in place moves every static point alike, whatever its depth, so the
/// glass slab's pixels take the same motion as the floor alone gives there.
#[test]
fn layer_guides_reproject_with_the_camera() {
    let realtime = paths(ViewLighting::default());
    let motion = |spec: Spec| {
        let mut bed = Bed::with(spec.clone());
        render_paths(&mut bed, &realtime, 1);
        let turned = spec.target + Vec3::new(0.15, 0.05, 0.0);
        bed.follow(spec.eye, turned, 1, &mut |list, f| realtime.record(list, f));
        bed.guides()
    };
    let glass = Analytic::DielectricSlab.spec();
    let floor = Spec {
        copies: vec![at(FLOOR, Vec3::ZERO)],
        ..glass.clone()
    };
    let (a, b) = (motion(glass), motion(floor));
    let mut layered = 0;
    for (i, (ma, mb)) in a.motion.iter().zip(&b.motion).enumerate() {
        assert!(
            (ma[0] - mb[0]).abs() < 2e-4 && (ma[1] - mb[1]).abs() < 2e-4,
            "pixel {i}: {ma:?} against {mb:?}"
        );
        if (a.depth[i][0] - b.depth[i][0]).abs() > 1e-4 {
            layered += 1;
        }
    }
    assert!(layered > 100, "only {layered} pixels took a layer's depth");
    assert_clean();
}

/// Rough refraction through the tinted slab over the emitter: the same light within 5% as
/// the smooth slab passes, spread over directions the uniform emitter fills alike.
#[test]
fn a_rough_slab_transmits_about_as_much_as_a_smooth_one() {
    let spec = |model| {
        from_above(
            vec![
                at(LIGHT_FLOOR, Vec3::ZERO),
                at(model, Vec3::new(0.0, 0.5, 0.0)),
            ],
            Vec::new(),
            3.0,
        )
    };
    let render = |model| {
        center(
            &render_paths(&mut Bed::with(spec(model)), &paths(reference(2)), 32),
            8,
        )
    };
    assert_close("rough slab", render(ROUGH_SLAB), render(TINTED_SLAB), 0.05);
    assert_clean();
}
