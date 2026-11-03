//! The dielectric dispatch on the bed: the interior list, Beer transmittance through a slab,
//! rough refraction, the glass furnace, single scattering against a CPU integral, an albedo-1
//! medium, presence layers, a camera inside water, layer guides under camera rotation,
//! the frosted-glass term of rough glass, dispersion: a white line through a prism against a
//! CPU model, a dispersive furnace, and paths without dispersive hits, the layer tiles
//! against tracing every tile, an emissive particle seen through glass, and each pixel's front
//! layer.

use lighting_suite::bed::{
    Accumulated, Analytic, Bed, CLOUD_SPHERE, Capture, DISPERSIVE_SPHERE, EMITTER_LUMINANCE,
    EXTENT, FLOOR, FURNACE_RADIANCE, FrostGuide, GLASS_SPHERE, Item, LIGHT_FLOOR, LINE,
    LINE_HALF_HEIGHT, LINE_HALF_WIDTH, MILK_SIGMA_S, MILK_SIGMA_T, MILKY_SLAB, OverlayFrontLayer,
    PRESENCE_VEIL, PRESENCE_VEIL_OPACITY, PRISM, PRISM_ABBE, PRISM_HALF_HEIGHT, PRISM_IOR,
    Particle, ParticleLook, Paths, ROUGH_SLAB, SLAB_THICKNESS, SMOOTH_FLOOR, SUN, SpareView, Spec,
    TINTED_SLAB, WATER_BOX, fresnel, prism_corners, reference, water_transmission,
};
use lighting_suite::shaders::interior_probe;
use lighting_suite::{assert_clean, device, pipeline, readback, render, require, run, upload};
use orr_gpu::{Hazards, KernelSet, Stages};
use orr_lighting::{
    DirectionalLight, DiscProfile, Light, LightingDebugView, PathEntry, ViewLighting,
};
use orr_math::{DMat3, DVec3, Mat3, Vec2, Vec3};
use orr_scene::Opacity;

/// `MATERIAL_NONE`.
const NONE: u32 = u32::MAX;

/// The front layer at the image's center after the camera turned for one frame, against the
/// depth and motion of an opaque floor in its plane: over a floor at y = 0, a smooth glass
/// slab's top or a veil of low coverage is the front layer, a particle above 0.1 coverage is
/// when it is nearest, and glass in front of a particle keeps it. A particle under the floor
/// or of low coverage, or a layer that is not there, leaves no front layer. The layer's weight
/// is the veil's coverage, the overlay's, and for glass the share of the light it reflects.
#[test]
fn the_front_layer_is_the_nearest_overlay_or_see_through_layer() {
    let mut realtime = paths(ViewLighting::default());
    realtime.overlay_front = true;
    let eye = Vec3::new(0.0, 5.0, 0.01);
    let turned = Vec3::new(0.15, 0.05, 0.0);
    let spec = |copies| from_above(copies, Vec::new(), eye.y);
    let [w, h] = EXTENT;
    let pixels: Vec<usize> = (h / 2 - 4..h / 2 + 4)
        .flat_map(|y| (w / 2 - 4..w / 2 + 4).map(move |x| (y * w + x) as usize))
        .collect();
    let front = |copies: Vec<Item>, particle: Option<(f32, f32)>| {
        let mut bed = Bed::with(spec(copies));
        if let Some((y, opacity)) = particle {
            bed.set_particles(
                &[Particle {
                    center: Vec3::new(0.0, y, 0.0),
                    size: 0.3,
                }],
                ParticleLook {
                    albedo: [0.5; 3],
                    emission: [0.0; 3],
                    opacity,
                    anisotropy: 0.0,
                },
                Opacity::Particle,
            );
        }
        render_paths(&mut bed, &realtime, 1);
        bed.follow(eye, turned, 1, &mut |list, f| realtime.record(list, f));
        realtime.overlay_front(&mut bed)
    };
    // The depth and motion of an opaque floor at height y, as the G-buffer has them.
    let plane = |y: f32| {
        let mut bed = Bed::with(spec(vec![at(FLOOR, Vec3::new(0.0, y, 0.0))]));
        render_paths(&mut bed, &realtime, 1);
        bed.follow(eye, turned, 1, &mut |list, f| realtime.record(list, f));
        let g = bed.guides();
        (
            g.depth.iter().map(|d| d[0]).collect::<Vec<_>>(),
            g.motion.iter().map(|m| [m[0], m[1]]).collect::<Vec<_>>(),
        )
    };
    let floor = at(FLOOR, Vec3::ZERO);
    let slab = at(TINTED_SLAB, Vec3::new(0.0, 0.5, 0.0));
    let veil = at(PRESENCE_VEIL, Vec3::new(0.0, 1.0, 0.0));
    let slab_top = 0.5 + SLAB_THICKNESS * 0.5;

    let is_plane = |what: &str, front: &OverlayFrontLayer, y: f32| {
        let (depth, motion) = (&front.depth, &front.motion);
        let (want_depth, want_motion) = plane(y);
        for &i in &pixels {
            assert!(
                (depth[i] / want_depth[i] - 1.0).abs() < 1e-3,
                "{what}: pixel {i} depth {} against {}",
                depth[i],
                want_depth[i]
            );
            let (m, want) = (motion[i], want_motion[i]);
            assert!(
                (m[0] - want[0]).abs() < 2e-4 && (m[1] - want[1]).abs() < 2e-4,
                "{what}: pixel {i} motion {m:?} against {want:?}"
            );
        }
        assert!(want_motion[pixels[0]][0].abs() > 1e-3, "the camera turned");
    };
    let is_empty = |what: &str, front: OverlayFrontLayer| {
        for &i in &pixels {
            assert_eq!(front.depth[i], 0.0, "{what}: pixel {i}");
            assert_eq!(front.motion[i], [0.0; 2], "{what}: pixel {i}");
            assert_eq!(front.weight[i], 0.0, "{what}: pixel {i}");
        }
    };
    let weighs = |what: &str, front: &OverlayFrontLayer, range: std::ops::Range<f32>| {
        for &i in &pixels {
            let w = front.weight[i];
            assert!(range.contains(&w), "{what}: pixel {i} weight {w}");
        }
    };

    is_empty("an opaque floor", front(vec![floor], None));
    let glass = front(vec![floor, slab], None);
    is_plane("glass with what lies behind dominant", &glass, slab_top);
    weighs("glass", &glass, 0.005..0.5);
    let opacity = PRESENCE_VEIL_OPACITY;
    let veiled = front(vec![floor, veil], None);
    is_plane("a veil below half coverage", &veiled, 1.0);
    weighs("a veil", &veiled, opacity - 0.005..opacity + 0.005);
    let open = front(vec![floor], Some((1.0, 0.5)));
    is_plane("a particle in the open", &open, 1.0);
    weighs("a particle", &open, 0.495..0.505);
    let above = front(vec![floor, slab], Some((1.5, 0.5)));
    is_plane("a particle above glass", &above, 1.5);
    weighs("a particle above glass", &above, 0.495..0.505);
    let below = front(vec![floor, slab], Some((0.2, 0.5)));
    is_plane("a particle below glass", &below, slab_top);
    weighs(
        "a particle below glass, which the glass shows",
        &below,
        0.005..0.5,
    );
    is_empty(
        "a particle under the floor",
        front(vec![floor], Some((-1.0, 0.5))),
    );
    is_empty(
        "a particle below the coverage",
        front(vec![floor], Some((1.0, 0.05))),
    );
    assert_clean();
}

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), PathEntry::Compute, settings)
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
    a.around(Vec2::new(w as f32, h as f32) * 0.5, side / 2)
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
            steps: steps.gpu(),
            lists: lists.gpu(),
            outside: outside.gpu(),
            count: script.len() as u32,
            _pad0: 0,
        }],
        "interior root",
    );
    let probe = pipeline(&interior_probe::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.dispatch(&probe, root.gpu(), [1, 1, 1]);
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
    let leave = 1.0 - fresnel(1.5, c);
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
    let leave = 1.0 - fresnel(1.33, f64::from(sun.y));
    let transmitted = water_transmission();
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

/// Rough glass over an emitter's edge on a black floor: the frosted-glass term softens the
/// edge in the depth-of-field guide, darker on its lit side and lighter on its dark side. The guide keeps
/// what output wrote away from the edge, outside the glass and under smooth glass.
#[test]
fn rough_glass_adds_its_blur_into_the_depth_of_field_guide() {
    const FILL: f32 = 0.25;
    let [w, h] = EXTENT;
    let render = |model| {
        let spec = from_above(
            vec![
                at(SMOOTH_FLOOR, Vec3::ZERO),
                at(LIGHT_FLOOR, Vec3::new(1.0, 0.01, 0.0)),
                at(model, Vec3::new(0.0, 0.5, 0.0)),
            ],
            Vec::new(),
            5.0,
        );
        let mut p = paths(ViewLighting::default());
        p.frost_guide = Some(FrostGuide::new(FILL));
        let radiance = render_paths(&mut Bed::with(spec), &p, 1);
        (radiance, p.frost_guide.as_ref().unwrap().read())
    };
    let rows = h / 2 - 8..h / 2 + 8;
    let column = |v: &dyn Fn(usize) -> f32, x: u32| -> f32 {
        rows.clone().map(|y| v((y * w + x) as usize)).sum::<f32>() / rows.len() as f32
    };

    let (radiance, guide) = render(ROUGH_SLAB);
    let lit = column(&|i| radiance.mean[i][1], w / 2 + 12);
    assert!(lit > 0.05 * EMITTER_LUMINANCE, "the emitter shows at {lit}");
    let term = |x| column(&|i| guide[i] - FILL, x);
    assert!(term(w / 2) < -0.1 * lit, "lit side {}", term(w / 2));
    assert!(term(w / 2 - 1) > 0.1 * lit, "dark side {}", term(w / 2 - 1));
    for x in [w / 2 - 24, w / 2 + 24] {
        assert!(term(x).abs() < 0.01 * lit, "column {x}: {}", term(x));
    }
    for (i, &g) in guide[..(4 * w) as usize].iter().enumerate() {
        assert_eq!(g, FILL, "pixel {i} outside the glass");
    }

    let (_, smooth) = render(TINTED_SLAB);
    for (i, &g) in smooth.iter().enumerate() {
        assert_eq!(g, FILL, "pixel {i} under smooth glass");
    }
    assert_clean();
}

/// A clear dispersive glass sphere in a uniform environment keeps its energy and its color:
/// the wavelengths a path splits into return to white on average.
#[test]
fn a_dispersive_sphere_in_a_furnace_stays_white() {
    let spec = Spec {
        copies: vec![at(DISPERSIVE_SPHERE, Vec3::ZERO)],
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
    let image = render_paths(&mut Bed::with(spec), &paths(settings), 64);
    let mean = drawn_mean(&image);
    for c in 0..3 {
        let ratio = mean[c] / f64::from(FURNACE_RADIANCE[c]);
        assert!(
            (0.97..=1.02).contains(&ratio),
            "channel {c}: {ratio} of the environment"
        );
    }
    assert_clean();
}

/// Without a dispersive material, turning dispersion off changes no pixel's radiance or ray
/// count. The scene has no light the light set samples, whose presampling differs between
/// beds.
#[test]
fn paths_without_dispersive_hits_ignore_the_dispersion_setting() {
    let spec = from_above(
        vec![
            at(LIGHT_FLOOR, Vec3::ZERO),
            at(ROUGH_SLAB, Vec3::new(0.0, 0.5, 0.0)),
            at(GLASS_SPHERE, Vec3::new(0.6, 1.2, 0.0)),
        ],
        Vec::new(),
        3.0,
    );
    let mut bed = Bed::with(spec);
    let mut paths = paths(reference(2));
    let mut render = |dispersion, debug_view| {
        paths.settings = ViewLighting {
            dispersion,
            debug_view,
            ..reference(2)
        };
        render_paths(&mut bed, &paths, 2)
    };
    for view in [LightingDebugView::None, LightingDebugView::RaysPerPixel] {
        let (on, off, again) = (render(true, view), render(false, view), render(true, view));
        assert!(on.mean.iter().any(|p| p[3] > 0.5), "nothing drawn");
        assert!(on == again, "{view:?} differs between identical runs");
        assert!(on == off, "{view:?} differs");
    }
    assert_clean();
}

/// The CPU model's wavelength step, nm, over 360-830 nm.
const STEP_NM: f64 = 0.25;

fn wavelengths() -> impl Iterator<Item = f64> {
    (0..=(470.0 / STEP_NM) as usize).map(|i| 360.0 + i as f64 * STEP_NM)
}

/// The Rec.2020 color each nm of a white spectrum adds, as the kernels take it back from
/// the sampled wavelengths: the color matching functions times the inverse of the round
/// trip's integral.
fn spectral_colors() -> impl Fn(f64) -> DVec3 {
    let to_rgb = orr_color::RgbPrimaries::REC2020.xyz_to_rgb();
    let rgb = move |nm: f64| to_rgb * orr_color::cie_xyz(nm);
    let basis = move |nm: f64| {
        let c = rgb(nm).max(DVec3::ZERO);
        c / c.element_sum()
    };
    let sum = wavelengths().map(|nm| rgb(nm) * STEP_NM).sum::<DVec3>();
    let mut round_trip = DMat3::ZERO;
    for nm in wavelengths() {
        let c = rgb(nm) / sum * STEP_NM;
        let b = basis(nm);
        round_trip += DMat3::from_cols(c * b.x, c * b.y, c * b.z);
    }
    let back = round_trip.inverse() * DMat3::from_diagonal(1.0 / sum);
    move |nm| back * rgb(nm)
}

/// The prism glass's IOR at `nm`: OpenPBR's Cauchy fit through its IOR at 587.6 nm and its
/// Abbe number over 486.1 and 656.3 nm.
fn prism_ior(nm: f64) -> f64 {
    let (d, f, c) = (587.6f64, 486.1f64, 656.3f64);
    let n_d = f64::from(PRISM_IOR);
    let b = (n_d - 1.0) / (f64::from(PRISM_ABBE) * (f.powi(-2) - c.powi(-2)));
    n_d + b * (nm.powi(-2) - d.powi(-2))
}

/// Refraction of unit `d` through a surface of unit normal `n` facing against it, from IOR
/// `from` to `to`; none on total internal reflection.
fn refract(d: DVec3, n: DVec3, from: f64, to: f64) -> Option<DVec3> {
    let eta = from / to;
    let c = -d.dot(n);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    (k >= 0.0).then(|| eta * d + (eta * c - k.sqrt()) * n)
}

/// The prism's side faces: two corners and the outward normal.
fn prism_faces() -> [(DVec3, DVec3, DVec3); 3] {
    let corners = prism_corners().map(Vec3::as_dvec3);
    std::array::from_fn(|i| {
        let (p, q) = (corners[i], corners[(i + 1) % 3]);
        let n = DVec3::Y.cross(q - p).normalize();
        (p, q, if n.dot(p + q) < 0.0 { -n } else { n })
    })
}

/// A ray from `o` along `d` through the prism at IOR `ior`: where it leaves, its direction
/// and the Fresnel transmission of both faces.
fn through_prism(o: DVec3, d: DVec3, ior: f64) -> Option<(DVec3, DVec3, f64)> {
    let faces = prism_faces();
    // The nearest side face the ray meets from outside, or from inside.
    let hit = |o: DVec3, d: DVec3, inside: bool| {
        faces
            .iter()
            .filter_map(|&(p, q, n)| {
                let facing = d.dot(n);
                if facing == 0.0 || (facing > 0.0) != inside {
                    return None;
                }
                let t = (p - o).dot(n) / facing;
                let x = o + d * t;
                let along = (x - p).dot(q - p) / (q - p).length_squared();
                let on = (0.0..=1.0).contains(&along) && x.y.abs() < f64::from(PRISM_HALF_HEIGHT);
                (t > 1e-9 && on).then_some((t, x, n))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
    };
    let (_, x1, n1) = hit(o, d, false)?;
    let inner = refract(d, n1, 1.0, ior)?;
    let (_, x2, n2) = hit(x1, inner, true)?;
    let out = refract(inner, -n2, ior, 1.0)?;
    let t = (1.0 - fresnel(ior, -d.dot(n1))) * (1.0 - fresnel(1.0 / ior, inner.dot(n2)));
    Some((x2, out, t))
}

/// A thin white line seen through a dispersive prism near minimum deviation splits into its
/// spectrum. Each pixel's ray, refracted at each wavelength's IOR, reaches the line over a
/// band of wavelengths; the CPU model adds their colors per pixel. The distance between the
/// blue and red channels' centroids along the rows, which the spread between 486 and 656 nm
/// sets, matches the model's within 2%, and each channel's light within 3%.
#[test]
fn a_white_line_seen_through_a_prism_splits_by_the_abbe_number() {
    let ior = f64::from(PRISM_IOR);
    // The ray that refracts at 587.6 nm into a path parallel to the base.
    let [(_, _, entry), (base_start, base_end, _), _] = prism_faces();
    let inside = (base_end - base_start).normalize();
    let forward = -refract(-inside, -entry, ior, 1.0).expect("leaves the entry face");
    let eye = -forward * 3.0;
    let (exit, out, _) = through_prism(eye, forward, ior).expect("crosses the prism");
    let line_at = exit + out * 4.0;
    let facing = -out;
    let line = Item {
        model: LINE,
        linear: Mat3::from_rotation_y(facing.x.atan2(facing.z) as f32),
        at: line_at.as_vec3(),
    };
    let fov_y = 0.16f32;
    let spec = Spec {
        copies: vec![at(PRISM, Vec3::ZERO), line],
        lights: Vec::new(),
        environment: None,
        eye: eye.as_vec3(),
        target: Vec3::ZERO,
        fov_y,
        focus: None,
    };
    let image = render_paths(&mut Bed::with(spec), &paths(reference(0)), 256);

    let [w, h] = EXTENT;
    let (wf, hf) = (f64::from(w), f64::from(h));
    let back = -forward;
    let right = DVec3::Y.cross(back).normalize();
    let up = back.cross(right);
    let tan = f64::from(fov_y * 0.5).tan();
    let lateral = DVec3::Y.cross(facing);
    let color = spectral_colors();
    let luminance = f64::from(EMITTER_LUMINANCE);
    let mut model = vec![DVec3::ZERO; w as usize];
    let mut rendered = vec![DVec3::ZERO; w as usize];
    for y in h / 2 - 8..h / 2 + 8 {
        for x in 0..w {
            let p = image.mean[(y * w + x) as usize];
            rendered[x as usize] += DVec3::new(p[0].into(), p[1].into(), p[2].into());
            let sx = (2.0 * (f64::from(x) + 0.5) / wf - 1.0) * tan * wf / hf;
            let sy = (1.0 - 2.0 * (f64::from(y) + 0.5) / hf) * tan;
            let d = (forward + right * sx + up * sy).normalize();
            for nm in wavelengths() {
                let Some((o, d, t)) = through_prism(eye, d, prism_ior(nm)) else {
                    continue;
                };
                let toward = d.dot(-facing);
                if toward <= 0.0 {
                    continue;
                }
                let off = o + d * ((line_at - o).dot(-facing) / toward) - line_at;
                if off.dot(lateral).abs() < f64::from(LINE_HALF_WIDTH)
                    && off.y.abs() < f64::from(LINE_HALF_HEIGHT)
                {
                    model[x as usize] += color(nm) * (t * luminance * STEP_NM);
                }
            }
        }
    }
    // The columns the model's spectrum spans. Light that reaches the line by other routes,
    // such as a reflection off the base between the two refractions, which undoes the
    // dispersion, lands elsewhere and is left out.
    let lit: Vec<usize> = (0..w as usize)
        .filter(|&x| model[x].abs().max_element() > 0.0)
        .collect();
    let span = lit[0]..=lit[lit.len() - 1];
    // Where along the row channel `c`'s light lies, in pixels, and how much there is.
    let centroid = |row: &[DVec3], c: usize| {
        let sum: f64 = span.clone().map(|x| row[x][c]).sum();
        let at: f64 = span.clone().map(|x| x as f64 * row[x][c]).sum();
        (at / sum, sum)
    };
    let (model_blue, model_blue_sum) = centroid(&model, 2);
    let (model_red, model_red_sum) = centroid(&model, 0);
    let (blue, blue_sum) = centroid(&rendered, 2);
    let (red, red_sum) = centroid(&rendered, 0);
    let (wanted, got) = (model_blue - model_red, blue - red);
    assert!(
        wanted.abs() > 10.0,
        "the model's blue and red lie {wanted} pixels apart"
    );
    assert!(
        (got - wanted).abs() <= 0.02 * wanted.abs(),
        "blue and red lie {got} pixels apart, in the model {wanted}"
    );
    for (what, a, b) in [
        ("red", red_sum, model_red_sum),
        ("blue", blue_sum, model_blue_sum),
    ] {
        assert!(
            (a - b).abs() <= 0.03 * b,
            "{what} light {a}, in the model {b}"
        );
    }
    assert_clean();
}

/// Tracing only the tiles the layers rastered into, and those beside them, leaves the
/// radiance, the guides and the frosted-glass term bit for bit as tracing every tile does,
/// on scenes whose layers cover part of the image, through the compute and raygen entries,
/// both into views of their own in the same frame. The reference's paths read no cache, so
/// they repeat exactly within a frame.
#[test]
fn layer_tiles_trace_what_every_tile_traces() {
    const TILES: u64 = (EXTENT[0].div_ceil(8) * EXTENT[1].div_ceil(8)) as u64;
    let scenes = [
        Analytic::DielectricSlab.spec(),
        from_above(
            vec![
                at(SMOOTH_FLOOR, Vec3::ZERO),
                at(LIGHT_FLOOR, Vec3::new(1.0, 0.01, 0.0)),
                at(ROUGH_SLAB, Vec3::new(0.0, 0.5, 0.0)),
            ],
            Vec::new(),
            5.0,
        ),
        from_above(
            vec![
                at(LIGHT_FLOOR, Vec3::ZERO),
                at(PRESENCE_VEIL, Vec3::new(0.0, 0.5, 0.0)),
            ],
            Vec::new(),
            16.0,
        ),
        from_above(
            vec![
                at(FLOOR, Vec3::ZERO),
                at(GLASS_SPHERE, Vec3::new(0.4, 0.5, -0.3)),
            ],
            vec![Light::Directional(SUN)],
            16.0,
        ),
    ];
    let mut entries = vec![PathEntry::Compute];
    if require(
        device().caps().ray_dispatch,
        "the raygen entry needs ray dispatch",
    ) {
        entries.push(PathEntry::RayDispatch);
    }
    for entry in entries {
        let paths = |every_tile| {
            let mut p = Paths::new(&KernelSet::default(), entry, reference(2));
            p.frost_guide = Some(FrostGuide::new(0.25));
            p.every_tile = every_tile;
            p
        };
        let (tiled, every) = (paths(false), paths(true));
        for (s, spec) in scenes.iter().enumerate() {
            let what = format!("{entry:?} scene {s}");
            let marks = readback::<u32>(TILES + 1, "layer tiles");
            let mut bed = Bed::with(spec.clone());
            let mut spare = SpareView::new();
            let (a, b) = (Capture::new(bed.view()), Capture::new(&spare.view));
            let done = Stages::COMPUTE | Stages::RAY | Stages::TRANSFER;
            bed.accumulate(1, 1, &mut |list, f| {
                list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
                list.copy(
                    marks.gpu().cast(),
                    f.view.targets.layer_tiles.cast(),
                    4 * (TILES + 1),
                );
                tiled.record(list, f);
                every.record_on(list, f, &mut spare.view);
                list.barrier(done, Stages::COMPUTE | Stages::TRANSFER, Hazards::empty());
                a.record(list, f.target);
                b.record(list, &spare.view);
            });
            let marked = marks.as_slice().unwrap()[..TILES as usize]
                .iter()
                .filter(|&&m| m != 0)
                .count();
            assert!(
                marked > 0 && marked < TILES as usize,
                "{what}: layers marked {marked} of {TILES} tiles"
            );
            let differ = a
                .radiance()
                .iter()
                .zip(b.radiance())
                .filter(|(x, y)| x != y)
                .count();
            assert_eq!(differ, 0, "{what}: radiance differs on {differ} pixels");
            let frost = |p: &Paths| p.frost_guide.as_ref().unwrap().read();
            assert!(
                frost(&tiled) == frost(&every),
                "{what}: frosted-glass term differs"
            );
            let (ga, gb) = (a.guides(), b.guides());
            for (name, x, y) in [
                ("diffuse albedo", &ga.diffuse_albedo, &gb.diffuse_albedo),
                ("specular albedo", &ga.specular_albedo, &gb.specular_albedo),
                ("normal", &ga.normal_roughness, &gb.normal_roughness),
                ("specular motion", &ga.specular_motion, &gb.specular_motion),
                ("depth", &ga.depth, &gb.depth),
                ("motion", &ga.motion, &gb.motion),
            ] {
                assert!(x == y, "{what}: {name} guide differs");
            }
        }
    }
    assert_clean();
}

/// An emissive particle under the tinted slab, seen from above through it and beside it: the
/// slab passes the particle's light by its Fresnel and Beer share, as the dielectric path
/// meets the particle by its emission, and the transparency hook draws none of it over the
/// slab.
#[test]
fn particles_behind_glass_show_through_it() {
    let look = ParticleLook {
        albedo: [0.0; 3],
        emission: [1000.0; 3],
        opacity: 0.5,
        anisotropy: 0.0,
    };
    let particle = [Particle {
        center: Vec3::new(0.0, 0.1, 0.0),
        size: 2.0,
    }];
    let paths = paths(reference(2));
    let seen = |slab: bool| {
        let mut copies = vec![at(FLOOR, Vec3::new(0.0, -50.0, 0.0))];
        if slab {
            copies.push(at(TINTED_SLAB, Vec3::new(0.0, 0.5, 0.0)));
        }
        let mut bed = Bed::with(from_above(copies, Vec::new(), 3.0));
        bed.set_particles(&particle, look, Opacity::Particle);
        center(&render_paths(&mut bed, &paths, 16), 8)
    };
    let open = seen(false);
    let through = seen(true);
    assert_close("beside the slab", open, [500.0; 3], 0.01);
    let (f, t) = (fresnel(1.5, 1.0), 0.5);
    let share = (1.0 - f) * (1.0 - f) * t / (1.0 - f * f * t * t);
    assert_close("through the slab", through, open.map(|v| v * share), 0.03);
    assert_clean();
}
