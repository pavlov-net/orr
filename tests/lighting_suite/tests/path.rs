//! The path kernel on the bed's analytic scenes: the furnace, convergence, the three
//! estimators against each other, pixel subsets, directional lights, ray counts, ray
//! origins, the thin lens and the background layer behind foreground, and emissive particles
//! as lights.

use lighting_suite::bed::{
    Accumulated, Analytic, BLACK_SPHERE, BUMPY_FLOOR, Bed, COATED_EMITTER, EMITTER, EMITTER_FIELD,
    EXTENT, FLOOR, FURNACE_RADIANCE, FloorDecal, GLASS_SPHERE, GLOSSY_FLOOR, Item, MIRROR, NEAR,
    Particle, ParticleLook, Paths, QUADRANT_EMITTER, SKINNED_EMITTER, SKY, SMOOTH_FLOOR, SUN, Spec,
    StrandRow, TILED_FLOOR, TINTED_FLOOR, TROUGH, WHITE_FLOOR, assert_close, image_mean, luminance,
    mean_where, pixel_of, reference, surface_mean,
};
use lighting_suite::shaders::origins;
use lighting_suite::welch::{assert_mean, assert_same_mean};
use lighting_suite::{
    assert_clean, device, pipeline, read, readback, render, run, texture, upload,
};
use orr_gpu::{Hazards, KernelSet, Stages};
use orr_lighting::types::LocalLights;
use orr_lighting::{
    DirectionalLight, DiscProfile, DofBackgroundPass, Light, LightId, LightSetGpu,
    LightingDebugView, OriginProbes, PathEntry, ReferenceSettings, SceneLightingSettings, ThinLens,
    ViewLighting,
};
use orr_math::{Affine3, DVec3, Mat3, Quat, Vec2, Vec3};
use orr_scene::{MediumProxy, MediumProxyId, Opacity, Projection, ProxyDensity, RendererDesc};

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), PathEntry::Compute, settings)
}

fn render_paths(bed: &mut Bed, paths: &Paths, frames: u32, seed: u32) -> Accumulated {
    bed.accumulate(frames, seed, &mut |list, f| paths.record(list, f))
}

fn debug(view: LightingDebugView, max_bounces: u32) -> ViewLighting {
    ViewLighting {
        debug_view: view,
        ..reference(max_bounces)
    }
}

fn uniform_disc(direction: Vec3, illuminance: f32, angular_radius: f32) -> DirectionalLight {
    DirectionalLight {
        direction_to_light: direction.normalize(),
        illuminance: Vec3::splat(illuminance),
        angular_radius,
        disc: DiscProfile::Uniform,
    }
}

fn directional(direction: Vec3, illuminance: f32, angular_radius: f32) -> Light {
    Light::Directional(uniform_disc(direction, illuminance, angular_radius))
}

fn at(model: &'static str, linear: Mat3, at: Vec3) -> Item {
    Item { model, linear, at }
}

fn sphere_on_ground(lights: Vec<Light>, environment: Option<[f32; 3]>) -> Spec {
    Spec {
        lights,
        environment,
        ..Analytic::SunAndGround.spec()
    }
}

#[test]
fn an_albedo_one_sphere_in_a_uniform_environment_returns_the_environment() {
    let mut bed = Bed::new(Analytic::Furnace);
    let p = paths(reference(6));
    let image = render_paths(&mut bed, &p, 64, 1);
    let expected = FURNACE_RADIANCE.map(f64::from);
    assert_close("furnace sphere", surface_mean(&image), expected, 0.005);
    assert_close(
        "furnace sky",
        mean_where(&image, |p| p[3] < 0.5),
        expected,
        1e-3,
    );
    assert_clean();
}

#[test]
fn reference_error_halves_with_four_times_the_frames() {
    let mut bed = Bed::new(Analytic::CornellBox);
    let p = paths(reference(4));
    let truth = render_paths(&mut bed, &p, 256, 100);
    let mut error = |frames, seed| {
        let a = render_paths(&mut bed, &p, frames, seed);
        let sum: f64 = a
            .mean
            .iter()
            .zip(&truth.mean)
            .map(|(a, t)| (luminance(a) - luminance(t)).powi(2))
            .sum();
        (sum / a.mean.len() as f64).sqrt()
    };
    let e4 = error(4, 1);
    let e16 = error(16, 2);
    assert!(
        e4 / e16 > 1.7,
        "rms error {e4} at 4 frames and {e16} at 16 frames"
    );
    assert_clean();
}

#[test]
fn half_the_pixels_trace_half_the_rays_and_converge_to_every_pixel() {
    let mut bed = Bed::new(Analytic::CornellBox);
    let subset = |p| ViewLighting {
        pixel_fraction: p,
        ..reference(4)
    };
    let every = image_mean(&render_paths(&mut bed, &paths(subset(1.0)), 256, 1));
    let half = image_mean(&render_paths(&mut bed, &paths(subset(0.5)), 256, 2));
    assert_close("half the pixels", half, every, 0.01);
    let mut rays = |p| {
        let settings = ViewLighting {
            debug_view: LightingDebugView::RaysPerPixel,
            ..subset(p)
        };
        image_mean(&render_paths(&mut bed, &paths(settings), 8, 3))[0]
    };
    let ratio = rays(0.5) / rays(1.0);
    assert!(
        (0.45..0.55).contains(&ratio),
        "rays per pixel ratio {ratio}"
    );
    assert_clean();
}

#[test]
fn next_event_and_bsdf_estimators_agree_with_their_combination() {
    let spec = sphere_on_ground(
        vec![directional(Vec3::new(0.4, 0.8, 0.3), 500.0, 0.15)],
        Some(SKY.map(|c| c / 50.0)),
    );
    let mut bed = Bed::with(spec);
    let full = image_mean(&render_paths(&mut bed, &paths(reference(2)), 128, 1));
    let nee = render_paths(
        &mut bed,
        &paths(debug(LightingDebugView::NeeOnly, 2)),
        128,
        2,
    );
    let bsdf = render_paths(
        &mut bed,
        &paths(debug(LightingDebugView::BsdfOnly, 2)),
        512,
        3,
    );
    assert_close("next-event only", image_mean(&nee), full, 0.01);
    assert_close("BSDF only", image_mean(&bsdf), full, 0.01);
    assert_clean();
}

#[test]
fn a_directional_light_glints_on_a_mirror_smooth_dielectric() {
    let to_light = Vec3::new(0.35, 0.876, 0.263).normalize();
    let spec = Spec {
        copies: vec![at(SMOOTH_FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        lights: vec![directional(to_light, 1000.0, 0.05)],
        environment: None,
        eye: Vec3::new(-to_light.x, to_light.y, -to_light.z) * 3.0,
        target: Vec3::ZERO,
        fov_y: 0.8,
        focus: None,
    };
    let mut bed = Bed::with(spec);
    let image = render_paths(&mut bed, &paths(ViewLighting::default()), 4, 1);
    let disc = 1000.0 / (std::f64::consts::PI * 0.05f64.sin().powi(2));
    let brightest = image.mean.iter().map(luminance).fold(0.0, f64::max);
    assert!(
        brightest > 0.02 * disc,
        "brightest pixel {brightest} nits, disc {disc} nits"
    );
    assert_clean();
}

#[test]
fn grazing_normal_mapped_surfaces_have_no_black_texels() {
    let spec = Spec {
        copies: vec![at(BUMPY_FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        eye: Vec3::new(0.0, 0.3, 5.0),
        target: Vec3::new(0.0, 0.0, -5.0),
        ..sphere_on_ground(vec![Light::Directional(SUN)], Some(SKY))
    };
    let mut bed = Bed::with(spec);
    bed.set_pre_exposure(1.0 / 64.0);
    let image = render_paths(&mut bed, &paths(ViewLighting::default()), 16, 1);
    let mut lit: Vec<f64> = image
        .mean
        .iter()
        .filter(|p| p[3] > 0.5)
        .map(luminance)
        .collect();
    lit.sort_by(f64::total_cmp);
    let median = lit[lit.len() / 2];
    let black = lit.iter().filter(|&&l| l <= 1e-3 * median).count();
    assert_eq!(
        black,
        0,
        "{black} of {} surface pixels are black",
        lit.len()
    );
    assert_clean();
}

#[test]
fn bright_and_faint_directional_lights_add_up() {
    let lights = [
        directional(Vec3::new(0.35, 0.876, 0.263), 100_000.0, 4.65e-3),
        directional(Vec3::new(-0.5, 0.6, 0.2), 100_000.0, 4.65e-3),
        directional(Vec3::new(0.1, 0.3, -0.9), 0.3, 4.65e-3),
    ];
    let p = paths(reference(3));
    let render_with = |lights: Vec<Light>| {
        let mut bed = Bed::with(sphere_on_ground(lights, None));
        bed.set_pre_exposure(1.0 / 64.0);
        image_mean(&render_paths(&mut bed, &p, 64, 1))
    };
    let all = render_with(lights.to_vec());
    let mut sum = [0.0; 3];
    for l in lights {
        let one = render_with(vec![l]);
        for c in 0..3 {
            sum[c] += one[c];
        }
    }
    assert_close("three lights against their sum", all, sum, 0.01);
    assert_clean();
}

#[test]
fn rays_per_pixel_do_not_grow_with_directional_lights() {
    let albedo = texture(2, 2, &[[0.6, 0.6, 0.6, 1.0]; 4], "moon albedo");
    let mut lights = Vec::new();
    for k in 0..2 {
        lights.push(Light::Directional(DirectionalLight {
            disc: DiscProfile::LimbDarkened,
            ..uniform_disc(Vec3::new(0.3 + k as f32, 0.8, 0.2), 50_000.0, 4.65e-3)
        }));
    }
    for k in 0..5 {
        lights.push(Light::Directional(DirectionalLight {
            direction_to_light: Vec3::new(-0.4 + 0.2 * k as f32, 0.5, -0.6).normalize(),
            illuminance: Vec3::splat(0.5),
            angular_radius: 0.01,
            disc: DiscProfile::Textured {
                albedo: albedo.index,
                up: Vec3::Y,
                lit_by: [LightId(0), LightId::NONE, LightId::NONE, LightId::NONE],
            },
        }));
    }
    let settings = debug(LightingDebugView::RaysPerPixel, 3);
    let p = paths(settings);
    let rays = |lights| {
        let mut bed = Bed::with(sphere_on_ground(lights, Some(SKY)));
        let image = render_paths(&mut bed, &p, 8, 1);
        let max = image.mean.iter().map(|p| p[0]).fold(0.0, f32::max);
        assert!(max <= 2.0 * (settings.max_bounces + 1) as f32, "{max} rays");
        image_mean(&image)[0]
    };
    let with = rays(lights);
    let without = rays(Vec::new());
    assert!(
        (with - without).abs() <= 0.01 * without,
        "{with} rays per pixel with seven directional lights, {without} with none"
    );
    assert_clean();
}

#[test]
fn every_analytic_scene_traces_finite_radiance() {
    let p = paths(debug(LightingDebugView::RaysPerPixel, 4));
    for analytic in Analytic::ALL {
        let image = render_paths(&mut Bed::new(analytic), &p, 4, 1);
        let bad = image.mean.iter().filter(|p| p[1] != 0.0).count();
        assert_eq!(
            bad, 0,
            "{analytic:?}: {bad} pixels with non-finite radiance"
        );
    }
    assert_clean();
}

#[test]
fn rays_leave_surfaces_from_the_origin_to_30_km() {
    let module = pipeline(&origins::MODULE, "main");
    let mut cases = Vec::new();
    let mut seed = 0x9e37_79b9u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32 * 2.0 - 1.0
    };
    for distance in [0.0f32, 2_000.0, 30_000.0] {
        for i in 0..512u32 {
            let n = Vec3::new(next(), next(), next()).normalize();
            let d = Vec3::new(next(), next(), next()).normalize();
            let p = Vec3::new(next(), next(), next()).normalize() * distance
                + Vec3::new(next(), next(), next());
            cases.push(origins::OriginCase {
                position: p.to_array(),
                policy: i % 3,
                geometric_normal: n.to_array(),
                error: if i.is_multiple_of(2) { 0.0 } else { 0.01 },
                dir: d.to_array(),
                camera_distance: if i % 4 < 2 { 1.0 } else { p.length() },
            });
        }
    }
    let input = upload(&cases, "origin cases");
    let out = readback::<[f32; 4]>(cases.len() as u64, "origins");
    let root = upload(
        &[origins::Root {
            cases: input.gpu(),
            origins: out.gpu(),
            count: cases.len() as u32,
            _pad0: 0,
            _pad1: 0,
        }],
        "origins root",
    );
    let mut list = render().command_list().unwrap();
    list.dispatch(&module, root.gpu(), [cases.len().div_ceil(64) as u32, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);
    for (c, o) in cases.iter().zip(out.as_slice().unwrap()) {
        let p = DVec3::from(c.position.map(f64::from));
        let o = DVec3::from([o[0], o[1], o[2]].map(f64::from));
        let n = DVec3::from(c.geometric_normal.map(f64::from)).normalize();
        let d = DVec3::from(c.dir.map(f64::from));
        let side = if n.dot(d) < 0.0 { -n } else { n };
        let floor = 1e-3 * (1.0 + f64::from(c.camera_distance));
        let wanted = if c.policy == 0 {
            floor.max(f64::from(c.error))
        } else {
            floor
        };
        let moved = (o - p).dot(side);
        let precision = p.abs().max_element() / 1048576.0;
        assert!(
            moved >= wanted && moved <= wanted + 3.0 * precision + 1e-6,
            "{c:?}: origin {moved} m off the surface, wanted {wanted} m"
        );
    }
    assert_clean();
}

#[test]
fn a_coarser_traced_cut_casts_no_self_shadow_speckle() {
    let angle = 5f32.to_radians();
    let eye = Vec3::new(0.0, 3.0, -6.0);
    let fov_y = 0.8;
    // The view sees only the valley: the trough's near end lies below the frame.
    let spec = Spec {
        copies: vec![at(TROUGH, Mat3::IDENTITY, Vec3::ZERO)],
        lights: vec![directional(
            Vec3::new(0.0, angle.sin(), angle.cos()),
            1000.0,
            4.65e-3,
        )],
        environment: None,
        eye,
        target: Vec3::new(0.0, 0.3, 0.0),
        fov_y,
        focus: Some((
            eye,
            Projection { fov_y, near: NEAR }.pixels_per_radian(EXTENT[1]),
        )),
    };
    // The raster draws the finest level; ray tracing holds the coarsest, whose error stays
    // under its threshold even at the trough's nearest point to the eye.
    let renderer = RendererDesc {
        raster_error_pixels: 0.01,
        rt_error_pixels: 16.0,
        ..RendererDesc::default()
    };
    let mut bed = Bed::with_renderer(spec, &renderer);
    let image = render_paths(
        &mut bed,
        &paths(debug(LightingDebugView::DirectOnly, 0)),
        4,
        1,
    );
    let lit: Vec<f64> = image
        .mean
        .iter()
        .filter(|p| p[3] > 0.5)
        .map(luminance)
        .collect();
    let mean = lit.iter().sum::<f64>() / lit.len() as f64;
    let speckle = lit.iter().filter(|&&l| l < 0.25 * mean).count();
    assert!(
        speckle * 1000 <= lit.len(),
        "{speckle} of {} pixels shadow themselves",
        lit.len()
    );
    let bias_paths = paths(debug(LightingDebugView::RayOriginBias, 0));
    let mut probes = OriginProbes::default();
    let center = EXTENT.map(|e| e / 2);
    let bias = bed.accumulate(1, 1, &mut |list, f| {
        bias_paths.record(list, f);
        let v = f.view;
        probes
            .record(
                device(),
                &KernelSet::default(),
                list,
                f.frame,
                v.scene,
                v.view,
                v.gbuffer,
                center,
            )
            .unwrap();
    });
    let probe = probes.take(bed.last_frame()).pop().unwrap();
    let g = probe.gpu;
    assert!(
        g.raster_error == 0.0 && g.traced_error > 2.0 * g.floor,
        "the traced cut is not coarser than the raster at the center pixel:\n{probe}"
    );
    let bias = surface_mean(&bias)[0];
    let floor = 1e-3 * (1.0 + 6.0);
    assert!(
        bias > 2.0 * floor,
        "x1 offsets {bias} m on average: the paths do not offset by the traced cut's error"
    );
    assert_clean();
}

/// Pearson correlation of `pairs`.
fn correlation(pairs: impl Iterator<Item = (f64, f64)> + Clone) -> f64 {
    let n = pairs.clone().count() as f64;
    let (ma, mb) = pairs
        .clone()
        .fold((0.0, 0.0), |(x, y), (a, b)| (x + a / n, y + b / n));
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (a, b) in pairs {
        ab += (a - ma) * (b - mb);
        aa += (a - ma) * (a - ma);
        bb += (b - mb) * (b - mb);
    }
    ab / (aa * bb).sqrt()
}

#[test]
fn noise_on_a_lit_plane_is_uncorrelated_across_pixels_and_frames() {
    let spec = Spec {
        copies: vec![at(FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        lights: Vec::new(),
        environment: Some(SKY),
        eye: Vec3::new(0.0, 4.0, 1.0),
        target: Vec3::ZERO,
        fov_y: 0.6,
        focus: None,
    };
    let mut bed = Bed::with(spec);
    bed.set_pre_exposure(1.0 / 16.0);
    let p = paths(ViewLighting::default());
    let first = render_paths(&mut bed, &p, 1, 5);
    let second = bed.accumulate_from(1, 1, 5, &mut |list, f| p.record(list, f));
    let l = |a: &Accumulated, i: usize| luminance(&a.mean[i]);
    let [w, h] = EXTENT.map(|v| v as usize);
    let drawn = |i: usize| first.mean[i][3] > 0.5;
    let right = (0..w * h).filter(|&i| i % w + 1 < w && drawn(i) && drawn(i + 1));
    let down = (0..w * (h - 1)).filter(|&i| drawn(i) && drawn(i + w));
    let frames = (0..w * h).filter(|&i| drawn(i));
    let across = correlation(right.map(|i| (l(&first, i), l(&first, i + 1))));
    let below = correlation(down.map(|i| (l(&first, i), l(&first, i + w))));
    let over_time = correlation(frames.map(|i| (l(&first, i), l(&second, i))));
    for (what, r) in [("across", across), ("down", below)] {
        assert!(r.abs() < 0.05, "noise correlation {what}: {r}");
    }
    // Consecutive frames take consecutive points of each pixel's scrambled Sobol sequence,
    // which are stratified against each other: their correlation is negative by design.
    assert!(
        over_time < 0.05,
        "noise correlation frame to frame: {over_time}"
    );
    assert_clean();
}

#[test]
fn a_thin_lens_with_no_aperture_matches_the_gbuffer() {
    let mut bed = Bed::new(Analytic::SunAndGround);
    bed.set_pre_exposure(1.0 / 64.0);
    let gbuffer = render_paths(&mut bed, &paths(reference(2)), 64, 1);
    let lens = ViewLighting {
        reference: Some(ReferenceSettings {
            lens: Some(ThinLens {
                aperture_radius: 1e-5,
                focus_distance: 5.0,
            }),
            ..ReferenceSettings::default()
        }),
        ..reference(2)
    };
    let traced = render_paths(&mut bed, &paths(lens), 64, 1);
    assert_close(
        "thin lens against the G-buffer",
        surface_mean(&traced),
        surface_mean(&gbuffer),
        0.02,
    );
    let e = traced.flip(&gbuffer, 1.0 / 500.0);
    assert!(e.mean < 0.05, "thin lens against the G-buffer: {e:?}");
    assert_clean();
}

#[test]
fn the_background_layer_sees_past_the_foreground_as_a_lens_does_without_it() {
    let lens = ThinLens {
        aperture_radius: 0.05,
        focus_distance: 5.0,
    };
    let background = Analytic::SunAndGround.spec();
    let mut without = Bed::with(background.clone());
    without.set_pre_exposure(1.0 / 64.0);
    let truth = render_paths(
        &mut without,
        &paths(ViewLighting {
            reference: Some(ReferenceSettings {
                lens: Some(lens),
                ..ReferenceSettings::default()
            }),
            ..reference(2)
        }),
        256,
        1,
    );
    drop(without);

    // A small mirror 1.2 m in front of the eye hides part of the sphere and the ground; its
    // shadow falls outside the view.
    let forward = (background.target - background.eye).normalize();
    let mut spec = background;
    spec.copies.push(Item {
        model: MIRROR,
        linear: Mat3::from_diagonal(Vec3::splat(0.3)),
        at: spec.eye + 1.2 * forward + Vec3::new(-0.25, 0.0, 0.0),
    });
    let mut with = Bed::with(spec);
    with.set_pre_exposure(1.0 / 64.0);
    let hidden = render_paths(&mut with, &paths(reference(2)), 16, 1);
    assert!(
        hidden.flip(&truth, 1.0 / 500.0).mean > 0.06,
        "the foreground hides too little of the background"
    );

    // Pixels left of `MARKED` start their background rays 2 m along the view axis, past the
    // mirror and before the ground.
    const MARKED: u32 = 96;
    let start = NEAR / 2.0;
    let mask: Vec<[f32; 4]> = (0..EXTENT[0] * EXTENT[1])
        .map(|i| {
            [
                if i % EXTENT[0] < MARKED { start } else { 0.0 },
                0.0,
                0.0,
                0.0,
            ]
        })
        .collect();
    let mask = texture(EXTENT[0], EXTENT[1], &mask, "dof mask");
    let focus = upload(&[lens.focus_distance], "focus distance");
    let mut layer_paths = paths(reference(2));
    layer_paths.dof_background = Some(DofBackgroundPass {
        mask: mask.index,
        aperture_radius: lens.aperture_radius,
        focus_distance: focus.gpu(),
    });
    with.accumulate_background_layer(true);
    let mut layer = render_paths(&mut with, &layer_paths, 256, 1);

    for (i, p) in layer.mean.iter_mut().enumerate() {
        if i as u32 % EXTENT[0] < MARKED {
            assert!(
                (0.0..=start * 1.001).contains(&p[3]),
                "pixel {i}: layer depth {} not past the mask's {start}",
                p[3]
            );
        } else {
            assert_eq!(*p, [0.0, 0.0, 0.0, -1.0], "unmarked pixel {i}");
            *p = truth.mean[i];
        }
    }
    let e = layer.flip(&truth, 1.0 / 500.0);
    assert!(
        e.mean < 0.06,
        "background layer against the lens reference without the foreground: {e:?}"
    );
    assert_clean();
}

/// A ring of colored sphere lights, two spots and three rects around the sphere on the
/// ground, under a dim sky and no directional light.
fn many_lights() -> Spec {
    let mut lights = Vec::new();
    for k in 0..24 {
        let a = f64::from(k) * std::f64::consts::TAU / 24.0;
        let r = 1.6 + 0.4 * f64::from(k % 3);
        let hue = Vec3::new(
            1.0 + (k % 2) as f32,
            1.0 + (k % 3) as f32 * 0.5,
            1.0 + (k % 5) as f32 * 0.25,
        );
        lights.push(Light::Sphere {
            position: DVec3::new(r * a.cos(), 0.3 + 0.2 * f64::from(k % 4), r * a.sin()),
            radius: 0.05,
            intensity: hue * (2.0 + (k % 7) as f32 * 3.0),
            cone: None,
        });
    }
    for x in [-0.8f64, 0.8] {
        lights.push(Light::Sphere {
            position: DVec3::new(x, 2.5, 0.8),
            radius: 0.1,
            intensity: Vec3::splat(150.0),
            cone: Some(orr_scene::Cone {
                direction: Vec3::new(-x as f32 * 0.3, -1.0, -0.2).normalize(),
                inner: 0.3,
                outer: 0.45,
            }),
        });
    }
    for (k, z) in [-1.5f64, 0.0, 1.5].into_iter().enumerate() {
        lights.push(Light::Rect {
            center: DVec3::new(-2.5, 1.0, z),
            edge_x: Vec3::new(0.0, 0.0, 0.6),
            edge_y: Vec3::new(0.0, 0.4, 0.0),
            luminance: Vec3::splat(40.0 * (k + 1) as f32),
        });
    }
    Spec {
        lights,
        environment: Some(SKY.map(|c| c / 2000.0)),
        ..Analytic::SunAndGround.spec()
    }
}

/// Mean rgb of each quarter of the image.
fn quarter_means(a: &Accumulated) -> Vec<[f64; 3]> {
    let [w, h] = EXTENT.map(|e| e as usize);
    let mut out = Vec::new();
    for (qx, qy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let mut sum = [0.0f64; 3];
        for y in qy * h / 2..(qy + 1) * h / 2 {
            for x in qx * w / 2..(qx + 1) * w / 2 {
                let p = a.mean[y * w + x];
                for c in 0..3 {
                    sum[c] += f64::from(p[c]);
                }
            }
        }
        out.push(sum.map(|s| s / (w * h / 4) as f64));
    }
    out
}

#[test]
fn light_feedback_and_local_samplers_keep_direct_light_unbiased() {
    let mut bed = Bed::with(many_lights());
    let direct = ViewLighting {
        debug_view: LightingDebugView::DirectOnly,
        ..ViewLighting::default()
    };
    let by_power = SceneLightingSettings {
        feedback_share: 0.0,
        ..*bed.lighting.settings()
    };
    bed.lighting.set_settings(&by_power).unwrap();
    let plain = render_paths(
        &mut bed,
        &paths(ViewLighting {
            local_light_share: 0.0,
            ..direct
        }),
        96,
        1,
    );
    bed.lighting
        .set_settings(&SceneLightingSettings::default())
        .unwrap();
    let guided = render_paths(&mut bed, &paths(direct), 96, 2);
    let stats = bed.lighting.stats(bed.last_frame()).unwrap();
    assert!(stats.feedback_pixels > 0, "no pixel fed back a light");
    assert_close("image", image_mean(&guided), image_mean(&plain), 0.015);
    for (q, (g, p)) in quarter_means(&guided)
        .into_iter()
        .zip(quarter_means(&plain))
        .enumerate()
    {
        assert_close(&format!("quarter {q}"), g, p, 0.04);
    }
    assert_clean();
}

/// Tiles of the bed view's local samplers that hold a slot for the light with feedback id `id`.
fn tiles_naming(bed: &Bed, id: u32) -> usize {
    let (local, [columns, rows]) = bed.view().local_lights();
    let tiles: Vec<LocalLights> = read(local, u64::from(columns * rows));
    tiles
        .iter()
        .filter(|t| t.lights[..t.count as usize].iter().any(|l| l.id == id))
        .count()
}

/// A sphere light and the sky feed back from the ground they light, and the frame after the
/// light table and the emissive list grow finds both in the view's local samplers as the frame
/// before did: the environment's feedback id moved with the two, and the samplers read what
/// the view fed back by the ids of the tables it was written under. The scene's selection
/// still counts the feedback of the frame that grew the tables.
#[test]
fn a_views_light_feedback_follows_its_lights_as_the_light_tables_grow() {
    let lamp = |intensity: f32, x: f64| Light::Sphere {
        position: DVec3::new(x, 1.5, 1.0),
        radius: 0.1,
        intensity: Vec3::splat(intensity),
        cone: None,
    };
    let mut bed = Bed::with(sphere_on_ground(
        vec![lamp(200.0, 0.0)],
        Some(SKY.map(|c| c / 20.0)),
    ));
    let p = paths(ViewLighting {
        debug_view: LightingDebugView::DirectOnly,
        ..ViewLighting::default()
    });
    let environment =
        |bed: &Bed| read::<LightSetGpu>(bed.lighting.light_set(), 1)[0].candidate_count;
    bed.accumulate(4, 1, &mut |list, f| p.record(list, f));
    let held = environment(&bed);
    let before = [0, held].map(|id| tiles_naming(&bed, id));
    assert!(
        before.iter().all(|&tiles| tiles > 0),
        "tiles naming the lamp and the sky: {before:?}"
    );

    bed.set_light(LightId(40), Some(lamp(0.0, 30.0)));
    // A dark particle out of sight takes the emissive list's first entry.
    let speck = Particle {
        center: Vec3::new(0.0, -50.0, 0.0),
        size: 0.01,
    };
    let dark = ParticleLook {
        albedo: [0.0; 3],
        emission: [0.0; 3],
        opacity: 1.0,
        anisotropy: 0.0,
    };
    bed.set_particles(&[speck], dark, Opacity::Particle);
    bed.accumulate_on(4, 1, 1, &mut |list, f| p.record(list, f));
    let grown = environment(&bed);
    assert!(
        grown > held + 40,
        "the light table and the emissive list never grew"
    );
    let after = [0, grown].map(|id| tiles_naming(&bed, id));
    for (what, (before, after)) in ["lamp", "sky"]
        .into_iter()
        .zip(before.into_iter().zip(after))
    {
        assert!(
            2 * after > before,
            "{what}: {after} tiles name it after the table grew, {before} before"
        );
    }
    bed.accumulate_on(5, 1, 1, &mut |list, f| p.record(list, f));
    let stats = bed.lighting.stats(bed.last_frame()).unwrap();
    assert!(
        stats.feedback_pixels > 0,
        "no pixel of the frame that grew the table fed back a light"
    );
    assert_clean();
}

/// The floor in front of an emissive particle of opacity 0.5 is lit as in front of an opaque
/// emitter of half its emission: light sampling takes the particle's emission at its opacity,
/// and the rays leaving the floor meet it there, weighted against light sampling.
#[test]
fn an_emissive_particle_lights_as_an_emitter_of_its_opacity() {
    let spec = Spec {
        copies: vec![at(FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 2.0, 4.0),
        target: Vec3::ZERO,
        fov_y: 0.8,
        focus: None,
    };
    let quad = [Particle {
        center: Vec3::new(0.0, 0.6, -1.0),
        size: 0.5,
    }];
    let paths = paths(ViewLighting::default());
    let [w, h] = EXTENT;
    let floor = |emission: f32, opacity: f32, class: Opacity| {
        let mut bed = Bed::with(spec.clone());
        let look = ParticleLook {
            albedo: [0.0; 3],
            emission: [emission; 3],
            opacity,
            anisotropy: 0.0,
        };
        bed.set_particles(&quad, look, class);
        let a = render_paths(&mut bed, &paths, 32, 1);
        let rows = &a.mean[((h - 16) * w) as usize..];
        let mut sum = [0.0f64; 3];
        for p in rows {
            for c in 0..3 {
                sum[c] += f64::from(p[c]);
            }
        }
        sum.map(|s| s / rows.len() as f64)
    };
    let particle = floor(2000.0, 0.5, Opacity::Particle);
    let opaque = floor(1000.0, 1.0, Opacity::Opaque);
    assert_close("floor", particle, opaque, 0.03);
    assert_clean();
}

/// A quad with a back of its own lights the floor on each side by both techniques: light
/// sampling takes each side by its emission and reads what that side shows, and BSDF rays weigh
/// what they meet against it, so a black back leaves the floor behind dark to both. A mirrored
/// placement keeps the sides.
#[test]
fn light_sampling_takes_each_side_of_a_surface_with_a_back() {
    let spec = Spec {
        copies: vec![at(FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 2.0, 4.0),
        target: Vec3::ZERO,
        fov_y: 0.8,
        focus: None,
    };
    let quad = [Particle {
        center: Vec3::new(0.0, 1.0, 0.0),
        size: 1.0,
    }];
    let look = |emission: f32| ParticleLook {
        albedo: [0.0; 3],
        emission: [emission; 3],
        opacity: 1.0,
        anisotropy: 0.0,
    };
    let before = pixel_of(&spec, Vec3::new(0.0, 0.0, 1.4));
    let behind = pixel_of(&spec, Vec3::new(1.5, 0.0, -1.0));
    let full = paths(reference(1));
    let nee = paths(debug(LightingDebugView::NeeOnly, 1));
    let bsdf = paths(debug(LightingDebugView::BsdfOnly, 1));
    for (back, mirror) in [(0.0, 1.0), (2000.0, 1.0), (2000.0, -1.0)] {
        let mut bed = Bed::with(spec.clone());
        let place = Affine3::from_scale(Vec3::new(mirror, 1.0, 1.0));
        bed.set_sided_quads(&quad, look(1000.0), look(back), place);
        let lit = |bed: &mut Bed, paths: &Paths, frames| {
            let a = render_paths(bed, paths, frames, 1);
            [a.around(before, 4), a.around(behind, 4)]
        };
        let by_full = lit(&mut bed, &full, 128);
        let by_nee = lit(&mut bed, &nee, 128);
        let by_bsdf = lit(&mut bed, &bsdf, 512);
        assert!(by_full[0][0] > 1.0, "the floor in front: {by_full:?}");
        assert_close("next-event in front", by_nee[0], by_full[0], 0.05);
        assert_close("BSDF in front", by_bsdf[0], by_full[0], 0.05);
        assert_eq!(by_nee[1][0] > 1.0, back > 0.0, "next-event behind");
        assert_eq!(by_bsdf[1][0] > 1.0, back > 0.0, "BSDF behind");
        if back > 0.0 {
            assert_close("next-event behind", by_nee[1], by_full[1], 0.1);
            assert_close("BSDF behind", by_bsdf[1], by_full[1], 0.1);
        }
    }
    assert_clean();
}

/// Per-render means of green over the `2 half` square pixels around `at`, each of `frames`
/// frames from its own seed.
fn patch_means(
    bed: &mut Bed,
    paths: &Paths,
    at: Vec2,
    renders: u32,
    frames: u32,
    seed: u32,
) -> Vec<f64> {
    (0..renders)
        .map(|k| render_paths(bed, paths, frames, seed * 1000 + k).around(at, 6)[1])
        .collect()
}

/// The floor below `copy` seen from the side, lit by it alone.
fn over_floor(floor: &'static str, copy: Item, lights: Vec<Light>) -> Spec {
    Spec {
        copies: vec![at(floor, Mat3::IDENTITY, Vec3::ZERO), copy],
        lights,
        environment: None,
        eye: Vec3::new(0.0, 1.5, 4.0),
        target: Vec3::new(0.0, 0.3, 0.0),
        fov_y: 0.8,
        focus: None,
    }
}

/// A cooked emitter tilted and stretched unevenly over the floor lights it alike by next-event
/// estimation and by BSDF rays, by Welch's test over renders: light sampling draws the copy's
/// traced triangles at the density that hits on them read, and the light set names its entry.
/// Under a coat and fuzz, next-event estimation passes the winner's light through them along
/// its shadow ray, as BSDF rays see it; with an emission texture, it reads the texel at the uv
/// of the point it draws, as the hit there does.
#[test]
fn a_cooked_emitter_lights_the_floor_alike_by_next_event_and_bsdf_rays() {
    let linear = Mat3::from_rotation_x(-0.6) * Mat3::from_diagonal(Vec3::new(1.6, 1.0, 0.7));
    for emitter in [EMITTER, COATED_EMITTER, QUADRANT_EMITTER] {
        let spec = over_floor(
            FLOOR,
            at(emitter, linear, Vec3::new(0.0, 1.0, 0.0)),
            Vec::new(),
        );
        let patch = pixel_of(&spec, Vec3::new(0.0, 0.0, 0.6));
        let mut bed = Bed::with(spec);
        let nee = patch_means(
            &mut bed,
            &paths(debug(LightingDebugView::NeeOnly, 1)),
            patch,
            16,
            8,
            1,
        );
        let stats = bed.lighting.stats(bed.last_frame()).unwrap();
        assert_eq!(stats.emissive_entries, 1, "{emitter}: emissive entries");
        let bsdf = patch_means(
            &mut bed,
            &paths(debug(LightingDebugView::BsdfOnly, 1)),
            patch,
            16,
            32,
            2,
        );
        let full = patch_means(&mut bed, &paths(reference(1)), patch, 16, 8, 3);
        assert!(
            nee.iter().sum::<f64>() > 0.0,
            "{emitter}: next-event lit nothing"
        );
        assert_same_mean(&format!("{emitter}: next-event against BSDF"), &nee, &bsdf);
        assert_same_mean(&format!("{emitter}: full against BSDF"), &full, &bsdf);
    }
    assert_clean();
}

/// A floor over an emitter that faces its underside stays dark by next-event estimation and by
/// BSDF rays, though its normal map turns texels toward directions below it: a surface takes
/// light only from the side of its triangle the path is on.
#[test]
fn a_normal_map_takes_no_light_through_its_surface() {
    let up = Mat3::from_rotation_x(std::f32::consts::PI) * Mat3::from_diagonal(Vec3::splat(6.0));
    let spec = over_floor(
        BUMPY_FLOOR,
        at(EMITTER, up, Vec3::new(0.0, -0.5, 0.0)),
        Vec::new(),
    );
    let mut bed = Bed::with(spec);
    for view in [LightingDebugView::NeeOnly, LightingDebugView::BsdfOnly] {
        let image = render_paths(&mut bed, &paths(debug(view, 1)), 16, 1);
        let brightest = image.mean.iter().map(luminance).fold(0.0, f64::max);
        assert_eq!(brightest, 0.0, "{view:?}: the brightest pixel");
    }
    assert_clean();
}

/// The floor under a single-sided emitter is lit by next-event estimation and by BSDF rays where
/// the emitter faces it, and stays dark to both where its back does: a back emits nothing.
#[test]
fn a_single_sided_emitters_back_lights_nothing() {
    for (down, linear) in [
        (true, Mat3::IDENTITY),
        (false, Mat3::from_diagonal(Vec3::new(1.0, -1.0, -1.0))),
    ] {
        let spec = over_floor(FLOOR, at(EMITTER, linear, Vec3::Y), Vec::new());
        let patch = pixel_of(&spec, Vec3::new(0.0, 0.0, 0.6));
        let mut bed = Bed::with(spec);
        for view in [LightingDebugView::NeeOnly, LightingDebugView::BsdfOnly] {
            let green = render_paths(&mut bed, &paths(debug(view, 1)), 16, 1).around(patch, 4)[1];
            assert_eq!(
                green > 0.0,
                down,
                "{view:?}, facing down {down}: green {green}"
            );
        }
    }
    assert_clean();
}

/// The single-sided floor turned over, so the sun lights its back, shows in a mirror as its
/// front does there: a ray that reaches a single-sided surface's back shades it as the front.
#[test]
fn a_single_sided_floor_shows_its_back_in_a_mirror_lit_as_its_front() {
    const MIRROR_Z: f32 = -1.5;
    let spec = |floor: Mat3| Spec {
        copies: vec![
            at(FLOOR, floor, Vec3::ZERO),
            at(MIRROR, Mat3::IDENTITY, Vec3::new(0.0, 1.0, MIRROR_Z)),
        ],
        lights: vec![Light::Directional(SUN)],
        environment: None,
        eye: Vec3::new(2.0, 1.6, 3.0),
        target: Vec3::new(0.3, 0.5, MIRROR_Z),
        fov_y: 0.8,
        focus: None,
    };
    let upright = spec(Mat3::IDENTITY);
    let patch = pixel_of(&upright, Vec3::new(0.0, 0.0, 2.0 * MIRROR_Z - 0.6));
    let p = paths(reference(2));
    let lit = |spec: Spec| {
        let mut bed = Bed::with(spec);
        bed.set_pre_exposure(1.0 / 64.0);
        render_paths(&mut bed, &p, 16, 1).around(patch, 4)
    };
    let front = lit(upright);
    let back = lit(spec(Mat3::from_diagonal(Vec3::new(1.0, -1.0, -1.0))));
    assert!(front[1] > 0.0, "the front in the mirror: {front:?}");
    assert_close("the back in the mirror", back, front, 0.03);
    assert_clean();
}

/// A floor whose normal map tilts texels away from a grazing view keeps every texel sunlit where
/// a mirror shows it, so its vertex is resolved from a hit: the shading normal leans toward the
/// ray on the triangle's front, and only a ray behind the triangle's plane turns the frame. The
/// sun alone lights it, since a texel turned below the floor would still see a sky.
#[test]
fn a_grazing_normal_mapped_floor_in_a_mirror_has_no_black_texels() {
    const MIRROR_Z: f32 = -1.5;
    let spec = Spec {
        copies: vec![
            at(BUMPY_FLOOR, Mat3::IDENTITY, Vec3::ZERO),
            at(MIRROR, Mat3::IDENTITY, Vec3::new(0.0, 1.0, MIRROR_Z)),
        ],
        eye: Vec3::new(0.0, 0.3, 1.0),
        target: Vec3::new(0.0, 0.12, MIRROR_Z),
        fov_y: 0.3,
        ..sphere_on_ground(vec![Light::Directional(SUN)], None)
    };
    // The mirror's rows that show the floor about 3 to 8 m from the eye's image, 2 to 6 degrees
    // under its view.
    let rows = [0.2, 0.03].map(|y| pixel_of(&spec, Vec3::new(0.0, y, MIRROR_Z)).y.round() as u32);
    let mut bed = Bed::with(spec);
    bed.set_pre_exposure(1.0 / 64.0);
    let image = render_paths(&mut bed, &paths(ViewLighting::default()), 16, 1);
    let width = EXTENT[0] as usize;
    let mut lit: Vec<f64> = image.mean[rows[0] as usize * width..rows[1] as usize * width]
        .iter()
        .map(luminance)
        .collect();
    lit.sort_by(f64::total_cmp);
    let median = lit[lit.len() / 2];
    let black = lit.iter().filter(|&&l| l <= 1e-3 * median).count();
    assert_eq!(black, 0, "{black} of {} texels are black", lit.len());
    assert_clean();
}

/// A single-sided sheet over the floor casts the sun's shadow whichever side faces the sun: rays
/// meet a single-sided surface's back.
#[test]
fn a_single_sided_sheet_shadows_from_either_side() {
    let sheet = Mat3::from_diagonal(Vec3::splat(0.1));
    let spec = |linear: Mat3| {
        over_floor(
            FLOOR,
            at(FLOOR, linear, Vec3::Y),
            vec![Light::Directional(SUN)],
        )
    };
    let to_sun = SUN.direction_to_light;
    let under = spec(sheet);
    let shadowed = pixel_of(&under, Vec3::Y - to_sun / to_sun.y);
    let open = pixel_of(&under, Vec3::new(1.5, 0.0, 1.0));
    let p = paths(debug(LightingDebugView::NeeOnly, 1));
    for (side, linear) in [
        ("front", sheet),
        (
            "back",
            Mat3::from_diagonal(Vec3::new(1.0, -1.0, -1.0)) * sheet,
        ),
    ] {
        let mut bed = Bed::with(spec(linear));
        bed.set_pre_exposure(1.0 / 64.0);
        let a = render_paths(&mut bed, &p, 4, 1);
        let [dark, lit] = [shadowed, open].map(|at| a.around(at, 2)[1]);
        assert!(lit > 0.0, "{side} to the sun: the open floor {lit}");
        assert!(
            dark <= 1e-3 * lit,
            "{side} to the sun: green {dark} in the shadow, {lit} beside it"
        );
    }
    assert_clean();
}

/// The floor seen in a mirror takes a cooked emitter's light alike by next-event estimation and
/// by BSDF rays, under a coat and fuzz too: the floor is then the path's second vertex, which
/// the plain opaque shade takes, and its next-event estimate passes the winner's light through
/// the emitter's coat and fuzz as the BSDF ray that hits the emitter sees it.
#[test]
fn a_coated_emitter_lights_a_floor_seen_in_a_mirror_alike_by_next_event_and_bsdf_rays() {
    let linear = Mat3::from_rotation_x(-0.6) * Mat3::from_diagonal(Vec3::new(1.6, 1.0, 0.7));
    const MIRROR_Z: f32 = -1.5;
    for emitter in [EMITTER, COATED_EMITTER] {
        let spec = Spec {
            copies: vec![
                at(FLOOR, Mat3::IDENTITY, Vec3::ZERO),
                at(emitter, linear, Vec3::new(0.0, 1.0, 0.0)),
                at(MIRROR, Mat3::IDENTITY, Vec3::new(0.0, 1.0, MIRROR_Z)),
            ],
            lights: Vec::new(),
            environment: None,
            eye: Vec3::new(2.0, 1.6, 3.0),
            target: Vec3::new(0.3, 0.5, MIRROR_Z),
            fov_y: 0.8,
            focus: None,
        };
        // The floor point the emitter lights, where the mirror shows it.
        let lit = Vec3::new(0.0, 0.0, 0.6);
        let patch = pixel_of(&spec, Vec3::new(lit.x, lit.y, 2.0 * MIRROR_Z - lit.z));
        let mut bed = Bed::with(spec);
        let nee = patch_means(
            &mut bed,
            &paths(debug(LightingDebugView::NeeOnly, 2)),
            patch,
            16,
            8,
            1,
        );
        let bsdf = patch_means(
            &mut bed,
            &paths(debug(LightingDebugView::BsdfOnly, 2)),
            patch,
            16,
            32,
            2,
        );
        let full = patch_means(&mut bed, &paths(reference(2)), patch, 16, 8, 3);
        assert!(
            nee.iter().sum::<f64>() > 0.0,
            "{emitter}: next-event lit nothing"
        );
        assert_same_mean(&format!("{emitter}: next-event against BSDF"), &nee, &bsdf);
        assert_same_mean(&format!("{emitter}: full against BSDF"), &full, &bsdf);
    }
    assert_clean();
}

/// Pipelines compiled for what a scene may hold read what its surfaces read at a hit: a floor
/// whose vertex colors tint its base color, under a decal, seen in a mirror, so the path's
/// second vertex is resolved from its hit and not from the G-buffer. The pipelines that hold
/// everything show the tint and the decal there, and the scene's own render the same image.
#[test]
fn specialized_pipelines_read_vertex_colors_and_decals_at_a_hit() {
    const MIRROR_Z: f32 = -1.5;
    let spec = Spec {
        copies: vec![
            at(TINTED_FLOOR, Mat3::IDENTITY, Vec3::ZERO),
            at(MIRROR, Mat3::IDENTITY, Vec3::new(0.0, 1.0, MIRROR_Z)),
        ],
        lights: Vec::new(),
        environment: Some(FURNACE_RADIANCE),
        eye: Vec3::new(2.0, 1.6, 3.0),
        target: Vec3::new(0.3, 0.5, MIRROR_Z),
        fov_y: 0.8,
        focus: None,
    };
    // A black decal on the floor, and the floor points left of it the mirror shows.
    let decal = FloorDecal {
        material: BLACK_SPHERE,
        center: Vec3::new(0.0, 0.0, 0.6),
        half_extent: Vec3::new(0.8, 0.8, 0.5),
    };
    let seen = |x: f32| pixel_of(&spec, Vec3::new(x, 0.0, 2.0 * MIRROR_Z - decal.center.z));
    let mut paths = paths(ViewLighting {
        debug_view: LightingDebugView::BsdfOnly,
        max_bounces: 2,
        ..ViewLighting::default()
    });
    let mut render = |specialize| {
        paths.specialize = specialize;
        let mut bed = Bed::with_features(spec.clone(), &[decal]);
        render_paths(&mut bed, &paths, 8, 1)
    };
    let general = render(false);
    let [under, near, far] = [0.0, -1.4, -2.2].map(|x| general.around(seen(x), 2));
    assert!(
        under[1] < 0.1 * near[1],
        "the decal: green {} under it, {} beside it",
        under[1],
        near[1]
    );
    // The floor's vertex colors go from red at -x to green at +x.
    let (near, far) = (near[0] / near[1], far[0] / far[1]);
    assert!(
        far > 1.2 * near,
        "the tint: red over green {far} at -2.2 m, {near} at -1.4 m"
    );
    let specialized = render(true);
    assert!(
        bits(&general) == bits(&specialized),
        "the scene's own pipelines changed the image"
    );
    assert_clean();
}

/// The image as bits, for renders that must agree exactly.
fn bits(a: &Accumulated) -> Vec<[u32; 4]> {
    a.mean.iter().map(|p| p.map(f32::to_bits)).collect()
}

/// A floor at `floor_x` along x and the mirror standing over it at `mirror_z`, in a furnace,
/// from an eye that sees the floor in front of the mirror reflected in it.
fn mirrored_floor(floor: &'static str, floor_x: f32, mirror_z: f32) -> Spec {
    Spec {
        copies: vec![
            at(floor, Mat3::IDENTITY, Vec3::new(floor_x, 0.0, 0.0)),
            at(MIRROR, Mat3::IDENTITY, Vec3::new(0.0, 1.0, mirror_z)),
        ],
        lights: Vec::new(),
        environment: Some(FURNACE_RADIANCE),
        eye: Vec3::new(2.0, 1.6, 3.0),
        target: Vec3::new(0.3, 0.5, mirror_z),
        fov_y: 0.8,
        focus: None,
    }
}

/// Pipelines compiled for what a scene may hold read a tile set at a hit: a floor whose base
/// color is two tiles, a dark one and a bright one, seen in a mirror, so the path's second
/// vertex is resolved from its hit. The pipelines that hold everything show each tile where its
/// square of uv lies, and the scene's own render the same image.
#[test]
fn specialized_pipelines_read_a_tile_set_at_a_hit() {
    const MIRROR_Z: f32 = -1.5;
    // The tiles meet under x = -1.
    let spec = mirrored_floor(TILED_FLOOR, -1.0, MIRROR_Z);
    let seen = |x: f32| pixel_of(&spec, Vec3::new(x, 0.0, 2.0 * MIRROR_Z - 0.6));
    let mut paths = paths(ViewLighting {
        debug_view: LightingDebugView::BsdfOnly,
        max_bounces: 2,
        ..ViewLighting::default()
    });
    let mut render = |specialize| {
        paths.specialize = specialize;
        let mut bed = Bed::with_features(spec.clone(), &[]);
        render_paths(&mut bed, &paths, 8, 1)
    };
    let general = render(false);
    let [dark, bright] = [-1.8, -0.3].map(|x| general.around(seen(x), 2)[1]);
    assert!(
        bright > 5.0 * dark,
        "the tiles: green {dark} on the dark one, {bright} on the bright one"
    );
    let specialized = render(true);
    assert!(
        bits(&general) == bits(&specialized),
        "the scene's own pipelines changed the image"
    );
    assert_clean();
}

/// Pipelines compiled for what a scene may hold shade a strand at a hit: a fence of dark
/// strands on the floor, seen in a mirror, so the path's second vertex is resolved from its
/// hit on a strand. The pipelines that hold everything show the fence darker than the floor
/// beside it, and the scene's own render the same image.
#[test]
fn specialized_pipelines_shade_strands_at_a_hit() {
    const MIRROR_Z: f32 = -1.5;
    let spec = mirrored_floor(FLOOR, 0.0, MIRROR_Z);
    // Strands that touch, from x = -0.5 to 0.1 in front of the mirror.
    let row = StrandRow {
        count: 16,
        first: Vec3::new(-0.5, 0.0, 0.6),
        step: Vec3::new(0.04, 0.0, 0.0),
        height: 1.0,
        radius: 0.03,
        albedo: [0.05; 3],
    };
    let seen = |x: f32| pixel_of(&spec, Vec3::new(x, 0.5, 2.0 * MIRROR_Z - row.first.z));
    let mut paths = paths(ViewLighting {
        debug_view: LightingDebugView::BsdfOnly,
        max_bounces: 2,
        ..ViewLighting::default()
    });
    let mut render = |specialize| {
        paths.specialize = specialize;
        let mut bed = Bed::with(spec.clone());
        bed.set_strands(row);
        render_paths(&mut bed, &paths, 8, 1)
    };
    let general = render(false);
    let [fence, beside] = [-0.2, -1.4].map(|x| general.around(seen(x), 2)[1]);
    assert!(
        fence < 0.5 * beside,
        "the strands: green {fence} on them, {beside} beside them"
    );
    let specialized = render(true);
    assert!(
        bits(&general) == bits(&specialized),
        "the scene's own pipelines changed the image"
    );
    assert_clean();
}

/// A view whose frames stop tracing photons keeps none of their pipelines among those its
/// frames ran: the bed checks each frame on its scene's own pipelines against what it asked.
#[test]
fn a_view_that_stops_tracing_photons_runs_only_what_it_asks_for() {
    let mut bed = Bed::new(Analytic::CornellBox);
    let mut real_time = paths(ViewLighting::default());
    real_time.specialize = true;
    render_paths(&mut bed, &real_time, 2, 1);
    let mut reference = paths(ViewLighting {
        reference: Some(ReferenceSettings {
            photons: 0,
            ..ReferenceSettings::default()
        }),
        ..reference(2)
    });
    reference.specialize = true;
    render_paths(&mut bed, &reference, 2, 1);
    assert_clean();
}

/// Green of each of the `2 half` square pixels around `at`, row by row.
fn patch_pixels(a: &Accumulated, at: Vec2, half: u32) -> Vec<f64> {
    let w = EXTENT[0];
    let (cx, cy) = (at.x.round() as u32, at.y.round() as u32);
    (cy - half..cy + half)
        .flat_map(|y| (cx - half..cx + half).map(move |x| (y * w + x) as usize))
        .map(|i| f64::from(a.mean[i][1]))
        .collect()
}

/// A skinned emitter bent off its rest pose lights the floor alike by next-event estimation and
/// by BSDF rays: light sampling draws the triangles of the cut its skinned BLAS traces, as
/// skinned, at the density hits on them read.
#[test]
fn a_skinned_emitter_off_its_rest_pose_lights_the_floor_alike_by_next_event_and_bsdf_rays() {
    let spec = Spec {
        copies: vec![at(FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 1.5, 4.0),
        target: Vec3::new(0.0, 0.3, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let patch = pixel_of(&spec, Vec3::new(0.3, 0.0, 0.4));
    let mut bed = Bed::with(spec);
    // The second joint turns the strip's +x half down toward the floor.
    bed.add_skinned(
        SKINNED_EMITTER,
        Affine3::from_translation(Vec3::new(0.0, 0.8, 0.0)),
        &[Affine3::IDENTITY, Affine3::from_rotation_z(-0.7)],
    );
    let nee = patch_means(
        &mut bed,
        &paths(debug(LightingDebugView::NeeOnly, 1)),
        patch,
        16,
        8,
        1,
    );
    let stats = bed.lighting.stats(bed.last_frame()).unwrap();
    assert_eq!(stats.emissive_entries, 1, "emissive entries");
    assert_ne!(
        bed.skin_state(0).blas,
        0,
        "the strip traces no skinned BLAS"
    );
    let bsdf = patch_means(
        &mut bed,
        &paths(debug(LightingDebugView::BsdfOnly, 1)),
        patch,
        16,
        32,
        2,
    );
    assert!(nee.iter().sum::<f64>() > 0.0, "next-event lit nothing");
    assert_same_mean("next-event against BSDF", &nee, &bsdf);
    assert_clean();
}

/// A sparse field of small emitters close over the floor lights it alike by next-event
/// estimation with and without the local samplers, which refine the field's tree toward the
/// floor each tile sees, by the two weighed against BSDF rays, and by BSDF rays alone; the
/// refinement lowers the variance of next-event estimation, as the nodes near a tile's surfaces
/// carry most of its light.
#[test]
fn a_sparse_emitter_field_lights_alike_by_every_technique_and_refines_toward_the_floor() {
    const RENDERS: u32 = 24;
    const FRAMES: u32 = 2;
    const WARM: u32 = 4;
    let spec = over_floor(
        FLOOR,
        at(EMITTER_FIELD, Mat3::IDENTITY, Vec3::new(0.0, 0.4, 0.0)),
        Vec::new(),
    );
    let patch = pixel_of(&spec, Vec3::new(0.0, 0.0, 1.5));
    let mut bed = Bed::with(spec);
    // Local samplers run outside the reference, on the light feedback of the frames before, so
    // each technique's renders continue one view after a camera cut and frames that feed it.
    let refined = ViewLighting {
        debug_view: LightingDebugView::NeeOnly,
        reference: None,
        ..reference(1)
    };
    let flat = ViewLighting {
        local_light_share: 0.0,
        ..refined
    };
    let weighed = ViewLighting {
        debug_view: LightingDebugView::DirectOnly,
        ..refined
    };
    let mut continued = |settings: ViewLighting, seed: u32| -> Vec<Accumulated> {
        let p = paths(settings);
        render_paths(&mut bed, &p, WARM, seed);
        (0..RENDERS)
            .map(|k| {
                let first = WARM + k * FRAMES;
                bed.accumulate_on(first, FRAMES, seed, &mut |list, f| p.record(list, f))
            })
            .collect()
    };
    let by_refined = continued(refined, 1);
    let by_flat = continued(flat, 2);
    let by_weighed = continued(weighed, 3);
    let means =
        |a: &[Accumulated]| -> Vec<f64> { a.iter().map(|a| a.around(patch, 6)[1]).collect() };
    let by_bsdf = patch_means(
        &mut bed,
        &paths(debug(LightingDebugView::BsdfOnly, 1)),
        patch,
        RENDERS,
        16,
        4,
    );
    assert!(by_bsdf.iter().sum::<f64>() > 0.0, "the field lit nothing");
    assert_same_mean("refined against BSDF", &means(&by_refined), &by_bsdf);
    assert_same_mean("flat against BSDF", &means(&by_flat), &by_bsdf);
    assert_same_mean("weighed against BSDF", &means(&by_weighed), &by_bsdf);
    // Each pixel's variance over the renders, pooled over the patch.
    let variance = |a: &[Accumulated]| {
        let pixels: Vec<Vec<f64>> = a.iter().map(|a| patch_pixels(a, patch, 6)).collect();
        let n = pixels.len() as f64;
        let mut pooled = 0.0;
        for i in 0..pixels[0].len() {
            let m = pixels.iter().map(|p| p[i]).sum::<f64>() / n;
            pooled += pixels.iter().map(|p| (p[i] - m).powi(2)).sum::<f64>() / (n - 1.0);
        }
        pooled / pixels[0].len() as f64
    };
    let (vr, vf) = (variance(&by_refined), variance(&by_flat));
    assert!(vr < vf, "refined variance {vr} against flat {vf}");
    assert_clean();
}

/// A sphere light and a rect light over a glossy floor show in its reflection alike by
/// next-event estimation and by BSDF rays, which meet the lights' proxies and weigh them against
/// light sampling: as placed, and once the lights have moved to ids past their table and a
/// medium proxy set after them has moved their instances in the PTLAS.
#[test]
fn sphere_and_rect_lights_reach_bsdf_rays_through_their_proxies() {
    let lights = vec![
        Light::Sphere {
            position: DVec3::new(-0.6, 1.2, -0.5),
            radius: 0.15,
            intensity: Vec3::splat(40.0),
            cone: None,
        },
        Light::Rect {
            center: DVec3::new(0.6, 1.2, -0.5),
            edge_x: Vec3::new(0.4, 0.0, 0.0),
            edge_y: Vec3::new(0.0, 0.0, 0.3),
            luminance: Vec3::splat(200.0),
        },
    ];
    // The rect faces edge_x x edge_y: down.
    let spec = Spec {
        copies: vec![at(GLOSSY_FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        lights,
        environment: None,
        eye: Vec3::new(0.0, 1.0, 3.0),
        target: Vec3::new(0.0, 0.0, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let mut bed = Bed::with(spec.clone());
    let lit = |bed: &mut Bed, when: &str| {
        for (what, under) in [
            ("sphere", Vec3::new(-0.6, 0.0, 0.2)),
            ("rect", Vec3::new(0.6, 0.0, 0.2)),
        ] {
            let patch = pixel_of(&spec, under);
            let nee = patch_means(
                bed,
                &paths(debug(LightingDebugView::NeeOnly, 1)),
                patch,
                16,
                8,
                1,
            );
            let bsdf = patch_means(
                bed,
                &paths(debug(LightingDebugView::BsdfOnly, 1)),
                patch,
                16,
                16,
                2,
            );
            let full = patch_means(bed, &paths(reference(1)), patch, 16, 8, 3);
            assert!(
                bsdf.iter().sum::<f64>() > 0.0,
                "{when}, {what}: BSDF rays met no light"
            );
            assert_same_mean(
                &format!("{when}, {what}: next-event against BSDF"),
                &nee,
                &bsdf,
            );
            assert_same_mean(&format!("{when}, {what}: full against BSDF"), &full, &bsdf);
        }
    };
    lit(&mut bed, "as placed");
    // The lights move to ids past their table and past the proxies the PTLAS holds for it. A
    // frame later a medium proxy that nothing meets moves the light proxies' instances, which
    // the PTLAS made for it writes anew.
    for (from, to) in [(0, 23), (1, 57)] {
        bed.set_light(LightId(from), None);
        bed.set_light(LightId(to), Some(spec.lights[from as usize]));
    }
    render_paths(&mut bed, &paths(reference(1)), 1, 0);
    bed.set_medium_proxy(
        MediumProxyId(6),
        Some(MediumProxy {
            position: DVec3::new(0.0, 500.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            density: ProxyDensity::Uniform {
                half_extent: Vec3::ONE,
            },
            extinction: Vec3::ZERO,
            albedo: Vec3::ZERO,
            anisotropy: 0.0,
        }),
    );
    lit(&mut bed, "after the tables grew");
    assert_clean();
}

/// A sphere light lights the floor below it alike at any radius that keeps it clear of the
/// floor's horizon, a radius below the smallest half included.
#[test]
fn a_sphere_light_smaller_than_any_half_lights_like_a_larger_one() {
    let spec = |radius: f32| Spec {
        copies: vec![at(WHITE_FLOOR, Mat3::IDENTITY, Vec3::ZERO)],
        lights: vec![Light::Sphere {
            position: DVec3::new(0.0, 1.0, 0.0),
            radius,
            intensity: Vec3::splat(40.0),
            cone: None,
        }],
        environment: None,
        eye: Vec3::new(0.0, 1.0, 3.0),
        target: Vec3::new(0.0, 0.0, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let patch = pixel_of(&spec(0.1), Vec3::new(0.3, 0.0, 0.4));
    let mut means = Vec::new();
    for radius in [0.1, 1e-9] {
        let mut bed = Bed::with(spec(radius));
        means.push(patch_means(
            &mut bed,
            &paths(debug(LightingDebugView::NeeOnly, 1)),
            patch,
            16,
            8,
            1,
        ));
    }
    let larger = means[0].iter().sum::<f64>() / means[0].len() as f64;
    assert!(larger > 0.0, "the light lit nothing");
    // The nearly noiseless point takes a tolerance: tile samples round each light's radiance to
    // 9 bits, differently at each radius.
    assert_mean("1 nm against 10 cm", &means[1], larger, 0.005);
    assert_clean();
}

/// A sphere light behind a glass sphere lights the glossy floor below alike by the full
/// estimator and by BSDF rays alone: rays bent through the glass meet the light's proxy, and
/// what reaches the floor through the glass is owned once.
#[test]
fn a_sphere_light_behind_glass_lights_a_glossy_floor_alike_by_full_and_bsdf_rays() {
    let spec = Spec {
        copies: vec![
            at(GLOSSY_FLOOR, Mat3::IDENTITY, Vec3::ZERO),
            at(
                GLASS_SPHERE,
                Mat3::from_diagonal(Vec3::splat(0.3)),
                Vec3::new(0.0, 0.6, 0.0),
            ),
        ],
        lights: vec![Light::Sphere {
            position: DVec3::new(0.0, 1.5, 0.0),
            radius: 0.12,
            intensity: Vec3::splat(60.0),
            cone: None,
        }],
        environment: None,
        eye: Vec3::new(0.0, 1.2, 2.5),
        target: Vec3::new(0.0, 0.2, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let patch = pixel_of(&spec, Vec3::new(0.0, 0.0, 0.3));
    let mut bed = Bed::with(spec);
    let full = patch_means(&mut bed, &paths(reference(3)), patch, 16, 16, 1);
    let bsdf = patch_means(
        &mut bed,
        &paths(debug(LightingDebugView::BsdfOnly, 3)),
        patch,
        16,
        64,
        2,
    );
    assert!(bsdf.iter().sum::<f64>() > 0.0, "BSDF rays met no light");
    assert_same_mean("full against BSDF", &full, &bsdf);
    assert_clean();
}
