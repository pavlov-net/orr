//! The opaque path kernel on the bed's analytic scenes: the furnace, convergence, the three
//! estimators against each other, directional lights, ray counts, ray origins, linked
//! surfaces at every hit, and the raygen against the compute entry.

use lighting_suite::bed::{
    Accumulated, Analytic, BUMPY_FLOOR, Bed, EXTENT, FLOOR, FURNACE_RADIANCE, Item, MIRROR, PANEL,
    Paths, SKY, SMOOTH_FLOOR, SUN, Spec, TROUGH, WHITE_SPHERE,
};
use lighting_suite::shaders::origins;
use lighting_suite::{
    assert_clean, device, linked, pipeline, readback, render, require, run, upload,
};
use orr_gpu::{Hazards, KernelSet, Stages};
use orr_lighting::{
    DirectionalLight, DiscProfile, Light, LightId, LightingDebugView, PathEntry, ReferenceSettings,
    ThinLens, ViewLighting,
};
use orr_math::{DVec3, Mat3, Vec3};
use orr_scene::RendererDesc;

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), PathEntry::Fastest, settings)
}

fn render_paths(bed: &mut Bed, paths: &Paths, frames: u32, seed: u32) -> Accumulated {
    bed.accumulate(frames, seed, &mut |list, f| paths.record(list, f))
}

fn reference(max_bounces: u32) -> ViewLighting {
    ViewLighting {
        max_bounces,
        reference: Some(ReferenceSettings::default()),
        ..ViewLighting::default()
    }
}

fn debug(view: LightingDebugView, max_bounces: u32) -> ViewLighting {
    ViewLighting {
        debug_view: view,
        ..reference(max_bounces)
    }
}

fn luminance(p: &[f32; 4]) -> f64 {
    0.2627 * f64::from(p[0]) + 0.6780 * f64::from(p[1]) + 0.0593 * f64::from(p[2])
}

/// Mean rgb over the pixels that pass `keep`.
fn mean_where(a: &Accumulated, keep: impl Fn(&[f32; 4]) -> bool) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut n = 0.0;
    for p in a.mean.iter().filter(|p| keep(p)) {
        for c in 0..3 {
            sum[c] += f64::from(p[c]);
        }
        n += 1.0;
    }
    assert!(n > 0.0, "no pixels selected");
    sum.map(|s| s / n)
}

fn surface_mean(a: &Accumulated) -> [f64; 3] {
    mean_where(a, |p| p[3] > 0.5)
}

fn image_mean(a: &Accumulated) -> [f64; 3] {
    mean_where(a, |_| true)
}

fn assert_close(what: &str, a: [f64; 3], b: [f64; 3], tolerance: f64) {
    for c in 0..3 {
        let e = (a[c] - b[c]).abs() / b[c].abs().max(1e-12);
        assert!(e <= tolerance, "{what}: channel {c} {a:?} against {b:?}");
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
    let albedo = lighting_suite::texture(2, 2, &[[0.6, 0.6, 0.6, 1.0]; 4], "moon albedo");
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
                error: if i % 2 == 0 { 0.0 } else { 0.01 },
                dir: d.to_array(),
                camera_distance: if i % 4 < 2 { 1.0 } else { p.length() },
            });
        }
    }
    let input = upload(&cases, "origin cases");
    let out = readback::<[f32; 4]>(cases.len() as u64, "origins");
    let root = upload(
        &[origins::Root {
            cases: input.gpu,
            origins: out.gpu,
            count: cases.len() as u32,
            _pad0: 0,
            _pad1: 0,
        }],
        "origins root",
    );
    let mut list = render().command_list().unwrap();
    list.dispatch(&module, root.gpu, [cases.len().div_ceil(64) as u32, 1, 1]);
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
        fov_y: 0.8,
        focus: Some((eye, 0.05)),
    };
    // The raster draws the finest level; ray tracing holds the coarsest.
    let renderer = RendererDesc {
        raster_error_pixels: 0.01,
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
    let bias = render_paths(
        &mut bed,
        &paths(debug(LightingDebugView::RayOriginBias, 0)),
        1,
        1,
    );
    let bias = surface_mean(&bias)[0];
    let floor = 1e-3 * (1.0 + 6.0);
    assert!(
        bias > 2.0 * floor,
        "x1 offsets {bias} m on average: the traced cut is not coarser than the raster"
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
fn raygen_and_compute_entries_trace_the_same_light() {
    if !require(device().caps().ray_dispatch, "no ray-dispatch pipelines") {
        return;
    }
    let mut bed = Bed::new(Analytic::SunAndGround);
    bed.set_pre_exposure(1.0 / 64.0);
    let settings = ViewLighting::default();
    let compute = Paths::new(&KernelSet::default(), PathEntry::Compute, settings);
    let rays = Paths::new(&KernelSet::default(), PathEntry::RayDispatch, settings);
    // Every frame redraws the light tiles, so runs differ by their noise; the raygen run must
    // sit as close to a compute run as another compute run does.
    let a = render_paths(&mut bed, &compute, 32, 1);
    let again = render_paths(&mut bed, &compute, 32, 1);
    let b = render_paths(&mut bed, &rays, 32, 1);
    assert_close(
        "raygen against compute",
        image_mean(&b),
        image_mean(&a),
        0.005,
    );
    let noise = again.flip(&a, 1.0 / 500.0);
    let e = b.flip(&a, 1.0 / 500.0);
    assert!(
        e.mean < 1.25 * noise.mean,
        "raygen against compute: {e:?}, compute against itself: {noise:?}"
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

/// The bed with `spec`, its panels' surface linked as `surfaces/context.slang`.
fn linked_bed(spec: Spec) -> Bed {
    Bed::with_renderer(
        spec,
        &RendererDesc {
            kernels: &linked::KERNELS,
            ..RendererDesc::default()
        },
    )
}

fn linked_paths(settings: ViewLighting) -> Paths {
    Paths::new(&linked::KERNELS, PathEntry::Fastest, settings)
}

#[test]
fn a_linked_surface_looks_the_same_in_a_mirror() {
    let panel_facing_back = Mat3::from_rotation_y(std::f32::consts::PI);
    let direct = Spec {
        copies: vec![at(PANEL, Mat3::IDENTITY, Vec3::new(0.0, 1.0, 0.0))],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 1.0, 3.0),
        target: Vec3::new(0.0, 1.0, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let mirrored = Spec {
        copies: vec![
            at(MIRROR, Mat3::IDENTITY, Vec3::new(0.0, 1.0, 0.0)),
            at(PANEL, panel_facing_back, Vec3::new(0.0, 1.0, 6.0)),
        ],
        ..direct.clone()
    };
    let color = |spec| {
        let mut bed = linked_bed(spec);
        let p = linked_paths(reference(2));
        let image = render_paths(&mut bed, &p, 4, 1);
        let m = mean_where(&image, |p| luminance(p) > 1.0);
        let total = m[0] + m[1] + m[2];
        m.map(|c| c / total)
    };
    let seen = color(direct);
    let reflected = color(mirrored);
    assert!(
        seen[0] > 0.7,
        "the panel seen directly is not red: {seen:?}"
    );
    assert_close("panel in the mirror", reflected, seen, 0.02);
    assert_clean();
}

#[test]
fn a_linked_surface_switches_form_only_from_the_second_bounce() {
    let spec = Spec {
        copies: vec![
            at(FLOOR, Mat3::IDENTITY, Vec3::ZERO),
            at(PANEL, Mat3::IDENTITY, Vec3::new(0.0, 0.8, -1.0)),
            at(WHITE_SPHERE, Mat3::IDENTITY, Vec3::new(1.6, 1.0, 0.0)),
        ],
        lights: Vec::new(),
        environment: None,
        eye: Vec3::new(0.0, 2.0, 4.0),
        target: Vec3::new(0.0, 0.5, 0.0),
        fov_y: 0.8,
        focus: None,
    };
    let mut bed = linked_bed(spec);
    let mut light = |view| {
        let p = linked_paths(debug(view, 3));
        surface_mean(&render_paths(&mut bed, &p, 32, 1))
    };
    let direct = light(LightingDebugView::DirectOnly);
    let indirect = light(LightingDebugView::IndirectOnly);
    assert!(
        direct[1] < 0.15 * direct[0],
        "light from the first bounce holds the far form: {direct:?}"
    );
    assert!(
        indirect[0] < 0.15 * indirect[1],
        "light from later bounces holds the near form: {indirect:?}"
    );
    assert_clean();
}
