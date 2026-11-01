//! The subsurface walk against openpbr_ref's CPU walk: in analytic slabs through
//! `walk_probe.slang` (the walk alone, its cap tail and its numerics), and through the opaque
//! path on the bed's slabs and spheres (entry, exit and geometry).

use lighting_suite::bed::{
    Accumulated, Bed, EXTENT, FURNACE_RADIANCE, Item, Paths, Spec, WALK_FURNACE, WALK_SLABS,
    WALK_SPHERE, look,
};
use lighting_suite::shaders::walk_probe;
use lighting_suite::{assert_clean, device, pipeline, readback, render, run, upload};
use openpbr_ref::Inputs;
use openpbr_ref::walk::{Interface, Light as CpuLight, Medium, Shape, Walks, top_exit};
use orr_gpu::{Hazards, KernelSet, Stages};
use orr_lighting::{
    CapTailTable, DirectionalLight, DiscProfile, Light, LightingDebugView, PathEntry,
    ReferenceSettings, ViewLighting,
};
use orr_math::{DVec3, Mat3, Vec3};

/// Lanes per case of `walk_probe.slang`.
const LANES: u64 = 256;

/// One probe case: subsurface color, anisotropy, mean free path per channel (m), slab
/// thickness (m) and step cap.
#[derive(Clone, Copy, Debug)]
struct Case {
    color: f32,
    anisotropy: f32,
    radius: [f32; 3],
    thickness: f32,
    max_steps: u32,
}

impl Case {
    fn inputs(&self) -> Inputs {
        Inputs {
            subsurface_weight: 1.0,
            subsurface_color: [self.color; 3],
            subsurface_radius: 1.0,
            subsurface_radius_scale: self.radius,
            subsurface_scatter_anisotropy: self.anisotropy,
            ..Inputs::default()
        }
    }
}

/// Per case, the mean throughput leaving through the slab's top and the walks whose
/// throughput was not finite, over `walks` walks per lane.
fn probe(cases: &[Case], walks: u32) -> Vec<(DVec3, u32)> {
    let tail = CapTailTable::new(device()).unwrap();
    let module = pipeline(&walk_probe::MODULE, "main");
    let gpu_cases: Vec<walk_probe::WalkCase> = cases
        .iter()
        .map(|c| walk_probe::WalkCase {
            color: [c.color; 3],
            anisotropy: c.anisotropy,
            radius: c.radius,
            thickness: c.thickness,
            max_steps: c.max_steps,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        })
        .collect();
    let input = upload(&gpu_cases, "walk cases");
    let sums = readback::<[f32; 4]>(cases.len() as u64 * LANES, "walk sums");
    let root = upload(
        &[walk_probe::Root {
            cases: input.gpu(),
            sums: sums.gpu(),
            tail: tail.gpu(),
            walks,
            seed: 7,
        }],
        "walk probe root",
    );
    let mut list = render().command_list().unwrap();
    list.dispatch(
        &module,
        root.gpu(),
        [(LANES / 64) as u32, cases.len() as u32, 1],
    );
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);
    let s = sums.as_slice().unwrap();
    let total = f64::from(walks) * LANES as f64;
    (0..cases.len())
        .map(|i| {
            let lanes = &s[i * LANES as usize..(i + 1) * LANES as usize];
            let mean = lanes
                .iter()
                .map(|l| Vec3::from_slice(l).as_dvec3())
                .sum::<DVec3>()
                / total;
            (mean, lanes.iter().map(|l| l[3] as u32).sum())
        })
        .collect()
}

fn cpu_top(case: &Case, walks: usize) -> DVec3 {
    top_exit(
        f64::from(case.thickness),
        &Medium::of(&case.inputs()),
        DVec3::NEG_Z,
        walks,
        u32::MAX,
        11,
    )
}

fn relative(a: DVec3, b: DVec3) -> f64 {
    (a / b - 1.0).abs().max_element()
}

#[test]
fn walks_match_the_cpu_walk_from_a_quarter_to_16_mean_free_paths() {
    let cases: Vec<Case> = [0.0025f32, 0.01, 0.04, 0.16]
        .iter()
        .map(|&thickness| Case {
            color: 0.8,
            anisotropy: 0.0,
            radius: [0.01, 0.005, 0.0025],
            thickness,
            max_steps: 1024,
        })
        .collect();
    for (case, (gpu, bad)) in cases.iter().zip(probe(&cases, 256)) {
        let cpu = cpu_top(case, 400_000);
        let e = relative(gpu, cpu);
        assert!(
            bad == 0 && e < 0.02,
            "{case:?}: GPU {gpu:?}, CPU {cpu:?} ({e:.4}), {bad} not finite"
        );
    }
    assert_clean();
}

#[test]
fn the_cap_tail_keeps_what_capped_walks_would_carry_out() {
    let mut cases = Vec::new();
    for color in [0.8f32, 0.9] {
        for anisotropy in [0.0f32, 0.8] {
            for max_steps in [16, 32, 64] {
                cases.push(Case {
                    color,
                    anisotropy,
                    radius: [0.01; 3],
                    thickness: 1.0,
                    max_steps,
                });
            }
        }
    }
    for (case, (gpu, _)) in cases.iter().zip(probe(&cases, 256)) {
        let cpu = cpu_top(case, 400_000);
        let e = relative(gpu, cpu);
        assert!(
            e < 0.02,
            "{case:?}: GPU {gpu:?}, uncapped CPU {cpu:?} ({e:.4})"
        );
    }
    assert_clean();
}

#[test]
fn walks_stay_finite_and_within_energy() {
    let mut cases = Vec::new();
    for radius in [1e-6f32, 1e-3, 1.0, 10.0] {
        for anisotropy in [-0.999f32, 0.0, 0.999] {
            for color in [0.0f32, 1.0] {
                cases.push(Case {
                    color,
                    anisotropy,
                    radius: [radius, radius * 0.5, radius * 0.25],
                    thickness: 0.1,
                    max_steps: 16,
                });
            }
        }
    }
    for (case, (top, bad)) in cases.iter().zip(probe(&cases, 64)) {
        assert_eq!(bad, 0, "{case:?}: walks not finite");
        assert!(
            top.max_element() <= 1.02,
            "{case:?}: {top:?} leaves the top"
        );
    }
    assert_clean();
}

/// The CPU walk's inputs for the bed's walk models.
fn walk_inputs(specular_weight: f32, radius_scale: [f32; 3]) -> Inputs {
    Inputs {
        subsurface_weight: 1.0,
        subsurface_color: [0.8; 3],
        subsurface_radius: 0.01,
        subsurface_radius_scale: radius_scale,
        specular_roughness: 0.4,
        specular_ior: 1.4,
        specular_weight,
        ..Inputs::default()
    }
}

const SUN_DIRECTION: Vec3 = Vec3::new(0.5, 0.8, -0.3);
const SUN_LUX: f32 = 1000.0;

fn sun() -> DirectionalLight {
    DirectionalLight {
        direction_to_light: SUN_DIRECTION.normalize(),
        illuminance: Vec3::splat(SUN_LUX),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    }
}

/// The reference with at most one bounce after the first vertex, showing `debug_view`.
fn settings(debug_view: LightingDebugView) -> ViewLighting {
    ViewLighting {
        debug_view,
        sss_max_steps: 1024,
        max_bounces: 1,
        reference: Some(ReferenceSettings::default()),
        ..ViewLighting::default()
    }
}

/// `spec` through `paths`, accumulated over `frames`.
fn render_spec(paths: &Paths, spec: Spec, frames: u32) -> Accumulated {
    let mut bed = Bed::with(spec);
    bed.accumulate(frames, 5, &mut |list, f| paths.record(list, f))
}

#[test]
fn the_opaque_path_walks_as_the_cpu_walk_does() {
    let mut paths = Paths::new(
        &KernelSet::default(),
        PathEntry::Compute,
        settings(LightingDebugView::Subsurface),
    );
    lit_slabs_match_the_cpu_walk(&paths);
    a_lit_sphere_matches_the_cpu_walk(&paths);
    paths.settings = settings(LightingDebugView::None);
    paths.settings.sss_max_steps = 16;
    a_furnace_sphere_returns_the_subsurface_color(&paths);
    assert_clean();
}

/// The unit direction of pixel `(x, y)`'s center ray from a camera at `eye` looking at
/// `target` with vertical field of view `fov_y`.
fn pixel_ray(eye: Vec3, target: Vec3, fov_y: f32, x: f64, y: f64) -> Vec3 {
    let r = look(eye, target);
    let t = (fov_y * 0.5).tan();
    let aspect = EXTENT[0] as f32 / EXTENT[1] as f32;
    let u = (x as f32 / EXTENT[0] as f32) * 2.0 - 1.0;
    let v = 1.0 - (y as f32 / EXTENT[1] as f32) * 2.0;
    (r * Vec3::new(u * t * aspect, v * t, -1.0)).normalize()
}

/// World to the CPU walk's frame, whose +z is world +y.
fn local(v: Vec3) -> DVec3 {
    DVec3::new(f64::from(v.x), f64::from(-v.z), f64::from(v.y))
}

fn cpu_sun() -> CpuLight {
    CpuLight::Sun {
        direction: local(SUN_DIRECTION).normalize(),
        illuminance: DVec3::splat(f64::from(SUN_LUX)),
    }
}

fn mean_rgb(a: &Accumulated, pixels: impl Iterator<Item = usize>) -> DVec3 {
    let mut sum = DVec3::ZERO;
    let mut n = 0.0;
    for i in pixels {
        sum += Vec3::from_slice(&a.mean[i]).as_dvec3();
        n += 1.0;
    }
    sum / n
}

fn lit_slabs_match_the_cpu_walk(paths: &Paths) {
    let (eye, target, fov_y) = (Vec3::new(0.0, 1.0, 0.25), Vec3::ZERO, 0.6);
    let inputs = walk_inputs(1.0, [1.0, 0.5, 0.25]);
    let interface = Interface::new(&inputs, 20_000);
    // The window of pixels around the image's center that the comparison averages.
    let half = 16u32;
    let (cx, cy) = (EXTENT[0] / 2, EXTENT[1] / 2);
    for (model, thickness) in WALK_SLABS {
        let spec = Spec {
            copies: vec![Item {
                model,
                linear: Mat3::IDENTITY,
                at: Vec3::ZERO,
            }],
            lights: vec![Light::Directional(sun())],
            environment: None,
            eye,
            target,
            fov_y,
            focus: None,
        };
        let gpu = render_spec(paths, spec, 256);
        let window = (cy - half..cy + half)
            .flat_map(|y| (cx - half..cx + half).map(move |x| (y * EXTENT[0] + x) as usize));
        let gpu = mean_rgb(&gpu, window);
        let walks = Walks::new(
            &inputs,
            Shape::Slab {
                thickness: f64::from(thickness),
            },
            &interface,
        );
        let cpu = walks.mean(1_000_000, cpu_sun(), 13, |rng| {
            let x = f64::from(cx - half) + (rng.f64() * f64::from(2 * half)).floor() + 0.5;
            let y = f64::from(cy - half) + (rng.f64() * f64::from(2 * half)).floor() + 0.5;
            let d = pixel_ray(eye, target, fov_y, x, y);
            (DVec3::ZERO, DVec3::Z, local(-d))
        });
        let e = relative(gpu, cpu);
        assert!(e < 0.02, "{model}: GPU {gpu:?}, CPU {cpu:?} ({e:.4})");
    }
}

/// The pixels whose center ray from `eye` toward `target` meets the sphere of `radius` at the
/// origin, with the boundary point, its normal and the view there.
fn sphere_pixels(
    eye: Vec3,
    target: Vec3,
    fov_y: f32,
    radius: f32,
) -> Vec<(usize, DVec3, DVec3, DVec3)> {
    let mut out = Vec::new();
    let e = local(eye);
    for y in 0..EXTENT[1] {
        for x in 0..EXTENT[0] {
            let d = local(pixel_ray(
                eye,
                target,
                fov_y,
                f64::from(x) + 0.5,
                f64::from(y) + 0.5,
            ));
            let b = e.dot(d);
            let c = e.length_squared() - f64::from(radius * radius);
            let disc = b * b - c;
            // Pixels within a hair of the silhouette are left out: the mesh and the sphere
            // disagree there.
            if disc > 1e-3 * f64::from(radius * radius) {
                let t = -b - disc.sqrt();
                let p = e + d * t;
                out.push(((y * EXTENT[0] + x) as usize, p, p.normalize(), -d));
            }
        }
    }
    out
}

fn a_lit_sphere_matches_the_cpu_walk(paths: &Paths) {
    let radius = 0.04f32;
    let (eye, target, fov_y) = (Vec3::new(0.0, 0.05, 0.25), Vec3::ZERO, 0.4);
    let spec = Spec {
        copies: vec![Item {
            model: WALK_SPHERE,
            linear: Mat3::from_diagonal(Vec3::splat(radius)),
            at: Vec3::ZERO,
        }],
        lights: vec![Light::Directional(sun())],
        environment: None,
        eye,
        target,
        fov_y,
        focus: None,
    };
    let gpu = render_spec(paths, spec, 128);
    let pixels = sphere_pixels(eye, target, fov_y, radius);
    let gpu = mean_rgb(&gpu, pixels.iter().map(|p| p.0));
    let inputs = walk_inputs(1.0, [1.0, 0.5, 0.25]);
    let interface = Interface::new(&inputs, 20_000);
    let walks = Walks::new(
        &inputs,
        Shape::Sphere {
            radius: f64::from(radius),
        },
        &interface,
    );
    let cpu = walks.mean(2_000_000, cpu_sun(), 17, |rng| {
        let (_, p, n, v) = pixels[(rng.f64() * pixels.len() as f64) as usize];
        (p, n, v)
    });
    let e = relative(gpu, cpu);
    assert!(e < 0.02, "sphere: GPU {gpu:?}, CPU {cpu:?} ({e:.4})");
}

/// With the default cap, so the cap tail takes part.
fn a_furnace_sphere_returns_the_subsurface_color(paths: &Paths) {
    let (eye, target, fov_y) = (Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, 0.6);
    let spec = Spec {
        copies: vec![Item {
            model: WALK_FURNACE,
            linear: Mat3::IDENTITY,
            at: Vec3::ZERO,
        }],
        lights: Vec::new(),
        environment: Some(FURNACE_RADIANCE),
        eye,
        target,
        fov_y,
        focus: None,
    };
    let image = render_spec(paths, spec, 64);
    let pixels = sphere_pixels(eye, target, fov_y, 1.0);
    let mean = mean_rgb(&image, pixels.iter().map(|p| p.0));
    let wanted = f64::from(FURNACE_RADIANCE[0]) * 0.8;
    let e = relative(mean, DVec3::splat(wanted));
    assert!(
        e < 0.01,
        "the furnace sphere returns {mean:?}, wanted {wanted}"
    );
}
