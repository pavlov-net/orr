//! Caustic photons: the grid's irradiance under a slab against Fresnel and Beer's law, under a
//! ball lens against its 1D map from impact parameter to radius, and the reference's photons
//! against BSDF rays alone, by Welch's test, through a glass sphere from a light photons own and
//! from the environment, which they do not.

use lighting_suite::assert_clean;
use lighting_suite::bed::{
    Accumulated, Bed, EMITTER, GLASS_SPHERE, Item, Paths, SKY, SUN, Spec, TINTED_SLAB, WHITE_FLOOR,
    WHITE_FLOOR_ALBEDO, fresnel, pixel_of, reference,
};
use lighting_suite::welch::{assert_mean, assert_same_mean};
use orr_gpu::KernelSet;
use orr_lighting::{
    DirectionalLight, DiscProfile, Light, LightingDebugView, PathEntry, ViewLighting,
};
use orr_math::{DVec2, Mat3, Vec2, Vec3};

fn paths(settings: ViewLighting) -> Paths {
    Paths::new(&KernelSet::default(), PathEntry::Compute, settings)
}

fn render_paths(bed: &mut Bed, paths: &Paths, frames: u32, seed: u32) -> Accumulated {
    bed.accumulate(frames, seed, &mut |list, f| paths.record(list, f))
}

fn at(model: &'static str, at: Vec3) -> Item {
    Item {
        model,
        linear: Mat3::IDENTITY,
        at,
    }
}

/// What the first vertex gathers of the grid, in real time.
fn gathered() -> ViewLighting {
    ViewLighting {
        debug_view: LightingDebugView::Caustics,
        ..ViewLighting::default()
    }
}

/// The first vertex's light in real time: its next-event estimation, what its BSDF ray finds and
/// what it gathers of the grid.
fn first_vertex() -> ViewLighting {
    ViewLighting {
        max_bounces: 0,
        ..ViewLighting::default()
    }
}

/// The white floor's irradiance where it shows radiance `radiance` by its diffuse albedo.
fn irradiance(radiance: f64) -> f64 {
    radiance * std::f64::consts::PI / f64::from(WHITE_FLOOR_ALBEDO)
}

/// Sunlight through the tinted slab reaches the floor below it through both faces' Fresnel and
/// Beer's law along the refracted path, by next-event estimation through the slab as a plane
/// stack, and through the internal reflections between the faces, which the grid holds, on a
/// level floor and on one tilted 30 degrees toward the sun, whose plane crosses the grid's
/// voxels. The patch lies under a point of the slab past the bands where the faces facing away
/// from the sun fold light onto the floor by total internal reflection.
#[test]
fn a_slab_passes_the_sun_to_the_floor_by_fresnel_and_beer_law() {
    let sun = SUN.direction_to_light.normalize();
    let height = 1.0f32;
    let mu = f64::from(sun.y);
    let mu_t = (1.0 - (1.0 - mu * mu) / (1.5 * 1.5)).sqrt();
    let f = fresnel(1.5, mu);
    let t = 0.5f64.powf(1.0 / mu_t);
    let through = (1.0 - f) * (1.0 - f) * t / (1.0 - f * f * t * t);
    for tilt in [0.0f32, 30.0] {
        let linear = Mat3::from_rotation_x(tilt.to_radians());
        let normal = linear * Vec3::Y;
        let spec = Spec {
            copies: vec![
                Item {
                    model: WHITE_FLOOR,
                    linear,
                    at: Vec3::ZERO,
                },
                at(TINTED_SLAB, Vec3::new(0.0, height, 0.0)),
            ],
            lights: vec![Light::Directional(SUN)],
            environment: None,
            eye: Vec3::new(0.0, 4.0, 0.01),
            target: Vec3::ZERO,
            fov_y: 0.6,
            focus: None,
        };
        // Where the sun through `through_at`, in the slab's middle plane, meets the floor.
        let cosine = normal.dot(sun);
        let through_at = Vec3::new(0.4, height, 0.27);
        let under = through_at - sun * (normal.dot(through_at) / cosine);
        let patch = pixel_of(&spec, under);
        let mut bed = Bed::with(spec);
        let image = render_paths(&mut bed, &paths(first_vertex()), 32, 1);
        let got = image.around(patch, 4);
        for (c, &radiance) in got.iter().enumerate() {
            let wanted = f64::from(SUN.illuminance[c]) * f64::from(cosine) * through;
            let e = irradiance(radiance);
            assert!(
                (e - wanted).abs() <= 0.05 * wanted,
                "tilt {tilt}, channel {c}: {e} lux under the slab, wanted {wanted}"
            );
        }
        let stats = bed.lighting.stats(bed.last_frame()).unwrap().caustics;
        assert!(stats.deposits > 0, "tilt {tilt}: no photon deposited");
        // At any load a few new keys find every slot of their probe full: such a cell sits out
        // the frame, and reads take the level beside it.
        assert!(
            stats.overflows * 10_000 <= stats.deposits,
            "tilt {tilt}: {} of {} deposits overflowed the grid",
            stats.overflows,
            stats.deposits
        );
    }
    assert_clean();
}

/// `d` (unit) refracted through a surface of unit normal `n` facing it, from IOR `from` to `to`;
/// `None` past the critical angle.
fn refract(d: DVec2, n: DVec2, from: f64, to: f64) -> Option<DVec2> {
    let eta = from / to;
    let c = -n.dot(d);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    (k >= 0.0).then(|| eta * d + (eta * c - k.sqrt()) * n)
}

/// Where a vertical ray `b` m from the axis of a unit ball lens of IOR 1.5 centered `height` m
/// over the floor meets the floor after its two refractions, as m from the axis, and the share
/// of its light the two faces let through.
fn ball_lens(b: f64, height: f64) -> Option<(f64, f64)> {
    let center = DVec2::new(0.0, height);
    let d0 = DVec2::new(0.0, -1.0);
    let p1 = DVec2::new(b, height + (1.0 - b * b).sqrt());
    let n1 = p1 - center;
    let d1 = refract(d0, n1, 1.0, 1.5)?;
    // The chord's other end.
    let p2 = p1 - 2.0 * d1.dot(n1) * d1;
    let n2 = p2 - center;
    let d2 = refract(d1, -n2, 1.5, 1.0)?;
    if d2.y >= 0.0 {
        return None;
    }
    let floor = p2 + d2 * (-p2.y / d2.y);
    let f = fresnel(1.5, -d0.dot(n1));
    Some((floor.x.abs(), (1.0 - f) * (1.0 - f)))
}

/// A ball lens under a sun straight overhead lights the floor below it by its 1D map from impact
/// parameter to radius: the grid's irradiance in rings where that map is smooth matches the flux
/// the map sends into each ring over its area, within 10%.
#[test]
fn a_ball_lens_lights_the_floor_by_its_impact_parameter_map() {
    const HEIGHT: f64 = 2.2;
    const ILLUMINANCE: f32 = 100_000.0;
    let sun = DirectionalLight {
        direction_to_light: Vec3::Y,
        illuminance: Vec3::splat(ILLUMINANCE),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    };
    let spec = Spec {
        copies: vec![
            at(WHITE_FLOOR, Vec3::ZERO),
            at(GLASS_SPHERE, Vec3::new(0.0, HEIGHT as f32, 0.0)),
        ],
        lights: vec![Light::Directional(sun)],
        environment: None,
        eye: Vec3::new(0.0, 7.0, 0.01),
        target: Vec3::ZERO,
        fov_y: 0.45,
        focus: None,
    };
    // The map's flux per ring of 1 cm, over the ring's area.
    const RING: f64 = 0.01;
    const RINGS: usize = 150;
    let mut flux = vec![0.0f64; RINGS];
    let steps = 2_000_000u32;
    for k in 0..steps {
        let b = (f64::from(k) + 0.5) / f64::from(steps);
        if let Some((r, t)) = ball_lens(b, HEIGHT) {
            let ring = (r / RING) as usize;
            if ring < RINGS {
                flux[ring] += t * 2.0 * std::f64::consts::PI * b / f64::from(steps);
            }
        }
    }
    let wanted: Vec<f64> = flux
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let (r0, r1) = (i as f64 * RING, (i + 1) as f64 * RING);
            f64::from(ILLUMINANCE) * f / (std::f64::consts::PI * (r1 * r1 - r0 * r0))
        })
        .collect();
    let mut bed = Bed::with(spec.clone());
    let image = render_paths(&mut bed, &paths(gathered()), 48, 1);
    let mut compared = 0;
    for ring in (20..RINGS - 10).step_by(10) {
        // A ring the map changes little across, so the grid's cells blur it little.
        let near = &wanted[ring - 3..=ring + 3];
        let (lo, hi) = near.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &e| {
            (lo.min(e), hi.max(e))
        });
        if lo <= 0.0 || hi > 1.2 * lo {
            continue;
        }
        let r = (ring as f64 + 0.5) * RING;
        let samples: Vec<f64> = (0..16u32)
            .map(|k| {
                let a = f64::from(k) / 16.0 * std::f64::consts::TAU;
                let p = Vec3::new((r * a.cos()) as f32, 0.0, (r * a.sin()) as f32);
                irradiance(image.around(pixel_of(&spec, p), 1)[1])
            })
            .collect();
        assert_mean(&format!("ring at {r:.2} m"), &samples, wanted[ring], 0.1);
        compared += 1;
    }
    assert!(compared >= 3, "{compared} smooth rings compared");
    assert_clean();
}

/// Per-render means of green over the 12 pixel square around `at`, each of `frames` frames from
/// its own seed.
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

/// The floor under a glass sphere, seen beside it, takes the same light from the reference
/// with its photons as from BSDF rays alone, by Welch's test: from an emitter above, a light
/// photons own and BSDF rays through the sphere then leave to them, and from the sky, which
/// stays with BSDF rays.
#[test]
fn photons_and_bsdf_rays_carry_the_same_light_through_a_glass_sphere() {
    let sphere = at(GLASS_SPHERE, Vec3::new(0.0, 1.3, 0.0));
    let floor = at(WHITE_FLOOR, Vec3::ZERO);
    for (what, copies, environment) in [
        (
            "owned emitter",
            vec![floor, sphere, at(EMITTER, Vec3::new(0.0, 3.0, 0.0))],
            None,
        ),
        ("sky, not owned", vec![floor, sphere], Some(SKY)),
    ] {
        let spec = Spec {
            copies,
            lights: Vec::new(),
            environment,
            eye: Vec3::new(0.0, 1.0, 4.0),
            target: Vec3::new(0.0, 0.2, 0.0),
            fov_y: 0.9,
            focus: None,
        };
        let under = pixel_of(&spec, Vec3::ZERO);
        let beside = pixel_of(&spec, Vec3::new(0.5, 0.0, 0.4));
        let mut bed = Bed::with(spec);
        let full_paths = paths(reference(2));
        let bsdf_paths = paths(ViewLighting {
            debug_view: LightingDebugView::BsdfOnly,
            ..reference(2)
        });
        for (where_, patch) in [("under", under), ("beside", beside)] {
            let full = patch_means(&mut bed, &full_paths, patch, 16, 8, 1);
            let bsdf = patch_means(&mut bed, &bsdf_paths, patch, 16, 32, 2);
            assert!(
                full.iter().sum::<f64>() > 0.0,
                "{what}, {where_}: nothing lit"
            );
            assert_same_mean(
                &format!("{what}, {where_}: full against BSDF"),
                &full,
                &bsdf,
            );
        }
    }
    assert_clean();
}
