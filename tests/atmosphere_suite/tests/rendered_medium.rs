//! `AtmosphereMedium` linked into orr's kernels and rendered: lighting's aerial-perspective LUT
//! on surfaces and path segments in a mirror, against the f64 reference's march of the same rays
//! under lights at 5 and 30 deg, height fog off and on, within 5%. Both sides are single
//! scattering. A 1 m `near_range` leaves the camera segment to the LUT; the froxels would add the
//! sky's light, which single scattering does not have.

use std::f32::consts::FRAC_PI_2;
use std::sync::Mutex;

use atmosphere_suite::linked::KERNELS;
use atmosphere_suite::{dir, earth, foggy};
use lighting_suite::bed::{
    BLACK_SPHERE, Bed, EXTENT, Item, MIRROR, Paths, Spec, WHITE_SPHERE, look, pixel_of,
};
use lighting_suite::{assert_clean, device};
use orr_atmosphere::presets::solar_disc;
use orr_atmosphere::reference::{
    Channels, Geometry, Light as ReferenceLight, Precision, Reference, Run,
};
use orr_atmosphere::spectrum::SOLAR_ILLUMINANCE;
use orr_atmosphere::{Atmosphere, AtmosphereState};
use orr_lighting::{Light, LightId, LightingDebugView, PathEntry, ViewLighting};
use orr_math::{DVec3, Mat3, Quat, Vec2, Vec3};
use orr_scene::RendererDesc;

const SUN: LightId = LightId(0);
const EYE: Vec3 = Vec3::new(0.0, 2.0, 0.0);
/// A little above the horizon, so the direct view of every surface stays above the ground.
const VIEW: Vec3 = Vec3::new(0.0, 0.03, -1.0);
const FOV_Y: f32 = 0.8;
/// Each sphere's radius over its distance.
const RADIUS: f32 = 0.06;
/// Keeps the in-scatter near the light under the radiance target's ceiling.
const PRE_EXPOSURE: f32 = 0.01;
const NEAR_RANGE: f32 = 1.0;
/// Light elevation and azimuth from +x toward +z, in degrees, with the view along -z.
const LIGHTS: [(f64, f64); 3] = [(5.0, -65.0), (5.0, 60.0), (30.0, 30.0)];
/// Half the side of the mirror on the ground.
const MIRROR_HALF: f32 = 200.0;

const PRECISION: Precision = Precision {
    relative_error: 0.005,
    max_samples: 0,
};

/// The kernels, compiled once for all tests in the binary.
static PATHS: Mutex<Option<Paths>> = Mutex::new(None);

/// A bed with `copies` seen through `atmosphere`, its LUTs recorded every frame.
fn bed(atmosphere: &Atmosphere, copies: Vec<Item>) -> Bed {
    let spec = Spec {
        copies,
        lights: vec![Light::Directional(solar_disc(Vec3::Y))],
        environment: None,
        eye: EYE,
        target: EYE + VIEW.normalize() * 10.0,
        fov_y: FOV_Y,
        focus: Some((EYE, 1000.0)),
    };
    let renderer = RendererDesc {
        kernels: &KERNELS,
        ..RendererDesc::default()
    };
    let mut bed = Bed::with_rt_range(spec, &renderer, 200_000.0);
    bed.set_near_range(NEAR_RANGE);
    let mut sky = AtmosphereState::new(device(), atmosphere, 0).unwrap();
    sky.set_lights(&[SUN]);
    sky.set_multiple_scattering(false);
    bed.set_medium(sky.data());
    bed.set_environment(sky.data());
    bed.set_scene_compute(Box::new(move |cx| {
        sky.skip_fades();
        sky.record(cx, &[]).unwrap();
    }));
    bed.set_pre_exposure(PRE_EXPOSURE);
    bed
}

/// The unit ray through image point `p`, in pixels from the top left.
fn pixel_ray(spec: &Spec, p: Vec2) -> DVec3 {
    let t = (0.5 * spec.fov_y).tan();
    let aspect = EXTENT[0] as f32 / EXTENT[1] as f32;
    let ndc = Vec2::new(
        2.0 * p.x / EXTENT[0] as f32 - 1.0,
        1.0 - 2.0 * p.y / EXTENT[1] as f32,
    );
    (look(spec.eye, spec.target) * Vec3::new(ndc.x * t * aspect, ndc.y * t, -1.0))
        .normalize()
        .as_dvec3()
}

/// Distance from the point `o` along unit `ray` to the sphere of `center` and `radius`.
fn sphere_hit(o: DVec3, ray: DVec3, center: Vec3, radius: f32) -> f64 {
    let to_center = center.as_dvec3() - o;
    let along = to_center.dot(ray);
    let miss = to_center.length_squared() - along * along;
    along - (f64::from(radius) * f64::from(radius) - miss).sqrt()
}

/// In the reference's Rec.2020 nits under a solar disc toward `sun`, the in-scatter and
/// transmittance over the segment.
fn segment(run: &Run<'_>, o: DVec3, ray: DVec3, t: f64, sun: DVec3) -> ([f64; 3], [f64; 3]) {
    let source = ReferenceLight::solar(sun);
    let s = run.segment(o, ray, t, &source, Geometry::Plane, &PRECISION);
    let inscatter = run
        .rec2020(&s.inscatter.values)
        .map(|v| v * SOLAR_ILLUMINANCE);
    (inscatter, [0, 1, 2].map(|c| s.transmittance[c]))
}

/// Renders `bed` under each of [`LIGHTS`] with the given debug view and holds the 2 x 2 pixels
/// around each of `points` within 5% of `want`'s mean over the upper and lower pixel rows at the
/// point: `want` takes the point's index, the row's unit ray and the light's direction.
fn check(
    bed: &mut Bed,
    settings: ViewLighting,
    points: &[Vec2],
    want: &dyn Fn(usize, DVec3, DVec3) -> [f64; 3],
) {
    let mut paths = PATHS.lock().unwrap();
    let paths = paths
        .get_or_insert_with(|| Paths::new(&KERNELS, PathEntry::Fastest, ViewLighting::default()));
    paths.settings = settings;
    for (elevation, azimuth) in LIGHTS {
        let sun = dir(elevation, azimuth);
        bed.set_light(SUN, Some(Light::Directional(solar_disc(sun.as_vec3()))));
        let image = bed.accumulate(8, 3, &mut |list, f| paths.record(list, f));
        for (i, &at) in points.iter().enumerate() {
            let at = at.round();
            let got = image.around(at, 1).map(|c| c / f64::from(PRE_EXPOSURE));
            let mut mean = [0.0; 3];
            for side in [-0.5, 0.5] {
                let ray = pixel_ray(&bed.spec, at + Vec2::new(0.0, side));
                let w = want(i, ray, sun);
                for c in 0..3 {
                    mean[c] += 0.5 * w[c];
                }
            }
            for c in 0..3 {
                assert!(
                    (got[c] - mean[c]).abs() < 0.05 * mean[c],
                    "light {elevation} az {azimuth}, point {i}, channel {c}: {} against {}",
                    got[c],
                    mean[c]
                );
            }
        }
    }
    assert_clean();
}

/// Surfaces 1, 5 and 30 km away through the aerial-perspective LUT.
#[test]
fn surfaces_match_the_reference() {
    // The distance of each near face and its angle from the view axis.
    let surfaces: [(f32, f32); 3] = [(1_000.0, -0.35), (5_000.0, 0.0), (30_000.0, 0.35)];
    let center = |(distance, angle): (f32, f32)| {
        EYE + Quat::from_rotation_y(-angle) * VIEW.normalize() * distance * (1.0 + RADIUS)
    };
    for atmosphere in [earth(), foggy()] {
        let reference = Reference::new(&atmosphere);
        let run = reference.run(Channels::Rgb);
        let copies = surfaces
            .iter()
            .map(|&s| Item {
                model: WHITE_SPHERE,
                linear: Mat3::from_diagonal(Vec3::splat(RADIUS * s.0)),
                at: center(s),
            })
            .collect();
        let mut bed = bed(&atmosphere, copies);
        let points = surfaces.map(|s| pixel_of(&bed.spec, center(s)));
        let settings = ViewLighting {
            debug_view: LightingDebugView::Media,
            ..ViewLighting::default()
        };
        let eye = EYE.as_dvec3();
        check(&mut bed, settings, &points, &|i, ray, sun| {
            let s = surfaces[i];
            let t = sphere_hit(eye, ray, center(s), RADIUS * s.0);
            segment(&run, eye, ray, t, sun).0
        });
    }
}

/// A mirror on the ground reflecting black surfaces 5 and 20 km away: in the mirror, the medium
/// from the eye to the mirror and on to the surface.
#[test]
fn mirror_shows_the_medium_on_long_segments() {
    // Distance, elevation from the ground and azimuth from the view axis, in radians.
    let surfaces: [(f32, f32, f32); 3] = [
        (5_000.0, 0.035, -0.3),
        (20_000.0, 0.035, 0.3),
        (5_000.0, 0.087, 0.0),
    ];
    let center = |(d, e, a): (f32, f32, f32)| {
        Vec3::new(d * e.cos() * a.sin(), d * e.sin(), -d * e.cos() * a.cos())
    };
    for atmosphere in [earth(), foggy()] {
        let reference = Reference::new(&atmosphere);
        let run = reference.run(Channels::Rgb);
        let mut copies: Vec<Item> = surfaces
            .iter()
            .map(|&s| Item {
                model: BLACK_SPHERE,
                linear: Mat3::from_diagonal(Vec3::splat(RADIUS * s.0)),
                at: center(s),
            })
            .collect();
        // The model faces +z.
        copies.push(Item {
            model: MIRROR,
            linear: Mat3::from_rotation_x(-FRAC_PI_2) * MIRROR_HALF,
            at: Vec3::new(0.0, 0.0, -0.75 * MIRROR_HALF),
        });
        let mut bed = bed(&atmosphere, copies);
        let points = surfaces.map(|s| {
            let image = center(s) * Vec3::new(1.0, -1.0, 1.0);
            pixel_of(&bed.spec, image)
        });
        let eye = EYE.as_dvec3();
        check(
            &mut bed,
            ViewLighting::default(),
            &points,
            &|i, ray, sun| {
                let s = surfaces[i];
                let to_mirror = eye.y / -ray.y;
                let hit = eye + ray * to_mirror;
                let reflected = ray * DVec3::new(1.0, -1.0, 1.0);
                let t = sphere_hit(hit, reflected, center(s), RADIUS * s.0);
                let (near, through) = segment(&run, eye, ray, to_mirror, sun);
                let (far, _) = segment(&run, hit, reflected, t, sun);
                [0, 1, 2].map(|c| near[c] + through[c] * far[c])
            },
        );
    }
}
