//! `AtmosphereMedium` against the f64 reference on the same plane-altitude segments:
//! closed-form transmittance, a mirror's long reflection segments, and the step at the horizon
//! between flat ground and the sky just above it.

use atmosphere_suite::{Probe, Rig, dir, earth, foggy, probe, sky_radiance, solar_rec2020};
use orr_atmosphere::Atmosphere;
use orr_atmosphere::reference::{Channels, Geometry, Light, Precision, Reference};
use orr_atmosphere::spectrum::SOLAR_ILLUMINANCE;
use orr_math::DVec3;
use orr_scene::testing::assert_clean;

const EYE: DVec3 = DVec3::new(0.0, 2.0, 0.0);
/// `SceneRoot::near_range` by default.
const NEAR_RANGE: f64 = 256.0;

const SINGLE: Precision = Precision {
    relative_error: 0.005,
    max_samples: 0,
};

/// A segment: its origin, length and direction.
type Segment = (DVec3, f64, DVec3);

/// The pairs the probes read: origin and `t`, direction.
fn input(segments: &[Segment]) -> Vec<[f32; 4]> {
    segments
        .iter()
        .flat_map(|&(o, t, d)| {
            [
                [o.x as f32, o.y as f32, o.z as f32, t as f32],
                [d.x as f32, d.y as f32, d.z as f32, 0.0],
            ]
        })
        .collect()
}

fn rgb(c: [f32; 4]) -> [f64; 3] {
    [c[0], c[1], c[2]].map(f64::from)
}

/// `AtmosphereMedium.transmittance` over each segment.
fn transmittance(rig: &Rig, segments: &[Segment]) -> Vec<[f64; 3]> {
    let n = segments.len() as u32;
    probe(
        rig.block(),
        "medium_transmittance_along",
        n.into(),
        Probe {
            threads: n,
            count: n,
            input: &input(segments),
            ..Probe::default()
        },
    )
    .into_iter()
    .map(rgb)
    .collect()
}

/// `segment_inscatter` over each segment in lighting's 16 steps: its in-scatter, nits under
/// `solar_disc`, and transmittance.
fn segments(rig: &Rig, segments: &[Segment]) -> Vec<([f64; 3], [f64; 3])> {
    let n = segments.len() as u32;
    let lighting = rig.lighting.as_ref().expect("a rig with lighting");
    let out = probe(
        rig.block(),
        "medium_segment",
        2 * u64::from(n),
        Probe {
            threads: n,
            count: n,
            input: &input(segments),
            layer: 16,
            set: lighting.light_set().address().0,
            ..Probe::default()
        },
    );
    out.chunks(2).map(|c| (rgb(c[0]), rgb(c[1]))).collect()
}

/// The worst channel's relative difference.
fn worst(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3)
        .map(|c| (a[c] - b[c]).abs() / b[c].abs().max(1e-30))
        .fold(0.0, f64::max)
}

/// Segments up to 100 km, elevations 0 to 90 deg, start altitudes 0 to 5 km, with height fog:
/// the closed form within 1% relative or 1e-4 absolute of the f64 integral of the same bands.
#[test]
fn finite_transmittance_against_the_reference() {
    let atmosphere = foggy();
    let mut rig = Rig::new(&atmosphere);
    rig.frame(Some(dir(30.0, 0.0)), EYE);
    let reference = Reference::new(&atmosphere);
    let run = reference.run(Channels::Rgb);
    let mut cases = Vec::new();
    for altitude in [0.0, 2.0, 300.0, 1_000.0, 5_000.0] {
        for elevation in [0.0, 0.1, 0.5, 2.0, 5.0, 20.0, 45.0, 90.0] {
            for t in [10.0, 100.0, 1_000.0, 5_000.0, 20_000.0, 50_000.0, 100_000.0] {
                cases.push((
                    DVec3::new(0.0, altitude, 0.0),
                    t,
                    dir(elevation, 37.0).normalize(),
                ));
            }
        }
    }
    let got = transmittance(&rig, &cases);
    for (&(o, t, d), g) in cases.iter().zip(&got) {
        let want = run.transmittance(o, d, t, Geometry::Plane);
        for c in 0..3 {
            let error = (g[c] - want[c]).abs();
            assert!(
                error < 1e-4 || error < 0.01 * want[c],
                "at {} m, {t} m along {d}: channel {c} {} against {}",
                o.y,
                g[c],
                want[c]
            );
        }
    }
    assert_clean();
}

/// Toward a light: the transmittance LUT at the point's plane altitude; and transmittance stays
/// a number on segments far outside the atmosphere's layers (a medium with a zero coefficient,
/// eyes in space and points below the plane).
#[test]
fn transmittance_toward_lights_and_on_extreme_segments() {
    let atmosphere = foggy();
    let mut rig = Rig::new(&atmosphere);
    rig.frame(Some(dir(30.0, 0.0)), EYE);
    let reference = Reference::new(&atmosphere);
    let run = reference.run(Channels::Groups);

    // Fog stays in the reference's column table as in the LUT's closed form.
    let mut cases = Vec::new();
    for altitude in [2.0, 1_000.0, 5_000.0] {
        for elevation in [2.0, 10.0, 30.0, 90.0] {
            cases.push((
                DVec3::new(25_000.0, altitude, -9_000.0),
                f64::INFINITY,
                dir(elevation, 11.0),
            ));
        }
    }
    let got = transmittance(&rig, &cases);
    for (&(o, _, d), g) in cases.iter().zip(&got) {
        let origin = DVec3::new(0.0, o.y, 0.0);
        let values = run.transmittance(origin, d, f64::INFINITY, Geometry::Sphere);
        let sunlight = run.rec2020(&values);
        let solar = *orr_atmosphere::spectrum::SOLAR_COLOR;
        let want = [0, 1, 2].map(|c| sunlight[c] / solar[c]);
        let error = worst(*g, want);
        assert!(
            error < 0.02,
            "at {} m toward {d}: {g:?} against {want:?}",
            o.y
        );
    }

    let down = DVec3::new(0.0, -1.0, 0.0);
    let extreme = [
        // Space to far under the plane.
        (DVec3::new(0.0, 400_000.0, 0.0), 500_000.0, down),
        (DVec3::new(0.0, 400_000.0, 0.0), 4_000_000.0, down),
        // Below the plane, level and upward.
        (DVec3::new(0.0, -3_000.0, 0.0), 20_000.0, dir(0.0, 0.0)),
        (DVec3::new(0.0, -3_000.0, 0.0), 20_000.0, dir(10.0, 0.0)),
        // So far below it that the fog's density overflows at both ends.
        (DVec3::new(0.0, -40_000.0, 0.0), 20_000.0, dir(10.0, 0.0)),
        // Below the plane and above the top, toward a light.
        (
            DVec3::new(0.0, -3_000.0, 0.0),
            f64::INFINITY,
            dir(10.0, 0.0),
        ),
        (
            DVec3::new(0.0, 400_000.0, 0.0),
            f64::INFINITY,
            dir(-10.0, 0.0),
        ),
    ];
    let got = transmittance(&rig, &extreme);
    for (case, g) in extreme.iter().zip(&got) {
        assert!(
            g.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
            "{case:?}: {g:?}"
        );
    }
    assert!(got[1].iter().all(|&c| c < 1e-6), "{:?}", got[1]);
    assert_clean();
}

/// From a mirror on the ground up to terrain 5 and 20 km away, rising at 0.5 to 5 deg, lights at
/// 5, 30 and 60 deg ahead of and behind it, height fog off and on: lighting's march through the
/// medium within 5% of the reference's segment, single scattering on both sides.
#[test]
fn reflection_segments_match_the_reference() {
    for atmosphere in [earth(), foggy()] {
        check_reflection_segments(&atmosphere);
    }
    assert_clean();
}

fn check_reflection_segments(atmosphere: &Atmosphere) {
    let mut rig = Rig::with_lighting(atmosphere);
    rig.sky.set_multiple_scattering(false);
    let reference = Reference::new(atmosphere);
    let run = reference.run(Channels::Rgb);
    let mirror = DVec3::new(0.0, 0.0, -50.0);
    for (light, azimuth) in [
        (5.0, 20.0),
        (30.0, 20.0),
        (60.0, 20.0),
        (5.0, 150.0),
        (30.0, 150.0),
    ] {
        let sun = dir(light, azimuth);
        rig.frame(Some(sun), EYE);
        let mut cases = Vec::new();
        for t in [5_000.0, 20_000.0] {
            for elevation in [0.5, 2.0, 5.0] {
                cases.push((mirror, t, dir(elevation, 0.0)));
            }
        }
        for (&(o, t, d), (s, through)) in cases.iter().zip(&segments(&rig, &cases)) {
            let want = run.segment(o, d, t, &Light::solar(sun), Geometry::Plane, &SINGLE);
            let want_s = run
                .rec2020(&want.inscatter.values)
                .map(|v| v * SOLAR_ILLUMINANCE);
            let want_t = [0, 1, 2].map(|c| want.transmittance[c]);
            let (e, f) = (worst(*s, want_s), worst(*through, want_t));
            assert!(
                e < 0.05 && f < 0.05,
                "light {light} az {azimuth}, {t} m along {d}: in-scatter off by {e}, \
                 transmittance by {f}"
            );
        }
    }
}

/// Flat ground 32 km away and the sky pixel just above it: the ground's radiance through the
/// medium (plane altitude) against the medium to `near_range` and `AtmosphereSky` past it
/// (spherical). The sky beyond 32 km shows through the air, so the reference has a step here
/// too, up to 10% in red under a low light; the medium and the sky add at most 5 points to it,
/// height fog off and on, single scattering on both sides.
#[test]
fn flat_ground_meets_the_sky_at_the_horizon() {
    let distance = 32_000.0;
    let below = (EYE.y / distance).atan().to_degrees();
    for atmosphere in [earth(), foggy()] {
        let fog = atmosphere.height_fog.is_some();
        let mut rig = Rig::with_lighting(&atmosphere);
        rig.sky.set_multiple_scattering(false);
        let reference = Reference::new(&atmosphere);
        let (bands, groups) = (
            reference.run(Channels::Rgb),
            reference.run(Channels::Groups),
        );
        let albedo = f64::from(atmosphere.ground_albedo.x);
        let solar = solar_rec2020();
        for (light, azimuth) in [
            (5.0, 0.0),
            (5.0, 90.0),
            (5.0, 180.0),
            (30.0, 0.0),
            (30.0, 90.0),
            (60.0, 0.0),
        ] {
            let sun = dir(light, azimuth);
            rig.frame(Some(sun), EYE);
            let to_ground = dir(-below, 0.0);
            let above = dir(0.05, 0.0);
            let got = segments(
                &rig,
                &[(EYE, distance, to_ground), (EYE, NEAR_RANGE, above)],
            );
            let lit = transmittance(&rig, &[(EYE + to_ground * distance, f64::INFINITY, sun)])[0];
            let sky = sky_radiance(&rig, EYE, &[above])[0];
            let (ground_s, ground_t) = got[0];
            let (near_s, near_t) = got[1];
            let source = Light::solar(sun);
            let want = bands.segment(EYE, to_ground, distance, &source, Geometry::Plane, &SINGLE);
            let want_s = bands.rec2020(&want.inscatter.values);
            let want_sky = groups.rec2020(&groups.sky(EYE, above, &source, 0.0, &SINGLE).values);
            for c in 0..3 {
                let surface = albedo / std::f64::consts::PI * solar[c] * sun.y * lit[c];
                let ground = ground_s[c] + ground_t[c] * surface;
                let upper = near_s[c] + near_t[c] * sky[c];
                let step = (ground - upper) / upper;
                let want_ground = want_s[c] * SOLAR_ILLUMINANCE + want.transmittance[c] * surface;
                let want_upper = want_sky[c] * SOLAR_ILLUMINANCE;
                let want_step = (want_ground - want_upper) / want_upper;
                assert!(
                    (step - want_step).abs() < 0.05,
                    "fog {fog} light {light} az {azimuth} channel {c}: step {step}, reference \
                     {want_step}"
                );
            }
        }
    }
    assert_clean();
}
