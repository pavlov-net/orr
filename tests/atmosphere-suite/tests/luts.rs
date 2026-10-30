#![allow(clippy::print_stderr)]

//! The LUT passes against the f64 reference in the same wavelength groups.

use atmosphere_suite::{Probe, Rig, dir, probe, solar_rec2020};
use glam::{DVec3, Vec3};
use orr_atmosphere::colour::luminance;
use orr_atmosphere::reference::{Channels, Geometry, Light, Precision, Reference};
use orr_atmosphere::spectrum::{SOLAR_ILLUMINANCE, SPECTRAL_GROUPS};
use orr_atmosphere::{Atmosphere, HeightFog, SKY_VIEW_SIZE, TRANSMITTANCE_SIZE};
use orr_scene::testing::assert_clean;

const EYE: DVec3 = DVec3::new(0.0, 2.0, 0.0);
/// `SceneRoot::near_range` by default: the sky-view LUTs start there.
const NEAR_RANGE: f64 = 256.0;
const PASSES: u32 = SPECTRAL_GROUPS as u32 / 3;

fn earth() -> Atmosphere {
    Atmosphere::earth(DVec3::ZERO, Vec3::Y)
}

fn foggy() -> Atmosphere {
    Atmosphere {
        height_fog: Some(HeightFog {
            density: 1e-4,
            falloff_height: 300.0,
            base_height: 0.0,
            albedo: Vec3::splat(0.9),
            droplet_diameter: 10.0,
        }),
        ..earth()
    }
}

#[test]
fn transmittance_lut_matches_reference() {
    let atmosphere = earth();
    let mut rig = Rig::new(&atmosphere);
    rig.frame(Some(dir(30.0, 0.0)), EYE);
    let block = rig.block();
    let n = TRANSMITTANCE_SIZE[0] * TRANSMITTANCE_SIZE[1];
    let all = Probe {
        threads: n,
        ..Probe::default()
    };
    let points = probe(block, "transmittance_texels", n.into(), all);
    let layers: Vec<_> = (0..PASSES)
        .map(|layer| probe(block, "transmittance_lut", n.into(), Probe { layer, ..all }))
        .collect();
    let reference = Reference::new(&atmosphere);
    let groups = reference.run(Channels::Groups);
    let (mut worst, mut worst_at) = (0.0f64, (0.0, 0.0));
    for y in (0..TRANSMITTANCE_SIZE[1]).step_by(3) {
        for x in (0..TRANSMITTANCE_SIZE[0]).step_by(3) {
            let i = (y * TRANSMITTANCE_SIZE[0] + x) as usize;
            let (h, mu) = (f64::from(points[i][0]), f64::from(points[i][1]));
            let d = DVec3::new((1.0 - mu * mu).max(0.0).sqrt(), mu, 0.0);
            let expected =
                groups.transmittance(DVec3::new(0.0, h, 0.0), d, f64::INFINITY, Geometry::Sphere);
            // The ray from the ground along its tangent plane: the reference counts it as
            // reaching the ground.
            if expected.iter().all(|&e| e == 0.0) {
                continue;
            }
            for (k, e) in expected.iter().enumerate() {
                let got = f64::from(layers[k / 3][i][k % 3]);
                // Relative where the light is not gone; a tiny absolute floor below that.
                let error = (got - e).abs() / e.max(1e-3);
                if error > worst {
                    worst = error;
                    worst_at = (h, mu);
                }
            }
        }
    }
    eprintln!(
        "transmittance: worst {:.3}% at {:.0} m, mu {:.5}",
        worst * 100.0,
        worst_at.0,
        worst_at.1
    );
    assert!(worst < 0.02, "transmittance off by {:.2}%", worst * 100.0);
    assert_clean();
}

/// Texel directions to compare: (elevation, azimuth from the light) in degrees. The last
/// two look at the virtual ground.
const SKY_DIRECTIONS: [(f64, f64); 9] = [
    (90.0, 0.0),
    (60.0, 90.0),
    (30.0, 0.0),
    (20.0, 135.0),
    (5.0, 0.0),
    (5.0, 90.0),
    (5.0, 180.0),
    (-5.0, 0.0),
    (-30.0, 135.0),
];

/// The worst per-channel error of the sky-view LUT against the reference over
/// `SKY_DIRECTIONS` at light elevations -12, -5, 5, 30 and 60 deg, height fog off and on.
/// `multiple`: with multiple scattering, else single scattering on both sides.
fn sky_view_error(multiple: bool, precision: &Precision) -> f64 {
    let n = SKY_VIEW_SIZE[0] * SKY_VIEW_SIZE[1];
    let mut worst = 0.0f64;
    for atmosphere in [earth(), foggy()] {
        let fog = atmosphere.height_fog.is_some();
        let mut rig = Rig::new(&atmosphere);
        rig.sky.set_multiple_scattering(multiple);
        let reference = Reference::new(&atmosphere);
        let groups = reference.run(Channels::Groups);
        for elevation in [-12.0, -5.0, 5.0, 30.0, 60.0] {
            let sun = dir(elevation, 0.0);
            rig.frame(Some(sun), EYE);
            let all = Probe {
                threads: n,
                light: [0.0, 0.0, 0.0, EYE.y as f32],
                ..Probe::default()
            };
            let texels = probe(rig.block(), "sky_view_texels", n.into(), all);
            let lut = probe(rig.block(), "sky_view_lut", n.into(), all);
            let direction = |i: usize| {
                let t = texels[i];
                DVec3::new(t[0].into(), t[1].into(), t[2].into()).normalize()
            };
            for (e, a) in SKY_DIRECTIONS {
                let target = dir(e, a);
                let i = (0..texels.len())
                    .max_by(|&i, &j| {
                        direction(i)
                            .dot(target)
                            .total_cmp(&direction(j).dot(target))
                    })
                    .unwrap();
                let r = groups.sky(EYE, direction(i), &Light::solar(sun), NEAR_RANGE, precision);
                let expected = groups.rec2020(&r.values).map(|v| v * SOLAR_ILLUMINANCE);
                let got = [0, 1, 2].map(|c| f64::from(lut[i][c]));
                let y = luminance(expected);
                // Per channel, relative to the channel or, for a channel near zero, to the
                // luminance; a view the light does not reach must stay dark.
                let error = if y > 0.0 {
                    (0..3)
                        .map(|c| (got[c] - expected[c]).abs() / expected[c].abs().max(0.05 * y))
                        .fold(0.0, f64::max)
                } else if luminance(got) > 1e-9 {
                    1.0
                } else {
                    0.0
                };
                eprintln!(
                    "fog {fog:>5}, light {elevation:>5} deg, view ({e:>4}, {a:>4}): {y:.3e} cd/m^2, \
                     worst channel off by {:.2}%",
                    error * 100.0
                );
                worst = worst.max(error);
            }
        }
    }
    worst
}

/// Single scattering, the march, fog, phases, the fold and the ground's direct light.
#[test]
fn sky_view_single_scattering_matches_reference() {
    let only_single = Precision {
        relative_error: 1.0,
        max_samples: 0,
    };
    let worst = sky_view_error(false, &only_single);
    assert!(worst < 0.05, "sky radiance off by {:.2}%", worst * 100.0);
    assert_clean();
}

/// The full sky against the reference's multiple scattering: measures the error of the
/// multiple-scattering LUT model (atmosphere.md 6).
#[test]
#[ignore = "measurement, about 7 minutes: run with --ignored --nocapture"]
fn sky_view_multiple_scattering_error() {
    let precision = Precision {
        relative_error: 0.01,
        max_samples: 1 << 22,
    };
    let worst = sky_view_error(true, &precision);
    eprintln!("worst channel {:.1}%", worst * 100.0);
    assert_clean();
}

/// Sky irradiance on the ground plus the direct light stays below the light above the
/// atmosphere.
#[test]
fn ground_irradiance_below_top_of_atmosphere() {
    let colour = solar_rec2020().map(|c| c as f32);
    for atmosphere in [earth(), foggy()] {
        let mut rig = Rig::new(&atmosphere);
        for elevation in [2.0, 5.0, 30.0, 60.0, 90.0] {
            let sun = dir(elevation, 0.0);
            rig.frame(Some(sun), DVec3::ZERO);
            let out = probe(
                rig.block(),
                "irradiance",
                2,
                Probe {
                    threads: 64,
                    count: 1 << 16,
                    input: &[[0.0, 0.0, 0.0, 0.0]],
                    light: [sun.x as f32, sun.y as f32, sun.z as f32, 0.0],
                    colour: [colour[0], colour[1], colour[2], 0.0],
                    ..Probe::default()
                },
            );
            let lux = |c: [f32; 4]| luminance([c[0].into(), c[1].into(), c[2].into()]);
            let (sky, direct) = (lux(out[0]), lux(out[1]));
            let top = SOLAR_ILLUMINANCE * sun.y;
            eprintln!(
                "fog {}, light {elevation} deg: sky {sky:.0} lx + direct {direct:.0} lx of {top:.0} lx",
                atmosphere.height_fog.is_some()
            );
            assert!(sky > 0.0 && direct >= 0.0);
            assert!(sky + direct <= top, "{sky} + {direct} lx exceeds {top} lx");
        }
    }
    assert_clean();
}
