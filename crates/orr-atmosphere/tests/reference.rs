#![allow(clippy::print_stderr)]

use glam::{DVec3, Vec3};
use orr_atmosphere::colour::{REC2020_TO_XYZ, delta_uv, mul};
use orr_atmosphere::reference::{Channels, Geometry, Light, Precision, Reference, Run};
use orr_atmosphere::spectrum::{SOLAR_COLOUR, SOLAR_ILLUMINANCE};
use orr_atmosphere::{Aerosol, Atmosphere, HeightFog};

fn earth() -> Atmosphere {
    Atmosphere::earth(DVec3::ZERO, Vec3::Y)
}

/// Unit direction at an elevation and an azimuth from +x toward +z, in degrees.
fn dir(elevation: f64, azimuth: f64) -> DVec3 {
    let (e, a) = (elevation.to_radians(), azimuth.to_radians());
    DVec3::new(e.cos() * a.cos(), e.sin(), e.cos() * a.sin())
}

#[test]
fn solar_illuminance_from_spectrum() {
    let error = (SOLAR_ILLUMINANCE - 132_600.0).abs() / 132_600.0;
    assert!(error < 1e-3, "{SOLAR_ILLUMINANCE} lx");
    let y = mul(&REC2020_TO_XYZ, SOLAR_COLOUR)[1];
    assert!((y - 1.0).abs() < 1e-6, "solar colour luminance {y}");
}

/// Uniform fog filling the atmosphere and nothing else.
fn slab(sigma: f32) -> Atmosphere {
    let mut a = earth();
    a.rayleigh.density = 0.0;
    a.aerosol.optical_depth_550 = 0.0;
    a.ozone.column = 0.0;
    a.height_fog = Some(HeightFog {
        density: sigma,
        falloff_height: f32::INFINITY,
        base_height: 0.0,
        albedo: Vec3::ONE,
        droplet_diameter: 10.0,
    });
    a
}

#[test]
fn homogeneous_slab_transmittance() {
    let sigma = 2e-5;
    let atmosphere = slab(sigma as f32);
    let reference = Reference::new(&atmosphere);
    let run = reference.run(Channels::Groups);
    let sigma = f64::from(sigma as f32);
    let (radius, top) = (
        f64::from(atmosphere.planet_radius),
        f64::from(atmosphere.top_radius),
    );
    let eye = DVec3::new(0.0, 1500.0, 0.0);
    for elevation in [90.0, 30.0, 2.0, 0.0, -0.5, -10.0] {
        let d = dir(elevation, 40.0);
        for t in [10.0, 3_000.0, 80_000.0, f64::INFINITY] {
            // Sphere: chord to the top or the ground.
            let q = eye + DVec3::Y * radius;
            let b = q.dot(d);
            let ground = b * b - (q.length_squared() - radius * radius);
            let hits_ground = b < 0.0 && ground >= 0.0;
            let exit = if hits_ground {
                -b - ground.sqrt()
            } else {
                -b + (b * b - (q.length_squared() - top * top)).sqrt()
            };
            let expected = if hits_ground && t > exit {
                0.0
            } else {
                (-sigma * t.min(exit)).exp()
            };
            for v in run.transmittance(eye, d, t, Geometry::Sphere) {
                assert!(
                    (v - expected).abs() <= 1e-9 * expected,
                    "sphere {elevation} deg, {t} m: {v} vs {expected}"
                );
            }
            // Plane: up to the top height or down to y = 0.
            let exit = if d.y > 0.0 {
                (top - radius - eye.y) / d.y
            } else if d.y < 0.0 {
                -eye.y / d.y
            } else {
                f64::INFINITY
            };
            let expected = if d.y < 0.0 && t > exit {
                0.0
            } else {
                (-sigma * t.min(exit)).exp()
            };
            for v in run.transmittance(eye, d, t, Geometry::Plane) {
                assert!(
                    (v - expected).abs() <= 1e-9 * expected + 1e-30,
                    "plane {elevation} deg, {t} m: {v} vs {expected}"
                );
            }
        }
    }
}

/// Zenith; 5 deg above the horizon toward, across and away from the light; 30 deg toward
/// it.
fn sky_directions() -> [DVec3; 5] {
    [
        DVec3::Y,
        dir(5.0, 0.0),
        dir(5.0, 90.0),
        dir(5.0, 180.0),
        dir(30.0, 0.0),
    ]
}

const EYE: DVec3 = DVec3::new(0.0, 2.0, 0.0);

#[test]
fn groups_match_spectral_colour() {
    let reference = Reference::new(&earth());
    let (spectral, groups) = (
        reference.run(Channels::Spectral),
        reference.run(Channels::Groups),
    );
    let elevations = [30.0, 10.0, 5.0, 2.0, 0.0, -2.0, -4.0, -6.0, -9.0, -12.0];
    let precision = Precision::default();
    let mut worst = 0.0f64;
    for elevation in elevations {
        let light = Light::solar(dir(elevation, 0.0));
        let sky: Vec<_> = sky_directions()
            .iter()
            .map(|&d| {
                (
                    spectral.sky(EYE, d, &light, 0.0, &precision),
                    groups.sky(EYE, d, &light, 0.0, &precision),
                )
            })
            .collect();
        let max = sky
            .iter()
            .map(|(s, g)| delta_uv(s.xyz, g.xyz))
            .fold(0.0, f64::max);
        let zenith = sky[0].0.xyz[1] * SOLAR_ILLUMINANCE;
        let samples = sky
            .iter()
            .map(|(s, g)| s.samples.max(g.samples))
            .max()
            .unwrap();
        let direct = |run: &Run| {
            run.xyz(&run.transmittance(EYE, light.direction, f64::INFINITY, Geometry::Sphere))
        };
        let (ds, dg) = (direct(&spectral), direct(&groups));
        let direct_duv = if ds[1] > 0.0 { delta_uv(ds, dg) } else { 0.0 };
        eprintln!(
            "light {elevation:>5} deg: sky max du'v' {max:.4}, direct {direct_duv:.4}, zenith {zenith:.4e} cd/m^2, {samples} paths"
        );
        assert!(
            direct_duv < 0.005,
            "direct light at {elevation} deg: {direct_duv}"
        );
        worst = worst.max(max);
    }
    assert!(worst < 0.005, "sky du'v' {worst}");
}

#[test]
fn zenith_blue_red_ratio() {
    let mut atmosphere = earth();
    atmosphere.aerosol = Aerosol::continental_average();
    let reference = Reference::new(&atmosphere);
    let (spectral, groups) = (
        reference.run(Channels::Spectral),
        reference.run(Channels::Groups),
    );
    let elevations = [60.0, 30.0, 10.0, 2.0];
    for e in elevations {
        let light = Light::solar(dir(e, 0.0));
        let ratio = |run: &Run| {
            let c = run.rec2020(
                &run.sky(EYE, DVec3::Y, &light, 0.0, &Precision::default())
                    .values,
            );
            c[2] / c[0]
        };
        let (s, g) = (ratio(&spectral), ratio(&groups));
        eprintln!("light {e} deg: zenith blue/red spectral {s:.4}, groups {g:.4}");
        assert!(
            (s - g).abs() < 0.1,
            "light {e} deg: spectral {s}, groups {g}"
        );
    }
}

/// The three Rec.2020 bands of the medium's `sample` and finite `t` against spectral on
/// horizontal segments: prints the worst in-scatter and transmitted-sunlight colour error
/// per light elevation and length over the aerosol presets (atmosphere.md 6).
#[test]
#[ignore = "measurement: run with --ignored --nocapture"]
fn rgb_bands_on_segments() {
    let presets = [
        Aerosol::default(),
        Aerosol::clean_continental(),
        Aerosol::continental_average(),
        Aerosol::urban(),
        Aerosol::desert(),
        Aerosol::maritime_clean(),
    ];
    let precision = Precision::default();
    let references: Vec<Reference> = presets
        .iter()
        .map(|&aerosol| Reference::new(&Atmosphere { aerosol, ..earth() }))
        .collect();
    for elevation in [5.0, 30.0] {
        let light = Light::solar(dir(elevation, 0.0));
        for length in [1_000.0, 5_000.0, 30_000.0] {
            let (mut inscatter, mut transmitted) = (0.0f64, 0.0f64);
            for reference in &references {
                let (spectral, rgb) = (
                    reference.run(Channels::Spectral),
                    reference.run(Channels::Rgb),
                );
                for azimuth in [0.0, 90.0, 180.0] {
                    let view = dir(0.0, azimuth);
                    let s =
                        spectral.segment(EYE, view, length, &light, Geometry::Plane, &precision);
                    let r = rgb.segment(EYE, view, length, &light, Geometry::Plane, &precision);
                    inscatter = inscatter.max(delta_uv(s.inscatter.xyz, r.inscatter.xyz));
                    transmitted = transmitted.max(delta_uv(
                        spectral.xyz(&s.transmittance),
                        rgb.xyz(&r.transmittance),
                    ));
                }
            }
            eprintln!(
                "light {elevation} deg, {length} m: in-scatter du'v' {inscatter:.4}, transmitted sunlight {transmitted:.4}"
            );
        }
    }
}
