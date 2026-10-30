#![allow(clippy::print_stderr)]

//! `lunar_disc` against ROLO's level and the reference: a Lambertian disc of the lunar albedo
//! at 5 deg phase, a high moon's illuminance at the ground, and the moonlit sky's colour.

use atmosphere_suite::{Probe, Rig, SUN, dir, probe, sky_radiance, uniform_texture};
use glam::{DVec3, Vec3};
use orr::abi::TextureIndex;
use orr::{DirectionalLight, DiscProfile, LightId};
use orr_atmosphere::Atmosphere;
use orr_atmosphere::colour::{REC709_TO_REC2020, REC2020_TO_XYZ, delta_uv, luminance, mul};
use orr_atmosphere::presets::{LUNAR_ALBEDO, lunar_disc, lunar_reflectance, solar_disc};
use orr_atmosphere::reference::{Channels, Geometry, Light, Precision, Reference};
use orr_atmosphere::spectrum::SPECTRAL;
use orr_scene::testing::{assert_clean, read_buffer};

const EYE: DVec3 = DVec3::new(0.0, 2.0, 0.0);
const MOON: LightId = LightId(1);
/// ROLO's full moon at the top of the atmosphere at 5 deg phase, lux.
const ROLO_5_DEG: f64 = 0.30;
/// `SceneRoot::near_range` by default.
const NEAR_RANGE: f64 = 256.0;

fn earth() -> Atmosphere {
    Atmosphere::earth(DVec3::ZERO, Vec3::Y)
}

fn rec2020(light: &DirectionalLight) -> [f64; 3] {
    mul(
        &REC709_TO_REC2020,
        light.illuminance.to_array().map(f64::from),
    )
}

/// The sun direction that puts a moon at `moon` at 5 deg phase: 5 deg off the point opposite
/// the moon, below the horizon for a moon above it.
fn full_moon(moon: DVec3) -> DVec3 {
    let axis = moon.cross(DVec3::Y).try_normalize().unwrap_or(DVec3::X);
    glam::DQuat::from_axis_angle(axis, 5f64.to_radians()) * -moon
}

/// The phase curve and a uniform texture of the lunar albedo with zero illuminance both
/// give ROLO's 0.30 lx at 5 deg phase.
#[test]
fn lunar_albedo_matches_the_phase_curve() {
    let moon_dir = dir(60.0, 0.0);
    let sun = solar_disc(full_moon(moon_dir).as_vec3());
    let curve = lunar_disc(
        moon_dir.as_vec3(),
        TextureIndex::NONE,
        Vec3::Y,
        &[(SUN, &sun)],
    );
    let curve_lux = luminance(rec2020(&curve));

    let grey = LUNAR_ALBEDO as f32;
    let albedo = uniform_texture([grey, grey, grey, 1.0], "lunar albedo");
    let lambertian = DirectionalLight {
        illuminance: Vec3::ZERO,
        disc: DiscProfile::Textured {
            albedo: albedo.index,
            up: Vec3::Y,
            lit_by: [SUN, LightId::NONE, LightId::NONE, LightId::NONE],
        },
        ..curve
    };
    let mut rig = Rig::with_lighting(&earth());
    rig.frame_lights(&[(SUN, sun), (MOON, lambertian)], EYE);
    let states = rig.lighting.as_ref().unwrap().directional_states();
    let texture_lux = luminance(read_buffer(states, 2)[1].integral.map(f64::from));
    eprintln!(
        "5 deg phase: curve {curve_lux:.4} lx, albedo {LUNAR_ALBEDO} texture {texture_lux:.4} lx"
    );
    for lux in [curve_lux, texture_lux] {
        assert!(
            (lux / ROLO_5_DEG - 1.0).abs() < 0.02,
            "{lux} lx against {ROLO_5_DEG}"
        );
    }
    assert_clean();
}

/// Moonlight's transmittance from the reference: the spectral run weighted by the lunar
/// spectrum.
fn lunar_transmittance(reference: &Reference, origin: DVec3, dir: DVec3) -> f64 {
    let spectral = reference.run(Channels::Spectral);
    let t = spectral.transmittance(origin, dir, f64::INFINITY, Geometry::Sphere);
    let (mut through, mut total) = (0.0, 0.0);
    for ((t, w), lambda) in t.iter().zip(SPECTRAL.rec2020).zip(SPECTRAL.wavelength) {
        let y = luminance(w) * lunar_reflectance(lambda);
        through += t * y;
        total += y;
    }
    through / total
}

/// A high moon's disc through the atmosphere gives its top-of-atmosphere illuminance times
/// the reference's zenith transmittance of moonlight.
#[test]
fn high_moon_at_the_ground() {
    let atmosphere = earth();
    let moon_dir = DVec3::Y;
    let sun = solar_disc(full_moon(moon_dir).as_vec3());
    let moon = lunar_disc(
        moon_dir.as_vec3(),
        TextureIndex::NONE,
        Vec3::Z,
        &[(SUN, &sun)],
    );
    let mut rig = Rig::with_lighting(&atmosphere);
    rig.frame_lights(&[(SUN, sun), (MOON, moon)], EYE);
    let set = rig.lighting.as_ref().unwrap().light_set().address().0;
    let out = probe(
        rig.block(),
        "disc",
        1,
        Probe {
            threads: 64,
            count: 256,
            layer: MOON.0,
            light: atmosphere_suite::eye_light(EYE),
            set,
            ..Probe::default()
        },
    );
    let ground = luminance([out[0][0], out[0][1], out[0][2]].map(f64::from));
    let top = luminance(rec2020(&moon));
    let t = lunar_transmittance(&Reference::new(&atmosphere), EYE, moon_dir);
    eprintln!(
        "high moon: {top:.4} lx above the atmosphere, {ground:.4} lx at the ground, \
         reference {:.4} lx (transmittance {t:.4})",
        top * t
    );
    assert!(
        (ground / (top * t) - 1.0).abs() < 0.02,
        "{ground} lx against {}",
        top * t
    );
    assert_clean();
}

/// The sky lit by the moon alone has the colour of the spectral reference lit by the lunar
/// spectrum: the solar-weighted groups hold for moonlight. Single scattering on both sides
/// (the multiple-scattering model's error is atmosphere.md 6's), no night background.
#[test]
fn moonlit_sky_colour() {
    let atmosphere = Atmosphere {
        night_background: false,
        ..earth()
    };
    let single = Precision {
        relative_error: 1.0,
        max_samples: 0,
    };
    let mut rig = Rig::new(&atmosphere);
    rig.sky.set_multiple_scattering(false);
    rig.sky.set_lights(&[MOON]);
    let reference = Reference::new(&atmosphere);
    let spectral = reference.run(Channels::Spectral);
    let views = [
        DVec3::Y,
        dir(30.0, 0.0),
        dir(30.0, 180.0),
        dir(5.0, 0.0),
        dir(5.0, 90.0),
        dir(5.0, 180.0),
        dir(-10.0, 0.0),
    ];
    let mut worst = 0.0f64;
    for elevation in [20.0, 45.0, 75.0] {
        let moon_dir = dir(elevation, 0.0);
        let sun = solar_disc(full_moon(moon_dir).as_vec3());
        let moon = lunar_disc(
            moon_dir.as_vec3(),
            TextureIndex::NONE,
            Vec3::Y,
            &[(SUN, &sun)],
        );
        rig.frame_lights(&[(SUN, sun), (MOON, moon)], EYE);
        let gpu = sky_radiance(&rig, EYE, &views);
        let light = Light {
            direction: moon_dir,
            tint: lunar_reflectance,
        };
        for (d, got) in views.iter().zip(&gpu) {
            let expected = spectral.xyz(&spectral.sky(EYE, *d, &light, NEAR_RANGE, &single).values);
            let error = delta_uv(mul(&REC2020_TO_XYZ, *got), expected);
            worst = worst.max(error);
        }
        eprintln!("moon at {elevation} deg: worst Delta u'v' {worst:.4}");
    }
    assert!(worst < 0.005, "moonlit sky off by Delta u'v' {worst:.4}");
    assert_clean();
}
