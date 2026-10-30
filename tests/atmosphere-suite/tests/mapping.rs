#![allow(clippy::print_stderr)]

//! LUT parameterisation round trips and f32 altitudes on the GPU.

use atmosphere_suite::{Probe, Rig, dir, probe};
use glam::{DVec3, Vec3};
use orr_atmosphere::{Atmosphere, MULTIPLE_SCATTERING_SIZE, SKY_VIEW_SIZE, TRANSMITTANCE_SIZE};
use orr_scene::testing::{assert_clean, read_buffer};

fn texels(size: [u32; 2]) -> u32 {
    size[0] * size[1]
}

#[test]
fn lut_parameterisations_round_trip() {
    let atmosphere = Atmosphere::earth(DVec3::ZERO, Vec3::Y);
    let mut rig = Rig::new(&atmosphere);
    rig.frame(Some(dir(30.0, 0.0)), DVec3::new(0.0, 2.0, 0.0));
    let block = rig.block();
    let radius = f64::from(atmosphere.planet_radius);

    for (entry, size) in [
        ("transmittance_texels", TRANSMITTANCE_SIZE),
        ("multiple_scattering_texels", MULTIPLE_SCATTERING_SIZE),
    ] {
        let n = texels(size);
        let out = probe(
            block,
            entry,
            n.into(),
            Probe {
                threads: n,
                ..Probe::default()
            },
        );
        let worst = out.iter().map(|t| t[2].max(t[3])).fold(0.0f32, f32::max);
        assert!(worst < 1e-2, "{entry}: round trip off by {worst} texels");
        if entry == "transmittance_texels" {
            // The last column is the horizon: the ray grazes the ground.
            for row in 0..size[1] {
                let t = out[(row * size[0] + size[0] - 1) as usize];
                let (h, mu) = (f64::from(t[0]), f64::from(t[1]));
                let horizon = -(h * (2.0 * radius + h)).sqrt() / (radius + h);
                assert!(
                    (mu - horizon).abs() < 1e-4,
                    "row {row} at {h} m: last column {mu}, horizon {horizon}"
                );
            }
        }
    }

    // The horizon sits between rows 99 and 100 at every eye altitude.
    let n = texels(SKY_VIEW_SIZE);
    for altitude in [2.0f32, 1000.0, 50_000.0] {
        let out = probe(
            block,
            "sky_view_texels",
            n.into(),
            Probe {
                threads: n,
                light: [0.0, 0.0, 0.0, altitude],
                ..Probe::default()
            },
        );
        let worst = out.iter().map(|t| t[3]).fold(0.0f32, f32::max);
        assert!(
            worst < 1e-2,
            "sky view at {altitude} m: round trip off by {worst} texels"
        );
        let h = f64::from(altitude);
        let horizon = -(h * (2.0 * radius + h)).sqrt() / (radius + h);
        let mid = SKY_VIEW_SIZE[1] / 2;
        for col in 0..SKY_VIEW_SIZE[0] {
            let above = out[((mid - 1) * SKY_VIEW_SIZE[0] + col) as usize][1];
            let below = out[(mid * SKY_VIEW_SIZE[0] + col) as usize][1];
            assert!(
                f64::from(above) > horizon && f64::from(below) < horizon,
                "sky view at {altitude} m, column {col}: rows either side of the horizon at \
                 {above} and {below}, horizon {horizon}"
            );
        }
    }
    assert_clean();
}

/// Plane and spherical altitude in f32 on the GPU against f64, over a 32 km world around
/// the planet top, from two render origins.
#[test]
fn altitudes_against_f64() {
    let top = DVec3::new(12_345.678, -40.25, -20_000.5);
    let up = Vec3::new(0.1, 1.0, 0.05).normalize();
    let atmosphere = Atmosphere::earth(top, up);
    let radius = f64::from(atmosphere.planet_radius);
    let mut rig = Rig::new(&atmosphere);
    let up = up.as_dvec3().normalize();
    let east = up.cross(DVec3::Z).normalize();
    let north = east.cross(up);
    let centre = top - up * radius;

    let mut world = Vec::new();
    for i in 0..=16 {
        for j in 0..=16 {
            for altitude in [0.0, 1.5, 120.0, 2_000.0, 10_000.0] {
                let x = -16_000.0 + 2_000.0 * f64::from(i);
                let z = -16_000.0 + 2_000.0 * f64::from(j);
                world.push(top + east * x + north * z + up * altitude);
            }
        }
    }
    for focus in [
        top + DVec3::new(15_000.0, 0.0, -14_000.0),
        top + DVec3::new(-16_000.0, 3_000.0, 9_000.0),
    ] {
        let root = rig.frame(None, focus);
        let cell = read_buffer(root, 1)[0].origin_cell;
        let origin = DVec3::new(f64::from(cell[0]), f64::from(cell[1]), f64::from(cell[2]))
            * f64::from(1u32 << cell[3]);
        let render: Vec<[f32; 4]> = world
            .iter()
            .map(|p| {
                let r = (*p - origin).as_vec3();
                [r.x, r.y, r.z, 0.0]
            })
            .collect();
        let n = render.len() as u32;
        let out = probe(
            rig.block(),
            "altitude",
            n.into(),
            Probe {
                threads: n,
                count: n,
                input: &render,
                ..Probe::default()
            },
        );
        let (mut plane, mut sphere) = (0.0f64, 0.0f64);
        for (p, o) in world.iter().zip(&out) {
            plane = plane.max((f64::from(o[0]) - (*p - top).dot(up)).abs());
            sphere = sphere.max((f64::from(o[1]) - ((*p - centre).length() - radius)).abs());
        }
        assert!(
            plane < 1.0 && sphere < 1.0,
            "plane {plane} m, sphere {sphere} m"
        );
        eprintln!("origin {origin}: plane altitude off by {plane:.4} m, spherical {sphere:.4} m");
    }
    assert_clean();
}
