#![allow(clippy::print_stderr)]

//! `AtmosphereSky` as orr's linked environment: the horizon, the night background, a light's
//! disc through the atmosphere, the sky-view LUT composed with the medium inside
//! `near_range`, the fold of several lights, and lighting's bake of the linked sky.

use atmosphere_suite::{Probe, Rig, SUN, dir, probe};
use glam::{DVec3, Vec3};
use orr::abi::TextureIndex;
use orr::{DirectionalLight, DiscProfile, LightId};
use orr_atmosphere::colour::{REC2020_LUMINANCE, REC2020_TO_REC709, mul};
use orr_atmosphere::presets::solar_disc;
use orr_atmosphere::reference::{Channels, Geometry, Light, Precision, Reference};
use orr_atmosphere::spectrum::{SOLAR_COLOUR, SOLAR_ILLUMINANCE};
use orr_atmosphere::{Atmosphere, HeightFog, SKY_VIEW_SIZE};
use orr_lighting::ENVIRONMENT_SIZE;
use orr_scene::testing::{assert_clean, read_buffer};

const EYE: DVec3 = DVec3::new(0.0, 2.0, 0.0);
/// `SceneRoot::near_range` by default.
const NEAR_RANGE: f64 = 256.0;

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

fn luminance(c: [f64; 3]) -> f64 {
    (0..3).map(|i| REC2020_LUMINANCE[i] * c[i]).sum()
}

fn rgb(c: [f32; 4]) -> [f64; 3] {
    [c[0].into(), c[1].into(), c[2].into()]
}

fn eye_light() -> [f32; 4] {
    [EYE.x as f32, EYE.y as f32, EYE.z as f32, EYE.y as f32]
}

/// `AtmosphereSky.radiance` from [`EYE`] along each of `dirs`.
fn radiance(rig: &Rig, dirs: &[DVec3]) -> Vec<[f64; 3]> {
    let input: Vec<[f32; 4]> = dirs
        .iter()
        .map(|d| [d.x as f32, d.y as f32, d.z as f32, 0.0])
        .collect();
    let n = dirs.len() as u32;
    let out = probe(
        rig.block(),
        "sky_radiance",
        n.into(),
        Probe {
            threads: n,
            count: n,
            input: &input,
            light: eye_light(),
            ..Probe::default()
        },
    );
    let out: Vec<_> = out.into_iter().map(rgb).collect();
    assert!(out.iter().flatten().all(|c| c.is_finite()), "{out:?}");
    out
}

/// The direction of every sky-view texel, row by row, from [`EYE`], once a frame has written
/// the block.
fn texel_directions(rig: &Rig) -> Vec<DVec3> {
    let n = SKY_VIEW_SIZE[0] * SKY_VIEW_SIZE[1];
    let texels = probe(
        rig.block(),
        "sky_view_texels",
        n.into(),
        Probe {
            threads: n,
            light: eye_light(),
            ..Probe::default()
        },
    );
    texels
        .iter()
        .map(|t| DVec3::new(t[0].into(), t[1].into(), t[2].into()).normalize())
        .collect()
}

/// The rows either side of the horizon match the reference within the 5% sky gate, single
/// scattering on both sides, so the LUT adds no step of its own. The sky still steps there:
/// just above the horizon the ray grazes the planet for hundreds of kilometres, just below
/// it meets the ground about 5 km away (from 2 m).
#[test]
fn horizon_matches_reference() {
    let [w, h] = SKY_VIEW_SIZE;
    let single = Precision {
        relative_error: 1.0,
        max_samples: 0,
    };
    let mut worst = 0.0f64;
    for atmosphere in [earth(), foggy()] {
        // The reference has no night background.
        let atmosphere = Atmosphere {
            night_background: false,
            ..atmosphere
        };
        let fog = atmosphere.height_fog.is_some();
        let mut rig = Rig::new(&atmosphere);
        rig.sky.set_multiple_scattering(false);
        rig.frame(None, EYE);
        let texels = texel_directions(&rig);
        let reference = Reference::new(&atmosphere);
        let groups = reference.run(Channels::Groups);
        // Three azimuths 90 deg apart in the two rows above the horizon and the two below.
        let rows = [h / 2 - 2, h / 2 - 1, h / 2, h / 2 + 1];
        let columns = [0, w / 4, w / 2];
        let dirs: Vec<DVec3> = rows
            .iter()
            .flat_map(|r| columns.iter().map(move |c| (r * w + c) as usize))
            .map(|i| texels[i])
            .collect();
        for elevation in [-5.0, 5.0, 30.0, 60.0] {
            let sun = dir(elevation, 0.0);
            rig.frame(Some(sun), EYE);
            let gpu = radiance(&rig, &dirs);
            for (d, got) in dirs.iter().zip(&gpu) {
                let r = groups.sky(EYE, *d, &Light::solar(sun), NEAR_RANGE, &single);
                let expected = groups.rec2020(&r.values).map(|c| c * SOLAR_ILLUMINANCE);
                let y = luminance(expected);
                let error = if y > 0.0 {
                    (0..3)
                        .map(|c| (got[c] - expected[c]).abs() / expected[c].abs().max(0.05 * y))
                        .fold(0.0, f64::max)
                } else if luminance(*got) > 1e-9 {
                    1.0
                } else {
                    0.0
                };
                worst = worst.max(error);
            }
            eprintln!(
                "fog {fog:>5}, light {elevation:>4} deg: worst {:.2}% off the reference",
                worst * 100.0
            );
        }
    }
    assert!(worst < 0.05, "horizon rows off by {:.2}%", worst * 100.0);
    assert_clean();
}

/// With every light more than 30 deg below the horizon only the night background is left:
/// 2e-4 cd/m^2 at the ground zenith, and 1.32 x the zenith value at air mass 1.5 before
/// extinction.
#[test]
fn night_background() {
    let atmosphere = earth();
    let mut rig = Rig::new(&atmosphere);
    rig.frame(Some(dir(-35.0, 0.0)), EYE);

    let zenith = radiance(&rig, &[DVec3::Y])[0];
    // The froxels' share: transmittance over the first `near_range`.
    let reference = Reference::new(&atmosphere);
    let groups = reference.run(Channels::Groups);
    let t = groups.rec2020(&groups.transmittance(EYE, DVec3::Y, NEAR_RANGE, Geometry::Sphere));
    let ground = luminance(std::array::from_fn(|c| zenith[c] * t[c] / SOLAR_COLOUR[c]));
    eprintln!("night zenith at the ground {ground:.4e} cd/m^2");
    assert!(
        (ground / 2e-4 - 1.0).abs() < 0.1,
        "night zenith {ground:.3e} cd/m^2"
    );

    let mu = [[1.0, 0.0, 0.0, 0.0], [1.0 / 1.5, 0.0, 0.0, 0.0]];
    let out = probe(
        rig.block(),
        "night_background",
        2,
        Probe {
            threads: 2,
            count: 2,
            input: &mu,
            light: eye_light(),
            ..Probe::default()
        },
    );
    let ratio = f64::from(out[1][0] / out[0][0]);
    eprintln!("night background at air mass 1.5: {ratio:.4} x zenith");
    assert!((ratio - 1.32).abs() <= 0.05, "air mass 1.5: {ratio}");
    assert_clean();
}

/// `solar_disc`'s limb-darkened radiance through the atmosphere integrates to its
/// illuminance times the transmittance along its axis.
#[test]
fn disc_integral_after_the_medium() {
    let mut rig = Rig::with_lighting(&earth());
    let mut worst = 0.0f64;
    for elevation in [5.0, 30.0, 60.0, 90.0] {
        rig.frame(Some(dir(elevation, 20.0)), EYE);
        let set = rig.lighting.as_ref().unwrap().light_set().address().0;
        let out = probe(
            rig.block(),
            "disc",
            2,
            Probe {
                threads: 64,
                count: 256,
                layer: SUN.0,
                light: eye_light(),
                set,
                ..Probe::default()
            },
        );
        let (integral, expected) = (rgb(out[0]), rgb(out[1]));
        let error = (0..3)
            .map(|c| (integral[c] / expected[c] - 1.0).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "light {elevation} deg: disc {:.1} lx of {:.1} lx, worst channel {:.3}%",
            luminance(integral),
            luminance(expected),
            error * 100.0
        );
        worst = worst.max(error);
    }
    assert!(worst < 0.005, "disc integral off by {:.3}%", worst * 100.0);
    assert_clean();
}

/// The worst per-channel error of the sky-view LUT, composed with the reference over
/// [0, `near_range`] along the same ray, against the reference's whole path, at light 5, 30
/// and 60 deg toward the zenith and the horizon, height fog off and on.
fn composition_error(multiple: bool, precision: &Precision) -> f64 {
    let [w, h] = SKY_VIEW_SIZE;
    let mut worst = 0.0f64;
    for atmosphere in [earth(), foggy()] {
        let fog = atmosphere.height_fog.is_some();
        let mut rig = Rig::new(&atmosphere);
        rig.sky.set_multiple_scattering(multiple);
        rig.frame(None, EYE);
        let texels = texel_directions(&rig);
        // The zenith, and the lowest row above the horizon toward the light and away.
        let row = &texels[(h / 2 - 1) as usize * w as usize..][..w as usize];
        let along = |sign: f64| {
            *row.iter()
                .max_by(|a, b| (sign * a.x).total_cmp(&(sign * b.x)))
                .unwrap()
        };
        let views = [
            ("zenith", texels[0]),
            ("horizon toward", along(1.0)),
            ("horizon away", along(-1.0)),
        ];
        let reference = Reference::new(&atmosphere);
        let groups = reference.run(Channels::Groups);
        for elevation in [5.0, 30.0, 60.0] {
            let sun = dir(elevation, 0.0);
            rig.frame(Some(sun), EYE);
            let dirs: Vec<DVec3> = views.iter().map(|v| v.1).collect();
            let lut = radiance(&rig, &dirs);
            let light = Light::solar(sun);
            for ((name, d), lut) in views.iter().zip(&lut) {
                let nits = |v: &[f64]| groups.rec2020(v).map(|c| c * SOLAR_ILLUMINANCE);
                let full = nits(&groups.sky(EYE, *d, &light, 0.0, precision).values);
                let near = groups.segment(EYE, *d, NEAR_RANGE, &light, Geometry::Sphere, precision);
                let near_inscatter = nits(&near.inscatter.values);
                let t = groups.rec2020(&near.transmittance);
                let composed: [f64; 3] =
                    std::array::from_fn(|c| near_inscatter[c] + t[c] / SOLAR_COLOUR[c] * lut[c]);
                let y = luminance(full);
                let error = (0..3)
                    .map(|c| (composed[c] - full[c]).abs() / full[c].abs().max(0.05 * y))
                    .fold(0.0, f64::max);
                eprintln!(
                    "fog {fog:>5}, light {elevation:>4} deg, {name:>14}: {y:.3e} cd/m^2 \
                     ({:.1}% inside near_range), worst channel off by {:.2}%",
                    luminance(near_inscatter) / y * 100.0,
                    error * 100.0
                );
                worst = worst.max(error);
            }
        }
    }
    worst
}

/// The LUT starts where the froxels stop: nothing inside `near_range` is counted twice or
/// dropped. Single scattering on both sides, which the A2 suite holds to the reference.
#[test]
fn near_range_composition() {
    let single = Precision {
        relative_error: 1.0,
        max_samples: 0,
    };
    let worst = composition_error(false, &single);
    assert!(worst < 0.05, "composed sky off by {:.2}%", worst * 100.0);
    assert_clean();
}

/// The same with multiple scattering: carries the multiple-scattering LUT's model error
/// (atmosphere.md 6).
#[test]
#[ignore = "measurement, about 20 s: run with --ignored --nocapture"]
fn near_range_composition_full_sky() {
    let precision = Precision {
        relative_error: 0.01,
        max_samples: 1 << 22,
    };
    let worst = composition_error(true, &precision);
    eprintln!("worst channel {:.1}%", worst * 100.0);
    assert_clean();
}

/// A light of `colour` x the sun's illuminance from `direction`.
fn tinted(direction: DVec3, colour: Vec3) -> DirectionalLight {
    let sun = solar_disc(direction.as_vec3());
    DirectionalLight {
        illuminance: sun.illuminance * colour,
        ..sun
    }
}

/// The whole sky-view LUT.
fn lut(rig: &Rig) -> Vec<[f64; 3]> {
    let n = SKY_VIEW_SIZE[0] * SKY_VIEW_SIZE[1];
    let all = Probe {
        threads: n,
        ..Probe::default()
    };
    let out: Vec<_> = probe(rig.block(), "sky_view_lut", n.into(), all)
        .into_iter()
        .map(rgb)
        .collect();
    assert!(out.iter().flatten().all(|c| c.is_finite()));
    out
}

/// The worst per-texel channel difference of `a` from `b`, relative to the channel or to 5%
/// of the luminance for a channel near zero.
fn lut_difference(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let floor = 0.05 * luminance(*b);
            (0..3)
                .map(|c| (a[c] - b[c]).abs() / b[c].abs().max(floor).max(1e-12))
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

/// Lights at full weight add; a candidate with zero weight changes nothing.
#[test]
fn folding() {
    let lights = [
        (LightId(0), solar_disc(dir(30.0, 0.0).as_vec3())),
        (
            LightId(1),
            tinted(dir(10.0, 120.0), Vec3::new(0.3, 0.2, 0.1)),
        ),
        (
            LightId(2),
            tinted(dir(50.0, 240.0), Vec3::new(0.02, 0.03, 0.05)),
        ),
        // More than 18 deg below the horizon: never folded.
        (LightId(3), tinted(dir(-25.0, 60.0), Vec3::splat(1.0))),
    ];
    let mut rig = Rig::new(&earth());
    let mut single = Vec::new();
    for (id, _) in &lights[..3] {
        rig.sky.set_lights(&[*id]);
        rig.frame_lights(&lights, EYE);
        single.push(lut(&rig));
    }
    rig.sky.set_lights(&[LightId(0), LightId(1), LightId(2)]);
    rig.frame_lights(&lights, EYE);
    let sum: Vec<[f64; 3]> = (0..single[0].len())
        .map(|i| std::array::from_fn(|c| single.iter().map(|l| l[i][c]).sum()))
        .collect();
    let added = lut_difference(&lut(&rig), &sum);
    eprintln!(
        "three lights against the sum of each: {:.4}%",
        added * 100.0
    );
    assert!(
        added < 0.01,
        "folded lights off their sum by {:.3}%",
        added * 100.0
    );

    rig.sky.set_lights(&[LightId(0), LightId(3)]);
    rig.frame_lights(&lights, EYE);
    let below = lut_difference(&lut(&rig), &single[0]);
    rig.sky.set(&Atmosphere {
        sky_lights: 1,
        ..earth()
    });
    rig.sky.set_lights(&[LightId(0), LightId(1)]);
    rig.frame_lights(&lights, EYE);
    let ranked_out = lut_difference(&lut(&rig), &single[0]);
    eprintln!("zero-weight candidates: below the cutoff {below:.2e}, ranked out {ranked_out:.2e}");
    assert!(below < 1e-5 && ranked_out < 1e-5, "{below} {ranked_out}");
    assert_clean();
}

/// A `Textured` candidate with zero illuminance folds in with its disc integral, as a
/// `Uniform` light of that illuminance does.
#[test]
fn textured_candidate_folds_its_disc_integral() {
    let sun = (SUN, solar_disc(dir(30.0, 0.0).as_vec3()));
    let moon_dir = dir(40.0, 150.0).as_vec3();
    let moon = DirectionalLight {
        direction_to_light: moon_dir,
        illuminance: Vec3::ZERO,
        angular_radius: 4.52e-3,
        disc: DiscProfile::Textured {
            albedo: TextureIndex::NONE,
            up: Vec3::Y,
            lit_by: [SUN, LightId::NONE, LightId::NONE, LightId::NONE],
        },
    };
    let mut rig = Rig::with_lighting(&earth());
    rig.sky.set_lights(&[LightId(1)]);
    rig.frame_lights(&[sun, (LightId(1), moon)], EYE);
    let textured = lut(&rig);
    let states = rig.lighting.as_ref().unwrap().directional_states();
    let integral = read_buffer(states, 2)[1].integral;
    let rec709 = mul(&REC2020_TO_REC709, integral.map(f64::from));
    let uniform = DirectionalLight {
        illuminance: Vec3::from(rec709.map(|c| c as f32)),
        disc: DiscProfile::Uniform,
        ..moon
    };
    rig.frame_lights(&[sun, (LightId(1), uniform)], EYE);
    let difference = lut_difference(&textured, &lut(&rig));
    eprintln!(
        "disc integral {:.4} lx; textured against uniform {difference:.2e}",
        luminance(integral.map(f64::from))
    );
    assert!(luminance(integral.map(f64::from)) > 0.0);
    assert!(difference < 1e-4, "{difference}");
    assert_clean();
}

/// Lighting's environment bake through orr's kernels linked to `AtmosphereSky` holds the
/// sky.
#[test]
fn lighting_bakes_the_linked_sky() {
    let mut rig = Rig::with_lighting(&foggy());
    for elevation in [-35.0, 5.0, 30.0] {
        rig.frame(Some(dir(elevation, 0.0)), EYE);
        let expected = probe(
            rig.block(),
            "bake_mean",
            1,
            Probe {
                threads: 64,
                count: ENVIRONMENT_SIZE,
                light: eye_light(),
                ..Probe::default()
            },
        )[0][3];
        let baked = rig.lighting_stats().sky.dome_luminance;
        eprintln!("light {elevation} deg: baked dome {baked:.4e}, sky {expected:.4e} cd/m^2");
        assert!(
            (baked / expected - 1.0).abs() < 1e-4,
            "baked {baked} against {expected}"
        );
    }
    assert_clean();
}
