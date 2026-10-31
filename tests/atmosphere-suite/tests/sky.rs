#![allow(clippy::print_stderr)]

//! `AtmosphereSky` as orr's linked environment: the horizon, the night background, a light's
//! disc through the atmosphere, the sky-view LUT composed with the medium inside
//! `near_range`, the fold of several lights and its fades, and lighting's bake of the linked
//! sky.

use atmosphere_suite::{Probe, Rig, SUN, dir, probe, sky_radiance};
use glam::{DVec3, Vec3};
use orr::abi::TextureIndex;
use orr::{DirectionalLight, DiscProfile, LightId};
use orr_atmosphere::colour::{REC2020_TO_REC709, luminance, mul};
use orr_atmosphere::presets::solar_disc;
use orr_atmosphere::reference::{Channels, Geometry, Light, Precision, Reference};
use orr_atmosphere::spectrum::{SOLAR_COLOUR, SOLAR_ILLUMINANCE};
use orr_atmosphere::{Atmosphere, FoldGpu, HeightFog, MAX_FOLDED_LIGHTS, SKY_VIEW_SIZE};
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

fn rgb(c: [f32; 4]) -> [f64; 3] {
    [c[0].into(), c[1].into(), c[2].into()]
}

fn eye_light() -> [f32; 4] {
    atmosphere_suite::eye_light(EYE)
}

/// `AtmosphereSky.radiance` from [`EYE`] along each of `dirs`.
fn radiance(rig: &Rig, dirs: &[DVec3]) -> Vec<[f64; 3]> {
    sky_radiance(rig, EYE, dirs)
}

/// The direction of every sky-view texel, row by row, from [`EYE`], once a frame has written
/// the block.
fn texel_directions(rig: &Rig) -> Vec<DVec3> {
    texel_directions_from(rig, EYE)
}

/// The direction of every sky-view texel of a LUT at the altitude of `eye`.
fn texel_directions_from(rig: &Rig, eye: DVec3) -> Vec<DVec3> {
    let n = SKY_VIEW_SIZE[0] * SKY_VIEW_SIZE[1];
    let texels = probe(
        rig.block(),
        "sky_view_texels",
        n.into(),
        Probe {
            threads: n,
            light: atmosphere_suite::eye_light(eye),
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
/// just above the horizon the ray grazes the planet for hundreds of kilometers, just below
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
/// and 60 deg toward the zenith and the horizon, height fog off and on, seen from each of
/// `eyes`, the rig's other views, with the focus at [`EYE`].
fn composition_error(multiple: bool, precision: &Precision, eyes: &[DVec3]) -> f64 {
    let [w, h] = SKY_VIEW_SIZE;
    let mut worst = 0.0f64;
    for atmosphere in [earth(), foggy()] {
        let fog = atmosphere.height_fog.is_some();
        let mut rig = Rig::new(&atmosphere);
        rig.sky.set_multiple_scattering(multiple);
        rig.eyes = eyes.to_vec();
        rig.frame(None, EYE);
        let reference = Reference::new(&atmosphere);
        let groups = reference.run(Channels::Groups);
        for &eye in eyes {
            let texels = texel_directions_from(&rig, eye);
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
            for elevation in [5.0, 30.0, 60.0] {
                let sun = dir(elevation, 0.0);
                rig.frame(Some(sun), EYE);
                let dirs: Vec<DVec3> = views.iter().map(|v| v.1).collect();
                let lut = sky_radiance(&rig, eye, &dirs);
                let light = Light::solar(sun);
                for ((name, d), lut) in views.iter().zip(&lut) {
                    let nits = |v: &[f64]| groups.rec2020(v).map(|c| c * SOLAR_ILLUMINANCE);
                    let full = nits(&groups.sky(eye, *d, &light, 0.0, precision).values);
                    let near =
                        groups.segment(eye, *d, NEAR_RANGE, &light, Geometry::Sphere, precision);
                    let near_inscatter = nits(&near.inscatter.values);
                    let t = groups.rec2020(&near.transmittance);
                    let composed: [f64; 3] = std::array::from_fn(|c| {
                        near_inscatter[c] + t[c] / SOLAR_COLOUR[c] * lut[c]
                    });
                    let y = luminance(full);
                    let error = (0..3)
                        .map(|c| (composed[c] - full[c]).abs() / full[c].abs().max(0.05 * y))
                        .fold(0.0, f64::max);
                    eprintln!(
                        "eye {:>7.0} m, fog {fog:>5}, light {elevation:>4} deg, {name:>14}:                          {y:.3e} cd/m^2 ({:.1}% inside near_range), worst channel off by {:.2}%",
                        eye.y,
                        luminance(near_inscatter) / y * 100.0,
                        error * 100.0
                    );
                    worst = worst.max(error);
                }
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
    let worst = composition_error(false, &single, &[EYE]);
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
    let worst = composition_error(true, &precision, &[EYE]);
    eprintln!("worst channel {:.1}%", worst * 100.0);
    assert_clean();
}

/// Eyes 50 m, 100 m, 1 km and 10 km above the focus at [`EYE`].
const HIGH_EYES: [DVec3; 4] = [
    DVec3::new(0.0, 52.0, 0.0),
    DVec3::new(30.0, 102.0, 0.0),
    DVec3::new(0.0, 1002.0, 40.0),
    DVec3::new(-50.0, 10_002.0, 0.0),
];

/// Views well above the focus get sky-view LUTs at their own altitudes and meet the 5% sky
/// gate there (single scattering on both sides, as in `near_range_composition`).
#[test]
fn extra_views_meet_the_sky_gate() {
    let single = Precision {
        relative_error: 1.0,
        max_samples: 0,
    };
    let mut rig = Rig::new(&earth());
    rig.eyes = HIGH_EYES.to_vec();
    rig.frame(None, EYE);
    let block = read_buffer(rig.block(), 1)[0];
    let mut altitudes = block.sky_view_altitude[1..block.sky_view_count as usize].to_vec();
    altitudes.sort_by(f32::total_cmp);
    assert_eq!(altitudes, HIGH_EYES.map(|e| e.y as f32), "one LUT per eye");

    let worst = composition_error(false, &single, &HIGH_EYES);
    assert!(worst < 0.05, "composed sky off by {:.2}%", worst * 100.0);
    assert_clean();
}

/// An eye moving 0.2 m in one frame through 25 m above the focus, where it gets its own LUT,
/// and through 50 m, where that LUT stops being blended with the focus LUT, changes no sky
/// pixel by more than 2%. Below the horizon the virtual ground still changes by far more:
/// from 2 m it lies inside `near_range` near the horizon, from 27 m kilometers away.
#[test]
fn crossing_the_view_thresholds() {
    let elevations = [0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 60.0, 89.0];
    let dirs: Vec<DVec3> = elevations
        .iter()
        .flat_map(|&e| (0..16).map(move |a| dir(e, f64::from(a) * 22.5)))
        .collect();
    let ground: Vec<DVec3> = [-0.5, -1.0, -3.0, -10.0]
        .iter()
        .flat_map(|&e| (0..16).map(move |a| dir(e, f64::from(a) * 22.5)))
        .collect();
    let mut worst = 0.0f64;
    for atmosphere in [earth(), foggy()] {
        let fog = atmosphere.height_fog.is_some();
        let mut rig = Rig::new(&atmosphere);
        for elevation in [5.0, 30.0] {
            let sun = Some(dir(elevation, 0.0));
            for (from, to, luts) in [(24.9, 25.1, [1, 2]), (49.9, 50.1, [2, 2])] {
                let mut sky = Vec::new();
                let mut below = Vec::new();
                for (above, luts) in [(from, luts[0]), (to, luts[1])] {
                    let eye = EYE + DVec3::Y * above;
                    rig.eyes = vec![eye];
                    rig.frame(sun, EYE);
                    let count = read_buffer(rig.block(), 1)[0].sky_view_count;
                    assert_eq!(count, luts, "{above} m above the focus");
                    sky.push(sky_radiance(&rig, eye, &dirs));
                    below.push(sky_radiance(&rig, eye, &ground));
                }
                let change = lut_difference(&sky[1], &sky[0]);
                eprintln!(
                    "fog {fog:>5}, light {elevation:>4} deg, {from} to {to} m: sky {:.2}%, \
                     virtual ground {:.1}%",
                    change * 100.0,
                    lut_difference(&below[1], &below[0]) * 100.0
                );
                worst = worst.max(change);
            }
        }
    }
    assert!(worst < 0.02, "sky changed by {:.2}%", worst * 100.0);
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

/// A fold weight's change in one 1/60 s frame.
const FADE_STEP: f32 = 1.0 / 30.0;

fn fold_state(rig: &Rig) -> FoldGpu {
    let block = read_buffer(rig.block(), 1)[0];
    read_buffer(block.fold, 1)[0]
}

fn fading_rig(sky_lights: u32) -> Rig {
    let mut rig = Rig::new(&Atmosphere {
        sky_lights,
        ..earth()
    });
    rig.fades = true;
    rig
}

/// Weights move by `delta_time / 0.5 s` toward 1 for members and 0 for the rest, stay in
/// [0, 1], and at most `MAX_FOLDED_LIGHTS` are non-zero, members first.
#[test]
fn fold_weights_fade() {
    let mut rig = fading_rig(1);
    let a = |s: f32| (LightId(0), tinted(dir(30.0, 0.0), Vec3::splat(s)));
    let b = |s: f32| (LightId(1), tinted(dir(10.0, 180.0), Vec3::splat(s)));
    rig.sky.set_lights(&[LightId(0), LightId(1)]);
    rig.frame_lights(&[a(1.0), b(0.5)], EYE);
    let f = fold_state(&rig);
    assert_eq!((f.weight[0], f.weight[1], f.folded_count), (1.0, 0.0, 1));
    for k in 1..=32 {
        rig.frame_lights(&[a(1.0), b(2.0)], EYE);
        let f = fold_state(&rig);
        let rate = (k as f32 * FADE_STEP).min(1.0);
        assert!(
            (f.weight[1] - rate).abs() < 1e-5 && (f.weight[0] - (1.0 - rate)).abs() < 1e-5,
            "frame {k}: weights {:?}",
            &f.weight[..2]
        );
        assert!(f.weight.iter().all(|w| (0.0..=1.0).contains(w)));
        assert_eq!(f.folded[0], 1, "the member folds first");
        assert_eq!(f.folded_count, if f.weight[0] > 0.0 { 2 } else { 1 });
    }
    let f = fold_state(&rig);
    assert_eq!((f.weight[0], f.weight[1]), (0.0, 1.0));

    // Reordering the candidates keeps each light's weight.
    rig.frame_lights(&[a(3.0), b(2.0)], EYE);
    let before = fold_state(&rig);
    rig.sky.set_lights(&[LightId(1), LightId(0)]);
    rig.frame_lights(&[a(3.0), b(2.0)], EYE);
    let after = fold_state(&rig);
    assert!((after.weight[1] - (before.weight[0] + FADE_STEP)).abs() < 1e-5);
    assert!((after.weight[0] - (before.weight[1] - FADE_STEP)).abs() < 1e-5);

    // 4 members leaving and 4 joining fold 8; 8 leaving and 8 joining keep the 8 joining.
    let many = |bright: &dyn Fn(u32) -> bool| -> Vec<_> {
        (0..16)
            .map(|i| {
                let s = if bright(i) { 1.0 } else { 0.1 };
                let d = dir(10.0 + f64::from(i) * 4.0, f64::from(i) * 22.5);
                (LightId(i), tinted(d, Vec3::splat(s)))
            })
            .collect()
    };
    let ids: Vec<_> = (0..16).map(LightId).collect();
    for (sky_lights, candidates) in [(4_u32, 8_usize), (8, 16)] {
        let mut rig = fading_rig(sky_lights);
        rig.sky.set_lights(&ids[..candidates]);
        rig.frame_lights(&many(&|i| i < sky_lights), EYE);
        rig.frame_lights(&many(&|i| i >= sky_lights), EYE);
        let f = fold_state(&rig);
        let weights = &f.weight[..candidates];
        assert_eq!(f.folded_count, MAX_FOLDED_LIGHTS);
        assert_eq!(
            weights.iter().filter(|w| **w > 0.0).count(),
            MAX_FOLDED_LIGHTS as usize
        );
        for (i, w) in weights.iter().enumerate() {
            let i = i as u32;
            let expected = if (sky_lights..2 * sky_lights).contains(&i) {
                FADE_STEP
            } else if 2 * sky_lights <= MAX_FOLDED_LIGHTS {
                1.0 - FADE_STEP
            } else {
                0.0
            };
            assert!(
                (w - expected).abs() < 1e-5,
                "{sky_lights} sky lights: {weights:?}"
            );
        }
    }
    assert_clean();
}

/// The worst per-texel channel change from `last` to `next`, relative to the sky with every
/// candidate folded at full weight, `full`, or to 5% of its luminance for a channel near
/// zero.
fn step_against(next: &[[f64; 3]], last: &[[f64; 3]], full: &[[f64; 3]]) -> f64 {
    next.iter()
        .zip(last)
        .zip(full)
        .map(|((n, l), f)| {
            let floor = 0.05 * luminance(*f);
            (0..3)
                .map(|c| (n[c] - l[c]).abs() / f[c].abs().max(floor).max(1e-12))
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

/// Steps of a fading sky with one sky light and two candidates.
struct Steps {
    /// Against the sky with both candidates folded.
    full: f64,
    /// Against the texel itself.
    texel: f64,
}

/// Runs `frames` fading frames of a one-sky-light rig whose candidates are lights 0 and 1,
/// lit by `at(frame)`, and returns the worst step between consecutive frames, the frame
/// whose fold first held light 1 and each frame's LUT.
fn fade_run(
    frames: u32,
    at: &dyn Fn(u32) -> Vec<(LightId, DirectionalLight)>,
) -> (Steps, Option<u32>, Vec<Vec<[f64; 3]>>) {
    let mut rig = fading_rig(1);
    let mut both = Rig::new(&Atmosphere {
        sky_lights: 2,
        ..earth()
    });
    for r in [&mut rig, &mut both] {
        r.sky.set_lights(&[LightId(0), LightId(1)]);
    }
    let mut steps = Steps {
        full: 0.0,
        texel: 0.0,
    };
    let mut joined = None;
    let mut luts: Vec<Vec<[f64; 3]>> = Vec::new();
    for n in 0..frames {
        rig.frame_lights(&at(n), EYE);
        both.frame_lights(&at(n), EYE);
        if joined.is_none() && fold_state(&rig).weight[1] > 0.0 {
            joined = Some(n);
        }
        let next = lut(&rig);
        if let Some(last) = luts.last() {
            steps.full = steps.full.max(step_against(&next, last, &lut(&both)));
            steps.texel = steps.texel.max(lut_difference(&next, last));
        }
        luts.push(next);
    }
    (steps, joined, luts)
}

/// With one sky light and two candidates crossing in brightness, the sky changes by under
/// 5% of the sky with both folded each frame at 60 fps, and 0.5 s after the fold changes
/// member it holds the new member alone.
#[test]
fn crossing_in_brightness() {
    // Equal at frame 30; light 1 is 2% brighter, enough to take the fold, near frame 36.
    let at = |n: u32| {
        let t = n as f32 / 60.0;
        let a = tinted(dir(30.0, 0.0), Vec3::splat(1.05 - 0.1 * t));
        let b = tinted(dir(20.0, 150.0), Vec3::splat(0.95 + 0.1 * t));
        vec![(LightId(0), a), (LightId(1), b)]
    };
    let (steps, joined, luts) = fade_run(80, &at);
    let joined = joined.expect("light 1 joins the fold");
    let later = joined + 30;
    let mut alone = Rig::new(&earth());
    alone.sky.set_lights(&[LightId(1)]);
    alone.frame_lights(&at(later), EYE);
    let difference = lut_difference(&luts[later as usize], &lut(&alone));
    eprintln!(
        "crossing: joined at frame {joined}; worst frame-to-frame {:.2}% of the full sky, {:.1}% of the texel; \
         0.5 s later against the new member alone {difference:.2e}",
        steps.full * 100.0,
        steps.texel * 100.0
    );
    assert!((33..=40).contains(&joined), "{joined}");
    assert!(
        steps.full < 0.05,
        "sky stepped by {:.2}%",
        steps.full * 100.0
    );
    assert!(difference < 0.01, "{difference}");
    assert_clean();
}

/// A candidate sinking through the 18 deg cutoff fades out over 0.5 s and fades back in
/// when it rises through it, with no step in the sky.
#[test]
fn crossing_the_cutoff() {
    let mut rig = fading_rig(2);
    rig.sky.set_lights(&[LightId(0), LightId(1)]);
    // -17.5 deg down to -18.5 deg over frames 0-10, back up over frames 50-60.
    let at = |n: u32| {
        let e = if n < 50 {
            -17.5 - f64::from(n.min(10)) * 0.1
        } else {
            -18.5 + f64::from((n - 50).min(10)) * 0.1
        };
        vec![
            (LightId(0), tinted(dir(30.0, 0.0), Vec3::splat(0.2))),
            (LightId(1), solar_disc(dir(e, 90.0).as_vec3())),
        ]
    };
    let mut weights = Vec::new();
    let mut last: Option<Vec<[f64; 3]>> = None;
    let mut worst = 0.0_f64;
    for n in 0..100 {
        rig.frame_lights(&at(n), EYE);
        weights.push(fold_state(&rig).weight[1]);
        let next = lut(&rig);
        if let Some(last) = &last {
            worst = worst.max(lut_difference(&next, last));
        }
        last = Some(next);
    }
    eprintln!("cutoff: worst frame-to-frame {:.2}%", worst * 100.0);
    for pair in weights.windows(2) {
        assert!((pair[1] - pair[0]).abs() <= FADE_STEP + 1e-5, "{weights:?}");
    }
    assert_eq!(weights[0], 1.0);
    let out = weights.iter().position(|w| *w == 0.0).unwrap();
    assert!(
        (35..=40).contains(&out),
        "faded out at frame {out}: {weights:?}"
    );
    assert_eq!(weights[99], 1.0);
    assert!(worst < 0.05, "sky stepped by {:.2}%", worst * 100.0);
    assert_clean();
}

/// Two candidates of equal brightness swapping rank every frame give a steady sky.
#[test]
fn equal_candidates_swapping_rank() {
    let at = |n: u32| {
        let s = if n.is_multiple_of(2) { 1e-3 } else { -1e-3 };
        vec![
            (LightId(0), tinted(dir(30.0, 0.0), Vec3::splat(1.0 + s))),
            (LightId(1), tinted(dir(30.0, 180.0), Vec3::splat(1.0 - s))),
        ]
    };
    let (steps, joined, _) = fade_run(60, &at);
    eprintln!(
        "swapping: worst frame-to-frame {:.3}% of the texel",
        steps.texel * 100.0
    );
    assert_eq!(joined, None, "light 1 never joins the fold");
    assert!(
        steps.texel < 0.005,
        "sky stepped by {:.3}%",
        steps.texel * 100.0
    );
    assert_clean();
}
