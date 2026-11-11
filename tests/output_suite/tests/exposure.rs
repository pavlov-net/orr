//! Auto and manual exposure on the GPU against CPU models of the meter and adapt.

use orr_color::Chromaticity;
use orr_output::{
    AutoExposure, CalibrationPattern, DisplayRequest, Exposure, ExposureTelemetry, Grade, MID_GRAY,
    MeterWeights, OutputDebug, OutputFormat, Physiological, ViewOutput, ev100_from_exposure,
    ev100_from_luminance, exposure_from_ev100, luminance_from_ev100,
};
use orr_scene::SkyStats;
use orr_scene::testing::assert_clean;
use output_suite::{Harness, adapt_step, keyed, natural_image, window_mean_ev100};

const SIDE: u32 = 128;
const FILTER: (f64, f64) = (0.10, 0.90);

/// A cut, whose frame sets the exposure, and a frame lit with it.
fn settle(h: &mut Harness, scale: f32, output: &ViewOutput) -> ExposureTelemetry {
    h.cut_next();
    h.frame(scale, output, None);
    h.frame(scale, output, None)
}

/// Exposed stops away from MID_GRAY of luminance `nits` at telemetry `t`.
fn stops_from_mid_gray(nits: f32, t: &ExposureTelemetry) -> f32 {
    (nits * exposure_from_ev100(t.adapted_ev100) / MID_GRAY).log2()
}

#[test]
fn a_cut_shows_a_uniform_field_at_mid_gray_on_its_own_frame() {
    let mut h = Harness::new(SIDE, SIDE);
    // Alternating extremes: lit with the exposure of the scene before it, a cut's frame would
    // clip the bright scenes at the HDR ceiling and underflow RGBA16F on the dark ones.
    for nits in [1e-4f32, 1e5, 1e-2, 1e3, 1e-3, 1e4, 0.1, 100.0, 1.0, 10.0] {
        h.set_image(&vec![nits; h.pixels()]);
        h.cut_next();
        let t = h.frame(1.0, &ViewOutput::default(), None);
        let off = stops_from_mid_gray(nits, &t);
        assert!(off.abs() < 0.01, "{nits} cd/m2: {off} stops from mid gray");
    }
    assert_clean();
}

/// A view sets its exposure once and then adapts. Through a history that keeps the cut's
/// frame as it was lit, at the reference, the frames after a cut meter many stops off; the
/// exposure moves from the cut's no faster than the adaptation speeds.
#[test]
fn after_a_cut_the_exposure_only_adapts() {
    let mut h = Harness::new(SIDE, SIDE);
    let output = ViewOutput::default();
    let auto = AutoExposure::default();
    let fastest = auto.speed_brighten.max(auto.speed_darken) * h.dt;
    h.history = 0.9;
    // Twelve stops brighter and ten darker than the reference keys.
    for nits in [1e4f32, 1e-3] {
        h.set_image(&vec![nits; h.pixels()]);
        h.cut_next();
        let cut = h.frame(1.0, &output, None);
        let off = stops_from_mid_gray(nits, &cut);
        assert!(off.abs() < 0.01, "{nits} cd/m2: {off} stops from mid gray");
        let mut last = cut.adapted_ev100;
        for f in 1..=30 {
            let t = h.frame(1.0, &output, None);
            let step = (t.adapted_ev100 - last).abs();
            assert!(
                step <= 1.001 * fastest,
                "{nits} cd/m2, frame {f} after the cut: {step} EV in one frame"
            );
            last = t.adapted_ev100;
        }
    }
    assert_clean();
}

/// An adapting view whose scene falls under RGBA16F at its pre-exposure holds its exposure on
/// the black frame and meters the light on the next, lit eight stops higher.
#[test]
fn an_adapting_view_finds_light_that_underflowed() {
    let mut h = Harness::new(SIDE, SIDE);
    let nits = 1e4f32;
    h.set_image(&vec![nits; h.pixels()]);
    let output = ViewOutput::default();
    let settled = settle(&mut h, 1.0, &output);
    let scale = (-26.0f32).exp2();
    let black = h.frame(scale, &output, None);
    assert_eq!(black.metered_ev100, None);
    assert_eq!(black.adapted_ev100, settled.adapted_ev100);
    let found = h.frame(scale, &output, None);
    let metered = found
        .metered_ev100
        .expect("light at the raised pre-exposure");
    // The light sits a dozen steps above zero in RGBA16F.
    let off = metered - ev100_from_luminance(nits * scale);
    assert!(off.abs() < 0.1, "metered {off} EV from the scene");
    assert!(found.adapted_ev100 < settled.adapted_ev100);
    assert_clean();
}

#[test]
fn bright_and_dark_scenes_fit_rgba16f() {
    let mut h = Harness::new(SIDE, SIDE);
    // Gray cards under 100,000 lux and 0.1 lux, spread +-8 stops.
    for lux in [1e5f32, 0.1] {
        let card = MID_GRAY * lux / std::f32::consts::PI;
        let nits = natural_image(h.pixels(), ev100_from_luminance(card), 2.0);
        h.set_image(&nits);
        let before = settle(&mut h, 1.0, &ViewOutput::default());
        h.frame(1.0, &ViewOutput::default(), None);
        // Every pixel is a normal RGBA16F value: within one f16 step of the exact
        // pre-exposed radiance.
        for (got, l) in h.radiance().iter().zip(&nits) {
            let exact = l * before.pre_exposure;
            let error = (got[0] / exact - 1.0).abs();
            assert!(error <= 1.0 / 1024.0, "{lux} lux: {got:?} against {exact}");
        }
    }
    assert_clean();
}

#[test]
fn black_first_frames_do_not_initialize() {
    let mut h = Harness::new(SIDE, SIDE);
    let nits = 50.0;
    h.set_image(&vec![nits; h.pixels()]);
    let auto = ViewOutput::default();
    for _ in 0..2 {
        let t = h.frame(0.0, &auto, None);
        assert_eq!(t.metered_ev100, None);
        assert_eq!(t.adapted_ev100, 0.0);
    }
    // The first frame with light, lit at the reference the black frames kept, takes its
    // target.
    let t = h.frame(1.0, &auto, None);
    let off = stops_from_mid_gray(nits, &t);
    assert!(off.abs() < 0.01, "{off} stops from mid gray");
    assert_clean();
}

#[test]
fn adaptation_follows_the_analytic_curve() {
    let mut h = Harness::new(SIDE, SIDE);
    let auto = AutoExposure::default();
    let output = ViewOutput::default();
    let dt = f64::from(h.dt);
    let nits = 10.0f32;
    h.set_image(&vec![nits; h.pixels()]);
    let mut prev = settle(&mut h, 1.0, &output);
    // Four stops brighter for 1.5 s, then four stops darker than the start for 4 s.
    for (scale, frames) in [(16.0f32, 45), (1.0 / 16.0, 120)] {
        let target = keyed(f64::from(ev100_from_luminance(nits * scale)), 0.0);
        for _ in 0..frames {
            let t = h.frame(scale, &output, None);
            assert!((f64::from(t.target_ev100) - target).abs() < 0.01);
            let from = f64::from(prev.adapted_ev100);
            let model = adapt_step(
                from,
                f64::from(t.target_ev100),
                f64::from(auto.speed_brighten),
                f64::from(auto.speed_darken),
                f64::from(auto.exponential_ev),
                dt,
            ) - from;
            let step = f64::from(t.adapted_ev100) - from;
            assert!(
                (step - model).abs() <= 0.02 * model.abs() + 2e-5,
                "step {step} against the model's {model}"
            );
            prev = t;
        }
    }

    // A 0.1 EV/s ramp moves exposure smoothly.
    let nits = natural_image(h.pixels(), 7.0, 1.5);
    h.set_image(&nits);
    let mut prev = settle(&mut h, 1.0, &output);
    for f in 1..=90 {
        let scale = (0.1 * f as f32 * h.dt).exp2();
        let t = h.frame(scale, &output, None);
        let step = (t.adapted_ev100 - prev.adapted_ev100).abs();
        assert!(step <= 0.01, "frame {f}: {step} EV step");
        prev = t;
    }
    assert_clean();
}

fn gray_sky(dome_ev100: f32, card_ev100: f32) -> SkyStats {
    let illuminance = luminance_from_ev100(card_ev100) * std::f32::consts::PI / MID_GRAY;
    SkyStats {
        dome_luminance: luminance_from_ev100(dome_ev100),
        dome_xy: [Chromaticity::D65.x as f32, Chromaticity::D65.y as f32],
        _pad: 0.0,
        directional: [illuminance, illuminance, illuminance, 0.0],
    }
}

#[test]
fn references_match_the_cpu_mix() {
    let mut h = Harness::new(SIDE, SIDE);
    let nits = natural_image(h.pixels(), 6.0, 1.5);
    let metered = window_mean_ev100(&nits, FILTER);
    h.set_image(&nits);
    let (dome, card) = (14.0f32, 4.0f32);
    let sky = gray_sky(dome, card);
    h.set_sky(sky);
    let weights = MeterWeights {
        histogram: 1.0,
        dome: 2.0,
        directional: 1.0,
    };
    let mix = (metered + 2.0 * f64::from(dome) + f64::from(card)) / 4.0;

    let single = ViewOutput {
        exposure: Exposure::Auto(AutoExposure {
            references: Some(weights),
            ..AutoExposure::default()
        }),
        ..ViewOutput::default()
    };
    let t = settle(&mut h, 1.0, &single);
    let metered_gpu = f64::from(t.metered_ev100.unwrap());
    assert!(
        (metered_gpu - metered).abs() < 0.01,
        "{metered_gpu} against {metered}"
    );
    let expected = keyed(mix, 0.0);
    assert!((f64::from(t.adapted_ev100) - expected).abs() < 0.05);
    assert_eq!(t.sky.dome_luminance, sky.dome_luminance);
    assert_eq!(t.sky.directional_lux, [sky.directional[0]; 3]);

    // Two stages: the mix sets the long-term exposure, which bounds the histogram's target.
    let physiological = Physiological {
        bound_brighten: 1.0,
        bound_darken: 1.5,
        ..Physiological::default()
    };
    let two = ViewOutput {
        exposure: Exposure::Auto(AutoExposure {
            references: Some(weights),
            physiological: Some(physiological),
            ..AutoExposure::default()
        }),
        ..ViewOutput::default()
    };
    h.cut_next();
    let t = h.frame(1.0, &two, None);
    let long_term = f64::from(t.long_term_ev100);
    assert!((long_term - expected).abs() < 0.05);
    // The histogram alone keys about 4.5 stops darker than the mix.
    assert!(keyed(metered, 0.0) < long_term - 1.5);
    let target = f64::from(t.target_ev100);
    assert!(
        (target - (long_term - 1.5)).abs() < 1e-4,
        "{target}, long term {long_term}"
    );
    assert_eq!(t.adapted_ev100, t.target_ev100);
    assert_clean();
}

#[test]
fn exposure_stays_inside_the_clamp_window() {
    let mut h = Harness::new(SIDE, SIDE);
    h.set_image(&natural_image(h.pixels(), 0.0, 1.0));
    let window = 4.0f32..=6.0;
    for physiological in [None, Some(Physiological::default())] {
        let output = ViewOutput {
            exposure: Exposure::Auto(AutoExposure {
                clamp_ev100: window.clone(),
                physiological,
                ..AutoExposure::default()
            }),
            ..ViewOutput::default()
        };
        settle(&mut h, 1.0, &output);
        // The scene sweeps 12 stops up and back, a quarter stop a frame.
        for f in (0..=48).chain((0..48).rev()) {
            let t = h.frame((0.25 * f as f32).exp2(), &output, None);
            for ev in [t.adapted_ev100, t.target_ev100] {
                assert!(window.contains(&ev), "frame {f}: {ev} outside {window:?}");
            }
        }
    }
    assert_clean();
}

#[test]
fn manual_exposure_shows_from_the_step_frame() {
    let mut h = Harness::new(SIDE, SIDE);
    let nits = 100.0f32;
    h.set_image(&vec![nits; h.pixels()]);
    let display = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
    // The grade's exposure and matrix together halve the image.
    let grade = Grade {
        exposure_ev: 1.0,
        matrix: [[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]],
    };
    for (frame, ev100) in [(1, 8.0f32), (2, 8.0), (3, 8.0), (4, 10.0), (5, 10.0)] {
        let output = ViewOutput {
            exposure: Exposure::Manual { ev100 },
            grade,
            debug: OutputDebug::Linear,
            ..ViewOutput::default()
        };
        let t = h.frame(1.0, &output, Some(&display));
        assert_eq!(t.adapted_ev100, ev100);
        let expected = 0.5 * nits * exposure_from_ev100(ev100);
        for p in h.display_linear() {
            for c in &p[..3] {
                assert!(
                    (c / expected - 1.0).abs() < 2e-3,
                    "frame {frame}: {c} against {expected}"
                );
            }
        }
    }
    assert_clean();
}

/// The scene's scale on frame `f` of a ramp: `stops` down over `frames`, held as long, then
/// back up over `frames` and held again.
fn down_and_up(f: u32, frames: u32, stops: f32) -> f32 {
    let at = match f / frames {
        0 => f as f32 / frames as f32,
        1 => 1.0,
        2 => 1.0 - (f - 2 * frames) as f32 / frames as f32,
        _ => 0.0,
    };
    (-stops * at).exp2()
}

#[test]
fn a_moving_pre_exposure_never_shows() {
    let mut h = Harness::new(SIDE, SIDE);
    let nits = natural_image(h.pixels(), 6.0, 1.0);
    h.set_image(&nits);
    let display = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
    // The scene moves six stops down at 3 stops/s and back; adapting as fast both ways, the
    // exposure follows it nearly all the way.
    let output = ViewOutput {
        exposure: Exposure::Auto(AutoExposure {
            speed_brighten: 3.0,
            speed_darken: 3.0,
            ..AutoExposure::default()
        }),
        debug: OutputDebug::Linear,
        ..ViewOutput::default()
    };
    let mut prev = settle(&mut h, 1.0, &output);
    let (mut lowest, mut highest) = (prev.adapted_ev100, prev.adapted_ev100);
    for f in 0..240 {
        let scale = down_and_up(f, 60, 6.0);
        let t = h.frame(scale, &output, Some(&display));
        // This frame, lit at the last one's pre-exposure, shows its own adapted exposure.
        let shown = h.display_linear()[0][1];
        let expected = nits[0] * scale * exposure_from_ev100(t.adapted_ev100);
        assert!(
            (shown / expected - 1.0).abs() < 2e-3,
            "frame {f}: {shown} against {expected}, lit at {}",
            prev.pre_exposure
        );
        lowest = lowest.min(t.adapted_ev100);
        highest = highest.max(t.adapted_ev100);
        prev = t;
    }
    assert!(highest - lowest > 5.0, "exposure moved {lowest}..{highest}");
    assert_clean();
}

/// Through a history that keeps its last image whatever its pre-exposure, as ray
/// reconstruction's does, the image changes from frame to frame by no more than the scene
/// and the adaptation rate move it. A step in pre-exposure would show as a flash.
#[test]
fn a_history_unaware_of_pre_exposure_shows_no_flash() {
    const MARGIN_EV: f32 = 0.05;
    let mut h = Harness::new(SIDE, SIDE);
    let nits = 50.0;
    h.set_image(&vec![nits; h.pixels()]);
    let display = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
    let output = ViewOutput {
        debug: OutputDebug::Linear,
        ..ViewOutput::default()
    };
    let auto = AutoExposure::default();
    let fastest = auto.speed_brighten.max(auto.speed_darken) * h.dt;
    // The history starts on frames lit with the exposure the cut set.
    settle(&mut h, 1.0, &output);
    h.history = 0.9;
    h.frame(1.0, &output, Some(&display));
    let (mut scene, mut shown) = (1.0f32, h.display_linear()[0][1]);
    let mut worst = 0.0f32;
    // Four stops down at 2 stops/s and back.
    for f in 0..240 {
        let next = down_and_up(f, 60, 4.0);
        h.frame(next, &output, Some(&display));
        let now = h.display_linear()[0][1];
        let change = (now / shown).log2().abs();
        let allowed = (next / scene).log2().abs() + fastest + MARGIN_EV;
        assert!(
            change <= allowed,
            "frame {f}: the image moved {change} EV, {allowed} allowed"
        );
        worst = worst.max(change - allowed + MARGIN_EV);
        (scene, shown) = (next, now);
    }
    log::info!("largest change past the scene's and the adaptation's: {worst} EV");
    assert_clean();
}

#[test]
fn debug_views() {
    let mut h = Harness::new(SIDE, SIDE / 2);
    let nits = 100.0f32;
    h.set_image(&vec![nits; h.pixels()]);
    let display = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
    // Half a stop above mid gray: an eighth of the way from green to red, at 85%.
    let ev100 = ev100_from_exposure(MID_GRAY * 0.5f32.exp2() / nits);
    let heatmap = ViewOutput {
        exposure: Exposure::Manual { ev100 },
        debug: OutputDebug::ExposureHeatmap,
        ..ViewOutput::default()
    };
    h.frame(1.0, &heatmap, Some(&display));
    let expected = [0.85 / 12.0, 0.85 * (1.0 - 0.9 / 12.0), 0.0];
    for p in h.display_linear() {
        for c in 0..3 {
            assert!(
                (p[c] - expected[c]).abs() < 2e-3,
                "{p:?} against {expected:?}"
            );
        }
    }

    let pattern = ViewOutput {
        debug: OutputDebug::Pattern(CalibrationPattern::PaperWhite { nits: 200.0 }),
        ..heatmap
    };
    h.frame(1.0, &pattern, Some(&display));
    let levels = CalibrationPattern::PaperWhite { nits: 200.0 }.levels(&display);
    let image = h.display_linear();
    let at = |x: u32, y: u32| image[(y * SIDE + x) as usize][0];
    let step = |v: f32| v / 1024.0;
    assert!((at(SIDE / 2, SIDE / 4) - levels.patch).abs() <= step(levels.patch));
    assert!((at(0, 0) - levels.surround).abs() <= step(levels.surround));
    assert_clean();
}

mod local_exposure {
    use orr_output::{
        DisplayRequest, Exposure, LocalExposure, OutputDebug, OutputFormat, ViewOutput,
        exposure_from_ev100, keyed_ev100, luminance_from_ev100,
    };
    use orr_scene::testing::assert_clean;
    use output_suite::Harness;

    use super::settle;

    /// EV100 `ev` with a quarter-stop checkerboard over it.
    fn checker(x: u32, y: u32, ev: f32) -> f32 {
        let d = if (x + y).is_multiple_of(2) {
            0.125
        } else {
            -0.125
        };
        luminance_from_ev100(ev + d)
    }

    fn with(strength: Option<f32>, output: &ViewOutput) -> ViewOutput {
        ViewOutput {
            local_exposure: strength.map(|strength| LocalExposure { strength }),
            ..output.clone()
        }
    }

    /// Manual exposure keying `ev100` to mid gray, shown linear.
    fn manual(ev100: f32) -> ViewOutput {
        ViewOutput {
            exposure: Exposure::Manual {
                ev100: keyed_ev100(ev100, 0.0),
            },
            debug: OutputDebug::Linear,
            ..ViewOutput::default()
        }
    }

    /// A quarter of sky, half middle ground and a quarter of deep shadow, each a
    /// quarter-stop checkerboard: with the meter on target, sky clips at the SDR peak and
    /// the shadow sits below 2^-10 of paper white.
    #[test]
    fn compresses_a_wide_scene() {
        let (w, h) = (256, 128);
        let mut harness = Harness::new(w, h);
        let mid = 10.0;
        let image: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| {
                let ev = match y * 4 / h {
                    0 => mid + 3.2,
                    3 => mid - 11.8,
                    _ => mid,
                };
                checker(x, y, ev)
            })
            .collect();
        harness.set_image(&image);
        let sdr = DisplayRequest::default().resolve(&[OutputFormat::Sdr], None, false);
        let shares = |harness: &mut Harness, strength| {
            let output = with(strength, &ViewOutput::default());
            settle(harness, 1.0, &output);
            harness.frame(1.0, &output, Some(&sdr));
            let image = harness.display_linear();
            let n = image.len() as f32;
            let peak = image.iter().filter(|p| p[1] >= 0.99).count() as f32 / n;
            let dark = image.iter().filter(|p| p[1] < 2f32.powi(-10)).count() as f32 / n;
            (peak, dark)
        };
        let (peak, dark) = shares(&mut harness, None);
        assert!(
            peak > 0.05 && dark > 0.05,
            "global: {peak} at peak, {dark} dark"
        );
        let (peak, dark) = shares(&mut harness, Some(0.5));
        assert!(
            peak < 0.05 && dark < 0.05,
            "local: {peak} at peak, {dark} dark"
        );
        assert_clean();
    }

    #[test]
    fn keeps_an_image_of_uniform_base() {
        let (w, h) = (128, 96);
        let mut harness = Harness::new(w, h);
        let hdr = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
        // EV100 9 is a grid bin's center.
        let ev = 9.0;
        let uniform = vec![luminance_from_ev100(ev); harness.pixels()];
        let checkered: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| checker(x, y, ev)))
            .collect();
        for image in [uniform, checkered] {
            harness.set_image(&image);
            let mut shown = |strength| {
                let output = with(strength, &manual(ev));
                harness.frame(1.0, &output, Some(&hdr));
                harness.frame(1.0, &output, Some(&hdr));
                harness.display_linear()
            };
            let global = shown(None);
            for strength in [0.5, 1.0] {
                for (a, b) in shown(Some(strength)).iter().zip(&global) {
                    let change = (a[1] / b[1] - 1.0).abs();
                    assert!(
                        change < 1e-3,
                        "strength {strength}: {} against {}",
                        a[1],
                        b[1]
                    );
                }
            }
        }
        assert_clean();
    }

    #[test]
    fn keeps_a_step_edge_sharp() {
        let (w, h) = (288, 64);
        let edge = 144;
        let mut harness = Harness::new(w, h);
        let hdr = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
        // Grid bin centers, ten stops apart, with the pivot between them.
        let (low, high) = (5.0, 15.0);
        let image: Vec<f32> = (0..h)
            .flat_map(|_| (0..w).map(|x| luminance_from_ev100(if x < edge { low } else { high })))
            .collect();
        harness.set_image(&image);
        let output = with(Some(0.5), &manual(0.5 * (low + high)));
        harness.frame(1.0, &output, Some(&hdr));
        let t = harness.frame(1.0, &output, Some(&hdr));
        let s = exposure_from_ev100(t.adapted_ev100);
        let shown = harness.display_linear();
        // Stops each pixel moved from global exposure.
        let moved = |x: u32, y: u32| {
            let i = (y * w + x) as usize;
            (shown[i][1] / (image[i] * s)).log2()
        };
        let across = moved(w - 1, h / 2) - moved(0, h / 2) + (high - low);
        assert!(
            (across - 5.0).abs() < 0.05,
            "{across} stops across the edge"
        );
        for y in 0..h {
            for x in (0..w).filter(|&x| x.abs_diff(edge) > 64) {
                let side = if x < edge { 2.5 } else { -2.5 };
                let off = moved(x, y) - side;
                assert!(off.abs() < 0.1, "({x}, {y}): {off} EV from its side");
            }
        }
        assert_clean();
    }

    #[test]
    fn turning_it_off_frees_the_grid() {
        let mut harness = Harness::new(64, 64);
        harness.set_image(&vec![100.0; harness.pixels()]);
        harness.frame(1.0, &with(Some(0.5), &ViewOutput::default()), None);
        assert!(harness.output().local_exposure_bytes() > 0);
        harness.frame(1.0, &ViewOutput::default(), None);
        assert_eq!(harness.output().local_exposure_bytes(), 0);
        assert_clean();
    }
}
