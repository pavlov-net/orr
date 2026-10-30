//! Auto and manual exposure on the GPU against CPU models of the meter and adapt.

use orr_output::{
    AutoExposure, CalibrationPattern, DisplayRequest, Exposure, ExposureTelemetry, Grade, MID_GREY,
    MeterWeights, OutputDebug, OutputFormat, Physiological, ViewOutput, ev100_from_exposure,
    ev100_from_luminance, exposure_from_ev100, luminance_from_ev100,
};
use orr_scene::SkyStats;
use orr_scene::testing::assert_clean;
use output_suite::{Harness, adapt_step, keyed, natural_image, window_mean_ev100};

const SIDE: u32 = 128;
const FILTER: (f64, f64) = (0.10, 0.90);

/// A cut, then four frames: enough for the view to find light through underflow, snap until
/// two snaps agree, and adapt.
fn settle(h: &mut Harness, scale: f32, output: &ViewOutput) -> ExposureTelemetry {
    let cut = ViewOutput {
        cut: true,
        ..output.clone()
    };
    let mut t = h.frame(scale, &cut, None);
    for _ in 0..4 {
        t = h.frame(scale, output, None);
    }
    t
}

/// Exposed stops away from MID_GREY of luminance `nits` at telemetry `t`.
fn stops_from_mid_grey(nits: f32, t: &ExposureTelemetry) -> f32 {
    (nits * exposure_from_ev100(t.adapted_ev100) / MID_GREY).log2()
}

#[test]
fn uniform_field_converges_to_mid_grey_after_a_cut() {
    let mut h = Harness::new(SIDE, SIDE);
    // Alternating extremes, so a cut's first frame is lit with a pre-exposure many stops off:
    // the bright scenes clip at the HDR ceiling and the dark ones underflow RGBA16F.
    for nits in [1e-4f32, 1e5, 1e-2, 1e3, 1e-3, 1e4, 0.1, 100.0, 1.0, 10.0] {
        h.set_image(&vec![nits; h.pixels()]);
        let t = settle(&mut h, 1.0, &ViewOutput::default());
        let off = stops_from_mid_grey(nits, &t);
        assert!(off.abs() < 0.01, "{nits} cd/m2: {off} stops from mid grey");
    }
    assert_clean();
}

#[test]
fn bright_and_dark_scenes_fit_rgba16f() {
    let mut h = Harness::new(SIDE, SIDE);
    // Grey cards under 100,000 lux and 0.1 lux, spread +-8 stops.
    for lux in [1e5f32, 0.1] {
        let card = MID_GREY * lux / std::f32::consts::PI;
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
fn black_first_frames_do_not_initialise() {
    let mut h = Harness::new(SIDE, SIDE);
    let nits = 50.0;
    h.set_image(&vec![nits; h.pixels()]);
    let auto = ViewOutput::default();
    for _ in 0..2 {
        let t = h.frame(0.0, &auto, None);
        assert_eq!(t.metered_ev100, None);
        assert_eq!(t.adapted_ev100, 0.0);
    }
    // The first frame with light snaps to it, lit with the pre-exposure the black frames
    // raised, and the next corrects that snap.
    let t = h.frame(1.0, &auto, None);
    assert_eq!(t.adapted_ev100, t.target_ev100);
    let t = h.frame(1.0, &auto, None);
    let off = stops_from_mid_grey(nits, &t);
    assert!(off.abs() < 0.01, "{off} stops from mid grey");
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

fn grey_sky(dome_ev100: f32, card_ev100: f32) -> SkyStats {
    let illuminance = luminance_from_ev100(card_ev100) * std::f32::consts::PI / MID_GREY;
    SkyStats {
        dome_luminance: luminance_from_ev100(dome_ev100),
        dome_xy: [0.3127, 0.3290],
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
    let sky = grey_sky(dome, card);
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
    let t = settle(&mut h, 1.0, &two);
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

#[test]
fn pre_exposure_moves_in_whole_stops_and_never_shows() {
    let mut h = Harness::new(SIDE, SIDE);
    let nits = natural_image(h.pixels(), 6.0, 1.0);
    h.set_image(&nits);
    let display = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
    let output = ViewOutput {
        debug: OutputDebug::Linear,
        ..ViewOutput::default()
    };
    let mut prev = settle(&mut h, 1.0, &output);
    let mut steps = 0;
    // 3 stops/s up for 3 s, then down.
    for f in (0..=90).chain((0..90).rev()) {
        let scale = (3.0 * f as f32 * h.dt).exp2();
        let t = h.frame(scale, &output, Some(&display));
        let stops = (t.pre_exposure / prev.pre_exposure).log2();
        assert_eq!(stops.fract(), 0.0, "pre-exposure moved {stops} stops");
        steps += u32::from(stops != 0.0);
        let s = exposure_from_ev100(t.adapted_ev100);
        assert!((s / t.pre_exposure).log2().abs() <= 1.0 + 1e-4);
        // The image shows the adapted exposure whatever the pre-exposure.
        let shown = h.display_linear()[0][1];
        let expected = nits[0] * scale * s;
        assert!(
            (shown / expected - 1.0).abs() < 2e-3,
            "{shown} against {expected}"
        );
        prev = t;
    }
    assert!(steps >= 4, "{steps} pre-exposure steps");
    assert_clean();
}

#[test]
fn debug_views() {
    let mut h = Harness::new(SIDE, SIDE / 2);
    let nits = 100.0f32;
    h.set_image(&vec![nits; h.pixels()]);
    let display = DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false);
    // Half a stop above mid grey: an eighth of the way from green to red, at 85%.
    let ev100 = ev100_from_exposure(MID_GREY * 0.5f32.exp2() / nits);
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
