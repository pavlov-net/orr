//! Auto white balance on the GPU against the CPU model of adapt's white point and CAT16.

use orr_output::{
    AutoWhiteBalance, D65_XY, DisplayRequest, Exposure, ExposureTelemetry, MeterWeights,
    OutputDebug, OutputFormat, ResolvedDisplay, ViewOutput, WHITE_BALANCE_CCT, WHITE_BALANCE_DUV,
    cct_duv_from_xy, exposure_from_ev100, xy_from_cct_duv,
};
use orr_scene::SkyStats;
use orr_scene::testing::assert_clean;
use output_suite::{
    Harness, REC709_TO_XYZ, REC2020_TO_XYZ, cat16_rec2020, mul3, rec2020_from_xy, uv_from_xy,
    white_target, xy_from_xyz,
};

const SIDE: u32 = 64;
const FILTER: (f64, f64) = (0.10, 0.90);
/// CIE illuminant A, 2856 K.
const A_XY: [f64; 2] = [0.44757, 0.40745];
const HISTOGRAM_ONLY: MeterWeights = MeterWeights {
    histogram: 1.0,
    dome: 0.0,
    directional: 0.0,
};

fn d65() -> [f64; 2] {
    D65_XY.map(f64::from)
}

fn hdr() -> ResolvedDisplay {
    DisplayRequest::default().resolve(&[OutputFormat::Hdr10Pq], None, false)
}

fn balanced(white: AutoWhiteBalance) -> ViewOutput {
    ViewOutput {
        white_balance: Some(white),
        debug: OutputDebug::Linear,
        ..ViewOutput::default()
    }
}

/// A cut, then four frames, tonemapped.
fn settle(h: &mut Harness, output: &ViewOutput, display: &ResolvedDisplay) -> ExposureTelemetry {
    let cut = ViewOutput {
        cut: true,
        ..output.clone()
    };
    let mut t = h.frame(1.0, &cut, Some(display));
    for _ in 0..4 {
        t = h.frame(1.0, output, Some(display));
    }
    t
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn xy(v: [f32; 2]) -> [f64; 2] {
    v.map(f64::from)
}

/// The XYZ of a display-linear Rec.709 texel.
fn shown_xyz(p: [f32; 4]) -> [f64; 3] {
    mul3(&REC709_TO_XYZ, [p[0], p[1], p[2]].map(f64::from))
}

fn no_sky() -> SkyStats {
    bytemuck::Zeroable::zeroed()
}

#[test]
fn tungsten_lit_grey_shows_as_d65() {
    let mut h = Harness::new(SIDE, SIDE);
    h.colour = rec2020_from_xy(A_XY);
    let nits = 100.0;
    h.set_image(&vec![nits; h.pixels()]);
    let display = hdr();
    // Manual exposure meters the chromaticity alone.
    for exposure in [Exposure::default(), Exposure::Manual { ev100: 9.0 }] {
        let output = ViewOutput {
            exposure,
            ..balanced(AutoWhiteBalance::default())
        };
        let t = settle(&mut h, &output, &display);
        let model = white_target(&[(nits, h.colour)], FILTER, HISTOGRAM_ONLY, 1.0, &no_sky());
        let white = xy(t.white_point_xy);
        assert!(distance(white, model) < 5e-4, "{white:?} against {model:?}");
        for p in h.display_linear() {
            let shown = xy_from_xyz(shown_xyz(p));
            assert!(distance(shown, d65()) < 0.005, "shown at {shown:?}");
        }
    }
    assert_clean();
}

#[test]
fn d65_field_is_left_unchanged() {
    let mut h = Harness::new(SIDE, SIDE);
    let hues = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.6, 0.4],
    ];
    // Grey at 100 cd/m2, and in the top 5% of luminance (outside the filter) saturated
    // colours, which show whether the matrix is the identity.
    let coloured = h.pixels() / 20;
    let mut nits = vec![100.0; h.pixels()];
    let mut colours = vec![[1.0; 3]; h.pixels()];
    for i in 0..coloured {
        nits[i] = 1000.0;
        colours[i] = hues[i % hues.len()];
    }
    h.set_image(&nits);
    h.set_colours(Some(&colours));
    let display = hdr();
    let output = balanced(AutoWhiteBalance::default());
    let plain = ViewOutput {
        white_balance: None,
        ..output.clone()
    };
    settle(&mut h, &output, &display);
    h.frame(1.0, &plain, Some(&display));
    let reference = h.display_linear();
    let t = h.frame(1.0, &output, Some(&display));
    assert!(distance(xy(t.white_point_xy), d65()) < 1e-4);
    for (got, want) in h.display_linear().iter().zip(&reference) {
        let scale = want[..3].iter().fold(0.0f32, |m, c| m.max(c.abs()));
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() <= 1e-3 * scale,
                "{got:?} against {want:?}"
            );
        }
    }
    assert_clean();
}

#[test]
fn white_point_and_correction_match_the_cpu_model() {
    let mut h = Harness::new(SIDE, SIDE);
    let display = hdr();
    // Clamped: below and above the CCT range, and above and below the Duv limit.
    let fields = [
        xy_from_cct_duv(1800.0, 0.0),
        xy_from_cct_duv(9000.0, 0.0),
        xy_from_cct_duv(5000.0, 0.08),
        xy_from_cct_duv(4000.0, -0.08),
    ];
    for field in fields {
        let field = xy(field);
        h.colour = rec2020_from_xy(field);
        h.set_image(&vec![100.0; h.pixels()]);
        let t = settle(&mut h, &balanced(AutoWhiteBalance::default()), &display);
        let model = white_target(&[(100.0, h.colour)], FILTER, HISTOGRAM_ONLY, 1.0, &no_sky());
        let white = xy(t.white_point_xy);
        assert!(distance(white, model) < 5e-4, "{white:?} against {model:?}");
        let (cct, duv) = cct_duv_from_xy(t.white_point_xy);
        let on_bound = (cct - WHITE_BALANCE_CCT.start()).abs() < 10.0
            || (cct - WHITE_BALANCE_CCT.end()).abs() < 10.0
            || (duv.abs() - WHITE_BALANCE_DUV).abs() < 1e-3;
        assert!(on_bound, "{field:?} held at {cct} K, Duv {duv}");
    }

    // Four patches of their own colour and luminance, with the sky's dome and directional
    // light mixed in.
    let patches = [
        (30.0f32, xy_from_cct_duv(2700.0, 0.004)),
        (120.0, [0.30, 0.34]),
        (500.0, [0.35, 0.37]),
        (2000.0, [0.28, 0.29]),
    ];
    let pixel = |i: usize| {
        let (nits, xy) = patches[i * patches.len() / (SIDE * SIDE) as usize];
        (nits, rec2020_from_xy(xy.map(f64::from)))
    };
    let pixels: Vec<(f32, [f32; 3])> = (0..h.pixels()).map(pixel).collect();
    h.set_image(&pixels.iter().map(|p| p.0).collect::<Vec<_>>());
    h.set_colours(Some(&pixels.iter().map(|p| p.1).collect::<Vec<_>>()));
    let sun = rec2020_from_xy(xy(xy_from_cct_duv(5000.0, 0.0))).map(|c| 5e4 * c);
    let sky = SkyStats {
        dome_luminance: 3000.0,
        dome_xy: [0.27, 0.28],
        _pad: 0.0,
        directional: [sun[0], sun[1], sun[2], 0.0],
    };
    h.set_sky(sky);
    let weights = MeterWeights {
        histogram: 1.0,
        dome: 0.5,
        directional: 0.25,
    };
    let white = AutoWhiteBalance {
        references: Some(weights),
        virtual_light_nits: 20.0,
        ..AutoWhiteBalance::default()
    };
    let t = settle(&mut h, &balanced(white), &display);
    let model = white_target(&pixels, FILTER, weights, 20.0, &sky);
    let got = xy(t.white_point_xy);
    assert!(distance(got, model) < 5e-4, "{got:?} against {model:?}");

    // Each patch shows as CAT16 from the white point to D65 of its exposed radiance.
    let m = cat16_rec2020(got, d65());
    let s = f64::from(exposure_from_ev100(t.adapted_ev100));
    let image = h.display_linear();
    for (p, (nits, colour)) in image.iter().zip(&pixels) {
        let exposed = colour.map(|c| f64::from(c * nits) * s);
        let want = mul3(&REC2020_TO_XYZ, mul3(&m, exposed));
        let shown = shown_xyz(*p);
        for c in 0..3 {
            assert!(
                (shown[c] - want[c]).abs() <= 2e-3 * want[1],
                "{shown:?} against {want:?}"
            );
        }
    }
    assert_clean();
}

#[test]
fn white_point_adapts_at_its_speed_holds_in_the_dark_and_snaps_on_a_cut() {
    let mut h = Harness::new(SIDE, SIDE);
    h.set_image(&vec![100.0; h.pixels()]);
    let display = hdr();
    let speed = 2.0;
    let output = balanced(AutoWhiteBalance {
        speed,
        ..AutoWhiteBalance::default()
    });
    let mut prev = settle(&mut h, &output, &display);
    assert!(distance(xy(prev.white_point_xy), d65()) < 1e-4);

    h.colour = rec2020_from_xy(A_XY);
    let k = 1.0 - (-f64::from(speed * h.dt)).exp();
    for f in 0..30 {
        let t = h.frame(1.0, &output, Some(&display));
        let (from, to) = (
            uv_from_xy(xy(prev.white_point_xy)),
            uv_from_xy(xy(t.white_target_xy)),
        );
        let want = [
            from[0] + k * (to[0] - from[0]),
            from[1] + k * (to[1] - from[1]),
        ];
        let got = uv_from_xy(xy(t.white_point_xy));
        assert!(
            distance(got, want) < 2e-5,
            "frame {f}: {got:?} against {want:?}"
        );
        prev = t;
    }
    assert!(distance(xy(prev.white_point_xy), xy(prev.white_target_xy)) > 1e-3);

    let t = h.frame(0.0, &output, Some(&display));
    assert_eq!(t.white_point_xy, prev.white_point_xy);

    h.colour = [1.0; 3];
    let cut = ViewOutput {
        cut: true,
        ..output.clone()
    };
    let t = h.frame(1.0, &cut, Some(&display));
    assert_eq!(t.white_point_xy, t.white_target_xy);
    assert!(distance(xy(t.white_point_xy), d65()) < 1e-4);
    assert_clean();
}
