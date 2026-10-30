//! Screenshot capture on the GPU: scene-referred captures read absolute luminance through the
//! pre-exposure, display captures hold the tonemapped image, and every capture carries the
//! id of the frame it holds.

use orr_output::{DisplayRequest, OutputFormat, Primaries, ScreenshotRequest, ViewOutput};
use orr_scene::testing::assert_clean;
use output_suite::{Harness, natural_image};

const REC2020_Y: [f64; 3] = [0.262_700_2, 0.677_998_1, 0.059_301_7];
const REC709_Y: [f64; 3] = [0.212_639, 0.715_168_7, 0.072_192_3];

fn luminance(p: &[f32], y: [f64; 3]) -> f64 {
    (0..3).map(|c| y[c] * f64::from(p[c])).sum()
}

#[test]
fn captures_carry_their_frames() {
    let mut h = Harness::new(64, 48);
    let nits = natural_image(h.pixels(), 12.0, 1.0);
    h.set_image(&nits);
    let display = DisplayRequest {
        hdr: false,
        ..DisplayRequest::default()
    }
    .resolve(&[OutputFormat::Sdr], None, false);
    let plain = ViewOutput::default();
    let mut telemetry = h.frame(1.0, &plain, Some(&display));
    for _ in 0..20 {
        telemetry = h.frame(1.0, &plain, Some(&display));
    }
    // The divide by pre-exposure is exercised.
    assert!(telemetry.pre_exposure.log2().abs() > 4.0);
    let first = telemetry.frame + 1;

    let shoot = |request| ViewOutput {
        screenshot: Some(request),
        ..ViewOutput::default()
    };
    let requests = [
        (1.0, ScreenshotRequest::SceneReferred { rec709: false }),
        (2.0, ScreenshotRequest::SceneReferred { rec709: true }),
        (1.0, ScreenshotRequest::Display { include_ui: false }),
        (1.5, ScreenshotRequest::Display { include_ui: false }),
    ];
    for (scale, request) in requests {
        h.frame(scale, &shoot(request), Some(&display));
    }
    let shots = h.screenshots();
    assert!(h.screenshots().is_empty());
    let frames: Vec<u64> = shots.iter().map(|s| s.frame).collect();
    assert_eq!(frames, (first..first + 4).collect::<Vec<_>>());

    for (shot, (scale, request)) in shots.iter().zip(requests) {
        assert_eq!(shot.size, [64, 48]);
        let ScreenshotRequest::SceneReferred { rec709 } = request else {
            continue;
        };
        assert!(shot.scene_referred);
        let (primaries, y) = if rec709 {
            (Primaries::Rec709, REC709_Y)
        } else {
            (Primaries::Rec2020, REC2020_Y)
        };
        assert_eq!(shot.primaries, primaries);
        for (i, p) in shot.rgba.chunks_exact(4).enumerate() {
            let want = f64::from(nits[i] * scale);
            let got = luminance(p, y);
            assert!(
                (got / want - 1.0).abs() < 1e-3,
                "frame {} texel {i}: {got} cd/m2 against {want}",
                shot.frame
            );
        }
    }

    // The last capture holds the last frame's tonemapped image as SDR shows it.
    let last = &shots[3];
    assert!(!last.scene_referred && !last.hdr);
    assert_eq!(last.primaries, Primaries::Rec709);
    assert_eq!(last.paper_white_nits, display.paper_white_nits);
    for (i, (got, linear)) in last
        .rgba
        .chunks_exact(4)
        .zip(h.display_linear())
        .enumerate()
    {
        for c in 0..3 {
            let want = linear[c].clamp(0.0, 1.0);
            assert!(
                (got[c] - want).abs() <= 1e-5,
                "display texel {i}: {got:?} against {linear:?}"
            );
        }
    }
    // Each display capture is its own frame's image.
    let mean = |s: &orr_output::Screenshot| s.rgba.iter().map(|&v| f64::from(v)).sum::<f64>();
    assert!(mean(&shots[3]) > mean(&shots[2]));
    assert_clean();
}
