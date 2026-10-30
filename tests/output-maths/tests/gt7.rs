use orr_output::{GT7_REFERENCE_NITS, Gt7Curve, Gt7Params};
use output_maths::{gt7_reference, gt7_tonemap};

/// 4096 samples per channel and on the grey axis, log-spaced over 1e-4..1e4 frame-buffer
/// units (0.01 nits to 1e6 nits), with zero first.
fn ramps() -> Vec<[f32; 3]> {
    let mut v = Vec::new();
    for axis in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ] {
        for i in 0..4096 {
            let x = if i == 0 {
                0.0
            } else {
                10f32.powf(-4.0 + 8.0 * (i - 1) as f32 / 4094.0)
            };
            v.push(axis.map(|a: f32| a * x));
        }
    }
    v
}

/// Against Polyphony's sample built in f64. f32 PQ round trips limit both the Slang port and
/// the published f32 sample to about 1e-4 of the peak, so the port must do no worse than
/// the sample.
#[test]
fn gt7_matches_polyphony() {
    let params = Gt7Params::default();
    let input = ramps();
    // HDR at paper white 100 nits leaves the sample's frame-buffer units unscaled.
    for peak in [None, Some(1000.0), Some(4000.0), Some(10000.0)] {
        let curve = match peak {
            None => Gt7Curve::sdr(&params),
            Some(nits) => Gt7Curve::new(&params, nits, GT7_REFERENCE_NITS),
        };
        let out_peak = curve.peak * curve.output_scale;
        let got = gt7_tonemap(&curve, &input);
        let wide = gt7_reference(peak, true, &input);
        let narrow = gt7_reference(peak, false, &input);
        let worst = |a: &[[f32; 3]]| {
            let mut worst = (0.0f32, [0.0; 3]);
            for ((g, w), x) in a.iter().zip(&wide).zip(&input) {
                for c in 0..3 {
                    let e = (g[c] - w[c]).abs() / out_peak;
                    if e > worst.0 {
                        worst = (e, *x);
                    }
                }
            }
            worst
        };
        let (ours, sample) = (worst(&got), worst(&narrow).0);
        assert!(
            ours.0 < 2.5e-4 && ours.0 <= sample,
            "peak {peak:?}: error {} of the peak at {:?} (the f32 sample's {sample})",
            ours.0,
            ours.1
        );
    }
}
