//! Meter and adapt model (output.md M4) on a 0.1 EV/s luminance ramp.
//!
//! Meter: 128 bins over EV100 -12..20 (0.25 EV), mean EV inside the 10-90% percentile
//! window with fractional edge bins. Adapt: speed 3 stops/s brightening, 1 darkening,
//! linear beyond 1.5 EV of the target and exponential inside it. The analytic curve
//! adapts to the exact window mean of the same pixels. Meter variants: bin centres,
//! per-bin sums of log luminance in f32, and per-bin sums of the in-bin offset in 8-bit
//! fixed point (fits u32 atomics at 4K).

const BINS: usize = 128;
const EV_MIN: f64 = -12.0;
const EV_MAX: f64 = 20.0;
const BIN_W: f64 = (EV_MAX - EV_MIN) / BINS as f64;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Meter {
    Exact,
    BinCentre,
    BinSum,
    BinSumFixed8,
}

fn meter(evs: &[f64], m: Meter) -> f64 {
    let mut count = [0u64; BINS];
    let mut sum = [0.0f64; BINS];
    let mut fixed = [0u64; BINS];
    for &e in evs {
        let x = ((e - EV_MIN) / BIN_W).clamp(0.0, BINS as f64 - 1e-9);
        let b = x as usize;
        count[b] += 1;
        sum[b] += e;
        fixed[b] += ((x - b as f64) * 256.0).floor().min(255.0) as u64;
    }
    if m == Meter::Exact {
        let mut s: Vec<f64> = evs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        let (lo, hi) = ((n as f64 * 0.1) as usize, (n as f64 * 0.9) as usize);
        return s[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
    }
    let n = evs.len() as f64;
    let (lo, hi) = (0.1 * n, 0.9 * n);
    let (mut cum, mut acc, mut wsum) = (0.0, 0.0, 0.0);
    for b in 0..BINS {
        let c = count[b] as f64;
        if c == 0.0 {
            continue;
        }
        let w = (cum + c).min(hi) - cum.max(lo);
        cum += c;
        if w <= 0.0 {
            continue;
        }
        let ev = match m {
            Meter::BinCentre => EV_MIN + (b as f64 + 0.5) * BIN_W,
            Meter::BinSum => sum[b] / c,
            Meter::BinSumFixed8 => EV_MIN + (b as f64 + (fixed[b] as f64 / c + 0.5) / 256.0) * BIN_W,
            Meter::Exact => unreachable!(),
        };
        acc += w * ev;
        wsum += w;
    }
    acc / wsum
}

fn adapt(cur: f64, target: f64, dt: f64) -> f64 {
    let d = target - cur;
    let speed = if d > 0.0 { 3.0 } else { 1.0 };
    let t = 1.5;
    if d.abs() > t { cur + d.signum() * speed * dt } else { cur + d * (1.0 - (-speed * dt / t).exp()) }
}

/// Deterministic pixel EVs for a distribution: quantiles of a normal in EV (sigma), or
/// a mix of two.
fn image(kind: &str, n: usize) -> Vec<f64> {
    let inv = |p: f64| {
        // Acklam's inverse normal CDF.
        let a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
        let b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
        let c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
        let d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
        let pl = 0.02425;
        if p < pl {
            let q = (-2.0 * p.ln()).sqrt();
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        } else if p <= 1.0 - pl {
            let q = p - 0.5;
            let r = q * q;
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
    };
    // Shuffle-free stratified quantiles; a per-index jitter avoids aligned ties.
    let q = |i: usize| (i as f64 + 0.5 + 0.3 * ((i as f64 * 0.618034).fract() - 0.5)) / n as f64;
    match kind {
        "uniform" => vec![7.0; n],
        "narrow" => (0..n).map(|i| 7.0 + 0.1 * inv(q(i))).collect(),
        "natural" => (0..n).map(|i| 7.0 + 1.5 * inv(q(i))).collect(),
        "sky_ground" => (0..n).map(|i| if i % 10 < 3 { 11.0 + 0.3 * inv(q(i)) } else { 6.0 + 0.8 * inv(q(i)) }).collect(),
        _ => unreachable!(),
    }
}

pub fn run() {
    let n = 1 << 16;
    println!("0.1 EV/s ramp over 30 s after a settled start; per-frame step of the adapted EV against the analytic curve");
    println!("image      fps  meter          | max |step| | max |step - analytic step| | max |EV - analytic EV| | frames with deviation > 0.01");
    for kind in ["uniform", "narrow", "natural", "sky_ground"] {
        let base = image(kind, n);
        for fps in [30.0f64, 60.0, 120.0] {
            let dt = 1.0 / fps;
            let frames = (30.0 * fps) as usize;
            for m in [Meter::BinCentre, Meter::BinSum, Meter::BinSumFixed8] {
                // The image shifts uniformly, so the exact window mean is the base mean plus
                // the shift.
                let exact0 = meter(&base, Meter::Exact);
                let mut a = exact0;
                let mut a_m = meter(&base, m);
                // Settle both at the start of the ramp.
                for _ in 0..2000 {
                    a = adapt(a, exact0, dt);
                    a_m = adapt(a_m, meter(&base, m), dt);
                }
                let (mut max_step, mut max_dev, mut max_err, mut bad) = (0.0f64, 0.0f64, 0.0f64, 0);
                // Odd frame offsets keep the ramp from sitting on bin edges.
                for f in 1..=frames {
                    let shift = 0.1 * f as f64 * dt + 0.013;
                    let evs: Vec<f64> = base.iter().map(|e| e + shift).collect();
                    let na = adapt(a, exact0 + shift, dt);
                    let nm = adapt(a_m, meter(&evs, m), dt);
                    let (s, sa) = (nm - a_m, na - a);
                    max_step = max_step.max(s.abs());
                    max_dev = max_dev.max((s - sa).abs());
                    max_err = max_err.max((nm - na).abs());
                    if (s - sa).abs() > 0.01 {
                        bad += 1;
                    }
                    a = na;
                    a_m = nm;
                }
                println!(
                    "{kind:10} {fps:4.0} {:14} | {max_step:.4} | {max_dev:.4} | {max_err:.4} | {bad}",
                    format!("{m:?}")
                );
            }
        }
    }
}
