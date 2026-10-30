//! SDR gamut handling after GT7: clipping, ACES 1.3 Reference Gamut Compression, and
//! chroma compression in GT7's UCS (ICtCp), on Rec.2020 primary and secondary ramps.
//! Hue drift is the ICtCp hue of the displayed colour against the input's hue.

// GT7 reference (Polyphony Digital sample, MIT), ported for the CPU study.

const REF_LUM: f32 = 100.0;

fn smooth_step(x: f32, e0: f32, e1: f32) -> f32 {
    let t = (x - e0) / (e1 - e0);
    if x < e0 {
        0.0
    } else if x > e1 {
        1.0
    } else {
        t * t * (3.0 - 2.0 * t)
    }
}

struct Curve {
    peak: f32,
    mid: f32,
    linear: f32,
    toe: f32,
    ka: f32,
    kb: f32,
    kc: f32,
}
impl Curve {
    fn new(peak: f32, alpha: f32, mid: f32, linear: f32, toe: f32) -> Self {
        let k = (linear - 1.0) / (alpha - 1.0);
        Curve { peak, mid, linear, toe, ka: peak * linear + peak * k, kb: -peak * k * (linear / k).exp(), kc: -1.0 / (k * peak) }
    }
    fn eval(&self, x: f32) -> f32 {
        if x < 0.0 {
            return 0.0;
        }
        let wl = smooth_step(x, 0.0, self.mid);
        let wt = 1.0 - wl;
        let shoulder = self.ka + self.kb * (x * self.kc).exp();
        if x < self.linear * self.peak {
            let toe = self.mid * (x / self.mid).powf(self.toe);
            wt * toe + wl * x
        } else {
            shoulder
        }
    }
}

fn pq_inv(v: f32) -> f32 {
    let (m1, m2, c1, c2, c3) = (0.1593017578125f32, 78.84375f32, 0.8359375f32, 18.8515625f32, 18.6875f32);
    let y = (v * REF_LUM / 10000.0).max(0.0);
    let ym = y.powf(m1);
    (m2 * ((c1 + c2 * ym).log2() - (1.0 + c3 * ym).log2())).exp2()
}
fn pq(n: f32) -> f32 {
    let (m1, m2, c1, c2, c3) = (0.1593017578125f32, 78.84375f32, 0.8359375f32, 18.8515625f32, 18.6875f32);
    let n = n.clamp(0.0, 1.0);
    let np = n.powf(1.0 / m2);
    let l = ((np - c1).max(0.0) / (c2 - c3 * np)).powf(1.0 / m1);
    l * 10000.0 / REF_LUM
}
/// Linear Rec.2020 frame-buffer units (1.0 = 100 nits) to ICtCp (ITU-T T.302).
pub fn ictcp(rgb: [f32; 3]) -> [f32; 3] {
    let l = (rgb[0] * 1688.0 + rgb[1] * 2146.0 + rgb[2] * 262.0) / 4096.0;
    let m = (rgb[0] * 683.0 + rgb[1] * 2951.0 + rgb[2] * 462.0) / 4096.0;
    let s = (rgb[0] * 99.0 + rgb[1] * 309.0 + rgb[2] * 3688.0) / 4096.0;
    let (l, m, s) = (pq_inv(l), pq_inv(m), pq_inv(s));
    [(2048.0 * l + 2048.0 * m) / 4096.0, (6610.0 * l - 13613.0 * m + 7003.0 * s) / 4096.0, (17933.0 * l - 17390.0 * m - 543.0 * s) / 4096.0]
}
/// No clamp, so out-of-gamut values survive for the gamut search.
fn ictcp_inv(c: [f32; 3]) -> [f32; 3] {
    let l = c[0] + 0.00860904 * c[1] + 0.11103 * c[2];
    let m = c[0] - 0.00860904 * c[1] - 0.11103 * c[2];
    let s = c[0] + 0.560031 * c[1] - 0.320627 * c[2];
    let (l, m, s) = (pq(l), pq(m), pq(s));
    [3.43661 * l - 2.50645 * m + 0.0698454 * s, -0.79133 * l + 1.9836 * m - 0.192271 * s, -0.0259499 * l - 0.0989137 * m + 1.12486 * s]
}

pub struct Gt7 {
    curve: Curve,
    sdr_scale: f32,
    target: f32,
    target_ucs: f32,
}
impl Gt7 {
    pub fn sdr() -> Self {
        let target = 250.0 / REF_LUM;
        let curve = Curve::new(target, 0.25, 0.538, 0.444, 1.280);
        Gt7 { curve, sdr_scale: 1.0 / target, target, target_ucs: ictcp([target; 3])[0] }
    }
    /// Linear Rec.2020 frame-buffer units in; SDR Rec.2020 out, 1.0 = paper white.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let ucs = ictcp(rgb);
        let skew = rgb.map(|c| self.curve.eval(c));
        let sk = ictcp(skew);
        let cs = 1.0 - smooth_step(ucs[0] / self.target_ucs, 0.98, 1.16);
        let mut scaled = ictcp_inv([sk[0], ucs[1] * cs, ucs[2] * cs]);
        for v in &mut scaled {
            *v = v.max(0.0);
        }
        [0, 1, 2].map(|i| self.sdr_scale * (0.4 * skew[i] + 0.6 * scaled[i]).min(self.target))
    }
}

const BT2020_TO_709: [[f32; 3]; 3] = [[1.660491, -0.587641, -0.072850], [-0.124550, 1.132900, -0.008349], [-0.018151, -0.100579, 1.118730]];
const BT709_TO_2020: [[f32; 3]; 3] = [[0.627404, 0.329283, 0.043313], [0.069097, 0.919540, 0.011362], [0.016391, 0.088013, 0.895595]];
fn mat(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// ACES 1.3 Reference Gamut Compression on linear RGB (distance limits per channel).
fn rgc(rgb: [f32; 3], thr: [f32; 3], lim: [f32; 3]) -> [f32; 3] {
    let p = 1.2f32;
    let ach = rgb[0].max(rgb[1]).max(rgb[2]);
    if ach == 0.0 {
        return rgb;
    }
    let mut out = rgb;
    for i in 0..3 {
        let d = (ach - rgb[i]) / ach.abs();
        let s = (lim[i] - thr[i]) / (((1.0 - thr[i]) / (lim[i] - thr[i])).powf(-p) - 1.0).powf(1.0 / p);
        let dc = if d < thr[i] { d } else { thr[i] + s * ((d - thr[i]) / s) / (1.0 + ((d - thr[i]) / s).powf(p)).powf(1.0 / p) };
        out[i] = ach - dc * ach.abs();
    }
    out
}

/// Largest chroma scale at constant ICtCp intensity that fits Rec.709 [0, 1]; the
/// intensity is first lowered to that of the clipped colour's maximum white.
fn ucs_compress(sdr2020: [f32; 3]) -> [f32; 3] {
    let fits = |c: [f32; 3]| {
        let r = mat(&BT2020_TO_709, c);
        r.iter().all(|&v| (-1e-5..=1.0 + 1e-5).contains(&v))
    };
    if fits(sdr2020) {
        return sdr2020;
    }
    // ICtCp in absolute units: SDR 1.0 = 250 nits = 2.5 frame-buffer units.
    let mut u = ictcp(sdr2020.map(|v| v * 2.5));
    let white = ictcp([2.5; 3])[0];
    u[0] = u[0].min(white);
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        let c = ictcp_inv([u[0], u[1] * m, u[2] * m]).map(|v| v / 2.5);
        if fits(c) { lo = m } else { hi = m }
    }
    ictcp_inv([u[0], u[1] * lo, u[2] * lo]).map(|v| v / 2.5)
}

fn hue(c: [f32; 3]) -> (f32, f32) {
    (c[2].atan2(c[1]).to_degrees(), (c[1] * c[1] + c[2] * c[2]).sqrt())
}
fn hue_diff(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

/// Rec.2020 RGC distance limits: the largest distance any Rec.2020 colour reaches in
/// Rec.709 per channel (from the Rec.2020 primaries).
fn rec2020_limits() -> [f32; 3] {
    let mut lim = [1.0f32; 3];
    for p in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]] {
        let r = mat(&BT2020_TO_709, p);
        let ach = r[0].max(r[1]).max(r[2]);
        for i in 0..3 {
            lim[i] = lim[i].max((ach - r[i]) / ach + 0.001);
        }
    }
    lim
}

pub fn run() {
    let gt7 = Gt7::sdr();
    let lim2020 = rec2020_limits();
    println!("Rec.2020 boundary distances in Rec.709 (RGC limits): {lim2020:.3?}");
    let aces_thr = [0.815, 0.803, 0.880];
    let aces_lim = [1.147, 1.264, 1.312];
    let ramps: [(&str, [f32; 3]); 6] = [
        ("red", [1.0, 0.0, 0.0]),
        ("green", [0.0, 1.0, 0.0]),
        ("blue", [0.0, 0.0, 1.0]),
        ("cyan", [0.0, 1.0, 1.0]),
        ("magenta", [1.0, 0.0, 1.0]),
        ("yellow", [1.0, 1.0, 0.0]),
    ];
    let lum = |c: [f32; 3]| 0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2];
    let methods = ["clip", "rgc_aces", "rgc_2020", "ucs"];
    println!("\nmax / mean ICtCp hue drift (deg) over 0.01-10,000 nits, points with displayed chroma >= 0.02");
    println!("{:8} {:>16} {:>16} {:>16} {:>16} {:>16}", "ramp", methods[0], methods[1], methods[2], methods[3], "gt7 unclipped");
    let mut totals = [[0.0f32; 2]; 5];
    for (name, c) in ramps {
        let mut stats = [[0.0f32, 0.0, 0.0]; 5]; // max, sum, n
        let steps = 241;
        for k in 0..steps {
            let nits = 10f32.powf(-2.0 + 6.0 * k as f32 / (steps - 1) as f32);
            let input = c.map(|v| v * nits / REF_LUM / lum(c));
            let sdr = gt7.apply(input);
            let r709 = mat(&BT2020_TO_709, sdr);
            let outs: [[f32; 3]; 5] = [
                r709.map(|v| v.clamp(0.0, 1.0)),
                rgc(r709, aces_thr, aces_lim).map(|v| v.clamp(0.0, 1.0)),
                rgc(r709, aces_thr, lim2020).map(|v| v.clamp(0.0, 1.0)),
                mat(&BT2020_TO_709, ucs_compress(sdr)).map(|v| v.clamp(0.0, 1.0)),
                r709,
            ];
            for (m, o) in outs.iter().enumerate() {
                // Displayed colour back to absolute Rec.2020 (1.0 = 250 nits), against the
                // input chromaticity at the displayed luminance.
                let disp = mat(&BT709_TO_2020, *o).map(|v| v * 2.5);
                let (h, chroma) = hue(ictcp(disp));
                let (h_in, _) = hue(ictcp(input.map(|v| v * lum(disp) / lum(input))));
                if chroma >= 0.02 {
                    let d = hue_diff(h, h_in);
                    stats[m][0] = stats[m][0].max(d);
                    stats[m][1] += d;
                    stats[m][2] += 1.0;
                }
            }
        }
        let cell = |s: [f32; 3]| format!("{:6.2} / {:5.2}", s[0], s[1] / s[2].max(1.0));
        println!("{name:8} {:>16} {:>16} {:>16} {:>16} {:>16}", cell(stats[0]), cell(stats[1]), cell(stats[2]), cell(stats[3]), cell(stats[4]));
        for m in 0..5 {
            totals[m][0] = totals[m][0].max(stats[m][0]);
            totals[m][1] += stats[m][1] / stats[m][2].max(1.0) / 6.0;
        }
    }
    println!("all      max {:.2} {:.2} {:.2} {:.2} (gt7 alone {:.2}); mean of ramp means {:.2} {:.2} {:.2} {:.2}", totals[0][0], totals[1][0], totals[2][0], totals[3][0], totals[4][0], totals[0][1], totals[1][1], totals[2][1], totals[3][1]);
    // Port check against Polyphony's sample on the same ramps.
    unsafe extern "C" {
        fn orr_gt7_ref_sdr(i: *const f32, o: *mut f32);
    }
    let mut max_err = 0.0f32;
    for (_, c) in ramps {
        for k in 0..241 {
            let nits = 10f32.powf(-2.0 + 6.0 * k as f32 / 240.0);
            let input = c.map(|v| v * nits / REF_LUM / lum(c));
            let mut r = [0.0f32; 3];
            unsafe { orr_gt7_ref_sdr(input.as_ptr(), r.as_mut_ptr()) };
            let o = gt7.apply(input);
            for i in 0..3 {
                max_err = max_err.max((o[i] - r[i]).abs());
            }
        }
    }
    println!("GT7 port vs Polyphony sample, max abs error over the ramps: {max_err:.2e}");
}

/// Chroma compression in ICtCp at constant intensity and hue with a knee: the ratio of
/// chroma to the Rec.709 boundary chroma at that intensity and hue passes unchanged below
/// `knee` and approaches 1 above it (tanh). The boundary is found by bisection.
fn ucs_knee(sdr2020: [f32; 3], knee: f32) -> [f32; 3] {
    let fits = |c: [f32; 3]| {
        let r = mat(&BT2020_TO_709, c);
        r.iter().all(|&v| (-1e-6..=1.0 + 1e-6).contains(&v))
    };
    let mut u = ictcp(sdr2020.map(|v| v * 2.5));
    let white = ictcp([2.5; 3])[0];
    u[0] = u[0].min(white);
    let chroma = (u[1] * u[1] + u[2] * u[2]).sqrt();
    if chroma == 0.0 {
        return ictcp_inv(u).map(|v| v / 2.5);
    }
    // Boundary chroma along the hue at this intensity.
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    let at = |c: f32| ictcp_inv([u[0], u[1] / chroma * c, u[2] / chroma * c]).map(|v| v / 2.5);
    while fits(at(hi)) && hi < 64.0 {
        hi *= 2.0;
    }
    for _ in 0..40 {
        let m = 0.5 * (lo + hi);
        if fits(at(m)) { lo = m } else { hi = m }
    }
    let r = chroma / lo;
    let out = if r <= knee { r } else { knee + (1.0 - knee) * ((r - knee) / (1.0 - knee)).tanh() };
    at(out * lo)
}

/// Saturation sweep: at fixed ICtCp hue and intensity, chroma rises from 0 to the Rec.2020
/// boundary; each input goes through GT7 SDR and a gamut method. Reports, over the samples
/// whose GT7 output lies outside Rec.709 [0, 1], how many consecutive pairs collapse to one
/// displayed colour, how many reverse the order of displayed chroma, and the share of the
/// displayed chroma range the out-of-gamut part spans. Runs over every out-of-gamut sample,
/// then over those with a channel below zero and none above one (what gamut compression
/// handles; the [0, 1] clamp takes the rest).
pub fn sweep() {
    sweep_over(false);
    sweep_over(true);
}

fn sweep_over(below_zero_only: bool) {
    let gt7 = Gt7::sdr();
    let aces_thr = [0.815, 0.803, 0.880];
    let aces_lim = [1.147, 1.264, 1.312];
    let lim2020 = rec2020_limits();
    let hues: Vec<[f32; 3]> = vec![
        [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.5, 1.0],
        [0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.5],
    ];
    let levels = [1.0f32, 10.0, 50.0, 100.0, 250.0, 1000.0, 4000.0];
    let lum = |c: [f32; 3]| 0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2];
    let names = ["clip", "rgc_aces", "ucs_knee_0.8", "rgc_2020"];
    let steps = 512;
    // per method: oog samples, collapsed pairs, order reversals, span sum, span n, hue drift sum, hue n, hue max
    let mut acc = [[0.0f64; 8]; 4];
    let mut worst_rev = [0.0f32; 4];
    for &h in &hues {
        for &nits in &levels {
            let base = h.map(|v| v * nits / REF_LUM / lum(h));
            let ub = ictcp(base.map(|v| v * REF_LUM / 100.0));
            let mut prev: Option<([f32; 3], f32, [[f32; 3]; 4], [f32; 4])> = None;
            let mut span: [(f32, f32); 4] = [(f32::INFINITY, f32::NEG_INFINITY); 4];
            let mut total_range: [(f32, f32); 4] = [(f32::INFINITY, f32::NEG_INFINITY); 4];
            for k in 0..=steps {
                let s = k as f32 / steps as f32;
                let input = ictcp_inv([ub[0], ub[1] * s, ub[2] * s]).map(|v| v.max(0.0));
                let sdr = gt7.apply(input);
                let r709 = mat(&BT2020_TO_709, sdr);
                let (_, c_pre) = hue(ictcp(sdr.map(|v| v * 2.5)));
                let oog = r709.iter().any(|&v| !(-1e-6..=1.0 + 1e-6).contains(&v));
                let oog = oog && (!below_zero_only || r709.iter().all(|&v| v <= 1.0 + 1e-6));
                let outs: [[f32; 3]; 4] = [
                    r709.map(|v| v.clamp(0.0, 1.0)),
                    rgc(r709, aces_thr, aces_lim).map(|v| v.clamp(0.0, 1.0)),
                    mat(&BT2020_TO_709, ucs_knee(sdr, 0.8)).map(|v| v.clamp(0.0, 1.0)),
                    rgc(r709, aces_thr, lim2020).map(|v| v.clamp(0.0, 1.0)),
                ];
                let mut cs = [0.0f32; 4];
                for (m, o) in outs.iter().enumerate() {
                    let disp = mat(&BT709_TO_2020, *o).map(|v| v * 2.5);
                    let (hd, c) = hue(ictcp(disp));
                    cs[m] = c;
                    total_range[m] = (total_range[m].0.min(c), total_range[m].1.max(c));
                    if !oog {
                        continue;
                    }
                    span[m] = (span[m].0.min(c), span[m].1.max(c));
                    if c >= 0.02 && s > 0.0 {
                        let (hb, _) = hue(ub);
                        let d = hue_diff(hd, hb);
                        acc[m][5] += f64::from(d);
                        acc[m][6] += 1.0;
                        acc[m][7] = acc[m][7].max(f64::from(d));
                    }
                    if let Some((p709, pc_pre, pouts, pcs)) = prev {
                        let pre_diff = (0..3).map(|i| (r709[i] - p709[i]).abs()).fold(0.0f32, f32::max);
                        if pre_diff >= 0.5 / 1023.0 {
                            acc[m][0] += 1.0;
                            let diff = (0..3).map(|i| (o[i] - pouts[m][i]).abs()).fold(0.0f32, f32::max);
                            // A channel below zero in both samples that GT7 moved but the
                            // method leaves equal (pinned at the boundary).
                            let pinned = (0..3).any(|i| {
                                r709[i] < 0.0 && p709[i] < 0.0 && (r709[i] - p709[i]).abs() > 1e-5 && (o[i] - pouts[m][i]).abs() < 1e-7
                            });
                            if diff < 1e-6 || pinned {
                                acc[m][1] += 1.0;
                            }
                        }
                        if c_pre > pc_pre + 1e-4 && c < pcs[m] - 1e-4 {
                            acc[m][2] += 1.0;
                            worst_rev[m] = worst_rev[m].max(pcs[m] - c);
                        }
                    }
                }
                prev = oog.then_some((r709, c_pre, outs, cs));
            }
            for m in 0..4 {
                if span[m].1 > span[m].0 && total_range[m].1 > 0.0 {
                    acc[m][3] += f64::from((span[m].1 - span[m].0) / total_range[m].1);
                    acc[m][4] += 1.0;
                }
            }
        }
    }
    let domain = if below_zero_only { "below-zero samples" } else { "all out-of-gamut samples" };
    println!("
saturation sweep, {domain}: 12 hues x {} levels (nits), {} chroma steps to the Rec.2020 boundary, GT7 SDR", levels.len(), steps);
    println!("pairs = consecutive out-of-gamut samples GT7 keeps apart by half a 10-bit code; collapsed = of those, displayed equal, or a below-zero channel GT7 moved pinned at the boundary; reversed = GT7 chroma rises, displayed chroma falls (by 1e-4)");
    println!("{:14} {:>8} {:>10} {:>9} {:>11} {:>12} {:>14}", "method", "pairs", "collapsed", "reversed", "worst rev", "oog span", "hue mean/max");
    for m in 0..4 {
        let a = acc[m];
        println!(
            "{:14} {:>8} {:>9.1}% {:>9} {:>11.2e} {:>11.1}% {:>7.2}/{:>5.2}",
            names[m], a[0], 100.0 * a[1] / a[0].max(1.0), a[2], worst_rev[m], 100.0 * a[3] / a[4].max(1.0), a[5] / a[6].max(1.0), a[7]
        );
    }
}
