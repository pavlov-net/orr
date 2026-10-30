use glam::{DMat3, DVec3};
use shader_maths::stats::Rng;
use shader_maths::*;

/// RGB to XYZ from primaries and white point chromaticities.
fn rgb_to_xyz(primaries: [[f64; 2]; 3], white: [f64; 2]) -> DMat3 {
    let xyz = |[x, y]: [f64; 2]| DVec3::new(x / y, 1.0, (1.0 - x - y) / y);
    let p = DMat3::from_cols(xyz(primaries[0]), xyz(primaries[1]), xyz(primaries[2]));
    let s = p.inverse() * xyz(white);
    DMat3::from_cols(p.x_axis * s.x, p.y_axis * s.y, p.z_axis * s.z)
}

const D65: [f64; 2] = [0.3127, 0.3290];
const REC709: [[f64; 2]; 3] = [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]];
const P3: [[f64; 2]; 3] = [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]];
const REC2020: [[f64; 2]; 3] = [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]];

fn d3(v: [f32; 3]) -> DVec3 {
    DVec3::from(v.map(f64::from))
}

#[test]
fn matrices_derive_from_the_primaries() {
    let (r709, p3, r2020) = (
        rgb_to_xyz(REC709, D65),
        rgb_to_xyz(P3, D65),
        rgb_to_xyz(REC2020, D65),
    );
    let want = [
        r709,
        r709.inverse(),
        r2020,
        r2020.inverse(),
        p3,
        p3.inverse(),
        r2020.inverse() * r709,
        r709.inverse() * r2020,
        r2020.inverse() * p3,
        p3.inverse() * r2020,
    ];
    for (i, w) in want.iter().enumerate() {
        let got = color_matrix(i);
        for r in 0..3 {
            for c in 0..3 {
                let (g, e) = (f64::from(got[3 * r + c]), w.row(r)[c]);
                assert!(
                    (g - e).abs() <= 1e-7 * e.abs().max(1.0),
                    "{} [{r}][{c}]: {g} vs {e}",
                    COLOR_MATRICES[i]
                );
            }
        }
    }
    let mut rng = Rng::new(40);
    for _ in 0..1000 {
        let c = [rng.f32(), rng.f32(), rng.f32()];
        assert!((f64::from(luminance_rec709(c)) - r709.row(1).dot(d3(c))).abs() < 1e-6);
        assert!((f64::from(luminance_rec2020(c)) - r2020.row(1).dot(d3(c))).abs() < 1e-6);
    }
}

fn srgb_encode_ref(x: f64) -> f64 {
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_decode_ref(x: f64) -> f64 {
    if x <= 0.040_45 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

fn pq_encode_ref(nits: f64) -> f64 {
    let y = (nits / 10000.0).clamp(0.0, 1.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y)).powf(PQ_M2)
}

fn pq_decode_ref(e: f64) -> f64 {
    let p = e.clamp(0.0, 1.0).powf(1.0 / PQ_M2);
    10000.0 * ((p - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * p)).powf(1.0 / PQ_M1)
}

#[test]
fn transfer_functions_match_their_standards() {
    for i in 0..=4096 {
        let x = f64::from(i) / 4096.0;
        let c = [x as f32, (x * 0.5) as f32, (-x * 0.01) as f32];
        let (e, d) = (srgb_encode(c), srgb_decode(c));
        for k in 0..3 {
            let v = f64::from(c[k]);
            assert!(
                (f64::from(e[k]) - srgb_encode_ref(v)).abs() < 1e-6,
                "srgb encode {v}"
            );
            assert!(
                (f64::from(d[k]) - srgb_decode_ref(v)).abs() < 1e-6,
                "srgb decode {v}"
            );
        }
    }
    let (mut worst_signal, mut worst_trip) = (0.0f64, 0.0f64);
    for i in 0..=4096 {
        let nits = 10f64.powf(-3.0 + 7.0 * f64::from(i) / 4096.0);
        let s = pq_encode([nits as f32; 3])[0];
        worst_signal = worst_signal.max((f64::from(s) - pq_encode_ref(nits)).abs());
        let back = pq_decode([s; 3])[0];
        let exact = pq_decode_ref(f64::from(s));
        worst_trip = worst_trip.max((f64::from(back) - exact).abs() / exact);
    }
    // In f32 the exponent 78.8 amplifies rounding in the encode, and 1 / 0.159 the
    // cancellation in the decode: 1.3e-5 of signal (0.013 of a 10-bit code) and 3.6e-5
    // relative at worst.
    assert!(worst_signal < 2e-5, "PQ signal error {worst_signal}");
    assert!(worst_trip < 5e-5, "PQ decode error {worst_trip}");
}

fn ictcp_ref(nits: DVec3) -> DVec3 {
    let lms = DMat3::from_cols_array(&[
        1688.0, 683.0, 99.0, 2146.0, 2951.0, 309.0, 262.0, 462.0, 3688.0,
    ]) / 4096.0;
    let ictcp = DMat3::from_cols_array(&[
        2048.0, 6610.0, 17933.0, 2048.0, -13613.0, -17390.0, 0.0, 7003.0, -543.0,
    ]) / 4096.0;
    let l = lms * nits;
    ictcp * DVec3::new(pq_encode_ref(l.x), pq_encode_ref(l.y), pq_encode_ref(l.z))
}

#[test]
fn ictcp_matches_bt2100() {
    let mut rng = Rng::new(41);
    for _ in 0..100_000 {
        let nits = [0; 3].map(|_: u8| 10f32.powf(rng.range(-2.0, 4.0)));
        let got = d3(ictcp_from_rec2020(nits));
        let want = ictcp_ref(d3(nits));
        // The PQ encode's error (1.3e-5), through ICtCp rows that sum to 8.6 in magnitude.
        assert!(
            (got - want).abs().max_element() < 1.5e-4,
            "{nits:?}: {got} vs {want}"
        );
        let back = d3(rec2020_from_ictcp(ictcp_from_rec2020(nits)));
        let n = d3(nits);
        // The PQ decode's error, spread by the inverse matrix over all three channels.
        assert!(
            ((back - n).abs() / n.max_element()).max_element() < 5e-4,
            "{nits:?} -> {back}"
        );
    }
}

const THRESHOLD: [f64; 3] = [0.815, 0.803, 0.880];
const LIMIT: [f64; 3] = [1.593_869_6, 1.087_499_8, 1.116_575_8];

/// ACES 1.3 Reference Gamut Compression on linear RGB with orr's limits.
fn gamut_compress_ref(rgb: [f64; 3]) -> [f64; 3] {
    let p = 1.2;
    let ach = rgb[0].max(rgb[1]).max(rgb[2]);
    if ach == 0.0 {
        return rgb;
    }
    std::array::from_fn(|i| {
        let (t, l) = (THRESHOLD[i], LIMIT[i]);
        let d = (ach - rgb[i]) / ach.abs();
        if d < t {
            return rgb[i];
        }
        let s = (l - t) / (((1.0 - t) / (l - t)).powf(-p) - 1.0).powf(1.0 / p);
        let x = (d - t) / s;
        let dc = t + s * x / (1.0 + x.powf(p)).powf(1.0 / p);
        ach - dc * ach.abs()
    })
}

#[test]
fn gamut_compression_matches_rgc_and_keeps_order() {
    let mut rng = Rng::new(42);
    for _ in 0..100_000 {
        let scale = 10f32.powf(rng.range(-3.0, 2.0));
        let c = [
            rng.range(-0.6, 1.0),
            rng.range(-0.6, 1.0),
            rng.range(-0.6, 1.0),
        ]
        .map(|v| v * scale);
        let got = gamut_compress(c);
        let want = gamut_compress_ref(c.map(f64::from));
        for i in 0..3 {
            assert!(
                (f64::from(got[i]) - want[i]).abs() < 2e-6 * f64::from(scale),
                "{c:?}: {got:?} vs {want:?}"
            );
            let ach = c.iter().copied().fold(f32::MIN, f32::max);
            if (ach - c[i]) / ach.abs() < THRESHOLD[i] as f32 {
                assert_eq!(
                    got[i].to_bits(),
                    c[i].to_bits(),
                    "{c:?} channel {i} changed"
                );
            }
        }
    }
    // A saturation sweep at one hue: distinct inputs stay distinct and ordered, and the
    // limit lands on the gamut boundary.
    for channel in 0..3 {
        let mut last = f32::INFINITY;
        for k in 0..=2000 {
            let d = 1.6 * k as f32 / 2000.0;
            let mut c = [1.0f32; 3];
            c[channel] = 1.0 - d;
            let out = gamut_compress(c)[channel];
            assert!(out < last, "channel {channel} d {d}: {out} after {last}");
            last = out;
        }
        let mut c = [1.0f32; 3];
        c[channel] = 1.0 - LIMIT[channel] as f32;
        assert!(
            gamut_compress(c)[channel].abs() < 1e-5,
            "channel {channel} limit"
        );
    }
}

#[test]
fn gamut_compression_brings_rec2020_inside_rec709() {
    let to709 = rgb_to_xyz(REC709, D65).inverse() * rgb_to_xyz(REC2020, D65);
    // The Rec.2020 cube's faces, where every distance limit is reached.
    let n = 400;
    let mut worst = f64::MAX;
    for face in 0..3 {
        for k in 0..=n {
            for j in 0..=n {
                let (a, b) = (f64::from(k) / f64::from(n), f64::from(j) / f64::from(n));
                let mut c = [0.0; 3];
                c[(face + 1) % 3] = a;
                c[(face + 2) % 3] = b;
                if a.max(b) == 0.0 {
                    continue;
                }
                let r = (to709 * DVec3::from(c)).to_array().map(|v| v as f32);
                let out = gamut_compress(r);
                worst = worst.min(out.iter().map(|&v| f64::from(v)).fold(f64::MAX, f64::min));
            }
        }
    }
    assert!(
        worst > -1e-6,
        "a Rec.2020 colour stays outside Rec.709: {worst}"
    );
}
