use orr_color::RgbPrimaries;
use orr_math::{DMat3, DVec3};
use shader_math::stats::Rng;
use shader_math::*;

fn d3(v: [f32; 3]) -> DVec3 {
    DVec3::from(v.map(f64::from))
}

/// The functions of one Slang color module: `orr.color`, or orr_color's `orr_color.*`.
struct Module {
    name: &'static str,
    matrix: fn(usize) -> [f32; 9],
    luminance_rec709: fn([f32; 3]) -> f32,
    luminance_rec2020: fn([f32; 3]) -> f32,
    srgb_encode: fn([f32; 3]) -> [f32; 3],
    srgb_decode: fn([f32; 3]) -> [f32; 3],
    pq_encode: fn([f32; 3]) -> [f32; 3],
    pq_decode: fn([f32; 3]) -> [f32; 3],
}

const MODULES: [Module; 2] = [
    Module {
        name: "orr.color",
        matrix: color_matrix,
        luminance_rec709,
        luminance_rec2020,
        srgb_encode,
        srgb_decode,
        pq_encode,
        pq_decode,
    },
    Module {
        name: "orr_color",
        matrix: color::matrix,
        luminance_rec709: color::luminance_rec709,
        luminance_rec2020: color::luminance_rec2020,
        srgb_encode: color::srgb_encode,
        srgb_decode: color::srgb_decode,
        pq_encode: color::pq_encode,
        pq_decode: color::pq_decode,
    },
];

#[test]
fn matrices_and_luminance_match_orr_color() {
    let (r709, p3, r2020) = (
        RgbPrimaries::REC709,
        RgbPrimaries::DISPLAY_P3,
        RgbPrimaries::REC2020,
    );
    let want = [
        r709.rgb_to_xyz(),
        r709.xyz_to_rgb(),
        r2020.rgb_to_xyz(),
        r2020.xyz_to_rgb(),
        p3.rgb_to_xyz(),
        p3.xyz_to_rgb(),
        r709.matrix_to(r2020),
        r2020.matrix_to(r709),
        p3.matrix_to(r2020),
        r2020.matrix_to(p3),
    ];
    for m in MODULES {
        for (i, w) in want.iter().enumerate() {
            let got = DMat3::from_cols_array(&(m.matrix)(i).map(f64::from)).transpose();
            let close = (got - *w)
                .to_cols_array()
                .iter()
                .zip(w.to_cols_array())
                .all(|(d, e)| d.abs() <= 1e-7 * e.abs().max(1.0));
            assert!(close, "{} {}: {got} vs {w}", m.name, COLOR_MATRICES[i]);
        }
        let w709 = r709.luminance_weights();
        let mut rng = Rng::new(40);
        for _ in 0..1000 {
            let c = [rng.f32(), rng.f32(), rng.f32()];
            let y709 = w709.dot(d3(c));
            let y2020 = orr_color::luminance(d3(c));
            assert!((f64::from((m.luminance_rec709)(c)) - y709).abs() < 1e-6);
            assert!((f64::from((m.luminance_rec2020)(c)) - y2020).abs() < 1e-6);
        }
    }
}

#[test]
fn transfer_functions_match_orr_color() {
    for m in MODULES {
        for i in 0..=4096 {
            let x = f64::from(i) / 4096.0;
            let c = [x as f32, (x * 0.5) as f32, (-x * 0.01) as f32];
            let (e, d) = ((m.srgb_encode)(c), (m.srgb_decode)(c));
            for k in 0..3 {
                let v = f64::from(c[k]);
                let (name, e, d) = (m.name, f64::from(e[k]), f64::from(d[k]));
                assert!(
                    (e - orr_color::srgb_encode(v)).abs() < 1e-6,
                    "{name} sRGB encode {v}"
                );
                assert!(
                    (d - orr_color::srgb_decode(v)).abs() < 1e-6,
                    "{name} sRGB decode {v}"
                );
            }
        }
        let (mut worst_signal, mut worst_trip) = (0.0f64, 0.0f64);
        for i in 0..=4096 {
            let nits = 10f64.powf(-3.0 + 7.0 * f64::from(i) / 4096.0);
            let s = (m.pq_encode)([nits as f32; 3])[0];
            worst_signal = worst_signal.max((f64::from(s) - orr_color::pq_encode(nits)).abs());
            let back = (m.pq_decode)([s; 3])[0];
            let exact = orr_color::pq_decode(f64::from(s));
            worst_trip = worst_trip.max((f64::from(back) - exact).abs() / exact);
        }
        // In f32 the exponent 78.8 amplifies rounding in the encode, and 1 / 0.159 the
        // cancellation in the decode: 1.3e-5 of signal (0.013 of a 10-bit code) and 3.6e-5
        // relative at worst.
        assert!(
            worst_signal < 2e-5,
            "{} PQ signal error {worst_signal}",
            m.name
        );
        assert!(worst_trip < 5e-5, "{} PQ decode error {worst_trip}", m.name);
    }
}

#[test]
fn hlg_matches_orr_color() {
    for i in 0..=4096 {
        let x = 1.2 * f64::from(i) / 4096.0;
        let e = color::hlg_encode([x as f32; 3])[0];
        assert!(
            (f64::from(e) - orr_color::hlg_encode(x)).abs() < 1e-6,
            "encode {x}"
        );
        let d = color::hlg_decode([x as f32; 3])[0];
        let want = orr_color::hlg_decode(x);
        assert!(
            (f64::from(d) - want).abs() < 2e-6 * want.max(1.0),
            "decode {x}"
        );
    }
}

fn ictcp_ref(nits: DVec3) -> DVec3 {
    let lms = DMat3::from_cols_array(&[
        1688.0, 683.0, 99.0, 2146.0, 2951.0, 309.0, 262.0, 462.0, 3688.0,
    ]) / 4096.0;
    let ictcp = DMat3::from_cols_array(&[
        2048.0, 6610.0, 17933.0, 2048.0, -13613.0, -17390.0, 0.0, 7003.0, -543.0,
    ]) / 4096.0;
    let l = lms * nits;
    ictcp
        * DVec3::new(
            orr_color::pq_encode(l.x),
            orr_color::pq_encode(l.y),
            orr_color::pq_encode(l.z),
        )
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
    let to709 = RgbPrimaries::REC2020.matrix_to(RgbPrimaries::REC709);
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
        "a Rec.2020 color stays outside Rec.709: {worst}"
    );
}
