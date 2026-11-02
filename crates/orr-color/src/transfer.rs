//! The sRGB, PQ and HLG transfer functions.

const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

const HLG_A: f64 = 0.178_832_77;
const HLG_B: f64 = 1.0 - 4.0 * HLG_A;

/// The luminance of a PQ signal of 1.0, in nits.
pub const PQ_MAX_NITS: f64 = 10000.0;

/// The luminance of scRGB's 1.0, in nits. scRGB is linear and unclamped: a value is nits
/// divided by this.
pub const SCRGB_WHITE_NITS: f64 = 80.0;

/// The sRGB transfer function (IEC 61966-2-1), linear to non-linear. Values below 0 stay on
/// the linear segment.
pub fn srgb_encode(linear: f64) -> f64 {
    if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// The inverse of [`srgb_encode`].
pub fn srgb_decode(signal: f64) -> f64 {
    if signal <= 0.040_45 {
        signal / 12.92
    } else {
        ((signal + 0.055) / 1.055).powf(2.4)
    }
}

/// SMPTE ST 2084 (PQ): absolute luminance in nits, clamped to 0..=10000, to a signal in
/// 0..=1.
pub fn pq_encode(nits: f64) -> f64 {
    let y = (nits / PQ_MAX_NITS).clamp(0.0, 1.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y)).powf(PQ_M2)
}

/// The inverse of [`pq_encode`]: a signal, clamped to 0..=1, to nits.
pub fn pq_decode(signal: f64) -> f64 {
    let e = signal.clamp(0.0, 1.0).powf(1.0 / PQ_M2);
    PQ_MAX_NITS * ((e - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * e)).powf(1.0 / PQ_M1)
}

/// The HLG OETF (ITU-R BT.2100): scene-linear light, 1 at the nominal peak and clamped at
/// 0, to a signal in 0..=1.
pub fn hlg_encode(linear: f64) -> f64 {
    let e = linear.max(0.0);
    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        HLG_A * (12.0 * e - HLG_B).ln() + hlg_c()
    }
}

fn hlg_c() -> f64 {
    0.5 - HLG_A * (4.0 * HLG_A).ln()
}

/// The inverse of [`hlg_encode`].
pub fn hlg_decode(signal: f64) -> f64 {
    let s = signal.max(0.0);
    if s <= 0.5 {
        s * s / 3.0
    } else {
        (((s - hlg_c()) / HLG_A).exp() + HLG_B) / 12.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_inverts() {
        for i in -100..=1100 {
            let x = f64::from(i) / 1000.0;
            assert!((srgb_decode(srgb_encode(x)) - x).abs() < 1e-12, "{x}");
        }
    }

    #[test]
    fn pq_inverts_and_hits_reference_points() {
        for i in 0..=1000 {
            let nits = PQ_MAX_NITS * (f64::from(i) / 1000.0).powi(4);
            let back = pq_decode(pq_encode(nits));
            assert!(
                (back - nits).abs() <= 1e-9 * nits.max(1e-3),
                "{nits} -> {back}"
            );
        }
        assert!((pq_encode(PQ_MAX_NITS) - 1.0).abs() < 1e-12);
        // BT.2100: 100 cd/m2 is a PQ signal of about 0.508.
        assert!((pq_encode(100.0) - 0.508_078).abs() < 1e-6);
    }

    #[test]
    fn slang_constants_are_the_rust_ones() {
        let slang = include_str!("../shaders/orr_color/transfer.slang");
        for (name, v) in [
            ("PQ_MAX_NITS", PQ_MAX_NITS),
            ("SCRGB_WHITE_NITS", SCRGB_WHITE_NITS),
        ] {
            let decl = format!("public static const float {name} = {v:?};");
            assert!(slang.contains(&decl), "{decl}");
        }
    }

    #[test]
    fn hlg_inverts_and_its_segments_meet() {
        for i in 0..=1200 {
            let x = f64::from(i) / 1000.0;
            assert!((hlg_decode(hlg_encode(x)) - x).abs() < 1e-12, "{x}");
        }
        // The log segment starts where the square root ends: 1/12 at signal 0.5.
        let joint = 1.0 / 12.0 + 1e-12;
        assert!((hlg_encode(joint) - 0.5).abs() < 1e-10);
        assert!((hlg_decode(0.5 + 1e-12) - 1.0 / 12.0).abs() < 1e-10);
        assert!((hlg_encode(1.0) - 1.0).abs() < 1e-8);
    }
}
