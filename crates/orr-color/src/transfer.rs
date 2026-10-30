const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

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
}
