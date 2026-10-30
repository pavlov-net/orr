//! Transfer functions for writing images on the CPU, in f64.

const PQ_M1: f64 = 2610.0 / 16384.0;
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0;
const PQ_C1: f64 = 3424.0 / 4096.0;
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0;
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0;

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
    let y = (nits / 10000.0).clamp(0.0, 1.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y)).powf(PQ_M2)
}

/// The inverse of [`pq_encode`]: a signal, clamped to 0..=1, to nits.
pub fn pq_decode(signal: f64) -> f64 {
    let e = signal.clamp(0.0, 1.0).powf(1.0 / PQ_M2);
    10000.0 * ((e - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * e)).powf(1.0 / PQ_M1)
}
