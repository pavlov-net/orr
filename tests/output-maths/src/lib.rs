//! orr-output's Slang maths compiled for the CPU (`slang/output.slang`), with safe wrappers
//! for the tests, and Polyphony's GT7 sample as the reference.

use orr_output::{Gt7Curve, OutputFormat};

/// `gpu/output.slang` compiled to SPIR-V; building it checks its generated layouts.
pub mod gpu {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

mod ffi {
    use orr_output::Gt7Curve;

    unsafe extern "C" {
        pub(crate) fn orr_gt7_tonemap(
            curve: *const Gt7Curve,
            rgb: *const f32,
            out: *mut f32,
            count: u32,
        );
        pub(crate) fn orr_gt7_display_linear(
            curve: *const Gt7Curve,
            rgb: *const f32,
            out: *mut f32,
            count: u32,
            sdr: u32,
        );
        pub(crate) fn orr_encode_display(
            rgb: *const f32,
            out: *mut f32,
            count: u32,
            encoding: u32,
            paper_white_nits: f32,
        );
        pub(crate) fn orr_ev100_from_luminance(l: f32) -> f32;
        pub(crate) fn orr_luminance_from_ev100(e: f32) -> f32;
        pub(crate) fn orr_exposure_from_ev100(e: f32) -> f32;
        pub(crate) fn orr_ev100_from_exposure(s: f32) -> f32;
        pub(crate) fn orr_srgb_encode(x: f32) -> f32;
        pub(crate) fn orr_srgb_decode(x: f32) -> f32;
        pub(crate) fn orr_pq_encode(nits: f32) -> f32;
        pub(crate) fn orr_pq_decode(signal: f32) -> f32;
        pub(crate) fn gt7_reference_f32(peak_nits: f32, rgb: *const f32, out: *mut f32, count: u32);
        pub(crate) fn gt7_reference_f64(peak_nits: f32, rgb: *const f32, out: *mut f32, count: u32);
    }
}

/// `gt7_tonemap` on the Slang CPU target.
pub fn gt7_tonemap(curve: &Gt7Curve, rgb: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; rgb.len()];
    // SAFETY: `rgb` and `out` hold `count` triples of f32; `Gt7Curve` is `repr(C)` like the
    // Slang struct.
    unsafe {
        ffi::orr_gt7_tonemap(
            curve,
            rgb.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            count(rgb),
        );
    }
    out
}

/// `gt7_display_linear` on the Slang CPU target: exposed linear Rec.2020 to display-linear
/// Rec.709.
pub fn gt7_display_linear(curve: &Gt7Curve, rgb: &[[f32; 3]], sdr: bool) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; rgb.len()];
    // SAFETY: as in `gt7_tonemap`.
    unsafe {
        ffi::orr_gt7_display_linear(
            curve,
            rgb.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            count(rgb),
            u32::from(sdr),
        );
    }
    out
}

/// `encode_display` on the Slang CPU target.
pub fn encode_display(
    rgb: &[[f32; 3]],
    format: OutputFormat,
    paper_white_nits: f32,
) -> Vec<[f32; 3]> {
    // ENCODING_* in display.slang.
    let encoding = match format {
        OutputFormat::Sdr => 0,
        OutputFormat::Hdr10Pq => 1,
        OutputFormat::ScRgb => 2,
    };
    let mut out = vec![[0.0; 3]; rgb.len()];
    // SAFETY: `rgb` and `out` hold `count` triples of f32.
    unsafe {
        ffi::orr_encode_display(
            rgb.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            count(rgb),
            encoding,
            paper_white_nits,
        );
    }
    out
}

/// Polyphony's sample: `None` is its SDR calibration, `Some(peak)` its HDR one. `wide`
/// runs it in double precision.
pub fn gt7_reference(peak_nits: Option<f32>, wide: bool, rgb: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0; 3]; rgb.len()];
    let f = if wide {
        ffi::gt7_reference_f64
    } else {
        ffi::gt7_reference_f32
    };
    // SAFETY: as above.
    unsafe {
        f(
            peak_nits.unwrap_or(0.0),
            rgb.as_ptr().cast(),
            out.as_mut_ptr().cast(),
            count(rgb),
        );
    }
    out
}

fn count(rgb: &[[f32; 3]]) -> u32 {
    u32::try_from(rgb.len()).expect("too many samples")
}

macro_rules! scalar {
    ($($name:ident => $ffi:ident),* $(,)?) => {$(
        pub fn $name(x: f32) -> f32 {
            // SAFETY: a pure function of its argument.
            unsafe { ffi::$ffi(x) }
        }
    )*};
}

scalar! {
    ev100_from_luminance => orr_ev100_from_luminance,
    luminance_from_ev100 => orr_luminance_from_ev100,
    exposure_from_ev100 => orr_exposure_from_ev100,
    ev100_from_exposure => orr_ev100_from_exposure,
    srgb_encode => orr_srgb_encode,
    srgb_decode => orr_srgb_decode,
    pq_encode => orr_pq_encode,
    pq_decode => orr_pq_decode,
}
