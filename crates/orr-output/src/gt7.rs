//! The GT7 tone mapper's parameters and the per-display constants `gt7.slang` reads.

use crate::encode::pq_encode;

/// Nits of GT7 frame-buffer value 1.0.
pub const GT7_REFERENCE_NITS: f32 = 100.0;
/// SDR paper white and peak in nits. GT7's curve parameters assume this SDR white.
pub const GT7_SDR_NITS: f32 = 250.0;

/// GT7's tone-mapping parameters. The defaults are Polyphony's reference values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gt7Params {
    /// Weight of the chroma-preserving (ICtCp) mapping against the per-channel curve.
    pub blend_ratio: f32,
    /// Intensity, relative to the peak's, where chroma starts fading to white.
    pub fade_start: f32,
    /// Intensity, relative to the peak's, where chroma has faded out.
    pub fade_end: f32,
    /// Shoulder convergence.
    pub alpha: f32,
    /// End of the toe, in frame-buffer units.
    pub mid_point: f32,
    /// Start of the shoulder as a fraction of the peak.
    pub linear_section: f32,
    /// Toe exponent.
    pub toe_strength: f32,
}

impl Default for Gt7Params {
    fn default() -> Self {
        Self {
            blend_ratio: 0.6,
            fade_start: 0.98,
            fade_end: 1.16,
            alpha: 0.25,
            mid_point: 0.538,
            linear_section: 0.444,
            toe_strength: 1.28,
        }
    }
}

/// [`Gt7Params`] resolved for one display, laid out like `Gt7Curve` in `gt7.slang`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gt7Curve {
    pub peak: f32,
    pub mid_point: f32,
    pub linear_section: f32,
    pub toe_strength: f32,
    pub k_a: f32,
    pub k_b: f32,
    pub k_c: f32,
    pub blend_ratio: f32,
    pub fade_start: f32,
    pub fade_end: f32,
    pub peak_intensity: f32,
    pub output_scale: f32,
}

impl Gt7Curve {
    /// GT7 for a display with `peak_nits` and `paper_white_nits`. The tone mapper's output
    /// is display-linear: 1.0 = paper white, up to `peak_nits / paper_white_nits`. SDR uses
    /// [`GT7_SDR_NITS`] for both; HDR peaks below it are raised to it.
    pub fn new(params: &Gt7Params, peak_nits: f32, paper_white_nits: f32) -> Self {
        let peak = peak_nits.max(GT7_SDR_NITS) / GT7_REFERENCE_NITS;
        let k = (params.linear_section - 1.0) / (params.alpha - 1.0);
        Self {
            peak,
            mid_point: params.mid_point,
            linear_section: params.linear_section,
            toe_strength: params.toe_strength,
            k_a: peak * params.linear_section + peak * k,
            k_b: -peak * k * (params.linear_section / k).exp(),
            k_c: -1.0 / (k * peak),
            blend_ratio: params.blend_ratio,
            fade_start: params.fade_start,
            fade_end: params.fade_end,
            // The LMS of a grey equals the grey, so its ICtCp intensity is its PQ signal.
            peak_intensity: pq_encode(f64::from(peak * GT7_REFERENCE_NITS)) as f32,
            output_scale: GT7_REFERENCE_NITS / paper_white_nits,
        }
    }

    /// GT7's SDR calibration: peak = paper white = 250 nits.
    pub fn sdr(params: &Gt7Params) -> Self {
        Self::new(params, GT7_SDR_NITS, GT7_SDR_NITS)
    }
}
