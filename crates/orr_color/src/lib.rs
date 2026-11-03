//! Color for orr: RGB primaries and the matrices between them and CIE XYZ, luminance, and
//! the sRGB, PQ and HLG transfer functions.
//!
//! orr renders in linear Rec.2020. Matrices are [`DMat3`](orr_math::DMat3)s derived in f64
//! from the chromaticities and act on linear RGB as `m * rgb`:
//!
//! ```
//! use orr_color::RgbPrimaries;
//! use orr_math::DVec3;
//!
//! let to_rec2020 = RgbPrimaries::REC709.matrix_to(RgbPrimaries::REC2020);
//! let rec2020 = to_rec2020 * DVec3::new(1.0, 0.5, 0.25);
//! ```
//!
//! The same constants and functions exist in Slang, in f32, as the modules
//! `orr_color.primaries`, `orr_color.luminance` and `orr_color.transfer` under
//! [`SHADER_DIR`], an import root of every shader build.

#![forbid(unsafe_code)]

mod luminance;
mod primaries;
mod transfer;

pub use luminance::{LUMINOUS_EFFICACY, luminance};
pub use primaries::{Chromaticity, Primaries, RgbPrimaries, delta_uv, uv};
pub use transfer::{
    PQ_MAX_NITS, SCRGB_WHITE_NITS, hlg_decode, hlg_encode, pq_decode, pq_encode, srgb_decode,
    srgb_encode,
};

/// The import root of the crate's Slang modules (`orr_color.*`).
pub const SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
