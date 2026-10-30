//! The lookup tables and specialisation constants of `orr.openpbr`, the OpenPBR 1.1.1
//! reference BSDF from adobe/openpbr-bsdf.
//!
//! Kernels that import `orr.openpbr` read the tables from eight consecutive heap slots, in
//! [`LUTS`] order, starting at the slot given by specialisation constant [`SPEC_LUT_SLOT`].
//! The shader filters them itself with texel loads, so no sampler is involved and any view
//! of the listed format works. The data is derived from openpbr-bsdf, which is under the
//! Apache License 2.0: binaries that embed it or kernels built from `orr.openpbr` carry
//! [`LICENSE`].

/// The texel format of a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutFormat {
    /// One 16-bit unsigned normalised channel (`R16_UNORM`).
    R16Unorm,
    /// Four 32-bit float channels (`R32G32B32A32_SFLOAT`); alpha is 0.
    Rgba32Float,
}

impl LutFormat {
    /// Bytes per texel.
    pub const fn texel_size(self) -> usize {
        match self {
            Self::R16Unorm => 2,
            Self::Rgba32Float => 16,
        }
    }
}

/// One table: texels in x-fastest order, little-endian.
#[derive(Clone, Copy, Debug)]
pub struct Lut {
    /// `OpenPBR_LutId`: the table's offset from the first slot.
    pub id: u32,
    pub name: &'static str,
    /// Width, height, depth. Depth above 1 means a 3D image; the rest are 2D.
    pub extent: [u32; 3],
    pub format: LutFormat,
    pub bytes: &'static [u8],
}

/// The eight tables in `OpenPBR_LutId` order.
pub static LUTS: [Lut; 8] = include!(concat!(env!("OUT_DIR"), "/openpbr/luts.rs"));

/// Specialisation constant ids of the feature toggles (`bool`, default true); turning one
/// off removes that part of the model from the kernel.
pub const SPEC_ENABLE_SHEEN_AND_COAT: u32 = 100;
pub const SPEC_ENABLE_DISPERSION: u32 = 101;
pub const SPEC_ENABLE_TRANSLUCENCY: u32 = 102;
pub const SPEC_ENABLE_METALLIC: u32 = 103;
/// Specialisation constant id of the first table's heap slot (`uint`, default 0).
pub const SPEC_LUT_SLOT: u32 = 104;

/// adobe/openpbr-bsdf's licence text (Apache License 2.0).
pub static LICENSE: &str = include_str!(concat!(env!("OUT_DIR"), "/openpbr/LICENSE"));
