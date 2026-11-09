//! The lookup tables of `orr_lighting.openpbr`, the OpenPBR reference BSDF from
//! adobe/openpbr-bsdf.
//!
//! Kernels that import `orr_lighting.openpbr` read the tables from eight consecutive heap slots, in
//! [`LUTS`] order, starting at the slot given by specialization constant `openpbr_lut_slot` (id
//! 104, default [`OPENPBR_LUT_SLOT`](orr_shaders::abi::OPENPBR_LUT_SLOT)).
//! The shader filters them itself with texel loads, so no sampler is involved and any view
//! of the listed format works. The data is derived from openpbr-bsdf, which is under the
//! Apache License 2.0: binaries that embed it or kernels built from `orr_lighting.openpbr` carry
//! [`LICENSE`].

/// The texel format of a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutFormat {
    /// One 16-bit unsigned normalized channel (`R16_UNORM`).
    R16Unorm,
    /// Four 32-bit float channels (`R32G32B32A32_SFLOAT`); alpha is 0.
    Rgba32Float,
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
pub static LUTS: [Lut; orr_shaders::abi::OPENPBR_LUT_SLOTS as usize] =
    include!(concat!(env!("OUT_DIR"), "/openpbr/luts.rs"));

/// adobe/openpbr-bsdf's license text (Apache License 2.0).
pub static LICENSE: &str = include_str!(concat!(env!("OUT_DIR"), "/openpbr/LICENSE"));
