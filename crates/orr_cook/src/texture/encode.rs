//! Mip chains to stored texels: BCn through ctt with named encoders, uncompressed formats
//! directly.

use ctt::encoders::Encoder;
use ctt::encoders::bc7enc::Bc7encSettings;
use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, Format, FormatDesc, ImageRef,
    PipelineOutput, Quality, SurfaceRef, TargetFormat, TextureKind,
};
use half::f16;
use orr_pack::texture::{FormatInfo, vk};

use super::mips::settled;
use super::{DataFormat, Img, MipFilter, Usage, linear_to_srgb};
use crate::simd;

/// Stored format of a cooked texture.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Encoding {
    /// RGB as sRGB, alpha linear.
    Bc7Srgb,
    Bc7,
    Bc5,
    /// Tangent-space normal maps store x and y as `orr_pack::texture::normal` says.
    Bc5Snorm,
    Bc4,
    Bc4Snorm,
    Bc6h,
    Raw(DataFormat),
}

impl Encoding {
    /// The format of `usage` from a source of signed normalized values or of floats.
    pub(super) fn of(usage: Usage, signed: bool, float: bool) -> Self {
        match usage {
            Usage::Color | Usage::ColorAlpha => Self::Bc7Srgb,
            Usage::Normal => Self::Bc5Snorm,
            Usage::Linear2 if signed => Self::Bc5Snorm,
            Usage::Linear2 => Self::Bc5,
            Usage::Linear1 if signed => Self::Bc4Snorm,
            Usage::Linear1 => Self::Bc4,
            Usage::Linear4 => Self::Bc7,
            Usage::Hdr | Usage::Environment { .. } => Self::Bc6h,
            // Half floats step 32 times coarser than R16 near 1.
            Usage::Displacement if float => Self::Raw(DataFormat::R32f),
            Usage::Displacement if signed => Self::Raw(DataFormat::R16Snorm),
            Usage::Displacement => Self::Raw(DataFormat::R16),
            Usage::Data(f) => Self::Raw(f),
        }
    }

    /// The range of values this stores; for formats of floats, the finite ones a filter can sum.
    pub(super) fn range(self) -> (f32, f32) {
        use DataFormat as D;
        let half = f16::MAX.to_f32();
        match self {
            Self::Bc4Snorm | Self::Bc5Snorm | Self::Raw(D::R16Snorm) => (-1.0, 1.0),
            Self::Bc6h => (0.0, half),
            Self::Raw(D::R16f | D::Rg16f | D::Rgba16f) => (-half, half),
            Self::Raw(D::R32f | D::Rg32f | D::Rgba32f) => (f32::MIN / 16.0, f32::MAX / 16.0),
            _ => (0.0, 1.0),
        }
    }

    /// The filter that makes this format's mips from `asked`. Kaiser's negative lobes are clipped
    /// at 0 in non-negative images, which adds energy, and around an HDR texel that is more than
    /// the mean of the image: one texel of 65504 in a white 64x64 more than doubles the 1x1.
    pub(super) fn mip_filter(self, asked: MipFilter) -> MipFilter {
        if self == Self::Bc6h {
            MipFilter::Box
        } else {
            asked
        }
    }

    pub(crate) fn vk_format(self) -> u32 {
        use vk::*;
        match self {
            Self::Bc7Srgb => BC7_SRGB,
            Self::Bc7 => BC7_UNORM,
            Self::Bc5 => BC5_UNORM,
            Self::Bc5Snorm => BC5_SNORM,
            Self::Bc4 => BC4_UNORM,
            Self::Bc4Snorm => BC4_SNORM,
            Self::Bc6h => BC6H_UFLOAT,
            Self::Raw(f) => match f {
                DataFormat::R8 => R8_UNORM,
                DataFormat::Rg8 => R8G8_UNORM,
                DataFormat::Rgba8 => R8G8B8A8_UNORM,
                DataFormat::Rgba8Srgb => R8G8B8A8_SRGB,
                DataFormat::R16 => R16_UNORM,
                DataFormat::R16Snorm => R16_SNORM,
                DataFormat::Rg16 => R16G16_UNORM,
                DataFormat::Rgba16 => R16G16B16A16_UNORM,
                DataFormat::R16f => R16_SFLOAT,
                DataFormat::Rg16f => R16G16_SFLOAT,
                DataFormat::Rgba16f => R16G16B16A16_SFLOAT,
                DataFormat::R32f => R32_SFLOAT,
                DataFormat::Rg32f => R32G32_SFLOAT,
                DataFormat::Rgba32f => R32G32B32A32_SFLOAT,
            },
        }
    }
}

/// Encodes `chains[layer][mip]`; returns tight block rows in the same shape.
pub(crate) fn encode(
    e: Encoding,
    q: Quality,
    chains: &[Vec<Img>],
) -> Result<Vec<Vec<Vec<u8>>>, String> {
    let _z = crate::profile::zone!("texture.encode");
    let f = FormatInfo::of(e.vk_format()).expect("known format");
    let out = match e {
        Encoding::Raw(d) => chains
            .iter()
            .map(|c| c.iter().map(|m| raw(d, m)).collect())
            .collect(),
        _ => bc(e, q, chains)?,
    };
    for (c, layer) in chains.iter().zip(&out) {
        for (m, bytes) in c.iter().zip(layer) {
            if bytes.len() != f.mip_bytes(m.w, m.h) {
                return Err(format!(
                    "encoder gave {} bytes for a {}x{} mip of {e:?}",
                    bytes.len(),
                    m.w,
                    m.h
                ));
            }
        }
    }
    Ok(out)
}

fn unorm8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn unorm16(v: f32) -> [u8; 2] {
    ((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes()
}

fn snorm8(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 127.0).round() as i8).cast_unsigned()
}

fn snorm16(v: f32) -> [u8; 2] {
    ((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()
}

/// `v` as a half float, at the largest finite one past them: a Kaiser mip overshoots a step
/// beside the largest half, which would otherwise store as infinity.
fn half(v: f32) -> [u8; 2] {
    let max = f16::MAX.to_f32();
    f16::from_f32(v.clamp(-max, max)).to_le_bytes()
}

/// `f` of every texel of `m`, one after another.
fn texel_bytes<const N: usize>(m: &Img, f: impl Fn(&[f32; 4]) -> [u8; N] + Sync) -> Vec<u8>
where
    [u8; N]: Default,
{
    simd::mapped(&m.px, f).into_flattened()
}

fn raw(d: DataFormat, m: &Img) -> Vec<u8> {
    use DataFormat as D;
    let le = |v: f32| v.to_le_bytes();
    match d {
        D::R8 => texel_bytes(m, |p| [unorm8(p[0])]),
        D::Rg8 => texel_bytes(m, |p| [unorm8(p[0]), unorm8(p[1])]),
        D::Rgba8 => texel_bytes(m, |p| p.map(unorm8)),
        D::Rgba8Srgb => texel_bytes(m, |p| {
            let [r, g, b] = [p[0], p[1], p[2]].map(|v| unorm8(linear_to_srgb(v)));
            [r, g, b, unorm8(p[3])]
        }),
        D::R16 => texel_bytes(m, |p| unorm16(p[0])),
        D::R16Snorm => texel_bytes(m, |p| snorm16(p[0])),
        D::Rg16 => texel_bytes::<4>(m, |p| bytemuck::cast([unorm16(p[0]), unorm16(p[1])])),
        D::Rgba16 => texel_bytes::<8>(m, |p| bytemuck::cast(p.map(unorm16))),
        D::R16f => texel_bytes(m, |p| half(p[0])),
        D::Rg16f => texel_bytes::<4>(m, |p| bytemuck::cast([half(p[0]), half(p[1])])),
        D::Rgba16f => texel_bytes::<8>(m, |p| bytemuck::cast(p.map(half))),
        D::R32f => texel_bytes(m, |p| le(p[0])),
        D::Rg32f => texel_bytes::<8>(m, |p| bytemuck::cast([le(p[0]), le(p[1])])),
        D::Rgba32f => texel_bytes::<16>(m, |p| bytemuck::cast(p.map(le))),
    }
}

/// ctt encoder, preset and input layout for each BCn encoding: bc7e for BC7, Compressonator
/// for BC4 and BC5, the only one with signed forms, Intel ISPC for BC6H. Encoders are always
/// named, never picked automatically.
fn bc(e: Encoding, quality: Quality, chains: &[Vec<Img>]) -> Result<Vec<Vec<Vec<u8>>>, String> {
    let amd = || Encoder::Amd(Default::default());
    let (target, encoder, input, bpp) = match e {
        Encoding::Bc7Srgb => (
            Format::BC7_SRGB_BLOCK,
            Encoder::Bc7enc(Bc7encSettings::default()),
            Format::R8G8B8A8_SRGB,
            4,
        ),
        Encoding::Bc7 => (
            Format::BC7_UNORM_BLOCK,
            Encoder::Bc7enc(Bc7encSettings {
                perceptual: false,
                ..Bc7encSettings::default()
            }),
            Format::R8G8B8A8_UNORM,
            4,
        ),
        Encoding::Bc5 => (Format::BC5_UNORM_BLOCK, amd(), Format::R8G8_UNORM, 2),
        Encoding::Bc5Snorm => (Format::BC5_SNORM_BLOCK, amd(), Format::R8G8_SNORM, 2),
        Encoding::Bc4 => (Format::BC4_UNORM_BLOCK, amd(), Format::R8_UNORM, 1),
        Encoding::Bc4Snorm => (Format::BC4_SNORM_BLOCK, amd(), Format::R8_SNORM, 1),
        Encoding::Bc6h => (
            Format::BC6H_UFLOAT_BLOCK,
            Encoder::Intel(Default::default()),
            Format::R32G32B32A32_SFLOAT,
            16,
        ),
        Encoding::Raw(_) => unreachable!("raw formats skip ctt"),
    };
    let srgb = e == Encoding::Bc7Srgb;
    let (lo, hi) = e.range();
    let texels = |m: &Img| match e {
        Encoding::Bc6h => texel_bytes::<16>(m, |p| {
            let [r, g, b] = [p[0], p[1], p[2]].map(|v| settled(v, lo, hi));
            bytemuck::cast([r, g, b, 1.0].map(f32::to_le_bytes))
        }),
        Encoding::Bc4 => raw(DataFormat::R8, m),
        Encoding::Bc5 => raw(DataFormat::Rg8, m),
        Encoding::Bc4Snorm => texel_bytes(m, |p| [snorm8(p[0])]),
        Encoding::Bc5Snorm => texel_bytes(m, |p| [snorm8(p[0]), snorm8(p[1])]),
        Encoding::Bc7Srgb => raw(DataFormat::Rgba8Srgb, m),
        _ => raw(DataFormat::Rgba8, m),
    };
    let inputs: Vec<Vec<Vec<u8>>> = chains
        .iter()
        .map(|c| c.iter().map(texels).collect())
        .collect();
    let image = ImageRef {
        surfaces: chains
            .iter()
            .zip(&inputs)
            .map(|(c, layer)| {
                c.iter()
                    .zip(layer)
                    .map(|(m, data)| SurfaceRef {
                        data,
                        width: m.w,
                        height: m.h,
                        depth: 1,
                        stride: m.w * bpp as u32,
                        slice_stride: 0,
                    })
                    .collect()
            })
            .collect(),
        kind: TextureKind::Texture2D,
        desc: FormatDesc {
            format: input,
            color_space: if srgb {
                ColorSpace::Srgb
            } else {
                ColorSpace::Linear
            },
            // Straight would premultiply and divide back, losing color where alpha is 0.
            alpha: AlphaMode::Opaque,
        },
    };
    let out = ctt::convert(
        image,
        ConvertSettings {
            format: Some(TargetFormat::Compressed {
                format: target,
                encoder,
            }),
            container: Container::Raw,
            quality,
            mipmap: false,
            ..Default::default()
        },
    )
    .map_err(|err| format!("ctt: {err}"))?;
    let PipelineOutput::Raw(img) = out else {
        return Err("ctt returned a container for raw output".into());
    };
    Ok(img
        .surfaces
        .into_iter()
        .map(|layer| layer.into_iter().map(|s| s.data).collect())
        .collect())
}
