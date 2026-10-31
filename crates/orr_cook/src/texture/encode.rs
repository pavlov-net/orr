//! Mip chains to stored texels: BCn through ctt with named encoders, uncompressed formats
//! directly.

use ctt::encoders::Encoder;
use ctt::encoders::bc7enc::Bc7encSettings;
use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, Format, FormatDesc, ImageRef,
    PipelineOutput, SurfaceRef, TargetFormat, TextureKind,
};
use half::f16;
use orr_pack::texture::{FormatInfo, vk};

use super::{DataFormat, Img, Quality, Usage, linear_to_srgb};

/// Stored format of a cooked texture.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Encoding {
    /// RGB as sRGB, alpha linear.
    Bc7Srgb,
    Bc7,
    /// Normal maps store x and y as `orr_pack::texture::normal` encodes them.
    Bc5,
    Bc4,
    Bc6h,
    Raw(DataFormat),
}

impl Encoding {
    pub(crate) fn of(usage: Usage) -> Self {
        match usage {
            Usage::Color | Usage::ColorAlpha => Self::Bc7Srgb,
            Usage::Normal | Usage::Linear2 => Self::Bc5,
            Usage::Linear1 => Self::Bc4,
            Usage::Linear4 => Self::Bc7,
            Usage::Hdr | Usage::Environment { .. } => Self::Bc6h,
            Usage::Displacement => Self::Raw(DataFormat::R16),
            Usage::Data(f) => Self::Raw(f),
        }
    }

    pub(crate) fn vk_format(self) -> u32 {
        use vk::*;
        match self {
            Self::Bc7Srgb => BC7_SRGB,
            Self::Bc7 => BC7_UNORM,
            Self::Bc5 => BC5_UNORM,
            Self::Bc4 => BC4_UNORM,
            Self::Bc6h => BC6H_UFLOAT,
            Self::Raw(f) => match f {
                DataFormat::R8 => R8_UNORM,
                DataFormat::Rg8 => R8G8_UNORM,
                DataFormat::Rgba8 => R8G8B8A8_UNORM,
                DataFormat::Rgba8Srgb => R8G8B8A8_SRGB,
                DataFormat::R16 => R16_UNORM,
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

fn raw(d: DataFormat, m: &Img) -> Vec<u8> {
    let mut out = Vec::new();
    for p in &m.px {
        match d {
            DataFormat::R8 => out.push(unorm8(p[0])),
            DataFormat::Rg8 => out.extend(p[..2].iter().map(|&v| unorm8(v))),
            DataFormat::Rgba8 => out.extend(p.iter().map(|&v| unorm8(v))),
            DataFormat::Rgba8Srgb => {
                out.extend(p[..3].iter().map(|&v| unorm8(linear_to_srgb(v))));
                out.push(unorm8(p[3]));
            }
            DataFormat::R16 => out.extend(unorm16(p[0])),
            DataFormat::Rg16 => out.extend(p[..2].iter().flat_map(|&v| unorm16(v))),
            DataFormat::Rgba16 => out.extend(p.iter().flat_map(|&v| unorm16(v))),
            DataFormat::R16f => out.extend(f16::from_f32(p[0]).to_le_bytes()),
            DataFormat::Rg16f => {
                out.extend(p[..2].iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()));
            }
            DataFormat::Rgba16f => {
                out.extend(p.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()));
            }
            DataFormat::R32f => out.extend(p[0].to_le_bytes()),
            DataFormat::Rg32f => out.extend(p[..2].iter().flat_map(|v| v.to_le_bytes())),
            DataFormat::Rgba32f => out.extend(p.iter().flat_map(|v| v.to_le_bytes())),
        }
    }
    out
}

/// ctt encoder, preset and input layout for each BCn encoding: bc7e for BC7, Intel ISPC
/// for BC4, BC5 and BC6H. Encoders are always named, never picked automatically.
fn bc(e: Encoding, q: Quality, chains: &[Vec<Img>]) -> Result<Vec<Vec<Vec<u8>>>, String> {
    let quality = match q {
        Quality::Fast => ctt::Quality::Fast,
        Quality::Release => ctt::Quality::Slow,
    };
    let (target, encoder) = match e {
        Encoding::Bc7Srgb => (
            Format::BC7_SRGB_BLOCK,
            Encoder::Bc7enc(Bc7encSettings::default()),
        ),
        Encoding::Bc7 => (
            Format::BC7_UNORM_BLOCK,
            Encoder::Bc7enc(Bc7encSettings {
                perceptual: false,
                ..Bc7encSettings::default()
            }),
        ),
        Encoding::Bc5 => (Format::BC5_UNORM_BLOCK, Encoder::Intel(Default::default())),
        Encoding::Bc4 => (Format::BC4_UNORM_BLOCK, Encoder::Intel(Default::default())),
        Encoding::Bc6h => (
            Format::BC6H_UFLOAT_BLOCK,
            Encoder::Intel(Default::default()),
        ),
        Encoding::Raw(_) => unreachable!("raw formats skip ctt"),
    };
    let float = e == Encoding::Bc6h;
    let srgb = e == Encoding::Bc7Srgb;
    let inputs: Vec<Vec<Vec<u8>>> = chains
        .iter()
        .map(|c| {
            c.iter()
                .map(|m| {
                    if float {
                        m.px.iter()
                            .flat_map(|p| [p[0].max(0.0), p[1].max(0.0), p[2].max(0.0), 1.0])
                            .flat_map(f32::to_le_bytes)
                            .collect()
                    } else {
                        m.px.iter()
                            .flat_map(|p| {
                                let rgb = |v: f32| unorm8(if srgb { linear_to_srgb(v) } else { v });
                                [rgb(p[0]), rgb(p[1]), rgb(p[2]), unorm8(p[3])]
                            })
                            .collect()
                    }
                })
                .collect()
        })
        .collect();
    let bpp = if float { 16 } else { 4 };
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
                        stride: m.w * bpp,
                        slice_stride: 0,
                    })
                    .collect()
            })
            .collect(),
        kind: TextureKind::Texture2D,
        desc: FormatDesc {
            format: match (float, srgb) {
                (true, _) => Format::R32G32B32A32_SFLOAT,
                (false, true) => Format::R8G8B8A8_SRGB,
                (false, false) => Format::R8G8B8A8_UNORM,
            },
            color_space: if srgb {
                ColorSpace::Srgb
            } else {
                ColorSpace::Linear
            },
            alpha: if float {
                AlphaMode::Opaque
            } else {
                AlphaMode::Straight
            },
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
            allow_discarding_alpha: true,
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
