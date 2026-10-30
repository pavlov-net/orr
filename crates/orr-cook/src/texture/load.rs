//! Image files to float texels: PNG, JPEG, Radiance HDR and OpenEXR through `image`, KTX2
//! and DDS through ctt's readers, block-compressed texels decoded with bcdec.

use std::io::Cursor;

use ctt::input::{InputOverrides, decode_container};
use ctt::{ColorSpace, FormatExt, TextureKind};
use orr_pack::texture::{FormatInfo, full_mip_count, mip_size, vk};

use super::{Img, Primaries};

/// A decoded image file.
pub(crate) struct Source {
    /// Mip 0 of each layer, values as stored: unorm data in 0..1, floats as they are.
    pub layers: Vec<Img>,
    /// Six layers in the Vulkan face order.
    pub cube: bool,
    /// RGB values are sRGB-encoded.
    pub srgb: bool,
    /// Primaries the file is tagged with.
    pub primaries: Option<Primaries>,
    /// The block-compressed texels of a compressed file, kept so an encoding that matches
    /// can pass them through.
    pub blocks: Option<Blocks>,
}

pub(crate) struct Blocks {
    pub vk_format: u32,
    pub w: u32,
    pub h: u32,
    /// `levels[layer][mip]`, tight block rows.
    pub levels: Vec<Vec<Vec<u8>>>,
}

impl Blocks {
    /// Every mip down to 1x1 is present.
    pub(crate) fn complete(&self) -> bool {
        self.levels[0].len() as u32 == full_mip_count(self.w, self.h)
    }
}

pub(crate) fn decode(bytes: &[u8]) -> Result<Source, String> {
    match decode_container(bytes, InputOverrides::default()).map_err(|e| e.to_string())? {
        Some(image) => container(image),
        None => file(bytes),
    }
}

fn container(image: ctt::Image) -> Result<Source, String> {
    let cube = match image.kind {
        TextureKind::Texture2D => false,
        TextureKind::Cubemap if image.surfaces.len() == 6 => true,
        TextureKind::Cubemap => return Err("cube arrays are not supported".into()),
        TextureKind::Texture3D => return Err("3D textures are not supported".into()),
    };
    let format = image.desc.format;
    let srgb = image.desc.color_space == ColorSpace::Srgb;
    let (w, h) = (image.surfaces[0][0].width, image.surfaces[0][0].height);
    if format.is_compressed() {
        let vk_format = format.value();
        let f = FormatInfo::of(vk_format)
            .filter(|f| f.block == 4)
            .ok_or_else(|| format!("compressed format {format:?} is not a BCn format"))?;
        let mut levels = Vec::new();
        for layer in &image.surfaces {
            let mut mips = Vec::new();
            for (m, s) in layer.iter().enumerate() {
                let [mw, mh] = mip_size(w, h, m as u32);
                let row = mw.div_ceil(4) as usize * f.bytes as usize;
                let rows = mh.div_ceil(4) as usize;
                let mut tight = Vec::with_capacity(row * rows);
                for r in 0..rows {
                    let at = r * s.stride as usize;
                    tight.extend_from_slice(
                        s.data
                            .get(at..at + row)
                            .ok_or("compressed surface too short")?,
                    );
                }
                mips.push(tight);
            }
            levels.push(mips);
        }
        let layers = levels
            .iter()
            .map(|l| decode_blocks(vk_format, w, h, &l[0]))
            .collect::<Result<_, _>>()?;
        return Ok(Source {
            layers,
            cube,
            srgb,
            primaries: None,
            blocks: Some(Blocks {
                vk_format,
                w,
                h,
                levels,
            }),
        });
    }
    // Uncompressed: ctt converts to f32 RGBA in the file's colour space.
    let mip0 = ctt::Image {
        surfaces: image.surfaces.iter().map(|l| vec![l[0].clone()]).collect(),
        kind: TextureKind::Texture2D,
        desc: image.desc,
    };
    let out = ctt::convert(
        mip0.to_ref(),
        ctt::ConvertSettings {
            format: Some(ctt::TargetFormat::Uncompressed(
                ctt::Format::R32G32B32A32_SFLOAT,
            )),
            container: ctt::Container::Raw,
            allow_discarding_alpha: true,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let ctt::PipelineOutput::Raw(img) = out else {
        return Err("ctt returned a container for raw output".into());
    };
    let layers = img
        .surfaces
        .iter()
        .map(|l| {
            let s = &l[0];
            let mut out = Img::new(s.width, s.height);
            for (y, row) in out.px.chunks_exact_mut(s.width as usize).enumerate() {
                let src = &s.data[y * s.stride as usize..][..s.width as usize * 16];
                for (p, b) in row.iter_mut().zip(src.chunks_exact(16)) {
                    *p = std::array::from_fn(|c| {
                        f32::from_le_bytes(b[c * 4..c * 4 + 4].try_into().expect("4 bytes"))
                    });
                }
            }
            out
        })
        .collect();
    Ok(Source {
        layers,
        cube,
        srgb,
        primaries: None,
        blocks: None,
    })
}

fn file(bytes: &[u8]) -> Result<Source, String> {
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let is_png = reader.format() == Some(image::ImageFormat::Png);
    let img = reader.decode().map_err(|e| e.to_string())?;
    let hdr = matches!(
        img,
        image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_)
    );
    let (mut srgb, mut primaries) = (!hdr, None);
    if is_png && let Some(cicp) = png_cicp(bytes)? {
        primaries = Some(match cicp.color_primaries {
            1 => Primaries::Rec709,
            9 => Primaries::Rec2020,
            12 => Primaries::DisplayP3,
            p => return Err(format!("PNG cICP colour primaries {p} are not supported")),
        });
        srgb = match cicp.transfer_function {
            13 => true,
            8 => false,
            t => return Err(format!("PNG cICP transfer function {t} is not supported")),
        };
    }
    let rgba = img.to_rgba32f();
    let (w, h) = rgba.dimensions();
    let px = rgba
        .into_raw()
        .chunks_exact(4)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    Ok(Source {
        layers: vec![Img { w, h, px }],
        cube: false,
        srgb,
        primaries,
        blocks: None,
    })
}

fn png_cicp(bytes: &[u8]) -> Result<Option<png::CodingIndependentCodePoints>, String> {
    let reader = png::Decoder::new(Cursor::new(bytes))
        .read_info()
        .map_err(|e| e.to_string())?;
    Ok(reader.info().coding_independent_code_points)
}

/// Decodes one layer's BCn mip to float texels (unorm values in 0..1, BC6H as floats).
pub(crate) fn decode_blocks(vk_format: u32, w: u32, h: u32, data: &[u8]) -> Result<Img, String> {
    let f = FormatInfo::of(vk_format).ok_or("unknown format")?;
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    if data.len() != bw * bh * f.bytes as usize {
        return Err("compressed mip size".into());
    }
    let mut out = Img::new(w, h);
    for by in 0..bh {
        for bx in 0..bw {
            let block = &data[(by * bw + bx) * f.bytes as usize..][..f.bytes as usize];
            let texels = decode_block(vk_format, block)?;
            for (i, t) in texels.iter().enumerate() {
                let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
                if x < w as usize && y < h as usize {
                    out.px[y * w as usize + x] = *t;
                }
            }
        }
    }
    Ok(out)
}

fn decode_block(vk_format: u32, block: &[u8]) -> Result<[[f32; 4]; 16], String> {
    use vk::*;
    let unorm = |b: &[u8], channels: usize| -> [[f32; 4]; 16] {
        std::array::from_fn(|i| {
            let mut p = [0.0, 0.0, 0.0, 1.0];
            for c in 0..channels {
                p[c] = f32::from(b[i * channels + c]) / 255.0;
            }
            p
        })
    };
    let mut b8 = [0u8; 64];
    Ok(match vk_format {
        BC1_RGB_UNORM | BC1_RGB_SRGB | BC1_RGBA_UNORM | BC1_RGBA_SRGB => {
            bcdec_rs::bc1(block, &mut b8, 16);
            unorm(&b8, 4)
        }
        BC2_UNORM | BC2_SRGB => {
            bcdec_rs::bc2(block, &mut b8, 16);
            unorm(&b8, 4)
        }
        BC3_UNORM | BC3_SRGB => {
            bcdec_rs::bc3(block, &mut b8, 16);
            unorm(&b8, 4)
        }
        BC7_UNORM | BC7_SRGB => {
            bcdec_rs::bc7(block, &mut b8, 16);
            unorm(&b8, 4)
        }
        BC4_UNORM => {
            bcdec_rs::bc4(block, &mut b8[..16], 4, false);
            unorm(&b8, 1)
        }
        BC5_UNORM => {
            bcdec_rs::bc5(block, &mut b8[..32], 8, false);
            unorm(&b8, 2)
        }
        BC6H_UFLOAT | BC6H_SFLOAT => {
            let mut f = [0.0f32; 48];
            bcdec_rs::bc6h_float(block, &mut f, 12, vk_format == BC6H_SFLOAT);
            std::array::from_fn(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2], 1.0])
        }
        _ => return Err(format!("format {vk_format} does not decode on the CPU")),
    })
}
