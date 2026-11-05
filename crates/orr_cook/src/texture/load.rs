//! Image files to float texels: PNG, JPEG, WebP, Radiance HDR and OpenEXR through `image`,
//! KTX2 and DDS through ctt's readers, block-compressed texels decoded with bcdec. KTX2 files
//! of BCn blocks, plain or zstd-compressed, are read here instead, each level decompressed
//! once straight into its blocks. Formats are recognized by their bytes. An animated WebP
//! decodes to its first frame.

use std::borrow::Cow;
use std::io::{BufRead, BufReader, Cursor, Read, Seek};
use std::path::Path;

use ctt::input::{InputOverrides, decode_container};
use ctt::{ColorSpace, FormatExt, TextureKind};
use image::ImageDecoder;
use image::codecs::webp::WebPDecoder;
use orr_pack::texture::{FormatInfo, full_mip_count, mip_size, vk};
use rayon::prelude::*;

use super::{Img, Primaries};

/// A decoded image file.
pub(super) struct Source {
    /// Mip 0 of each layer, values as stored: unorm data in 0..1, snorm data in -1..1, floats
    /// as they are.
    pub(super) layers: Vec<Img>,
    /// Six layers in the Vulkan face order.
    pub(super) cube: bool,
    /// RGB values are sRGB-encoded.
    pub(super) srgb: bool,
    /// Primaries the file is tagged with.
    pub(super) primaries: Option<Primaries>,
    /// The block-compressed texels of a compressed file, kept so an encoding that matches
    /// can pass them through.
    pub(super) blocks: Option<Blocks>,
    /// The file holds two channels, red and green.
    pub(super) rg: bool,
    /// The file stores signed normalized values.
    pub(super) signed: bool,
}

pub(super) struct Blocks {
    pub(super) vk_format: u32,
    pub(super) w: u32,
    pub(super) h: u32,
    /// `levels[layer][mip]`, tight block rows.
    pub(super) levels: Vec<Vec<Vec<u8>>>,
}

impl Blocks {
    /// Every mip down to 1x1 is present.
    pub(super) fn complete(&self) -> bool {
        self.levels[0].len() as u32 == full_mip_count(self.w, self.h)
    }
}

/// Decodes an image file, recognized by its first bytes.
pub(super) fn decode(bytes: &[u8]) -> Result<Source, String> {
    let mut src = open(bytes)?;
    decode_layers(&mut src)?;
    Ok(src)
}

/// Fills the layers of an opened block-compressed file with its decoded mip 0.
pub(super) fn decode_layers(src: &mut Source) -> Result<(), String> {
    if let Some(b) = &src.blocks
        && src.layers.is_empty()
    {
        src.layers = b
            .levels
            .iter()
            .map(|l| decode_blocks(b.vk_format, b.w, b.h, &l[0]))
            .collect::<Result<_, _>>()?;
    }
    Ok(())
}

/// Opens an image file like [`decode`], but leaves `layers` empty when the file holds
/// block-compressed texels.
pub(super) fn open(bytes: &[u8]) -> Result<Source, String> {
    const KTX1: &[u8] = b"\xABKTX 11\xBB\r\n\x1A\n";
    if bytes.starts_with(KTX1) {
        return Err("KTX1 images are not supported; convert them to KTX2".into());
    }
    if let Some(k) = Ktx2Blocks::of(bytes) {
        return k.open();
    }
    match decode_container(bytes, InputOverrides::default()).map_err(|e| e.to_string())? {
        Some(image) => container(image),
        None => file(bytes),
    }
}

/// What [`open`] tells of a KTX2 file of BCn blocks holding one 2D image, with the range of
/// its mip 0 as [`block_range`] finds it, decompressing only as much as the range needs.
pub(crate) struct Scan {
    pub(super) w: u32,
    pub(super) h: u32,
    pub(super) vk_format: u32,
    /// Every mip down to 1x1 is present.
    pub(super) complete: bool,
    pub(super) srgb: bool,
    pub(super) rg: bool,
    pub(super) signed: bool,
    pub(super) range: [(f32, f32); 4],
}

/// The [`Scan`] of a KTX2 file of BCn blocks; `None` for every other file.
pub(crate) fn scan_ktx2(bytes: &[u8], spread: f32) -> Option<Result<Scan, String>> {
    Ktx2Blocks::of(bytes).map(|k| k.scan(spread))
}

/// A KTX2 file whose levels hold BCn blocks, stored plain or with zstd.
struct Ktx2Blocks<'a> {
    reader: ktx2::Reader<&'a [u8]>,
    header: ktx2::Header,
    format: ktx2::Format,
    info: FormatInfo,
    zstd: bool,
}

impl<'a> Ktx2Blocks<'a> {
    /// The file as such, or `None` when ctt's reader takes it (and reports its errors):
    /// other formats, 3D textures, and levels whose sizes do not match the header.
    fn of(bytes: &'a [u8]) -> Option<Self> {
        let reader = ktx2::Reader::new(bytes).ok()?;
        let h = reader.header();
        let format = h.format?;
        let info = FormatInfo::of(format.value()).filter(|f| f.block == 4)?;
        let zstd = match h.supercompression_scheme {
            None => false,
            Some(ktx2::SupercompressionScheme::Zstandard) => true,
            Some(_) => return None,
        };
        let (w, hh) = (h.pixel_width, h.pixel_height);
        let plain = w > 0
            && hh > 0
            && h.pixel_depth == 0
            && (h.face_count == 1 || h.face_count == 6 && w == hh)
            && h.level_count.max(1) <= full_mip_count(w, hh);
        let k = plain.then_some(Self {
            reader,
            header: h,
            format,
            info,
            zstd,
        })?;
        let images = u64::from(h.layer_count.max(1)) * u64::from(h.face_count);
        let sized = k.reader.levels().enumerate().all(|(m, level)| {
            let stored = if zstd {
                level.uncompressed_byte_length
            } else {
                level.data.len() as u64
            };
            Some(stored) == (k.image_bytes(m as u32) as u64).checked_mul(images)
        });
        sized.then_some(k)
    }

    fn vk_format(&self) -> u32 {
        self.format.value()
    }

    /// Bytes of one image of level `m`.
    fn image_bytes(&self, m: u32) -> usize {
        let [w, h] = mip_size(self.header.pixel_width, self.header.pixel_height, m);
        self.info.mip_bytes(w, h)
    }

    fn srgb(&self) -> bool {
        self.format.normalize().1 == ColorSpace::Srgb
    }

    fn open(self) -> Result<Source, String> {
        let h = &self.header;
        let cube = h.face_count == 6;
        if cube && h.layer_count > 1 {
            return Err("cube arrays are not supported".into());
        }
        // Each level holds every layer's faces in turn.
        let images = h.layer_count.max(1) as usize * h.face_count as usize;
        let mut levels = vec![Vec::new(); images];
        for (m, level) in self.reader.levels().enumerate() {
            let size = self.image_bytes(m as u32);
            let data = if self.zstd {
                let data = zstd::bulk::decompress(level.data, size * images)
                    .map_err(|e| format!("KTX2 level {m}: {e}"))?;
                if data.len() != size * images {
                    return Err(format!("KTX2 level {m} holds {} bytes", data.len()));
                }
                Cow::Owned(data)
            } else {
                Cow::Borrowed(level.data)
            };
            if images == 1 {
                levels[0].push(data.into_owned());
            } else {
                for (l, image) in levels.iter_mut().zip(data.chunks_exact(size)) {
                    l.push(image.to_vec());
                }
            }
        }
        Ok(Source {
            layers: Vec::new(),
            cube,
            srgb: self.srgb(),
            primaries: None,
            rg: two_channels(self.vk_format()),
            signed: signed_blocks(self.vk_format()),
            blocks: Some(Blocks {
                vk_format: self.vk_format(),
                w: h.pixel_width,
                h: h.pixel_height,
                levels,
            }),
        })
    }

    fn scan(self, spread: f32) -> Result<Scan, String> {
        let h = &self.header;
        if h.face_count != 1 || h.layer_count > 1 {
            return Err("not a 2D image".into());
        }
        let (w, hh) = (h.pixel_width, h.pixel_height);
        let level = self.reader.levels().next().ok_or("KTX2 without levels")?;
        let vk_format = self.vk_format();
        let range = if self.zstd {
            // Whole chunks per decompress call, not one block at a time.
            let d =
                zstd::stream::read::Decoder::with_buffer(level.data).map_err(|e| e.to_string())?;
            block_range(
                vk_format,
                w,
                hh,
                &mut BufReader::with_capacity(1 << 16, d),
                spread,
            )?
        } else {
            block_range(vk_format, w, hh, &mut &level.data[..], spread)?
        };
        Ok(Scan {
            w,
            h: hh,
            vk_format,
            complete: h.level_count.max(1) == full_mip_count(w, hh),
            srgb: self.srgb(),
            rg: two_channels(vk_format),
            signed: signed_blocks(vk_format),
            range,
        })
    }
}

/// Whether blocks of `vk_format` hold two channels, red and green.
fn two_channels(vk_format: u32) -> bool {
    matches!(vk_format, vk::BC5_UNORM | vk::BC5_SNORM)
}

fn signed_blocks(vk_format: u32) -> bool {
    matches!(vk_format, vk::BC4_SNORM | vk::BC5_SNORM)
}

/// Per channel, the lowest and highest value of a `w` x `h` mip of `vk_format` blocks read
/// from `data`, decoded block by block until every channel the format stores spans more
/// than `spread`.
pub(super) fn block_range(
    vk_format: u32,
    w: u32,
    h: u32,
    data: &mut impl Read,
    spread: f32,
) -> Result<[(f32, f32); 4], String> {
    let f = FormatInfo::of(vk_format).ok_or("unknown format")?;
    let bw = w.div_ceil(4) as usize;
    let blocks = bw * h.div_ceil(4) as usize;
    // Channels past these hold the same value in every block of the format.
    let stored = match vk_format {
        vk::BC4_UNORM | vk::BC4_SNORM => 1,
        vk::BC5_UNORM | vk::BC5_SNORM => 2,
        vk::BC1_RGB_UNORM | vk::BC1_RGB_SRGB | vk::BC6H_UFLOAT | vk::BC6H_SFLOAT => 3,
        _ => 4,
    };
    let bc7 = matches!(vk_format, vk::BC7_UNORM | vk::BC7_SRGB);
    let mut range = [(f32::MAX, f32::MIN); 4];
    let mut buf = [0; 16];
    let block = &mut buf[..f.bytes as usize];
    for i in 0..blocks {
        if range[..stored].iter().all(|r| r.1 - r.0 > spread) {
            break;
        }
        data.read_exact(block)
            .map_err(|_| "compressed mip size".to_string())?;
        let block = &*block;
        // Once color varies, only alpha is left to find, and opaque BC7 blocks give 1.
        if bc7 && bc7_opaque(block) && range[..3].iter().all(|r| r.1 - r.0 > spread) {
            range[3] = (range[3].0.min(1.0), range[3].1.max(1.0));
            continue;
        }
        let texels = decode_block(vk_format, block)?;
        let (bx, by) = (i % bw * 4, i / bw * 4);
        for (k, t) in texels.iter().enumerate() {
            if bx + k % 4 < w as usize && by + k / 4 < h as usize {
                for (r, v) in range.iter_mut().zip(t) {
                    *r = (r.0.min(*v), r.1.max(*v));
                }
            }
        }
    }
    Ok(range)
}

/// Whether every texel of a BC7 block decodes with alpha 255: modes 0 to 3 store no
/// alpha, and a mode 6 block whose alpha endpoints and p-bits are all ones holds 255 at
/// both ends.
fn bc7_opaque(block: &[u8]) -> bool {
    let bits = u64::from_le_bytes(block[..8].try_into().expect("16-byte block"));
    block[0] & 0x0f != 0 || (block[0] & 0x7f == 0x40 && bits >> 49 == 0x7fff && block[8] & 1 == 1)
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
        return Ok(Source {
            layers: Vec::new(),
            cube,
            srgb,
            primaries: None,
            blocks: Some(Blocks {
                vk_format,
                w,
                h,
                levels,
            }),
            rg: two_channels(vk_format),
            signed: signed_blocks(vk_format),
        });
    }
    // Uncompressed: ctt converts to f32 RGBA in the file's color space. Read as straight alpha
    // it would premultiply and divide back, which loses color where alpha is 0 or negative.
    let mip0 = ctt::Image {
        surfaces: image.surfaces.iter().map(|l| vec![l[0].clone()]).collect(),
        kind: TextureKind::Texture2D,
        desc: ctt::FormatDesc {
            alpha: ctt::AlphaMode::Opaque,
            ..image.desc
        },
    };
    let out = ctt::convert(
        mip0.to_ref(),
        ctt::ConvertSettings {
            format: Some(ctt::TargetFormat::Uncompressed(
                ctt::Format::R32G32B32A32_SFLOAT,
            )),
            container: ctt::Container::Raw,
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
        rg: format.channel_count() == Some(2),
        signed: is_snorm(format),
    })
}

/// Whether uncompressed `format` stores signed normalized channels.
fn is_snorm(format: ctt::Format) -> bool {
    use ctt::Format as F;
    matches!(
        format,
        F::R8_SNORM
            | F::R8G8_SNORM
            | F::R8G8B8_SNORM
            | F::B8G8R8_SNORM
            | F::R8G8B8A8_SNORM
            | F::B8G8R8A8_SNORM
            | F::R16_SNORM
            | F::R16G16_SNORM
            | F::R16G16B16_SNORM
            | F::R16G16B16A16_SNORM
            | F::A2R10G10B10_SNORM_PACK32
            | F::A2B10G10R10_SNORM_PACK32
    )
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
            p => return Err(format!("PNG cICP color primaries {p} are not supported")),
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
        rg: false,
        signed: false,
    })
}

/// An image's size and whether it is an animated WebP, from its headers; `None` when `image`
/// does not read the file, and decoding it reports why.
fn header<R: BufRead + Seek>(reader: image::ImageReader<R>) -> Option<([u32; 2], bool)> {
    if reader.format() == Some(image::ImageFormat::WebP) {
        let d = WebPDecoder::new(reader.into_inner()).ok()?;
        let (w, h) = d.dimensions();
        return Some(([w, h], d.has_animation()));
    }
    let (w, h) = reader.into_dimensions().ok()?;
    Some(([w, h], false))
}

/// [`header`] of the image file at `path`.
pub(crate) fn file_header(path: &Path) -> Option<([u32; 2], bool)> {
    image::ImageReader::open(path)
        .and_then(image::ImageReader::with_guessed_format)
        .ok()
        .and_then(header)
}

/// Whether an image file's bytes are an animated WebP.
pub(super) fn is_animated(bytes: &[u8]) -> bool {
    image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()
        .and_then(header)
        .is_some_and(|h| h.1)
}

fn png_cicp(bytes: &[u8]) -> Result<Option<png::CodingIndependentCodePoints>, String> {
    let reader = png::Decoder::new(Cursor::new(bytes))
        .read_info()
        .map_err(|e| e.to_string())?;
    Ok(reader.info().coding_independent_code_points)
}

/// Decodes one layer's BCn mip to float texels (unorm values in 0..1, snorm values in -1..1,
/// BC6H as floats).
pub(crate) fn decode_blocks(vk_format: u32, w: u32, h: u32, data: &[u8]) -> Result<Img, String> {
    decode_blocks_with(vk_format, w, h, data, |t| t)
}

/// [`decode_blocks`] with each texel passed through `f`.
pub(super) fn decode_blocks_with(
    vk_format: u32,
    w: u32,
    h: u32,
    data: &[u8],
    f: impl Fn([f32; 4]) -> [f32; 4] + Sync,
) -> Result<Img, String> {
    let mut out = Img::new(w, h);
    decode_rows(vk_format, w, h, data, &mut out.px, f)?;
    Ok(out)
}

/// Channel `c` of one layer's BCn mip, decoded as [`decode_blocks`] decodes it.
pub(crate) fn decode_channel(
    vk_format: u32,
    w: u32,
    h: u32,
    data: &[u8],
    c: usize,
) -> Result<Vec<f32>, String> {
    let mut out = vec![0.0; w as usize * h as usize];
    decode_rows(vk_format, w, h, data, &mut out, |t| t[c])?;
    Ok(out)
}

/// Decodes a `w` x `h` mip of blocks into row-major `out`, one row of blocks per task, each
/// texel through `f`.
fn decode_rows<T: Send>(
    vk_format: u32,
    w: u32,
    h: u32,
    data: &[u8],
    out: &mut [T],
    f: impl Fn([f32; 4]) -> T + Sync,
) -> Result<(), String> {
    let info = FormatInfo::of(vk_format).ok_or("unknown format")?;
    if data.len() != info.mip_bytes(w, h) {
        return Err("compressed mip size".into());
    }
    let (w, bytes) = (w as usize, info.bytes as usize);
    let bw = w.div_ceil(4);
    out.par_chunks_mut(4 * w)
        .zip(data.par_chunks(bw * bytes))
        .try_for_each(|(rows, blocks)| {
            for (bx, block) in blocks.chunks_exact(bytes).enumerate() {
                let texels = decode_block(vk_format, block)?;
                for (i, t) in texels.into_iter().enumerate() {
                    let (x, y) = (bx * 4 + i % 4, i / 4);
                    if x < w && y * w < rows.len() {
                        rows[y * w + x] = f(t);
                    }
                }
            }
            Ok::<_, String>(())
        })?;
    Ok(())
}

/// The 16 texels of `channels` bytes each in `b`, every byte read by `value`.
fn block_texels(b: &[u8], channels: usize, value: impl Fn(u8) -> f32) -> [[f32; 4]; 16] {
    std::array::from_fn(|i| {
        let mut p = [0.0, 0.0, 0.0, 1.0];
        for c in 0..channels {
            p[c] = value(b[i * channels + c]);
        }
        p
    })
}

fn decode_block(vk_format: u32, block: &[u8]) -> Result<[[f32; 4]; 16], String> {
    use vk::*;
    let unorm = |b: &[u8], channels| block_texels(b, channels, |v| f32::from(v) / 255.0);
    // Both -128 and -127 stand for -1.
    let snorm = |b: &[u8], channels| {
        block_texels(b, channels, |v| {
            (f32::from(v.cast_signed()) / 127.0).max(-1.0)
        })
    };
    let mut b8 = [0u8; 64];
    Ok(match vk_format {
        BC1_RGB_UNORM | BC1_RGB_SRGB | BC1_RGBA_UNORM | BC1_RGBA_SRGB => {
            bcdec_rs::bc1(block, &mut b8, 16);
            // bcdec reads index 3 of a c0 <= c1 block as transparent; only RGBA formats are.
            if matches!(vk_format, BC1_RGB_UNORM | BC1_RGB_SRGB) {
                b8.iter_mut().skip(3).step_by(4).for_each(|a| *a = 255);
            }
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
        BC4_SNORM => {
            bcdec_rs::bc4(block, &mut b8[..16], 4, true);
            snorm(&b8, 1)
        }
        BC5_SNORM => {
            bcdec_rs::bc5(block, &mut b8[..32], 8, true);
            snorm(&b8, 2)
        }
        BC6H_UFLOAT | BC6H_SFLOAT => {
            let mut f = [0.0f32; 48];
            bcdec_rs::bc6h_float(block, &mut f, 12, vk_format == BC6H_SFLOAT);
            std::array::from_fn(|i| [f[i * 3], f[i * 3 + 1], f[i * 3 + 2], 1.0])
        }
        _ => return Err(format!("format {vk_format} does not decode on the CPU")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 32x32, each quadrant a flat color and alpha, from libwebp at quality 90 with lossless
    /// alpha: a VP8X file with ALPH and VP8 chunks.
    const LOSSY_WEBP: &str = concat!(
        "524946469e00000057454250565038580a000000100000001f00001f0000414c5048140000000120",
        "d2b6b132a9777e383c161113703ddd1bba2f56503820640000005005009d012a200020003e291086",
        "42a1a10dfa000c01425a00274ca11c0de01f801fb01feab9c3340bb01fb019df617016898000fef6",
        "9597fe74f49d257ff5b2ff05592a462084c37acf1af816dfeb454dfd8886267da1eeffda136acacd",
        "97545137c000",
    );

    #[test]
    fn lossy_webp_decodes_its_colors_and_alpha() {
        let bytes: Vec<u8> = (0..LOSSY_WEBP.len() / 2)
            .map(|i| u8::from_str_radix(&LOSSY_WEBP[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        let src = decode(&bytes).unwrap();
        assert!(src.srgb && !src.rg && src.blocks.is_none() && !is_animated(&bytes));
        let img = &src.layers[0];
        assert_eq!((img.w, img.h), (32, 32));
        let quadrants: [[u8; 4]; 4] = [
            [200, 40, 40, 255],
            [40, 200, 40, 128],
            [40, 40, 200, 255],
            [220, 220, 60, 32],
        ];
        for y in 0..32usize {
            for x in 0..32usize {
                // Chroma is stored at half size, so colors blend across quadrant edges.
                if x.abs_diff(16) < 4 || y.abs_diff(16) < 4 {
                    continue;
                }
                let want = quadrants[usize::from(y >= 16) * 2 + usize::from(x >= 16)];
                let got = img.px[y * 32 + x].map(|c| c * 255.0);
                for c in 0..3 {
                    let error = (got[c] - f32::from(want[c])).abs();
                    assert!(error <= 6.0, "({x}, {y}) channel {c}: {got:?} for {want:?}");
                }
                // The alpha chunk is lossless.
                assert_eq!(got[3].round(), f32::from(want[3]), "({x}, {y})");
            }
        }
    }

    #[test]
    fn bc5_snorm_decodes_to_signed_values() {
        // Red: index 1 of endpoints 127 and -127 in every texel. Green: endpoints of 0.
        let mut block = [0u8; 16];
        block[..8].copy_from_slice(&[127, 0x81, 0x49, 0x92, 0x24, 0x49, 0x92, 0x24]);
        let texels = decode_block(vk::BC5_SNORM, &block).unwrap();
        assert!(texels.iter().all(|t| t[..2] == [-1.0, 0.0]), "{texels:?}");
    }

    #[test]
    fn bc1_rgb_blocks_have_no_alpha() {
        // c0 <= c1 with every index 3: black, and transparent only where the format has alpha.
        let block = [0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
        let texels = decode_block(vk::BC1_RGB_UNORM, &block).unwrap();
        assert!(
            texels.iter().all(|t| *t == [0.0, 0.0, 0.0, 1.0]),
            "{texels:?}"
        );
        let texels = decode_block(vk::BC1_RGBA_UNORM, &block).unwrap();
        assert!(texels.iter().all(|t| t[3] == 0.0), "{texels:?}");
    }

    #[test]
    fn opaque_bc7_blocks_decode_with_alpha_255() {
        let mut seed = 3u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as u8
        };
        let mut opaque = 0;
        for k in 0..200_000 {
            let mut block: [u8; 16] = std::array::from_fn(|_| next());
            // Every mode, and mode 6 blocks with their alpha bits mostly set.
            let mode = k % 8;
            block[0] = (block[0] & !(((2u16 << mode) - 1) as u8)) | (1 << mode);
            if mode == 6 && k % 3 != 0 {
                block[6] |= 0xfe;
                block[7] = 0xff;
                block[8] |= u8::from(k % 5 != 0);
            }
            let mut out = [0u8; 64];
            bcdec_rs::bc7(&block, &mut out, 16);
            let decoded = out.chunks_exact(4).all(|t| t[3] == 255);
            if bc7_opaque(&block) {
                opaque += 1;
                assert!(decoded, "{block:02x?}");
            }
        }
        assert!(opaque > 80_000);
    }
}
