//! Image files to float texels: PNG, JPEG, WebP, Radiance HDR and OpenEXR through `image`,
//! KTX2 and DDS through ctt's readers, block-compressed texels decoded with bcdec. KTX2 files
//! of BCn blocks, plain or zstd-compressed, are read here instead, each level decompressed
//! once straight into its blocks. Formats are recognized by their bytes. An animated WebP
//! decodes to its first frame. The spec leaves `KTXswizzle` to applications; the cook applies
//! it, because the file states it: decoded texels are the swizzled ones, and a material that
//! keeps a file's blocks reads its channels through the swizzle ([`open_blocks`]).
//!
//! A [`Source`] holds a file's values as stored, beside what the file states of them
//! ([`FileColor`]); the importers' rules decide which of that applies. Radiance's `EXPOSURE` and
//! `COLORCORR` are the exception: they state how the values were scaled, which no rule reads past,
//! so they are undone here.

use std::borrow::Cow;
use std::io::{BufRead, BufReader, Cursor, Read, Seek};
use std::path::Path;

use crate::par::prelude::*;
use crate::simd;
use ctt::input::{InputOverrides, decode_container};
use ctt::{ColorSpace, FormatExt, TextureKind};
use image::ImageDecoder;
use image::codecs::hdr::HdrDecoder;
use image::codecs::webp::WebPDecoder;
use orr_pack::texture::{FormatInfo, full_mip_count, mip_size, vk};

use super::color::{self, Codes, FileColor, Gamut, ImageColor, Transfer};
use super::material::Layer;
use super::{Img, Wrap};

/// What a file of one channel, or of gray and alpha, holds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum Gray {
    No,
    /// Gray alone, which decoded texels spread over RGB.
    Alone,
    /// Gray and alpha, likewise.
    WithAlpha,
    /// One channel, which decoded texels hold in red alone.
    Red,
}

/// A decoded image file.
pub(super) struct Source {
    /// Mip 0 of each layer, values as stored: unorm data in 0..1, snorm data in -1..1, floats
    /// as they are.
    pub(super) layers: Vec<Img>,
    /// Six layers in the Vulkan face order.
    pub(super) cube: bool,
    pub(super) color: FileColor,
    /// The block-compressed texels of a compressed file, kept so an encoding that matches
    /// can pass them through.
    pub(super) blocks: Option<Blocks>,
    /// The file holds two channels, red and green.
    pub(super) rg: bool,
    /// The file stores signed normalized values.
    pub(super) signed: bool,
    /// The file stores floats, whose range no format of fixed span holds.
    pub(super) float: bool,
    /// The file's format can hold values below 0.
    pub(super) negatives: bool,
    pub(super) gray: Gray,
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

/// What the `wrapmodes` attribute of an OpenEXR file says it reads past 0..1, per axis: one
/// keyword for both, or two separated by a comma.
pub(super) fn exr_wrap(bytes: &[u8]) -> Option<[Wrap; 2]> {
    if !bytes.starts_with(&exr::meta::magic_number::BYTES) {
        return None;
    }
    let meta = exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false).ok()?;
    let modes = meta
        .headers
        .first()?
        .own_attributes
        .wrap_mode_name
        .as_ref()?;
    let mode = |m: &str| match m.trim() {
        "black" => Some(Wrap::Border),
        "clamp" => Some(Wrap::Clamp),
        "periodic" => Some(Wrap::Repeat),
        "mirror" => Some(Wrap::Mirror),
        _ => None,
    };
    let modes = modes.to_string();
    let (u, v) = modes.split_once(',').unwrap_or((&modes, &modes));
    Some([mode(u)?, mode(v)?])
}

/// The short name of `channel` of OpenEXR part `part`, if it lies directly in layer `name`: a
/// channel states its layers in its own name, and a part's name may lead them.
fn in_layer(
    name: &str,
    part: Option<&exr::meta::attribute::Text>,
    channel: &str,
) -> Option<String> {
    let under = |full: &str| {
        let short = full.strip_prefix(name)?.strip_prefix('.')?;
        (!short.contains('.')).then(|| short.to_string())
    };
    under(channel).or_else(|| under(&format!("{}.{channel}", part?)))
}

/// The layer of an OpenEXR file that [`Layer::Implied`] reads, if the file has one.
fn implied_layer(bytes: &[u8]) -> Option<&'static str> {
    if !bytes.starts_with(&exr::meta::magic_number::BYTES) {
        return None;
    }
    let meta = exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false).ok()?;
    ["default", "rgba"].into_iter().find(|name| {
        meta.headers.iter().any(|h| {
            let part = h.own_attributes.layer_name.as_ref();
            let held = |c: &exr::meta::attribute::ChannelDescription| {
                in_layer(name, part, &c.name.to_string()).is_some()
            };
            h.channels.list.iter().any(held)
        })
    })
}

/// What `layer` reads of an image file, as a file of its own, or none where that is the file as
/// it stands. Only OpenEXR files hold layers. The new file keeps the display window, data window
/// and attributes of the part that holds the layer. Its red, green, blue and alpha are the
/// channels of those names, or where the layer has no color, its other channels in file order
/// from red. `rgba` names a file's own channels where no layer has that name.
pub(super) fn layer_of(bytes: &[u8], layer: &Layer) -> Result<Option<Vec<u8>>, String> {
    use exr::image::{AnyChannel, Encoding, FlatSamples};
    use exr::prelude::{Image, ReadChannels, ReadLayers, SpecificChannels, WritableImage, read};
    let name = match layer {
        Layer::Whole => return Ok(None),
        Layer::Named(name) => name.as_str(),
        Layer::Implied => match implied_layer(bytes) {
            Some(name) => name,
            None => return Ok(None),
        },
    };
    let absent = || match name {
        "rgba" => Ok(None),
        _ => Err(format!("it has no layer {name}")),
    };
    if !bytes.starts_with(&exr::meta::magic_number::BYTES) {
        return absent();
    }
    let file = read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .all_layers()
        .all_attributes()
        .non_parallel()
        .from_buffered(Cursor::new(bytes))
        .map_err(|e| e.to_string())?;
    let held = file.layer_data.iter().find_map(|part| {
        let held: Vec<_> = part
            .channel_data
            .list
            .iter()
            .filter_map(|c| {
                let short = in_layer(
                    name,
                    part.attributes.layer_name.as_ref(),
                    &c.name.to_string(),
                );
                Some((short?, c))
            })
            .collect();
        (!held.is_empty()).then_some((part, held))
    });
    let Some((part, held)) = held else {
        return absent();
    };
    let mut rgba: [Option<&AnyChannel<FlatSamples>>; 4] = [None; 4];
    let mut others = Vec::new();
    for (short, channel) in held {
        match short.to_ascii_lowercase().as_str() {
            "r" | "red" => rgba[0] = Some(channel),
            "g" | "green" => rgba[1] = Some(channel),
            "b" | "blue" => rgba[2] = Some(channel),
            "a" | "alpha" => rgba[3] = Some(channel),
            _ => others.push(channel),
        }
    }
    if rgba[..3].iter().all(Option::is_none) {
        if others.is_empty() {
            others.extend(rgba[3].take());
        }
        for (slot, channel) in rgba.iter_mut().zip(others).take(3) {
            *slot = Some(channel);
        }
    }
    let [red, green, blue, alpha] = rgba;
    let width = part.size.width();
    let sample = |c: Option<&AnyChannel<FlatSamples>>, pos: exr::prelude::Vec2<usize>, or: f32| {
        c.map_or(or, |c| {
            let i = pos.y() * width + pos.x();
            c.sample_data.value_by_flat_index(i).to_f32()
        })
    };
    let rgb = |pos| {
        (
            sample(red, pos, 0.0),
            sample(green, pos, 0.0),
            sample(blue, pos, 0.0),
        )
    };
    let attributes = || file.attributes.clone();
    let own = || part.attributes.clone();
    let mut out = Cursor::new(Vec::new());
    // Without alpha the window around the data stays opaque, as it does reading the whole file.
    match alpha {
        Some(_) => {
            let channels = SpecificChannels::rgba(|pos| {
                let (r, g, b) = rgb(pos);
                (r, g, b, sample(alpha, pos, 1.0))
            });
            let layer = exr::image::Layer::new(part.size, own(), Encoding::UNCOMPRESSED, channels);
            Image::new(attributes(), layer)
                .write()
                .non_parallel()
                .to_buffered(&mut out)
        }
        None => {
            let channels = SpecificChannels::rgb(rgb);
            let layer = exr::image::Layer::new(part.size, own(), Encoding::UNCOMPRESSED, channels);
            Image::new(attributes(), layer)
                .write()
                .non_parallel()
                .to_buffered(&mut out)
        }
    }
    .map_err(|e| e.to_string())?;
    Ok(Some(out.into_inner()))
}

/// Decodes the mip 0 of each layer of an image file, recognized by its first bytes. A KTX2 file's
/// other levels stay packed.
pub(super) fn decode(bytes: &[u8]) -> Result<Source, String> {
    let mut src = open_levels(bytes, 1)?;
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

/// Whether `test` holds of an image `src` stores: mip 0 of each layer, and of blocks every mip,
/// which a cook can keep, each block's texels read through `swizzled`. Each is decoded for the
/// test, so it stops at the first that holds.
pub(super) fn any_stored(
    src: &Source,
    swizzled: Option<Swizzle>,
    mut test: impl FnMut(&Img) -> bool,
) -> Result<bool, String> {
    let Some(b) = &src.blocks else {
        return Ok(src.layers.iter().any(test));
    };
    for layer in &b.levels {
        for (m, bytes) in layer.iter().enumerate() {
            let [w, h] = mip_size(b.w, b.h, m as u32);
            let mut img = decode_blocks(b.vk_format, w, h, bytes)?;
            if let Some(s) = swizzled {
                swizzle(&mut img.px, s);
            }
            if test(&img) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Per channel, the lowest and highest finite value of every image `src` stores, its blocks read
/// through `swizzled`.
pub(super) fn stored_range(
    src: &Source,
    swizzled: Option<Swizzle>,
) -> Result<[(f32, f32); 4], String> {
    let mut range = [(f32::MAX, f32::MIN); 4];
    any_stored(src, None, |l| {
        for (c, r) in range.iter_mut().enumerate() {
            let (lo, hi) = l.finite_range(c);
            *r = (r.0.min(lo), r.1.max(hi));
        }
        false
    })?;
    Ok(swizzled.map_or(range, |s| swizzle_range(range, s)))
}

/// Per channel, the range of a channel read through `s` from channels of `range`.
fn swizzle_range(range: [(f32, f32); 4], s: Swizzle) -> [(f32, f32); 4] {
    s.map(|c| match c {
        Component::Channel(i) => range[i],
        Component::Value(v) => (v, v),
    })
}

/// Opens an image file with every mip it stores, leaving `layers` empty when the file holds
/// block-compressed texels.
pub(super) fn open(bytes: &[u8]) -> Result<Source, String> {
    open_levels(bytes, usize::MAX)
}

/// [`open`] with its blocks as stored, and what each channel reads of them where the file's
/// `KTXswizzle` changes any. Its blocks decode to the stored texels: [`swizzle`] makes them
/// what [`open`] decodes. A file without blocks is as [`open`] reads it.
pub(super) fn open_blocks(bytes: &[u8]) -> Result<(Source, Option<Swizzle>), String> {
    let src = open_stored(bytes, usize::MAX)?;
    match ktx2_swizzle(bytes) {
        Some(s) if src.blocks.is_none() => Ok((swizzled(src, s)?, None)),
        s => Ok((src, s)),
    }
}

/// Texels `px` as swizzle `s` reads them.
pub(super) fn swizzle(px: &mut [[f32; 4]], s: Swizzle) {
    simd::update(px, |p| {
        let t = *p;
        *p = s.map(|c| match c {
            Component::Channel(i) => t[i],
            Component::Value(v) => v,
        });
    });
}

/// [`open`], taking at most `levels` mips of a KTX2 file.
fn open_levels(bytes: &[u8], levels: usize) -> Result<Source, String> {
    let src = open_stored(bytes, levels)?;
    match ktx2_swizzle(bytes) {
        Some(s) => swizzled(src, s),
        None => Ok(src),
    }
}

/// `src` decoded and read through swizzle `s`, without its blocks.
fn swizzled(mut src: Source, s: Swizzle) -> Result<Source, String> {
    decode_layers(&mut src)?;
    src.blocks = None;
    src.gray = Gray::No;
    // Past the file's two channels, only what decoding fills in leaves it two.
    src.rg &= s[2].keeps(2, 2) && s[3].keeps(3, 2);
    for l in &mut src.layers {
        swizzle(&mut l.px, s);
    }
    Ok(src)
}

/// [`open_levels`] without the file's `KTXswizzle`.
fn open_stored(bytes: &[u8], levels: usize) -> Result<Source, String> {
    const KTX1: &[u8] = b"\xABKTX 11\xBB\r\n\x1A\n";
    if bytes.starts_with(KTX1) {
        return Err("KTX1 images are not supported; convert them to KTX2".into());
    }
    Ok(match Ktx2Blocks::of(bytes) {
        Some(k) => k.open(levels)?,
        None => {
            let dds = dds_with_known_blocks(bytes);
            match decode_container(&dds, InputOverrides::default()).map_err(|e| e.to_string())? {
                Some(image) => container(image, bytes)?,
                None => file(bytes)?,
            }
        }
    })
}

const DDS_MAGIC: &[u8] = b"DDS ";
const DDS_FOUR_CC: usize = 84;
const DDS_ALPHA_MODE: usize = 144;
const DDS_ALPHA_MODE_PREMULTIPLIED: u32 = 2;

/// A DDS file's bytes with DXT2 and DXT4 as DXT3 and DXT5, whose blocks they share: they differ
/// in premultiplied alpha, which [`dds_premultiplied`] reads, and the DDS reader knows only those.
fn dds_with_known_blocks(bytes: &[u8]) -> Cow<'_, [u8]> {
    let same = match bytes.get(DDS_FOUR_CC..DDS_FOUR_CC + 4) {
        _ if !bytes.starts_with(DDS_MAGIC) => return Cow::Borrowed(bytes),
        Some(b"DXT2") => b"DXT3",
        Some(b"DXT4") => b"DXT5",
        _ => return Cow::Borrowed(bytes),
    };
    let mut patched = bytes.to_vec();
    patched[DDS_FOUR_CC..DDS_FOUR_CC + 4].copy_from_slice(same);
    Cow::Owned(patched)
}

/// Whether a DDS file states premultiplied alpha: by its legacy block format, or by the alpha
/// mode in `miscFlags2` of its `DDS_HEADER_DXT10`.
fn dds_premultiplied(bytes: &[u8]) -> bool {
    match bytes.get(DDS_FOUR_CC..DDS_FOUR_CC + 4) {
        Some(b"DXT2" | b"DXT4") => true,
        Some(b"DX10") => bytes
            .get(DDS_ALPHA_MODE..DDS_ALPHA_MODE + 4)
            .is_some_and(|b| {
                let flags = u32::from_le_bytes(b.try_into().expect("4 bytes"));
                flags & 7 == DDS_ALPHA_MODE_PREMULTIPLIED
            }),
        _ => false,
    }
}

/// What a KTX2 file's DFD states of `format`. A file's `_SRGB` format decodes on any sampler, so
/// it stands over a transfer function that disagrees.
fn ktx2_color(reader: &ktx2::Reader<&[u8]>, format: ctt::Format, numeric: bool) -> FileColor {
    let stated = reader
        .transfer_function()
        .and_then(color::ktx2_transfer)
        .unwrap_or(Transfer::Linear);
    FileColor {
        transfer: Some(if format.normalize().1 == ColorSpace::Srgb {
            Transfer::Srgb
        } else {
            stated
        }),
        gamut: reader.color_primaries().and_then(color::ktx2_gamut),
        premultiplied: reader.is_alpha_premultiplied() == Some(true),
        numeric,
        ..Default::default()
    }
}

/// What the KTX2 or DDS file `bytes` states of `format`.
fn container_color(bytes: &[u8], format: ctt::Format, srgb: bool, numeric: bool) -> FileColor {
    match ktx2::Reader::new(bytes) {
        Ok(reader) => ktx2_color(&reader, format, numeric),
        Err(_) => FileColor {
            transfer: Some(if srgb {
                Transfer::Srgb
            } else {
                Transfer::Linear
            }),
            premultiplied: dds_premultiplied(bytes),
            numeric,
            ..Default::default()
        },
    }
}

/// What each channel reads under a `KTXswizzle`.
pub(super) type Swizzle = [Component; 4];

/// What one channel of a `KTXswizzle` mapping reads.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum Component {
    Channel(usize),
    Value(f32),
}

impl Component {
    /// Whether this leaves channel `c` of a file of `held` channels as it decodes: absent channels
    /// decode to 0, with alpha 1.
    fn keeps(self, c: usize, held: usize) -> bool {
        let absent = if c == 3 { 1.0 } else { 0.0 };
        self == Self::Channel(c) || (c >= held && self == Self::Value(absent))
    }
}

/// The `KTXswizzle` of a KTX2 file, unless it leaves every channel as it decodes.
fn ktx2_swizzle(bytes: &[u8]) -> Option<Swizzle> {
    let reader = ktx2::Reader::new(bytes).ok()?;
    let (_, value) = reader
        .key_value_data()
        .find(|(key, _)| *key == "KTXswizzle")?;
    let text = value.strip_suffix(b"\0").unwrap_or(value);
    let read = |c: u8| match c {
        b'r' => Some(Component::Channel(0)),
        b'g' => Some(Component::Channel(1)),
        b'b' => Some(Component::Channel(2)),
        b'a' => Some(Component::Channel(3)),
        b'0' => Some(Component::Value(0.0)),
        b'1' => Some(Component::Value(1.0)),
        _ => None,
    };
    let s = match *text {
        [r, g, b, a] => [read(r)?, read(g)?, read(b)?, read(a)?],
        _ => return None,
    };
    let held = reader
        .header()
        .format
        .and_then(|f| f.channel_count())
        .unwrap_or(4);
    (!(0..4).all(|c| s[c].keeps(c, held))).then_some(s)
}

/// What [`open`] tells of a KTX2 file of BCn blocks holding one 2D image, with the range of
/// its mip 0 as [`block_range`] finds it, decompressing only as much as the range needs.
pub(crate) struct Scan {
    pub(super) w: u32,
    pub(super) h: u32,
    pub(super) vk_format: u32,
    /// Every mip down to 1x1 is present.
    pub(super) complete: bool,
    pub(super) color: FileColor,
    pub(super) rg: bool,
    pub(super) signed: bool,
    pub(super) float: bool,
    pub(super) negatives: bool,
    pub(super) gray: Gray,
    /// Of the channels as swizzled, as every field above but `vk_format`.
    pub(super) range: [(f32, f32); 4],
    /// What each channel reads of the blocks, where the file's `KTXswizzle` changes any.
    pub(super) swizzle: Option<Swizzle>,
}

/// The [`Scan`] of a KTX2 file of BCn blocks; `None` for every other file.
pub(crate) fn scan_ktx2(bytes: &[u8], spread: f32) -> Option<Result<Scan, String>> {
    let scan = Ktx2Blocks::of(bytes)?.scan(spread);
    let Some(s) = ktx2_swizzle(bytes) else {
        return Some(scan);
    };
    Some(scan.map(|mut scan| {
        scan.range = swizzle_range(scan.range, s);
        scan.gray = Gray::No;
        scan.rg &= s[2].keeps(2, 2) && s[3].keeps(3, 2);
        scan.swizzle = Some(s);
        scan
    }))
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

    fn color(&self) -> FileColor {
        let vk_format = self.vk_format();
        let numeric = float_blocks(vk_format) || signed_blocks(vk_format);
        ktx2_color(&self.reader, self.format, numeric)
    }

    fn open(self, levels_kept: usize) -> Result<Source, String> {
        let h = &self.header;
        let cube = h.face_count == 6;
        if cube && h.layer_count > 1 {
            return Err("cube arrays are not supported".into());
        }
        // Each level holds every layer's faces in turn.
        let images = h.layer_count.max(1) as usize * h.face_count as usize;
        let mut levels = vec![Vec::new(); images];
        for (m, level) in self.reader.levels().take(levels_kept).enumerate() {
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
            color: self.color(),
            rg: two_channels(self.vk_format()),
            signed: signed_blocks(self.vk_format()),
            float: float_blocks(self.vk_format()),
            negatives: negative_blocks(self.vk_format()),
            gray: gray_blocks(self.vk_format()),
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
            color: self.color(),
            rg: two_channels(vk_format),
            signed: signed_blocks(vk_format),
            float: float_blocks(vk_format),
            negatives: negative_blocks(vk_format),
            gray: gray_blocks(vk_format),
            range,
            swizzle: None,
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

fn float_blocks(vk_format: u32) -> bool {
    matches!(vk_format, vk::BC6H_UFLOAT | vk::BC6H_SFLOAT)
}

fn gray_blocks(vk_format: u32) -> Gray {
    if matches!(vk_format, vk::BC4_UNORM | vk::BC4_SNORM) {
        Gray::Red
    } else {
        Gray::No
    }
}

fn negative_blocks(vk_format: u32) -> bool {
    signed_blocks(vk_format) || vk_format == vk::BC6H_SFLOAT
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
    // No fixed span says how far floats reach, so their readers need all of the range.
    let spread = if float_blocks(vk_format) {
        f32::INFINITY
    } else {
        spread
    };
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

fn container(image: ctt::Image, bytes: &[u8]) -> Result<Source, String> {
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
            color: container_color(
                bytes,
                format,
                srgb,
                float_blocks(vk_format) || signed_blocks(vk_format),
            ),
            blocks: Some(Blocks {
                vk_format,
                w,
                h,
                levels,
            }),
            rg: two_channels(vk_format),
            signed: signed_blocks(vk_format),
            float: float_blocks(vk_format),
            negatives: negative_blocks(vk_format),
            gray: gray_blocks(vk_format),
        });
    }
    // Uncompressed: ctt converts to f32 RGBA, values as stored. Read as sRGB it would decode and
    // encode again, and as straight alpha premultiply and divide back, which loses color where
    // alpha is 0 or negative; the file's color reads as for any other file.
    let (stored, _) = format.normalize();
    let mip0 = ctt::Image {
        surfaces: image.surfaces.iter().map(|l| vec![l[0].clone()]).collect(),
        kind: TextureKind::Texture2D,
        desc: ctt::FormatDesc {
            format: stored,
            color_space: ColorSpace::Linear,
            alpha: ctt::AlphaMode::Opaque,
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
    let codes = match stored.channel_kind() {
        _ if is_snorm(stored) => None,
        Some(ctt::ChannelKind::U8) => Some(Codes::EIGHT),
        Some(ctt::ChannelKind::U16) => Some(Codes::SIXTEEN),
        _ => None,
    };
    let layers = img
        .surfaces
        .iter()
        .map(|l| {
            let s = &l[0];
            let mut out = Img::new(s.width, s.height);
            out.codes = codes;
            for (y, row) in out.px.chunks_exact_mut(s.width as usize).enumerate() {
                let src = &s.data[y * s.stride as usize..][..s.width as usize * 16];
                for (p, b) in row
                    .iter_mut()
                    .zip(src.as_chunks::<4>().0.as_chunks::<4>().0)
                {
                    *p = b.map(f32::from_le_bytes);
                }
            }
            out
        })
        .collect();
    Ok(Source {
        layers,
        cube,
        color: container_color(bytes, format, srgb, is_float(format) || is_snorm(format)),
        blocks: None,
        rg: format.channel_count() == Some(2),
        signed: is_snorm(format),
        float: is_float(format),
        negatives: is_snorm(format) || (is_float(format) && !is_ufloat(format)),
        gray: if format.channel_count() == Some(1) {
            Gray::Red
        } else {
            Gray::No
        },
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

/// Whether uncompressed `format` stores floats.
fn is_float(format: ctt::Format) -> bool {
    use ctt::{ChannelKind, Format as F};
    matches!(
        format.channel_kind(),
        Some(ChannelKind::F16 | ChannelKind::F32)
    ) || matches!(
        format,
        F::R64_SFLOAT
            | F::R64G64_SFLOAT
            | F::R64G64B64_SFLOAT
            | F::R64G64B64A64_SFLOAT
            | F::B10G11R11_UFLOAT_PACK32
            | F::E5B9G9R9_UFLOAT_PACK32
    )
}

/// Whether uncompressed `format` stores floats no lower than 0.
fn is_ufloat(format: ctt::Format) -> bool {
    use ctt::Format as F;
    matches!(
        format,
        F::B10G11R11_UFLOAT_PACK32 | F::E5B9G9R9_UFLOAT_PACK32
    )
}

fn file(bytes: &[u8]) -> Result<Source, String> {
    use image::{ColorType as C, DynamicImage, ImageFormat};
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let format = reader.format();
    let radiance = format == Some(ImageFormat::Hdr);
    let mut decoder = reader.into_decoder().map_err(|e| e.to_string())?;
    let icc = decoder.icc_profile().ok().flatten();
    let img = DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    let hdr = matches!(
        img,
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_)
    );
    let gray = match img.color() {
        C::L8 | C::L16 => Gray::Alone,
        C::La8 | C::La16 => Gray::WithAlpha,
        _ => Gray::No,
    };
    let mut color = FileColor {
        numeric: hdr,
        rgb8: matches!(img.color(), C::Rgb8 | C::Rgba8),
        // OpenEXR: "By convention, all color channels are premultiplied by alpha".
        premultiplied: format == Some(ImageFormat::OpenExr) && img.color().has_alpha(),
        ..Default::default()
    };
    let mut scale = [1.0; 3];
    match format {
        Some(ImageFormat::Png) => png_color(&mut color, bytes, icc.as_deref(), gray != Gray::No)?,
        Some(ImageFormat::OpenExr) => exr_color(&mut color, bytes),
        Some(ImageFormat::Hdr) => scale = radiance_header(&mut color, bytes)?,
        _ => {
            if let Some(icc) = &icc {
                color::read_icc(&mut color, icc, gray != Gray::No);
            }
        }
    }
    let (w, h) = (img.width(), img.height());
    let (codes, mut px) = widened(&img);
    if scale != [1.0; 3] {
        for p in &mut px {
            *p = [p[0] * scale[0], p[1] * scale[1], p[2] * scale[2], p[3]];
        }
    }
    Ok(Source {
        layers: vec![Img { w, h, px, codes }],
        cube: false,
        color,
        blocks: None,
        rg: false,
        signed: false,
        float: hdr,
        negatives: hdr && !radiance,
        gray,
    })
}

/// The texels of `img` as RGBA floats, gray spread over RGB and absent alpha 1. Integers widen as
/// [`Codes`], which they state.
fn widened(img: &image::DynamicImage) -> (Option<Codes>, Vec<[f32; 4]>) {
    fn texels<T: Copy + Into<u32>>(
        codes: Codes,
        samples: &[T],
        channels: usize,
    ) -> (Option<Codes>, Vec<[f32; 4]>) {
        let texel = |c: &[T]| {
            let v = |i: usize| codes.value(c[i].into());
            match c.len() {
                1 => [v(0), v(0), v(0), 1.0],
                2 => [v(0), v(0), v(0), v(1)],
                3 => [v(0), v(1), v(2), 1.0],
                _ => [v(0), v(1), v(2), v(3)],
            }
        };
        (
            Some(codes),
            samples.chunks_exact(channels).map(texel).collect(),
        )
    }
    let channels = usize::from(img.color().channel_count());
    match (img.as_flat_samples_u8(), img.as_flat_samples_u16()) {
        (Some(s), _) => texels(Codes::EIGHT, s.samples, channels),
        (_, Some(s)) => texels(Codes::SIXTEEN, s.samples, channels),
        _ => (None, img.to_rgba32f().as_chunks::<4>().0.to_vec()),
    }
}

/// States what a PNG's chunks say on `color`, the chunk of lowest priority number in PNG §4.3
/// taking precedence: `cICP`, then `iCCP` (`icc`), `sRGB`, and `cHRM` and `gAMA` last.
fn png_color(
    color: &mut FileColor,
    bytes: &[u8],
    icc: Option<&[u8]>,
    gray: bool,
) -> Result<(), String> {
    let reader = png::Decoder::new(Cursor::new(bytes))
        .read_info()
        .map_err(|e| e.to_string())?;
    let info = reader.info();
    if let Some(c) = info.coding_independent_code_points {
        color.transfer = color::cicp_transfer(c.transfer_function);
        color.gamut = color::cicp_gamut(c.color_primaries);
        if !c.is_video_full_range_image {
            // H.273: `D = 2^(n - 8) (219 E + 16)`, on the n-bit code that 0..1 stores.
            color.narrow = Some(if info.bit_depth == png::BitDepth::Sixteen {
                [-4096.0 / 56064.0, 65535.0 / 56064.0]
            } else {
                [-16.0 / 219.0, 255.0 / 219.0]
            });
        }
        if color.transfer.is_some() || color.gamut.is_some() {
            return Ok(());
        }
    }
    if icc.is_some_and(|icc| color::read_icc(color, icc, gray)) {
        return Ok(());
    }
    if info.srgb.is_some() {
        color.transfer = Some(Transfer::Srgb);
        color.gamut = Some(Gamut::REC709);
        return Ok(());
    }
    if let Some(gamma) = info
        .gama_chunk
        .map(png::ScaledFloat::into_value)
        .filter(|g| *g > 0.0)
    {
        color.transfer = Some(Transfer::Gamma(1.0 / gamma));
    }
    if let Some(c) = info.chrm_chunk {
        let xy =
            |p: (png::ScaledFloat, png::ScaledFloat)| [p.0, p.1].map(|v| f64::from(v.into_value()));
        color.gamut = color::stated_gamut([xy(c.red), xy(c.green), xy(c.blue), xy(c.white)]);
    }
    Ok(())
}

/// States what an OpenEXR header says on `color`: `colorInteropID`, which supersedes
/// `chromaticities`, or `chromaticities`.
fn exr_color(color: &mut FileColor, bytes: &[u8]) {
    use exr::meta::attribute::{AttributeValue, Text};
    let Ok(meta) = exr::meta::MetaData::read_from_buffered(bytes, false) else {
        return;
    };
    // The part `image` decodes.
    let rgb = |h: &&exr::meta::header::Header| {
        let holds = |c: &str| h.channels.find_index_of_channel(&Text::from(c)).is_some();
        !h.deep && ["R", "G", "B"].into_iter().all(holds)
    };
    let Some(header) = meta.headers.iter().find(rgb) else {
        return;
    };
    let name = Text::from("colorInteropID");
    let interop = [
        &header.own_attributes.other,
        &header.shared_attributes.other,
    ]
    .into_iter()
    .find_map(|attributes| match attributes.get(&name) {
        Some(AttributeValue::Text(id)) => ImageColor::named(&id.to_string()),
        _ => None,
    });
    match interop {
        Some(ImageColor::Named(s)) => {
            color.transfer = Some(s.transfer);
            color.gamut = Some(s.gamut);
        }
        Some(_) => color.transfer = Some(Transfer::Linear),
        None => {
            if let Some(c) = &header.shared_attributes.chromaticities {
                let xy = |v: exr::math::Vec2<f32>| [f64::from(v.0), f64::from(v.1)];
                color.gamut =
                    color::stated_gamut([xy(c.red), xy(c.green), xy(c.blue), xy(c.white)]);
            }
        }
    }
}

/// States a Radiance header's `PRIMARIES` on `color`, and gives what its texels multiply by to
/// undo every `EXPOSURE` and `COLORCORR`, which accumulate. No `PRIMARIES` stays Rec.709 rather
/// than Radiance's default system, whose white is near E: untagged images are authored against
/// neutrals, and would tint.
fn radiance_header(color: &mut FileColor, bytes: &[u8]) -> Result<[f32; 3], String> {
    let decoder = HdrDecoder::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let mut scale = [1.0f32; 3];
    let numbers = |v: &str| -> Vec<f32> {
        v.split_whitespace()
            .map_while(|n| n.parse().ok())
            .filter(|n: &f32| n.is_finite())
            .collect()
    };
    for (key, value) in &decoder.metadata().custom_attributes {
        match (key.as_str(), &numbers(value)[..]) {
            ("EXPOSURE", &[e]) if e > 0.0 => scale = scale.map(|s| s / e),
            ("COLORCORR", &[r, g, b]) if r > 0.0 && g > 0.0 && b > 0.0 => {
                scale = [scale[0] / r, scale[1] / g, scale[2] / b];
            }
            ("PRIMARIES", &[rx, ry, gx, gy, bx, by, wx, wy]) => {
                let xy = |x: f32, y: f32| [f64::from(x), f64::from(y)];
                color.gamut = color::stated_gamut([xy(rx, ry), xy(gx, gy), xy(bx, by), xy(wx, wy)]);
            }
            _ => {}
        }
    }
    Ok(scale)
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

/// Decodes one layer's BCn mip to float texels (unorm values in 0..1, snorm values in -1..1,
/// BC6H as floats).
pub(crate) fn decode_blocks(vk_format: u32, w: u32, h: u32, data: &[u8]) -> Result<Img, String> {
    let mut out = Img::new(w, h);
    decode_rows(vk_format, w, h, data, &mut out.px, |t| t)?;
    out.codes = block_codes(vk_format);
    Ok(out)
}

/// The codes blocks of `vk_format` decode to: BC7 interpolates to 8 bits, and the others in
/// real numbers.
pub(super) fn block_codes(vk_format: u32) -> Option<Codes> {
    matches!(vk_format, vk::BC7_UNORM | vk::BC7_SRGB).then_some(Codes::EIGHT)
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

/// The 16 texels of `channels` floats each in `f`.
fn block_texels(f: &[f32], channels: usize) -> [[f32; 4]; 16] {
    std::array::from_fn(|i| {
        let mut p = [0.0, 0.0, 0.0, 1.0];
        p[..channels].copy_from_slice(&f[i * channels..][..channels]);
        p
    })
}

/// The 16 texels of a BC1 color block, interpolated in real numbers as S3TC states; bcdec rounds
/// to 8 bits. With `four`, the block has four colors whatever the order of its endpoints, as BC2
/// and BC3 do. Without, one with `c0 <= c1` has three and index 3 is transparent black.
fn color_block(b: &[u8], four: bool) -> [[f32; 4]; 16] {
    let [c0, c1] = [0, 2].map(|i| u16::from_le_bytes([b[i], b[i + 1]]));
    let rgb = |c: u16| {
        let [r, g, b] = [c >> 11, (c >> 5) & 63, c & 31].map(f32::from);
        [r / 31.0, g / 63.0, b / 31.0, 1.0]
    };
    let (e0, e1) = (rgb(c0), rgb(c1));
    let mix = |k: f32| std::array::from_fn(|c| k * e0[c] + (1.0 - k) * e1[c]);
    let colors: [[f32; 4]; 4] = if four || c0 > c1 {
        [e0, e1, mix(2.0 / 3.0), mix(1.0 / 3.0)]
    } else {
        [e0, e1, mix(0.5), [0.0; 4]]
    };
    let indices = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    std::array::from_fn(|i| colors[(indices >> (2 * i)) as usize & 3])
}

fn decode_block(vk_format: u32, block: &[u8]) -> Result<[[f32; 4]; 16], String> {
    use vk::*;
    let mut f = [0.0f32; 48];
    Ok(match vk_format {
        BC1_RGB_UNORM | BC1_RGB_SRGB => {
            color_block(block, false).map(|[r, g, b, _]| [r, g, b, 1.0])
        }
        BC1_RGBA_UNORM | BC1_RGBA_SRGB => color_block(block, false),
        BC2_UNORM | BC2_SRGB => {
            let mut t = color_block(&block[8..], true);
            for (i, p) in t.iter_mut().enumerate() {
                p[3] = f32::from((block[i / 2] >> (4 * (i % 2))) & 15) / 15.0;
            }
            t
        }
        BC3_UNORM | BC3_SRGB => {
            bcdec_rs::bc4_float(&block[..8], &mut f[..16], 4, false);
            let mut t = color_block(&block[8..], true);
            for (p, a) in t.iter_mut().zip(f) {
                p[3] = a;
            }
            t
        }
        BC7_UNORM | BC7_SRGB => {
            let mut b8 = [0u8; 64];
            bcdec_rs::bc7(block, &mut b8, 16);
            std::array::from_fn(|i| std::array::from_fn(|c| f32::from(b8[i * 4 + c]) / 255.0))
        }
        BC4_UNORM | BC4_SNORM => {
            bcdec_rs::bc4_float(block, &mut f[..16], 4, vk_format == BC4_SNORM);
            block_texels(&f, 1)
        }
        BC5_UNORM | BC5_SNORM => {
            bcdec_rs::bc5_float(block, &mut f[..32], 8, vk_format == BC5_SNORM);
            block_texels(&f, 2)
        }
        BC6H_UFLOAT | BC6H_SFLOAT => {
            bcdec_rs::bc6h_float(block, &mut f, 12, vk_format == BC6H_SFLOAT);
            block_texels(&f, 3)
        }
        _ => return Err(format!("format {vk_format} does not decode on the CPU")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A layer in a window of its frame, in a named part of a file with stated primaries, reads
    /// as that part of the whole file does.
    #[test]
    fn a_layer_keeps_the_frame_and_color_of_its_file() {
        use exr::image::{AnyChannel, AnyChannels, Encoding, FlatSamples, Layer as ExrLayer};
        use exr::meta::attribute::{Chromaticities, IntegerBounds};
        use exr::meta::header::{ImageAttributes, LayerAttributes};
        use exr::prelude::{Image, Vec2, WritableImage};
        let plane = |name: &str, v: f32| AnyChannel::new(name, FlatSamples::F32(vec![v; 16]));
        let mut image = ImageAttributes::new(IntegerBounds::from_dimensions((8, 8)));
        image.chromaticities = Some(Chromaticities {
            red: Vec2(0.708, 0.292),
            green: Vec2(0.170, 0.797),
            blue: Vec2(0.131, 0.046),
            white: Vec2(0.3127, 0.3290),
        });
        let mut part = LayerAttributes::named("beauty");
        part.layer_position = Vec2(2, 2);
        let channels = AnyChannels::sort(
            vec![
                plane("P.X", 0.25),
                plane("P.Y", 0.5),
                plane("P.Z", 0.75),
                plane("mask.A", 0.5),
                plane("mask.Y", 0.125),
            ]
            .into(),
        );
        let layer = ExrLayer::new((4, 4), part, Encoding::UNCOMPRESSED, channels);
        let mut file = Cursor::new(Vec::new());
        Image::new(image, layer)
            .write()
            .to_buffered(&mut file)
            .unwrap();
        let file = file.into_inner();

        let read = |name: &str| {
            let layer = Layer::Named(name.into());
            decode(&layer_of(&file, &layer).unwrap().unwrap()).unwrap()
        };
        let p = read("P");
        let img = &p.layers[0];
        assert_eq!((img.w, img.h), (8, 8));
        assert_eq!(img.px[2 * 8 + 2], [0.25, 0.5, 0.75, 1.0]);
        assert_eq!(img.px[0], [0.0, 0.0, 0.0, 1.0]);
        assert!(p.color.gamut.is_some());
        assert_eq!(
            read("beauty.mask").layers[0].px[2 * 8 + 2],
            [0.125, 0.0, 0.0, 0.5]
        );
    }

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
        assert!(src.color.rgb8 && !src.rg && src.blocks.is_none() && !is_animated(&bytes));
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

    /// A PNG of `w` x `h` `texels` with `chunks` ahead of its image data.
    fn png(
        color: png::ColorType,
        depth: png::BitDepth,
        [w, h]: [u32; 2],
        texels: &[u8],
        chunks: &[(&[u8; 4], &[u8])],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        let mut info = png::Info::with_size(w, h);
        (info.color_type, info.bit_depth) = (color, depth);
        let mut writer = png::Encoder::with_info(&mut out, info)
            .unwrap()
            .write_header()
            .unwrap();
        for (name, data) in chunks {
            writer
                .write_chunk(png::chunk::ChunkType(**name), data)
                .unwrap();
        }
        writer.write_image_data(texels).unwrap();
        writer.finish().unwrap();
        out
    }

    /// An uncompressed 256 x 1 KTX2 of `texels` in R8G8B8A8_SRGB, premultiplied if it says so.
    fn srgb_ktx2(texels: &[u8], premultiplied: bool) -> Vec<u8> {
        let image = ctt::ImageRef {
            surfaces: vec![vec![ctt::SurfaceRef {
                data: texels,
                width: 256,
                height: 1,
                depth: 1,
                stride: 256 * 4,
                slice_stride: 0,
            }]],
            kind: TextureKind::Texture2D,
            desc: ctt::FormatDesc {
                format: ctt::Format::R8G8B8A8_SRGB,
                color_space: ColorSpace::Srgb,
                alpha: ctt::AlphaMode::Straight,
            },
        };
        let settings = ctt::ConvertSettings {
            container: ctt::Container::ktx2(),
            ..Default::default()
        };
        let ctt::PipelineOutput::Encoded(mut bytes) = ctt::convert(image, settings).unwrap() else {
            panic!("container output");
        };
        // KTX 2.0 §3.10: bit 0 of the DFD's flags states premultiplied alpha.
        let dfd = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
        bytes[dfd + 15] = u8::from(premultiplied);
        bytes
    }

    /// 8-bit sRGB codes read as the f32 nearest their exact value, through each loader and gray,
    /// and premultiplied ones before their alpha divides them. The f32 nearest a code over 255
    /// read through the curve misses it for some codes, which this tells apart.
    #[test]
    fn eight_bit_srgb_codes_read_exactly() {
        let exact = |k: usize| orr_color::srgb8_decode(k as u8);
        let near = |k: usize| orr_color::srgb_decode(f64::from(k as f32 / 255.0)) as f32;
        assert!((0..256).any(|k| near(k).to_bits() != exact(k).to_bits()));

        let rgb: Vec<u8> = (0..=255).flat_map(|k| [k, k, k]).collect();
        let gray: Vec<u8> = (0..=255).collect();
        let rgba: Vec<u8> = (0..=255).flat_map(|k| [k, k, k, 255]).collect();
        let half: Vec<u8> = (0..=255).flat_map(|k| [k, k, k, 128]).collect();
        let (eight, rgb8) = (png::BitDepth::Eight, png::ColorType::Rgb);
        let files = [
            (png(rgb8, eight, [256, 1], &rgb, &[]), 1.0),
            (
                png(
                    png::ColorType::Grayscale,
                    eight,
                    [256, 1],
                    &gray,
                    &[(b"sRGB", &[0])],
                ),
                1.0,
            ),
            (srgb_ktx2(&rgba, false), 1.0),
            (srgb_ktx2(&half, true), 128.0 / 255.0),
        ];
        for (i, (file, alpha)) in files.iter().enumerate() {
            let src = decode(file).unwrap();
            let (read, ..) = ImageColor::File.color(&src.color, None, |_| false);
            let img = &src.layers[0];
            let (texels, values) = (read.on(img), read.values(img.codes));
            for (k, t) in img.px.iter().enumerate() {
                let want = (exact(k) / alpha).to_bits();
                for c in 0..3 {
                    assert_eq!(texels.value(*t, c).to_bits(), want, "file {i} code {k}");
                    assert_eq!(values.value(*t, c).to_bits(), want, "file {i} code {k}");
                }
            }
        }
    }

    /// Every curve but 8-bit sRGB reads a code as it reads the value that stores it, bit for bit,
    /// through a table or without: a gamma chunk, narrow-range PQ, HLG, and 16-bit gray sRGB.
    #[test]
    fn other_curves_read_codes_as_their_values() {
        let codes: Vec<u8> = (0..=255).flat_map(|k| [k, 255 - k, k / 2]).collect();
        let wide: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_be_bytes).collect();
        let gamma = (0.45455f64 * 100_000.0).round() as u32;
        let (rgb8, eight) = (png::ColorType::Rgb, png::BitDepth::Eight);
        let files = [
            png(
                rgb8,
                eight,
                [256, 1],
                &codes,
                &[(b"gAMA", &gamma.to_be_bytes())],
            ),
            png(rgb8, eight, [256, 1], &codes, &[(b"cICP", &[9, 16, 0, 0])]),
            png(rgb8, eight, [256, 1], &codes, &[(b"cICP", &[9, 18, 0, 1])]),
            png(
                png::ColorType::Grayscale,
                png::BitDepth::Sixteen,
                [256, 256],
                &wide,
                &[(b"sRGB", &[0])],
            ),
        ];
        for (i, file) in files.iter().enumerate() {
            let src = decode(file).unwrap();
            let (read, ..) = ImageColor::File.color(&src.color, None, |_| false);
            assert_ne!(read.decode.transfer, Transfer::Linear, "file {i}");
            let img = &src.layers[0];
            let (texels, values) = (read.on(img), read.values(None));
            let one = Img {
                w: 1,
                h: 1,
                px: img.px[..1].to_vec(),
                codes: img.codes,
            };
            let untabled = read.on(&one);
            for t in &img.px {
                for c in 0..3 {
                    let want = values.value(*t, c).to_bits();
                    assert_eq!(texels.value(*t, c).to_bits(), want, "file {i} {t:?}");
                    assert_eq!(untabled.value(*t, c).to_bits(), want, "file {i} {t:?}");
                }
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
            let decoded = out.as_chunks::<4>().0.iter().all(|t| t[3] == 255);
            if bc7_opaque(&block) {
                opaque += 1;
                assert!(decoded, "{block:02x?}");
            }
        }
        assert!(opaque > 80_000);
    }
}
