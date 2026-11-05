//! Texture cook: import, mip chains, encoding, and the tiled archive layout.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) mod color;
mod encode;
mod env;
mod load;
pub(crate) mod material;
mod mips;
pub(crate) mod neural;
pub(crate) mod tiles;

pub use orr_color::Primaries;
use orr_math::Vec3;
use orr_pack::material::wrap;
use orr_pack::texture::{
    ExtractedLight, FormatInfo, TILE_BYTES, TextureHeader, TextureInfo, full_mip_count, mip_size,
    normal, tail_first_mip, tile,
};
use orr_pack::{ArchiveWriter, AssetId, BcLayout, BlobIn, Class, Filter, Hash32, Span};
use serde::Deserialize;

use color::{BELOW_ZERO, Decode, ImageColor, Read};
use ctt::Quality;
pub(crate) use encode::{Encoding, encode as encode_chains};
use load::{Blocks, Source};
pub(crate) use load::{decode_blocks, decode_channel, file_header};
use mips::{MipSpec, Toksvig};

use crate::{Error, Result};

/// How one texture is cooked.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureSettings {
    pub usage: Usage,
    /// Alpha-tested coverage cutoff. With it, every mip the cook makes has its alpha scaled
    /// so the fraction of the texture at or above the cutoff matches mip 0; mips kept from
    /// the source stay as they are. Applies to the alpha of [`Usage::ColorAlpha`] and
    /// [`Usage::Linear4`] and to the value of [`Usage::Linear1`].
    pub alpha_cutoff: Option<f32>,
    /// Largest mip 0 edge; larger mips are dropped. For [`Usage::Environment`] the map edge,
    /// rounded down to a power of two (1024 when unset).
    pub max_size: Option<u32>,
    /// Cells per axis of an atlas that divides the texture evenly. Mips whose cells are a
    /// whole number of texels and at least 4x4 filter each cell alone and match alpha
    /// coverage per cell; smaller mips filter the whole texture.
    pub atlas_grid: Option<[u16; 2]>,
    /// Source primaries of color texels; overrides what the file states. Files that state none
    /// are Rec. 709, and primaries with no id convert to Rec. 2020, as does a file that reads below
    /// 0 unless these are set.
    pub primaries: Option<Primaries>,
    pub filter: MipFilter,
    /// What the mip filter reads past each edge, per axis (u, v).
    pub wrap: [Wrap; 2],
    /// Store BCn blocks split into endpoint and index streams, which zstd compresses
    /// better. `None` takes the format's default.
    pub bc_split: Option<bool>,
    /// Whether the texture has a mip chain under mip 0; `None` takes the usage's default
    /// ([`Usage::has_mips`]). Without one the texture is the first mip, of the source or
    /// made, that fits `max_size`. [`Usage::Environment`] maps always have one.
    pub mips: Option<bool>,
}

impl Default for TextureSettings {
    fn default() -> Self {
        Self {
            usage: Usage::Color,
            alpha_cutoff: None,
            max_size: None,
            atlas_grid: None,
            primaries: None,
            filter: MipFilter::Kaiser,
            wrap: [Wrap::Repeat; 2],
            bc_split: None,
            mips: None,
        }
    }
}

/// What a texture holds; sets its format. Where a usage has a signed or float form, the format
/// follows the source's as sRGB follows the file: nothing else tells the consumer the domain of
/// what it samples. A float source counts as signed for `Linear1` and `Linear2` when the channels
/// they keep hold negative values.
#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Usage {
    /// sRGB color (RGB) to BC7 sRGB.
    Color,
    /// sRGB color with linear alpha to BC7 sRGB.
    ColorAlpha,
    /// Tangent-space normal (RGB, or x and y alone from a two-channel source) to BC5 holding x
    /// and y; z is reconstructed. Unsigned sources store components at `0.5 v + 0.5` and signed
    /// ones as they are. No format says which a float source that can hold negatives does, so
    /// its vectors decide.
    Normal,
    /// One linear channel (R) to BC4, or BC4 snorm from a signed source without `alpha_cutoff`.
    Linear1,
    /// Two linear channels (RG) to BC5, or BC5 snorm from a signed source.
    Linear2,
    /// Four linear channels to BC7.
    Linear4,
    /// Linear HDR color to BC6H.
    Hdr,
    /// Height (R) to R16_UNORM, R16_SNORM from a signed source, or R32_SFLOAT from a float
    /// source.
    Displacement,
    /// An equirectangular (2:1) or cube HDR map resampled to an equal-area octahedral map,
    /// BC6H with mips. Each connected region over `light_threshold` (luminance), brightest
    /// first up to `max_lights`, moves into one directional light of equal energy.
    Environment {
        #[serde(default)]
        light_threshold: Option<f32>,
        max_lights: u8,
    },
    /// Uncompressed, the source channels the format holds, clamped to its range.
    Data(DataFormat),
}

impl Usage {
    /// Whether textures of this usage have mips unless their settings say otherwise: all
    /// but [`Usage::Data`], where filtering would blend texels (lookup tables, ids) into
    /// values the data never held.
    pub fn has_mips(self) -> bool {
        !matches!(self, Self::Data(_))
    }
}

/// Uncompressed formats for [`Usage::Data`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    R8,
    Rg8,
    Rgba8,
    Rgba8Srgb,
    R16,
    R16Snorm,
    Rg16,
    Rgba16,
    R16f,
    Rg16f,
    Rgba16f,
    R32f,
    Rg32f,
    Rgba32f,
}

/// Mip downsampling filter, applied to linear values.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MipFilter {
    /// Area average.
    Box,
    /// Kaiser-windowed sinc, 3 lobes, alpha 4. BC6H mips use [`Self::Box`] instead.
    #[default]
    Kaiser,
}

/// What a texture reads past 0..1.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
    /// Transparent black, which filtering blends with the edge texels.
    Border,
    /// A value of the texture's binding, and [`Self::Clamp`] within 0..1. Nothing blends across
    /// the edge, so mips filter as under [`Self::Clamp`].
    Constant,
}

impl Wrap {
    /// Whether reads past 0..1 leave the texture for a value of their own.
    pub(crate) fn bounded(self) -> bool {
        matches!(self, Self::Border | Self::Constant)
    }

    /// The `TextureBinding::wrap` value.
    pub(crate) fn binding(self) -> u8 {
        match self {
            Self::Repeat => wrap::REPEAT,
            Self::Clamp => wrap::CLAMP,
            Self::Mirror => wrap::MIRROR,
            Self::Border => wrap::BORDER,
            Self::Constant => wrap::CONSTANT,
        }
    }

    /// The mode of a `TextureBinding::wrap` value.
    pub(crate) fn of_binding(v: u8) -> Self {
        match v {
            wrap::CLAMP => Self::Clamp,
            wrap::MIRROR => Self::Mirror,
            wrap::BORDER => Self::Border,
            wrap::CONSTANT => Self::Constant,
            _ => Self::Repeat,
        }
    }
}

/// Linear float RGBA texels, row-major.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Img {
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) px: Vec<[f32; 4]>,
}

impl Img {
    fn new(w: u32, h: u32) -> Self {
        Self {
            w,
            h,
            px: vec![[0.0; 4]; w as usize * h as usize],
        }
    }

    /// Bytes of the texels.
    pub(crate) fn bytes(&self) -> u64 {
        (self.px.len() * size_of::<[f32; 4]>()) as u64
    }

    /// The lowest and highest finite value of channel `c`, or `(f32::MAX, f32::MIN)` with none.
    fn finite_range(&self, c: usize) -> (f32, f32) {
        let finite = self.px.iter().map(|p| p[c]).filter(|v| v.is_finite());
        finite.fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)))
    }

    /// Bilinear resize; used to bring a packed texture's inputs to one size. A non-finite texel
    /// takes an end of its channel's range, or every tap beside it would read it.
    fn resized(self: Arc<Self>, w: u32, h: u32) -> Arc<Self> {
        if (w, h) == (self.w, self.h) {
            return self;
        }
        let held = self.px.iter().flatten().any(|v| !v.is_finite()).then(|| {
            let ranges: [_; 4] = std::array::from_fn(|c| self.finite_range(c));
            let mut held = (*self).clone();
            for p in &mut held.px {
                for (v, (lo, hi)) in p.iter_mut().zip(ranges) {
                    if !v.is_finite() {
                        *v = if lo <= hi { v.max(lo).min(hi) } else { 0.0 };
                    }
                }
            }
            held
        });
        let src = held.as_ref().unwrap_or(&self);
        let mut out = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f32 + 0.5) * self.w as f32 / w as f32 - 0.5;
                let sy = (y as f32 + 0.5) * self.h as f32 / h as f32 - 0.5;
                out.px[(y * w + x) as usize] = src.bilinear(sx, sy);
            }
        }
        Arc::new(out)
    }

    /// Bilinear sample at texel coordinates (texel centers at integers), clamped at edges.
    fn bilinear(&self, x: f32, y: f32) -> [f32; 4] {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let cx = |v: f32| (v as i64).clamp(0, i64::from(self.w) - 1) as usize;
        let cy = |v: f32| (v as i64).clamp(0, i64::from(self.h) - 1) as usize;
        let at = |xx: usize, yy: usize| self.px[yy * self.w as usize + xx];
        let (a, b) = (at(cx(x0), cy(y0)), at(cx(x0 + 1.0), cy(y0)));
        let (c, d) = (at(cx(x0), cy(y0 + 1.0)), at(cx(x0 + 1.0), cy(y0 + 1.0)));
        std::array::from_fn(|i| {
            let top = a[i] + (b[i] - a[i]) * fx;
            let bot = c[i] + (d[i] - c[i]) * fx;
            top + (bot - top) * fy
        })
    }
}

pub(crate) fn srgb_to_linear(v: f32) -> f32 {
    orr_color::srgb_decode(f64::from(v)) as f32
}

/// The sRGB encoding of `v`, clamped to 0..1 first.
pub(crate) fn linear_to_srgb(v: f32) -> f32 {
    orr_color::srgb_encode(f64::from(v.clamp(0.0, 1.0))) as f32
}

/// What to cook from prepared linear layers.
struct Plan {
    encoding: Encoding,
    cube: bool,
    primaries: Primaries,
    mips: MipSpec,
    quality: Quality,
    max_size: Option<u32>,
    /// Mips down to 1x1; without them the texture is the first mip that fits `max_size`.
    full_chain: bool,
    toksvig: Option<Toksvig>,
    lights: Vec<ExtractedLight>,
}

/// A cooked texture: blob 0 and the blobs after it, stored with `filter`.
pub(crate) struct Built {
    info: TextureInfo,
    blobs: Vec<Vec<u8>>,
    filter: Filter,
    /// Its source holds colors a texture can't.
    pub(crate) clips: bool,
}

/// The report's warning for a source that reads the animated image `image`.
pub(crate) fn first_frame_warning(image: &str) -> String {
    format!("image {image} is animated; only its first frame is cooked")
}

/// The report's warning for `image`, which can't be read for `why`, so its inputs read their
/// fallback.
pub(crate) fn unread_warning(image: &str, why: &str) -> String {
    format!("image {image} reads as its input's fallback: {why}")
}

/// The report's warning for file `image`, which names a tile no set holds.
pub(crate) fn past_tile_warning(image: &str) -> String {
    format!("image {image} is not read: tiles stop at column and row 65535")
}

/// The report's warning for `images`, with colors a texture can't hold.
pub(crate) fn gamut_warning(images: &str) -> String {
    format!("colors of {images} below 0 or outside Rec.2020 clip")
}

/// The report's warning for a USD normal map `image` with no scale or bias, which an unsigned
/// file can't mean literally.
pub(crate) fn unscaled_normal_warning(image: &str) -> String {
    format!("normal map {image} states no scale or bias; decoding it as 2v - 1")
}

impl Built {
    /// Stores the blocks split, or not, as `bc_split` asks; `None` takes the format's
    /// default.
    fn with_split(mut self, bc_split: Option<bool>) -> Self {
        let layout = BcLayout::of(self.info.header.vk_format);
        self.filter = match layout {
            Some(l) if bc_split.unwrap_or_else(|| split_by_default(l)) => Filter::BcSplit(l),
            _ => Filter::None,
        };
        self
    }

    fn with_clips(self, clips: &AtomicBool) -> Self {
        Self {
            clips: clips.load(Ordering::Relaxed),
            ..self
        }
    }
}

/// Whether a format's blocks are split when the settings leave it open.
fn split_by_default(_: BcLayout) -> bool {
    true
}

/// Mips a `w` x `h` texture keeps: down to 1x1 with `full_chain`, else down to the first
/// that fits `max_size`.
fn kept_mip_count(w: u32, h: u32, max_size: Option<u32>, full_chain: bool) -> u32 {
    if full_chain {
        full_mip_count(w, h)
    } else {
        skipped_mips(w, h, max_size) + 1
    }
}

/// Mips dropped from the top of a `w` x `h` chain so mip 0 fits `max_size`.
fn skipped_mips(w: u32, h: u32, max_size: Option<u32>) -> u32 {
    max_size.map_or(0, |max| {
        (0..full_mip_count(w, h))
            .take_while(|&m| mip_size(w, h, m).into_iter().any(|e| e > max.max(1)))
            .count() as u32
    })
}

/// The mip chains of `layers` (mip 0 each, values linear) as they are encoded: filtered,
/// without the mips over `max_size`, with Toksvig roughness from kept mip `first` on.
fn chains(layers: Vec<Img>, plan: &Plan, first: usize) -> Result<Vec<Vec<Img>>, String> {
    let (w, h) = (layers[0].w, layers[0].h);
    if layers.iter().any(|l| (l.w, l.h) != (w, h)) {
        return Err("layers differ in size".into());
    }
    let skip = skipped_mips(w, h, plan.max_size) as usize;
    let count = kept_mip_count(w, h, plan.max_size, plan.full_chain);
    // Normals and heights hold their own ranges.
    let settles = count > 1 && !plan.mips.normal && !plan.mips.height;
    let (lo, hi) = plan.encoding.range();
    let mut chains: Vec<Vec<Img>> = layers
        .into_iter()
        .map(|mut l| {
            if settles {
                mips::settle(&mut l, lo, hi);
            }
            mips::chain(l, &plan.mips, count)
        })
        .collect();
    if let Some(t) = &plan.toksvig {
        for c in &mut chains {
            t.apply(c, &plan.mips, skip + first)?;
        }
    }
    for c in &mut chains {
        c.drain(..skip);
    }
    Ok(chains)
}

/// Mip chains, encoding and layout for `layers` (mip 0 each, values linear).
fn build(layers: Vec<Img>, plan: &Plan) -> Result<Built, String> {
    finish(chains(layers, plan, 0)?, plan)
}

/// How an image stores normals.
#[derive(Copy, Clone, Debug)]
struct NormalSource {
    /// It holds x and y alone.
    rg: bool,
    /// Components are stored as they are, not at `0.5 v + 0.5`.
    signed: bool,
    /// The texel channel holding y.
    y: usize,
    /// The texel channel holding z, where it holds one.
    z: usize,
    /// How its channels read as numbers.
    decode: Decode,
}

impl NormalSource {
    /// The normal of texel `t`. Without a stored z it is rebuilt as the shader rebuilds it.
    fn read(self, t: [f32; 4]) -> [f32; 3] {
        let component = |v: f32| if self.signed { v } else { 2.0 * v - 1.0 };
        let [x, y] = [component(t[0]), component(t[self.y])];
        let z = if self.rg {
            normal::z(x, y)
        } else {
            component(t[self.z])
        };
        [x, y, z]
    }
}

/// Whether float `layers` hold normals as they are rather than at `0.5 v + 0.5`, the reading
/// tools give them. With z, the reading closer to unit length on average, each texel counting
/// for at most 1 so that no stray value outweighs the image. Without, both can fit inside the
/// unit circle, as a flat map at 0.5 does, so it takes most texels falling outside it at
/// `2v - 1`: a few ringing texels must not flip a whole image.
fn stores_normals(layers: &[Img], rg: bool) -> bool {
    let texels = || {
        let finite = |p: &&[f32; 4]| p[..3].iter().all(|v| v.is_finite());
        layers.iter().flat_map(|l| &l.px).filter(finite)
    };
    let [kept, moved] = [true, false].map(|signed| NormalSource {
        rg: false,
        signed,
        y: 1,
        z: 2,
        decode: Decode::STORED,
    });
    if rg {
        let outside = |s: NormalSource| {
            let beyond = |p: &&[f32; 4]| {
                let [x, y, _] = s.read(**p);
                x * x + y * y > 1.0
            };
            texels().filter(beyond).count()
        };
        let (kept, moved) = (outside(kept), outside(moved));
        return 2 * moved > texels().count() && kept < moved;
    }
    let error = |s: NormalSource| {
        texels()
            .map(|p| f64::from((Vec3::from(s.read(*p)).length() - 1.0).abs().min(1.0)))
            .sum::<f64>()
    };
    error(kept) < error(moved)
}

/// Encoding and layout for the mip chains of `layers` (values linear, normals unit).
fn finish(mut chains: Vec<Vec<Img>>, plan: &Plan) -> Result<Built, String> {
    if plan.mips.normal {
        for p in chains.iter_mut().flatten().flat_map(|m| &mut m.px) {
            p[0] = normal::encode(p[0]);
            p[1] = normal::encode(p[1]);
        }
    }
    let [w, h] = [chains[0][0].w, chains[0][0].h];
    let encoded = encode::encode(plan.encoding, plan.quality, &chains)?;
    let alpha_cutoff = plan.mips.coverage.map_or(f32::NAN, |(_, c)| c);
    Ok(layout(
        plan.encoding.vk_format(),
        [w, h],
        plan.cube,
        plan.primaries,
        alpha_cutoff,
        plan.lights.clone(),
        &encoded,
    ))
}

/// Archive blobs of an encoded texture: `levels[layer][mip]` holds tight block rows.
fn layout(
    vk_format: u32,
    [w, h]: [u32; 2],
    cube: bool,
    primaries: Primaries,
    alpha_cutoff: f32,
    lights: Vec<ExtractedLight>,
    levels: &[Vec<Vec<u8>>],
) -> Built {
    let f = FormatInfo::of(vk_format).expect("cooked formats are known");
    let mip_count = levels[0].len() as u32;
    let tail = tail_first_mip(f, w, h, mip_count);
    let header = TextureHeader {
        vk_format,
        width: w,
        height: h,
        layers: levels.len() as u16,
        kind: if cube {
            TextureHeader::CUBE
        } else {
            TextureHeader::D2
        },
        mip_count: mip_count as u8,
        tail_first_mip: tail as u8,
        primaries: primaries.id(),
        _pad: [0; 2],
        alpha_cutoff,
        lights: Span::default(),
    };
    let mut blobs = Vec::new();
    for m in 0..tail {
        let [mw, mh] = mip_size(w, h, m);
        let mut out = Vec::new();
        for layer in levels {
            tile(f, mw, mh, &layer[m as usize], &mut out);
        }
        blobs.push(out);
    }
    if tail < mip_count {
        let mut out = Vec::new();
        for m in tail..mip_count {
            for layer in levels {
                out.extend_from_slice(&layer[m as usize]);
            }
        }
        blobs.push(out);
    }
    Built {
        info: TextureInfo { header, lights },
        blobs,
        filter: Filter::None,
        clips: false,
    }
}

/// Adds a cooked texture: blob 0 and the tail in the mount section, one 64 KiB tile per
/// chunk in the mip blobs. Every blob after blob 0 takes the texture's filter.
pub(crate) fn add(w: &mut ArchiveWriter, name: &str, key: Hash32, t: &Built) -> Result<AssetId> {
    let _z = crate::profile::zone!("archive.add");
    let head = t.info.to_bytes();
    let tail = t.info.header.tail_first_mip < t.info.header.mip_count;
    let mut blobs = vec![BlobIn {
        mount: true,
        ..BlobIn::new(&head)
    }];
    let last = t.blobs.len().saturating_sub(1);
    blobs.extend(t.blobs.iter().enumerate().map(|(i, b)| {
        if tail && i == last {
            BlobIn {
                mount: true,
                filter: t.filter,
                ..BlobIn::new(b)
            }
        } else {
            BlobIn {
                chunk_len: TILE_BYTES as u32,
                filter: t.filter,
                ..BlobIn::new(b)
            }
        }
    }));
    Ok(w.add(
        name,
        Class::Texture as u16,
        orr_pack::texture::SCHEMA,
        key,
        &blobs,
    )?)
}

/// Cooks `Source::Texture`: one texture entry from one image file.
pub(crate) fn cook_file(
    asset: &str,
    bytes: &[u8],
    s: &TextureSettings,
    quality: Quality,
) -> Result<Built> {
    load::open(bytes)
        .and_then(|src| cook_source(src, s, quality))
        .map(|b| b.with_split(s.bc_split))
        .map_err(|message| Error::Texture {
            asset: asset.into(),
            message,
        })
}

/// The encoders and their preset, for cook keys; their crate versions enter through
/// [`crate::key`].
pub(crate) fn encoders(quality: Quality) -> String {
    format!("bc7e for BC7, amd for BC4 BC5, intel for BC6H, preset {quality:?}")
}

/// Whether `usage` holds color, not numbers.
fn reads_color(usage: Usage) -> bool {
    matches!(
        usage,
        Usage::Color
            | Usage::ColorAlpha
            | Usage::Hdr
            | Usage::Data(DataFormat::Rgba8Srgb)
            | Usage::Environment { .. }
    )
}

fn cook_source(mut src: Source, s: &TextureSettings, quality: Quality) -> Result<Built, String> {
    let color = reads_color(s.usage);
    let negatives = color && (src.negatives || src.color.can_read_below());
    let lowest = if negatives {
        load::stored_range(&src)?.map(|r| r.0)
    } else {
        [0.0; 4]
    };
    let (read, primaries, gamut) = ImageColor::File.color(&src.color, s.primaries, |read| {
        lowest[..3].iter().any(|&v| read.alone(v) < BELOW_ZERO)
    });
    let gamut = gamut.as_ref();
    let measures = negatives || (color && read.reaches_past(gamut));
    let clipped = AtomicBool::new(false);
    // The conversion says whether a color clips, and the first that does settles it.
    let measure = |read: &Read, p: [f32; 4]| {
        if measures && !clipped.load(Ordering::Relaxed) && read.clips(p, gamut) {
            clipped.store(true, Ordering::Relaxed);
        }
    };
    if let Usage::Environment {
        light_threshold,
        max_lights,
    } = s.usage
    {
        if s.mips == Some(false) {
            return Err("environment maps have mips".into());
        }
        load::decode_layers(&mut src)?;
        // Nothing lies behind an environment, so its stored color is the light it gives.
        let read = Read {
            straight: false,
            ..read
        };
        for p in src.layers.iter_mut().flat_map(|l| &mut l.px) {
            measure(&read, *p);
            *p = read.color(*p, gamut);
        }
        return env::cook(src, s, primaries, light_threshold, max_lights, quality)
            .map(|b| b.with_clips(&clipped));
    }
    let full_chain = s.mips.unwrap_or(s.usage.has_mips());
    let kept_levels = |b: &Blocks| kept_mip_count(b.w, b.h, s.max_size, full_chain) as usize;
    if let Some(b) = &mut src.blocks {
        let n = kept_levels(b);
        b.levels.iter_mut().for_each(|l| l.truncate(n));
    }
    let kept = match s.usage {
        Usage::Linear1 => 1,
        Usage::Linear2 => 2,
        _ => 0,
    };
    // A cutoff makes the channel an alpha, 0..1 whatever the source stores.
    let signed = s.alpha_cutoff.is_none() && {
        if src.float && kept > 0 {
            load::decode_layers(&mut src)?;
        }
        let negative = |l: &Img| l.px.iter().any(|p| p[..kept].iter().any(|&v| v < 0.0));
        src.signed || (src.float && kept > 0 && src.layers.iter().any(negative))
    };
    let encoding = Encoding::of(s.usage, signed, src.float);
    let coverage = match (s.alpha_cutoff, s.usage) {
        (None, _) => None,
        (Some(c), Usage::ColorAlpha | Usage::Linear4) => Some((3, c)),
        (Some(c), Usage::Linear1) => Some((0, c)),
        (Some(_), u) => return Err(format!("alpha_cutoff does not apply to {u:?}")),
    };
    let normal = s.usage == Usage::Normal;
    let vectors = normal && src.float && src.negatives;
    if vectors {
        load::decode_layers(&mut src)?;
    }
    let source = NormalSource {
        rg: src.rg,
        signed: src.signed || (vectors && stores_normals(&src.layers, src.rg)),
        y: 1,
        z: 2,
        decode: Decode::STORED,
    };
    let linear = |p: &mut [f32; 4]| {
        if color {
            measure(&read, *p);
            *p = read.color(*p, gamut);
        }
        if normal {
            let [x, y, z] = source.read(*p);
            *p = [x, y, z, 1.0];
        }
    };
    let plan = Plan {
        encoding,
        cube: src.cube,
        primaries,
        mips: MipSpec {
            filter: encoding.mip_filter(s.filter),
            wrap: s.wrap,
            grid: s.atlas_grid.map(|g| g.map(u32::from)),
            normal,
            coverage,
            height: s.usage == Usage::Displacement,
        },
        quality,
        max_size: s.max_size,
        full_chain,
        toksvig: None,
        lights: Vec::new(),
    };
    // A source with every mip in a block format keeps its mips, unless they are remade.
    let remakes = crate::quality::remakes_mips(quality);
    if let Some(b) = src
        .blocks
        .as_ref()
        .filter(|b| b.levels[0].len() == kept_levels(b) && s.atlas_grid.is_none())
    {
        let stored = if encoding == Encoding::Bc7Srgb {
            Decode::SRGB
        } else {
            Decode::STORED
        };
        if b.vk_format == encoding.vk_format()
            && (coverage.is_none() || !remakes)
            && (!color || gamut.is_none() && read.is(stored))
        {
            let (w, h) = (b.w, b.h);
            let skip = skipped_mips(w, h, s.max_size);
            let levels: Vec<Vec<Vec<u8>>> = src
                .blocks
                .take()
                .expect("blocks")
                .levels
                .into_iter()
                .map(|mut l| l.split_off(skip as usize))
                .collect();
            return Ok(layout(
                encoding.vk_format(),
                mip_size(w, h, skip),
                src.cube,
                primaries,
                coverage.map_or(f32::NAN, |c| c.1),
                Vec::new(),
                &levels,
            ));
        }
        if !remakes {
            let mut chains = source_chains(b, s.max_size, |mut p| {
                linear(&mut p);
                if normal { mips::normalized(p) } else { p }
            })?;
            if plan.mips.height {
                load::decode_layers(&mut src)?;
                for (chain, l) in chains.iter_mut().zip(&mut src.layers) {
                    mips::keep_heights(chain, mips::settle_heights(l));
                }
            }
            return finish(chains, &plan).map(|b| b.with_clips(&clipped));
        }
    }
    load::decode_layers(&mut src)?;
    let layers = src
        .layers
        .into_iter()
        .map(|mut l| {
            l.px.iter_mut().for_each(linear);
            l
        })
        .collect();
    build(layers, &plan).map(|b| b.with_clips(&clipped))
}

/// Every layer's mips of `b` as stored, without those over `max_size`, decoded and passed
/// texel by texel through `f`.
fn source_chains(
    b: &Blocks,
    max_size: Option<u32>,
    f: impl Fn([f32; 4]) -> [f32; 4] + Sync,
) -> Result<Vec<Vec<Img>>, String> {
    let skip = skipped_mips(b.w, b.h, max_size) as usize;
    b.levels
        .iter()
        .map(|l| {
            l.iter()
                .enumerate()
                .skip(skip)
                .map(|(m, bytes)| {
                    let [w, h] = mip_size(b.w, b.h, m as u32);
                    load::decode_blocks_with(b.vk_format, w, h, bytes, &f)
                })
                .collect()
        })
        .collect()
}
