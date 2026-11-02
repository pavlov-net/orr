//! Texture cook: import, mip chains, encoding, and the tiled archive layout.

use std::sync::Arc;

mod encode;
mod env;
mod load;
pub(crate) mod material;
mod mips;
pub(crate) mod neural;

use orr_pack::texture::{
    ExtractedLight, FormatInfo, TILE_BYTES, TextureHeader, TextureInfo, full_mip_count, mip_size,
    normal, tail_first_mip, tile,
};
use orr_pack::{ArchiveWriter, AssetId, BcLayout, BlobIn, Class, Filter, Hash32, Span};
use serde::Deserialize;

use ctt::Quality;
pub(crate) use encode::{Encoding, encode as encode_chains};
pub(crate) use load::{Blocks, Source, decode_blocks, decode_channel, file_header};
pub(crate) use mips::{MipSpec, Toksvig};

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
    /// Source primaries of color texels; overrides the file's tag. Untagged files are
    /// Rec. 709.
    pub primaries: Option<Primaries>,
    pub filter: MipFilter,
    /// Edge handling of the mip filter per axis (u, v).
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

/// What a texture holds; sets its format.
#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Usage {
    /// sRGB color (RGB) to BC7 sRGB.
    Color,
    /// sRGB color with linear alpha to BC7 sRGB.
    ColorAlpha,
    /// Tangent-space normal (RGB as stored, `2v - 1`) to BC5 holding x and y; z is
    /// reconstructed.
    Normal,
    /// One linear channel (R) to BC4.
    Linear1,
    /// Two linear channels (RG) to BC5.
    Linear2,
    /// Four linear channels to BC7.
    Linear4,
    /// Linear HDR color to BC6H.
    Hdr,
    /// Height (R) to R16_UNORM.
    Displacement,
    /// An equirectangular (2:1) or cube HDR map resampled to an equal-area octahedral map,
    /// BC6H with mips. Each connected region over `light_threshold` (luminance), brightest
    /// first up to `max_lights`, moves into one directional light of equal energy.
    Environment {
        #[serde(default)]
        light_threshold: Option<f32>,
        max_lights: u8,
    },
    /// Uncompressed, the source channels the format holds.
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
    Rg16,
    Rgba16,
    R16f,
    Rg16f,
    Rgba16f,
    R32f,
    Rg32f,
    Rgba32f,
}

/// Color primaries.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Primaries {
    Rec709,
    DisplayP3,
    Rec2020,
}

impl Primaries {
    fn header(self) -> u8 {
        match self {
            Self::Rec709 => TextureHeader::REC709,
            Self::DisplayP3 => TextureHeader::DISPLAY_P3,
            Self::Rec2020 => TextureHeader::REC2020,
        }
    }

    /// Luminance weights of linear RGB in these primaries.
    fn luminance(self) -> [f32; 3] {
        orr_color::Primaries::from(self)
            .rgb()
            .luminance_weights()
            .as_vec3()
            .to_array()
    }
}

impl From<Primaries> for orr_color::Primaries {
    fn from(p: Primaries) -> Self {
        match p {
            Primaries::Rec709 => Self::Rec709,
            Primaries::DisplayP3 => Self::DisplayP3,
            Primaries::Rec2020 => Self::Rec2020,
        }
    }
}

/// Mip downsampling filter, applied to linear values.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MipFilter {
    /// Area average.
    Box,
    /// Kaiser-windowed sinc, 3 lobes, alpha 4.
    #[default]
    Kaiser,
}

/// Filter edge handling.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

/// Linear float RGBA texels, row-major.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Img {
    pub w: u32,
    pub h: u32,
    pub px: Vec<[f32; 4]>,
}

impl Img {
    pub(crate) fn new(w: u32, h: u32) -> Self {
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

    /// Bilinear resize; used to bring a packed texture's inputs to one size.
    pub(crate) fn resized(self: Arc<Self>, w: u32, h: u32) -> Arc<Self> {
        if (w, h) == (self.w, self.h) {
            return self;
        }
        let mut out = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f32 + 0.5) * self.w as f32 / w as f32 - 0.5;
                let sy = (y as f32 + 0.5) * self.h as f32 / h as f32 - 0.5;
                out.px[(y * w + x) as usize] = self.bilinear(sx, sy);
            }
        }
        Arc::new(out)
    }

    /// Bilinear sample at texel coordinates (texel centers at integers), clamped at edges.
    pub(crate) fn bilinear(&self, x: f32, y: f32) -> [f32; 4] {
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
pub(crate) struct Plan {
    pub encoding: Encoding,
    pub cube: bool,
    pub primaries: Primaries,
    pub mips: MipSpec,
    pub quality: Quality,
    pub max_size: Option<u32>,
    /// Mips down to 1x1; without them the texture is the first mip that fits `max_size`.
    pub full_chain: bool,
    pub toksvig: Option<Toksvig>,
    pub lights: Vec<ExtractedLight>,
}

/// A cooked texture: blob 0 and the blobs after it, stored with `filter`.
pub(crate) struct Built {
    pub info: TextureInfo,
    pub blobs: Vec<Vec<u8>>,
    pub filter: Filter,
}

/// The report's warning for a source that reads the animated image `image`.
pub(crate) fn first_frame_warning(image: &str) -> String {
    format!("image {image} is animated; only its first frame is cooked")
}

impl Built {
    /// Stores the blocks split, or not, as `bc_split` asks; `None` takes the format's
    /// default.
    pub(crate) fn with_split(mut self, bc_split: Option<bool>) -> Self {
        let layout = BcLayout::of(self.info.header.vk_format);
        self.filter = match layout {
            Some(l) if bc_split.unwrap_or_else(|| split_by_default(l)) => Filter::BcSplit(l),
            _ => Filter::None,
        };
        self
    }
}

/// Whether a format's blocks are split when the settings leave it open.
fn split_by_default(_: BcLayout) -> bool {
    true
}

/// The mips of a stored `w` x `h` chain that fit `max_size`, and the size of the first.
pub(crate) fn kept_mips(
    mut mips: Vec<Vec<u8>>,
    w: u32,
    h: u32,
    max_size: Option<u32>,
) -> ([u32; 2], Vec<Vec<u8>>) {
    let skip = skipped_mips(w, h, max_size);
    (mip_size(w, h, skip), mips.split_off(skip as usize))
}

/// Mips a `w` x `h` texture keeps: down to 1x1 with `full_chain`, else down to the first
/// that fits `max_size`.
pub(crate) fn kept_mip_count(w: u32, h: u32, max_size: Option<u32>, full_chain: bool) -> u32 {
    if full_chain {
        full_mip_count(w, h)
    } else {
        skipped_mips(w, h, max_size) + 1
    }
}

/// Mips dropped from the top of a `w` x `h` chain so mip 0 fits `max_size`.
pub(crate) fn skipped_mips(w: u32, h: u32, max_size: Option<u32>) -> u32 {
    max_size.map_or(0, |max| {
        (0..full_mip_count(w, h))
            .take_while(|&m| mip_size(w, h, m).into_iter().any(|e| e > max.max(1)))
            .count() as u32
    })
}

/// The mip chains of `layers` (mip 0 each, values linear) as they are encoded: filtered,
/// without the mips over `max_size`, with Toksvig roughness from kept mip `first` on.
pub(crate) fn chains(layers: Vec<Img>, plan: &Plan, first: usize) -> Result<Vec<Vec<Img>>, String> {
    let (w, h) = (layers[0].w, layers[0].h);
    if layers.iter().any(|l| (l.w, l.h) != (w, h)) {
        return Err("layers differ in size".into());
    }
    let skip = skipped_mips(w, h, plan.max_size) as usize;
    let count = kept_mip_count(w, h, plan.max_size, plan.full_chain);
    let mut chains: Vec<Vec<Img>> = layers
        .into_iter()
        .map(|l| mips::chain(l, &plan.mips, count))
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
pub(crate) fn build(layers: Vec<Img>, plan: &Plan) -> Result<Built, String> {
    finish(chains(layers, plan, 0)?, plan)
}

/// Encoding and layout for the mip chains of `layers` (values linear, normals unit).
pub(crate) fn finish(mut chains: Vec<Vec<Img>>, plan: &Plan) -> Result<Built, String> {
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
pub(crate) fn layout(
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
        primaries: primaries.header(),
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
    format!("bc7e for BC7, intel for BC4 BC5 BC6H, preset {quality:?}")
}

fn cook_source(mut src: Source, s: &TextureSettings, quality: Quality) -> Result<Built, String> {
    let primaries = s.primaries.or(src.primaries).unwrap_or(Primaries::Rec709);
    if let Usage::Environment {
        light_threshold,
        max_lights,
    } = s.usage
    {
        if s.mips == Some(false) {
            return Err("environment maps have mips".into());
        }
        load::decode_layers(&mut src)?;
        return env::cook(src, s, primaries, light_threshold, max_lights, quality);
    }
    let full_chain = s.mips.unwrap_or(s.usage.has_mips());
    let kept_levels = |b: &Blocks| kept_mip_count(b.w, b.h, s.max_size, full_chain) as usize;
    if let Some(b) = &mut src.blocks {
        let n = kept_levels(b);
        b.levels.iter_mut().for_each(|l| l.truncate(n));
    }
    let encoding = Encoding::of(s.usage);
    let coverage = match (s.alpha_cutoff, s.usage) {
        (None, _) => None,
        (Some(c), Usage::ColorAlpha | Usage::Linear4) => Some((3, c)),
        (Some(c), Usage::Linear1) => Some((0, c)),
        (Some(_), u) => return Err(format!("alpha_cutoff does not apply to {u:?}")),
    };
    let color = matches!(
        s.usage,
        Usage::Color | Usage::ColorAlpha | Usage::Hdr | Usage::Data(DataFormat::Rgba8Srgb)
    );
    let normal = s.usage == Usage::Normal;
    let srgb = src.srgb;
    let linear = |p: &mut [f32; 4]| {
        if color && srgb {
            for c in &mut p[..3] {
                *c = srgb_to_linear(*c);
            }
        }
        if normal {
            *p = [p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0, p[2] * 2.0 - 1.0, 1.0];
        }
    };
    let plan = Plan {
        encoding,
        cube: src.cube,
        primaries,
        mips: MipSpec {
            filter: s.filter,
            wrap: s.wrap,
            grid: s.atlas_grid.map(|g| g.map(u32::from)),
            normal,
            coverage,
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
        if b.vk_format == encoding.vk_format() && (coverage.is_none() || !remakes) {
            let (w, h) = (b.w, b.h);
            let size = mip_size(w, h, skipped_mips(w, h, s.max_size));
            let levels: Vec<Vec<Vec<u8>>> = src
                .blocks
                .take()
                .expect("blocks")
                .levels
                .into_iter()
                .map(|l| kept_mips(l, w, h, s.max_size).1)
                .collect();
            return Ok(layout(
                encoding.vk_format(),
                size,
                src.cube,
                primaries,
                coverage.map_or(f32::NAN, |c| c.1),
                Vec::new(),
                &levels,
            ));
        }
        if !remakes {
            let chains = source_chains(b, s.max_size, |mut p| {
                linear(&mut p);
                if normal { mips::normalized(p) } else { p }
            })?;
            return finish(chains, &plan);
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
    build(layers, &plan)
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
