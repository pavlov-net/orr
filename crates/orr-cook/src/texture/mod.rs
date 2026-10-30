//! Texture cook: import, mip chains, encoding, and the tiled archive layout.

mod encode;
mod env;
mod load;
pub(crate) mod material;
mod mips;

use orr_pack::texture::{
    ExtractedLight, FormatInfo, TILE_BYTES, TextureHeader, TextureInfo, full_mip_count, mip_size,
    tail_first_mip, tile,
};
use orr_pack::{ArchiveWriter, AssetId, BlobIn, Class, Hash32, Span};
use serde::Deserialize;

pub(crate) use encode::{Encoding, encode as encode_chains};
pub(crate) use load::{Source, decode_blocks};
pub(crate) use mips::{MipSpec, Toksvig};

use crate::{Error, Result};

/// How one texture is cooked.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureSettings {
    pub usage: Usage,
    /// Alpha-tested coverage cutoff. With it, every mip's alpha is scaled so the fraction of
    /// the texture at or above the cutoff matches mip 0. Applies to the alpha of
    /// [`Usage::ColorAlpha`] and [`Usage::Linear4`] and to the value of [`Usage::Linear1`].
    pub alpha_cutoff: Option<f32>,
    pub quality: Quality,
    /// Largest mip 0 edge; larger mips are dropped. For [`Usage::Environment`] the map edge,
    /// rounded down to a power of two (1024 when unset).
    pub max_size: Option<u32>,
    /// Cells per axis of an atlas that divides the texture evenly. Mips whose cells are a
    /// whole number of texels and at least 4x4 filter each cell alone and match alpha
    /// coverage per cell; smaller mips filter the whole texture.
    pub atlas_grid: Option<[u16; 2]>,
    /// Source primaries of colour texels; overrides the file's tag. Untagged files are
    /// Rec. 709.
    pub primaries: Option<Primaries>,
    pub filter: MipFilter,
    /// Edge handling of the mip filter per axis (u, v).
    pub wrap: [Wrap; 2],
}

impl Default for TextureSettings {
    fn default() -> Self {
        Self {
            usage: Usage::Color,
            alpha_cutoff: None,
            quality: Quality::Release,
            max_size: None,
            atlas_grid: None,
            primaries: None,
            filter: MipFilter::Kaiser,
            wrap: [Wrap::Repeat; 2],
        }
    }
}

/// What a texture holds; sets its format.
#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Usage {
    /// sRGB colour (RGB) to BC7 sRGB.
    Color,
    /// sRGB colour with linear alpha to BC7 sRGB.
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
    /// Linear HDR colour to BC6H.
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

/// Encoder effort.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    /// For development cooks.
    Fast,
    /// For release cooks.
    #[default]
    Release,
}

/// Colour primaries.
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
        match self {
            Self::Rec709 => [0.2126, 0.7152, 0.0722],
            Self::DisplayP3 => [0.2290, 0.6917, 0.0793],
            Self::Rec2020 => [0.2627, 0.6780, 0.0593],
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

    /// Bilinear resize; used to bring a packed texture's inputs to one size.
    pub(crate) fn resized(&self, w: u32, h: u32) -> Self {
        if (w, h) == (self.w, self.h) {
            return self.clone();
        }
        let mut out = Self::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let sx = (x as f32 + 0.5) * self.w as f32 / w as f32 - 0.5;
                let sy = (y as f32 + 0.5) * self.h as f32 / h as f32 - 0.5;
                out.px[(y * w + x) as usize] = self.bilinear(sx, sy);
            }
        }
        out
    }

    /// Bilinear sample at texel coordinates (texel centres at integers), clamped at edges.
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
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// What to cook from prepared linear layers.
pub(crate) struct Plan {
    pub encoding: Encoding,
    pub cube: bool,
    pub primaries: Primaries,
    pub mips: MipSpec,
    pub quality: Quality,
    pub max_size: Option<u32>,
    pub toksvig: Option<Toksvig>,
    pub lights: Vec<ExtractedLight>,
}

/// A cooked texture: blob 0 and the blobs after it.
pub(crate) struct Built {
    pub info: TextureInfo,
    pub blobs: Vec<Vec<u8>>,
}

/// Mip chains, encoding and layout for `layers` (mip 0 each, values linear).
pub(crate) fn build(layers: Vec<Img>, plan: &Plan) -> Result<Built, String> {
    let (w, h) = (layers[0].w, layers[0].h);
    if layers.iter().any(|l| (l.w, l.h) != (w, h)) {
        return Err("layers differ in size".into());
    }
    let mut chains: Vec<Vec<Img>> = layers
        .into_iter()
        .map(|l| mips::chain(l, &plan.mips))
        .collect();
    if let Some(t) = &plan.toksvig {
        for c in &mut chains {
            t.apply(c, &plan.mips)?;
        }
    }
    let skip = plan.max_size.map_or(0, |max| {
        (0..full_mip_count(w, h))
            .take_while(|&m| mip_size(w, h, m).into_iter().any(|e| e > max.max(1)))
            .count()
    });
    for c in &mut chains {
        c.drain(..skip);
    }
    let [w, h] = mip_size(w, h, skip as u32);
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
    }
}

/// Adds a cooked texture: blob 0 and the tail in the mount section, one 64 KiB tile per
/// chunk in the mip blobs.
pub(crate) fn add(w: &mut ArchiveWriter, name: &str, key: Hash32, t: &Built) -> Result<AssetId> {
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
                ..BlobIn::new(b)
            }
        } else {
            BlobIn {
                chunk_len: TILE_BYTES as u32,
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
pub(crate) fn cook_file(asset: &str, bytes: &[u8], s: &TextureSettings) -> Result<Built> {
    load::decode(bytes)
        .and_then(|src| cook_source(src, s))
        .map_err(|message| Error::Texture {
            asset: asset.into(),
            message,
        })
}

/// The encoders, for cook keys; their crate versions enter through [`crate::key`].
pub(crate) const ENCODERS: &str = "bc7e for BC7, intel for BC4 BC5 BC6H";

fn cook_source(src: Source, s: &TextureSettings) -> Result<Built, String> {
    let primaries = s.primaries.or(src.primaries).unwrap_or(Primaries::Rec709);
    if let Usage::Environment {
        light_threshold,
        max_lights,
    } = s.usage
    {
        return env::cook(src, s, primaries, light_threshold, max_lights);
    }
    let encoding = Encoding::of(s.usage);
    if let Some(b) = &src.blocks
        && b.vk_format == encoding.vk_format()
        && b.complete()
        && s.atlas_grid.is_none()
        && s.alpha_cutoff.is_none()
        && s.max_size.is_none_or(|m| b.w.max(b.h) <= m)
    {
        return Ok(layout(
            b.vk_format,
            [b.w, b.h],
            src.cube,
            primaries,
            f32::NAN,
            Vec::new(),
            &b.levels,
        ));
    }
    let coverage = match (s.alpha_cutoff, s.usage) {
        (None, _) => None,
        (Some(c), Usage::ColorAlpha | Usage::Linear4) => Some((3, c)),
        (Some(c), Usage::Linear1) => Some((0, c)),
        (Some(_), u) => return Err(format!("alpha_cutoff does not apply to {u:?}")),
    };
    let colour = matches!(
        s.usage,
        Usage::Color | Usage::ColorAlpha | Usage::Hdr | Usage::Data(DataFormat::Rgba8Srgb)
    );
    let normal = s.usage == Usage::Normal;
    let layers = src
        .layers
        .into_iter()
        .map(|mut l| {
            for p in &mut l.px {
                if colour && src.srgb {
                    for c in &mut p[..3] {
                        *c = srgb_to_linear(*c);
                    }
                }
                if normal {
                    *p = [p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0, p[2] * 2.0 - 1.0, 1.0];
                }
            }
            l
        })
        .collect();
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
        quality: s.quality,
        max_size: s.max_size,
        toksvig: None,
        lights: Vec::new(),
    };
    build(layers, &plan)
}
