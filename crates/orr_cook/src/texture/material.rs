//! A material's canonical channel set: its source images repacked into the textures the
//! renderer samples, with constant images collapsed into factors.
//!
//! - Base color: BC7 sRGB (RGB). Under `MASK` and `BLEND`, base alpha times the opacity
//!   factor becomes a BC4 opacity texture, with alpha-coverage mips at the cutoff under
//!   `MASK` unless it keeps its source's mips.
//! - Normal: BC5, with the glTF normal scale applied.
//! - Roughness and metalness share a BC5 (BC4 when only one varies); occlusion gets a BC4.
//!   Factors are folded into these texels and the material's factors set to 1.
//! - Roughness takes the normal map's variance per mip (Toksvig), so a material with a normal
//!   map gets a roughness texture even when its roughness is constant. At the quality levels
//!   that keep source mips, roughness that keeps them, and roughness beside a normal map
//!   that keeps them, takes none.
//! - Emission and subsurface color: BC7 sRGB.
//! - Specular weight (the specular texture's alpha), coat and transmission weights (red):
//!   BC4 each. Coat and transmission factors are folded into their texels and set to 1;
//!   the specular weight texture multiplies `specular_weight`.
//! - Displacement: R16 unorm from red, never collapsed; displacement is
//!   `displacement_scale x texel + displacement_offset`.
//! - An image whose texels all lie within 1/255 of one value collapses into the material's
//!   factor; a constant normal or occlusion image, which has no factor, cooks at 4x4.
//!
//! A bound texture's value multiplies the matching factor; normal, occlusion and opacity
//! textures have no factor.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use ctt::Quality;
use orr_color::RgbPrimaries;
use orr_math::DVec3;
use orr_pack::material::{Material, TextureBinding, slot};
use orr_pack::texture::{full_mip_count, mip_size};
use orr_pack::{AssetId, Class, Hash32};
use rayon::prelude::*;
use serde::Deserialize;

use super::load::{self, Source};
use super::mips::{MipSpec, Toksvig, normalized};
use super::neural::{self, NeuralJob, NeuralSettings};
use super::{Built, DataFormat, Encoding, Img, MipFilter, Plan, Primaries, Wrap, srgb_to_linear};
use crate::key::Key;
use crate::state::{self, Memo};
use crate::{Error, Result};

/// Texture settings for images a glTF file references.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureRules {
    /// Largest mip 0 edge of every texture.
    pub max_size: Option<u32>,
    pub filter: MipFilter,
    /// See [`super::TextureSettings::bc_split`].
    pub bc_split: Option<bool>,
    /// Applied in order to every texture made from an image whose name or URI contains the
    /// pattern.
    pub overrides: Vec<(String, TextureOverride)>,
    /// With it, each material also cooks a neural texture set of its textures.
    pub neural: Option<NeuralSettings>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureOverride {
    pub max_size: Option<u32>,
    pub atlas_grid: Option<[u16; 2]>,
    pub primaries: Option<Primaries>,
    pub bc_split: Option<bool>,
    /// The source channel each material input reads from the matching images, where they
    /// are laid out differently from what the importer assumes.
    pub channels: InputChannels,
}

/// Source channels (0 red .. 3 alpha) of scalar material inputs; `None` keeps the
/// importer's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InputChannels {
    pub opacity: Option<u8>,
    pub roughness: Option<u8>,
    pub metalness: Option<u8>,
    pub occlusion: Option<u8>,
    pub specular: Option<u8>,
    pub coat: Option<u8>,
    pub transmission: Option<u8>,
    pub displacement: Option<u8>,
}

/// One image a material reads.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TexRef {
    pub image: usize,
    /// The source channel of a scalar input; color inputs read RGB.
    pub channel: u8,
    /// The channel read instead from an image that holds only red and green.
    pub rg_channel: Option<u8>,
    pub uv_set: u8,
    pub wrap: [Wrap; 2],
    /// The uv transform the texture is sampled with (`Material::uv_transforms` form).
    pub uv: [f32; 6],
    /// Whether color texels are sRGB-encoded; `None` takes the file's own encoding.
    pub srgb: Option<bool>,
}

impl TexRef {
    /// Whether both read their textures at the same uvs.
    fn samples_like(&self, o: &TexRef) -> bool {
        self.uv_set == o.uv_set && self.uv == o.uv
    }
}

/// The images one material reads, per input.
#[derive(Clone, Debug, Default)]
pub(crate) struct MaterialTextures {
    pub base_color: Option<TexRef>,
    /// Read under `MASK` and `BLEND` only.
    pub opacity: Option<TexRef>,
    pub roughness: Option<TexRef>,
    pub metalness: Option<TexRef>,
    /// With its scale per tangent axis; a negative scale flips the axis.
    pub normal: Option<(TexRef, [f32; 2])>,
    /// With its strength.
    pub occlusion: Option<(TexRef, f32)>,
    pub emissive: Option<TexRef>,
    pub specular: Option<TexRef>,
    pub coat: Option<TexRef>,
    pub transmission: Option<TexRef>,
    pub subsurface_color: Option<TexRef>,
    pub displacement: Option<TexRef>,
}

impl MaterialTextures {
    /// Every image reference.
    pub(crate) fn refs(&self) -> impl Iterator<Item = &TexRef> {
        [
            self.base_color.as_ref(),
            self.opacity.as_ref(),
            self.roughness.as_ref(),
            self.metalness.as_ref(),
            self.normal.as_ref().map(|n| &n.0),
            self.occlusion.as_ref().map(|o| &o.0),
            self.emissive.as_ref(),
            self.specular.as_ref(),
            self.coat.as_ref(),
            self.transmission.as_ref(),
            self.subsurface_color.as_ref(),
            self.displacement.as_ref(),
        ]
        .into_iter()
        .flatten()
    }
}

pub(crate) struct ImageFile {
    /// glTF image name, else its URI, else its index.
    pub name: String,
    pub data: ImageData,
}

pub(crate) enum ImageData {
    Bytes(Vec<u8>),
    /// A file, read when needed. [`image_facts`] reads it first and records its hash and
    /// length, which guard the later reads.
    File {
        path: PathBuf,
        read: Option<(Hash32, u64)>,
    },
}

impl ImageFile {
    /// The file's bytes; a file must not have changed since its facts were read.
    pub(crate) fn bytes(&self) -> Result<Cow<'_, [u8]>> {
        match &self.data {
            ImageData::Bytes(b) => Ok(Cow::Borrowed(b)),
            ImageData::File {
                path,
                read: Some((hash, _)),
            } => Ok(Cow::Owned(state::read_checked(path, *hash)?)),
            ImageData::File { read: None, .. } => {
                unreachable!("an image's facts are read before its jobs")
            }
        }
    }

    pub(crate) fn len(&self) -> u64 {
        match &self.data {
            ImageData::Bytes(b) => b.len() as u64,
            ImageData::File { read, .. } => read.map_or(0, |r| r.1),
        }
    }
}

/// What packing needs to know of one source image; its texels are decoded again by the
/// texture jobs that read them.
pub(crate) struct ImageFacts {
    w: u32,
    h: u32,
    srgb: bool,
    primaries: Option<Primaries>,
    /// Per channel, the value every texel lies within 1/255 of, if there is one.
    constant: [Option<f32>; 4],
    hash: Hash32,
    /// The block format of a block-compressed file, and whether it has every mip.
    blocks: Option<(u32, bool)>,
    /// The file holds two channels, red and green.
    rg: bool,
}

/// Decodes a 2D image: its only layer and the file's color facts.
fn decode(bytes: &[u8]) -> Result<(Img, bool, Option<Primaries>), String> {
    let Source {
        mut layers,
        cube,
        srgb,
        primaries,
        ..
    } = load::decode(bytes)?;
    if cube || layers.len() != 1 {
        return Err("not a 2D image".into());
    }
    Ok((layers.remove(0), srgb, primaries))
}

/// Facts of the images `used`, read in parallel. File images are read through the memo, and
/// each one's hash and length are recorded in its [`ImageData::File`].
pub(crate) fn image_facts(
    asset: &str,
    images: &mut [ImageFile],
    used: &[usize],
    memo: &Memo,
) -> Result<HashMap<usize, ImageFacts>> {
    let fail = |message: String| Error::Texture {
        asset: asset.into(),
        message,
    };
    if let Some(&i) = used.iter().find(|&&i| i >= images.len()) {
        return Err(fail(format!("image {i} does not exist")));
    }
    let facts: HashMap<usize, (ImageFacts, Option<u64>)> = used
        .par_iter()
        .map(|&i| {
            let file = &images[i];
            let (bytes, hash, read) = match &file.data {
                ImageData::Bytes(b) => (Cow::Borrowed(b.as_slice()), Hash32::of(b), None),
                ImageData::File { path, .. } => {
                    let (bytes, hash) = memo.read(path)?;
                    let len = bytes.len() as u64;
                    (Cow::Owned(bytes), hash, Some(len))
                }
            };
            let facts =
                facts_of(&bytes, hash).map_err(|e| fail(format!("image {}: {e}", file.name)))?;
            Ok((i, (facts, read)))
        })
        .collect::<Result<_>>()?;
    let mut out = HashMap::with_capacity(facts.len());
    for (i, (f, len)) in facts {
        if let (ImageData::File { read, .. }, Some(len)) = (&mut images[i].data, len) {
            *read = Some((f.hash, len));
        }
        out.insert(i, f);
    }
    Ok(out)
}

/// The facts of one image file's bytes.
fn facts_of(bytes: &[u8], hash: Hash32) -> Result<ImageFacts, String> {
    let _z = crate::profile::zone!("texture.facts");
    // A channel is constant when it spans at most one UNORM8 step.
    const SPREAD: f32 = 1.0 / 255.0 + 1e-6;
    let constant = |range: [(f32, f32); 4]| {
        range.map(|(lo, hi)| (hi - lo <= SPREAD).then_some(0.5 * (lo + hi)))
    };
    if let Some(scan) = load::scan_ktx2(bytes, SPREAD) {
        let s = scan?;
        return Ok(ImageFacts {
            w: s.w,
            h: s.h,
            srgb: s.srgb,
            primaries: None,
            constant: constant(s.range),
            hash,
            blocks: Some((s.vk_format, s.complete)),
            rg: s.rg,
        });
    }
    let src = load::open(bytes)?;
    if src.cube || src.layers.len() > 1 || src.blocks.as_ref().is_some_and(|b| b.levels.len() != 1)
    {
        return Err("not a 2D image".into());
    }
    let (w, h, range) = match &src.blocks {
        Some(b) => {
            let range = load::block_range(b.vk_format, b.w, b.h, &mut &b.levels[0][0][..], SPREAD)?;
            (b.w, b.h, range)
        }
        None => {
            let img = &src.layers[0];
            let range = std::array::from_fn(|c| {
                img.px.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                    (lo.min(p[c]), hi.max(p[c]))
                })
            });
            (img.w, img.h, range)
        }
    };
    Ok(ImageFacts {
        w,
        h,
        srgb: src.srgb,
        primaries: src.primaries,
        constant: constant(range),
        hash,
        blocks: src.blocks.as_ref().map(|b| (b.vk_format, b.complete())),
        rg: src.rg,
    })
}

/// Plans the textures of a glTF file's materials, sharing identical ones. The textures
/// are cooked later by their [`TextureJob`]s.
pub(crate) struct Packer<'a> {
    asset: &'a str,
    images: &'a [ImageFile],
    facts: &'a HashMap<usize, ImageFacts>,
    rules: &'a TextureRules,
    quality: Quality,
    made: HashMap<Hash32, AssetId>,
    pub jobs: Vec<TextureJob>,
    /// Job index of each planned texture.
    by_id: HashMap<AssetId, usize>,
    /// Member textures of each planned neural set.
    sets: HashSet<Vec<AssetId>>,
    pub neural: Vec<NeuralJob>,
}

/// A texture to make from one or more source channels.
#[derive(Clone, Debug)]
struct Pack {
    role: &'static str,
    encoding: Encoding,
    /// Channel sources in output order.
    channels: Vec<Channel>,
    refs: Vec<TexRef>,
    coverage: Option<(usize, f32)>,
    /// Output channel, normal map and its scale.
    toksvig: Option<(usize, TexRef, f32)>,
    normal: bool,
    shrink: bool,
}

/// Where one output channel comes from.
#[derive(Copy, Clone, Debug)]
enum Channel {
    /// Source channel of an image, `a + b * value` (after sRGB decode when `srgb`).
    Image {
        image: usize,
        channel: usize,
        srgb: bool,
        a: f32,
        b: f32,
    },
    Constant(f32),
    /// Never read: stored as 1, or as the source has it where the blocks pass through.
    Unused,
}

/// One packed texture to cook.
#[derive(Clone)]
pub(crate) struct TextureJob {
    pub name: String,
    pub key: Hash32,
    asset: String,
    pack: Pack,
    /// Source images, sorted.
    images: Vec<usize>,
    size: [u32; 2],
    quality: Quality,
    max_size: Option<u32>,
    grid: Option<[u16; 2]>,
    primaries: Primaries,
    filter: MipFilter,
    bc_split: Option<bool>,
    /// Keeps the source's blocks.
    pass: Option<Pass>,
    /// Encodes the source images' own mips instead of making them from mip 0.
    source_mips: bool,
}

/// A planned texture and, per packed channel, the channel of the texture that holds it.
#[derive(Copy, Clone, Debug)]
struct Made {
    id: AssetId,
    channels: [u8; 4],
}

/// Binds slot `s` to packed channel `channel` of `t`, sampled as `r` is.
fn bind(m: &mut Material, s: usize, t: Made, channel: usize, r: &TexRef) {
    m.textures[s] = TextureBinding {
        texture: t.id,
        channels: t.channels[channel],
        uv_set: r.uv_set,
        _pad: [0; 2],
    };
    m.uv_transforms[s] = r.uv;
}

/// How a planned texture keeps its source image's blocks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Pass {
    image: usize,
    /// Every mip as stored, or only mip 0 with the rest made from it.
    all_mips: bool,
    /// The half (0 red, 1 green) of a BC5 source that a BC4 texture takes.
    half: Option<u8>,
}

impl<'a> Packer<'a> {
    pub(crate) fn new(
        asset: &'a str,
        images: &'a [ImageFile],
        facts: &'a HashMap<usize, ImageFacts>,
        rules: &'a TextureRules,
        quality: Quality,
    ) -> Self {
        Self {
            asset,
            images,
            facts,
            rules,
            quality,
            made: HashMap::new(),
            jobs: Vec::new(),
            by_id: HashMap::new(),
            sets: HashSet::new(),
            neural: Vec::new(),
        }
    }

    fn image(&self, i: usize) -> &ImageFacts {
        &self.facts[&i]
    }

    /// The value of `channel` if every texel lies within 1/255 of one value.
    fn constant(&self, i: usize, channel: usize) -> Option<f32> {
        self.image(i).constant[channel]
    }

    /// The primaries of a color texture made from image `i`: the last override that sets
    /// them, else the file's tag, else Rec.709.
    fn color_primaries(&self, i: usize) -> Primaries {
        self.rules
            .overrides
            .iter()
            .filter(|(pattern, _)| self.images[i].name.contains(pattern.as_str()))
            .fold(None, |p, (_, o)| o.primaries.or(p))
            .or(self.image(i).primaries)
            .unwrap_or(Primaries::Rec709)
    }

    /// The channel `r` reads for an input: the last override for its image that sets the
    /// input's channel, else its two-channel reading on a red-green image, else its own.
    fn channel_of(&self, r: TexRef, input: fn(&InputChannels) -> Option<u8>) -> TexRef {
        let name = &self.images[r.image].name;
        let set = self
            .rules
            .overrides
            .iter()
            .filter(|(pattern, _)| name.contains(pattern.as_str()))
            .fold(None, |c, (_, o)| input(&o.channels).or(c));
        let rg = r.rg_channel.filter(|_| self.image(r.image).rg);
        TexRef {
            channel: set.or(rg).unwrap_or(r.channel),
            ..r
        }
    }

    /// Cooks the material's textures, binds them and folds constants into its factors.
    pub(crate) fn material(
        &mut self,
        suffix: &str,
        m: &mut Material,
        t: &MaterialTextures,
    ) -> Result<()> {
        let remap = |r: Option<TexRef>, input| r.map(|r| self.channel_of(r, input));
        let t = &MaterialTextures {
            opacity: remap(t.opacity, |c| c.opacity),
            roughness: remap(t.roughness, |c| c.roughness),
            metalness: remap(t.metalness, |c| c.metalness),
            occlusion: t
                .occlusion
                .map(|(r, s)| (self.channel_of(r, |c| c.occlusion), s)),
            specular: remap(t.specular, |c| c.specular),
            coat: remap(t.coat, |c| c.coat),
            transmission: remap(t.transmission, |c| c.transmission),
            displacement: remap(t.displacement, |c| c.displacement),
            ..t.clone()
        };
        if let Some(r) = &t.base_color {
            // Only an alpha-tested or blended material reads base alpha as opacity, and only
            // where no opacity slot is bound.
            let alpha_free = m.alpha_mode == Material::OPAQUE
                || t.opacity
                    .is_some_and(|o| self.constant(o.image, usize::from(o.channel)).is_none());
            self.color(suffix, m, r, slot::BASE_COLOR, alpha_free)?;
        }
        if m.alpha_mode != Material::OPAQUE
            && let Some(r) = &t.opacity
        {
            let channel = usize::from(r.channel);
            match self.constant(r.image, channel) {
                Some(a) => m.geometry_opacity *= a,
                None => {
                    let coverage = (m.alpha_mode == Material::MASK).then_some((0, m.alpha_cutoff));
                    let id = self.make(
                        suffix,
                        Pack {
                            role: "opacity",
                            encoding: Encoding::Bc4,
                            channels: vec![Channel::Image {
                                image: r.image,
                                channel,
                                srgb: false,
                                a: 0.0,
                                b: m.geometry_opacity,
                            }],
                            refs: vec![*r],
                            coverage,
                            toksvig: None,
                            normal: false,
                            shrink: false,
                        },
                    )?;
                    m.geometry_opacity = 1.0;
                    bind(m, slot::OPACITY, id, 0, r);
                }
            }
        }

        let mut normal_packed = None;
        if let Some((r, scale)) = &t.normal {
            let flat = [0.5, 0.5, 1.0];
            let consts: Vec<Option<f32>> = (0..3).map(|c| self.constant(r.image, c)).collect();
            let is_flat = consts
                .iter()
                .zip(flat)
                .all(|(c, f)| c.is_some_and(|v| (v - f).abs() <= 1.0 / 255.0));
            if !is_flat {
                let channel = |c: usize, s: f32| Channel::Image {
                    image: r.image,
                    channel: c,
                    srgb: false,
                    a: -s,
                    b: 2.0 * s,
                };
                let id = self.make(
                    suffix,
                    Pack {
                        role: "normal",
                        encoding: Encoding::Bc5,
                        channels: vec![
                            channel(0, scale[0]),
                            channel(1, scale[1]),
                            channel(2, 1.0),
                            Channel::Constant(1.0),
                        ],
                        refs: vec![*r],
                        coverage: None,
                        toksvig: None,
                        normal: true,
                        shrink: consts.iter().all(Option::is_some),
                    },
                )?;
                bind(m, slot::NORMAL, id, 0, r);
                // A normal map whose source mips the cook keeps instead of remaking them adds
                // no Toksvig roughness.
                let toksvig = !self.jobs[self.by_id[&id.id]].source_mips;
                if toksvig && consts.iter().any(Option::is_none) {
                    normal_packed = Some((*r, scale[0]));
                }
            }
        }

        // Roughness, metalness, occlusion.
        let (rough, metal) = (t.roughness, t.metalness);
        let constant = |r: &Option<TexRef>| match r {
            Some(r) => self.constant(r.image, usize::from(r.channel)),
            None => Some(1.0),
        };
        let (rough_const, metal_const) = (constant(&rough), constant(&metal));
        let rough_varies = rough_const.is_none() || normal_packed.is_some();
        let metal_varies = metal_const.is_none();
        let occ = t
            .occlusion
            .map(|(r, s)| (r, s, self.constant(r.image, usize::from(r.channel))));
        let occ_varies = occ.as_ref().is_some_and(|o| o.2.is_none());
        let rough_channel = |m: &Material| match (&rough, rough_const) {
            (Some(r), None) => Channel::Image {
                image: r.image,
                channel: usize::from(r.channel),
                srgb: false,
                a: 0.0,
                b: m.specular_roughness,
            },
            (_, c) => Channel::Constant(m.specular_roughness * c.unwrap_or(1.0)),
        };
        if !rough_varies {
            m.specular_roughness *= rough_const.unwrap_or(1.0);
        }
        if !metal_varies {
            m.metalness *= metal_const.unwrap_or(1.0);
        }
        let occ_channel = |o: &(TexRef, f32, Option<f32>)| Channel::Image {
            image: o.0.image,
            channel: usize::from(o.0.channel),
            srgb: false,
            a: 1.0 - o.1,
            b: o.1,
        };
        let metal_channel = |m: &Material, r: &TexRef| Channel::Image {
            image: r.image,
            channel: usize::from(r.channel),
            srgb: false,
            a: 0.0,
            b: m.metalness,
        };
        let normal_ref = normal_packed.as_ref().map(|n| n.0);
        // Per texture: its (slot, source) channels and the references it samples like.
        // Roughness and metalness share one texture when they sample alike.
        type Slots = Vec<(usize, Channel)>;
        let mut packs: Vec<(Slots, Vec<TexRef>)> = Vec::new();
        if rough_varies {
            let refs = rough.or(normal_ref).into_iter().collect();
            packs.push((vec![(slot::ROUGHNESS, rough_channel(m))], refs));
        }
        if metal_varies {
            let r = metal.expect("varying metalness has an image");
            let entry = (slot::METALNESS, metal_channel(m, &r));
            let alike = |a: &TexRef| a.samples_like(&r) && a.wrap == r.wrap;
            match packs.first_mut() {
                Some((channels, refs)) if refs.first().is_none_or(alike) => {
                    channels.push(entry);
                    if refs.is_empty() {
                        refs.push(r);
                    }
                }
                _ => packs.push((vec![entry], vec![r])),
            }
        }
        for (rm, refs) in packs {
            let (encoding, role) = match rm.as_slice() {
                [(slot::ROUGHNESS, _)] => (Encoding::Bc4, "roughness"),
                [_] => (Encoding::Bc4, "metalness"),
                _ => (Encoding::Bc5, "roughness_metalness"),
            };
            let mut channels: Vec<Channel> = rm.iter().map(|c| c.1).collect();
            channels.resize(4, Channel::Constant(0.0));
            // Toksvig pairs roughness and normal texels, so both must sample alike.
            let toksvig = match (&normal_packed, rm[0].0 == slot::ROUGHNESS) {
                (Some((r, s)), true) if refs[0].samples_like(r) => Some((0, *r, *s)),
                _ => None,
            };
            let id = self.make(
                suffix,
                Pack {
                    role,
                    encoding,
                    channels,
                    refs: refs.clone(),
                    coverage: None,
                    toksvig,
                    normal: false,
                    shrink: false,
                },
            )?;
            for (i, (s, _)) in rm.iter().enumerate() {
                bind(m, *s, id, i, &refs[0]);
                if *s == slot::ROUGHNESS {
                    m.specular_roughness = 1.0;
                } else {
                    m.metalness = 1.0;
                }
            }
        }
        if let Some(o) = &occ
            && (occ_varies || o.2.is_some_and(|c| 1.0 - o.1 + o.1 * c < 1.0 - 1.0 / 255.0))
        {
            let id = self.make(
                suffix,
                Pack {
                    role: "occlusion",
                    encoding: Encoding::Bc4,
                    channels: vec![occ_channel(o)],
                    refs: vec![o.0],
                    coverage: None,
                    toksvig: None,
                    normal: false,
                    shrink: !occ_varies,
                },
            )?;
            bind(m, slot::OCCLUSION, id, 0, &o.0);
        }

        if let Some(r) = &t.emissive {
            self.color(suffix, m, r, slot::EMISSION, true)?;
        }
        if let Some(r) = &t.subsurface_color {
            self.color(suffix, m, r, slot::SUBSURFACE_COLOR, true)?;
        }
        if let Some(r) = &t.specular {
            match self.constant(r.image, usize::from(r.channel)) {
                Some(v) => m.specular_weight *= v,
                None => {
                    let id = self.single(suffix, r, 1.0, "specular_weight")?;
                    bind(m, slot::SPECULAR_WEIGHT, id, 0, r);
                }
            }
        }
        type Weight = fn(&mut Material) -> &mut f32;
        let weights: [(_, _, _, Weight); 2] = [
            (&t.coat, "coat", slot::COAT, |m| &mut m.coat_weight),
            (&t.transmission, "transmission", slot::TRANSMISSION, |m| {
                &mut m.transmission_weight
            }),
        ];
        for (r, role, s, weight) in weights {
            let Some(r) = r else { continue };
            let factor = *weight(m);
            *weight(m) = match self.constant(r.image, usize::from(r.channel)) {
                Some(v) => factor * v,
                None => {
                    let id = self.single(suffix, r, factor, role)?;
                    bind(m, s, id, 0, r);
                    1.0
                }
            };
        }
        if let Some(r) = &t.displacement {
            let id = self.make(
                suffix,
                Pack {
                    role: "displacement",
                    encoding: Encoding::Raw(DataFormat::R16),
                    channels: vec![Channel::Image {
                        image: r.image,
                        channel: usize::from(r.channel),
                        srgb: false,
                        a: 0.0,
                        b: 1.0,
                    }],
                    refs: vec![*r],
                    coverage: None,
                    toksvig: None,
                    normal: false,
                    shrink: self.constant(r.image, usize::from(r.channel)).is_some(),
                },
            )?;
            bind(m, slot::DISPLACEMENT, id, 0, r);
        }
        if let Some(settings) = &self.rules.neural {
            self.neural_set(suffix, m, settings)?;
        }
        Ok(())
    }

    /// Plans the neural set of the material's textures, unless one with the same textures
    /// is planned.
    fn neural_set(&mut self, suffix: &str, m: &Material, settings: &NeuralSettings) -> Result<()> {
        let mut ids: Vec<AssetId> = Vec::new();
        for (s, b) in m.textures.iter().enumerate() {
            if s != slot::OPACITY && b.texture != AssetId::NONE && !ids.contains(&b.texture) {
                ids.push(b.texture);
            }
        }
        let jobs: Vec<&TextureJob> = ids.iter().map(|id| &self.jobs[self.by_id[id]]).collect();
        let Some(set) = neural::plan(self.asset, suffix, &jobs, settings)? else {
            return Ok(());
        };
        if self.sets.insert(set.members()) {
            self.neural.push(set);
        }
        Ok(())
    }

    /// A color texture whose value multiplies the material color of slot `s` (base color,
    /// emission or subsurface color); a constant image folds into that color, multiplied in
    /// the image's primaries.
    fn color(
        &mut self,
        suffix: &str,
        m: &mut Material,
        r: &TexRef,
        s: usize,
        alpha_free: bool,
    ) -> Result<()> {
        let alpha = if alpha_free {
            Channel::Unused
        } else {
            Channel::Constant(1.0)
        };
        let (role, color): (_, fn(&mut Material) -> &mut [f32; 3]) = match s {
            slot::BASE_COLOR => ("base_color", |m| &mut m.base_color),
            slot::EMISSION => ("emission", |m| &mut m.emission_color),
            _ => ("subsurface_color", |m| &mut m.subsurface_color),
        };
        let srgb = r.srgb.unwrap_or(self.image(r.image).srgb);
        let rgb: Vec<Option<f32>> = (0..3).map(|c| self.constant(r.image, c)).collect();
        if rgb.iter().all(Option::is_some) {
            let texel: [f32; 3] = std::array::from_fn(|c| {
                let v = rgb[c].expect("constant");
                if srgb { srgb_to_linear(v) } else { v }
            });
            let target = color(m);
            // Rec.709 multiplies directly, so its factors stay exact.
            *target = match self.color_primaries(r.image) {
                Primaries::Rec709 => std::array::from_fn(|c| target[c] * texel[c]),
                p => {
                    let (rec709, p) = (RgbPrimaries::REC709, orr_color::Primaries::from(p).rgb());
                    let f = rec709.matrix_to(p) * DVec3::from(target.map(f64::from));
                    let product = f * DVec3::from(texel.map(f64::from));
                    (p.matrix_to(rec709) * product).as_vec3().to_array()
                }
            };
            return Ok(());
        }
        let image = |channel| Channel::Image {
            image: r.image,
            channel,
            srgb,
            a: 0.0,
            b: 1.0,
        };
        let id = self.make(
            suffix,
            Pack {
                role,
                encoding: Encoding::Bc7Srgb,
                channels: vec![image(0), image(1), image(2), alpha],
                refs: vec![*r],
                coverage: None,
                toksvig: None,
                normal: false,
                shrink: false,
            },
        )?;
        bind(m, s, id, 0, r);
        Ok(())
    }

    /// A BC4 texture of the reference's channel times `scale`.
    fn single(&mut self, suffix: &str, r: &TexRef, scale: f32, role: &'static str) -> Result<Made> {
        self.make(
            suffix,
            Pack {
                role,
                encoding: Encoding::Bc4,
                channels: vec![Channel::Image {
                    image: r.image,
                    channel: usize::from(r.channel),
                    srgb: false,
                    a: 0.0,
                    b: scale,
                }],
                refs: vec![*r],
                coverage: None,
                toksvig: None,
                normal: false,
                shrink: false,
            },
        )
    }

    /// How pack `p` keeps the blocks of its one source image, and per packed channel the
    /// source channel that holds it; `None` when it needs its texels made. The image's blocks
    /// must be the pack's format (or a BC5 whose half a BC4 takes), with every mip, each
    /// packed channel a source channel as stored. Mip 0 alone passes when the pack makes its
    /// other mips from it (Toksvig roughness, alpha coverage unless `source_mips`) and keeps
    /// its full size.
    fn passes(
        &self,
        p: &Pack,
        size: [u32; 2],
        grid: Option<[u16; 2]>,
        skipped: bool,
        source_mips: bool,
    ) -> Option<(Pass, [u8; 4])> {
        use orr_pack::texture::vk;
        let count = match p.encoding {
            Encoding::Bc7Srgb => 4,
            Encoding::Bc5 => 2,
            Encoding::Bc4 => 1,
            _ => return None,
        };
        if p.shrink || grid.is_some() {
            return None;
        }
        let mut image = None;
        let mut channels = [0u8, 1, 2, 3];
        for (c, ch) in p.channels.iter().take(count).enumerate() {
            match *ch {
                Channel::Image {
                    image: i,
                    channel,
                    srgb,
                    a,
                    b,
                } => {
                    let identity = if p.normal {
                        (a, b) == (-1.0, 2.0)
                    } else {
                        (a, b) == (0.0, 1.0)
                    };
                    // Color channels keep their sRGB encoding; alpha and data are linear.
                    let color = p.encoding == Encoding::Bc7Srgb && c < 3;
                    if !identity || *image.get_or_insert(i) != i || srgb != color {
                        return None;
                    }
                    channels[c] = channel as u8;
                }
                Channel::Unused => {}
                // A color texture's alpha of 1 holds where the source's is 1.
                Channel::Constant(v) if p.encoding == Encoding::Bc7Srgb && c == 3 => {
                    let opaque = self.image(image?).constant[3]
                        .is_some_and(|a| (a - 1.0).abs() <= 1.0 / 255.0);
                    if v != 1.0 || !opaque {
                        return None;
                    }
                }
                Channel::Constant(_) => return None,
            }
        }
        let i = image?;
        let f = self.image(i);
        let (format, complete) = f.blocks?;
        if [f.w, f.h] != size {
            return None;
        }
        let half = match (p.encoding, format) {
            (e, f) if f == e.vk_format() => None,
            (Encoding::Bc4, vk::BC5_UNORM) => Some(channels[0]),
            _ => return None,
        };
        // Slots read color and normal channels together from their first one on.
        let in_order = match p.encoding {
            Encoding::Bc7Srgb => channels[..3] == [0, 1, 2],
            Encoding::Bc5 if p.normal => channels[..2] == [0, 1],
            _ => true,
        };
        let held = match format {
            vk::BC4_UNORM => 1,
            vk::BC5_UNORM => 2,
            _ => 4,
        };
        // A neural set trains on the packed channels in pack order.
        let reordered = channels[..count]
            .iter()
            .enumerate()
            .any(|(c, &s)| usize::from(s) != c);
        if !in_order
            || channels[..count].iter().any(|&c| c >= held)
            || reordered && half.is_none() && self.rules.neural.is_some()
        {
            return None;
        }
        if half.is_some() {
            channels[0] = 0;
        }
        let derived = p.toksvig.is_some() || p.coverage.is_some() && !source_mips;
        if !complete || derived && skipped {
            return None;
        }
        Some((
            Pass {
                image: i,
                all_mips: !derived,
                half,
            },
            channels,
        ))
    }

    /// Plans one pack, or returns the one an identical pack planned before.
    fn make(&mut self, suffix: &str, mut p: Pack) -> Result<Made> {
        let mut images: Vec<usize> = p
            .channels
            .iter()
            .filter_map(|c| match c {
                Channel::Image { image, .. } => Some(*image),
                Channel::Constant(_) | Channel::Unused => None,
            })
            .collect();
        images.sort_unstable();
        images.dedup();
        // Unless mips are remade, channels whose images all hold every mip as blocks at one
        // size keep those mips, without Toksvig roughness or alpha coverage.
        let kept = |i: usize| {
            let f = self.image(i);
            f.blocks.is_some_and(|b| b.1).then_some((f.w, f.h))
        };
        let kept_size = images.first().and_then(|&i| kept(i));
        let source_mips = !crate::quality::remakes_mips(self.quality)
            && !p.shrink
            && kept_size.is_some()
            && images.iter().all(|&i| kept(i) == kept_size)
            && self.overridden(&images, |o| o.atlas_grid).is_none();
        if source_mips {
            p.toksvig = None;
        }
        if let Some((_, r, _)) = &p.toksvig
            && !images.contains(&r.image)
        {
            images.push(r.image);
            images.sort_unstable();
        }
        let max_size = self
            .overridden(&images, |o| o.max_size)
            .or(self.rules.max_size);
        let grid = self.overridden(&images, |o| o.atlas_grid);
        let primaries = self.overridden(&images, |o| o.primaries);
        let bc_split = self
            .overridden(&images, |o| o.bc_split)
            .or(self.rules.bc_split);
        let mut size = [1u32, 1];
        for &i in &images {
            let f = self.image(i);
            size = [size[0].max(f.w), size[1].max(f.h)];
        }
        if p.shrink {
            size = [4, 4];
        }
        let primaries = if p.encoding == Encoding::Bc7Srgb {
            self.color_primaries(images[0])
        } else {
            primaries.unwrap_or(Primaries::Rec709)
        };
        let skipped = super::skipped_mips(size[0], size[1], max_size) > 0;
        let pass = self.passes(&p, size, grid, skipped, source_mips);

        let mut recipe = format!(
            "{} {:?} {:?} {:?} {:?} {:?} {max_size:?} {grid:?} {primaries:?} {bc_split:?} {:?} {:?} {:?} {source_mips}",
            p.role,
            p.encoding,
            p.channels,
            p.coverage,
            p.toksvig.as_ref().map(|t| (t.0, t.1.image, t.2)),
            p.refs[0].wrap,
            self.rules.filter,
            size,
            pass,
        )
        .into_bytes();
        for &i in &images {
            recipe.extend_from_slice(&self.image(i).hash.0);
        }
        let same = Hash32::of(&recipe);
        let channels = pass.map_or([0, 1, 2, 3], |p| p.1);
        if let Some(id) = self.made.get(&same) {
            return Ok(Made { id: *id, channels });
        }
        let name = match suffix {
            "" => format!("{}/texture/{}", self.asset, p.role),
            _ => format!("{}/texture/{suffix}/{}", self.asset, p.role),
        };
        let key = Key::new(Class::Texture, &name)
            .str(&super::encoders(self.quality))
            .bytes(&recipe)
            .finish();
        let id = AssetId::from_name(&name);
        self.made.insert(same, id);
        self.by_id.insert(id, self.jobs.len());
        self.jobs.push(TextureJob {
            name,
            key,
            asset: self.asset.into(),
            pack: p,
            images,
            size,
            quality: self.quality,
            max_size,
            grid,
            primaries,
            filter: self.rules.filter,
            bc_split,
            pass: pass.map(|p| p.0),
            source_mips,
        });
        Ok(Made { id, channels })
    }
}

impl Packer<'_> {
    /// The last value `field` sets among the overrides that match any of `images`.
    fn overridden<T>(
        &self,
        images: &[usize],
        field: impl Fn(&TextureOverride) -> Option<T>,
    ) -> Option<T> {
        self.rules
            .overrides
            .iter()
            .filter(|(pattern, _)| {
                images
                    .iter()
                    .any(|&i| self.images[i].name.contains(pattern.as_str()))
            })
            .fold(None, |v, (_, o)| field(o).or(v))
    }
}

impl TextureJob {
    /// Estimated peak bytes: 8 per compressed source byte for the decoded sources, 48 per
    /// output texel for the packed image and its chain.
    pub(crate) fn working_set(&self, images: &[ImageFile]) -> u64 {
        let sources: u64 = self.images.iter().map(|&i| images[i].len() * 8).sum();
        sources + u64::from(self.size[0]) * u64::from(self.size[1]) * 48
    }

    pub(crate) fn run(&self, images: &[ImageFile]) -> Result<Built> {
        let fail = |e: String| self.fail(format!("texture {}: {e}", self.pack.role));
        let built = match self.pass {
            Some(pass) => self.pass_through(pass, images).map_err(fail)?,
            None => super::finish(vec![self.chain(images)?], &self.plan(None)).map_err(fail)?,
        };
        Ok(built.with_split(self.bc_split))
    }

    /// The texture from the source's blocks: every mip as stored (without those over
    /// `max_size`), or mip 0 as stored and the rest made from it.
    fn pass_through(&self, pass: Pass, images: &[ImageFile]) -> Result<Built, String> {
        let src = {
            let _z = crate::profile::zone!("texture.open");
            let bytes = images[pass.image].bytes().map_err(|e| e.to_string())?;
            load::open(&bytes)?
        };
        let b = src
            .blocks
            .as_ref()
            .ok_or("the image no longer holds blocks")?;
        // A BC5 block is the red BC4 block, then the green one.
        let half = |level: &[u8]| -> Vec<u8> {
            match pass.half {
                Some(h) => {
                    let h = usize::from(h) * 8;
                    level
                        .chunks_exact(16)
                        .flat_map(|b| &b[h..h + 8])
                        .copied()
                        .collect()
                }
                None => level.to_vec(),
            }
        };
        let (size, mips) = if pass.all_mips {
            let (size, kept) = super::kept_mips(b.levels[0].clone(), b.w, b.h, self.max_size);
            (size, kept.iter().map(|m| half(m)).collect())
        } else {
            let first = half(&b.levels[0][0]);
            let (img, plan) = self
                .prepare(images, Some((pass.image, src)))
                .map_err(|e| e.to_string())?;
            // Mip 0 keeps the source's blocks.
            let mut chain = super::chains(vec![img], &plan, 1)?.remove(0);
            let size = [chain[0].w, chain[0].h];
            chain.remove(0);
            let mut mips = super::encode_chains(plan.encoding, plan.quality, &[chain])?.remove(0);
            mips.insert(0, first);
            (size, mips)
        };
        Ok(super::layout(
            self.pack.encoding.vk_format(),
            size,
            false,
            self.primaries,
            self.pack.coverage.map_or(f32::NAN, |c| c.1),
            Vec::new(),
            &[mips],
        ))
    }

    /// The mip chain [`Self::run`] encodes.
    pub(crate) fn chain(&self, images: &[ImageFile]) -> Result<Vec<Img>> {
        if self.source_mips {
            return self.source_chain(images);
        }
        let (img, plan) = self.prepare(images, None)?;
        super::chains(vec![img], &plan, 0)
            .map(|mut c| c.remove(0))
            .map_err(|e| self.fail(format!("texture {}: {e}", self.pack.role)))
    }

    /// Mip 0 size of the cooked texture.
    pub(crate) fn cooked_size(&self) -> [u32; 2] {
        let [w, h] = self.size;
        mip_size(w, h, super::skipped_mips(w, h, self.max_size))
    }

    pub(crate) fn encoding(&self) -> Encoding {
        self.pack.encoding
    }

    /// Whether the texture holds a tangent-space normal map.
    pub(crate) fn is_normal(&self) -> bool {
        self.pack.normal
    }

    fn fail(&self, message: String) -> Error {
        Error::Texture {
            asset: self.asset.clone(),
            message,
        }
    }

    /// The packed mip 0 and how it cooks; `opened` is one source image already opened.
    fn prepare(
        &self,
        images: &[ImageFile],
        mut opened: Option<(usize, Source)>,
    ) -> Result<(Img, Plan)> {
        let _z = crate::profile::zone!("texture.prepare");
        let fail = |message: String| self.fail(message);
        let mut decoded = HashMap::new();
        for &i in &self.images {
            let img = match opened.take_if(|o| o.0 == i) {
                Some((_, mut src)) => {
                    load::decode_layers(&mut src).map_err(&fail)?;
                    src.layers.remove(0)
                }
                None => decode(&images[i].bytes()?).map(|d| d.0).map_err(&fail)?,
            };
            decoded.insert(i, img.resized(self.size[0], self.size[1]));
        }
        let img = self.pack(&decoded, self.size);
        let toksvig = self.pack.toksvig.as_ref().map(|(channel, r, scale)| {
            let mut n = decoded
                .remove(&r.image)
                .expect("the normal map is a source image");
            for t in &mut n.px {
                *t = [
                    (2.0 * t[0] - 1.0) * scale,
                    (2.0 * t[1] - 1.0) * scale,
                    2.0 * t[2] - 1.0,
                    1.0,
                ];
            }
            Toksvig {
                channel: *channel,
                normals: n,
            }
        });
        drop(decoded);
        Ok((img, self.plan(toksvig)))
    }

    /// The packed chain from the source images' own mips, without those over `max_size`.
    fn source_chain(&self, images: &[ImageFile]) -> Result<Vec<Img>> {
        let _z = crate::profile::zone!("texture.source_mips");
        let fail = |message: String| self.fail(message);
        let mut sources = Vec::with_capacity(self.images.len());
        for &i in &self.images {
            let src = load::open(&images[i].bytes()?).map_err(&fail)?;
            let b = src
                .blocks
                .ok_or_else(|| fail("the image no longer holds blocks".into()))?;
            if [b.w, b.h] != self.size || !b.complete() {
                return Err(fail("the image's mips changed".into()));
            }
            sources.push((i, b));
        }
        let [w, h] = self.size;
        (super::skipped_mips(w, h, self.max_size)..full_mip_count(w, h))
            .map(|m| {
                let [mw, mh] = mip_size(w, h, m);
                let mut decoded = HashMap::new();
                for (i, b) in &sources {
                    let img = load::decode_blocks(b.vk_format, mw, mh, &b.levels[0][m as usize])
                        .map_err(&fail)?;
                    decoded.insert(*i, img);
                }
                let mut img = self.pack(&decoded, [mw, mh]);
                if self.pack.normal {
                    img.px.par_iter_mut().for_each(|p| *p = normalized(*p));
                }
                Ok(img)
            })
            .collect()
    }

    /// The packed texture of `decoded` source images, each `w` x `h`.
    fn pack(&self, decoded: &HashMap<usize, Img>, [w, h]: [u32; 2]) -> Img {
        let mut img = Img::new(w, h);
        for (c, ch) in self.pack.channels.iter().enumerate() {
            match *ch {
                Channel::Constant(v) => img.px.par_iter_mut().for_each(|t| t[c] = v),
                Channel::Unused => img.px.par_iter_mut().for_each(|t| t[c] = 1.0),
                Channel::Image {
                    image,
                    channel,
                    srgb,
                    a,
                    b,
                } => {
                    img.px
                        .par_iter_mut()
                        .zip(&decoded[&image].px)
                        .for_each(|(t, s)| {
                            let v = if srgb {
                                srgb_to_linear(s[channel])
                            } else {
                                s[channel]
                            };
                            t[c] = a + b * v;
                        });
                }
            }
        }
        img
    }

    /// How the texture cooks, with Toksvig roughness from `toksvig`.
    fn plan(&self, toksvig: Option<Toksvig>) -> Plan {
        let p = &self.pack;
        Plan {
            encoding: p.encoding,
            cube: false,
            primaries: self.primaries,
            mips: MipSpec {
                filter: self.filter,
                wrap: p.refs[0].wrap,
                grid: self.grid.map(|g| g.map(u32::from)),
                normal: p.normal,
                coverage: p.coverage,
            },
            full_chain: true,
            quality: self.quality,
            max_size: self.max_size,
            toksvig,
            lights: Vec::new(),
        }
    }
}
