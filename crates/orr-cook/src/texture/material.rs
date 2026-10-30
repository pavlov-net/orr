//! A material's canonical channel set: its source images repacked into the textures the
//! renderer samples, with constant images collapsed into factors.
//!
//! - Base colour: BC7 sRGB (RGB). Under `MASK` and `BLEND`, base alpha times the opacity
//!   factor becomes a BC4 opacity texture, with alpha-coverage mips at the cutoff under
//!   `MASK`.
//! - Normal: BC5, with the glTF normal scale applied.
//! - Roughness and metalness share a BC5 (BC4 when only one varies); occlusion gets a BC4.
//!   Factors are folded into these texels and the material's factors set to 1.
//! - Roughness takes the normal map's variance per mip (Toksvig), so a material with a normal
//!   map gets a roughness texture even when its roughness is constant.
//! - Emission and subsurface colour: BC7 sRGB.
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

use std::collections::HashMap;

use orr_pack::material::{Material, TextureBinding, slot};
use orr_pack::{AssetId, Class, Hash32};
use rayon::prelude::*;
use serde::Deserialize;

use super::load::{self, Source};
use super::mips::{MipSpec, Toksvig};
use super::{
    Built, DataFormat, Encoding, Img, MipFilter, Plan, Primaries, Quality, Wrap, srgb_to_linear,
};
use crate::key::Key;
use crate::{Error, Result};

/// Texture settings for images a glTF file references.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureRules {
    pub quality: Quality,
    /// Largest mip 0 edge of every texture.
    pub max_size: Option<u32>,
    pub filter: MipFilter,
    /// Applied in order to every texture made from an image whose name or URI contains the
    /// pattern.
    pub overrides: Vec<(String, TextureOverride)>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureOverride {
    pub quality: Option<Quality>,
    pub max_size: Option<u32>,
    pub atlas_grid: Option<[u16; 2]>,
    pub primaries: Option<Primaries>,
}

/// One image a material reads.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TexRef {
    pub image: usize,
    pub uv_set: u8,
    pub wrap: [Wrap; 2],
}

/// The images one glTF material reads.
#[derive(Clone, Debug, Default)]
pub(crate) struct MaterialTextures {
    pub base_color: Option<TexRef>,
    /// Roughness in G, metalness in B.
    pub metal_rough: Option<TexRef>,
    pub normal: Option<(TexRef, f32)>,
    /// With its strength.
    pub occlusion: Option<(TexRef, f32)>,
    pub emissive: Option<TexRef>,
    /// Specular weight in A.
    pub specular: Option<TexRef>,
    /// Coat weight in R.
    pub coat: Option<TexRef>,
    /// Transmission weight in R.
    pub transmission: Option<TexRef>,
    pub subsurface_color: Option<TexRef>,
    /// Displacement in R.
    pub displacement: Option<TexRef>,
}

impl MaterialTextures {
    /// Every image reference.
    pub(crate) fn refs(&self) -> impl Iterator<Item = &TexRef> {
        [
            self.base_color.as_ref(),
            self.metal_rough.as_ref(),
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
    pub bytes: Vec<u8>,
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
}

/// Decodes a 2D image: its only layer and the file's colour facts.
fn decode(file: &ImageFile) -> Result<(Img, bool, Option<Primaries>), String> {
    let Source {
        mut layers,
        cube,
        srgb,
        primaries,
        ..
    } = load::decode(&file.bytes)?;
    if cube || layers.len() != 1 {
        return Err("not a 2D image".into());
    }
    Ok((layers.remove(0), srgb, primaries))
}

/// Facts of the images `used`, decoded in parallel.
pub(crate) fn image_facts(
    asset: &str,
    images: &[ImageFile],
    used: &[usize],
) -> Result<HashMap<usize, ImageFacts>> {
    used.par_iter()
        .map(|&i| {
            let fail = |message: String| Error::Texture {
                asset: asset.into(),
                message,
            };
            let file = images
                .get(i)
                .ok_or_else(|| fail(format!("image {i} does not exist")))?;
            let (img, srgb, primaries) =
                decode(file).map_err(|e| fail(format!("image {}: {e}", file.name)))?;
            let constant = std::array::from_fn(|c| {
                let (lo, hi) = img.px.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                    (lo.min(p[c]), hi.max(p[c]))
                });
                (hi - lo <= 1.0 / 255.0 + 1e-6).then_some(0.5 * (lo + hi))
            });
            Ok((
                i,
                ImageFacts {
                    w: img.w,
                    h: img.h,
                    srgb,
                    primaries,
                    constant,
                    hash: Hash32::of(&file.bytes),
                },
            ))
        })
        .collect()
}

/// Plans the textures of a glTF file's materials, sharing identical ones. The textures
/// are cooked later by their [`TextureJob`]s.
pub(crate) struct Packer<'a> {
    asset: &'a str,
    images: &'a [ImageFile],
    facts: &'a HashMap<usize, ImageFacts>,
    rules: &'a TextureRules,
    made: HashMap<Hash32, AssetId>,
    pub jobs: Vec<TextureJob>,
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
}

/// One packed texture to cook.
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
}

fn bind(m: &mut Material, s: usize, id: AssetId, channel: u8, r: &TexRef) {
    m.textures[s] = TextureBinding {
        texture: id,
        channels: channel,
        uv_set: r.uv_set,
        _pad: [0; 2],
    };
}

impl<'a> Packer<'a> {
    pub(crate) fn new(
        asset: &'a str,
        images: &'a [ImageFile],
        facts: &'a HashMap<usize, ImageFacts>,
        rules: &'a TextureRules,
    ) -> Self {
        Self {
            asset,
            images,
            facts,
            rules,
            made: HashMap::new(),
            jobs: Vec::new(),
        }
    }

    fn image(&self, i: usize) -> &ImageFacts {
        &self.facts[&i]
    }

    /// The value of `channel` if every texel lies within 1/255 of one value.
    fn constant(&self, i: usize, channel: usize) -> Option<f32> {
        self.image(i).constant[channel]
    }

    /// Cooks the material's textures, binds them and folds constants into its factors.
    pub(crate) fn material(
        &mut self,
        suffix: &str,
        m: &mut Material,
        t: &MaterialTextures,
    ) -> Result<()> {
        if let Some(r) = &t.base_color {
            let srgb = self.image(r.image).srgb;
            let rgb: Vec<Option<f32>> = (0..3).map(|c| self.constant(r.image, c)).collect();
            if rgb.iter().all(Option::is_some) {
                for (c, v) in rgb.into_iter().enumerate() {
                    let v = v.expect("constant");
                    m.base_color[c] *= if srgb { srgb_to_linear(v) } else { v };
                }
            } else {
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
                        role: "base_color",
                        encoding: Encoding::Bc7Srgb,
                        channels: vec![image(0), image(1), image(2), Channel::Constant(1.0)],
                        refs: vec![*r],
                        coverage: None,
                        toksvig: None,
                        normal: false,
                        shrink: false,
                    },
                )?;
                bind(m, slot::BASE_COLOR, id, 0, r);
            }
            if m.alpha_mode != Material::OPAQUE {
                match self.constant(r.image, 3) {
                    Some(a) => m.geometry_opacity *= a,
                    None => {
                        let coverage =
                            (m.alpha_mode == Material::MASK).then_some((0, m.alpha_cutoff));
                        let id = self.make(
                            suffix,
                            Pack {
                                role: "opacity",
                                encoding: Encoding::Bc4,
                                channels: vec![Channel::Image {
                                    image: r.image,
                                    channel: 3,
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
                            channel(0, *scale),
                            channel(1, *scale),
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
                if consts.iter().any(Option::is_none) {
                    normal_packed = Some((*r, *scale));
                }
            }
        }

        // Roughness, metalness, occlusion; specular weight has no glTF core source.
        let mr = t.metal_rough;
        let (rough_const, metal_const) = match &mr {
            Some(r) => (self.constant(r.image, 1), self.constant(r.image, 2)),
            None => (Some(1.0), Some(1.0)),
        };
        let rough_varies = rough_const.is_none() || normal_packed.is_some();
        let metal_varies = metal_const.is_none();
        let occ = t.occlusion.map(|(r, s)| (r, s, self.constant(r.image, 0)));
        let occ_varies = occ.as_ref().is_some_and(|o| o.2.is_none());
        let rough_channel = |m: &Material| match (&mr, rough_const) {
            (Some(r), None) => Channel::Image {
                image: r.image,
                channel: 1,
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
            channel: 0,
            srgb: false,
            a: 1.0 - o.1,
            b: o.1,
        };
        let metal_channel = |m: &Material, r: &TexRef| Channel::Image {
            image: r.image,
            channel: 2,
            srgb: false,
            a: 0.0,
            b: m.metalness,
        };
        let mut rm = Vec::new();
        let mut refs: Vec<TexRef> = Vec::new();
        if rough_varies {
            rm.push(("roughness", slot::ROUGHNESS, rough_channel(m)));
            refs.extend(mr);
        }
        if metal_varies {
            let r = mr.expect("varying metalness has an image");
            rm.push(("metalness", slot::METALNESS, metal_channel(m, &r)));
            if refs.is_empty() {
                refs.push(r);
            }
        }
        let normal_ref = normal_packed.as_ref().map(|n| n.0);
        if let Some(n) = normal_ref
            && refs.is_empty()
        {
            refs.push(n);
        }
        if !rm.is_empty() {
            let (encoding, role) = match rm.len() {
                1 => (Encoding::Bc4, rm[0].0),
                _ => (Encoding::Bc5, "roughness_metalness"),
            };
            let mut channels: Vec<Channel> = rm.iter().map(|c| c.2).collect();
            channels.resize(4, Channel::Constant(0.0));
            let toksvig = match (&normal_packed, rough_varies) {
                (Some((r, s)), true) => Some((0, *r, *s)),
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
            let uv = refs[0];
            for (i, (_, s, _)) in rm.iter().enumerate() {
                bind(m, *s, id, i as u8, &uv);
            }
            if rough_varies {
                m.specular_roughness = 1.0;
            }
            if metal_varies {
                m.metalness = 1.0;
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
            self.colour(suffix, m, r, "emission", slot::EMISSION, |m| {
                &mut m.emission_color
            })?;
        }
        if let Some(r) = &t.subsurface_color {
            self.colour(
                suffix,
                m,
                r,
                "subsurface_color",
                slot::SUBSURFACE_COLOR,
                |m| &mut m.subsurface_color,
            )?;
        }
        if let Some(r) = &t.specular {
            match self.constant(r.image, 3) {
                Some(v) => m.specular_weight *= v,
                None => {
                    let id = self.single(suffix, r, 3, 1.0, "specular_weight")?;
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
            *weight(m) = match self.constant(r.image, 0) {
                Some(v) => factor * v,
                None => {
                    let id = self.single(suffix, r, 0, factor, role)?;
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
                        channel: 0,
                        srgb: false,
                        a: 0.0,
                        b: 1.0,
                    }],
                    refs: vec![*r],
                    coverage: None,
                    toksvig: None,
                    normal: false,
                    shrink: self.constant(r.image, 0).is_some(),
                },
            )?;
            bind(m, slot::DISPLACEMENT, id, 0, r);
        }
        Ok(())
    }

    /// A colour texture whose value multiplies the material colour of slot `s`; a constant
    /// image folds into that colour.
    fn colour(
        &mut self,
        suffix: &str,
        m: &mut Material,
        r: &TexRef,
        role: &'static str,
        s: usize,
        colour: fn(&mut Material) -> &mut [f32; 3],
    ) -> Result<()> {
        let srgb = self.image(r.image).srgb;
        let rgb: Vec<Option<f32>> = (0..3).map(|c| self.constant(r.image, c)).collect();
        if rgb.iter().all(Option::is_some) {
            let target = colour(m);
            for (c, v) in rgb.into_iter().enumerate() {
                let v = v.expect("constant");
                target[c] *= if srgb { srgb_to_linear(v) } else { v };
            }
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
                channels: vec![image(0), image(1), image(2), Channel::Constant(1.0)],
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

    /// A BC4 texture of one source channel times `scale`.
    fn single(
        &mut self,
        suffix: &str,
        r: &TexRef,
        channel: usize,
        scale: f32,
        role: &'static str,
    ) -> Result<AssetId> {
        self.make(
            suffix,
            Pack {
                role,
                encoding: Encoding::Bc4,
                channels: vec![Channel::Image {
                    image: r.image,
                    channel,
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

    /// Plans one pack, or returns the id of an identical one planned before.
    fn make(&mut self, suffix: &str, p: Pack) -> Result<AssetId> {
        let mut images: Vec<usize> = p
            .channels
            .iter()
            .filter_map(|c| match c {
                Channel::Image { image, .. } => Some(*image),
                Channel::Constant(_) => None,
            })
            .collect();
        if let Some((_, r, _)) = &p.toksvig {
            images.push(r.image);
        }
        images.sort_unstable();
        images.dedup();
        let mut quality = self.rules.quality;
        let mut max_size = self.rules.max_size;
        let mut grid = None;
        let mut primaries = None;
        for (pattern, o) in &self.rules.overrides {
            if images
                .iter()
                .any(|&i| self.images[i].name.contains(pattern.as_str()))
            {
                quality = o.quality.unwrap_or(quality);
                max_size = o.max_size.or(max_size);
                grid = o.atlas_grid.or(grid);
                primaries = o.primaries.or(primaries);
            }
        }
        let mut size = [1u32, 1];
        for &i in &images {
            let f = self.image(i);
            size = [size[0].max(f.w), size[1].max(f.h)];
        }
        if p.shrink {
            size = [4, 4];
        }
        let colour = p.encoding == Encoding::Bc7Srgb;
        let primaries = primaries
            .or(if colour {
                self.image(images[0]).primaries
            } else {
                None
            })
            .unwrap_or(Primaries::Rec709);

        let mut recipe = format!(
            "{} {:?} {:?} {:?} {:?} {:?} {quality:?} {max_size:?} {grid:?} {primaries:?} {:?} {:?}",
            p.role,
            p.encoding,
            p.channels,
            p.coverage,
            p.toksvig.as_ref().map(|t| (t.0, t.1.image, t.2)),
            p.refs[0].wrap,
            self.rules.filter,
            size,
        )
        .into_bytes();
        for &i in &images {
            recipe.extend_from_slice(&self.image(i).hash.0);
        }
        let same = Hash32::of(&recipe);
        if let Some(id) = self.made.get(&same) {
            return Ok(*id);
        }
        let name = format!("{}/texture/{suffix}/{}", self.asset, p.role);
        let key = Key::new(Class::Texture, &name)
            .str(super::ENCODERS)
            .bytes(&recipe)
            .finish();
        let id = AssetId::from_name(&name);
        self.made.insert(same, id);
        self.jobs.push(TextureJob {
            name,
            key,
            asset: self.asset.into(),
            pack: p,
            images,
            size,
            quality,
            max_size,
            grid,
            primaries,
            filter: self.rules.filter,
        });
        Ok(id)
    }
}

impl TextureJob {
    /// Estimated peak bytes: 8 per compressed source byte for the decoded sources, 48 per
    /// output texel for the packed image and its chain.
    pub(crate) fn working_set(&self, images: &[ImageFile]) -> u64 {
        let sources: u64 = self
            .images
            .iter()
            .map(|&i| images[i].bytes.len() as u64 * 8)
            .sum();
        sources + u64::from(self.size[0]) * u64::from(self.size[1]) * 48
    }

    pub(crate) fn run(&self, images: &[ImageFile]) -> Result<Built> {
        let fail = |message: String| Error::Texture {
            asset: self.asset.clone(),
            message,
        };
        let mut decoded = HashMap::new();
        for &i in &self.images {
            let (img, ..) =
                decode(&images[i]).map_err(|e| fail(format!("image {}: {e}", images[i].name)))?;
            decoded.insert(i, img.resized(self.size[0], self.size[1]));
        }
        let p = &self.pack;
        let mut img = Img::new(self.size[0], self.size[1]);
        for (c, ch) in p.channels.iter().enumerate() {
            match *ch {
                Channel::Constant(v) => img.px.iter_mut().for_each(|t| t[c] = v),
                Channel::Image {
                    image,
                    channel,
                    srgb,
                    a,
                    b,
                } => {
                    for (t, s) in img.px.iter_mut().zip(&decoded[&image].px) {
                        let v = if srgb {
                            srgb_to_linear(s[channel])
                        } else {
                            s[channel]
                        };
                        t[c] = a + b * v;
                    }
                }
            }
        }
        let toksvig = p.toksvig.as_ref().map(|(channel, r, scale)| {
            let mut n = decoded[&r.image].clone();
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
        let plan = Plan {
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
            quality: self.quality,
            max_size: self.max_size,
            toksvig,
            lights: Vec::new(),
        };
        super::build(vec![img], &plan).map_err(|e| fail(format!("texture {}: {e}", p.role)))
    }
}
