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
//! - Emission: BC7 sRGB.
//! - An image whose texels all lie within 1/255 of one value collapses into the material's
//!   factor; a constant normal or occlusion image, which has no factor, cooks at 4x4.
//!
//! A bound texture's value multiplies the matching factor; normal, occlusion and opacity
//! textures have no factor.

use std::collections::HashMap;

use orr_pack::material::{Material, TextureBinding, slot};
use orr_pack::{ArchiveWriter, AssetId, Class, Hash32};

use super::load::{self, Source};
use super::mips::{MipSpec, Toksvig};
use super::{Encoding, Img, MipFilter, Plan, Primaries, Quality, Wrap, srgb_to_linear};
use crate::{Error, Result};

/// Texture settings for images a glTF file references.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextureRules {
    pub quality: Quality,
    /// Largest mip 0 edge of every texture.
    pub max_size: Option<u32>,
    pub filter: MipFilter,
    /// Applied in order to every texture made from an image whose name or URI contains the
    /// pattern.
    pub overrides: Vec<(String, TextureOverride)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
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
}

pub(crate) struct ImageFile {
    /// glTF image name, else its URI, else its index.
    pub name: String,
    pub bytes: Vec<u8>,
}

struct Decoded {
    img: Img,
    srgb: bool,
    primaries: Option<Primaries>,
    hash: Hash32,
}

/// Cooks the textures of a glTF file's materials, sharing identical ones.
pub(crate) struct Packer<'a> {
    asset: &'a str,
    images: &'a [ImageFile],
    rules: &'a TextureRules,
    decoded: HashMap<usize, Decoded>,
    made: HashMap<Hash32, AssetId>,
}

/// A texture to make from one or more source channels.
struct Pack<'p> {
    role: &'static str,
    encoding: Encoding,
    /// Channel sources in output order.
    channels: Vec<Channel>,
    refs: Vec<TexRef>,
    coverage: Option<(usize, f32)>,
    toksvig: Option<(usize, &'p TexRef, f32)>,
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

impl<'a> Packer<'a> {
    pub(crate) fn new(asset: &'a str, images: &'a [ImageFile], rules: &'a TextureRules) -> Self {
        Self {
            asset,
            images,
            rules,
            decoded: HashMap::new(),
            made: HashMap::new(),
        }
    }

    fn fail(&self, message: String) -> Error {
        Error::Texture {
            asset: self.asset.into(),
            message,
        }
    }

    fn image(&mut self, i: usize) -> Result<&Decoded> {
        if !self.decoded.contains_key(&i) {
            let file = self
                .images
                .get(i)
                .ok_or_else(|| self.fail(format!("image {i} does not exist")))?;
            let Source {
                mut layers,
                cube,
                srgb,
                primaries,
                ..
            } = load::decode(&file.bytes)
                .map_err(|e| self.fail(format!("image {}: {e}", file.name)))?;
            if cube || layers.len() != 1 {
                return Err(self.fail(format!("image {}: not a 2D image", file.name)));
            }
            let d = Decoded {
                img: layers.remove(0),
                srgb,
                primaries,
                hash: Hash32::of(&file.bytes),
            };
            self.decoded.insert(i, d);
        }
        Ok(&self.decoded[&i])
    }

    /// The value of `channel` if every texel lies within 1/255 of one value.
    fn constant(&mut self, i: usize, channel: usize) -> Result<Option<f32>> {
        let img = &self.image(i)?.img;
        let (lo, hi) = img.px.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
            (lo.min(p[channel]), hi.max(p[channel]))
        });
        Ok((hi - lo <= 1.0 / 255.0 + 1e-6).then_some(0.5 * (lo + hi)))
    }

    /// Cooks the material's textures, binds them and folds constants into its factors.
    pub(crate) fn material(
        &mut self,
        w: &mut ArchiveWriter,
        suffix: &str,
        m: &mut Material,
        t: &MaterialTextures,
    ) -> Result<()> {
        let bind = |m: &mut Material, s: usize, id: AssetId, channel: u8, r: &TexRef| {
            m.textures[s] = TextureBinding {
                texture: id,
                channels: channel,
                uv_set: r.uv_set,
                _pad: [0; 2],
            };
        };

        if let Some(r) = &t.base_color {
            let srgb = self.image(r.image)?.srgb;
            let rgb: Vec<Option<f32>> = (0..3)
                .map(|c| self.constant(r.image, c))
                .collect::<Result<_>>()?;
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
                    w,
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
                match self.constant(r.image, 3)? {
                    Some(a) => m.geometry_opacity *= a,
                    None => {
                        let coverage =
                            (m.alpha_mode == Material::MASK).then_some((0, m.alpha_cutoff));
                        let id = self.make(
                            w,
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
            let consts: Vec<Option<f32>> = (0..3)
                .map(|c| self.constant(r.image, c))
                .collect::<Result<_>>()?;
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
                    w,
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
            Some(r) => (self.constant(r.image, 1)?, self.constant(r.image, 2)?),
            None => (Some(1.0), Some(1.0)),
        };
        let rough_varies = rough_const.is_none() || normal_packed.is_some();
        let metal_varies = metal_const.is_none();
        let occ = match &t.occlusion {
            Some((r, s)) => Some((*r, *s, self.constant(r.image, 0)?)),
            None => None,
        };
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
                (Some((r, s)), true) => Some((0, r, *s)),
                _ => None,
            };
            let id = self.make(
                w,
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
                w,
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
            let srgb = self.image(r.image)?.srgb;
            let rgb: Vec<Option<f32>> = (0..3)
                .map(|c| self.constant(r.image, c))
                .collect::<Result<_>>()?;
            if rgb.iter().all(Option::is_some) {
                for (c, v) in rgb.into_iter().enumerate() {
                    let v = v.expect("constant");
                    m.emission_color[c] *= if srgb { srgb_to_linear(v) } else { v };
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
                    w,
                    suffix,
                    Pack {
                        role: "emission",
                        encoding: Encoding::Bc7Srgb,
                        channels: vec![image(0), image(1), image(2), Channel::Constant(1.0)],
                        refs: vec![*r],
                        coverage: None,
                        toksvig: None,
                        normal: false,
                        shrink: false,
                    },
                )?;
                bind(m, slot::EMISSION, id, 0, r);
            }
        }
        Ok(())
    }

    /// Cooks one pack, or returns the id of an identical one cooked before.
    fn make(&mut self, w: &mut ArchiveWriter, suffix: &str, p: Pack) -> Result<AssetId> {
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
            let img = &self.image(i)?.img;
            size = [size[0].max(img.w), size[1].max(img.h)];
        }
        if p.shrink {
            size = [4, 4];
        }

        let mut key = format!(
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
            key.extend_from_slice(&self.image(i)?.hash.0);
        }
        key.extend_from_slice(super::encoder_key_base());
        let key = crate::cook_key(Class::Texture, &[&key]);
        if let Some(id) = self.made.get(&key) {
            return Ok(*id);
        }

        let mut img = Img::new(size[0], size[1]);
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
                    let src = self.image(image)?.img.resized(size[0], size[1]);
                    for (t, s) in img.px.iter_mut().zip(&src.px) {
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
        let toksvig = match &p.toksvig {
            Some((channel, r, scale)) => {
                let mut n = self.image(r.image)?.img.resized(size[0], size[1]);
                for t in &mut n.px {
                    *t = [
                        (2.0 * t[0] - 1.0) * scale,
                        (2.0 * t[1] - 1.0) * scale,
                        2.0 * t[2] - 1.0,
                        1.0,
                    ];
                }
                Some(Toksvig {
                    channel: *channel,
                    normals: n,
                })
            }
            None => None,
        };
        let colour = p.encoding == Encoding::Bc7Srgb;
        let primaries = primaries
            .or(if colour {
                self.image(images[0])?.primaries
            } else {
                None
            })
            .unwrap_or(Primaries::Rec709);
        let plan = Plan {
            encoding: p.encoding,
            cube: false,
            primaries,
            mips: MipSpec {
                filter: self.rules.filter,
                wrap: p.refs[0].wrap,
                grid: grid.map(|g| g.map(u32::from)),
                normal: p.normal,
                coverage: p.coverage,
            },
            quality,
            max_size,
            toksvig,
            lights: Vec::new(),
        };
        let built = super::build(vec![img], &plan)
            .map_err(|e| self.fail(format!("texture {}: {e}", p.role)))?;
        let id = super::add(
            w,
            &format!("{}/texture/{suffix}/{}", self.asset, p.role),
            key,
            &built,
        )?;
        self.made.insert(key, id);
        Ok(id)
    }
}
