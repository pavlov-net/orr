//! A material's canonical channel set: its source images repacked into the textures the
//! renderer samples, with constant images collapsed into factors.
//!
//! - Base color: BC7 sRGB (RGB). Under `MASK` and `BLEND`, base alpha times the opacity
//!   factor becomes a BC4 opacity texture, with alpha-coverage mips at the cutoff under
//!   `MASK` unless it keeps its source's mips.
//! - Normal: BC5, with the glTF normal scale applied.
//! - Roughness and metalness share a BC5 (BC4 when only one varies). Factors are folded into
//!   these texels and the material's factors set to 1.
//! - Roughness takes the normal map's variance per mip (Toksvig), so a material with a normal
//!   map gets a roughness texture even when its roughness is constant. At the quality levels
//!   that keep source mips, roughness that keeps them, and roughness beside a normal map
//!   that keeps them, takes none.
//! - Emission and subsurface color: BC7 sRGB. Emission that can read past 1 is BC6H, as
//!   [`Packer::past_one`] says.
//! - Specular weight (the specular texture's alpha), coat and transmission weights (red):
//!   BC4 each. Coat and transmission factors are folded into their texels and set to 1;
//!   the specular weight texture multiplies `specular_weight`. A specular weight that can read
//!   past 1 is R16_SFLOAT.
//! - Displacement: R16 unorm from red, never collapsed; displacement is
//!   `displacement_scale x texel + displacement_offset`. A height `h` stores across the
//!   range `lo..hi` the image measures, as `(h - lo) / (hi - lo)`, with the scale and offset
//!   compensated; a range past 0..1 widens the declared one. Heights along u and v cook alike,
//!   each from its own channel, into [`Material::displacement_tangent`].
//! - An image whose texels all lie within 1/255 of one value collapses into the material's
//!   factor, clamped to the input's range as a texture stores it; a constant normal image, which
//!   has no factor, cooks at 4x4. Under `MASK` an opacity image collapses only when its texels
//!   fall on one side of the cutoff. An image read past 0..1 as something else ([`Wrap::Border`],
//!   [`Wrap::Constant`]) never collapses.
//! - A slot's wrap and filter go to its binding, as does what it reads past 0..1: the image's
//!   transparent black or stated value through the same factors and remaps as its texels.
//!   Under [`Wrap::Border`] a texture that reads anything but black there takes
//!   [`Wrap::Constant`], which the shader checks: a hardware border holds only black.
//!
//! - A set of tiles ([`ImageData::Tiles`]) cooks each tile into a texture of its own, at its own
//!   size and with its own mips, packed as an image would be, under what the whole set decides:
//!   the format, the primaries, the scale and offset of heights. A
//!   [`TextureTiles`](orr_pack::tiles::TextureTiles) asset holds them, and the slot binds that.
//!   Two inputs share a texture, or pair for Toksvig roughness, only with tiles at the same cells.
//!   A token picks the tile, so no point lies past a tile's edge: tiles clamp whatever wrap the
//!   input states, and mips filter each tile alone. What the input reads where there is no tile
//!   goes to the binding, and a set never collapses into a factor.
//!
//! - The slots from [`slot::MORE`] on take their inputs from [`MaterialTextures::more`]: a weight
//!   as a coat weight is, a value that can read past 1 as a specular weight is, a color as
//!   subsurface color is, the coat normal as the normal is but for Toksvig roughness, and a
//!   direction as BC5 of its first two channels.
//!
//! A bound texture's value multiplies the matching factor; normal and opacity textures have no
//! factor.
//!
//! No importer reads occlusion: a path tracer computes it, and glTF and USD say to ignore the
//! texture there.

use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use ctt::Quality;
use orr_color::RgbPrimaries;
use orr_math::{DMat3, DVec3};
use orr_pack::material::{Material, TextureBinding, slot};
use orr_pack::texture::{full_mip_count, mip_size, normal};
use orr_pack::tiles::{TextureTiles, Tile};
use orr_pack::{AssetId, Class, Hash32};
use rayon::prelude::*;
use serde::Deserialize;

use super::color::{BELOW_ZERO, Decode, FileColor, ImageColor, Read, clipped, multiply};
use super::load::{self, Blocks, Gray, Source};
use super::mips::{MipSpec, Toksvig, normalized, settled};
use super::neural::{self, NeuralJob, NeuralSettings};
use super::tiles::Cell;
use super::{
    Built, DataFormat, Encoding, Img, MipFilter, NormalSource, Plan, Primaries, Wrap,
    stores_normals,
};
use crate::decoded::Decoded;
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
    /// pattern. A tile of a set matches by its own name or the set's.
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
    pub specular: Option<u8>,
    pub coat: Option<u8>,
    pub transmission: Option<u8>,
    pub displacement: Option<u8>,
}

/// Whose rules read an image: formats differ on what a file's channels and values mean.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ImageRules {
    Gltf,
    Usd,
    MaterialX,
}

/// One image a material reads.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TexRef {
    pub(crate) image: usize,
    pub(crate) rules: ImageRules,
    /// The source channel of a scalar input; color inputs read RGB.
    pub(crate) channel: u8,
    /// The channel read instead from an image that holds only red and green.
    pub(crate) rg_channel: Option<u8>,
    /// The input whose channel overrides set [`Self::channel`], which each tile of a set takes up
    /// by its own name.
    pub(crate) input: Option<fn(&InputChannels) -> Option<u8>>,
    pub(crate) uv_set: u8,
    pub(crate) wrap: [Wrap; 2],
    /// The axes whose [`Self::wrap`] gives way to what the file's metadata states.
    pub(crate) file_wrap: [bool; 2],
    /// `TextureBinding::filter` bits.
    pub(crate) filter: u8,
    /// What each channel of the image reads past 0..1 under [`Wrap::Border`] and
    /// [`Wrap::Constant`], as its texels decode: colors linear Rec.709.
    pub(crate) outside: [f32; 4],
    /// What each channel reads where the image can't be read, as a linear value: the `fallback`
    /// or `default` of the input.
    pub(crate) missing: [f32; 4],
    /// The uv transform the texture is sampled with (`Material::uv_transforms` form).
    pub(crate) uv: [f32; 6],
    /// What the importer's rules say the image's texels are.
    pub(crate) color: ImageColor,
    /// Per axis (a scalar input's being the first), `[a, b]` of the `a + b * texel` the input
    /// reads, below any factor; a normal map reads it in place of its encoding.
    pub(crate) remap: Option<[[f32; 2]; 3]>,
}

impl TexRef {
    /// What the input reads of `texel` on `axis`.
    fn read(&self, axis: usize, texel: f32) -> f32 {
        self.remap
            .map_or(texel, |m| m[axis][0] + m[axis][1] * texel)
    }

    /// `[a, b]` of `a + b * read`, with `read` this input's value on `axis`, of a source value.
    fn through(&self, axis: usize, a: f32, b: f32) -> [f32; 2] {
        let [ra, rb] = self.remap.map_or([0.0, 1.0], |m| m[axis]);
        [a + b * ra, b * rb]
    }

    /// The source channel `channel` as axis `axis` of this input, under `a + b * value`.
    fn source(&self, axis: usize, channel: usize, read: Read, a: f32, b: f32) -> Texels {
        let [a, b] = self.through(axis, a, b);
        Texels::Image {
            image: self.image,
            channel,
            read,
            a,
            b,
        }
    }

    /// The normal in texel `t`, with [`Packer::normal_source`]'s remap, of an image laid out as
    /// `source` says.
    fn normal(&self, source: NormalSource, t: [f32; 4]) -> [f32; 3] {
        let value = |c: usize| source.decode.value(t[c]);
        let [x, y] = [self.read(0, value(0)), self.read(1, value(source.y))];
        let z = if source.rg {
            normal::z(x, y)
        } else {
            self.read(2, value(source.z))
        };
        [x, y, z]
    }

    /// Whether both read their textures at the same uvs, past the edges too.
    fn samples_like(&self, o: &TexRef) -> bool {
        self.uv_set == o.uv_set && self.uv == o.uv && self.wrap == o.wrap
    }

    /// Whether reads past 0..1 leave the image, so that an image of one value isn't one value
    /// everywhere.
    fn bounded(&self) -> bool {
        self.wrap.iter().any(|w| w.bounded())
    }
}

/// The images one material reads, per input.
#[derive(Clone, Debug, Default)]
pub(crate) struct MaterialTextures {
    pub(crate) base_color: Option<TexRef>,
    /// Read under `MASK` and `BLEND` only.
    pub(crate) opacity: Option<TexRef>,
    pub(crate) roughness: Option<TexRef>,
    pub(crate) metalness: Option<TexRef>,
    /// With its scale per tangent axis; a negative scale flips the axis.
    pub(crate) normal: Option<(TexRef, [f32; 2])>,
    pub(crate) emissive: Option<TexRef>,
    pub(crate) specular: Option<TexRef>,
    pub(crate) coat: Option<TexRef>,
    pub(crate) transmission: Option<TexRef>,
    pub(crate) subsurface_color: Option<TexRef>,
    pub(crate) displacement: Option<TexRef>,
    /// With its scale per tangent axis, as `normal`.
    pub(crate) coat_normal: Option<(TexRef, [f32; 2])>,
    /// The inputs of the slots from [`slot::MORE`] on, by slot, but for the coat normal.
    pub(crate) more: BTreeMap<usize, Option<TexRef>>,
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
            self.emissive.as_ref(),
            self.specular.as_ref(),
            self.coat.as_ref(),
            self.transmission.as_ref(),
            self.subsurface_color.as_ref(),
            self.displacement.as_ref(),
            self.coat_normal.as_ref().map(|n| &n.0),
        ]
        .into_iter()
        .flatten()
        .chain(self.more.values().flatten())
    }

    fn refs_mut(&mut self) -> impl Iterator<Item = &mut TexRef> {
        [
            self.base_color.as_mut(),
            self.opacity.as_mut(),
            self.roughness.as_mut(),
            self.metalness.as_mut(),
            self.normal.as_mut().map(|n| &mut n.0),
            self.emissive.as_mut(),
            self.specular.as_mut(),
            self.coat.as_mut(),
            self.transmission.as_mut(),
            self.subsurface_color.as_mut(),
            self.displacement.as_mut(),
            self.coat_normal.as_mut().map(|n| &mut n.0),
        ]
        .into_iter()
        .flatten()
        .chain(self.more.values_mut().flatten())
    }

    /// The references of color inputs.
    pub(crate) fn colors(&self) -> impl Iterator<Item = &TexRef> {
        let more = self
            .more
            .iter()
            .filter(|(s, _)| More::of(**s) == More::Color);
        [&self.base_color, &self.emissive, &self.subsurface_color]
            .into_iter()
            .flatten()
            .chain(more.flat_map(|(_, r)| r))
    }

    /// The images of normal maps that [`ImageFacts::cant_read`].
    pub(crate) fn unscaled_normals<'a>(
        &'a self,
        facts: &'a HashMap<usize, ImageFacts>,
    ) -> impl Iterator<Item = usize> + 'a {
        let maps = [&self.normal, &self.coat_normal].into_iter().flatten();
        maps.filter(|(r, _)| facts[&r.image].cant_read(r))
            .map(|(r, _)| r.image)
    }
}

/// How the input of a slot from [`slot::MORE`] on packs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum More {
    /// A value of 0..1, with its role, whose factor folds into the texels.
    Weight(&'static str),
    /// A value from 0 up, with its role, which multiplies the field.
    Ranged(&'static str),
    /// A value of -1..1, with its role, stored at `0.5 v + 0.5` in the field's place.
    Signed(&'static str),
    Color,
    /// The first two channels of a direction, with its role.
    Direction(&'static str),
    /// A height along tangent axis 0 or 1, with its role.
    Height(usize, &'static str),
}

impl More {
    fn of(s: usize) -> Self {
        match s {
            slot::SPECULAR_COLOR
            | slot::TRANSMISSION_COLOR
            | slot::TRANSMISSION_SCATTER
            | slot::SUBSURFACE_RADIUS_SCALE
            | slot::FUZZ_COLOR
            | slot::COAT_COLOR => Self::Color,
            slot::SPECULAR_IOR => Self::Ranged("specular_ior"),
            slot::COAT_IOR => Self::Ranged("coat_ior"),
            slot::THIN_FILM_IOR => Self::Ranged("thin_film_ior"),
            slot::THIN_FILM_THICKNESS => Self::Ranged("thin_film_thickness"),
            slot::SUBSURFACE_RADIUS => Self::Ranged("subsurface_radius"),
            slot::EMISSION_LUMINANCE => Self::Ranged("emission_luminance"),
            slot::TRANSMISSION_DEPTH => Self::Ranged("transmission_depth"),
            slot::DISPERSION_ABBE => Self::Ranged("dispersion_abbe"),
            slot::SUBSURFACE_ANISOTROPY => Self::Signed("subsurface_anisotropy"),
            slot::TRANSMISSION_SCATTER_ANISOTROPY => Self::Signed("transmission_anisotropy"),
            slot::TANGENT => Self::Direction("tangent"),
            slot::COAT_TANGENT => Self::Direction("coat_tangent"),
            slot::DISPLACEMENT_U => Self::Height(0, "displacement_u"),
            slot::DISPLACEMENT_V => Self::Height(1, "displacement_v"),
            slot::BASE_WEIGHT => Self::Weight("base_weight"),
            slot::DISPERSION_SCALE => Self::Weight("dispersion_scale"),
            slot::DIFFUSE_ROUGHNESS => Self::Weight("diffuse_roughness"),
            slot::ANISOTROPY => Self::Weight("anisotropy"),
            slot::SUBSURFACE_WEIGHT => Self::Weight("subsurface_weight"),
            slot::FUZZ_WEIGHT => Self::Weight("fuzz_weight"),
            slot::FUZZ_ROUGHNESS => Self::Weight("fuzz_roughness"),
            slot::COAT_ROUGHNESS => Self::Weight("coat_roughness"),
            slot::COAT_DARKENING => Self::Weight("coat_darkening"),
            slot::COAT_ANISOTROPY => Self::Weight("coat_anisotropy"),
            slot::THIN_FILM_WEIGHT => Self::Weight("thin_film_weight"),
            _ => unreachable!("slot {s} is no input of `more`"),
        }
    }
}

/// The field a scalar slot from [`slot::MORE`] on multiplies.
pub(crate) fn scalar_field(m: &mut Material, s: usize) -> &mut f32 {
    match s {
        slot::SPECULAR_IOR => &mut m.specular_ior,
        slot::COAT_IOR => &mut m.coat_ior,
        slot::THIN_FILM_IOR => &mut m.thin_film_ior,
        slot::THIN_FILM_THICKNESS => &mut m.thin_film_thickness,
        slot::SUBSURFACE_RADIUS => &mut m.subsurface_radius,
        slot::EMISSION_LUMINANCE => &mut m.emission_luminance,
        slot::TRANSMISSION_DEPTH => &mut m.transmission_depth,
        slot::DISPERSION_ABBE => &mut m.transmission_dispersion_abbe_number,
        slot::DISPERSION_SCALE => &mut m.transmission_dispersion_scale,
        slot::TRANSMISSION_SCATTER_ANISOTROPY => &mut m.transmission_scatter_anisotropy,
        slot::SUBSURFACE_ANISOTROPY => &mut m.subsurface_scatter_anisotropy,
        slot::BASE_WEIGHT => &mut m.base_weight,
        slot::DIFFUSE_ROUGHNESS => &mut m.diffuse_roughness,
        slot::ANISOTROPY => &mut m.specular_anisotropy,
        slot::SUBSURFACE_WEIGHT => &mut m.subsurface_weight,
        slot::FUZZ_WEIGHT => &mut m.fuzz_weight,
        slot::FUZZ_ROUGHNESS => &mut m.fuzz_roughness,
        slot::COAT_ROUGHNESS => &mut m.coat_roughness,
        slot::COAT_DARKENING => &mut m.coat_darkening,
        slot::COAT_ANISOTROPY => &mut m.coat_anisotropy,
        slot::THIN_FILM_WEIGHT => &mut m.thin_film_weight,
        _ => unreachable!("slot {s} is no scalar"),
    }
}

pub(crate) struct ImageFile {
    /// glTF image name, else its URI, else its index.
    pub(crate) name: String,
    pub(crate) data: ImageData,
}

/// What an image reads of its file.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Layer {
    /// All of it.
    Whole,
    /// The layer of this name, and a file without one reads as a file that can't be read.
    Named(String),
    /// Layer `default`, else layer `rgba`, else all of it.
    Implied,
}

pub(crate) enum ImageData {
    Bytes(Vec<u8>),
    /// A file, read when needed. [`image_facts`] reads it first and records its hash and
    /// length, which guard the later reads.
    File {
        path: PathBuf,
        read: Option<(Hash32, u64)>,
        layer: Layer,
    },
    /// No file to read, and why.
    Unread(String),
    /// One texel of these values.
    Solid([f32; 4]),
    /// The images of a file name that names one per unit square of uv space, in cell order.
    Tiles(Vec<ImageTile>),
}

/// One tile of a set: an image of the scene and its cell.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ImageTile {
    pub(crate) cell: Cell,
    pub(crate) image: usize,
}

impl ImageFile {
    /// An image that reads as the input's fallback, for why it has no file.
    pub(crate) fn unread(name: String, why: String) -> Self {
        Self {
            name,
            data: ImageData::Unread(why),
        }
    }

    /// The tiles of a set.
    pub(crate) fn tiles(&self) -> Option<&[ImageTile]> {
        match &self.data {
            ImageData::Tiles(t) => Some(t),
            _ => None,
        }
    }

    /// The file's bytes; a file must not have changed since its facts were read.
    fn bytes(&self) -> Result<Cow<'_, [u8]>> {
        match &self.data {
            ImageData::Bytes(b) => Ok(Cow::Borrowed(b)),
            ImageData::File {
                path,
                read: Some((hash, _)),
                layer,
            } => {
                let bytes = state::read_checked(path, *hash)?;
                let layer = load::layer_of(&bytes, layer).map_err(|message| Error::Texture {
                    asset: self.name.clone(),
                    message,
                })?;
                Ok(Cow::Owned(layer.unwrap_or(bytes)))
            }
            ImageData::File { read: None, .. }
            | ImageData::Unread(_)
            | ImageData::Solid(_)
            | ImageData::Tiles(_) => {
                unreachable!("only an image with facts from bytes reads them")
            }
        }
    }

    fn len(&self) -> u64 {
        match &self.data {
            ImageData::Bytes(b) => b.len() as u64,
            ImageData::File { read, .. } => read.map_or(0, |r| r.1),
            ImageData::Unread(_) | ImageData::Solid(_) | ImageData::Tiles(_) => 0,
        }
    }

    /// The image's only layer.
    fn decoded(&self) -> Result<Img, String> {
        match self.data {
            ImageData::Solid(texel) => Ok(Img {
                w: 1,
                h: 1,
                px: vec![texel],
            }),
            _ => decode(&self.bytes().map_err(|e| e.to_string())?),
        }
    }
}

/// What packing needs to know of one source image; its texels are decoded again by the
/// texture jobs that read them.
pub(crate) struct ImageFacts {
    w: u32,
    h: u32,
    color: FileColor,
    /// Per channel, the value every texel lies within 1/255 of, if there is one.
    constant: [Option<f32>; 4],
    /// Per channel, the lowest and highest value of mip 0, or of every mip the file stores of a
    /// color that can read below 0. Only a float image or one read with all of its range is sure
    /// to have all of it: reading blocks stops once every channel varies.
    range: [(f32, f32); 4],
    /// A color channel holds negative infinity, which [`Self::range`] leaves out: it has no color
    /// to convert, and clips.
    minus_infinity: bool,
    hash: Hash32,
    /// The block format of a block-compressed file, and whether it has every mip.
    blocks: Option<(u32, bool)>,
    /// The file holds two channels, red and green.
    rg: bool,
    /// The file stores signed normalized values.
    signed: bool,
    /// The file stores floats.
    float: bool,
    /// The file's format can hold values below 0.
    negatives: bool,
    /// Read as a normal map, the file holds normals as they are, not at `0.5 v + 0.5`, with what
    /// its format says or, under glTF, a float image's vectors.
    stored_normals: bool,
    gray: Gray,
    /// The file is animated; only its first frame is read.
    pub(crate) animated: bool,
    /// What the file's metadata says it reads past 0..1, per axis.
    wrap: Option<[Wrap; 2]>,
}

impl ImageFacts {
    /// The facts of a texel of float values.
    fn solid(texel: [f32; 4]) -> Self {
        Self {
            w: 1,
            h: 1,
            color: FileColor::default(),
            constant: texel.map(Some),
            range: texel.map(|v| (v, v)),
            minus_infinity: false,
            hash: Hash32::of(bytemuck::cast_slice(&texel)),
            blocks: None,
            rg: false,
            signed: false,
            float: true,
            negatives: true,
            stored_normals: false,
            gray: Gray::No,
            animated: false,
            wrap: None,
        }
    }

    /// What a set of tiles holds in all, for what the whole set decides. Nothing collapses, so
    /// it has no constant, and its color and values are its tiles' to read.
    fn of_tiles<'a>(tiles: impl Iterator<Item = &'a Self> + Clone) -> Self {
        let all = |f: fn(&Self) -> bool| tiles.clone().all(f);
        let any = |f: fn(&Self) -> bool| tiles.clone().any(f);
        Self {
            w: tiles.clone().map(|t| t.w).max().unwrap_or(1),
            h: tiles.clone().map(|t| t.h).max().unwrap_or(1),
            color: FileColor::default(),
            constant: [None; 4],
            range: [(0.0, 0.0); 4],
            minus_infinity: any(|t| t.minus_infinity),
            hash: Hash32::of(&tiles.clone().flat_map(|t| t.hash.0).collect::<Vec<u8>>()),
            blocks: None,
            rg: all(|t| t.rg),
            signed: all(|t| t.signed),
            float: any(|t| t.float),
            negatives: all(|t| t.negatives),
            stored_normals: all(|t| t.stored_normals),
            gray: Gray::No,
            animated: false,
            wrap: None,
        }
    }

    /// Whether normal map `r` is to read as it stands where this image can't hold the signed
    /// vectors that would mean.
    fn cant_read(&self, r: &TexRef) -> bool {
        !self.negatives && r.remap == Some([[0.0, 1.0]; 3])
    }
}

/// Decodes a 2D image: its only layer.
fn decode(bytes: &[u8]) -> Result<Img, String> {
    let Source {
        mut layers, cube, ..
    } = load::decode(bytes)?;
    if cube || layers.len() != 1 {
        return Err("not a 2D image".into());
    }
    Ok(layers.remove(0))
}

/// How a scene reads its images, by index.
pub(crate) struct ImageReads {
    /// Read with all of their range.
    pub(crate) whole: HashSet<usize>,
    /// Normal maps that say nothing of their encoding.
    pub(crate) normals: HashSet<usize>,
    /// Read as color.
    pub(crate) colors: HashSet<usize>,
    /// Their inputs state what to read where they can't be read.
    pub(crate) optional: HashSet<usize>,
}

/// Facts of the images `used`, read as `reads` says, in parallel, and why each optional image that
/// can't be read can't; any other fails. File images are read through the memo, and each one's
/// hash and length are recorded in its [`ImageData::File`].
pub(crate) fn image_facts(
    asset: &str,
    images: &mut [ImageFile],
    used: &[usize],
    reads: &ImageReads,
    memo: &Memo,
) -> Result<(HashMap<usize, ImageFacts>, BTreeMap<usize, String>)> {
    type Outcome = (
        std::result::Result<ImageFacts, String>,
        Option<(Hash32, u64)>,
    );
    let fail = |message: String| Error::Texture {
        asset: asset.into(),
        message,
    };
    if let Some(&i) = used.iter().find(|&&i| i >= images.len()) {
        return Err(fail(format!("image {i} does not exist")));
    }
    let read: HashMap<usize, Outcome> = used
        .par_iter()
        .map(|&i| {
            let file = &images[i];
            let (bytes, hash, read) = match &file.data {
                ImageData::Bytes(b) => (Cow::Borrowed(b.as_slice()), Hash32::of(b), None),
                ImageData::File { path, layer, .. } => match memo.read(path) {
                    Ok((bytes, hash)) => {
                        let read = Some((hash, bytes.len() as u64));
                        match load::layer_of(&bytes, layer) {
                            Ok(None) => (Cow::Owned(bytes), hash, read),
                            Ok(Some(layer)) => {
                                let hash = Hash32::of(&layer);
                                (Cow::Owned(layer), hash, read)
                            }
                            Err(why) => return Ok((i, (Err(why), read))),
                        }
                    }
                    Err(e) if reads.optional.contains(&i) => {
                        return Ok((i, (Err(e.to_string()), None)));
                    }
                    Err(e) => return Err(e),
                },
                ImageData::Unread(why) => return Ok((i, (Err(why.clone()), None))),
                &ImageData::Solid(texel) => return Ok((i, (Ok(ImageFacts::solid(texel)), None))),
                ImageData::Tiles(_) => unreachable!("a set's tiles are read in its place"),
            };
            let uses = [&reads.whole, &reads.normals, &reads.colors].map(|s| s.contains(&i));
            let facts = facts_of(&bytes, hash, uses);
            Ok((i, (facts, read)))
        })
        .collect::<Result<_>>()?;
    let mut out = HashMap::with_capacity(read.len());
    let mut unread = BTreeMap::new();
    for (i, (facts, file)) in read {
        if let (ImageData::File { read, .. }, Some(file)) = (&mut images[i].data, file) {
            *read = Some(file);
        }
        match facts {
            Ok(f) => {
                out.insert(i, f);
            }
            Err(why) if reads.optional.contains(&i) => {
                unread.insert(i, why);
            }
            Err(e) => return Err(fail(format!("image {}: {e}", images[i].name))),
        }
    }
    Ok((out, unread))
}

/// Drops the tiles of `unread` from the sets among `sets` and gives each set its facts; a set left
/// with no tile joins `unread`.
pub(crate) fn settle_tiles(
    sets: &[usize],
    images: &mut [ImageFile],
    facts: &mut HashMap<usize, ImageFacts>,
    unread: &mut BTreeMap<usize, String>,
) {
    for &set in sets {
        let ImageData::Tiles(tiles) = &mut images[set].data else {
            continue;
        };
        tiles.retain(|t| !unread.contains_key(&t.image));
        if tiles.is_empty() {
            unread.insert(set, "it has no tile that can be read".into());
        } else {
            let merged = ImageFacts::of_tiles(tiles.iter().map(|t| &facts[&t.image]));
            facts.insert(set, merged);
        }
    }
}

/// Points every reference of `textures` to an image of `unread` at a solid image of what it
/// reads in its place: the input's fallback, everywhere and as it stands. `solids` shares them
/// by value.
pub(crate) fn stand_in(
    textures: &mut MaterialTextures,
    unread: &BTreeMap<usize, String>,
    images: &mut Vec<ImageFile>,
    facts: &mut HashMap<usize, ImageFacts>,
    solids: &mut HashMap<[u32; 4], usize>,
) {
    for r in textures
        .refs_mut()
        .filter(|r| unread.contains_key(&r.image))
    {
        let image = *solids
            .entry(r.missing.map(f32::to_bits))
            .or_insert_with(|| {
                images.push(ImageFile {
                    name: format!("fallback/{:?}", r.missing),
                    data: ImageData::Solid(r.missing),
                });
                facts.insert(images.len() - 1, ImageFacts::solid(r.missing));
                images.len() - 1
            });
        *r = TexRef {
            image,
            rg_channel: None,
            wrap: [Wrap::Repeat; 2],
            file_wrap: [false; 2],
            outside: [0.0; 4],
            color: ImageColor::Data,
            ..*r
        };
    }
}

/// The facts of one image file's bytes, with all of its range if it is read `whole`, how it
/// stores normals if it is a `normal` map, and what a `color` reads below 0 in any mip it stores.
fn facts_of(
    bytes: &[u8],
    hash: Hash32,
    [whole, normal, color]: [bool; 3],
) -> Result<ImageFacts, String> {
    let _z = crate::profile::zone!("texture.facts");
    // A channel is constant when it spans at most one UNORM8 step.
    const SPREAD: f32 = 1.0 / 255.0 + 1e-6;
    let spread = if whole { f32::INFINITY } else { SPREAD };
    let constant = |range: [(f32, f32); 4]| {
        range.map(|(lo, hi)| (hi - lo <= SPREAD).then_some(0.5 * (lo + hi)))
    };
    let every_mip = |negatives, f: &FileColor| color && (whole || negatives || f.can_read_below());
    if let Some(scan) = load::scan_ktx2(bytes, spread) {
        let mut s = scan?;
        let mut opened = None;
        if every_mip(s.negatives, &s.color) {
            let src = load::open(bytes)?;
            s.range = load::stored_range(&src)?;
            opened = Some(src);
        }
        let vectors = normal && s.float && s.negatives;
        let stored_normals = s.signed
            || (vectors && {
                let src = match opened {
                    Some(mut src) => {
                        load::decode_layers(&mut src)?;
                        src
                    }
                    None => load::decode(bytes)?,
                };
                stores_normals(&src.layers, s.rg)
            });
        return Ok(ImageFacts {
            w: s.w,
            h: s.h,
            color: s.color,
            constant: constant(s.range),
            range: s.range,
            minus_infinity: false,
            hash,
            blocks: Some((s.vk_format, s.complete)),
            rg: s.rg,
            signed: s.signed,
            float: s.float,
            negatives: s.negatives,
            stored_normals,
            gray: s.gray,
            animated: false,
            wrap: None,
        });
    }
    let mut src = load::open(bytes)?;
    if src.cube || src.layers.len() > 1 || src.blocks.as_ref().is_some_and(|b| b.levels.len() != 1)
    {
        return Err("not a 2D image".into());
    }
    let vectors = normal && src.float && src.negatives;
    if vectors {
        load::decode_layers(&mut src)?;
    }
    let (w, h, range) = match &src.blocks {
        Some(b) if every_mip(src.negatives, &src.color) => (b.w, b.h, load::stored_range(&src)?),
        Some(b) => {
            let range = load::block_range(b.vk_format, b.w, b.h, &mut &b.levels[0][0][..], spread)?;
            (b.w, b.h, range)
        }
        None => {
            let img = &src.layers[0];
            // An infinite texel stores at an end of the range, and must not make it infinite.
            (img.w, img.h, std::array::from_fn(|c| img.finite_range(c)))
        }
    };
    let mut texels = src.layers.iter().flat_map(|l| &l.px);
    let minus_infinity = src.float && texels.any(|p| p[..3].contains(&f32::NEG_INFINITY));
    Ok(ImageFacts {
        w,
        h,
        color: src.color,
        constant: constant(range),
        range,
        minus_infinity,
        hash,
        blocks: src.blocks.as_ref().map(|b| (b.vk_format, b.complete())),
        rg: src.rg,
        signed: src.signed,
        float: src.float,
        negatives: src.negatives,
        stored_normals: src.signed || (vectors && stores_normals(&src.layers, src.rg)),
        gray: src.gray,
        animated: load::is_animated(bytes),
        wrap: load::exr_wrap(bytes),
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
    pub(crate) jobs: Vec<TextureJob>,
    /// Job index of each planned texture.
    by_id: HashMap<AssetId, usize>,
    /// Member textures of each planned neural set.
    sets: HashSet<Vec<AssetId>>,
    pub(crate) neural: Vec<NeuralJob>,
    pub(crate) tile_sets: Vec<TileSetJob>,
    /// Tiles of each planned tile set, and which set holds those tiles.
    tiles_of: HashMap<AssetId, Vec<Tile>>,
    made_sets: HashMap<Hash32, AssetId>,
    /// The set of each tile image, whose name overrides match as well as the tile's.
    set_of: HashMap<usize, usize>,
}

/// A planned [`TextureTiles`] asset.
pub(crate) struct TileSetJob {
    pub(crate) name: String,
    pub(crate) key: Hash32,
    pub(crate) bytes: Vec<u8>,
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
    /// Output channel, normal map, its scale per tangent axis and how it stores normals.
    toksvig: Option<(usize, TexRef, [f32; 2], NormalSource)>,
    normal: bool,
    shrink: bool,
    /// The cell of a tile, and the primaries its set shares, for a color tile.
    tile: Option<(Cell, Option<Primaries>)>,
}

/// One output channel.
#[derive(Clone, Debug)]
struct Channel {
    texels: Texels,
    /// What it reads past 0..1 under [`Wrap::Border`] and [`Wrap::Constant`].
    outside: Outside,
}

/// A channel's reading past 0..1: source value `value`, read as a texel is, `a + b * value`.
#[derive(Copy, Clone, Debug)]
struct Outside {
    value: f32,
    a: f32,
    b: f32,
}

impl Outside {
    /// The value `v` itself.
    const fn of(v: f32) -> Self {
        Self {
            value: 0.0,
            a: v,
            b: 0.0,
        }
    }
}

impl Channel {
    const UNUSED: Self = Self {
        texels: Texels::Unused,
        outside: Outside::of(0.0),
    };

    /// The value `v`, past 0..1 too.
    fn constant(v: f32) -> Self {
        Self {
            texels: Texels::Constant(v),
            outside: Outside::of(v),
        }
    }
}

/// Source value `v` of a channel of `a + b * value`, under [`Pack::limit`] `limit`.
fn stored(v: f32, a: f32, b: f32, limit: Option<f32>) -> f32 {
    let read = a + b * limit.map_or(v, |l| v.clamp(0.0, l));
    if limit == Some(f32::INFINITY) {
        read.max(0.0)
    } else {
        read
    }
}

/// Where one output channel's texels come from.
#[derive(Clone, Debug)]
enum Texels {
    /// Source channel of an image, `a + b * value`, the value as `read` has it.
    Image {
        image: usize,
        channel: usize,
        read: Read,
        a: f32,
        b: f32,
    },
    /// The z of a two-channel normal image, which holds none, rebuilt from its x and y.
    /// With `encoded`, at `0.5 z + 0.5`.
    NormalZ {
        encoded: bool,
        image: usize,
        source: NormalSource,
    },
    Constant(f32),
    /// Never read: stored as 1, or as the source has it where the blocks pass through.
    Unused,
}

/// One packed texture to cook.
#[derive(Clone)]
pub(crate) struct TextureJob {
    pub(crate) name: String,
    pub(crate) key: Hash32,
    asset: String,
    pack: Pack,
    /// Source images, sorted, with their content hashes.
    images: Vec<(usize, Hash32)>,
    size: [u32; 2],
    quality: Quality,
    max_size: Option<u32>,
    grid: Option<[u16; 2]>,
    primaries: Primaries,
    /// Takes linear colors into `primaries`, where the source's have no id.
    gamut: Option<DMat3>,
    filter: MipFilter,
    bc_split: Option<bool>,
    /// Keeps the source's blocks.
    pass: Option<Pass>,
    /// Encodes the source images' own mips instead of making them from mip 0.
    source_mips: bool,
}

impl Pack {
    /// Where a source value stops. Apart from normals, BC4 and BC5 hold weights, and a float
    /// weight has no upper limit: a source texel clamps before its factor, as a constant does. A
    /// float weight's remap can reach below 0 where no format of 0..1 would store it, and a
    /// negative weight has no IOR.
    fn limit(&self) -> Option<f32> {
        match self.encoding {
            Encoding::Bc4 | Encoding::Bc5 if !self.normal => Some(1.0),
            Encoding::Raw(DataFormat::R16f) => Some(f32::INFINITY),
            _ => None,
        }
    }

    /// What the texture reads past 0..1 per packed channel, as its texels sample: filtered
    /// values stay in the format's range, and a normal is stored x and y.
    fn outside(&self) -> [f32; 4] {
        let limit = self.limit();
        let mut v = [0.0; 4];
        for (o, c) in v.iter_mut().zip(&self.channels) {
            let Outside { value, a, b } = c.outside;
            *o = stored(value, a, b, limit);
        }
        if self.normal {
            let [x, y, ..] = normalized(v);
            return [normal::encode(x), normal::encode(y), 0.0, 0.0];
        }
        let (lo, hi) = self.encoding.range();
        v.map(|c| settled(c, lo, hi))
    }
}

/// A planned texture, per packed channel the channel of the texture that holds it, what it
/// reads past 0..1 and how it gets there.
#[derive(Copy, Clone, Debug)]
struct Made {
    id: AssetId,
    channels: [u8; 4],
    wrap: [Wrap; 2],
    outside: [f32; 4],
    /// It is a set of tiles, which read `outside` where they lie on no tile.
    tiled: bool,
    /// Keeps source mips in place of remaking them.
    source_mips: bool,
}

/// The name of the texture, or tile set, of `role` in a material.
fn texture_name(asset: &str, suffix: &str, role: &str) -> String {
    match suffix {
        "" => format!("{asset}/texture/{role}"),
        _ => format!("{asset}/texture/{suffix}/{role}"),
    }
}

/// The span from the lowest to the highest of `ranges`.
fn widest(ranges: impl Iterator<Item = (f32, f32)>) -> (f32, f32) {
    ranges.fold((f32::INFINITY, f32::NEG_INFINITY), |a, b| {
        (a.0.min(b.0), a.1.max(b.1))
    })
}

/// Binds slot `s` to packed channel `channel` of `t`, sampled as `r` is.
fn bind(m: &mut Material, s: usize, t: Made, channel: usize, r: &TexRef) {
    let mut outside = [0.0; 4];
    if t.tiled || t.wrap.contains(&Wrap::Constant) {
        outside[..4 - channel].copy_from_slice(&t.outside[channel..]);
    }
    m.textures[s] = TextureBinding {
        texture: t.id,
        channels: t.channels[channel],
        uv_set: r.uv_set,
        wrap: t.wrap.map(Wrap::binding),
        filter: r.filter,
        vertex: m.textures[s].vertex,
        vertex_complement: m.textures[s].vertex_complement,
        _pad: [0; 1],
        outside,
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
            tile_sets: Vec::new(),
            tiles_of: HashMap::new(),
            made_sets: HashMap::new(),
            set_of: images
                .iter()
                .enumerate()
                .flat_map(|(s, f)| f.tiles().into_iter().flatten().map(move |t| (t.image, s)))
                .collect(),
        }
    }

    fn image(&self, i: usize) -> &ImageFacts {
        &self.facts[&i]
    }

    /// The cell of each tile `r` reads, in order, or `None` for an image.
    fn cells(&self, r: &TexRef) -> Option<Vec<Cell>> {
        let tiles = self.images[r.image].tiles()?;
        Some(tiles.iter().map(|t| t.cell).collect())
    }

    /// Whether `r` reads a set of tiles.
    fn tiled(&self, r: &TexRef) -> bool {
        self.images[r.image].tiles().is_some()
    }

    /// `r` as it reads the tile at `cell`, or `r` where it reads no set.
    fn at(&self, r: &TexRef, cell: Option<Cell>) -> TexRef {
        let (Some(cell), Some(tiles)) = (cell, self.images[r.image].tiles()) else {
            return *r;
        };
        let tile = tiles.binary_search_by_key(&[cell[1], cell[0]], |t| [t.cell[1], t.cell[0]]);
        let tile = tile.expect("those sharing a texture hold the same cells");
        self.tile(r, &tiles[tile])
    }

    /// `r`, of a set, as it reads `tile`.
    fn tile(&self, r: &TexRef, tile: &ImageTile) -> TexRef {
        let image = tile.image;
        self.channel_override(TexRef { image, ..*r })
    }

    /// `r`, or each tile of the set it reads.
    fn each(&self, r: &TexRef) -> Vec<TexRef> {
        match self.images[r.image].tiles() {
            Some(tiles) => tiles.iter().map(|t| self.tile(r, t)).collect(),
            None => vec![*r],
        }
    }

    /// Channel `c` of `r` as its rules read the file: a channel of the decoded image, or the
    /// value of one the file doesn't hold. MaterialX reads a file's N channels as they are, zero
    /// past them and 1 in the fourth. USD repeats the first of one or two over RGB, with the second
    /// as alpha, and glTF repeats one.
    fn read(&self, r: &TexRef, c: usize) -> Result<usize, f32> {
        let f = self.image(r.image);
        match (r.rules, f.gray, c) {
            (_, _, 0) => Ok(0),
            (ImageRules::MaterialX, Gray::WithAlpha, 1) => Ok(3),
            (ImageRules::MaterialX, Gray::Alone | Gray::WithAlpha, 3) => Err(1.0),
            (ImageRules::MaterialX, Gray::Alone | Gray::WithAlpha, _) => Err(0.0),
            (ImageRules::MaterialX, ..) => Ok(c),
            (_, Gray::Red, 1 | 2) => Ok(0),
            (ImageRules::Usd, ..) if f.rg => Ok(if c == 3 { 1 } else { 0 }),
            _ => Ok(c),
        }
    }

    /// How `r` reads a number from its image: an input that isn't a color takes no primaries.
    fn decode(&self, r: &TexRef) -> Decode {
        r.color.decode(&self.image(r.image).color)
    }

    /// Stored value `v` of channel `c` of `r`, as [`Self::read`] numbers it, as an input that
    /// isn't a color reads it. Alpha is no color, whichever channel of the file holds it.
    fn number(&self, r: &TexRef, c: usize, v: f32) -> f32 {
        if c == 3 { v } else { self.decode(r).value(v) }
    }

    /// [`TexRef::source`] of channel `c`, as [`Self::read`] has it, through `read`.
    fn read_source(
        &self,
        r: &TexRef,
        axis: usize,
        c: usize,
        read: Read,
        a: f32,
        b: f32,
    ) -> Channel {
        let texels = match self.read(r, c) {
            Ok(held) if c == 3 => r.source(axis, held, Read::of(Decode::STORED), a, b),
            Ok(held) => r.source(axis, held, read, a, b),
            Err(v) => Texels::Constant(a + b * r.read(axis, v)),
        };
        let [a, b] = r.through(axis, a, b);
        let value = r.outside[c];
        let outside = Outside { value, a, b };
        Channel { texels, outside }
    }

    /// [`Self::read_source`] of an input that isn't a color.
    fn source(&self, r: &TexRef, axis: usize, c: usize, a: f32, b: f32) -> Channel {
        self.read_source(r, axis, c, Read::of(self.decode(r)), a, b)
    }

    /// The lowest and highest value of channel `c` of `r`.
    fn range(&self, r: &TexRef, c: usize) -> (f32, f32) {
        if self.tiled(r) {
            return widest(self.each(r).into_iter().map(|t| self.range(&t, c)));
        }
        match self.read(r, c) {
            Ok(held) => {
                let (lo, hi) = self.image(r.image).range[held];
                (self.number(r, c, lo), self.number(r, c, hi))
            }
            Err(v) => (v, v),
        }
    }

    /// The lowest and highest value scalar input `r` reads, each tile of a set through its own
    /// channel.
    fn input_range(&self, r: &TexRef) -> (f32, f32) {
        let each = self.each(r).into_iter();
        widest(each.map(|t| self.range(&t, usize::from(t.channel))))
    }

    /// Whether some reads of `r` leave its texels: past the edges, or past a set's tiles.
    fn bounded(&self, r: &TexRef) -> bool {
        r.bounded() || self.tiled(r)
    }

    /// The stored value of channel `c` of `r` if every texel lies within 1/255 of one value, and
    /// nothing else reads past them.
    fn stored_constant(&self, r: &TexRef, c: usize) -> Option<f32> {
        if self.bounded(r) {
            return None;
        }
        match self.read(r, c) {
            Ok(c) => self.image(r.image).constant[c],
            Err(v) => Some(v),
        }
    }

    /// [`Self::stored_constant`] as an input that isn't a color reads it.
    fn constant(&self, r: &TexRef, c: usize) -> Option<f32> {
        let v = self.stored_constant(r, c)?;
        Some(match self.read(r, c) {
            Ok(_) => self.number(r, c, v),
            Err(_) => v,
        })
    }

    /// Value `v` of `r` as a texture up to `top` stores it. A stated value is stored nowhere, and
    /// stands whole, as a constant input does.
    fn held(&self, r: &TexRef, v: f32, top: f32) -> f32 {
        match self.images[r.image].data {
            ImageData::Solid(_) => v,
            _ => v.clamp(0.0, top),
        }
    }

    /// [`Self::constant`] of an input with a 0..1 meaning, clamped as its texture stores it.
    fn constant_weight(&self, r: &TexRef, c: usize) -> Option<f32> {
        self.constant(r, c).map(|v| self.held(r, v, 1.0))
    }

    /// Whether emission or specular weight `r` can read past 1: USD and MaterialX range them over
    /// 0 and up, where glTF's stop at 1, and a float image, a remap past it or a value stated past
    /// the edges reaches there. A color's remap stops at 1, short of rounding, with its field
    /// holding the rest.
    fn past_one(&self, r: &TexRef) -> bool {
        let reaches = |[a, b]: [f32; 2]| a.max(a + b) > 1.0 + 1e-6;
        let outside = (self.tiled(r) || r.wrap.contains(&Wrap::Constant))
            && (0..3).any(|c| r.read(c, r.outside[c]) > 1.0 + 1e-6);
        r.rules != ImageRules::Gltf
            && (outside
                || r.remap.is_some_and(|m| m.into_iter().any(reaches))
                || self
                    .each(r)
                    .iter()
                    .any(|t| self.image(t.image).float || self.decode(t).value(1.0) > 1.0 + 1e-6))
    }

    /// [`Self::constant`] of an input [`Self::past_one`], from 0 up, with a float image's texels
    /// also within 1/255 of their own size: no 8-bit step limits them, and a dim image can be
    /// multiplied by any luminance.
    fn constant_ranged(&self, r: &TexRef, c: usize) -> Option<f32> {
        let (lo, hi) = self.range(r, c);
        let held = !self.image(r.image).float || hi - lo <= hi.abs() / 255.0;
        self.constant(r, c).filter(|_| held).map(|v| v.max(0.0))
    }

    /// [`Self::constant_weight`] of the channel of scalar input `r`, as the input reads it.
    fn constant_input(&self, r: &TexRef) -> Option<f32> {
        self.constant_weight(r, usize::from(r.channel))
            .map(|v| self.held(r, r.read(0, v), 1.0))
    }

    /// [`Self::constant_input`] of opacity `r` of material `m`. A folded opacity tests the same
    /// for the whole surface, so under `MASK` texels either side of the cutoff stay a texture.
    fn constant_opacity(&self, m: &Material, r: &TexRef) -> Option<f32> {
        let (lo, hi) = self.input_range(r);
        let read = |t: f32| self.held(r, r.read(0, self.held(r, t, 1.0)), 1.0);
        let passes = |t: f32| m.geometry_opacity * read(t) >= m.alpha_cutoff;
        let one_side = m.alpha_mode != Material::MASK || passes(lo) == passes(hi);
        self.constant_input(r).filter(|_| one_side)
    }

    /// Normal map `r` with the remap it reads through, and its layout: what its material states
    /// where the image can hold that. A signed image holds the vectors themselves, so a pair that
    /// centers 0..1 at 0 doesn't apply to it, and one with no negatives can't, so reading it as it
    /// stands doesn't either.
    fn normal_source(&self, r: &TexRef) -> (TexRef, NormalSource) {
        const ENCODED: [[f32; 2]; 3] = [[-1.0, 2.0]; 3];
        let f = self.image(r.image);
        let m = match r.remap {
            None if f.stored_normals => [[0.0, 1.0]; 3],
            None => ENCODED,
            Some(_) if f.cant_read(r) => ENCODED,
            Some(m) if f.signed => m.map(|[a, b]| {
                if a + 0.5 * b == 0.0 {
                    [0.0, 0.5 * b]
                } else {
                    [a, b]
                }
            }),
            Some(m) => m,
        };
        let r = self.file_channels(*r);
        // A channel the file doesn't hold reads as the same decode of its fill value.
        let remap = Some(std::array::from_fn(|c| match self.read(&r, c) {
            Ok(_) => m[c],
            Err(v) => [m[c][0] + m[c][1] * v, 0.0],
        }));
        let source = NormalSource {
            rg: f.rg,
            signed: f.stored_normals,
            y: self.read(&r, 1).unwrap_or(1),
            z: self.read(&r, 2).unwrap_or(2),
            decode: self.decode(&r),
        };
        (TexRef { remap, ..r }, source)
    }

    /// `r` reading the channels a two-channel file holds, not what USD's outputs make of them:
    /// a normal's x and y, or a channel an override names.
    fn file_channels(&self, r: TexRef) -> TexRef {
        let rules = match r.rules {
            ImageRules::Usd if self.image(r.image).rg => ImageRules::MaterialX,
            rules => rules,
        };
        TexRef { rules, ..r }
    }

    /// How color input `r` reads, and the id and matrix of what it reads as: what its rules say
    /// of its image, under the last override that sets its primaries.
    fn color_read(&self, r: &TexRef) -> (Read, Primaries, Option<DMat3>) {
        let over = self.overridden(&[r.image], |o| o.primaries);
        let color = &self.image(r.image).color;
        r.color.color(color, over, |read| self.reads_below(r, read))
    }

    /// The primaries a texture of color input `r` holds: a slot has one id, so tiles that read as
    /// different ones all take Rec. 2020, which covers them.
    fn slot_primaries(&self, r: &TexRef) -> Primaries {
        let mut each = self.each(r).into_iter().map(|t| self.color_read(&t).1);
        let first = each.next().expect("a set has a tile");
        if each.all(|p| p == first) {
            first
        } else {
            Primaries::Rec2020
        }
    }

    /// Whether color input `r`, through `read`, holds a value below 0, in the primaries of its file
    /// and through its remap.
    fn reads_below(&self, r: &TexRef, read: &Read) -> bool {
        let f = self.image(r.image);
        (0..3).any(|c| match self.read(r, c) {
            Ok(held) => {
                let (lo, hi) = f.range[held];
                let below = |&v: &f32| r.read(c, read.alone(v)) < BELOW_ZERO;
                lo <= hi && [lo, hi].iter().any(below)
            }
            Err(v) => r.read(c, v) < BELOW_ZERO,
        })
    }

    /// Whether any texel of `img` still clips once color input `r` has read and converted it.
    fn texels_clip(&self, r: &TexRef, img: &Img) -> bool {
        let (read, _, gamut) = self.color_read(r);
        let channels = [0, 1, 2].map(|c| self.read(r, c));
        img.px.par_iter().any(|t| {
            let reach = read.reach(*t);
            let rgb = std::array::from_fn(|c| {
                r.read(c, channels[c].map_or_else(|v| v, |held| reach[held]))
            });
            clipped(gamut.as_ref().map_or(rgb, |m| multiply(m, rgb)))
        })
    }

    /// The images of `materials`' color inputs with colors a texture can't hold, measured through
    /// what they convert to. An image is decoded here only if a value below 0 or primaries past
    /// Rec.2020 can clip.
    pub(crate) fn clipped<'m>(
        &self,
        materials: impl Iterator<Item = &'m MaterialTextures>,
    ) -> Result<BTreeSet<usize>> {
        // Per file, the image the report names, a tile's being its set, and how it reads.
        let mut reads: BTreeMap<usize, (usize, Vec<TexRef>)> = BTreeMap::new();
        // A stated value has no texels, and stands whole.
        let filed = |r: &&TexRef| !matches!(self.images[r.image].data, ImageData::Solid(_));
        for r in materials.flat_map(MaterialTextures::colors).filter(filed) {
            for t in self.each(r) {
                let (read, _, gamut) = self.color_read(&t);
                let below = self.image(t.image).minus_infinity || self.reads_below(&t, &read);
                if below || read.reaches_past(gamut.as_ref()) {
                    let (_, refs) = reads.entry(t.image).or_insert((r.image, Vec::new()));
                    let same =
                        |o: &TexRef| (o.color, o.rules, o.remap) == (t.color, t.rules, t.remap);
                    if !refs.iter().any(same) {
                        refs.push(t);
                    }
                }
            }
        }
        let reads: Vec<_> = reads.into_iter().collect();
        let found: Vec<Option<usize>> = reads
            .par_iter()
            .map(|(i, (named, refs))| {
                let file = &self.images[*i];
                let fail = |message| Error::Texture {
                    asset: self.asset.into(),
                    message: format!("image {}: {message}", file.name),
                };
                let src = load::open(&file.bytes()?).map_err(fail)?;
                let clips = load::any_stored(&src, |l| refs.iter().any(|r| self.texels_clip(r, l)));
                Ok(clips.map_err(fail)?.then_some(*named))
            })
            .collect::<Result<_>>()?;
        Ok(found.into_iter().flatten().collect())
    }

    /// `r` reading the channel the last override for its image sets for its input, if one does.
    fn channel_override(&self, r: TexRef) -> TexRef {
        let set = |input: fn(&InputChannels) -> Option<u8>| {
            self.overridden(&[r.image], |o| input(&o.channels))
        };
        match r.input.and_then(set) {
            Some(channel) => TexRef {
                channel,
                ..self.file_channels(r)
            },
            None => r,
        }
    }

    /// The channel `r` reads for an input: the last override for its image that sets the
    /// input's channel, else its two-channel reading on a red-green image, else its own.
    fn channel_of(&self, r: TexRef, input: fn(&InputChannels) -> Option<u8>) -> Result<TexRef> {
        let rg = r.rg_channel.filter(|_| self.image(r.image).rg);
        let r = self.channel_override(TexRef {
            channel: rg.unwrap_or(r.channel),
            input: Some(input),
            ..r
        });
        // The set's own channel reads where it has no tile.
        match self.each(&r).iter().chain(Some(&r)).find(|t| t.channel > 3) {
            Some(t) => Err(Error::Texture {
                asset: self.asset.into(),
                message: format!(
                    "image {}: no channel {}",
                    self.images[t.image].name, t.channel
                ),
            }),
            None => Ok(r),
        }
    }

    /// Cooks the material's textures, binds them and folds constants into its factors.
    pub(crate) fn material(
        &mut self,
        suffix: &str,
        m: &mut Material,
        t: &MaterialTextures,
    ) -> Result<()> {
        let mut read = t.clone();
        for r in read.refs_mut() {
            if let Some(wrap) = self.image(r.image).wrap {
                r.wrap = std::array::from_fn(|a| if r.file_wrap[a] { wrap[a] } else { r.wrap[a] });
            }
            if self.tiled(r) {
                r.wrap = [Wrap::Clamp; 2];
                r.outside = r.missing;
            }
        }
        let t = &read;
        let remap = |r: Option<TexRef>, input| r.map(|r| self.channel_of(r, input)).transpose();
        let t = &MaterialTextures {
            opacity: remap(t.opacity, |c| c.opacity)?,
            roughness: remap(t.roughness, |c| c.roughness)?,
            metalness: remap(t.metalness, |c| c.metalness)?,
            specular: remap(t.specular, |c| c.specular)?,
            coat: remap(t.coat, |c| c.coat)?,
            transmission: remap(t.transmission, |c| c.transmission)?,
            // The channels of a vector are its axes, which no scalar input's channel stands for.
            displacement: match t.displacement {
                Some(r) if t.more.contains_key(&slot::DISPLACEMENT_U) => Some(r),
                r => remap(r, |c| c.displacement)?,
            },
            ..t.clone()
        };
        if let Some(r) = &t.base_color {
            // Only an alpha-tested or blended material reads base alpha as opacity, and only
            // where no opacity slot is bound.
            let alpha_free = m.alpha_mode == Material::OPAQUE
                || t.opacity
                    .is_some_and(|o| self.constant_opacity(m, &o).is_none());
            self.color(suffix, m, r, slot::BASE_COLOR, alpha_free)?;
        }
        if m.alpha_mode != Material::OPAQUE
            && let Some(r) = &t.opacity
        {
            match self.constant_opacity(m, r) {
                Some(a) => m.geometry_opacity *= a,
                None => {
                    let coverage = (m.alpha_mode == Material::MASK).then_some((0, m.alpha_cutoff));
                    let opacity = m.geometry_opacity;
                    let id = self.make_each(suffix, r, |s, cell| {
                        let r = s.at(r, cell);
                        let channel = usize::from(r.channel);
                        Pack {
                            role: "opacity",
                            encoding: Encoding::Bc4,
                            channels: vec![s.source(&r, 0, channel, 0.0, opacity)],
                            refs: vec![r],
                            coverage,
                            toksvig: None,
                            normal: false,
                            shrink: false,
                            tile: None,
                        }
                    })?;
                    m.geometry_opacity = 1.0;
                    bind(m, slot::OPACITY, id, 0, r);
                }
            }
        }

        let mut normal_packed = None;
        if let Some((raw, scale)) = t.normal {
            // A normal map whose source mips the cook keeps instead of remaking them adds no
            // Toksvig roughness.
            if self.normal_map(suffix, m, &raw, scale, slot::NORMAL, "normal")? {
                normal_packed = Some((raw, scale));
            }
        }
        if let Some((raw, scale)) = t.coat_normal {
            self.normal_map(suffix, m, &raw, scale, slot::COAT_NORMAL, "coat_normal")?;
        }

        // Roughness and metalness.
        let (rough, metal) = (t.roughness, t.metalness);
        let constant = |r: &Option<TexRef>| match r {
            Some(r) => self.constant_input(r),
            None => Some(1.0),
        };
        let (rough_const, metal_const) = (constant(&rough), constant(&metal));
        let rough_varies = rough_const.is_none() || normal_packed.is_some();
        let metal_varies = metal_const.is_none();
        if !rough_varies {
            m.specular_roughness *= rough_const.unwrap_or(1.0);
        }
        if !metal_varies {
            m.metalness *= metal_const.unwrap_or(1.0);
        }
        let normal_ref = normal_packed.map(|n| n.0);
        // Per texture: per channel its slot, reference and value, and the references it samples
        // like. Roughness and metalness share one texture when they sample alike.
        type Slots = Vec<(usize, Option<TexRef>, f32)>;
        let mut packs: Vec<(Slots, Vec<TexRef>)> = Vec::new();
        if rough_varies {
            let refs = rough.or(normal_ref).into_iter().collect();
            let entry = match (rough, rough_const) {
                (Some(r), None) => (slot::ROUGHNESS, Some(r), m.specular_roughness),
                (_, c) => (
                    slot::ROUGHNESS,
                    None,
                    m.specular_roughness * c.unwrap_or(1.0),
                ),
            };
            packs.push((vec![entry], refs));
        }
        if metal_varies {
            let r = metal.expect("varying metalness has an image");
            let entry = (slot::METALNESS, Some(r), m.metalness);
            let alike = |a: &TexRef| self.alike(a, &r) && a.filter == r.filter;
            match packs.first_mut() {
                Some((channels, refs)) if alike(&refs[0]) => channels.push(entry),
                _ => packs.push((vec![entry], vec![r])),
            }
        }
        for (rm, refs) in packs {
            let (encoding, role) = match rm.as_slice() {
                [(slot::ROUGHNESS, ..)] => (Encoding::Bc4, "roughness"),
                [_] => (Encoding::Bc4, "metalness"),
                _ => (Encoding::Bc5, "roughness_metalness"),
            };
            // Toksvig pairs roughness and normal texels, so both must sample alike.
            let paired = match (&normal_packed, rm[0].0 == slot::ROUGHNESS) {
                (Some((r, s)), true) if self.alike(&refs[0], r) => Some((*r, *s)),
                _ => None,
            };
            let id = self.make_each(suffix, &refs[0], |s, cell| {
                let channel = |(_, r, v): &(usize, Option<TexRef>, f32)| match r {
                    Some(r) => {
                        let r = s.at(r, cell);
                        s.source(&r, 0, usize::from(r.channel), 0.0, *v)
                    }
                    None => Channel::constant(*v),
                };
                let mut channels: Vec<Channel> = rm.iter().map(channel).collect();
                channels.resize(4, Channel::constant(0.0));
                let toksvig = paired.map(|(r, scale)| {
                    let (r, source) = s.normal_source(&s.at(&r, cell));
                    (0, r, scale, source)
                });
                Pack {
                    role,
                    encoding,
                    channels,
                    refs: refs.iter().map(|r| s.at(r, cell)).collect(),
                    coverage: None,
                    toksvig,
                    normal: false,
                    shrink: false,
                    tile: None,
                }
            })?;
            for (i, (s, ..)) in rm.iter().enumerate() {
                bind(m, *s, id, i, &refs[0]);
                if *s == slot::ROUGHNESS {
                    m.specular_roughness = 1.0;
                } else {
                    m.metalness = 1.0;
                }
            }
        }

        if let Some(r) = &t.emissive {
            self.color(suffix, m, r, slot::EMISSION, true)?;
        }
        if let Some(r) = &t.subsurface_color {
            self.color(suffix, m, r, slot::SUBSURFACE_COLOR, true)?;
        }
        if let Some(r) = &t.specular {
            let hdr = self.past_one(r);
            let constant = if hdr {
                self.constant_ranged(r, usize::from(r.channel))
                    .map(|v| r.read(0, v).max(0.0))
            } else {
                self.constant_input(r)
            };
            match constant {
                Some(v) => m.specular_weight *= v,
                None => {
                    let encoding = if hdr {
                        Encoding::Raw(DataFormat::R16f)
                    } else {
                        Encoding::Bc4
                    };
                    let id = self.single(suffix, r, 1.0, "specular_weight", encoding)?;
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
            *weight(m) = match self.constant_input(r) {
                Some(v) => factor * v,
                None => {
                    let id = self.single(suffix, r, factor, role, Encoding::Bc4)?;
                    bind(m, s, id, 0, r);
                    1.0
                }
            };
        }
        if let Some(r) = &t.displacement {
            let mut h = [
                m.displacement_scale,
                m.displacement_offset,
                m.displacement_min,
                m.displacement_max,
            ];
            self.height(suffix, m, r, slot::DISPLACEMENT, "displacement", &mut h)?;
            [
                m.displacement_scale,
                m.displacement_offset,
                m.displacement_min,
                m.displacement_max,
            ] = h;
        }
        for (&s, r) in &t.more {
            if let Some(r) = r {
                self.more(suffix, m, s, r)?;
            }
        }
        if let Some(settings) = &self.rules.neural {
            self.neural_set(suffix, m, settings)?;
        }
        Ok(())
    }

    /// Cooks normal map `raw` at `scale` into slot `s` unless it is flat, and returns whether it
    /// varies through mips the cook made.
    fn normal_map(
        &mut self,
        suffix: &str,
        m: &mut Material,
        raw: &TexRef,
        scale: [f32; 2],
        s: usize,
        role: &'static str,
    ) -> Result<bool> {
        if m.object_normal(s) {
            return self.object_normal_map(suffix, m, raw, s, role);
        }
        let (r, source) = self.normal_source(raw);
        let r = &r;
        let map = r.remap.expect("a normal map reads through a remap");
        let flat = [0, 1, 2].map(|c| ([0.0, 0.0, 1.0][c] - map[c][0]) / map[c][1]);
        let mut consts: Vec<Option<f32>> = (0..3).map(|c| self.constant(r, c)).collect();
        // A two-channel image's blue is no z, so x and y alone say whether it is flat.
        if source.rg {
            consts[2] = Some(flat[2]);
        }
        let is_flat = consts
            .iter()
            .zip(flat)
            .all(|(c, f)| c.is_some_and(|v| (v - f).abs() <= 1.0 / 255.0));
        if is_flat {
            return Ok(false);
        }
        let shrink = consts.iter().all(Option::is_some);
        let id = self.make_each(suffix, r, |this, cell| {
            let (r, source) = this.normal_source(&this.at(raw, cell));
            let channel = |c: usize, scale: f32| this.source(&r, c, c, 0.0, scale);
            Pack {
                role,
                encoding: Encoding::Bc5,
                channels: vec![
                    channel(0, scale[0]),
                    channel(1, scale[1]),
                    if source.rg {
                        Channel {
                            texels: Texels::NormalZ {
                                encoded: false,
                                image: r.image,
                                source,
                            },
                            outside: Outside::of(normal::z(
                                r.read(0, r.outside[0]),
                                r.read(1, r.outside[1]),
                            )),
                        }
                    } else {
                        channel(2, 1.0)
                    },
                    Channel::constant(1.0),
                ],
                refs: vec![r],
                coverage: None,
                toksvig: None,
                normal: true,
                shrink,
                tile: None,
            }
        })?;
        bind(m, s, id, 0, r);
        Ok(!id.source_mips && consts.iter().any(Option::is_none))
    }

    /// A normal map of mesh space directions. Unlike the tangent plane's they reach every side, so
    /// each holds three channels at `0.5 v + 0.5`, and none is flat. Roughness takes no variance
    /// from it.
    fn object_normal_map(
        &mut self,
        suffix: &str,
        m: &mut Material,
        raw: &TexRef,
        s: usize,
        role: &'static str,
    ) -> Result<bool> {
        let (r, source) = self.normal_source(raw);
        // A two-channel image's z follows from its x and y.
        let held = if source.rg { 2 } else { 3 };
        let shrink = (0..held).all(|c| self.constant(&r, c).is_some());
        let id = self.make_each(suffix, &r, |this, cell| {
            let (r, source) = this.normal_source(&this.at(raw, cell));
            let channel = |c: usize| this.source(&r, c, c, 0.5, 0.5);
            Pack {
                role,
                encoding: Encoding::Bc7,
                channels: vec![
                    channel(0),
                    channel(1),
                    if source.rg {
                        let z = normal::z(r.read(0, r.outside[0]), r.read(1, r.outside[1]));
                        Channel {
                            texels: Texels::NormalZ {
                                encoded: true,
                                image: r.image,
                                source,
                            },
                            outside: Outside::of(0.5 * z + 0.5),
                        }
                    } else {
                        channel(2)
                    },
                    Channel::constant(1.0),
                ],
                refs: vec![r],
                coverage: None,
                toksvig: None,
                normal: false,
                shrink,
                tile: None,
            }
        })?;
        bind(m, s, id, 0, &r);
        Ok(false)
    }

    /// Packs height texture `r` into slot `s`, whose `h` is scale, offset and declared range.
    fn height(
        &mut self,
        suffix: &str,
        m: &mut Material,
        r: &TexRef,
        s: usize,
        role: &'static str,
        h: &mut [f32; 4],
    ) -> Result<()> {
        let channel = usize::from(r.channel);
        // An image with no finite texels leaves no span to measure.
        let (lo, hi) = match self.input_range(r) {
            (lo, hi) if (hi - lo).is_finite() => (lo, hi),
            _ => (0.0, 1.0),
        };
        let (lo, hi) = if self.bounded(r) {
            (lo.min(r.outside[channel]), hi.max(r.outside[channel]))
        } else {
            (lo, hi)
        };
        let span = hi - lo;
        // A constant image stores 0, and the offset holds its height.
        let b = if span.recip().is_finite() {
            span.recip()
        } else {
            0.0
        };
        let shrink = self.constant(r, channel).is_some();
        let id = self.make_each(suffix, r, |s, cell| {
            let r = s.at(r, cell);
            let channel = usize::from(r.channel);
            Pack {
                role,
                encoding: Encoding::Raw(DataFormat::R16),
                channels: vec![s.source(&r, 0, channel, -lo * b, b)],
                refs: vec![r],
                coverage: None,
                toksvig: None,
                normal: false,
                shrink,
                tile: None,
            }
        })?;
        bind(m, s, id, 0, r);
        h[1] += h[0] * lo;
        h[0] *= span;
        // Importers declare the span of a 0..1 texel. A height past it widens that, and any
        // range a patch declared, which otherwise stands for the mesh cook to check.
        if lo < 0.0 || hi > 1.0 {
            let (a, b) = (h[1], h[1] + h[0]);
            h[2] = h[2].min(a.min(b));
            h[3] = h[3].max(a.max(b));
        }
        Ok(())
    }

    /// Cooks input `r` of slot `s`, from [`slot::MORE`] on.
    fn more(&mut self, suffix: &str, m: &mut Material, s: usize, r: &TexRef) -> Result<()> {
        match More::of(s) {
            More::Color => self.color(suffix, m, r, s, true),
            More::Height(axis, role) => {
                let mut h = m.displacement_tangent[axis];
                self.height(suffix, m, r, s, role, &mut h)?;
                m.displacement_tangent[axis] = h;
                Ok(())
            }
            More::Weight(role) => {
                let factor = *scalar_field(m, s);
                *scalar_field(m, s) = match self.constant_input(r) {
                    Some(v) => factor * v,
                    None => {
                        let id = self.single(suffix, r, factor, role, Encoding::Bc4)?;
                        bind(m, s, id, 0, r);
                        1.0
                    }
                };
                Ok(())
            }
            More::Ranged(role) => {
                let hdr = self.past_one(r);
                let constant = if hdr {
                    self.constant_ranged(r, usize::from(r.channel))
                        .map(|v| r.read(0, v).max(0.0))
                } else {
                    self.constant_input(r)
                };
                match constant {
                    Some(v) => *scalar_field(m, s) *= v,
                    None => {
                        let encoding = if hdr {
                            Encoding::Raw(DataFormat::R16f)
                        } else {
                            Encoding::Bc4
                        };
                        let id = self.single(suffix, r, 1.0, role, encoding)?;
                        bind(m, s, id, 0, r);
                    }
                }
                Ok(())
            }
            More::Signed(role) => {
                let factor = *scalar_field(m, s);
                let channel = usize::from(r.channel);
                match self.constant(r, channel).filter(|_| !self.bounded(r)) {
                    Some(v) => *scalar_field(m, s) = factor * r.read(0, v),
                    None => {
                        let id = self.make_each(suffix, r, |this, cell| {
                            let r = this.at(r, cell);
                            let channel = usize::from(r.channel);
                            Pack {
                                role,
                                encoding: Encoding::Bc4,
                                channels: vec![this.source(&r, 0, channel, 0.5, 0.5 * factor)],
                                refs: vec![r],
                                coverage: None,
                                toksvig: None,
                                normal: false,
                                shrink: false,
                                tile: None,
                            }
                        })?;
                        bind(m, s, id, 0, r);
                    }
                }
                Ok(())
            }
            More::Direction(role) => {
                let id = self.make_each(suffix, r, |this, cell| {
                    let r = this.at(r, cell);
                    let channels = vec![
                        this.source(&r, 0, 0, 0.0, 1.0),
                        this.source(&r, 1, 1, 0.0, 1.0),
                        Channel::constant(0.0),
                        Channel::constant(0.0),
                    ];
                    Pack {
                        role,
                        encoding: Encoding::Bc5,
                        channels,
                        refs: vec![r],
                        coverage: None,
                        toksvig: None,
                        normal: false,
                        shrink: false,
                        tile: None,
                    }
                })?;
                bind(m, s, id, 0, r);
                Ok(())
            }
        }
    }

    /// Plans the neural set of the material's textures, unless one with the same textures
    /// is planned.
    fn neural_set(&mut self, suffix: &str, m: &Material, settings: &NeuralSettings) -> Result<()> {
        for (cell, ids) in self.neural_members(m) {
            let suffix = match cell {
                Some([u, v]) => format!("{suffix}/tile/{u}_{v}"),
                None => suffix.to_string(),
            };
            let jobs: Vec<&TextureJob> = ids.iter().map(|id| &self.jobs[self.by_id[id]]).collect();
            let Some(set) = neural::plan(self.asset, &suffix, &jobs, settings)? else {
                continue;
            };
            if self.sets.insert(set.members()) {
                self.neural.push(set);
            }
        }
        Ok(())
    }

    /// The textures of `m` that a neural set may hold, in slot order. The textures of a set have
    /// one size, so each cell of tiles makes a set of its own, apart from `m`'s plain textures.
    fn neural_members(&self, m: &Material) -> BTreeMap<Option<Cell>, Vec<AssetId>> {
        let mut ids: BTreeMap<Option<Cell>, Vec<AssetId>> = BTreeMap::new();
        let mut add = |cell, id| {
            let member = ids.entry(cell).or_default();
            if !member.contains(&id) {
                member.push(id);
            }
        };
        for (s, b) in m.textures.iter().enumerate() {
            if s == slot::OPACITY || b.texture == AssetId::NONE {
                continue;
            }
            match self.tiles_of.get(&b.texture) {
                Some(tiles) => {
                    for t in tiles {
                        add(Some([t.column, t.row]), t.texture);
                    }
                }
                None => add(None, b.texture),
            }
        }
        ids
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
        let fourth = if alpha_free {
            Channel::UNUSED
        } else {
            Channel::constant(1.0)
        };
        let (role, color): (_, fn(&mut Material) -> &mut [f32; 3]) = match s {
            slot::BASE_COLOR => ("base_color", |m| &mut m.base_color),
            slot::EMISSION => ("emission", |m| &mut m.emission_color),
            slot::SPECULAR_COLOR => ("specular_color", |m| &mut m.specular_color),
            slot::TRANSMISSION_COLOR => ("transmission_color", |m| &mut m.transmission_color),
            slot::TRANSMISSION_SCATTER => ("transmission_scatter", |m| &mut m.transmission_scatter),
            slot::SUBSURFACE_RADIUS_SCALE => ("subsurface_radius_scale", |m| {
                &mut m.subsurface_radius_scale
            }),
            slot::FUZZ_COLOR => ("fuzz_color", |m| &mut m.fuzz_color),
            slot::COAT_COLOR => ("coat_color", |m| &mut m.coat_color),
            _ => ("subsurface_color", |m| &mut m.subsurface_color),
        };
        let (read, _, gamut) = self.color_read(r);
        let primaries = self.slot_primaries(r);
        let hdr = s == slot::EMISSION && self.past_one(r);
        let top = if hdr { f32::INFINITY } else { 1.0 };
        let held = (0..3).all(|c| {
            if hdr {
                self.constant_ranged(r, c).is_some()
            } else {
                self.stored_constant(r, c).is_some()
            }
        });
        // A premultiplied file's color varies with its alpha.
        let alpha = self.image(r.image).constant[3];
        if held && (alpha.is_some() || !read.straight) {
            let [x, y, z] = [0, 1, 2].map(|c| self.stored_constant(r, c).expect("constant"));
            let t = [x, y, z, alpha.unwrap_or(1.0)];
            let mut texel: [f32; 3] = std::array::from_fn(|c| r.read(c, read.value(t, c)));
            if let Some(m) = &gamut {
                texel = multiply(m, texel);
            }
            let texel = texel.map(|v| self.held(r, v, top));
            let target = color(m);
            // Rec.709 multiplies directly, so its factors stay exact.
            *target = match primaries {
                Primaries::Rec709 => std::array::from_fn(|c| target[c] * texel[c]),
                p => {
                    let (rec709, p) = (RgbPrimaries::REC709, p.rgb());
                    let f = rec709.matrix_to(p) * DVec3::from(target.map(f64::from));
                    let product = f * DVec3::from(texel.map(f64::from));
                    (p.matrix_to(rec709) * product).as_vec3().to_array()
                }
            };
            return Ok(());
        }
        // Texels read in their texture's primaries, and a slot's value past 0..1 with them.
        let outside = match primaries {
            Primaries::Rec709 => r.outside,
            p => {
                let [x, y, z, w] = r.outside;
                let rgb =
                    RgbPrimaries::REC709.matrix_to(p.rgb()) * DVec3::from([x, y, z].map(f64::from));
                let [x, y, z] = rgb.as_vec3().to_array();
                [x, y, z, w]
            }
        };
        let id = self.make_each(suffix, r, |this, cell| {
            let r = this.at(r, cell);
            let (read, ..) = this.color_read(&r);
            let texture = TexRef { outside, ..r };
            let image = |c| this.read_source(&texture, c, c, read.clone(), 0.0, 1.0);
            Pack {
                role,
                encoding: if hdr {
                    Encoding::Bc6h
                } else {
                    Encoding::Bc7Srgb
                },
                channels: vec![image(0), image(1), image(2), fourth.clone()],
                refs: vec![r],
                coverage: None,
                toksvig: None,
                normal: false,
                shrink: false,
                tile: cell.map(|c| (c, Some(primaries))),
            }
        })?;
        bind(m, s, id, 0, r);
        Ok(())
    }

    /// A one-channel texture of the reference's channel times `scale`.
    fn single(
        &mut self,
        suffix: &str,
        r: &TexRef,
        scale: f32,
        role: &'static str,
        encoding: Encoding,
    ) -> Result<Made> {
        self.make_each(suffix, r, |s, cell| {
            let r = s.at(r, cell);
            Pack {
                role,
                encoding,
                channels: vec![s.source(&r, 0, usize::from(r.channel), 0.0, scale)],
                refs: vec![r],
                coverage: None,
                toksvig: None,
                normal: false,
                shrink: false,
                tile: None,
            }
        })
    }

    /// Whether both read at the same uvs, and any tiles at the same cells.
    fn alike(&self, a: &TexRef, b: &TexRef) -> bool {
        a.samples_like(b) && self.cells(a) == self.cells(b)
    }

    /// The pack `build` gives for `r`: planned as it stands for an image, and for a set of tiles
    /// once per tile, by its cell, into a [`TextureTiles`] asset. Where no tile lies the set reads
    /// what the pack of the set as a whole says, which no one tile's file or overrides change.
    fn make_each(
        &mut self,
        suffix: &str,
        r: &TexRef,
        build: impl Fn(&Self, Option<Cell>) -> Pack,
    ) -> Result<Made> {
        let Some(cells) = self.cells(r) else {
            return self.make(suffix, build(self, None));
        };
        let mut tiles = Vec::with_capacity(cells.len());
        let mut keys = Vec::new();
        let mut set = None;
        let mut source_mips = false;
        for cell in cells {
            let mut p = build(self, Some(cell));
            let primaries = p.tile.and_then(|t| t.1);
            p.tile = Some((cell, primaries));
            let role = p.role;
            let made = self.make(suffix, p)?;
            keys.extend_from_slice(&self.jobs[self.by_id[&made.id]].key.0);
            tiles.push((cell[0], cell[1], made.id));
            source_mips |= made.source_mips;
            set = Some((made, role));
        }
        let (made, role) = set.expect("a set has a tile");
        let id = self.tile_set(suffix, role, &tiles, &keys);
        Ok(Made {
            id,
            channels: [0, 1, 2, 3],
            outside: build(self, None).outside(),
            tiled: true,
            source_mips,
            ..made
        })
    }

    /// Plans the tile set of `tiles`, whose textures' keys are `keys`, or returns the one planned
    /// with the same tiles.
    fn tile_set(
        &mut self,
        suffix: &str,
        role: &str,
        tiles: &[(u16, u16, AssetId)],
        keys: &[u8],
    ) -> AssetId {
        let set = TextureTiles::new(tiles);
        let bytes = set.to_bytes();
        *self.made_sets.entry(Hash32::of(&bytes)).or_insert_with(|| {
            let name = texture_name(self.asset, suffix, role);
            let key = Key::new(Class::TextureTiles, &name)
                .bytes(&bytes)
                .bytes(keys)
                .finish();
            let id = AssetId::from_name(&name);
            self.tiles_of.insert(id, set.tiles);
            self.tile_sets.push(TileSetJob { name, key, bytes });
            id
        })
    }

    /// How pack `p` keeps the blocks of its one source image, and per packed channel the
    /// source channel that holds it; `None` when it needs its texels made. The image's blocks
    /// must be the pack's format (or a BC5 whose half a BC4 takes), with every mip, each
    /// packed channel a source channel as stored. Mip 0 alone passes when the pack makes its
    /// other mips from it (Toksvig roughness, alpha coverage unless `source_mips`), keeps its
    /// full size and holds its channels in pack order, as the made mips do.
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
            Encoding::Bc7Srgb | Encoding::Bc7 => 4,
            Encoding::Bc6h => 3,
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
            match &ch.texels {
                Texels::Image {
                    image: i,
                    channel,
                    read,
                    a,
                    b,
                } => {
                    let identity = if p.normal {
                        (*a, *b) == (-1.0, 2.0)
                    } else {
                        (*a, *b) == (0.0, 1.0)
                    };
                    // Color channels keep their sRGB encoding; alpha and data are linear.
                    let stored = if p.encoding == Encoding::Bc7Srgb && c < 3 {
                        Decode::SRGB
                    } else {
                        Decode::STORED
                    };
                    if !identity || *image.get_or_insert(*i) != *i || !read.is(stored) {
                        return None;
                    }
                    channels[c] = *channel as u8;
                }
                Texels::Unused => {}
                // A color texture's alpha of 1 holds where the source's is 1.
                Texels::Constant(v) if count == 4 && c == 3 => {
                    let opaque = self.image(image?).constant[3]
                        .is_some_and(|a| (a - 1.0).abs() <= 1.0 / 255.0);
                    if *v != 1.0 || !opaque {
                        return None;
                    }
                }
                Texels::Constant(_) | Texels::NormalZ { .. } => return None,
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
            Encoding::Bc7Srgb | Encoding::Bc7 | Encoding::Bc6h => channels[..3] == [0, 1, 2],
            Encoding::Bc5 if p.normal => channels[..2] == [0, 1],
            _ => true,
        };
        let held = match format {
            vk::BC4_UNORM => 1,
            vk::BC5_UNORM => 2,
            _ => 4,
        };
        let derived = p.toksvig.is_some() || p.coverage.is_some() && !source_mips;
        // A neural set trains on the packed channels in pack order.
        let reordered = channels[..count]
            .iter()
            .enumerate()
            .any(|(c, &s)| usize::from(s) != c);
        if !in_order
            || channels[..count].iter().any(|&c| c >= held)
            || reordered && half.is_none() && (derived || self.rules.neural.is_some())
            || !complete
            || derived && skipped
        {
            return None;
        }
        if half.is_some() {
            channels[0] = 0;
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
            .filter_map(|c| match c.texels {
                Texels::Image { image, .. } | Texels::NormalZ { image, .. } => Some(image),
                Texels::Constant(_) | Texels::Unused => None,
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
        if let Some((_, r, ..)) = &p.toksvig
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
        let (primaries, gamut) = if matches!(p.encoding, Encoding::Bc7Srgb | Encoding::Bc6h) {
            let (_, primaries, gamut) = self.color_read(&p.refs[0]);
            match p.tile.and_then(|t| t.1).filter(|set| *set != primaries) {
                Some(set) => {
                    let to = primaries.rgb().matrix_to(set.rgb());
                    (set, Some(gamut.map_or(to, |g| to * g)))
                }
                None => (primaries, gamut),
            }
        } else {
            // A set holds one id, which an override of some of its tiles can't split.
            let primaries = primaries.filter(|_| p.tile.is_none());
            (primaries.unwrap_or(Primaries::Rec709), None)
        };
        let skipped = super::skipped_mips(size[0], size[1], max_size) > 0;
        // A slot reads one set of channels, whichever tile it lands on.
        let pass = self
            .passes(&p, size, grid, skipped, source_mips)
            .filter(|_| gamut.is_none())
            .filter(|(_, channels)| p.tile.is_none() || *channels == [0, 1, 2, 3]);
        // A hardware border reads transparent black, whatever else a slot of this texture reads
        // past 0..1 takes the shader's check.
        let outside = p.outside();
        if outside != [0.0; 4] {
            let wrap = &mut p.refs[0].wrap;
            *wrap = wrap.map(|w| if w == Wrap::Border { Wrap::Constant } else { w });
        }
        let wrap = p.refs[0].wrap;
        // Constant's texels and mips are Clamp's, and what it reads past them is the binding's.
        let mip_wrap = wrap.map(|w| if w == Wrap::Constant { Wrap::Clamp } else { w });

        // A rebuilt z reads the remap alone, which a normal's channels hold times the scale.
        let mut recipe = format!(
            "{} {:?} {:?} {:?} {:?} {:?} {:?} {max_size:?} {grid:?} {primaries:?} {gamut:?} {bc_split:?} {:?} {:?} {:?} {source_mips}",
            p.role,
            p.encoding,
            p.channels.iter().map(|c| &c.texels).collect::<Vec<_>>(),
            p.coverage,
            p.toksvig.as_ref().map(|t| (t.0, t.1.image, t.1.remap, t.2, t.3)),
            p.normal.then_some(p.refs[0].remap),
            mip_wrap,
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
        let made = |id| Made {
            id,
            channels,
            wrap,
            outside,
            tiled: false,
            source_mips,
        };
        if let Some(id) = self.made.get(&same) {
            return Ok(made(*id));
        }
        let mut name = texture_name(self.asset, suffix, p.role);
        if let Some(([u, v], _)) = p.tile {
            name = format!("{name}/tile/{u}_{v}");
        }
        let key = Key::new(Class::Texture, &name)
            .str(&super::encoders(self.quality))
            .bytes(&recipe)
            .finish();
        let id = AssetId::from_name(&name);
        let job_images = images.iter().map(|&i| (i, self.image(i).hash)).collect();
        self.made.insert(same, id);
        self.by_id.insert(id, self.jobs.len());
        self.jobs.push(TextureJob {
            name,
            key,
            asset: self.asset.into(),
            pack: p,
            images: job_images,
            size,
            quality: self.quality,
            max_size,
            grid,
            primaries,
            gamut,
            filter: self.rules.filter,
            bc_split,
            pass: pass.map(|p| p.0),
            source_mips,
        });
        Ok(made(id))
    }
}

impl Packer<'_> {
    /// The last value `field` sets among the overrides that match any of `images`, a tile by its
    /// own name or its set's.
    fn overridden<T>(
        &self,
        images: &[usize],
        field: impl Fn(&TextureOverride) -> Option<T>,
    ) -> Option<T> {
        let named = |i: usize, pattern: &str| self.images[i].name.contains(pattern);
        let matches = |i: usize, pattern: &str| {
            named(i, pattern) || self.set_of.get(&i).is_some_and(|&s| named(s, pattern))
        };
        self.rules
            .overrides
            .iter()
            .filter(|(pattern, _)| images.iter().any(|&i| matches(i, pattern)))
            .fold(None, |v, (_, o)| field(o).or(v))
    }
}

impl TextureJob {
    /// Estimated peak bytes: 8 per compressed source byte for the decoded sources, 48 per
    /// output texel for the packed image and its chain.
    pub(crate) fn working_set(&self, images: &[ImageFile]) -> u64 {
        let sources: u64 = self.images.iter().map(|&(i, _)| images[i].len() * 8).sum();
        sources + u64::from(self.size[0]) * u64::from(self.size[1]) * 48
    }

    pub(crate) fn run(&self, images: &[ImageFile], decoded: &Decoded<Img>) -> Result<Built> {
        let fail = |e: String| self.fail(format!("texture {}: {e}", self.pack.role));
        let built = match self.pass {
            Some(pass) => self.pass_through(pass, images, decoded).map_err(fail)?,
            None => {
                let chain = self.chain(images, decoded)?;
                super::finish(vec![chain], &self.plan(None)).map_err(fail)?
            }
        };
        Ok(built.with_split(self.bc_split))
    }

    /// The texture from the source's blocks: every mip as stored (without those over
    /// `max_size`), or mip 0 as stored and the rest made from it.
    fn pass_through(
        &self,
        pass: Pass,
        images: &[ImageFile],
        decoded: &Decoded<Img>,
    ) -> Result<Built, String> {
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
            let skip = super::skipped_mips(b.w, b.h, self.max_size);
            let kept = &b.levels[0][skip as usize..];
            (
                mip_size(b.w, b.h, skip),
                kept.iter().map(|m| half(m)).collect(),
            )
        } else {
            let first = half(&b.levels[0][0]);
            let (img, plan) = self
                .prepare(images, decoded, Some((pass.image, src)))
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
    pub(super) fn chain(&self, images: &[ImageFile], decoded: &Decoded<Img>) -> Result<Vec<Img>> {
        if self.source_mips {
            return self.source_chain(images, decoded);
        }
        let (img, plan) = self.prepare(images, decoded, None)?;
        super::chains(vec![img], &plan, 0)
            .map(|mut c| c.remove(0))
            .map_err(|e| self.fail(format!("texture {}: {e}", self.pack.role)))
    }

    /// Mip 0 size of the cooked texture.
    pub(super) fn cooked_size(&self) -> [u32; 2] {
        let [w, h] = self.size;
        mip_size(w, h, super::skipped_mips(w, h, self.max_size))
    }

    pub(super) fn encoding(&self) -> Encoding {
        self.pack.encoding
    }

    /// Whether the texture holds a tangent-space normal map.
    pub(super) fn is_normal(&self) -> bool {
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
        decoded: &Decoded<Img>,
        mut opened: Option<(usize, Source)>,
    ) -> Result<(Img, Plan)> {
        let _z = crate::profile::zone!("texture.prepare");
        let fail = |message: String| self.fail(message);
        let mut sources = HashMap::new();
        for &(i, hash) in &self.images {
            let opened = opened.take_if(|o| o.0 == i);
            let img = decoded.get((hash, 0), || match opened {
                Some((_, mut src)) => {
                    load::decode_layers(&mut src).map_err(&fail)?;
                    Ok(src.layers.remove(0))
                }
                None => images[i].decoded().map_err(&fail),
            })?;
            sources.insert(i, img.resized(self.size[0], self.size[1]));
        }
        let img = self.pack(&sources, self.size);
        let toksvig = self
            .pack
            .toksvig
            .as_ref()
            .map(|(channel, r, scale, source)| {
                let n = &sources[&r.image];
                let px = n.px.iter().map(|t| {
                    let [x, y, z] = r.normal(*source, *t);
                    [x * scale[0], y * scale[1], z, 1.0]
                });
                Toksvig {
                    channel: *channel,
                    normals: Img {
                        w: n.w,
                        h: n.h,
                        px: px.collect(),
                    },
                }
            });
        Ok((img, self.plan(toksvig)))
    }

    /// The packed chain from the source images' own mips, without those over `max_size`.
    fn source_chain(&self, images: &[ImageFile], decoded: &Decoded<Img>) -> Result<Vec<Img>> {
        let _z = crate::profile::zone!("texture.source_mips");
        let fail = |message: String| self.fail(message);
        let mut sources = HashMap::new();
        let [w, h] = self.size;
        (super::skipped_mips(w, h, self.max_size)..full_mip_count(w, h))
            .map(|m| {
                let [mw, mh] = mip_size(w, h, m);
                let mut mips = HashMap::new();
                for &(i, hash) in &self.images {
                    let img = decoded.get((hash, m), || {
                        if let Entry::Vacant(e) = sources.entry(i) {
                            e.insert(self.source_blocks(images, i)?);
                        }
                        let b: &Blocks = &sources[&i];
                        load::decode_blocks(b.vk_format, mw, mh, &b.levels[0][m as usize])
                            .map_err(&fail)
                    })?;
                    mips.insert(i, img);
                }
                let mut img = self.pack(&mips, [mw, mh]);
                if self.pack.normal {
                    img.px.par_iter_mut().for_each(|p| *p = normalized(*p));
                }
                Ok(img)
            })
            .collect()
    }

    /// The blocks of source image `i`, which must still hold every mip at the texture's size.
    fn source_blocks(&self, images: &[ImageFile], i: usize) -> Result<Blocks> {
        let src = load::open(&images[i].bytes()?).map_err(|m| self.fail(m))?;
        let b = src
            .blocks
            .ok_or_else(|| self.fail("the image no longer holds blocks".into()))?;
        if [b.w, b.h] != self.size || !b.complete() {
            return Err(self.fail("the image's mips changed".into()));
        }
        Ok(b)
    }

    /// The packed texture of `decoded` source images, each `w` x `h`.
    fn pack(&self, decoded: &HashMap<usize, Arc<Img>>, [w, h]: [u32; 2]) -> Img {
        let limit = self.pack.limit();
        let mut img = Img::new(w, h);
        // One ICC transform gives all three channels of a texel.
        let mut profiled: HashMap<usize, Vec<[f32; 4]>> = HashMap::new();
        let as_stored = Read::of(Decode::STORED);
        for (c, ch) in self.pack.channels.iter().enumerate() {
            match &ch.texels {
                &Texels::Constant(v) => img.px.par_iter_mut().for_each(|t| t[c] = v),
                Texels::Unused => img.px.par_iter_mut().for_each(|t| t[c] = 1.0),
                &Texels::NormalZ {
                    encoded,
                    image,
                    source,
                } => {
                    let r = self.pack.refs[0];
                    let (a, b) = if encoded { (0.5, 0.5) } else { (0.0, 1.0) };
                    img.px
                        .par_iter_mut()
                        .zip(&decoded[&image].px)
                        .for_each(|(t, s)| t[c] = a + b * r.normal(source, *s)[2]);
                }
                Texels::Image {
                    image,
                    channel,
                    read,
                    a,
                    b,
                } => {
                    let source = &decoded[image].px;
                    let (texels, read) = if read.icc.is_some() {
                        let texels = profiled.entry(*image).or_insert_with(|| {
                            source.par_iter().map(|s| read.color(*s, None)).collect()
                        });
                        (&*texels, &as_stored)
                    } else {
                        (source, read)
                    };
                    img.px.par_iter_mut().zip(texels).for_each(|(t, s)| {
                        t[c] = stored(read.value(*s, *channel), *a, *b, limit);
                    });
                }
            }
        }
        if let Some(m) = &self.gamut {
            img.px.par_iter_mut().for_each(|t| {
                let [r, g, b] = multiply(m, [t[0], t[1], t[2]]);
                *t = [r, g, b, t[3]];
            });
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
                filter: p.encoding.mip_filter(self.filter),
                wrap: p.refs[0].wrap,
                grid: self.grid.map(|g| g.map(u32::from)),
                normal: p.normal,
                coverage: p.coverage,
                height: p.role.starts_with("displacement"),
            },
            full_chain: true,
            quality: self.quality,
            max_size: self.max_size,
            toksvig,
            lights: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(shift: u32) -> ImageData {
        let img = image::RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba([(x * 32 + shift) as u8, (y * 32) as u8, 0, 255])
        });
        let mut bytes = std::io::Cursor::new(Vec::new());
        img.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        ImageData::Bytes(bytes.into_inner())
    }

    /// Tiles of one cell go to one neural set whichever slots they fill, so that a tap of any
    /// tile finds what stands in for it at that tile's size, and a slot of plain textures stays
    /// out of them. Opacity never joins a set, on tiles or off.
    #[test]
    fn each_cell_of_tiles_plans_a_neural_set_of_its_own() {
        let image = |name: &str, data| ImageFile {
            name: name.into(),
            data,
        };
        let tiles = |images: [usize; 2]| {
            ImageData::Tiles(vec![
                ImageTile {
                    cell: [0, 0],
                    image: images[0],
                },
                ImageTile {
                    cell: [1, 0],
                    image: images[1],
                },
            ])
        };
        let mut images = vec![
            image("c.1001", gradient(0)),
            image("c.1002", gradient(1)),
            image("r.1001", gradient(2)),
            image("r.1002", gradient(3)),
            image("o.1001", gradient(4)),
            image("o.1002", gradient(5)),
            image("plain", gradient(6)),
        ];
        images.push(image("c.<UDIM>", tiles([0, 1])));
        images.push(image("r.<UDIM>", tiles([2, 3])));
        images.push(image("o.<UDIM>", tiles([4, 5])));
        let (color, rough, opacity, plain) = (7, 8, 9, 6);
        let reads = ImageReads {
            whole: HashSet::new(),
            normals: HashSet::new(),
            colors: HashSet::new(),
            optional: HashSet::new(),
        };
        let used: Vec<usize> = (0..7).collect();
        let (mut facts, mut unread) =
            image_facts("a", &mut images, &used, &reads, &Memo::default()).unwrap();
        settle_tiles(
            &[color, rough, opacity],
            &mut images,
            &mut facts,
            &mut unread,
        );
        let at = |image, channel| TexRef {
            image,
            rules: ImageRules::MaterialX,
            channel,
            rg_channel: None,
            input: None,
            uv_set: 0,
            wrap: [Wrap::Repeat; 2],
            file_wrap: [false; 2],
            filter: 0,
            outside: [0.0; 4],
            missing: [0.0; 4],
            uv: [0.0; 6],
            color: ImageColor::Data,
            remap: None,
        };
        let textures = MaterialTextures {
            base_color: Some(at(color, 0)),
            roughness: Some(at(rough, 0)),
            opacity: Some(at(opacity, 0)),
            specular: Some(at(plain, 0)),
            ..MaterialTextures::default()
        };
        let mut m = Material {
            alpha_mode: Material::BLEND,
            ..Material::default()
        };
        let rules = TextureRules::default();
        let mut packer = Packer::new("a", &images, &facts, &rules, Quality::UltraFast);
        packer.material("m", &mut m, &textures).unwrap();

        let id = |role: &str| AssetId::from_name(&format!("a/texture/m/{role}"));
        let want: BTreeMap<Option<Cell>, Vec<AssetId>> = [
            (None, vec![id("specular_weight")]),
            (
                Some([0, 0]),
                vec![id("base_color/tile/0_0"), id("roughness/tile/0_0")],
            ),
            (
                Some([1, 0]),
                vec![id("base_color/tile/1_0"), id("roughness/tile/1_0")],
            ),
        ]
        .into();
        assert_eq!(packer.neural_members(&m), want);
    }
}
