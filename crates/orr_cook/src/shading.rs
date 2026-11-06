//! Shading networks, as UsdShade and MaterialX documents author them, mapped onto
//! [`Material`] and its textures.
//!
//! Two surfaces map:
//! - MaterialX `open_pbr_surface`, input for field: `base_metalness` is `metalness`,
//!   `base_diffuse_roughness` `diffuse_roughness`, `specular_roughness_anisotropy`
//!   `specular_anisotropy`, `geometry_thin_walled` `thin_walled`, `thin_film_thickness`
//!   (micrometers) the field in nanometers, `coat_roughness_anisotropy` `coat_anisotropy`, and
//!   every other input the field of its name. `geometry_coat_normal` reads as `geometry_normal`
//!   does, and a `normalmap` `space` of `object` in either states directions in the mesh's space,
//!   which turn with what places it and with a transform an importer bakes into the mesh
//!   ([`Material::object_frame`]), but not with a skin's joints, which warns.
//!   `geometry_tangent` and `geometry_coat_tangent` take a
//!   `tangent` node, the mesh's, or a texture of a direction in the tangent plane; any other warns
//!   and reads the mesh's. The base and the coat each follow the uv set the `index` of their
//!   `tangent` nodes names. Mesh tangents hold one of the two, and a coat on another set takes its
//!   tangent from the triangle's uvs. A `geometry_opacity` below 1 or textured blends, as the
//!   linear presence weight OpenPBR defines.
//! - `UsdPreviewSurface`, mapped like glTF metallic-roughness: `diffuseColor` is
//!   `base_color`, `metallic`, `roughness`, `ior` and `emissiveColor` (read as nits) their
//!   fields, `clearcoat` and `clearcoatRoughness` the coat at `ior`, and `normal` and
//!   `displacement` their textures, a constant reading a solid image ([`solid_image`]).
//!   `occlusion` isn't read: a path tracer computes it. `opacityThreshold` above 0 alpha-tests
//!   `opacity` at it; otherwise `opacityMode` `presence` blends an opacity below 1 or textured,
//!   and `transparent` makes the rest of it thin-walled transmission, which leaves specular and
//!   emission at full weight. The specular workflow's `specularColor` is F0, with `ior` left to
//!   refraction and the coat.
//!
//! An input is a constant, or reads a texture through these nodes: MaterialX `image` and
//! `tiledimage` (with `texcoord`, `geompropvalue` and `place2d` on their coordinates,
//! `extract`, `separate2`, `separate3` or `separate4` picking a channel, `extract` a row of a
//! constant matrix, `normalmap` under `geometry_normal`), and `UsdUVTexture` (with
//! `UsdPrimvarReader_float2` and `UsdTransform2d` on `st`, whose default is the constant
//! (0, 0)). `constant` and `dot` nodes
//! pass on what they read to MaterialX nodes. A primvar reader, or a `geompropvalue` or `geomcolor`
//! off a coordinate, gives the value of a primvar constant over the mesh, and its `fallback` or
//! `default` where the mesh has no such primvar. One that varies over the mesh takes channels of
//! the mesh's vertex color stream, four between all that one material reads, and multiplies its
//! input's texture or stands in for it ([`VertexRead`]); `geomcolor` reads `displayColor`, or
//! with no mesh to ask the stream as it stands, and at any `index` but 0 a color no mesh holds,
//! which is 0. Any other node fails the import, naming it, as
//! does a connection cycle. A `geompropvalue` on a coordinate whose property no uv set holds reads
//! its `default` at every point, as an unconnected `place2d` reads 0, 0. Textures go to the
//! material's slots. An interior medium is one for the whole material, so a textured
//! `transmission_depth`, `transmission_scatter`, `transmission_scatter_anisotropy` or dispersion
//! input fails, and a `transmission_color` texture under a `transmission_depth` above 0 warns and
//! reads the field, unless the material is thin-walled. A texture value multiplies the field, which takes the reader's scale
//! (`UsdUVTexture` `scale`; 1 otherwise); a bias folds scale and bias into the texels
//! ([`TexRef::remap`]) and leaves the field 1, or for a color what it reaches past 1. Normals
//! read `scale * t + bias` per axis; a pair of the form `s (2 t - 1)` is the tangent-space
//! encoding at strength `s`. A `normalmap` decodes the reader's `v` as `2 v - 1`, where such a
//! pair is the encoding stated, not a second decode. Both hold as far as the image's format can
//! hold what they state. Coordinate nodes set each slot's `Material::uv_transforms`,
//! conjugated by the flip of t between st and the cook's top-left uvs.
//!
//! A value that states a [`Unit`] converts to meters or degrees. `transmission_depth`,
//! `subsurface_radius` (its default too) and both kinds of `displacement` are lengths: without a
//! unit they are in the scene's length unit ([`Host::meters_per_unit`]). They are world space
//! lengths, so no object scale enters them, and a displacement keeps its length under a
//! placement's scale ([`Material::displacement_world`]). `thin_film_thickness` without a unit is
//! in micrometers, as OpenPBR states it, whatever the scene's unit.
//!
//! A texture whose file can't be read, or which names none, reads what its node states for that
//! everywhere: `UsdUVTexture` `fallback`, default (0, 0, 0, 1), taking the texture's place in
//! `scale * t + bias`, or `image` and `tiledimage` `default`, default 0. Those values are linear
//! and take no decoding. One that is no constant reads as that default there, and a `default`
//! past the edges under `constant` fails.
//!
//! A file name holding a tile token is a set of images, one to each unit square of st
//! ([`crate::texture::tiles`]). Those values are what a square with no tile reads, as a point there
//! names a file that isn't. A token, not an address mode, picks the tile, so `wrapS`, `wrapT`,
//! `uaddressmode` and `vaddressmode` have nothing to state of a set: tiles clamp at their edges,
//! as USD's Storm has them.
//!
//! A `UsdUVTexture` reads as its `sourceColorSpace` says, or under `auto` as the file's
//! `colorSpace` metadata names, and a MaterialX image of a color type as its `colorspace` names;
//! [`crate::texture::color`] has what each rule does. An unknown `colorspace` warns and reads as
//! "data". A channel picked from a color image takes the space's curve and none of its primaries:
//! a renderer working in the document's space would read the channel as it stands.

use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use orr_math::{Affine2, Vec2};
use orr_pack::material::{Material, filter, slot};

use crate::texture::Wrap;
use crate::texture::color::{ImageColor, Space};
use crate::texture::material::{ImageData, ImageFile, ImageRules, MaterialTextures, TexRef};

/// A shading network: nodes and the connections between their inputs and outputs.
#[derive(Default)]
pub(crate) struct Network {
    pub(crate) nodes: Vec<Node>,
    /// Geometric properties a document defines as a uv set, by name.
    pub(crate) uv_sets: HashMap<String, u8>,
}

pub(crate) struct Node {
    /// Prim path or document name, for messages.
    pub(crate) name: String,
    /// MaterialX category (`image`, `open_pbr_surface`) or UsdShade id (`UsdUVTexture`).
    pub(crate) kind: String,
    /// MaterialX output type (`float`, `color3`); empty for UsdShade nodes.
    pub(crate) ty: String,
    pub(crate) inputs: BTreeMap<String, Input>,
}

pub(crate) enum Input {
    Value(Vec<f32>, Unit),
    Text(String),
    /// A file input: an image of the scene, its authored color space and the unit of its values.
    Image {
        image: usize,
        color_space: Option<String>,
        unit: Unit,
    },
    Connect {
        node: usize,
        output: String,
    },
}

/// The unit a value states.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Unit {
    Unstated,
    /// Meters per unit.
    Distance(f64),
    /// Degrees per unit.
    Angle(f64),
}

impl Unit {
    /// `v` in the scene's unit of this type.
    fn scene(self, v: f32) -> f32 {
        match self {
            Self::Unstated => v,
            Self::Distance(s) | Self::Angle(s) => scaled(v, s),
        }
    }
}

fn scaled(v: f32, by: f64) -> f32 {
    (f64::from(v) * by) as f32
}

/// What a `constant` or `dot` without a value reads.
static ZERO: LazyLock<Input> = LazyLock::new(|| Input::Value(vec![0.0], Unit::Unstated));

/// What a matrix `constant` or `dot` without a value reads: its nodedef's identity.
static IDENTITY33: LazyLock<Input> = LazyLock::new(|| identity(3));
static IDENTITY44: LazyLock<Input> = LazyLock::new(|| identity(4));

fn identity(n: usize) -> Input {
    let cell = |i| if i % (n + 1) == 0 { 1.0 } else { 0.0 };
    Input::Value((0..n * n).map(cell).collect(), Unit::Unstated)
}

fn is_pick(node: &Node) -> bool {
    matches!(
        node.kind.as_str(),
        "extract" | "separate2" | "separate3" | "separate4"
    )
}

/// Whether `node` is an `extract` of a matrix's row, whose output type tells it from one of a
/// float channel.
fn gives_row(node: &Node) -> bool {
    node.kind == "extract" && matches!(node.ty.as_str(), "vector3" | "vector4")
}

/// What `input` reads past `constant` and `dot` nodes, which pass a value on unchanged. A ring
/// of them leaves a connection.
pub(crate) fn through<'n>(net: &'n Network, mut input: &'n Input) -> &'n Input {
    for _ in 0..net.nodes.len() {
        let Input::Connect { node, .. } = input else {
            break;
        };
        let node = &net.nodes[*node];
        let name = match node.kind.as_str() {
            "constant" => "value",
            "dot" => "in",
            _ => break,
        };
        input = node.inputs.get(name).unwrap_or(match node.ty.as_str() {
            "matrix33" => &IDENTITY33,
            "matrix44" => &IDENTITY44,
            _ => &ZERO,
        });
    }
    input
}

impl Network {
    /// The primvars the network's readers name.
    pub(crate) fn primvar_reads(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .nodes
            .iter()
            .filter_map(|n| match n.kind.as_str() {
                "geomcolor" => (color_index(n) == 0.0).then(|| DISPLAY_COLOR.to_string()),
                "geompropvalue" => text_of(n, "geomprop"),
                k if k.starts_with("UsdPrimvarReader_") => text_of(n, "varname"),
                _ => None,
            })
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    }
}

/// The primvar `geomcolor` reads at index 0.
const DISPLAY_COLOR: &str = "displayColor";

/// The `index` of `geomcolor` node `n`, a uniform input.
fn color_index(n: &Node) -> f32 {
    match n.inputs.get("index") {
        Some(Input::Value(v, _)) => v[0],
        _ => 0.0,
    }
}

fn text_of(n: &Node, input: &str) -> Option<String> {
    match n.inputs.get(input) {
        Some(Input::Text(t)) if !t.is_empty() => Some(t.clone()),
        _ => None,
    }
}

/// What a mesh holds for a primvar a reader names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Primvar {
    /// One of its uv sets.
    Uv(u8),
    /// One value for the whole mesh.
    Constant(Vec<f32>),
    /// Values that vary over the mesh, of `width` channels, at most the stream's four: a reader
    /// takes a wider primvar's first ones. The stream holds them over `scale`
    /// ([`stream_scale`]), `None` on a mesh that holds no stream and so reads 1.
    Varying {
        width: u8,
        scale: Option<f32>,
    },
    Missing,
}

/// A primvar a material reads through the vertex color stream.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VertexRead {
    pub(crate) name: String,
    /// The stream's channel of its first value.
    pub(crate) first: u8,
    pub(crate) width: u8,
    /// What the reader gives per channel where the mesh holds no value.
    pub(crate) fallback: [f32; 4],
    /// What the stream holds its values over ([`stream_scale`]).
    pub(crate) scale: f32,
}

/// The half floats' largest value.
const HALF_MAX: f64 = 65504.0;

/// What the vertex color stream divides values of magnitude up to `peak` by: 1 where its half
/// floats hold them, else the least power of two that brings `peak` within them, which leaves
/// every mantissa as it is and the most room below for small values. A value no `f32` field
/// reaches gets 1, for [`crate::mesh::check`] to refuse.
pub(crate) fn stream_scale(peak: f64) -> f32 {
    if peak <= HALF_MAX || peak > f64::from(f32::MAX) {
        return 1.0;
    }
    (peak / HALF_MAX).log2().ceil().exp2() as f32
}

/// The field slot `s` reads the stream through, whose channels multiply it.
fn stream_field(m: &mut Material, s: usize) -> &mut [f32] {
    match s {
        slot::BASE_COLOR => &mut m.base_color,
        slot::EMISSION => &mut m.emission_color,
        slot::SPECULAR_COLOR => &mut m.specular_color,
        slot::TRANSMISSION_COLOR => &mut m.transmission_color,
        slot::SUBSURFACE_COLOR => &mut m.subsurface_color,
        slot::FUZZ_COLOR => &mut m.fuzz_color,
        slot::COAT_COLOR => &mut m.coat_color,
        slot::OPACITY => std::slice::from_mut(&mut m.geometry_opacity),
        slot::METALNESS => std::slice::from_mut(&mut m.metalness),
        slot::ROUGHNESS => std::slice::from_mut(&mut m.specular_roughness),
        slot::SPECULAR_WEIGHT => std::slice::from_mut(&mut m.specular_weight),
        slot::TRANSMISSION => std::slice::from_mut(&mut m.transmission_weight),
        slot::COAT => std::slice::from_mut(&mut m.coat_weight),
        s => std::slice::from_mut(crate::texture::material::scalar_field(m, s)),
    }
}

/// Gives each slot of `m` that reads the stream back what `reads` hold it over: the field for a
/// slot it multiplies, which is the scale's whole effect on `m`, or a complement's own scale.
/// `m`'s fields are final by then, so a patch of one states what multiplies the primvar.
pub(crate) fn fold_stream_scales(m: &mut Material, reads: &[VertexRead]) {
    for s in 0..slot::COUNT {
        let channel = m.textures[s].vertex.wrapping_sub(1);
        let held = |r: &&VertexRead| (r.first..r.first + r.width).contains(&channel);
        let Some(scale) = reads.iter().find(held).map(|r| r.scale) else {
            continue;
        };
        if m.textures[s].vertex_complement != 0 {
            m.textures[s].vertex_complement = 1 + scale.log2() as u8;
        } else {
            stream_field(m, s).iter_mut().for_each(|f| *f *= scale);
        }
    }
}

/// What mapping asks of the scene a material lands in.
pub(crate) struct Host<'a> {
    /// The scene's length unit, which a length that states none is in.
    pub(crate) meters_per_unit: f64,
    /// The mesh primvar of each uv set, which geometric properties name.
    pub(crate) primvars: &'a [&'a str],
    /// What the mesh holds for each primvar of [`Network::primvar_reads`].
    pub(crate) reads: &'a [(String, Primvar)],
    /// No mesh is at hand to ask: `geomcolor` reads whatever the vertex color stream holds.
    pub(crate) any_mesh: bool,
    /// Adds an image to the scene, unless it holds one of that name; returns its index.
    pub(crate) add: &'a mut dyn FnMut(ImageFile) -> usize,
}

/// An image of one linear color, for inputs the material reads only through a texture.
pub(crate) fn solid_image(rgb: [f32; 3]) -> ImageFile {
    let texel = rgb.map(|c| (c.clamp(0.0, 1.0) * 65535.0).round() as u16);
    let image = image::ImageBuffer::from_pixel(1, 1, image::Rgb(texel));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, image::ImageFormat::Png)
        .expect("a PNG writes to memory");
    ImageFile {
        name: format!("solid/{texel:04x?}"),
        data: ImageData::Bytes(bytes.into_inner()),
    }
}

/// Whether a MaterialX type is one that color management covers.
pub(crate) fn is_color(ty: &str) -> bool {
    matches!(ty, "color3" | "color4")
}

/// The MaterialX category of a node definition id: `ND_image_color3` is `image`.
pub(crate) fn category(id: &str) -> &str {
    const KNOWN: &[&str] = &[
        "open_pbr_surface",
        "surfacematerial",
        "geompropvalue",
        "geomcolor",
        "displacement",
        "tiledimage",
        "normalmap",
        "bitangent",
        "tangent",
        "separate2",
        "separate3",
        "separate4",
        "texcoord",
        "constant",
        "extract",
        "place2d",
        "image",
        "dot",
    ];
    let Some(rest) = id.strip_prefix("ND_") else {
        return id;
    };
    KNOWN
        .iter()
        .find(|k| rest == **k || rest.strip_prefix(**k).is_some_and(|r| r.starts_with('_')))
        .copied()
        .unwrap_or(id)
}

pub(crate) struct Mapped {
    pub(crate) material: Material,
    pub(crate) textures: MaterialTextures,
    /// The primvars its vertex color stream holds.
    pub(crate) vertex: Vec<VertexRead>,
    /// What the material asks for that it does not get.
    pub(crate) warnings: Vec<String>,
}

/// Which channels of a texture an input reads.
#[derive(Copy, Clone, PartialEq)]
enum Channels {
    Rgb,
    One(u8),
}

#[derive(Copy, Clone)]
struct Tex {
    image: usize,
    rules: ImageRules,
    channels: Channels,
    uv_set: u8,
    wrap: [Wrap; 2],
    color: ImageColor,
    file_wrap: [bool; 2],
    filter: u8,
    outside: [f32; 4],
    missing: [f32; 4],
    uv: Affine2,
    scale: [f32; 4],
    bias: [f32; 4],
    unit: Unit,
}

fn point2(v: &[f32]) -> Vec2 {
    let at = |i: usize| v.get(i).copied().unwrap_or(0.0);
    Vec2::new(at(0), at(1))
}

impl Tex {
    fn tex_ref(&self) -> TexRef {
        TexRef {
            rg_channel: None,
            input: None,
            uv: slot_transform(self.uv),
            image: self.image,
            rules: self.rules,
            channel: self.channel() as u8,
            uv_set: self.uv_set,
            wrap: self.wrap,
            color: self.color,
            file_wrap: self.file_wrap,
            filter: self.filter,
            outside: self.outside,
            missing: self.missing,
            remap: None,
        }
    }

    /// [`Self::tex_ref`] reading `scale * t + bias` on `axes`.
    fn remapped(&self, axes: [usize; 3]) -> TexRef {
        TexRef {
            remap: Some(axes.map(|c| [self.bias[c], self.scale[c]])),
            ..self.tex_ref()
        }
    }

    /// [`Self::tex_ref`] under a `normalmap`.
    fn normalmap_ref(&self) -> TexRef {
        let axis = |c: usize| match (self.bias[c], self.scale[c]) {
            (b, s) if b == -0.5 * s => [b, s],
            (b, s) => [2.0 * b - 1.0, 2.0 * s],
        };
        TexRef {
            remap: Some(std::array::from_fn(axis)),
            ..self.tex_ref()
        }
    }

    /// The reference of a scalar input, with the field's value: the reader's scale, unless a
    /// bias folds it into the texels.
    fn scalar(&self) -> (TexRef, f32) {
        let c = self.channel();
        match self.bias[c] {
            0.0 => (self.tex_ref(), self.scale[c]),
            _ => (self.remapped([c; 3]), 1.0),
        }
    }

    fn channel(&self) -> usize {
        match self.channels {
            Channels::Rgb => 0,
            Channels::One(c) => usize::from(c),
        }
    }
}

enum Value {
    Constant(Vec<f32>, Unit),
    Texture(Tex),
    /// Channels of the vertex color stream, from `first` on.
    Vertex {
        first: u8,
        width: u8,
    },
}

/// The slot an input stands on, which a vertex color stream multiplies.
fn input_slot(input: &str) -> Option<usize> {
    Some(match input {
        "base_color" | "diffuseColor" => slot::BASE_COLOR,
        "base_weight" => slot::BASE_WEIGHT,
        "base_metalness" | "metallic" => slot::METALNESS,
        "base_diffuse_roughness" => slot::DIFFUSE_ROUGHNESS,
        "specular_weight" => slot::SPECULAR_WEIGHT,
        "specular_color" | "specularColor" => slot::SPECULAR_COLOR,
        "specular_roughness" | "roughness" => slot::ROUGHNESS,
        "specular_ior" | "ior" => slot::SPECULAR_IOR,
        "specular_roughness_anisotropy" => slot::ANISOTROPY,
        "transmission_weight" => slot::TRANSMISSION,
        "transmission_color" => slot::TRANSMISSION_COLOR,
        "subsurface_weight" => slot::SUBSURFACE_WEIGHT,
        "subsurface_color" => slot::SUBSURFACE_COLOR,
        "fuzz_weight" => slot::FUZZ_WEIGHT,
        "fuzz_color" => slot::FUZZ_COLOR,
        "fuzz_roughness" => slot::FUZZ_ROUGHNESS,
        "coat_weight" | "clearcoat" => slot::COAT,
        "coat_color" => slot::COAT_COLOR,
        "coat_roughness" | "clearcoatRoughness" => slot::COAT_ROUGHNESS,
        "coat_ior" => slot::COAT_IOR,
        "coat_darkening" => slot::COAT_DARKENING,
        "coat_roughness_anisotropy" => slot::COAT_ANISOTROPY,
        "thin_film_weight" => slot::THIN_FILM_WEIGHT,
        "thin_film_ior" => slot::THIN_FILM_IOR,
        "emission_color" | "emissiveColor" => slot::EMISSION,
        "emission_luminance" => slot::EMISSION_LUMINANCE,
        "geometry_opacity" | "opacity" => slot::OPACITY,
        _ => return None,
    })
}

/// The coordinate `at`, wherever on the mesh it is read.
fn fixed(at: Vec2) -> Affine2 {
    Affine2::from_cols(Vec2::ZERO, Vec2::ZERO, at)
}

/// `t`, between MaterialX coordinates, in `Material::uv_transforms` form, between uvs with the
/// cook's top-left origin.
fn slot_transform(t: Affine2) -> [f32; 6] {
    let flip = Affine2::from_cols(Vec2::X, -Vec2::Y, Vec2::Y);
    let t = flip * t * flip;
    let [a, d, b, e] = t.matrix2.to_cols_array();
    // Adding 0 turns -0 into 0, so equal transforms store equal bytes.
    [a - 1.0, b, t.translation.x, d, e - 1.0, t.translation.y].map(|x| x + 0.0)
}

/// Fails on a connection cycle among the nodes `roots` reach. Every later walk of the
/// connections relies on it, and none of them recurses along a chain, so a long one is safe.
fn acyclic(net: &Network, roots: impl IntoIterator<Item = usize>) -> Result<(), String> {
    #[derive(Copy, Clone, PartialEq)]
    enum Visit {
        New,
        Open,
        Done,
    }
    let upstream = |n: usize| {
        net.nodes[n].inputs.values().filter_map(|i| match i {
            Input::Connect { node, .. } => Some(*node),
            _ => None,
        })
    };
    let mut visits = vec![Visit::New; net.nodes.len()];
    for root in roots {
        if visits[root] != Visit::New {
            continue;
        }
        visits[root] = Visit::Open;
        let mut stack = vec![(root, upstream(root))];
        while let Some((n, next)) = stack.last_mut() {
            let n = *n;
            let Some(u) = next.next() else {
                visits[n] = Visit::Done;
                stack.pop();
                continue;
            };
            match visits[u] {
                Visit::Open => {
                    return Err(format!(
                        "node {} is in a connection cycle",
                        net.nodes[u].name
                    ));
                }
                Visit::New => {
                    visits[u] = Visit::Open;
                    stack.push((u, upstream(u)));
                }
                Visit::Done => {}
            }
        }
    }
    Ok(())
}

/// Maps the surface node (and an optional MaterialX `displacement` node) of `net`.
pub(crate) fn map(
    net: &Network,
    surface: usize,
    displacement: Option<usize>,
    host: &mut Host<'_>,
) -> Result<Mapped, String> {
    acyclic(net, std::iter::once(surface).chain(displacement))?;
    let mut m = Mapper {
        net,
        host,
        // PS `frame:stPrimvarName` and MaterialX `Tworld` both default to the first set.
        material: Material {
            frame_uv: Material::FRAME_UV0,
            ..Material::default()
        },
        textures: MaterialTextures::default(),
        vertex: Vec::new(),
        frames: [None; 2],
        warnings: Vec::new(),
    };
    let node = &net.nodes[surface];
    match node.kind.as_str() {
        "open_pbr_surface" => m.open_pbr(surface)?,
        "UsdPreviewSurface" => m.preview(surface)?,
        k => {
            return Err(format!(
                "{}: surface {k} is not open_pbr_surface or UsdPreviewSurface",
                node.name
            ));
        }
    }
    if let Some(d) = displacement {
        m.displacement(d)?;
    }
    Ok(Mapped {
        material: m.material,
        textures: m.textures,
        vertex: m.vertex,
        warnings: m.warnings,
    })
}

struct Mapper<'a, 'h> {
    net: &'a Network,
    host: &'a mut Host<'h>,
    material: Material,
    textures: MaterialTextures,
    vertex: Vec<VertexRead>,
    /// The uv set the [`BASE`] and [`COAT`] frames follow, where a node names one.
    frames: [Option<f32>; 2],
    warnings: Vec<String>,
}

/// Indices of [`Mapper::frames`].
const BASE: usize = 0;
const COAT: usize = 1;

type Slot = fn(&mut MaterialTextures) -> &mut Option<TexRef>;

/// Slot `S` of [`MaterialTextures::more`].
fn more<const S: usize>(t: &mut MaterialTextures) -> &mut Option<TexRef> {
    t.more.entry(S).or_default()
}

impl Mapper<'_, '_> {
    fn node(&self, n: usize) -> &Node {
        &self.net.nodes[n]
    }

    /// Has `frame` follow uv set `index`, as node `n` says. One frame serves a layer's normal map
    /// and its tangent.
    fn follow(&mut self, n: usize, frame: usize, index: f32) {
        let name = &self.node(n).name;
        if index != 0.0 && index != 1.0 {
            let warning = format!("{name}: uv set {index} is past the two a mesh holds");
            self.warnings.push(warning);
            self.frames[frame].get_or_insert(0.0);
        } else if self.frames[frame].is_some_and(|set| set != index) {
            let warning =
                format!("{name}: reads another uv set than its layer's frame, which stands");
            self.warnings.push(warning);
        } else {
            self.frames[frame] = Some(index);
        }
    }

    /// The uv set of the `tangent` and `bitangent` nodes feeding `normalmap` node `map` for layer
    /// `frame`; by default `Tworld`, index 0. A frame from anything else has no field: meshes
    /// hold one tangent per vertex.
    fn normal_frame(&mut self, map: usize, frame: usize) -> Result<(), String> {
        let mut index = None;
        for (input, kind) in [("tangent", "tangent"), ("bitangent", "bitangent")] {
            let Some(source) = self.source(map, input) else {
                continue;
            };
            match source {
                Input::Connect { node, .. } if self.node(*node).kind == kind => {
                    let node = *node;
                    let i = self.float(node, "index", 0.0)?;
                    if index.is_some_and(|set| set != i) {
                        let name = &self.node(map).name;
                        self.warnings.push(format!(
                            "{name}: its bitangent reads another uv set than its tangent, which \
                             stands"
                        ));
                    }
                    index.get_or_insert(i);
                }
                _ => {
                    let name = &self.node(map).name;
                    self.warnings.push(format!(
                        "{name}: {input} is not a {kind} node; the mesh's own stands in for it"
                    ));
                }
            }
        }
        self.follow(map, frame, index.unwrap_or(0.0));
        Ok(())
    }

    /// Settles the material's frames. The mesh's tangents go to the base's frame, or to the
    /// coat's where nothing reads the base's, and the coat holds its own only where the two part.
    fn settle_frames(&mut self) {
        let m = &self.material;
        let bound = |s: usize| self.textures.more.get(&s).is_some_and(Option::is_some);
        let reads_base =
            self.frames[BASE].is_some() || m.specular_anisotropy > 0.0 || bound(slot::TANGENT);
        let reads_coat =
            self.frames[COAT].is_some() || m.coat_anisotropy > 0.0 || bound(slot::COAT_TANGENT);
        let uv = |set: Option<f32>| {
            if set == Some(1.0) {
                Material::FRAME_UV1
            } else {
                Material::FRAME_UV0
            }
        };
        let (base, coat) = (uv(self.frames[BASE]), uv(self.frames[COAT]));
        let (frame, coat_frame) = match (reads_base, reads_coat) {
            (false, true) => (coat, Material::FRAME_NORMAL_MAP),
            (true, true) if coat != base => (base, coat),
            _ => (base, Material::FRAME_NORMAL_MAP),
        };
        self.material.frame_uv = frame;
        self.material.coat_frame_uv = coat_frame;
    }

    fn input<'n>(&'n self, n: usize, name: &str) -> Option<&'n Input> {
        self.node(n).inputs.get(name)
    }

    /// What input `name` of node `n` reads past `constant` and `dot` nodes.
    fn source<'n>(&'n self, n: usize, name: &str) -> Option<&'n Input> {
        self.input(n, name).map(|i| through(self.net, i))
    }

    /// A constant input's value as authored and its unit, or `None` when unauthored.
    fn measure(&self, n: usize, name: &str) -> Result<Option<(&[f32], Unit)>, String> {
        match self.source(n, name) {
            None => Ok(None),
            Some(Input::Value(v, unit)) => Ok(Some((v, *unit))),
            Some(_) => Err(format!(
                "{}: input {name} must be a constant",
                self.node(n).name
            )),
        }
    }

    /// A constant input's value in the scene's units, or `None` when unauthored.
    fn number(&self, n: usize, name: &str) -> Result<Option<Vec<f32>>, String> {
        let converted = |(v, unit): (&[f32], Unit)| v.iter().map(|&x| unit.scene(x)).collect();
        Ok(self.measure(n, name)?.map(converted))
    }

    /// Meters per `unit` of length input `name` of node `n`.
    fn meters(&self, n: usize, name: &str, unit: Unit) -> Result<f64, String> {
        match unit {
            Unit::Unstated => Ok(self.host.meters_per_unit),
            Unit::Distance(s) => Ok(s),
            Unit::Angle(_) => Err(format!(
                "{} input {name}: a length in a unit of angle",
                self.node(n).name
            )),
        }
    }

    /// A constant angle input in degrees, 0 when unauthored. A connection brings a value here
    /// unchecked against the unit type the input declares.
    fn degrees(&self, n: usize, name: &str) -> Result<f32, String> {
        match self.measure(n, name)? {
            None => Ok(0.0),
            Some((_, Unit::Distance(_))) => Err(format!(
                "{} input {name}: an angle in a unit of length",
                self.node(n).name
            )),
            Some((v, unit)) => Ok(unit.scene(v[0])),
        }
    }

    /// A constant size input, 1 when unauthored, checked as [`Self::degrees`] is.
    fn size(&self, n: usize, name: &str) -> Result<Vec2, String> {
        if let Some((_, unit)) = self.measure(n, name)? {
            self.meters(n, name, unit)?;
        }
        self.vec2(n, name, Vec2::ONE)
    }

    fn text<'n>(&'n self, n: usize, name: &str) -> Option<&'n str> {
        match self.source(n, name) {
            Some(Input::Text(t)) => Some(t),
            _ => None,
        }
    }

    fn float(&self, n: usize, name: &str, default: f32) -> Result<f32, String> {
        Ok(self.number(n, name)?.map_or(default, |v| v[0]))
    }

    fn vec2(&self, n: usize, name: &str, default: Vec2) -> Result<Vec2, String> {
        Ok(self.number(n, name)?.map_or(default, |v| {
            Vec2::new(v[0], v.get(1).copied().unwrap_or(v[0]))
        }))
    }

    /// The row an `extract` node takes of a matrix on its input `in`; `None` for a node that
    /// picks a channel. Nodes `extract`: "the node extracts a row vector by index", and a matrix
    /// value lists its rows in order. An unconnected input is the nodedef's zero matrix.
    fn matrix_row(&mut self, n: usize) -> Result<Option<Value>, String> {
        let node = &self.net.nodes[n];
        if !gives_row(node) {
            return Ok(None);
        }
        let width = if node.ty == "vector3" { 3 } else { 4 };
        let index = self.float(n, "index", 0.0)?;
        if index.fract() != 0.0 || !(0.0..width as f32).contains(&index) {
            return Err(format!(
                "{}: row {index} of a matrix of {width} rows",
                node.name
            ));
        }
        let no_matrix = || format!("{}: a row of {width} reads no matrix", node.name);
        // Refusing a pick here keeps a chain of them from recursing.
        if let Some(Input::Connect { node: up, .. }) = self.source(n, "in")
            && is_pick(self.node(*up))
        {
            return Err(no_matrix());
        }
        match self.value(n, "in")? {
            None => Ok(Some(Value::Constant(vec![0.0; width], Unit::Unstated))),
            Some(Value::Constant(v, unit)) if v.len() == width * width => {
                let row = v[index as usize * width..][..width].to_vec();
                Ok(Some(Value::Constant(row, unit)))
            }
            Some(_) => Err(no_matrix()),
        }
    }

    /// The channels of the stream on input `in` of an `extract` node.
    fn components(&self, n: usize) -> Result<usize, String> {
        match self.input(n, "in") {
            None => Ok(4),
            Some(Input::Value(v, _)) if v.len() <= 4 => Ok(v.len()),
            Some(Input::Connect { node, output }) => Ok(match self.node(*node).ty.as_str() {
                "float" => 1,
                "vector2" => 2,
                "color3" | "vector3" => 3,
                "" if output == "rgb" => 3,
                _ => 4,
            }),
            Some(_) => Err(format!(
                "{}: extract reads a color or vector",
                self.node(n).name
            )),
        }
    }

    /// What input `name` of node `n` resolves to; `None` when unauthored.
    fn value(&mut self, n: usize, name: &str) -> Result<Option<Value>, String> {
        let what = format!("{} input {name}", self.node(n).name);
        let v = match self.source(n, name) {
            None => return Ok(None),
            Some(Input::Value(v, unit)) => Value::Constant(v.clone(), *unit),
            Some(Input::Text(_) | Input::Image { .. }) => {
                return Err(format!("{what}: not a number"));
            }
            Some(Input::Connect { node, output }) => {
                let (node, output) = (*node, output.clone());
                self.connected(node, &output)
                    .map_err(|e| format!("{what}: {e}"))?
            }
        };
        Ok(Some(v))
    }

    /// The value output `output` of node `n` gives.
    fn connected(&mut self, n: usize, output: &str) -> Result<Value, String> {
        let node = &self.net.nodes[n];
        let channel = |out: &str| match out {
            "outr" | "outx" | "r" => Some(0),
            "outg" | "outy" | "g" => Some(1),
            "outb" | "outz" | "b" => Some(2),
            "outa" | "outw" | "a" => Some(3),
            _ => None,
        };
        match node.kind.as_str() {
            "image" | "tiledimage" => {
                let mut t = self.image(n)?;
                t.channels = match node.ty.as_str() {
                    "float" => Channels::One(0),
                    _ => Channels::Rgb,
                };
                if let Some(Input::Image {
                    color_space: Some(space),
                    ..
                }) = self.source(n, "file")
                    && is_color(&node.ty)
                    && ImageColor::named(space).is_none()
                {
                    let warning = format!("{}: colorspace {space} reads as data", node.name);
                    self.warnings.push(warning);
                }
                Ok(Value::Texture(t))
            }
            "UsdUVTexture" => {
                let mut t = self.uv_texture(n)?;
                t.channels = match output {
                    "rgb" => Channels::Rgb,
                    o => Channels::One(
                        channel(o).ok_or_else(|| format!("{}: no output {o}", node.name))?,
                    ),
                };
                Ok(Value::Texture(t))
            }
            "extract" | "separate2" | "separate3" | "separate4" => {
                if node.kind == "extract"
                    && let Some(row) = self.matrix_row(n)?
                {
                    return Ok(row);
                }
                let c = match node.kind.as_str() {
                    "extract" => {
                        let width = self.components(n)?;
                        let index = self.float(n, "index", 0.0)?;
                        if index.fract() != 0.0 || !(0.0..width as f32).contains(&index) {
                            return Err(format!(
                                "{}: index {index} of a stream of {width} channels",
                                node.name
                            ));
                        }
                        index as u8
                    }
                    kind => {
                        let width = match kind {
                            "separate2" => 2,
                            "separate3" => 3,
                            _ => 4,
                        };
                        channel(output)
                            .filter(|c| *c < width)
                            .ok_or_else(|| format!("{}: no output {output}", node.name))?
                    }
                };
                let name = node.name.clone();
                // A picked channel holds none to pick, though a matrix's row holds several.
                // Refusing it here keeps a chain of picks from recursing.
                if let Some(Input::Connect { node, .. }) = self.source(n, "in")
                    && is_pick(self.node(*node))
                    && !gives_row(self.node(*node))
                {
                    return Err(format!("{name}: picks channel {c} of a single channel"));
                }
                match self.value(n, "in")? {
                    Some(Value::Texture(mut t)) if t.channels == Channels::Rgb => {
                        t.channels = Channels::One(c);
                        Ok(Value::Texture(t))
                    }
                    Some(Value::Constant(v, unit)) => Ok(Value::Constant(
                        vec![v.get(usize::from(c)).copied().unwrap_or(0.0)],
                        unit,
                    )),
                    Some(Value::Vertex { first, width }) if c < width => Ok(Value::Vertex {
                        first: first + c,
                        width: 1,
                    }),
                    None => Ok(Value::Constant(vec![0.0], Unit::Unstated)),
                    _ => Err(format!(
                        "{name}: picks channel {c} of nothing a texture holds"
                    )),
                }
            }
            k if k.starts_with("UsdPrimvarReader_") => {
                let name = self.text(n, "varname").unwrap_or_default().to_string();
                self.primvar_value(n, &name, "fallback")
            }
            "geompropvalue" => {
                let name = self.text(n, "geomprop").unwrap_or_default().to_string();
                self.primvar_value(n, &name, "default")
            }
            "geomcolor" => {
                let width = match node.ty.as_str() {
                    "float" => 1,
                    "color4" => 4,
                    _ => 3,
                };
                let index = color_index(node);
                if index != 0.0 {
                    let message = format!(
                        "{}: color {index} is past the one a mesh holds; it reads 0",
                        node.name
                    );
                    self.warnings.push(message);
                    return Ok(Value::Constant(vec![0.0; width], Unit::Unstated));
                }
                if self.host.any_mesh {
                    return Ok(Value::Vertex {
                        first: 0,
                        width: width as u8,
                    });
                }
                self.primvar_value(n, DISPLAY_COLOR, "default")
            }
            k => Err(format!(
                "node {} ({k}) is not a constant, image or texture channel pick",
                node.name
            )),
        }
    }

    /// What the mesh holds for primvar `name`.
    fn held(&self, name: &str) -> Primvar {
        let read = self.host.reads.iter().find(|r| r.0 == name);
        read.map_or(Primvar::Missing, |r| r.1.clone())
    }

    /// What the mesh holds for the primvar of reader `n`.
    fn fetch(&self, n: usize) -> Primvar {
        self.held(self.text(n, "varname").unwrap_or_default())
    }

    /// The value of primvar `name` to reader `n`, whose input `missing` stands in where the mesh
    /// holds none.
    fn primvar_value(&mut self, n: usize, name: &str, missing: &str) -> Result<Value, String> {
        match self.held(name) {
            Primvar::Constant(v) => Ok(Value::Constant(v, Unit::Unstated)),
            Primvar::Missing => Ok(Value::Constant(
                self.number(n, missing)?.unwrap_or_else(|| vec![0.0; 4]),
                Unit::Unstated,
            )),
            Primvar::Uv(_) => Err(format!(
                "{}: {name} is a uv set, which only texture coordinates read",
                self.node(n).name
            )),
            Primvar::Varying { width, scale } => {
                let fallback = self.fallback(n, missing);
                let first = self.vertex_channels(n, name, width, scale, fallback)?;
                Ok(Value::Vertex { first, width })
            }
        }
    }

    /// What input `missing` of reader `n` gives each channel of an element a mesh doesn't hold:
    /// a single number all of them, as a float `default` does for an image.
    fn fallback(&mut self, n: usize, missing: &str) -> [f32; 4] {
        match self.number(n, missing) {
            Ok(None) => [0.0; 4],
            Ok(Some(v)) if v.len() == 1 => [v[0]; 4],
            Ok(Some(v)) => std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0)),
            Err(e) => {
                self.warnings.push(format!("{e}; 0 stands in for it"));
                [0.0; 4]
            }
        }
    }

    /// The first channel of the vertex color stream that holds primvar `name`, with `fallback`
    /// where the mesh holds none of it.
    fn vertex_channels(
        &mut self,
        n: usize,
        name: &str,
        width: u8,
        scale: Option<f32>,
        fallback: [f32; 4],
    ) -> Result<u8, String> {
        let same = |r: &&VertexRead| r.name == name && r.width == width;
        let exact = |r: &&VertexRead| same(r) && r.fallback == fallback;
        if let Some(r) = self.vertex.iter().find(exact) {
            return Ok(r.first);
        }
        let first = self.vertex.iter().map(|r| r.width).sum::<u8>();
        if first + width > 4 {
            if let Some(first) = self.vertex.iter().find(same).map(|r| r.first) {
                let reader = &self.node(n).name;
                self.warnings.push(format!(
                    "{reader}: no channel is left for its fallback, so {name} reads another \
                     reader's"
                ));
                return Ok(first);
            }
            return Err(format!(
                "{}: {name} takes {width} channels of the vertex color stream's 4, {first} of \
                 which other primvars of the material hold",
                self.node(n).name
            ));
        }
        self.vertex.push(VertexRead {
            name: name.to_string(),
            first,
            width,
            fallback,
            scale: scale.map_or(1.0, |s| {
                s.max(stream_scale(
                    fallback.iter().fold(0.0, |p, &x| p.max(f64::from(x.abs()))),
                ))
            }),
        });
        Ok(first)
    }

    /// Input `input` of `n` multiplied by channels of the vertex color stream from `first` on.
    fn vertex_slot(&mut self, n: usize, input: &str, first: u8) -> Result<(), String> {
        let s = input_slot(input).ok_or_else(|| {
            format!(
                "{} input {input} varies over the mesh, which no slot reads",
                self.node(n).name
            )
        })?;
        self.material.textures[s].vertex = first + 1;
        Ok(())
    }

    /// A MaterialX `image` or `tiledimage` node, its channels left for the caller.
    fn image(&mut self, n: usize) -> Result<Tex, String> {
        let (image, color_space, unit) = match self.source(n, "file") {
            Some(Input::Image {
                image,
                color_space,
                unit,
            }) => (*image, color_space.clone(), *unit),
            None => (self.no_file(n), None, Unit::Unstated),
            Some(_) => return Err(format!("{}: file is not a filename", self.node(n).name)),
        };
        let node = self.node(n);
        let wrap = |mode: Option<&str>| match mode {
            None | Some("periodic") => Ok(Wrap::Repeat),
            Some("clamp") => Ok(Wrap::Clamp),
            Some("constant") => Ok(Wrap::Constant),
            Some("mirror") => Ok(Wrap::Mirror),
            Some(m) => Err(format!("{}: address mode {m}", node.name)),
        };
        let wrap = [
            wrap(self.text(n, "uaddressmode"))?,
            wrap(self.text(n, "vaddressmode"))?,
        ];
        // Color management covers color types, and a document without a space has none.
        let color = color_space
            .as_deref()
            .filter(|_| is_color(&node.ty))
            .and_then(ImageColor::named)
            .unwrap_or(ImageColor::Data);
        // Past the edges under `constant` the default must read; where only a missing file would
        // need it, one that isn't a constant gives the zero of no default.
        let default = match self.number(n, "default") {
            Err(e) if wrap.contains(&Wrap::Constant) => return Err(e),
            d => d.unwrap_or_default().unwrap_or_default(),
        };
        // A float default holds whichever channel of the file a slot reads.
        let missing = match default[..] {
            [v] => [v; 4],
            ref v => std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0)),
        };
        let outside = if wrap.contains(&Wrap::Constant) {
            missing
        } else {
            [0.0; 4]
        };
        let filter = match self.text(n, "filtertype") {
            Some("closest") => filter::MAG_NEAREST | filter::MIN_NEAREST | filter::MIP_NEAREST,
            _ => 0,
        };
        let (uv_set, mut uv) = self.texcoord(n, "texcoord", (0, Affine2::IDENTITY))?;
        if node.kind == "tiledimage" {
            let tiling = self.vec2(n, "uvtiling", Vec2::ONE)?;
            let offset = self.vec2(n, "uvoffset", Vec2::ZERO)?;
            let image = self.size(n, "realworldimagesize")?;
            let tile = self.size(n, "realworldtilesize")?;
            uv = Affine2::from_scale(tile / image)
                * Affine2::from_translation(-offset)
                * Affine2::from_scale(tiling)
                * uv;
        }
        Ok(Tex {
            image,
            rules: ImageRules::MaterialX,
            channels: Channels::Rgb,
            uv_set,
            wrap,
            color,
            file_wrap: [false; 2],
            filter,
            outside,
            missing,
            uv,
            scale: [1.0; 4],
            bias: [0.0; 4],
            unit,
        })
    }

    /// A `UsdUVTexture` node, its channels left for the caller.
    fn uv_texture(&mut self, n: usize) -> Result<Tex, String> {
        let (image, color_space) = match self.node(n).inputs.get("file") {
            Some(Input::Image {
                image, color_space, ..
            }) => (*image, color_space.clone()),
            None => (self.no_file(n), None),
            Some(_) => return Err(format!("{}: file is not an asset", self.node(n).name)),
        };
        let node = self.node(n);
        let wrap = |mode: Option<&str>| match mode {
            None | Some("black" | "useMetadata") => Ok(Wrap::Border),
            Some("repeat") => Ok(Wrap::Repeat),
            Some("clamp") => Ok(Wrap::Clamp),
            Some("mirror") => Ok(Wrap::Mirror),
            Some(m) => Err(format!("{}: wrap mode {m}", node.name)),
        };
        let modes = [self.text(n, "wrapS"), self.text(n, "wrapT")];
        let wrap = [wrap(modes[0])?, wrap(modes[1])?];
        let file_wrap = modes.map(|m| matches!(m, None | Some("useMetadata")));
        let stated = match self.text(n, "sourceColorSpace") {
            Some(s @ ("sRGB" | "raw")) => Some(s),
            _ => color_space.as_deref(),
        };
        let color = match stated {
            Some("sRGB") => ImageColor::Named(Space::SRGB),
            s => s.and_then(ImageColor::named).unwrap_or(ImageColor::UsdAuto),
        };
        let four = |v: Option<Vec<f32>>, d: [f32; 4]| {
            v.map_or(d, |v| {
                std::array::from_fn(|i| v.get(i).copied().unwrap_or(d[i]))
            })
        };
        let scale = four(self.number(n, "scale")?, [1.0; 4]);
        let bias = four(self.number(n, "bias")?, [0.0; 4]);
        // Only a file that can't be read needs the fallback, so one that isn't a constant gives
        // its default.
        let missing = four(
            self.number(n, "fallback").unwrap_or_default(),
            [0.0, 0.0, 0.0, 1.0],
        );
        let (uv_set, uv) = self.texcoord(n, "st", (0, fixed(Vec2::ZERO)))?;
        Ok(Tex {
            image,
            rules: ImageRules::Usd,
            channels: Channels::Rgb,
            uv_set,
            wrap,
            color,
            file_wrap,
            filter: 0,
            outside: [0.0; 4],
            missing,
            uv,
            scale,
            bias,
            unit: Unit::Unstated,
        })
    }

    /// The image of texture node `n`, which names no file.
    fn no_file(&mut self, n: usize) -> usize {
        let name = self.node(n).name.clone();
        (self.host.add)(ImageFile::unread(name, "it names no file".into()))
    }

    /// A texture of `rgb`, for a constant input read through a slot.
    fn solid(&mut self, rgb: [f32; 3]) -> Tex {
        Tex {
            image: (self.host.add)(solid_image(rgb)),
            rules: ImageRules::Usd,
            channels: Channels::Rgb,
            uv_set: 0,
            wrap: [Wrap::Repeat; 2],
            color: ImageColor::Data,
            file_wrap: [false; 2],
            filter: 0,
            outside: [0.0; 4],
            missing: [0.0; 4],
            uv: Affine2::IDENTITY,
            scale: [1.0; 4],
            bias: [0.0; 4],
            unit: Unit::Unstated,
        }
    }

    /// The uv set and transform coordinate input `name` of `n` gives; `unset` when unauthored.
    fn texcoord(
        &self,
        n: usize,
        name: &str,
        unset: (u8, Affine2),
    ) -> Result<(u8, Affine2), String> {
        match self.source(n, name) {
            None => Ok(unset),
            Some(Input::Connect { node, .. }) => self.coordinates(*node),
            Some(_) => Ok((0, fixed(self.vec2(n, name, Vec2::ZERO)?))),
        }
    }

    /// The uv set and transform a texture-coordinate node gives.
    fn coordinates(&self, mut n: usize) -> Result<(u8, Affine2), String> {
        // The transforms walked so far, of the coordinates the next node gives.
        let mut outer = Affine2::IDENTITY;
        loop {
            let node = self.node(n);
            let (transform, input, unset) = match node.kind.as_str() {
                "texcoord" => {
                    return match self.float(n, "index", 0.0)? {
                        0.0 => Ok((0, outer)),
                        1.0 => Ok((1, outer)),
                        i => Err(format!(
                            "{}: texcoord index {i}; uv sets are 0 and 1",
                            node.name
                        )),
                    };
                }
                "geompropvalue" => {
                    let name = self.text(n, "geomprop").unwrap_or_default();
                    let set = self
                        .host
                        .primvars
                        .iter()
                        .position(|p| *p == name)
                        .map(|s| s as u8)
                        .or((name == "UV0").then_some(0))
                        .or(self.net.uv_sets.get(name).copied())
                        .filter(|s| *s < 2);
                    match set {
                        Some(s) => return Ok((s, outer)),
                        None => (Affine2::IDENTITY, "default", (0, fixed(Vec2::ZERO))),
                    }
                }
                "UsdPrimvarReader_float2" => {
                    return match self.fetch(n) {
                        Primvar::Uv(set) => Ok((set, outer)),
                        Primvar::Constant(v) => Ok((0, outer * fixed(point2(&v)))),
                        Primvar::Missing => {
                            let fallback = self.vec2(n, "fallback", Vec2::ZERO)?;
                            Ok((0, outer * fixed(fallback)))
                        }
                        Primvar::Varying { .. } => Err(format!(
                            "{}: reads {}; the mesh uv sets are {}",
                            node.name,
                            self.text(n, "varname").unwrap_or_default(),
                            self.host.primvars.join(", "),
                        )),
                    };
                }
                "place2d" => {
                    let pivot = self.vec2(n, "pivot", Vec2::ZERO)?;
                    let scale = self.vec2(n, "scale", Vec2::ONE)?;
                    let offset = self.vec2(n, "offset", Vec2::ZERO)?;
                    // MaterialX rotate2d turns clockwise by its angle in degrees.
                    let rotate = Affine2::from_angle(-self.degrees(n, "rotate")?.to_radians());
                    let srt = self.float(n, "operationorder", 0.0)? == 0.0;
                    let to = Affine2::from_translation(-pivot);
                    let back = Affine2::from_translation(pivot);
                    let s = Affine2::from_scale(scale.recip());
                    let t = Affine2::from_translation(-offset);
                    let place = if srt {
                        back * t * rotate * s * to
                    } else {
                        back * s * rotate * t * to
                    };
                    (place, "texcoord", (0, fixed(Vec2::ZERO)))
                }
                "UsdTransform2d" => {
                    let scale = self.vec2(n, "scale", Vec2::ONE)?;
                    let rotation = self.degrees(n, "rotation")?.to_radians();
                    let translation = self.vec2(n, "translation", Vec2::ZERO)?;
                    let t = Affine2::from_scale_angle_translation(scale, rotation, translation);
                    (t, "in", (0, fixed(Vec2::ZERO)))
                }
                k => {
                    return Err(format!(
                        "node {} ({k}) is not a texture coordinate reader or transform",
                        node.name
                    ));
                }
            };
            outer *= transform;
            match self.source(n, input) {
                Some(Input::Connect { node, .. }) => n = *node,
                _ => {
                    let (set, uv) = self.texcoord(n, input, unset)?;
                    return Ok((set, outer * uv));
                }
            }
        }
    }

    /// A scalar field: a constant sets it, a texture sets it to the reader's scale and
    /// binds the slot.
    fn scalar(
        &mut self,
        n: usize,
        input: &str,
        field: fn(&mut Material) -> &mut f32,
        slot: Option<Slot>,
    ) -> Result<(), String> {
        match self.value(n, input)? {
            None => {}
            Some(Value::Constant(v, unit)) => *field(&mut self.material) = unit.scene(v[0]),
            Some(Value::Texture(t)) => {
                let slot = slot.ok_or_else(|| self.no_slot(n, input))?;
                if t.channels == Channels::Rgb {
                    return Err(format!(
                        "{} input {input}: a scalar reads one channel",
                        self.node(n).name
                    ));
                }
                let (r, value) = t.scalar();
                *field(&mut self.material) = t.unit.scene(value);
                *slot(&mut self.textures) = Some(r);
            }
            Some(Value::Vertex { first, .. }) => {
                *field(&mut self.material) = 1.0;
                self.vertex_slot(n, input, first)?;
            }
        }
        Ok(())
    }

    /// A length field, in meters. Its default is a length in the scene's unit as well.
    fn length(
        &mut self,
        n: usize,
        input: &str,
        field: fn(&mut Material) -> &mut f32,
    ) -> Result<(), String> {
        let (v, unit) = match self.value(n, input)? {
            None => (*field(&mut self.material), Unit::Unstated),
            Some(Value::Constant(v, unit)) => (v[0], unit),
            Some(Value::Texture(_) | Value::Vertex { .. }) => return Err(self.no_slot(n, input)),
        };
        *field(&mut self.material) = scaled(v, self.meters(n, input, unit)?);
        Ok(())
    }

    /// A color field, set or bound like [`Self::scalar`].
    fn color(
        &mut self,
        n: usize,
        input: &str,
        field: fn(&mut Material) -> &mut [f32; 3],
        slot: Option<Slot>,
    ) -> Result<(), String> {
        match self.value(n, input)? {
            None => {}
            Some(Value::Constant(v, _)) => {
                *field(&mut self.material) =
                    std::array::from_fn(|i| v.get(i).copied().unwrap_or(v[0]));
            }
            Some(Value::Texture(t)) => {
                let slot = slot.ok_or_else(|| self.no_slot(n, input))?;
                if t.channels != Channels::Rgb {
                    return Err(format!(
                        "{} input {input}: a color reads RGB",
                        self.node(n).name
                    ));
                }
                let (r, value) = if t.bias[..3].iter().any(|&b| b != 0.0) {
                    // Texels store 0..1, so the field holds what the input reaches past that.
                    let peak = [0, 1, 2].map(|c| t.bias[c].max(t.bias[c] + t.scale[c]).max(1.0));
                    let remap = [0, 1, 2].map(|c| [t.bias[c] / peak[c], t.scale[c] / peak[c]]);
                    let r = TexRef {
                        remap: Some(remap),
                        ..t.tex_ref()
                    };
                    (r, peak)
                } else {
                    (t.tex_ref(), [t.scale[0], t.scale[1], t.scale[2]])
                };
                *field(&mut self.material) = value;
                *slot(&mut self.textures) = Some(r);
            }
            Some(Value::Vertex { first, width }) => {
                if width < 3 {
                    return Err(format!(
                        "{} input {input}: a color reads RGB",
                        self.node(n).name
                    ));
                }
                *field(&mut self.material) = [1.0; 3];
                self.vertex_slot(n, input, first)?;
            }
        }
        Ok(())
    }

    fn no_slot(&self, n: usize, input: &str) -> String {
        format!(
            "{} input {input} is textured; the material has no texture slot for it",
            self.node(n).name
        )
    }

    fn open_pbr(&mut self, n: usize) -> Result<(), String> {
        type F = fn(&mut Material) -> &mut f32;
        type C = fn(&mut Material) -> &mut [f32; 3];
        let scalars: &[(&str, F, Option<Slot>)] = &[
            (
                "base_weight",
                |m| &mut m.base_weight,
                Some(more::<{ slot::BASE_WEIGHT }>),
            ),
            (
                "base_diffuse_roughness",
                |m| &mut m.diffuse_roughness,
                Some(more::<{ slot::DIFFUSE_ROUGHNESS }>),
            ),
            (
                "base_metalness",
                |m| &mut m.metalness,
                Some(|t| &mut t.metalness),
            ),
            (
                "specular_weight",
                |m| &mut m.specular_weight,
                Some(|t| &mut t.specular),
            ),
            (
                "specular_roughness",
                |m| &mut m.specular_roughness,
                Some(|t| &mut t.roughness),
            ),
            (
                "specular_ior",
                |m| &mut m.specular_ior,
                Some(more::<{ slot::SPECULAR_IOR }>),
            ),
            (
                "specular_roughness_anisotropy",
                |m| &mut m.specular_anisotropy,
                Some(more::<{ slot::ANISOTROPY }>),
            ),
            (
                "transmission_weight",
                |m| &mut m.transmission_weight,
                Some(|t| &mut t.transmission),
            ),
            (
                "transmission_scatter_anisotropy",
                |m| &mut m.transmission_scatter_anisotropy,
                None,
            ),
            (
                "transmission_dispersion_scale",
                |m| &mut m.transmission_dispersion_scale,
                None,
            ),
            (
                "transmission_dispersion_abbe_number",
                |m| &mut m.transmission_dispersion_abbe_number,
                None,
            ),
            (
                "subsurface_weight",
                |m| &mut m.subsurface_weight,
                Some(more::<{ slot::SUBSURFACE_WEIGHT }>),
            ),
            (
                "subsurface_scatter_anisotropy",
                |m| &mut m.subsurface_scatter_anisotropy,
                Some(more::<{ slot::SUBSURFACE_ANISOTROPY }>),
            ),
            (
                "fuzz_weight",
                |m| &mut m.fuzz_weight,
                Some(more::<{ slot::FUZZ_WEIGHT }>),
            ),
            (
                "fuzz_roughness",
                |m| &mut m.fuzz_roughness,
                Some(more::<{ slot::FUZZ_ROUGHNESS }>),
            ),
            ("coat_weight", |m| &mut m.coat_weight, Some(|t| &mut t.coat)),
            (
                "coat_roughness",
                |m| &mut m.coat_roughness,
                Some(more::<{ slot::COAT_ROUGHNESS }>),
            ),
            (
                "coat_ior",
                |m| &mut m.coat_ior,
                Some(more::<{ slot::COAT_IOR }>),
            ),
            (
                "coat_darkening",
                |m| &mut m.coat_darkening,
                Some(more::<{ slot::COAT_DARKENING }>),
            ),
            (
                "coat_roughness_anisotropy",
                |m| &mut m.coat_anisotropy,
                Some(more::<{ slot::COAT_ANISOTROPY }>),
            ),
            (
                "thin_film_weight",
                |m| &mut m.thin_film_weight,
                Some(more::<{ slot::THIN_FILM_WEIGHT }>),
            ),
            (
                "thin_film_ior",
                |m| &mut m.thin_film_ior,
                Some(more::<{ slot::THIN_FILM_IOR }>),
            ),
            (
                "emission_luminance",
                |m| &mut m.emission_luminance,
                Some(more::<{ slot::EMISSION_LUMINANCE }>),
            ),
            (
                "geometry_opacity",
                |m| &mut m.geometry_opacity,
                Some(|t| &mut t.opacity),
            ),
        ];
        let colors: &[(&str, C, Option<Slot>)] = &[
            (
                "base_color",
                |m| &mut m.base_color,
                Some(|t| &mut t.base_color),
            ),
            (
                "specular_color",
                |m| &mut m.specular_color,
                Some(more::<{ slot::SPECULAR_COLOR }>),
            ),
            (
                "transmission_color",
                |m| &mut m.transmission_color,
                Some(more::<{ slot::TRANSMISSION_COLOR }>),
            ),
            (
                "transmission_scatter",
                |m| &mut m.transmission_scatter,
                None,
            ),
            (
                "subsurface_color",
                |m| &mut m.subsurface_color,
                Some(|t| &mut t.subsurface_color),
            ),
            (
                "subsurface_radius_scale",
                |m| &mut m.subsurface_radius_scale,
                Some(more::<{ slot::SUBSURFACE_RADIUS_SCALE }>),
            ),
            (
                "fuzz_color",
                |m| &mut m.fuzz_color,
                Some(more::<{ slot::FUZZ_COLOR }>),
            ),
            (
                "coat_color",
                |m| &mut m.coat_color,
                Some(more::<{ slot::COAT_COLOR }>),
            ),
            (
                "emission_color",
                |m| &mut m.emission_color,
                Some(|t| &mut t.emissive),
            ),
        ];
        let names: Vec<String> = self.node(n).inputs.keys().cloned().collect();
        for name in &names {
            let known = scalars.iter().any(|s| s.0 == name)
                || colors.iter().any(|c| c.0 == name)
                || matches!(
                    name.as_str(),
                    "transmission_depth"
                        | "subsurface_radius"
                        | "thin_film_thickness"
                        | "geometry_thin_walled"
                        | "geometry_normal"
                        | "geometry_coat_normal"
                        | "geometry_tangent"
                        | "geometry_coat_tangent"
                );
            if !known {
                return Err(format!(
                    "{}: open_pbr_surface has no input {name}",
                    self.node(n).name
                ));
            }
        }
        for &(name, field, slot) in scalars {
            self.scalar(n, name, field, slot)?;
        }
        for &(name, field, slot) in colors {
            self.color(n, name, field, slot)?;
        }
        self.length(n, "transmission_depth", |m| &mut m.transmission_depth)?;
        self.textured_length(n, "subsurface_radius", slot::SUBSURFACE_RADIUS, |m| {
            &mut m.subsurface_radius
        })?;
        self.film_thickness(n)?;
        if let Some(v) = self.number(n, "geometry_thin_walled")? {
            self.material.thin_walled = u8::from(v[0] != 0.0);
        }
        if self
            .textures
            .more
            .get(&slot::TRANSMISSION_COLOR)
            .is_some_and(Option::is_some)
            && self.material.transmission_depth > 0.0
            && self.material.thin_walled == 0
        {
            self.warnings.push(format!(
                "{}: an interior medium is one for the whole material, so a transmission_color \
                 texture reads its factor under a transmission_depth above 0",
                self.node(n).name
            ));
        }
        if self.textures.opacity.is_some()
            || self.material.geometry_opacity < 1.0
            || self.material.textures[slot::OPACITY].vertex != 0
        {
            self.material.alpha_mode = Material::BLEND;
        }
        self.textures.normal =
            self.normal_input(n, "geometry_normal", Material::OBJECT_NORMAL, BASE)?;
        self.textures.coat_normal = self.normal_input(
            n,
            "geometry_coat_normal",
            Material::OBJECT_COAT_NORMAL,
            COAT,
        )?;
        self.tangent_input(n, "geometry_tangent", slot::TANGENT, BASE)?;
        self.tangent_input(n, "geometry_coat_tangent", slot::COAT_TANGENT, COAT)?;
        self.settle_frames();
        Ok(())
    }

    /// The normal map of input `name` of `n`, from a `normalmap` node. A 1.38 `space` of `object`
    /// sets `object` in [`Material::object_normals`]: as in the library's upgrade of it, the node
    /// reads `2 in - 1` and no tangent frame or `scale` has a part.
    fn normal_input(
        &mut self,
        n: usize,
        name: &str,
        object: u8,
        frame: usize,
    ) -> Result<Option<(TexRef, [f32; 2])>, String> {
        match self.source(n, name) {
            None => Ok(None),
            Some(Input::Connect { node, .. }) if self.node(*node).kind == "normalmap" => {
                let map = *node;
                let scale = if self.text(map, "space") == Some("object") {
                    self.material.object_normals |= object;
                    Vec2::ONE
                } else {
                    self.normal_frame(map, frame)?;
                    self.vec2(map, "scale", Vec2::ONE)?
                };
                match self.value(map, "in")? {
                    Some(Value::Texture(t)) if t.channels == Channels::Rgb => {
                        Ok(Some((t.normalmap_ref(), scale.to_array())))
                    }
                    _ => Err(format!("{}: in must read an image", self.node(map).name)),
                }
            }
            Some(_) => Err(format!(
                "{}: {name} must come from a normalmap node",
                self.node(n).name
            )),
        }
    }

    /// Tangent input `name` of `n`: a `tangent` node is the mesh's own in its `index`'s uv set
    /// for layer `frame`, and a texture a direction in the tangent plane. Anything else stands
    /// for no frame a mesh holds.
    fn tangent_input(
        &mut self,
        n: usize,
        name: &str,
        s: usize,
        frame: usize,
    ) -> Result<(), String> {
        let source = match self.source(n, name) {
            None => return Ok(()),
            Some(Input::Connect { node, .. }) => Some(*node),
            Some(_) => None,
        };
        if let Some(node) = source.filter(|&node| self.node(node).kind == "tangent") {
            let index = self.float(node, "index", 0.0)?;
            self.follow(node, frame, index);
            return Ok(());
        }
        let texture = source
            .is_some_and(|node| matches!(self.node(node).kind.as_str(), "image" | "tiledimage"));
        match texture.then(|| self.value(n, name)).transpose()?.flatten() {
            Some(Value::Texture(t)) if t.channels == Channels::Rgb => {
                self.textures.more.insert(s, Some(t.tex_ref()));
            }
            _ => {
                let warning = format!(
                    "{}: {name} is neither a tangent node nor a texture; the mesh's tangent \
                     stands in for it",
                    self.node(n).name
                );
                self.warnings.push(warning);
            }
        }
        Ok(())
    }

    /// A length input with a slot: [`Self::length`], or a texture that multiplies the field.
    fn textured_length(
        &mut self,
        n: usize,
        input: &str,
        s: usize,
        field: fn(&mut Material) -> &mut f32,
    ) -> Result<(), String> {
        match self.value(n, input)? {
            Some(Value::Texture(t)) => {
                if t.channels == Channels::Rgb {
                    return Err(format!(
                        "{} input {input}: a scalar reads one channel",
                        self.node(n).name
                    ));
                }
                let (r, value) = t.scalar();
                *field(&mut self.material) = scaled(value, self.meters(n, input, t.unit)?);
                self.textures.more.insert(s, Some(r));
                Ok(())
            }
            _ => self.length(n, input, field),
        }
    }

    /// `thin_film_thickness`, in nanometers, whose values without a unit are in micrometers.
    fn film_thickness(&mut self, n: usize) -> Result<(), String> {
        let input = "thin_film_thickness";
        let nanometers = |this: &Self, unit| match unit {
            Unit::Unstated => Ok(1e3),
            unit => Ok::<_, String>(this.meters(n, input, unit)? * 1e9),
        };
        match self.value(n, input)? {
            None => self.material.thin_film_thickness = 500.0,
            Some(Value::Constant(v, unit)) => {
                self.material.thin_film_thickness = scaled(v[0], nanometers(self, unit)?);
            }
            Some(Value::Texture(t)) => {
                if t.channels == Channels::Rgb {
                    return Err(format!(
                        "{} input {input}: a scalar reads one channel",
                        self.node(n).name
                    ));
                }
                let (r, value) = t.scalar();
                self.material.thin_film_thickness = scaled(value, nanometers(self, t.unit)?);
                self.textures
                    .more
                    .insert(slot::THIN_FILM_THICKNESS, Some(r));
            }
            Some(Value::Vertex { first, .. }) => {
                self.material.thin_film_thickness = scaled(1.0, nanometers(self, Unit::Unstated)?);
                self.material.textures[slot::THIN_FILM_THICKNESS].vertex = first + 1;
            }
        }
        Ok(())
    }

    fn preview(&mut self, n: usize) -> Result<(), String> {
        let m = &mut self.material;
        m.model = Material::MODEL_PREVIEW;
        m.coat_darkening = 0.0;
        m.base_color = [0.18; 3];
        m.metalness = 0.0;
        m.specular_roughness = 0.5;
        m.emission_color = [0.0; 3];
        self.color(
            n,
            "diffuseColor",
            |m| &mut m.base_color,
            Some(|t| &mut t.base_color),
        )?;
        self.color(
            n,
            "emissiveColor",
            |m| &mut m.emission_color,
            Some(|t| &mut t.emissive),
        )?;
        let emits = self.textures.emissive.is_some()
            || self.material.emission_color.iter().any(|&c| c > 0.0);
        self.material.emission_luminance = if emits { 1.0 } else { 0.0 };
        let specular = self
            .number(n, "useSpecularWorkflow")?
            .is_some_and(|v| v[0] != 0.0);
        self.scalar(
            n,
            "ior",
            |m| &mut m.specular_ior,
            Some(more::<{ slot::SPECULAR_IOR }>),
        )?;
        let ior = self.material.specular_ior;
        if specular {
            self.material.model = Material::MODEL_PREVIEW_SPECULAR;
            self.material.specular_color = [0.0; 3];
            self.color(
                n,
                "specularColor",
                |m| &mut m.specular_color,
                Some(more::<{ slot::SPECULAR_COLOR }>),
            )?;
        } else {
            self.scalar(
                n,
                "metallic",
                |m| &mut m.metalness,
                Some(|t| &mut t.metalness),
            )?;
        }
        self.scalar(
            n,
            "roughness",
            |m| &mut m.specular_roughness,
            Some(|t| &mut t.roughness),
        )?;
        self.scalar(
            n,
            "clearcoat",
            |m| &mut m.coat_weight,
            Some(|t| &mut t.coat),
        )?;
        if self.material.coat_weight > 0.0 || self.textures.coat.is_some() {
            self.material.coat_roughness = 0.01;
            self.scalar(
                n,
                "clearcoatRoughness",
                |m| &mut m.coat_roughness,
                Some(more::<{ slot::COAT_ROUGHNESS }>),
            )?;
            // "including the clearcoat if clearcoat > 0": what `ior` reads holds for both.
            self.material.coat_ior = ior;
            let shared = self
                .textures
                .more
                .get(&slot::SPECULAR_IOR)
                .copied()
                .flatten();
            self.textures.more.insert(slot::COAT_IOR, shared);
            self.material.textures[slot::COAT_IOR].vertex =
                self.material.textures[slot::SPECULAR_IOR].vertex;
            self.material.coat_color = [1.0; 3];
        }
        self.preview_opacity(n)?;
        match self.value(n, "normal")? {
            None => {}
            Some(Value::Constant(v, _)) => {
                let at = |i: usize, d: f32| v.get(i).copied().unwrap_or(d);
                let normal = [at(0, 0.0), at(1, 0.0), at(2, 1.0)];
                if normal != [0.0, 0.0, 1.0] {
                    let t = self.solid(normal.map(|c| 0.5 * c + 0.5));
                    self.textures.normal = Some((t.tex_ref(), [1.0; 2]));
                }
            }
            Some(Value::Vertex { .. }) => return Err(self.no_slot(n, "normal")),
            Some(Value::Texture(t)) => {
                if t.channels != Channels::Rgb {
                    return Err(format!("{}: a normal texture reads rgb", self.node(n).name));
                }
                // An axis of `s (2 t - 1)` is the tangent-space encoding at strength `s`, which
                // stands for a signed image's values too.
                let strength =
                    |c: usize| (t.bias[c] == -0.5 * t.scale[c]).then(|| 0.5 * t.scale[c]);
                self.textures.normal = Some(match (strength(0), strength(1), strength(2)) {
                    (Some(x), Some(y), Some(1.0)) => (t.tex_ref(), [x, y]),
                    _ => (t.remapped([0, 1, 2]), [1.0; 2]),
                });
            }
        }
        match self.value(n, "displacement")? {
            None => Ok(()),
            Some(height) => self.displaced(n, height, 1.0),
        }
    }

    /// `opacity`, as `opacityThreshold` and `opacityMode` have it.
    fn preview_opacity(&mut self, n: usize) -> Result<(), String> {
        let threshold = self.number(n, "opacityThreshold")?.map_or(0.0, |v| v[0]);
        if threshold > 0.0 || self.text(n, "opacityMode") == Some("presence") {
            self.scalar(
                n,
                "opacity",
                |m| &mut m.geometry_opacity,
                Some(|t| &mut t.opacity),
            )?;
            if threshold > 0.0 {
                self.material.alpha_mode = Material::MASK;
                self.material.alpha_cutoff = threshold;
            } else if self.textures.opacity.is_some()
                || self.material.geometry_opacity < 1.0
                || self.material.textures[slot::OPACITY].vertex != 0
            {
                self.material.alpha_mode = Material::BLEND;
            }
            return Ok(());
        }
        match self.value(n, "opacity")? {
            None => {}
            Some(Value::Constant(v, _)) if v[0] >= 1.0 => {}
            Some(Value::Constant(v, _)) => {
                self.material.transmission_weight = 1.0 - v[0].max(0.0);
                self.material.thin_walled = 1;
            }
            Some(Value::Texture(t)) if t.channels == Channels::Rgb => {
                return Err(format!(
                    "{} input opacity: a scalar reads one channel",
                    self.node(n).name
                ));
            }
            Some(Value::Texture(t)) => {
                let c = t.channel();
                self.textures.transmission = Some(TexRef {
                    remap: Some([[1.0 - t.bias[c], -t.scale[c]]; 3]),
                    ..t.tex_ref()
                });
                self.material.transmission_weight = 1.0;
                self.material.thin_walled = 1;
            }
            Some(Value::Vertex { first, .. }) => {
                let binding = &mut self.material.textures[slot::TRANSMISSION];
                binding.vertex = first + 1;
                binding.vertex_complement = 1;
                self.material.transmission_weight = 1.0;
                self.material.thin_walled = 1;
            }
        }
        Ok(())
    }

    /// A MaterialX `displacement` node.
    fn displacement(&mut self, n: usize) -> Result<(), String> {
        let node = self.node(n);
        if node.kind != "displacement" {
            return Err(format!(
                "displacement node {} ({}) is not displacement",
                node.name, node.kind
            ));
        }
        let scale = self.float(n, "scale", 1.0)?;
        let height = self.value(n, "displacement")?;
        match height {
            None => Ok(()),
            // Nodes: "in tangent/normal space using (dPdu, dPdv, N) coordinates".
            Some(Value::Constant(v, unit)) if v.len() > 1 && v.iter().any(|&c| c != 0.0) => {
                let along =
                    |c: usize| Value::Constant(vec![v.get(c).copied().unwrap_or(0.0)], unit);
                self.tangent_height(n, 0, along(0), scale)?;
                self.tangent_height(n, 1, along(1), scale)?;
                self.normal_height(n, along(2), scale)
            }
            Some(Value::Texture(t)) if t.channels == Channels::Rgb => {
                let channel = |c: u8| {
                    Value::Texture(Tex {
                        channels: Channels::One(c),
                        ..t
                    })
                };
                self.tangent_height(n, 0, channel(0), scale)?;
                self.tangent_height(n, 1, channel(1), scale)?;
                self.normal_height(n, channel(2), scale)
            }
            Some(h) => self.displaced(n, h, scale),
        }
    }

    /// `height` times `scale` along tangent `axis`, 0 for u and 1 for v.
    fn tangent_height(
        &mut self,
        n: usize,
        axis: usize,
        height: Value,
        scale: f32,
    ) -> Result<(), String> {
        let s = [slot::DISPLACEMENT_U, slot::DISPLACEMENT_V][axis];
        let (t, heights) = self.height(n, height, scale)?;
        self.material.displacement_tangent[axis] = heights;
        self.textures.more.insert(s, Some(t.tex_ref()));
        Ok(())
    }

    /// [`Self::displaced`] where a zero stands too: the tangents read beside it.
    fn normal_height(&mut self, n: usize, height: Value, scale: f32) -> Result<(), String> {
        let (t, [scale, offset, lo, hi]) = self.height(n, height, scale)?;
        let m = &mut self.material;
        m.displacement_world = 1;
        m.displacement_scale = scale;
        m.displacement_offset = offset;
        m.displacement_min = lo;
        m.displacement_max = hi;
        self.textures.displacement = Some(t.tex_ref());
        Ok(())
    }

    /// Sets the displacement of node `n` to `height` times `scale`, in meters. A constant height
    /// reads a solid image at the offset.
    fn displaced(&mut self, n: usize, height: Value, scale: f32) -> Result<(), String> {
        match height {
            Value::Constant(v, _) if v[0] == 0.0 => Ok(()),
            height => self.normal_height(n, height, scale),
        }
    }

    /// What reads `height` times `scale`, in meters: its texture, and the texture's scale and
    /// offset, then the lowest and highest it reaches.
    fn height(&mut self, n: usize, height: Value, scale: f32) -> Result<(Tex, [f32; 4]), String> {
        let (t, unit, heights) = match height {
            Value::Constant(v, unit) => (self.solid([0.0; 3]), unit, (0.0, v[0])),
            Value::Texture(t) => {
                let c = t.channel();
                (t, t.unit, (t.scale[c], t.bias[c]))
            }
            Value::Vertex { .. } => return Err(self.no_slot(n, "displacement")),
        };
        let meters = f64::from(scale) * self.meters(n, "displacement", unit)?;
        let (scale, offset) = (scaled(heights.0, meters), scaled(heights.1, meters));
        let (lo, hi) = (offset.min(offset + scale), offset.max(offset + scale));
        Ok((t, [scale, offset, lo, hi]))
    }
}
