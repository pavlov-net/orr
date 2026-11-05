//! Shading networks, as UsdShade and MaterialX documents author them, mapped onto
//! [`Material`] and its textures.
//!
//! Two surfaces map:
//! - MaterialX `open_pbr_surface`, input for field: `base_metalness` is `metalness`,
//!   `base_diffuse_roughness` `diffuse_roughness`, `specular_roughness_anisotropy`
//!   `specular_anisotropy`, `geometry_thin_walled` `thin_walled`, `thin_film_thickness`
//!   (micrometers) the field in nanometers when `thin_film_weight` is above 0 (0 otherwise),
//!   and every other input the field of its name. `coat_darkening`,
//!   `coat_roughness_anisotropy` and the coat normal and tangents have no field and are not
//!   read. A `geometry_opacity` below 1 or textured blends, as the linear presence weight
//!   OpenPBR defines.
//! - `UsdPreviewSurface`, mapped like glTF metallic-roughness: `diffuseColor` is
//!   `base_color`, `metallic`, `roughness`, `ior` and `emissiveColor` (read as nits) their
//!   fields, `clearcoat` and `clearcoatRoughness` the coat at `ior`, `occlusion` the occlusion
//!   texture, and `normal` and `displacement` their textures, a constant reading a solid image
//!   ([`solid_image`]). `opacityThreshold` above 0 alpha-tests `opacity` at it; otherwise
//!   `opacityMode` `presence` blends an opacity below 1 or textured, and `transparent` makes
//!   the rest of it thin-walled transmission, which leaves specular and emission at full
//!   weight. The specular workflow's `specularColor` is F0: `specular_ior` gives its largest
//!   channel and `specular_color` the others, so a colored F0 tints the edge as well.
//!
//! An input is a constant, or reads a texture through these nodes: MaterialX `image` and
//! `tiledimage` (with `texcoord`, `geompropvalue` and `place2d` on their coordinates,
//! `extract`, `separate2`, `separate3` or `separate4` picking a channel, `normalmap` under
//! `geometry_normal`), and `UsdUVTexture` (with `UsdPrimvarReader_float2` and
//! `UsdTransform2d` on `st`, whose default is the constant (0, 0)). `constant` and `dot` nodes
//! pass on what they read to MaterialX nodes. A primvar reader gives the value of a primvar
//! constant over the mesh, and its `fallback` where the mesh has no such primvar; a varying one
//! other than a uv set fails, as meshes hold no values for it. Any other node fails the import,
//! naming it, as does a connection cycle. A `geompropvalue` whose property no uv set holds reads
//! its `default` at every point, as an unconnected `place2d` reads 0, 0. Textures go to the
//! material's slots (base color, opacity, roughness, metalness, normal, occlusion, emission,
//! specular weight, subsurface color, coat, transmission, displacement); a textured input
//! without a slot fails. A texture value multiplies the field, which takes the reader's scale
//! (`UsdUVTexture` `scale`; 1 otherwise); a bias folds scale and bias into the texels
//! ([`TexRef::remap`]) and leaves the field 1, or for a color what it reaches past 1. Normals
//! read `scale * t + bias` per axis; a pair of the form `s (2 t - 1)` is the tangent-space
//! encoding at strength `s`. A `normalmap` decodes the reader's `v` as `2 v - 1`, where such a
//! pair is the encoding stated, not a second decode. Both hold as far as the image's format can
//! hold what they state. Coordinate nodes set each slot's `Material::uv_transforms`,
//! conjugated by the flip of t between st and the cook's top-left uvs.

use std::collections::{BTreeMap, HashMap};
use std::sync::LazyLock;

use orr_math::{Affine2, Vec2};
use orr_pack::material::Material;

use crate::texture::Wrap;
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
    Value(Vec<f32>),
    Text(String),
    /// A file input: an image of the scene and its authored color space.
    Image {
        image: usize,
        color_space: Option<String>,
    },
    Connect {
        node: usize,
        output: String,
    },
}

/// What a `constant` or `dot` without a value reads.
static ZERO: LazyLock<Input> = LazyLock::new(|| Input::Value(vec![0.0]));

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
        input = node.inputs.get(name).unwrap_or(&ZERO);
    }
    input
}

impl Network {
    /// The primvars the network's readers name.
    pub(crate) fn primvar_reads(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .nodes
            .iter()
            .filter(|n| n.kind.starts_with("UsdPrimvarReader_"))
            .filter_map(|n| match n.inputs.get("varname") {
                Some(Input::Text(t)) if !t.is_empty() => Some(t.clone()),
                _ => None,
            })
            .collect();
        names.sort_unstable();
        names.dedup();
        names
    }
}

/// What a mesh holds for a primvar a reader names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Primvar {
    /// One of its uv sets.
    Uv(u8),
    /// One value for the whole mesh.
    Constant(Vec<f32>),
    /// Values that vary over the mesh.
    Varying,
    Missing,
}

/// What mapping asks of the scene a material lands in.
pub(crate) struct Host<'a> {
    /// The mesh primvar of each uv set, which geometric properties name.
    pub(crate) primvars: &'a [&'a str],
    /// What the mesh holds for each primvar of [`Network::primvar_reads`].
    pub(crate) reads: &'a [(String, Primvar)],
    /// Adds a [`solid_image`] of a color to the scene; returns its index.
    pub(crate) solid: &'a mut dyn FnMut([f32; 3]) -> usize,
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

/// The MaterialX category of a node definition id: `ND_image_color3` is `image`.
pub(crate) fn category(id: &str) -> &str {
    const KNOWN: &[&str] = &[
        "open_pbr_surface",
        "surfacematerial",
        "geompropvalue",
        "displacement",
        "tiledimage",
        "normalmap",
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
    srgb: Option<bool>,
    uv: Affine2,
    scale: [f32; 4],
    bias: [f32; 4],
}

fn point2(v: &[f32]) -> Vec2 {
    let at = |i: usize| v.get(i).copied().unwrap_or(0.0);
    Vec2::new(at(0), at(1))
}

impl Tex {
    fn tex_ref(&self) -> TexRef {
        TexRef {
            rg_channel: None,
            uv: slot_transform(self.uv),
            image: self.image,
            rules: self.rules,
            channel: self.channel() as u8,
            uv_set: self.uv_set,
            wrap: self.wrap,
            srgb: self.srgb,
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
    Constant(Vec<f32>),
    Texture(Tex),
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
        material: Material::default(),
        textures: MaterialTextures::default(),
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
        warnings: m.warnings,
    })
}

struct Mapper<'a, 'h> {
    net: &'a Network,
    host: &'a mut Host<'h>,
    material: Material,
    textures: MaterialTextures,
    warnings: Vec<String>,
}

type Slot = fn(&mut MaterialTextures) -> &mut Option<TexRef>;

impl Mapper<'_, '_> {
    fn node(&self, n: usize) -> &Node {
        &self.net.nodes[n]
    }

    fn input<'n>(&'n self, n: usize, name: &str) -> Option<&'n Input> {
        self.node(n).inputs.get(name)
    }

    /// What input `name` of node `n` reads past `constant` and `dot` nodes.
    fn source<'n>(&'n self, n: usize, name: &str) -> Option<&'n Input> {
        self.input(n, name).map(|i| through(self.net, i))
    }

    /// A constant input's value, or `None` when unauthored.
    fn number(&self, n: usize, name: &str) -> Result<Option<Vec<f32>>, String> {
        match self.source(n, name) {
            None => Ok(None),
            Some(Input::Value(v)) => Ok(Some(v.clone())),
            Some(_) => Err(format!(
                "{}: input {name} must be a constant",
                self.node(n).name
            )),
        }
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

    /// The channels of the stream on input `in` of an `extract` node.
    fn components(&self, n: usize) -> Result<usize, String> {
        match self.input(n, "in") {
            None => Ok(4),
            Some(Input::Value(v)) if v.len() <= 4 => Ok(v.len()),
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
            Some(Input::Value(v)) => Value::Constant(v.clone()),
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
        let node = self.node(n);
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
                // A picked channel holds none to pick, and refusing it here keeps a chain of
                // picks from recursing.
                if let Some(Input::Connect { node, .. }) = self.source(n, "in")
                    && matches!(
                        self.node(*node).kind.as_str(),
                        "extract" | "separate2" | "separate3" | "separate4"
                    )
                {
                    return Err(format!("{name}: picks channel {c} of a single channel"));
                }
                match self.value(n, "in")? {
                    Some(Value::Texture(mut t)) if t.channels == Channels::Rgb => {
                        t.channels = Channels::One(c);
                        Ok(Value::Texture(t))
                    }
                    Some(Value::Constant(v)) => Ok(Value::Constant(vec![
                        v.get(usize::from(c)).copied().unwrap_or(0.0),
                    ])),
                    None => Ok(Value::Constant(vec![0.0])),
                    _ => Err(format!(
                        "{name}: picks channel {c} of nothing a texture holds"
                    )),
                }
            }
            k if k.starts_with("UsdPrimvarReader_") => match self.fetch(n) {
                Primvar::Constant(v) => Ok(Value::Constant(v)),
                Primvar::Missing => Ok(Value::Constant(
                    self.number(n, "fallback")?.unwrap_or_else(|| vec![0.0; 4]),
                )),
                Primvar::Uv(_) | Primvar::Varying => Err(format!(
                    "{}: its primvar varies over the mesh, which holds no values for it",
                    node.name
                )),
            },
            k => Err(format!(
                "node {} ({k}) is not a constant, image or texture channel pick",
                node.name
            )),
        }
    }

    /// What the mesh holds for the primvar of reader `n`.
    fn fetch(&self, n: usize) -> Primvar {
        let name = self.text(n, "varname").unwrap_or_default();
        let read = self.host.reads.iter().find(|r| r.0 == name);
        read.map_or(Primvar::Missing, |r| r.1.clone())
    }

    /// A MaterialX `image` or `tiledimage` node, its channels left for the caller.
    fn image(&self, n: usize) -> Result<Tex, String> {
        let node = self.node(n);
        let Some(Input::Image { image, color_space }) = self.source(n, "file") else {
            return Err(format!("{}: no file", node.name));
        };
        let wrap = |mode: Option<&str>| match mode {
            None | Some("periodic") => Ok(Wrap::Repeat),
            Some("clamp" | "constant") => Ok(Wrap::Clamp),
            Some("mirror") => Ok(Wrap::Mirror),
            Some(m) => Err(format!("{}: address mode {m}", node.name)),
        };
        let wrap = [
            wrap(self.text(n, "uaddressmode"))?,
            wrap(self.text(n, "vaddressmode"))?,
        ];
        let srgb = Some(color_space.as_deref() == Some("srgb_texture"));
        let (uv_set, mut uv) = self.texcoord(n, "texcoord", (0, Affine2::IDENTITY))?;
        if node.kind == "tiledimage" {
            let tiling = self.vec2(n, "uvtiling", Vec2::ONE)?;
            let offset = self.vec2(n, "uvoffset", Vec2::ZERO)?;
            let image = self.vec2(n, "realworldimagesize", Vec2::ONE)?;
            let tile = self.vec2(n, "realworldtilesize", Vec2::ONE)?;
            uv = Affine2::from_scale(tile / image)
                * Affine2::from_translation(-offset)
                * Affine2::from_scale(tiling)
                * uv;
        }
        Ok(Tex {
            image: *image,
            rules: ImageRules::MaterialX,
            channels: Channels::Rgb,
            uv_set,
            wrap,
            srgb,
            uv,
            scale: [1.0; 4],
            bias: [0.0; 4],
        })
    }

    /// A `UsdUVTexture` node, its channels left for the caller.
    fn uv_texture(&self, n: usize) -> Result<Tex, String> {
        let node = self.node(n);
        let Some(Input::Image { image, .. }) = node.inputs.get("file") else {
            return Err(format!("{}: no file", node.name));
        };
        let wrap = |mode: Option<&str>| match mode {
            None | Some("repeat" | "useMetadata") => Ok(Wrap::Repeat),
            Some("clamp" | "black") => Ok(Wrap::Clamp),
            Some("mirror") => Ok(Wrap::Mirror),
            Some(m) => Err(format!("{}: wrap mode {m}", node.name)),
        };
        let wrap = [wrap(self.text(n, "wrapS"))?, wrap(self.text(n, "wrapT"))?];
        let srgb = match self.text(n, "sourceColorSpace") {
            Some("sRGB") => Some(true),
            Some("raw") => Some(false),
            _ => None,
        };
        let four = |v: Option<Vec<f32>>, d: f32| {
            v.map_or([d; 4], |v| {
                std::array::from_fn(|i| v.get(i).copied().unwrap_or(d))
            })
        };
        let scale = four(self.number(n, "scale")?, 1.0);
        let bias = four(self.number(n, "bias")?, 0.0);
        let (uv_set, uv) = self.texcoord(n, "st", (0, fixed(Vec2::ZERO)))?;
        Ok(Tex {
            image: *image,
            rules: ImageRules::Usd,
            channels: Channels::Rgb,
            uv_set,
            wrap,
            srgb,
            uv,
            scale,
            bias,
        })
    }

    /// A texture of `rgb`, for a constant input read through a slot.
    fn solid(&mut self, rgb: [f32; 3]) -> Tex {
        Tex {
            image: (self.host.solid)(rgb),
            rules: ImageRules::Usd,
            channels: Channels::Rgb,
            uv_set: 0,
            wrap: [Wrap::Repeat; 2],
            srgb: Some(false),
            uv: Affine2::IDENTITY,
            scale: [1.0; 4],
            bias: [0.0; 4],
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
                        Primvar::Varying => Err(format!(
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
                    let rotate = Affine2::from_angle(-self.float(n, "rotate", 0.0)?.to_radians());
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
                    let rotation = self.float(n, "rotation", 0.0)?.to_radians();
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
            Some(Value::Constant(v)) => *field(&mut self.material) = v[0],
            Some(Value::Texture(t)) => {
                let slot = slot.ok_or_else(|| self.no_slot(n, input))?;
                if t.channels == Channels::Rgb {
                    return Err(format!(
                        "{} input {input}: a scalar reads one channel",
                        self.node(n).name
                    ));
                }
                let (r, value) = t.scalar();
                *field(&mut self.material) = value;
                *slot(&mut self.textures) = Some(r);
            }
        }
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
            Some(Value::Constant(v)) => {
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
            ("base_weight", |m| &mut m.base_weight, None),
            ("base_diffuse_roughness", |m| &mut m.diffuse_roughness, None),
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
            ("specular_ior", |m| &mut m.specular_ior, None),
            (
                "specular_roughness_anisotropy",
                |m| &mut m.specular_anisotropy,
                None,
            ),
            (
                "transmission_weight",
                |m| &mut m.transmission_weight,
                Some(|t| &mut t.transmission),
            ),
            ("transmission_depth", |m| &mut m.transmission_depth, None),
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
            ("subsurface_weight", |m| &mut m.subsurface_weight, None),
            ("subsurface_radius", |m| &mut m.subsurface_radius, None),
            (
                "subsurface_scatter_anisotropy",
                |m| &mut m.subsurface_scatter_anisotropy,
                None,
            ),
            ("fuzz_weight", |m| &mut m.fuzz_weight, None),
            ("fuzz_roughness", |m| &mut m.fuzz_roughness, None),
            ("coat_weight", |m| &mut m.coat_weight, Some(|t| &mut t.coat)),
            ("coat_roughness", |m| &mut m.coat_roughness, None),
            ("coat_ior", |m| &mut m.coat_ior, None),
            ("thin_film_ior", |m| &mut m.thin_film_ior, None),
            ("emission_luminance", |m| &mut m.emission_luminance, None),
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
            ("specular_color", |m| &mut m.specular_color, None),
            ("transmission_color", |m| &mut m.transmission_color, None),
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
                None,
            ),
            ("fuzz_color", |m| &mut m.fuzz_color, None),
            ("coat_color", |m| &mut m.coat_color, None),
            (
                "emission_color",
                |m| &mut m.emission_color,
                Some(|t| &mut t.emissive),
            ),
        ];
        const UNREAD: &[&str] = &[
            "coat_darkening",
            "coat_roughness_anisotropy",
            "geometry_coat_normal",
            "geometry_tangent",
            "geometry_coat_tangent",
        ];
        let names: Vec<String> = self.node(n).inputs.keys().cloned().collect();
        for name in &names {
            let known = scalars.iter().any(|s| s.0 == name)
                || colors.iter().any(|c| c.0 == name)
                || UNREAD.contains(&name.as_str())
                || matches!(
                    name.as_str(),
                    "thin_film_weight"
                        | "thin_film_thickness"
                        | "geometry_thin_walled"
                        | "geometry_normal"
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
        let film = self.number(n, "thin_film_weight")?.map_or(0.0, |v| v[0]);
        let thickness = self.number(n, "thin_film_thickness")?.map_or(0.5, |v| v[0]);
        self.material.thin_film_thickness = if film > 0.0 { thickness * 1000.0 } else { 0.0 };
        if let Some(v) = self.number(n, "geometry_thin_walled")? {
            self.material.thin_walled = u8::from(v[0] != 0.0);
        }
        if self.textures.opacity.is_some() || self.material.geometry_opacity < 1.0 {
            self.material.alpha_mode = Material::BLEND;
        }
        match self.source(n, "geometry_normal") {
            None => {}
            Some(Input::Connect { node, .. }) if self.node(*node).kind == "normalmap" => {
                let map = *node;
                if self.text(map, "space") == Some("object") {
                    self.warnings.push(format!(
                        "{}: a normalmap space of object is not honored yet; it reads as \
                         tangent space",
                        self.node(map).name
                    ));
                }
                let scale = self.vec2(map, "scale", Vec2::ONE)?;
                match self.value(map, "in")? {
                    Some(Value::Texture(t)) if t.channels == Channels::Rgb => {
                        self.textures.normal = Some((t.normalmap_ref(), scale.to_array()));
                    }
                    _ => {
                        return Err(format!("{}: in must read an image", self.node(map).name));
                    }
                }
            }
            Some(_) => {
                return Err(format!(
                    "{}: geometry_normal must come from a normalmap node",
                    self.node(n).name
                ));
            }
        }
        Ok(())
    }

    fn preview(&mut self, n: usize) -> Result<(), String> {
        let m = &mut self.material;
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
        self.scalar(n, "ior", |m| &mut m.specular_ior, None)?;
        let ior = self.material.specular_ior;
        if specular {
            self.color(n, "specularColor", |m| &mut m.specular_color, None)?;
            let authored = self.input(n, "specularColor").is_some();
            let f0 = if authored {
                self.material.specular_color
            } else {
                [0.0; 3]
            };
            // Past this the IOR leaves what a float holds.
            let peak = f0.into_iter().fold(0.0, f32::max).clamp(0.0, 0.9999);
            let r = peak.sqrt();
            self.material.specular_ior = (1.0 + r) / (1.0 - r);
            self.material.specular_color = match peak {
                0.0 => [1.0; 3],
                _ => f0.map(|c| c.max(0.0) / peak),
            };
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
            self.scalar(n, "clearcoatRoughness", |m| &mut m.coat_roughness, None)?;
            self.material.coat_ior = ior;
            self.material.coat_color = [1.0; 3];
        }
        self.preview_opacity(n)?;
        match self.value(n, "occlusion")? {
            None | Some(Value::Constant(_)) => {}
            Some(Value::Texture(t)) => {
                self.textures.occlusion = Some((t.scalar().0, 1.0));
            }
        }
        match self.value(n, "normal")? {
            None => {}
            Some(Value::Constant(v)) => {
                let at = |i: usize, d: f32| v.get(i).copied().unwrap_or(d);
                let normal = [at(0, 0.0), at(1, 0.0), at(2, 1.0)];
                if normal != [0.0, 0.0, 1.0] {
                    let t = self.solid(normal.map(|c| 0.5 * c + 0.5));
                    self.textures.normal = Some((t.tex_ref(), [1.0; 2]));
                }
            }
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
            None => {}
            Some(Value::Constant(v)) if v[0] == 0.0 => {}
            Some(Value::Constant(v)) => {
                let t = self.solid([0.0; 3]);
                self.displaced(t, 0.0, v[0]);
            }
            Some(Value::Texture(t)) => {
                let c = t.channel();
                self.displaced(t, t.scale[c], t.bias[c]);
            }
        }
        Ok(())
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
            } else if self.textures.opacity.is_some() || self.material.geometry_opacity < 1.0 {
                self.material.alpha_mode = Material::BLEND;
            }
            return Ok(());
        }
        match self.value(n, "opacity")? {
            None => {}
            Some(Value::Constant(v)) if v[0] >= 1.0 => {}
            Some(Value::Constant(v)) => {
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
        let scale = self.number(n, "scale")?.map_or(1.0, |v| v[0]);
        match self.value(n, "displacement")? {
            None => Ok(()),
            Some(Value::Texture(t)) if t.channels != Channels::Rgb => {
                self.displaced(t, scale, 0.0);
                Ok(())
            }
            _ => Err(format!(
                "{}: displacement must read one channel of an image",
                self.node(n).name
            )),
        }
    }

    fn displaced(&mut self, t: Tex, scale: f32, offset: f32) {
        let m = &mut self.material;
        m.displacement_scale = scale;
        m.displacement_offset = offset;
        m.displacement_min = offset.min(offset + scale);
        m.displacement_max = offset.max(offset + scale);
        self.textures.displacement = Some(t.tex_ref());
    }
}
