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
//!   read. A textured `geometry_opacity` is alpha-tested at 0.5, a constant one below 1
//!   blended.
//! - `UsdPreviewSurface`, mapped like glTF metallic-roughness: `diffuseColor` is
//!   `base_color`, `metallic`, `roughness`, `ior` and `emissiveColor` (read as nits) their
//!   fields, `clearcoat` and `clearcoatRoughness` the coat at IOR 1.5, `occlusion` the
//!   occlusion texture, and `displacement` the displacement texture with its reader's scale
//!   and bias. `opacityThreshold` above 0 alpha-tests `opacity` at it; otherwise an opacity
//!   below 1 or textured blends. The specular workflow's `specularColor` becomes
//!   `specular_color` at metalness 0.
//!
//! An input is a constant, or reads a texture through these nodes: MaterialX `image` (with
//! `texcoord`, `geompropvalue` and `place2d` on its coordinates, `extract`, `separate3` or
//! `separate4` picking a channel, `normalmap` under `geometry_normal`, `constant`), and
//! `UsdUVTexture` (with `UsdPrimvarReader_float2` and `UsdTransform2d`). Any other node
//! fails the import, naming it. Textures go to the material's slots (base color, opacity,
//! roughness, metalness, normal, occlusion, emission, specular weight, subsurface color,
//! coat, transmission, displacement); a textured input without a slot fails. A texture
//! value multiplies the field, which takes the reader's scale (`UsdUVTexture` `scale`; 1
//! otherwise); bias is refused except under normals (`UsdUVTexture` scale 2 and bias -1,
//! the tangent-space encoding, with x or y flipped by scale -2 and bias 1) and
//! displacement. The uv transform of `place2d` or `UsdTransform2d` has no material field:
//! the importer applies it to the mesh uvs, so every texture of one material must share it.

use std::collections::BTreeMap;

use orr_math::{Affine2, Vec2};
use orr_pack::material::Material;

use crate::texture::Wrap;
use crate::texture::material::{MaterialTextures, TexRef};

/// A shading network: nodes and the connections between their inputs and outputs.
#[derive(Default)]
pub(crate) struct Network {
    pub nodes: Vec<Node>,
}

pub(crate) struct Node {
    /// Prim path or document name, for messages.
    pub name: String,
    /// MaterialX category (`image`, `open_pbr_surface`) or UsdShade id (`UsdUVTexture`).
    pub kind: String,
    /// MaterialX output type (`float`, `color3`); empty for UsdShade nodes.
    pub ty: String,
    pub inputs: BTreeMap<String, Input>,
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

/// The MaterialX category of a node definition id: `ND_image_color3` is `image`.
pub(crate) fn category(id: &str) -> &str {
    const KNOWN: &[&str] = &[
        "open_pbr_surface",
        "surfacematerial",
        "geompropvalue",
        "displacement",
        "normalmap",
        "separate3",
        "separate4",
        "texcoord",
        "constant",
        "extract",
        "place2d",
        "image",
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
    pub material: Material,
    pub textures: MaterialTextures,
    /// The uv transform every texture shares, when it is not the identity.
    pub uv: Option<Affine2>,
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
    channels: Channels,
    uv_set: u8,
    wrap: [Wrap; 2],
    srgb: Option<bool>,
    uv: Affine2,
    scale: [f32; 4],
    bias: [f32; 4],
}

impl Tex {
    fn tex_ref(&self) -> TexRef {
        TexRef {
            image: self.image,
            channel: self.channel() as u8,
            uv_set: self.uv_set,
            wrap: self.wrap,
            srgb: self.srgb,
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

/// Maps the surface node (and an optional MaterialX `displacement` node) of `net`.
/// `uv_set` names the uv set a primvar or geometric property reads.
pub(crate) fn map(
    net: &Network,
    surface: usize,
    displacement: Option<usize>,
    uv_set: &dyn Fn(&str) -> Result<u8, String>,
) -> Result<Mapped, String> {
    let mut m = Mapper {
        net,
        uv_set,
        material: Material::default(),
        textures: MaterialTextures::default(),
        uv: Vec::new(),
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
    let first = m.uv.first().copied();
    if let Some(t) = first
        && m.uv.iter().any(|u| !u.abs_diff_eq(t, 1e-6))
    {
        return Err(format!(
            "{}: its textures use different uv transforms; one material holds one",
            node.name
        ));
    }
    Ok(Mapped {
        material: m.material,
        textures: m.textures,
        uv: first.filter(|t| !t.abs_diff_eq(Affine2::IDENTITY, 1e-6)),
    })
}

struct Mapper<'a> {
    net: &'a Network,
    uv_set: &'a dyn Fn(&str) -> Result<u8, String>,
    material: Material,
    textures: MaterialTextures,
    /// The uv transform of every texture read.
    uv: Vec<Affine2>,
}

type Slot = fn(&mut MaterialTextures) -> &mut Option<TexRef>;

impl Mapper<'_> {
    fn node(&self, n: usize) -> &Node {
        &self.net.nodes[n]
    }

    fn input<'n>(&'n self, n: usize, name: &str) -> Option<&'n Input> {
        self.node(n).inputs.get(name)
    }

    /// A constant input's value, or `None` when unauthored.
    fn number(&self, n: usize, name: &str) -> Result<Option<Vec<f32>>, String> {
        match self.input(n, name) {
            None => Ok(None),
            Some(Input::Value(v)) => Ok(Some(v.clone())),
            Some(Input::Connect { node, .. }) if self.node(*node).kind == "constant" => {
                self.number(*node, "value")
            }
            Some(_) => Err(format!(
                "{}: input {name} must be a constant",
                self.node(n).name
            )),
        }
    }

    fn text<'n>(&'n self, n: usize, name: &str) -> Option<&'n str> {
        match self.input(n, name) {
            Some(Input::Text(t)) => Some(t),
            _ => None,
        }
    }

    /// What input `name` of node `n` resolves to; `None` when unauthored.
    fn value(&mut self, n: usize, name: &str) -> Result<Option<Value>, String> {
        let what = format!("{} input {name}", self.node(n).name);
        let v = match self.input(n, name) {
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
        if let Value::Texture(t) = &v {
            self.uv.push(t.uv);
        }
        Ok(Some(v))
    }

    /// The value output `output` of node `n` gives.
    fn connected(&mut self, n: usize, output: &str) -> Result<Value, String> {
        let node = self.node(n);
        let channel = |out: &str| match out {
            "outr" | "r" => Some(0),
            "outg" | "g" => Some(1),
            "outb" | "b" => Some(2),
            "outa" | "a" => Some(3),
            _ => None,
        };
        match node.kind.as_str() {
            "constant" => Ok(Value::Constant(
                self.number(n, "value")?.unwrap_or_else(|| vec![0.0]),
            )),
            "image" => {
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
            "extract" | "separate3" | "separate4" => {
                let c = match node.kind.as_str() {
                    "extract" => self
                        .number(n, "index")?
                        .map_or(0, |v| v.first().copied().unwrap_or(0.0) as u8),
                    _ => channel(output)
                        .ok_or_else(|| format!("{}: no output {output}", node.name))?,
                };
                let name = node.name.clone();
                match self.value(n, "in")? {
                    Some(Value::Texture(mut t)) if t.channels == Channels::Rgb && c < 4 => {
                        t.channels = Channels::One(c);
                        Ok(Value::Texture(t))
                    }
                    Some(Value::Constant(v)) => Ok(Value::Constant(vec![
                        v.get(usize::from(c)).copied().unwrap_or(0.0),
                    ])),
                    _ => Err(format!(
                        "{name}: picks channel {c} of nothing a texture holds"
                    )),
                }
            }
            k => Err(format!(
                "node {} ({k}) is not a constant, image or texture channel pick",
                node.name
            )),
        }
    }

    /// A MaterialX `image` node, its channels left for the caller.
    fn image(&self, n: usize) -> Result<Tex, String> {
        let node = self.node(n);
        let Some(Input::Image { image, color_space }) = node.inputs.get("file") else {
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
        let (uv_set, uv) = match self.input(n, "texcoord") {
            None => (0, Affine2::IDENTITY),
            Some(Input::Connect { node, .. }) => self.coordinates(*node)?,
            Some(_) => return Err(format!("{}: texcoord must be connected", node.name)),
        };
        Ok(Tex {
            image: *image,
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
        let (uv_set, uv) = match self.input(n, "st") {
            None => (0, Affine2::IDENTITY),
            Some(Input::Connect { node, .. }) => self.coordinates(*node)?,
            Some(_) => return Err(format!("{}: st must be connected", node.name)),
        };
        Ok(Tex {
            image: *image,
            channels: Channels::Rgb,
            uv_set,
            wrap,
            srgb,
            uv,
            scale,
            bias,
        })
    }

    /// The uv set and transform a texture-coordinate node gives.
    fn coordinates(&self, n: usize) -> Result<(u8, Affine2), String> {
        let node = self.node(n);
        let named = |name: Option<&str>| {
            (self.uv_set)(name.unwrap_or("st")).map_err(|e| format!("{}: {e}", node.name))
        };
        let inner = |input: &str| match self.input(n, input) {
            None => Ok((0, Affine2::IDENTITY)),
            Some(Input::Connect { node, .. }) => self.coordinates(*node),
            Some(_) => Err(format!("{}: {input} must be connected", node.name)),
        };
        let vec2 = |input: &str, d: Vec2| -> Result<Vec2, String> {
            Ok(self
                .number(n, input)?
                .map_or(d, |v| Vec2::new(v[0], v.get(1).copied().unwrap_or(v[0]))))
        };
        let float = |input: &str, d: f32| -> Result<f32, String> {
            Ok(self.number(n, input)?.map_or(d, |v| v[0]))
        };
        match node.kind.as_str() {
            "texcoord" => match float("index", 0.0)? {
                0.0 => Ok((0, Affine2::IDENTITY)),
                1.0 => Ok((1, Affine2::IDENTITY)),
                i => Err(format!(
                    "{}: texcoord index {i}; uv sets are 0 and 1",
                    node.name
                )),
            },
            "geompropvalue" => Ok((named(self.text(n, "geomprop"))?, Affine2::IDENTITY)),
            "UsdPrimvarReader_float2" => Ok((named(self.text(n, "varname"))?, Affine2::IDENTITY)),
            "place2d" => {
                let (set, inner) = inner("texcoord")?;
                let pivot = vec2("pivot", Vec2::ZERO)?;
                let scale = vec2("scale", Vec2::ONE)?;
                let offset = vec2("offset", Vec2::ZERO)?;
                // MaterialX rotate2d turns clockwise by its angle in degrees.
                let rotate = Affine2::from_angle(-float("rotate", 0.0)?.to_radians());
                let srt = float("operationorder", 0.0)? == 0.0;
                let to = Affine2::from_translation(-pivot);
                let back = Affine2::from_translation(pivot);
                let s = Affine2::from_scale(scale.recip());
                let t = Affine2::from_translation(-offset);
                let place = if srt {
                    back * t * rotate * s * to
                } else {
                    back * s * rotate * t * to
                };
                Ok((set, place * inner))
            }
            "UsdTransform2d" => {
                let (set, inner) = inner("in")?;
                let scale = vec2("scale", Vec2::ONE)?;
                let rotation = float("rotation", 0.0)?.to_radians();
                let translation = vec2("translation", Vec2::ZERO)?;
                let t = Affine2::from_scale_angle_translation(scale, rotation, translation);
                Ok((set, t * inner))
            }
            k => Err(format!(
                "node {} ({k}) is not a texture coordinate reader or transform",
                node.name
            )),
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
                self.no_bias(n, input, &t)?;
                *field(&mut self.material) = t.scale[t.channel()];
                *slot(&mut self.textures) = Some(t.tex_ref());
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
                self.no_bias(n, input, &t)?;
                *field(&mut self.material) = [t.scale[0], t.scale[1], t.scale[2]];
                *slot(&mut self.textures) = Some(t.tex_ref());
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

    fn no_bias(&self, n: usize, input: &str, t: &Tex) -> Result<(), String> {
        if t.bias.iter().any(|&b| b != 0.0) {
            return Err(format!(
                "{} input {input}: a texture bias has no material field",
                self.node(n).name
            ));
        }
        Ok(())
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
        if self.textures.opacity.is_some() {
            self.material.alpha_mode = Material::MASK;
        } else if self.material.geometry_opacity < 1.0 {
            self.material.alpha_mode = Material::BLEND;
        }
        match self.input(n, "geometry_normal") {
            None => {}
            Some(Input::Connect { node, .. }) if self.node(*node).kind == "normalmap" => {
                let map = *node;
                let scale = self.number(map, "scale")?.map_or(1.0, |v| v[0]);
                match self.value(map, "in")? {
                    Some(Value::Texture(t)) if t.channels == Channels::Rgb => {
                        self.textures.normal = Some((t.tex_ref(), [scale; 2]));
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
        if specular {
            self.color(n, "specularColor", |m| &mut m.specular_color, None)?;
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
        self.scalar(n, "ior", |m| &mut m.specular_ior, None)?;
        self.scalar(
            n,
            "clearcoat",
            |m| &mut m.coat_weight,
            Some(|t| &mut t.coat),
        )?;
        self.scalar(n, "clearcoatRoughness", |m| &mut m.coat_roughness, None)?;
        if self.material.coat_weight > 0.0 || self.textures.coat.is_some() {
            self.material.coat_ior = 1.5;
            self.material.coat_color = [1.0; 3];
        }
        self.scalar(
            n,
            "opacity",
            |m| &mut m.geometry_opacity,
            Some(|t| &mut t.opacity),
        )?;
        let threshold = self.number(n, "opacityThreshold")?.map_or(0.0, |v| v[0]);
        if threshold > 0.0 {
            self.material.alpha_mode = Material::MASK;
            self.material.alpha_cutoff = threshold;
        } else if self.textures.opacity.is_some() || self.material.geometry_opacity < 1.0 {
            self.material.alpha_mode = Material::BLEND;
        }
        match self.value(n, "occlusion")? {
            None | Some(Value::Constant(_)) => {}
            Some(Value::Texture(t)) => {
                self.no_bias(n, "occlusion", &t)?;
                self.textures.occlusion = Some((t.tex_ref(), 1.0));
            }
        }
        match self.value(n, "normal")? {
            None | Some(Value::Constant(_)) => {}
            Some(Value::Texture(t)) => {
                // Each axis 2t - 1, or 1 - 2t to flip x or y.
                let sign = |c: usize| match (t.scale[c], t.bias[c]) {
                    (2.0, -1.0) => Some(1.0),
                    (-2.0, 1.0) if c < 2 => Some(-1.0),
                    _ => None,
                };
                match (t.channels, sign(0), sign(1), sign(2)) {
                    (Channels::Rgb, Some(x), Some(y), Some(_)) => {
                        self.textures.normal = Some((t.tex_ref(), [x, y]));
                    }
                    _ => {
                        return Err(format!(
                            "{}: a normal texture reads rgb at scale 2 and bias -1 (x and y \
                             may flip: scale -2, bias 1)",
                            self.node(n).name
                        ));
                    }
                }
            }
        }
        match self.value(n, "displacement")? {
            None => {}
            Some(Value::Constant(v)) if v[0] == 0.0 => {}
            Some(Value::Constant(_)) => {
                return Err(format!(
                    "{}: a constant displacement has no texture to displace with",
                    self.node(n).name
                ));
            }
            Some(Value::Texture(t)) => {
                let c = t.channel();
                self.displaced(t, t.scale[c], t.bias[c]);
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
