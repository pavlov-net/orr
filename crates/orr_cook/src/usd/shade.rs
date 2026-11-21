//! UsdShade materials as shading networks: the surface a `Material` prim outputs (its
//! `mtlx` render context, else the universal one), its MaterialX displacement
//! (`outputs:mtlx:displacement`), and every shader feeding them. Each input reads what
//! `UsdShadeUtils::GetValueProducingAttributes` finds for it: a shader output, or a value authored
//! on the input itself or on the `NodeGraph` or `Material` interface it connects through. An input
//! that finds nothing keeps its node's default. A texture file is in the color space
//! `UsdColorSpaceAPI` assigns it.

use std::collections::{BTreeMap, HashMap};

use openusd::{sdf, usd, vt};
use openusd_schemas::shade::{
    self, Connectable, ConnectableContext, ProducerFilter, ShadingAttribute,
};

use super::read::{self, PrimReader, numbers};
use crate::shading::{self, Input, Network, Node, Unit};
use crate::texture::color::ImageColor;

/// The render context whose terminals a material outputs first.
const MTLX: &str = "mtlx";

struct Builder<'a> {
    time: usd::TimeCode,
    /// Loads the image an asset value names; returns its index.
    load: &'a mut dyn FnMut(&sdf::AssetPath) -> usize,
    spaces: &'a mut usd::ColorSpaceCache,
    context: ConnectableContext,
    net: Network,
    by_prim: HashMap<String, usize>,
    /// The material's path, which warnings name.
    material: String,
    warnings: Vec<String>,
}

/// The network of a material.
pub(super) struct Shaded {
    pub(super) net: Network,
    pub(super) surface: usize,
    pub(super) displacement: Option<usize>,
    /// Inputs whose connections reach nothing that gives a value.
    pub(super) warnings: Vec<String>,
}

/// The network of `material`. `spaces` keeps the color spaces of the prims it reads.
pub(super) fn network(
    material: &shade::Material,
    time: usd::TimeCode,
    load: &mut dyn FnMut(&sdf::AssetPath) -> usize,
    spaces: &mut usd::ColorSpaceCache,
) -> Result<Shaded, String> {
    let path = material.path().as_str();
    let mut b = Builder {
        time,
        load,
        spaces,
        context: ConnectableContext::default(),
        net: Network::default(),
        by_prim: HashMap::new(),
        material: path.to_string(),
        warnings: Vec::new(),
    };
    let surface = material
        .surface_source(&[MTLX])
        .map_err(|e| format!("{path}: {e}"))?
        .ok_or_else(|| format!("{path} has no surface output"))?;
    let surface = &surface.sources()[0];
    let Input::Connect { node: surface, .. } = b.produced(surface.attribute())? else {
        return Err(format!("{path}: the surface output reads no shader"));
    };
    let displacement = material
        .displacement_output_for(MTLX)
        .map_err(|e| format!("{path}: {e}"))?
        .value_producing_attributes(ProducerFilter::ShaderOutputsOnly)
        .map_err(|e| format!("{path}: {e}"))?;
    let displacement = match displacement.first() {
        Some(producer) => match b.produced(producer)? {
            Input::Connect { node, .. } => Some(node),
            _ => None,
        },
        None => None,
    };
    Ok(Shaded {
        net: b.net,
        surface,
        displacement,
        warnings: b.warnings,
    })
}

/// The output type of a node definition whose id ends in `suffix`, and the type a `convert` reads.
/// An `extract` id names the type it reads, where other ids name the type they give, and a
/// `convert` id names both, as `ND_convert_color4_color3`.
fn types<'s>(kind: &str, suffix: &'s str) -> (&'s str, &'s str) {
    match (kind, suffix) {
        ("extract", "matrix33") => ("vector3", ""),
        ("extract", "matrix44") => ("vector4", ""),
        ("extract", _) => ("float", ""),
        ("convert", s) => s.split_once('_').map_or((s, ""), |(from, to)| (to, from)),
        _ => (suffix, ""),
    }
}

impl Builder<'_> {
    /// What `producer`, an attribute that produces a value, gives: a shader output, or the value
    /// authored on it.
    fn produced(&mut self, producer: &ShadingAttribute) -> Result<Input, String> {
        let attr = producer.attribute();
        let prim = attr.prim();
        let path = attr.path();
        let container = self.context.is_container(&prim);
        let container = container.map_err(|e| format!("{path}: {e}"))?;
        if let ShadingAttribute::Output(output) = producer
            && !container
        {
            let node = self.shader(&prim)?;
            return Ok(Input::Connect {
                node,
                output: output.base_name().to_string(),
            });
        }
        let reader = PrimReader {
            prim: &prim,
            time: self.time,
        };
        let name = attr.name();
        match reader.value(name)? {
            Some(v) => self.value(&reader, attr, v),
            None => Err(format!("{path} has no value")),
        }
    }

    /// The node of a shader prim.
    fn shader(&mut self, prim: &usd::Prim) -> Result<usize, String> {
        let path = prim.path().as_str().to_string();
        if let Some(&n) = self.by_prim.get(&path) {
            return Ok(n);
        }
        let shader = shade::Shader::from_prim_unchecked(prim.clone());
        let id = shader.node_def().shader_id();
        let id = id
            .map_err(|e| format!("{path}: {e}"))?
            .ok_or_else(|| format!("{path}: shader without info:id"))?;
        let id = id.as_str();
        let kind = shading::category(id).to_string();
        let (ty, in_ty) = id
            .strip_prefix("ND_")
            .and_then(|r| r.strip_prefix(kind.as_str()))
            .and_then(|r| r.strip_prefix('_'))
            .map_or_default(|suffix| types(&kind, suffix));
        let n = self.net.nodes.len();
        self.net.nodes.push(Node {
            name: path.clone(),
            ty: ty.to_string(),
            in_ty: in_ty.to_string(),
            kind,
            inputs: BTreeMap::new(),
        });
        self.by_prim.insert(path.clone(), n);
        let reader = PrimReader {
            prim,
            time: self.time,
        };
        let mut inputs = BTreeMap::new();
        for input in shader
            .authored_inputs()
            .map_err(|e| format!("{path}: {e}"))?
        {
            let fail =
                |e: openusd_schemas::SchemaError| format!("{}: {e}", input.attribute().path());
            let producers = input.value_producing_attributes(ProducerFilter::Any);
            let Some(producer) = producers.map_err(fail)?.into_iter().next() else {
                let to = input.attribute().connections();
                let to = to.map_err(|e| format!("{}: {e}", input.attribute().path()))?;
                if !to.is_empty() {
                    let to: Vec<String> = to.iter().map(ToString::to_string).collect();
                    self.warnings.push(format!(
                        "material {}: {} connects to {}, which gives no value; it keeps its \
                         default",
                        self.material,
                        input.attribute().path(),
                        to.join(", ")
                    ));
                }
                continue;
            };
            let value = match &producer {
                // A value on the input itself, read where the shader stands.
                ShadingAttribute::Input(own)
                    if own.attribute().path() == input.attribute().path() =>
                {
                    match reader.value(own.attribute().name())? {
                        Some(vt::Value::AssetPath(a)) if a.asset_path().is_empty() => continue,
                        Some(v) => self.value(&reader, own.attribute(), v)?,
                        None => continue,
                    }
                }
                _ => self.produced(&producer)?,
            };
            inputs.insert(input.base_name().to_string(), value);
        }
        self.net.nodes[n].inputs = inputs;
        Ok(n)
    }

    /// An authored input value; a color's in linear Rec.709 from the space
    /// [`PrimReader::color_space`] resolves, or as it stands, with a warning, where the cook
    /// doesn't know that space.
    fn value(
        &mut self,
        reader: &PrimReader<'_>,
        attr: &usd::Attribute,
        v: vt::Value,
    ) -> Result<Input, String> {
        let name = attr.name();
        Ok(match v {
            vt::Value::AssetPath(a) => {
                let space = usd::ColorSpaceAPI::compute_attribute_name(attr, Some(self.spaces));
                let space = space.map_err(|e| format!("{} {name}: {e}", reader.path()))?;
                Input::Image {
                    image: (self.load)(&a),
                    color_space: space.map(|s| s.as_str().to_string()),
                    unit: Unit::Unstated,
                }
            }
            vt::Value::Token(t) => Input::Text(t.as_str().to_string()),
            vt::Value::String(s) => Input::Text(s),
            v => {
                let mut v = numbers(&v)
                    .ok_or_else(|| format!("{} {name}: unsupported value", reader.path()))?;
                let role = attr.role();
                let role = role.map_err(|e| format!("{} {name}: {e}", reader.path()))?;
                if let (Some(sdf::Role::Color), [r, g, b, ..]) = (role, &v[..]) {
                    let space = reader.color_space(name)?;
                    match ImageColor::named(&space) {
                        Some(c) => {
                            let rgb = read::to_rec709(c, orr_math::DVec3::new(*r, *g, *b));
                            v[..3].copy_from_slice(&rgb.to_array());
                        }
                        None => self.warnings.push(format!(
                            "{} {name}: color space {space} reads as data",
                            reader.path()
                        )),
                    }
                }
                Input::Value(v.into_iter().map(|x| x as f32).collect(), Unit::Unstated)
            }
        })
    }
}
