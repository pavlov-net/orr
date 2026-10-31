//! UsdShade materials as shading networks: the surface a `Material` prim outputs
//! (`outputs:mtlx:surface`, else `outputs:surface`), its MaterialX displacement
//! (`outputs:mtlx:displacement`), and every shader feeding them. Connections pass through
//! the inputs and outputs of `NodeGraph` and `Material` prims.

use std::collections::{BTreeMap, HashMap};

use openusd::{sdf, usd};

use super::read::{PrimReader, numbers};
use crate::shading::{self, Input, Network, Node};

struct Builder<'a> {
    stage: &'a usd::Stage,
    time: usd::TimeCode,
    /// Loads the image an asset value names; returns its index.
    load: &'a mut dyn FnMut(&sdf::AssetPath) -> Result<usize, String>,
    net: Network,
    by_prim: HashMap<String, usize>,
}

/// The network of material prim `material`, its surface node and its displacement node.
pub(super) fn network(
    stage: &usd::Stage,
    time: usd::TimeCode,
    material: &sdf::Path,
    load: &mut dyn FnMut(&sdf::AssetPath) -> Result<usize, String>,
) -> Result<(Network, usize, Option<usize>), String> {
    let mut b = Builder {
        stage,
        time,
        load,
        net: Network::default(),
        by_prim: HashMap::new(),
    };
    let prim = stage.prim(material.clone()).map_err(|e| e.to_string())?;
    let output = |name: &str| -> Result<Option<sdf::Path>, String> {
        let attr = prim.attribute(name);
        let targets = attr
            .connections()
            .map_err(|e| format!("{material} {name}: {e}"))?;
        Ok(targets.into_iter().next())
    };
    // MaterialX outputs first, then the universal render context.
    let surface = match output("outputs:mtlx:surface")? {
        Some(t) => Some(t),
        None => output("outputs:surface")?,
    };
    let surface = surface.ok_or_else(|| format!("{material} has no surface output"))?;
    let Input::Connect { node: surface, .. } = b.source(&surface)? else {
        return Err(format!("{material}: the surface output reads no shader"));
    };
    let displacement = match output("outputs:mtlx:displacement")? {
        Some(t) => match b.source(&t)? {
            Input::Connect { node, .. } => Some(node),
            _ => None,
        },
        None => None,
    };
    Ok((b.net, surface, displacement))
}

impl Builder<'_> {
    /// What a connection to property `target` reads.
    fn source(&mut self, target: &sdf::Path) -> Result<Input, String> {
        let (prim_path, property) = target
            .split_property()
            .ok_or_else(|| format!("connection {target} is not to a property"))?;
        let property = property.to_string();
        let prim = self
            .stage
            .prim(prim_path.clone())
            .map_err(|e| e.to_string())?;
        let ty = prim
            .type_name()
            .map_err(|e| e.to_string())?
            .map(|t| t.as_str().to_string())
            .unwrap_or_default();
        if ty == "Shader" {
            let output = property
                .strip_prefix("outputs:")
                .ok_or_else(|| format!("connection {target} is not to a shader output"))?;
            let node = self.shader(&prim)?;
            return Ok(Input::Connect {
                node,
                output: output.to_string(),
            });
        }
        // A node graph or material: follow its interface attribute.
        let attr = prim.attribute(property.as_str());
        let connections = attr.connections().map_err(|e| format!("{target}: {e}"))?;
        if let Some(next) = connections.into_iter().next() {
            return self.source(&next);
        }
        let reader = PrimReader {
            prim: &prim,
            time: self.time,
        };
        match reader.value(&property)? {
            Some(v) => self.value(&reader, &property, v),
            None => Err(format!("{target} has no value or connection")),
        }
    }

    /// The node of a shader prim.
    fn shader(&mut self, prim: &usd::Prim) -> Result<usize, String> {
        let path = prim.path().as_str().to_string();
        if let Some(&n) = self.by_prim.get(&path) {
            return Ok(n);
        }
        let reader = PrimReader {
            prim,
            time: self.time,
        };
        let id = reader
            .token("info:id")?
            .ok_or_else(|| format!("{path}: shader without info:id"))?;
        let kind = shading::category(&id).to_string();
        let ty = id
            .strip_prefix("ND_")
            .and_then(|r| r.strip_prefix(kind.as_str()))
            .and_then(|r| r.strip_prefix('_'))
            .unwrap_or_default()
            .to_string();
        let n = self.net.nodes.len();
        self.net.nodes.push(Node {
            name: path.clone(),
            kind,
            ty,
            inputs: BTreeMap::new(),
        });
        self.by_prim.insert(path.clone(), n);
        let attrs = prim
            .authored_attributes_in_namespace("inputs")
            .map_err(|e| format!("{path}: {e}"))?;
        let mut inputs = BTreeMap::new();
        for attr in attrs {
            let full = attr.name().to_string();
            let name = full.trim_start_matches("inputs:").to_string();
            let connections = attr
                .connections()
                .map_err(|e| format!("{path} {full}: {e}"))?;
            let input = match connections.into_iter().next() {
                Some(t) => self.source(&t)?,
                None => match reader.value(&full)? {
                    Some(v) => self.value(&reader, &full, v)?,
                    None => continue,
                },
            };
            inputs.insert(name, input);
        }
        self.net.nodes[n].inputs = inputs;
        Ok(n)
    }

    /// An authored input value.
    fn value(
        &mut self,
        reader: &PrimReader<'_>,
        name: &str,
        v: sdf::Value,
    ) -> Result<Input, String> {
        Ok(match v {
            sdf::Value::AssetPath(a) => Input::Image {
                image: (self.load)(&a).map_err(|e| format!("{} {name}: {e}", reader.path()))?,
                color_space: reader.metadata_token(name, "colorSpace"),
            },
            sdf::Value::Token(t) => Input::Text(t.as_str().to_string()),
            sdf::Value::String(s) => Input::Text(s),
            v => Input::Value(
                numbers(&v)
                    .ok_or_else(|| format!("{} {name}: unsupported value", reader.path()))?
                    .into_iter()
                    .map(|x| x as f32)
                    .collect(),
            ),
        })
    }
}
