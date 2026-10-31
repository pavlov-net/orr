//! MaterialX documents (`.mtlx`): one `surfacematerial` and the nodes and node graphs
//! feeding its `surfaceshader` and `displacementshader`, mapped as [`crate::shading`] lists.
//!
//! Inputs connect by `nodename` (and `output`), by `nodegraph` and `output`, or inside a node
//! graph by `interfacename`. File values take the `fileprefix` of their element and its
//! ancestors and resolve against the document's directory; their `colorspace` (inherited
//! likewise) is `srgb_texture` for sRGB, anything else linear. A standalone material has no
//! mesh, so its textures read uv set 0 or 1 untransformed; geometric properties name `st`.

use std::collections::HashMap;
use std::path::Path;

use roxmltree::{Document, Node as XmlNode};

use crate::import::{Scene, SceneMaterial};
use crate::shading::{self, Input, Network, Node};
use crate::state::{Input as FileInput, Memo};
use crate::texture::material::ImageFile;
use crate::{Error, Result};

/// Elements that are not nodes.
const NOT_NODES: &[&str] = &[
    "input",
    "output",
    "nodedef",
    "nodegraph",
    "typedef",
    "token",
    "look",
    "materialassign",
    "collection",
    "geominfo",
    "propertyset",
    "variantset",
    "implementation",
    "unittypedef",
];

pub(crate) fn import(asset: &str, path: &Path, memo: &Memo, material: &str) -> Result<Scene> {
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message: format!("{}: {message}", path.display()),
    };
    let (bytes, hash) = memo.read(path)?;
    let mut files = vec![FileInput {
        path: path.to_owned(),
        hash,
    }];
    let text = String::from_utf8(bytes).map_err(|e| fail(e.to_string()))?;
    let doc = Document::parse(&text).map_err(|e| fail(e.to_string()))?;
    let dir = path.parent().unwrap_or(Path::new(""));
    let mut images = Vec::new();
    let mut by_path: HashMap<std::path::PathBuf, usize> = HashMap::new();
    let mut load = |file: &str| -> Result<usize> {
        let p = dir.join(file);
        if let Some(&i) = by_path.get(&p) {
            return Ok(i);
        }
        let (bytes, hash) = memo.read(&p)?;
        files.push(FileInput {
            path: p.clone(),
            hash,
        });
        images.push(ImageFile {
            name: file.to_string(),
            bytes,
        });
        by_path.insert(p, images.len() - 1);
        Ok(images.len() - 1)
    };
    let (net, surface, displacement) = network(&doc, material, &mut load).map_err(|e| match e {
        Failure::Read(e) => e,
        Failure::Document(m) => fail(m),
    })?;
    let uv_set = |name: &str| match name {
        "st" => Ok(0),
        n => Err(format!(
            "geometric property {n}; a standalone material reads st"
        )),
    };
    let mapped = shading::map(&net, surface, displacement, &uv_set)
        .map_err(|m| fail(format!("material {material}: {m}")))?;
    if mapped.uv.is_some() {
        return Err(fail(format!(
            "material {material}: a uv transform has no material field, and there is no mesh \
             to apply it to"
        )));
    }
    Ok(Scene {
        models: Vec::new(),
        materials: vec![SceneMaterial {
            suffix: String::new(),
            material: mapped.material,
            textures: mapped.textures,
        }],
        images,
        skeletons: Vec::new(),
        instances: Vec::new(),
        files,
    })
}

enum Failure {
    Read(Error),
    Document(String),
}

impl From<String> for Failure {
    fn from(m: String) -> Self {
        Self::Document(m)
    }
}

/// The network of `material` in `doc`: every node, the surface shader and the displacement
/// shader. `load` reads an image file (relative to the document) and returns its index.
fn network(
    doc: &Document<'_>,
    material: &str,
    load: &mut dyn FnMut(&str) -> Result<usize>,
) -> std::result::Result<(Network, usize, Option<usize>), Failure> {
    let root = doc.root_element();
    if root.tag_name().name() != "materialx" {
        return Err(format!("root element {} is not materialx", root.tag_name().name()).into());
    }
    // Nodes by (graph, name); the top level is graph "".
    let mut index: HashMap<(String, String), usize> = HashMap::new();
    let mut elements = Vec::new();
    let is_node = |e: &XmlNode<'_, '_>| {
        e.is_element() && e.has_attribute("type") && !NOT_NODES.contains(&e.tag_name().name())
    };
    for e in root.children().filter(XmlNode::is_element) {
        if e.tag_name().name() == "nodegraph" {
            let graph = e.attribute("name").unwrap_or_default().to_string();
            for n in e.children().filter(is_node) {
                let name = n.attribute("name").unwrap_or_default().to_string();
                index.insert((graph.clone(), name), elements.len());
                elements.push((graph.clone(), n));
            }
        } else if is_node(&e) {
            let name = e.attribute("name").unwrap_or_default().to_string();
            index.insert((String::new(), name), elements.len());
            elements.push((String::new(), e));
        }
    }
    let graph_of = |name: &str| {
        root.children()
            .find(|e| e.tag_name().name() == "nodegraph" && e.attribute("name") == Some(name))
    };
    // The connection an element with nodename/nodegraph/output attributes makes, in `graph`.
    let connect = |graph: &str,
                   e: &XmlNode<'_, '_>|
     -> std::result::Result<Option<Input>, String> {
        let output = e.attribute("output").unwrap_or("out").to_string();
        if let Some(n) = e.attribute("nodename") {
            let node = *index
                .get(&(graph.to_string(), n.to_string()))
                .ok_or_else(|| {
                    format!("{}: no node {n}", e.attribute("name").unwrap_or_default())
                })?;
            return Ok(Some(Input::Connect { node, output }));
        }
        if let Some(g) = e.attribute("nodegraph") {
            let ge = graph_of(g).ok_or_else(|| format!("no nodegraph {g}"))?;
            let out = ge
                .children()
                .find(|o| o.tag_name().name() == "output" && o.attribute("name") == Some(&output))
                .ok_or_else(|| format!("nodegraph {g} has no output {output}"))?;
            let n = out
                .attribute("nodename")
                .ok_or_else(|| format!("nodegraph {g} output {output} names no node"))?;
            let node = *index
                .get(&(g.to_string(), n.to_string()))
                .ok_or_else(|| format!("nodegraph {g}: no node {n}"))?;
            let output = out.attribute("output").unwrap_or("out").to_string();
            return Ok(Some(Input::Connect { node, output }));
        }
        Ok(None)
    };
    let mut net = Network::default();
    for (graph, e) in &elements {
        let name = e.attribute("name").unwrap_or_default();
        let mut inputs = std::collections::BTreeMap::new();
        for i in e.children().filter(|c| c.tag_name().name() == "input") {
            let input_name = i.attribute("name").unwrap_or_default().to_string();
            // Inside a graph, an interface name reads the graph's own input.
            let source = match (i.attribute("interfacename"), graph.as_str()) {
                (Some(iface), g) if !g.is_empty() => {
                    let ge = graph_of(g).ok_or_else(|| format!("no nodegraph {g}"))?;
                    let gi = ge
                        .children()
                        .find(|c| {
                            c.tag_name().name() == "input" && c.attribute("name") == Some(iface)
                        })
                        .ok_or_else(|| format!("nodegraph {g} has no input {iface}"))?;
                    (gi, "")
                }
                _ => (i, graph.as_str()),
            };
            let value = match connect(source.1, &source.0)? {
                Some(c) => c,
                None => match source.0.attribute("value") {
                    None => continue,
                    Some(v) => value(&source.0, v, load)?,
                },
            };
            inputs.insert(input_name, value);
        }
        net.nodes.push(Node {
            name: if graph.is_empty() {
                name.to_string()
            } else {
                format!("{graph}/{name}")
            },
            kind: e.tag_name().name().to_string(),
            ty: e.attribute("type").unwrap_or_default().to_string(),
            inputs,
        });
    }
    let m = *index
        .get(&(String::new(), material.to_string()))
        .ok_or_else(|| format!("no material {material}"))?;
    if net.nodes[m].kind != "surfacematerial" {
        return Err(format!(
            "{material} is a {}, not a surfacematerial",
            net.nodes[m].kind
        )
        .into());
    }
    let shader = |input: &str| match net.nodes[m].inputs.get(input) {
        Some(Input::Connect { node, .. }) => Some(*node),
        _ => None,
    };
    let surface =
        shader("surfaceshader").ok_or_else(|| format!("{material} has no surfaceshader"))?;
    let displacement = shader("displacementshader");
    Ok((net, surface, displacement))
}

/// An input's authored value: an image for filenames, numbers, or text.
fn value(
    e: &XmlNode<'_, '_>,
    v: &str,
    load: &mut dyn FnMut(&str) -> Result<usize>,
) -> std::result::Result<Input, Failure> {
    let inherited = |attr: &str| e.ancestors().find_map(|a| a.attribute(attr));
    match e.attribute("type").unwrap_or_default() {
        "filename" => {
            let prefix: String = {
                let mut parts: Vec<&str> = e
                    .ancestors()
                    .filter_map(|a| a.attribute("fileprefix"))
                    .collect();
                parts.reverse();
                parts.concat()
            };
            let image = load(&format!("{prefix}{v}")).map_err(Failure::Read)?;
            Ok(Input::Image {
                image,
                color_space: inherited("colorspace").map(str::to_string),
            })
        }
        "string" => Ok(Input::Text(v.to_string())),
        "boolean" => Ok(Input::Value(vec![if v == "true" { 1.0 } else { 0.0 }])),
        _ => v
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map(Input::Value)
            .or_else(|_| Ok(Input::Text(v.to_string()))),
    }
}
