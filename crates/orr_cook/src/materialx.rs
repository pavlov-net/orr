//! MaterialX documents (`.mtlx`): one `surfacematerial` and the nodes and node graphs
//! feeding its `surfaceshader` and `displacementshader`, mapped as [`crate::shading`] lists.
//!
//! Inputs connect by `nodename` (and `output`), by `nodegraph` (and `output`, which a graph of
//! one output ignores), or inside a node graph by `interfacename`. File values take the
//! `fileprefix` of their nearest scope and resolve against the directory of their document;
//! their `colorspace` (inherited likewise) is `srgb_texture` for sRGB, anything else linear.
//! A standalone material has no mesh, so each texture keeps its uv set (0 or 1) and transform
//! in its slot; geometric properties name `st`, `UV0` or a `geompropdef` of a texcoord.
//!
//! A `namespace` qualifies the names of its scope: references within it and to the global
//! namespace need no qualifier, others name `namespace:element`, and so may the material.
//!
//! Not honored yet, and reported as warnings: a `backsurfaceshader` other than the
//! `surfaceshader`, and a `normalmap` `space` of `object` from 1.38 documents, which reads as
//! tangent space.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use roxmltree::{Document, Node as XmlNode};

use crate::import::{Scene, SceneMaterial};
use crate::shading::{self, Input, Network, Node, through};
use crate::state::{Input as FileInput, Memo};
use crate::texture::material::{ImageData, ImageFile};
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
    "geompropdef",
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
    let files = vec![FileInput {
        path: path.to_owned(),
        hash,
    }];
    let text = String::from_utf8(bytes).map_err(|e| fail(e.to_string()))?;
    let docs = [Doc {
        doc: Document::parse(&text).map_err(|e| fail(e.to_string()))?,
        dir: path.parent().unwrap_or(Path::new("")),
        parent: None,
    }];
    let mut images = Vec::new();
    let mut by_path: HashMap<PathBuf, usize> = HashMap::new();
    let mut load = |dir: &Path, file: &str| -> usize {
        let p = dir.join(file);
        *by_path.entry(p.clone()).or_insert_with(|| {
            images.push(ImageFile {
                name: file.to_string(),
                data: ImageData::File {
                    path: p,
                    read: None,
                },
            });
            images.len() - 1
        })
    };
    let graph = network(&docs, &[0], material, &mut load).map_err(fail)?;
    let mapped = shading::map_per_slot(&graph.net, graph.surface, graph.displacement, &["st"])
        .map_err(|m| fail(format!("material {material}: {m}")))?;
    let mut warnings = graph.warnings;
    warnings.extend(mapped.warnings);
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
        warnings,
    })
}

/// One parsed document.
struct Doc<'t> {
    doc: Document<'t>,
    dir: &'t Path,
    parent: Option<usize>,
}

/// Attribute `attr` of `e` in document `d`, from its nearest scope, then the roots of the
/// documents including it.
fn inherited<'a>(
    docs: &'a [Doc<'_>],
    d: usize,
    e: &XmlNode<'a, '_>,
    attr: &str,
) -> Option<&'a str> {
    let mut found = e.ancestors().find_map(|a| a.attribute(attr));
    let mut parent = docs[d].parent;
    while let (None, Some(p)) = (found, parent) {
        found = docs[p].doc.root_element().attribute(attr);
        parent = docs[p].parent;
    }
    found
}

/// `name` in `namespace`, unless it has one.
fn qualified(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(ns) if !ns.is_empty() && !name.contains(':') => format!("{ns}:{name}"),
        _ => name.to_string(),
    }
}

/// A material's nodes and what the material reads of them.
struct Graph {
    net: Network,
    surface: usize,
    displacement: Option<usize>,
    warnings: Vec<String>,
}

/// The network of `material` in `docs`, the first of them the document cooked, walked in
/// `order`. `load` reads an image file, relative to a directory, and returns its index.
fn network(
    docs: &[Doc<'_>],
    order: &[usize],
    material: &str,
    load: &mut dyn FnMut(&Path, &str) -> usize,
) -> std::result::Result<Graph, String> {
    let mut net = Network::default();
    // Nodes by (graph, name); the top level is graph "".
    let mut index: HashMap<(String, String), usize> = HashMap::new();
    let mut elements = Vec::new();
    let mut graphs = HashMap::new();
    let is_node = |e: &XmlNode<'_, '_>| {
        e.is_element() && e.has_attribute("type") && !NOT_NODES.contains(&e.tag_name().name())
    };
    let mut add = |graph: &str, name: String, n, d| {
        if let Entry::Vacant(v) = index.entry((graph.to_string(), name)) {
            v.insert(elements.len());
            elements.push((graph.to_string(), n, d));
        }
    };
    for &d in order {
        let root = docs[d].doc.root_element();
        if root.tag_name().name() != "materialx" {
            return Err(format!(
                "root element {} is not materialx",
                root.tag_name().name()
            ));
        }
        for e in root.children().filter(XmlNode::is_element) {
            let name = e.attribute("name").unwrap_or_default();
            let full_name = qualified(inherited(docs, d, &e, "namespace"), name);
            match e.tag_name().name() {
                "nodegraph" => {
                    if let Entry::Vacant(v) = graphs.entry(full_name.clone()) {
                        v.insert((e, d));
                        for n in e.children().filter(is_node) {
                            let name = n.attribute("name").unwrap_or_default().to_string();
                            add(&full_name, name, n, d);
                        }
                    }
                }
                "geompropdef" if e.attribute("geomprop") == Some("texcoord") => {
                    let set = e.attribute("index").map_or(Some(0), |i| i.parse().ok());
                    if let Some(set) = set {
                        net.uv_sets.entry(name.to_string()).or_insert(set);
                    }
                }
                _ if is_node(&e) => add("", full_name, e, d),
                _ => {}
            }
        }
    }
    let find = |graph: &str, namespace: Option<&str>, name: &str| {
        index
            .get(&(graph.to_string(), qualified(namespace, name)))
            .or_else(|| index.get(&(graph.to_string(), name.to_string())))
            .copied()
    };
    // The connection an element with nodename/nodegraph/output attributes makes, in `graph`.
    let connect = |graph: &str,
                   e: &XmlNode<'_, '_>,
                   d: usize|
     -> std::result::Result<Option<Input>, String> {
        let namespace = inherited(docs, d, e, "namespace");
        let output = e.attribute("output").unwrap_or("out").to_string();
        if let Some(n) = e.attribute("nodename") {
            let node = find(graph, namespace, n).ok_or_else(|| {
                format!("{}: no node {n}", e.attribute("name").unwrap_or_default())
            })?;
            return Ok(Some(Input::Connect { node, output }));
        }
        if let Some(g) = e.attribute("nodegraph") {
            let key = qualified(namespace, g);
            let (key, (ge, _)) = graphs
                .get_key_value(&key)
                .or_else(|| graphs.get_key_value(g))
                .ok_or_else(|| format!("no nodegraph {g}"))?;
            let outputs: Vec<_> = ge
                .children()
                .filter(|o| o.tag_name().name() == "output")
                .collect();
            let out = match outputs[..] {
                [only] => only,
                _ => *outputs
                    .iter()
                    .find(|o| o.attribute("name") == Some(&output))
                    .ok_or_else(|| format!("nodegraph {g} has no output {output}"))?,
            };
            let name = out.attribute("name").unwrap_or_default();
            let n = out
                .attribute("nodename")
                .ok_or_else(|| format!("nodegraph {g} output {name} names no node"))?;
            let node = find(key, None, n).ok_or_else(|| format!("nodegraph {g}: no node {n}"))?;
            let output = out.attribute("output").unwrap_or("out").to_string();
            return Ok(Some(Input::Connect { node, output }));
        }
        Ok(None)
    };
    net.nodes = elements
        .iter()
        .map(|(graph, e, _)| {
            let name = e.attribute("name").unwrap_or_default();
            Node {
                name: if graph.is_empty() {
                    name.to_string()
                } else {
                    format!("{graph}/{name}")
                },
                kind: e.tag_name().name().to_string(),
                ty: e.attribute("type").unwrap_or_default().to_string(),
                inputs: BTreeMap::new(),
            }
        })
        .collect();
    let namespace = docs[0].doc.root_element().attribute("namespace");
    let m = find("", namespace, material).ok_or_else(|| format!("no material {material}"))?;
    if net.nodes[m].kind != "surfacematerial" {
        return Err(format!(
            "{material} is a {}, not a surfacematerial",
            net.nodes[m].kind
        ));
    }
    // Only what the material reads resolves: the rest of an included library may rest on
    // definitions this document never sees, such as the nodedef behind a functional graph.
    let mut resolved = vec![false; elements.len()];
    let mut pending = vec![m];
    // Filename inputs fed by a connection, which may end in a string.
    let mut filenames = Vec::new();
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut resolved[n], true) {
            continue;
        }
        let (graph, e, d) = &elements[n];
        let mut inputs = BTreeMap::new();
        for i in e.children().filter(|c| c.tag_name().name() == "input") {
            let input_name = i.attribute("name").unwrap_or_default().to_string();
            // Inside a graph, an interface name reads the graph's own input.
            let source = match (i.attribute("interfacename"), graph.as_str()) {
                (Some(iface), g) if !g.is_empty() => {
                    let (ge, _) = graphs.get(g).ok_or_else(|| format!("no nodegraph {g}"))?;
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
            let value = match connect(source.1, &source.0, *d)? {
                Some(c) => {
                    if source.0.attribute("type") == Some("filename") {
                        filenames.push((n, input_name.clone(), source.0));
                    }
                    if let Input::Connect { node, .. } = c {
                        pending.push(node);
                    }
                    c
                }
                None => match source.0.attribute("value") {
                    None => continue,
                    Some(v) => value(docs, *d, &source.0, v, load),
                },
            };
            inputs.insert(input_name, value);
        }
        net.nodes[n].inputs = inputs;
    }
    for (n, input, e) in filenames {
        let d = elements[n].2;
        if let Input::Text(file) = through(&net, &net.nodes[n].inputs[&input]) {
            let image = file_input(docs, d, &e, file, load);
            net.nodes[n].inputs.insert(input, image);
        }
    }
    let shader = |input: &str| match net.nodes[m].inputs.get(input).map(|i| through(&net, i)) {
        Some(Input::Connect { node, .. }) => Some(*node),
        _ => None,
    };
    let surface =
        shader("surfaceshader").ok_or_else(|| format!("{material} has no surfaceshader"))?;
    let displacement = shader("displacementshader");
    let mut warnings = Vec::new();
    if shader("backsurfaceshader").is_some_and(|b| b != surface) {
        warnings.push(format!(
            "material {material}: backsurfaceshader is not honored yet; both sides read \
             surfaceshader"
        ));
    }
    Ok(Graph {
        net,
        surface,
        displacement,
        warnings,
    })
}

/// A filename value of element `e` in document `d`.
fn file_input(
    docs: &[Doc<'_>],
    d: usize,
    e: &XmlNode<'_, '_>,
    file: &str,
    load: &mut dyn FnMut(&Path, &str) -> usize,
) -> Input {
    let prefix = inherited(docs, d, e, "fileprefix").unwrap_or_default();
    Input::Image {
        image: load(docs[d].dir, &format!("{prefix}{file}")),
        color_space: inherited(docs, d, e, "colorspace").map(str::to_string),
    }
}

/// An input's authored value: an image for filenames, numbers, or text.
fn value(
    docs: &[Doc<'_>],
    d: usize,
    e: &XmlNode<'_, '_>,
    v: &str,
    load: &mut dyn FnMut(&Path, &str) -> usize,
) -> Input {
    match e.attribute("type").unwrap_or_default() {
        "filename" => file_input(docs, d, e, v, load),
        "string" => Input::Text(v.to_string()),
        "boolean" => Input::Value(vec![if v == "true" { 1.0 } else { 0.0 }]),
        _ => v
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_or_else(|_| Input::Text(v.to_string()), Input::Value),
    }
}
