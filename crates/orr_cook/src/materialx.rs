//! MaterialX documents (`.mtlx`): one `surfacematerial` and the nodes and node graphs
//! feeding its `surfaceshader` and `displacementshader`, mapped as [`crate::shading`] lists.
//!
//! Inputs connect by `nodename` (and `output`), by `nodegraph` (and `output`, which a graph of
//! one output ignores), or inside a node graph by `interfacename`. File values take the
//! `fileprefix` of their nearest scope and resolve against the directory of their document.
//! Images and values of color types read in their `colorspace`, inherited likewise, so that the
//! document's working space is the last scope, and "data" where no scope names one: values
//! convert here to the linear Rec.709 of [`Material`](orr_pack::material::Material), images when
//! they cook. A name that isn't a known space warns and reads as "data".
//! A standalone material has no mesh, so each texture keeps its uv set (0 or 1) and transform
//! in its slot; geometric properties name `st`, `UV0` or a `geompropdef` of a texcoord.
//!
//! The scene's units are the meter and the degree. A value's unit is its input's `unit`, else
//! the one its nodedef's input declares, and it stays with the value along connections; as in
//! the MaterialX library, a `unit` on an input that is connected states nothing, whatever it
//! names. A filename input reading a string is the exception: a string holds no unit, so the
//! input states the unit of the file. A unit must be a unit of its nodedef's `unittype`, else of
//! the input's. With neither, which the specification does not permit but which still states a
//! unit, any type holding the name serves, `distance` and `angle` first. An empty `unit` or
//! `unittype` states nothing. `unitdef`s add to the standard units, and a type other than
//! `distance` and `angle` has no scene unit, so it converts nothing. The root of a document
//! states no unit.
//!
//! An `xi:include` adds the children of the root of the document it names, read before the
//! including document, so that the first of two like-named elements wins, as in the MaterialX
//! library. Root attributes an included document leaves out come from the document including
//! it. A `namespace` qualifies the names of its scope: references within it and to the global
//! namespace need no qualifier, others name `namespace:element`, and so may the material.
//!
//! Not honored yet, and reported as warnings: a `backsurfaceshader` other than the
//! `surfaceshader`, and a `normalmap` `space` of `object` from 1.38 documents, which reads as
//! tangent space.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use roxmltree::{Document, Node as XmlNode, NodeId};

use crate::import::{Scene, SceneMaterial};
use crate::shading::{self, Input, Network, Node, Unit, is_color, through};
use crate::state::{Input as FileInput, Memo};
use crate::texture::color::ImageColor;
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

/// The units every document has, from MaterialX's `stdlib_defs.mtlx` (Apache-2.0): unit type,
/// unit and scale.
const UNITS: &[(&str, &str, f64)] = &[
    ("distance", "nanometer", 0.000000001),
    ("distance", "micron", 0.000001),
    ("distance", "millimeter", 0.001),
    ("distance", "centimeter", 0.01),
    ("distance", "inch", 0.0254),
    ("distance", "foot", 0.3048),
    ("distance", "yard", 0.9144),
    ("distance", "meter", 1.0),
    ("distance", "kilometer", 1000.0),
    ("distance", "mile", 1609.344),
    ("angle", "degree", 1.0),
    ("angle", "radian", 57.295779513),
];

/// What the standard nodedefs of the nodes read here declare: node, input and unit type. Each
/// expects the scene's unit of that type.
const DECLARED: &[(&str, &str, &str)] = &[
    ("place2d", "rotate", "angle"),
    ("tiledimage", "realworldimagesize", "distance"),
    ("tiledimage", "realworldtilesize", "distance"),
];

const XINCLUDE: &str = "http://www.w3.org/2001/XInclude";

fn is_xinclude(e: &XmlNode<'_, '_>, name: &str) -> bool {
    e.tag_name().namespace() == Some(XINCLUDE) && e.tag_name().name() == name
}

pub(crate) fn import(asset: &str, path: &Path, memo: &Memo, material: &str) -> Result<Scene> {
    let fail = |message: String| Error::Import {
        asset: asset.into(),
        message: format!("{}: {message}", path.display()),
    };
    let mut reader = Reader {
        memo,
        texts: Vec::new(),
        order: Vec::new(),
        open: Vec::new(),
        seen: HashSet::new(),
        files: Vec::new(),
    };
    reader.read(path, None).map_err(|e| match e {
        Failure::Read(e) => e,
        Failure::Document(m) => fail(m),
    })?;
    let docs = reader
        .texts
        .iter()
        .map(|t| {
            Ok(Doc {
                doc: Document::parse(&t.text).map_err(|e| fail(e.to_string()))?,
                dir: t.path.parent().unwrap_or(Path::new("")),
                parent: t.parent,
                fallbacks: &t.fallbacks,
            })
        })
        .collect::<Result<Vec<_>>>()?;
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
    let graph = network(&docs, &reader.order, material, &mut load).map_err(fail)?;
    let mut solid = |rgb| {
        images.push(shading::solid_image(rgb));
        images.len() - 1
    };
    let mut host = shading::Host {
        meters_per_unit: 1.0,
        primvars: &["st"],
        reads: &[],
        solid: &mut solid,
    };
    let mapped = shading::map(&graph.net, graph.surface, graph.displacement, &mut host)
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
        files: reader.files,
        warnings,
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

/// One document's text.
struct Text {
    path: PathBuf,
    /// Its entry in [`Reader::seen`].
    key: PathBuf,
    text: String,
    /// The document that first included it.
    parent: Option<usize>,
    /// `xi:fallback` elements taking the place of includes that could not be read.
    fallbacks: Vec<NodeId>,
}

/// An `xi:include` of a document.
struct Include {
    id: NodeId,
    href: String,
    fallback: Option<NodeId>,
    /// The include whose fallback holds it, which it is read only in place of.
    within: Option<usize>,
}

/// One parsed document.
struct Doc<'t> {
    doc: Document<'t>,
    dir: &'t Path,
    parent: Option<usize>,
    fallbacks: &'t [NodeId],
}

/// Reads a document and all it includes.
struct Reader<'a> {
    memo: &'a Memo,
    texts: Vec<Text>,
    /// Indices of `texts`, each after those it includes.
    order: Vec<usize>,
    /// The chain of includes being read.
    open: Vec<PathBuf>,
    seen: HashSet<PathBuf>,
    files: Vec<FileInput>,
}

impl Reader<'_> {
    fn read(&mut self, path: &Path, parent: Option<usize>) -> std::result::Result<(), Failure> {
        let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_owned());
        if self.open.contains(&key) {
            return Err(format!("{} includes itself in a cycle", path.display()).into());
        }
        if self.seen.contains(&key) {
            return Ok(());
        }
        let (bytes, hash) = match self.memo.read(path) {
            Ok(read) => read,
            Err(e) => {
                self.files.push(FileInput {
                    path: path.to_owned(),
                    hash: FileInput::ABSENT,
                });
                return Err(Failure::Read(e));
            }
        };
        self.seen.insert(key.clone());
        self.files.push(FileInput {
            path: path.to_owned(),
            hash,
        });
        let in_file = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
        let mut text = String::from_utf8(bytes).map_err(|e| in_file(&e))?;
        let undeclared = matches!(
            Document::parse(&text),
            Err(roxmltree::Error::UnknownNamespace(prefix, _)) if prefix == "xi"
        );
        if undeclared && let Some(declared) = declare_xi(&text) {
            text = declared;
        }
        let doc = Document::parse(&text).map_err(|e| in_file(&e))?;
        let mut includes: Vec<Include> = Vec::new();
        for e in doc.descendants().filter(|e| is_xinclude(e, "include")) {
            let href = e
                .attribute("href")
                .ok_or_else(|| in_file(&"xi:include without an href"))?;
            let outer = e.ancestors().skip(1).find(|a| is_xinclude(a, "include"));
            includes.push(Include {
                id: e.id(),
                href: href.to_string(),
                fallback: e
                    .children()
                    .find(|c| is_xinclude(c, "fallback"))
                    .map(|f| f.id()),
                within: outer.and_then(|o| includes.iter().position(|i| i.id == o.id())),
            });
        }
        drop(doc);
        let id = self.texts.len();
        self.texts.push(Text {
            path: path.to_owned(),
            key: key.clone(),
            text,
            parent,
            fallbacks: Vec::new(),
        });
        self.open.push(key);
        let dir = path.parent().unwrap_or(Path::new(""));
        let mut fell_back = vec![false; includes.len()];
        let mut result = Ok(());
        for (i, include) in includes.iter().enumerate() {
            if include.within.is_some_and(|w| !fell_back[w]) {
                continue;
            }
            let (texts, order) = (self.texts.len(), self.order.len());
            match (
                self.read(&dir.join(&include.href), Some(id)),
                include.fallback,
            ) {
                (Err(Failure::Read(_)), Some(f)) => {
                    // What the failed include read is not part of the document, though
                    // another include may yet bring it in.
                    self.order.truncate(order);
                    for t in self.texts.drain(texts..) {
                        self.seen.remove(&t.key);
                    }
                    self.texts[id].fallbacks.push(f);
                    fell_back[i] = true;
                }
                (Err(e), _) => {
                    result = Err(e);
                    break;
                }
                (Ok(()), _) => {}
            }
        }
        self.open.pop();
        self.order.push(id);
        result
    }
}

/// `text` with the `xi` prefix declared on its root, which the specification's own snippets
/// leave out.
fn declare_xi(text: &str) -> Option<String> {
    let mut rest = text.trim_start_matches('\u{feff}');
    loop {
        rest = rest.trim_start();
        let end = if rest.starts_with("<?") {
            "?>"
        } else if rest.starts_with("<!--") {
            "-->"
        } else {
            break;
        };
        rest = &rest[rest.find(end)? + end.len()..];
    }
    let name = rest.find(|c: char| c.is_whitespace() || c == '>' || c == '/')?;
    let (head, tail) = text.split_at(text.len() - rest.len() + name);
    Some(format!("{head} xmlns:xi=\"{XINCLUDE}\"{tail}"))
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

/// Attribute `name` of `e`, which states nothing when empty.
fn stated<'a>(e: &XmlNode<'a, '_>, name: &str) -> Option<&'a str> {
    e.attribute(name).filter(|v| !v.is_empty())
}

/// The `unittype` and `unit` of a nodedef's input.
type Declared<'a> = (Option<&'a str>, Option<&'a str>);

/// The units of a document.
#[derive(Default)]
struct Units<'a> {
    /// Scales by unit type and unit.
    scales: BTreeMap<(&'a str, &'a str), f64>,
    /// What nodedefs declare, by node and input.
    declared: HashMap<(&'a str, &'a str), Declared<'a>>,
}

impl<'a> Units<'a> {
    /// Adds the units of a `unitdef`; the first definition of a unit stands. A scale that is not
    /// a number defines nothing, so it fails only a material that reads the unit.
    fn define(&mut self, e: &XmlNode<'a, '_>) {
        let unit_type = e.attribute("unittype").unwrap_or_default();
        for u in e.children().filter(|c| c.tag_name().name() == "unit") {
            let name = u.attribute("name").unwrap_or_default();
            if let Ok(scale) = u.attribute("scale").map_or(Ok(1.0), str::parse::<f64>) {
                self.scales.entry((unit_type, name)).or_insert(scale);
            }
        }
    }

    /// Adds what a `nodedef` declares; the first declaration of an input stands.
    fn declare(&mut self, e: &XmlNode<'a, '_>) {
        let node = e.attribute("node").unwrap_or_default();
        for i in e.children().filter(|c| c.tag_name().name() == "input") {
            let name = i.attribute("name").unwrap_or_default();
            let units = (stated(&i, "unittype"), stated(&i, "unit"));
            if units != (None, None) {
                self.declared.entry((node, name)).or_insert(units);
            }
        }
    }

    /// The unit of element `e`, input `input` of a `kind` node.
    fn of(
        &self,
        e: &XmlNode<'_, '_>,
        kind: &'a str,
        input: &'a str,
    ) -> std::result::Result<Unit, String> {
        let (declared_type, declared_unit) = self
            .declared
            .get(&(kind, input))
            .copied()
            .unwrap_or_default();
        let Some(unit) = stated(e, "unit").or(declared_unit) else {
            return Ok(Unit::Unstated);
        };
        let unit_type = declared_type.or(stated(e, "unittype"));
        let found = self
            .scales
            .iter()
            .filter(|((t, u), _)| *u == unit && unit_type.is_none_or(|ty| ty == *t))
            .min_by_key(|((t, _), _)| !matches!(*t, "distance" | "angle"));
        match found {
            Some((("distance", _), &s)) => Ok(Unit::Distance(s)),
            Some((("angle", _), &s)) => Ok(Unit::Angle(s)),
            Some(_) => Ok(Unit::Unstated),
            None => Err(format!(
                "{input}: no unit {unit} of type {}",
                unit_type.unwrap_or("any")
            )),
        }
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
    let mut units = Units::default();
    for &(unit_type, unit, scale) in UNITS {
        units.scales.insert((unit_type, unit), scale);
    }
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
        let fallbacks = docs[d]
            .fallbacks
            .iter()
            .filter_map(|&f| docs[d].doc.get_node(f))
            .flat_map(|f| f.children());
        for e in fallbacks.chain(root.children()).filter(XmlNode::is_element) {
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
                "unitdef" => units.define(&e),
                "nodedef" => units.declare(&e),
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
    for &(node, input, unit_type) in DECLARED {
        let declared = units.declared.entry((node, input));
        declared.or_insert((Some(unit_type), None));
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
    let mut warnings = Vec::new();
    // Filename inputs fed by a connection, which may end in a string, and their units.
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
            let unit = || {
                units
                    .of(&source.0, e.tag_name().name(), &input_name)
                    .map_err(|m| format!("{}: {m}", net.nodes[n].name))
            };
            let value = match connect(source.1, &source.0, *d)? {
                Some(c) => {
                    if source.0.attribute("type") == Some("filename") {
                        filenames.push((n, input_name.clone(), source.0, unit()?));
                    }
                    if let Input::Connect { node, .. } = c {
                        pending.push(node);
                    }
                    c
                }
                None => match source.0.attribute("value") {
                    None => continue,
                    Some(v) => value(docs, *d, &source.0, v, unit()?, load, &mut warnings),
                },
            };
            inputs.insert(input_name, value);
        }
        net.nodes[n].inputs = inputs;
    }
    for (n, input, e, unit) in filenames {
        let d = elements[n].2;
        if let Input::Text(file) = through(&net, &net.nodes[n].inputs[&input]) {
            let image = file_input(docs, d, &e, file, unit, load);
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
    unit: Unit,
    load: &mut dyn FnMut(&Path, &str) -> usize,
) -> Input {
    let prefix = inherited(docs, d, e, "fileprefix").unwrap_or_default();
    Input::Image {
        image: load(docs[d].dir, &format!("{prefix}{file}")),
        color_space: inherited(docs, d, e, "colorspace").map(str::to_string),
        unit,
    }
}

/// An input's authored value: an image for filenames, numbers, or text.
fn value(
    docs: &[Doc<'_>],
    d: usize,
    e: &XmlNode<'_, '_>,
    v: &str,
    unit: Unit,
    load: &mut dyn FnMut(&Path, &str) -> usize,
    warnings: &mut Vec<String>,
) -> Input {
    let ty = e.attribute("type").unwrap_or_default();
    match ty {
        "filename" => file_input(docs, d, e, v, unit, load),
        "string" => Input::Text(v.to_string()),
        "boolean" => Input::Value(vec![if v == "true" { 1.0 } else { 0.0 }], unit),
        _ => {
            let numbers = v
                .split(',')
                .map(|x| x.trim().parse::<f32>())
                .collect::<std::result::Result<Vec<_>, _>>();
            let Ok(mut numbers) = numbers else {
                return Input::Text(v.to_string());
            };
            if let (true, Some(space), &[r, g, b, ..]) = (
                is_color(ty),
                inherited(docs, d, e, "colorspace"),
                &numbers[..],
            ) {
                match ImageColor::named(space) {
                    Some(ImageColor::Named(s)) => {
                        numbers[..3].copy_from_slice(&s.to_rec709([r, g, b]));
                    }
                    Some(_) => {}
                    None => warnings.push(format!(
                        "{}: colorspace {space} reads as data",
                        e.attribute("name").unwrap_or_default()
                    )),
                }
            }
            Input::Value(numbers, unit)
        }
    }
}
