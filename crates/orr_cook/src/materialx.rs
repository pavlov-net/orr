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
//! An image's `layer` reads that layer of its file, and a file without one reads as a file that
//! can't be read. An image with none reads layer `default`, then `rgba`, then the whole file. A
//! filename's `<UDIM>` and `<UVTILE>` make it a set of tiles ([`crate::texture::tiles`]), and
//! `[token]` reads the value of a `token` of the nearest scope, then of the nodedef the graph
//! implements. A token with no value, a `{hostattr}` or `{frame}` (no host states one) and any
//! other `<geometry token>` (a standalone material has no geometry) leave the file unresolved, so
//! the image reads its `default`. A raw `<` in an attribute reads as the specification's own
//! filenames have it, though XML would have it escaped.
//!
//! A scope holds one list of names: the first of two elements of one name wins, whatever their
//! categories, as in the MaterialX library. An `xi:include` among the root's children adds the
//! children of the root of the document it names, read before the including document. Its
//! `xi:fallback` stands where it stood, reading what it includes first; an include anywhere else
//! reads nothing and warns, as the library reads nothing there. Root attributes an included
//! document leaves out come from the document including it. A `namespace` qualifies the names of
//! its scope: references within it and to the global namespace need no qualifier, others name
//! `namespace:element`, and so may the material.
//!
//! Not honored yet, and reported as a warning: a `backsurfaceshader` other than the
//! `surfaceshader`.

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use roxmltree::{Document, Node as XmlNode, NodeId};

use crate::import::{Scene, SceneMaterial};
use crate::shading::{self, Input, Network, Node, Unit, is_color, through};
use crate::state::{Input as FileInput, Memo};
use crate::texture::color::ImageColor;
use crate::texture::material::{ImageData, ImageFile, ImageTile, Layer};
use crate::texture::{self, tiles};
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

/// The `xi:fallback` an include reads when its document can't be read.
fn fallback_of<'a, 't>(include: &XmlNode<'a, 't>) -> Option<XmlNode<'a, 't>> {
    include.children().find(|c| is_xinclude(c, "fallback"))
}

/// The include whose fallback holds `e`.
fn outer_include<'a, 't>(e: &XmlNode<'a, 't>) -> Option<XmlNode<'a, 't>> {
    let fallback = e.parent_element().filter(|p| is_xinclude(p, "fallback"))?;
    let include = fallback.parent_element()?;
    (is_xinclude(&include, "include") && fallback_of(&include) == Some(fallback)).then_some(include)
}

/// Whether include `e` stands among the children of `root`, or in the fallback of one that does.
fn in_place<'a, 't>(e: &XmlNode<'a, 't>, root: &XmlNode<'a, 't>) -> bool {
    e.parent_element().as_ref() == Some(root)
        || outer_include(e).is_some_and(|outer| in_place(&outer, root))
}

/// The elements under `parent`, the first of each name.
fn first_named<'a, 't>(parent: XmlNode<'a, 't>) -> impl Iterator<Item = XmlNode<'a, 't>> {
    let mut names = HashSet::new();
    parent
        .children()
        .filter(XmlNode::is_element)
        .filter(move |e| {
            let name = e.attribute("name").unwrap_or_default();
            name.is_empty() || names.insert(name)
        })
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
        warnings: Vec::new(),
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
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut images = Vec::new();
    let mut by_path: HashMap<(PathBuf, Layer), usize> = HashMap::new();
    let mut listed = Vec::new();
    let mut past = Vec::new();
    let mut load = |dir: &Path, file: FileName, layer: Layer| -> usize {
        let (file, unresolved) = match file {
            Ok(file) => (file, None),
            Err((file, why)) => (file, Some(why)),
        };
        let p = dir.join(&file);
        let key = (p.clone(), layer);
        if let Some(&i) = by_path.get(&key) {
            return i;
        }
        let image = if let Some(why) = unresolved {
            ImageFile::unread(file, why)
        } else if tiles::tiled(&file) {
            let scan = tiles::scan(&p);
            past.extend(scan.past.iter().map(|f| texture::past_tile_warning(f)));
            listed.extend(scan.holders.into_iter().map(|path| {
                let hash = memo.hash(&path).unwrap_or(FileInput::ABSENT);
                FileInput { path, hash }
            }));
            let tiles = scan
                .files
                .into_iter()
                .map(|(cell, path)| {
                    images.push(ImageFile {
                        name: tiles::named(&file, cell).to_string_lossy().into_owned(),
                        data: ImageData::File {
                            path,
                            read: None,
                            layer: key.1.clone(),
                        },
                    });
                    let image = images.len() - 1;
                    ImageTile { cell, image }
                })
                .collect();
            ImageFile {
                name: file,
                data: ImageData::Tiles(tiles),
            }
        } else {
            ImageFile {
                name: file,
                data: ImageData::File {
                    path: p,
                    read: None,
                    layer: key.1.clone(),
                },
            }
        };
        images.push(image);
        by_path.insert(key, images.len() - 1);
        images.len() - 1
    };
    let graph = network(&docs, &reader.order, material, &mut load).map_err(fail)?;
    reader.files.append(&mut listed);
    let mut added: HashMap<String, usize> = HashMap::new();
    let mut add = |image: ImageFile| {
        *added.entry(image.name.clone()).or_insert_with(|| {
            images.push(image);
            images.len() - 1
        })
    };
    let mut host = shading::Host {
        meters_per_unit: 1.0,
        primvars: &["st"],
        reads: &[],
        any_mesh: true,
        add: &mut add,
    };
    let mapped = shading::map(&graph.net, graph.surface, graph.displacement, &mut host)
        .map_err(|m| fail(format!("material {material}: {m}")))?;
    let mut warnings = reader.warnings;
    warnings.extend(graph.warnings);
    warnings.extend(past);
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
}

/// What is read at one place in the order: the children of a document's root, or of an
/// `xi:fallback` of it taking the place of an include that could not be read.
#[derive(Clone, Copy)]
struct Part {
    doc: usize,
    fallback: Option<NodeId>,
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
}

impl<'t> Doc<'t> {
    /// The elements under the root, or under `fallback`.
    fn children(&self, fallback: Option<NodeId>) -> impl Iterator<Item = XmlNode<'_, 't>> {
        let root = self.doc.root_element();
        let parent = fallback.and_then(|f| self.doc.get_node(f)).unwrap_or(root);
        parent.children().filter(XmlNode::is_element)
    }
}

/// Reads a document and all it includes.
struct Reader<'a> {
    memo: &'a Memo,
    texts: Vec<Text>,
    /// What each of `texts` reads, each after what it includes.
    order: Vec<Part>,
    /// The chain of includes being read.
    open: Vec<PathBuf>,
    seen: HashSet<PathBuf>,
    files: Vec<FileInput>,
    warnings: Vec<String>,
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
        text = escape_attributes(&text);
        let undeclared = matches!(
            Document::parse(&text),
            Err(roxmltree::Error::UnknownNamespace(prefix, _)) if prefix == "xi"
        );
        if undeclared && let Some(declared) = declare_xi(&text) {
            text = declared;
        }
        let doc = Document::parse(&text).map_err(|e| in_file(&e))?;
        let mut includes: Vec<Include> = Vec::new();
        let root = doc.root_element();
        for e in doc.descendants().filter(|e| is_xinclude(e, "include")) {
            if !in_place(&e, &root) {
                self.warnings.push(in_file(
                    &"an xi:include below the root's children reads nothing",
                ));
                continue;
            }
            let href = e
                .attribute("href")
                .ok_or_else(|| in_file(&"xi:include without an href"))?;
            let outer = outer_include(&e);
            includes.push(Include {
                id: e.id(),
                href: href.to_string(),
                fallback: fallback_of(&e).map(|f| f.id()),
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
        });
        self.open.push(key);
        let dir = path.parent().unwrap_or(Path::new(""));
        let result = self.includes(id, dir, &includes, None);
        self.open.pop();
        self.order.push(Part {
            doc: id,
            fallback: None,
        });
        result
    }

    /// Reads the includes of document `id` that stand in the fallback of include `within`, or
    /// at its root.
    fn includes(
        &mut self,
        id: usize,
        dir: &Path,
        includes: &[Include],
        within: Option<usize>,
    ) -> std::result::Result<(), Failure> {
        for (i, include) in includes.iter().enumerate() {
            if include.within != within {
                continue;
            }
            let (texts, order, warnings) =
                (self.texts.len(), self.order.len(), self.warnings.len());
            match (
                self.read(&dir.join(&include.href), Some(id)),
                include.fallback,
            ) {
                (Err(Failure::Read(_)), Some(f)) => {
                    // What the failed include read is not part of the document, though
                    // another include may yet bring it in.
                    self.order.truncate(order);
                    self.warnings.truncate(warnings);
                    for t in self.texts.drain(texts..) {
                        self.seen.remove(&t.key);
                    }
                    self.includes(id, dir, includes, Some(i))?;
                    self.order.push(Part {
                        doc: id,
                        fallback: Some(f),
                    });
                }
                (result, _) => result?,
            }
        }
        Ok(())
    }
}

/// `text` with each `<` in an attribute value escaped.
fn escape_attributes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        let (before, tag) = rest.split_at(open);
        out.push_str(before);
        let skipped = [("<!--", "-->"), ("<![CDATA[", "]]>"), ("<?", "?>")]
            .into_iter()
            .find(|(start, _)| tag.starts_with(start))
            .map(|(_, end)| tag.find(end).map_or(tag.len(), |at| at + end.len()));
        if let Some(len) = skipped {
            out.push_str(&tag[..len]);
            rest = &tag[len..];
            continue;
        }
        let mut quote = None;
        let mut end = tag.len();
        for (at, c) in tag.char_indices() {
            match (quote, c) {
                (None, '>') => {
                    out.push('>');
                    end = at + 1;
                    break;
                }
                (None, '"' | '\'') => quote = Some(c),
                (Some(q), _) if q == c => quote = None,
                _ => {}
            }
            if c == '<' && quote.is_some() {
                out.push_str("&lt;");
            } else {
                out.push(c);
            }
        }
        rest = &tag[end..];
    }
    out.push_str(rest);
    out
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
        for u in first_named(*e).filter(|c| c.tag_name().name() == "unit") {
            let name = u.attribute("name").unwrap_or_default();
            if let Ok(scale) = u.attribute("scale").map_or(Ok(1.0), str::parse::<f64>) {
                self.scales.entry((unit_type, name)).or_insert(scale);
            }
        }
    }

    /// Adds what a `nodedef` declares; the first declaration of an input stands.
    fn declare(&mut self, e: &XmlNode<'a, '_>) {
        let node = e.attribute("node").unwrap_or_default();
        for i in first_named(*e).filter(|c| c.tag_name().name() == "input") {
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

/// The nodedefs read, each by its name and by its qualified name; the first read of a name stands.
type Nodedefs<'a, 't> = HashMap<String, XmlNode<'a, 't>>;

/// A filename with its substitutions made, or as it stands, with why it can't be resolved.
type FileName = std::result::Result<String, (String, String)>;

/// Adds the image of a filename, relative to a directory, read at a layer of its file, and returns
/// its index.
type Load<'a> = dyn FnMut(&Path, FileName, Layer) -> usize + 'a;

/// The network of `material` in `docs`, the first of them the document cooked, walked in
/// `order`.
fn network(
    docs: &[Doc<'_>],
    order: &[Part],
    material: &str,
    load: &mut Load<'_>,
) -> std::result::Result<Graph, String> {
    let mut net = Network::default();
    // Nodes by (graph, name); the top level is graph "".
    let mut index: HashMap<(String, String), usize> = HashMap::new();
    let mut elements = Vec::new();
    let mut graphs = HashMap::new();
    let mut nodedefs = Nodedefs::new();
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
    // The top level holds one list of names, whatever the elements' categories.
    let mut names = HashSet::new();
    for &Part { doc: d, fallback } in order {
        let root = docs[d].doc.root_element();
        if root.tag_name().name() != "materialx" {
            return Err(format!(
                "root element {} is not materialx",
                root.tag_name().name()
            ));
        }
        for e in docs[d].children(fallback) {
            let name = e.attribute("name").unwrap_or_default();
            let full_name = qualified(inherited(docs, d, &e, "namespace"), name);
            if !name.is_empty() && !names.insert(full_name.clone()) {
                continue;
            }
            match e.tag_name().name() {
                // Graph "" is the top level, so a graph with no name holds nothing to read.
                "nodegraph" if !name.is_empty() => {
                    graphs.insert(full_name.clone(), (e, d));
                    for n in first_named(e).filter(is_node) {
                        let name = n.attribute("name").unwrap_or_default().to_string();
                        add(&full_name, name, n, d);
                    }
                }
                "unitdef" => units.define(&e),
                "nodedef" => {
                    units.declare(&e);
                    nodedefs.entry(name.to_string()).or_insert(e);
                    nodedefs.entry(full_name).or_insert(e);
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
            let outputs: Vec<_> = first_named(*ge)
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
    // Filename inputs and their units, read once every node is: a connection may end in a string,
    // and what an image reads of its file is another of its inputs.
    let mut filenames = Vec::new();
    while let Some(n) = pending.pop() {
        if std::mem::replace(&mut resolved[n], true) {
            continue;
        }
        let (graph, e, d) = &elements[n];
        let mut inputs = BTreeMap::new();
        for i in first_named(*e).filter(|c| c.tag_name().name() == "input") {
            let input_name = i.attribute("name").unwrap_or_default().to_string();
            // Inside a graph, an interface name reads the graph's own input.
            let source = match (i.attribute("interfacename"), graph.as_str()) {
                (Some(iface), g) if !g.is_empty() => {
                    let (ge, _) = graphs.get(g).ok_or_else(|| format!("no nodegraph {g}"))?;
                    let gi = first_named(*ge)
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
                    Some("") if source.0.attribute("type") == Some("filename") => continue,
                    Some(v) if source.0.attribute("type") == Some("filename") => {
                        filenames.push((n, input_name.clone(), source.0, unit()?));
                        Input::Text(v.to_string())
                    }
                    Some(v) => value(docs, *d, &source.0, v, unit()?, &mut warnings),
                },
            };
            inputs.insert(input_name, value);
        }
        net.nodes[n].inputs = inputs;
    }
    let mut files = Vec::new();
    for f in &filenames {
        let read = through(&net, &net.nodes[f.0].inputs[&f.1]);
        let file = match read {
            Input::Text(file) if !file.is_empty() => file,
            Input::Connect { .. } => continue,
            // A constant or dot that holds no name leaves the input naming no file.
            _ => {
                files.push((f.0, &f.1, None));
                continue;
            }
        };
        // A value keeps the scope of the input that states it, wherever it is read.
        let (n, _, e, unit) = filenames
            .iter()
            .find(|s| std::ptr::eq(&net.nodes[s.0].inputs[&s.1], read))
            .unwrap_or(f);
        let layer = match net.nodes[f.0].inputs.get("layer").map(|l| through(&net, l)) {
            Some(Input::Text(l)) if !l.is_empty() => Layer::Named(l.clone()),
            _ => Layer::Implied,
        };
        let d = elements[*n].2;
        let name = file_name(docs, &nodedefs, d, e, file);
        let mut image = file_input(docs, d, e, name, *unit, layer, load);
        // A colorspace is how the input reading the file decodes it, wherever the name comes from.
        if let (Input::Image { color_space, .. }, Some(own)) =
            (&mut image, f.2.attribute("colorspace"))
        {
            *color_space = Some(own.to_string());
        }
        files.push((f.0, &f.1, Some(image)));
    }
    for (n, input, image) in files {
        match image {
            Some(image) => net.nodes[n].inputs.insert(input.clone(), image),
            None => net.nodes[n].inputs.remove(input),
        };
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

/// The value of `token` for element `e`: a `token` holding it on the nearest scope, which may name
/// one further out through `interfacename`, else the default of the nodedef a graph implements.
fn interface_token(
    nodedefs: &Nodedefs<'_, '_>,
    e: &XmlNode<'_, '_>,
    token: &str,
) -> Option<String> {
    let named = |scope: &XmlNode<'_, '_>, name: &str| {
        let is_it = |c: &XmlNode<'_, '_>| {
            c.tag_name().name() == "token" && c.attribute("name") == Some(name)
        };
        first_named(*scope).find(is_it).map(|t| {
            (
                t.attribute("value").map(str::to_string),
                t.attribute("interfacename").map(str::to_string),
            )
        })
    };
    let mut name = token.to_string();
    for scope in e.ancestors() {
        if let Some((value, interface)) = named(&scope, &name) {
            if value.is_some() {
                return value;
            }
            name = interface.unwrap_or(name);
        }
        if let Some(def) = scope.attribute("nodedef").and_then(|n| nodedefs.get(n))
            && let Some((Some(value), _)) = named(def, &name)
        {
            return Some(value);
        }
    }
    None
}

/// `file` with its substitutions made, all but the tile tokens each point picks from, or the
/// reason one has no value.
fn substituted(
    nodedefs: &Nodedefs<'_, '_>,
    e: &XmlNode<'_, '_>,
    file: &str,
) -> std::result::Result<String, String> {
    let mut out = String::new();
    let mut rest = file;
    while let Some(at) = rest.find(['[', '{', '<']) {
        let close = match rest[at..].chars().next() {
            Some('[') => ']',
            Some('{') => '}',
            _ => '>',
        };
        let Some(len) = rest[at..].find(close) else {
            break;
        };
        let whole = &rest[at..=at + len];
        out.push_str(&rest[..at]);
        match close {
            ']' => {
                let value = interface_token(nodedefs, e, &whole[1..whole.len() - 1]);
                out.push_str(&value.ok_or_else(|| format!("{whole} has no value"))?);
            }
            '>' if tiles::tiled(whole) => out.push_str(whole),
            _ => return Err(format!("{whole} has no value")),
        }
        rest = &rest[at + len + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// The file a filename value of element `e` in document `d` names.
fn file_name(
    docs: &[Doc<'_>],
    nodedefs: &Nodedefs<'_, '_>,
    d: usize,
    e: &XmlNode<'_, '_>,
    file: &str,
) -> FileName {
    let prefix = inherited(docs, d, e, "fileprefix").unwrap_or_default();
    let file = format!("{prefix}{file}");
    substituted(nodedefs, e, &file).map_err(|why| (file, why))
}

/// The image of file `name`, read by element `e` in document `d`.
fn file_input(
    docs: &[Doc<'_>],
    d: usize,
    e: &XmlNode<'_, '_>,
    name: FileName,
    unit: Unit,
    layer: Layer,
    load: &mut Load<'_>,
) -> Input {
    Input::Image {
        image: load(docs[d].dir, name, layer),
        color_space: inherited(docs, d, e, "colorspace").map(str::to_string),
        unit,
    }
}

/// An input's authored value, other than a filename: numbers or text.
fn value(
    docs: &[Doc<'_>],
    d: usize,
    e: &XmlNode<'_, '_>,
    v: &str,
    unit: Unit,
    warnings: &mut Vec<String>,
) -> Input {
    let ty = e.attribute("type").unwrap_or_default();
    match ty {
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
