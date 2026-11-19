//! Entry names of what a scene source cooks, from which runtime code finds an entry by
//! [`AssetId::from_name`](crate::AssetId::from_name).
//!
//! A scene source `N` writes its whole-scene model as `N`, with that model's own entries under
//! it, and everything else under a kind: `N/model/...`, `N/material/...`, `N/texture/...`,
//! `N/neural/...`, `N/instances...` and `N/layout`. A model's own entries are `<model>/mesh`,
//! `<model>/subd`, `<model>/skeleton` and `<model>/clip/<clip>`. Every component that comes from
//! source data (a node or prim name, a prim path, a clip name) is [`escape`]d into exactly one
//! component; literals and indices are written as they are. So no two structured names meet,
//! whatever the source names hold: a model named by the source is one component under
//! `N/model`, one named by an index (`node/<i>`, `mesh/<i>`) has its index where a derived or
//! own entry has a literal, and the literals of kinds, own entries, derived models and
//! material variants differ from each other.

use std::borrow::Cow;
use std::fmt;

/// `s` as one name component: `%` becomes `%25` and `/` becomes `%2F`.
pub fn escape(s: &str) -> Cow<'_, str> {
    if !s.contains(['%', '/']) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        match c {
            '%' => out.push_str("%25"),
            '/' => out.push_str("%2F"),
            c => out.push(c),
        }
    }
    Cow::Owned(out)
}

/// The mesh of model entry `model`.
pub fn mesh(model: &str) -> String {
    format!("{model}/mesh")
}

/// A model of a scene source, relative to the source: what names it and what derives it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelName {
    base: ModelBase,
    derived: Vec<Derived>,
}

/// What names a model.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModelBase {
    /// The whole scene: `N`.
    Whole,
    /// A name from the source, such as a node or prim name or a prim path: `N/model/<name>`.
    Source(String),
    /// A node by index, where its name is empty or taken: `N/model/node/<i>`.
    Node(u32),
    /// A mesh of the source by index: `N/model/mesh/<i>`.
    Mesh(u32),
    /// The part of the scene no instancer places that layout partition `p` holds, for a
    /// partition held by a group: `N/model/partition/<p>`.
    Partition(u32),
}

/// How the importer derives a model from another, appended to that model's name.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Derived {
    /// Copies whose linear transform the mesh holds: `baked/<k>`.
    Baked(u32),
    /// Another set of morph weights: `morph/<k>`.
    Morph(u32),
    /// The meshes of one skeleton binding, by the binding prim's path: `binding/<path>`.
    Binding(String),
    /// What moves beside a prototype's skins: `moved`.
    Moved,
    /// A prototype whose prim's name and path other models take: `prototype`.
    Prototype,
    /// A repeated rigid piece of the mesh: `unit/<i>`.
    Unit(u32),
    /// What layout partition `p` holds of it, for a partition held by a group:
    /// `partition/<p>`.
    Partition(u32),
}

impl ModelName {
    pub const WHOLE: Self = Self::new(ModelBase::Whole);

    pub const fn new(base: ModelBase) -> Self {
        Self {
            base,
            derived: Vec::new(),
        }
    }

    pub fn source(name: impl Into<String>) -> Self {
        Self::new(ModelBase::Source(name.into()))
    }

    pub fn base(&self) -> &ModelBase {
        &self.base
    }

    pub fn is_whole(&self) -> bool {
        *self == Self::WHOLE
    }

    /// The layout partition whose part of the scene that no instancer places the model holds:
    /// 0 for the whole scene, `p` for [`ModelBase::Partition`]`(p)`.
    pub fn partition(&self) -> Option<u32> {
        match self.base {
            _ if !self.derived.is_empty() => None,
            ModelBase::Whole => Some(0),
            ModelBase::Partition(p) => Some(p),
            _ => None,
        }
    }

    /// The model derived from this one by `d`.
    pub fn with(&self, d: Derived) -> Self {
        let mut out = self.clone();
        out.derived.push(d);
        out
    }

    /// The model's entry in source `asset`.
    pub fn entry(&self, asset: &str) -> String {
        let mut out = asset.to_string();
        match &self.base {
            ModelBase::Whole => {}
            ModelBase::Source(s) => push(&mut out, &["model", &escape(s)]),
            ModelBase::Node(i) => push(&mut out, &["model", "node", &i.to_string()]),
            ModelBase::Mesh(i) => push(&mut out, &["model", "mesh", &i.to_string()]),
            ModelBase::Partition(p) => push(&mut out, &["model", "partition", &p.to_string()]),
        }
        for d in &self.derived {
            match d {
                Derived::Baked(k) => push(&mut out, &["baked", &k.to_string()]),
                Derived::Morph(k) => push(&mut out, &["morph", &k.to_string()]),
                Derived::Binding(p) => push(&mut out, &["binding", &escape(p)]),
                Derived::Moved => push(&mut out, &["moved"]),
                Derived::Prototype => push(&mut out, &["prototype"]),
                Derived::Unit(i) => push(&mut out, &["unit", &i.to_string()]),
                Derived::Partition(p) => push(&mut out, &["partition", &p.to_string()]),
            }
        }
        out
    }

    pub fn mesh(&self, asset: &str) -> String {
        mesh(&self.entry(asset))
    }

    pub fn subd(&self, asset: &str) -> String {
        self.entry(asset) + "/subd"
    }

    pub fn skeleton(&self, asset: &str) -> String {
        self.entry(asset) + "/skeleton"
    }

    /// The clip named `clip` of the model's skeleton.
    pub fn clip(&self, asset: &str, clip: &str) -> String {
        let mut out = self.entry(asset);
        push(&mut out, &["clip", &escape(clip)]);
        out
    }
}

/// The model's source name and what derives it, `/`-separated and unescaped, for people and
/// name patterns; empty for the whole scene.
impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        match &self.base {
            ModelBase::Whole => {}
            ModelBase::Source(s) => parts.push(s.clone()),
            ModelBase::Node(i) => parts.push(format!("node{i}")),
            ModelBase::Mesh(i) => parts.push(format!("mesh/{i}")),
            ModelBase::Partition(p) => parts.push(format!("partition/{p}")),
        }
        parts.extend(self.derived.iter().map(|d| match d {
            Derived::Baked(k) => format!("baked/{k}"),
            Derived::Morph(k) => format!("morph/{k}"),
            Derived::Binding(p) => p.clone(),
            Derived::Moved => "moved".into(),
            Derived::Prototype => "prototype".into(),
            Derived::Unit(i) => format!("unit/{i}"),
            Derived::Partition(p) => format!("partition/{p}"),
        }));
        f.write_str(&parts.join("/"))
    }
}

/// A material of a source, relative to the source.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialName {
    base: MaterialBase,
    variants: Vec<Variant>,
}

/// What names a material.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MaterialBase {
    /// The source's one material: `N`.
    Own,
    /// A material by index, or a section's: `N/material/<i>`.
    Index(u32),
    /// What faces without a material take: `N/material/default`.
    Default,
    /// A material prim by its absolute path, which starts with `/`: `N/material/<path>`.
    Path(String),
}

/// A copy of a material that differs in what a mesh needs of it, appended to its name.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Variant {
    /// The back of a material whose sides differ: `back`.
    Back,
    /// For double-sided meshes: `double_sided`.
    DoubleSided,
    /// For meshes that differ in what its primvar readers fetch: `primvars/<k>`.
    Primvars(u32),
    /// With another object space for its maps: `object/<n>`.
    Object(u32),
    /// Without its displacement, which a mesh holds: `static`.
    Static,
}

impl MaterialName {
    pub const OWN: Self = Self::new(MaterialBase::Own);

    pub const fn new(base: MaterialBase) -> Self {
        Self {
            base,
            variants: Vec::new(),
        }
    }

    /// Material `i` of the source, or that of section `i`: `N/material/<i>`.
    pub const fn index(i: u32) -> Self {
        Self::new(MaterialBase::Index(i))
    }

    /// The material's copy that differs by `v`.
    pub fn with(&self, v: Variant) -> Self {
        let mut out = self.clone();
        out.variants.push(v);
        out
    }

    /// Its components under `N/material`, `N/texture` and `N/neural`.
    fn components(&self) -> String {
        let mut out = String::new();
        match &self.base {
            MaterialBase::Own => {}
            MaterialBase::Index(i) => push(&mut out, &[&i.to_string()]),
            MaterialBase::Default => push(&mut out, &["default"]),
            MaterialBase::Path(p) => push(&mut out, &[&escape(p)]),
        }
        for v in &self.variants {
            match v {
                Variant::Back => push(&mut out, &["back"]),
                Variant::DoubleSided => push(&mut out, &["double_sided"]),
                Variant::Primvars(k) => push(&mut out, &["primvars", &k.to_string()]),
                Variant::Object(n) => push(&mut out, &["object", &n.to_string()]),
                Variant::Static => push(&mut out, &["static"]),
            }
        }
        out
    }

    /// The material's entry in source `asset`.
    pub fn entry(&self, asset: &str) -> String {
        match self.components() {
            c if c.is_empty() => asset.to_string(),
            c => format!("{asset}/material{c}"),
        }
    }

    /// The texture, or set of tiles, of `role` that the material reads.
    pub fn texture(&self, asset: &str, role: &str) -> String {
        format!("{asset}/texture{}/{role}", self.components())
    }

    /// The neural texture set of the material's textures, or of those of tile `cell`.
    pub fn neural(&self, asset: &str, cell: Option<[u16; 2]>) -> String {
        let out = format!("{asset}/neural{}", self.components());
        match cell {
            Some(c) => tile(&out, c),
            None => out,
        }
    }
}

/// Tile `[u, v]` of texture or neural set `name`.
pub fn tile(name: &str, [u, v]: [u16; 2]) -> String {
    format!("{name}/tile/{u}_{v}")
}

/// The latents texture of neural set `set`.
pub fn latents(set: &str) -> String {
    format!("{set}/latents")
}

/// A scene source's layout ([`crate::layout`]): `N/layout`.
pub fn layout(asset: &str) -> String {
    format!("{asset}/layout")
}

/// The `k`th instance source of scene source `asset`.
pub fn instances(asset: &str, k: usize) -> String {
    match k {
        0 => format!("{asset}/instances"),
        k => format!("{asset}/instances/{k}"),
    }
}

/// Page `i` of world cell `cell` of instance source `source`.
pub fn page(source: &str, [x, y, z]: [i32; 3], i: u32) -> String {
    format!("{source}/{x}_{y}_{z}/{i}")
}

/// The instance source, cell and index of `page` if it names a page.
pub fn parse_page(page: &str) -> Option<(&str, [i32; 3], u32)> {
    let (rest, index) = page.rsplit_once('/')?;
    let (source, cell) = rest.rsplit_once('/')?;
    let mut it = cell.split('_').map(str::parse::<i32>);
    let c = [it.next()?.ok()?, it.next()?.ok()?, it.next()?.ok()?];
    Some((source, c, index.parse().ok()?))
}

fn push(out: &mut String, components: &[&str]) {
    for c in components {
        out.push('/');
        out.push_str(c);
    }
}
