//! Link points: consumer Slang types linked into orr's kernels at build time.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::Error;
use super::lint::tokens;
use super::model;
use super::roots::Root;
use super::slang::Session;
use super::sys;

/// A link-time extension point of orr's kernels. A kernel uses a point when its source
/// declares `extern struct <Name> : I<Point> = <Default>;`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkPoint {
    /// `ISurface`: one switch on the surface id, `NoSurface` for id 0, which the kernels'
    /// `ISurfaces` takes through `Surfaces`.
    Surface,
    /// `IDisplacement`: one switch on the displacement id, `TextureDisplacement` for id 0.
    Displacement,
    /// `IDrawn`: one switch on the drawn id, `NoDrawn` for id 0.
    Drawn,
    /// `IEnvironment`: at most one type per build.
    Environment,
    /// `IMedium`: at most one type per build.
    Medium,
}

/// One method of the linked types' interface that the generated switch forwards.
struct Method {
    /// The declaration without `static`.
    decl: &'static str,
    name: &'static str,
    /// The arguments it forwards.
    args: &'static str,
    /// The interface gives it a body, so the switch type marks it `override`.
    optional: bool,
}

/// A point whose linked types are chosen per call by an id.
struct Switch {
    /// The expression that reads the id; every method's first parameter is `i`.
    id: &'static str,
    /// The type for id 0 and every id nothing links.
    default: &'static str,
    /// The interface the linked types implement, with the methods the switch forwards.
    linked: &'static str,
    methods: &'static [Method],
    /// The generic over the switch that the kernels' interface takes, where the kernels use
    /// another interface than the linked types.
    wrap: Option<&'static str>,
}

/// The surface interface's methods, every one optional: `Surfaces` over the switch calls the
/// standard surface once at a site for every linked type that leaves a hit to it.
const SURFACE: Switch = Switch {
    id: "i.surface",
    default: "NoSurface",
    linked: "ISurface",
    methods: &[
        Method {
            decl: "bool surface(SurfaceInput i, out GSurface g, out float depth_offset)",
            name: "surface",
            args: "i, g, depth_offset",
            optional: true,
        },
        Method {
            decl: "void adjust(SurfaceInput i, inout GSurface g, inout float depth_offset)",
            name: "adjust",
            args: "i, g, depth_offset",
            optional: true,
        },
        Method {
            decl: "bool coverage(SurfaceInput i, out float coverage)",
            name: "coverage",
            args: "i, coverage",
            optional: true,
        },
        Method {
            decl: "void feedback(SurfaceInput i)",
            name: "feedback",
            args: "i",
            optional: true,
        },
        Method {
            decl: "void coverage_feedback(SurfaceInput i)",
            name: "coverage_feedback",
            args: "i",
            optional: true,
        },
    ],
    wrap: Some("Surfaces"),
};

const DISPLACEMENT: Switch = Switch {
    id: "i.displacement",
    default: "TextureDisplacement",
    linked: "IDisplacement",
    methods: &[Method {
        decl: "float3 displace(DisplacementInput i)",
        name: "displace",
        args: "i",
        optional: false,
    }],
    wrap: None,
};

const DRAWN: Switch = Switch {
    id: "i.drawn",
    default: "NoDrawn",
    linked: "IDrawn",
    methods: &[Method {
        decl: "DrawnTriangle triangle(DrawnInput i)",
        name: "triangle",
        args: "i",
        optional: false,
    }],
    wrap: None,
};

impl LinkPoint {
    const ALL: [Self; 5] = [
        Self::Surface,
        Self::Displacement,
        Self::Drawn,
        Self::Environment,
        Self::Medium,
    ];

    /// The interface of the kernels' `extern struct`.
    fn interface(self) -> &'static str {
        match self {
            Self::Surface => "ISurfaces",
            Self::Displacement => "IDisplacement",
            Self::Drawn => "IDrawn",
            Self::Environment => "IEnvironment",
            Self::Medium => "IMedium",
        }
    }

    /// The module that declares the interface.
    fn module(self) -> &'static str {
        match self {
            Self::Surface => "orr_scene.surface",
            Self::Displacement => "orr_scene.displacement",
            Self::Drawn => "orr_scene.drawn",
            Self::Environment => "orr_lighting.environment",
            Self::Medium => "orr_lighting.media",
        }
    }

    fn switch(self) -> Option<&'static Switch> {
        match self {
            Self::Surface => Some(&SURFACE),
            Self::Displacement => Some(&DISPLACEMENT),
            Self::Drawn => Some(&DRAWN),
            Self::Environment | Self::Medium => None,
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Displacement => "displacement",
            Self::Drawn => "drawn",
            Self::Environment => "environment",
            Self::Medium => "medium",
        }
    }
}

/// One `Build::link` call.
#[derive(Clone, Debug)]
pub(super) struct Link {
    pub(crate) point: LinkPoint,
    pub(crate) name: String,
    pub(super) module: PathBuf,
    pub(super) ty: String,
    /// The name `module` is imported by when it is under a prefixed root; set by [`resolve`].
    import: Option<String>,
}

impl Link {
    pub(crate) fn new(point: LinkPoint, name: &str, module: &Path, ty: &str) -> Self {
        Self {
            point,
            name: name.to_owned(),
            module: module.to_path_buf(),
            ty: ty.to_owned(),
            import: None,
        }
    }

    fn import(&self) -> String {
        self.import.clone().unwrap_or_else(|| {
            self.module
                .file_stem()
                .map_or_default(|s| s.to_string_lossy().into_owned())
        })
    }
}

/// A point a kernel declares, with the name of its `extern struct`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) struct Declared {
    pub(crate) point: LinkPoint,
    pub(crate) name: String,
}

/// One kernel file in `link_points.json`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(super) struct KernelPoints {
    /// The kernel's path under its root, without extension.
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(super) root: Root,
    pub(super) points: Vec<Declared>,
}

/// `link_points.json`: every kernel file that declares a link point.
#[derive(Default, Serialize, Deserialize)]
pub(super) struct LinkPoints {
    pub(super) kernels: Vec<KernelPoints>,
}

impl LinkPoints {
    pub(super) fn read(path: &Path) -> Result<Self, Error> {
        serde_json::from_str(&super::compile::read(path)?).map_err(|e| Error::Link {
            message: format!("{} is not a link-points file: {e}", path.display()),
        })
    }

    /// Adds `other`'s kernels; a kernel file listed twice is kept once.
    pub(super) fn merge(&mut self, other: Self) {
        self.kernels.extend(other.kernels);
        self.kernels.sort();
        self.kernels.dedup();
    }
}

/// The link points `src` declares.
pub(super) fn declared(src: &str) -> Vec<Declared> {
    let t = tokens(src);
    let mut out = Vec::new();
    for w in t.windows(4) {
        if w[0].0 != "extern" || w[1].0 != "struct" {
            continue;
        }
        if let Some(point) = LinkPoint::ALL.into_iter().find(|p| p.interface() == w[3].0) {
            out.push(Declared {
                point,
                name: w[2].0.to_owned(),
            });
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The `link` calls with their module paths resolved against `CARGO_MANIFEST_DIR` and their
/// import names against `roots`, after checking names, ids and counts.
pub(super) fn resolve(links: &[Link], roots: &[Root]) -> Result<Vec<Link>, Error> {
    check(links)?;
    links
        .iter()
        .map(|l| {
            let module = resolve_module(&l.module)?;
            Ok(Link {
                import: roots.iter().find_map(|r| r.module_name(&module)),
                module,
                ..l.clone()
            })
        })
        .collect()
}

fn check(links: &[Link]) -> Result<(), Error> {
    let fail = |message: String| Err(Error::Link { message });
    let ident = |s: &str| {
        s.starts_with(|c: char| c.is_ascii_alphabetic())
            && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    let mut names = BTreeSet::new();
    for l in links {
        if !ident(&l.name) || l.name.chars().any(|c| c.is_ascii_uppercase()) {
            return fail(format!(
                "link name {:?} is not a lower-case identifier",
                l.name
            ));
        }
        if !ident(&l.ty) {
            return fail(format!("{:?} is not a Slang type name", l.ty));
        }
        if !names.insert((l.point, l.name.as_str())) {
            return fail(format!("{} {:?} is linked twice", l.point.label(), l.name));
        }
    }
    for point in LinkPoint::ALL {
        let n = links.iter().filter(|l| l.point == point).count();
        let max = if point.switch().is_some() { 255 } else { 1 };
        if n > max {
            return fail(format!(
                "{n} {} types are linked; at most {max} fit",
                point.label()
            ));
        }
    }
    Ok(())
}

/// The id constants of each switch point, in link order from 1.
pub(super) fn ids(links: &[Link]) -> BTreeMap<LinkPoint, Vec<(&str, u8)>> {
    let mut out: BTreeMap<LinkPoint, Vec<(&str, u8)>> = BTreeMap::new();
    for point in LinkPoint::ALL.into_iter().filter(|p| p.switch().is_some()) {
        out.insert(point, Vec::new());
    }
    for l in links {
        if let Some(list) = out.get_mut(&l.point) {
            let id = u8::try_from(list.len() + 1).expect("checked: at most 255");
            list.push((&l.name, id));
        }
    }
    out
}

/// The type linked at `point`, which holds at most one.
pub(super) fn single_type(links: &[Link], point: LinkPoint) -> Option<&str> {
    links
        .iter()
        .find(|l| l.point == point)
        .map(|l| l.ty.as_str())
}

/// The width of the coherence key: 3 plus the bit length of the highest
/// linked surface id.
fn key_bits(links: &[Link]) -> u32 {
    let surfaces = links
        .iter()
        .filter(|l| l.point == LinkPoint::Surface)
        .count() as u32;
    3 + surfaces.bit_width()
}

/// A generated link module: one per point and declared name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct LinkModule {
    /// The Slang module name.
    pub(super) module: String,
    pub(crate) path: PathBuf,
    pub(super) source: String,
    pub(super) point: LinkPoint,
    /// The linked types, each with the canonical path of its module.
    pub(super) types: Vec<(String, PathBuf)>,
}

/// Writes the link module for `decl` under `dir` and returns it; `None` when nothing links
/// to its point.
pub(super) fn write_module(
    decl: &Declared,
    links: &[Link],
    dir: &Path,
) -> Result<Option<LinkModule>, Error> {
    let linked: Vec<&Link> = links.iter().filter(|l| l.point == decl.point).collect();
    if linked.is_empty() {
        return Ok(None);
    }
    let module = format!("orr_link_{}_{}", decl.point.label(), decl.name);
    let path = dir.join(format!("{module}.slang"));
    let source = source(decl, &linked, key_bits(links));
    super::compile::write(&path, source.as_bytes())?;
    Ok(Some(LinkModule {
        module,
        path,
        source,
        point: decl.point,
        types: linked
            .iter()
            .map(|l| (l.ty.clone(), l.module.clone()))
            .collect(),
    }))
}

/// The Slang of one link module.
fn source(decl: &Declared, linked: &[&Link], key_bits: u32) -> String {
    let point = decl.point;
    let mut out = format!(
        "// Generated by orr_shaders::build: the {} link point. Do not edit.\nimport {};\n",
        point.label(),
        point.module()
    );
    let imports: BTreeSet<String> = linked.iter().map(|l| l.import()).collect();
    for i in imports {
        let _ = writeln!(out, "import {i};");
    }
    let Some(switch) = point.switch() else {
        let _ = writeln!(
            out,
            "\nexport struct {} : {} = {};",
            decl.name,
            point.interface(),
            linked[0].ty
        );
        return out;
    };
    // The switch is the kernels' type, or with `wrap` the type the kernels' type wraps.
    let (ty, export) = match switch.wrap {
        Some(_) => (format!("{}Links", decl.name), ""),
        None => (decl.name.clone(), "export "),
    };
    let _ = writeln!(out, "\n{export}struct {ty} : {}\n{{", switch.linked);
    for m in switch.methods {
        let modifier = if m.optional { "override " } else { "" };
        // A void method's cases call, then return: `return` takes no void expression.
        let (call, end) = if m.decl.starts_with("void ") {
            ("", " return;")
        } else {
            ("return ", "")
        };
        let _ = writeln!(
            out,
            "    {modifier}static {}\n    {{\n        switch ({})\n        {{",
            m.decl, switch.id
        );
        for (i, l) in linked.iter().enumerate() {
            let _ = writeln!(
                out,
                "        case {}: {call}{}.{}({});{end}",
                i + 1,
                l.ty,
                m.name,
                m.args
            );
        }
        let _ = writeln!(
            out,
            "        default: {call}{}.{}({});{end}\n        }}\n    }}",
            switch.default, m.name, m.args
        );
    }
    out.push_str("}\n");
    if let Some(wrap) = switch.wrap {
        let _ = writeln!(
            out,
            "\nexport struct {} : {} = {wrap}<{ty}>;",
            decl.name,
            point.interface()
        );
    }
    if point == LinkPoint::Surface {
        let _ = writeln!(out, "\nexport static const uint KEY_BITS = {key_bits};");
    }
    out
}

/// Checks that where a linked type of `m` declares methods under the name of an optional method
/// of its interface, one of them overrides it. One with another signature is a method of its
/// own: Slang takes it without a word, and the switch calls the interface's default in its
/// place. `session` has loaded `m` and the modules it imports.
pub(super) fn check_overrides(session: &Session, m: &LinkModule) -> Result<(), String> {
    use sys::SlangDeclKind as K;
    let Some(switch) = m.point.switch() else {
        return Ok(());
    };
    if !switch.methods.iter().any(|x| x.optional) {
        return Ok(());
    }
    let api = &session.lib.api;
    let mut wrong = Vec::new();
    let mut stack: Vec<*mut sys::SlangReflectionDecl> = session
        .loaded_modules()
        .iter()
        .map(super::slang::Module::reflection)
        .filter(|d| !d.is_null())
        .collect();
    // SAFETY: decls come from modules owned by the live session.
    unsafe {
        while let Some(decl) = stack.pop() {
            for i in 0..api.spReflectionDecl_getChildrenCount(decl) {
                let child = api.spReflectionDecl_getChild(decl, i);
                if child.is_null() {
                    continue;
                }
                let kind = api.spReflectionDecl_getKind(child);
                if kind == K::SLANG_DECL_KIND_NAMESPACE {
                    stack.push(child);
                }
                if kind != K::SLANG_DECL_KIND_STRUCT {
                    continue;
                }
                // A linked type is the struct of its name in its own module.
                let ty = model::text(api.spReflectionDecl_getName(child));
                let linked = m.types.iter().any(|(t, module)| {
                    *t == ty
                        && session.decl_location(child).is_some_and(|(file, _)| {
                            super::compile::canonical_lossy(&file) == *module
                        })
                });
                if !linked {
                    continue;
                }
                // The signatures the type declares under each optional method's name.
                let mut declared: Vec<(&Method, String)> = Vec::new();
                for j in 0..api.spReflectionDecl_getChildrenCount(child) {
                    let f = api.spReflectionDecl_getChild(child, j);
                    if f.is_null() || api.spReflectionDecl_getKind(f) != K::SLANG_DECL_KIND_FUNC {
                        continue;
                    }
                    let f = api.spReflectionDecl_castToFunction(f);
                    let name = model::text(api.spReflectionFunction_GetName(f));
                    if let Some(method) =
                        switch.methods.iter().find(|x| x.optional && x.name == name)
                    {
                        declared.push((method, signature(api, f)));
                    }
                }
                for method in switch.methods.iter().filter(|x| x.optional) {
                    let wanted = declared_signature(method.decl);
                    let found = declared.iter().filter(|(x, _)| x.name == method.name);
                    if found.clone().any(|(_, s)| *s == wanted) {
                        continue;
                    }
                    for (_, s) in found {
                        wrong.push(format!(
                            "{ty}.{} is `{s}`, where {} declares `{wanted}`",
                            method.name, switch.linked
                        ));
                    }
                }
            }
        }
    }
    if wrong.is_empty() {
        return Ok(());
    }
    wrong.sort();
    wrong.dedup();
    Err(format!(
        "linked {} types declare methods named as the interface's of which none overrides it, \
         so the switch calls the interface's default: {}",
        m.point.label(),
        wrong.join("; ")
    ))
}

/// `f`'s signature as `ret name(modifier type, ...)`, without parameter names.
///
/// # Safety
/// `f` is a live function reflection of a session `api` belongs to.
unsafe fn signature(api: &sys::SlangLib, f: *mut sys::SlangReflectionFunction) -> String {
    use sys::SlangModifierID as M;
    // SAFETY: the caller's.
    unsafe {
        let ret =
            model::text(api.spReflectionType_GetName(api.spReflectionFunction_GetResultType(f)));
        let name = model::text(api.spReflectionFunction_GetName(f));
        let params: Vec<String> = (0..api.spReflectionFunction_GetParameterCount(f))
            .map(|k| {
                let p = api.spReflectionFunction_GetParameter(f, k);
                let modifier = if !api
                    .spReflectionVariable_FindModifier(p, M::SLANG_MODIFIER_INOUT)
                    .is_null()
                {
                    "inout "
                } else if !api
                    .spReflectionVariable_FindModifier(p, M::SLANG_MODIFIER_OUT)
                    .is_null()
                {
                    "out "
                } else {
                    ""
                };
                let ty =
                    model::text(api.spReflectionType_GetName(api.spReflectionVariable_GetType(p)));
                format!("{modifier}{ty}")
            })
            .collect();
        format!("{ret} {name}({})", params.join(", "))
    }
}

/// A method table declaration's signature as [`signature`] gives it.
fn declared_signature(decl: &str) -> String {
    let (head, params) = decl.split_once('(').expect("a declaration has parameters");
    let params: Vec<&str> = params
        .trim_end_matches(')')
        .split(',')
        .map(|p| {
            p.trim()
                .rsplit_once(' ')
                .expect("a declared parameter has a name")
                .0
        })
        .collect();
    format!("{head}({})", params.join(", "))
}

/// The directories of the linked modules imported by their file stems, as import roots.
pub(super) fn module_dirs(links: &[Link]) -> Vec<Root> {
    let mut dirs: Vec<Root> = links
        .iter()
        .filter(|l| l.import.is_none())
        .filter_map(|l| l.module.parent().map(Root::plain))
        .collect();
    dirs.sort();
    dirs.dedup();
    dirs
}

fn resolve_module(module: &Path) -> Result<PathBuf, Error> {
    let path = if module.is_absolute() {
        module.to_path_buf()
    } else {
        std::env::var_os("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .ok_or(Error::NoManifestDir)?
            .join(module)
    };
    super::compile::canonical(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_points_skip_comments() {
        let src = "// extern struct Old : ISurfaces = StandardSurface;\n\
                   import orr_scene.surface;\nextern struct Surface : ISurfaces = StandardSurface;\n\
                   extern struct Medium : IMedium = NoMedium;\n";
        assert_eq!(
            declared(src),
            [
                Declared {
                    point: LinkPoint::Surface,
                    name: "Surface".into()
                },
                Declared {
                    point: LinkPoint::Medium,
                    name: "Medium".into()
                },
            ]
        );
    }
}
