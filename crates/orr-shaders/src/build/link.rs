//! Link points: consumer Slang types linked into orr's kernels at build time.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::Error;
use super::lint::tokens;

/// A link-time extension point of orr's kernels. A kernel uses a point when its source
/// declares `extern struct <Name> : I<Point> = <Default>;`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkPoint {
    /// `ISurface`: one switch on the surface id, `StandardSurface` for id 0.
    Surface,
    /// `IDisplacement`: one switch on the displacement id, `TextureDisplacement` for id 0.
    Displacement,
    /// `IEnvironment`: at most one type per build.
    Environment,
    /// `IMedium`: at most one type per build.
    Medium,
}

/// One interface method the generated switch forwards.
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
    default: &'static str,
    methods: &'static [Method],
}

/// The interface's methods as contracts.md 6 defines them. A link-point interface change
/// edits this table in the same change.
const SURFACE: Switch = Switch {
    id: "i.surface",
    default: "StandardSurface",
    methods: &[
        Method {
            decl: "GSurface surface(SurfaceInput i, out float depth_offset)",
            name: "surface",
            args: "i, depth_offset",
            optional: false,
        },
        Method {
            decl: "float coverage(SurfaceInput i)",
            name: "coverage",
            args: "i",
            optional: true,
        },
    ],
};

const DISPLACEMENT: Switch = Switch {
    id: "i.displacement",
    default: "TextureDisplacement",
    methods: &[Method {
        decl: "float3 displace(DisplacementInput i)",
        name: "displace",
        args: "i",
        optional: false,
    }],
};

impl LinkPoint {
    const ALL: [Self; 4] = [
        Self::Surface,
        Self::Displacement,
        Self::Environment,
        Self::Medium,
    ];

    fn interface(self) -> &'static str {
        match self {
            Self::Surface => "ISurface",
            Self::Displacement => "IDisplacement",
            Self::Environment => "IEnvironment",
            Self::Medium => "IMedium",
        }
    }

    /// The module that declares the interface.
    fn module(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Displacement => "displacement",
            Self::Environment => "environment",
            Self::Medium => "media",
        }
    }

    fn switch(self) -> Option<&'static Switch> {
        match self {
            Self::Surface => Some(&SURFACE),
            Self::Displacement => Some(&DISPLACEMENT),
            Self::Environment | Self::Medium => None,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Displacement => "displacement",
            Self::Environment => "environment",
            Self::Medium => "medium",
        }
    }
}

/// One `Build::link` call.
#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub(crate) point: LinkPoint,
    pub(crate) name: String,
    pub(crate) module: PathBuf,
    pub(crate) ty: String,
}

impl Link {
    pub(crate) fn new(point: LinkPoint, name: &str, module: &Path, ty: &str) -> Self {
        Self {
            point,
            name: name.to_owned(),
            module: module.to_path_buf(),
            ty: ty.to_owned(),
        }
    }

    fn import(&self) -> String {
        self.module
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

/// A point a kernel declares, with the name of its `extern struct`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct Declared {
    pub(crate) point: LinkPoint,
    pub(crate) name: String,
}

/// One kernel file in `link_points.json`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub(crate) struct KernelPoints {
    /// The kernel's path under its root, without extension.
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) root: PathBuf,
    pub(crate) points: Vec<Declared>,
}

/// `link_points.json`: every kernel file that declares a link point.
#[derive(Default, Serialize, Deserialize)]
pub(crate) struct LinkPoints {
    pub(crate) kernels: Vec<KernelPoints>,
}

impl LinkPoints {
    pub(crate) fn read(path: &Path) -> Result<Self, Error> {
        serde_json::from_str(&super::compile::read(path)?).map_err(|e| Error::Link {
            message: format!("{} is not a link-points file: {e}", path.display()),
        })
    }

    /// Adds `other`'s kernels; a kernel file listed twice is kept once.
    pub(crate) fn merge(&mut self, other: Self) {
        self.kernels.extend(other.kernels);
        self.kernels.sort();
        self.kernels.dedup();
    }
}

/// The link points `src` declares.
pub(crate) fn declared(src: &str) -> Vec<Declared> {
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

/// The `link` calls with their module paths resolved against `CARGO_MANIFEST_DIR`, after
/// checking names, ids and counts.
pub(crate) fn resolve(links: &[Link]) -> Result<Vec<Link>, Error> {
    check(links)?;
    links
        .iter()
        .map(|l| {
            Ok(Link {
                module: resolve_module(&l.module)?,
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
pub(crate) fn ids(links: &[Link]) -> BTreeMap<LinkPoint, Vec<(&str, u8)>> {
    let mut out: BTreeMap<LinkPoint, Vec<(&str, u8)>> = BTreeMap::new();
    for point in [LinkPoint::Surface, LinkPoint::Displacement] {
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

/// The width of the coherence key (contracts.md 3): 3 plus the bit length of the highest
/// linked surface id.
pub(crate) fn key_bits(links: &[Link]) -> u32 {
    let surfaces = links
        .iter()
        .filter(|l| l.point == LinkPoint::Surface)
        .count() as u32;
    3 + (u32::BITS - surfaces.leading_zeros())
}

/// A generated link module: one per point and declared name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct LinkModule {
    /// The Slang module name.
    pub(crate) module: String,
    pub(crate) path: PathBuf,
    pub(crate) source: String,
}

/// Writes the link module for `decl` under `dir` and returns it; `None` when nothing links
/// to its point.
pub(crate) fn write_module(
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
    let _ = writeln!(
        out,
        "\nexport struct {} : {}\n{{",
        decl.name,
        point.interface()
    );
    for m in switch.methods {
        let modifier = if m.optional { "override " } else { "" };
        let _ = writeln!(
            out,
            "    {modifier}static {}\n    {{\n        switch ({})\n        {{",
            m.decl, switch.id
        );
        for (i, l) in linked.iter().enumerate() {
            let _ = writeln!(
                out,
                "        case {}: return {}.{}({});",
                i + 1,
                l.ty,
                m.name,
                m.args
            );
        }
        let _ = writeln!(
            out,
            "        default: return {}.{}({});\n        }}\n    }}",
            switch.default, m.name, m.args
        );
    }
    out.push_str("}\n");
    if point == LinkPoint::Surface {
        let _ = writeln!(out, "\nexport static const uint KEY_BITS = {key_bits};");
    }
    out
}

/// The directories of the linked modules, for the include path.
pub(crate) fn module_dirs(links: &[Link]) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = links
        .iter()
        .filter_map(|l| l.module.parent().map(Path::to_path_buf))
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
        let src = "// extern struct Old : ISurface = StandardSurface;\n\
                   import surface;\nextern struct Surface : ISurface = StandardSurface;\n\
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
