//! Import roots: where a dotted Slang module name is found.
//!
//! Slang only maps a name to a path under a search path, so a root with a prefix stays off the
//! search paths: its modules that a compile reaches load by name, before the code that imports
//! them, from their real files. Diagnostics, debug info and dependency lists keep those paths.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::{fs, io};

use serde::{Deserialize, Serialize};

use super::Error;
use super::compile::{canonical, canonical_lossy};
use super::slang::Session;

/// A directory of Slang modules. With a prefix, `<prefix>.<path>` names `<dir>/<path>.slang`;
/// without, `<path>` does.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub(crate) struct Root {
    pub(crate) prefix: Option<String>,
    pub(crate) dir: PathBuf,
}

impl Root {
    pub(crate) fn plain(dir: impl Into<PathBuf>) -> Self {
        Self {
            prefix: None,
            dir: dir.into(),
        }
    }

    pub(crate) fn prefixed(prefix: &str, dir: impl Into<PathBuf>) -> Self {
        Self {
            prefix: Some(prefix.to_owned()),
            dir: dir.into(),
        }
    }

    /// This root with its directory canonical.
    pub(crate) fn canonical(&self) -> Result<Self, Error> {
        Ok(Self {
            prefix: self.prefix.clone(),
            dir: canonical(&self.dir)?,
        })
    }

    /// An entry of a `DEP_*_SHADER_DIRS` value: `<prefix>=<dir>` or `<dir>`. An `=` after
    /// anything but a module name is part of a directory.
    fn parse(entry: &str) -> Self {
        let is_name = |s: &str| {
            !s.is_empty()
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
        };
        match entry.split_once('=') {
            Some((prefix, dir)) if is_name(prefix) => Self::prefixed(prefix, dir),
            _ => Self::plain(entry),
        }
    }

    /// This root as a `DEP_*_SHADER_DIRS` entry.
    pub(crate) fn entry(&self) -> String {
        match &self.prefix {
            Some(prefix) => format!("{prefix}={}", self.dir.display()),
            None => self.dir.display().to_string(),
        }
    }

    /// The file this root gives `name`, if it exists.
    fn file(&self, name: &str) -> Option<PathBuf> {
        let rest = match &self.prefix {
            Some(prefix) => name.strip_prefix(prefix.as_str())?.strip_prefix('.')?,
            None => name,
        };
        let mut path = self.dir.join(PathBuf::from_iter(rest.split('.')));
        path.set_extension("slang");
        path.is_file().then_some(path)
    }

    /// The module name of the file at `path` under this root, when it has a prefix.
    pub(crate) fn module_name(&self, path: &Path) -> Option<String> {
        let prefix = self.prefix.as_deref()?;
        let rest = path.strip_prefix(&self.dir).ok()?.with_extension("");
        let parts: Vec<_> = rest
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect();
        Some(format!("{prefix}.{}", parts.join(".")))
    }
}

/// The directory of orr_shaders' Slang modules, imported as `orr_shaders.<name>`.
pub(super) const LIB_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");

/// The roots of every build: orr_color's and orr_shaders' modules, and orr_shaders' headers, whose
/// directory holds nothing else so that the search paths give no module a name without its prefix.
pub(crate) fn library() -> Result<[Root; 3], Error> {
    Ok([
        Root::prefixed("orr_color", orr_color::SHADER_DIR).canonical()?,
        Root::prefixed("orr_shaders", LIB_DIR).canonical()?,
        Root::plain(Path::new(LIB_DIR).join("include")).canonical()?,
    ])
}

/// The roots of a `DEP_*_SHADER_DIRS` value (`;`-separated).
pub(crate) fn parse_list(list: &str) -> impl Iterator<Item = Root> + '_ {
    list.split(';').filter(|s| !s.is_empty()).map(Root::parse)
}

/// `roots` as a `DEP_*_SHADER_DIRS` value.
pub(crate) fn format_list(roots: &[Root]) -> String {
    roots.iter().map(Root::entry).collect::<Vec<_>>().join(";")
}

/// The file an import of `name` loads when nothing beside the importer has it.
pub(crate) fn find(roots: &[Root], name: &str) -> Option<PathBuf> {
    roots
        .iter()
        .find_map(|r| r.file(name))
        .map(|p| canonical_lossy(&p))
}

/// The file of `name` in the `DEP_*_SHADER_DIRS` value `list`. For a build script that links a
/// dependency's module (`module_file(&roots, "dep.sky")`).
pub fn module_file(list: &str, name: &str) -> Option<PathBuf> {
    find(&parse_list(list).collect::<Vec<_>>(), name)
}

/// The modules under prefixed roots that `sources` (path and text) reach through imports, each
/// after those it imports, with its text, for [`load_prefixed`].
fn prefixed_modules(
    roots: &[Root],
    sources: &[(&Path, &str)],
) -> Vec<(String, PathBuf, io::Result<String>)> {
    fn visit(
        roots: &[Root],
        dir: Option<&Path>,
        text: &str,
        seen: &mut BTreeSet<String>,
        out: &mut Vec<(String, PathBuf, io::Result<String>)>,
    ) {
        for name in imports(text) {
            if !seen.insert(name.to_owned()) {
                continue;
            }
            let found = roots
                .iter()
                .find_map(|r| Some((r.prefix.is_some(), r.file(name)?)))
                .or_else(|| Some((false, beside(dir, name)?)));
            let Some((prefixed, file)) = found else {
                continue;
            };
            let read = fs::read_to_string(&file);
            if let Ok(text) = &read {
                visit(roots, file.parent(), text, seen, out);
            }
            if prefixed {
                out.push((name.to_owned(), canonical_lossy(&file), read));
            }
        }
    }
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for (path, text) in sources {
        visit(roots, path.parent(), text, &mut seen, &mut out);
    }
    out
}

/// Loads the modules under prefixed roots that `sources` (path and text) reach into `session`,
/// those `skip` accepts aside. Errors carry Slang's diagnostics.
pub(crate) fn load_prefixed(
    session: &Session,
    roots: &[Root],
    sources: &[(&Path, &str)],
    skip: impl Fn(&str) -> bool,
) -> Result<(), String> {
    for (name, file, text) in prefixed_modules(roots, sources) {
        if skip(&name) {
            continue;
        }
        let text = text.map_err(|e| format!("{}: {e}", file.display()))?;
        // The name an import looks a module up by has `/` where the source has `.`.
        session.load(&name.replace('.', "/"), &file, &text)?;
    }
    Ok(())
}

/// The file an import of `name` loads from the importer's directory, if any.
pub(crate) fn beside(importer_dir: Option<&Path>, name: &str) -> Option<PathBuf> {
    let mut rel = PathBuf::from_iter(name.split('.'));
    rel.set_extension("slang");
    let path = importer_dir?.join(rel);
    path.is_file().then(|| canonical_lossy(&path))
}

/// The dotted module names of a source's `import` declarations.
pub(crate) fn imports(text: &str) -> Vec<&str> {
    text.lines()
        .flat_map(|line| {
            let code = line.split("//").next().unwrap_or_default();
            // What follows the last `;` is no whole statement.
            let mut statements: Vec<&str> = code.split(';').collect();
            statements.pop();
            statements
        })
        .filter_map(|statement| {
            let statement = statement.trim();
            let statement = statement
                .strip_prefix("__exported ")
                .or_else(|| statement.strip_prefix("public "))
                .unwrap_or(statement)
                .trim_start();
            let name = statement.strip_prefix("import ")?.trim();
            name.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
                .then_some(name)
                .filter(|n| !n.is_empty())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_declarations() {
        let text = "import orr_shaders.abi;\n  __exported import orr_lighting.openpbr_bsdf;\npublic import lights; // x\n\
            // import gone;\nimport \"orr/path\";\nimport a.b; import c;\nimplementing x;\n";
        assert_eq!(
            imports(text),
            [
                "orr_shaders.abi",
                "orr_lighting.openpbr_bsdf",
                "lights",
                "a.b",
                "c"
            ]
        );
    }

    #[test]
    fn root_entries_round_trip() {
        let entries = "orr_scene=C:\\a=b\\src;C:\\b=c\\shaders;/c/d=e/shaders";
        let roots: Vec<_> = parse_list(entries).collect();
        assert_eq!(roots[0].prefix.as_deref(), Some("orr_scene"));
        assert_eq!(roots[0].dir, PathBuf::from("C:\\a=b\\src"));
        assert_eq!(roots[1].prefix, None);
        assert_eq!(roots[2].prefix, None);
        assert_eq!(format_list(&roots), entries);
    }
}
