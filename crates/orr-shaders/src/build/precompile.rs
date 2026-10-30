//! Library modules compiled once per build into `.slang-module` files.
//!
//! Slang prefers a `.slang-module` on the search path over a source beside the importer and
//! never checks it against the source, so the directory holds only modules whose sources
//! match their records; the rest are deleted and load from source. A kernel's dependency
//! list still names the sources of the precompiled modules it loads.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::compile::{canonical_lossy, write};
use super::slang::{GlobalSession, Library, Session};
use super::{Error, SLANG_VERSION};

/// The precompiled modules of one option set.
#[derive(Default)]
pub(crate) struct Precompiled {
    /// Each module file (canonical) with its record.
    modules: BTreeMap<PathBuf, ModuleRecord>,
}

impl Precompiled {
    /// The import names of the precompiled modules loaded into `session`.
    pub(crate) fn used(&self, session: &Session) -> Vec<String> {
        let mut used: Vec<String> = session
            .loaded_modules()
            .iter()
            .filter_map(|m| self.modules.get(&canonical_lossy(&m.file()?)))
            .map(|e| e.name.clone())
            .collect();
        used.sort();
        used
    }

    /// The source file of a precompiled module file, which keeps no declaration locations.
    pub(crate) fn source(&self, module: &Path) -> Option<&Path> {
        self.modules
            .get(&canonical_lossy(module))
            .map(|e| e.file.as_path())
    }
}

#[derive(Serialize, Deserialize)]
struct ModuleRecord {
    key: String,
    /// The import name.
    name: String,
    /// The module's source file.
    file: PathBuf,
    /// Its dependency files.
    inputs: Vec<PathBuf>,
}

/// Precompiles every library module that `sources` import, directly or through other
/// libraries, into `dir`, reusing the modules whose records are current.
///
/// `sources` are the files the build compiles (path and text); `include_dirs` the session's
/// search paths, which `dir` must not be among; `key_parts` everything besides the inputs
/// that changes a module. A module is precompiled only where every importer resolves its
/// name to the file the search paths give, since the precompiled copy wins over a file
/// beside the importer. A module that fails to compile is left to load from source, so the
/// kernel's compile reports the error.
pub(crate) fn precompile(
    lib: &Arc<Library>,
    options: &[String],
    key_parts: &[&str],
    include_dirs: &[PathBuf],
    sources: &[(PathBuf, String)],
    dir: &Path,
) -> Result<Precompiled, Error> {
    let candidates = libraries(include_dirs, sources);
    fs::create_dir_all(dir).map_err(|source| Error::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    let dir = &canonical_lossy(dir);
    let mut modules = BTreeMap::new();
    let mut stale = Vec::new();
    for (name, file) in &candidates {
        let out = module_path(dir, name);
        let record = out.with_extension("json");
        let current = fs::read(&record)
            .ok()
            .and_then(|b| serde_json::from_slice::<ModuleRecord>(&b).ok())
            .filter(|r| {
                &r.name == name
                    && &r.file == file
                    && out.is_file()
                    && module_key(key_parts, name, &r.inputs).is_some_and(|k| k == r.key)
            });
        match current {
            Some(r) => {
                modules.insert(out, r);
            }
            None => stale.push((name, file, out, record)),
        }
    }
    if !stale.is_empty() {
        let slang_err = |message| Error::Slang {
            module: "precompiled modules".to_owned(),
            message,
        };
        let global = GlobalSession::new(Arc::clone(lib)).map_err(slang_err)?;
        let session = global.session(options).map_err(slang_err)?;
        // Each module loads through an `import` in a module of its own, which resolves the
        // name the way a kernel's import does. The importers sit in `dir`'s parent, which
        // holds no Slang files.
        let importers = dir.parent().unwrap_or(dir);
        for (i, (name, file, out, record)) in stale.into_iter().enumerate() {
            let _ = fs::remove_file(&out);
            let importer = format!("orr_precompile_{i}");
            let module = session
                .load(
                    &importer,
                    &importers.join(format!("{importer}.slang")),
                    &format!("import {name};\n"),
                )
                .ok()
                .and_then(|_| {
                    session
                        .loaded_modules()
                        .into_iter()
                        .find(|m| m.file().is_some_and(|f| canonical_lossy(&f) == *file))
                });
            let Some(module) = module else {
                let _ = fs::remove_file(&record);
                continue;
            };
            let mut inputs: Vec<PathBuf> = module
                .dependencies()
                .iter()
                .map(|p| canonical_lossy(p))
                .collect();
            inputs.sort();
            inputs.dedup();
            let key = module_key(key_parts, name, &inputs);
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent).map_err(|source| Error::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
            let (Some(key), Ok(())) = (key, module.write(&out)) else {
                let _ = fs::remove_file(&out);
                let _ = fs::remove_file(&record);
                continue;
            };
            let r = ModuleRecord {
                key,
                name: name.clone(),
                file: file.clone(),
                inputs,
            };
            write(
                &record,
                &serde_json::to_vec(&r).expect("module records serialise"),
            )?;
            modules.insert(out, r);
        }
    }
    remove_others(dir, &modules);
    Ok(Precompiled { modules })
}

fn module_path(dir: &Path, name: &str) -> PathBuf {
    let mut p = dir.to_path_buf();
    p.extend(name.split('.'));
    p.set_extension("slang-module");
    p
}

fn module_key(key_parts: &[&str], name: &str, inputs: &[PathBuf]) -> Option<String> {
    let mut h = blake3::Hasher::new();
    h.update(SLANG_VERSION.as_bytes());
    for part in key_parts {
        h.update(part.as_bytes());
        h.update(&[0]);
    }
    h.update(name.as_bytes());
    h.update(&[1]);
    for input in inputs {
        h.update(input.to_string_lossy().as_bytes());
        h.update(&[0]);
        h.update(&fs::read(input).ok()?);
    }
    Some(h.finalize().to_hex().to_string())
}

/// Deletes every module file and record in `dir` that is not one of `keep`.
fn remove_others(dir: &Path, keep: &BTreeMap<PathBuf, ModuleRecord>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for path in entries.filter_map(Result::ok).map(|e| e.path()) {
            if path.is_dir() {
                stack.push(path);
            } else if !keep.contains_key(&path.with_extension("slang-module")) {
                let _ = fs::remove_file(&path);
            }
        }
    }
}

/// The library modules reachable from `sources` by dotted `import` names, by name, with
/// the file the search paths give for each.
fn libraries(include_dirs: &[PathBuf], sources: &[(PathBuf, String)]) -> Vec<(String, PathBuf)> {
    let search = |name: &str| resolve(None, name, include_dirs);
    // name -> (searched file, imported names); `None` when an importer sees another file.
    let mut found: BTreeMap<String, Option<(PathBuf, Vec<String>)>> = BTreeMap::new();
    let mut queue: VecDeque<(PathBuf, String)> = sources.iter().cloned().collect();
    let mut read = BTreeSet::new();
    while let Some((importer, text)) = queue.pop_front() {
        let here = importer.parent();
        for name in imports(&text) {
            let searched = search(name);
            let seen = resolve(here, name, include_dirs);
            let usable = searched.is_some() && seen == searched;
            match found.get_mut(name) {
                Some(entry) if !usable => *entry = None,
                Some(_) => {}
                None => {
                    let entry = searched.filter(|_| usable).and_then(|file| {
                        let text = fs::read_to_string(&file).ok()?;
                        if text.contains("[shader(") {
                            return None;
                        }
                        let names = imports(&text).into_iter().map(str::to_owned).collect();
                        if read.insert(file.clone()) {
                            queue.push_back((file.clone(), text));
                        }
                        Some((file, names))
                    });
                    found.insert(name.to_owned(), entry);
                }
            }
        }
    }
    // A module whose imports load from source is left to source as well.
    loop {
        let excluded: BTreeSet<String> = found
            .iter()
            .filter(|(_, e)| e.is_none())
            .map(|(n, _)| n.clone())
            .collect();
        let mut changed = false;
        for entry in found.values_mut() {
            if entry
                .as_ref()
                .is_some_and(|(_, names)| names.iter().any(|n| excluded.contains(n)))
            {
                *entry = None;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    found
        .into_iter()
        .filter_map(|(name, e)| e.map(|(file, _)| (name, file)))
        .collect()
}

/// The file an import of `name` loads: beside the importer, else on the first search path
/// that has it.
fn resolve(importer_dir: Option<&Path>, name: &str, include_dirs: &[PathBuf]) -> Option<PathBuf> {
    let mut rel = PathBuf::from_iter(name.split('.'));
    rel.set_extension("slang");
    importer_dir
        .into_iter()
        .chain(include_dirs.iter().map(PathBuf::as_path))
        .map(|d| d.join(&rel))
        .find(|p| p.is_file())
        .map(|p| canonical_lossy(&p))
}

/// The dotted module names of a source's `import` declarations.
fn imports(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|line| {
            let line = line.split("//").next()?.trim();
            let line = line
                .strip_prefix("__exported ")
                .or_else(|| line.strip_prefix("public "))
                .unwrap_or(line)
                .trim_start();
            let name = line
                .strip_prefix("import ")?
                .trim()
                .strip_suffix(';')?
                .trim();
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
        let text = "import orr.abi;\n  __exported import orr.openpbr_bsdf;\npublic import lights; // x\n\
            // import gone;\nimport \"orr/path\";\nimport a.b; import c;\nimplementing x;\n";
        assert_eq!(imports(text), ["orr.abi", "orr.openpbr_bsdf", "lights"]);
    }
}
