//! Hot reload: recompiles watched modules from source when their inputs change.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use std::{fs, ptr};

use super::compile::{self, Manifest, Settings, Unit};
use super::link::LinkModule;
use super::precompile::Precompiled;
use super::slang::{GlobalSession, Library};
use super::{Error, spirv, toolchain};
use crate::ShaderModule;

/// Watches the inputs of compiled-in modules and recompiles a module when one of them
/// changes, with the options its build manifest records. Owns no thread: [`scan`] compiles
/// on the calling thread.
///
/// [`scan`]: Self::scan
#[derive(Default)]
pub struct Watcher {
    manifests: HashMap<&'static str, Arc<Manifest>>,
    modules: Vec<Watched>,
    lib: Option<Arc<Library>>,
}

struct Watched {
    module: &'static ShaderModule,
    manifest: Arc<Manifest>,
    /// The module's entry in `manifest.modules`.
    index: usize,
    inputs: Vec<PathBuf>,
    /// The write time of each input when it was last compiled or first watched.
    stamps: Vec<Option<SystemTime>>,
    /// The hash of the SPIR-V the module's pipelines run.
    hash: u64,
}

/// One recompiled module.
pub enum Reload {
    /// New SPIR-V for `module`, whose entry points and interface are unchanged. Its debug
    /// info is embedded.
    Module {
        module: &'static ShaderModule,
        spirv: Arc<[u32]>,
        /// Empty when the module has no ray query.
        spirv_omm: Arc<[u32]>,
        hash: u64,
    },
    /// The module failed to compile, with Slang's file and line, or its entry points or
    /// interface changed, which needs a rebuild. Its pipelines keep the old SPIR-V.
    Rejected {
        module: &'static ShaderModule,
        reason: String,
    },
}

impl Reload {
    pub fn module(&self) -> &'static ShaderModule {
        match self {
            Self::Module { module, .. } | Self::Rejected { module, .. } => module,
        }
    }
}

impl fmt::Debug for Reload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module {
                module,
                spirv,
                hash,
                ..
            } => write!(
                f,
                "Module({}, {} words, {hash:#018x})",
                module.name,
                spirv.len()
            ),
            Self::Rejected { module, reason } => write!(f, "Rejected({}: {reason})", module.name),
        }
    }
}

impl Watcher {
    pub fn new() -> Self {
        Self::default()
    }

    /// Watches every input of `module` as its build manifest lists them. Idempotent.
    pub fn watch(&mut self, module: &'static ShaderModule) -> Result<(), Error> {
        if self.modules.iter().any(|w| ptr::eq(w.module, module)) {
            return Ok(());
        }
        let manifest = match self.manifests.get(module.manifest) {
            Some(m) => Arc::clone(m),
            None => {
                let m = Arc::new(read_manifest(Path::new(module.manifest))?);
                self.manifests.insert(module.manifest, Arc::clone(&m));
                m
            }
        };
        let index = find(&manifest, module).ok_or_else(|| Error::Manifest {
            path: module.manifest.into(),
            message: format!("lists no module {}", module.name),
        })?;
        let inputs = manifest.modules[index].inputs.clone();
        self.modules.push(Watched {
            module,
            stamps: inputs.iter().map(|p| stamp(p)).collect(),
            inputs,
            manifest,
            index,
            hash: module.hash,
        });
        Ok(())
    }

    /// Blocking. Reads the write time of every watched input, recompiles the modules whose
    /// inputs changed and returns one result per module that compiled to new SPIR-V or
    /// was rejected. Runs as a loading job, never on the render thread.
    pub fn scan(&mut self) -> Vec<Reload> {
        let mut seen: HashMap<PathBuf, Option<SystemTime>> = HashMap::new();
        let mut now = |p: &Path| *seen.entry(p.to_path_buf()).or_insert_with(|| stamp(p));
        let mut changed = Vec::new();
        for (i, w) in self.modules.iter_mut().enumerate() {
            let stamps: Vec<_> = w.inputs.iter().map(|p| now(p)).collect();
            if stamps != w.stamps {
                w.stamps = stamps;
                changed.push(i);
            }
        }
        if changed.is_empty() {
            return Vec::new();
        }
        let library = self.modules[changed[0]].manifest.library.clone();
        let global = self.session(&library);
        let mut out = Vec::new();
        for i in changed {
            let w = &mut self.modules[i];
            let result = match &global {
                Ok(global) => recompile(global, w, &mut now),
                Err(reason) => Err(reason.clone()),
            };
            match result {
                Ok(Some(reload)) => out.push(reload),
                Ok(None) => {}
                Err(reason) => out.push(Reload::Rejected {
                    module: w.module,
                    reason,
                }),
            }
        }
        out
    }

    /// A global session over the Slang library at `path`, loaded once.
    fn session(&mut self, path: &Path) -> Result<GlobalSession, String> {
        let lib = match &self.lib {
            Some(lib) => Arc::clone(lib),
            None => {
                let lib = toolchain::load_library(path).map_err(|e| e.to_string())?;
                self.lib = Some(Arc::clone(&lib));
                lib
            }
        };
        GlobalSession::new(lib)
    }
}

/// Compiles `w` again from source. `None` when the SPIR-V is what its pipelines run.
fn recompile(
    global: &GlobalSession,
    w: &mut Watched,
    now: &mut impl FnMut(&Path) -> Option<SystemTime>,
) -> Result<Option<Reload>, String> {
    let manifest = &*w.manifest;
    let entry = &manifest.modules[w.index];
    let mut options = if entry.linked {
        manifest.linked_options.clone()
    } else {
        manifest.options.clone()
    };
    options.retain(|o| o != "-separate-debug-info");
    let session = global.session(&options)?;
    let read = |path: &Path| compile::read(path).map_err(|e| e.to_string());
    let source = read(&entry.kernel)?;
    let links = entry
        .links
        .iter()
        .map(|l| {
            Ok(LinkModule {
                module: l.module.clone(),
                path: l.path.clone(),
                source: read(&l.path)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let precompiled = Precompiled::default();
    let settings = Settings {
        tool: "",
        options: &options,
        metal: None,
        asserts: manifest.asserts,
        separate_debug: false,
        precompiled: &precompiled,
    };
    let unit = Unit {
        name: &entry.name,
        path: &entry.kernel,
        source: &source,
        links: &links,
    };
    let compiled =
        compile::compile_kernel(&session, &settings, &unit).map_err(|e| e.to_string())?;
    w.stamps = compiled.inputs.iter().map(|p| now(p)).collect();
    w.inputs = compiled.inputs;

    let module = w.module;
    let entries = &compiled.model.entries;
    let same_entries = entries.len() == module.entry_points.len()
        && module.entry_points.iter().all(|e| {
            entries.iter().any(|n| {
                n.name.as_bytes() == e.name.to_bytes()
                    && n.stage == e.stage
                    && n.workgroup == e.workgroup
            })
        });
    if !same_entries {
        return Err(format!(
            "{}: rebuild required: its entry points, stages or workgroup sizes changed",
            module.name
        ));
    }
    let interface = compile::interface_hash(&compiled.model);
    if let Some(e) = module
        .entry_points
        .iter()
        .find(|e| e.interface != interface)
    {
        return Err(format!(
            "{}: rebuild required: the root layout or specialization constants of {:?} changed",
            module.name, e.name
        ));
    }
    let hash = compile::hash64(&compile::words_to_bytes(&compiled.words));
    if hash == w.hash {
        return Ok(None);
    }
    w.hash = hash;
    let omm = spirv::omm_variant(&compiled.words).unwrap_or_default();
    Ok(Some(Reload::Module {
        module,
        spirv: compiled.words.into(),
        spirv_omm: omm.into(),
        hash,
    }))
}

fn read_manifest(path: &Path) -> Result<Manifest, Error> {
    let bytes = fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Manifest {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// The manifest entry of `module`: by name, and by hash where a build lists a kernel
/// both as its own and linked.
fn find(manifest: &Manifest, module: &ShaderModule) -> Option<usize> {
    let named: Vec<usize> = (0..manifest.modules.len())
        .filter(|&i| manifest.modules[i].name == module.name)
        .collect();
    match named[..] {
        [i] => Some(i),
        _ => named
            .into_iter()
            .find(|&i| manifest.modules[i].hash == module.hash),
    }
}

fn stamp(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}
