use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{env, fs, thread};

use serde::{Deserialize, Serialize};

use super::codegen::{self, Kernel, Owner, Shared};
use super::layout::{Plan, Planner};
use super::lint::{self, Lint, LintError};
use super::model::{self, Model, StructDef, Ty};
use super::slang::{GlobalSession, Library, Session};
use super::{Error, SLANG_VERSION, SLANGC_OPTIONS, metal, spirv};

/// Compiles a crate's Slang kernels and generates Rust for them. Used from `build.rs`.
///
/// ```no_run
/// # fn main() -> Result<(), orr_shaders::build::Error> {
/// let src = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
/// let out = orr_shaders::build::Build::new()?.kernels(&src).compile()?;
/// println!("cargo:shader_dirs={}", out.shader_dirs);
/// # Ok(())
/// # }
/// ```
///
/// and in the crate: `mod shaders { include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs")); }`.
pub struct Build {
    lib: Arc<Library>,
    kernels: Option<PathBuf>,
    includes: Vec<PathBuf>,
    externs: BTreeMap<String, String>,
    debug_info: bool,
    fp_precise: bool,
    asserts: bool,
    metal_check: bool,
    allowed: BTreeSet<Lint>,
    out_dir: Option<PathBuf>,
}

/// What [`Build::compile`] produced.
#[derive(Debug)]
pub struct Output {
    /// The crate's shader root followed by every root it was given, `;`-separated, for
    /// `cargo:shader_dirs`.
    pub shader_dirs: String,
    /// The modules compiled by this run; the others were up to date.
    pub compiled: Vec<String>,
    /// The build manifest: every module with its inputs, options and outputs.
    pub manifest: PathBuf,
}

impl Build {
    /// Locates and version-checks the Slang release.
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            lib: super::toolchain::load()?,
            kernels: None,
            includes: Vec::new(),
            externs: BTreeMap::new(),
            debug_info: env::var("PROFILE").map_or(true, |p| p != "release"),
            fp_precise: false,
            asserts: false,
            metal_check: env::var_os("ORR_METAL_CHECK").is_some_and(|v| v == "1"),
            allowed: BTreeSet::new(),
            out_dir: None,
        })
    }

    /// The crate's shader root. Every `.slang` file under it that contains `[shader(`
    /// becomes one module.
    pub fn kernels(mut self, root: impl AsRef<Path>) -> Self {
        self.kernels = Some(root.as_ref().to_path_buf());
        self
    }

    /// One more import root. [`LIB_DIR`](crate::LIB_DIR) is always present.
    pub fn include(mut self, dir: impl AsRef<Path>) -> Self {
        self.includes.push(dir.as_ref().to_path_buf());
        self
    }

    /// Every root in a `DEP_*_SHADER_DIRS` value (`;`-separated).
    pub fn roots(mut self, list: &str) -> Self {
        self.includes
            .extend(list.split(';').filter(|s| !s.is_empty()).map(PathBuf::from));
        self
    }

    /// Uses an existing `#[repr(C)]` Rust type for a Slang struct instead of generating
    /// one; the generated code asserts its size and field offsets.
    pub fn extern_type(mut self, slang_name: &str, rust_path: &str) -> Self {
        self.externs
            .insert(slang_name.to_owned(), rust_path.to_owned());
        self
    }

    /// Debug info in the SPIR-V (`-g2`). On by default unless the profile is `release`.
    pub fn debug_info(mut self, on: bool) -> Self {
        self.debug_info = on;
        self
    }

    /// `-fp-mode precise`, for entry-parity test kernels.
    pub fn fp_precise(mut self, on: bool) -> Self {
        self.fp_precise = on;
        self
    }

    /// Links `orr.debug`'s `ASSERTS = true`, so checks such as `Tlas.get()` on address 0
    /// abort. Off by default; the device then needs `shaderAbort`.
    pub fn asserts(mut self, on: bool) -> Self {
        self.asserts = on;
        self
    }

    /// Also compiles each module's compute and mesh-pipeline entries to MSL and checks them
    /// ([`Lint::Metal`]). Defaults to on when `ORR_METAL_CHECK=1`. The `.metal` files are
    /// written beside the SPIR-V.
    pub fn metal_check(mut self, on: bool) -> Self {
        self.metal_check = on;
        self
    }

    /// Turns a lint off for this build.
    pub fn allow(mut self, lint: Lint) -> Self {
        self.allowed.insert(lint);
        self
    }

    /// Writes outputs here instead of `OUT_DIR`.
    pub fn out_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.out_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Compiles changed modules in parallel, checks layouts and lints, and writes
    /// `orr_shaders.rs`, the `.spv` files with their `.omm.spv` variants and
    /// `shaders.manifest.json` to the output directory. Prints `rerun-if-changed` lines.
    pub fn compile(self) -> Result<Output, Error> {
        let started = Instant::now();
        let out_dir = match &self.out_dir {
            Some(d) => d.clone(),
            None => env::var_os("OUT_DIR")
                .map(PathBuf::from)
                .ok_or(Error::NoOutDir)?,
        };
        let lib_dir = canonical_lossy(Path::new(crate::LIB_DIR));
        let own_root = self.kernels.as_deref().map(canonical).transpose()?;
        let mut roots: Vec<PathBuf> = own_root.iter().cloned().collect();
        for dir in &self.includes {
            let dir = canonical(dir)?;
            if !roots.contains(&dir) && dir != lib_dir {
                roots.push(dir);
            }
        }
        let shader_dirs = roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(";");
        if lib_dir.is_dir() {
            roots.push(lib_dir.clone());
        }

        let mut options: Vec<String> = SLANGC_OPTIONS.iter().map(|s| (*s).to_owned()).collect();
        for root in &roots {
            options.push("-I".into());
            options.push(root.display().to_string());
        }
        if self.debug_info {
            options.push("-g2".into());
        }
        if self.fp_precise {
            options.extend(["-fp-mode".into(), "precise".into()]);
        }

        let kernels = match &own_root {
            Some(root) => find_kernels(root)?,
            None => Vec::new(),
        };
        let metal_options = self.metal_check.then(|| {
            let mut o: Vec<String> = metal::OPTIONS.iter().map(|s| (*s).to_owned()).collect();
            for root in &roots {
                o.extend(["-I".to_owned(), root.display().to_string()]);
            }
            o
        });
        let settings = Settings {
            tool: tool_hash(),
            options: &options,
            metal: metal_options.as_deref(),
            asserts: self.asserts,
        };
        let records = out_dir.join("orr-shaders");
        let mut jobs = Vec::new();
        let mut modules = BTreeMap::new();
        for (name, path) in kernels {
            let record_path = records.join(format!("{name}.json"));
            match load_record(&record_path, &settings, &path) {
                Some(record) => {
                    modules.insert(name, record);
                }
                None => jobs.push((name, path, record_path)),
            }
        }
        let compiled: Vec<String> = jobs.iter().map(|(n, ..)| n.clone()).collect();
        let fresh = compile_all(&self.lib, &settings, jobs, &out_dir)?;
        modules.extend(fresh);

        let mut lints = lint::duplicate_imports(&roots);
        for (name, record) in &modules {
            lints.extend(record.metal.iter().map(|message| LintError {
                lint: Lint::Metal,
                module: name.clone(),
                message: message.clone(),
            }));
        }
        let mut seen_files = BTreeSet::new();
        for record in modules.values() {
            for input in &record.inputs {
                if seen_files.insert(input.clone())
                    && let Ok(src) = fs::read_to_string(input)
                {
                    lints.extend(lint::source_lints(input, &src, &lib_dir));
                }
            }
        }

        // Owners, then sizes merged across modules for every shared struct.
        let mut owners: HashMap<String, HashMap<String, Owner>> = HashMap::new();
        let mut shared_defs: BTreeMap<String, (StructDef, String)> = BTreeMap::new();
        for (name, record) in &modules {
            let mut module_owners = HashMap::new();
            let kernel_file = canonical_lossy(&record.kernel);
            for (s, def) in &record.model.structs {
                let owner = if let Some(path) = self.externs.get(s) {
                    Owner::Extern(path.clone())
                } else {
                    match def.file.as_deref().map(canonical_lossy) {
                        Some(file) if file == kernel_file => Owner::Kernel,
                        Some(file) if own_root.as_ref().is_some_and(|r| file.starts_with(r)) => {
                            Owner::Types
                        }
                        None => Owner::Types,
                        Some(file) => {
                            let root = roots.iter().find(|r| file.starts_with(r));
                            lints.push(LintError {
                                lint: Lint::Ownership,
                                module: name.clone(),
                                message: format!(
                                    "struct {s} is declared in {} under {}; map it with .extern_type({s:?}, \"<crate>::<path>::{}\")",
                                    file.display(),
                                    root.map_or("no shader root".to_owned(), |r| format!("shader root {}", r.display())),
                                    codegen::rust_name(s)
                                ),
                            });
                            Owner::Types
                        }
                    }
                };
                if owner != Owner::Kernel {
                    merge_shared(&mut shared_defs, s, def, name)?;
                }
                module_owners.insert(s.clone(), owner);
            }
            owners.insert(name.clone(), module_owners);
        }

        let mut plans: HashMap<String, HashMap<String, Plan>> = HashMap::new();
        let mut models: HashMap<String, Model> = HashMap::new();
        for (name, record) in &modules {
            let mut model = record.model.clone();
            for (s, def) in model.structs.iter_mut() {
                if let Some((shared, _)) = shared_defs.get(s) {
                    def.size = shared.size;
                }
            }
            let module_plans =
                Planner::new(&model.structs)
                    .plan_all()
                    .map_err(|m| Error::Layout {
                        module: name.clone(),
                        structure: m.structure,
                        field: m.field,
                        message: m.message,
                    })?;
            lints.extend(lint::model_lints(name, &model, &module_plans, &lib_dir));
            plans.insert(name.clone(), module_plans);
            models.insert(name.clone(), model);
        }

        lints.retain(|l| !self.allowed.contains(&l.lint));
        if !lints.is_empty() {
            return Err(Error::Lints(lints));
        }

        let shared_plans = Planner::new(
            &shared_defs
                .iter()
                .map(|(k, (d, _))| (k.clone(), d.clone()))
                .collect(),
        )
        .plan_all()
        .map_err(|m| Error::Layout {
            module: shared_defs
                .get(&m.structure)
                .map_or_else(String::new, |(_, module)| module.clone()),
            structure: m.structure,
            field: m.field,
            message: m.message,
        })?;
        let mut shared = Shared {
            types: BTreeMap::new(),
            externs: BTreeMap::new(),
        };
        for (s, (def, _)) in &shared_defs {
            let plan = &shared_plans[s];
            match self.externs.get(s) {
                Some(path) => {
                    shared.externs.insert(s.clone(), (def, plan, path.clone()));
                }
                None => {
                    shared.types.insert(s.clone(), (def, plan));
                }
            }
        }

        let manifest_path = out_dir.join("shaders.manifest.json");
        let mut kernels_out = Vec::new();
        for (name, record) in &modules {
            let model = &models[name];
            kernels_out.push(Kernel {
                name,
                spv: &record.spv,
                omm: record.omm.as_deref(),
                hash: record.hash,
                interface: interface_hash(model),
                model,
                owners: &owners[name],
                plans: &plans[name],
            });
        }
        let rust = codegen::generate(&kernels_out, &shared, &manifest_path);
        write(&out_dir.join("orr_shaders.rs"), rust.as_bytes())?;

        let manifest = Manifest {
            slang: SLANG_VERSION,
            options: &options,
            asserts: self.asserts,
            modules: modules
                .iter()
                .map(|(name, r)| ManifestModule {
                    name,
                    kernel: &r.kernel,
                    spv: &r.spv,
                    omm: r.omm.as_deref(),
                    hash: r.hash,
                    inputs: &r.inputs,
                })
                .collect(),
        };
        write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest)
                .expect("the manifest serialises")
                .as_bytes(),
        )?;

        print_cargo(&roots, &modules);
        if env::var_os("ORR_SHADER_TIMINGS").is_some() {
            #[allow(clippy::print_stdout)]
            {
                println!(
                    "cargo:warning=orr-shaders: {} of {} modules compiled in {:.2} s",
                    compiled.len(),
                    modules.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
        Ok(Output {
            shader_dirs,
            compiled,
            manifest: manifest_path,
        })
    }
}

#[derive(Serialize)]
struct Manifest<'a> {
    slang: &'a str,
    options: &'a [String],
    asserts: bool,
    modules: Vec<ManifestModule<'a>>,
}

#[derive(Serialize)]
struct ManifestModule<'a> {
    name: &'a str,
    kernel: &'a Path,
    spv: &'a Path,
    omm: Option<&'a Path>,
    hash: u64,
    inputs: &'a [PathBuf],
}

/// One compiled module as the build remembers it between runs.
#[derive(Clone, Serialize, Deserialize)]
struct Record {
    key: String,
    kernel: PathBuf,
    inputs: Vec<PathBuf>,
    spv: PathBuf,
    omm: Option<PathBuf>,
    hash: u64,
    model: Model,
    /// MSL check failures; empty when the check passed or did not run.
    metal: Vec<String>,
}

/// What every module of one build is compiled with.
struct Settings<'a> {
    /// A hash of the running build script, so a changed build invalidates every record.
    tool: String,
    options: &'a [String],
    /// The Metal session's options when the MSL check runs.
    metal: Option<&'a [String]>,
    asserts: bool,
}

impl Settings<'_> {
    /// Everything besides the inputs that changes a module's outputs.
    fn key_parts(&self) -> Vec<&str> {
        let mut parts = vec![self.tool.as_str()];
        parts.extend(self.options.iter().map(String::as_str));
        if self.asserts {
            parts.push("+asserts");
        }
        if let Some(metal) = self.metal {
            parts.push("+metal");
            parts.extend(metal.iter().map(String::as_str));
        }
        parts
    }
}

fn print_cargo(roots: &[PathBuf], modules: &BTreeMap<String, Record>) {
    let mut lines = BTreeSet::new();
    for r in roots {
        lines.insert(r.clone());
    }
    for m in modules.values() {
        lines.extend(m.inputs.iter().cloned());
    }
    #[allow(clippy::print_stdout)]
    {
        for l in lines {
            println!("cargo:rerun-if-changed={}", l.display());
        }
        for var in ["ORR_SLANGC", "ORR_METAL_CHECK", "ORR_SHADER_TIMINGS"] {
            println!("cargo:rerun-if-env-changed={var}");
        }
    }
}

fn merge_shared(
    shared: &mut BTreeMap<String, (StructDef, String)>,
    name: &str,
    def: &StructDef,
    module: &str,
) -> Result<(), Error> {
    match shared.get_mut(name) {
        None => {
            shared.insert(name.to_owned(), (def.clone(), module.to_owned()));
        }
        Some((existing, first)) => {
            let same_fields = existing.fields == def.fields;
            let sizes_agree = match (existing.size, def.size) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            };
            if !same_fields || !sizes_agree {
                return Err(Error::StructConflict {
                    name: name.to_owned(),
                    first: first.clone(),
                    second: module.to_owned(),
                });
            }
            existing.size = existing.size.or(def.size);
        }
    }
    Ok(())
}

/// Every `.slang` file under `root` that declares an entry point, by its path under the
/// root without extension.
fn find_kernels(root: &Path) -> Result<Vec<(String, PathBuf)>, Error> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|source| Error::Io {
            path: dir.clone(),
            source,
        })?;
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "slang") {
                let src = fs::read_to_string(&path).map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?;
                if src.contains("[shader(") {
                    let rel = path
                        .strip_prefix(root)
                        .expect("walked from root")
                        .with_extension("");
                    let name = rel
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy())
                        .collect::<Vec<_>>()
                        .join("/");
                    out.push((name, path));
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

fn tool_hash() -> String {
    env::current_exe()
        .and_then(fs::read)
        .map(|bytes| blake3::hash(&bytes).to_hex().to_string())
        .unwrap_or_default()
}

fn cache_key(settings: &Settings<'_>, kernel: &Path, inputs: &[PathBuf]) -> Option<String> {
    let mut h = blake3::Hasher::new();
    h.update(SLANG_VERSION.as_bytes());
    for o in settings.key_parts() {
        h.update(o.as_bytes());
        h.update(&[0]);
    }
    h.update(kernel.to_string_lossy().as_bytes());
    for input in inputs {
        h.update(input.to_string_lossy().as_bytes());
        h.update(&[0]);
        h.update(&fs::read(input).ok()?);
    }
    Some(h.finalize().to_hex().to_string())
}

fn load_record(path: &Path, settings: &Settings<'_>, kernel: &Path) -> Option<Record> {
    let record: Record = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    let key = cache_key(settings, kernel, &record.inputs)?;
    let outputs_exist = record.spv.is_file() && record.omm.as_ref().is_none_or(|p| p.is_file());
    (record.key == key && record.kernel == kernel && outputs_exist).then_some(record)
}

fn compile_all(
    lib: &Arc<Library>,
    settings: &Settings<'_>,
    jobs: Vec<(String, PathBuf, PathBuf)>,
    out_dir: &Path,
) -> Result<BTreeMap<String, Record>, Error> {
    if jobs.is_empty() {
        return Ok(BTreeMap::new());
    }
    let threads = thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(jobs.len());
    let queue = Mutex::new(jobs);
    let results = Mutex::new(Vec::new());
    thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut global: Option<GlobalSession> = None;
                loop {
                    let Some((name, path, record_path)) = queue.lock().expect("queue").pop() else {
                        break;
                    };
                    let result = compile_one(lib, &mut global, settings, &name, &path, out_dir)
                        .and_then(|record| {
                            write(
                                &record_path,
                                &serde_json::to_vec(&record).expect("records serialise"),
                            )?;
                            Ok(record)
                        });
                    results.lock().expect("results").push((name, result));
                }
            });
        }
    });
    let mut out = BTreeMap::new();
    let mut results = results.into_inner().expect("results");
    results.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, result) in results {
        out.insert(name, result?);
    }
    Ok(out)
}

fn compile_one(
    lib: &Arc<Library>,
    global: &mut Option<GlobalSession>,
    settings: &Settings<'_>,
    name: &str,
    path: &Path,
    out_dir: &Path,
) -> Result<Record, Error> {
    let slang_err = |message: String| Error::Slang {
        module: name.to_owned(),
        message,
    };
    if global.is_none() {
        *global = Some(GlobalSession::new(Arc::clone(lib)).map_err(slang_err)?);
    }
    let global = global.as_ref().expect("created above");
    let session = global.session(settings.options).map_err(slang_err)?;
    let source = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let compile_err = |diagnostics: String| Error::Compile {
        module: name.to_owned(),
        diagnostics,
    };
    let module = session
        .load(&name.replace('/', "."), path, &source)
        .map_err(compile_err)?;
    let extra = link_modules(&session, settings, out_dir).map_err(compile_err)?;
    let program = session
        .link(&module, &extra, |_| true)
        .map_err(compile_err)?;
    let words = program.spirv().map_err(compile_err)?;
    let spv = out_dir.join(format!("{name}.spv"));
    write(&spv, &words_to_bytes(&words))?;
    let layout = program.layout().map_err(compile_err)?;
    let model = model::extract(&session, layout, &words).map_err(|message| Error::Reflection {
        module: name.to_owned(),
        message,
    })?;

    let mut inputs: Vec<PathBuf> = module
        .dependencies()
        .iter()
        .map(|p| canonical_lossy(p))
        .collect();
    inputs.sort();
    inputs.dedup();
    let metal = match settings.metal {
        Some(metal_options) => {
            let session = global.session(metal_options).map_err(slang_err)?;
            metal_issues(&session, settings, &model, name, path, &source, out_dir)?
        }
        None => Vec::new(),
    };
    let key = cache_key(settings, path, &inputs).ok_or_else(|| Error::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other("an input vanished during the build"),
    })?;

    let omm = match spirv::omm_variant(&words) {
        Some(variant) => {
            let p = out_dir.join(format!("{name}.omm.spv"));
            write(&p, &words_to_bytes(&variant))?;
            Some(p)
        }
        None => None,
    };
    Ok(Record {
        key,
        kernel: path.to_path_buf(),
        inputs,
        spv,
        omm,
        hash: hash64(&words_to_bytes(&words)),
        model,
        metal,
    })
}

/// The modules linked into every kernel besides its own: `ASSERTS = true` when asserts are
/// on.
fn link_modules(
    session: &Session,
    settings: &Settings<'_>,
    out_dir: &Path,
) -> Result<Vec<super::slang::Module>, String> {
    if !settings.asserts {
        return Ok(Vec::new());
    }
    let path = out_dir.join("orr-shaders").join("asserts.slang");
    Ok(vec![session.load(
        "orr_asserts",
        &path,
        "export static const bool ASSERTS = true;\n",
    )?])
}

/// Loads the kernel at `path` once more into the Metal `session`, compiles its non-raygen
/// entries to MSL and returns what the MSL check finds. Writes `<name>.metal`.
fn metal_issues(
    session: &Session,
    settings: &Settings<'_>,
    model: &Model,
    name: &str,
    path: &Path,
    source: &str,
    out_dir: &Path,
) -> Result<Vec<String>, Error> {
    let entries: BTreeSet<&str> = model
        .entries
        .iter()
        .filter(|e| e.stage != crate::Stage::RayGen)
        .map(|e| e.name.as_str())
        .collect();
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let text = metal::without_raygen(source);
    let compiled = (|| {
        let module = session.load(&name.replace('/', "."), path, &text)?;
        let extra = link_modules(session, settings, out_dir)?;
        let program = session.link(&module, &extra, |e| entries.contains(e))?;
        Ok::<_, String>((program.text()?, program))
    })();
    let (msl, program) = match compiled {
        Ok(r) => r,
        Err(diagnostics) => return Ok(vec![format!("Metal emission failed:\n{diagnostics}")]),
    };
    write(&out_dir.join(format!("{name}.metal")), msl.as_bytes())?;
    let mut issues = metal::msl_issues(&msl);
    match program.layout() {
        Ok(layout) => issues.extend(metal::layout_issues(&session.lib.api, layout, model)),
        Err(diagnostics) => issues.push(format!("no Metal layout: {diagnostics}")),
    }
    Ok(issues)
}

/// The interface of a module's entry points: the root's layout, through pointees, and the
/// specialisation constants' ids and types. Names do not count.
fn interface_hash(model: &Model) -> u64 {
    fn walk(ty: &Ty, model: &Model, seen: &mut Vec<String>, out: &mut String) {
        match ty {
            Ty::Struct(name) => {
                if let Some(i) = seen.iter().position(|s| s == name) {
                    out.push_str(&format!("^{i}"));
                    return;
                }
                seen.push(name.clone());
                let def = &model.structs[name];
                out.push_str(&format!("{{{:?}", def.size));
                for f in &def.fields {
                    out.push_str(&format!("@{}:", f.offset));
                    walk(&f.ty, model, seen, out);
                }
                out.push('}');
            }
            Ty::Pointer(p) => {
                out.push('*');
                walk(p, model, seen, out);
            }
            Ty::Array {
                element,
                len,
                stride,
            } => {
                out.push_str(&format!("[{len};{stride}]"));
                walk(element, model, seen, out);
            }
            other => out.push_str(&format!("{other:?}")),
        }
    }
    let mut text = String::new();
    if let Some(root) = &model.root {
        walk(&Ty::Struct(root.clone()), model, &mut Vec::new(), &mut text);
    }
    for s in &model.spec {
        text.push_str(&format!("|{}:{:?}", s.id, s.scalar));
    }
    hash64(text.as_bytes())
}

fn hash64(bytes: &[u8]) -> u64 {
    let h = blake3::hash(bytes);
    u64::from_le_bytes(h.as_bytes()[..8].try_into().expect("8 bytes"))
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let io = |source| Error::Io {
        path: path.to_path_buf(),
        source,
    };
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io)?;
    }
    if fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    fs::write(path, bytes).map_err(io)
}

fn canonical(path: &Path) -> Result<PathBuf, Error> {
    path.canonicalize()
        .map(strip_verbatim)
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn canonical_lossy(path: &Path) -> PathBuf {
    path.canonicalize()
        .map(strip_verbatim)
        .unwrap_or_else(|_| path.to_path_buf())
}

/// Drops Windows' `\\?\` prefix so paths stay readable and comparable with user paths.
fn strip_verbatim(p: PathBuf) -> PathBuf {
    match p.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(s) if !s.starts_with("UNC\\") => PathBuf::from(s),
        _ => p,
    }
}
