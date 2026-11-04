use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;
use std::{env, fs, thread};

use jobserver::{Acquired, Client};
use serde::{Deserialize, Serialize};

use super::codegen::{self, Kernel, Owner, Shared};
use super::layout::{Plan, Planner};
use super::link::{self, KernelPoints, Link, LinkModule, LinkPoint, LinkPoints};
use super::lint::{self, Lint, LintError};
use super::model::{self, Model, StructDef, Ty};
use super::precompile::{self, Precompiled};
use super::roots::{self, Root};
use super::slang::{GlobalSession, Library, Session};
use super::{Error, SLANG_VERSION, SLANGC_OPTIONS, metal, spirv};

/// Compiles a crate's Slang kernels and generates Rust for them. Used from `build.rs`.
///
/// ```no_run
/// # fn main() -> Result<(), orr_shaders::build::Error> {
/// let src = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("src");
/// orr_shaders::build::Build::new()?.kernels(&src).compile()?;
/// # Ok(())
/// # }
/// ```
///
/// and in the crate: `mod shaders { include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs")); }`.
pub struct Build {
    lib: Arc<Library>,
    kernels: Option<PathBuf>,
    prefix: Option<String>,
    includes: Vec<Root>,
    deps: Vec<String>,
    externs: BTreeMap<String, String>,
    debug_info: DebugInfo,
    fp_precise: bool,
    asserts: bool,
    omm: bool,
    neural: bool,
    metal_check: bool,
    precompile: bool,
    allowed: BTreeSet<Lint>,
    out_dir: Option<PathBuf>,
    link_points: Vec<PathBuf>,
    links: Vec<Link>,
    type_maps: Vec<PathBuf>,
    publish_types: Option<String>,
}

/// How much debug info a build puts in the SPIR-V, and where.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DebugInfo {
    Off,
    /// Line tables only (`-g1`): `OpLine` instructions, no variables, types or scopes.
    Lines,
    /// In the SPIR-V (`-g2`).
    Embedded,
    /// In a sidecar beside each module (`-g2 -separate-debug-info`): the module is stripped
    /// but keeps its `DebugBuildIdentifier`, and `<name>.dbg.spv` holds the debug info, the
    /// form Aftermath maps crash dumps through.
    Separate,
}

/// The session option of builds with micromaps, which `orr_root.h` and
/// `orr_shaders.trace` test.
const OMM_OPTION: &str = "-DORR_OMM_BUILD";

/// The session option of builds with neural texture decoding, which `orr_scene.material`
/// tests.
const NEURAL_OPTION: &str = "-DORR_NEURAL_BUILD";

/// What [`Build::compile`] produced.
#[derive(Debug)]
pub struct Output {
    /// The crate's shader root followed by every root it was given, `;`-separated, each
    /// `<prefix>=<dir>` or `<dir>`. `compile` prints it as `cargo::metadata=shader_dirs`.
    pub shader_dirs: String,
    /// The modules compiled by this run, linked kernels as `linked/<kernel>`; the others
    /// were up to date.
    pub compiled: Vec<String>,
    /// The build manifest: every module with its inputs, options and outputs.
    pub manifest: PathBuf,
    /// `link_points.json`: the kernels of this build and of every file given to
    /// [`Build::link_points`] that declare a link point. `compile` prints it as
    /// `cargo::metadata=link_points`.
    pub link_points: PathBuf,
    /// `types.json`, with [`Build::publish_types`]: the Rust type of every struct this
    /// build generated or mapped. `compile` prints it as `cargo::metadata=types`.
    pub types: Option<PathBuf>,
    /// The build was with opacity micromaps ([`Build::omm`]).
    pub omm: bool,
    /// The build decodes neural textures ([`Build::neural`]).
    pub neural: bool,
}

impl Build {
    /// Locates and version-checks the Slang release.
    pub fn new() -> Result<Self, Error> {
        let release = env::var("PROFILE").is_ok_and(|p| p == "release");
        // `1` or `0` from `var`, else `default`.
        let flag = |var: &str, default: bool| env::var_os(var).map_or(default, |v| v == "1");
        let capture = flag("ORR_SHADER_CAPTURE", false);
        Ok(Self {
            lib: super::toolchain::load()?,
            kernels: None,
            prefix: None,
            includes: Vec::new(),
            deps: Vec::new(),
            externs: BTreeMap::new(),
            debug_info: match env::var("ORR_SHADER_DEBUG_INFO").as_deref() {
                Err(_) | Ok("lines") => DebugInfo::Lines,
                Ok("off") => DebugInfo::Off,
                Ok("embedded") => DebugInfo::Embedded,
                Ok("separate") => DebugInfo::Separate,
                Ok(other) => return Err(Error::DebugInfoVar(other.to_owned())),
            },
            fp_precise: false,
            asserts: flag("ORR_SHADER_ASSERTS", !release && !capture),
            omm: flag("ORR_SHADER_OMM", !capture),
            neural: flag("ORR_SHADER_NEURAL", true),
            metal_check: env::var_os("ORR_METAL_CHECK").is_some_and(|v| v == "1"),
            precompile: env::var_os("ORR_SHADER_PRECOMPILE").is_none_or(|v| v != "0"),
            allowed: BTreeSet::new(),
            out_dir: None,
            link_points: Vec::new(),
            links: Vec::new(),
            type_maps: Vec::new(),
            publish_types: None,
        })
    }

    /// The crate's shader root. Every `.slang` file under it that contains `[shader(`
    /// becomes one module. The crate's files import each other, and dependents import them,
    /// as `<crate>.<path>`: `src/media/froxel.slang` under `orr_lighting` is
    /// `orr_lighting.media.froxel`, and `<crate>` has `_` where the package name has `-`.
    pub fn kernels(mut self, root: impl AsRef<Path>) -> Self {
        self.kernels = Some(root.as_ref().to_path_buf());
        self
    }

    /// The name [`kernels`](Self::kernels)' root is imported under, instead of the package name.
    pub fn prefix(mut self, prefix: &str) -> Self {
        self.prefix = Some(prefix.to_owned());
        self
    }

    /// One more import root whose modules are imported by their path under it.
    /// `orr_shaders`' and `orr_color`'s Slang roots are always present, and so is
    /// [`openpbr_dir`](super::openpbr_dir) when it is found.
    pub fn include(mut self, dir: impl AsRef<Path>) -> Self {
        self.includes.push(Root::plain(dir.as_ref()));
        self
    }

    /// One more import root whose modules are imported as `<prefix>.<path>`.
    pub fn include_as(mut self, prefix: &str, dir: impl AsRef<Path>) -> Self {
        self.includes.push(Root::prefixed(prefix, dir.as_ref()));
        self
    }

    /// Every root in a `DEP_*_SHADER_DIRS` value (`;`-separated).
    pub fn roots(mut self, list: &str) -> Self {
        self.includes.extend(roots::parse_list(list));
        self
    }

    /// Everything the direct dependency with `links = "<links>"` publishes: its shader roots,
    /// the Rust types of its Slang structs, as [`types`](Self::types) maps them, and its link
    /// points, as [`link_points`](Self::link_points) reads them. A dependency's types and
    /// link points are optional. Repeatable.
    pub fn dep(mut self, links: &str) -> Self {
        self.deps.push(links.to_owned());
        self
    }

    /// Every `link_points.json` in a `DEP_*_LINK_POINTS` value (`;`-separated): the kernels
    /// that [`link`](Self::link) compiles. Repeatable.
    pub fn link_points(mut self, list: &str) -> Self {
        self.link_points
            .extend(list.split(';').filter(|s| !s.is_empty()).map(PathBuf::from));
        self
    }

    /// Links Slang type `ty` from `module` (relative to `CARGO_MANIFEST_DIR`) into every
    /// kernel of the [`link_points`](Self::link_points) files that declares `point`.
    ///
    /// Each such kernel is compiled once more with a generated module that exports the
    /// kernel's `extern struct`, and goes into `KERNELS` in `orr_kernels.rs`. Surfaces and
    /// displacements become one `switch` on the input's id: ids run from 1 in call order,
    /// id 0 and unknown ids call the default type, and `orr_kernels.rs` holds the ids as
    /// `surface::<NAME>` and `displacement::<NAME>`. Environment and medium take one call
    /// each at most. A `module` under a prefixed root is imported by its name there; any other
    /// is imported by its file stem, and its directory joins the include path.
    pub fn link(
        mut self,
        point: LinkPoint,
        name: &str,
        module: impl AsRef<Path>,
        ty: &str,
    ) -> Self {
        self.links.push(Link::new(point, name, module.as_ref(), ty));
        self
    }

    /// Uses an existing `#[repr(C)]` Rust type for a Slang struct instead of generating
    /// one; the generated code asserts its size and field offsets.
    pub fn extern_type(mut self, slang_name: &str, rust_path: &str) -> Self {
        self.externs
            .insert(slang_name.to_owned(), rust_path.to_owned());
        self
    }

    /// Maps every struct of each `types.json` in a `DEP_*_TYPES` value (`;`-separated) to
    /// the Rust type a dependency published for it, as [`extern_type`](Self::extern_type)
    /// does. An `extern_type` call for the same struct wins. Repeatable.
    pub fn types(mut self, list: &str) -> Self {
        self.type_maps
            .extend(list.split(';').filter(|s| !s.is_empty()).map(PathBuf::from));
        self
    }

    /// Publishes this crate's Rust types for its Slang structs to dependents, which read
    /// them with [`types`](Self::types). `path` names the module dependents find them in
    /// (`::my_crate::types`): the generated structs must be re-exported there, as must every
    /// `extern_type` path starting with `crate::`, which is published under it. Mapped types from
    /// [`types`](Self::types) are published too, so a dependent of this crate needs no
    /// other map.
    pub fn publish_types(mut self, path: &str) -> Self {
        self.publish_types = Some(path.trim_end_matches("::").to_owned());
        self
    }

    /// How much debug info the SPIR-V carries, overriding `ORR_SHADER_DEBUG_INFO` (`off`,
    /// `lines`, `embedded` or `separate`). Without either it is [`DebugInfo::Lines`] in
    /// every profile.
    pub fn debug_info(mut self, debug_info: DebugInfo) -> Self {
        self.debug_info = debug_info;
        self
    }

    /// `-fp-mode precise`, for entry-parity test kernels.
    pub fn fp_precise(mut self, on: bool) -> Self {
        self.fp_precise = on;
        self
    }

    /// Links `orr_shaders.debug`'s `ASSERTS = true`, so `orr_assert` and checks such as `Tlas.get()`
    /// on address 0 abort. Pipelines from modules that abort need a device with
    /// `shaderAbort`. On by default unless the profile is `release` or the build is for a
    /// capture session (`ORR_SHADER_CAPTURE=1`), whose devices leave `shaderAbort` off;
    /// `ORR_SHADER_ASSERTS=0` or `1` overrides the default.
    pub fn asserts(mut self, on: bool) -> Self {
        self.asserts = on;
        self
    }

    /// Builds with opacity micromaps: `ORR_OMM(entry)` from `orr_root.h` declares the
    /// micromap execution mode on the entry points of ray-query modules, and
    /// `RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE` reaches the ray queries. Pipelines from
    /// these modules need a device with micromaps. Without, both drop out and alpha-tested
    /// geometry is tested in the candidate loop. On by default unless the build is for a
    /// capture session (`ORR_SHADER_CAPTURE=1`), since captures do not support micromaps;
    /// `ORR_SHADER_OMM=0` or `1` overrides the default. The Metal check builds without.
    pub fn omm(mut self, on: bool) -> Self {
        self.omm = on;
        self
    }

    /// Builds with neural texture decoding: kernels that sample materials decode neural
    /// texture sets with cooperative vectors, so pipelines from them need a device with
    /// cooperative vectors. On by default; `ORR_SHADER_NEURAL=0` or `1` overrides it. The
    /// Metal check builds without.
    pub fn neural(mut self, on: bool) -> Self {
        self.neural = on;
        self
    }

    /// Also compiles each module's compute and mesh-pipeline entries to MSL and checks them
    /// ([`Lint::Metal`]). Defaults to on when `ORR_METAL_CHECK=1`. The `.metal` files are
    /// written beside the SPIR-V.
    pub fn metal_check(mut self, on: bool) -> Self {
        self.metal_check = on;
        self
    }

    /// Compiles the library modules that kernels import once per build into
    /// `.slang-module` files, which every kernel then loads instead of the source. The
    /// SPIR-V is the same either way. On unless `ORR_SHADER_PRECOMPILE=0`.
    pub fn precompile(mut self, on: bool) -> Self {
        self.precompile = on;
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
    /// `orr_shaders.rs`, the `.spv` files, `shaders.manifest.json` and `link_points.json` to
    /// the output directory. With [`link`](Self::link) calls it also compiles the linked
    /// kernels under `linked/`; `orr_kernels.rs` holds their `KERNELS` set, empty without
    /// links. Prints `rerun-if-changed` lines and `cargo::metadata=link_points`.
    pub fn compile(mut self) -> Result<Output, Error> {
        let started = Instant::now();
        for dep in std::mem::take(&mut self.deps) {
            let var = |what: &str| env::var(dep_var(&dep, what));
            let dirs = var("SHADER_DIRS").map_err(|_| Error::NoDep {
                var: dep_var(&dep, "SHADER_DIRS"),
            })?;
            self = self.roots(&dirs);
            if let Ok(types) = var("TYPES") {
                self = self.types(&types);
            }
            if let Ok(points) = var("LINK_POINTS") {
                self = self.link_points(&points);
            }
        }
        for path in &self.type_maps {
            for (slang, rust) in read_types(path)? {
                self.externs.entry(slang).or_insert(rust);
            }
        }
        let out_dir = out_dir(self.out_dir.as_deref())?;
        let own_root = self.kernels.as_deref().map(canonical).transpose()?;
        let prefix = self
            .prefix
            .clone()
            .or_else(|| Some(env::var("CARGO_PKG_NAME").ok()?.replace('-', "_")));
        let mut roots: Vec<Root> = own_root
            .iter()
            .map(|dir| Root {
                prefix: prefix.clone(),
                dir: dir.clone(),
            })
            .collect();
        for root in &self.includes {
            let root = root.canonical()?;
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        let shader_dirs = roots::format_list(&roots);
        roots.extend(roots::library()?);
        let mut lint_roots = roots.clone();
        roots.extend(super::openpbr_dir().ok().map(Root::plain));

        let links = link::resolve(&self.links, &roots)?;
        let mut listed = LinkPoints::default();
        for path in &self.link_points {
            listed.merge(LinkPoints::read(path)?);
        }

        let mut kernels = Vec::new();
        let mut own_points = LinkPoints::default();
        if let Some(root) = &own_root {
            kernels = find_kernels(root)?;
            for (name, path) in &kernels {
                let points = link::declared(&read(path)?);
                if !points.is_empty() {
                    own_points.kernels.push(KernelPoints {
                        name: name.clone(),
                        path: path.clone(),
                        root: Root {
                            prefix: prefix.clone(),
                            dir: root.clone(),
                        },
                        points,
                    });
                }
            }
        }

        let tool = tool_hash();
        let records = out_dir.join("orr_shaders");
        let jobs = kernels
            .into_iter()
            .map(|(name, path)| Job {
                record: records.join(format!("{name}.json")),
                spv: out_dir.join(format!("{name}.spv")),
                name,
                path,
                links: Vec::new(),
            })
            .collect();
        let Run {
            modules,
            mut compiled,
        } = self.run(&tool, &roots, &records, jobs)?;

        // The listed kernels compiled once more with the linked types.
        let mut linked_modules = BTreeMap::new();
        let mut linked_roots = roots.clone();
        if !links.is_empty() {
            let jobs = linked_jobs(&links, &listed, &out_dir)?;
            let module_dirs = link::module_dirs(&links);
            let kernel_roots = listed.kernels.iter().map(|k| &k.root);
            for root in module_dirs.iter().chain(kernel_roots) {
                if !linked_roots.contains(root) {
                    linked_roots.push(root.clone());
                }
            }
            for root in module_dirs {
                if !lint_roots.contains(&root) {
                    lint_roots.push(root);
                }
            }
            let run = self.run(&tool, &linked_roots, &records, jobs)?;
            linked_modules = run.modules;
            compiled.extend(run.compiled.into_iter().map(|n| format!("linked/{n}")));
        }

        let mut lints = lint::duplicate_imports(&lint_roots);
        for (name, record) in modules.iter().chain(&linked_modules) {
            lints.extend(record.metal.iter().map(|message| LintError {
                lint: Lint::Metal,
                module: name.clone(),
                message: message.clone(),
            }));
            if self.omm {
                lints.extend(record.model.no_omm_mode.iter().map(|entry| LintError {
                    lint: Lint::Micromap,
                    module: name.clone(),
                    message: no_omm_mode(entry),
                }));
            }
        }
        let mut seen_files = BTreeSet::new();
        for record in modules.values().chain(linked_modules.values()) {
            for input in &record.inputs {
                if seen_files.insert(input.clone())
                    && let Ok(src) = fs::read_to_string(input)
                {
                    lints.extend(lint::source_lints(input, &src));
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
                            let root = roots.iter().find(|r| file.starts_with(&r.dir));
                            lints.push(LintError {
                                lint: Lint::Ownership,
                                module: name.clone(),
                                message: format!(
                                    "struct {s} is declared in {} under {}; map it with .extern_type({s:?}, \"<crate>::<path>::{}\")",
                                    file.display(),
                                    root.map_or("no shader root".to_owned(), |r| format!("shader root {}", r.dir.display())),
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
            let module_plans = plan(name, &model)?;
            lints.extend(lint::model_lints(name, &model, &module_plans));
            plans.insert(name.clone(), module_plans);
            models.insert(name.clone(), model);
        }
        let mut linked_plans: HashMap<String, HashMap<String, Plan>> = HashMap::new();
        for (name, record) in &linked_modules {
            let module_plans = plan(name, &record.model)?;
            lints.extend(lint::model_lints(name, &record.model, &module_plans));
            linked_plans.insert(name.clone(), module_plans);
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
                debug_info: record.debug_info.as_deref(),
                hash: record.hash,
                // From the module's own model, as a reload of the module alone computes it.
                interface: interface_hash(&record.model),
                model,
                owners: &owners[name],
                plans: &plans[name],
            });
        }
        let rust = codegen::generate(&kernels_out, &shared, &manifest_path);
        write(&out_dir.join("orr_shaders.rs"), rust.as_bytes())?;

        let no_owners = HashMap::new();
        let linked_out: Vec<Kernel<'_>> = linked_modules
            .iter()
            .map(|(name, record)| Kernel {
                name,
                spv: &record.spv,
                debug_info: record.debug_info.as_deref(),
                hash: record.hash,
                interface: interface_hash(&record.model),
                model: &record.model,
                owners: &no_owners,
                plans: &linked_plans[name],
            })
            .collect();
        let rust = codegen::kernel_set(&linked_out, &link::ids(&links), &manifest_path);
        write(&out_dir.join("orr_kernels.rs"), rust.as_bytes())?;

        let manifest = Manifest {
            slang: SLANG_VERSION.to_owned(),
            library: self.lib.path.clone(),
            options: self.options(&include_dirs(&roots)),
            linked_options: self.options(&include_dirs(&linked_roots)),
            roots: roots.clone(),
            linked_roots,
            asserts: self.asserts,
            omm: self.omm,
            modules: modules
                .iter()
                .map(|(name, r)| (name, r, false))
                .chain(linked_modules.iter().map(|(name, r)| (name, r, true)))
                .map(|(name, r, linked)| ManifestModule {
                    name: name.clone(),
                    kernel: r.kernel.clone(),
                    linked,
                    spv: r.spv.clone(),
                    debug_info: r.debug_info.clone(),
                    hash: r.hash,
                    inputs: r.inputs.clone(),
                    links: r.links.clone(),
                    precompiled: r.precompiled.clone(),
                })
                .collect(),
        };
        write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest)
                .expect("the manifest serializes")
                .as_bytes(),
        )?;

        let link_points_path = out_dir.join("link_points.json");
        own_points.merge(listed);
        write(
            &link_points_path,
            serde_json::to_string_pretty(&own_points)
                .expect("link points serialize")
                .as_bytes(),
        )?;

        let types_path = match &self.publish_types {
            Some(module) => {
                let mut published: BTreeMap<String, String> = self
                    .externs
                    .iter()
                    .map(|(slang, rust)| {
                        let rust = match rust.strip_prefix("crate::") {
                            Some(rest) => format!("{module}::{rest}"),
                            None => rust.clone(),
                        };
                        (slang.clone(), rust)
                    })
                    .collect();
                for slang in shared.types.keys() {
                    published.insert(
                        slang.clone(),
                        format!("{module}::{}", codegen::rust_name(slang)),
                    );
                }
                let path = out_dir.join("types.json");
                write_types(&path, &published)?;
                Some(path)
            }
            None => None,
        };

        let mut watched: Vec<PathBuf> = roots.into_iter().map(|r| r.dir).collect();
        watched.extend(self.link_points.iter().cloned());
        watched.extend(links.iter().map(|l| l.module.clone()));
        watched.extend(self.type_maps.iter().cloned());
        print_cargo(
            &shader_dirs,
            &watched,
            modules.values().chain(linked_modules.values()),
            &link_points_path,
            types_path.as_deref(),
        );
        if env::var_os("ORR_SHADER_TIMINGS").is_some() {
            #[allow(clippy::print_stdout)]
            {
                println!(
                    "cargo::warning=orr_shaders: {} of {} modules compiled in {:.2} s",
                    compiled.len(),
                    modules.len() + linked_modules.len(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
        Ok(Output {
            shader_dirs,
            compiled,
            manifest: manifest_path,
            link_points: link_points_path,
            types: types_path,
            omm: self.omm,
            neural: self.neural,
        })
    }

    /// Precompiles the libraries the jobs import, loads up-to-date records and compiles the
    /// rest.
    fn run(
        &self,
        tool: &str,
        roots: &[Root],
        records: &Path,
        jobs: Vec<Job>,
    ) -> Result<Run, Error> {
        let include_dirs = include_dirs(roots);
        let options = self.options(&include_dirs);
        let (precompiled, kernel_options) = if self.precompile && !jobs.is_empty() {
            let root_list = roots::format_list(roots);
            let identity = format!("{}\n{root_list}", options.join("\n"));
            let dir = records
                .join("modules")
                .join(format!("{:016x}", hash64(identity.as_bytes())));
            let mut key_parts = vec![tool, root_list.as_str()];
            key_parts.extend(options.iter().map(String::as_str));
            let mut sources = Vec::new();
            for job in &jobs {
                sources.push((job.path.clone(), read(&job.path)?));
                sources.extend(job.links.iter().map(|l| (l.path.clone(), l.source.clone())));
            }
            let precompiled =
                precompile::precompile(&self.lib, &options, &key_parts, roots, &sources, &dir)?;
            let mut dirs = vec![canonical_lossy(&dir)];
            dirs.extend(include_dirs.iter().cloned());
            (precompiled, self.options(&dirs))
        } else {
            (Precompiled::default(), options)
        };
        let metal_options = self
            .metal_check
            .then(|| metal_session_options(&include_dirs));
        let settings = Settings {
            tool,
            options: &kernel_options,
            metal: metal_options.as_deref(),
            asserts: self.asserts,
            separate_debug: self.debug_info == DebugInfo::Separate,
            precompiled: &precompiled,
            roots,
        };
        let mut modules = BTreeMap::new();
        let mut todo = Vec::new();
        for job in jobs {
            match load_record(&settings, &job) {
                Some(record) => {
                    modules.insert(job.name, record);
                }
                None => todo.push(job),
            }
        }
        let compiled = todo.iter().map(|j| j.name.clone()).collect();
        modules.extend(compile_all(&self.lib, &settings, todo)?);
        Ok(Run { modules, compiled })
    }

    /// The session options for kernels that import from `include_dirs`.
    fn options(&self, include_dirs: &[PathBuf]) -> Vec<String> {
        let mut options: Vec<String> = SLANGC_OPTIONS.iter().map(|s| (*s).to_owned()).collect();
        for dir in include_dirs {
            options.push("-I".into());
            options.push(dir.display().to_string());
        }
        match self.debug_info {
            DebugInfo::Off => {}
            DebugInfo::Lines => options.push("-g1".into()),
            DebugInfo::Embedded => options.push("-g2".into()),
            DebugInfo::Separate => {
                options.extend(["-g2".into(), "-separate-debug-info".into()]);
            }
        }
        if self.fp_precise {
            options.extend(["-fp-mode".into(), "precise".into()]);
        }
        if self.omm {
            options.push(OMM_OPTION.into());
        }
        if self.neural {
            options.push(NEURAL_OPTION.into());
        }
        options
    }
}

/// The search paths of `roots`: those without a prefix.
fn include_dirs(roots: &[Root]) -> Vec<PathBuf> {
    roots
        .iter()
        .filter(|r| r.prefix.is_none())
        .map(|r| r.dir.clone())
        .collect()
}

/// The environment variable that holds what the dependency with `links = "<links>"` published.
fn dep_var(links: &str, what: &str) -> String {
    format!("DEP_{}_{what}", links.to_uppercase().replace('-', "_"))
}

fn metal_session_options(include_dirs: &[PathBuf]) -> Vec<String> {
    let mut o: Vec<String> = metal::OPTIONS.iter().map(|s| (*s).to_owned()).collect();
    for dir in include_dirs {
        o.extend(["-I".to_owned(), dir.display().to_string()]);
    }
    o
}

/// One job per listed kernel that declares a linked point, with its generated link
/// modules.
fn linked_jobs(links: &[Link], listed: &LinkPoints, out_dir: &Path) -> Result<Vec<Job>, Error> {
    let declared: BTreeSet<LinkPoint> = listed
        .kernels
        .iter()
        .flat_map(|k| k.points.iter().map(|d| d.point))
        .collect();
    if let Some(l) = links.iter().find(|l| !declared.contains(&l.point)) {
        return Err(Error::Link {
            message: format!(
                "the {} point is linked, but no kernel in the link-points files declares it \
                 (they list {} kernels); name the renderer's crate with Build::dep, or pass its \
                 link points to Build::link_points",
                l.point.label(),
                listed.kernels.len()
            ),
        });
    }
    let modules_dir = out_dir.join("orr_shaders").join("link");
    let records = out_dir.join("orr_shaders").join("linked");
    let mut generated: BTreeMap<&link::Declared, Option<LinkModule>> = BTreeMap::new();
    let mut jobs: Vec<Job> = Vec::new();
    for k in &listed.kernels {
        let mut modules = Vec::new();
        for decl in &k.points {
            if !generated.contains_key(decl) {
                generated.insert(decl, link::write_module(decl, links, &modules_dir)?);
            }
            modules.extend(generated[decl].clone());
        }
        if modules.is_empty() {
            continue;
        }
        if jobs.iter().any(|j| j.name == k.name) {
            return Err(Error::Link {
                message: format!("two link-points files list a kernel named {}", k.name),
            });
        }
        jobs.push(Job {
            name: k.name.clone(),
            path: k.path.clone(),
            record: records.join(format!("{}.json", k.name)),
            spv: out_dir.join("linked").join(format!("{}.spv", k.name)),
            links: modules,
        });
    }
    Ok(jobs)
}

/// The layout plans of every struct of a module.
fn plan(name: &str, model: &Model) -> Result<HashMap<String, Plan>, Error> {
    Planner::new(&model.structs)
        .plan_all()
        .map_err(|m| Error::Layout {
            module: name.to_owned(),
            structure: m.structure,
            field: m.field,
            message: m.message,
        })
}

/// `shaders.manifest.json`: every module of a build with what compiling it again from
/// source takes.
#[derive(Serialize, Deserialize)]
pub(super) struct Manifest {
    pub(super) slang: String,
    /// The Slang library the build loaded.
    pub(super) library: PathBuf,
    /// The source options of the crate's own kernels.
    pub(super) options: Vec<String>,
    /// The source options of the linked kernels.
    pub(super) linked_options: Vec<String>,
    /// The import roots of the crate's own kernels.
    pub(super) roots: Vec<Root>,
    /// The import roots of the linked kernels.
    pub(super) linked_roots: Vec<Root>,
    pub(super) asserts: bool,
    pub(super) omm: bool,
    pub(super) modules: Vec<ManifestModule>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct ManifestModule {
    pub(super) name: String,
    pub(super) kernel: PathBuf,
    /// Compiled with this build's linked types; goes into `KERNELS`.
    pub(super) linked: bool,
    spv: PathBuf,
    /// The separated debug info of `spv`.
    debug_info: Option<PathBuf>,
    pub(super) hash: u64,
    pub(super) inputs: Vec<PathBuf>,
    /// The generated link modules composed with the kernel.
    pub(super) links: Vec<LinkFile>,
    /// The library modules the compile loaded precompiled, by import name.
    precompiled: Vec<String>,
}

/// A generated link module as the manifest lists it.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct LinkFile {
    pub(super) module: String,
    pub(super) path: PathBuf,
}

/// One compiled module as the build remembers it between runs.
#[derive(Clone, Serialize, Deserialize)]
struct Record {
    key: String,
    kernel: PathBuf,
    inputs: Vec<PathBuf>,
    spv: PathBuf,
    debug_info: Option<PathBuf>,
    hash: u64,
    links: Vec<LinkFile>,
    precompiled: Vec<String>,
    model: Model,
    /// MSL check failures; empty when the check passed or did not run.
    metal: Vec<String>,
}

/// The modules of one option set.
struct Run {
    modules: BTreeMap<String, Record>,
    /// The names of those compiled by this run; the others were up to date.
    compiled: Vec<String>,
}

/// What every module of one build is compiled with.
pub(super) struct Settings<'a> {
    /// A hash of the running build script, so a changed build invalidates every record.
    pub(super) tool: &'a str,
    /// The kernel session's options, led by the precompiled modules' directory when there
    /// is one.
    pub(super) options: &'a [String],
    /// The Metal session's options when the MSL check runs.
    pub(super) metal: Option<&'a [String]>,
    pub(super) asserts: bool,
    /// `options` hold `-separate-debug-info`.
    pub(super) separate_debug: bool,
    pub(super) precompiled: &'a Precompiled,
    /// The import roots.
    pub(super) roots: &'a [Root],
}

impl Settings<'_> {
    /// Everything besides the inputs that changes a module's outputs.
    fn key_parts(&self) -> Vec<&str> {
        let mut parts = vec![self.tool];
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

/// The work of compiling one kernel file.
struct Job {
    name: String,
    path: PathBuf,
    record: PathBuf,
    spv: PathBuf,
    /// Generated link modules composed with the kernel.
    links: Vec<LinkModule>,
}

fn print_cargo<'a>(
    shader_dirs: &str,
    watched: &[PathBuf],
    modules: impl Iterator<Item = &'a Record>,
    link_points: &Path,
    types: Option<&Path>,
) {
    let mut lines: BTreeSet<PathBuf> = watched.iter().cloned().collect();
    for m in modules {
        lines.extend(m.inputs.iter().cloned());
    }
    #[allow(clippy::print_stdout)]
    {
        println!("cargo::metadata=shader_dirs={shader_dirs}");
        println!("cargo::metadata=link_points={}", link_points.display());
        if let Some(types) = types {
            println!("cargo::metadata=types={}", types.display());
        }
        for l in lines {
            println!("cargo::rerun-if-changed={}", l.display());
        }
        for var in [
            "ORR_SLANGC",
            "ORR_OPENPBR_BSDF",
            "ORR_METAL_CHECK",
            "ORR_SHADER_PRECOMPILE",
            "ORR_SHADER_TIMINGS",
            "ORR_SHADER_ASSERTS",
            "ORR_SHADER_DEBUG_INFO",
            "ORR_SHADER_OMM",
            "ORR_SHADER_NEURAL",
            "ORR_SHADER_CAPTURE",
        ] {
            println!("cargo::rerun-if-env-changed={var}");
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

fn cache_key(settings: &Settings<'_>, job: &Job, inputs: &[PathBuf]) -> Option<String> {
    let mut h = blake3::Hasher::new();
    h.update(SLANG_VERSION.as_bytes());
    for o in settings.key_parts() {
        h.update(o.as_bytes());
        h.update(&[0]);
    }
    // Prefixed roots are in no option, and an import's file depends on them.
    for root in settings.roots {
        h.update(root.entry().as_bytes());
        h.update(&[0]);
    }
    h.update(job.path.to_string_lossy().as_bytes());
    for l in &job.links {
        h.update(&[0]);
        h.update(l.module.as_bytes());
    }
    h.update(&[1]);
    for input in inputs {
        h.update(input.to_string_lossy().as_bytes());
        h.update(&[0]);
        h.update(&fs::read(input).ok()?);
    }
    Some(h.finalize().to_hex().to_string())
}

fn load_record(settings: &Settings<'_>, job: &Job) -> Option<Record> {
    let record: Record = serde_json::from_slice(&fs::read(&job.record).ok()?).ok()?;
    let key = cache_key(settings, job, &record.inputs)?;
    let outputs_exist = [Some(&record.spv), record.debug_info.as_ref()]
        .into_iter()
        .flatten()
        .all(|p| p.is_file());
    (record.key == key && record.kernel == job.path && outputs_exist).then_some(record)
}

fn compile_all(
    lib: &Arc<Library>,
    settings: &Settings<'_>,
    jobs: Vec<Job>,
) -> Result<BTreeMap<String, Record>, Error> {
    if jobs.is_empty() {
        return Ok(BTreeMap::new());
    }
    let max = jobs.len();
    let queue = Mutex::new(jobs);
    let results = Mutex::new(Vec::new());
    let work = || {
        let mut global: Option<GlobalSession> = None;
        loop {
            let Some(job) = queue.lock().expect("queue").pop() else {
                break;
            };
            let result = compile_one(lib, &mut global, settings, &job).and_then(|record| {
                write(
                    &job.record,
                    &serde_json::to_vec(&record).expect("records serialize"),
                )?;
                Ok(record)
            });
            results.lock().expect("results").push((job.name, result));
        }
    };
    thread::scope(|scope| match parallelism() {
        Parallelism::Threads(n) => {
            for _ in 0..n.min(max) {
                scope.spawn(work);
            }
        }
        Parallelism::Jobserver(client) => spawn_with_tokens(scope, client, max, work),
    });
    let mut out = BTreeMap::new();
    let mut results = results.into_inner().expect("results");
    results.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, result) in results {
        out.insert(name, result?);
    }
    Ok(out)
}

enum Parallelism {
    Threads(usize),
    /// Cargo's jobserver, so the modules of every build script share cargo's `-j` limit.
    Jobserver(Client),
}

/// In a build script (cargo sets `NUM_JOBS` only there), cargo's jobserver or else
/// `NUM_JOBS` threads; elsewhere one thread per core.
fn parallelism() -> Parallelism {
    let Ok(jobs) = env::var("NUM_JOBS") else {
        return Parallelism::Threads(thread::available_parallelism().map_or(1, NonZero::get));
    };
    // SAFETY: `from_env` duplicates the descriptors (or opens the fifo or semaphore) named in
    // CARGO_MAKEFLAGS, which this process leaves as cargo passed them.
    match unsafe { Client::from_env() } {
        Some(client) => Parallelism::Jobserver(client),
        None => Parallelism::Threads(jobs.parse().map_or(1, NonZero::<usize>::get)),
    }
}

/// Runs `work` on up to `max` threads: one on the build script's implicit token and one
/// more for each token cargo hands out before the jobs run out.
fn spawn_with_tokens<'scope>(
    scope: &'scope thread::Scope<'scope, '_>,
    client: Client,
    max: usize,
    work: impl FnOnce() + Copy + Send + 'scope,
) {
    /// Tells the dispatcher that a worker found the queue empty, also when it panics.
    struct Done(mpsc::Sender<Option<Acquired>>);
    impl Drop for Done {
        fn drop(&mut self) {
            let _ = self.0.send(None);
        }
    }
    let (tx, rx) = mpsc::channel();
    let spawn = |token: Option<Acquired>| {
        let done = Done(tx.clone());
        scope.spawn(move || {
            let _done = done;
            let _token = token;
            work();
        });
    };
    spawn(None);
    let tokens = tx.clone();
    let Ok(helper) = client.into_helper_thread(move |token| {
        if let Ok(token) = token {
            let _ = tokens.send(Some(token));
        }
    }) else {
        return;
    };
    for _ in 1..max {
        helper.request_token();
    }
    while let Ok(Some(token)) = rx.recv() {
        spawn(Some(token));
    }
}

fn compile_one(
    lib: &Arc<Library>,
    global: &mut Option<GlobalSession>,
    settings: &Settings<'_>,
    job: &Job,
) -> Result<Record, Error> {
    let (name, path) = (job.name.as_str(), job.path.as_path());
    let slang_err = |message: String| Error::Slang {
        module: name.to_owned(),
        message,
    };
    if global.is_none() {
        *global = Some(GlobalSession::new(Arc::clone(lib)).map_err(slang_err)?);
    }
    let global = global.as_ref().expect("created above");
    let session = global.session(settings.options).map_err(slang_err)?;
    let source = read(path)?;
    let unit = Unit {
        name,
        path,
        source: &source,
        links: &job.links,
    };
    let compiled = compile_kernel(&session, settings, &unit)?;
    let debug_info = match &compiled.debug {
        Some(debug) => {
            let path = job.spv.with_extension("dbg.spv");
            write(&path, bytemuck::cast_slice(debug))?;
            Some(path)
        }
        None => None,
    };
    let bytes: &[u8] = bytemuck::cast_slice(&compiled.words);
    write(&job.spv, bytes)?;
    let metal = match settings.metal {
        Some(metal_options) => {
            let session = global.session(metal_options).map_err(slang_err)?;
            metal_issues(&session, settings, &compiled.model, job, &source)?
        }
        None => Vec::new(),
    };
    let key = cache_key(settings, job, &compiled.inputs).ok_or_else(|| Error::Io {
        path: path.to_path_buf(),
        source: std::io::Error::other("an input vanished during the build"),
    })?;
    Ok(Record {
        key,
        kernel: path.to_path_buf(),
        inputs: compiled.inputs,
        spv: job.spv.clone(),
        debug_info,
        hash: hash64(bytes),
        links: job
            .links
            .iter()
            .map(|l| LinkFile {
                module: l.module.clone(),
                path: l.path.clone(),
            })
            .collect(),
        precompiled: compiled.precompiled,
        model: compiled.model,
        metal,
    })
}

/// One kernel file to compile: its source and the link modules composed with it.
pub(super) struct Unit<'a> {
    pub(super) name: &'a str,
    pub(super) path: &'a Path,
    pub(super) source: &'a str,
    pub(super) links: &'a [LinkModule],
}

/// A compiled kernel, before anything is written.
pub(super) struct Compiled {
    /// With separated debug info: stripped, its strings restored.
    pub(super) words: Vec<u32>,
    /// The separated debug info, when `settings.separate_debug` is set.
    debug: Option<Vec<u32>>,
    pub(super) model: Model,
    /// Every file the kernel and its link modules read.
    pub(super) inputs: Vec<PathBuf>,
    /// The library modules loaded precompiled.
    precompiled: Vec<String>,
}

/// Compiles `unit` in `session`, which holds `settings.options`, and reads its model.
pub(super) fn compile_kernel(
    session: &Session,
    settings: &Settings<'_>,
    unit: &Unit<'_>,
) -> Result<Compiled, Error> {
    let compile_err = |diagnostics: String| Error::Compile {
        module: unit.name.to_owned(),
        diagnostics,
    };
    let mut sources = vec![(unit.path, unit.source)];
    sources.extend(
        unit.links
            .iter()
            .map(|l| (l.path.as_path(), l.source.as_str())),
    );
    roots::load_prefixed(session, settings.roots, &sources, |name| {
        settings.precompiled.holds(name)
    })
    .map_err(compile_err)?;
    let module = session
        .load(&unit.name.replace('/', "."), unit.path, unit.source)
        .map_err(compile_err)?;
    let extra = link_modules(session, settings, unit.links).map_err(compile_err)?;
    let program = session
        .link(&module, &extra, |_| true)
        .map_err(compile_err)?;
    // A stripped module has no names, which the model reads; the sidecar is the whole
    // module.
    let (words, debug) = if settings.separate_debug {
        let (stripped, debug) = program.spirv_separated().map_err(compile_err)?;
        (spirv::restore_strings(stripped, &debug), Some(debug))
    } else {
        (program.spirv().map_err(compile_err)?, None)
    };
    let layout = program.layout().map_err(compile_err)?;
    let named = debug.as_deref().unwrap_or(&words);
    let model =
        model::extract(session, layout, named, settings.precompiled).map_err(|message| {
            Error::Reflection {
                module: unit.name.to_owned(),
                message,
            }
        })?;

    // Dependencies reached through a precompiled library can resolve against the importing
    // link module's directory and name a file outside every root, so those files come from the
    // libraries' records instead.
    let mut inputs: Vec<PathBuf> = module
        .dependencies()
        .into_iter()
        .chain(
            extra[extra.len() - unit.links.len()..]
                .iter()
                .flat_map(super::slang::Module::dependencies),
        )
        .map(|p| canonical_lossy(&p))
        .filter(|p| {
            settings.precompiled.is_empty()
                || p == unit.path
                || unit.links.iter().any(|l| p == &l.path)
                || settings.roots.iter().any(|r| p.starts_with(&r.dir))
        })
        .chain(settings.precompiled.inputs(session))
        .collect();
    inputs.sort();
    inputs.dedup();
    Ok(Compiled {
        words,
        debug,
        model,
        inputs,
        precompiled: settings.precompiled.used(session),
    })
}

/// The modules composed with a kernel besides its own: `ASSERTS = true` when asserts are
/// on, and the job's link modules.
fn link_modules(
    session: &Session,
    settings: &Settings<'_>,
    links: &[LinkModule],
) -> Result<Vec<super::slang::Module>, String> {
    let mut out = Vec::new();
    if settings.asserts {
        out.push(session.load(
            "orr_asserts",
            Path::new("orr_asserts.slang"),
            "export static const bool ASSERTS = true;\n",
        )?);
    }
    for l in links {
        out.push(session.load(&l.module, &l.path, &l.source)?);
    }
    Ok(out)
}

/// Loads the kernel at `path` once more into the Metal `session`, compiles its non-raygen
/// entries to MSL and returns what the MSL check finds. Writes `<name>.metal`.
fn metal_issues(
    session: &Session,
    settings: &Settings<'_>,
    model: &Model,
    job: &Job,
    source: &str,
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
        let mut sources = vec![(job.path.as_path(), text.as_ref())];
        sources.extend(
            job.links
                .iter()
                .map(|l| (l.path.as_path(), l.source.as_str())),
        );
        roots::load_prefixed(session, settings.roots, &sources, |_| false)?;
        let module = session.load(&job.name.replace('/', "."), &job.path, &text)?;
        let extra = link_modules(session, settings, &job.links)?;
        let program = session.link(&module, &extra, |e| entries.contains(e))?;
        Ok::<_, String>((program.text()?, program))
    })();
    let (msl, program) = match compiled {
        Ok(r) => r,
        Err(diagnostics) => return Ok(vec![format!("Metal emission failed:\n{diagnostics}")]),
    };
    write(&job.spv.with_extension("metal"), msl.as_bytes())?;
    let mut issues = metal::msl_issues(&msl);
    match program.layout() {
        Ok(layout) => issues.extend(metal::layout_issues(&session.lib.api, layout, model)),
        Err(diagnostics) => issues.push(format!("no Metal layout: {diagnostics}")),
    }
    Ok(issues)
}

/// The [`Lint::Micromap`] message for `entry`.
pub(super) fn no_omm_mode(entry: &str) -> String {
    format!(
        "entry point {entry} traces rays without the opacity-micromap mode; begin it with `ORR_OMM({entry});` from orr_root.h"
    )
}

/// The interface of a module's entry points: the root's layout, through pointees, and the
/// specialization constants' ids and types. Names do not count.
pub(super) fn interface_hash(model: &Model) -> u64 {
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
    let mut text = format!("inline {}|", model.inline_root);
    if let Some(root) = &model.root {
        walk(&Ty::Struct(root.clone()), model, &mut Vec::new(), &mut text);
    }
    for s in &model.spec {
        text.push_str(&format!("|{}:{:?}", s.id, s.scalar));
    }
    hash64(text.as_bytes())
}

pub(super) fn hash64(bytes: &[u8]) -> u64 {
    let h = blake3::hash(bytes);
    u64::from_le_bytes(h.as_bytes()[..8].try_into().expect("8 bytes"))
}

/// `dir`, else `OUT_DIR`.
pub(super) fn out_dir(dir: Option<&Path>) -> Result<PathBuf, Error> {
    match dir {
        Some(d) => Ok(d.to_path_buf()),
        None => env::var_os("OUT_DIR")
            .map(PathBuf::from)
            .ok_or(Error::NoOutDir),
    }
}

pub(super) fn write(path: &Path, bytes: &[u8]) -> Result<(), Error> {
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

pub(super) fn canonical(path: &Path) -> Result<PathBuf, Error> {
    path.canonicalize()
        .map(strip_verbatim)
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })
}

/// A `types.json`: Slang struct name to Rust path.
fn read_types(path: &Path) -> Result<BTreeMap<String, String>, Error> {
    serde_json::from_str(&read(path)?).map_err(|e| Error::Manifest {
        path: path.to_path_buf(),
        message: format!("not a types file: {e}"),
    })
}

fn write_types(path: &Path, types: &BTreeMap<String, String>) -> Result<(), Error> {
    write(
        path,
        serde_json::to_string_pretty(types)
            .expect("types serialize")
            .as_bytes(),
    )
}

/// Publishes the union of what the direct dependencies with the `links` values `deps` publish,
/// for a crate that re-exports them, so its dependents read one dependency. Each `(from, to)`
/// of `moved` rewrites a type's Rust path under `from` (`::dep`) to `to` (`::my_crate::dep`).
pub fn publish_union(deps: &[&str], moved: &[(&str, &str)]) -> Result<(), Error> {
    let mut roots: Vec<Root> = Vec::new();
    let mut points = Vec::new();
    let mut types = BTreeMap::new();
    let required = |links: &str, what: &str| {
        let var = dep_var(links, what);
        env::var(&var).map_err(|_| Error::NoDep { var })
    };
    for &dep in deps {
        for root in roots::parse_list(&required(dep, "SHADER_DIRS")?) {
            if !roots.contains(&root) {
                roots.push(root);
            }
        }
        points.push(required(dep, "LINK_POINTS")?);
        let list = required(dep, "TYPES")?;
        for path in list.split(';').filter(|s| !s.is_empty()).map(Path::new) {
            for (slang, rust) in read_types(path)? {
                let rust = moved
                    .iter()
                    .find_map(|(from, to)| {
                        let rest = rust.strip_prefix(from.trim_end_matches("::"))?;
                        rest.starts_with("::")
                            .then(|| format!("{}{rest}", to.trim_end_matches("::")))
                    })
                    .unwrap_or(rust);
                types.insert(slang, rust);
            }
            #[allow(clippy::print_stdout)]
            {
                println!("cargo::rerun-if-changed={}", path.display());
            }
        }
    }
    let path = out_dir(None)?.join("types.json");
    write_types(&path, &types)?;
    #[allow(clippy::print_stdout)]
    {
        println!("cargo::metadata=shader_dirs={}", roots::format_list(&roots));
        println!("cargo::metadata=link_points={}", points.join(";"));
        println!("cargo::metadata=types={}", path.display());
    }
    Ok(())
}

pub(super) fn read(path: &Path) -> Result<String, Error> {
    fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

pub(super) fn canonical_lossy(path: &Path) -> PathBuf {
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
