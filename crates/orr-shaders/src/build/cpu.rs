use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{env, fs};

use super::Error;
use super::compile::{canonical, out_dir, write};
use super::slang::{GlobalSession, Library};

/// Options of the C++ session; include roots follow.
const OPTIONS: &[&str] = &[
    "-target",
    "cpp",
    "-matrix-layout-row-major",
    "-warnings-as-errors",
    "all",
];

/// Compiles Slang modules for the CPU and links them into the crate as a static library,
/// so tests can call shader code. Used from `build.rs`.
///
/// Functions marked `export __extern_cpp` are callable from Rust through `extern "C"` with
/// C types (`float`, `uint32_t`, pointers to `#[repr(C)]` data). Matrices are row-major,
/// as on the GPU. Results differ from the GPU in FMA contraction, denormals and
/// transcendental precision, so CPU tests compare against references and GPU tests compare
/// against the CPU.
///
/// ```no_run
/// # fn main() -> Result<(), orr_shaders::build::Error> {
/// orr_shaders::build::CpuLibrary::new()?
///     .source("slang/packing.slang")
///     .compile("shader_cpu")?;
/// # Ok(())
/// # }
/// ```
pub struct CpuLibrary {
    lib: Arc<Library>,
    sources: Vec<PathBuf>,
    includes: Vec<PathBuf>,
    defines: Vec<String>,
    out_dir: Option<PathBuf>,
}

impl CpuLibrary {
    /// Locates and version-checks the Slang release, like [`Build::new`](super::Build::new).
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            lib: super::toolchain::load()?,
            sources: Vec::new(),
            includes: Vec::new(),
            defines: Vec::new(),
            out_dir: None,
        })
    }

    /// One more module to compile. Relative paths are under `CARGO_MANIFEST_DIR`.
    pub fn source(mut self, path: impl AsRef<Path>) -> Self {
        self.sources.push(path.as_ref().to_path_buf());
        self
    }

    /// One more import root. [`LIB_DIR`](crate::LIB_DIR) is always present, and so is
    /// [`openpbr_dir`](super::openpbr_dir) when it is found.
    pub fn include(mut self, dir: impl AsRef<Path>) -> Self {
        self.includes.push(dir.as_ref().to_path_buf());
        self
    }

    /// Defines a preprocessor macro for every source, as `-D name=value`.
    pub fn define(mut self, name: &str, value: &str) -> Self {
        self.defines.push(format!("{name}={value}"));
        self
    }

    /// Writes the C++ here instead of `OUT_DIR`.
    pub fn out_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.out_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// Emits one C++ file for all sources (Slang's C++ prelude inlined), compiles it into
    /// the static library `name` and links it into the crate. Prints `rerun-if-changed`
    /// lines for every Slang input.
    pub fn compile(self, name: &str) -> Result<(), Error> {
        let manifest = env::var_os("CARGO_MANIFEST_DIR")
            .map(PathBuf::from)
            .ok_or(Error::NoManifestDir)?;
        let out_dir = out_dir(self.out_dir.as_deref())?;
        let mut options: Vec<String> = OPTIONS.iter().map(|s| (*s).to_owned()).collect();
        for dir in self
            .includes
            .iter()
            .map(PathBuf::as_path)
            .chain([Path::new(crate::LIB_DIR)])
        {
            options.extend(["-I".to_owned(), canonical(dir)?.display().to_string()]);
        }
        if let Ok(dir) = super::openpbr_dir() {
            options.extend(["-I".to_owned(), dir.display().to_string()]);
        }
        for define in &self.defines {
            options.extend(["-D".to_owned(), define.clone()]);
        }
        let slang_err = |message: String| Error::Slang {
            module: name.to_owned(),
            message,
        };
        let compile_err = |diagnostics: String| Error::Compile {
            module: name.to_owned(),
            diagnostics,
        };
        let global = GlobalSession::new(Arc::clone(&self.lib)).map_err(slang_err)?;
        let session = global.whole_program_session(&options).map_err(slang_err)?;
        let mut modules = Vec::new();
        let mut inputs = Vec::new();
        for source in &self.sources {
            let path = canonical(&manifest.join(source))?;
            let text = fs::read_to_string(&path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let module = session.load(&stem, &path, &text).map_err(compile_err)?;
            inputs.extend(module.dependencies());
            modules.push(module);
        }
        let Some((first, rest)) = modules.split_first() else {
            return Err(slang_err("no sources".to_owned()));
        };
        let program = session.link(first, rest, |_| false).map_err(compile_err)?;
        let cpp = program.text().map_err(compile_err)?;
        if cpp.contains("KernelContext") {
            return Err(slang_err(
                "the exported functions read global shader parameters (specialisation constants, \
                 resources or uniforms), which the C++ target passes as an extra KernelContext \
                 argument that `extern \"C\"` callers cannot supply"
                    .to_owned(),
            ));
        }
        let cpp_path = out_dir.join(format!("{name}.cpp"));
        write(&cpp_path, cpp.as_bytes())?;

        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .warnings(false)
            .file(&cpp_path)
            .try_compile(name)
            .map_err(|e| Error::Slang {
                module: name.to_owned(),
                message: format!("C++ compile of {} failed: {e}", cpp_path.display()),
            })?;

        inputs.sort();
        inputs.dedup();
        #[allow(clippy::print_stdout)]
        {
            for input in inputs {
                println!("cargo:rerun-if-changed={}", input.display());
            }
            println!("cargo:rerun-if-env-changed=ORR_SLANGC");
            println!("cargo:rerun-if-env-changed=ORR_OPENPBR_BSDF");
        }
        Ok(())
    }
}
