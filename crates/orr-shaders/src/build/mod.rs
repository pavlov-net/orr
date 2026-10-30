//! Build-script support: compiles a crate's Slang kernels with the pinned Slang release in
//! process, checks layouts and lints, and generates Rust.

mod codegen;
mod compile;
mod layout;
mod lint;
mod locate;
mod model;
mod slang;
mod spirv;
mod sys;
mod toolchain;

pub use compile::{Build, Output};
pub use lint::{Lint, LintError};
pub use locate::SLANG_VERSION;
pub use toolchain::SLANGC_OPTIONS;

use std::io;
use std::path::PathBuf;

/// A build failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `ORR_SLANGC` is unset and no `.tools` directory above the crate holds the release.
    #[error(
        "Slang {SLANG_VERSION} not found: no {} in or above {}; run tools/fetch-slang.sh or set ORR_SLANGC to the slangc executable",
        locate::tools_slangc().display(),
        searched_from.display()
    )]
    SlangcNotFound { searched_from: PathBuf },
    /// `ORR_SLANGC` is not the absolute path of an existing file.
    #[error("ORR_SLANGC is {}, which is not the absolute path of an existing file", path.display())]
    SlangcMissing { path: PathBuf },
    /// The release's Slang library could not be loaded.
    #[error("cannot load the Slang library at {}: {source}", path.display())]
    LibraryLoad {
        path: PathBuf,
        source: libloading::Error,
    },
    /// The Slang library reports a version other than [`SLANG_VERSION`].
    #[error(
        "Slang library at {} is version {found:?}; orr needs exactly {SLANG_VERSION} (run tools/fetch-slang.sh or point ORR_SLANGC at a {SLANG_VERSION} release)",
        path.display()
    )]
    SlangVersion { path: PathBuf, found: String },
    /// The build runs outside cargo: `CARGO_MANIFEST_DIR` is unset.
    #[error(
        "CARGO_MANIFEST_DIR is unset; orr_shaders::build runs from a cargo build script or test"
    )]
    NoManifestDir,
    /// `OUT_DIR` is unset and [`Build::out_dir`] was not called.
    #[error("OUT_DIR is unset; run from a build script or call Build::out_dir")]
    NoOutDir,
    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io { path: PathBuf, source: io::Error },
    /// The Slang library failed outside compiling source.
    #[error("{module}: {message}")]
    Slang { module: String, message: String },
    /// Slang rejected the module; warnings count as errors.
    #[error("{module} failed to compile:\n{diagnostics}")]
    Compile { module: String, diagnostics: String },
    /// The module uses a type the build cannot map to Rust, or reflection and SPIR-V
    /// disagree about it.
    #[error("{module}: {message}")]
    Reflection { module: String, message: String },
    /// Generated Rust cannot reproduce the SPIR-V layout.
    #[error("{module}: layout mismatch in struct {structure}, field {field}: {message}")]
    Layout {
        module: String,
        structure: String,
        field: String,
        message: String,
    },
    /// Two modules of one crate define the same struct name with different layouts.
    #[error("struct {name} differs between modules {first} and {second}")]
    StructConflict {
        name: String,
        first: String,
        second: String,
    },
    /// One or more lints failed.
    #[error("{}", lint_lines(.0))]
    Lints(Vec<LintError>),
}

fn lint_lines(lints: &[LintError]) -> String {
    lints
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
}
