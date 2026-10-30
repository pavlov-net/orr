//! Build-script support: compiling Slang to SPIR-V.

mod toolchain;

pub use toolchain::{SLANG_VERSION, SLANGC_OPTIONS, Toolchain};

use std::io;
use std::path::PathBuf;

/// A build failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `ORR_SLANGC` is unset and no `.tools` directory above the crate holds the release.
    #[error(
        "slangc {SLANG_VERSION} not found: no {} in or above {}; run tools/fetch-slang.sh or set ORR_SLANGC to the slangc executable",
        toolchain::tools_slangc().display(),
        searched_from.display()
    )]
    SlangcNotFound { searched_from: PathBuf },
    /// `ORR_SLANGC` is not the absolute path of an existing file.
    #[error("ORR_SLANGC is {}, which is not the absolute path of an existing file", path.display())]
    SlangcMissing { path: PathBuf },
    /// slangc could not be started.
    #[error("cannot run slangc at {}: {source}", path.display())]
    SlangcSpawn { path: PathBuf, source: io::Error },
    /// slangc reports a version other than [`SLANG_VERSION`].
    #[error(
        "slangc at {} is version {found:?}; orr needs exactly {SLANG_VERSION} (run tools/fetch-slang.sh or point ORR_SLANGC at a {SLANG_VERSION} release)",
        path.display()
    )]
    SlangcVersion { path: PathBuf, found: String },
    /// The build runs outside cargo: `CARGO_MANIFEST_DIR` is unset.
    #[error(
        "CARGO_MANIFEST_DIR is unset; orr_shaders::build runs from a cargo build script or test"
    )]
    NoManifestDir,
}
