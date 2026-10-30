// Shared by the crate's build.rs (through `#[path]`) and `Build::new`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The Slang release orr builds with. The check is exact: precompiled Slang modules load
/// only in the compiler version that wrote them.
pub const SLANG_VERSION: &str = "2026.20";

pub(crate) enum Missing {
    /// `ORR_SLANGC` is set but is not the absolute path of an existing file.
    BadEnv(PathBuf),
    /// No `.tools` release in or above the start directory.
    NotFound,
}

/// The slangc path under a tools directory.
pub(crate) fn tools_slangc() -> PathBuf {
    Path::new(".tools")
        .join(format!("slang-{SLANG_VERSION}"))
        .join("bin")
        .join(format!("slangc{}", std::env::consts::EXE_SUFFIX))
}

/// `ORR_SLANGC` when set, else `.tools/slang-<version>/bin/slangc` in or above `start`.
pub(crate) fn find_slangc(orr_slangc: Option<OsString>, start: &Path) -> Result<PathBuf, Missing> {
    if let Some(path) = orr_slangc.filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        return if path.is_absolute() && path.is_file() {
            Ok(path)
        } else {
            Err(Missing::BadEnv(path))
        };
    }
    let rel = tools_slangc();
    start
        .ancestors()
        .map(|dir| dir.join(&rel))
        .find(|p| p.is_file())
        .ok_or(Missing::NotFound)
}

/// The release directory of a slangc executable: `bin/`, `lib/` and `include/` sit under it.
pub(crate) fn release_dir(slangc: &Path) -> &Path {
    slangc
        .parent()
        .and_then(Path::parent)
        .expect("slangc lies in the release's bin directory")
}

/// adobe/openpbr-bsdf: `ORR_OPENPBR_BSDF` when set, else `.tools/openpbr-bsdf` in or above
/// `start`. `None` when neither holds `openpbr.h`.
pub(crate) fn find_openpbr(orr_openpbr_bsdf: Option<OsString>, start: &Path) -> Option<PathBuf> {
    if let Some(dir) = orr_openpbr_bsdf.filter(|d| !d.is_empty()) {
        let dir = PathBuf::from(dir);
        return dir.join("openpbr.h").is_file().then_some(dir);
    }
    start
        .ancestors()
        .map(|d| d.join(".tools").join("openpbr-bsdf"))
        .find(|d| d.join("openpbr.h").is_file())
}
