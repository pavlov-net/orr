use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};

use super::Error;
use super::locate::{self, Missing, SLANG_VERSION};
use super::slang::Library;

/// The compiler options every kernel is built with, in slangc's command-line form. The
/// build hands them to the Slang library unchanged; per-module options (include roots,
/// debug info, floating-point mode) follow them.
pub const SLANGC_OPTIONS: &[&str] = &[
    "-target",
    "spirv",
    "-profile",
    "spirv_1_6",
    "-emit-spirv-directly",
    "-capability",
    "spvDescriptorHeapEXT+spvRayQueryKHR+spvMeshShadingEXT+spvShaderInvocationReorderEXT+spvAbort+spvCooperativeVectorNV+spvRayQueryPositionFetchKHR",
    "-fvk-use-c-layout",
    "-matrix-layout-row-major",
    "-fvk-use-entrypoint-name",
    "-warnings-as-errors",
    "all",
];

/// Finds the release, loads its Slang library and checks that it is exactly
/// [`SLANG_VERSION`].
///
/// `ORR_SLANGC` (absolute path to the slangc executable) wins when set. Otherwise the
/// directories from `CARGO_MANIFEST_DIR` upward are searched for
/// `.tools/slang-2026.20/bin/slangc`, where `tools/fetch-slang.sh` puts it. The library is
/// loaded from the same release: `bin/` on Windows, `lib/` elsewhere.
pub(crate) fn load() -> Result<Arc<Library>, Error> {
    let start = env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or(Error::NoManifestDir)?;
    let slangc = locate::find_slangc(env::var_os("ORR_SLANGC"), &start).map_err(|m| match m {
        Missing::BadEnv(path) => Error::SlangcMissing { path },
        Missing::NotFound => Error::SlangcNotFound {
            searched_from: start,
        },
    })?;
    enable_spirv_validation();
    let library = library_path(locate::release_dir(&slangc));
    let lib = Library::load(&library).map_err(|source| Error::LibraryLoad {
        path: library.clone(),
        source,
    })?;
    check_version(&library, &lib.build_tag())?;
    Ok(lib)
}

/// Slang runs its bundled SPIR-V validator only when `SLANG_RUN_SPIRV_VALIDATION` is set.
fn enable_spirv_validation() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if env::var_os(VALIDATION_VAR).is_none_or(|v| v != "1") {
            // SAFETY: runs once, before any Slang library in this process compiles, and
            // build scripts read no environment concurrently with locating the toolchain.
            unsafe { env::set_var(VALIDATION_VAR, "1") };
        }
    });
}

const VALIDATION_VAR: &str = "SLANG_RUN_SPIRV_VALIDATION";

fn library_path(release: &Path) -> PathBuf {
    if cfg!(windows) {
        release.join("bin").join("slang.dll")
    } else if cfg!(target_os = "macos") {
        release.join("lib").join("libslang.dylib")
    } else {
        release.join("lib").join("libslang.so")
    }
}

fn check_version(library: &Path, found: &str) -> Result<(), Error> {
    let found = found.trim();
    if found == SLANG_VERSION {
        Ok(())
    } else {
        Err(Error::SlangVersion {
            path: library.to_path_buf(),
            found: found.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_must_match_exactly() {
        let path = Path::new("/sdk/bin/slang.dll");
        assert!(check_version(path, "2026.20").is_ok());
        assert!(check_version(path, "2026.20\r\n").is_ok());
        for wrong in ["2026.1-52-gc8ddf20bb", "2026.201", "2026.2", ""] {
            let message = check_version(path, wrong).unwrap_err().to_string();
            assert!(message.contains("/sdk/bin/slang.dll"), "{message}");
            assert!(message.contains(&format!("{wrong:?}")), "{message}");
            assert!(message.contains("exactly 2026.20"), "{message}");
        }
    }
}
