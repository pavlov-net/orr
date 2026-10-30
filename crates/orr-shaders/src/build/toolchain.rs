use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::Error;

/// The Slang release orr builds with. The check is exact: precompiled Slang modules load
/// only in the compiler version that wrote them.
pub const SLANG_VERSION: &str = "2026.20";

/// The compiler options every kernel is built with. Per-file arguments (input, include
/// roots, linked modules, debug info, outputs) follow them.
pub const SLANGC_OPTIONS: &[&str] = &[
    "-target",
    "spirv",
    "-profile",
    "spirv_1_6",
    "-emit-spirv-directly",
    "-capability",
    "spvDescriptorHeapEXT+spvRayQueryKHR+spvMeshShadingEXT+spvShaderInvocationReorderEXT+spvAbort",
    "-fvk-use-c-layout",
    "-matrix-layout-row-major",
    "-fvk-use-entrypoint-name",
    "-warnings-as-errors",
    "all",
];

/// A located, version-checked Slang release.
#[derive(Clone, Debug)]
pub struct Toolchain {
    slangc: PathBuf,
}

impl Toolchain {
    /// Finds slangc and checks that it is exactly [`SLANG_VERSION`].
    ///
    /// `ORR_SLANGC` (absolute path to the executable) wins when set. Otherwise the
    /// directories from `CARGO_MANIFEST_DIR` upward are searched for
    /// `.tools/slang-2026.20/bin/slangc`, where `tools/fetch-slang.sh` puts it. Build scripts
    /// should print `cargo:rerun-if-env-changed=ORR_SLANGC`.
    pub fn locate() -> Result<Self, Error> {
        let slangc = find(
            env::var_os("ORR_SLANGC"),
            env::var_os("CARGO_MANIFEST_DIR").map(PathBuf::from),
        )?;
        let output = Command::new(&slangc)
            .arg("-version")
            .output()
            .map_err(|source| Error::SlangcSpawn {
                path: slangc.clone(),
                source,
            })?;
        // slangc prints its version to stderr.
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        check_version(&slangc, &text)?;
        Ok(Self { slangc })
    }

    /// The slangc executable.
    pub fn slangc(&self) -> &Path {
        &self.slangc
    }

    /// The release directory: `bin/`, `lib/` and `include/` sit under it.
    pub fn release_dir(&self) -> &Path {
        self.slangc
            .parent()
            .and_then(Path::parent)
            .expect("slangc lies in the release's bin directory")
    }

    /// A slangc command with [`SLANGC_OPTIONS`] already applied.
    pub fn command(&self) -> Command {
        let mut cmd = Command::new(&self.slangc);
        cmd.args(SLANGC_OPTIONS);
        cmd
    }
}

/// The slangc path under a tools directory.
pub(super) fn tools_slangc() -> PathBuf {
    Path::new(".tools")
        .join(format!("slang-{SLANG_VERSION}"))
        .join("bin")
        .join(format!("slangc{}", env::consts::EXE_SUFFIX))
}

fn find(orr_slangc: Option<OsString>, manifest_dir: Option<PathBuf>) -> Result<PathBuf, Error> {
    if let Some(path) = orr_slangc.filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        return if path.is_absolute() && path.is_file() {
            Ok(path)
        } else {
            Err(Error::SlangcMissing { path })
        };
    }
    let start = manifest_dir.ok_or(Error::NoManifestDir)?;
    let rel = tools_slangc();
    start
        .ancestors()
        .map(|dir| dir.join(&rel))
        .find(|p| p.is_file())
        .ok_or(Error::SlangcNotFound {
            searched_from: start,
        })
}

fn check_version(slangc: &Path, output: &str) -> Result<(), Error> {
    let found = output.trim();
    if found == SLANG_VERSION {
        Ok(())
    } else {
        Err(Error::SlangcVersion {
            path: slangc.to_path_buf(),
            found: found.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_must_match_exactly() {
        let path = Path::new("/sdk/bin/slangc");
        assert!(check_version(path, "2026.20").is_ok());
        assert!(check_version(path, "2026.20\r\n").is_ok());
        for wrong in ["2026.1-52-gc8ddf20bb", "2026.201", "2026.2", ""] {
            let message = check_version(path, wrong).unwrap_err().to_string();
            assert!(message.contains("/sdk/bin/slangc"), "{message}");
            assert!(message.contains(&format!("{wrong:?}")), "{message}");
            assert!(message.contains("exactly 2026.20"), "{message}");
        }
    }
}
