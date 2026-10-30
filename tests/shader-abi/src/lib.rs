//! Kernels over `orr.abi`, `orr.trace`, `orr.coherence` and `orr.atomic`, built with the MSL
//! check: the build passing is the check. The kernels under `asserts` link `ASSERTS = true`.
//! The tests under `tests/` run them on the GPU against CPU references.

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

pub mod assert_shaders {
    include!(concat!(env!("OUT_DIR"), "/asserts/orr_shaders.rs"));
}

#[cfg(test)]
mod tests {
    use core::mem::{offset_of, size_of};
    use std::path::Path;
    use std::process::Command;

    use super::{assert_shaders, shaders};

    const OP_ABORT: u32 = 5121;

    fn aborts(spirv: &[u32]) -> bool {
        let mut at = 5;
        while let Some(&first) = spirv.get(at) {
            if first & 0xffff == OP_ABORT {
                return true;
            }
            at += (first >> 16).max(1) as usize;
        }
        false
    }

    #[test]
    fn asserts_abort_only_when_linked() {
        assert!(aborts(assert_shaders::null_tlas::MODULE.spirv));
        assert!(!aborts(shaders::alpha::MODULE.spirv));
        assert!(!aborts(shaders::clusters::MODULE.spirv));
    }

    /// Every opacity-micromap variant validates under `spirv-val` from `ORR_SPIRV_TOOLS`
    /// (v2026.4 or newer knows the micromap execution mode).
    #[test]
    #[allow(clippy::print_stderr)]
    fn omm_variants_validate() {
        let Some(tools) = std::env::var_os("ORR_SPIRV_TOOLS") else {
            eprintln!("ORR_SPIRV_TOOLS unset; spirv-val not run");
            return;
        };
        let val = Path::new(&tools).join(format!("spirv-val{}", std::env::consts::EXE_SUFFIX));
        let out = Path::new(env!("OUT_DIR"));
        let mut checked = 0;
        for dir in [out.to_path_buf(), out.join("asserts")] {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if !path.to_string_lossy().ends_with(".omm.spv") {
                    continue;
                }
                let run = Command::new(&val)
                    .args(["--target-env", "vulkan1.4", "--scalar-block-layout"])
                    .arg(&path)
                    .output()
                    .unwrap_or_else(|e| panic!("{}: {e}", val.display()));
                assert!(
                    run.status.success(),
                    "spirv-val rejected {}:\n{}{}",
                    path.display(),
                    String::from_utf8_lossy(&run.stdout),
                    String::from_utf8_lossy(&run.stderr)
                );
                checked += 1;
            }
        }
        assert!(checked >= 4, "{checked} variants found");
    }

    #[test]
    fn records_keep_c_layout() {
        use shaders::clusters::ClusterHit;
        assert_eq!(offset_of!(ClusterHit, positions), 16);
        assert_eq!(size_of::<ClusterHit>(), 52);
    }
}
