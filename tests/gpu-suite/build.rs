//! Builds the suite's kernels, and the entry-parity kernels with precise floating point. The
//! kernels are Vulkan-only, so the MSL check stays off whatever `ORR_METAL_CHECK` says.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, DebugInfo};

/// The `rtas` records the kernels use, mirrored in `orr_gpu::rtas`.
const RTAS_RECORDS: &[&str] = &[
    "ClasBuild",
    "TemplateBuild",
    "Instantiate",
    "ClusterBlas",
    "Move",
    "PtlasWrite",
    "PtlasUpdate",
    "PtlasTranslate",
    "PtlasCommand",
];

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let gpu = env::var("DEP_ORR_GPU_SHADER_DIRS").expect("orr-gpu shader roots");
    let built = Build::new()
        .and_then(|b| {
            let mut b = b
                .kernels(dir.join("src"))
                .roots(&gpu)
                .debug_info(DebugInfo::Embedded)
                .metal_check(false);
            for t in RTAS_RECORDS {
                b = b.extern_type(t, &format!("::orr_gpu::rtas::{t}"));
            }
            b.extern_type("ClassicInstance", "::orr_gpu::ClassicInstance")
                .compile()
        })
        .and_then(|_| {
            Build::new()?
                .kernels(dir.join("parity"))
                .fp_precise(true)
                .metal_check(false)
                .out_dir(out.join("parity"))
                .compile()
        });
    if let Err(e) = built {
        panic!("{e}");
    }
}
