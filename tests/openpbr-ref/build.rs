//! Builds the two CPU BSDFs the harness compares, `orr.openpbr` through Slang's C++ target
//! (`slang/openpbr_cpu.slang`, array LUTs) and adobe/openpbr-bsdf's own C++ with GLM
//! (`cpp/reference.cpp`), and compiles the GPU kernels under `gpu/` to SPIR-V.
//!
//! GLM comes from `ORR_GLM`, else `.tools/glm` in or above the crate.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, CpuLibrary, openpbr_dir};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let fail = |e: orr_shaders::build::Error| -> ! { panic!("{e}") };
    let openpbr = openpbr_dir().unwrap_or_else(|e| fail(e));
    let glm = glm_dir(&manifest);

    CpuLibrary::new()
        .and_then(|b| {
            b.source("slang/openpbr_cpu.slang")
                .include(manifest.join("gpu"))
                .define("ORR_OPENPBR_ARRAY_LUTS", "1")
                .compile("openpbr_slang")
        })
        .unwrap_or_else(|e| fail(e));

    println!("cargo:rerun-if-changed=cpp/reference.cpp");
    println!("cargo:rerun-if-env-changed=ORR_GLM");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&openpbr)
        .include(&glm)
        .warnings(false)
        .file("cpp/reference.cpp")
        .compile("openpbr_ref");

    Build::new()
        .and_then(|b| b.kernels(manifest.join("gpu")).compile())
        .unwrap_or_else(|e| fail(e));
}

fn glm_dir(manifest: &std::path::Path) -> PathBuf {
    if let Some(dir) = env::var_os("ORR_GLM").filter(|d| !d.is_empty()) {
        return PathBuf::from(dir);
    }
    manifest
        .ancestors()
        .map(|d| d.join(".tools").join("glm"))
        .find(|d| d.join("glm").join("glm.hpp").is_file())
        .unwrap_or_else(|| {
            panic!(
                "GLM not found: run tools/fetch-slang.sh, or set ORR_GLM to a GLM 1.0.1 checkout"
            )
        })
}
