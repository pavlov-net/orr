//! Compiles orr-lighting's `bsdf`, with the orr-scene modules it imports, for the CPU through
//! `slang/bsdf_cpu.slang` (array LUTs), and for SPIR-V and Metal through
//! `gpu/bsdf_cases.slang` (heap LUTs). The harness's `Inputs` come from `tests/openpbr-ref/gpu`.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, CpuLibrary};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let lighting = manifest.join("../../crates/orr-lighting/src");
    let scene = manifest.join("../../crates/orr-scene/src");
    let inputs = manifest.join("../openpbr-ref/gpu");
    let result = CpuLibrary::new()
        .and_then(|b| {
            b.source("slang/bsdf_cpu.slang")
                .include(&lighting)
                .include(&scene)
                .include(&inputs)
                .include(manifest.join("gpu"))
                .define("ORR_OPENPBR_ARRAY_LUTS", "1")
                .compile("lighting_bsdf")
        })
        .and_then(|()| Build::new())
        .and_then(|b| {
            b.kernels(manifest.join("gpu"))
                .include(&lighting)
                .include(&scene)
                .include(&inputs)
                .extern_type("Inputs", "::openpbr_ref::Inputs")
                .metal_check(true)
                .compile()
        });
    if let Err(e) = result {
        panic!("{e}");
    }
}
