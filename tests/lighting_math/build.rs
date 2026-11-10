//! Compiles orr_lighting's `bsdf`, with the orr_scene and orr_pack modules it imports, for
//! the CPU through `slang/bsdf_cpu.slang` (array LUTs), and for SPIR-V through
//! `gpu/bsdf_cases.slang` (heap LUTs). The harness's `Inputs` come from `tests/openpbr_ref/gpu`.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, CpuLibrary, fetched};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let lighting = manifest.join("../../crates/orr_lighting/src");
    let scene = manifest.join("../../crates/orr_scene/src");
    let pack = manifest.join("../../crates/orr_pack/src");
    let inputs = manifest.join("../openpbr_ref/gpu");
    let openpbr = fetched("ORR_OPENPBR_BSDF", "openpbr-bsdf").unwrap_or_else(|e| panic!("{e}"));
    let result = CpuLibrary::new()
        .and_then(|b| {
            b.source("slang/bsdf_cpu.slang")
                .include_as("orr_lighting", &lighting)
                .include_as("orr_scene", &scene)
                .include_as("orr_pack", &pack)
                .include(&openpbr)
                .include_as("openpbr_ref", &inputs)
                .include_as("lighting_math", manifest.join("gpu"))
                .define("ORR_OPENPBR_ARRAY_LUTS", "1")
                .define("ORR_UNSPECIALIZED", "1")
                .compile("lighting_bsdf")
        })
        .and_then(|()| Build::new())
        .and_then(|b| {
            b.kernels(manifest.join("gpu"))
                .include_as("orr_lighting", &lighting)
                .include_as("orr_scene", &scene)
                .include_as("orr_pack", &pack)
                .include(&openpbr)
                .include_as("openpbr_ref", &inputs)
                .extern_type("Inputs", "::openpbr_ref::Inputs")
                .compile()
        });
    if let Err(e) = result {
        panic!("{e}");
    }
}
