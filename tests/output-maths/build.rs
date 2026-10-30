//! Compiles orr-output's Slang maths for the CPU through `slang/output.slang`, and for
//! SPIR-V and Metal through `gpu/output.slang`; links Polyphony's GT7 sample as the
//! reference. `GT7_SAMPLE` names its directory, else `.tools/display/gt7` in the nearest
//! directory above the crate that has it (where `tools/fetch-slang.sh` puts it).

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, CpuLibrary};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let output_src = manifest.join("../../crates/orr-output/src");
    let result = CpuLibrary::new()
        .and_then(|b| {
            b.source("slang/output.slang")
                .include(&output_src)
                .compile("output_maths")
        })
        .and_then(|()| Build::new())
        .and_then(|b| {
            b.kernels(manifest.join("gpu"))
                .include(&output_src)
                .extern_type("Gt7Curve", "orr_output::Gt7Curve")
                .metal_check(true)
                .compile()
        });
    if let Err(e) = result {
        panic!("{e}");
    }

    println!("cargo:rerun-if-env-changed=GT7_SAMPLE");
    println!("cargo:rerun-if-changed=cpp/gt7_reference.cpp");
    let sample = env::var_os("GT7_SAMPLE")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            manifest
                .ancestors()
                .map(|d| d.join(".tools/display/gt7"))
                .find(|d| d.join("gt7_tone_mapping.cpp").is_file())
        })
        .unwrap_or_else(|| {
            panic!(
                "GT7 sample not found: run tools/fetch-slang.sh, or set GT7_SAMPLE to the \
                 directory holding gt7_tone_mapping.cpp"
            )
        });
    println!(
        "cargo:rerun-if-changed={}",
        sample.join("gt7_tone_mapping.cpp").display()
    );
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .file("cpp/gt7_reference.cpp")
        .include(&sample)
        .compile("gt7_reference");
}
