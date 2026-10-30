//! Compiles the probe shaders with the pinned standalone Slang (`ORR_SLANGC`, default
//! `.tools/slang-2026.20`), validating each with the Vulkan SDK's spirv-val.

use std::path::PathBuf;
use std::process::Command;

fn run(cmd: &mut Command) {
    let status = cmd.status().unwrap_or_else(|e| panic!("failed to spawn {cmd:?}: {e}"));
    assert!(status.success(), "{cmd:?} failed");
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shaders = manifest.join("shaders");
    let slangc = std::env::var("ORR_SLANGC")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("../../.tools/slang-2026.20/bin/slangc.exe"));
    let sdk_bin = PathBuf::from(std::env::var("VULKAN_SDK").expect("VULKAN_SDK")).join("Bin");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=ORR_SLANGC");
    for name in ["frame", "read", "sample", "expand"] {
        let dst = out.join(format!("{name}.spv"));
        run(Command::new(&slangc)
            .arg(shaders.join(format!("{name}.slang")))
            .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly"])
            .args(["-fvk-use-entrypoint-name", "-fvk-use-c-layout"])
            .args(["-entry", "main", "-stage", "compute", "-o"])
            .arg(&dst));
        run(Command::new(sdk_bin.join("spirv-val.exe"))
            .args(["--target-env", "vulkan1.4", "--scalar-block-layout"])
            .arg(&dst));
    }
}
