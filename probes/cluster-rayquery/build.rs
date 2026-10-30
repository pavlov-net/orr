//! Compiles the probe shaders with the pinned standalone Slang (`ORR_SLANGC`, default
//! `.tools/slang-2026.20`) and the Vulkan SDK's glslangValidator, validating each with spirv-val.

use std::path::{Path, PathBuf};
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

    let slang = |file: &str, defines: &[&str], name: &str| {
        let dst = out.join(format!("{name}.spv"));
        let mut cmd = Command::new(&slangc);
        cmd.arg(shaders.join(file))
            .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly"])
            .args(["-fvk-use-entrypoint-name", "-fvk-use-c-layout", "-matrix-layout-row-major"])
            .args(["-entry", "main", "-stage", "compute", "-o"])
            .arg(&dst);
        for d in defines {
            cmd.arg(format!("-D{d}"));
        }
        run(&mut cmd);
        validate(&sdk_bin, &dst);
    };
    slang("args.slang", &[], "args");
    slang("trace.slang", &["WITH_CLUSTER=1"], "trace_cluster");
    slang("trace.slang", &[], "trace_plain");

    let dst = out.join("trace_glsl.spv");
    run(Command::new(sdk_bin.join("glslangValidator.exe"))
        .args(["-V", "--target-env", "vulkan1.4"])
        .arg(shaders.join("trace.comp"))
        .arg("-o")
        .arg(&dst));
    validate(&sdk_bin, &dst);
}

fn validate(sdk_bin: &Path, spv: &Path) {
    run(Command::new(sdk_bin.join("spirv-val.exe"))
        .args(["--target-env", "vulkan1.4", "--scalar-block-layout"])
        .arg(spv));
}
