//! Compiles the probe shaders with the pinned standalone Slang (`ORR_SLANGC`, default
//! `.tools/slang-2026.20`), validates each with the Vulkan SDK's spirv-val, and generates NGX
//! bindings from the SDK at `DLSS_SDK`.

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
    let vulkan_sdk = PathBuf::from(std::env::var("VULKAN_SDK").expect("VULKAN_SDK"));
    let sdk_bin = vulkan_sdk.join("Bin");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=ORR_SLANGC");
    println!("cargo:rerun-if-env-changed=DLSS_SDK");

    for (file, entry) in [
        ("scene.slang", "inputs_main"),
        ("scene.slang", "reference_main"),
        ("scene.slang", "backbuffer_main"),
        ("scene.slang", "copy_main"),
        ("scene.slang", "rows_main"),
    ] {
        let dst = out.join(format!("{entry}.spv"));
        run(Command::new(&slangc)
            .arg(shaders.join(file))
            .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly"])
            .args(["-fvk-use-entrypoint-name", "-fvk-use-c-layout", "-matrix-layout-row-major"])
            .args(["-capability", "spvDescriptorHeapEXT"])
            .args(["-entry", entry, "-stage", "compute", "-o"])
            .arg(&dst));
        validate(&sdk_bin, &dst);
    }
    let dst = out.join("classic_main.spv");
    run(Command::new(&slangc)
        .arg(shaders.join("classic.slang"))
        .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly", "-fvk-use-entrypoint-name"])
        .args(["-entry", "classic_main", "-stage", "compute", "-o"])
        .arg(&dst));
    validate(&sdk_bin, &dst);

    let sdk = PathBuf::from(std::env::var("DLSS_SDK").expect("DLSS_SDK (absolute path to the DLSS SDK)"));
    println!("cargo:rustc-link-search=native={}", sdk.join("lib/Windows_x86_64/x64").display());
    println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
    println!("cargo:rustc-link-lib=advapi32");
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-env=PROBE_NGX_DLLS={}", sdk.join("lib/Windows_x86_64/rel").display());
    bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", sdk.join("include").display()))
        .clang_arg(format!("-I{}", vulkan_sdk.join("Include").display()))
        .allowlist_function("NVSDK_NGX_.*")
        .allowlist_type("NVSDK_NGX_.*")
        .allowlist_var("NVSDK_NGX_.*")
        .layout_tests(false)
        .generate()
        .expect("NGX bindings")
        .write_to_file(out.join("ngx.rs"))
        .unwrap();
}

fn validate(sdk_bin: &Path, spv: &Path) {
    run(Command::new(sdk_bin.join("spirv-val.exe"))
        .args(["--target-env", "vulkan1.4", "--scalar-block-layout"])
        .arg(spv));
}
