//! Compiles the probe shaders with the pinned standalone Slang (`ORR_SLANGC`, default
//! `.tools/slang-2026.20`), validating each with the Vulkan SDK's spirv-val, and generates NGX
//! bindings from the SDK at `DLSS_SDK` (cfg `ngx`; without it the NGX steps are skipped).

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
    println!("cargo::rustc-check-cfg=cfg(ngx)");

    let slang = |file: &str, entry: &str, stage: &str, caps: &[&str], name: &str| {
        let dst = out.join(format!("{name}.spv"));
        let mut cmd = Command::new(&slangc);
        cmd.arg(shaders.join(file))
            .args(["-target", "spirv", "-profile", "spirv_1_6", "-emit-spirv-directly"])
            .args(["-fvk-use-entrypoint-name", "-fvk-use-c-layout"])
            .args(["-entry", entry, "-stage", stage, "-o"])
            .arg(&dst);
        for cap in caps {
            cmd.args(["-capability", cap]);
        }
        run(&mut cmd);
        validate(&sdk_bin, &dst);
    };
    let mesh = "spvMeshShadingEXT";
    slang("pass.slang", "main", "compute", &[], "pass");
    slang("mesh.slang", "mesh_main", "mesh", &[mesh], "mesh_mesh");
    slang("mesh.slang", "pixel_main", "fragment", &[mesh], "mesh_pixel");
    slang("compose.slang", "main", "compute", &[], "compose");

    if let Ok(sdk) = std::env::var("DLSS_SDK") {
        let sdk = PathBuf::from(sdk);
        println!("cargo:rustc-cfg=ngx");
        println!("cargo:rustc-link-search=native={}", sdk.join("lib/Windows_x86_64/x64").display());
        println!("cargo:rustc-link-lib=static=nvsdk_ngx_d");
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
}

fn validate(sdk_bin: &Path, spv: &Path) {
    run(Command::new(sdk_bin.join("spirv-val.exe"))
        .args(["--target-env", "vulkan1.4", "--scalar-block-layout"])
        .arg(spv));
}
