//! Compiles orr-gpu's own kernels and publishes its shader root, which holds the `rtas` and
//! `sequence` libraries, and generates the Nsight Perf SDK bindings that counter frames use
//! when the SDK is found.

use std::env;
use std::path::{Path, PathBuf};

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    match Build::new().and_then(|b| b.kernels(&src).compile()) {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
    perf_bindings();
}

fn perf_bindings() {
    println!("cargo::rustc-check-cfg=cfg(nvperf)");
    println!("cargo::rerun-if-env-changed=NSIGHT_PERF_SDK");
    println!("cargo::rerun-if-env-changed=VULKAN_SDK");
    let Some(sdk) = env::var_os("NSIGHT_PERF_SDK")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
    else {
        return;
    };
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        println!(
            "cargo::warning=NSIGHT_PERF_SDK is set, but orr-gpu loads the Perf SDK on Windows only"
        );
        return;
    }
    let find = |name: &str| {
        find_file(&sdk, name)
            .unwrap_or_else(|| panic!("NSIGHT_PERF_SDK is {} but holds no {name}", sdk.display()))
    };
    let library = find("nvperf_grfx_host.dll");
    let include = find("nvperf_host.h");
    let include = include.parent().unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut builder = bindgen::Builder::default()
        .header_contents(
            "nvperf.h",
            "#include <vulkan/vulkan.h>\n#include <nvperf_host.h>\n#include <nvperf_target.h>\n\
             #include <nvperf_vulkan_host.h>\n#include <nvperf_vulkan_target.h>\n",
        )
        .clang_arg(format!("-I{}", include.display()))
        .allowlist_type("NV(PW|PA)_.*")
        .allowlist_var("NV(PW|PA)_.*")
        .layout_tests(false);
    if let Some(vulkan) = env::var_os("VULKAN_SDK").map(PathBuf::from) {
        builder = builder.clang_arg(format!("-I{}", vulkan.join("Include").display()));
    }
    builder
        .generate()
        .expect("Nsight Perf SDK bindings (needs clang and the Vulkan SDK headers at VULKAN_SDK)")
        .write_to_file(out.join("nvperf.rs"))
        .expect("write Nsight Perf SDK bindings");
    println!("cargo::rustc-env=ORR_NVPERF_LIBRARY={}", library.display());
    println!("cargo::rustc-cfg=nvperf");
}

/// The first file called `name` under `dir`, depth first.
fn find_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).ok()?.flatten().collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    entries.iter().find_map(|e| {
        let path = e.path();
        if path.is_dir() {
            find_file(&path, name)
        } else {
            (e.file_name() == name).then_some(path)
        }
    })
}
