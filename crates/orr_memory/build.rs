//! Compiles Vulkan Memory Allocator's implementation. The header `vk_mem_alloc.h` is found in
//! `ORR_VMA` when set, else in `vma/include` under `ORR_TOOLS` or the nearest `.tools` in or
//! above the crate, where `tools/fetch-slang.sh` puts the pinned version, else in
//! `$VULKAN_SDK/Include/vma`. It must be VMA 3.5.0, the version `src/bindings.rs` is generated
//! from. The Vulkan headers come from the Vulkan SDK at `VULKAN_SDK`, or the system include
//! path without it.

use std::env;
use std::path::{Path, PathBuf};

const HEADER: &str = "vk_mem_alloc.h";

fn vma_dir(manifest: &Path) -> PathBuf {
    if let Some(dir) = env::var_os("ORR_VMA").filter(|d| !d.is_empty()) {
        let dir = PathBuf::from(dir);
        assert!(
            dir.join(HEADER).is_file(),
            "ORR_VMA is {}, which holds no {HEADER}",
            dir.display()
        );
        return dir;
    }
    let tools = env::var_os("ORR_TOOLS")
        .filter(|d| !d.is_empty())
        .map(|d| PathBuf::from(d).join("vma").join("include"));
    let nearest = manifest
        .ancestors()
        .map(|d| d.join(".tools").join("vma").join("include"));
    let sdk = env::var_os("VULKAN_SDK")
        .filter(|d| !d.is_empty())
        .map(|d| PathBuf::from(d).join("Include").join("vma"));
    tools
        .into_iter()
        .chain(nearest)
        .chain(sdk)
        .find(|d| d.join(HEADER).is_file())
        .unwrap_or_else(|| {
            panic!(
                "{HEADER} not found in ORR_VMA, .tools/vma/include or $VULKAN_SDK/Include/vma; run tools/fetch-slang.sh"
            )
        })
}

fn vulkan_include() -> Option<PathBuf> {
    let sdk = PathBuf::from(env::var_os("VULKAN_SDK").filter(|d| !d.is_empty())?);
    ["Include", "include"]
        .iter()
        .map(|d| sdk.join(d))
        .find(|d| d.join("vulkan").join("vulkan.h").is_file())
}

fn main() {
    for var in ["ORR_VMA", "VULKAN_SDK", "ORR_TOOLS"] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    println!("cargo::rerun-if-changed=src/vma.h");
    println!("cargo::rerun-if-changed=src/vma.cpp");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let vma = vma_dir(&manifest);
    println!("cargo::rerun-if-changed={}", vma.join(HEADER).display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(manifest.join("src/vma.cpp"))
        .include(&vma)
        .include(manifest.join("src"))
        .warnings(false);
    if let Some(dir) = vulkan_include() {
        build.include(dir);
    }
    if env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some() {
        build.define("ORR_MEMORY_CHECKS", None);
    }
    build.compile("orr_vma");
}
