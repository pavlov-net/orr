//! Compiles Vulkan Memory Allocator's implementation and generates bindings for its `vma*`
//! API. The header `vk_mem_alloc.h` is found in `ORR_VMA` when set, else in
//! `$VULKAN_SDK/Include/vma`, else in `vma/include` under `ORR_TOOLS` or the nearest `.tools`
//! in or above the crate, where `tools/fetch-slang.sh` puts it. The Vulkan headers come from
//! the Vulkan SDK at `VULKAN_SDK`, or the system include path without it.

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
    let sdk = env::var_os("VULKAN_SDK")
        .filter(|d| !d.is_empty())
        .map(|d| PathBuf::from(d).join("Include").join("vma"));
    let tools = env::var_os("ORR_TOOLS")
        .filter(|d| !d.is_empty())
        .map(|d| PathBuf::from(d).join("vma").join("include"));
    let nearest = manifest
        .ancestors()
        .map(|d| d.join(".tools").join("vma").join("include"));
    sdk.into_iter()
        .chain(tools)
        .chain(nearest)
        .find(|d| d.join(HEADER).is_file())
        .unwrap_or_else(|| {
            panic!(
                "{HEADER} not found in ORR_VMA, $VULKAN_SDK/Include/vma or .tools/vma/include; run tools/fetch-slang.sh"
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
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let vma = vma_dir(&manifest);
    println!("cargo::rerun-if-changed={}", vma.join(HEADER).display());
    let vulkan = vulkan_include();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(manifest.join("src/vma.cpp"))
        .include(&vma)
        .include(manifest.join("src"))
        .warnings(false);
    if let Some(dir) = &vulkan {
        build.include(dir);
    }
    if env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some() {
        build.define("ORR_MEMORY_CHECKS", None);
    }
    build.compile("orr_vma");

    let mut bindings = bindgen::Builder::default()
        .header(manifest.join("src/vma.h").to_string_lossy())
        .clang_arg(format!("-I{}", vma.display()))
        .allowlist_function("vma.*")
        .allowlist_type("Vma.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .prepend_enum_name(false)
        .derive_default(true)
        .layout_tests(false)
        .rust_edition(bindgen::RustEdition::Edition2024);
    if let Some(dir) = &vulkan {
        bindings = bindings.clang_arg(format!("-I{}", dir.display()));
    }
    bindings
        .generate()
        .expect("VMA bindings (needs libclang and the Vulkan headers)")
        .write_to_file(out.join("vma.rs"))
        .expect("write VMA bindings");
}
