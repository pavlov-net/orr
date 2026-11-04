//! Compiles orr_gpu's own kernels, publishes its shader root, which holds the `rtas` and
//! `sequence` libraries, and the Rust types of the `rtas` records, sets `cfg(orr_omm)` when the
//! shaders are built with micromaps, and generates the Nsight Perf SDK bindings that counter
//! frames use when the SDK is found. Publishes `vulkan` for targets with the Vulkan backend, for
//! crates built on its interop.

use std::env;
use std::path::{Path, PathBuf};

use orr_shaders::build::Build;

/// The `rtas` records, mirrored in `crate::rtas`.
const RTAS_RECORDS: &[&str] = &[
    "ClasBuild",
    "TemplateBuild",
    "Instantiate",
    "ClusterBlas",
    "Move",
    "PtlasWrite",
    "PtlasUpdate",
    "PtlasTranslate",
    "PtlasCommand",
];

fn main() {
    if env::var("CARGO_CFG_TARGET_VENDOR").as_deref() != Ok("apple") {
        println!("cargo::metadata=vulkan=1");
    }
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    println!("cargo::rustc-check-cfg=cfg(orr_omm)");
    let built = Build::new().and_then(|b| {
        RTAS_RECORDS
            .iter()
            .fold(b.kernels(&src), |b, t| {
                b.extern_type(t, &format!("crate::rtas::{t}"))
            })
            .extern_type("ClassicInstance", "crate::ClassicInstance")
            .publish_types("::orr_gpu")
            .compile()
    });
    match built {
        Ok(out) => {
            if out.omm {
                println!("cargo::rustc-cfg=orr_omm");
            }
        }
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
            "cargo::warning=NSIGHT_PERF_SDK is set, but orr_gpu loads the Perf SDK on Windows only"
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
