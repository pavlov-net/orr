//! Generates NGX bindings from the DLSS SDK at `DLSS_SDK` and links NGX's static library for
//! the target's C runtime. With the `mock` feature, or without `DLSS_SDK`, nothing is linked.
//! Publishes `built`, so a dependent's build script can tell that Cargo resolved this crate.
//! Sets `cfg(vulkan)` where orr_gpu has its Vulkan backend; elsewhere the crate is empty.

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(ngx)");
    println!("cargo::rustc-check-cfg=cfg(vulkan)");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=wrapper.h");
    println!("cargo::rerun-if-env-changed=DLSS_SDK");
    println!("cargo::rerun-if-env-changed=VULKAN_SDK");
    if std::env::var_os("DEP_ORR_GPU_VULKAN").is_none() {
        return;
    }
    println!("cargo::rustc-cfg=vulkan");
    println!("cargo::metadata=built=1");
    if std::env::var_os("CARGO_FEATURE_MOCK").is_some() {
        return;
    }
    let Some(sdk) = std::env::var_os("DLSS_SDK")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
    else {
        println!(
            "cargo::warning=DLSS_SDK is not set: orr_dlss builds without NGX and reports DLSS unavailable"
        );
        return;
    };
    assert!(
        std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows"),
        "orr_dlss links NGX on Windows only; build with the `mock` feature elsewhere"
    );
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());

    let static_crt = std::env::var("CARGO_CFG_TARGET_FEATURE")
        .is_ok_and(|f| f.split(',').any(|f| f == "crt-static"));
    let lib = if static_crt {
        "nvsdk_ngx_s"
    } else {
        "nvsdk_ngx_d"
    };
    println!(
        "cargo::rustc-link-search=native={}",
        sdk.join("lib/Windows_x86_64/x64").display()
    );
    println!("cargo::rustc-link-lib=static={lib}");
    println!("cargo::rustc-link-lib=advapi32");
    println!("cargo::rustc-link-lib=user32");
    println!("cargo::rustc-cfg=ngx");

    let dlls = if std::env::var_os("CARGO_FEATURE_DEBUG_OVERLAY").is_some() {
        "dev"
    } else {
        "rel"
    };
    println!(
        "cargo::rustc-env=ORR_DLSS_BUILD_DLLS={}",
        sdk.join("lib/Windows_x86_64").join(dlls).display()
    );

    let mut builder = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", sdk.join("include").display()))
        .allowlist_function("NVSDK_NGX_.*")
        .allowlist_type("NVSDK_NGX_.*")
        .allowlist_var("NVSDK_NGX_.*")
        .layout_tests(false);
    if let Some(include) = vulkan_include() {
        builder = builder.clang_arg(format!("-I{}", include.display()));
    }
    builder
        .generate()
        .expect("NGX bindings (needs clang and the Vulkan SDK headers at VULKAN_SDK)")
        .write_to_file(out.join("ngx.rs"))
        .expect("write NGX bindings");
}

/// The Windows SDK names it `Include`; the Linux SDK and Vulkan-Headers name it `include`.
fn vulkan_include() -> Option<PathBuf> {
    let sdk = PathBuf::from(std::env::var_os("VULKAN_SDK").filter(|s| !s.is_empty())?);
    ["Include", "include"]
        .into_iter()
        .map(|d| sdk.join(d))
        .find(|d| d.is_dir())
}
