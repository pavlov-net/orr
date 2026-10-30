//! Emits C++ from the Slang modules with the pinned slangc (`ORR_SLANGC`, default
//! `.tools/slang-2026.20/bin/slangc[.exe]`), builds it and the OpenPBR C++ reference with `cc`.
//! `ORR_TOOLS` overrides the `.tools` directory (openpbr-bsdf, glm).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let tools = std::env::var("ORR_TOOLS").map(PathBuf::from).unwrap_or_else(|_| manifest.join("../../../../.tools"));
    let exe = if cfg!(windows) { "slangc.exe" } else { "slangc" };
    let slangc = std::env::var("ORR_SLANGC").map(PathBuf::from).unwrap_or_else(|_| tools.join("slang-2026.20/bin").join(exe));
    let slang_include = slangc.parent().unwrap().join("../include");
    let openpbr = tools.join("openpbr-bsdf");
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=ref.cpp");
    println!("cargo:rerun-if-env-changed=ORR_SLANGC");
    println!("cargo:rerun-if-env-changed=ORR_TOOLS");

    let version = Command::new(&slangc).arg("-version").output().unwrap_or_else(|e| panic!("{}: {e}", slangc.display()));
    let v = String::from_utf8_lossy(&version.stdout).to_string() + &String::from_utf8_lossy(&version.stderr);
    assert!(v.trim().starts_with("2026.20"), "slangc -version: {v:?}");

    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").include(&slang_include).warnings(false);
    for (file, extra) in [("pack.slang", vec![]), ("openpbr_wrap.slang", vec!["-I".to_string(), openpbr.display().to_string()])] {
        let dst = out.join(file.replace(".slang", ".cpp"));
        let status = Command::new(&slangc)
            .arg(manifest.join("shaders").join(file))
            .args(["-target", "cpp", "-o"])
            .arg(&dst)
            .args(&extra)
            .status()
            .unwrap();
        assert!(status.success(), "slangc -target cpp {file}");
        build.file(&dst);
    }
    build.compile("slang_cpu");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&openpbr)
        .include(tools.join("glm"))
        .warnings(false)
        .file(manifest.join("ref.cpp"))
        .compile("openpbr_ref");
}
