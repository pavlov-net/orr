//! Puts the locked versions of the libraries that shape cooked bytes into environment
//! variables that enter cook keys: `ORR_COOK_ZSTD_LIBS` for every entry,
//! `ORR_COOK_MESH_LIBS` for meshes, `ORR_COOK_TEXTURE_LIBS` for textures and
//! `ORR_COOK_RASTER_LIBS` for raster sets.
//!
//! With the `ntc` feature, compiles the LibNTC shim against the headers of the RTXNTC-Library
//! checkout in `ORR_NTC_SDK` (else `RTXNTC-Library` under `ORR_TOOLS` or the nearest
//! `.tools`) and sets `ORR_COOK_NTC_SDK`, `ORR_COOK_NTC_INTERFACE` (the headers'
//! `InterfaceVersion`) and `ORR_COOK_NTC_SCHEMA` (the NTC file schema version the checkout
//! writes).

use std::path::Path;

const ZSTD: &[&str] = &["zstd", "zstd-safe", "zstd-sys"];
const MESH: &[&str] = &["meshopt", "gltf", "gltf-json", "openusd", "roxmltree"];
const TEXTURE: &[&str] = &["bcdec_rs", "image", "png", "half"];

fn main() {
    ntc();
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let lock = Path::new(&dir)
        .ancestors()
        .map(|d| d.join("Cargo.lock"))
        .find(|p| p.exists());
    let mut locked = Vec::new();
    if let Some(lock) = lock {
        println!("cargo:rerun-if-changed={}", lock.display());
        let text = std::fs::read_to_string(&lock).expect("Cargo.lock");
        let mut name = None;
        for line in text.lines() {
            if let Some(n) = line.strip_prefix("name = ") {
                name = Some(n.trim_matches('"').to_string());
            } else if let (Some(v), Some(n)) = (line.strip_prefix("version = "), name.take()) {
                locked.push((n, v.trim_matches('"').to_string()));
            }
        }
    }
    let versions = |pick: &dyn Fn(&str) -> bool| {
        let mut v: Vec<String> = locked
            .iter()
            .filter(|(n, _)| pick(n))
            .map(|(n, v)| format!("{n} {v}"))
            .collect();
        v.sort();
        v.join(", ")
    };
    println!(
        "cargo:rustc-env=ORR_COOK_ZSTD_LIBS={}",
        versions(&|n| ZSTD.contains(&n))
    );
    println!(
        "cargo:rustc-env=ORR_COOK_MESH_LIBS={}",
        versions(&|n| MESH.contains(&n))
    );
    let texture = |n: &str| n == "ctt" || n.starts_with("ctt-") || TEXTURE.contains(&n);
    println!(
        "cargo:rustc-env=ORR_COOK_TEXTURE_LIBS={}",
        versions(&texture)
    );
    println!(
        "cargo:rustc-env=ORR_COOK_RASTER_LIBS={}",
        versions(&|n| n == "tiff" || texture(n))
    );
}

/// The value after `marker` up to the next `;`, with digit separators removed.
#[cfg(feature = "ntc")]
fn constant(text: &str, marker: &str, file: &Path) -> String {
    let at = text
        .find(marker)
        .unwrap_or_else(|| panic!("{} has no `{marker}`", file.display()));
    let rest = &text[at + marker.len()..];
    rest[..rest.find(';').expect("constant ends")]
        .trim()
        .replace('\'', "")
}

#[cfg(feature = "ntc")]
fn ntc() {
    let sdk = orr_shaders::build::fetched("ORR_NTC_SDK", "RTXNTC-Library")
        .unwrap_or_else(|e| panic!("the ntc feature needs an RTXNTC-Library checkout: {e}"));
    let header = sdk.join("include/libntc/ntc.h");
    let schema = sdk.join("src/JsonFileFormat.h");
    for f in [&header, &schema] {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    let read =
        |f: &Path| std::fs::read_to_string(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
    let interface = constant(&read(&header), "uint32_t InterfaceVersion =", &header);
    let schema_version = constant(&read(&schema), "uint32_t SchemaVersion =", &schema);
    println!("cargo:rustc-env=ORR_COOK_NTC_SDK={}", sdk.display());
    println!("cargo:rustc-env=ORR_COOK_NTC_INTERFACE={interface}");
    println!("cargo:rustc-env=ORR_COOK_NTC_SCHEMA={schema_version}");
    println!("cargo:rerun-if-changed=src/texture/ntc.cpp");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(sdk.join("include"))
        .define("NTC_STATIC", None)
        .file("src/texture/ntc.cpp")
        .compile("orr_ntc");
}

#[cfg(not(feature = "ntc"))]
fn ntc() {}
