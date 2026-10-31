//! Puts the locked versions of the libraries that shape cooked bytes into environment
//! variables that enter cook keys: `ORR_COOK_ZSTD_LIBS` for every entry,
//! `ORR_COOK_MESH_LIBS` for meshes, `ORR_COOK_TEXTURE_LIBS` for textures,
//! `ORR_COOK_RASTER_LIBS` for raster sets and `ORR_COOK_VOLUME_LIBS` for volumes.
//!
//! With the `ntc` feature, compiles the LibNTC shim against the headers of the RTXNTC-Library
//! checkout in `ORR_NTC_SDK` (else `RTXNTC-Library` under `ORR_TOOLS` or the nearest
//! `.tools`) and sets `ORR_COOK_NTC_SDK`, `ORR_COOK_NTC_INTERFACE` (the headers'
//! `InterfaceVersion`) and `ORR_COOK_NTC_SCHEMA` (the NTC file schema version the checkout
//! writes).
//!
//! With the `subd` feature, compiles OpenSubdiv's Sdc, Vtr and Far sources from the release
//! checkout in `ORR_OPENSUBDIV` (else `OpenSubdiv-3.8.0` under `ORR_TOOLS` or the nearest
//! `.tools`) with the Far shim, and sets `ORR_COOK_SUBD_LIBS` to the release it read.
//!
//! With the `gdeflate` feature, compiles the GDeflate codec of NVIDIA's libdeflate fork from
//! the checkout in `ORR_GDEFLATE` (else `libdeflate-gdeflate` under `ORR_TOOLS` or the
//! nearest `.tools`, where `tools/fetch-slang.sh` puts it) and sets `ORR_COOK_GDEFLATE` to
//! its commit.

use std::path::Path;

const ZSTD: &[&str] = &["zstd", "zstd-safe", "zstd-sys"];
const MESH: &[&str] = &["meshopt", "gltf", "gltf-json", "openusd", "roxmltree"];
const TEXTURE: &[&str] = &["bcdec_rs", "image", "png", "half"];
const VOLUME: &[&str] = &["vdb-rs", "blosc-src", "half"];

fn main() {
    ntc();
    subd();
    gdeflate();
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
    println!(
        "cargo:rustc-env=ORR_COOK_VOLUME_LIBS={}",
        versions(&|n| VOLUME.contains(&n))
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

/// The OpenSubdiv release the cook is pinned to: its checkout directory and version number.
#[cfg(feature = "subd")]
const OPENSUBDIV: (&str, &str) = ("OpenSubdiv-3.8.0", "30800");

#[cfg(feature = "subd")]
fn subd() {
    let src = orr_shaders::build::fetched("ORR_OPENSUBDIV", OPENSUBDIV.0)
        .unwrap_or_else(|e| panic!("the subd feature needs an OpenSubdiv checkout: {e}"));
    let version = src.join("opensubdiv/version.h");
    let text =
        std::fs::read_to_string(&version).unwrap_or_else(|e| panic!("{}: {e}", version.display()));
    let number = constant_define(&text, "OPENSUBDIV_VERSION_NUMBER", &version);
    assert_eq!(
        number,
        OPENSUBDIV.1,
        "{} is OpenSubdiv {number}; the cook is pinned to {}",
        src.display(),
        OPENSUBDIV.1
    );
    println!("cargo:rustc-env=ORR_COOK_SUBD_LIBS=OpenSubdiv {number}");
    println!("cargo:rerun-if-changed=src/subd/osd.cpp");
    let mut files = Vec::new();
    for dir in ["sdc", "vtr", "far"] {
        let dir = src.join("opensubdiv").join(dir);
        let entries = std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        for e in entries {
            let path = e.expect("directory entry").path();
            if path.extension().is_some_and(|x| x == "cpp") {
                files.push(path);
            }
        }
    }
    files.sort();
    let mut b = cc::Build::new();
    b.cpp(true)
        .std("c++17")
        .include(&src)
        .define("_USE_MATH_DEFINES", None)
        .define("NOMINMAX", None)
        .warnings(false)
        .files(&files)
        .file("src/subd/osd.cpp");
    b.compile("orr_osd");
}

#[cfg(feature = "subd")]
fn constant_define(text: &str, name: &str, file: &Path) -> String {
    text.lines()
        .find_map(|l| l.trim().strip_prefix("#define ")?.strip_prefix(name))
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| panic!("{} has no {name}", file.display()))
}

#[cfg(not(feature = "subd"))]
fn subd() {
    println!("cargo:rustc-env=ORR_COOK_SUBD_LIBS=");
}

#[cfg(feature = "gdeflate")]
fn gdeflate() {
    let src =
        orr_shaders::build::fetched("ORR_GDEFLATE", "libdeflate-gdeflate").unwrap_or_else(|e| {
            panic!("the gdeflate feature needs a NVIDIA/libdeflate gdeflate checkout: {e}")
        });
    println!("cargo:rustc-env=ORR_COOK_GDEFLATE={}", commit(&src));
    let cpu = if cfg!(any(target_arch = "aarch64", target_arch = "arm")) {
        "lib/arm/cpu_features.c"
    } else {
        "lib/x86/cpu_features.c"
    };
    // The GDeflate page bound calls the plain DEFLATE bound in `deflate_compress.c`.
    let files = [
        "lib/gdeflate_compress.c",
        "lib/gdeflate_decompress.c",
        "lib/deflate_compress.c",
        "lib/utils.c",
        cpu,
    ];
    for f in files {
        println!("cargo:rerun-if-changed={}", src.join(f).display());
    }
    cc::Build::new()
        .include(&src)
        .files(files.map(|f| src.join(f)))
        .warnings(false)
        .compile("orr_gdeflate");
}

/// The commit a git checkout has checked out, from its `HEAD` and refs.
#[cfg(feature = "gdeflate")]
fn commit(checkout: &Path) -> String {
    let git = checkout.join(".git");
    let read = |f: &Path| std::fs::read_to_string(f).map(|s| s.trim().to_owned());
    let head = read(&git.join("HEAD"))
        .unwrap_or_else(|e| panic!("{} is not a git checkout: {e}", checkout.display()));
    println!("cargo:rerun-if-changed={}", git.join("HEAD").display());
    let Some(name) = head.strip_prefix("ref: ") else {
        return head;
    };
    if let Ok(id) = read(&git.join(name)) {
        return id;
    }
    read(&git.join("packed-refs"))
        .ok()
        .and_then(|refs| {
            refs.lines()
                .find_map(|l| l.strip_suffix(name)?.strip_suffix(' ').map(str::to_owned))
        })
        .unwrap_or_else(|| panic!("{}: cannot resolve {name}", checkout.display()))
}

#[cfg(not(feature = "gdeflate"))]
fn gdeflate() {}
