//! Puts the locked versions of the libraries that shape cooked bytes into environment
//! variables that enter cook keys: `ORR_COOK_ZSTD_LIBS` for every entry,
//! `ORR_COOK_MESH_LIBS` for meshes and `ORR_COOK_TEXTURE_LIBS` for textures.

use std::path::Path;

const ZSTD: &[&str] = &["zstd", "zstd-safe", "zstd-sys"];
const MESH: &[&str] = &["meshopt", "gltf", "gltf-json"];
const TEXTURE: &[&str] = &["bcdec_rs", "image", "png", "half"];

fn main() {
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
    println!(
        "cargo:rustc-env=ORR_COOK_TEXTURE_LIBS={}",
        versions(&|n| n == "ctt" || n.starts_with("ctt-") || TEXTURE.contains(&n))
    );
}
