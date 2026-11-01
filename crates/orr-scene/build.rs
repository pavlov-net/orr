//! Compiles orr-scene's Slang kernels against orr-pack's and orr-gpu's shader roots and
//! publishes all three, with the Rust types of orr-scene's Slang structs, and writes the
//! tessellation pattern table.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use orr_shaders::build::Build;

#[path = "src/pattern.rs"]
pub mod pattern;

/// `patterns.bin`: per pattern its first vertex, first triangle, vertex and triangle counts
/// (four u32), then every vertex as `w1 | w2 << 16`, then every triangle as three u8, padded
/// to 4 bytes.
fn write_patterns(out: &Path) {
    let table = (0..pattern::PATTERNS).map(|i| pattern::generate(pattern::factors(i)));
    let mut heads = Vec::new();
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for p in table {
        for x in [
            vertices.len() / 4,
            triangles.len() / 3,
            p.vertices.len(),
            p.triangles.len(),
        ] {
            heads.extend((x as u32).to_le_bytes());
        }
        for v in &p.vertices {
            vertices.extend((u32::from(v[0]) | u32::from(v[1]) << 16).to_le_bytes());
        }
        triangles.extend(p.triangles.iter().flatten());
    }
    triangles.resize(triangles.len().next_multiple_of(4), 0);
    let blob = [heads, vertices, triangles].concat();
    fs::write(out.join("patterns.bin"), blob).unwrap();
}

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=src/pattern.rs");
    write_patterns(&out_dir);
    let pack = env::var("DEP_ORR_PACK_SHADER_DIRS").expect("orr-pack shader roots");
    let gpu = env::var("DEP_ORR_GPU_SHADER_DIRS").expect("orr-gpu shader roots");
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .roots(&pack)
            .roots(&gpu)
            .extern_type("ViewConstants", "crate::ViewConstants")
            .extern_type("ExposureState", "crate::ExposureState")
            .publish_types("::orr_scene")
            .compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}
