//! Compiles the crate's kernels against orr_scene's shader root and adobe/openpbr-bsdf, publishes
//! them and the Rust types of the crate's Slang structs to dependents, and converts
//! openpbr-bsdf's lookup tables into little-endian blobs in `OUT_DIR` with the `LUTS` array over
//! them.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use orr_shaders::build::{Build, fetched};

/// File stem under `impl/data/`, extent and whether texels are RGBA `f32` (else `u16`
/// unorm), in `OpenPBR_LutId` order.
const OPENPBR_LUTS: [(&str, [u32; 3], bool); 8] = [
    ("ideal_dielectric_energy_complement", [32, 32, 32], false),
    ("ideal_dielectric_avg_energy_complement", [32, 32, 1], false),
    ("ideal_dielectric_reflection_ratio", [32, 32, 1], false),
    ("opaque_dielectric_energy_complement", [32, 32, 32], false),
    (
        "opaque_dielectric_avg_energy_complement",
        [32, 32, 1],
        false,
    ),
    ("ideal_metal_energy_complement", [32, 32, 1], false),
    ("ideal_metal_avg_energy_complement", [32, 1, 1], false),
    ("ltc", [32, 32, 1], true),
];

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let openpbr = fetched("ORR_OPENPBR_BSDF", "openpbr-bsdf").unwrap_or_else(|e| panic!("{e}"));
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .include(&openpbr)
            .dep("orr_scene")
            .publish_types("::orr_lighting::types")
            .compile()
    });
    if let Err(e) = out {
        panic!("{e}");
    }
    openpbr_luts(&openpbr);
}

/// Writes each table as a blob, the `LUTS` array that includes them, and the license.
fn openpbr_luts(dir: &Path) {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("openpbr");
    fs::create_dir_all(&out).unwrap_or_else(|e| panic!("{}: {e}", out.display()));

    let mut luts = String::from("[\n");
    for (id, (stem, extent, rgba)) in OPENPBR_LUTS.into_iter().enumerate() {
        let path = dir.join("impl/data").join(format!("openpbr_{stem}_data.h"));
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let values = lut_values(&text);
        let texels = extent.iter().product::<u32>() as usize;
        let expected = if rgba { texels * 3 } else { texels };
        assert_eq!(
            values.len(),
            expected,
            "{} holds {} values, expected {expected}",
            path.display(),
            values.len()
        );
        let mut bytes = Vec::new();
        if rgba {
            for rgb in values.chunks_exact(3) {
                for v in [rgb[0], rgb[1], rgb[2], 0.0] {
                    bytes.extend_from_slice(&(v as f32).to_le_bytes());
                }
            }
        } else {
            for v in values {
                assert!(
                    v.fract() == 0.0 && (0.0..=65535.0).contains(&v),
                    "{}: {v} is not a 16-bit unorm value",
                    path.display()
                );
                bytes.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
        let blob = out.join(format!("{stem}.bin"));
        fs::write(&blob, bytes).unwrap_or_else(|e| panic!("{}: {e}", blob.display()));
        let format = if rgba { "Rgba32Float" } else { "R16Unorm" };
        let _ = writeln!(
            luts,
            "    Lut {{ id: {id}, name: {stem:?}, extent: {extent:?}, format: LutFormat::{format}, \
             bytes: include_bytes!(concat!(env!(\"OUT_DIR\"), \"/openpbr/{stem}.bin\")) }},"
        );
    }
    luts.push(']');
    let rs = out.join("luts.rs");
    fs::write(&rs, luts).unwrap_or_else(|e| panic!("{}: {e}", rs.display()));

    let license = dir.join("LICENSE");
    fs::copy(&license, out.join("LICENSE"))
        .unwrap_or_else(|e| panic!("{}: {e}", license.display()));
}

/// The comma-separated numbers of a data header, outside comments, with `vec3(...)`
/// wrappers dropped.
fn lut_values(text: &str) -> Vec<f64> {
    text.lines()
        .flat_map(|line| line.split("//").next().unwrap_or("").split(','))
        .map(|t| {
            t.trim()
                .trim_start_matches("vec3(")
                .trim_end_matches(')')
                .trim()
        })
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.trim_end_matches('f')
                .parse()
                .unwrap_or_else(|e| panic!("bad LUT value {t:?}: {e}"))
        })
        .collect()
}
