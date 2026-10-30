//! With the `build` feature, generates bindings to the pinned Slang release's `slang.h`.
//! With the `openpbr` feature, converts adobe/openpbr-bsdf's lookup tables into
//! little-endian blobs in `OUT_DIR` and generates the `LUTS` array over them.

#[cfg(any(feature = "build", feature = "openpbr"))]
#[allow(dead_code, unreachable_pub)]
#[path = "src/build/locate.rs"]
mod locate;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "build")]
    bindings();
    #[cfg(feature = "openpbr")]
    openpbr_luts();
}

/// File stem under `impl/data/`, extent and whether texels are RGBA `f32` (else `u16`
/// unorm), in `OpenPBR_LutId` order.
#[cfg(feature = "openpbr")]
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

/// Writes each table as a blob, the `LUTS` array that includes them, and the licence.
#[cfg(feature = "openpbr")]
fn openpbr_luts() {
    use std::env;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=ORR_OPENPBR_BSDF");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let dir = locate::find_openpbr(env::var_os("ORR_OPENPBR_BSDF"), &manifest).unwrap_or_else(|| {
        panic!(
            "adobe/openpbr-bsdf not found: no .tools/openpbr-bsdf/openpbr.h in or above {} and ORR_OPENPBR_BSDF does not name a checkout; run tools/fetch-slang.sh",
            manifest.display()
        )
    });
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("openpbr");
    fs::create_dir_all(&out).unwrap_or_else(|e| panic!("{}: {e}", out.display()));

    let mut luts = String::from("[\n");
    for (id, (stem, extent, rgba)) in OPENPBR_LUTS.into_iter().enumerate() {
        let path = dir.join("impl/data").join(format!("openpbr_{stem}_data.h"));
        println!("cargo:rerun-if-changed={}", path.display());
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

    let licence = dir.join("LICENSE");
    println!("cargo:rerun-if-changed={}", licence.display());
    fs::copy(&licence, out.join("LICENSE"))
        .unwrap_or_else(|e| panic!("{}: {e}", licence.display()));
}

/// The comma-separated numbers of a data header, outside comments, with `vec3(...)`
/// wrappers dropped.
#[cfg(feature = "openpbr")]
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

#[cfg(feature = "build")]
fn bindings() {
    use std::env;
    use std::path::PathBuf;

    println!("cargo:rerun-if-env-changed=ORR_SLANGC");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let slangc = match locate::find_slangc(env::var_os("ORR_SLANGC"), &manifest) {
        Ok(path) => path,
        Err(locate::Missing::BadEnv(path)) => panic!(
            "ORR_SLANGC is {}, which is not the absolute path of an existing file",
            path.display()
        ),
        Err(locate::Missing::NotFound) => panic!(
            "Slang {} not found: no {} in or above {}; run tools/fetch-slang.sh or set ORR_SLANGC",
            locate::SLANG_VERSION,
            locate::tools_slangc().display(),
            manifest.display()
        ),
    };
    let include = locate::release_dir(&slangc).join("include");
    let tag = include.join("slang-tag-version.h");
    println!("cargo:rerun-if-changed={}", tag.display());
    let found = std::fs::read_to_string(&tag)
        .ok()
        .and_then(|s| {
            let line = s.lines().find(|l| l.contains("SLANG_TAG_VERSION"))?;
            Some(line.split('"').nth(1)?.to_owned())
        })
        .unwrap_or_default();
    if found != locate::SLANG_VERSION {
        panic!(
            "Slang at {} is version {found:?}; orr needs exactly {} (run tools/fetch-slang.sh or point ORR_SLANGC at a {} release)",
            locate::release_dir(&slangc).display(),
            locate::SLANG_VERSION,
            locate::SLANG_VERSION
        );
    }
    let header = include.join("slang.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_args(["-xc++", "-std=c++17"])
        .allowlist_type("slang::.*")
        .allowlist_type("Slang.*")
        .allowlist_function("spReflection.*")
        .allowlist_function("spGetBuildTagString")
        .allowlist_function("slang_createGlobalSession2")
        .allowlist_var("SLANG_.*")
        .with_codegen_config(
            bindgen::CodegenConfig::FUNCTIONS
                | bindgen::CodegenConfig::TYPES
                | bindgen::CodegenConfig::VARS,
        )
        .vtable_generation(true)
        .wrap_unsafe_ops(true)
        .layout_tests(false)
        .derive_default(true)
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .dynamic_library_name("SlangLib")
        .dynamic_link_require_all(true)
        .generate()
        .unwrap_or_else(|e| panic!("bindgen over {} failed: {e}", header.display()));
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("slang_sys.rs");
    bindings
        .write_to_file(&out)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
}
