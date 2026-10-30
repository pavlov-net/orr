//! With the `build` feature, generates bindings to the pinned Slang release's `slang.h`.

#[cfg(feature = "build")]
#[allow(dead_code, unreachable_pub)]
#[path = "src/build/locate.rs"]
mod locate;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "build")]
    bindings();
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
