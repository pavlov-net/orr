//! Publishes the crate's Slang root (`orr_pack.*`) as `DEP_ORR_PACK_SHADER_DIRS`.

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:shader_dirs={dir}/shaders");
    println!("cargo:rerun-if-changed=build.rs");
}
