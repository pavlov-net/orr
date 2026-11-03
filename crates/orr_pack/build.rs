//! Publishes the crate's Slang root (`orr_pack.*`) as `DEP_ORR_PACK_SHADER_DIRS` and the Rust
//! types of its Slang structs as `DEP_ORR_PACK_TYPES`.

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo::metadata=shader_dirs={dir}/shaders");
    println!("cargo::metadata=types={dir}/types.json");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=types.json");
}
