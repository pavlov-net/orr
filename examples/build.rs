//! Compiles the examples' kernels against orr's shader roots, with orr's Slang structs mapped
//! to the Rust types orr publishes.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_SHADER_DIRS").expect("orr shader roots");
    let types = env::var("DEP_ORR_TYPES").expect("orr types");
    let built = Build::new().and_then(|b| {
        b.kernels(dir.join("shaders"))
            .roots(&roots)
            .types(&types)
            .compile()
    });
    if let Err(e) = built {
        panic!("{e}");
    }
}
