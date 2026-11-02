//! Builds the suite's kernels against orr-pack's shader root, with orr-pack's page and
//! record types standing in for the generated ones.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_PACK_SHADER_DIRS").expect("orr-pack shader roots");
    let types = env::var("DEP_ORR_PACK_TYPES").expect("orr-pack type map");
    let result = Build::new().and_then(|b| {
        b.kernels(dir.join("src"))
            .roots(&roots)
            .types(&types)
            .compile()
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
