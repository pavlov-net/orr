//! Builds the suite's kernels against orr_lighting's shader roots, and orr_scene's and
//! orr_lighting's kernels with `surfaces/context.slang` linked as surface 1 and
//! `src/rings.slang`'s `Rings` as surface 2.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, LinkPoint};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let build = Build::new().and_then(|b| b.kernels(dir.join("src")).dep("orr_lighting").compile());
    if let Err(e) = build {
        panic!("{e}");
    }
    let linked = Build::new().and_then(|b| {
        b.dep("orr_lighting")
            .out_dir(out.join("linked"))
            .link(
                LinkPoint::Surface,
                "context",
                dir.join("surfaces/context.slang"),
                "Context",
            )
            .link(
                LinkPoint::Surface,
                "rings",
                dir.join("src/rings.slang"),
                "Rings",
            )
            .compile()
    });
    if let Err(e) = linked {
        panic!("{e}");
    }
}
