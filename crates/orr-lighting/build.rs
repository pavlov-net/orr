//! Compiles the crate's kernels against orr-scene's shader root and publishes both roots and
//! the Rust types of the crate's Slang structs to dependents. Also generates the subsurface
//! walk's cap tail table.

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};

use orr_shaders::build::Build;

#[path = "build/cap_tail.rs"]
mod cap_tail;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    cap_tail_table(&out_dir.join("cap_tail.rs"));
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let scene = env::var("DEP_ORR_SCENE_SHADER_DIRS").expect("orr-scene shader roots");
    let types = env::var("DEP_ORR_SCENE_TYPES").expect("orr-scene type map");
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .roots(&scene)
            .types(&types)
            .publish_types("::orr_lighting::shaders::types")
            .compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}

/// Writes the cap tail table to `path` as Rust constants, unless the file there came from
/// the same generator source.
fn cap_tail_table(path: &Path) {
    let source = include_str!("build/cap_tail.rs");
    let hash = source.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    let stamp = format!("// generator {hash:016x}\n");
    if fs::read_to_string(path).is_ok_and(|s| s.starts_with(&stamp)) {
        return;
    }
    let caps = cap_tail::CAPS.map(|c| c.to_string()).join(", ");
    let mut s = format!(
        "{stamp}pub(crate) const CAP_TAIL_CAPS: [u32; {}] = [{caps}];\n\
         pub(crate) const CAP_TAIL_ALBEDOS: u32 = {};\n\
         pub(crate) const CAP_TAIL_ANISOTROPIES: u32 = {};\n\
         pub(crate) const CAP_TAIL_G_MAX: f32 = {:?};\n\
         pub(crate) const CAP_TAIL_IORS: u32 = {};\n\
         pub(crate) const CAP_TAIL_IOR_SHARE_MAX: f32 = {:?};\n",
        cap_tail::CAPS.len(),
        cap_tail::ALBEDOS,
        cap_tail::ANISOTROPIES,
        cap_tail::G_MAX as f32,
        cap_tail::IORS,
        cap_tail::IOR_SHARE_MAX as f32,
    );
    let table = cap_tail::table();
    writeln!(s, "pub(crate) static CAP_TAIL: [f32; {}] = [", table.len()).unwrap();
    for v in table {
        writeln!(s, "    {v:?},").unwrap();
    }
    s.push_str("];\n");
    fs::write(path, s).unwrap();
}
