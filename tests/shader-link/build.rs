//! Builds the renderer fixture as a renderer crate would, then three consumer builds against
//! it: one linking every point, one linking nothing, and one against a copy of the renderer
//! whose resolve root has an extra field. The CPU library calls the same linked types.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use orr_shaders::build::{Build, CpuLibrary, Error, LinkPoint, Output};

/// Two surfaces, one overriding `coverage` and one omitting it, a displacement, an
/// environment and a medium.
const LINKS: [(LinkPoint, &str, &str, &str); 5] = [
    (
        LinkPoint::Surface,
        "ground",
        "consumer/ground.slang",
        "GroundBlend",
    ),
    (LinkPoint::Surface, "grass", "consumer/grass.slang", "Grass"),
    (
        LinkPoint::Displacement,
        "waves",
        "consumer/waves.slang",
        "Waves",
    ),
    (
        LinkPoint::Environment,
        "dome",
        "consumer/dome.slang",
        "Dome",
    ),
    (LinkPoint::Medium, "air", "consumer/air.slang", "Air"),
];

fn renderer(src: &Path, out: &Path) -> Result<Output, Error> {
    Build::new()?.kernels(src).out_dir(out).compile()
}

fn consumer(renderer: &Output, out: &Path, link: bool) -> Result<Output, Error> {
    let b = Build::new()?
        .roots(&renderer.shader_dirs)
        .link_points(&renderer.link_points.display().to_string())
        .out_dir(out);
    let b = if link {
        LINKS.iter().fold(b, |b, (point, name, module, ty)| {
            b.link(*point, name, module, ty)
        })
    } else {
        b
    };
    b.compile()
}

/// Copies the renderer fixture with one more field in the resolve root.
fn edited_copy(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let path = entry?.path();
        let mut text = fs::read_to_string(&path)?;
        if path.file_name().is_some_and(|n| n == "resolve.slang") {
            text = text.replace("uint pad2;", "uint pad2;\n    float4 tint;");
        }
        let target = dst.join(path.file_name().expect("a file"));
        if fs::read_to_string(&target).ok().as_deref() != Some(text.as_str()) {
            fs::write(target, text)?;
        }
    }
    Ok(())
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let src = manifest.join("renderer");
    let edited_src = out.join("edited-renderer-src");
    edited_copy(&src, &edited_src).unwrap_or_else(|e| panic!("{}: {e}", edited_src.display()));
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("consumer").display()
    );

    let result = (|| {
        let r = renderer(&src, &out.join("renderer"))?;
        consumer(&r, &out.join("linked"), true)?;
        consumer(&r, &out.join("unlinked"), false)?;
        let edited = renderer(&edited_src, &out.join("edited-renderer"))?;
        consumer(&edited, &out.join("edited"), true)?;
        LINKS
            .iter()
            .fold(
                CpuLibrary::new()?
                    .source("cpu/calls.slang")
                    .include("renderer"),
                |b, (point, name, module, ty)| b.link(*point, name, module, ty),
            )
            .compile("shader_link_cpu")
    })();
    if let Err(e) = result {
        panic!("{e}");
    }
}
