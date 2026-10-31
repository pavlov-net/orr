//! Builds the renderer fixture as a renderer crate would, then three consumer builds against
//! it: one linking every point, one linking nothing, and one against a copy of the renderer
//! whose resolve root has an extra field. The CPU library calls the same linked types. Two
//! more consumer builds link into orr's own kernels: one against orr as it is, one against a
//! copy of orr-lighting's shader root whose environment-bake root has an extra field. The
//! renderer is linked once more with copies of the consumer modules in `OUT_DIR`, which the
//! hot reload tests edit.

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

/// The points orr's own kernels declare.
const ORR_LINKS: [(LinkPoint, &str, &str, &str); 2] = [
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

fn consumer(
    roots: &str,
    link_points: &str,
    out: &Path,
    links: &[(LinkPoint, &str, &str, &str)],
) -> Result<Output, Error> {
    links
        .iter()
        .fold(
            Build::new()?
                .roots(roots)
                .link_points(link_points)
                .out_dir(out),
            |b, (point, name, module, ty)| b.link(*point, name, module, ty),
        )
        .compile()
}

fn fixture_consumer(renderer: &Output, out: &Path, link: bool) -> Result<Output, Error> {
    let links: &[_] = if link { &LINKS } else { &[] };
    consumer(
        &renderer.shader_dirs,
        &renderer.link_points.display().to_string(),
        out,
        links,
    )
}

/// Builds a copy of orr-lighting's shader root whose environment-bake root has one more
/// field, as orr-lighting's build does.
fn edited_orr(roots: &str, src: &Path, out: &Path) -> Result<Output, Error> {
    let (lighting, others): (Vec<&str>, Vec<&str>) = roots
        .split(';')
        .partition(|r| Path::new(r).join("environment/bake.slang").is_file());
    let [lighting] = lighting[..] else {
        panic!("one orr root holds the environment bake: {roots}");
    };
    copy_slang(Path::new(lighting), src, &|name, text| {
        if name == "environment.slang" {
            let edited = text.replace(
                "public float4 focus;",
                "public float4 focus;
    public float4 tint;",
            );
            assert_ne!(edited, text, "BakeRoot moved");
            edited
        } else {
            text
        }
    });
    let scene = env::var("DEP_ORR_TYPES").expect("orr's type map");
    Build::new()?
        .kernels(src)
        .roots(&others.join(";"))
        .types(&scene)
        .out_dir(out)
        .compile()
}

/// Copies every `.slang` file under `src` to `dst` through `edit`, writing only what changed
/// and removing what `src` no longer has.
fn copy_slang(src: &Path, dst: &Path, edit: &dyn Fn(&str, String) -> String) {
    for entry in fs::read_dir(dst).into_iter().flatten() {
        let path = entry.expect("the copied root").path();
        if !src.join(path.file_name().expect("a file")).exists() {
            if path.is_dir() {
                fs::remove_dir_all(&path).expect("the copied root");
            } else {
                fs::remove_file(&path).expect("the copied root");
            }
        }
    }
    for entry in fs::read_dir(src).expect("a shader root") {
        let path = entry.expect("a shader root").path();
        let name = path
            .file_name()
            .expect("a file")
            .to_string_lossy()
            .into_owned();
        let target = dst.join(&name);
        if path.is_dir() {
            copy_slang(&path, &target, edit);
        } else if name.ends_with(".slang") {
            let text = edit(&name, fs::read_to_string(&path).expect("a shader file"));
            fs::create_dir_all(dst).expect("the copied root");
            if fs::read_to_string(&target).ok().as_deref() != Some(text.as_str()) {
                fs::write(&target, text).expect("the copied root");
            }
        }
    }
}

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let src = manifest.join("renderer");
    let edited_src = out.join("edited-renderer-src");
    copy_slang(&src, &edited_src, &|name, text| {
        if name == "resolve.slang" {
            text.replace(
                "uint pad2;",
                "uint pad2;
    float4 tint;",
            )
        } else {
            text
        }
    });
    println!(
        "cargo:rerun-if-changed={}",
        manifest.join("consumer").display()
    );

    let orr_roots = env::var("DEP_ORR_SHADER_DIRS").expect("orr's shader roots");
    let orr_points = env::var("DEP_ORR_LINK_POINTS").expect("orr's link points");

    let hot_consumer = out.join("hot-consumer");
    copy_slang(&manifest.join("consumer"), &hot_consumer, &|_, text| text);
    let hot_modules: Vec<String> = LINKS
        .iter()
        .map(|(_, _, module, _)| {
            let file = Path::new(module).file_name().expect("a module file");
            hot_consumer.join(file).display().to_string()
        })
        .collect();
    let hot_links: Vec<_> = LINKS
        .iter()
        .zip(&hot_modules)
        .map(|((point, name, _, ty), module)| (*point, *name, module.as_str(), *ty))
        .collect();

    let result = (|| {
        let r = renderer(&src, &out.join("renderer"))?;
        consumer(
            &r.shader_dirs,
            &r.link_points.display().to_string(),
            &out.join("hot-linked"),
            &hot_links,
        )?;
        fixture_consumer(&r, &out.join("linked"), true)?;
        fixture_consumer(&r, &out.join("unlinked"), false)?;
        let edited = renderer(&edited_src, &out.join("edited-renderer"))?;
        fixture_consumer(&edited, &out.join("edited"), true)?;
        consumer(&orr_roots, &orr_points, &out.join("orr"), &ORR_LINKS)?;
        let edited = edited_orr(
            &orr_roots,
            &out.join("edited-orr-src"),
            &out.join("edited-orr-renderer"),
        )?;
        consumer(
            &edited.shader_dirs,
            &edited.link_points.display().to_string(),
            &out.join("edited-orr"),
            &ORR_LINKS[..1],
        )?;
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
