//! MaterialX documents against the MaterialX specification (`Spec`) and standard nodes
//! (`Nodes`), and OpenPBR. Each row names its clause.

#[allow(dead_code)]
#[path = "common/tagged.rs"]
mod tagged;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use orr_cook::{ArchiveSpec, AssetSpec, CookReport, Manifest, Source, TextureRules};
use orr_pack::material::{Material, filter, slot, wrap};
use orr_pack::{Archive, AssetId, Target};

fn png(f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(8, 8, |x, y| image::Rgba(f(x, y)));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn varying(x: u32, y: u32) -> [u8; 4] {
    [(x * 32) as u8, (y * 32) as u8, 200, 255]
}

/// A directory holding every image the rows name.
fn fixture() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("spec_materialx_{n}"));
    let _ = std::fs::remove_dir_all(&d);
    let files: [(&str, Vec<u8>); 9] = [
        ("v.png", png(varying)),
        (
            "flipped.png",
            png(|x, y| {
                let [r, g, b, a] = varying(x, y);
                [r, 255 - g, b, a]
            }),
        ),
        // Channels 0.2, 0.4, 0.6, 0.8.
        ("flat.png", png(|_, _| [51, 102, 153, 204])),
        ("gray30.png", png(|_, _| [77, 77, 77, 255])),
        ("textures/color/r.png", png(varying)),
        ("textures/color2/c.png", png(varying)),
        ("lib/sub/l.png", png(varying)),
        ("gray128.png", png(|_, _| [128, 128, 128, 255])),
        // States Display P3 in linear light, and a gamma of 1.
        (
            "tagged.png",
            tagged::varying_rgb(&[(b"gAMA", &tagged::gama(1.0)), (b"cICP", &[12, 8, 0, 1])]),
        ),
    ];
    for (name, bytes) in files {
        let path = d.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
    d
}

fn cook(path: &Path, material: &str) -> orr_cook::Result<(Archive, CookReport)> {
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![AssetSpec {
                name: "u".into(),
                source: Source::MaterialX {
                    path: path.into(),
                    material: material.into(),
                    textures: TextureRules::default(),
                },
            }],
        )],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default())?;
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    Ok((a, cooked.report))
}

/// `body`, with material `M` on shader `S` of `inputs`, as a document in `d`.
fn write_doc(d: &Path, name: &str, attrs: &str, body: &str, inputs: &str) -> PathBuf {
    let path = d.join(name);
    let doc = format!(
        r#"<?xml version="1.0"?>
<materialx version="1.39" {attrs}>
  {body}
  <open_pbr_surface name="S" type="surfaceshader">{inputs}</open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
  </surfacematerial>
</materialx>
"#
    );
    std::fs::write(&path, doc).unwrap();
    path
}

fn cook_body(body: &str, inputs: &str) -> orr_cook::Result<(Archive, CookReport)> {
    cook(&write_doc(&fixture(), "m.mtlx", "", body, inputs), "M")
}

fn material(a: &Archive) -> Material {
    let e = a.find(AssetId::from_name("u")).expect("no material");
    let b = &a.blobs(e)[0];
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    Material::from_bytes(&v).unwrap()
}

fn bound(m: &Material, s: usize) -> bool {
    m.textures[s].texture != AssetId::NONE
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

fn close6(a: [f32; 6], b: [f32; 6]) -> bool {
    a.iter().zip(b).all(|(x, y)| close(*x, y))
}

const COLOR: &str = r#"<image name="i" type="color3">
    <input name="file" type="filename" value="v.png"/></image>"#;
const FLOAT: &str = r#"<image name="i" type="float">
    <input name="file" type="filename" value="v.png"/></image>"#;
const FLAT3: &str = r#"<image name="i" type="vector3">
    <input name="file" type="filename" value="flat.png"/></image>"#;
const BASE_I: &str = r#"<input name="base_color" type="color3" nodename="i"/>"#;
const ROUGH_I: &str = r#"<input name="specular_roughness" type="float" nodename="i"/>"#;
const ROUGH_E: &str = r#"<input name="specular_roughness" type="float" nodename="e"/>"#;

struct Row {
    clause: &'static str,
    body: &'static str,
    inputs: &'static str,
    check: fn(&Material) -> bool,
}

#[test]
fn documents_map_as_the_clauses_say() {
    let rows = [
        Row {
            clause: "Spec File Prefixes: an inner fileprefix replaces the enclosing one",
            body: r#"<nodegraph name="g" fileprefix="textures/color/">
                <image name="in1" type="float">
                  <input name="file" type="filename" value="r.png"/></image>
                <image name="in2" type="color3">
                  <input name="file" type="filename" fileprefix="textures/"
                         value="color2/c.png"/></image>
                <output name="o1" type="float" nodename="in1"/>
                <output name="o2" type="color3" nodename="in2"/>
              </nodegraph>"#,
            inputs: r#"<input name="specular_roughness" type="float" nodegraph="g" output="o1"/>
                <input name="base_color" type="color3" nodegraph="g" output="o2"/>"#,
            check: |m| bound(m, slot::ROUGHNESS) && bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "OpenPBR Opacity: a textured geometry_opacity is a linear presence weight",
            body: FLOAT,
            inputs: r#"<input name="geometry_opacity" type="float" nodename="i"/>"#,
            check: |m| m.alpha_mode == Material::BLEND && bound(m, slot::OPACITY),
        },
        Row {
            clause: "OpenPBR Opacity: a uniform image weighs like the constant it holds",
            body: r#"<image name="i" type="float">
                <input name="file" type="filename" value="gray30.png"/></image>"#,
            inputs: r#"<input name="geometry_opacity" type="float" nodename="i"/>"#,
            check: |m| m.alpha_mode == Material::BLEND && close(m.geometry_opacity, 77.0 / 255.0),
        },
        Row {
            clause: "Spec Node Graphs: a functional graph's interface is its nodedef, unread here",
            body: r#"<nodegraph name="NG" nodedef="ND_x">
                  <image name="i" type="color3">
                    <input name="file" type="filename" interfacename="on_the_nodedef"/></image>
                  <output name="out" type="color3" nodename="i"/></nodegraph>
                <dot name="d" type="color3"><input name="in" type="color3" nodename="nope"/></dot>"#,
            inputs: "",
            check: |m| !bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "Spec Inputs: output is optional on a one-output nodegraph",
            body: r#"<nodegraph name="g"><image name="i" type="color3">
                <input name="file" type="filename" value="v.png"/></image>
                <output name="c" type="color3" nodename="i"/></nodegraph>"#,
            inputs: r#"<input name="base_color" type="color3" nodegraph="g"/>"#,
            check: |m| bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "Spec Inputs: output is ignored on a one-output nodegraph",
            body: r#"<nodegraph name="g"><image name="i" type="color3">
                <input name="file" type="filename" value="v.png"/></image>
                <output name="c" type="color3" nodename="i"/></nodegraph>"#,
            inputs: r#"<input name="base_color" type="color3" nodegraph="g" output="nope"/>"#,
            check: |m| bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "Nodes extract: index 3 of a 4-channel stream is valid",
            body: r#"<image name="i" type="color4">
                  <input name="file" type="filename" value="flat.png"/></image>
                <extract name="e" type="float">
                  <input name="in" type="color4" nodename="i"/>
                  <input name="index" type="integer" value="3"/></extract>"#,
            inputs: ROUGH_E,
            check: |m| close(m.specular_roughness, 0.8),
        },
        Row {
            clause: "Nodes separate3: outy of a vector3 is its second channel",
            body: r#"<image name="i" type="vector3">
                  <input name="file" type="filename" value="flat.png"/></image>
                <separate3 name="s" type="multioutput">
                  <input name="in" type="vector3" nodename="i"/></separate3>"#,
            inputs: r#"<input name="specular_roughness" type="float" nodename="s" output="outy"/>"#,
            check: |m| close(m.specular_roughness, 0.4),
        },
        Row {
            clause: "Nodes separate4: outw of a vector4 is its fourth channel",
            body: r#"<image name="i" type="vector4">
                  <input name="file" type="filename" value="flat.png"/></image>
                <separate4 name="s" type="multioutput">
                  <input name="in" type="vector4" nodename="i"/></separate4>"#,
            inputs: r#"<input name="specular_roughness" type="float" nodename="s" output="outw"/>"#,
            check: |m| close(m.specular_roughness, 0.8),
        },
        Row {
            clause: "Nodes separate2: outy of a constant vector2",
            body: r#"<separate2 name="s" type="multioutput">
                  <input name="in" type="vector2" value="0.25, 0.75"/></separate2>"#,
            inputs: r#"<input name="specular_roughness" type="float" nodename="s" output="outy"/>"#,
            check: |m| close(m.specular_roughness, 0.75),
        },
        Row {
            clause: "Spec Inputs: a uniform input reads a constant through dots",
            body: r#"<image name="i" type="vector3">
                  <input name="file" type="filename" value="flat.png"/></image>
                <constant name="k" type="integer">
                  <input name="value" type="integer" value="2"/></constant>
                <dot name="d1" type="integer"><input name="in" type="integer" nodename="k"/></dot>
                <dot name="d2" type="integer"><input name="in" type="integer" nodename="d1"/></dot>
                <extract name="e" type="float">
                  <input name="in" type="vector3" nodename="i"/>
                  <input name="index" type="integer" nodename="d2"/></extract>"#,
            inputs: ROUGH_E,
            check: |m| close(m.specular_roughness, 0.6),
        },
        Row {
            clause: "Spec Inputs: a filename input reads a filename constant through a dot",
            body: r#"<constant name="k" type="filename">
                  <input name="value" type="filename" value="v.png"/></constant>
                <dot name="d" type="filename"><input name="in" type="filename" nodename="k"/></dot>
                <image name="i" type="color3">
                  <input name="file" type="filename" nodename="d"/></image>"#,
            inputs: BASE_I,
            check: |m| bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "Spec Inputs: a string output may connect to a filename input",
            body: r#"<constant name="k" type="string">
                  <input name="value" type="string" value="color/r.png"/></constant>
                <image name="i" type="color3" fileprefix="textures/">
                  <input name="file" type="filename" nodename="k"/></image>"#,
            inputs: BASE_I,
            check: |m| bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "Nodes dot: a no-op on a varying stream",
            body: r#"<image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/></image>
                <dot name="d" type="color3"><input name="in" type="color3" nodename="i"/></dot>"#,
            inputs: r#"<input name="base_color" type="color3" nodename="d"/>"#,
            check: |m| bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "Nodes image: uaddressmode and vaddressmode periodic, clamp and mirror",
            body: r#"<image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="uaddressmode" type="string" value="clamp"/>
                  <input name="vaddressmode" type="string" value="mirror"/></image>"#,
            inputs: BASE_I,
            check: |m| m.textures[slot::BASE_COLOR].wrap == [wrap::CLAMP, wrap::MIRROR],
        },
        Row {
            clause: "Nodes image: address modes default to periodic",
            body: COLOR,
            inputs: BASE_I,
            check: |m| m.textures[slot::BASE_COLOR].wrap == [wrap::REPEAT; 2],
        },
        Row {
            clause: "Nodes image, constant: outside 0-1 reads the node's default",
            body: r#"<image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="default" type="color3" value="0.25, 0.5, 0.75"/>
                  <input name="uaddressmode" type="string" value="constant"/></image>"#,
            inputs: BASE_I,
            check: |m| {
                let b = m.textures[slot::BASE_COLOR];
                b.wrap == [wrap::CONSTANT, wrap::REPEAT] && b.outside == [0.25, 0.5, 0.75, 0.0]
            },
        },
        Row {
            clause: "Nodes image, constant: default is 0 in all channels when not defined",
            body: r#"<image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="vaddressmode" type="string" value="constant"/></image>"#,
            inputs: BASE_I,
            check: |m| {
                let b = m.textures[slot::BASE_COLOR];
                b.wrap == [wrap::REPEAT, wrap::CONSTANT] && b.outside == [0.0; 4]
            },
        },
        Row {
            clause: "Nodes image, constant: a float default reads through its own input",
            body: r#"<image name="i" type="float">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="default" type="float" value="0.75"/>
                  <input name="uaddressmode" type="string" value="constant"/>
                  <input name="vaddressmode" type="string" value="constant"/></image>"#,
            inputs: ROUGH_I,
            check: |m| {
                let b = m.textures[slot::ROUGHNESS];
                b.wrap == [wrap::CONSTANT; 2] && close(b.outside[0], 0.75)
            },
        },
        Row {
            clause: "Nodes image, constant: a default reads past what the image's texels reach",
            body: r#"<image name="i" type="float">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="default" type="float" value="1.5"/>
                  <input name="uaddressmode" type="string" value="constant"/></image>"#,
            inputs: r#"<input name="specular_weight" type="float" nodename="i"/>"#,
            check: |m| close(m.textures[slot::SPECULAR_WEIGHT].outside[0], 1.5),
        },
        Row {
            clause: "Nodes image, constant: an image of one value isn't one past its edges",
            body: r#"<image name="i" type="vector3">
                  <input name="file" type="filename" value="flat.png"/>
                  <input name="default" type="vector3" value="1, 1, 1"/>
                  <input name="uaddressmode" type="string" value="constant"/></image>"#,
            inputs: r#"<input name="base_color" type="color3" nodename="i"/>"#,
            check: |m| bound(m, slot::BASE_COLOR),
        },
        Row {
            clause: "Nodes image: filtertype closest is one texel",
            body: r#"<image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="filtertype" type="string" value="closest"/></image>"#,
            inputs: BASE_I,
            check: |m| {
                m.textures[slot::BASE_COLOR].filter
                    == filter::MAG_NEAREST | filter::MIN_NEAREST | filter::MIP_NEAREST
            },
        },
        Row {
            clause: "Spec Geometric Properties: UV0 is uv set 0",
            body: r#"<geompropvalue name="p" type="vector2">
                  <input name="geomprop" type="string" value="UV0"/></geompropvalue>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#,
            inputs: BASE_I,
            check: |m| {
                let t = m.textures[slot::BASE_COLOR];
                t.texture != AssetId::NONE
                    && t.uv_set == 0
                    && m.uv_transforms[slot::BASE_COLOR] == [0.0; 6]
            },
        },
        Row {
            clause: "Spec Geometric Properties: a geompropdef names a texcoord index",
            body: r#"<geompropdef name="uv1" type="vector2" geomprop="texcoord" index="1"/>
                <geompropvalue name="p" type="vector2">
                  <input name="geomprop" type="string" value="uv1"/></geompropvalue>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#,
            inputs: BASE_I,
            check: |m| m.textures[slot::BASE_COLOR].uv_set == 1,
        },
        Row {
            clause: "Nodes geompropvalue: an undefined geomprop reads default, at every point",
            body: r#"<geompropvalue name="p" type="vector2">
                  <input name="geomprop" type="string" value="nope"/>
                  <input name="default" type="vector2" value="0.25, 0.75"/></geompropvalue>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#,
            inputs: BASE_I,
            // Top-left origin: MaterialX v 0.75 is 0.25.
            check: |m| {
                close6(
                    m.uv_transforms[slot::BASE_COLOR],
                    [-1.0, 0.0, 0.25, 0.0, -1.0, 0.25],
                )
            },
        },
        Row {
            clause: "Nodes tiledimage: ((uv * uvtiling - uvoffset) / imagesize) * tilesize",
            body: r#"<tiledimage name="i" type="color3">
                <input name="file" type="filename" value="v.png"/>
                <input name="uvtiling" type="vector2" value="2, 4"/>
                <input name="uvoffset" type="vector2" value="0.5, 0.25"/>
                <input name="realworldimagesize" type="vector2" value="2, 2"/>
                </tiledimage>"#,
            inputs: BASE_I,
            // u' = u - 0.25; v' = 2 v - 0.125 from the bottom, so 2 v - 0.875 from the top.
            check: |m| {
                close6(
                    m.uv_transforms[slot::BASE_COLOR],
                    [0.0, 0.0, -0.25, 0.0, 1.0, -0.875],
                )
            },
        },
        Row {
            clause: "Nodes place2d: scale divides the coordinates, about the bottom-left origin",
            body: r#"<texcoord name="t" type="vector2"/>
                <place2d name="p" type="vector2">
                  <input name="texcoord" type="vector2" nodename="t"/>
                  <input name="scale" type="vector2" value="2, 4"/></place2d>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#,
            inputs: BASE_I,
            check: |m| {
                close6(
                    m.uv_transforms[slot::BASE_COLOR],
                    [-0.5, 0.0, 0.0, 0.0, -0.75, 0.75],
                )
            },
        },
        Row {
            clause: "ND_place2d_vector2: texcoord defaults to 0, 0, not UV0",
            body: r#"<place2d name="p" type="vector2">
                  <input name="offset" type="vector2" value="-0.25, -0.5"/></place2d>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#,
            inputs: BASE_I,
            check: |m| {
                close6(
                    m.uv_transforms[slot::BASE_COLOR],
                    [-1.0, 0.0, 0.25, 0.0, -1.0, 0.5],
                )
            },
        },
        Row {
            clause: "Nodes image: each texture keeps its own coordinates",
            body: r#"<tiledimage name="a" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="uvtiling" type="vector2" value="2, 2"/></tiledimage>
                <tiledimage name="b" type="float">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="uvtiling" type="vector2" value="4, 4"/></tiledimage>"#,
            inputs: r#"<input name="base_color" type="color3" nodename="a"/>
                <input name="specular_roughness" type="float" nodename="b"/>"#,
            check: |m| {
                close(m.uv_transforms[slot::BASE_COLOR][0], 1.0)
                    && close(m.uv_transforms[slot::ROUGHNESS][0], 3.0)
            },
        },
        Row {
            clause: "Spec Inputs: a unit states the value's unit, converted into the scene's meters",
            body: "",
            inputs: r#"<input name="transmission_depth" type="float" value="5" unittype="distance"
                  unit="centimeter"/>
                <input name="subsurface_radius" type="float" value="2" unittype="distance"
                  unit="foot"/>"#,
            check: |m| close(m.transmission_depth, 0.05) && close(m.subsurface_radius, 0.6096),
        },
        Row {
            clause: "Spec Units: a unit names its own type, which open_pbr_surface does not declare",
            body: "",
            inputs: r#"<input name="transmission_depth" type="float" value="5" unit="centimeter"/>"#,
            check: |m| close(m.transmission_depth, 0.05),
        },
        Row {
            clause: "OpenPBR thin_film_thickness: a stated unit, not its micrometers",
            body: "",
            inputs: r#"<input name="thin_film_weight" type="float" value="1"/>
                <input name="thin_film_thickness" type="float" value="500" unit="nanometer"/>"#,
            check: |m| m.thin_film_thickness == 500.0,
        },
        Row {
            clause: "Spec Units: an unstated type takes the name from a type the scene has",
            body: r#"<unitdef name="UD_a" unittype="altitude"><unit name="foot" scale="2"/></unitdef>"#,
            inputs: r#"<input name="transmission_depth" type="float" value="2" unit="foot"/>"#,
            check: |m| close(m.transmission_depth, 0.6096),
        },
        Row {
            clause: "Spec Units: a unit no input reads needs no scale that parses",
            body: r#"<unitdef name="UD_b" unittype="distance"><unit name="x" scale="1/3"/></unitdef>"#,
            inputs: r#"<input name="transmission_depth" type="float" value="2" unit="foot"/>"#,
            check: |m| close(m.transmission_depth, 0.6096),
        },
        Row {
            clause: "MaterialX library: a unit on a connected input states nothing",
            body: r#"<constant name="c" type="float">
                  <input name="value" type="float" value="4"/></constant>"#,
            inputs: r#"<input name="transmission_depth" type="float" nodename="c" unit="cubit"/>"#,
            check: |m| close(m.transmission_depth, 4.0),
        },
        Row {
            clause: "Spec NodeDef: an empty unit declares none",
            body: r#"<nodedef name="ND_e" node="open_pbr_surface">
                  <input name="transmission_depth" type="float" unittype="" unit=""/>
                  <output name="out" type="surfaceshader"/></nodedef>"#,
            inputs: r#"<input name="transmission_depth" type="float" value="2"/>"#,
            check: |m| close(m.transmission_depth, 2.0),
        },
        Row {
            clause: "Spec Inputs: the constant example's unit reaches what reads the constant",
            body: r#"<constant name="boxwidth" type="float">
                  <input name="value" type="float" value="2.39" unittype="distance" unit="foot"/>
                </constant>
                <dot name="d" type="float"><input name="in" type="float" nodename="boxwidth"/></dot>"#,
            inputs: r#"<input name="transmission_depth" type="float" nodename="d"/>"#,
            check: |m| close(m.transmission_depth, 0.728472),
        },
        Row {
            clause: "Spec Units: another unitdef adds units to a type",
            body: r#"<unitdef name="UD_marine" unittype="distance">
                  <unit name="fathom" scale="1.8288"/></unitdef>"#,
            inputs: r#"<input name="transmission_depth" type="float" value="2" unittype="distance"
                  unit="fathom"/>"#,
            check: |m| close(m.transmission_depth, 3.6576),
        },
        Row {
            clause: "Spec Inputs: a nodegraph interface input holds the unit of the value it gives",
            body: r#"<nodegraph name="g">
                  <input name="w" type="float" value="3" unittype="distance" unit="millimeter"/>
                  <constant name="c" type="float">
                    <input name="value" type="float" interfacename="w"/></constant>
                  <output name="o" type="float" nodename="c"/>
                </nodegraph>"#,
            inputs: r#"<input name="transmission_depth" type="float" nodegraph="g"/>"#,
            check: |m| close(m.transmission_depth, 0.003),
        },
        Row {
            clause: "Spec NodeDef: a unit on a nodedef input is the one its values are in",
            body: r#"<nodedef name="ND_mm" node="open_pbr_surface">
                  <input name="transmission_depth" type="float" unittype="distance"
                    unit="millimeter"/>
                  <output name="out" type="surfaceshader"/></nodedef>"#,
            inputs: r#"<input name="transmission_depth" type="float" value="7"/>
                <input name="subsurface_radius" type="float" value="2"/>"#,
            check: |m| close(m.transmission_depth, 0.007) && m.subsurface_radius == 2.0,
        },
        Row {
            clause: "ND_place2d_vector2: rotate declares degrees, so radians convert to them",
            body: r#"<texcoord name="t" type="vector2"/>
                <place2d name="p" type="vector2">
                  <input name="texcoord" type="vector2" nodename="t"/>
                  <input name="rotate" type="float" value="1.57079632679" unittype="angle"
                    unit="radian"/></place2d>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#,
            inputs: BASE_I,
            check: |m| {
                let [a, b, _, d, e, _] = m.uv_transforms[slot::BASE_COLOR];
                close(a + 1.0, 0.0) && close(e + 1.0, 0.0) && close(b.abs(), 1.0) && close(d, -b)
            },
        },
        Row {
            clause: "Nodes tiledimage: real-world sizes of different units convert before they divide",
            body: r#"<tiledimage name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="realworldimagesize" type="vector2" value="100, 100"
                    unittype="distance" unit="centimeter"/>
                  <input name="realworldtilesize" type="vector2" value="0.5, 0.25"
                    unittype="distance" unit="meter"/>
                </tiledimage>"#,
            inputs: BASE_I,
            check: |m| {
                let t = m.uv_transforms[slot::BASE_COLOR];
                close(t[0], -0.5) && close(t[4], -0.75)
            },
        },
    ];
    let failed: Vec<String> = rows
        .iter()
        .filter_map(|r| match cook_body(r.body, r.inputs) {
            Ok((a, _)) => (!(r.check)(&material(&a))).then(|| r.clause.to_string()),
            Err(e) => Some(format!("{}: {e}", r.clause)),
        })
        .collect();
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn invalid_documents_fail() {
    let extract = |index: &'static str| {
        format!(
            r#"<extract name="e" type="float">
              <input name="in" type="color3" nodename="i"/>
              <input name="index" type="integer" value="{index}"/></extract>"#
        )
    };
    let rows = [
        (
            "Nodes extract: index 3 of a color3",
            format!("{COLOR}{}", extract("3")),
            ROUGH_E,
        ),
        (
            "Nodes extract: a negative index",
            format!("{COLOR}{}", extract("-1")),
            ROUGH_E,
        ),
        (
            "Nodes extract: index 3 of a constant color3",
            r#"<extract name="e" type="float">
              <input name="in" type="color3" value="0.1, 0.2, 0.3"/>
              <input name="index" type="integer" value="3"/></extract>"#
                .to_string(),
            ROUGH_E,
        ),
        (
            "Nodes separate3: no fourth output",
            format!(
                r#"{FLAT3}<separate3 name="s" type="multioutput">
                  <input name="in" type="vector3" nodename="i"/></separate3>"#
            ),
            r#"<input name="specular_roughness" type="float" nodename="s" output="outw"/>"#,
        ),
        (
            "Nodes image: an address mode outside the enum, through a constant",
            r#"<constant name="k" type="string">
                <input name="value" type="string" value="nope"/></constant>
              <image name="i" type="color3">
                <input name="file" type="filename" value="v.png"/>
                <input name="uaddressmode" type="string" nodename="k"/></image>"#
                .to_string(),
            BASE_I,
        ),
        (
            "Spec Units: a unit no unitdef defines",
            String::new(),
            r#"<input name="transmission_depth" type="float" value="5" unit="cubit"/>"#,
        ),
        (
            "Spec Inputs: a unit outside the input's unittype",
            String::new(),
            r#"<input name="transmission_depth" type="float" value="5" unittype="angle"
                  unit="meter"/>"#,
        ),
        (
            "ND_place2d_vector2: rotate declares angle, so a distance is outside it",
            r#"<place2d name="p" type="vector2">
                  <input name="rotate" type="float" value="5" unit="meter"/></place2d>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#
                .to_string(),
            BASE_I,
        ),
        (
            "ND_place2d_vector2: a connection does not carry a distance past rotate's angle",
            r#"<constant name="c" type="float">
                  <input name="value" type="float" value="5" unit="meter"/></constant>
                <place2d name="p" type="vector2">
                  <input name="rotate" type="float" nodename="c"/></place2d>
                <image name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="texcoord" type="vector2" nodename="p"/></image>"#
                .to_string(),
            BASE_I,
        ),
        (
            "Nodes tiledimage: a connection does not carry an angle past a real-world size",
            r#"<constant name="c" type="vector2">
                  <input name="value" type="vector2" value="1, 1" unit="radian"/></constant>
                <tiledimage name="i" type="color3">
                  <input name="file" type="filename" value="v.png"/>
                  <input name="realworldimagesize" type="vector2" nodename="c"/></tiledimage>"#
                .to_string(),
            BASE_I,
        ),
        (
            "OpenPBR transmission_depth: a distance, not an angle",
            String::new(),
            r#"<input name="transmission_depth" type="float" value="5" unit="degree"/>"#,
        ),
    ];
    let cooked: Vec<&str> = rows
        .iter()
        .filter(|(_, body, inputs)| cook_body(body, inputs).is_ok())
        .map(|r| r.0)
        .collect();
    assert!(cooked.is_empty(), "{cooked:#?}");
}

/// The cook of a material whose `displacementshader` is node `D` of `body`.
fn cook_displaced(body: &str) -> orr_cook::Result<(Archive, CookReport)> {
    let path = fixture().join("m.mtlx");
    let doc = format!(
        r#"<?xml version="1.0"?>
<materialx version="1.39">
  {body}
  <open_pbr_surface name="S" type="surfaceshader"/>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
    <input name="displacementshader" type="displacementshader" nodename="D"/>
  </surfacematerial>
</materialx>
"#
    );
    std::fs::write(&path, doc).unwrap();
    cook(&path, "M")
}

fn displaced(body: &str) -> Material {
    material(&cook_displaced(body).unwrap().0)
}

/// Spec Units: "images and the quantities they represent such as displacement amount" convert
/// from an absolute size, and a unit on a filename applies to the values read from the file.
/// `flat.png` holds a height of 0.2 of its full scale, which the cook's measured range moves
/// into the offset.
#[test]
fn displacement_units_reach_the_material() {
    let image = |unit: &str| {
        format!(
            r#"<image name="i" type="float">
              <input name="file" type="filename" value="flat.png" {unit}/></image>"#
        )
    };
    let node = |inputs: &str| {
        format!(r#"<displacement name="D" type="displacementshader">{inputs}</displacement>"#)
    };
    let from_image = r#"<input name="displacement" type="float" nodename="i"/>"#;

    let m = displaced(&format!("{}{}", image(""), node(from_image)));
    assert!(close(m.displacement_offset, 0.2), "{m:?}");

    let unit = r#"unittype="distance" unit="centimeter""#;
    let m = displaced(&format!("{}{}", image(unit), node(from_image)));
    assert!(close(m.displacement_offset, 0.002), "{m:?}");
    assert!(close(m.displacement_max, 0.01), "{m:?}");

    let scale = r#"<input name="scale" type="float" value="10" unit="millimeter"/>"#;
    let inputs = format!("{from_image}{scale}");
    let m = displaced(&format!("{}{}", image(""), node(&inputs)));
    assert!(close(m.displacement_offset, 0.002), "{m:?}");

    let m = displaced(&node(
        r#"<input name="displacement" type="float" value="3" unit="centimeter"/>
           <input name="scale" type="float" value="2"/>"#,
    ));
    assert!(bound(&m, slot::DISPLACEMENT));
    assert!(close(m.displacement_offset, 0.06), "{m:?}");
    assert!(close(m.displacement_max, 0.06), "{m:?}");

    // ND_displacement_vector3 has no reading along the normal.
    let vector = r#"<input name="displacement" type="vector3" value="0.1, 0.2, 0.3"/>"#;
    assert!(cook_displaced(&node(vector)).is_err());
}

/// The content of the base color texture of `body`.
fn base_texture(body: &str, inputs: &str) -> orr_pack::Hash32 {
    let (a, _) = cook_body(body, inputs).unwrap();
    let m = material(&a);
    a.find(m.textures[slot::BASE_COLOR].texture)
        .unwrap()
        .content_hash
}

#[test]
fn address_mode_through_a_constant_is_the_mode_it_holds() {
    let image = |mode: &str| {
        format!(
            r#"<constant name="k" type="string">
              <input name="value" type="string" value="clamp"/></constant>
            <image name="i" type="color3">
              <input name="file" type="filename" value="v.png"/>
              {mode}</image>"#
        )
    };
    let direct = |m: &str| {
        image(&format!(
            r#"<input name="uaddressmode" type="string" value="{m}"/>
            <input name="vaddressmode" type="string" value="{m}"/>"#
        ))
    };
    let through = image(
        r#"<input name="uaddressmode" type="string" nodename="k"/>
        <input name="vaddressmode" type="string" nodename="k"/>"#,
    );
    let clamp = base_texture(&direct("clamp"), BASE_I);
    assert_ne!(clamp, base_texture(&direct("periodic"), BASE_I));
    assert_eq!(clamp, base_texture(&through, BASE_I));
}

#[test]
fn normalmap_vector2_scale_flips_only_y() {
    // Nodes normalmap: scale is float or vector2.
    let normal = |file: &str, scale: &str| {
        let body = format!(
            r#"<image name="i" type="vector3">
              <input name="file" type="filename" value="{file}"/></image>
            <normalmap name="n" type="vector3">
              <input name="in" type="vector3" nodename="i"/>
              <input name="scale" {scale}/></normalmap>"#
        );
        let inputs = r#"<input name="geometry_normal" type="vector3" nodename="n"/>"#;
        let (a, _) = cook_body(&body, inputs).unwrap();
        let m = material(&a);
        a.find(m.textures[slot::NORMAL].texture)
            .unwrap()
            .content_hash
    };
    assert_eq!(
        normal("v.png", r#"type="vector2" value="1, -1""#),
        normal("flipped.png", r#"type="float" value="1""#)
    );
}

#[test]
fn toksvig_roughness_follows_the_normal_scale_of_each_axis() {
    let roughness = |scale: &str| {
        let body = format!(
            r#"<image name="i" type="vector3">
              <input name="file" type="filename" value="ripple.png"/></image>
            <normalmap name="n" type="vector3">
              <input name="in" type="vector3" nodename="i"/>
              <input name="scale" type="vector2" value="{scale}"/></normalmap>"#
        );
        let inputs = r#"<input name="geometry_normal" type="vector3" nodename="n"/>"#;
        let d = fixture();
        std::fs::write(
            d.join("ripple.png"),
            png(|_, y| [128, if y % 2 == 0 { 40 } else { 216 }, 200, 255]),
        )
        .unwrap();
        let (a, _) = cook(&write_doc(&d, "m.mtlx", "", &body, inputs), "M").unwrap();
        let m = material(&a);
        a.find(m.textures[slot::ROUGHNESS].texture)
            .unwrap()
            .content_hash
    };
    assert_ne!(roughness("0, 1"), roughness("0, 0.1"));
}

#[test]
fn xinclude_adds_the_children_of_the_included_root() {
    // Spec File Format; the spec's snippet leaves the xi prefix undeclared.
    let d = fixture();
    std::fs::write(
        d.join("lib/lib.mtlx"),
        r#"<?xml version="1.0"?>
<materialx version="1.39" namespace="stdmaterials" fileprefix="sub/">
  <image name="i" type="color3"><input name="file" type="filename" value="l.png"/></image>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" nodename="i"/>
  </open_pbr_surface>
  <surfacematerial name="wood" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
  </surfacematerial>
</materialx>
"#,
    )
    .unwrap();
    std::fs::write(
        d.join("lib/nodes.mtlx"),
        r#"<?xml version="1.0"?>
<materialx version="1.39">
  <image name="shared" type="color3">
    <input name="file" type="filename" value="r.png"/>
  </image>
</materialx>
"#,
    )
    .unwrap();
    let top = write_doc(
        &d,
        "top.mtlx",
        r#"fileprefix="textures/color/""#,
        r#"<xi:include href="lib/lib.mtlx"/>"#,
        "",
    );
    let (a, report) = cook(&top, "stdmaterials:wood").unwrap();
    assert!(bound(&material(&a), slot::BASE_COLOR));
    let inputs = &report.assets[0].inputs;
    assert!(
        inputs.iter().any(|p| p.ends_with("lib.mtlx")),
        "an included document is an input of the cook: {inputs:?}"
    );

    // No fileprefix of its own: the includer's is distributed to the included children,
    // and a declared xi prefix reads the same. The image resolves against the included
    // document, and the includer connects to it by name.
    let top = write_doc(
        &d,
        "shared.mtlx",
        r#"xmlns:xi="http://www.w3.org/2001/XInclude" fileprefix="../textures/color/""#,
        r#"<xi:include href="lib/nodes.mtlx"/>"#,
        r#"<input name="base_color" type="color3" nodename="shared"/>"#,
    );
    let (a, _) = cook(&top, "M").unwrap();
    assert!(bound(&material(&a), slot::BASE_COLOR));
}

#[test]
fn xinclude_targets_must_exist_and_end() {
    let d = fixture();
    let missing = write_doc(&d, "a.mtlx", "", r#"<xi:include href="nope.mtlx"/>"#, "");
    assert!(cook(&missing, "M").is_err(), "a missing include");
    let fallback = write_doc(
        &d,
        "b.mtlx",
        "",
        r#"<xi:include href="nope.mtlx"><xi:fallback/></xi:include>"#,
        "",
    );
    assert!(
        cook(&fallback, "M").is_ok(),
        "a fallback replaces a missing include"
    );

    let one = write_doc(
        &d,
        "one.mtlx",
        "",
        r#"<xi:include href="sub/../two.mtlx"/>"#,
        "",
    );
    write_doc(&d, "two.mtlx", "", r#"<xi:include href="one.mtlx"/>"#, "");
    std::fs::create_dir_all(d.join("sub")).unwrap();
    let e = cook(&one, "M").err().expect("an include cycle cooks");
    assert!(e.to_string().contains("cycle"), "{e}");
}

#[test]
fn xinclude_fallbacks_stand_where_the_include_failed() {
    let d = fixture();
    let image = r#"<image name="shared" type="color3">
        <input name="file" type="filename" value="v.png"/></image>"#;
    let base = r#"<input name="base_color" type="color3" nodename="shared"/>"#;
    std::fs::write(
        d.join("good.mtlx"),
        format!(r#"<?xml version="1.0"?><materialx version="1.39">{image}</materialx>"#),
    )
    .unwrap();
    std::fs::write(
        d.join("broken.mtlx"),
        r#"<?xml version="1.0"?><materialx version="1.39">
          <xi:include href="good.mtlx"/><xi:include href="nope.mtlx"/></materialx>"#,
    )
    .unwrap();

    let body = format!(
        r#"<xi:include href="nope.mtlx"><xi:fallback/></xi:include>
        <xi:include href="nope.mtlx"><xi:fallback>{image}</xi:fallback></xi:include>"#
    );
    let again = write_doc(&d, "again.mtlx", "", &body, base);
    assert!(
        cook(&again, "M").is_ok(),
        "a second failed include takes its own fallback"
    );
    let body = r#"<xi:include href="nope.mtlx"><xi:fallback/></xi:include>
        <xi:include href="nope.mtlx"/>"#;
    let unforgiven = write_doc(&d, "unforgiven.mtlx", "", body, "");
    assert!(
        cook(&unforgiven, "M").is_err(),
        "a fallback excuses only its own include"
    );

    let body = r#"<xi:include href="broken.mtlx"><xi:fallback/></xi:include>
        <xi:include href="good.mtlx"/>"#;
    let partial = write_doc(&d, "partial.mtlx", "", body, base);
    assert!(
        cook(&partial, "M").is_ok(),
        "a document a failed include read is still there to include"
    );

    let body = r#"<xi:include href="good.mtlx">
        <xi:fallback><xi:include href="nope.mtlx"/></xi:fallback></xi:include>"#;
    let unused = write_doc(&d, "unused.mtlx", "", body, base);
    assert!(
        cook(&unused, "M").is_ok(),
        "an include in an unused fallback is not read"
    );
    let body = r#"<xi:include href="nope.mtlx">
        <xi:fallback><xi:include href="good.mtlx"/></xi:fallback></xi:include>"#;
    let nested = write_doc(&d, "nested.mtlx", "", body, base);
    assert!(
        cook(&nested, "M").is_ok(),
        "an include in a used fallback is read"
    );
}

#[test]
fn a_fallback_stands_only_until_the_include_appears() {
    let d = fixture();
    let doc = write_doc(
        &d,
        "m.mtlx",
        "",
        r#"<xi:include href="late.mtlx"><xi:fallback>
          <constant name="c" type="color3">
            <input name="value" type="color3" value="1, 0, 0"/></constant>
          </xi:fallback></xi:include>"#,
        r#"<input name="base_color" type="color3" nodename="c"/>"#,
    );
    let out = d.join("out.orr");
    let cook_file = || {
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::File(out.clone()),
                vec![AssetSpec {
                    name: "u".into(),
                    source: Source::MaterialX {
                        path: doc.clone(),
                        material: "M".into(),
                        textures: TextureRules::default(),
                    },
                }],
            )],
        };
        orr_cook::cook(&manifest, &Default::default()).unwrap();
        material(&Archive::open(&out).unwrap()).base_color
    };
    assert_eq!(cook_file(), [1.0, 0.0, 0.0]);
    assert_eq!(cook_file(), [1.0, 0.0, 0.0], "the record links");
    std::fs::write(
        d.join("late.mtlx"),
        r#"<?xml version="1.0"?><materialx version="1.39">
          <constant name="c" type="color3">
            <input name="value" type="color3" value="0, 1, 0"/></constant></materialx>"#,
    )
    .unwrap();
    assert_eq!(cook_file(), [0.0, 1.0, 0.0]);
}

/// Runs a cook that a cycle could keep from ending.
fn bounded(body: &'static str, inputs: &'static str) -> String {
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = send.send(cook_body(body, inputs).map(|_| ()));
    });
    match receive.recv_timeout(Duration::from_secs(60)) {
        Ok(Ok(())) => panic!("a cycle cooks"),
        Ok(Err(e)) => e.to_string(),
        Err(e) => panic!("a cycle hangs the cook: {e}"),
    }
}

#[test]
fn connection_cycles_fail() {
    // Spec: a node graph is a directed acyclic graph.
    let rows: [(&str, &str, &str); 5] = [
        (
            "a constant on itself",
            r#"<constant name="c" type="float">
                <input name="value" type="float" nodename="c"/></constant>"#,
            r#"<input name="specular_roughness" type="float" nodename="c"/>"#,
        ),
        (
            "dots in a ring",
            r#"<dot name="a" type="color3"><input name="in" type="color3" nodename="b"/></dot>
              <dot name="b" type="color3"><input name="in" type="color3" nodename="a"/></dot>"#,
            r#"<input name="base_color" type="color3" nodename="a"/>"#,
        ),
        (
            "an extract under itself",
            r#"<extract name="e" type="float"><input name="in" type="color3" nodename="e"/>
                </extract>"#,
            ROUGH_E,
        ),
        (
            "a place2d under itself",
            r#"<place2d name="p" type="vector2">
                <input name="texcoord" type="vector2" nodename="p"/></place2d>
              <image name="i" type="color3">
                <input name="file" type="filename" value="v.png"/>
                <input name="texcoord" type="vector2" nodename="p"/></image>"#,
            BASE_I,
        ),
        (
            "a uniform input through a ring",
            r#"<dot name="a" type="string"><input name="in" type="string" nodename="a"/></dot>
              <image name="i" type="float">
                <input name="file" type="filename" value="v.png"/>
                <input name="uaddressmode" type="string" nodename="a"/></image>"#,
            ROUGH_I,
        ),
    ];
    for (what, body, inputs) in rows {
        let e = bounded(body, inputs);
        assert!(e.contains("cycle"), "{what}: {e}");
    }
}

#[test]
fn inputs_not_yet_honored_warn() {
    let object = format!(
        r#"{COLOR}<normalmap name="n" type="vector3">
          <input name="in" type="vector3" nodename="i"/>
          <input name="space" type="string" value="object"/></normalmap>"#
    );
    let inputs = r#"<input name="geometry_normal" type="vector3" nodename="n"/>"#;
    let (_, report) = cook_body(&object, inputs).unwrap();
    assert!(
        report.assets[0]
            .warnings
            .iter()
            .any(|w| w.contains("space")),
        "{:?}",
        report.assets[0].warnings
    );

    let d = fixture();
    let doc = d.join("back.mtlx");
    std::fs::write(
        &doc,
        r#"<?xml version="1.0"?>
<materialx version="1.39">
  <open_pbr_surface name="F" type="surfaceshader"/>
  <open_pbr_surface name="B" type="surfaceshader"/>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="F"/>
    <input name="backsurfaceshader" type="surfaceshader" nodename="B"/>
  </surfacematerial>
  <surfacematerial name="Same" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="F"/>
    <input name="backsurfaceshader" type="surfaceshader" nodename="F"/>
  </surfacematerial>
</materialx>
"#,
    )
    .unwrap();
    let (_, report) = cook(&doc, "M").unwrap();
    assert!(
        report.assets[0]
            .warnings
            .iter()
            .any(|w| w.contains("backsurfaceshader")),
        "{:?}",
        report.assets[0].warnings
    );
    let (_, report) = cook(&doc, "Same").unwrap();
    assert!(report.assets[0].warnings.is_empty());
}

#[test]
fn long_connection_chains_cook() {
    // Nothing in the spec limits a graph's depth, so a chain must not exhaust the stack.
    const LENGTH: usize = 20_000;
    let mut body = String::from(r#"<texcoord name="n0" type="vector2"/>"#);
    for i in 1..LENGTH {
        let (kind, input, extra) = match i % 3 {
            0 => ("dot", "in", ""),
            1 => ("place2d", "texcoord", ""),
            _ => (
                "geompropvalue",
                "default",
                r#"<input name="geomprop" type="string" value="nope"/>"#,
            ),
        };
        let last = i - 1;
        body += &format!(
            r#"<{kind} name="n{i}" type="vector2">{extra}
              <input name="{input}" type="vector2" nodename="n{last}"/></{kind}>"#
        );
    }
    let last = LENGTH - 1;
    body += &format!(
        r#"<image name="i" type="color3"><input name="file" type="filename" value="v.png"/>
          <input name="texcoord" type="vector2" nodename="n{last}"/></image>"#
    );
    let (a, _) = cook_body(&body, BASE_I).unwrap();
    assert_eq!(material(&a).uv_transforms[slot::BASE_COLOR], [0.0; 6]);
}

/// An `image` `i` of `ty` reading `file`, the file's input carrying `attrs`.
fn image_node(ty: &str, file: &str, attrs: &str) -> String {
    format!(
        r#"<image name="i" type="{ty}"><input name="file" type="filename" value="{file}" {attrs}/></image>"#
    )
}

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 2e-3
}

/// Spec Color Spaces: the table of spaces, with "`none` remains an acceptable alias for `data`",
/// and L200 "values of type color3 and color4 are always associated with a particular color
/// space", which leaves a `float` image in a color scope as it is. PNG 128 through current and
/// deprecated names.
#[test]
fn image_color_spaces_decode_as_named() {
    let color = r#"<input name="base_color" type="color3" nodename="i"/>"#;
    let stored: f32 = 128.0 / 255.0;
    let srgb = orr_color::srgb_decode(f64::from(stored)) as f32;
    for (space, want) in [
        ("srgb_rec709_scene", srgb),
        ("srgb_texture", srgb),
        ("g22_rec709_scene", stored.powf(2.2)),
        ("g24_rec709_scene", stored.powf(2.4)),
        ("g18_rec709", stored.powf(1.8)),
        ("lin_rec709_scene", stored),
        ("data", stored),
        ("none", stored),
    ] {
        let image = image_node("color3", "gray128.png", &format!(r#"colorspace="{space}""#));
        let (a, _) = cook_body(&image, color).unwrap();
        let got = material(&a).base_color;
        assert!(
            got.iter().all(|c| near(*c, want)),
            "{space}: {got:?}, not {want}"
        );
    }

    let image = image_node("float", "gray128.png", r#"colorspace="srgb_rec709_scene""#);
    let input = r#"<input name="specular_roughness" type="float" nodename="i"/>"#;
    let (a, _) = cook_body(&image, input).unwrap();
    assert!(near(material(&a).specular_roughness, stored));

    // L364 leaves other spaces to color management systems, which the cook isn't: it says so.
    let image = image_node("color3", "gray128.png", r#"colorspace="studio_log""#);
    let (a, report) = cook_body(&image, color).unwrap();
    assert!(near(material(&a).base_color[0], stored));
    assert!(
        report.assets[0]
            .warnings
            .iter()
            .any(|w| w.contains("studio_log"))
    );
}

/// Spec L381: "data" means "no color space conversion should be applied". What a file states
/// doesn't apply under any space a document names either: L364 has the application transform from
/// the space named. `acescg`, which has no id, converts to Rec.2020; `lin_p3d65_scene` keeps its.
#[test]
fn named_spaces_stand_over_what_the_file_states() {
    let color = r#"<input name="base_color" type="color3" nodename="i"/>"#;
    let primaries = |a: &Archive| {
        let e = a
            .find(material(a).textures[slot::BASE_COLOR].texture)
            .unwrap();
        let b = &a.blobs(e)[0];
        let mut v = vec![0; b.raw_len as usize];
        a.read(b, &mut v).unwrap();
        orr_pack::texture::TextureInfo::parse(&v)
            .unwrap()
            .header
            .primaries
    };
    for (attrs, want) in [
        (r#"colorspace="data""#, 0),
        ("", 0),
        (r#"colorspace="srgb_rec709_scene""#, 0),
        (r#"colorspace="lin_p3d65_scene""#, 1),
        (r#"colorspace="acescg""#, 2),
    ] {
        let image = image_node("color3", "tagged.png", attrs);
        let (a, report) = cook_body(&image, color).unwrap();
        assert_eq!(primaries(&a), want, "{attrs}");
        let warnings = &report.assets[0].warnings;
        let clipped = warnings.iter().filter(|w| w.contains("Rec.2020")).count();
        assert_eq!(clipped, usize::from(want == 2), "{attrs}: {warnings:?}");
    }
}

/// Spec L184: "implementations are expected to translate those color values", and L331's working
/// space is "the default color space for all image inputs and color values". The audit's rows:
/// 0.5 under an sRGB name reads 0.214, a document in ACEScg converts what it doesn't mark, and
/// "data" (L381) converts nothing.
#[test]
fn color_values_read_in_their_space() {
    let (a, _) = cook_body(
        "",
        r#"<input name="base_color" type="color3" value="0.5,0.5,0.5" colorspace="srgb_rec709_scene"/>
           <input name="base_weight" type="float" value="0.5" colorspace="srgb_rec709_scene"/>"#,
    )
    .unwrap();
    let m = material(&a);
    assert!(
        m.base_color.iter().all(|c| near(*c, 0.214)),
        "{:?}",
        m.base_color
    );
    assert!(close(m.base_weight, 0.5));

    let doc = write_doc(
        &fixture(),
        "m.mtlx",
        r#"colorspace="acescg""#,
        "",
        r#"<input name="base_color" type="color3" value="0.5,0.25,0.125"/>
           <input name="specular_color" type="color3" value="0.5,0.25,0.125" colorspace="data"/>"#,
    );
    let (a, _) = cook(&doc, "M").unwrap();
    let m = material(&a);
    // By the published ACEScg to linear Rec.709 matrix.
    let want = [0.6867, 0.2188, 0.0999];
    assert!(
        m.base_color.iter().zip(want).all(|(c, w)| near(*c, w)),
        "{:?}",
        m.base_color
    );
    assert_eq!(m.specular_color, [0.5, 0.25, 0.125]);
}
