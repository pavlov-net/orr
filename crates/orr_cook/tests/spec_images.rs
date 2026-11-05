//! What the specs say of images, per importer and per encoding. Each row cites its clause; rows
//! without one pin this project's own rule and say so.

#[path = "common/float_ktx2.rs"]
mod float_ktx2;
#[allow(dead_code)]
#[path = "common/tagged.rs"]
mod tagged;
#[path = "common/texels.rs"]
mod texels;

use std::path::{Path, PathBuf};

use float_ktx2::float_ktx2;
use orr_cook::{
    ArchiveSpec, AssetSpec, CookReport, DataFormat, GltfSettings, InputChannels, Manifest,
    MipFilter, Source, TextureOverride, TextureRules, TextureSettings, Usage, UsdSettings,
};
use orr_pack::material::{Material, slot};
use orr_pack::texture::{mip_size, vk};
use orr_pack::{Archive, AssetId, Target};
use texels::{decode, mips, read_blob, write_glb};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cook(source: Source) -> (Archive, CookReport) {
    let assets = vec![AssetSpec {
        name: "u".into(),
        source,
    }];
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(Target::Memory, assets)],
    };
    let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
    let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
    (a, cooked.report)
}

fn material(a: &Archive, name: &str) -> Material {
    let e = a.find(AssetId::from_name(name)).unwrap();
    Material::from_bytes(&read_blob(a, e, 0)).unwrap()
}

/// Cooks a MaterialX document of `nodes`, under an `open_pbr_surface` with `inputs`.
fn mtlx(d: &Path, nodes: &str, inputs: &str, textures: TextureRules) -> (Archive, Material) {
    let doc = d.join("m.mtlx");
    std::fs::write(
        &doc,
        format!(
            r#"<?xml version="1.0"?>
<materialx version="1.39">
  {nodes}
  <open_pbr_surface name="S" type="surfaceshader">{inputs}</open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
  </surfacematerial>
</materialx>
"#
        ),
    )
    .unwrap();
    let (a, _) = cook(Source::MaterialX {
        path: doc,
        material: "M".into(),
        textures,
    });
    let m = material(&a, "u");
    (a, m)
}

/// Cooks a MaterialX document with one `image` node `i` of `kind` reading `file`, under an
/// `open_pbr_surface` with `inputs`.
fn materialx(d: &Path, file: &str, kind: &str, inputs: &str) -> (Archive, Material) {
    let image = format!(
        r#"<image name="i" type="{kind}"><input name="file" type="filename" value="{file}"/></image>"#
    );
    mtlx(d, &image, inputs, TextureRules::default())
}

/// A `UsdUVTexture` `i` of `kind` reading `file` with its `scale` and `bias`.
fn reader(file: &str, kind: &str, [scale, bias]: [&str; 2]) -> String {
    format!(
        r#"<UsdUVTexture name="i" type="{kind}">
    <input name="file" type="filename" value="{file}"/>
    <input name="scale" type="vector4" value="{scale}"/>
    <input name="bias" type="vector4" value="{bias}"/>
    <input name="wrapS" type="string" value="repeat"/>
    <input name="wrapT" type="string" value="repeat"/>
  </UsdUVTexture>"#
    )
}

/// Cooks a MaterialX document whose `open_pbr_surface` has `inputs`, over a [`reader`] and any
/// further `nodes`.
fn usd_reader(
    d: &Path,
    file: &str,
    kind: &str,
    pair: [&str; 2],
    nodes: &str,
    inputs: &str,
) -> (Archive, Material) {
    let nodes = format!("{}\n  {nodes}", reader(file, kind, pair));
    mtlx(d, &nodes, inputs, TextureRules::default())
}

/// Cooks a USD quad whose `UsdPreviewSurface` input `surface` connects to output `output` of a
/// `UsdUVTexture` reading `file`, with `reader` lines of its own.
fn usd(
    d: &Path,
    file: &str,
    surface: &str,
    output: &str,
    reader: &str,
) -> (Archive, Material, CookReport) {
    let path = d.join("m.usda");
    std::fs::write(
        &path,
        format!(
            r#"#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Xform "World"
{{
    def Mesh "Quad"
    {{
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </World/Looks/M>
    }}

    def Scope "Looks"
    {{
        def Material "M"
        {{
            token outputs:surface.connect = </World/Looks/M/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                {surface}.connect = </World/Looks/M/Reader.outputs:{output}>
                token outputs:surface
            }}
            def Shader "Reader"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{file}@
                float2 inputs:st.connect = </World/Looks/M/St.outputs:result>
                token inputs:wrapS = "repeat"
                token inputs:wrapT = "repeat"
                {reader}
                float3 outputs:rgb
                float outputs:r
                float outputs:a
            }}
            def Shader "St"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}
    }}
}}
"#
        ),
    )
    .unwrap();
    let (a, report) = cook(Source::Usd {
        path,
        settings: UsdSettings::default(),
    });
    let m = material(&a, "u/material/World/Looks/M");
    (a, m, report)
}

/// The first material of a GLB of `json` reading `image`.
fn gltf(d: &Path, json: &str, image: Vec<u8>) -> (Archive, Material) {
    let path = d.join("m.glb");
    write_glb(&path, &[json.into()], &[image]);
    let (a, _) = cook(Source::Gltf {
        path,
        settings: GltfSettings::default(),
    });
    let m = material(&a, "u/material/0");
    (a, m)
}

/// Format and mip 0 texels of the texture in slot `s`, `n` wide.
fn stored(a: &Archive, m: &Material, s: usize, n: u32) -> (u32, Vec<[f32; 4]>) {
    let e = a.find(m.textures[s].texture).unwrap();
    let (info, levels) = mips(a, a.name(e));
    let format = info.header.vk_format;
    (format, decode(format, n, n, &levels[0][0]))
}

fn f32_rgba() -> ctt::TargetFormat {
    ctt::TargetFormat::Uncompressed(ctt::Format::R32G32B32A32_SFLOAT)
}

fn gray_png(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    let img = image::GrayImage::from_fn(w, h, |x, y| image::Luma([f(x, y)]));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// OpenPBR Emission: `emission_color` ranges over [0, inf), and "the emission_color components
/// may exceed 1, in order to be able to plug in an HDR texture". USD sets no limit on
/// `emissiveColor`. glTF's own emissive range is held by `constants_fold_into_factors_clamped_to_their_domain`.
#[test]
fn hdr_emission_keeps_values_above_1() {
    let d = dir("spec-emission");
    let open_pbr = r#"<input name="emission_color" type="color3" nodename="i"/>
        <input name="emission_luminance" type="float" value="100"/>"#;

    std::fs::write(
        d.join("uniform.ktx2"),
        float_ktx2(8, 8, f32_rgba(), false, |_, _| [5.0, 5.0, 5.0, 1.0]),
    )
    .unwrap();
    let (_, m) = materialx(&d, "uniform.ktx2", "color3", open_pbr);
    assert_eq!(m.emission_color, [5.0; 3]);
    assert_eq!(m.textures[slot::EMISSION].texture, AssetId::NONE);

    let varying = |x: u32, _| [if x < 4 { 5.0 } else { 0.5 }, 0.25, 0.25, 1.0];
    std::fs::write(
        d.join("varying.ktx2"),
        float_ktx2(8, 8, f32_rgba(), false, varying),
    )
    .unwrap();
    let (a, m) = materialx(&d, "varying.ktx2", "color3", open_pbr);
    let (format, texels) = stored(&a, &m, slot::EMISSION, 8);
    assert_eq!(format, vk::BC6H_UFLOAT);
    assert!((texels[0][0] - 5.0).abs() < 0.1, "{:?}", texels[0]);
    assert!((texels[7][0] - 0.5).abs() < 0.02, "{:?}", texels[7]);

    let (a, m, _) = usd(
        &d,
        "varying.ktx2",
        "color3f inputs:emissiveColor",
        "rgb",
        "",
    );
    let (format, texels) = stored(&a, &m, slot::EMISSION, 8);
    assert_eq!(format, vk::BC6H_UFLOAT);
    assert!((texels[0][0] - 5.0).abs() < 0.1, "{:?}", texels[0]);

    // A bias past 1 on 8-bit texels goes to the field, so they keep to BC7.
    let ramp = image::RgbImage::from_fn(8, 8, |x, _| image::Rgb([if x < 4 { 0 } else { 255 }; 3]));
    ramp.save(d.join("ramp.png")).unwrap();
    let pair = ["2, 2, 2, 1", "0.25, 0.25, 0.25, 0"];
    let input = r#"<input name="emission_color" type="color3" nodename="i" output="rgb"/>"#;
    let (a, m) = usd_reader(&d, "ramp.png", "color3", pair, "", input);
    assert_eq!(m.emission_color, [2.25; 3]);
    assert_eq!(stored(&a, &m, slot::EMISSION, 8).0, vk::BC7_SRGB);

    // BC6H blocks that hold every mip are what the texture stores.
    let intel = ctt::encoders::Encoder::Intel(Default::default());
    let format = ctt::TargetFormat::Compressed {
        format: ctt::Format::BC6H_UFLOAT_BLOCK,
        encoder: intel,
    };
    let file = float_ktx2(16, 16, format, true, |x, _| [x as f32, 0.5, 2.0, 1.0]);
    std::fs::write(d.join("blocks.ktx2"), &file).unwrap();
    let (a, m) = materialx(&d, "blocks.ktx2", "color3", open_pbr);
    let e = a.find(m.textures[slot::EMISSION].texture).unwrap();
    let (_, levels) = mips(&a, a.name(e));
    let src = ctt::input::decode_container(&file, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(levels[0].len(), src.surfaces[0].len());
    assert_eq!(levels[0][0][..], src.surfaces[0][0].data[..]);

    // A float image has no 8-bit step: dim texels vary however bright luminance makes them.
    std::fs::write(
        d.join("dim.ktx2"),
        float_ktx2(8, 8, f32_rgba(), false, |x, _| {
            [0.001 * (x + 1) as f32 / 8.0, 0.0, 0.0, 1.0]
        }),
    )
    .unwrap();
    let (_, m) = materialx(&d, "dim.ktx2", "color3", open_pbr);
    assert_ne!(m.textures[slot::EMISSION].texture, AssetId::NONE);
}

/// OpenPBR Specular: the dielectric's `specular_weight` ranges over [0, inf), soft [0, 1], and
/// "we also allow the specular_weight to exceed 1".
#[test]
fn open_pbr_specular_weight_keeps_values_above_1() {
    let d = dir("spec-specular");
    let input = r#"<input name="specular_weight" type="float" nodename="i"/>"#;

    std::fs::write(
        d.join("uniform.ktx2"),
        float_ktx2(8, 8, f32_rgba(), false, |_, _| [1.5; 4]),
    )
    .unwrap();
    let (_, m) = materialx(&d, "uniform.ktx2", "float", input);
    assert_eq!(m.specular_weight, 1.5);
    assert_eq!(m.textures[slot::SPECULAR_WEIGHT].texture, AssetId::NONE);

    std::fs::write(
        d.join("varying.ktx2"),
        float_ktx2(8, 8, f32_rgba(), false, |x, _| {
            [if x < 4 { 1.5 } else { -0.25 }; 4]
        }),
    )
    .unwrap();
    let (a, m) = materialx(&d, "varying.ktx2", "float", input);
    let e = a.find(m.textures[slot::SPECULAR_WEIGHT].texture).unwrap();
    let (info, levels) = mips(&a, a.name(e));
    assert_eq!(info.header.vk_format, vk::R16_SFLOAT);
    let texel = |i: usize| {
        let b = &levels[0][0][2 * i..2 * i + 2];
        half::f16::from_le_bytes([b[0], b[1]]).to_f32()
    };
    assert_eq!(m.specular_weight, 1.0);
    assert_eq!(texel(0), 1.5);
    assert_eq!(texel(7), 0.0, "the range starts at 0");

    let weight = |a: &Archive, m: &Material, i: usize| {
        let e = a.find(m.textures[slot::SPECULAR_WEIGHT].texture).unwrap();
        let (info, levels) = mips(a, a.name(e));
        assert_eq!(info.header.vk_format, vk::R16_SFLOAT);
        let b = &levels[0][0][2 * i..2 * i + 2];
        half::f16::from_le_bytes([b[0], b[1]]).to_f32()
    };
    let input = r#"<input name="specular_weight" type="float" nodename="i" output="r"/>"#;

    // A bias reaches below the texel's floor: 0.1 - 0.25 reads 0 at the range's start, as a
    // constant image folds, and a negative weight has no IOR.
    std::fs::write(
        d.join("low.ktx2"),
        float_ktx2(8, 8, f32_rgba(), false, |x, _| {
            [if x < 4 { 0.1 } else { 1.5 }; 4]
        }),
    )
    .unwrap();
    let pair = ["1, 1, 1, 1", "-0.25, -0.25, -0.25, -0.25"];
    let (a, m) = usd_reader(&d, "low.ktx2", "float", pair, "", input);
    assert_eq!(weight(&a, &m, 0), 0.0);
    assert_eq!(weight(&a, &m, 7), 1.25);
    let b = m.textures[slot::SPECULAR_WEIGHT];
    assert_eq!(b.outside, [0.0; 4], "the floor holds past the edges too");

    // A scale and bias take an 8-bit image past 1 as they do a float one.
    let ramp = image::GrayImage::from_fn(8, 8, |x, _| image::Luma([if x < 4 { 0 } else { 255 }]));
    ramp.save(d.join("ramp.png")).unwrap();
    let pair = ["2, 2, 2, 1", "0.25, 0.25, 0.25, 0"];
    let (a, m) = usd_reader(&d, "ramp.png", "float", pair, "", input);
    assert_eq!(weight(&a, &m, 7), 2.25);
}

/// BC6H: "encoders ... should convert that data to the maximum allowable non-INF representation
/// value, and map NaN to 0 prior to compression". The clause covers BC6H encoders; the mip filter
/// comes before them, so the cook applies it there, and to every format with that format's range.
/// White 64x64 with one bad red texel has 4095 ones and one replaced value, so a stored value `v`
/// leaves the 1x1 at `(4095 + v) / 4096`.
#[test]
fn non_finite_texels_stay_out_of_the_mips() {
    let d = dir("spec-non-finite");
    let hdr = (Usage::Hdr, vk::BC6H_UFLOAT);
    let rows = [
        (hdr, f32::INFINITY, (4095.0 + 65504.0) / 4096.0),
        (hdr, f32::NEG_INFINITY, 4095.0 / 4096.0),
        (hdr, f32::NAN, 4095.0 / 4096.0),
        ((Usage::Color, vk::BC7_SRGB), f32::INFINITY, 1.0),
        ((Usage::Color, vk::BC7_SRGB), f32::NAN, 4095.0 / 4096.0),
        (
            (Usage::Linear4, vk::BC7_UNORM),
            f32::NEG_INFINITY,
            4095.0 / 4096.0,
        ),
        ((Usage::Linear1, vk::BC4_UNORM), f32::INFINITY, 1.0),
        ((Usage::Linear1, vk::BC4_UNORM), f32::NAN, 4095.0 / 4096.0),
    ];
    for ((usage, format), bad, want) in rows {
        for filter in [MipFilter::Kaiser, MipFilter::Box] {
            let path = d.join("bad.ktx2");
            std::fs::write(
                &path,
                float_ktx2(64, 64, f32_rgba(), false, |x, y| {
                    [if (x, y) == (9, 9) { bad } else { 1.0 }, 1.0, 1.0, 1.0]
                }),
            )
            .unwrap();
            let settings = TextureSettings {
                usage,
                filter,
                ..TextureSettings::default()
            };
            let (a, _) = cook(Source::Texture { path, settings });
            let (info, levels) = mips(&a, "u");
            assert_eq!(info.header.vk_format, format);
            let last = levels[0].len() - 1;
            assert_eq!(mip_size(64, 64, last as u32), [1, 1]);
            let texel = decode(format, 1, 1, &levels[0][last])[0];
            // BC6H keeps about 11 bits, BC7 and BC4 about 8.
            let tolerance = (want * 0.02f32).max(0.01);
            assert!(
                (texel[0] - want).abs() < tolerance,
                "{usage:?} {filter:?} {bad}: {texel:?}, want {want}"
            );
            for (i, m) in levels[0].iter().enumerate().skip(1) {
                let [w, h] = mip_size(64, 64, i as u32);
                let px = decode(format, w, h, m);
                assert!(
                    px.iter().all(|p| p[0].is_finite() && p[0] > 0.5),
                    "{usage:?} {filter:?} {bad}: mip {i} went dark"
                );
            }
        }
    }
}

/// BC6H's clause again, where no filter runs: a texture of one mip reaches the encoder as it is.
/// A half float's largest value beside 0 is a step a Kaiser mip overshoots, past what a half holds.
#[test]
fn encoders_hold_what_no_filter_settled() {
    let d = dir("spec-encoders");
    let path = d.join("bad.ktx2");
    std::fs::write(
        &path,
        float_ktx2(16, 16, f32_rgba(), false, |x, _| {
            [if x == 3 { f32::INFINITY } else { 1.0 }, 1.0, 1.0, 1.0]
        }),
    )
    .unwrap();
    let settings = TextureSettings {
        usage: Usage::Hdr,
        mips: Some(false),
        ..TextureSettings::default()
    };
    let (a, _) = cook(Source::Texture { path, settings });
    let (_, levels) = mips(&a, "u");
    let px = decode(vk::BC6H_UFLOAT, 16, 16, &levels[0][0]);
    assert!((px[3][0] - 65504.0).abs() < 1000.0, "{:?}", px[3]);

    let path = d.join("step.ktx2");
    std::fs::write(
        &path,
        float_ktx2(16, 16, f32_rgba(), false, |x, _| {
            [if x < 8 { 65504.0 } else { 0.0 }, 0.0, 0.0, 1.0]
        }),
    )
    .unwrap();
    let settings = TextureSettings {
        usage: Usage::Data(DataFormat::R16f),
        mips: Some(true),
        ..TextureSettings::default()
    };
    let (a, _) = cook(Source::Texture { path, settings });
    let (_, levels) = mips(&a, "u");
    for m in &levels[0] {
        for b in m.chunks_exact(2) {
            let v = half::f16::from_le_bytes([b[0], b[1]]);
            assert!(v.is_finite(), "{v}");
        }
    }
}

/// MaterialX `image`: "the output will contain zero values in all channels beyond the N channels
/// of the image file, aside from the fourth channel which is populated with 1." glTF and USD read
/// a gray file as gray.
#[test]
fn one_channel_files_fill_as_their_importer_says() {
    let d = dir("spec-channels");
    std::fs::write(d.join("flat.png"), gray_png(8, 8, |_, _| 128)).unwrap();
    std::fs::write(d.join("ramp.png"), gray_png(8, 8, |x, _| (x * 32) as u8)).unwrap();
    let base = r#"<input name="base_color" type="color3" nodename="i"/>"#;
    let gray = 128.0 / 255.0;

    let (_, m) = materialx(&d, "flat.png", "color3", base);
    assert_eq!(m.base_color, [gray, 0.0, 0.0]);

    let (a, m) = materialx(&d, "ramp.png", "color3", base);
    let (_, texels) = stored(&a, &m, slot::BASE_COLOR, 8);
    let last = texels[7];
    assert!(last[0] > 0.4, "{last:?}");
    assert!(last[1] < 0.01 && last[2] < 0.01, "{last:?}");
    assert!((last[3] - 1.0).abs() < 0.01, "{last:?}");

    // Gray with alpha is two channels: (gray, alpha, 0, 1).
    let two = image::GrayAlphaImage::from_fn(8, 8, |_, _| image::LumaA([128, 64]));
    two.save(d.join("two.png")).unwrap();
    let (_, m) = materialx(&d, "two.png", "color3", base);
    assert_eq!(m.base_color, [gray, 64.0 / 255.0, 0.0]);

    let srgb = |v: f32| orr_color::srgb_decode(f64::from(v)) as f32;
    let (_, m, _) = usd(
        &d,
        "flat.png",
        "color3f inputs:diffuseColor",
        "rgb",
        r#"token inputs:sourceColorSpace = "raw""#,
    );
    assert_eq!(m.base_color, [gray; 3]);

    let json = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
    let (_, m) = gltf(&d, json, gray_png(8, 8, |_, _| 128));
    assert_eq!(m.base_color, [srgb(gray); 3]);
}

/// USD Preview Surface `UsdUVTexture`: "If a single-channel texture is fed into a UsdUVTexture,
/// the r, g, and b components of the rgb output will repeat the channel's value, while the single
/// a output will be set to 1.0. If a two-channel texture is fed into a UsdUVTexture, the r, g, and
/// b components of the rgb output will repeat the first channel's value, while the single a output
/// will be set to the second channel's value." This holds of every container, where a block or
/// KTX2 format decodes as (r, 0, 0, 1) or (r, g, 0, 1); MaterialX keeps those zeros.
#[test]
fn few_channel_containers_read_as_their_importer_says() {
    let d = dir("spec-container-channels");
    let raw = |f| ctt::TargetFormat::Uncompressed(f);
    let texel = |_: u32, _: u32| [0.5, 0.25, 0.0, 1.0];
    let bc4 = ctt::TargetFormat::Compressed {
        format: ctt::Format::BC4_UNORM_BLOCK,
        encoder: ctt::encoders::Encoder::Amd(Default::default()),
    };
    let files = [("r8.ktx2", raw(ctt::Format::R8_UNORM)), ("bc4.ktx2", bc4)];
    let color = "color3f inputs:diffuseColor";
    let base = r#"<input name="base_color" type="color3" nodename="i"/>"#;
    let raw_space = r#"token inputs:sourceColorSpace = "raw""#;
    for (file, format) in files {
        std::fs::write(d.join(file), float_ktx2(8, 8, format, false, texel)).unwrap();
        let (_, m, _) = usd(&d, file, color, "rgb", raw_space);
        let [r, g, b] = m.base_color;
        assert!(
            (r - 0.5).abs() < 0.01 && r == g && r == b,
            "{file}: {r} {g} {b}"
        );
        let (_, m) = materialx(&d, file, "color3", base);
        assert_eq!(m.base_color[1..], [0.0, 0.0], "{file}");
    }

    std::fs::write(
        d.join("rg8.ktx2"),
        float_ktx2(8, 8, raw(ctt::Format::R8G8_UNORM), false, texel),
    )
    .unwrap();
    let (_, m, _) = usd(&d, "rg8.ktx2", color, "rgb", raw_space);
    let [r, g, b] = m.base_color;
    assert!((r - 0.5).abs() < 0.01 && r == g && r == b, "{r} {g} {b}");
    let (_, m, _) = usd(&d, "rg8.ktx2", "float inputs:opacity", "a", raw_space);
    // Opacity under the default `transparent` mode is 1 - transmission.
    assert!((m.transmission_weight - 0.75).abs() < 0.01, "{m:?}");
    // The second channel is alpha there, which no curve touches.
    let srgb_space = r#"token inputs:sourceColorSpace = "sRGB""#;
    let (_, m, _) = usd(&d, "rg8.ktx2", "float inputs:opacity", "a", srgb_space);
    assert!((m.transmission_weight - 0.75).abs() < 0.01, "{m:?}");
    let varying = |x: u32, _: u32| [0.5, if x < 4 { 0.25 } else { 0.75 }, 0.0, 1.0];
    std::fs::write(
        d.join("rg8_varying.ktx2"),
        float_ktx2(8, 8, raw(ctt::Format::R8G8_UNORM), false, varying),
    )
    .unwrap();
    let (a, m, _) = usd(
        &d,
        "rg8_varying.ktx2",
        "float inputs:opacity",
        "a",
        srgb_space,
    );
    let (_, texels) = stored(&a, &m, slot::TRANSMISSION, 8);
    assert!((texels[0][0] - 0.75).abs() < 0.02, "{:?}", texels[0]);

    // An override names a channel of the file, past what the outputs make of it.
    let channels = InputChannels {
        roughness: Some(1),
        ..Default::default()
    };
    let overrides = vec![(
        "rg8".to_string(),
        TextureOverride {
            channels,
            ..Default::default()
        },
    )];
    let textures = TextureRules {
        overrides,
        ..Default::default()
    };
    let (_, m) = mtlx(
        &d,
        &reader("rg8.ktx2", "float", ["1, 1, 1, 1", "0, 0, 0, 0"]),
        r#"<input name="specular_roughness" type="float" nodename="i" output="r"/>"#,
        textures,
    );
    assert!((m.specular_roughness - 0.25).abs() < 0.01, "{m:?}");

    // A two-channel normal map holds x and y, which repeating x over rgb would lose.
    let tilted = |px: u32, _: u32| [if px < 4 { 0.8 } else { 0.2 }, 0.5, 0.0, 1.0];
    std::fs::write(
        d.join("n.ktx2"),
        float_ktx2(8, 8, raw(ctt::Format::R8G8_UNORM), false, tilted),
    )
    .unwrap();
    let pair = r#"float4 inputs:scale = (2, 2, 2, 1)
                float4 inputs:bias = (-1, -1, -1, 0)"#;
    let (a, m, _) = usd(&d, "n.ktx2", "normal3f inputs:normal", "rgb", pair);
    let (_, texels) = stored(&a, &m, slot::NORMAL, 8);
    assert!((texels[0][1] - 0.5).abs() < 0.02, "{:?}", texels[0]);
}

/// A KTX2 whose key/value data starts with `KTXswizzle` set to `swizzle`, with a padding entry
/// that keeps the levels' alignment.
fn with_swizzle(file: &[u8], swizzle: &str) -> Vec<u8> {
    let entry = |key: &str, value: &[u8]| {
        let mut e = (key.len() as u32 + 1 + value.len() as u32)
            .to_le_bytes()
            .to_vec();
        e.extend_from_slice(key.as_bytes());
        e.push(0);
        e.extend_from_slice(value);
        e.resize(e.len().next_multiple_of(4), 0);
        e
    };
    let mut added = entry("KTXswizzle", format!("{swizzle}\0").as_bytes());
    let padding = added.len().next_multiple_of(16) - added.len();
    if padding >= 8 {
        added.extend(entry("pad", &vec![0; padding - 8]));
    } else if padding > 0 {
        added.extend(entry("padding", &vec![0; padding + 8]));
    }
    let word = |at: usize| u32::from_le_bytes(file[at..at + 4].try_into().unwrap());
    let long = |at: usize| u64::from_le_bytes(file[at..at + 8].try_into().unwrap());
    let (dfd, dfd_len) = (word(48), word(52));
    let (kvd, kvd_len) = (word(56), word(60));
    let at = if kvd_len > 0 { kvd } else { dfd + dfd_len } as usize;
    let levels = word(40).max(1) as usize;
    let mut out = file[..at].to_vec();
    out.extend_from_slice(&added);
    out.extend_from_slice(&file[at..]);
    out[56..60].copy_from_slice(&(at as u32).to_le_bytes());
    out[60..64].copy_from_slice(&(kvd_len + added.len() as u32).to_le_bytes());
    for l in 0..levels {
        let offset = 80 + 24 * l;
        let moved = long(offset) + added.len() as u64;
        out[offset..offset + 8].copy_from_slice(&moved.to_le_bytes());
    }
    let sgd_len = long(72);
    if sgd_len > 0 {
        let moved = long(64) + added.len() as u64;
        out[64..72].copy_from_slice(&moved.to_le_bytes());
    }
    out
}

/// KTX 2.0 "Desired component mapping for a texture can be indicated with the key `KTXswizzle`".
/// The spec lays no duty on loaders; applying it is this project's rule, because the file states
/// it. Rows are the Common Mappings table's Luminance8 and Alpha8, Luminance8Alpha8, and the
/// identity on a two-channel block format, which must keep its blocks.
#[test]
fn ktx_swizzle_is_applied() {
    let d = dir("spec-swizzle");
    let raw_rgba = Usage::Data(DataFormat::Rgba32f);
    let value = 0.5_f32;
    let rows = [
        (ctt::Format::R8_UNORM, "rrr1", [value, value, value, 1.0]),
        (ctt::Format::R8_UNORM, "000r", [0.0, 0.0, 0.0, value]),
        (ctt::Format::R8G8_UNORM, "rrrg", [value, value, value, 0.25]),
    ];
    for (format, swizzle, want) in rows {
        let format = ctt::TargetFormat::Uncompressed(format);
        let file = float_ktx2(4, 4, format, false, |_, _| [value, 0.25, 0.0, 1.0]);
        let path = d.join("s.ktx2");
        std::fs::write(&path, with_swizzle(&file, swizzle)).unwrap();
        let settings = TextureSettings {
            usage: raw_rgba,
            ..TextureSettings::default()
        };
        let (a, _) = cook(Source::Texture { path, settings });
        let (_, levels) = mips(&a, "u");
        let got: Vec<f32> = levels[0][0][..16]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 0.01, "{swizzle}: {got:?}, want {want:?}");
        }
    }

    let amd = ctt::encoders::Encoder::Amd(Default::default());
    let file = float_ktx2(
        16,
        16,
        ctt::TargetFormat::Compressed {
            format: ctt::Format::BC5_UNORM_BLOCK,
            encoder: amd,
        },
        true,
        |x, y| [x as f32 / 16.0, y as f32 / 16.0, 0.0, 1.0],
    );
    let path = d.join("identity.ktx2");
    std::fs::write(&path, with_swizzle(&file, "rg01")).unwrap();
    let settings = TextureSettings {
        usage: Usage::Linear2,
        ..TextureSettings::default()
    };
    let (a, _) = cook(Source::Texture { path, settings });
    let (_, levels) = mips(&a, "u");
    let src = ctt::input::decode_container(&file, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(levels[0][0][..], src.surfaces[0][0].data[..]);
}

/// Decodes one 4x4 block file through the cook, as floats.
fn decoded_block(d: &Path, format: u32, block: &[u8]) -> Vec<[f32; 4]> {
    let mut f = float_ktx2(
        4,
        4,
        ctt::TargetFormat::Compressed {
            format: ctt::Format::BC4_UNORM_BLOCK,
            encoder: ctt::encoders::Encoder::Amd(Default::default()),
        },
        false,
        |_, _| [0.0; 4],
    );
    // Reuses the file's layout, and its DFD, which the cook doesn't read for block formats.
    let level = u64::from_le_bytes(f[80..88].try_into().unwrap()) as usize;
    f.truncate(level);
    f.extend_from_slice(block);
    f[12..16].copy_from_slice(&format.to_le_bytes());
    f[88..96].copy_from_slice(&(block.len() as u64).to_le_bytes());
    f[96..104].copy_from_slice(&(block.len() as u64).to_le_bytes());
    let path = d.join("block.ktx2");
    std::fs::write(&path, f).unwrap();
    let settings = TextureSettings {
        usage: Usage::Data(DataFormat::Rgba32f),
        ..TextureSettings::default()
    };
    let (a, _) = cook(Source::Texture { path, settings });
    let (_, levels) = mips(&a, "u");
    levels[0][0]
        .chunks_exact(16)
        .map(|t| std::array::from_fn(|c| f32::from_le_bytes(t[c * 4..][..4].try_into().unwrap())))
        .collect()
}

/// RGTC "(6*RED0+RED1)/7, if red0 > red1 and code(x,y) == 2", and S3TC "RGB2 = (2*RGB0 + RGB1)/3":
/// real-valued, with no rounding to 8 bits.
#[test]
fn block_interpolants_are_real_valued() {
    let d = dir("spec-blocks");
    // Every texel has code 2.
    let code2 = [0x92u8, 0x24, 0x49, 0x92, 0x24, 0x49];
    let mut bc4 = vec![127u8, 0x81];
    bc4.extend_from_slice(&code2);
    let texels = decoded_block(&d, vk::BC4_SNORM, &bc4);
    assert!((texels[0][0] - 5.0 / 7.0).abs() < 1e-5, "{:?}", texels[0]);

    let mut bc5 = vec![100u8, 10];
    bc5.extend_from_slice(&code2);
    bc5.extend_from_slice(&[255, 0]);
    bc5.extend_from_slice(&code2);
    let texels = decoded_block(&d, vk::BC5_UNORM, &bc5);
    let want = (6.0 * 100.0 + 10.0) / 7.0 / 255.0;
    assert!((texels[0][0] - want).abs() < 1e-5, "{:?}", texels[0]);
    assert!((texels[0][1] - 6.0 / 7.0).abs() < 1e-5, "{:?}", texels[0]);

    // Red 1 of 31 and 0, every index 2.
    let bc1 = [0x00u8, 0x08, 0x00, 0x00, 0xaa, 0xaa, 0xaa, 0xaa];
    let texels = decoded_block(&d, vk::BC1_RGB_UNORM, &bc1);
    let want = 2.0 / 3.0 / 31.0;
    assert!((texels[0][0] - want).abs() < 1e-5, "{:?}", texels[0]);
}

/// The vectors `x`, 0 and 0.8 stored in `format` as snorm values.
fn signed_normals(format: ctt::Format) -> Vec<u8> {
    float_ktx2(
        16,
        16,
        ctt::TargetFormat::Uncompressed(format),
        false,
        |px, _| [if px < 8 { 0.6 } else { -0.6 }, 0.0, 0.8, 1.0],
    )
}

fn normal_hash(a: &Archive, m: &Material) -> orr_pack::Hash32 {
    a.find(m.textures[slot::NORMAL].texture)
        .unwrap()
        .content_hash
}

/// An image's domain is certain where its format states it. USD Preview Surface `normal`: "If the
/// texture has 8 bits per component, then scale and bias must be adjusted": an unsigned image can't
/// hold a signed vector, so with no scale or bias it reads as `2v - 1`, with a warning. A snorm image
/// already holds the vector, so USD's (2, -1) and a MaterialX `normalmap` read it as stored.
#[test]
fn normal_encodings_yield_to_what_the_image_can_hold() {
    let d = dir("spec-normals");
    let surface = "normal3f inputs:normal";
    let unsigned = |x: u32, _: u32| {
        let v = if x < 8 { 204 } else { 51 };
        [v, 128, 230, 255]
    };
    let png = image::RgbaImage::from_fn(16, 16, |x, y| image::Rgba(unsigned(x, y)));
    png.save(d.join("n.png")).unwrap();
    let decode_pair = r#"float4 inputs:scale = (2, 2, 2, 1)
                float4 inputs:bias = (-1, -1, -1, 0)"#;

    let (a, m, report) = usd(&d, "n.png", surface, "rgb", "");
    let warnings: Vec<&String> = report.assets.iter().flat_map(|e| &e.warnings).collect();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("n.png"), "{warnings:?}");
    let unscaled = normal_hash(&a, &m);
    let (a, m, report) = usd(&d, "n.png", surface, "rgb", decode_pair);
    assert_eq!(normal_hash(&a, &m), unscaled);
    assert!(report.assets.iter().all(|e| e.warnings.is_empty()));

    let snorm = signed_normals(ctt::Format::R8G8B8A8_SNORM);
    std::fs::write(d.join("s.ktx2"), snorm).unwrap();
    let (a, m, _) = usd(&d, "s.ktx2", surface, "rgb", decode_pair);
    let stated = normal_hash(&a, &m);
    let (a, m, _) = usd(&d, "s.ktx2", surface, "rgb", "");
    assert_eq!(normal_hash(&a, &m), stated, "as stored either way");

    let flip = r#"float4 inputs:scale = (-2, 2, 2, 1)
                float4 inputs:bias = (1, -1, -1, 0)"#;
    let (a, m, _) = usd(&d, "s.ktx2", surface, "rgb", flip);
    let (_, texels) = stored(&a, &m, slot::NORMAL, 16);
    assert!(texels[0][0] < 0.4, "the flip still holds: {:?}", texels[0]);

    let doc = d.join("n.mtlx");
    std::fs::write(
        &doc,
        r#"<?xml version="1.0"?>
<materialx version="1.39">
  <image name="i" type="vector3"><input name="file" type="filename" value="s.ktx2"/></image>
  <normalmap name="n" type="vector3"><input name="in" type="vector3" nodename="i"/></normalmap>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="geometry_normal" type="vector3" nodename="n"/>
  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
  </surfacematerial>
</materialx>
"#,
    )
    .unwrap();
    let (a, _) = cook(Source::MaterialX {
        path: doc,
        material: "M".into(),
        textures: TextureRules::default(),
    });
    assert_eq!(normal_hash(&a, &material(&a, "u")), stated);

    // A `normalmap` states the encoding a `UsdUVTexture` leaves open, for a float image too.
    let encoded = |px: u32, _: u32| [if px < 8 { 0.8 } else { 0.2 }, 0.5, 0.9, 1.0];
    let float = float_ktx2(16, 16, f32_rgba(), false, encoded);
    std::fs::write(d.join("f.ktx2"), float).unwrap();
    let mapped = |reader: &str, output: &str| {
        let doc = d.join("f.mtlx");
        std::fs::write(
            &doc,
            format!(
                r#"<?xml version="1.0"?>
<materialx version="1.39">
  <{reader} name="i" type="vector3"><input name="file" type="filename" value="f.ktx2"/><input name="wrapS" type="string" value="repeat"/><input name="wrapT" type="string" value="repeat"/></{reader}>
  <normalmap name="n" type="vector3"><input name="in" type="vector3" nodename="i"{output}/></normalmap>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="geometry_normal" type="vector3" nodename="n"/>
  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
  </surfacematerial>
</materialx>
"#
            ),
        )
        .unwrap();
        let (a, _) = cook(Source::MaterialX {
            path: doc,
            material: "M".into(),
            textures: TextureRules::default(),
        });
        normal_hash(&a, &material(&a, "u"))
    };
    assert_eq!(
        mapped("UsdUVTexture", r#" output="rgb""#),
        mapped("image", "")
    );

    // A `normalmap` over a reader means `2 (s t + b) - 1`: scale 0.5 and bias 0.25 on x is
    // `t - 0.5`, as USD states it with scale 1 and bias -0.5. A stated (2, -1) is the encoding
    // itself, not `4 t - 3`.
    let surface_input = r#"<input name="geometry_normal" type="vector3" nodename="n"/>"#;
    let map = r#"<normalmap name="n" type="vector3"><input name="in" type="vector3" nodename="i" output="rgb"/></normalmap>"#;
    let read = |pair: [&str; 2]| {
        let (a, m) = usd_reader(&d, "f.ktx2", "vector3", pair, map, surface_input);
        normal_hash(&a, &m)
    };
    let half_pair = read(["0.5, 1, 1, 1", "0.25, 0, 0, 0"]);
    let (a, m, _) = usd(
        &d,
        "f.ktx2",
        surface,
        "rgb",
        r#"float4 inputs:scale = (1, 2, 2, 1)
                float4 inputs:bias = (-0.5, -1, -1, 0)"#,
    );
    assert_eq!(half_pair, normal_hash(&a, &m));
    assert_ne!(half_pair, mapped("image", ""));
    assert_eq!(
        read(["2, 2, 2, 1", "-1, -1, -1, 0"]),
        mapped("image", ""),
        "the encoding, stated"
    );
}

/// BC6H_UFLOAT, B10G11R11 and E5B9G9R9 store floats no lower than 0, so like an unsigned integer
/// image they can't hold a signed vector: with no scale or bias they read as `2 v - 1`.
#[test]
fn unsigned_floats_cant_hold_normals_as_they_are() {
    let d = dir("spec-ufloat-normals");
    let encoded = |px: u32, _: u32| [if px < 8 { 0.8 } else { 0.2 }, 0.5, 0.9, 1.0];
    let format = ctt::TargetFormat::Compressed {
        format: ctt::Format::BC6H_UFLOAT_BLOCK,
        encoder: ctt::encoders::Encoder::Intel(Default::default()),
    };
    std::fs::write(d.join("n.ktx2"), float_ktx2(16, 16, format, false, encoded)).unwrap();
    let surface = "normal3f inputs:normal";
    let (a, m, report) = usd(&d, "n.ktx2", surface, "rgb", "");
    assert_eq!(
        report.assets.iter().flat_map(|e| &e.warnings).count(),
        1,
        "{report:?}"
    );
    let unscaled = normal_hash(&a, &m);
    let pair = r#"float4 inputs:scale = (2, 2, 2, 1)
                float4 inputs:bias = (-1, -1, -1, 0)"#;
    let (a, m, _) = usd(&d, "n.ktx2", surface, "rgb", pair);
    assert_eq!(unscaled, normal_hash(&a, &m));
}

/// A file's flat color read back: the id of its primaries and the texel at (0, 0) and (4, 0).
fn read_back(d: &Path, bytes: &[u8]) -> (u8, [[f32; 4]; 2]) {
    let path = d.join("image");
    std::fs::write(&path, bytes).unwrap();
    let settings = TextureSettings {
        usage: Usage::Hdr,
        mips: Some(false),
        ..Default::default()
    };
    let (a, _) = cook(Source::Texture { path, settings });
    let (info, levels) = mips(&a, "u");
    let (w, h) = (info.header.width, info.header.height);
    let texels = decode(info.header.vk_format, w, h, &levels[0][0]);
    (info.header.primaries, [texels[0], texels[4]])
}

/// Color past Rec.2020 clips, so a file of primaries that reach there warns, once, where it is
/// read as color.
#[test]
fn primaries_past_rec2020_warn_where_read_as_color() {
    let d = dir("past-rec2020");
    let path = d.join("image");
    let wide = moxcms::ColorProfile::new_aces_aces_2065_1_linear();
    let narrow = moxcms::ColorProfile::new_display_p3();
    for (profile, usage, want) in [
        (&wide, Usage::Hdr, 1),
        (&wide, Usage::Linear4, 0),
        (&narrow, Usage::Hdr, 0),
    ] {
        let icc = profile.encode().unwrap();
        let red = [255u8, 0, 0].repeat(64);
        let image = tagged::png(tagged::RGB8, [8, 8], &red, Some(&icc), &[]);
        std::fs::write(&path, image).unwrap();
        let settings = TextureSettings {
            usage,
            ..Default::default()
        };
        let (_, report) = cook(Source::Texture {
            path: path.clone(),
            settings,
        });
        let entries = report.assets.iter().flat_map(|a| &a.entries);
        let warnings: Vec<&String> = entries.flat_map(|e| &e.warnings).collect();
        assert_eq!(warnings.len(), want, "{usage:?}: {warnings:?}");
    }
}

/// What converting to Rec.2020 leaves below 0 clips, and only that warns: `[-0.5, 1, 0]` in Rec.709
/// is inside Rec.2020, and `[-1, 1, 0]` is not.
#[test]
fn color_below_zero_warns_past_rec2020() {
    let d = dir("below-zero");
    let path = d.join("image.exr");
    let warnings = |texel: [f32; 3], usage| {
        let img = image::Rgb32FImage::from_fn(8, 8, |_, _| image::Rgb(texel));
        image::DynamicImage::ImageRgb32F(img)
            .save_with_format(&path, image::ImageFormat::OpenExr)
            .unwrap();
        let settings = TextureSettings {
            usage,
            ..Default::default()
        };
        let (_, report) = cook(Source::Texture {
            path: path.clone(),
            settings,
        });
        let entries = report.assets.iter().flat_map(|a| &a.entries);
        entries.flat_map(|e| &e.warnings).count()
    };
    assert_eq!(warnings([-0.5, 1.0, 0.0], Usage::Hdr), 0);
    assert_eq!(warnings([-1.0, 1.0, 0.0], Usage::Hdr), 1);
    assert_eq!(warnings([-1.0, 1.0, 0.0], Usage::Linear4), 0);
    assert_eq!(warnings([0.5, 1.0, 0.0], Usage::Hdr), 0);
}

/// Linear `rgb` in primaries `from` as primaries `to`.
fn converted(
    from: orr_color::RgbPrimaries,
    to: orr_color::RgbPrimaries,
    rgb: [f32; 3],
) -> [f32; 3] {
    let rgb = orr_math::DVec3::from(rgb.map(f64::from));
    (from.matrix_to(to) * rgb).as_vec3().to_array()
}

/// `rgb` in `primaries` as linear Rec.2020.
fn in_rec2020(primaries: orr_color::RgbPrimaries, rgb: [f32; 3]) -> [f32; 3] {
    converted(primaries, orr_color::RgbPrimaries::REC2020, rgb)
}

/// How many warnings say that colors clip.
fn clip_warnings(report: &CookReport) -> usize {
    let entries = report.assets.iter().flat_map(|a| &a.entries);
    let warnings = entries.flat_map(|e| &e.warnings);
    let all = warnings.chain(report.assets.iter().flat_map(|a| &a.warnings));
    all.filter(|w| w.contains("Rec.2020")).count()
}

/// Rec.709's `[-0.3, 1, 0]`, inside Rec.2020.
const BELOW: [f32; 3] = [-0.3, 1.0, 0.0];

fn flat_exr(
    [r, g, b]: [f32; 3],
    edit: impl FnOnce(&mut exr::meta::header::LayerAttributes, &mut exr::meta::header::ImageAttributes),
) -> Vec<u8> {
    exr_image(|_| (r, g, b, 1.0), edit)
}

fn display_p3() -> exr::meta::attribute::Chromaticities {
    use exr::prelude::Vec2;
    exr::meta::attribute::Chromaticities {
        red: Vec2(0.68, 0.32),
        green: Vec2(0.265, 0.69),
        blue: Vec2(0.15, 0.06),
        white: Vec2(0.3127, 0.329),
    }
}

/// This project's rule: a file is what it states unless the image can't be. No texture holds a
/// value below 0, so a float image in narrower primaries converts to Rec.2020, the widest id.
#[test]
fn a_texture_file_below_zero_converts_to_rec2020() {
    use exr::meta::attribute::{AttributeValue, Text};
    use orr_color::RgbPrimaries as P;
    let d = dir("below-zero-file");
    let (id, [t, _]) = read_back(&d, &flat_exr(BELOW, |_, _| {}));
    assert_eq!(id, 2);
    assert_rgb(t, in_rec2020(P::REC709, BELOW), "Rec.709");

    let p3 = |_: &mut _, image: &mut exr::meta::header::ImageAttributes| {
        image.chromaticities = Some(display_p3());
    };
    let below_p3 = [-0.2, 1.0, 0.0];
    let (id, [t, _]) = read_back(&d, &flat_exr(below_p3, p3));
    assert_eq!(id, 2);
    assert_rgb(t, in_rec2020(P::DISPLAY_P3, below_p3), "Display P3");
    assert_eq!(read_back(&d, &flat_exr([0.3, 1.0, 0.0], p3)).0, 1);

    // A narrow-range file's code 16 is black, and its codes below that are not.
    let linear_709: &[u8] = &[1, 8, 0, 0];
    let narrow = |r: u8| tagged::flat_rgb([r, 235, 16], &[(b"cICP", linear_709)]);
    assert_eq!(read_back(&d, &narrow(16)).0, 0);
    let (id, [t, _]) = read_back(&d, &narrow(10));
    assert_eq!(id, 2);
    assert_rgb(t, in_rec2020(P::REC709, [-6.0 / 219.0, 1.0, 0.0]), "narrow");

    // Rec.2020 has nowhere wider to go.
    let rec2020 = flat_exr(BELOW, |layer, _| {
        layer.other.insert(
            Text::from("colorInteropID"),
            AttributeValue::Text(Text::from("lin_rec2020_scene")),
        );
    });
    let (id, [t, _]) = read_back(&d, &rec2020);
    assert_eq!(id, 2);
    assert_rgb(t, [0.0, 1.0, 0.0], "Rec.2020");

    // An override takes the texels as they are, so what they can't hold clips.
    let path = d.join("image");
    std::fs::write(&path, flat_exr(BELOW, |_, _| {})).unwrap();
    let settings = TextureSettings {
        usage: Usage::Hdr,
        mips: Some(false),
        primaries: Some(orr_color::Primaries::Rec709),
        ..Default::default()
    };
    let (a, report) = cook(Source::Texture { path, settings });
    let (info, levels) = mips(&a, "u");
    assert_eq!(info.header.primaries, 0);
    let texels = decode(info.header.vk_format, 8, 8, &levels[0][0]);
    assert_rgb(texels[0], [0.0, 1.0, 0.0], "override");
    assert_eq!(clip_warnings(&report), 1);
}

/// The rule above of an environment map, which reads through its own path: its texels and the
/// primaries the lights take from it.
#[test]
fn an_environment_below_zero_converts_to_rec2020() {
    let d = dir("below-zero-environment");
    let path = d.join("sky.exr");
    let usage = Usage::Environment {
        light_threshold: None,
        max_lights: 0,
    };
    let cooked = |texel: [f32; 3]| {
        let img = image::Rgb32FImage::from_fn(16, 8, |_, _| image::Rgb(texel));
        image::DynamicImage::ImageRgb32F(img)
            .save_with_format(&path, image::ImageFormat::OpenExr)
            .unwrap();
        let settings = TextureSettings {
            usage,
            max_size: Some(8),
            ..Default::default()
        };
        let (a, report) = cook(Source::Texture {
            path: path.clone(),
            settings,
        });
        let (info, levels) = mips(&a, "u");
        let [w, h] = [info.header.width, info.header.height];
        let texels = decode(info.header.vk_format, w, h, &levels[0][0]);
        (info.header.primaries, texels[0], clip_warnings(&report))
    };
    let (id, t, warnings) = cooked(BELOW);
    assert_eq!((id, warnings), (2, 0));
    assert_rgb(t, in_rec2020(orr_color::RgbPrimaries::REC709, BELOW), "sky");
    let (id, _, warnings) = cooked([-1.0, 1.0, 0.0]);
    assert_eq!((id, warnings), (2, 1));
    assert_eq!(cooked([0.3, 1.0, 0.0]).0, 0);
}

/// The rule above of a material's color inputs: texels and constants convert alike, a constant's
/// factor is still Rec.709, and a remap that reaches below 0 counts as what the input reads.
#[test]
fn material_colors_below_zero_convert_to_rec2020() {
    use orr_color::RgbPrimaries as P;
    let d = dir("below-zero-material");
    let surface = "color3f inputs:diffuseColor";
    let write = |name: &str, bytes: Vec<u8>| std::fs::write(d.join(name), bytes).unwrap();
    let close = |got: [f32; 3], want: [f32; 3]| {
        assert!(
            (0..3).all(|c| (got[c] - want[c]).abs() < 2e-3),
            "{got:?}, not {want:?}"
        );
    };

    write("flat.exr", flat_exr(BELOW, |_, _| {}));
    let (_, m, report) = usd(&d, "flat.exr", surface, "rgb", "");
    assert_eq!(m.textures[slot::BASE_COLOR].texture, AssetId::NONE);
    close(m.base_color, BELOW);
    assert_eq!(clip_warnings(&report), 0);

    // A stored constant reads through its remap before anything clamps it.
    let remapped = r#"token inputs:sourceColorSpace = "raw"
                float4 inputs:bias = (-0.5, 0, 0, 0)"#;
    write("remapped.png", tagged::flat_rgb([51, 255, 0], &[]));
    let (_, m, report) = usd(&d, "remapped.png", surface, "rgb", remapped);
    close(m.base_color, BELOW);
    assert_eq!(clip_warnings(&report), 0);

    write("past.exr", flat_exr([-1.0, 1.0, 0.0], |_, _| {}));
    let (_, m, report) = usd(&d, "past.exr", surface, "rgb", "");
    let held = in_rec2020(P::REC709, [-1.0, 1.0, 0.0]).map(|v| v.clamp(0.0, 1.0));
    close(m.base_color, converted(P::REC2020, P::REC709, held));
    assert_eq!(clip_warnings(&report), 1);

    // Negative infinity has no color to convert, and clips in the primaries the file states.
    let infinite = |x: usize| (if x == 0 { f32::NEG_INFINITY } else { 0.5 }, 0.5, 0.5, 1.0);
    write("infinite.exr", exr_image(infinite, |_, _| {}));
    let (_, _, report) = usd(&d, "infinite.exr", surface, "rgb", "");
    assert_eq!(clip_warnings(&report), 1);

    let varying = |x: usize| {
        let [r, g, b] = if x < 4 { BELOW } else { [0.5, 0.25, 0.125] };
        (r, g, b, 1.0)
    };
    write("varying.exr", exr_image(varying, |_, _| {}));
    let emissive = "color3f inputs:emissiveColor";
    let (a, m, report) = usd(&d, "varying.exr", emissive, "rgb", "");
    let e = a.find(m.textures[slot::EMISSION].texture).unwrap();
    assert_eq!(mips(&a, a.name(e)).0.header.primaries, 2);
    let (_, texels) = stored(&a, &m, slot::EMISSION, 8);
    assert_rgb(texels[0], in_rec2020(P::REC709, BELOW), "below 0");
    let inside = in_rec2020(P::REC709, [0.5, 0.25, 0.125]);
    assert_rgb(texels[7], inside, "inside Rec.709");
    assert_eq!(clip_warnings(&report), 0);
}

/// A slot holds one primaries id, so whether a set converts is one answer for all of its tiles: a
/// tile with nothing below 0 converts beside one that has, or the two would read in different
/// primaries. And a set is one image to the report, however many tiles clip.
#[test]
fn a_tile_set_converts_and_warns_as_one_image() {
    use orr_color::RgbPrimaries as P;
    use orr_pack::tiles::TextureTiles;
    let d = dir("below-zero-tiles");
    let write = |name: &str, bytes: Vec<u8>| std::fs::write(d.join(name), bytes).unwrap();
    let inside = [0.5, 0.25, 0.125];
    let surface = "color3f inputs:emissiveColor";

    let varying = |x: usize| {
        let [r, g, b] = if x < 4 { BELOW } else { inside };
        (r, g, b, 1.0)
    };
    write("mixed.1001.exr", exr_image(varying, |_, _| {}));
    write("mixed.1002.exr", flat_exr(inside, |_, _| {}));
    let (a, m, report) = usd(&d, "mixed.<UDIM>.exr", surface, "rgb", "");
    let set = a.find(m.textures[slot::EMISSION].texture).unwrap();
    let b = &a.blobs(set)[0];
    let mut bytes = vec![0; b.raw_len as usize];
    a.read(b, &mut bytes).unwrap();
    let tiles = TextureTiles::parse(&bytes).unwrap().tiles;
    let read = |t: usize| {
        let e = a.find(tiles[t].texture).unwrap();
        let (info, levels) = mips(&a, a.name(e));
        let h = info.header;
        (h.primaries, decode(h.vk_format, 8, 8, &levels[0][0]))
    };
    let (primaries, texels) = read(0);
    assert_eq!(primaries, 2);
    assert_rgb(texels[0], in_rec2020(P::REC709, BELOW), "below 0");
    let (primaries, texels) = read(1);
    assert_eq!(primaries, 2, "the tile with nothing below 0 converts too");
    assert_rgb(texels[0], in_rec2020(P::REC709, inside), "inside Rec.709");
    assert_eq!(clip_warnings(&report), 0);

    write("past.1001.exr", flat_exr([-1.0, 1.0, 0.0], |_, _| {}));
    write("past.1002.exr", flat_exr([-1.0, 1.0, 0.0], |_, _| {}));
    let (_, _, report) = usd(&d, "past.<UDIM>.exr", surface, "rgb", "");
    assert_eq!(clip_warnings(&report), 1);
    let mut warnings = report.assets.iter().flat_map(|a| &a.warnings);
    let warning = warnings.find(|w| w.contains("Rec.2020")).unwrap();
    assert!(
        warning.contains("past.<UDIM>.exr") && !warning.contains("more"),
        "{warning}"
    );
}

/// A warning follows what an image holds, not what its primaries could reach: gray in AP0 reads
/// inside Rec.2020, and a saturated red doesn't, through either path.
#[test]
fn primaries_past_rec2020_warn_of_the_texels_that_clip() {
    use exr::prelude::Vec2;
    let d = dir("past-rec2020-texels");
    let ap0 = exr::meta::attribute::Chromaticities {
        red: Vec2(0.7347, 0.2653),
        green: Vec2(0.0, 1.0),
        blue: Vec2(0.0001, -0.077),
        white: Vec2(0.32168, 0.33767),
    };
    let path = d.join("ap0.exr");
    for (rgb, want) in [([0.5; 3], 0), ([1.0, 0.0, 0.0], 1)] {
        let file = flat_exr(rgb, |_, image| image.chromaticities = Some(ap0));
        std::fs::write(&path, file).unwrap();
        let settings = TextureSettings {
            usage: Usage::Hdr,
            ..Default::default()
        };
        let (_, report) = cook(Source::Texture {
            path: path.clone(),
            settings,
        });
        assert_eq!(clip_warnings(&report), want, "{rgb:?} as a file");
        let (_, _, report) = usd(&d, "ap0.exr", "color3f inputs:diffuseColor", "rgb", "");
        assert_eq!(clip_warnings(&report), want, "{rgb:?} as a material");
    }
}

/// A signed BC6H block of one color: mode 10, whose endpoints hold 10 bits, all indices 0.
fn bc6h_block(rgb: [i32; 3]) -> [u8; 16] {
    let [r, g, b] = rgb.map(|v| (v as u128) & 0x3ff);
    (0b00011 | r << 5 | g << 15 | b << 25).to_le_bytes()
}

/// The rule above holds of every mip a file stores, which a cook can keep: here only the mips past
/// the first read below 0.
#[test]
fn stored_mips_below_zero_convert_to_rec2020() {
    let d = dir("below-zero-mips");
    let positive = bc6h_block([40, 40, 40]);
    let negative = bc6h_block([-40, 40, 0]);
    let data = [positive, negative, negative].concat();
    const DXGI_BC6H_SF16: u32 = 96;
    let mut file = dds(b"DX10", Some((DXGI_BC6H_SF16, 0)), [4, 4], &data);
    file[8..12].copy_from_slice(&(0x1007u32 | 0x20000).to_le_bytes());
    file[28..32].copy_from_slice(&3u32.to_le_bytes());
    assert_eq!(read_back(&d, &file).0, 2, "as a file");

    std::fs::write(d.join("mips.dds"), &file).unwrap();
    let (a, m, _) = usd(&d, "mips.dds", "color3f inputs:emissiveColor", "rgb", "");
    let e = a.find(m.textures[slot::EMISSION].texture).unwrap();
    assert_eq!(mips(&a, a.name(e)).0.header.primaries, 2, "as a material");

    // A BC1 block of one color, in a file whose remap takes its mips below 0.
    let bc1_block =
        |color: u16| [color.to_le_bytes(), color.to_le_bytes(), [0; 2], [0; 2]].concat();
    let data = [bc1_block(0xffff), bc1_block(0), bc1_block(0)].concat();
    const DXGI_BC1_UNORM: u32 = 71;
    let mut file = dds(b"DX10", Some((DXGI_BC1_UNORM, 0)), [4, 4], &data);
    file[8..12].copy_from_slice(&(0x1007u32 | 0x20000).to_le_bytes());
    file[28..32].copy_from_slice(&3u32.to_le_bytes());
    std::fs::write(d.join("remapped.dds"), &file).unwrap();
    let remap = r#"token inputs:sourceColorSpace = "raw"
                float4 inputs:bias = (-0.5, 0, 0, 0)"#;
    let (a, m, _) = usd(
        &d,
        "remapped.dds",
        "color3f inputs:emissiveColor",
        "rgb",
        remap,
    );
    let e = a.find(m.textures[slot::EMISSION].texture).unwrap();
    assert_eq!(mips(&a, a.name(e)).0.header.primaries, 2, "through a remap");
}

/// A curve can read an unsigned file's black below 0, and what the file stores says if it does.
#[test]
fn a_curve_below_zero_at_black_converts_what_reads_there() {
    const KHR_DF_TRANSFER_SLOG: u8 = 5;
    let d = dir("below-zero-curve");
    let slog = |[r, g, b]: [f32; 3]| {
        let format = ctt::TargetFormat::Uncompressed(ctt::Format::R8G8B8A8_UNORM);
        let mut file = float_ktx2(8, 8, format, false, |_, _| [r, g, b, 1.0]);
        let dfd = u32::from_le_bytes(file[48..52].try_into().unwrap()) as usize;
        file[dfd + 14] = KHR_DF_TRANSFER_SLOG;
        file
    };
    assert_eq!(read_back(&d, &slog([0.5; 3])).0, 0);
    assert_eq!(read_back(&d, &slog([0.0, 0.5, 0.5])).0, 2);

    let path = d.join("black");
    std::fs::write(&path, slog([0.0; 3])).unwrap();
    let settings = TextureSettings {
        usage: Usage::Hdr,
        mips: Some(false),
        ..Default::default()
    };
    let (_, report) = cook(Source::Texture { path, settings });
    assert_eq!(clip_warnings(&report), 1);
}

fn assert_rgb(got: [f32; 4], want: [f32; 3], what: &str) {
    let close = (0..3).all(|c| (got[c] - want[c]).abs() <= 0.012 * want[c].abs().max(0.25));
    assert!(close, "{what}: {got:?}, not {want:?}");
}

fn linear(code: f64) -> f32 {
    orr_color::srgb_decode(code / 255.0) as f32
}

/// `file`, a KTX2, with its DFD's `colorPrimaries`, `transferFunction` and `flags`.
fn with_dfd(file: &[u8], primaries: u8, transfer: u8, flags: u8) -> Vec<u8> {
    let mut out = file.to_vec();
    let dfd = u32::from_le_bytes(file[48..52].try_into().unwrap()) as usize;
    out[dfd + 13..dfd + 16].copy_from_slice(&[primaries, transfer, flags]);
    out
}

/// KTX 2.0 §3.10: the DFD's `colorPrimaries` and `transferFunction`. The spec's notes say which
/// files are valid and lay no duty on loaders; honoring them is this project's rule for files
/// that state them. Display P3 keeps its id, ACES AP0 has none and converts to Rec.2020, and
/// A2B10G10R10, which has no sRGB format, "may be used with `KHR_DF_TRANSFER_SRGB`".
#[test]
fn ktx2_dfd_states_primaries_and_transfer() {
    const P3: u8 = 10;
    const ACES: u8 = 6;
    const SRGB: u8 = 2;
    let d = dir("ktx2_dfd");
    let flat = |format| {
        float_ktx2(
            8,
            8,
            ctt::TargetFormat::Uncompressed(format),
            false,
            |_, _| [0.5, 0.25, 0.125, 1.0],
        )
    };
    let rgba = flat(ctt::Format::R16G16B16A16_UNORM);
    let (id, [t, _]) = read_back(&d, &with_dfd(&rgba, P3, 1, 0));
    assert_eq!(id, 1);
    assert_rgb(t, [0.5, 0.25, 0.125], "P3 stays as stored");

    let (id, [t, _]) = read_back(&d, &with_dfd(&rgba, ACES, 1, 0));
    assert_eq!(id, 2);
    let want = orr_color::RgbPrimaries::ACES_AP0.matrix_to(orr_color::RgbPrimaries::REC2020)
        * orr_math::DVec3::new(0.5, 0.25, 0.125);
    assert_rgb(t, want.as_vec3().to_array(), "AP0 converts");

    let packed = flat(ctt::Format::A2B10G10R10_UNORM_PACK32);
    let (_, [t, _]) = read_back(&d, &with_dfd(&packed, 1, SRGB, 0));
    let decoded = [0.5, 0.25, 0.125].map(|v| orr_color::srgb_decode(v) as f32);
    assert_rgb(t, decoded, "sRGB transfer on a format without an sRGB form");
}

/// A KTX2 of 8 x 8 BC7 sRGB blocks, every mip, of stored texels `f(x, y)`.
fn bc7_srgb(f: impl Fn(u32, u32) -> [f32; 4]) -> Vec<u8> {
    let bc7 = ctt::TargetFormat::Compressed {
        format: ctt::Format::BC7_UNORM_BLOCK,
        encoder: ctt::encoders::Encoder::Bc7enc(Default::default()),
    };
    let mut file = float_ktx2(8, 8, bc7, true, f);
    file[12..16].copy_from_slice(&vk::BC7_SRGB.to_le_bytes());
    file
}

/// What a block file states reaches materials through the scan of its blocks too, and a file's
/// Display P3 is the file's alone: USD's `auto` takes it and glTF ignores it (L2418), keeping the
/// blocks either way.
#[test]
fn ktx2_blocks_state_primaries_to_importers_that_read_them() {
    let d = dir("ktx2_blocks_primaries");
    let file = bc7_srgb(|x, y| [x as f32 / 8.0, y as f32 / 8.0, 0.5, 1.0]);
    let file = with_dfd(&file, 10, 2, 0);
    std::fs::write(d.join("p3.ktx2"), &file).unwrap();
    let (a, m, _) = usd(&d, "p3.ktx2", "color3f inputs:diffuseColor", "rgb", "");
    let e = a.find(m.textures[slot::BASE_COLOR].texture).unwrap();
    assert_eq!(mips(&a, a.name(e)).0.header.primaries, 1);

    let material = r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}"#;
    let (a, m) = gltf(&d, material, file);
    let e = a.find(m.textures[slot::BASE_COLOR].texture).unwrap();
    assert_eq!(mips(&a, a.name(e)).0.header.primaries, 0);
}

/// A DDS of `w` x `h` `data`: legacy FourCC `four_cc`, or with `dx10` the DXGI format and alpha
/// mode of a `DDS_HEADER_DXT10`.
fn dds(four_cc: &[u8; 4], dx10: Option<(u32, u32)>, [w, h]: [u32; 2], data: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8; 128];
    let mut put = |at: usize, v: u32| f[at..at + 4].copy_from_slice(&v.to_le_bytes());
    put(4, 124);
    put(8, 0x1007);
    put(12, h);
    put(16, w);
    put(76, 32);
    put(80, 4);
    put(108, 0x1000);
    f[..4].copy_from_slice(b"DDS ");
    f[84..88].copy_from_slice(four_cc);
    if let Some((format, alpha)) = dx10 {
        for v in [format, 3, 0, 1, alpha] {
            f.extend_from_slice(&v.to_le_bytes());
        }
    }
    f.extend_from_slice(data);
    f
}

/// An EXR of 8 x 8 texels `f(x)`.
fn exr_image(
    f: impl Fn(usize) -> (f32, f32, f32, f32) + Sync,
    edit: impl FnOnce(&mut exr::meta::header::LayerAttributes, &mut exr::meta::header::ImageAttributes),
) -> Vec<u8> {
    use exr::image::write::WritableImage;
    use exr::prelude::{Image, SpecificChannels, Vec2};
    let channels = SpecificChannels::rgba(|Vec2(x, _): Vec2<usize>| f(x));
    let mut image = Image::from_channels((8, 8), channels);
    edit(&mut image.layer_data.attributes, &mut image.attributes);
    let mut out = std::io::Cursor::new(Vec::new());
    image.write().non_parallel().to_buffered(&mut out).unwrap();
    out.into_inner()
}

/// The four places a file says its color is premultiplied: KTX 2.0 `KHR_DF_FLAG_ALPHA_PREMULTIPLIED`
/// ("the images' RGB components have been multiplied by their alpha components"),
/// `DDS_ALPHA_MODE_PREMULTIPLIED`, the legacy DXT2 and DXT4 ("The only legacy file formats that
/// indicate this information are 'DX2' and 'DX4'"), and OpenEXR ("By convention, all color channels
/// are premultiplied by alpha"). The material model's colors are straight, and the division is in
/// linear light. OpenEXR: "Premultiplied does not mean that pixels with zero alpha and non-zero
/// color channels are illegal", so alpha 0 keeps what is stored.
#[test]
fn premultiplied_files_read_straight() {
    let d = dir("premultiplied");
    let texels = |x: u32, y: u32, rgba: [u8; 4], zero: [u8; 4]| {
        let _ = y;
        (if x < 4 { rgba } else { zero }).map(|c| f32::from(c) / 255.0)
    };
    let (rgba, zero) = ([128, 64, 32, 128], [128, 0, 0, 0]);
    let file = float_ktx2(
        8,
        8,
        ctt::TargetFormat::Uncompressed(ctt::Format::R8G8B8A8_UNORM),
        false,
        |x, y| texels(x, y, rgba, zero),
    );
    let mut srgb = with_dfd(&file, 1, 2, 1);
    srgb[12..16].copy_from_slice(&43u32.to_le_bytes());
    let a = 128.0 / 255.0;
    let (_, [t, none]) = read_back(&d, &srgb);
    assert_rgb(
        t,
        [linear(128.0) / a, linear(64.0) / a, linear(32.0) / a],
        "KTX2",
    );
    assert_rgb(none, [linear(128.0), 0.0, 0.0], "KTX2 at alpha 0");
    let (_, [t, _]) = read_back(&d, &with_dfd(&file, 1, 1, 0));
    assert_rgb(t, [a, 64.0 / 255.0, 32.0 / 255.0], "straight KTX2");

    let data: Vec<u8> = (0..64u32)
        .flat_map(|i| if i % 8 < 4 { rgba } else { zero })
        .collect();
    let straight = [128.0 / 128.0, 64.0 / 128.0, 32.0 / 128.0];
    let (_, [t, none]) = read_back(&d, &dds(b"DX10", Some((28, 2)), [8, 8], &data));
    assert_rgb(t, straight, "DDS alpha mode");
    assert_rgb(none, [a, 0.0, 0.0], "DDS at alpha 0");
    let (_, [t, _]) = read_back(&d, &dds(b"DX10", Some((28, 1)), [8, 8], &data));
    assert_rgb(
        t,
        [a, 64.0 / 255.0, 32.0 / 255.0],
        "DDS straight alpha mode",
    );

    // One block, alpha 8/15 in DXT2 and 136/255 in DXT4 alike, over c0 = c1 = (16, 32, 8).
    let color = [0x08u8, 0x84, 0x08, 0x84, 0, 0, 0, 0];
    let mut dxt2 = vec![0x88u8; 8];
    dxt2.extend_from_slice(&color);
    let mut dxt4 = vec![136u8, 136, 0, 0, 0, 0, 0, 0];
    dxt4.extend_from_slice(&color);
    let want = [16.0 / 31.0, 32.0 / 63.0, 8.0 / 31.0]
        .map(|c: f64| orr_color::srgb_decode(c) as f32 / (136.0 / 255.0));
    for (name, block) in [(b"DXT2", dxt2), (b"DXT4", dxt4)] {
        let (_, [t, _]) = read_back(&d, &dds(name, None, [4, 4], &block));
        assert_rgb(t, want, "legacy block format");
    }

    let exr = exr_image(
        |x| {
            if x < 4 {
                (0.25, 0.125, 0.0625, 0.5)
            } else {
                (0.75, 0.0, 0.0, 0.0)
            }
        },
        |_, _| {},
    );
    let (_, [t, none]) = read_back(&d, &exr);
    assert_rgb(t, [0.5, 0.25, 0.125], "EXR");
    assert_rgb(none, [0.75, 0.0, 0.0], "EXR at alpha 0");
}

/// A premultiplied file's blocks hold what no material reads as stored, so they don't pass, and
/// a constant color folds only with its alpha: here alpha varies under flat RGB, which is no
/// constant color.
#[test]
fn premultiplied_blocks_do_not_pass_or_fold() {
    let d = dir("premultiplied_blocks");
    let file = bc7_srgb(|x, _| [0.25, 0.25, 0.25, if x < 4 { 0.5 } else { 1.0 }]);
    let file = with_dfd(&file, 1, 2, 1);
    std::fs::write(d.join("t.ktx2"), &file).unwrap();
    let (a, m, _) = usd(&d, "t.ktx2", "color3f inputs:diffuseColor", "rgb", "");
    assert_ne!(m.textures[slot::BASE_COLOR].texture, AssetId::NONE);
    let (_, texels) = stored(&a, &m, slot::BASE_COLOR, 8);
    assert!(
        texels[0][0] > texels[4][0] + 0.05,
        "{:?} {:?}",
        texels[0],
        texels[4]
    );
}

/// PNG §4.3, under this project's rule that plain files and USD's `auto` honor what they state:
/// "the chunk with the lowest Priority number should take precedence", of `cICP`, `iCCP`, `sRGB`,
/// then `cHRM` and `gAMA`. The same files through glTF are `image_color_metadata_is_ignored`.
#[test]
fn png_chunks_read_by_priority() {
    let d = dir("png_priority");
    let code = 128.0;
    let stored = code as f32 / 255.0;
    let one = tagged::gama(1.0);
    let gamma_1: (&[u8; 4], &[u8]) = (b"gAMA", &one);
    let read = |chunks: &[(&[u8; 4], &[u8])], icc: Option<&[u8]>| {
        let texels = [128u8; 3].repeat(64);
        let (id, [t, _]) = read_back(&d, &tagged::png(tagged::RGB8, [8, 8], &texels, icc, chunks));
        (id, t)
    };
    let (id, t) = read(&[], None);
    assert_rgb(t, [linear(code); 3], "no chunk");
    assert_eq!(id, 0);

    let (_, t) = read(&[gamma_1], None);
    assert_rgb(t, [stored; 3], "gAMA of 1");
    let (_, t) = read(&[(b"gAMA", &tagged::gama(1.0 / 1.8))], None);
    assert_rgb(t, [stored.powf(1.8); 3], "gAMA of 1/1.8");
    let (_, t) = read(&[gamma_1, (b"sRGB", &[0])], None);
    assert_rgb(t, [linear(code); 3], "sRGB over gAMA");
    let (_, t) = read(&[gamma_1, (b"cICP", &[1, 13, 0, 1])], None);
    assert_rgb(t, [linear(code); 3], "cICP over gAMA");

    // Display P3's chromaticities at 100000 per unit, white first.
    let chrm: Vec<u8> = [31270u32, 32900, 68000, 32000, 26500, 69000, 15000, 6000]
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let (id, _) = read(&[(b"cHRM", &chrm)], None);
    assert_eq!(id, 1, "cHRM");
    let (id, _) = read(&[(b"cHRM", &chrm), (b"sRGB", &[0])], None);
    assert_eq!(id, 0, "sRGB over cHRM");

    let p3 = moxcms::ColorProfile::new_display_p3().encode().unwrap();
    let srgb = moxcms::ColorProfile::new_srgb().encode().unwrap();
    let (id, t) = read(&[gamma_1], Some(&p3));
    assert_eq!(id, 2, "iCCP converts through linear Rec.2020");
    assert_rgb(t, [linear(code); 3], "neutrals keep through iCCP");
    let (id, t) = read(&[gamma_1], Some(&srgb));
    assert_eq!(id, 0, "an sRGB iCCP is sRGB");
    assert_rgb(t, [linear(code); 3], "iCCP over gAMA");
    let (id, t) = read(&[(b"cICP", &[1, 8, 0, 1])], Some(&p3));
    assert_eq!(id, 0, "cICP over iCCP");
    assert_rgb(t, [stored; 3], "cICP over iCCP");

    let (id, t) = {
        let red = [255u8, 0, 0].repeat(64);
        let (id, [t, _]) = read_back(&d, &tagged::png(tagged::RGB8, [8, 8], &red, Some(&p3), &[]));
        (id, t)
    };
    assert_eq!(id, 2);
    let want = orr_color::RgbPrimaries::DISPLAY_P3.matrix_to(orr_color::RgbPrimaries::REC2020)
        * orr_math::DVec3::X;
    assert_rgb(t, want.as_vec3().to_array(), "iCCP primaries");

    let gray = moxcms::ColorProfile::new_gray_with_gamma(1.8)
        .encode()
        .unwrap();
    let image = tagged::png(
        (png::ColorType::Grayscale, png::BitDepth::Eight),
        [8, 8],
        &[128; 64],
        Some(&gray),
        &[],
    );
    let (_, [t, _]) = read_back(&d, &image);
    assert_rgb(t, [stored.powf(1.8); 3], "gray iCCP");

    // Code points outside the three ids read, and neutrals stay.
    let (id, t) = read(&[(b"cICP", &[7, 4, 0, 1])], None);
    assert_eq!(id, 2);
    assert_rgb(t, [stored.powf(2.2); 3], "H.273 7 and 4");
}

/// PNG §11.3.2.6: "If Video Full Range Flag value is 0, then the image is a narrow-range image",
/// which H.273 scales as `D = 2^(n - 8) (219 E + 16)`. Alpha is no color signal.
#[test]
fn cicp_narrow_range_expands() {
    let d = dir("cicp_narrow");
    let linear_709: &[u8] = &[1, 8, 0, 0];
    let eight = [126u8, 16, 235].repeat(64);
    let image = tagged::png(tagged::RGB8, [8, 8], &eight, None, &[(b"cICP", linear_709)]);
    let (_, [t, _]) = read_back(&d, &image);
    assert_rgb(t, [110.0 / 219.0, 0.0, 1.0], "8 bits");
    let wide: Vec<u8> = [0x7d80u16, 4096, 60160]
        .repeat(64)
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let sixteen = (png::ColorType::Rgb, png::BitDepth::Sixteen);
    let image = tagged::png(sixteen, [8, 8], &wide, None, &[(b"cICP", linear_709)]);
    let (_, [t, _]) = read_back(&d, &image);
    assert_rgb(t, [(32128.0 - 4096.0) / 56064.0, 0.0, 1.0], "16 bits");

    // Codes past 235 are signal past 1, which emission keeps.
    let above = tagged::flat_rgb([255; 3], &[(b"cICP", linear_709)]);
    std::fs::write(d.join("above.png"), above).unwrap();
    let emissive = "color3f inputs:emissiveColor";
    let (_, m, _) = usd(&d, "above.png", emissive, "rgb", "");
    let want = 239.0 / 219.0;
    assert!((m.emission_color[0] - want).abs() < 0.01, "{m:?}");
}

/// OpenEXR standard attributes: `chromaticities` "specifies the CIE (x,y) chromaticities of the
/// primaries and the white point", "superseded by the colorInteropID". ACES AP0 has no id and
/// converts; the interop id stands over chromaticities; a file with neither is Rec.709, the
/// library's default.
#[test]
fn exr_states_chromaticities_and_interop_id() {
    use exr::meta::attribute::{AttributeValue, Chromaticities, Text};
    use exr::prelude::Vec2;
    let d = dir("exr_color");
    let flat = |_: usize| (0.5, 0.25, 0.125, 1.0);
    let ap0 = Chromaticities {
        red: Vec2(0.7347, 0.2653),
        green: Vec2(0.0, 1.0),
        blue: Vec2(0.0001, -0.077),
        white: Vec2(0.32168, 0.33767),
    };
    let (id, [t, _]) = read_back(&d, &exr_image(flat, |_, _| {}));
    assert_eq!(id, 0);
    assert_rgb(t, [0.5, 0.25, 0.125], "untagged");
    let (id, [t, _]) = read_back(
        &d,
        &exr_image(flat, |_, image| image.chromaticities = Some(ap0)),
    );
    assert_eq!(id, 2);
    let want = orr_color::RgbPrimaries::ACES_AP0.matrix_to(orr_color::RgbPrimaries::REC2020)
        * orr_math::DVec3::new(0.5, 0.25, 0.125);
    assert_rgb(t, want.as_vec3().to_array(), "AP0");

    let (id, [t, _]) = read_back(
        &d,
        &exr_image(flat, |layer, image| {
            image.chromaticities = Some(ap0);
            layer.other.insert(
                Text::from("colorInteropID"),
                AttributeValue::Text(Text::from("lin_rec2020_scene")),
            );
        }),
    );
    assert_eq!(id, 2);
    assert_rgb(t, [0.5, 0.25, 0.125], "colorInteropID");
}

/// A Radiance picture of 4 x 4 texels of RGBE (128, 128, 128, 129), 1.0, under header `lines`.
fn radiance(lines: &str) -> Vec<u8> {
    let mut f = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n{lines}\n-Y 4 +X 4\n").into_bytes();
    f.extend([128u8, 128, 128, 129].repeat(16));
    f
}

/// Radiance filefmts: EXPOSURE is "a multiplier that has been applied to all the pixels in the
/// file", radiances coming from "dividing by all the EXPOSURE settings multiplied together";
/// COLORCORR is "Similar to the EXPOSURE variable except given in three parts", and PRIMARIES
/// gives "primaries and the white point used to standardize the picture's color system". A file
/// without PRIMARIES is Rec.709 by this project's decision.
#[test]
fn radiance_header_variables_apply() {
    let d = dir("radiance");
    let (id, [t, _]) = read_back(&d, &radiance(""));
    assert_eq!(id, 0);
    assert_rgb(t, [1.0; 3], "no variables");
    let (_, [t, _]) = read_back(&d, &radiance("EXPOSURE=0.5\nEXPOSURE=0.5\n"));
    assert_rgb(t, [4.0; 3], "EXPOSURE");
    let (_, [t, _]) = read_back(&d, &radiance("COLORCORR=1 2 4\nCOLORCORR=0.5 1 1\n"));
    assert_rgb(t, [2.0, 0.5, 0.25], "COLORCORR");
    let p3 = "PRIMARIES=0.680 0.320 0.265 0.690 0.150 0.060 0.3127 0.3290\n";
    let (id, _) = read_back(&d, &radiance(p3));
    assert_eq!(id, 1, "PRIMARIES");
}
