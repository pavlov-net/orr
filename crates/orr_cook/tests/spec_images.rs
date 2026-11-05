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
  <{reader} name="i" type="vector3"><input name="file" type="filename" value="f.ktx2"/></{reader}>
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

fn assert_rgb(got: [f32; 4], want: [f32; 3], what: &str) {
    let close = (0..3).all(|c| (got[c] - want[c]).abs() <= 0.012 * want[c].abs().max(0.25));
    assert!(close, "{what}: {got:?}, not {want:?}");
}

fn linear(code: f64) -> f32 {
    orr_color::srgb_decode(code / 255.0) as f32
}

/// PNG §4.3, under this project's rule that plain files and USD's `auto` honor what they state:
/// "the chunk with the lowest Priority number should take precedence", of `cICP`, `sRGB`, then
/// `cHRM` and `gAMA`. The same files through glTF are `image_color_metadata_is_ignored`.
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
