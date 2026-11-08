#![allow(clippy::print_stdout)] // the tests report measured figures

#[path = "common/float_ktx2.rs"]
mod float_ktx2;
#[path = "common/texels.rs"]
mod texels;

use std::f32::consts::{PI, TAU};
use std::path::{Path, PathBuf};

use float_ktx2::float_ktx2;
use orr_color::RgbPrimaries;
use orr_cook::{
    ArchiveSpec, AssetSpec, CookOptions, DataFormat, GltfSettings, InputChannels, Manifest,
    MaterialPatch, Primaries, Quality, Source, TextureOverride, TextureRules, TextureSettings,
    Usage,
};
use orr_math::Vec3;
use orr_pack::material::{Material, slot};
use orr_pack::model::ModelInfo;
use orr_pack::texture::{FormatInfo, TextureInfo, octahedral, vk};
use orr_pack::{Archive, AssetId, BcLayout, Class, Codec, Filter, Target};
use texels::{decode, mips, read_blob, write_glb};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cook(assets: Vec<AssetSpec>) -> Archive {
    cook_at(assets, CookOptions::default().quality)
}

fn cook_at(assets: Vec<AssetSpec>, quality: Quality) -> Archive {
    let cooked = orr_cook::cook(
        &Manifest {
            archives: vec![ArchiveSpec {
                output: Target::Memory,
                cell_log2: ArchiveSpec::DEFAULT_CELL_LOG2,
                gdeflate: false,
                assets,
            }],
        },
        &CookOptions {
            quality,
            ..Default::default()
        },
    )
    .unwrap();
    Archive::from_memory(std::sync::Arc::clone(&cooked.memory[0])).unwrap()
}

fn texture(path: &Path, name: &str, settings: TextureSettings) -> AssetSpec {
    AssetSpec {
        name: name.into(),
        source: Source::Texture {
            path: path.into(),
            settings,
        },
    }
}

/// Cooks the GLB at `path` as asset `g`; the archive and its first material.
fn cook_glb_at(path: &Path, settings: GltfSettings, quality: Quality) -> (Archive, Material) {
    let a = cook_at(
        vec![AssetSpec {
            name: "g".into(),
            source: Source::Gltf {
                path: path.to_path_buf(),
                settings,
            },
        }],
        quality,
    );
    let e = a.find(AssetId::from_name("g/material/0")).unwrap();
    let m = Material::from_bytes(&read_blob(&a, e, 0)).unwrap();
    (a, m)
}

fn cook_glb(path: &Path, settings: GltfSettings) -> (Archive, Material) {
    cook_glb_at(path, settings, CookOptions::default().quality)
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    10.0 * (1.0 / mse).log10()
}

fn write_png(path: &Path, w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) {
    let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
    img.save(path).unwrap();
}

#[test]
fn bc7_mip0_keeps_the_source() {
    let d = dir("bc7");
    let (w, h) = (300, 200);
    let px = |x: u32, y: u32| {
        let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
        [
            (255.0 * (0.5 + 0.5 * (fx * 9.0).sin())) as u8,
            (255.0 * fy) as u8,
            (255.0 * (0.5 + 0.5 * ((fx + fy) * 5.0).cos())) as u8,
            255,
        ]
    };
    let path = d.join("color.png");
    write_png(&path, w, h, px);
    let a = cook(vec![texture(&path, "color", TextureSettings::default())]);
    let (info, levels) = mips(&a, "color");
    let hd = info.header;
    assert_eq!(
        (hd.vk_format, hd.width, hd.height, hd.mip_count),
        (vk::BC7_SRGB, w, h, 9)
    );
    assert_eq!(hd.tail_first_mip, 0);
    assert!(hd.alpha_cutoff.is_nan());
    let got = decode(vk::BC7_SRGB, w, h, &levels[0][0]);
    let (mut src, mut dec) = (Vec::new(), Vec::new());
    for y in 0..h {
        for x in 0..w {
            let p = px(x, y);
            let g = got[(y * w + x) as usize];
            for c in 0..3 {
                src.push(f32::from(p[c]) / 255.0);
                dec.push(g[c]);
            }
        }
    }
    let db = psnr(&src, &dec);
    assert!(db > 40.0, "BC7 mip 0 PSNR {db:.1} dB");
}

/// Data textures keep one mip unless asked for more, at the size limit; a texture asked for
/// no mips keeps only its source's mip 0.
#[test]
fn textures_keep_the_mips_their_usage_or_settings_ask_for() {
    let d = dir("mips");
    let (w, h) = (300, 200);
    let px = |x: u32, y: u32| [(x * 7) as u8, (y * 3) as u8, (x ^ y) as u8, (x + y) as u8];
    let path = d.join("data.png");
    write_png(&path, w, h, px);
    let (color, color_blocks) = ktx2(64, 64, ctt::Format::BC7_SRGB_BLOCK, |m, x, y| {
        [(x * 4) as u8, (y * 4) as u8, (m * 40) as u8, 255]
    });
    let color_path = d.join("color.ktx2");
    std::fs::write(&color_path, color).unwrap();
    let data = |max_size, mips| TextureSettings {
        usage: Usage::Data(DataFormat::Rgba8),
        max_size,
        mips,
        ..TextureSettings::default()
    };
    let a = cook(vec![
        texture(&path, "one", data(None, None)),
        texture(&path, "all", data(None, Some(true))),
        texture(&path, "small", data(Some(128), None)),
        texture(
            &color_path,
            "color",
            TextureSettings {
                mips: Some(false),
                ..TextureSettings::default()
            },
        ),
    ]);
    let (info, levels) = mips(&a, "one");
    assert_eq!(info.header.mip_count, 1);
    let texels: Vec<u8> = (0..w * h).flat_map(|i| px(i % w, i / w)).collect();
    assert_eq!(levels[0][0], texels);
    assert_eq!(mips(&a, "all").0.header.mip_count, 9);
    let small = mips(&a, "small").0.header;
    assert_eq!((small.width, small.height, small.mip_count), (75, 50, 1));
    let (info, levels) = mips(&a, "color");
    assert_eq!(info.header.mip_count, 1);
    assert_eq!(levels[0][0], color_blocks[0]);
}

#[test]
fn bc_split_keeps_the_blocks() {
    let d = dir("bc-split");
    let path = d.join("noise.png");
    write_png(&path, 520, 520, |x, y| {
        let v = (x * 7 + y * 13) ^ (x * y);
        [v as u8, (v >> 3) as u8, (x + y) as u8, (v >> 5) as u8]
    });
    for (usage, layout) in [
        (Usage::ColorAlpha, BcLayout::Bc7),
        (Usage::Normal, BcLayout::Bc5),
        (Usage::Linear1, BcLayout::Bc4),
    ] {
        let a = cook(
            [None, Some(false), Some(true)]
                .into_iter()
                .enumerate()
                .map(|(i, bc_split)| {
                    let settings = TextureSettings {
                        usage,
                        bc_split,
                        ..TextureSettings::default()
                    };
                    texture(&path, &i.to_string(), settings)
                })
                .collect(),
        );
        let levels = ["0", "1", "2"].map(|n| mips(&a, n).1);
        assert!(
            levels[0] == levels[1] && levels[1] == levels[2],
            "{usage:?}"
        );
        for (n, filter) in [
            ("0", Filter::BcSplit(layout)),
            ("1", Filter::None),
            ("2", Filter::BcSplit(layout)),
        ] {
            let e = a.find(AssetId::from_name(n)).unwrap();
            let blobs = a.blobs(e);
            assert_eq!(blobs[0].filter(), Filter::None);
            assert!(
                blobs[1..]
                    .iter()
                    .all(|b| b.filter() == filter || b.codec() == Codec::None),
                "{usage:?} {n}"
            );
            assert_eq!(blobs[1].filter(), filter, "{usage:?} {n}");
        }
    }
}

/// A BC6H cube with every mip, written as KTX2, cooks to the same blocks.
#[test]
fn bc6h_cube_from_ktx2_round_trips() {
    let d = dir("cube");
    let n = 512;
    let faces: Vec<Vec<u8>> = (0..6)
        .map(|f| {
            (0..n * n)
                .flat_map(|i| {
                    let (x, y) = ((i % n) as f32 / n as f32, (i / n) as f32 / n as f32);
                    let v = (f as f32 + 1.0) * (1.0 + 8.0 * x * y);
                    [v, v * 0.5 * (1.0 + y), 0.1 + x, 1.0]
                })
                .flat_map(f32::to_le_bytes)
                .collect()
        })
        .collect();
    let image = ctt::ImageRef {
        surfaces: faces
            .iter()
            .map(|f| {
                vec![ctt::SurfaceRef {
                    data: f,
                    width: n,
                    height: n,
                    depth: 1,
                    stride: n * 16,
                    slice_stride: 0,
                }]
            })
            .collect(),
        kind: ctt::TextureKind::Cubemap,
        desc: ctt::FormatDesc {
            format: ctt::Format::R32G32B32A32_SFLOAT,
            color_space: ctt::ColorSpace::Linear,
            alpha: ctt::AlphaMode::Opaque,
        },
    };
    let ctt::PipelineOutput::Encoded(ktx) = ctt::convert(
        image,
        ctt::ConvertSettings {
            format: Some(ctt::TargetFormat::Compressed {
                format: ctt::Format::BC6H_UFLOAT_BLOCK,
                encoder: ctt::encoders::Encoder::Intel(Default::default()),
            }),
            container: ctt::Container::ktx2(),
            quality: Quality::Fast,
            mipmap: true,
            ..Default::default()
        },
    )
    .unwrap() else {
        panic!("container output");
    };
    let path = d.join("cube.ktx2");
    std::fs::write(&path, &ktx).unwrap();
    let src = ctt::input::decode_container(&ktx, Default::default())
        .unwrap()
        .unwrap();

    let a = cook(vec![texture(
        &path,
        "cube",
        TextureSettings {
            usage: Usage::Hdr,
            ..TextureSettings::default()
        },
    )]);
    let (info, levels) = mips(&a, "cube");
    let hd = info.header;
    assert_eq!(
        (
            hd.vk_format,
            hd.kind,
            hd.layers,
            hd.mip_count,
            hd.tail_first_mip
        ),
        (
            vk::BC6H_UFLOAT,
            orr_pack::texture::TextureHeader::CUBE,
            6,
            10,
            2
        )
    );
    for (face, mips) in src.surfaces.iter().enumerate() {
        for (m, s) in mips.iter().enumerate() {
            assert_eq!(levels[face][m], s.data, "face {face} mip {m}");
        }
    }
}

/// Equirectangular radiance: a sky gradient, a ground and a few soft features.
fn sky(dir: [f32; 3]) -> [f32; 3] {
    let up = dir[1];
    let base = if up > 0.0 {
        [0.3 + 0.7 * up, 0.5 + 0.8 * up, 1.0 + 1.5 * up]
    } else {
        [0.25, 0.2, 0.15]
    };
    let blob = |c: [f32; 3], k: f32, s: f32| {
        let d = Vec3::from(dir).dot(Vec3::from(c));
        s * (k * (d - 1.0)).exp()
    };
    let a = blob([0.6, 0.3, -0.74], 40.0, 3.0);
    let b = blob([-0.7, 0.1, 0.7], 15.0, 1.5);
    [base[0] + a, base[1] + a + b, base[2] + b]
}

fn equirect_dir(u: f32, v: f32) -> [f32; 3] {
    let (theta, phi) = (v * PI, u * TAU - PI);
    [
        theta.sin() * phi.sin(),
        theta.cos(),
        -theta.sin() * phi.cos(),
    ]
}

fn write_hdr(path: &Path, w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 3]) {
    let px: Vec<image::Rgb<f32>> = (0..w * h).map(|i| image::Rgb(f(i % w, i / w))).collect();
    let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    image::codecs::hdr::HdrEncoder::new(file)
        .encode(&px, w as usize, h as usize)
        .unwrap();
}

/// Bilinear sample of a decoded octahedral map along `d`, folding across the map's edges.
fn sample_octahedral(map: &[[f32; 4]], n: u32, d: [f32; 3]) -> [f32; 3] {
    let [s, t] = octahedral::from_dir(d);
    let (x, y) = (s * n as f32 - 0.5, t * n as f32 - 0.5);
    let (x0, y0) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let n = i64::from(n);
    let at = |mut x: i64, mut y: i64| {
        if x < 0 {
            (x, y) = (0, n - 1 - y);
        } else if x >= n {
            (x, y) = (n - 1, n - 1 - y);
        }
        if y < 0 {
            (x, y) = (n - 1 - x, 0);
        } else if y >= n {
            (x, y) = (n - 1 - x, n - 1);
        }
        map[(y * n + x) as usize]
    };
    std::array::from_fn(|c| {
        let top = at(x0, y0)[c] * (1.0 - fx) + at(x0 + 1, y0)[c] * fx;
        let bot = at(x0, y0 + 1)[c] * (1.0 - fx) + at(x0 + 1, y0 + 1)[c] * fx;
        top * (1.0 - fy) + bot * fy
    })
}

/// PSNR of `x / (1 + x)` between a source equirect and the map resampled back, both at
/// `w` x `w / 2`.
fn equirect_round_trip_psnr(
    src: impl Fn(f32, f32) -> [f32; 3],
    map: &[[f32; 4]],
    n: u32,
    w: u32,
) -> f64 {
    let h = w / 2;
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let tm = |v: f32| v / (1.0 + v);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let s = src(u, v);
            let m = sample_octahedral(map, n, equirect_dir(u, v));
            for c in 0..3 {
                a.push(tm(s[c]));
                b.push(tm(m[c]));
            }
        }
    }
    psnr(&a, &b)
}

#[test]
fn environment_map_resamples_back_to_the_source() {
    let d = dir("env");
    let (w, h) = (4096, 2048);
    let path = d.join("sky.hdr");
    write_hdr(&path, w, h, |x, y| {
        sky(equirect_dir(
            (x as f32 + 0.5) / w as f32,
            (y as f32 + 0.5) / h as f32,
        ))
    });
    let a = cook(vec![texture(
        &path,
        "sky",
        TextureSettings {
            usage: Usage::Environment {
                light_threshold: None,
                max_lights: 4,
            },
            ..TextureSettings::default()
        },
    )]);
    let (info, levels) = mips(&a, "sky");
    let hd = info.header;
    assert_eq!(
        (hd.vk_format, hd.width, hd.height, hd.mip_count),
        (vk::BC6H_UFLOAT, 1024, 1024, 11)
    );
    assert!(info.lights.is_empty());
    let map = decode(vk::BC6H_UFLOAT, 1024, 1024, &levels[0][0]);
    // The source at the map's resolution (1024 x 512 equirect texels hold about as many
    // samples as the 1024^2 map).
    let db = equirect_round_trip_psnr(|u, v| sky(equirect_dir(u, v)), &map, 1024, 1024);
    println!("environment round trip: {db:.1} dB");
    assert!(db > 45.0, "round trip PSNR {db:.1} dB");
}

#[test]
fn environment_lights_keep_energy_and_direction() {
    let d = dir("discs");
    let (w, h) = (2048, 1024);
    // Two discs: a small bright one and a larger dim one, over a dim sky.
    let discs = [
        (
            [0.3f32, 0.8, -0.52],
            0.6f32.to_radians(),
            [9000.0f32, 8000.0, 7000.0],
        ),
        (
            [-0.8, 0.35, 0.49],
            2.5f32.to_radians(),
            [300.0, 320.0, 360.0],
        ),
    ];
    let discs = discs.map(|(c, r, v)| (Vec3::from(c).normalize(), r, v));
    let radiance = |dir: [f32; 3]| {
        let mut v = [0.4, 0.6, 1.0];
        for (c, r, e) in &discs {
            let cos = Vec3::from(dir).dot(*c);
            if cos >= r.cos() {
                v = *e;
            }
        }
        v
    };
    let path = d.join("discs.hdr");
    let texel = |x: u32, y: u32| {
        // 4x4 supersampled, so the source holds the discs' exact energy closely.
        let mut acc = [0.0f32; 3];
        for j in 0..4 {
            for i in 0..4 {
                let u = (x as f32 + (i as f32 + 0.5) / 4.0) / w as f32;
                let v = (y as f32 + (j as f32 + 0.5) / 4.0) / h as f32;
                let r = radiance(equirect_dir(u, v));
                for c in 0..3 {
                    acc[c] += r[c] / 16.0;
                }
            }
        }
        acc
    };
    let src: Vec<[f32; 3]> = (0..w * h).map(|i| texel(i % w, i / w)).collect();
    write_hdr(&path, w, h, |x, y| src[(y * w + x) as usize]);
    let n = 512;
    let a = cook(vec![texture(
        &path,
        "discs",
        TextureSettings {
            usage: Usage::Environment {
                light_threshold: Some(100.0),
                max_lights: 4,
            },
            max_size: Some(n),
            ..TextureSettings::default()
        },
    )]);
    let (info, levels) = mips(&a, "discs");
    assert_eq!(info.lights.len(), 2);
    let map = decode(vk::BC6H_UFLOAT, n, n, &levels[0][0]);

    // Energy: the source's radiance integral against the map's plus the lights'.
    let mut source = [0.0f64; 3];
    for y in 0..h {
        let theta = (f64::from(y) + 0.5) / f64::from(h) * std::f64::consts::PI;
        let sr = theta.sin()
            * (std::f64::consts::TAU / f64::from(w))
            * (std::f64::consts::PI / f64::from(h));
        for x in 0..w {
            for c in 0..3 {
                source[c] += f64::from(src[(y * w + x) as usize][c]) * sr;
            }
        }
    }
    let texel_sr = 4.0 * std::f64::consts::PI / f64::from(n * n);
    let mut cooked = [0.0f64; 3];
    for p in &map {
        for c in 0..3 {
            cooked[c] += f64::from(p[c]) * texel_sr;
        }
    }
    for l in &info.lights {
        for (c, e) in cooked.iter_mut().zip(l.illuminance) {
            *c += f64::from(e);
        }
    }
    for c in 0..3 {
        let rel = cooked[c] / source[c] - 1.0;
        println!(
            "channel {c}: map plus lights {:+.3}% of the source",
            rel * 100.0
        );
        assert!(
            rel.abs() < 0.01,
            "channel {c}: energy off by {:.3}%",
            rel * 100.0
        );
    }

    // Brightest first, each pointing at its disc within half a texel, with its energy.
    let half_texel = 0.5 * (4.0 * PI / (n * n) as f32).sqrt();
    for (l, (c, r, e)) in info.lights.iter().zip(&discs) {
        let d = l.direction_to_light;
        let cos = Vec3::from(d).dot(*c).min(1.0);
        assert!(
            cos.acos() <= half_texel,
            "light off its disc by {} rad",
            cos.acos()
        );
        let solid = TAU * (1.0 - r.cos());
        let want = e[1] * solid;
        println!(
            "light: {:.3} of its disc's energy, {:.3} texels off, radius {:.3} of the disc's",
            l.illuminance[1] / want,
            cos.acos() / (2.0 * half_texel),
            l.angular_radius / r
        );
        assert!(
            (l.illuminance[1] / want - 1.0).abs() < 0.1,
            "light {} against disc {want}",
            l.illuminance[1]
        );
        let texel = 2.0 * half_texel;
        assert!(
            (l.angular_radius - r).abs() < 1.5 * texel,
            "angular radius {} against {r}",
            l.angular_radius
        );
    }
}

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let img = image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y)));
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// Resident bytes of a full mip chain.
fn chain_bytes(format: u32, w: u32, h: u32) -> u64 {
    let f = FormatInfo::of(format).unwrap();
    (0..orr_pack::texture::full_mip_count(w, h))
        .map(|m| {
            let [mw, mh] = orr_pack::texture::mip_size(w, h, m);
            f.mip_bytes(mw, mh) as u64
        })
        .sum()
}

/// A KTX2 of `w` x `h` RGBA8 texels `f(mip, x, y)` in `format` with every mip, and its mips
/// as blocks.
fn ktx2(
    w: u32,
    h: u32,
    format: ctt::Format,
    f: impl Fn(u32, u32, u32) -> [u8; 4],
) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mips: Vec<([u32; 2], Vec<u8>)> = (0..orr_pack::texture::full_mip_count(w, h))
        .map(|m| {
            let [mw, mh] = orr_pack::texture::mip_size(w, h, m);
            let texels = (0..mw * mh).flat_map(|i| f(m, i % mw, i / mw)).collect();
            ([mw, mh], texels)
        })
        .collect();
    let srgb = format == ctt::Format::BC7_SRGB_BLOCK;
    let image = ctt::ImageRef {
        surfaces: vec![
            mips.iter()
                .map(|([mw, mh], texels)| ctt::SurfaceRef {
                    data: texels,
                    width: *mw,
                    height: *mh,
                    depth: 1,
                    stride: mw * 4,
                    slice_stride: 0,
                })
                .collect(),
        ],
        kind: ctt::TextureKind::Texture2D,
        desc: ctt::FormatDesc {
            format: if srgb {
                ctt::Format::R8G8B8A8_SRGB
            } else {
                ctt::Format::R8G8B8A8_UNORM
            },
            color_space: if srgb {
                ctt::ColorSpace::Srgb
            } else {
                ctt::ColorSpace::Linear
            },
            alpha: ctt::AlphaMode::Straight,
        },
    };
    let encoder = if srgb {
        ctt::encoders::Encoder::Bc7enc(Default::default())
    } else {
        ctt::encoders::Encoder::Intel(Default::default())
    };
    let ctt::PipelineOutput::Encoded(bytes) = ctt::convert(
        image,
        ctt::ConvertSettings {
            format: Some(ctt::TargetFormat::Compressed { format, encoder }),
            container: ctt::Container::ktx2(),
            quality: Quality::Fast,
            mipmap: false,
            ..Default::default()
        },
    )
    .unwrap() else {
        panic!("container output");
    };
    let src = ctt::input::decode_container(&bytes, Default::default())
        .unwrap()
        .unwrap();
    let blocks = src.surfaces[0].iter().map(|s| s.data.to_vec()).collect();
    (bytes, blocks)
}

/// A map of mesh space directions stores what a linear BC7 file holds: `0.5 v + 0.5` as it stands.
#[test]
fn object_normal_maps_keep_bc7_blocks() {
    let d = dir("object-normal-pass");
    let n = 64;
    let (file, blocks) = ktx2(n, n, ctt::Format::BC7_UNORM_BLOCK, |_, x, y| {
        [(4 * x) as u8, (4 * y) as u8, (2 * (x + y)) as u8, 255]
    });
    std::fs::write(d.join("o.ktx2"), file).unwrap();
    let doc = d.join("m.mtlx");
    let body = r#"<?xml version="1.0"?>
<materialx version="1.38">
  <image name="i" type="vector3"><input name="file" type="filename" value="o.ktx2"/></image>
  <normalmap name="n" type="vector3">
    <input name="in" type="vector3" nodename="i"/>
    <input name="space" type="string" value="object"/>
  </normalmap>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="geometry_normal" type="vector3" nodename="n"/>
  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="S"/>
  </surfacematerial>
</materialx>
"#;
    std::fs::write(&doc, body).unwrap();
    let a = cook_at(
        vec![AssetSpec {
            name: "u".into(),
            source: Source::MaterialX {
                path: doc,
                material: "M".into(),
                textures: TextureRules::default(),
            },
        }],
        Quality::Fast,
    );
    let e = a.find(AssetId::from_name("u")).unwrap();
    let m = Material::from_bytes(&read_blob(&a, e, 0)).unwrap();
    let t = m.textures[slot::NORMAL].texture;
    let (info, levels) = mips(&a, a.name(a.find(t).unwrap()));
    assert_eq!(info.header.vk_format, vk::BC7_UNORM);
    assert_eq!(levels[0], blocks);
}

/// BCn images that already hold what the material wants keep their blocks: every mip, or
/// at qualities that remake mips, mip 0 where the cook makes the other mips (Toksvig
/// roughness). Below those, a texture made from other channels of BCn images encodes the
/// images' own mips. An opaque material's base color keeps its alpha, which nothing reads
/// as opacity. Two-channel images hold roughness and metalness in red and green, and
/// overrides name other channels.
#[test]
fn bcn_material_textures_keep_their_blocks() {
    let d = dir("bcn-pass");
    let n = 64;
    let wave =
        |x: u32, y: u32, k: u32| (128.0 + 100.0 * ((x * k + y * 3) as f32 * 0.21).sin()) as u8;
    // Alpha below mip 0 is not what filtering mip 0 gives.
    let (color, color_blocks) = ktx2(n, n, ctt::Format::BC7_SRGB_BLOCK, |m, x, y| {
        let a = if m == 0 { (2 * (x + y)) as u8 } else { 200 };
        [wave(x, y, 1), wave(x, y, 2), wave(x, y, 3), a]
    });
    let (normal, normal_blocks) = ktx2(n, n, ctt::Format::BC5_UNORM_BLOCK, |_, x, y| {
        [wave(x, y, 5), wave(y, x, 4), 255, 255]
    });
    let (rm, rm_blocks) = ktx2(n, n, ctt::Format::BC5_UNORM_BLOCK, |_, x, y| {
        [wave(x, y, 2), wave(y, x, 7), 0, 255]
    });
    let (specular, specular_blocks) = ktx2(n, n, ctt::Format::BC4_UNORM_BLOCK, |_, x, y| {
        [wave(x, y, 6), 0, 0, 255]
    });
    // Constant metalness: roughness alone, a BC4 taken from the red half of the BC5.
    let (rough, rough_blocks) = ktx2(n, n, ctt::Format::BC5_UNORM_BLOCK, |_, x, y| {
        [wave(x, y, 3), 0, 0, 255]
    });
    let path = d.join("bcn.glb");
    write_glb(
        &path,
        &[
            r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0},"metallicRoughnessTexture":{"index":2}},
                "normalTexture":{"index":1}}"#
                .into(),
            r#"{"pbrMetallicRoughness":{"metallicRoughnessTexture":{"index":2}},
                "extensions":{"KHR_materials_specular":{"specularTexture":{"index":3}}}}"#
                .into(),
            r#"{"pbrMetallicRoughness":{"metallicRoughnessTexture":{"index":4}},
                "normalTexture":{"index":1}}"#
                .into(),
            r#"{"pbrMetallicRoughness":{"baseColorTexture":{"index":0}},"alphaMode":"MASK"}"#.into(),
        ],
        &[color, normal, rm, specular, rough],
    );
    // The specular image keeps its weight in red, not glTF's alpha.
    let overrides = vec![(
        "image 3".to_string(),
        TextureOverride {
            channels: InputChannels {
                specular: Some(0),
                ..InputChannels::default()
            },
            ..TextureOverride::default()
        },
    )];
    for quality in [Quality::Fast, Quality::Slow] {
        let remade = quality == Quality::Slow;
        let a = cook_at(
            vec![AssetSpec {
                name: "g".into(),
                source: Source::Gltf {
                    path: path.clone(),
                    settings: GltfSettings {
                        textures: TextureRules {
                            overrides: overrides.clone(),
                            ..TextureRules::default()
                        },
                        ..GltfSettings::default()
                    },
                },
            }],
            quality,
        );
        bcn_material_checks(
            &a,
            remade,
            [
                &color_blocks,
                &normal_blocks,
                &rm_blocks,
                &specular_blocks,
                &rough_blocks,
            ],
        );
    }
}

/// Mips made below a passed mip 0 hold their channels where mip 0 does, which is where the
/// material's bindings read them.
#[test]
fn swapped_roughness_and_metalness_keep_their_channels_below_mip_0() {
    let d = dir("bcn-swapped");
    let n = 64;
    let (metal, rough) = (50.0 / 255.0, 210.0 / 255.0);
    let (rm, _) = ktx2(n, n, ctt::Format::BC5_UNORM_BLOCK, |_, x, _| {
        let wobble = (x % 2 * 20) as u8;
        [40 + wobble, 200 + wobble, 0, 255]
    });
    let normal = png(n, n, |x, _| [118 + (x % 2 * 20) as u8, 128, 255, 255]);
    let path = d.join("swapped.glb");
    write_glb(
        &path,
        &[
            r#"{"pbrMetallicRoughness":{"metallicRoughnessTexture":{"index":0}},
            "normalTexture":{"index":1}}"#
                .into(),
        ],
        &[rm, normal],
    );
    let channels = InputChannels {
        roughness: Some(1),
        metalness: Some(0),
        ..InputChannels::default()
    };
    let overrides = vec![(
        "image 0".to_string(),
        TextureOverride {
            channels,
            ..TextureOverride::default()
        },
    )];
    let settings = GltfSettings {
        textures: TextureRules {
            overrides,
            ..TextureRules::default()
        },
        ..GltfSettings::default()
    };
    let (a, m) = cook_glb_at(&path, settings, Quality::Slow);
    let (r, t) = (m.textures[slot::ROUGHNESS], m.textures[slot::METALNESS]);
    assert_eq!(r.texture, t.texture);
    let (_, levels) = mips(&a, a.name(a.find(r.texture).unwrap()));
    for (mip, size) in [(0, n), (1, n / 2)] {
        let texels = decode(vk::BC5_UNORM, size, size, &levels[0][mip]);
        let mean =
            |c: u8| texels.iter().map(|t| t[usize::from(c)]).sum::<f32>() / texels.len() as f32;
        let (got_rough, got_metal) = (mean(r.channels), mean(t.channels));
        assert!(
            got_rough >= rough - 0.05,
            "mip {mip}: roughness {got_rough}"
        );
        assert!(
            (got_metal - metal).abs() < 0.05,
            "mip {mip}: metalness {got_metal}"
        );
    }
}

#[test]
fn two_channel_normals_rebuild_z_when_they_are_reencoded() {
    let d = dir("rg-normal");
    let n = 64;
    // Two halves tilted 45 degrees either way; the block format holds no z.
    let tilt = 0.5f32.sqrt();
    let (image, _) = ktx2(n, n, ctt::Format::BC5_UNORM_BLOCK, |m, x, _| {
        let stored = if x << m < n / 2 { 1.0 } else { -1.0 } * tilt * 0.5 + 0.5;
        [(stored * 255.0).round() as u8, 128, 0, 255]
    });
    let x_of = |a: &Archive, name: &str| {
        let (_, levels) = mips(a, name);
        2.0 * decode(vk::BC5_UNORM, n, n, &levels[0][0])[8 * n as usize + 8][0] - 1.0
    };

    // A normal scale stops the blocks passing through, whichever path remakes them.
    let path = d.join("scaled.glb");
    write_glb(
        &path,
        &[r#"{"normalTexture":{"index":0,"scale":0.5}}"#.into()],
        std::slice::from_ref(&image),
    );
    let scaled = 0.5 / 1.25f32.sqrt();
    for quality in [Quality::Fast, Quality::Slow] {
        let (a, m) = cook_glb_at(&path, GltfSettings::default(), quality);
        let name = a.name(a.find(m.textures[slot::NORMAL].texture).unwrap());
        let x = x_of(&a, name);
        assert!((x - scaled).abs() < 0.02, "{quality:?}: {x}, not {scaled}");
    }

    let path = d.join("standalone.ktx2");
    std::fs::write(&path, &image).unwrap();
    let settings = TextureSettings {
        usage: Usage::Normal,
        atlas_grid: Some([2, 2]),
        ..TextureSettings::default()
    };
    let x = x_of(&cook(vec![texture(&path, "n", settings)]), "n");
    assert!((x - tilt).abs() < 0.02, "{x}, not {tilt}");
}

#[test]
fn signed_two_channel_normals_read_their_own_values() {
    let d = dir("snorm-normal");
    let n = 64u32;
    let tilt = 0.5f32.sqrt();
    let format = ctt::TargetFormat::Uncompressed(ctt::Format::R8G8_SNORM);
    let bytes = float_ktx2(n, n, format, false, |x, _| {
        [if x < n / 2 { tilt } else { -tilt }, 0.0, 0.0, 1.0]
    });
    let path = d.join("signed.ktx2");
    std::fs::write(&path, &bytes).unwrap();
    let check = |a: &Archive, name: &str| {
        let (_, levels) = mips(a, name);
        let texels = decode(vk::BC5_UNORM, n, n, &levels[0][0]);
        for (x, want) in [(8, tilt), (56, -tilt)] {
            let got = texels[8 * n as usize + x].map(|c| 2.0 * c - 1.0);
            assert!((got[0] - want).abs() < 0.02, "{got:?}, not {want}");
            assert!(got[1].abs() < 0.02, "{got:?}");
        }
    };
    let settings = TextureSettings {
        usage: Usage::Normal,
        ..TextureSettings::default()
    };
    check(&cook(vec![texture(&path, "n", settings)]), "n");

    let glb = d.join("signed.glb");
    write_glb(&glb, &[r#"{"normalTexture":{"index":0}}"#.into()], &[bytes]);
    let (a, m) = cook_glb(&glb, GltfSettings::default());
    check(
        &a,
        a.name(a.find(m.textures[slot::NORMAL].texture).unwrap()),
    );

    // As plain data, a signed value is that value.
    let settings = TextureSettings {
        usage: Usage::Data(DataFormat::R32f),
        ..TextureSettings::default()
    };
    let a = cook(vec![texture(&path, "data", settings)]);
    let (_, levels) = mips(&a, "data");
    let value = |x: usize| {
        let at = (8 * n as usize + x) * 4;
        f32::from_le_bytes(levels[0][0][at..at + 4].try_into().unwrap())
    };
    assert!((value(8) - tilt).abs() < 0.02, "{}", value(8));
    assert!((value(56) + tilt).abs() < 0.02, "{}", value(56));
}

fn signed_blocks(format: ctt::Format) -> ctt::TargetFormat {
    ctt::TargetFormat::Compressed {
        format,
        encoder: ctt::encoders::Encoder::Amd(Default::default()),
    }
}

fn height_material(factor: f32, offset: f32) -> String {
    format!(
        r#"{{"extensions":{{"KHR_materials_displacement":{{"displacementGeometryTexture":{{"index":0}},
            "displacementGeometryFactor":{factor},"displacementGeometryOffset":{offset}}}}}}}"#
    )
}

/// R16 unorm texels of a material's displacement texture, read as 0..1.
fn stored_heights(a: &Archive, m: &Material) -> Vec<f32> {
    let texture = a.find(m.textures[slot::DISPLACEMENT].texture).unwrap();
    let (info, levels) = mips(a, a.name(texture));
    assert_eq!(info.header.vk_format, vk::R16_UNORM);
    levels[0][0]
        .chunks_exact(2)
        .map(|t| f32::from(u16::from_le_bytes([t[0], t[1]])) / 65535.0)
        .collect()
}

/// A material's height stores across the range the image measures, with the difference in its
/// factors: it reads back as the source height, inside the material's range, and the mesh cook
/// takes it. Blocks hold their extremes in the last blocks read, and the image of part of 0..1
/// gets a pair that rounds a step past the range the importer declared.
#[test]
fn material_heights_read_back_through_compensated_factors() {
    let path = dir("height-range").join("heights.glb");
    let n = 64u32;
    let ramp = |lo: f32, hi: f32| -> Vec<f32> {
        (0..n)
            .map(|x| lo + (hi - lo) * x as f32 / (n - 1) as f32)
            .collect()
    };
    let file =
        |format, h: &[f32]| float_ktx2(n, n, format, false, |x, _| [h[x as usize], 0.0, 0.0, 1.0]);
    let raw = ctt::TargetFormat::Uncompressed;
    let bytes: Vec<u8> = (0..n).map(|x| (16 + 239 * x / (n - 1)) as u8).collect();
    let partial: Vec<f32> = bytes.iter().map(|&v| f32::from(v) / 255.0).collect();
    let signed = ramp(-1.0, 1.0);
    let float = ramp(-2.0, 3.0);
    for (image, heights, factor, offset) in [
        (
            png(n, n, |x, _| [bytes[x as usize], 0, 0, 255]),
            &partial,
            0.2,
            0.0,
        ),
        (
            file(raw(ctt::Format::R32G32B32A32_SFLOAT), &float),
            &float,
            0.2,
            0.05,
        ),
        (
            file(signed_blocks(ctt::Format::BC4_SNORM_BLOCK), &signed),
            &signed,
            0.2,
            0.05,
        ),
        (
            file(raw(ctt::Format::R16_SNORM), &signed),
            &signed,
            0.2,
            0.05,
        ),
    ] {
        write_glb(&path, &[height_material(factor, offset)], &[image]);
        let (a, m) = cook_glb(&path, GltfSettings::default());
        let stored = stored_heights(&a, &m);
        assert_eq!(stored.iter().copied().fold(f32::MAX, f32::min), 0.0);
        assert_eq!(stored.iter().copied().fold(f32::MIN, f32::max), 1.0);
        for (i, u) in stored.iter().enumerate() {
            let got = m.displacement_scale * u + m.displacement_offset;
            let want = factor * heights[i % n as usize] + offset;
            assert!((got - want).abs() < 0.005, "texel {i}: {got}, not {want}");
            assert!(m.displacement_min - 1e-6 <= got && got <= m.displacement_max + 1e-6);
        }
    }

    // A range a patch declares past the texture's stays.
    let (_, m) = cook_glb(
        &path,
        GltfSettings {
            material_overrides: vec![(
                String::new(),
                MaterialPatch {
                    displacement_max: Some(1.0),
                    ..Default::default()
                },
            )],
            ..Default::default()
        },
    );
    assert_eq!(m.displacement_max, 1.0);
    assert!((m.displacement_min + 0.15).abs() < 1e-6);
}

/// A constant height has no span to store across: the offset holds it.
#[test]
fn constant_material_heights_fold_into_the_offset() {
    let path = dir("height-constant").join("constant.glb");
    let (factor, offset) = (0.2, 0.05);
    for (format, height) in [
        (ctt::Format::R16_SNORM, -0.5),
        (ctt::Format::R32G32B32A32_SFLOAT, 3.0),
    ] {
        let format = ctt::TargetFormat::Uncompressed(format);
        let image = float_ktx2(8, 8, format, false, |_, _| [height, 0.0, 0.0, 1.0]);
        write_glb(&path, &[height_material(factor, offset)], &[image]);
        let (a, m) = cook_glb(&path, GltfSettings::default());
        assert!(stored_heights(&a, &m).iter().all(|&u| u == 0.0));
        assert_eq!(m.displacement_scale, 0.0);
        let want = factor * height + offset;
        assert!((m.displacement_offset - want).abs() < 1e-4, "{m:?}");
        assert!(m.displacement_min <= want + 1e-4 && want - 1e-4 <= m.displacement_max);
    }
}

/// A float source's heights store as they are and displace geometry at every mip: non-finite
/// texels take finite values, and the filter's overshoot at a step stays in the source's range.
#[test]
fn float_standalone_heights_stay_finite_and_in_range_at_every_mip() {
    let path = dir("float-heights").join("h.ktx2");
    let n = 32u32;
    let format = ctt::TargetFormat::Uncompressed(ctt::Format::R32G32B32A32_SFLOAT);
    let bytes = float_ktx2(n, n, format, false, |x, y| {
        let height = match (x, y) {
            (5, 5) => f32::NAN,
            (9, 9) => f32::INFINITY,
            (25, 25) => f32::NEG_INFINITY,
            _ if x < n / 2 => -2.5,
            _ => 3.0,
        };
        [height, 0.0, 0.0, 1.0]
    });
    std::fs::write(&path, bytes).unwrap();
    let settings = TextureSettings {
        usage: Usage::Displacement,
        ..TextureSettings::default()
    };
    let (info, levels) = mips(&cook(vec![texture(&path, "t", settings)]), "t");
    assert_eq!(info.header.vk_format, vk::R32_SFLOAT);
    for (m, level) in levels[0].iter().enumerate() {
        for t in level.chunks_exact(4) {
            let h = f32::from_le_bytes(t.try_into().unwrap());
            assert!((-2.5..=3.0).contains(&h), "mip {m}: {h}");
        }
    }
}

/// A material's non-finite heights settle before filtering, which would spread them through
/// the mips: the smallest mip is the mean of the finite image, not a hole.
#[test]
fn non_finite_material_heights_stay_out_of_the_mips() {
    let path = dir("height-non-finite").join("heights.glb");
    let n = 32u32;
    let format = ctt::TargetFormat::Uncompressed(ctt::Format::R32G32B32A32_SFLOAT);
    let image = float_ktx2(n, n, format, false, |x, y| {
        let height = match (x, y) {
            (5, 5) => f32::NAN,
            (9, 9) => f32::INFINITY,
            (25, 25) => f32::NEG_INFINITY,
            _ if x < n / 2 => 1.0,
            _ => 3.0,
        };
        [height, 0.0, 0.0, 1.0]
    });
    write_glb(&path, &[height_material(0.2, 0.0)], &[image]);
    let (a, m) = cook_glb(&path, GltfSettings::default());
    let texture = a.find(m.textures[slot::DISPLACEMENT].texture).unwrap();
    let (_, levels) = mips(&a, a.name(texture));
    let last = levels[0].last().unwrap();
    let mean = f32::from(u16::from_le_bytes([last[0], last[1]])) / 65535.0;
    assert!((mean - 0.5).abs() < 0.05, "{mean}");
}

/// With no factor to compensate in, the stored format carries the sign. Every mip of these
/// negative red texels stays negative, down through the partial blocks.
#[test]
fn signed_standalone_sources_keep_their_values_in_signed_formats() {
    let d = dir("snorm-standalone");
    let (w, h) = (18u32, 10u32);
    let red = |x: u32| -0.7 + 0.4 * x as f32 / (w - 1) as f32;
    let green = |y: u32| 0.3 + 0.4 * y as f32 / (h - 1) as f32;
    let read = |vk_format: u32, w: u32, h: u32, data: &[u8]| -> Vec<[f32; 4]> {
        if vk_format == vk::R32_SFLOAT {
            return data
                .chunks_exact(4)
                .map(|t| [f32::from_le_bytes(t.try_into().unwrap()), 0.0, 0.0, 0.0])
                .collect();
        }
        if vk_format == vk::R16_SNORM {
            return data
                .chunks_exact(2)
                .map(|t| {
                    let v = f32::from(i16::from_le_bytes([t[0], t[1]])) / 32767.0;
                    [v, 0.0, 0.0, 0.0]
                })
                .collect();
        }
        decode(vk_format, w, h, data)
    };
    for (usage, source, vk_format, channels) in [
        (Usage::Linear1, ctt::Format::R8_SNORM, vk::BC4_SNORM, 1),
        (Usage::Linear2, ctt::Format::R8G8_SNORM, vk::BC5_SNORM, 2),
        (
            Usage::Linear2,
            ctt::Format::R32G32B32A32_SFLOAT,
            vk::BC5_SNORM,
            2,
        ),
        (
            Usage::Displacement,
            ctt::Format::R32G32B32A32_SFLOAT,
            vk::R32_SFLOAT,
            1,
        ),
        (
            Usage::Displacement,
            ctt::Format::R16_SNORM,
            vk::R16_SNORM,
            1,
        ),
    ] {
        let path = d.join(format!("{vk_format}.ktx2"));
        let format = ctt::TargetFormat::Uncompressed(source);
        let bytes = float_ktx2(w, h, format, false, |x, y| [red(x), green(y), 0.0, 1.0]);
        std::fs::write(&path, bytes).unwrap();
        let settings = TextureSettings {
            usage,
            filter: orr_cook::MipFilter::Box,
            ..TextureSettings::default()
        };
        let (info, levels) = mips(&cook(vec![texture(&path, "t", settings)]), "t");
        assert_eq!(info.header.vk_format, vk_format);
        assert_eq!(levels[0].len(), 5);
        let texels = read(vk_format, w, h, &levels[0][0]);
        for (i, t) in texels.iter().enumerate() {
            let want = [red(i as u32 % w), green(i as u32 / w)];
            for c in 0..channels {
                assert!(
                    (t[c] - want[c]).abs() < 0.02,
                    "texel {i}: {t:?}, not {want:?}"
                );
            }
        }
        for (m, level) in levels[0].iter().enumerate() {
            let [mw, mh] = orr_pack::texture::mip_size(w, h, m as u32);
            for t in read(vk_format, mw, mh, level) {
                assert!((t[0] + 0.5).abs() < 0.25, "mip {m}: {t:?}");
                assert!(channels == 1 || (t[1] - 0.5).abs() < 0.25, "mip {m}: {t:?}");
            }
        }
    }

    // A float source's values say whether it is signed, in the channels the usage keeps.
    let format = ctt::TargetFormat::Uncompressed(ctt::Format::R32G32B32A32_SFLOAT);
    let bytes = float_ktx2(w, h, format, false, |_, _| [0.5, -0.5, 0.0, 1.0]);
    let path = d.join("positive.ktx2");
    std::fs::write(&path, bytes).unwrap();
    let settings = TextureSettings {
        usage: Usage::Linear1,
        ..TextureSettings::default()
    };
    let (info, _) = mips(&cook(vec![texture(&path, "t", settings)]), "t");
    assert_eq!(info.header.vk_format, vk::BC4_UNORM);

    // A cutoff makes the channel an alpha, 0..1 whatever the source stores.
    let settings = TextureSettings {
        usage: Usage::Linear1,
        alpha_cutoff: Some(0.5),
        ..TextureSettings::default()
    };
    let path = d.join(format!("{}.ktx2", vk::BC4_SNORM));
    let (info, _) = mips(&cook(vec![texture(&path, "t", settings)]), "t");
    assert_eq!(info.header.vk_format, vk::BC4_UNORM);

    // A source already in the signed format keeps its blocks.
    for (usage, format) in [
        (Usage::Linear1, ctt::Format::BC4_SNORM_BLOCK),
        (Usage::Linear2, ctt::Format::BC5_SNORM_BLOCK),
    ] {
        let bytes = float_ktx2(16, 16, signed_blocks(format), true, |x, y| {
            [-(x as f32) / 16.0, -(y as f32) / 16.0, 0.0, 1.0]
        });
        let src = ctt::input::decode_container(&bytes, Default::default())
            .unwrap()
            .unwrap();
        let path = d.join("blocks.ktx2");
        std::fs::write(&path, &bytes).unwrap();
        let settings = TextureSettings {
            usage,
            ..TextureSettings::default()
        };
        let (_, levels) = mips(&cook(vec![texture(&path, "t", settings)]), "t");
        assert_eq!(levels[0].len(), src.surfaces[0].len());
        for (got, want) in levels[0].iter().zip(&src.surfaces[0]) {
            assert_eq!(got[..], want.data[..]);
        }
    }
}

/// A constant out of 0..1 folds into a factor as the value a texture of it would store.
#[test]
fn constants_fold_into_factors_clamped_to_their_domain() {
    let d = dir("constant-clamp");
    let material = r#"{"alphaMode":"BLEND",
        "pbrMetallicRoughness":{"baseColorTexture":{"index":0},"metallicRoughnessTexture":{"index":0},
            "baseColorFactor":[0.5,0.5,0.5,0.5],"metallicFactor":0.5,"roughnessFactor":0.5},
        "emissiveTexture":{"index":0},"emissiveFactor":[0.5,0.5,0.5],
        "extensions":{
            "KHR_materials_specular":{"specularTexture":{"index":0},"specularFactor":0.5},
            "KHR_materials_clearcoat":{"clearcoatTexture":{"index":0},"clearcoatFactor":0.5},
            "KHR_materials_transmission":{"transmissionTexture":{"index":0},"transmissionFactor":0.5}}}"#;
    for (format, value, want) in [
        (ctt::Format::R8G8B8A8_SNORM, -0.2, 0.0),
        (ctt::Format::R32G32B32A32_SFLOAT, 1.5, 0.5),
    ] {
        let format = ctt::TargetFormat::Uncompressed(format);
        let image = float_ktx2(8, 8, format, false, |_, _| [value; 4]);
        let path = d.join("constant.glb");
        write_glb(&path, &[material.into()], &[image]);
        let (_, m) = cook_glb(&path, GltfSettings::default());
        assert!(m.textures.iter().all(|b| b.texture == AssetId::NONE));
        for got in [
            m.base_color[0],
            m.geometry_opacity,
            m.emission_color[0],
            m.specular_roughness,
            m.metalness,
            m.specular_weight,
            m.coat_weight,
            m.transmission_weight,
        ] {
            assert_eq!(got, want, "{m:?}");
        }
    }
}

/// A weight's varying texels clamp before its factor, as a constant's do.
#[test]
fn weight_texels_clamp_before_their_factor() {
    let path = dir("weight-clamp").join("weight.glb");
    let material = r#"{"extensions":{"KHR_materials_clearcoat":
        {"clearcoatTexture":{"index":0},"clearcoatFactor":0.5}}}"#;
    let format = ctt::TargetFormat::Uncompressed(ctt::Format::R32G32B32A32_SFLOAT);
    let image = float_ktx2(8, 8, format, false, |x, _| {
        [if x < 4 { 1.5 } else { 0.25 }; 4]
    });
    write_glb(&path, &[material.into()], &[image]);
    let (a, m) = cook_glb(&path, GltfSettings::default());
    let texture = a.find(m.textures[slot::COAT].texture).unwrap();
    let (info, levels) = mips(&a, a.name(texture));
    let texels = decode(info.header.vk_format, 8, 8, &levels[0][0]);
    assert!((texels[0][0] - 0.5).abs() < 0.01, "{:?}", texels[0]);
}

/// No format says how float texels hold normals, so their vectors decide, on both paths: unit
/// vectors read as they are, and 0..1 values, stray texel or not, at `2v - 1`.
#[test]
fn float_normals_read_the_way_that_gives_unit_vectors() {
    let d = dir("float-normals");
    let n = 32u32;
    let (x, z) = (0.6f32, 0.8f32);
    let tilted = |px: u32| if px < n / 2 { x } else { -x };
    let packed = |v: f32| 0.5 * v + 0.5;
    let raw = ctt::TargetFormat::Uncompressed;
    let rgb = || raw(ctt::Format::R32G32B32A32_SFLOAT);
    let rg = || raw(ctt::Format::R16G16_SFLOAT);
    // One firefly, and one ringing texel inside the circle only as stored.
    let stray = |px: u32, py: u32, v: f32, s: f32| if (px, py) == (3, 3) { s } else { v };
    let files = [
        float_ktx2(n, n, rgb(), false, |px, _| [tilted(px), 0.0, z, 1.0]),
        float_ktx2(n, n, rg(), false, |px, _| [tilted(px), 0.0, 0.0, 1.0]),
        float_ktx2(n, n, rgb(), false, |px, py| {
            [stray(px, py, packed(tilted(px)), -6e4), 0.5, packed(z), 1.0]
        }),
        float_ktx2(n, n, rg(), false, |px, py| {
            [stray(px, py, packed(tilted(px)), -0.05), 0.5, 0.0, 1.0]
        }),
    ];
    let check = |a: &Archive, name: &str, file: usize| {
        let (_, levels) = mips(a, name);
        let texels = decode(vk::BC5_UNORM, n, n, &levels[0][0]);
        for (px, want) in [(8usize, x), (24, -x)] {
            let got = 2.0 * texels[24 * n as usize + px][0] - 1.0;
            assert!((got - want).abs() < 0.03, "file {file}: {got}, not {want}");
        }
    };
    // Toksvig roughness reads the same vectors from either storage.
    let mut roughness = Vec::new();
    for (i, bytes) in files.iter().enumerate() {
        let path = d.join(format!("{i}.ktx2"));
        std::fs::write(&path, bytes).unwrap();
        let settings = TextureSettings {
            usage: Usage::Normal,
            ..TextureSettings::default()
        };
        check(&cook(vec![texture(&path, "n", settings)]), "n", i);

        let glb = d.join(format!("{i}.glb"));
        let material =
            r#"{"normalTexture":{"index":0},"pbrMetallicRoughness":{"roughnessFactor":0.25}}"#;
        write_glb(&glb, &[material.into()], std::slice::from_ref(bytes));
        let (a, m) = cook_glb_at(&glb, GltfSettings::default(), Quality::Slow);
        check(
            &a,
            a.name(a.find(m.textures[slot::NORMAL].texture).unwrap()),
            i,
        );
        let (info, levels) = mips(
            &a,
            a.name(a.find(m.textures[slot::ROUGHNESS].texture).unwrap()),
        );
        let last = u32::from(info.header.mip_count) - 1;
        roughness.push(decode(vk::BC4_UNORM, 1, 1, &levels[0][last as usize])[0][0]);
    }
    assert!(roughness[0] > 0.3, "{roughness:?}");
    assert!(
        roughness.iter().all(|r| (r - roughness[0]).abs() < 0.02),
        "{roughness:?}"
    );
}

fn bcn_material_checks(a: &Archive, remade: bool, blocks: [&Vec<Vec<u8>>; 5]) {
    let [
        color_blocks,
        normal_blocks,
        rm_blocks,
        specular_blocks,
        rough_blocks,
    ] = blocks;
    let n = 64;
    let material = |i: usize| {
        let e = a
            .find(AssetId::from_name(&format!("g/material/{i}")))
            .unwrap();
        Material::from_bytes(&read_blob(a, e, 0)).unwrap()
    };
    let (m0, m1) = (material(0), material(1));
    let name = |t: AssetId| a.name(a.find(t).unwrap()).to_string();
    let all = |t: AssetId, want: &[Vec<u8>]| {
        let (info, levels) = mips(a, &name(t));
        assert_eq!(info.header.mip_count as usize, want.len());
        assert_eq!(levels[0], want, "{}", name(t));
    };
    all(m0.textures[slot::BASE_COLOR].texture, color_blocks);
    all(m0.textures[slot::NORMAL].texture, normal_blocks);
    all(m1.textures[slot::SPECULAR_WEIGHT].texture, specular_blocks);
    assert_eq!(m1.textures[slot::SPECULAR_WEIGHT].channels, 0);
    // Without a normal map every mip passes; with one, mip 0 does.
    let rough = |m: &Material| {
        let b = (m.textures[slot::ROUGHNESS], m.textures[slot::METALNESS]);
        assert_eq!(b.0.texture, b.1.texture);
        (
            (b.0.channels, b.1.channels),
            mips(a, &name(b.0.texture)).1.remove(0),
        )
    };
    let (channels, levels) = rough(&m1);
    assert_eq!((channels, levels), ((0, 1), rm_blocks.clone()));
    let (channels, levels) = rough(&m0);
    assert_eq!(channels, (0, 1));
    if remade {
        assert_eq!(levels[0], rm_blocks[0]);
        assert_ne!(
            levels[1], rm_blocks[1],
            "Toksvig raises roughness below mip 0"
        );
    } else {
        assert_eq!(levels, *rm_blocks);
    }
    let m2 = material(2);
    assert_eq!(m2.textures[slot::METALNESS].texture, AssetId::NONE);
    let (info, levels) = mips(a, &name(m2.textures[slot::ROUGHNESS].texture));
    assert_eq!(info.header.vk_format, vk::BC4_UNORM);
    assert_eq!(levels[0].len(), rough_blocks.len());
    let red: Vec<u8> = rough_blocks[0]
        .chunks_exact(16)
        .flat_map(|b| &b[..8])
        .copied()
        .collect();
    assert_eq!(levels[0][0], red);
    // Opacity from the base color's alpha: the source's own alpha mips unless remade.
    let m3 = material(3);
    let (info, levels) = mips(a, &name(m3.textures[slot::OPACITY].texture));
    assert_eq!(info.header.vk_format, vk::BC4_UNORM);
    assert_eq!(info.header.alpha_cutoff, 0.5);
    let s = n / 2;
    let err = decode(vk::BC4_UNORM, s, s, &levels[0][1])
        .iter()
        .map(|t| (t[0] - 200.0 / 255.0).abs())
        .fold(0.0, f32::max);
    assert_eq!(err < 0.05, !remade, "{err}");
}

#[test]
fn gltf_materials_cook_their_canonical_channel_set() {
    let d = dir("gltf-textures");
    let n = 256;
    let images = vec![
        // 0: base color with a leafy alpha.
        png(n, n, |x, y| {
            let (dx, dy) = (x as f32 - 128.0, y as f32 - 128.0);
            let a = if dx * dx / 9000.0 + dy * dy / 3000.0 < 1.0 {
                255
            } else {
                0
            };
            [(x % 256) as u8, (y % 256) as u8, 90, a]
        }),
        // 1: metallic-roughness: varying roughness (G), constant metalness 51/255 (B).
        png(n, n, |x, _| [0, (40 + x / 2) as u8, 51, 255]),
        // 2: a bumpy normal map.
        png(n, n, |x, y| {
            let (sx, sy) = ((x as f32 * 0.3).sin() * 0.4, (y as f32 * 0.2).cos() * 0.4);
            let l = (1.0 + sx * sx + sy * sy).sqrt();
            let e = |v: f32| ((v / l * 0.5 + 0.5) * 255.0).round() as u8;
            [e(-sx), e(-sy), e(1.0), 255]
        }),
        // 3: constant emissive, folded into the factor.
        png(8, 8, |_, _| [255, 128, 0, 255]),
    ];
    let mat = |name: &str, metal: f32| {
        format!(
            r#"{{"name":"{name}","alphaMode":"MASK","alphaCutoff":0.4,
               "pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}},"metallicFactor":{metal},
                 "roughnessFactor":0.8,"metallicRoughnessTexture":{{"index":1}}}},
               "normalTexture":{{"index":2,"scale":0.5}},
               "emissiveTexture":{{"index":3}},"emissiveFactor":[2,2,2]}}"#
        )
    };
    let path = d.join("quad.glb");
    // Material 1 differs only in its metalness factor, which folds into the material
    // because the metalness channel is constant, so the two share every texture.
    write_glb(&path, &[mat("a", 1.0), mat("b", 0.5)], &images);
    let a = cook(vec![AssetSpec {
        name: "quad".into(),
        source: Source::Gltf {
            path,
            settings: GltfSettings::default(),
        },
    }]);
    let model = a.find(AssetId::from_name("quad")).unwrap();
    let model = ModelInfo::parse(&read_blob(&a, model, 0)).unwrap();
    let mats: Vec<Material> = model
        .materials
        .iter()
        .map(|id| Material::from_bytes(&read_blob(&a, a.find(*id).unwrap(), 0)).unwrap())
        .collect();
    let [m0, m1] = [mats[0], mats[1]];
    for (m, metal) in [(m0, 1.0), (m1, 0.5)] {
        let t = |s: usize| m.textures[s].texture;
        assert_ne!(t(slot::BASE_COLOR), AssetId::NONE);
        assert_ne!(t(slot::OPACITY), AssetId::NONE);
        assert_ne!(t(slot::NORMAL), AssetId::NONE);
        assert_ne!(t(slot::ROUGHNESS), AssetId::NONE);
        assert_eq!(t(slot::METALNESS), AssetId::NONE);
        assert_eq!(t(slot::EMISSION), AssetId::NONE);
        assert!(
            (m.metalness - metal * 0.2).abs() < 1e-6,
            "metalness {}",
            m.metalness
        );
        assert_eq!(
            m.specular_roughness, 1.0,
            "roughness is folded into its texture"
        );
        assert_eq!(m.geometry_opacity, 1.0);
        let want = 2.0 * orr_color::srgb_decode(128.0 / 255.0) as f32;
        assert!((m.emission_color[1] - want).abs() < 1e-4);
        assert_eq!(m.emission_color[0], 2.0);
        assert_eq!(m.emission_color[2], 0.0);
    }
    for s in 0..slot::COUNT {
        assert_eq!(m0.textures[s], m1.textures[s], "slot {s} is shared");
    }
    let format = |s: usize| {
        let e = a.find(m0.textures[s].texture).unwrap();
        TextureInfo::parse(&read_blob(&a, e, 0)).unwrap().header
    };
    assert_eq!(format(slot::BASE_COLOR).vk_format, vk::BC7_SRGB);
    assert_eq!(format(slot::NORMAL).vk_format, vk::BC5_UNORM);
    assert_eq!(format(slot::ROUGHNESS).vk_format, vk::BC4_UNORM);
    let opacity = format(slot::OPACITY);
    assert_eq!(
        (opacity.vk_format, opacity.alpha_cutoff),
        (vk::BC4_UNORM, 0.4)
    );

    // Toksvig: the roughness texture's coarse mips are rougher than the base value.
    let (rough, levels) = mips(&a, "quad/texture/0/roughness");
    let [w, h] = rough.header.mip_size(4);
    let mip4 = decode(vk::BC4_UNORM, w, h, &levels[0][4]);
    let mip0 = decode(vk::BC4_UNORM, n, n, &levels[0][0]);
    let mean = |m: &[[f32; 4]]| m.iter().map(|p| p[0]).sum::<f32>() / m.len() as f32;
    assert!(mean(&mip4) > mean(&mip0) + 0.02);

    // Resident bytes of the packed set against one texture per image (BC7 for color and
    // packed data, BC5 for normals).
    let packed: u64 = a
        .entries()
        .iter()
        .filter(|e| e.class == Class::Texture as u16)
        .map(|e| {
            let h = TextureInfo::parse(&read_blob(&a, e, 0)).unwrap().header;
            chain_bytes(h.vk_format, h.width, h.height)
        })
        .sum();
    let per_image = chain_bytes(vk::BC7_SRGB, n, n)
        + chain_bytes(vk::BC7_UNORM, n, n)
        + chain_bytes(vk::BC5_UNORM, n, n)
        + chain_bytes(vk::BC7_SRGB, 8, 8);
    println!("texture bytes: packed {packed}, one texture per image {per_image}");
}

#[test]
fn a_constant_color_folds_into_its_factor_in_the_image_primaries() {
    let d = dir("gltf-constant-primaries");
    let path = d.join("quad.glb");
    let texel = [255u8, 128, 0];
    write_glb(
        &path,
        &[r#"{"emissiveTexture":{"index":0},"emissiveFactor":[2,1,0.5]}"#.into()],
        &[png(8, 8, |_, _| [texel[0], texel[1], texel[2], 255])],
    );
    let a = cook(vec![AssetSpec {
        name: "quad".into(),
        source: Source::Gltf {
            path,
            settings: GltfSettings {
                textures: TextureRules {
                    overrides: vec![(
                        "image 0".into(),
                        TextureOverride {
                            primaries: Some(Primaries::DisplayP3),
                            ..TextureOverride::default()
                        },
                    )],
                    ..TextureRules::default()
                },
                ..GltfSettings::default()
            },
        },
    }]);
    let model = a.find(AssetId::from_name("quad")).unwrap();
    let model = ModelInfo::parse(&read_blob(&a, model, 0)).unwrap();
    let m = Material::from_bytes(&read_blob(&a, a.find(model.materials[0]).unwrap(), 0)).unwrap();
    assert_eq!(m.textures[slot::EMISSION].texture, AssetId::NONE);
    let (rec709, p3) = (RgbPrimaries::REC709, RgbPrimaries::DISPLAY_P3);
    let factor = rec709.matrix_to(p3) * orr_math::DVec3::new(2.0, 1.0, 0.5);
    let texel = texel.map(|c| orr_color::srgb_decode(f64::from(c) / 255.0));
    let want = p3.matrix_to(rec709) * (factor * orr_math::DVec3::from(texel));
    for c in 0..3 {
        assert!(
            (f64::from(m.emission_color[c]) - want[c]).abs() < 1e-4,
            "emission {:?}, want {want:?}",
            m.emission_color
        );
    }
}
