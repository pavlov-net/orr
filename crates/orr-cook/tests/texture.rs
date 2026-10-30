#![allow(clippy::print_stdout)] // the tests report measured figures

use std::f32::consts::{PI, TAU};
use std::path::{Path, PathBuf};

use orr_cook::{
    ArchiveSpec, AssetSpec, GltfSettings, Manifest, Quality, Source, TextureSettings, Usage,
};
use orr_pack::material::{Material, slot};
use orr_pack::model::ModelInfo;
use orr_pack::texture::{FormatInfo, TILE_BYTES, TextureInfo, octahedral, tile_grid, vk};
use orr_pack::{Archive, AssetId, Class, Target};

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cook(assets: Vec<AssetSpec>) -> Archive {
    let cooked = orr_cook::cook(&Manifest {
        archives: vec![ArchiveSpec {
            output: Target::Memory,
            cell_log2: ArchiveSpec::DEFAULT_CELL_LOG2,
            assets,
        }],
    })
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

fn read_blob(a: &Archive, e: &orr_pack::Entry, i: usize) -> Vec<u8> {
    let b = &a.blobs(e)[i];
    let mut out = vec![0; b.raw_len as usize];
    a.read(b, &mut out).unwrap();
    out
}

/// Header and every mip of every layer, tight, from the archive.
fn mips(a: &Archive, name: &str) -> (TextureInfo, Vec<Vec<Vec<u8>>>) {
    let e = *a
        .expect(
            AssetId::from_name(name),
            Class::Texture as u16,
            orr_pack::texture::SCHEMA,
        )
        .unwrap();
    a.verify(&e).unwrap();
    let info = TextureInfo::parse(&read_blob(a, &e, 0)).unwrap();
    let h = info.header;
    assert_eq!(a.blobs(&e).len(), h.blob_count());
    for (i, len) in info.blob_lens().into_iter().enumerate() {
        let b = &a.blobs(&e)[i + 1];
        assert_eq!(b.raw_len as usize, len);
        let tail = i == h.blob_count() - 2 && h.tail_first_mip < h.mip_count;
        if tail {
            assert!(b.is_mount(), "the mip tail is resident from mount");
        } else {
            // One 64 KiB tile per chunk.
            assert!(!b.is_mount());
            let [gx, gy] = tile_grid(&h, i as u32);
            assert_eq!(u32::from(b.chunk_count), gx * gy * u32::from(h.layers));
            assert_eq!(b.chunk_raw_len() as usize, TILE_BYTES);
        }
    }
    let levels = (0..u32::from(h.layers))
        .map(|l| {
            (0..u32::from(h.mip_count))
                .map(|m| {
                    let blob = read_blob(a, &e, h.mip_blob(m));
                    info.mip_texels(&blob, m, l).unwrap()
                })
                .collect()
        })
        .collect();
    (info, levels)
}

fn decode(vk_format: u32, w: u32, h: u32, data: &[u8]) -> Vec<[f32; 4]> {
    let f = FormatInfo::of(vk_format).unwrap();
    let bw = w.div_ceil(4) as usize;
    let mut out = vec![[0.0; 4]; (w * h) as usize];
    for (bi, block) in data.chunks_exact(f.bytes as usize).enumerate() {
        let mut t = [[0.0f32; 4]; 16];
        let mut b8 = [0u8; 64];
        match vk_format {
            vk::BC7_SRGB | vk::BC7_UNORM => {
                bcdec_rs::bc7(block, &mut b8, 16);
                for i in 0..16 {
                    t[i] = std::array::from_fn(|c| f32::from(b8[i * 4 + c]) / 255.0);
                }
            }
            vk::BC4_UNORM => {
                bcdec_rs::bc4(block, &mut b8[..16], 4, false);
                for i in 0..16 {
                    t[i][0] = f32::from(b8[i]) / 255.0;
                }
            }
            vk::BC5_UNORM => {
                bcdec_rs::bc5(block, &mut b8[..32], 8, false);
                for i in 0..16 {
                    t[i][0] = f32::from(b8[i * 2]) / 255.0;
                    t[i][1] = f32::from(b8[i * 2 + 1]) / 255.0;
                }
            }
            vk::BC6H_UFLOAT => {
                let mut f = [0.0f32; 48];
                bcdec_rs::bc6h_float(block, &mut f, 12, false);
                for i in 0..16 {
                    t[i] = [f[i * 3], f[i * 3 + 1], f[i * 3 + 2], 1.0];
                }
            }
            _ => panic!("format {vk_format}"),
        }
        let (bx, by) = (bi % bw, bi / bw);
        for (i, p) in t.iter().enumerate() {
            let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
            if x < w as usize && y < h as usize {
                out[y * w as usize + x] = *p;
            }
        }
    }
    out
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
    let path = d.join("colour.png");
    write_png(&path, w, h, px);
    let a = cook(vec![texture(
        &path,
        "colour",
        TextureSettings {
            quality: Quality::Fast,
            ..TextureSettings::default()
        },
    )]);
    let (info, levels) = mips(&a, "colour");
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
            quality: ctt::Quality::Fast,
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
        let d = dir[0] * c[0] + dir[1] * c[1] + dir[2] * c[2];
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
            quality: Quality::Fast,
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
    let discs = discs.map(|(c, r, v)| {
        let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        (c.map(|x| x / l), r, v)
    });
    let radiance = |dir: [f32; 3]| {
        let mut v = [0.4, 0.6, 1.0];
        for (c, r, e) in &discs {
            let cos = dir[0] * c[0] + dir[1] * c[1] + dir[2] * c[2];
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
            quality: Quality::Fast,
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
        let cos = (d[0] * c[0] + d[1] * c[1] + d[2] * c[2]).min(1.0);
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

/// A GLB holding a quad, the given materials and PNG images.
fn write_glb(path: &Path, materials: &[String], images: &[Vec<u8>]) {
    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut view = |bin: &mut Vec<u8>, bytes: &[u8]| {
        views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            bin.len(),
            bytes.len()
        ));
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        views.len() - 1
    };
    let pos: [[f32; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let nrm = [[0.0f32, 0.0, 1.0]; 4];
    let uv: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let idx: [u32; 6] = [0, 1, 2, 0, 2, 3];
    let vp = view(&mut bin, bytemuck::cast_slice(&pos));
    let vn = view(&mut bin, bytemuck::cast_slice(&nrm));
    let vu = view(&mut bin, bytemuck::cast_slice(&uv));
    let vi = view(&mut bin, bytemuck::cast_slice(&idx));
    let image_views: Vec<usize> = images.iter().map(|i| view(&mut bin, i)).collect();
    let accessors = format!(
        r#"[{{"bufferView":{vp},"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
            {{"bufferView":{vn},"componentType":5126,"count":4,"type":"VEC3"}},
            {{"bufferView":{vu},"componentType":5126,"count":4,"type":"VEC2"}},
            {{"bufferView":{vi},"componentType":5125,"count":6,"type":"SCALAR"}}]"#
    );
    let prims: Vec<String> = (0..materials.len())
        .map(|m| {
            format!(
                r#"{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":3,"material":{m}}}"#
            )
        })
        .collect();
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
            "meshes":[{{"primitives":[{}]}}],"materials":[{}],
            "textures":[{}],"images":[{}],"samplers":[{{}}],
            "accessors":{accessors},"bufferViews":[{}],"buffers":[{{"byteLength":{}}}]}}"#,
        prims.join(","),
        materials.join(","),
        (0..images.len())
            .map(|i| format!(r#"{{"source":{i},"sampler":0}}"#))
            .collect::<Vec<_>>()
            .join(","),
        image_views
            .iter()
            .map(|v| format!(r#"{{"bufferView":{v},"mimeType":"image/png"}}"#))
            .collect::<Vec<_>>()
            .join(","),
        views.join(","),
        bin.len()
    );
    let mut json = json.into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);
    std::fs::write(path, glb).unwrap();
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

#[test]
fn gltf_materials_cook_their_canonical_channel_set() {
    let d = dir("gltf-textures");
    let n = 256;
    let images = vec![
        // 0: base colour with a leafy alpha.
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
        // 3: constant occlusion of 1: dropped.
        png(16, 16, |_, _| [255, 255, 255, 255]),
        // 4: constant emissive, folded into the factor.
        png(8, 8, |_, _| [255, 128, 0, 255]),
    ];
    let mat = |name: &str, metal: f32| {
        format!(
            r#"{{"name":"{name}","alphaMode":"MASK","alphaCutoff":0.4,
               "pbrMetallicRoughness":{{"baseColorTexture":{{"index":0}},"metallicFactor":{metal},
                 "roughnessFactor":0.8,"metallicRoughnessTexture":{{"index":1}}}},
               "normalTexture":{{"index":2,"scale":0.5}},"occlusionTexture":{{"index":3}},
               "emissiveTexture":{{"index":4}},"emissiveFactor":[2,2,2]}}"#
        )
    };
    let path = d.join("quad.glb");
    // Material 1 differs only in its metalness factor, which folds into the material
    // because the metalness channel is constant, so the two share every texture.
    write_glb(&path, &[mat("a", 1.0), mat("b", 0.5)], &images);
    let a = cook(vec![AssetSpec {
        name: "quad".into(),
        source: Source::Gltf {
            path: path.clone(),
            settings: GltfSettings {
                textures: orr_cook::TextureRules {
                    quality: Quality::Fast,
                    ..Default::default()
                },
                ..Default::default()
            },
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
        assert_eq!(t(slot::OCCLUSION), AssetId::NONE);
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
        let want = 2.0 * ((128.0f32 / 255.0 + 0.055) / 1.055).powf(2.4);
        assert!((m.emission_color[1] - want).abs() < 1e-4);
        assert_eq!(m.emission_color[0], 2.0);
        assert_eq!(m.emission_color[2], 0.0);
    }
    for s in 0..12 {
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

    // Resident bytes of the packed set against one texture per image (BC7 for colour and
    // packed data, BC5 for normals, BC4 for occlusion).
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
        + chain_bytes(vk::BC4_UNORM, 16, 16)
        + chain_bytes(vk::BC7_SRGB, 8, 8);
    println!("texture bytes: packed {packed}, one texture per image {per_image}");
}
