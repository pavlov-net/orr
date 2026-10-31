#![allow(clippy::print_stdout)] // the tests report measured figures

//! Cooks external texture sets; ignored unless asked for, since the sets live outside the
//! repository.
//!
//! - `ORR_TEXTURE_SET=<dir>`: every PNG under `dir`, usage from its file name. With
//!   `ORR_ATLAS=<dir name>:<cells>`, images in directories of that name are atlases of
//!   `cells` x `cells`. Checks alpha coverage per mip, and per atlas cell.
//!   `ORR_TEXTURE_ONLY=<substring>` keeps the paths that contain it.
//! - `ORR_HDR_SET=<dir>`: every `.hdr` under `dir` as a 1024^2 environment map; prints the
//!   PSNR of the map resampled back to a 1024 x 512 equirect against the source.
//!
//! Both check that every compressed BCn tile is stored split and decodes to its blocks.
//!
//! - `ORR_NTC_SET=<dir>` (with the `ntc` feature, LibNTC and a CUDA device): every glTF
//!   file under `dir` cooks neural texture sets beside its BC textures; every set holds no
//!   more bytes than the BC textures it stands in for and every texture meets the PSNR
//!   floor.
//!
//! `cargo test --release -p orr-cook --test texture_sets -- --ignored --nocapture`

use std::path::{Path, PathBuf};

use orr_cook::{ArchiveSpec, AssetSpec, Manifest, Source, TextureSettings, Usage};
use orr_pack::texture::{FormatInfo, TextureInfo, octahedral, vk};
use orr_pack::{Archive, AssetId, BcLayout, Codec, Filter, Target};

fn files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            files(&p, ext, out);
        } else if p.extension().is_some_and(|x| x == ext) {
            out.push(p);
        }
    }
}

/// Verifies every texture of `a`, which decodes each blob chunk by chunk through the BCn
/// split and checks the bytes, and returns the number of split chunks.
fn split_chunks(a: &Archive) -> u32 {
    let mut n = 0;
    for e in a.entries() {
        a.verify(e).unwrap();
        let info = TextureInfo::parse(&read_blob(a, e, 0)).unwrap();
        let Some(layout) = BcLayout::of(info.header.vk_format) else {
            continue;
        };
        for b in &a.blobs(e)[1..] {
            if b.codec() != Codec::None {
                assert_eq!(b.filter(), Filter::BcSplit(layout), "{}", a.name(e));
                n += u32::from(b.chunk_count);
            }
        }
    }
    n
}

fn read_blob(a: &Archive, e: &orr_pack::Entry, i: usize) -> Vec<u8> {
    let b = &a.blobs(e)[i];
    let mut out = vec![0; b.raw_len as usize];
    a.read(b, &mut out).unwrap();
    out
}

/// Channel `c` of a decoded BC7, BC4 or BC6H mip.
fn decode(vk_format: u32, w: u32, h: u32, data: &[u8], c: usize) -> Vec<f32> {
    let f = FormatInfo::of(vk_format).unwrap();
    let bw = w.div_ceil(4) as usize;
    let mut out = vec![0.0; (w * h) as usize];
    for (bi, block) in data.chunks_exact(f.bytes as usize).enumerate() {
        let mut t = [0.0f32; 16];
        match vk_format {
            vk::BC7_SRGB | vk::BC7_UNORM => {
                let mut b = [0u8; 64];
                bcdec_rs::bc7(block, &mut b, 16);
                t = std::array::from_fn(|i| f32::from(b[i * 4 + c]) / 255.0);
            }
            vk::BC4_UNORM => {
                let mut b = [0u8; 16];
                bcdec_rs::bc4(block, &mut b, 4, false);
                t = std::array::from_fn(|i| f32::from(b[i]) / 255.0);
            }
            vk::BC6H_UFLOAT => {
                let mut f = [0.0f32; 48];
                bcdec_rs::bc6h_float(block, &mut f, 12, false);
                t = std::array::from_fn(|i| f[i * 3 + c]);
            }
            _ => {}
        }
        let (bx, by) = (bi % bw, bi / bw);
        for (i, v) in t.iter().enumerate() {
            let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
            if x < w as usize && y < h as usize {
                out[y * w as usize + x] = *v;
            }
        }
    }
    out
}

/// The cook's coverage metric: 4x4 bilinear sub-samples per texel inside the rectangle.
fn coverage(a: &[f32], w: u32, [x0, y0, rw, rh]: [u32; 4], cutoff: f32) -> f32 {
    let at = |x: i64, y: i64| {
        let x = x.clamp(i64::from(x0), i64::from(x0 + rw) - 1) as usize;
        let y = y.clamp(i64::from(y0), i64::from(y0 + rh) - 1) as usize;
        a[y * w as usize + x]
    };
    let (mut hit, mut n) = (0u64, 0u64);
    for y in y0..y0 + rh {
        for x in x0..x0 + rw {
            for j in 0..4 {
                for i in 0..4 {
                    let px = x as f32 + (i as f32 + 0.5) / 4.0 - 0.5;
                    let py = y as f32 + (j as f32 + 0.5) / 4.0 - 0.5;
                    let (fx0, fy0) = (px.floor(), py.floor());
                    let (fx, fy) = (px - fx0, py - fy0);
                    let (ix, iy) = (fx0 as i64, fy0 as i64);
                    let top = at(ix, iy) * (1.0 - fx) + at(ix + 1, iy) * fx;
                    let bot = at(ix, iy + 1) * (1.0 - fx) + at(ix + 1, iy + 1) * fx;
                    hit += u64::from(top * (1.0 - fy) + bot * fy >= cutoff);
                    n += 1;
                }
            }
        }
    }
    hit as f32 / n as f32
}

#[test]
#[ignore = "needs ORR_TEXTURE_SET"]
fn texture_set_cooks() {
    let dir = PathBuf::from(std::env::var("ORR_TEXTURE_SET").expect("ORR_TEXTURE_SET"));
    let atlas = std::env::var("ORR_ATLAS").ok().map(|s| {
        let (d, n) = s.split_once(':').expect("ORR_ATLAS=<dir name>:<cells>");
        (d.to_string(), n.parse::<u16>().unwrap())
    });
    let mut pngs = Vec::new();
    files(&dir, "png", &mut pngs);
    if let Ok(only) = std::env::var("ORR_TEXTURE_ONLY") {
        pngs.retain(|p| p.to_string_lossy().replace('\\', "/").contains(&only));
    }
    pngs.sort();
    let mut assets = Vec::new();
    let mut checks = Vec::new();
    for p in &pngs {
        let name = p
            .strip_prefix(&dir)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let file = p.file_stem().unwrap().to_string_lossy().to_lowercase();
        let (usage, cutoff) = if file.contains("normal_depth") {
            (Usage::Linear4, None)
        } else if file.contains("normal") {
            (Usage::Normal, None)
        } else if file.contains("basecolor") || file.contains("albedo") {
            (Usage::ColorAlpha, Some(0.5))
        } else if ["occlusion", "visibility", "mask", "weight"]
            .iter()
            .any(|k| file.contains(k))
        {
            (Usage::Linear1, None)
        } else {
            (Usage::Linear4, None)
        };
        let grid = atlas.as_ref().and_then(|(d, n)| {
            let parent = p.parent()?.file_name()?.to_string_lossy();
            (parent == *d).then_some([*n, *n])
        });
        let (w, h) = image::image_dimensions(p).unwrap();
        let grid = grid.filter(|g| w % u32::from(g[0]) == 0 && h % u32::from(g[1]) == 0);
        if cutoff.is_some() {
            checks.push((name.clone(), grid));
        }
        assets.push(AssetSpec {
            name,
            source: Source::Texture {
                path: p.clone(),
                settings: TextureSettings {
                    usage,
                    alpha_cutoff: cutoff,
                    atlas_grid: grid,
                    ..TextureSettings::default()
                },
            },
        });
    }
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("texture-set.orr");
    let t = std::time::Instant::now();
    orr_cook::cook(
        &Manifest {
            archives: vec![ArchiveSpec {
                output: Target::File(out.clone()),
                cell_log2: ArchiveSpec::DEFAULT_CELL_LOG2,
                gdeflate: false,
                assets,
            }],
        },
        &orr_cook::CookOptions {
            force: true,
            ..Default::default()
        },
    )
    .unwrap();
    println!(
        "{} PNGs cooked in {:.1} s to {} MiB",
        pngs.len(),
        t.elapsed().as_secs_f64(),
        std::fs::metadata(&out).unwrap().len() >> 20
    );
    let a = Archive::open(&out).unwrap();
    println!(
        "{} tiles decode exact through the BCn split",
        split_chunks(&a)
    );
    let (mut worst, mut over, mut count) = ([0.0f32; 4], [0u32; 4], [0u32; 4]);
    for (name, grid) in checks {
        let e = a.find(AssetId::from_name(&name)).unwrap();
        let info = TextureInfo::parse(&read_blob(&a, e, 0)).unwrap();
        let h = info.header;
        let cutoff = h.alpha_cutoff;
        let mut target: Vec<f32> = Vec::new();
        for m in 0..u32::from(h.mip_count) {
            let [w, hh] = h.mip_size(m);
            let cells = grid
                .map(|g| g.map(u32::from))
                .filter(|g| w % g[0] == 0 && hh % g[1] == 0 && w / g[0] >= 4 && hh / g[1] >= 4);
            if w.min(hh) < 4 {
                break;
            }
            let blob = read_blob(&a, e, h.mip_blob(m));
            let alpha = decode(
                h.vk_format,
                w,
                hh,
                &info.mip_texels(&blob, m, 0).unwrap(),
                3,
            );
            let rects: Vec<[u32; 4]> = match cells {
                Some([gx, gy]) => (0..gy)
                    .flat_map(|cy| {
                        (0..gx).map(move |cx| [cx * w / gx, cy * hh / gy, w / gx, hh / gy])
                    })
                    .collect(),
                None => vec![[0, 0, w, hh]],
            };
            let got: Vec<f32> = rects
                .iter()
                .map(|&r| coverage(&alpha, w, r, cutoff))
                .collect();
            if m == 0 {
                target = got;
                continue;
            }
            let whole = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
            let (got, want, side) = if got.len() == target.len() {
                (got, target.clone(), rects[0][2].min(rects[0][3]))
            } else {
                (vec![whole(&got)], vec![whole(&target)], w.min(hh))
            };
            for (g, t) in got.iter().zip(&want) {
                let d = (g - t).abs();
                // Region edge in texels: 4, 8, 16, 32 and up.
                let bucket = (side.ilog2().clamp(2, 5) - 2) as usize;
                worst[bucket] = worst[bucket].max(d);
                over[bucket] += u32::from(d > 0.01);
                count[bucket] += 1;
            }
        }
    }
    for (b, edge) in ["4", "8", "16", "32+"].iter().enumerate() {
        println!(
            "decoded coverage against mip 0, regions of {edge} texels: {} regions, largest difference {:.4}, {} over 1%",
            count[b], worst[b], over[b]
        );
    }
    // The cooked chain holds 1% before encoding (unit tests); BC alpha rounding moves the
    // decoded coverage of small regions further.
    assert!(
        worst[2].max(worst[3]) <= 0.02,
        "regions of 16 texels or more: {:.4}",
        worst[2].max(worst[3])
    );
}

#[test]
#[ignore = "needs ORR_HDR_SET"]
fn hdr_set_round_trips() {
    let dir = PathBuf::from(std::env::var("ORR_HDR_SET").expect("ORR_HDR_SET"));
    let mut maps = Vec::new();
    files(&dir, "hdr", &mut maps);
    maps.sort();
    for p in maps {
        let src = image::open(&p).unwrap().to_rgb32f();
        let (sw, sh) = src.dimensions();
        if sw != 2 * sh {
            continue;
        }
        let cooked = orr_cook::cook(
            &Manifest {
                archives: vec![ArchiveSpec {
                    output: Target::Memory,
                    cell_log2: ArchiveSpec::DEFAULT_CELL_LOG2,
                    gdeflate: false,
                    assets: vec![AssetSpec {
                        name: "env".into(),
                        source: Source::Texture {
                            path: p.clone(),
                            settings: TextureSettings {
                                usage: Usage::Environment {
                                    light_threshold: None,
                                    max_lights: 0,
                                },
                                ..TextureSettings::default()
                            },
                        },
                    }],
                }],
            },
            &Default::default(),
        )
        .unwrap();
        let a = Archive::from_memory(std::sync::Arc::clone(&cooked.memory[0])).unwrap();
        let split = split_chunks(&a);
        let e = a.find(AssetId::from_name("env")).unwrap();
        let info = TextureInfo::parse(&read_blob(&a, e, 0)).unwrap();
        let n = info.header.width;
        let mip0 = info.mip_texels(&read_blob(&a, e, 1), 0, 0).unwrap();
        let map: Vec<[f32; 3]> = {
            let ch: Vec<Vec<f32>> = (0..3)
                .map(|c| decode(vk::BC6H_UFLOAT, n, n, &mip0, c))
                .collect();
            (0..(n * n) as usize)
                .map(|i| [ch[0][i], ch[1][i], ch[2][i]])
                .collect()
        };
        let (w, h) = (1024u32, 512u32);
        let (kx, ky) = (sw / w, sh / h);
        let tm = |v: f32| v / (1.0 + v);
        let mut mse = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let mut s = [0.0f32; 3];
                for j in 0..ky {
                    for i in 0..kx {
                        let p = src.get_pixel(x * kx + i, y * ky + j);
                        for c in 0..3 {
                            s[c] += p[c] / (kx * ky) as f32;
                        }
                    }
                }
                let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                let (theta, phi) = (
                    v * std::f32::consts::PI,
                    u * std::f32::consts::TAU - std::f32::consts::PI,
                );
                let d = [
                    theta.sin() * phi.sin(),
                    theta.cos(),
                    -theta.sin() * phi.cos(),
                ];
                let [ms, mt] = octahedral::from_dir(d);
                let (fx, fy) = (ms * n as f32 - 0.5, mt * n as f32 - 0.5);
                let (x0, y0) = (fx.floor(), fy.floor());
                let (ax, ay) = (fx - x0, fy - y0);
                let at = |x: f32, y: f32| {
                    let c = |v: f32| (v.max(0.0) as u32).min(n - 1);
                    map[(c(y) * n + c(x)) as usize]
                };
                let m: [f32; 3] = std::array::from_fn(|c| {
                    let top = at(x0, y0)[c] * (1.0 - ax) + at(x0 + 1.0, y0)[c] * ax;
                    let bot = at(x0, y0 + 1.0)[c] * (1.0 - ax) + at(x0 + 1.0, y0 + 1.0)[c] * ax;
                    top * (1.0 - ay) + bot * ay
                });
                for c in 0..3 {
                    mse += f64::from(tm(s[c]) - tm(m[c])).powi(2);
                }
            }
        }
        let db = 10.0 * (f64::from(w * h * 3) / mse).log10();
        println!(
            "{}: {sw}x{sh} -> {n}^2, round trip {db:.1} dB, {split} split tiles exact",
            p.file_name().unwrap().to_string_lossy()
        );
    }
}

#[cfg(feature = "ntc")]
#[test]
#[ignore = "needs ORR_NTC_SET, LibNTC and a CUDA device"]
fn neural_sets_meet_the_floor() {
    use orr_cook::{GltfSettings, NeuralSettings, TextureRules};
    use orr_pack::Class;
    use orr_pack::neural::NeuralInfo;

    let dir = PathBuf::from(std::env::var_os("ORR_NTC_SET").expect("ORR_NTC_SET"));
    let mut paths = Vec::new();
    files(&dir, "gltf", &mut paths);
    files(&dir, "glb", &mut paths);
    paths.sort();
    let floor = NeuralSettings::default().min_psnr;
    let mut sets = 0;
    for path in paths {
        let settings = GltfSettings {
            textures: TextureRules {
                neural: Some(NeuralSettings::default()),
                ..Default::default()
            },
            ..Default::default()
        };
        let manifest = Manifest {
            archives: vec![ArchiveSpec::new(
                Target::Memory,
                vec![AssetSpec {
                    name: "g".into(),
                    source: Source::Gltf {
                        path: path.clone(),
                        settings,
                    },
                }],
            )],
        };
        let mut cooked = orr_cook::cook(&manifest, &Default::default()).unwrap();
        let a = Archive::from_memory(cooked.memory.remove(0)).unwrap();
        for r in cooked.report.assets.iter().flat_map(|a| &a.entries) {
            let Some(n) = &r.neural else { continue };
            sets += 1;
            println!(
                "{} {}: {:.2} bpp, {} bytes against {} BC, {:.1} dB over {} runs",
                path.display(),
                r.name,
                n.bits_per_pixel,
                n.bytes,
                n.bc_bytes,
                n.psnr,
                n.runs
            );
            for t in &n.textures {
                println!("  {} {:?}", t.name, t.psnr);
            }
            assert!(n.bytes <= n.bc_bytes, "{}: more bytes than BC", r.name);
            for t in &n.textures {
                assert!(
                    t.psnr.iter().all(|&p| p >= floor),
                    "{}: {} under {floor} dB: {:?}",
                    r.name,
                    t.name,
                    t.psnr
                );
            }
            let e = a.find(AssetId::from_name(&r.name)).unwrap();
            assert_eq!(e.class, Class::NeuralTextureSet as u16);
            let info = NeuralInfo::parse(&read_blob(&a, e, 0)).unwrap();
            assert_eq!(info.textures.len(), n.textures.len());
            let latents = a.find(info.header.latents).unwrap();
            let t = TextureInfo::parse(&read_blob(&a, latents, 0)).unwrap();
            assert_eq!(t.header.vk_format, vk::A4R4G4B4_UNORM_PACK16);
            assert_eq!(
                u32::from(t.header.layers) * 4,
                u32::from(info.header.latent_features)
            );
        }
    }
    assert!(sets > 0, "no neural sets cooked");
}
