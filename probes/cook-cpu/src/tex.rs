//! P7 and the texture half of P5.
//!
//! `tex <dir> <quality> <N> [limit]`: every PNG under `dir`, mip 0, encoded by ctt 0.6.1 with
//! named encoders (bc7e_ for BC7, intel_ for BC4/BC5) three ways: one thread, N threads inside
//! each image (ctt's rayon block rows), and N images at once on the same pool. Output bytes
//! are hashed per image and compared. The BC data is then cut into 64 KiB tiles in the Vulkan
//! standard sparse block shape and zstd-compressed per tile and as one blob.
//!
//! `bc6h <dir> [quality]`: every .hdr under `dir` to BC6H UFLOAT with intel_ and amd_, decoded
//! with bcdec, error against the f32 source.

use ctt::encoders::Encoder;
use ctt::{
    AlphaMode, ColorSpace, Container, ConvertSettings, Format, FormatDesc, ImageRef, PipelineOutput, Quality, SurfaceRef,
    TargetFormat, TextureKind,
};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Copy)]
enum Kind {
    Bc7Srgb,
    Bc7,
    Bc5,
    Bc4,
}

fn kind(name: &str) -> Kind {
    if name.contains("normal_depth") {
        Kind::Bc7
    } else if name.contains("normal") {
        Kind::Bc5
    } else if name.contains("basecolor") || name.contains("albedo") {
        Kind::Bc7Srgb
    } else if name.contains("occlusion") || name.contains("visibility") || name.contains("mask") || name.contains("weight") {
        Kind::Bc4
    } else {
        Kind::Bc7
    }
}

fn pngs(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            pngs(&p, out);
        } else if p.extension().is_some_and(|x| x == "png") {
            out.push(p);
        }
    }
}

fn quality(s: &str) -> Quality {
    match s {
        "ultrafast" => Quality::UltraFast,
        "veryfast" => Quality::VeryFast,
        "fast" => Quality::Fast,
        "basic" => Quality::Basic,
        "slow" => Quality::Slow,
        "veryslow" => Quality::VerySlow,
        _ => panic!("quality {s}"),
    }
}

/// Encodes mip 0; returns the BC blocks (row-major) and the block size in bytes.
fn encode(rgba: &[u8], w: u32, h: u32, k: Kind, q: Quality) -> (Vec<u8>, usize) {
    let srgb = matches!(k, Kind::Bc7Srgb);
    let surface = SurfaceRef { data: rgba, width: w, height: h, depth: 1, stride: w * 4, slice_stride: 0 };
    let image = ImageRef {
        surfaces: vec![vec![surface]],
        kind: TextureKind::Texture2D,
        desc: FormatDesc {
            format: if srgb { Format::R8G8B8A8_SRGB } else { Format::R8G8B8A8_UNORM },
            color_space: if srgb { ColorSpace::Srgb } else { ColorSpace::Linear },
            alpha: AlphaMode::Straight,
        },
    };
    let (format, encoder, bs) = match k {
        Kind::Bc7Srgb => (Format::BC7_SRGB_BLOCK, Encoder::Bc7enc(Default::default()), 16),
        Kind::Bc7 => (Format::BC7_UNORM_BLOCK, Encoder::Bc7enc(Default::default()), 16),
        Kind::Bc5 => (Format::BC5_UNORM_BLOCK, Encoder::Intel(Default::default()), 16),
        Kind::Bc4 => (Format::BC4_UNORM_BLOCK, Encoder::Intel(Default::default()), 8),
    };
    let out = ctt::convert(
        image,
        ConvertSettings {
            format: Some(TargetFormat::Compressed { format, encoder }),
            container: Container::Raw,
            quality: q,
            allow_discarding_alpha: true,
            mipmap: false,
            ..Default::default()
        },
    )
    .unwrap();
    let PipelineOutput::Raw(img) = out else { panic!("raw output") };
    (img.surfaces[0][0].data.clone(), bs)
}

/// 64 KiB tiles in the standard sparse block shape: 64x64 blocks for 16-byte formats, 128x64
/// for 8-byte formats; edge tiles zero-padded; tiles row-major.
fn tiles(bc: &[u8], w: u32, h: u32, bs: usize) -> Vec<Vec<u8>> {
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let (tw, th) = if bs == 16 { (64, 64) } else { (128, 64) };
    let mut out = Vec::new();
    for ty in 0..bh.div_ceil(th) {
        for tx in 0..bw.div_ceil(tw) {
            let mut t = vec![0u8; 65536];
            for y in 0..th {
                let sy = ty * th + y;
                if sy >= bh {
                    break;
                }
                let x0 = tx * tw;
                let n = tw.min(bw - x0);
                let src = (sy * bw + x0) * bs;
                t[y * tw * bs..y * tw * bs + n * bs].copy_from_slice(&bc[src..src + n * bs]);
            }
            out.push(t);
        }
    }
    out
}

pub fn run(a: &[String]) {
    let mut files = Vec::new();
    pngs(Path::new(&a[0]), &mut files);
    files.sort();
    let q = quality(&a[1]);
    let n: usize = a[2].parse().unwrap();
    let limit: usize = a.get(3).map_or(usize::MAX, |s| s.parse().unwrap());
    files.truncate(limit);
    let t = Instant::now();
    let imgs: Vec<(PathBuf, Vec<u8>, u32, u32)> = files
        .par_iter()
        .map(|f| {
            let i = image::open(f).unwrap().to_rgba8();
            let (w, h) = i.dimensions();
            (f.clone(), i.into_raw(), w, h)
        })
        .collect();
    let texels: u64 = imgs.iter().map(|i| u64::from(i.2) * u64::from(i.3)).sum();
    println!("{} PNGs, {:.1} Mtexel, decoded in {:.1} s", imgs.len(), texels as f64 / 1e6, crate::secs(t));
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let many = rayon::ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    let enc = |i: &(PathBuf, Vec<u8>, u32, u32)| encode(&i.1, i.2, i.3, kind(&i.0.to_string_lossy()), q);
    let t = Instant::now();
    let r1: Vec<(Vec<u8>, usize)> = imgs.iter().map(|i| one.install(|| enc(i))).collect();
    let s1 = crate::secs(t);
    let t = Instant::now();
    let rn: Vec<(Vec<u8>, usize)> = imgs.iter().map(|i| many.install(|| enc(i))).collect();
    let sn = crate::secs(t);
    let t = Instant::now();
    let rx: Vec<(Vec<u8>, usize)> = many.install(|| imgs.par_iter().map(enc).collect());
    let sx = crate::secs(t);
    let (mut diff_n, mut diff_x) = (0, 0);
    for (k, (a1, (an, ax))) in r1.iter().zip(rn.iter().zip(&rx)).enumerate() {
        if a1.0 != an.0 {
            diff_n += 1;
            println!("  differs (N threads in image): {}", imgs[k].0.display());
        }
        if a1.0 != ax.0 {
            diff_x += 1;
            println!("  differs (N images at once): {}", imgs[k].0.display());
        }
    }
    let bytes: usize = r1.iter().map(|r| r.0.len()).sum();
    println!(
        "quality {}: jobs 1 {s1:.1} s ({:.2} Mtexel/s); {n} threads in image {sn:.1} s; {n} images at once {sx:.1} s; BC bytes {bytes}",
        a[1],
        texels as f64 / s1 / 1e6,
    );
    println!("images whose bytes differ from jobs 1: N threads in image {diff_n}, N images at once {diff_x}");
    // P5 textures: tile chunks vs one blob of the same tiles, and vs the linear mip.
    let res: Vec<[usize; 6]> = many.install(|| {
        imgs.par_iter()
            .zip(&r1)
            .map(|(i, (bc, bs))| {
                let t = tiles(bc, i.2, i.3, *bs);
                let blob: Vec<u8> = t.concat();
                let wl = zstd::bulk::compress(bc, 19).unwrap().len();
                let wt = zstd::bulk::compress(&blob, 19).unwrap().len();
                let pt: usize = t.iter().map(|x| zstd::bulk::compress(x, 19).unwrap().len()).sum();
                [wl, wt, pt, bc.len(), blob.len() - bc.len(), t.len()]
            })
            .collect()
    });
    let s = |k: usize| -> usize { res.iter().map(|r| r[k]).sum() };
    let loss = |c: usize, w: usize| (c as f64 / w as f64 - 1.0) * 100.0;
    println!(
        "mip 0 BC {} B in {} tiles (padding {} B, {:.2}%); zstd 19: linear blob {}, tiled blob {}, per 64 KiB tile {} ({:+.2}% vs tiled blob, {:+.2}% vs linear); zstd saves {:.1}% of BC bytes",
        s(3),
        s(5),
        s(4),
        s(4) as f64 / s(3) as f64 * 100.0,
        s(0),
        s(1),
        s(2),
        loss(s(2), s(1)),
        loss(s(2), s(0)),
        (1.0 - s(2) as f64 / (s(3) + s(4)) as f64) * 100.0
    );
}

fn bc6h(rgba: &[f32], w: u32, h: u32, encoder: Encoder, q: Quality) -> Vec<u8> {
    // SAFETY: f32 slice viewed as its bytes.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(rgba.as_ptr().cast(), rgba.len() * 4) };
    let surface = SurfaceRef { data: bytes, width: w, height: h, depth: 1, stride: w * 16, slice_stride: 0 };
    let image = ImageRef {
        surfaces: vec![vec![surface]],
        kind: TextureKind::Texture2D,
        desc: FormatDesc { format: Format::R32G32B32A32_SFLOAT, color_space: ColorSpace::Linear, alpha: AlphaMode::Opaque },
    };
    let out = ctt::convert(
        image,
        ConvertSettings {
            format: Some(TargetFormat::Compressed { format: Format::BC6H_UFLOAT_BLOCK, encoder }),
            container: Container::Raw,
            quality: q,
            allow_discarding_alpha: true,
            mipmap: false,
            ..Default::default()
        },
    )
    .unwrap();
    let PipelineOutput::Raw(img) = out else { panic!("raw output") };
    img.surfaces[0][0].data.clone()
}

/// (log2 RMSE, relative RMSE, max relative error on texels over 1% of the peak).
fn bc6h_error(bc: &[u8], src: &[f32], w: u32, h: u32) -> (f64, f64, f64) {
    let (bw, bh) = (w.div_ceil(4) as usize, h.div_ceil(4) as usize);
    let pitch = bw * 4 * 3;
    let mut dec = vec![0f32; pitch * bh * 4];
    for by in 0..bh {
        for bx in 0..bw {
            let blk = &bc[(by * bw + bx) * 16..(by * bw + bx) * 16 + 16];
            bcdec_rs::bc6h_float(blk, &mut dec[by * 4 * pitch + bx * 12..], pitch, false);
        }
    }
    let peak = src.chunks(4).map(|p| p[0].max(p[1]).max(p[2])).fold(0f32, f32::max);
    let (mut log2, mut rel, mut n, mut maxrel) = (0f64, 0f64, 0usize, 0f64);
    for y in 0..h as usize {
        for x in 0..w as usize {
            for c in 0..3 {
                let s = f64::from(src[(y * w as usize + x) * 4 + c].max(0.0));
                let d = f64::from(dec[y * pitch + x * 3 + c]);
                log2 += ((d + 1e-4).log2() - (s + 1e-4).log2()).powi(2);
                rel += ((d - s) / (s + 1e-3)).powi(2);
                if s > f64::from(peak) * 0.01 {
                    maxrel = maxrel.max((d - s).abs() / s);
                }
                n += 1;
            }
        }
    }
    ((log2 / n as f64).sqrt(), (rel / n as f64).sqrt(), maxrel)
}

pub fn run_bc6h(a: &[String]) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(&a[0])
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "hdr" || x == "exr"))
        .collect();
    files.sort();
    let qs = a.get(1).map_or("basic", String::as_str);
    let q = quality(qs);
    let mut wins = [0, 0];
    let (mut si, mut sa) = (0f64, 0f64);
    for f in &files {
        let i = image::open(f).unwrap().to_rgba32f();
        let (w, h) = i.dimensions();
        let src = i.into_raw();
        let peak = src.chunks(4).map(|p| p[0].max(p[1]).max(p[2])).fold(0f32, f32::max);
        let t = Instant::now();
        let bi = bc6h(&src, w, h, Encoder::Intel(Default::default()), q);
        let ti = crate::secs(t);
        let t = Instant::now();
        let ba = bc6h(&src, w, h, Encoder::Amd(Default::default()), q);
        let ta = crate::secs(t);
        let ei = bc6h_error(&bi, &src, w, h);
        let ea = bc6h_error(&ba, &src, w, h);
        wins[usize::from(ea.0 < ei.0)] += 1;
        si += ei.0;
        sa += ea.0;
        println!(
            "{} {w}x{h} peak {peak:.0}: intel log2-RMSE {:.4} relRMSE {:.4} maxrel {:.3} ({ti:.2} s); amd log2-RMSE {:.4} relRMSE {:.4} maxrel {:.3} ({ta:.2} s)",
            f.file_name().unwrap().to_string_lossy(),
            ei.0,
            ei.1,
            ei.2,
            ea.0,
            ea.1,
            ea.2
        );
    }
    println!(
        "quality {qs}: lower log2-RMSE: intel {} maps, amd {} maps; mean log2-RMSE intel {:.4}, amd {:.4}",
        wins[0],
        wins[1],
        si / files.len() as f64,
        sa / files.len() as f64
    );
}
