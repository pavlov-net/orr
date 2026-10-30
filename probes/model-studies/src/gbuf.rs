//! G-buffer packing of coat, fuzz and anisotropy fields.
//!
//! Renders each test material on a sphere under a studio environment with the
//! openpbr-bsdf reference, once with f32 fields and once per encoding with the fields
//! quantised as the G-buffer would store them. tools/flip_batch.py scores the pairs.

use crate::openpbr::{Mat, Prepared};
use crate::util::*;
use rayon::prelude::*;

const W: usize = 192;
const SPP: u32 = 2048;

/// Per-pixel lobe fields that the extended target has to carry.
#[derive(Clone, Copy)]
struct Fields {
    coat_w: f32,
    coat_rough: f32,
    coat_ior: f32,
    coat_col: V3,
    fuzz_w: f32,
    fuzz_rough: f32,
    fuzz_col: V3,
    aniso: f32,
    angle: f32, // radians, [0, pi)
}

struct Material {
    name: &'static str,
    base: fn(f32, f32) -> Mat,
    fields: fn(f32, f32) -> Fields,
}

fn none() -> Fields {
    Fields {
        coat_w: 0.0,
        coat_rough: 0.0,
        coat_ior: 1.6,
        coat_col: [1.0; 3],
        fuzz_w: 0.0,
        fuzz_rough: 0.5,
        fuzz_col: [1.0; 3],
        aniso: 0.0,
        angle: 0.0,
    }
}

fn lerp3(a: V3, b: V3, t: f32) -> V3 {
    add(mul(a, 1.0 - t), mul(b, t))
}

fn hue(h: f32, s: f32, v: f32) -> V3 {
    let h = h.rem_euclid(1.0) * 6.0;
    let f = |k: f32| {
        let k = (k + h).rem_euclid(6.0);
        v - v * s * (k.min(4.0 - k).min(1.0)).max(0.0)
    };
    [f(5.0), f(3.0), f(1.0)]
}

// x, y in [-1, 1] across the sphere image.
fn f1(x: f32, _y: f32) -> f32 {
    0.5 + 0.5 * x
}
fn f2(_x: f32, y: f32) -> f32 {
    0.5 + 0.5 * y
}
fn f3(x: f32, y: f32) -> f32 {
    0.5 + 0.5 * (7.0 * x + 4.0 * y).sin()
}
fn swirl(x: f32, y: f32) -> f32 {
    (y.atan2(x) + std::f32::consts::FRAC_PI_2).rem_euclid(std::f32::consts::PI)
}

fn materials() -> Vec<Material> {
    vec![
        Material {
            name: "car_paint",
            base: |_, _| Mat { base_color: [0.6, 0.05, 0.04], metalness: 0.5, spec_rough: 0.35, ..Mat::default() },
            fields: |x, y| Fields {
                coat_w: 1.0,
                coat_rough: 0.03 + 0.1 * f2(x, y),
                coat_ior: 1.5,
                coat_col: lerp3([1.0; 3], [0.95, 0.88, 0.75], f1(x, y)),
                ..none()
            },
        },
        Material {
            name: "clear_wood",
            base: |x, y| Mat { base_color: lerp3([0.35, 0.18, 0.08], [0.2, 0.09, 0.03], f3(x, y)), spec_rough: 0.6, ..Mat::default() },
            fields: |x, y| Fields { coat_w: 0.4 + 0.6 * f3(x, y), coat_rough: 0.05 + 0.25 * f1(x, y), coat_ior: 1.5, ..none() },
        },
        Material {
            name: "tinted_lacquer",
            base: |_, _| Mat { base_color: [0.7; 3], spec_rough: 0.4, ..Mat::default() },
            fields: |x, y| Fields {
                coat_w: 1.0,
                coat_rough: 0.03 + 0.1 * f2(x, y),
                coat_ior: 1.3 + 0.9 * f1(x, y),
                coat_col: hue(0.6 * f2(x, y), 0.8, 0.9),
                ..none()
            },
        },
        Material {
            name: "velvet",
            base: |_, _| Mat { base_color: [0.3, 0.02, 0.05], spec_rough: 0.7, ..Mat::default() },
            fields: |x, y| Fields {
                fuzz_w: 1.0,
                fuzz_rough: 0.2 + 0.7 * f2(x, y),
                fuzz_col: mul([0.9, 0.2, 0.3], 0.5 + 0.5 * f1(x, y)),
                ..none()
            },
        },
        Material {
            name: "sheen_cloth",
            base: |_, _| Mat { base_color: [0.05, 0.08, 0.3], spec_rough: 0.8, ..Mat::default() },
            fields: |x, y| Fields { fuzz_w: 0.3 + 0.7 * f3(x, y), fuzz_rough: 0.5, fuzz_col: [0.8, 0.85, 1.0], ..none() },
        },
        Material {
            name: "brushed_metal",
            base: |_, _| Mat { base_color: [0.9, 0.9, 0.92], metalness: 1.0, spec_rough: 0.3, ..Mat::default() },
            fields: |x, y| Fields { aniso: 0.3 + 0.6 * f1(x, y), angle: swirl(x, y), ..none() },
        },
        Material {
            name: "brushed_coated",
            base: |_, _| Mat { base_color: [0.8, 0.6, 0.3], metalness: 1.0, spec_rough: 0.35, ..Mat::default() },
            fields: |x, y| Fields {
                aniso: 0.8,
                angle: swirl(x, y),
                coat_w: 1.0,
                coat_rough: 0.03 + 0.05 * f2(x, y),
                coat_ior: 1.5,
                ..none()
            },
        },
        Material {
            name: "all_three",
            base: |_, _| Mat { base_color: [0.5, 0.3, 0.2], spec_rough: 0.4, ..Mat::default() },
            fields: |x, y| Fields {
                aniso: 0.6 * f1(x, y),
                angle: swirl(x, y),
                coat_w: f2(x, y),
                coat_rough: 0.1,
                coat_ior: 1.6,
                coat_col: hue(0.1 + 0.3 * f1(x, y), 0.5, 1.0),
                fuzz_w: f3(x, y),
                fuzz_rough: 0.4,
                fuzz_col: hue(0.55 + 0.2 * f2(x, y), 0.6, 0.9),
            },
        },
        Material {
            name: "two_colours",
            base: |_, _| Mat { base_color: [0.4; 3], spec_rough: 0.5, ..Mat::default() },
            fields: |_, _| Fields {
                coat_w: 1.0,
                coat_rough: 0.05,
                coat_ior: 1.5,
                coat_col: [0.2, 0.6, 1.0],
                fuzz_w: 0.7,
                fuzz_rough: 0.5,
                fuzz_col: [1.0, 0.5, 0.1],
                ..none()
            },
        },
        Material {
            name: "coat_ior_sweep",
            base: |_, _| Mat { base_color: [0.02; 3], spec_rough: 0.5, ..Mat::default() },
            fields: |x, y| Fields { coat_w: 1.0, coat_rough: 0.03, coat_ior: 1.2 + 1.2 * f1(x, y), ..none() },
        },
    ]
}

fn q(x: f32, bits: u32) -> f32 {
    let m = ((1u32 << bits) - 1) as f32;
    (x.clamp(0.0, 1.0) * m).round() / m
}
fn srgb_enc(x: f32) -> f32 {
    if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}
fn srgb_dec(x: f32) -> f32 {
    if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}
fn q_col(c: V3, bits: [u32; 3]) -> V3 {
    [0, 1, 2].map(|i| srgb_dec(q(srgb_enc(c[i]), bits[i])))
}
/// YCoCg on sRGB-encoded values: luma `ly` bits, chroma `lc` bits each.
fn q_ycocg(c: V3, ly: u32, lc: u32) -> V3 {
    let e = c.map(srgb_enc);
    let y = 0.25 * e[0] + 0.5 * e[1] + 0.25 * e[2];
    let co = 0.5 * e[0] - 0.5 * e[2];
    let cg = -0.25 * e[0] + 0.5 * e[1] - 0.25 * e[2];
    let y = q(y, ly);
    // Signed chroma on a grid with an exact zero.
    let m = ((1u32 << (lc - 1)) - 1) as f32;
    let co = (co * 2.0 * m).round() / (2.0 * m);
    let cg = (cg * 2.0 * m).round() / (2.0 * m);
    let r = y + co - cg;
    let g = y + cg;
    let b = y - co - cg;
    [r, g, b].map(|v| srgb_dec(v.clamp(0.0, 1.0)))
}
/// IOR through s = (n - 1) / (n + 1) on [0, 0.5] (n up to 3).
fn q_ior(n: f32, bits: u32) -> f32 {
    let s = q((n - 1.0) / (n + 1.0) / 0.5, bits) * 0.5;
    (1.0 + s) / (1.0 - s)
}
fn q_angle(a: f32, bits: u32) -> f32 {
    let steps = (1u32 << bits) as f32;
    ((a / std::f32::consts::PI * steps).round() % steps) / steps * std::f32::consts::PI
}

#[derive(Clone, Copy, PartialEq)]
enum Enc {
    F32,
    /// Every scalar 8 bits, colours RGB8 sRGB, IOR 8, angle 8: 104 bits.
    U8All,
    /// lighting.md L1: coat weight 8, coat roughness 8, angle 8 per pixel; the rest from
    /// the material (here: the field's mean over the sphere).
    PlanL1,
    /// Every scalar 8 bits including coat IOR (56 bits); colours from the material.
    Scalars8,
    /// RGB565 colours: 5+5+3+16 coat, 4+4+16 fuzz, 4+7 anisotropy = 64.
    Rgb565,
    /// RGB444 colours: 6+6+4+12 coat, 6+5+12 fuzz, 6+7 anisotropy = 64.
    Rgb444,
    /// YCoCg 6:4:4 colours: 5+6+3+14 coat, 5+5+14 fuzz, 5+7 anisotropy = 64.
    YCoCg644,
    /// One shared RGB777 colour (fuzz colour where fuzz is present): 21 + 6+7+4 + 6+6 + 6+8 = 64.
    Shared777,
}

impl Enc {
    fn name(self) -> &'static str {
        match self {
            Enc::F32 => "f32",
            Enc::U8All => "u8_all_104b",
            Enc::PlanL1 => "plan_l1_24b",
            Enc::Scalars8 => "scalars8_56b",
            Enc::Rgb565 => "rgb565_64b",
            Enc::Rgb444 => "rgb444_64b",
            Enc::YCoCg644 => "ycocg644_64b",
            Enc::Shared777 => "shared777_64b",
        }
    }
    fn apply(self, f: Fields, mean: &Fields) -> Fields {
        match self {
            Enc::F32 => f,
            Enc::U8All => Fields {
                coat_w: q(f.coat_w, 8),
                coat_rough: q(f.coat_rough, 8),
                coat_ior: q_ior(f.coat_ior, 8),
                coat_col: q_col(f.coat_col, [8; 3]),
                fuzz_w: q(f.fuzz_w, 8),
                fuzz_rough: q(f.fuzz_rough, 8),
                fuzz_col: q_col(f.fuzz_col, [8; 3]),
                aniso: q(f.aniso, 8),
                angle: q_angle(f.angle, 8),
            },
            Enc::PlanL1 => Fields {
                coat_w: q(f.coat_w, 8),
                coat_rough: q(f.coat_rough, 8),
                angle: q_angle(f.angle, 8),
                ..*mean
            },
            Enc::Scalars8 => Fields {
                coat_w: q(f.coat_w, 8),
                coat_rough: q(f.coat_rough, 8),
                coat_ior: q_ior(f.coat_ior, 8),
                fuzz_w: q(f.fuzz_w, 8),
                fuzz_rough: q(f.fuzz_rough, 8),
                aniso: q(f.aniso, 8),
                angle: q_angle(f.angle, 8),
                coat_col: mean.coat_col,
                fuzz_col: mean.fuzz_col,
            },
            Enc::Rgb565 => Fields {
                coat_w: q(f.coat_w, 5),
                coat_rough: q(f.coat_rough, 5),
                coat_ior: q_ior(f.coat_ior, 3),
                coat_col: q_col(f.coat_col, [5, 6, 5]),
                fuzz_w: q(f.fuzz_w, 4),
                fuzz_rough: q(f.fuzz_rough, 4),
                fuzz_col: q_col(f.fuzz_col, [5, 6, 5]),
                aniso: q(f.aniso, 4),
                angle: q_angle(f.angle, 7),
            },
            Enc::Rgb444 => Fields {
                coat_w: q(f.coat_w, 6),
                coat_rough: q(f.coat_rough, 6),
                coat_ior: q_ior(f.coat_ior, 4),
                coat_col: q_col(f.coat_col, [4; 3]),
                fuzz_w: q(f.fuzz_w, 6),
                fuzz_rough: q(f.fuzz_rough, 5),
                fuzz_col: q_col(f.fuzz_col, [4; 3]),
                aniso: q(f.aniso, 6),
                angle: q_angle(f.angle, 7),
            },
            Enc::YCoCg644 => Fields {
                coat_w: q(f.coat_w, 5),
                coat_rough: q(f.coat_rough, 6),
                coat_ior: q_ior(f.coat_ior, 3),
                coat_col: q_ycocg(f.coat_col, 6, 4),
                fuzz_w: q(f.fuzz_w, 5),
                fuzz_rough: q(f.fuzz_rough, 5),
                fuzz_col: q_ycocg(f.fuzz_col, 6, 4),
                aniso: q(f.aniso, 5),
                angle: q_angle(f.angle, 7),
            },
            Enc::Shared777 => {
                let c = q_col(if f.fuzz_w > 0.0 { f.fuzz_col } else { f.coat_col }, [7; 3]);
                Fields {
                    coat_w: q(f.coat_w, 6),
                    coat_rough: q(f.coat_rough, 7),
                    coat_ior: q_ior(f.coat_ior, 4),
                    coat_col: c,
                    fuzz_w: q(f.fuzz_w, 6),
                    fuzz_rough: q(f.fuzz_rough, 6),
                    fuzz_col: c,
                    aniso: q(f.aniso, 6),
                    angle: q_angle(f.angle, 8),
                }
            }
        }
    }
}

fn env(d: V3) -> V3 {
    let mut l = if d[1] > 0.0 { lerp3([0.25, 0.28, 0.32], [0.45, 0.55, 0.8], d[1]) } else { [0.08, 0.07, 0.06] };
    let boxes: [(V3, f32, f32, V3); 3] = [
        (norm([-0.5, 0.6, 0.6]), 0.30, 0.18, [20.0, 19.0, 18.0]),
        (norm([0.85, 0.1, 0.4]), 0.06, 0.45, [10.0, 10.0, 11.0]),
        (norm([0.1, 0.35, -1.0]), 0.25, 0.25, [15.0, 14.0, 12.0]),
    ];
    for (c, hw, hh, e) in boxes {
        let (t, b) = basis(c);
        if dot(d, c) > 0.0 {
            let u = dot(d, t) / dot(d, c);
            let v = dot(d, b) / dot(d, c);
            if u.abs() < hw && v.abs() < hh {
                l = e;
            }
        }
    }
    l
}

fn mean_fields(m: &Material) -> Fields {
    let mut acc = none();
    let (mut s, mut n) = ([0.0f64; 13], 0.0f64);
    for j in 0..W {
        for i in 0..W {
            let x = (i as f32 + 0.5) / W as f32 * 2.0 - 1.0;
            let y = 1.0 - (j as f32 + 0.5) / W as f32 * 2.0;
            if x * x + y * y >= 1.0 {
                continue;
            }
            let f = (m.fields)(x, y);
            let v = [
                f.coat_w, f.coat_rough, f.coat_ior, f.coat_col[0], f.coat_col[1], f.coat_col[2], f.fuzz_w, f.fuzz_rough,
                f.fuzz_col[0], f.fuzz_col[1], f.fuzz_col[2], f.aniso, f.angle,
            ];
            for k in 0..13 {
                s[k] += v[k] as f64;
            }
            n += 1.0;
        }
    }
    let a = s.map(|v| (v / n) as f32);
    acc.coat_w = a[0];
    acc.coat_rough = a[1];
    acc.coat_ior = a[2];
    acc.coat_col = [a[3], a[4], a[5]];
    acc.fuzz_w = a[6];
    acc.fuzz_rough = a[7];
    acc.fuzz_col = [a[8], a[9], a[10]];
    acc.aniso = a[11];
    acc.angle = a[12];
    acc
}

/// Renders the reference and every encoding with shared directions: each sample takes
/// its direction from the reference BSDF and weights every variant by eval / reference
/// pdf, so the images differ only by the BSDF values, not by sampling noise. All
/// roughnesses are at least 0.03, so no lobe is a delta.
fn render(m: &Material, encs: &[Enc], seed: u64) -> Vec<Vec<V3>> {
    let mean = mean_fields(m);
    let pixels: Vec<Vec<V3>> = (0..W * W)
        .into_par_iter()
        .map(|p| {
            let (i, j) = (p % W, p / W);
            let x = (i as f32 + 0.5) / W as f32 * 2.0 - 1.0;
            let y = 1.0 - (j as f32 + 0.5) / W as f32 * 2.0;
            let r2 = x * x + y * y;
            if r2 >= 1.0 {
                return vec![env([0.0, 0.0, -1.0]); encs.len()];
            }
            let n = [x, y, (1.0 - r2).sqrt()];
            let t = norm(cross([0.0, 1.0, 0.0], n));
            let b = cross(n, t);
            let preps: Vec<Prepared> = encs
                .iter()
                .map(|e| {
                    let f = e.apply((m.fields)(x, y), &mean);
                    let mut mat = (m.base)(x, y);
                    mat.t = t;
                    mat.b = b;
                    mat.n = n;
                    mat.coat_weight = f.coat_w;
                    mat.coat_rough = f.coat_rough;
                    mat.coat_ior = f.coat_ior;
                    mat.coat_color = f.coat_col;
                    mat.fuzz_weight = f.fuzz_w;
                    mat.fuzz_rough = f.fuzz_rough;
                    mat.fuzz_color = f.fuzz_col;
                    mat.spec_aniso = f.aniso;
                    mat.spec_rot_cs = [f.angle.cos(), f.angle.sin()];
                    Prepared::new(&mat, [0.0, 0.0, 1.0])
                })
                .collect();
            let h = hash(seed ^ (p as u64).wrapping_mul(0x9e3779b97f4a7c15));
            let shift = [(h & 0xffff) as f32 / 65536.0, ((h >> 16) & 0xffff) as f32 / 65536.0, ((h >> 32) & 0xffff) as f32 / 65536.0];
            let mut acc = vec![[0.0f32; 3]; encs.len()];
            for s in 0..SPP {
                let Some(smp) = preps[0].sample(r3(s, shift)) else { continue };
                let pdf = preps[0].pdf(smp.wi);
                if !(pdf > 0.0) || !pdf.is_finite() {
                    continue;
                }
                let l = env(smp.wi);
                for (k, pr) in preps.iter().enumerate() {
                    let f = pr.eval(smp.wi);
                    if f.iter().all(|v| v.is_finite()) {
                        acc[k] = add(acc[k], mul(hadamard(f, l), 1.0 / pdf));
                    }
                }
            }
            acc.into_iter().map(|a| mul(a, 1.0 / SPP as f32)).collect()
        })
        .collect();
    (0..encs.len()).map(|k| pixels.iter().map(|px| px[k]).collect()).collect()
}

pub fn run() {
    let dir = util_dir();
    let mut manifest = String::new();
    let encs = [Enc::F32, Enc::U8All, Enc::PlanL1, Enc::Scalars8, Enc::Rgb565, Enc::Rgb444, Enc::YCoCg644, Enc::Shared777];
    for m in materials() {
        let t0 = std::time::Instant::now();
        let imgs = render(&m, &encs, 1);
        let ref_path = dir.join(format!("{}_f32.bin", m.name));
        write_image(&ref_path, W, W, &imgs[0]);
        let floor = render(&m, &[Enc::F32], 2);
        let floor_path = dir.join(format!("{}_f32_seed2.bin", m.name));
        write_image(&floor_path, W, W, &floor[0]);
        manifest += &format!("{} {} {} noise_floor
", m.name, ref_path.display(), floor_path.display());
        for (k, e) in encs.iter().enumerate().skip(1) {
            let p = dir.join(format!("{}_{}.bin", m.name, e.name()));
            write_image(&p, W, W, &imgs[k]);
            manifest += &format!("{} {} {} {}
", m.name, ref_path.display(), p.display(), e.name());
        }
        eprintln!("{}: {:.1} s", m.name, t0.elapsed().as_secs_f32());
    }
    std::fs::write(dir.join("manifest.txt"), manifest).unwrap();
    println!("wrote {}", dir.join("manifest.txt").display());
}

fn util_dir() -> std::path::PathBuf {
    out_dir("gbuf")
}
