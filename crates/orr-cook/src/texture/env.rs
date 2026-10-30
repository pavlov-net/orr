//! Environment maps: equirectangular or cube HDR input resampled to an equal-area octahedral
//! map, bright regions extracted into directional lights, box mips, BC6H.
//!
//! Map space is the input's: for equirectangular input, row 0 is +Y and column `u` looks
//! along `(sin theta sin phi, cos theta, -sin theta cos phi)` with `phi = 2 pi u - pi`, so
//! the image centre looks along -Z and `u = 0.75` along +X. Cube input uses the Vulkan
//! face orientation.

use std::collections::HashSet;
use std::f32::consts::{PI, TAU};

use orr_pack::texture::{ExtractedLight, octahedral};
use rayon::prelude::*;

use super::load::Source;
use super::mips::MipSpec;
use super::{Encoding, Img, MipFilter, Plan, Primaries, TextureSettings, Wrap, srgb_to_linear};

const SUB: u32 = 4;

pub(super) fn cook(
    mut src: Source,
    s: &TextureSettings,
    primaries: Primaries,
    light_threshold: Option<f32>,
    max_lights: u8,
) -> Result<super::Built, String> {
    if src.srgb {
        for l in &mut src.layers {
            for p in &mut l.px {
                for c in &mut p[..3] {
                    *c = srgb_to_linear(*c);
                }
            }
        }
    }
    let n = s.max_size.unwrap_or(1024).max(4);
    let n = 1 << (31 - n.leading_zeros());
    let input = if src.cube {
        Input::Cube(&src.layers)
    } else {
        let l = &src.layers[0];
        if l.w != 2 * l.h {
            return Err(format!(
                "environment input is {}x{}: not 2:1 equirectangular or a cube",
                l.w, l.h
            ));
        }
        Input::Equirect(l)
    };
    let mut map = octahedral_map(&input, n);
    let lights = match light_threshold {
        Some(t) => extract_lights(&mut map, t, max_lights, primaries.luminance()),
        None => Vec::new(),
    };
    let plan = Plan {
        encoding: Encoding::Bc6h,
        cube: false,
        primaries,
        mips: MipSpec {
            filter: MipFilter::Box,
            wrap: [Wrap::Clamp; 2],
            grid: None,
            normal: false,
            coverage: None,
        },
        quality: s.quality,
        max_size: None,
        toksvig: None,
        lights,
    };
    super::build(vec![map], &plan)
}

pub(crate) enum Input<'a> {
    Equirect(&'a Img),
    /// Six faces, +X, -X, +Y, -Y, +Z, -Z.
    Cube(&'a [Img]),
}

impl Input<'_> {
    /// Bilinear radiance along unit direction `d`.
    pub(crate) fn sample(&self, d: [f32; 3]) -> [f32; 4] {
        match self {
            Self::Equirect(img) => {
                let theta = d[1].clamp(-1.0, 1.0).acos();
                let phi = d[0].atan2(-d[2]);
                let u = (phi + PI) / TAU;
                let v = theta / PI;
                let x = u * img.w as f32 - 0.5;
                let y = v * img.h as f32 - 0.5;
                // Wrap horizontally, clamp vertically.
                let x0 = x.floor();
                let fx = x - x0;
                let w = i64::from(img.w);
                let xa = (x0 as i64).rem_euclid(w) as f32;
                let xb = (x0 as i64 + 1).rem_euclid(w) as f32;
                let a = img.bilinear(xa, y);
                let b = img.bilinear(xb, y);
                std::array::from_fn(|c| a[c] + (b[c] - a[c]) * fx)
            }
            Self::Cube(faces) => {
                let [x, y, z] = d;
                let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
                let (face, sc, tc, ma) = if ax >= ay && ax >= az {
                    if x > 0.0 {
                        (0, -z, -y, ax)
                    } else {
                        (1, z, -y, ax)
                    }
                } else if ay >= az {
                    if y > 0.0 {
                        (2, x, z, ay)
                    } else {
                        (3, x, -z, ay)
                    }
                } else if z > 0.0 {
                    (4, x, -y, az)
                } else {
                    (5, -x, -y, az)
                };
                let f = &faces[face];
                let s = 0.5 * (sc / ma + 1.0);
                let t = 0.5 * (tc / ma + 1.0);
                f.bilinear(s * f.w as f32 - 0.5, t * f.h as f32 - 0.5)
            }
        }
    }
}

/// `n` x `n` equal-area octahedral map; each texel averages 4x4 stratified samples.
pub(crate) fn octahedral_map(input: &Input, n: u32) -> Img {
    let mut map = Img::new(n, n);
    map.px
        .par_chunks_mut(n as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, p) in row.iter_mut().enumerate() {
                let mut acc = [0.0f32; 4];
                for j in 0..SUB {
                    for i in 0..SUB {
                        let st = [
                            (x as f32 + (i as f32 + 0.5) / SUB as f32) / n as f32,
                            (y as f32 + (j as f32 + 0.5) / SUB as f32) / n as f32,
                        ];
                        let v = input.sample(octahedral::to_dir(st));
                        for c in 0..4 {
                            acc[c] += v[c];
                        }
                    }
                }
                let k = 1.0 / (SUB * SUB) as f32;
                *p = [acc[0] * k, acc[1] * k, acc[2] * k, 1.0];
            }
        });
    map
}

/// Texel neighbours on the sphere: edges of the octahedral square fold onto themselves.
fn neighbours(n: u32, x: u32, y: u32) -> [(u32, u32); 4] {
    let flip = |v: u32| n - 1 - v;
    [
        if x == 0 { (0, flip(y)) } else { (x - 1, y) },
        if x == n - 1 {
            (n - 1, flip(y))
        } else {
            (x + 1, y)
        },
        if y == 0 { (flip(x), 0) } else { (x, y - 1) },
        if y == n - 1 {
            (flip(x), n - 1)
        } else {
            (x, y + 1)
        },
    ]
}

/// Moves the brightest connected regions over `threshold` into lights. Each region's texels
/// keep the mean of the background around it (never more than they held), and the light
/// takes the rest of their energy.
fn extract_lights(
    map: &mut Img,
    threshold: f32,
    max_lights: u8,
    luma: [f32; 3],
) -> Vec<ExtractedLight> {
    let n = map.w;
    let texel_sr = 4.0 * PI / (n as f32 * n as f32);
    let y_of = |p: &[f32; 4]| luma[0] * p[0] + luma[1] * p[1] + luma[2] * p[2];
    let bright: Vec<bool> = map.px.iter().map(|p| y_of(p) > threshold).collect();
    let mut region = vec![u32::MAX; map.px.len()];
    let mut regions: Vec<(f64, Vec<u32>)> = Vec::new();
    for start in 0..map.px.len() {
        if !bright[start] || region[start] != u32::MAX {
            continue;
        }
        let id = regions.len() as u32;
        let mut texels = vec![start as u32];
        region[start] = id;
        let mut i = 0;
        while i < texels.len() {
            let t = texels[i];
            for (nx, ny) in neighbours(n, t % n, t / n) {
                let k = (ny * n + nx) as usize;
                if bright[k] && region[k] == u32::MAX {
                    region[k] = id;
                    texels.push(k as u32);
                }
            }
            i += 1;
        }
        let energy = texels
            .iter()
            .map(|&t| f64::from(y_of(&map.px[t as usize])))
            .sum();
        regions.push((energy, texels));
    }
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by(|&a, &b| regions[b].0.total_cmp(&regions[a].0));
    order.truncate(usize::from(max_lights));
    order
        .into_iter()
        .map(|r| {
            let texels = &regions[r].1;
            // The background under the region: texels two steps outside it, past the
            // partly covered edge.
            let mut seen: HashSet<u32> = texels.iter().copied().collect();
            let step = |from: &[u32], seen: &mut HashSet<u32>| -> Vec<u32> {
                let mut out = Vec::new();
                for &t in from {
                    for (nx, ny) in neighbours(n, t % n, t / n) {
                        let k = ny * n + nx;
                        if seen.insert(k) {
                            out.push(k);
                        }
                    }
                }
                out
            };
            let ring1 = step(texels, &mut seen);
            let ring2 = step(&ring1, &mut seen);
            let mut fill = [0.0f64; 3];
            let mut ring = 0u32;
            for &k in if ring2.is_empty() { &ring1 } else { &ring2 } {
                if !bright[k as usize] {
                    for (f, v) in fill.iter_mut().zip(map.px[k as usize]) {
                        *f += f64::from(v);
                    }
                    ring += 1;
                }
            }
            let fill = fill.map(|f| (f / f64::from(ring.max(1))) as f32);
            let mut e = [0.0f64; 3];
            let mut centroid = [0.0f64; 3];
            for &t in texels {
                let p = &mut map.px[t as usize];
                let excess: [f32; 3] = std::array::from_fn(|c| (p[c] - fill[c]).max(0.0));
                for c in 0..3 {
                    p[c] -= excess[c];
                    e[c] += f64::from(excess[c] * texel_sr);
                }
                let st = [
                    ((t % n) as f32 + 0.5) / n as f32,
                    ((t / n) as f32 + 0.5) / n as f32,
                ];
                let d = octahedral::to_dir(st);
                let wgt = f64::from(y_of(&[excess[0], excess[1], excess[2], 0.0]));
                for c in 0..3 {
                    centroid[c] += wgt * f64::from(d[c]);
                }
            }
            let len = centroid
                .iter()
                .map(|c| c * c)
                .sum::<f64>()
                .sqrt()
                .max(1e-30);
            let solid = texels.len() as f32 * texel_sr;
            ExtractedLight {
                direction_to_light: centroid.map(|c| (c / len) as f32),
                illuminance: e.map(|v| v as f32),
                angular_radius: (1.0 - solid / TAU).clamp(-1.0, 1.0).acos(),
                _pad: 0.0,
            }
        })
        .collect()
}
