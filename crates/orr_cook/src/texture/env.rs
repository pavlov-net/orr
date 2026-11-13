//! Environment maps: equirectangular or cube HDR input resampled to an equal-area octahedral
//! map, bright regions extracted into directional lights, box mips, BC6H.
//!
//! Map space is the input's: for equirectangular input, row 0 is +Y and column `u` looks
//! along `(sin theta sin phi, cos theta, -sin theta cos phi)` with `phi = 2 pi u - pi`, so
//! the image center looks along -Z and `u = 0.75` along +X. Cube input uses the Vulkan
//! face orientation.

use std::collections::HashSet;
use std::f32::consts::{PI, TAU};

use crate::par::prelude::*;
use ctt::Quality;
use orr_pack::texture::{ExtractedLight, octahedral};

use super::load::Source;
use super::mips::{MipSpec, settled};
use super::{Encoding, Img, MipFilter, Plan, Primaries, TextureSettings, Wrap};

const SUB: u32 = 4;

/// Bytes a source texel takes while it decodes: the decoder's image (at most an f32 RGBA), that
/// image's f32 RGBA conversion and the source's f32 RGBA made from it.
const DECODE_BYTES_PER_TEXEL: u64 = 16 + 16 + 16;

/// Bytes a source takes decoded per byte of a file whose header `image` does not read (KTX2 or
/// DDS), as other textures estimate it.
const DECODE_BYTES_PER_FILE_BYTE: u64 = 64;

/// Bytes a texel of the map's mip chain takes while it encodes: the chain's f32 RGBA, the
/// encoder's f32 RGBA copy of it, and inside ctt an f32 RGBA load of that beside its f16 RGBA
/// conversion, which outweighs the f16 copy and BC6H byte that follow.
const CHAIN_BYTES_PER_TEXEL: u64 = 16 + 16 + 16 + 8;

/// Texels a side of the octahedral map of an environment under `max_size`.
fn map_side(max_size: Option<u32>) -> u32 {
    let n = max_size.unwrap_or(1024).max(4);
    n.isolate_highest_one()
}

/// Estimated peak bytes of cooking an environment source of `file` bytes and `texels` texels,
/// if its header gives them, under `max_size`: the file, and the larger of the source's decode
/// and the map's encode. The resample between them, the source beside the map, stays under the
/// larger.
pub(crate) fn working_set(file: u64, texels: Option<u64>, max_size: Option<u32>) -> u64 {
    let n = u64::from(map_side(max_size));
    // A full chain of an n by n square, n a power of two.
    let chain = (4 * n * n - 1) / 3;
    let decode = texels.map_or(file * DECODE_BYTES_PER_FILE_BYTE, |t| {
        t * DECODE_BYTES_PER_TEXEL
    });
    file + decode.max(chain * CHAIN_BYTES_PER_TEXEL)
}

pub(super) fn cook(
    mut src: Source,
    s: &TextureSettings,
    primaries: Primaries,
    light_threshold: Option<f32>,
    max_lights: u8,
    quality: Quality,
) -> Result<super::Built, String> {
    // Finite values past the format's range stay, for the lights to take. Others would spread
    // through the resampling and the lights' energy.
    let (lo, hi) = Encoding::Bc6h.range();
    for v in src.layers.iter_mut().flat_map(|l| &mut l.px).flatten() {
        if !v.is_finite() {
            *v = settled(*v, lo, hi);
        }
    }
    let n = map_side(s.max_size);
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
        Some(t) => {
            let weights = primaries.rgb().luminance_weights().as_vec3().to_array();
            extract_lights(&mut map, &input, t, max_lights, weights)
        }
        None => Vec::new(),
    };
    drop(src);
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
            height: false,
        },
        full_chain: true,
        quality,
        max_size: None,
        toksvig: None,
        lights,
    };
    super::build(vec![map], &plan)
}

enum Input<'a> {
    Equirect(&'a Img),
    /// Six faces, +X, -X, +Y, -Y, +Z, -Z.
    Cube(&'a [Img]),
}

impl Input<'_> {
    /// Bilinear radiance along unit direction `d`.
    fn sample(&self, d: [f32; 3]) -> [f32; 4] {
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

/// Map point of sample `(i, j)` of the `SUB` x `SUB` that texel `(x, y)` of an `n` x `n` map
/// averages.
fn sample_point(n: u32, x: u32, y: u32, i: u32, j: u32) -> [f32; 2] {
    [
        (x as f32 + (i as f32 + 0.5) / SUB as f32) / n as f32,
        (y as f32 + (j as f32 + 0.5) / SUB as f32) / n as f32,
    ]
}

/// `n` x `n` equal-area octahedral map; each texel averages 4x4 stratified samples.
fn octahedral_map(input: &Input, n: u32) -> Img {
    let mut map = Img::new(n, n);
    map.px
        .par_chunks_mut(n as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, p) in row.iter_mut().enumerate() {
                let mut acc = [0.0f32; 4];
                for j in 0..SUB {
                    for i in 0..SUB {
                        let st = sample_point(n, x as u32, y as u32, i, j);
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

/// Texel neighbors on the sphere: edges of the octahedral square fold onto themselves.
fn neighbors(n: u32, x: u32, y: u32) -> [(u32, u32); 4] {
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
    input: &Input,
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
            for (nx, ny) in neighbors(n, t % n, t / n) {
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
                    for (nx, ny) in neighbors(n, t % n, t / n) {
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
            // The removed luminance and its sum of directions, at the samples the map averaged:
            // a texel's center would widen the spread by the texel's own footprint.
            let mut weight = 0.0f64;
            let mut centroid = [0.0f64; 3];
            for &t in texels {
                let p = &mut map.px[t as usize];
                let excess: [f32; 3] = std::array::from_fn(|c| (p[c] - fill[c]).max(0.0));
                for c in 0..3 {
                    p[c] -= excess[c];
                    e[c] += f64::from(excess[c] * texel_sr);
                }
                for j in 0..SUB {
                    for i in 0..SUB {
                        let st = sample_point(n, t % n, t / n, i, j);
                        let v = input.sample(octahedral::to_dir(st));
                        let above: [f32; 4] = std::array::from_fn(|c| {
                            if c < 3 {
                                (v[c] - fill[c]).max(0.0)
                            } else {
                                0.0
                            }
                        });
                        let w = f64::from(y_of(&above));
                        let d = octahedral::to_dir_f64(st.map(f64::from));
                        weight += w;
                        for c in 0..3 {
                            centroid[c] += w * d[c];
                        }
                    }
                }
            }
            let len = centroid
                .iter()
                .map(|c| c * c)
                .sum::<f64>()
                .sqrt()
                .max(1e-30);
            // Over a uniform cap of radius r about its axis the mean of `cos` is
            // `(1 + cos r) / 2`, so the mean direction's length gives r for a disc of any size,
            // and the disc of equal second moment for a blurred or darkened one. Rounding takes
            // a region whose energy lies along one sample's direction just past 1.
            let cos = (2.0 * len / weight - 1.0).min(1.0);
            ExtractedLight {
                direction_to_light: centroid.map(|c| (c / len) as f32),
                illuminance: e.map(|v| v as f32),
                angular_radius: cos.acos() as f32,
                _pad: 0.0,
            }
        })
        .collect()
}
