//! Texel formats of raster layers, the 2x2 reductions and value ranges.

use orr_pack::raster::RasterLayer;
use orr_pack::texture::vk;
use serde::Deserialize;

use crate::texture::{Img, linear_to_srgb, srgb_to_linear};

/// Texel format of a raster input. RGB8 is stored as RGBA8 with alpha 255.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RasterFormat {
    R32f,
    R16,
    R8,
    Rg8,
    Rgb8,
    Rgba8,
}

impl RasterFormat {
    /// Bytes per input texel.
    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::R16 | Self::Rg8 => 2,
            Self::Rgb8 => 3,
            Self::R32f | Self::Rgba8 => 4,
        }
    }

    pub(crate) fn stored(self) -> Texel {
        match self {
            Self::R32f => Texel::R32f,
            Self::R16 => Texel::R16,
            Self::R8 => Texel::R8,
            Self::Rg8 => Texel::Rg8,
            Self::Rgb8 | Self::Rgba8 => Texel::Rgba8,
        }
    }

    /// Input texels to stored texels, appended to `out`.
    pub(crate) fn expand(self, src: &[u8], out: &mut Vec<u8>) {
        if self == Self::Rgb8 {
            out.reserve(src.len() / 3 * 4);
            for p in src.chunks_exact(3) {
                out.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
        } else {
            out.extend_from_slice(src);
        }
    }

    /// Widens `range` to the finite channel values of input texels, UNORM normalised.
    pub(crate) fn range(self, src: &[u8], range: &mut [f32; 2]) {
        let mut add = |v: f32| {
            if v.is_finite() {
                range[0] = range[0].min(v);
                range[1] = range[1].max(v);
            }
        };
        match self {
            Self::R32f => src
                .chunks_exact(4)
                .for_each(|b| add(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
            Self::R16 => src
                .chunks_exact(2)
                .for_each(|b| add(f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0)),
            _ => {
                let (mut lo, mut hi) = (u8::MAX, u8::MIN);
                for &b in src {
                    lo = lo.min(b);
                    hi = hi.max(b);
                }
                if !src.is_empty() {
                    add(f32::from(lo) / 255.0);
                    add(f32::from(hi) / 255.0);
                }
            }
        }
    }
}

/// Stored texel format.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Texel {
    R32f,
    R16,
    R8,
    Rg8,
    Rgba8,
}

impl Texel {
    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::R16 | Self::Rg8 => 2,
            Self::R32f | Self::Rgba8 => 4,
        }
    }

    pub(crate) fn vk_format(self, srgb: bool) -> u32 {
        match self {
            Self::R32f => vk::R32_SFLOAT,
            Self::R16 => vk::R16_UNORM,
            Self::R8 => vk::R8_UNORM,
            Self::Rg8 => vk::R8G8_UNORM,
            Self::Rgba8 if srgb => vk::R8G8B8A8_SRGB,
            Self::Rgba8 => vk::R8G8B8A8_UNORM,
        }
    }

    /// Linear float texels for the BCn encoders; sRGB colour is decoded when `srgb`.
    pub(crate) fn to_img(self, srgb: bool, size: u32, bytes: &[u8]) -> Img {
        let unorm8 = |b: u8| f32::from(b) / 255.0;
        let colour = |b: u8| {
            if srgb {
                srgb_to_linear(unorm8(b))
            } else {
                unorm8(b)
            }
        };
        let px = bytes
            .chunks_exact(self.bytes())
            .map(|b| match self {
                Self::R32f => [f32::from_le_bytes([b[0], b[1], b[2], b[3]]), 0.0, 0.0, 1.0],
                Self::R16 => [
                    f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                Self::R8 => [unorm8(b[0]), 0.0, 0.0, 1.0],
                Self::Rg8 => [unorm8(b[0]), unorm8(b[1]), 0.0, 1.0],
                Self::Rgba8 => [colour(b[0]), colour(b[1]), colour(b[2]), unorm8(b[3])],
            })
            .collect();
        Img {
            w: size,
            h: size,
            px,
        }
    }
}

/// How a coarser LOD texel comes from its 2x2 block of finer texels.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reduction {
    /// f32: `(a + b + c + d) * 0.25` with `a` top-left, `b` right of it, `c` below it, `d`
    /// diagonal. Integer channels: the rounded mean; sRGB colour is averaged in linear
    /// light.
    #[default]
    Mean,
    /// Per channel; f32 NaN loses to any number.
    Min,
    Max,
    /// The top-left texel.
    Nearest,
}

impl Reduction {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Mean => RasterLayer::MEAN,
            Self::Min => RasterLayer::MIN,
            Self::Max => RasterLayer::MAX,
            Self::Nearest => RasterLayer::NEAREST,
        }
    }
}

/// Reduces two rows of finer texels into one row of half the width.
pub(crate) fn reduce_rows(
    t: Texel,
    r: Reduction,
    srgb: bool,
    top: &[u8],
    bottom: &[u8],
    out: &mut [u8],
) {
    let n = t.bytes();
    debug_assert_eq!(top.len(), out.len() * 2);
    if r == Reduction::Nearest {
        for (o, src) in out.chunks_exact_mut(n).zip(top.chunks_exact(2 * n)) {
            o.copy_from_slice(&src[..n]);
        }
        return;
    }
    match t {
        Texel::R32f => {
            let f = |b: &[u8], i: usize| f32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
            for (i, o) in out.chunks_exact_mut(4).enumerate() {
                let (a, b) = (f(top, 8 * i), f(top, 8 * i + 4));
                let (c, d) = (f(bottom, 8 * i), f(bottom, 8 * i + 4));
                let v = match r {
                    Reduction::Mean => (a + b + c + d) * 0.25,
                    Reduction::Min => a.min(b).min(c).min(d),
                    Reduction::Max => a.max(b).max(c).max(d),
                    Reduction::Nearest => unreachable!("handled above"),
                };
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
        Texel::R16 => {
            let u = |b: &[u8], i: usize| u32::from(u16::from_le_bytes([b[i], b[i + 1]]));
            for (i, o) in out.chunks_exact_mut(2).enumerate() {
                let v = [
                    u(top, 4 * i),
                    u(top, 4 * i + 2),
                    u(bottom, 4 * i),
                    u(bottom, 4 * i + 2),
                ];
                o.copy_from_slice(&(reduce_int(r, v) as u16).to_le_bytes());
            }
        }
        Texel::R8 | Texel::Rg8 | Texel::Rgba8 => {
            let lut = (srgb && r == Reduction::Mean && t == Texel::Rgba8).then(srgb_lut);
            for (i, o) in out.iter_mut().enumerate() {
                let (px, ch) = (i / n, i % n);
                let at = 2 * px * n + ch;
                let v = [top[at], top[at + n], bottom[at], bottom[at + n]];
                *o = match &lut {
                    Some(lut) if ch < 3 => {
                        let l = v.iter().map(|&b| lut[usize::from(b)]).sum::<f32>() * 0.25;
                        (linear_to_srgb(l) * 255.0).round() as u8
                    }
                    _ => reduce_int(r, v.map(u32::from)) as u8,
                };
            }
        }
    }
}

fn reduce_int(r: Reduction, v: [u32; 4]) -> u32 {
    match r {
        Reduction::Mean => (v.iter().sum::<u32>() + 2) / 4,
        Reduction::Min => v.into_iter().min().expect("four values"),
        Reduction::Max => v.into_iter().max().expect("four values"),
        Reduction::Nearest => v[0],
    }
}

fn srgb_lut() -> [f32; 256] {
    std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0))
}
