//! Texel formats of raster layers, the 2x2 reductions and value ranges.

use orr_color::{srgb8_decode, srgb8_decode_fixed, srgb8_encode_fixed};
use orr_pack::raster::RasterLayer;
use orr_pack::texture::vk;
use serde::Deserialize;

use crate::Result;
use crate::texture::Img;

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
    pub(super) fn bytes(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::R16 | Self::Rg8 => 2,
            Self::Rgb8 => 3,
            Self::R32f | Self::Rgba8 => 4,
        }
    }

    pub(super) fn stored(self) -> Texel {
        match self {
            Self::R32f => Texel::R32f,
            Self::R16 => Texel::R16,
            Self::R8 => Texel::R8,
            Self::Rg8 => Texel::Rg8,
            Self::Rgb8 | Self::Rgba8 => Texel::Rgba8,
        }
    }

    /// Input texels to stored texels, appended to `out`.
    pub(super) fn expand(self, src: &[u8], out: &mut Vec<u8>) {
        if self == Self::Rgb8 {
            out.reserve(src.len() / 3 * 4);
            for p in src.as_chunks::<3>().0 {
                out.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
        } else {
            out.extend_from_slice(src);
        }
    }

    /// Widens `range` to the finite channel values of input texels, UNORM normalized.
    pub(super) fn range(self, src: &[u8], range: &mut [f32; 2]) {
        let mut add = |v: f32| {
            if v.is_finite() {
                range[0] = range[0].min(v);
                range[1] = range[1].max(v);
            }
        };
        match self {
            Self::R32f => src
                .as_chunks::<4>()
                .0
                .iter()
                .for_each(|b| add(f32::from_le_bytes(*b))),
            Self::R16 => src
                .as_chunks::<2>()
                .0
                .iter()
                .for_each(|b| add(f32::from(u16::from_le_bytes(*b)) / 65535.0)),
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
pub(super) enum Texel {
    R32f,
    R16,
    R8,
    Rg8,
    Rgba8,
}

impl Texel {
    pub(super) fn bytes(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::R16 | Self::Rg8 => 2,
            Self::R32f | Self::Rgba8 => 4,
        }
    }

    pub(super) fn vk_format(self, srgb: bool) -> u32 {
        match self {
            Self::R32f => vk::R32_SFLOAT,
            Self::R16 => vk::R16_UNORM,
            Self::R8 => vk::R8_UNORM,
            Self::Rg8 => vk::R8G8_UNORM,
            Self::Rgba8 if srgb => vk::R8G8B8A8_SRGB,
            Self::Rgba8 => vk::R8G8B8A8_UNORM,
        }
    }

    /// Linear float texels for the BCn encoders; sRGB color is decoded when `srgb`.
    pub(super) fn to_img(self, srgb: bool, size: u32, bytes: &[u8]) -> Img {
        let unorm8 = |b: u8| f32::from(b) / 255.0;
        let color = |b: u8| if srgb { srgb8_decode(b) } else { unorm8(b) };
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
                Self::Rgba8 => [color(b[0]), color(b[1]), color(b[2]), unorm8(b[3])],
            })
            .collect();
        Img::of(size, size, px)
    }
}

/// How a coarser LOD texel comes from its 2x2 block of finer texels.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reduction {
    /// f32: `(a + b + c + d) * 0.25` with `a` top-left, `b` right of it, `c` below it, `d`
    /// diagonal. Integer channels: the mean of the finest texels under the texel, rounded
    /// once to the nearest value with ties to even; sRGB color is averaged in linear light.
    #[default]
    Mean,
    /// Per channel; f32 NaN loses to any number.
    Min,
    Max,
    /// The top-left texel.
    Nearest,
}

impl Reduction {
    pub(super) fn code(self) -> u8 {
        match self {
            Self::Mean => RasterLayer::MEAN,
            Self::Min => RasterLayer::MIN,
            Self::Max => RasterLayer::MAX,
            Self::Nearest => RasterLayer::NEAREST,
        }
    }
}

/// Builds the coarser levels of a raster from its finest rows as they arrive: level `k`
/// (1 for half the finest size) takes a row from every `2^k` finest rows.
pub(super) struct Cascade {
    t: Texel,
    r: Reduction,
    /// Whether color channels are sRGB codes that average in fixed-point linear light.
    linear: bool,
    /// By level, from the first coarser one: the upper row of the pair being formed.
    pending: Vec<Option<Row>>,
}

/// A row of one level: stored texels, or the exact sums of the finest texel values under
/// each of its texels' channels.
enum Row {
    Texels(Vec<u8>),
    Sums(Vec<u128>),
}

/// Bits below the point of the fixed-point linear light that sRGB means sum.
const LINEAR_BITS: u32 = 64;

impl Cascade {
    pub(super) fn new(t: Texel, r: Reduction, srgb: bool, coarser: usize) -> Self {
        Self {
            t,
            r,
            linear: srgb && r == Reduction::Mean && t == Texel::Rgba8,
            pending: (0..coarser).map(|_| None).collect(),
        }
    }

    /// Takes the next finest row; `emit(k, texels)` receives each row of level `k` it
    /// completes, finer levels first.
    pub(super) fn push(
        &mut self,
        finest: &[u8],
        mut emit: impl FnMut(usize, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let sums = self.t != Texel::R32f && self.r == Reduction::Mean;
        let mut row = if sums {
            Row::Sums(self.values(finest))
        } else {
            Row::Texels(finest.to_vec())
        };
        for k in 0..self.pending.len() {
            let Some(top) = self.pending[k].take() else {
                self.pending[k] = Some(row);
                break;
            };
            row = match (top, row) {
                (Row::Texels(top), Row::Texels(bottom)) => {
                    let mut out = vec![0; bottom.len() / 2];
                    reduce_rows(self.t, self.r, &top, &bottom, &mut out);
                    emit(k + 1, &out)?;
                    Row::Texels(out)
                }
                (Row::Sums(top), Row::Sums(bottom)) => {
                    let out = sum_rows(self.channels(), &top, &bottom);
                    emit(k + 1, &self.means(&out, k as u32 + 1))?;
                    Row::Sums(out)
                }
                _ => unreachable!("a cascade's rows are all texels or all sums"),
            };
        }
        Ok(())
    }

    fn channels(&self) -> usize {
        match self.t {
            Texel::R16 => 1,
            t => t.bytes(),
        }
    }

    /// The channel values of a row of stored texels, in linear light for sRGB color.
    fn values(&self, texels: &[u8]) -> Vec<u128> {
        match self.t {
            Texel::R16 => texels
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u128::from(u16::from_le_bytes(*b)))
                .collect(),
            _ => texels
                .iter()
                .enumerate()
                .map(|(i, &b)| {
                    if self.linear && i % 4 < 3 {
                        srgb8_decode_fixed(b, LINEAR_BITS)
                    } else {
                        u128::from(b)
                    }
                })
                .collect(),
        }
    }

    /// Stored texels of level `k` from its sums over `4^k` finest texels each.
    fn means(&self, sums: &[u128], k: u32) -> Vec<u8> {
        let shift = 2 * k;
        let int = |s: u128| {
            let (q, rest, half) = (s >> shift, s & ((1 << shift) - 1), 1 << (shift - 1));
            q + u128::from(rest > half || (rest == half && q & 1 == 1))
        };
        match self.t {
            Texel::R16 => sums
                .iter()
                .flat_map(|&s| (int(s) as u16).to_le_bytes())
                .collect(),
            _ => sums
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    if self.linear && i % 4 < 3 {
                        srgb8_encode_fixed(s, LINEAR_BITS + shift)
                    } else {
                        int(s) as u8
                    }
                })
                .collect(),
        }
    }
}

/// The 2x2 sums of two rows of `channels`-channel sums: a row of half the width.
fn sum_rows(channels: usize, top: &[u128], bottom: &[u128]) -> Vec<u128> {
    let n = channels;
    (0..top.len() / 2)
        .map(|i| {
            let (px, ch) = (i / n, i % n);
            let at = 2 * px * n + ch;
            top[at] + top[at + n] + bottom[at] + bottom[at + n]
        })
        .collect()
}

/// Reduces two rows of finer stored texels into one row of half the width, for every
/// reduction but the integer mean.
fn reduce_rows(t: Texel, r: Reduction, top: &[u8], bottom: &[u8], out: &mut [u8]) {
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
            for (i, o) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
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
            for (i, o) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
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
            for (i, o) in out.iter_mut().enumerate() {
                let (px, ch) = (i / n, i % n);
                let at = 2 * px * n + ch;
                let v = [top[at], top[at + n], bottom[at], bottom[at + n]];
                *o = reduce_int(r, v.map(u32::from)) as u8;
            }
        }
    }
}

fn reduce_int(r: Reduction, v: [u32; 4]) -> u32 {
    match r {
        Reduction::Mean => unreachable!("integer means reduce from sums"),
        Reduction::Min => v.into_iter().min().expect("four values"),
        Reduction::Max => v.into_iter().max().expect("four values"),
        Reduction::Nearest => v[0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coarser levels of the square `finest` (`size` texels a side) through the cascade,
    /// from level 1.
    fn cascade(t: Texel, srgb: bool, size: usize, coarser: usize, finest: &[u8]) -> Vec<Vec<u8>> {
        let mut c = Cascade::new(t, Reduction::Mean, srgb, coarser);
        let mut levels = vec![Vec::new(); coarser];
        for row in finest.chunks_exact(finest.len() / size) {
            c.push(row, |k, out| {
                levels[k - 1].extend_from_slice(out);
                Ok(())
            })
            .unwrap();
        }
        levels
    }

    /// The finest values of channel `ch` under texel `(x, y)` of level `k`.
    fn block(
        finest: &[u32],
        size: usize,
        channels: usize,
        k: usize,
        x: usize,
        y: usize,
        ch: usize,
    ) -> impl Iterator<Item = u32> + '_ {
        let n = 1 << k;
        (0..n * n).map(move |i| finest[((y * n + i / n) * size + x * n + i % n) * channels + ch])
    }

    fn noise(len: usize, mut s: u64) -> impl Iterator<Item = u64> {
        (0..len).map(move |_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
    }

    #[test]
    fn integer_means_are_the_rounded_means_of_the_finest_texels() {
        let (size, coarser) = (512, 9);
        let finest: Vec<u32> = noise(size * size, 0x9E37_79B9_7F4A_7C15)
            .map(|v| (v >> 48) as u32)
            .collect();
        let bytes: Vec<u8> = finest
            .iter()
            .flat_map(|&v| (v as u16).to_le_bytes())
            .collect();
        let levels = cascade(Texel::R16, false, size, coarser, &bytes);
        for (k, level) in (1..).zip(&levels) {
            let m = size >> k;
            for (i, got) in level.as_chunks::<2>().0.iter().enumerate() {
                let n = 1u64 << (2 * k);
                let sum: u64 = block(&finest, size, 1, k, i % m, i / m, 0)
                    .map(u64::from)
                    .sum();
                let (q, r) = (sum / n, sum % n);
                let want = q + u64::from(2 * r > n || (2 * r == n && q % 2 == 1));
                assert_eq!(
                    u64::from(u16::from_le_bytes(*got)),
                    want,
                    "level {k} texel {i}"
                );
            }
        }
    }

    #[test]
    fn srgb_means_average_the_finest_texels_in_linear_light() {
        let (size, coarser) = (128, 7);
        let finest: Vec<u32> = noise(size * size * 4, 0x2545_F491_4F6C_DD1D)
            .map(|v| (v >> 56) as u32)
            .collect();
        let bytes: Vec<u8> = finest.iter().map(|&v| v as u8).collect();
        let levels = cascade(Texel::Rgba8, true, size, coarser, &bytes);
        for (k, level) in (1..).zip(&levels) {
            let m = size >> k;
            for (i, &got) in level.iter().enumerate() {
                let (px, ch) = (i / 4, i % 4);
                let values = block(&finest, size, 4, k, px % m, px / m, ch);
                let n = (1u64 << (2 * k)) as f64;
                let want = if ch < 3 {
                    let linear: f64 = values
                        .map(|v| orr_color::srgb_decode(f64::from(v) / 255.0))
                        .sum();
                    (orr_color::srgb_encode(linear / n) * 255.0).round_ties_even()
                } else {
                    (values.map(f64::from).sum::<f64>() / n).round_ties_even()
                };
                assert_eq!(f64::from(got), want, "level {k} texel {px} channel {ch}");
            }
        }
    }
}
