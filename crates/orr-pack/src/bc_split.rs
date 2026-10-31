use crate::texture::vk;
use crate::{Error, Result};

/// Block layout of a BCn format, for [`crate::Filter::BcSplit`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BcLayout {
    Bc1 = 0,
    Bc2 = 1,
    Bc3 = 2,
    Bc4 = 3,
    Bc5 = 4,
    Bc6h = 5,
    Bc7 = 6,
}

/// Byte fields `(offset, len)` of a block, endpoints before indices.
const BC1: &[(usize, usize)] = &[(0, 4), (4, 4)];
const BC2: &[(usize, usize)] = &[(8, 4), (0, 8), (12, 4)];
const BC3: &[(usize, usize)] = &[(0, 2), (8, 4), (2, 6), (12, 4)];
const BC4: &[(usize, usize)] = &[(0, 2), (2, 6)];
const BC5: &[(usize, usize)] = &[(0, 2), (8, 2), (2, 6), (10, 6)];

/// Mode bytes run 0..MODES.
const MODES: usize = 32;
const MAX_FIELDS: usize = 8;

/// A BC6H or BC7 block mode: its mode bits and the widths of the fields after them.
struct Mode {
    bits: u32,
    value: u128,
    fields: &'static [u32],
}

impl BcLayout {
    pub const ALL: [Self; 7] = [
        Self::Bc1,
        Self::Bc2,
        Self::Bc3,
        Self::Bc4,
        Self::Bc5,
        Self::Bc6h,
        Self::Bc7,
    ];

    /// The layout of a BCn `VkFormat`, or `None` for any other format.
    pub fn of(vk_format: u32) -> Option<Self> {
        use vk::*;
        Some(match vk_format {
            BC1_RGB_UNORM | BC1_RGB_SRGB | BC1_RGBA_UNORM | BC1_RGBA_SRGB => Self::Bc1,
            BC2_UNORM | BC2_SRGB => Self::Bc2,
            BC3_UNORM | BC3_SRGB => Self::Bc3,
            BC4_UNORM | BC4_SNORM => Self::Bc4,
            BC5_UNORM | BC5_SNORM => Self::Bc5,
            BC6H_UFLOAT | BC6H_SFLOAT => Self::Bc6h,
            BC7_UNORM | BC7_SRGB => Self::Bc7,
            _ => return None,
        })
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Bc1 | Self::Bc4 => 8,
            _ => 16,
        }
    }

    fn byte_fields(self) -> Option<&'static [(usize, usize)]> {
        match self {
            Self::Bc1 => Some(BC1),
            Self::Bc2 => Some(BC2),
            Self::Bc3 => Some(BC3),
            Self::Bc4 => Some(BC4),
            Self::Bc5 => Some(BC5),
            Self::Bc6h | Self::Bc7 => None,
        }
    }

    /// Mode byte of a BC6H or BC7 block.
    fn mode_of(self, block: u128) -> u8 {
        let low = block as u8;
        match self {
            Self::Bc7 if low == 0 => 8,
            Self::Bc7 => low.trailing_zeros() as u8,
            _ if low & 3 < 2 => low & 3,
            _ => low & 31,
        }
    }

    fn mode(self, code: u8) -> Option<Mode> {
        let mode = |bits, value, fields| {
            Some(Mode {
                bits,
                value,
                fields,
            })
        };
        match (self, code) {
            // Partition, then R, G, B (and A) endpoints, p-bits, indices; mode 4 and 5 lead
            // with rotation (and the index mode) and split color and alpha.
            (Self::Bc7, 0) => mode(1, 1, &[4, 24, 24, 24, 6, 45]),
            (Self::Bc7, 1) => mode(2, 2, &[6, 24, 24, 24, 2, 46]),
            (Self::Bc7, 2) => mode(3, 4, &[6, 30, 30, 30, 29]),
            (Self::Bc7, 3) => mode(4, 8, &[6, 28, 28, 28, 4, 30]),
            (Self::Bc7, 4) => mode(5, 16, &[2, 1, 10, 10, 10, 12, 31, 47]),
            (Self::Bc7, 5) => mode(6, 32, &[2, 14, 14, 14, 16, 31, 31]),
            (Self::Bc7, 6) => mode(7, 64, &[14, 14, 14, 14, 2, 63]),
            (Self::Bc7, 7) => mode(8, 128, &[6, 20, 20, 20, 20, 4, 30]),
            (Self::Bc7, 8) => mode(8, 0, &[120]),
            // Endpoint bits, partition, indices.
            (Self::Bc6h, 0 | 1) => mode(2, u128::from(code), &[75, 5, 46]),
            (Self::Bc6h, c) if c < 32 && c & 3 == 2 => mode(5, u128::from(c), &[72, 5, 46]),
            (Self::Bc6h, c) if c < 16 && c & 3 == 3 => mode(5, u128::from(c), &[60, 63]),
            (Self::Bc6h, c) if c < 32 && c & 3 == 3 => mode(5, u128::from(c), &[123]),
            _ => None,
        }
    }

    /// Bytes a field of `bits` takes per block.
    fn field_bytes(bits: u32) -> usize {
        bits.div_ceil(8) as usize
    }

    /// Largest split of `raw` bytes.
    pub(crate) fn max_split_len(self, raw: usize) -> usize {
        if self.byte_fields().is_some() {
            return raw;
        }
        let most = (0..MODES as u8)
            .filter_map(|c| self.mode(c))
            .map(|m| {
                m.fields
                    .iter()
                    .map(|&w| Self::field_bytes(w))
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        raw / 16 * (1 + most) + raw % 16
    }

    pub(crate) fn split(self, src: &[u8], dst: &mut Vec<u8>) {
        let bb = self.block_bytes();
        let n = src.len() / bb;
        dst.clear();
        if let Some(fields) = self.byte_fields() {
            for &(o, len) in fields {
                for b in src[..n * bb].chunks_exact(bb) {
                    dst.extend_from_slice(&b[o..o + len]);
                }
            }
        } else {
            let blocks: Vec<u128> = src[..n * 16]
                .chunks_exact(16)
                .map(|b| u128::from_le_bytes(b.try_into().unwrap()))
                .collect();
            dst.extend(blocks.iter().map(|&b| self.mode_of(b)));
            let mut cursors = self.cursors(dst);
            let start = dst.len();
            dst.resize(start + cursors.1, 0);
            for &b in &blocks {
                let code = self.mode_of(b);
                let m = self
                    .mode(code)
                    .expect("mode_of gives mode bytes that have a mode");
                let mut v = b >> m.bits;
                for (f, &w) in m.fields.iter().enumerate() {
                    let fb = Self::field_bytes(w);
                    let at = start + cursors.0[usize::from(code)][f];
                    let x = v & ((1u128 << w) - 1);
                    dst[at..at + fb].copy_from_slice(&x.to_le_bytes()[..fb]);
                    cursors.0[usize::from(code)][f] += fb;
                    v >>= w;
                }
            }
        }
        dst.extend_from_slice(&src[n * bb..]);
    }

    /// Start of each mode's field streams after the mode bytes, and their total length.
    fn cursors(self, modes: &[u8]) -> ([[usize; MAX_FIELDS]; MODES], usize) {
        let mut count = [0usize; MODES];
        for &c in modes {
            count[usize::from(c)] += 1;
        }
        let mut at = [[0; MAX_FIELDS]; MODES];
        let mut off = 0;
        for (c, n) in count.iter().enumerate() {
            if let Some(m) = self.mode(c as u8) {
                for (f, &w) in m.fields.iter().enumerate() {
                    at[c][f] = off;
                    off += n * Self::field_bytes(w);
                }
            }
        }
        (at, off)
    }

    pub(crate) fn join(self, src: &[u8], dst: &mut [u8]) -> Result<()> {
        let bb = self.block_bytes();
        let n = dst.len() / bb;
        let tail = dst.len() - n * bb;
        let bad = || {
            Error::Decode(format!(
                "{self:?} split of {} bytes is malformed",
                src.len()
            ))
        };
        if let Some(fields) = self.byte_fields() {
            if src.len() != dst.len() {
                return Err(bad());
            }
            let mut at = 0;
            for &(o, len) in fields {
                for b in dst[..n * bb].chunks_exact_mut(bb) {
                    b[o..o + len].copy_from_slice(&src[at..at + len]);
                    at += len;
                }
            }
        } else {
            let modes = src.get(..n).ok_or_else(bad)?;
            if modes.iter().any(|&c| self.mode(c).is_none()) {
                return Err(bad());
            }
            let (mut cursors, len) = self.cursors(modes);
            if src.len() != n + len + tail {
                return Err(bad());
            }
            let streams = &src[n..n + len];
            for (b, &code) in dst[..n * 16].chunks_exact_mut(16).zip(modes) {
                let m = self.mode(code).ok_or_else(bad)?;
                let (mut v, mut shift) = (m.value, m.bits);
                for (f, &w) in m.fields.iter().enumerate() {
                    let fb = Self::field_bytes(w);
                    let at = &mut cursors[usize::from(code)][f];
                    let mut x = [0; 16];
                    x[..fb].copy_from_slice(&streams[*at..*at + fb]);
                    *at += fb;
                    let x = u128::from_le_bytes(x);
                    if x >> w != 0 {
                        return Err(bad());
                    }
                    v |= x << shift;
                    shift += w;
                }
                b.copy_from_slice(&v.to_le_bytes());
            }
        }
        dst[n * bb..].copy_from_slice(&src[src.len() - tail..]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    /// Random blocks, with the first byte forced through every mode byte the layout has.
    fn blocks(l: BcLayout, n: usize, tail: usize) -> Vec<u8> {
        let bb = l.block_bytes();
        let mut v = noise(n * bb + tail, l as u64 + 7);
        for (i, b) in v[..n * bb].chunks_exact_mut(bb).enumerate() {
            match l {
                BcLayout::Bc7 => {
                    b[0] = if i % 9 == 8 {
                        0
                    } else {
                        b[0] << (i % 9) | 1 << (i % 9)
                    }
                }
                BcLayout::Bc6h if i % 3 == 0 => b[0] &= !2,
                _ => {}
            }
        }
        v
    }

    #[test]
    fn round_trips_every_layout_and_mode() {
        for l in BcLayout::ALL {
            for (n, tail) in [(0, 0), (0, 5), (1, 0), (300, 0), (300, 7)] {
                let raw = blocks(l, n, tail);
                let mut split = Vec::new();
                l.split(&raw, &mut split);
                assert!(split.len() <= l.max_split_len(raw.len()), "{l:?}");
                let mut back = vec![0; raw.len()];
                l.join(&split, &mut back).unwrap();
                assert_eq!(back, raw, "{l:?} {n} blocks, {tail} tail bytes");
            }
        }
        let raw = blocks(BcLayout::Bc6h, 4000, 0);
        let modes: std::collections::HashSet<u8> = raw
            .chunks_exact(16)
            .map(|b| BcLayout::Bc6h.mode_of(u128::from_le_bytes(b.try_into().unwrap())))
            .collect();
        assert_eq!(modes.len(), 18, "every BC6H mode and reserved code");
    }

    #[test]
    fn refuses_bad_splits() {
        for l in [BcLayout::Bc5, BcLayout::Bc6h, BcLayout::Bc7] {
            let raw = blocks(l, 20, 3);
            let mut split = Vec::new();
            l.split(&raw, &mut split);
            let mut back = vec![0; raw.len()];
            assert!(l.join(&split[..split.len() - 1], &mut back).is_err());
            split.push(0);
            assert!(l.join(&split, &mut back).is_err());
        }
        let mut split = Vec::new();
        BcLayout::Bc7.split(&blocks(BcLayout::Bc7, 2, 0), &mut split);
        split[0] = 9;
        assert!(BcLayout::Bc7.join(&split, &mut [0; 32]).is_err());
        BcLayout::Bc6h.split(&blocks(BcLayout::Bc6h, 2, 0), &mut split);
        split[0] = 4;
        assert!(BcLayout::Bc6h.join(&split, &mut [0; 32]).is_err());
    }
}
