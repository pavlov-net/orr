//! Texture class: blob 0 holds a [`TextureHeader`] and its light table ([`TextureInfo`]);
//! then one blob per mip above the tail, finest first; then the mip tail.
//!
//! - A mip blob holds 64 KiB tiles in the Vulkan standard 2D sparse block shape
//!   ([`tile_shape`]), layer by layer, each layer's tiles row-major. A tile's block rows are
//!   tightly packed at the full tile width, and tiles at the right and bottom edges are
//!   padded with zeros, so every tile copies with a row length and image height of the full
//!   tile. Each tile is one chunk.
//! - The tail starts at the first mip smaller than one tile in either axis
//!   ([`tail_first_mip`]). It is one blob holding every tail mip, finest first, each mip's
//!   layers in order, each layer's block rows tightly packed.
//! - Cube textures have six layers in the Vulkan face order +X, -X, +Y, -Y, +Z, -Z.
//! - Environment maps are 2D equal-area octahedral maps ([`octahedral`]).

use bytemuck::{Pod, Zeroable};

use crate::blob::{put_table, read, table};
use crate::{Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Bytes per tile.
pub const TILE_BYTES: usize = 64 << 10;

/// Blob 0 header.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct TextureHeader {
    /// Vulkan format number (the [`vk`] constants).
    pub vk_format: u32,
    pub width: u32,
    pub height: u32,
    pub layers: u16,
    /// [`Self::D2`] or [`Self::CUBE`].
    pub kind: u8,
    pub mip_count: u8,
    /// Mips from this one down share the last blob.
    pub tail_first_mip: u8,
    /// Source primaries of colour texels: [`Self::REC709`], [`Self::DISPLAY_P3`] or
    /// [`Self::REC2020`].
    pub primaries: u8,
    pub _pad: [u8; 2],
    /// Cutoff the alpha-coverage mips were built for; NaN when not alpha-tested.
    pub alpha_cutoff: f32,
    /// [`ExtractedLight`] table, brightest first; spans are byte offsets in blob 0.
    pub lights: Span,
}

impl TextureHeader {
    pub const D2: u8 = 0;
    pub const CUBE: u8 = 1;
    pub const REC709: u8 = 0;
    pub const DISPLAY_P3: u8 = 1;
    pub const REC2020: u8 = 2;

    pub fn format(&self) -> Result<FormatInfo> {
        FormatInfo::of(self.vk_format).ok_or(Error::Malformed("texture format"))
    }

    /// Texels per axis of `mip`.
    pub fn mip_size(&self, mip: u32) -> [u32; 2] {
        mip_size(self.width, self.height, mip)
    }

    /// Blob holding `mip`: blob `1 + mip` above the tail, the last blob for tail mips.
    pub fn mip_blob(&self, mip: u32) -> usize {
        1 + mip.min(u32::from(self.tail_first_mip)) as usize
    }

    pub fn blob_count(&self) -> usize {
        let tail = u32::from(self.tail_first_mip) < u32::from(self.mip_count);
        1 + usize::from(self.tail_first_mip) + usize::from(tail)
    }
}

/// A bright region of an environment map moved into a directional light of equal energy.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct ExtractedLight {
    /// Unit direction toward the region's energy centroid, in map space.
    pub direction_to_light: [f32; 3],
    /// Integral of the removed radiance over the region's solid angle: map units times sr.
    pub illuminance: [f32; 3],
    /// Half angle of the cone whose solid angle equals the region's, in radians.
    pub angular_radius: f32,
    pub _pad: f32,
}

/// A texture blob 0: the header and its light table.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureInfo {
    pub header: TextureHeader,
    pub lights: Vec<ExtractedLight>,
}

impl TextureInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let header: TextureHeader = read(blob, 0).ok_or(Error::Malformed("texture header"))?;
        let f = header.format()?;
        let h = &header;
        let full = full_mip_count(h.width, h.height);
        if h.width == 0
            || h.height == 0
            || h.layers == 0
            || h.mip_count == 0
            || u32::from(h.mip_count) > full
            || h.kind > TextureHeader::CUBE
            || (h.kind == TextureHeader::CUBE && !h.layers.is_multiple_of(6))
            || u32::from(h.tail_first_mip)
                != tail_first_mip(f, h.width, h.height, u32::from(h.mip_count))
        {
            return Err(Error::Malformed("texture header"));
        }
        Ok(Self {
            lights: table(blob, header.lights)?,
            header,
        })
    }

    /// Blob bytes; the header's light span is set from the table.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut out = vec![0; size_of::<TextureHeader>()];
        h.lights = put_table(&mut out, &self.lights);
        out[..size_of::<TextureHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    /// Decoded length of every blob after blob 0, in order.
    pub fn blob_lens(&self) -> Vec<usize> {
        let h = &self.header;
        let f = h.format().expect("parsed header");
        let layers = usize::from(h.layers);
        let tail = u32::from(h.tail_first_mip);
        let mut lens: Vec<usize> = (0..tail)
            .map(|m| {
                let [gx, gy] = tile_grid(h, m);
                (gx * gy) as usize * layers * TILE_BYTES
            })
            .collect();
        if tail < u32::from(h.mip_count) {
            lens.push(
                (tail..u32::from(h.mip_count))
                    .map(|m| {
                        let [w, hh] = h.mip_size(m);
                        f.mip_bytes(w, hh) * layers
                    })
                    .sum(),
            );
        }
        lens
    }

    /// Tightly packed block rows of `layer` at `mip`, from the decoded bytes of the blob that
    /// holds the mip.
    pub fn mip_texels(&self, blob: &[u8], mip: u32, layer: u32) -> Result<Vec<u8>> {
        let h = &self.header;
        let f = h.format()?;
        if mip >= u32::from(h.mip_count) || layer >= u32::from(h.layers) {
            return Err(Error::Malformed("texture mip or layer out of range"));
        }
        let [w, hh] = h.mip_size(mip);
        let size = f.mip_bytes(w, hh);
        let layers = usize::from(h.layers);
        if mip >= u32::from(h.tail_first_mip) {
            let before: usize = (u32::from(h.tail_first_mip)..mip)
                .map(|m| {
                    let [w, hh] = h.mip_size(m);
                    f.mip_bytes(w, hh) * layers
                })
                .sum();
            let at = before + layer as usize * size;
            return blob
                .get(at..at + size)
                .map(<[u8]>::to_vec)
                .ok_or(Error::Malformed("texture tail blob too short"));
        }
        let [gx, gy] = tile_grid(h, mip);
        let per_layer = (gx * gy) as usize * TILE_BYTES;
        let at = layer as usize * per_layer;
        let tiles = blob
            .get(at..at + per_layer)
            .ok_or(Error::Malformed("texture mip blob too short"))?;
        Ok(untile(f, w, hh, tiles))
    }
}

/// Block size and bytes of a texture format.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FormatInfo {
    /// Texels per block edge: 4 for BCn, 1 for uncompressed formats.
    pub block: u32,
    /// Bytes per block or texel.
    pub bytes: u32,
}

impl FormatInfo {
    pub fn of(vk_format: u32) -> Option<Self> {
        use vk::*;
        let (block, bytes) = match vk_format {
            R8_UNORM => (1, 1),
            R8G8_UNORM | R16_UNORM | R16_SFLOAT => (1, 2),
            R8G8B8A8_UNORM | R8G8B8A8_SRGB | R16G16_UNORM | R16G16_SFLOAT | R32_SFLOAT => (1, 4),
            R16G16B16A16_UNORM | R16G16B16A16_SFLOAT | R32G32_SFLOAT => (1, 8),
            R32G32B32A32_SFLOAT => (1, 16),
            BC1_RGB_UNORM | BC1_RGB_SRGB | BC1_RGBA_UNORM | BC1_RGBA_SRGB | BC4_UNORM
            | BC4_SNORM => (4, 8),
            BC2_UNORM | BC2_SRGB | BC3_UNORM | BC3_SRGB | BC5_UNORM | BC5_SNORM | BC6H_UFLOAT
            | BC6H_SFLOAT | BC7_UNORM | BC7_SRGB => (4, 16),
            _ => return None,
        };
        Some(Self { block, bytes })
    }

    /// Bytes of a `w` x `h` mip with tightly packed block rows.
    pub fn mip_bytes(self, w: u32, h: u32) -> usize {
        w.div_ceil(self.block) as usize * h.div_ceil(self.block) as usize * self.bytes as usize
    }
}

/// Tile size in texels: the Vulkan standard 2D sparse block shape for single-sample images,
/// 64 KiB for every format.
pub fn tile_shape(f: FormatInfo) -> [u32; 2] {
    let [w, h] = match f.bytes {
        1 => [256, 256],
        2 => [256, 128],
        4 => [128, 128],
        8 => [128, 64],
        _ => [64, 64],
    };
    [w * f.block, h * f.block]
}

/// Tiles per axis of `mip`.
pub fn tile_grid(h: &TextureHeader, mip: u32) -> [u32; 2] {
    let f = h.format().expect("texture format");
    let [tw, th] = tile_shape(f);
    let [w, hh] = h.mip_size(mip);
    [w.div_ceil(tw), hh.div_ceil(th)]
}

pub fn mip_size(width: u32, height: u32, mip: u32) -> [u32; 2] {
    [(width >> mip).max(1), (height >> mip).max(1)]
}

/// Mips down to 1x1.
pub fn full_mip_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// The first mip smaller than one tile in either axis, or `mips` when there is none.
pub fn tail_first_mip(f: FormatInfo, width: u32, height: u32, mips: u32) -> u32 {
    let [tw, th] = tile_shape(f);
    (0..mips)
        .find(|&m| {
            let [w, h] = mip_size(width, height, m);
            w < tw || h < th
        })
        .unwrap_or(mips)
}

/// Cuts one layer's tightly packed mip into tiles, row-major, edge tiles zero-padded.
pub fn tile(f: FormatInfo, w: u32, h: u32, texels: &[u8], out: &mut Vec<u8>) {
    let [tw, th] = tile_shape(f).map(|t| (t / f.block) as usize);
    let (bw, bh) = (w.div_ceil(f.block) as usize, h.div_ceil(f.block) as usize);
    let bpb = f.bytes as usize;
    assert_eq!(texels.len(), bw * bh * bpb, "mip size");
    for ty in 0..bh.div_ceil(th) {
        for tx in 0..bw.div_ceil(tw) {
            let at = out.len();
            out.resize(at + TILE_BYTES, 0);
            let tile = &mut out[at..];
            let x0 = tx * tw;
            let n = tw.min(bw - x0) * bpb;
            for y in 0..th.min(bh - ty * th) {
                let src = ((ty * th + y) * bw + x0) * bpb;
                tile[y * tw * bpb..][..n].copy_from_slice(&texels[src..src + n]);
            }
        }
    }
}

/// The inverse of [`tile`]: one layer's tiles back to tightly packed block rows.
pub fn untile(f: FormatInfo, w: u32, h: u32, tiles: &[u8]) -> Vec<u8> {
    let [tw, th] = tile_shape(f).map(|t| (t / f.block) as usize);
    let (bw, bh) = (w.div_ceil(f.block) as usize, h.div_ceil(f.block) as usize);
    let bpb = f.bytes as usize;
    let gx = bw.div_ceil(tw);
    let mut out = vec![0; bw * bh * bpb];
    for (y, row) in out.chunks_exact_mut(bw * bpb).enumerate() {
        for tx in 0..gx {
            let tile = &tiles[((y / th) * gx + tx) * TILE_BYTES..][..TILE_BYTES];
            let x0 = tx * tw;
            let n = tw.min(bw - x0) * bpb;
            row[x0 * bpb..][..n].copy_from_slice(&tile[(y % th) * tw * bpb..][..n]);
        }
    }
    out
}

/// The equal-area octahedral sphere mapping of environment maps (Clarberg 2008).
///
/// Map coordinates `(s, t)` cover `[0, 1]^2`; `t` grows with the texel row. The centre of
/// the map is +Y, the corners are -Y, and the four edge midpoints lie on the horizon. Texels
/// of an `n` x `n` map have equal solid angle `4 pi / n^2`. Across the map's edges, the
/// neighbour of texel `(-1, y)` is `(0, n - 1 - y)`, and likewise on the other three edges.
pub mod octahedral {
    use std::f32::consts::{FRAC_PI_4, PI};

    /// Direction of map point `[s, t]`.
    pub fn to_dir([s, t]: [f32; 2]) -> [f32; 3] {
        let (u, v) = (2.0 * s - 1.0, 2.0 * t - 1.0);
        let (up, vp) = (u.abs(), v.abs());
        let signed = 1.0 - (up + vp);
        let r = 1.0 - signed.abs();
        let phi = if r == 0.0 { 1.0 } else { (vp - up) / r + 1.0 } * FRAC_PI_4;
        let y = (1.0 - r * r).copysign(signed);
        let k = r * (2.0 - r * r).max(0.0).sqrt();
        [
            phi.cos().abs().copysign(u) * k,
            y,
            phi.sin().abs().copysign(v) * k,
        ]
    }

    /// Map point of unit direction `d`.
    pub fn from_dir(d: [f32; 3]) -> [f32; 2] {
        let (x, z, y) = (d[0].abs(), d[2].abs(), d[1].abs());
        let r = (1.0 - y).max(0.0).sqrt();
        let (a, b) = (x.max(z), x.min(z));
        let b = if a == 0.0 { 0.0 } else { b / a };
        let mut phi = b.atan() * 2.0 / PI;
        if x < z {
            phi = 1.0 - phi;
        }
        let mut v = phi * r;
        let mut u = r - v;
        if d[1] < 0.0 {
            (u, v) = (1.0 - v, 1.0 - u);
        }
        [
            0.5 * (u.copysign(d[0]) + 1.0),
            0.5 * (v.copysign(d[2]) + 1.0),
        ]
    }
}

/// Vulkan format numbers of the formats textures use.
pub mod vk {
    pub const R8_UNORM: u32 = 9;
    pub const R8G8_UNORM: u32 = 16;
    pub const R8G8B8A8_UNORM: u32 = 37;
    pub const R8G8B8A8_SRGB: u32 = 43;
    pub const R16_UNORM: u32 = 70;
    pub const R16_SFLOAT: u32 = 76;
    pub const R16G16_UNORM: u32 = 77;
    pub const R16G16_SFLOAT: u32 = 83;
    pub const R16G16B16A16_UNORM: u32 = 91;
    pub const R16G16B16A16_SFLOAT: u32 = 97;
    pub const R32_SFLOAT: u32 = 100;
    pub const R32G32_SFLOAT: u32 = 103;
    pub const R32G32B32A32_SFLOAT: u32 = 109;
    pub const BC1_RGB_UNORM: u32 = 131;
    pub const BC1_RGB_SRGB: u32 = 132;
    pub const BC1_RGBA_UNORM: u32 = 133;
    pub const BC1_RGBA_SRGB: u32 = 134;
    pub const BC2_UNORM: u32 = 135;
    pub const BC2_SRGB: u32 = 136;
    pub const BC3_UNORM: u32 = 137;
    pub const BC3_SRGB: u32 = 138;
    pub const BC4_UNORM: u32 = 139;
    pub const BC4_SNORM: u32 = 140;
    pub const BC5_UNORM: u32 = 141;
    pub const BC5_SNORM: u32 = 142;
    pub const BC6H_UFLOAT: u32 = 143;
    pub const BC6H_SFLOAT: u32 = 144;
    pub const BC7_UNORM: u32 = 145;
    pub const BC7_SRGB: u32 = 146;
}

const _: () = {
    assert!(size_of::<TextureHeader>() == 32);
    assert!(size_of::<ExtractedLight>() == 32);
};
