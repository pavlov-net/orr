//! Raster set class: tiled, bordered LOD pyramids of one or more layers over a square of
//! the world. Blob 0 holds a [`RasterSetHeader`], the layer table, the layer names and the
//! tile table ([`RasterSetInfo`]); every later blob is one tile.
//!
//! - LOD 0 is the coarsest: one tile. LOD `l` has `2^l` tiles per axis, and its level is
//!   `tile_size << l` texels per axis. Each coarser level was built from the next finer one
//!   by the layer's reduction over 2x2 texel blocks.
//! - A tile holds `tile_size + 2 * border` texels per axis: tile `(x, y)` covers level
//!   texels `x * tile_size - border` to `(x + 1) * tile_size + border`, so border texels
//!   repeat the neighbours' edge texels, and outside the level the nearest edge texel
//!   repeats. Texels (or BC blocks) are row-major and tightly packed, so a tile copies into
//!   an image like a texture mip.
//! - Tiles are sorted by layer, LOD and the Morton code of `(x, y)`, and stored in that
//!   order. A set may omit tiles.
//! - Lossless layers store the input bits: decoding returns them unchanged.

use bytemuck::{Pod, Zeroable};

use crate::blob::{put_table, read, table};
use crate::texture::FormatInfo;
use crate::{Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Blob 0 header.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct RasterSetHeader {
    /// World x and z of the set's corner at texel `(0, 0)` of every level, in m.
    pub world_min: [f64; 2],
    /// Edge of the square the set covers, in m.
    pub world_size: f32,
    pub _pad: u32,
    /// [`RasterLayer`] table; spans are byte offsets in blob 0.
    pub layers: Span,
    /// Layer names, UTF-8, each layer's `name` a range of these bytes.
    pub names: Span,
    /// [`TileKey`] table.
    pub tiles: Span,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct RasterLayer {
    /// Byte range within the header's `names`.
    pub name: Span,
    /// Vulkan format number (the [`crate::texture::vk`] constants).
    pub vk_format: u32,
    /// Texels per tile and axis, borders excluded.
    pub tile_size: u32,
    pub border: u16,
    pub lod_count: u8,
    /// 1 when tiles hold the input bits; 0 for BCn layers.
    pub lossless: u8,
    /// How coarser LODs were built: [`Self::MEAN`], [`Self::MIN`], [`Self::MAX`] or
    /// [`Self::NEAREST`].
    pub reduction: u8,
    pub _pad: [u8; 3],
    /// Smallest and largest finite channel value over the input texels, as a shader reads
    /// them (UNORM normalised to 0-1); `[inf, -inf]` when there is none.
    pub value_range: [f32; 2],
}

impl RasterLayer {
    /// Average of the 2x2 block; for f32, `(a + b + c + d) * 0.25` in that order with `a`
    /// the top-left texel, `b` its right, `c` below, `d` diagonal.
    pub const MEAN: u8 = 0;
    pub const MIN: u8 = 1;
    pub const MAX: u8 = 2;
    /// The block's top-left texel.
    pub const NEAREST: u8 = 3;

    pub fn format(&self) -> Result<FormatInfo> {
        FormatInfo::of(self.vk_format).ok_or(Error::Malformed("raster layer format"))
    }

    /// Texels per tile and axis, borders included.
    pub fn tile_texels(&self) -> u32 {
        self.tile_size + 2 * u32::from(self.border)
    }

    /// Decoded bytes of one tile.
    pub fn tile_bytes(&self) -> usize {
        let t = self.tile_texels();
        self.format().map_or(0, |f| f.mip_bytes(t, t))
    }

    /// Texels per axis of the level at `lod`.
    pub fn level_size(&self, lod: u32) -> u32 {
        self.tile_size << lod
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct TileKey {
    pub layer: u16,
    pub lod: u8,
    pub _pad: u8,
    pub x: u32,
    pub y: u32,
    /// The entry blob holding the tile.
    pub blob: u32,
}

impl TileKey {
    /// Table order: layer, LOD, then the Morton code of the tile.
    pub fn order(&self) -> (u16, u8, u64) {
        (self.layer, self.lod, morton(self.x, self.y))
    }
}

/// Interleaves `x` (even bits) and `y` (odd bits).
pub fn morton(x: u32, y: u32) -> u64 {
    fn spread(v: u32) -> u64 {
        let mut v = u64::from(v);
        v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
        v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        (v | (v << 1)) & 0x5555_5555_5555_5555
    }
    spread(x) | (spread(y) << 1)
}

/// A raster set blob 0.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterSetInfo {
    pub header: RasterSetHeader,
    pub layers: Vec<RasterLayer>,
    /// Name of each layer.
    pub names: Vec<String>,
    pub tiles: Vec<TileKey>,
}

impl RasterSetInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let header: RasterSetHeader = read(blob, 0).ok_or(Error::Malformed("raster set header"))?;
        let layers: Vec<RasterLayer> = table(blob, header.layers)?;
        let name_bytes: Vec<u8> = table(blob, header.names)?;
        let tiles: Vec<TileKey> = table(blob, header.tiles)?;
        let mut names = Vec::with_capacity(layers.len());
        for l in &layers {
            l.format()?;
            if l.tile_size == 0 || l.lod_count == 0 || l.lod_count > 24 || l.reduction > 3 {
                return Err(Error::Malformed("raster layer"));
            }
            let at = l.name.first as usize;
            let name = at
                .checked_add(l.name.count as usize)
                .and_then(|end| name_bytes.get(at..end))
                .and_then(|b| std::str::from_utf8(b).ok())
                .ok_or(Error::Malformed("raster layer name"))?;
            names.push(name.to_owned());
        }
        for (i, t) in tiles.iter().enumerate() {
            let l = layers
                .get(usize::from(t.layer))
                .ok_or(Error::Malformed("raster tile layer"))?;
            let n = 1u32 << t.lod;
            if t.lod >= l.lod_count || t.x >= n || t.y >= n || t.blob == 0 {
                return Err(Error::Malformed("raster tile key"));
            }
            if i > 0 && tiles[i - 1].order() >= t.order() {
                return Err(Error::Malformed("raster tiles out of order"));
            }
        }
        Ok(Self {
            header,
            layers,
            names,
            tiles,
        })
    }

    /// Blob bytes; the header's spans and each layer's name span are set from the tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut layers = self.layers.clone();
        let mut names = Vec::new();
        for (l, n) in layers.iter_mut().zip(&self.names) {
            l.name = Span {
                first: names.len() as u32,
                count: n.len() as u32,
            };
            names.extend_from_slice(n.as_bytes());
        }
        let mut out = vec![0; size_of::<RasterSetHeader>()];
        h.layers = put_table(&mut out, &layers);
        h.names = put_table(&mut out, &names);
        h.tiles = put_table(&mut out, &self.tiles);
        out[..size_of::<RasterSetHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }

    /// Index of the layer named `name`.
    pub fn layer(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// The tile at `(x, y)` of `lod` in `layer`, if the set holds it.
    pub fn tile(&self, layer: u16, lod: u8, x: u32, y: u32) -> Option<&TileKey> {
        let key = (layer, lod, morton(x, y));
        self.tiles
            .binary_search_by(|t| t.order().cmp(&key))
            .ok()
            .map(|i| &self.tiles[i])
    }
}

const _: () = {
    assert!(size_of::<RasterSetHeader>() == 48);
    assert!(size_of::<RasterLayer>() == 32);
    assert!(size_of::<TileKey>() == 16);
};
