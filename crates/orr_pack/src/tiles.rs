//! Texture tiles class: one blob holding a [`TextureTiles`], the textures a material slot reads
//! across uv space, one to each unit square, as a UDIM set lays them out. Each tile is an ordinary
//! texture, of its own size and mips, that streams on its own. All tiles of a set have the same
//! primaries, which the material slot holds for all of them; a mount fails on a set that mixes them.
//!
//! Tile (`column`, `row`) covers u from `column` to `column + 1` and v from `-row` to `1 - row`:
//! uvs have a top-left origin and rows count up the image, so row 1 lies above row 0.
//!
//! A slot finds a tile through a hash of its key `row << 16 | column`, so a table's size follows
//! the tiles a set has and not how far apart their names lie. Bucket `(key * multiplier) >> shift`
//! holds up to [`BUCKET`] entries of a key and a heap index, one cache line in all, which a tap
//! reads whole. The cook decides where each tile goes, so a mount only fills in heap indices.

use std::collections::HashSet;

use bytemuck::{Pod, Zeroable};

use crate::blob::{put_table, read, table};
use crate::{AssetId, Error, Result, Span};

pub const SCHEMA: u16 = 1;

/// Entries of a bucket.
pub const BUCKET: u32 = 8;

/// Words of a table before its first bucket: the multiplier, the shift, then padding to a cache
/// line, on which every bucket starts.
pub const HEADER_WORDS: usize = 16;

/// Seeds a size of table tries before it doubles.
const TRIES: u32 = 64;

/// One tile of a set.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Tile {
    pub texture: AssetId,
    pub column: u16,
    pub row: u16,
    /// Its entry in the table: `bucket * BUCKET` and its place in the bucket.
    pub entry: u32,
}

/// Blob header; the span is a byte offset in the blob and a count of [`Tile`]s.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
struct Header {
    multiplier: u32,
    shift: u32,
    buckets: u32,
    tiles: Span,
}

/// A tile set's blob.
#[derive(Clone, Debug, PartialEq)]
pub struct TextureTiles {
    /// 0 for one bucket, which every key reaches, and otherwise odd.
    pub multiplier: u32,
    /// 32 less the log2 of the buckets: below 32, as a shift has to be.
    pub shift: u32,
    pub buckets: u32,
    pub tiles: Vec<Tile>,
}

fn key(column: u16, row: u16) -> u32 {
    u32::from(row) << 16 | u32::from(column)
}

fn hashed(key: u32, multiplier: u32, shift: u32) -> u32 {
    key.wrapping_mul(multiplier) >> shift
}

impl TextureTiles {
    /// The set of `tiles`, each a column, a row and a texture, in distinct cells: its table at a
    /// load of about 35%, in the first of a fixed run of multipliers that leaves no bucket over
    /// [`BUCKET`] keys, so the same tiles lay out the same.
    pub fn new(tiles: &[(u16, u16, AssetId)]) -> Self {
        let mut buckets = (tiles.len().div_ceil(3).next_power_of_two()) as u32;
        loop {
            if buckets == 1 {
                if let Some(set) = Self::placed(tiles, 0, 0, 1) {
                    return set;
                }
                buckets = 2;
            }
            let shift = 32 - buckets.trailing_zeros();
            let mut multiplier = 0x9e37_79b1u32;
            for _ in 0..TRIES {
                if let Some(set) = Self::placed(tiles, multiplier, shift, buckets) {
                    return set;
                }
                multiplier = (multiplier ^ multiplier >> 15).wrapping_mul(0x85eb_ca77) | 1;
            }
            buckets *= 2;
        }
    }

    /// `tiles` in `buckets` buckets, unless one holds too many.
    fn placed(
        tiles: &[(u16, u16, AssetId)],
        multiplier: u32,
        shift: u32,
        buckets: u32,
    ) -> Option<Self> {
        let mut used = vec![0; buckets as usize];
        let mut placed = Vec::with_capacity(tiles.len());
        for &(column, row, texture) in tiles {
            let bucket = hashed(key(column, row), multiplier, shift);
            let held = &mut used[bucket as usize];
            if *held == BUCKET {
                return None;
            }
            placed.push(Tile {
                texture,
                column,
                row,
                entry: bucket * BUCKET + *held,
            });
            *held += 1;
        }
        Some(Self {
            multiplier,
            shift,
            buckets,
            tiles: placed,
        })
    }

    pub fn parse(blob: &[u8]) -> Result<Self> {
        let h: Header = read(blob, 0).ok_or(Error::Malformed("texture tiles header"))?;
        let tiles: Vec<Tile> = table(blob, h.tiles)?;
        let set = Self {
            multiplier: h.multiplier,
            shift: h.shift,
            buckets: h.buckets,
            tiles,
        };
        if !set.well_formed() {
            return Err(Error::Malformed("texture tiles"));
        }
        Ok(set)
    }

    /// Whether every tile stands where a tap looks for it, and no shift or entry would reach past
    /// a word.
    fn well_formed(&self) -> bool {
        let hash = match self.buckets {
            1 => self.multiplier == 0 && self.shift == 0,
            b => {
                b.is_power_of_two()
                    && b <= u32::MAX / BUCKET
                    && self.multiplier % 2 == 1
                    && self.shift == 32 - b.trailing_zeros()
            }
        };
        let cells: HashSet<_> = self.tiles.iter().map(|t| (t.column, t.row)).collect();
        let entries: HashSet<_> = self.tiles.iter().map(|t| t.entry).collect();
        hash && !self.tiles.is_empty()
            && cells.len() == self.tiles.len()
            && entries.len() == self.tiles.len()
            && self.tiles.iter().all(|t| {
                t.texture != AssetId::NONE && t.entry / BUCKET == self.bucket(t.column, t.row)
            })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![0; size_of::<Header>()];
        let header = Header {
            multiplier: self.multiplier,
            shift: self.shift,
            buckets: self.buckets,
            tiles: put_table(&mut out, &self.tiles),
        };
        out[..size_of::<Header>()].copy_from_slice(bytemuck::bytes_of(&header));
        out
    }

    /// The bucket a tap of cell (`column`, `row`) reads.
    pub fn bucket(&self, column: u16, row: u16) -> u32 {
        hashed(key(column, row), self.multiplier, self.shift)
    }

    /// Words of the table a kernel reads.
    pub fn words(&self) -> usize {
        HEADER_WORDS + 2 * (self.buckets * BUCKET) as usize
    }

    /// The table a kernel reads, each entry the key and heap index of a tile, with `index` giving
    /// a texture's heap index, or the texture that has none. Empty entries hold 0, which is no
    /// heap index.
    pub fn table(
        &self,
        mut index: impl FnMut(AssetId) -> Option<u32>,
    ) -> std::result::Result<Vec<u32>, AssetId> {
        let mut words = vec![0; self.words()];
        words[..2].copy_from_slice(&[self.multiplier, self.shift]);
        for t in &self.tiles {
            let at = HEADER_WORDS + 2 * t.entry as usize;
            words[at] = key(t.column, t.row);
            words[at + 1] = index(t.texture).ok_or(t.texture)?;
        }
        Ok(words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(cells: impl Iterator<Item = (u16, u16)>) -> TextureTiles {
        let tiles: Vec<_> = cells
            .enumerate()
            .map(|(i, (u, v))| (u, v, AssetId::from_name(&i.to_string())))
            .collect();
        TextureTiles::new(&tiles)
    }

    /// A table holds every tile, in the bucket a tap of its cell reads, and survives its blob.
    fn check(s: &TextureTiles) {
        assert_eq!(TextureTiles::parse(&s.to_bytes()).unwrap(), *s);
        let words = s.table(|_| Some(7)).unwrap();
        assert_eq!(words.len(), s.words());
        for t in &s.tiles {
            let bucket = hashed(key(t.column, t.row), words[0], words[1]);
            let first = HEADER_WORDS + 2 * (bucket * BUCKET) as usize;
            let entries = words[first..first + 2 * BUCKET as usize].chunks_exact(2);
            let found = entries.filter(|e| e[0] == key(t.column, t.row) && e[1] == 7);
            assert_eq!(found.count(), 1, "{t:?}");
        }
    }

    #[test]
    fn tables_follow_the_tiles_a_set_has() {
        let udim = set((0..100).map(|i| (i % 10, i / 10)));
        check(&udim);
        assert_eq!(udim.buckets, 64);
        assert_eq!(4 * udim.words(), 4160, "bytes of a full UDIM set");
        let far = set([(0, 0), (39_999, 39_999)].into_iter());
        check(&far);
        assert_eq!(4 * far.words(), 128, "bytes of a far tile beside the first");
        check(&set(
            [(u16::MAX, u16::MAX), (0, u16::MAX), (u16::MAX, 0)].into_iter()
        ));
        // Keys that differ only high up, which a multiplier has to carry down.
        check(&set((0..2000).map(|i| (0, i * 32))));
        check(&set((0..2000).map(|i| (i * 32, i * 32))));
    }

    #[test]
    fn a_tile_where_no_tap_looks_is_malformed() {
        let good = set([(0, 0), (1, 0), (2, 1), (3, 1)].into_iter());
        check(&good);
        let read = |s: &TextureTiles| TextureTiles::parse(&s.to_bytes());
        let mut moved = good.clone();
        moved.tiles[0].entry = (moved.tiles[0].entry + BUCKET) % (good.buckets * BUCKET);
        assert!(read(&moved).is_err());
        let mut twice = good.clone();
        twice.tiles[1].column = twice.tiles[0].column;
        twice.tiles[1].row = twice.tiles[0].row;
        assert!(read(&twice).is_err());
        let mut even = good.clone();
        even.multiplier &= !1;
        assert!(read(&even).is_err());
        let mut wide = good.clone();
        wide.shift = 32;
        assert!(read(&wide).is_err());
        let mut huge = good;
        huge.buckets = 1 << 29;
        huge.shift = 3;
        for t in &mut huge.tiles {
            t.entry = hashed(key(t.column, t.row), huge.multiplier, huge.shift) * BUCKET;
        }
        assert!(read(&huge).is_err());
    }
}
