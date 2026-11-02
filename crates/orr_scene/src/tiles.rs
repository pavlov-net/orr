//! Streamed textures: sparse images whose 64 KiB tiles above the mip tail load on the
//! frames' texture feedback.
//!
//! A texture with mips above its tail streams. Its mip tails (one per layer, or one for the
//! whole image) are bound and filled at mount and never leave. Every tile above them is a
//! residency unit of the texture class that depends on the tile of the next coarser mip
//! covering it, so a resident tile always has its coarser tiles. Per tile of mip 0 in each
//! layer the residency map holds the finest mip resident there and at its eight neighbors;
//! sampling clamps its level of detail to it, and reports the level it wanted into the
//! frame's feedback cells. The render thread reads the feedback once the frame has
//! completed, requests the tiles the wanted levels miss and touches the ones they read.

use std::collections::BTreeMap;
use std::sync::Arc;

use orr_gpu::{
    Allocation, CommandList, Device, GpuPtr, Hazards, MemoryKind, SparseLayout, Stages, Texture,
    TextureIndex,
};

use crate::memory::{PoolSlot, RangeAllocator};
use crate::mount::Apply;
use crate::upload::Record;
use crate::{MountId, Result, TextureFeedbackGpu, TextureStreamGpu};

/// How a streamed texture's tiles are numbered: by layer, then mip finest first, then row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TileGrid {
    /// Tiles per axis of each mip above the tail.
    pub(crate) grids: Vec<[u32; 2]>,
    /// First tile of each mip within a layer, then the tiles of one layer.
    pub(crate) mip_first: Vec<u32>,
    pub(crate) layers: u32,
}

/// One tile of a streamed texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tile {
    pub(crate) mip: u32,
    pub(crate) layer: u32,
    pub(crate) xy: [u32; 2],
}

impl TileGrid {
    pub(crate) fn new(grids: Vec<[u32; 2]>, layers: u32) -> Self {
        let mut mip_first = vec![0];
        for g in &grids {
            mip_first.push(mip_first.last().unwrap() + g[0] * g[1]);
        }
        Self {
            grids,
            mip_first,
            layers,
        }
    }

    /// The mip the tail starts at.
    pub(crate) fn tail_mip(&self) -> u32 {
        self.grids.len() as u32
    }

    pub(crate) fn layer_tiles(&self) -> u32 {
        *self.mip_first.last().expect("a first tile per mip")
    }

    pub(crate) fn tile_count(&self) -> u32 {
        self.layer_tiles() * self.layers
    }

    /// Mip 0 tiles per axis: the residency map's cells per layer.
    pub(crate) fn cells(&self) -> [u32; 2] {
        self.grids[0]
    }

    pub(crate) fn tile(&self, t: u32) -> Tile {
        let (layer, r) = (t / self.layer_tiles(), t % self.layer_tiles());
        let mip = self.mip_first.partition_point(|&f| f <= r) as u32 - 1;
        let [gx, _] = self.grids[mip as usize];
        let i = r - self.mip_first[mip as usize];
        Tile {
            mip,
            layer,
            xy: [i % gx, i / gx],
        }
    }

    pub(crate) fn index_of(&self, tile: Tile) -> u32 {
        let [gx, _] = self.grids[tile.mip as usize];
        tile.layer * self.layer_tiles()
            + self.mip_first[tile.mip as usize]
            + tile.xy[1] * gx
            + tile.xy[0]
    }

    /// The tile of `mip` covering mip 0 tile `cell` of `layer`.
    pub(crate) fn covering(&self, layer: u32, mip: u32, cell: [u32; 2]) -> Tile {
        let [gx, gy] = self.grids[mip as usize];
        Tile {
            mip,
            layer,
            xy: [(cell[0] >> mip).min(gx - 1), (cell[1] >> mip).min(gy - 1)],
        }
    }

    /// The tile `t` depends on: the next coarser mip's tile covering it, none for the
    /// coarsest mip above the tail.
    pub(crate) fn parent(&self, t: u32) -> Option<u32> {
        let tile = self.tile(t);
        let mip = tile.mip + 1;
        (mip < self.tail_mip()).then(|| {
            let [gx, gy] = self.grids[mip as usize];
            self.index_of(Tile {
                mip,
                layer: tile.layer,
                xy: [(tile.xy[0] >> 1).min(gx - 1), (tile.xy[1] >> 1).min(gy - 1)],
            })
        })
    }

    /// The finest resident mip at each mip 0 tile of `layer` (`resident` by tile index),
    /// before the neighbors are taken in.
    pub(crate) fn finest(&self, resident: &[bool], layer: u32) -> Vec<u32> {
        let [cx, cy] = self.cells();
        let mut out = Vec::with_capacity((cx * cy) as usize);
        for y in 0..cy {
            for x in 0..cx {
                let mut m = self.tail_mip();
                while m > 0 && resident[self.index_of(self.covering(layer, m - 1, [x, y])) as usize]
                {
                    m -= 1;
                }
                out.push(m);
            }
        }
        out
    }
}

/// A streamed texture as its mount placed it. Shared by the render thread and the loading
/// side; it keeps the image and its tails' memory alive.
pub(crate) struct TexStream {
    pub(crate) texture: Arc<Texture>,
    pub(crate) index: TextureIndex,
    pub(crate) layout: SparseLayout,
    pub(crate) grid: TileGrid,
    /// The texture's archive entry and the blob of each mip above the tail.
    pub(crate) entry: u32,
    pub(crate) blobs: Vec<u32>,
    /// What kernels read: the record and its residency map.
    pub(crate) record: GpuPtr<TextureStreamGpu>,
    pub(crate) map: GpuPtr<u32>,
    /// Mip tails bound (one per layer, or one for the image) and their memory.
    pub(crate) tails: Option<(u32, Allocation<u8>)>,
}

/// The residency map of one layer from its finest resident mips: each cell takes the
/// coarsest of itself and its eight neighbors, so filtering across a tile edge reads only
/// resident tiles.
pub(crate) fn dilate(finest: &[u32], [cx, cy]: [u32; 2]) -> Vec<u32> {
    let at = |x: u32, y: u32| finest[(y * cx + x) as usize];
    let mut out = Vec::with_capacity(finest.len());
    for y in 0..cy {
        for x in 0..cx {
            let mut v = 0;
            for ny in y.saturating_sub(1)..(y + 2).min(cy) {
                for nx in x.saturating_sub(1)..(x + 2).min(cx) {
                    v = v.max(at(nx, ny));
                }
            }
            out.push(v);
        }
    }
    out
}

/// Which tile of a texture moves next so that its tiles sit in contiguous slots, and the slot
/// it should follow: a tile `movable` allows from the smallest other run of adjacent slots, to
/// the end of the largest run that has room after it. None once one run holds them, or when
/// nothing can move.
pub(crate) fn compaction_move(
    tiles: &[(u32, PoolSlot)],
    movable: impl Fn(u32) -> bool,
    room_after: impl Fn(PoolSlot) -> bool,
) -> Option<(u32, PoolSlot)> {
    let mut sorted: Vec<(u32, PoolSlot)> = tiles.to_vec();
    sorted.sort_by_key(|(_, s)| (s.block, s.first));
    let mut runs: Vec<&[(u32, PoolSlot)]> = sorted
        .chunk_by(|a, b| a.1.block == b.1.block && a.1.first + a.1.units == b.1.first)
        .collect();
    if runs.len() < 2 {
        return None;
    }
    runs.sort_by_key(|r| r.len());
    let at = runs.iter().rposition(|r| room_after(r[r.len() - 1].1))?;
    let target = runs[at].last()?.1;
    let from = runs
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != at)
        .flat_map(|(_, r)| r.iter().rev())
        .find(|(t, _)| movable(*t))?;
    Some((from.0, target))
}

/// The render thread's view of one streamed texture.
pub(crate) struct TexResidency {
    pub(crate) stream: Arc<TexStream>,
    /// Residency unit of tile 0; tile `t` is unit `first_unit + t`.
    pub(crate) first_unit: u32,
    pub(crate) resident: Vec<bool>,
    /// Finest resident mip per mip 0 tile, per layer.
    pub(crate) finest: Vec<u32>,
    pub(crate) first_cell: u32,
    pub(crate) dirty: bool,
}

impl TexResidency {
    /// Rewrites the residency map from `resident` when a tile came or went.
    pub(crate) fn write_map(&mut self, w: &mut Apply<'_>) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        self.dirty = false;
        let g = &self.stream.grid;
        let cells = g.cells();
        let mut map = Vec::with_capacity(self.finest.len());
        self.finest.clear();
        for layer in 0..g.layers {
            let finest = g.finest(&self.resident, layer);
            map.extend(dilate(&finest, cells));
            self.finest.extend(finest);
        }
        w.put(self.stream.map, &map)
    }
}

/// Words of one frame's feedback readback: the record count, then the records.
fn readback_words(capacity: u32) -> usize {
    1 + 2 * capacity as usize
}

/// Feedback the frame's texture taps write: per-cell wanted mips and the records of their
/// raises, copied back per frame in flight.
pub(crate) struct TextureFeedback {
    cells: Allocation<u32>,
    records: Allocation<[u32; 2]>,
    counts: Allocation<u32>,
    pub(crate) gpu: Allocation<TextureFeedbackGpu>,
    readback: Allocation<u32>,
    capacity: u32,
    cell_ranges: RangeAllocator,
    pub(crate) cell_capacity: u32,
    /// The texture each cell range belongs to, by first cell: its mount, index among the
    /// mount's streamed textures and cell count.
    owners: BTreeMap<u32, (MountId, u32, u32)>,
}

impl TextureFeedback {
    pub(crate) fn new(device: &Device, cells: u32, records: u32, fif: u32) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let fif = u64::from(fif.max(1));
        Ok(Self {
            cells: device.gpu_malloc(dl, u64::from(cells.max(1)), "texture feedback cells")?,
            records: device.gpu_malloc(
                dl,
                u64::from(records.max(1)) * fif,
                "texture feedback records",
            )?,
            counts: device.gpu_malloc(dl, fif, "texture feedback counts")?,
            gpu: device.gpu_malloc(dl, 1, "texture feedback")?,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                readback_words(records.max(1)) as u64 * fif,
                "texture feedback readback",
            )?,
            capacity: records.max(1),
            cell_ranges: RangeAllocator::new(cells),
            cell_capacity: cells,
            owners: BTreeMap::new(),
        })
    }

    /// The records that clear the cells, for the first frame.
    pub(crate) fn init(&self) -> Record {
        Record::fill(
            0,
            self.cells.gpu().address().0,
            self.cell_capacity.max(1),
            0,
        )
    }

    /// Cells for `count` cells of a texture.
    pub(crate) fn alloc(&mut self, owner: (MountId, u32), count: u32) -> Option<u32> {
        let first = self.cell_ranges.alloc(count)?;
        self.owners.insert(first, (owner.0, owner.1, count));
        Some(first)
    }

    pub(crate) fn free(&mut self, first: u32) {
        if let Some((_, _, count)) = self.owners.remove(&first) {
            self.cell_ranges.free(first, count);
        }
    }

    /// Starts `frame` in readback slot `slot`: its taps stamp the frame, its record count
    /// starts at 0, and the cells clear when the 24-bit frame stamp wraps.
    pub(crate) fn begin(&self, frame: u64, slot: usize, w: &mut Apply<'_>) -> Result<()> {
        let stamp = (frame & 0xff_ffff) as u32;
        if stamp == 0 {
            w.fill(self.cells.gpu(), self.cell_capacity.max(1), 0);
        }
        let record = TextureFeedbackGpu {
            cells: self.cells.gpu(),
            records: self.records.gpu(),
            counts: self.counts.gpu(),
            capacity: self.capacity,
            frame: stamp,
            slot: slot as u32,
            _pad: 0,
        };
        w.put(self.gpu.gpu(), &[record])?;
        w.put(self.counts.gpu().offset(slot as u64), &[0])
    }

    /// Copies slot `slot`'s count and records into its readback, after every tap of the
    /// frame on `list`.
    pub(crate) fn record_end(&self, list: &mut CommandList<'_>, slot: usize) {
        let words = readback_words(self.capacity);
        let dst = self.readback.gpu().offset((slot * words) as u64);
        list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
        list.copy(dst.cast(), self.counts.gpu().offset(slot as u64).cast(), 4);
        list.copy(
            dst.offset(1).cast(),
            self.records
                .gpu()
                .offset(slot as u64 * u64::from(self.capacity))
                .cast(),
            8 * u64::from(self.capacity),
        );
        list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    }

    /// The wanted mips frame `frame` reported from readback slot `slot`: the texture's
    /// mount and index, the cell's offset in the texture's cells, and the mip.
    pub(crate) fn read(&self, slot: usize, frame: u64) -> Vec<((MountId, u32), u32, u32)> {
        let words = readback_words(self.capacity);
        let data = &self.readback.as_slice().expect("readback is mapped")[slot * words..][..words];
        let n = data[0].min(self.capacity) as usize;
        let stamp = (frame & 0xff_ffff) as u32;
        let mut out = Vec::with_capacity(n);
        for r in data[1..1 + 2 * n].chunks_exact(2) {
            let (cell, value) = (r[0], r[1]);
            if value >> 8 != stamp {
                continue;
            }
            let Some((&first, &(mount, texture, count))) = self.owners.range(..=cell).next_back()
            else {
                continue;
            };
            if cell - first < count {
                out.push(((mount, texture), cell - first, 255 - (value & 255)));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_moves_the_smallest_runs_tiles_to_the_end_of_the_largest() {
        let slot = |block, first| PoolSlot {
            block,
            first,
            units: 1,
        };
        let tiles = [
            (0, slot(0, 0)),
            (1, slot(0, 1)),
            (2, slot(0, 2)),
            (3, slot(0, 5)),
            (4, slot(1, 0)),
            (5, slot(1, 1)),
        ];
        let room = |_| true;
        let (tile, after) = compaction_move(&tiles, |_| true, room).unwrap();
        assert_eq!(tile, 3, "the run of one moves first");
        assert_eq!(after, slot(0, 2));
        let (tile, _) = compaction_move(&tiles, |t| t != 3, room).unwrap();
        assert_eq!(tile, 5, "a run that cannot move leaves the next smallest");
        assert!(
            compaction_move(&tiles[..3], |_| true, room).is_none(),
            "one run"
        );
        assert!(
            compaction_move(&tiles, |t| t < 3, room).is_none(),
            "nothing movable"
        );
        let (tile, after) = compaction_move(&tiles, |_| true, |s| s != slot(0, 2)).unwrap();
        assert_eq!(tile, 3, "the next largest run with room takes it");
        assert_eq!(after, slot(1, 1));
        assert!(
            compaction_move(&tiles, |_| true, |_| false).is_none(),
            "no room anywhere"
        );
    }

    #[test]
    fn tiles_number_by_layer_mip_and_row_and_depend_on_the_coarser_tile() {
        let g = TileGrid::new(vec![[5, 3], [3, 2], [2, 1]], 2);
        assert_eq!(g.tile_count(), 2 * (15 + 6 + 2));
        for t in 0..g.tile_count() {
            assert_eq!(g.index_of(g.tile(t)), t);
        }
        let t = g.index_of(Tile {
            mip: 0,
            layer: 1,
            xy: [4, 2],
        });
        let p = g.tile(g.parent(t).unwrap());
        assert_eq!(
            p,
            Tile {
                mip: 1,
                layer: 1,
                xy: [2, 1]
            }
        );
        assert_eq!(g.tile(g.parent(g.index_of(p)).unwrap()).xy, [1, 0]);
        assert_eq!(
            g.parent(g.index_of(Tile {
                mip: 2,
                layer: 0,
                xy: [1, 0]
            })),
            None
        );
    }

    #[test]
    fn the_map_takes_the_finest_nested_mip_and_the_coarsest_neighbor() {
        let g = TileGrid::new(vec![[5, 3], [3, 2], [2, 1]], 2);
        let mut resident = vec![false; g.tile_count() as usize];
        let mut set = |mip, xy| resident[g.index_of(Tile { mip, layer: 1, xy }) as usize] = true;
        set(2, [0, 0]);
        set(1, [0, 0]);
        set(0, [1, 1]);
        assert_eq!(
            g.finest(&resident, 0),
            vec![3; 15],
            "layer 0 holds only its tail"
        );
        let finest = g.finest(&resident, 1);
        assert_eq!(
            finest,
            vec![1, 1, 2, 2, 3, 1, 0, 2, 2, 3, 2, 2, 2, 2, 3],
            "mip 0 (1, 1) sits under resident mip 1 and 2 tiles; mip 2 tile (1, 0) is absent"
        );
        let map = dilate(&finest, [5, 3]);
        assert_eq!(map, vec![1, 2, 2, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 3]);
    }
}
