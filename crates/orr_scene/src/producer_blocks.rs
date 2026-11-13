//! Producer entries' emitted clusters in blocks of [`BLOCK_CLUSTERS`], one workgroup each, for
//! kernels that weigh every cluster of a frame at once and gather per entry after
//! (`producer_blocks.slang`).

use orr_gpu::{Allocation, Device, DispatchArgs, GpuPtr, MemoryKind};

use crate::{LightTables, Result};

/// Clusters per block (`producer_blocks.slang` `BLOCK_CLUSTERS`).
pub const BLOCK_CLUSTERS: u32 = 256;

/// The layout's words before its first entry's (`producer_blocks.slang` `BLOCKS_FIRST`).
const BLOCKS_FIRST: u64 = 4;

/// A layout of producer entries' clusters in blocks, which a kernel writes each frame, and a
/// record of `T` per block, with room for what a scene's light tables can emit.
pub struct ClusterBlocks<T: 'static> {
    layout: Allocation<u32>,
    records: Allocation<T>,
    producers: u32,
    blocks: u64,
}

/// The blocks the producer entries of `tables` fill at most: their clusters' share, and a block
/// each that an entry ends in part.
fn max_blocks(tables: LightTables) -> u64 {
    u64::from(tables.producer_clusters).div_ceil(u64::from(BLOCK_CLUSTERS))
        + u64::from(tables.producers)
}

impl<T> ClusterBlocks<T> {
    /// Room for light tables `tables`. Fails without device memory.
    pub fn new(device: &Device, tables: LightTables, name: &str) -> Result<Self> {
        let blocks = max_blocks(tables);
        Ok(Self {
            layout: device.gpu_malloc(
                MemoryKind::DeviceLocal,
                BLOCKS_FIRST + u64::from(tables.producers) + 1,
                name,
            )?,
            records: device.gpu_malloc(MemoryKind::DeviceLocal, blocks.max(1), name)?,
            producers: tables.producers,
            blocks,
        })
    }

    /// Whether it has room for light tables `tables`.
    pub fn holds(&self, tables: LightTables) -> bool {
        self.producers >= tables.producers && self.blocks >= max_blocks(tables)
    }

    /// The layout: the workgroup counts that run one per block, then each entry's first block.
    pub fn layout(&self) -> GpuPtr<u32> {
        self.layout.gpu()
    }

    /// The workgroup counts at the layout's start, for an indirect dispatch over the blocks.
    pub fn groups(&self) -> GpuPtr<DispatchArgs> {
        self.layout.gpu().cast()
    }

    pub fn records(&self) -> GpuPtr<T> {
        self.records.gpu()
    }
}
