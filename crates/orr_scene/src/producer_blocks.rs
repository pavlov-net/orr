//! Lists of clusters, one per entry of a table, in blocks of [`BLOCK_CLUSTERS`], one workgroup
//! each, for kernels that weigh every cluster of a frame at once and gather per entry after
//! (`producer_blocks.slang`): the producer entries' emitted clusters, or the emissive list's
//! caster lists.

use orr_gpu::{Allocation, Device, DispatchArgs, GpuPtr, MemoryKind};

use crate::Result;

/// Clusters per block (`producer_blocks.slang` `BLOCK_CLUSTERS`).
pub const BLOCK_CLUSTERS: u32 = 256;

/// The layout's words before its first entry's (`producer_blocks.slang` `BLOCKS_FIRST`).
const BLOCKS_FIRST: u64 = 4;

/// A layout of the lists of a table's entries in blocks, which a kernel writes each frame, and a
/// record of `T` per block, with room for a number of entries and of clusters they hold.
pub struct ClusterBlocks<T: 'static> {
    layout: Allocation<u32>,
    records: Allocation<T>,
    lists: u32,
    blocks: u64,
}

/// The blocks `lists` lists of `clusters` clusters between them fill at most: the clusters'
/// share, and a block each that a list ends in part.
fn max_blocks(lists: u32, clusters: u32) -> u64 {
    u64::from(clusters).div_ceil(u64::from(BLOCK_CLUSTERS)) + u64::from(lists)
}

impl<T> ClusterBlocks<T> {
    /// Room for `lists` lists of `clusters` clusters between them. Fails without device memory.
    pub fn new(device: &Device, lists: u32, clusters: u32, name: &str) -> Result<Self> {
        let blocks = max_blocks(lists, clusters);
        Ok(Self {
            layout: device.gpu_malloc(
                MemoryKind::DeviceLocal,
                BLOCKS_FIRST + u64::from(lists) + 1,
                name,
            )?,
            records: device.gpu_malloc(MemoryKind::DeviceLocal, blocks.max(1), name)?,
            lists,
            blocks,
        })
    }

    /// Whether it has room for `lists` lists of `clusters` clusters between them.
    pub fn holds(&self, lists: u32, clusters: u32) -> bool {
        self.lists >= lists && self.blocks >= max_blocks(lists, clusters)
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

    /// The blocks it holds a record for.
    pub fn block_count(&self) -> u64 {
        self.blocks
    }
}
