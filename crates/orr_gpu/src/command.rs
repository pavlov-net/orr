//! Records kernels and indirect commands read from GPU memory, alike on every backend.

use bytemuck::{Pod, Zeroable};

use crate::{GpuAddress, TextureRegion};

/// One GDeflate region for [`CommandList::decompress`](crate::CommandList::decompress): device addresses and byte counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DecompressRegion {
    pub src: u64,
    pub dst: u64,
    pub compressed_bytes: u64,
    pub decompressed_bytes: u64,
}

/// Indirect dispatch arguments (`vkCmdDispatchIndirect`'s): workgroup counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DispatchArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Indirect mesh-draw arguments (`vkCmdDrawMeshTasksIndirectEXT`'s): task workgroup counts, or mesh workgroup
/// counts for pipelines without a task stage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MeshArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// One region of [`CommandList::copy_indirect`](crate::CommandList::copy_indirect)
/// (`VkCopyMemoryIndirectCommandKHR`, 24 B):
/// device addresses and a byte count, each a multiple of 4. A region of 0 bytes copies
/// nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MemoryCopy {
    pub src: u64,
    pub dst: u64,
    pub bytes: u64,
}

/// One region of
/// [`CommandList::copy_to_texture_indirect`](crate::CommandList::copy_to_texture_indirect)
/// (`VkCopyMemoryToImageIndirectCommandKHR`, 56 B). Kernels write it with the `copy`
/// module's `store`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct TextureCopy {
    src: u64,
    row_texels: u32,
    image_rows: u32,
    aspect: u32,
    mip: u32,
    base_layer: u32,
    layers: u32,
    offset: [u32; 3],
    extent: [u32; 3],
}

impl TextureCopy {
    /// Texels from `src`, whose rows start `row_texels` texels apart (0 packs them tightly),
    /// into `region` of a color texture. The region's extent is not empty.
    pub fn new(src: GpuAddress, region: &TextureRegion, row_texels: u32) -> Self {
        Self {
            src: src.0,
            row_texels,
            image_rows: 0,
            // VK_IMAGE_ASPECT_COLOR_BIT
            aspect: 1,
            mip: region.mip,
            base_layer: region.base_layer,
            layers: region.layers,
            offset: region.offset,
            extent: region.extent,
        }
    }
}
