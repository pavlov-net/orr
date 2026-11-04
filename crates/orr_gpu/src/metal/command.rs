use std::any::Any;

use bytemuck::{Pod, Zeroable};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTL4RenderCommandEncoder, MTL4VisibilityOptions,
    MTLBuffer, MTLComputePipelineState, MTLDevice, MTLSize, MTLStages,
};

use super::memory::{Allocation, MemoryKind, malloc};
use super::pipeline::ComputePipeline;
use super::queue::Queue;
use super::texture::Texture;
use super::unsupported::RayDispatchPipeline;
use super::{lock, not_implemented};
use crate::{Error, GpuAddress, GpuPtr, Hazards, Stages, TextureLayers, TextureRegion};

/// Bytes of each block inline roots are copied into.
const INLINE_BLOCK: u64 = 64 << 10;

/// Indirect dispatch arguments: workgroup counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DispatchArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Indirect mesh-draw arguments: task workgroup counts, or mesh workgroup counts for
/// pipelines without a task stage.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MeshArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// One GDeflate region for [`CommandList::decompress`]: device addresses and byte counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DecompressRegion {
    pub src: u64,
    pub dst: u64,
    pub compressed_bytes: u64,
    pub decompressed_bytes: u64,
}

/// One region of [`CommandList::copy_indirect`]: device addresses and a byte count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MemoryCopy {
    pub src: u64,
    pub dst: u64,
    pub bytes: u64,
}

/// One region of [`CommandList::copy_to_texture_indirect`], laid out as the Vulkan
/// backend's.
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
            aspect: 1,
            mip: region.mip,
            base_layer: region.base_layer,
            layers: region.layers,
            offset: region.offset,
            extent: region.extent,
        }
    }
}

type ComputeEncoder = ProtocolObject<dyn MTL4ComputeCommandEncoder>;
type RenderEncoder = ProtocolObject<dyn MTL4RenderCommandEncoder>;

/// The encoder commands go into now.
pub(super) enum Encoder {
    None,
    Compute(Retained<ComputeEncoder>),
    Render(Retained<RenderEncoder>),
}

/// What a list's commands use until its submit completes.
pub(super) struct Finished {
    pub(super) buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    _allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    _table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    _inline: Vec<Allocation<u8>>,
    _keep: Vec<Box<dyn Any + Send>>,
}

// SAFETY: a finished list is only dropped, on whichever thread collects garbage; Metal command
// buffers, allocators and argument tables may be released from any thread once no encoding is
// under way.
unsafe impl Send for Finished {}

/// A one-shot command list for one queue. Record on any thread, then hand it to the queue's
/// owner for [`Queue::submit`]. Dropping it unsubmitted discards it.
pub struct CommandList<'q> {
    pub(super) queue: &'q Queue,
    pub(super) buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    /// Buffer 0 holds the root, 1 the texture view pool's base ID, 2 the sampler IDs.
    pub(super) table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    pub(super) encoder: Encoder,
    /// Stages the next encoder waits for, and its stages that wait: a queue barrier.
    pending: Option<(MTLStages, MTLStages)>,
    inline: Vec<Allocation<u8>>,
    /// Bytes used of the last inline block.
    inline_used: u64,
    pub(super) keep: Vec<Box<dyn Any + Send>>,
    /// Set by `finish`, so dropping does not end the buffer twice.
    finished: bool,
}

// SAFETY: one thread at a time records a list: `CommandList` is not `Sync` and every method
// takes `&mut self`; Metal command buffers and encoders may move between threads.
unsafe impl Send for CommandList<'_> {}

impl<'q> CommandList<'q> {
    pub(super) fn new(queue: &'q Queue) -> Result<Self, Error> {
        let shared = &queue.shared;
        let fail = |call: &'static str| Error::Metal {
            call,
            message: format!("{:?} queue command list", queue.kind),
        };
        let allocator = shared
            .device
            .newCommandAllocator()
            .ok_or_else(|| fail("newCommandAllocator"))?;
        let buffer = shared
            .device
            .newCommandBuffer()
            .ok_or_else(|| fail("newCommandBuffer"))?;
        buffer.beginCommandBufferWithAllocator(&allocator);
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(3);
        let table = shared
            .device
            .newArgumentTableWithDescriptor_error(&desc)
            .map_err(|e| super::device::metal_error("newArgumentTableWithDescriptor", &e))?;
        let b = &shared.bindings;
        // SAFETY: both addresses are of live device buffers the residency set holds.
        unsafe {
            table.setAddress_atIndex(b.pool_base.gpuAddress(), 1);
            table.setAddress_atIndex(b.samplers.gpuAddress(), 2);
        }
        Ok(Self {
            queue,
            buffer,
            allocator,
            table,
            encoder: Encoder::None,
            pending: None,
            inline: Vec::new(),
            inline_used: INLINE_BLOCK,
            keep: Vec::new(),
            finished: false,
        })
    }

    /// Ends recording; the queue commits the buffer and keeps the rest until it completes.
    pub(super) fn finish(mut self) -> Finished {
        self.end_encoder();
        self.buffer.endCommandBuffer();
        self.finished = true;
        Finished {
            buffer: self.buffer.clone(),
            _allocator: self.allocator.clone(),
            _table: self.table.clone(),
            _inline: std::mem::take(&mut self.inline),
            _keep: std::mem::take(&mut self.keep),
        }
    }

    pub(super) fn end_encoder(&mut self) {
        match std::mem::replace(&mut self.encoder, Encoder::None) {
            Encoder::None => {}
            Encoder::Compute(e) => e.endEncoding(),
            Encoder::Render(e) => e.endEncoding(),
        }
    }

    /// Applies the pending queue barrier to a new encoder.
    pub(super) fn open_barrier(&mut self, encoder: &ProtocolObject<dyn MTL4CommandEncoder>) {
        if let Some((after_queue, before)) = self.pending.take() {
            encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                after_queue,
                before,
                MTL4VisibilityOptions::Device,
            );
        }
    }

    /// The compute encoder, opened if another or none is open.
    pub(super) fn compute(&mut self) -> Retained<ComputeEncoder> {
        if let Encoder::Compute(e) = &self.encoder {
            return e.clone();
        }
        self.end_encoder();
        let e = self
            .buffer
            .computeCommandEncoder()
            .expect("computeCommandEncoder");
        self.open_barrier(ProtocolObject::from_ref(&*e));
        e.setArgumentTable(Some(&self.table));
        self.encoder = Encoder::Compute(e.clone());
        e
    }

    /// The device address of a copy of `bytes`, for an inline root.
    fn inline_bytes(&mut self, bytes: &[u8]) -> u64 {
        let len = (bytes.len() as u64).next_multiple_of(16);
        if self.inline_used + len > INLINE_BLOCK {
            let block = malloc::<u8>(
                &self.queue.shared,
                MemoryKind::Upload,
                INLINE_BLOCK,
                "inline roots",
            )
            .expect("inline root block");
            self.inline.push(block);
            self.inline_used = 0;
        }
        let block = self.inline.last_mut().expect("an inline block");
        let at = self.inline_used as usize;
        block.as_mut_slice().expect("mapped")[at..at + bytes.len()].copy_from_slice(bytes);
        self.inline_used += len;
        block.gpu().address().0 + at as u64
    }

    /// Makes work of `before` stages, and its writes, visible to work of `after` stages
    /// recorded later on this queue.
    pub fn barrier(&mut self, before: Stages, after: Stages, hazards: Hazards) {
        let _ = hazards;
        let (b, a) = (mtl_stages(before), mtl_stages(after));
        if b.is_empty() || a.is_empty() {
            return;
        }
        self.end_encoder();
        let (pb, pa) = self
            .pending
            .unwrap_or((MTLStages::empty(), MTLStages::empty()));
        self.pending = Some((pb | b, pa | a));
    }

    /// Nothing on Metal: textures have no layouts.
    pub fn discard(&mut self, t: &Texture) {
        let _ = t;
    }

    /// Nothing on Metal: textures have no layouts.
    pub fn discard_after_use(&mut self, t: &Texture) {
        let _ = t;
    }

    /// Runs `p` over `groups` workgroups with the root at `root`.
    pub fn dispatch<R>(&mut self, p: &ComputePipeline, root: GpuPtr<R>, groups: [u32; 3]) {
        root_in_memory(p);
        self.dispatch_at(p, root.address().0, groups);
    }

    /// Runs `p`, whose kernel holds its root inline (`ORR_ROOT_INLINE`), with `root`.
    pub fn dispatch_inline<R: Pod>(&mut self, p: &ComputePipeline, root: &R, groups: [u32; 3]) {
        root_inline::<R>(p);
        let address = self.inline_bytes(bytemuck::bytes_of(root));
        self.dispatch_at(p, address, groups);
    }

    fn dispatch_at(&mut self, p: &ComputePipeline, root: u64, groups: [u32; 3]) {
        let e = self.bind_compute(p, root);
        e.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(p.workgroup()));
    }

    fn bind_compute(&mut self, p: &ComputePipeline, root: u64) -> Retained<ComputeEncoder> {
        let e = self.compute();
        e.setComputePipelineState(&p.raw);
        self.keep.push(Box::new(p.raw.clone()));
        // SAFETY: the root's address is the caller's to keep valid until the submit
        // completes, as on every backend.
        unsafe { self.table.setAddress_atIndex(root, 0) };
        e.setArgumentTable(Some(&self.table));
        e
    }

    /// Runs one of orr_gpu's own pipeline states over `groups` workgroups of `workgroup`
    /// threads, with a copy of `root` as its root.
    pub(super) fn dispatch_state(
        &mut self,
        state: &super::Obj<ProtocolObject<dyn MTLComputePipelineState>>,
        root: &[u8],
        groups: [u32; 3],
        workgroup: [u32; 3],
    ) {
        let address = self.inline_bytes(root);
        let e = self.compute();
        e.setComputePipelineState(state);
        // SAFETY: the root copy lives in the list's inline blocks until the submit completes.
        unsafe { self.table.setAddress_atIndex(address, 0) };
        e.setArgumentTable(Some(&self.table));
        e.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(workgroup));
        self.keep.push(Box::new(state.clone()));
    }

    /// Runs `p` over the workgroup counts the GPU reads from `args`.
    pub fn dispatch_indirect<R>(
        &mut self,
        p: &ComputePipeline,
        root: GpuPtr<R>,
        args: GpuPtr<DispatchArgs>,
    ) {
        root_in_memory(p);
        let e = self.bind_compute(p, root.address().0);
        // SAFETY: `args` is the caller's live memory, as on every backend.
        unsafe {
            e.dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(
                args.address().0,
                size(p.workgroup()),
            );
        }
    }

    /// [`CommandList::dispatch_indirect`] for kernels that hold the root inline.
    pub fn dispatch_indirect_inline<R: Pod>(
        &mut self,
        p: &ComputePipeline,
        root: &R,
        args: GpuPtr<DispatchArgs>,
    ) {
        root_inline::<R>(p);
        let root = self.inline_bytes(bytemuck::bytes_of(root));
        let e = self.bind_compute(p, root);
        // SAFETY: as in `dispatch_indirect`.
        unsafe {
            e.dispatchThreadgroupsWithIndirectBuffer_threadsPerThreadgroup(
                args.address().0,
                size(p.workgroup()),
            );
        }
    }

    /// No call creates a [`RayDispatchPipeline`] on Metal.
    pub fn dispatch_rays<R: Pod>(&mut self, p: &RayDispatchPipeline, root: &R, extent: [u32; 3]) {
        let _ = (root, extent);
        p.unreachable()
    }

    pub fn dispatch_rays_indirect<R: Pod>(
        &mut self,
        p: &RayDispatchPipeline,
        root: &R,
        args: GpuPtr<DispatchArgs>,
    ) {
        let _ = (root, args);
        p.unreachable()
    }

    /// Copies `bytes` bytes from `src` to `dst`.
    pub fn copy(&mut self, dst: GpuPtr<u8>, src: GpuPtr<u8>, bytes: u64) {
        let (sb, so) = self.buffer_at(src.address().0, bytes, "copy source");
        let (db, d) = self.buffer_at(dst.address().0, bytes, "copy destination");
        let e = self.compute();
        // SAFETY: both ranges lie within live buffers (checked by `buffer_at`).
        unsafe {
            e.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &sb,
                so as usize,
                &db,
                d as usize,
                bytes as usize,
            );
        }
    }

    /// The buffer holding `bytes` bytes at `address`, and the offset into it.
    pub(super) fn buffer_at(
        &self,
        address: u64,
        bytes: u64,
        what: &str,
    ) -> (super::Obj<super::Buffer>, u64) {
        lock(&self.queue.shared.buffers)
            .find(address, bytes)
            .unwrap_or_else(|| {
                panic!("{what}: {bytes} bytes at {address:#x} are in no live allocation")
            })
    }

    pub fn decompress(
        &mut self,
        regions: GpuPtr<DecompressRegion>,
        count: GpuPtr<u32>,
        max_count: u32,
    ) {
        let _ = (regions, count, max_count);
        not_implemented("CommandList::decompress")
    }

    pub fn copy_indirect(&mut self, regions: GpuPtr<MemoryCopy>, count: u32) {
        let _ = (regions, count);
        not_implemented("CommandList::copy_indirect")
    }

    pub fn copy_to_texture_indirect(
        &mut self,
        dst: &Texture,
        regions: GpuPtr<TextureCopy>,
        layers: &[TextureLayers],
    ) {
        let _ = (dst, regions, layers);
        not_implemented("CommandList::copy_to_texture_indirect")
    }

    /// Nothing on Metal.
    pub fn set_occupancy(&mut self, priority: f32, throttling: f32) {
        let _ = (priority, throttling);
    }

    /// Opens a named pass. Metal records no pass timings yet.
    pub fn begin_pass(&mut self, name: &'static str, stats: bool) {
        let _ = (name, stats);
    }

    pub fn end_pass(&mut self) {}
}

impl Drop for CommandList<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.end_encoder();
            self.buffer.endCommandBuffer();
        }
    }
}

fn root_in_memory(p: &ComputePipeline) {
    assert_eq!(
        p.inline_root_bytes, 0,
        "an ORR_ROOT_INLINE kernel runs with dispatch_inline or dispatch_indirect_inline"
    );
}

fn root_inline<R>(p: &ComputePipeline) {
    assert_eq!(
        p.inline_root_bytes as usize,
        size_of::<R>(),
        "the inline root is not the kernel's ORR_ROOT_INLINE type"
    );
}

pub(super) fn size([x, y, z]: [u32; 3]) -> MTLSize {
    MTLSize {
        width: x as usize,
        height: y as usize,
        depth: z as usize,
    }
}

/// The Metal stages of `s`; empty for the host, which needs no GPU barrier.
fn mtl_stages(s: Stages) -> MTLStages {
    if s.contains(Stages::ALL) {
        return MTLStages::All;
    }
    let mut m = MTLStages::empty();
    if s.intersects(Stages::TRANSFER) {
        m |= MTLStages::Blit;
    }
    if s.intersects(Stages::COMPUTE | Stages::RAY | Stages::DECOMPRESS | Stages::PREPROCESS) {
        m |= MTLStages::Dispatch;
    }
    if s.intersects(Stages::TASK) {
        m |= MTLStages::Object;
    }
    if s.intersects(Stages::MESH) {
        m |= MTLStages::Mesh;
    }
    if s.intersects(Stages::PIXEL | Stages::COLOR_OUT | Stages::DEPTH) {
        m |= MTLStages::Fragment;
    }
    if s.intersects(Stages::AS_BUILD) {
        m |= MTLStages::AccelerationStructure;
    }
    m
}
