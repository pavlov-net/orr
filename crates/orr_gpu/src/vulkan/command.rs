//! Command recording. Record calls read only their arguments and the device's fixed state.

use std::ffi::CString;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::diagnostics::next_pass_id;
use super::garbage::Item;
use super::memory::MemoryKind;
use super::mesh::{MeshPipeline, Rendering};
use super::pipeline::ComputePipeline;
use super::queue::{ListResources, MAX_PASSES, Queue};
use super::ray::RayDispatchPipeline;
use super::texture::{Texture, vk_extent, vk_offset};
use crate::{
    Caps, ClearColor, ColorLoad, DepthLoad, Error, GpuAddress, GpuPtr, Hazards, Stages,
    TextureRegion, TextureUsage,
};

/// What device address commands are told about every `gpu_malloc` and sparse range: their
/// buffers have storage usage. Sparse ranges may be partly bound, so none claims to be fully
/// bound.
pub(crate) const ADDRESS_FLAGS: vk::AddressCommandFlagsKHR =
    vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE;

pub(crate) fn address_range(address: GpuAddress, size: u64) -> vk::DeviceAddressRangeKHR {
    vk::DeviceAddressRangeKHR {
        address: address.0,
        size,
    }
}

/// `vkCmdDispatchIndirect` arguments: workgroup counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DispatchArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// `vkCmdDrawMeshTasksIndirectEXT` arguments: task workgroup counts, or mesh workgroup
/// counts for pipelines without a task stage.
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

pub(crate) struct PassRecord {
    pub id: u32,
    pub name: &'static str,
    pub stats: bool,
}

/// What a finished list hands to its submit.
pub(crate) struct Finished {
    pub res: ListResources,
    pub passes: Vec<PassRecord>,
    /// Objects the list's commands use, destroyed once its submit completes.
    pub keep: Vec<Item>,
}

/// A one-shot command list for one queue. Record on any thread, then hand it to the queue's
/// owner for [`Queue::submit`]. Dropping it unsubmitted discards it.
pub struct CommandList<'q> {
    queue: &'q Queue,
    res: Option<ListResources>,
    passes: Vec<PassRecord>,
    open: bool,
    /// Rendering extent while between `begin_rendering` and `end_rendering`.
    rendering: Option<[u32; 2]>,
    bound: vk::Pipeline,
    /// Raw commands ran since the heaps were last bound.
    rebind_heaps: bool,
    keep: Vec<Item>,
    /// Compacted-size queries recorded so far.
    pub(crate) compact_queries: u32,
}

impl<'q> CommandList<'q> {
    pub(crate) fn new(queue: &'q Queue, res: ListResources) -> Self {
        Self {
            queue,
            res: Some(res),
            passes: Vec::new(),
            open: false,
            rendering: None,
            bound: vk::Pipeline::null(),
            rebind_heaps: false,
            keep: Vec::new(),
            compact_queries: 0,
        }
    }

    pub(crate) fn queue(&self) -> &'q Queue {
        self.queue
    }

    pub(crate) fn in_rendering(&self) -> bool {
        self.rendering.is_some()
    }

    /// Raw commands may have bound anything: the next heap-mode command rebinds the heaps,
    /// its pipeline and its root.
    pub(crate) fn raw_recorded(&mut self) {
        self.rebind_heaps = true;
        self.bound = vk::Pipeline::null();
    }

    pub(crate) fn cmd(&self) -> vk::CommandBuffer {
        self.res().cmd
    }

    pub(crate) fn res(&self) -> &ListResources {
        self.res.as_ref().expect("command list already finished")
    }

    /// Upload memory holding `bytes`, alive until the list's submit completes.
    pub(crate) fn keep_upload(&mut self, bytes: &[u8], name: &str) -> GpuAddress {
        let shared = &self.queue.shared;
        let (raw, mapped) = shared
            .create_buffer(
                MemoryKind::Upload,
                bytes.len() as u64,
                shared.buffer_usage,
                16,
                name,
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ptr = mapped.expect("Upload memory is mapped");
        // SAFETY: the mapping holds at least `bytes.len()` bytes and nothing else uses it.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len()) };
        let address = GpuAddress(raw.address);
        self.keep.push(Item::Buffer(raw));
        address
    }

    /// Dispatches an internal pipeline with `root` as its root pointer.
    pub(crate) fn dispatch_raw(&mut self, p: vk::Pipeline, root: GpuAddress, groups: [u32; 3]) {
        self.bind(vk::PipelineBindPoint::COMPUTE, p, root.0);
        // SAFETY: recording with a compute pipeline bound.
        unsafe {
            self.device()
                .cmd_dispatch(self.cmd(), groups[0], groups[1], groups[2]);
        };
    }

    fn device(&self) -> &ash::Device {
        &self.queue.shared.device
    }

    /// Ends recording and hands the resources to the submit.
    pub(crate) fn finish(mut self) -> Result<Finished, Error> {
        assert!(
            !self.open,
            "pass {:?} was not ended",
            self.passes.last().map(|p| p.name)
        );
        assert!(self.rendering.is_none(), "rendering was not ended");
        let res = self.res.take().unwrap();
        // SAFETY: the buffer is recording and owned by this list.
        if let Err(e) = unsafe { self.device().end_command_buffer(res.cmd) } {
            self.queue.recycle(res);
            return Err(Error::vk("vkEndCommandBuffer")(e));
        }
        Ok(Finished {
            res,
            passes: std::mem::take(&mut self.passes),
            keep: std::mem::take(&mut self.keep),
        })
    }

    /// Moves a swapchain image into `GENERAL` after its acquire (`acquired`), or into
    /// `PRESENT_SRC` after the last write before its present.
    pub(crate) fn swapchain_barrier(&mut self, t: &Texture, acquired: bool) {
        let all = vk::PipelineStageFlags2::ALL_COMMANDS;
        let rw = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        let b = vk::ImageMemoryBarrier2::default()
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(t.image)
            .subresource_range(t.full_range());
        let b = if acquired {
            b.src_stage_mask(super::queue::SWAPCHAIN_WRITE_STAGES)
                .dst_stage_mask(all)
                .dst_access_mask(rw)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
        } else {
            b.src_stage_mask(all)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(all)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        };
        // SAFETY: recording.
        unsafe {
            self.device().cmd_pipeline_barrier2(
                self.cmd(),
                &vk::DependencyInfo::default().image_memory_barriers(&[b]),
            );
        }
    }

    /// Makes writes in `before` stages visible to `after` stages. [`Stages::RAY`] and
    /// [`Stages::DECOMPRESS`] are ignored without their `Caps`, since no such work can run
    /// then.
    pub fn barrier(&mut self, before: Stages, after: Stages, hazards: Hazards) {
        let caps = &self.queue.shared.caps;
        let mut src = vk_stages(before, caps);
        let mut dst = vk_stages(after, caps);
        let mut src_access = vk::AccessFlags2::MEMORY_WRITE;
        let mut dst_access = vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
        if hazards.contains(Hazards::DRAW_ARGUMENTS) {
            dst |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        }
        if hazards.contains(Hazards::ACCEL_STRUCT) {
            src_access |= vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR;
            dst_access |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
            src |= vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
        }
        if hazards.contains(Hazards::DESCRIPTORS) {
            dst_access |=
                vk::AccessFlags2::RESOURCE_HEAP_READ_EXT | vk::AccessFlags2::SAMPLER_HEAP_READ_EXT;
        }
        if before.contains(Stages::HOST) {
            src_access |= vk::AccessFlags2::HOST_WRITE;
        }
        if after.contains(Stages::HOST) {
            dst_access |= vk::AccessFlags2::HOST_READ;
        }
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(src)
            .src_access_mask(src_access)
            .dst_stage_mask(dst)
            .dst_access_mask(dst_access)];
        // SAFETY: recording.
        unsafe {
            self.device().cmd_pipeline_barrier2(
                self.cmd(),
                &vk::DependencyInfo::default().memory_barriers(&b),
            );
        };
    }

    /// Puts `t` in `GENERAL` with undefined contents. Required before a texture's first use
    /// and whenever its placed memory held another resource.
    pub fn discard(&mut self, t: &Texture) {
        let b = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(t.image)
            .subresource_range(t.full_range())];
        // SAFETY: recording.
        unsafe {
            self.device().cmd_pipeline_barrier2(
                self.cmd(),
                &vk::DependencyInfo::default().image_memory_barriers(&b),
            );
        }
    }

    fn bind(&mut self, point: vk::PipelineBindPoint, pipeline: vk::Pipeline, root: u64) {
        let cmd = self.cmd();
        if self.rebind_heaps {
            self.queue
                .shared
                .heaps()
                .bind(&self.queue.shared.heap_fn, cmd);
            self.rebind_heaps = false;
        }
        if self.bound != pipeline {
            // SAFETY: recording on a compute-capable queue.
            unsafe { self.device().cmd_bind_pipeline(cmd, point, pipeline) };
            self.bound = pipeline;
        }
        let data = root.to_le_bytes();
        let push = vk::PushDataInfoEXT {
            offset: 0,
            data: vk::HostAddressRangeConstEXT::default().address(&data),
            ..Default::default()
        };
        // SAFETY: recording; heaps are bound.
        unsafe { self.queue.shared.heap_fn.cmd_push_data(cmd, &push) };
    }

    /// Runs `p` over `groups` workgroups with `root` as its root pointer.
    pub fn dispatch<R>(&mut self, p: &ComputePipeline, root: GpuPtr<R>, groups: [u32; 3]) {
        self.bind(vk::PipelineBindPoint::COMPUTE, p.raw, root.address().0);
        // SAFETY: recording with a compute pipeline bound.
        unsafe {
            self.device()
                .cmd_dispatch(self.cmd(), groups[0], groups[1], groups[2]);
        };
    }

    /// Runs `p`'s raygen entry once per element of `extent` with `root` as its root pointer.
    pub fn dispatch_rays<R>(&mut self, p: &RayDispatchPipeline, root: GpuPtr<R>, extent: [u32; 3]) {
        self.bind(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            p.raw,
            root.address().0,
        );
        let rt = self
            .queue
            .shared
            .rt_fn
            .as_ref()
            .expect("a ray-dispatch pipeline exists");
        let empty = vk::StridedDeviceAddressRegionKHR::default();
        // SAFETY: recording with a ray-tracing pipeline bound; its raygen record is live
        // while the pipeline is.
        unsafe {
            rt.cmd_trace_rays(
                self.cmd(),
                &p.raygen,
                &empty,
                &empty,
                &empty,
                extent[0],
                extent[1],
                extent[2],
            );
        };
    }

    /// Starts drawing into `r`'s attachments, which stay in `GENERAL`. Viewport and scissor
    /// cover the extent. Order earlier writes to the attachments with a barrier before
    /// `COLOR_OUT` and `DEPTH`.
    pub fn begin_rendering(&mut self, r: &Rendering<'_>) {
        assert!(self.rendering.is_none(), "begin_rendering inside rendering");
        assert!(
            self.queue.graphics,
            "rendering needs a queue with graphics (Render or Present)"
        );
        let colors: Vec<_> = r
            .colors
            .iter()
            .map(|a| {
                let (load, clear) = match a.load {
                    ColorLoad::Keep => (vk::AttachmentLoadOp::LOAD, vk::ClearColorValue::default()),
                    ColorLoad::Discard => (
                        vk::AttachmentLoadOp::DONT_CARE,
                        vk::ClearColorValue::default(),
                    ),
                    ColorLoad::Clear(c) => (
                        vk::AttachmentLoadOp::CLEAR,
                        match c {
                            ClearColor::Float(v) => vk::ClearColorValue { float32: v },
                            ClearColor::Uint(v) => vk::ClearColorValue { uint32: v },
                            ClearColor::Sint(v) => vk::ClearColorValue { int32: v },
                        },
                    ),
                };
                vk::RenderingAttachmentInfo::default()
                    .image_view(a.texture.attachment(TextureUsage::COLOR))
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(load)
                    .store_op(store_op(a.store))
                    .clear_value(vk::ClearValue { color: clear })
            })
            .collect();
        let depth = r.depth.map(|a| {
            let (load, value) = match a.load {
                DepthLoad::Keep => (vk::AttachmentLoadOp::LOAD, 0.0),
                DepthLoad::Discard => (vk::AttachmentLoadOp::DONT_CARE, 0.0),
                DepthLoad::Clear(v) => (vk::AttachmentLoadOp::CLEAR, v),
            };
            vk::RenderingAttachmentInfo::default()
                .image_view(a.texture.attachment(TextureUsage::DEPTH))
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(load)
                .store_op(store_op(a.store))
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: value,
                        stencil: 0,
                    },
                })
        });
        let stencil = r
            .depth
            .filter(|a| a.texture.desc().format.has_stencil())
            .and(depth);
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: r.extent[0],
                height: r.extent[1],
            },
        };
        let mut info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&colors);
        if let Some(d) = &depth {
            info = info.depth_attachment(d);
        }
        if let Some(s) = &stencil {
            info = info.stencil_attachment(s);
        }
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: r.extent[0] as f32,
            height: r.extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let cmd = self.cmd();
        // SAFETY: recording on a graphics queue; the attachments are live views in GENERAL.
        unsafe {
            self.device().cmd_begin_rendering(cmd, &info);
            self.device().cmd_set_viewport(cmd, 0, &[viewport]);
            self.device().cmd_set_scissor(cmd, 0, &[area]);
        }
        self.rendering = Some(r.extent);
    }

    /// Limits drawing to the rectangle at `offset` of size `extent`, clamped to the
    /// rendering extent. Lasts until the next `set_scissor` or `end_rendering`.
    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        let full = self.rendering.expect("set_scissor outside rendering");
        let x = offset[0].min(full[0]);
        let y = offset[1].min(full[1]);
        let rect = vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y as i32,
            },
            extent: vk::Extent2D {
                width: extent[0].min(full[0] - x),
                height: extent[1].min(full[1] - y),
            },
        };
        // SAFETY: recording inside rendering.
        unsafe { self.device().cmd_set_scissor(self.cmd(), 0, &[rect]) };
    }

    pub fn end_rendering(&mut self) {
        assert!(
            self.rendering.take().is_some(),
            "end_rendering without begin_rendering"
        );
        // SAFETY: recording inside rendering begun in this list.
        unsafe { self.device().cmd_end_rendering(self.cmd()) };
    }

    /// Draws `p` over `groups` task workgroups (mesh workgroups without a task stage) with
    /// `root` as its root pointer. Inside rendering only.
    pub fn draw_mesh_tasks<R>(&mut self, p: &MeshPipeline, root: GpuPtr<R>, groups: [u32; 3]) {
        assert!(
            self.rendering.is_some(),
            "draw_mesh_tasks outside rendering"
        );
        self.bind(vk::PipelineBindPoint::GRAPHICS, p.raw, root.address().0);
        // SAFETY: recording inside rendering with a mesh pipeline bound.
        unsafe {
            self.queue.shared.mesh_fn.cmd_draw_mesh_tasks(
                self.cmd(),
                groups[0],
                groups[1],
                groups[2],
            );
        }
    }

    /// Draws `p` once per element of `args`, reading the draw count from `count` on the GPU
    /// and drawing at most `max`. Inside rendering only.
    pub fn draw_mesh_tasks_indirect_count<R>(
        &mut self,
        p: &MeshPipeline,
        root: GpuPtr<R>,
        args: GpuPtr<MeshArgs>,
        count: GpuPtr<u32>,
        max: u32,
    ) {
        assert!(
            self.rendering.is_some(),
            "draw_mesh_tasks_indirect_count outside rendering"
        );
        self.bind(vk::PipelineBindPoint::GRAPHICS, p.raw, root.address().0);
        let shared = &self.queue.shared;
        let stride = size_of::<MeshArgs>() as u64;
        let args_bytes = stride * u64::from(max);
        if let Some(dac) = &shared.dac_fn {
            let info = vk::DrawIndirectCount2InfoKHR::default()
                .address_range(vk::StridedDeviceAddressRangeKHR {
                    address: args.address().0,
                    size: args_bytes,
                    stride,
                })
                .address_flags(ADDRESS_FLAGS)
                .count_address_range(address_range(count.address(), 4))
                .count_address_flags(ADDRESS_FLAGS)
                .max_draw_count(max);
            // SAFETY: recording inside rendering with a mesh pipeline bound; the arguments and
            // the count are the caller's live allocations.
            unsafe { dac.cmd_draw_mesh_tasks_indirect_count2(self.cmd(), &info) };
            return;
        }
        let (args_buffer, args_offset) = shared.locate(args.address(), args_bytes);
        let (count_buffer, count_offset) = shared.locate(count.address(), 4);
        // SAFETY: recording inside rendering with a mesh pipeline bound; the arguments and
        // the count each lie in one buffer.
        unsafe {
            shared.mesh_fn.cmd_draw_mesh_tasks_indirect_count(
                self.cmd(),
                args_buffer,
                args_offset,
                count_buffer,
                count_offset,
                max,
                stride as u32,
            );
        }
    }

    /// Runs `p` with workgroup counts read from `args` on the GPU.
    pub fn dispatch_indirect<R>(
        &mut self,
        p: &ComputePipeline,
        root: GpuPtr<R>,
        args: GpuPtr<DispatchArgs>,
    ) {
        self.bind(vk::PipelineBindPoint::COMPUTE, p.raw, root.address().0);
        let shared = &self.queue.shared;
        let bytes = size_of::<DispatchArgs>() as u64;
        if let Some(dac) = &shared.dac_fn {
            let info = vk::DispatchIndirect2InfoKHR::default()
                .address_range(address_range(args.address(), bytes))
                .address_flags(ADDRESS_FLAGS);
            // SAFETY: recording with a compute pipeline bound; the arguments are the caller's
            // live allocation.
            unsafe { dac.cmd_dispatch_indirect2(self.cmd(), &info) };
            return;
        }
        let (buffer, offset) = shared.locate(args.address(), bytes);
        // SAFETY: recording with a compute pipeline bound; the arguments lie in one buffer.
        unsafe {
            self.device()
                .cmd_dispatch_indirect(self.cmd(), buffer, offset);
        };
    }

    /// Copies `bytes` bytes between GPU allocations.
    pub fn copy(&mut self, dst: GpuPtr<u8>, src: GpuPtr<u8>, bytes: u64) {
        let shared = &self.queue.shared;
        if let Some(dac) = &shared.dac_fn {
            let region = [vk::DeviceMemoryCopyKHR::default()
                .src_range(address_range(src.address(), bytes))
                .src_flags(ADDRESS_FLAGS)
                .dst_range(address_range(dst.address(), bytes))
                .dst_flags(ADDRESS_FLAGS)];
            let info = vk::CopyDeviceMemoryInfoKHR::default().regions(&region);
            // SAFETY: recording; both ranges are the caller's live allocations.
            unsafe { dac.cmd_copy_memory(self.cmd(), &info) };
            return;
        }
        let (s, so) = shared.locate(src.address(), bytes);
        let (d, d_off) = shared.locate(dst.address(), bytes);
        let region = [vk::BufferCopy2::default()
            .src_offset(so)
            .dst_offset(d_off)
            .size(bytes)];
        let info = vk::CopyBufferInfo2::default()
            .src_buffer(s)
            .dst_buffer(d)
            .regions(&region);
        // SAFETY: recording; both ranges lie inside live buffers.
        unsafe { self.device().cmd_copy_buffer2(self.cmd(), &info) };
    }

    /// Copies between `memory` and `region` of `t`, into the texture when `to_texture`.
    /// `row_texels`: texels from one row in memory to the next; 0 packs rows tightly.
    fn texture_copy(
        &self,
        t: &Texture,
        region: &TextureRegion,
        memory: GpuPtr<u8>,
        row_texels: u32,
        to_texture: bool,
    ) {
        let format = t.desc().format;
        let block = format.block_extent();
        assert!(
            row_texels == 0 || (row_texels >= region.extent[0] && row_texels.is_multiple_of(block)),
            "rows of {row_texels} texels for a region {} texels wide in {format:?}",
            region.extent[0]
        );
        let rows = TextureRegion {
            extent: [
                row_texels.max(region.extent[0]),
                region.extent[1],
                region.extent[2],
            ],
            ..*region
        };
        let bytes = rows.bytes(format);
        let shared = &self.queue.shared;
        let cmd = self.cmd();
        let layout = vk::ImageLayout::GENERAL;
        if let Some(dac) = &shared.dac_fn {
            let regions = [vk::DeviceMemoryImageCopyKHR::default()
                .address_range(address_range(memory.address(), bytes))
                .address_flags(ADDRESS_FLAGS)
                .address_row_length(row_texels)
                .image_subresource(t.layers(region))
                .image_layout(layout)
                .image_offset(vk_offset(region))
                .image_extent(vk_extent(region))];
            let info = vk::CopyDeviceMemoryImageInfoKHR::default()
                .image(t.image)
                .regions(&regions);
            // SAFETY: recording; the texture is in GENERAL and the memory range is the
            // caller's live allocation.
            unsafe {
                if to_texture {
                    dac.cmd_copy_memory_to_image(cmd, &info);
                } else {
                    dac.cmd_copy_image_to_memory(cmd, &info);
                }
            }
            return;
        }
        let (buffer, offset) = shared.locate(memory.address(), bytes);
        let regions = [vk::BufferImageCopy2::default()
            .buffer_offset(offset)
            .buffer_row_length(row_texels)
            .image_subresource(t.layers(region))
            .image_offset(vk_offset(region))
            .image_extent(vk_extent(region))];
        let device = self.device();
        // SAFETY: recording; the texture is in GENERAL and the range lies in one buffer.
        unsafe {
            if to_texture {
                let info = vk::CopyBufferToImageInfo2::default()
                    .src_buffer(buffer)
                    .dst_image(t.image)
                    .dst_image_layout(layout)
                    .regions(&regions);
                device.cmd_copy_buffer_to_image2(cmd, &info);
            } else {
                let info = vk::CopyImageToBufferInfo2::default()
                    .src_image(t.image)
                    .src_image_layout(layout)
                    .dst_buffer(buffer)
                    .regions(&regions);
                device.cmd_copy_image_to_buffer2(cmd, &info);
            }
        }
    }

    /// Copies tightly packed texels from `src` into `region` of `dst`.
    pub fn copy_to_texture(&mut self, dst: &Texture, region: &TextureRegion, src: GpuPtr<u8>) {
        self.copy_rows_to_texture(dst, region, src, 0);
    }

    /// Copies texels from `src`, whose rows start `row_texels` texels apart (a multiple of the
    /// format's block width, at least the region's width), into `region` of `dst`: an edge
    /// tile of a cooked texture, whose rows keep the full tile width.
    pub fn copy_rows_to_texture(
        &mut self,
        dst: &Texture,
        region: &TextureRegion,
        src: GpuPtr<u8>,
        row_texels: u32,
    ) {
        self.texture_copy(dst, region, src, row_texels, true);
    }

    /// Copies `region` of `src` into `dst`, tightly packed.
    pub fn copy_from_texture(&mut self, dst: GpuPtr<u8>, src: &Texture, region: &TextureRegion) {
        self.texture_copy(src, region, dst, 0, false);
    }

    /// Decompresses up to `max_count` GDeflate 1.0 regions listed at `regions`, as many as
    /// the GPU-written `count` says. Each region decompresses to at most 65,536 bytes; its
    /// addresses are multiples of 4. Later reads of the output wait on a barrier from
    /// [`Stages::DECOMPRESS`]. Needs `Caps::memory_decompression` and a graphics or compute
    /// queue.
    pub fn decompress(
        &mut self,
        regions: GpuPtr<DecompressRegion>,
        count: GpuPtr<u32>,
        max_count: u32,
    ) {
        let f = self
            .queue
            .shared
            .decompress_fn
            .as_ref()
            .expect("decompression needs Caps::memory_decompression");
        assert!(
            self.queue.shaders,
            "decompression runs on a graphics or compute queue"
        );
        let limit = self.queue.shared.caps.limits.max_decompress_regions;
        assert!(
            max_count <= limit,
            "{max_count} decompression regions, more than the device's {limit}"
        );
        // SAFETY: recording on a graphics or compute queue with the extension enabled; the
        // addresses are the caller's live allocations.
        unsafe {
            (f.fp().cmd_decompress_memory_indirect_count_ext)(
                self.cmd(),
                vk::MemoryDecompressionMethodFlagsEXT::GDEFLATE_1_0,
                regions.address().0,
                count.address().0,
                max_count,
                size_of::<DecompressRegion>() as u32,
            );
        }
    }

    /// Occupancy priority (0 lowest, 0.5 the default, 1 highest) and throttling (0 none, 1
    /// the most) of the dispatches recorded after it in this list, against work running
    /// beside them on other queues. Does nothing without `Caps::occupancy_priority`.
    pub fn set_occupancy(&mut self, priority: f32, throttling: f32) {
        assert!(
            (0.0..=1.0).contains(&priority) && (0.0..=1.0).contains(&throttling),
            "occupancy priority {priority} and throttling {throttling} outside 0..=1"
        );
        let Some(f) = &self.queue.shared.occupancy_fn else {
            return;
        };
        assert!(
            self.queue.shaders,
            "occupancy priority needs a compute queue"
        );
        let params = vk::ComputeOccupancyPriorityParametersNV::default()
            .occupancy_priority(priority)
            .occupancy_throttling(throttling);
        // SAFETY: recording on a compute queue with the extension enabled.
        unsafe { (f.fp().cmd_set_compute_occupancy_priority_nv)(self.cmd(), &params) };
    }

    /// Opens a named pass: a debug label and GPU timestamps, and with `stats` shader
    /// invocation counts. Passes do not nest; [`crate::Device::pass_results`] reads them.
    /// With `Caps::breadcrumbs` and `Caps::checkpoints` the pass's begin and end are also
    /// marked for [`crate::Device::diagnostics_report`] and crash dumps.
    pub fn begin_pass(&mut self, name: &'static str, stats: bool) {
        assert!(!self.open, "pass {name} begun inside another pass");
        assert!(
            (self.passes.len() as u32) < MAX_PASSES,
            "more than {MAX_PASSES} passes in one list"
        );
        let index = self.passes.len() as u32;
        let res = self.res.as_ref().unwrap();
        let (cmd, timestamps, stats_pool) = (res.cmd, res.timestamps, res.stats);
        let label = CString::new(name).unwrap_or_default();
        let id = next_pass_id();
        // SAFETY: recording; the queries were reset when the list was created.
        unsafe {
            self.queue.shared.debug_fn.cmd_begin_debug_utils_label(
                cmd,
                &vk::DebugUtilsLabelEXT::default().label_name(&label),
            );
            self.queue.shared.mark_pass(cmd, id, false);
            if timestamps != vk::QueryPool::null() {
                self.device().cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    timestamps,
                    2 * index,
                );
            }
            if stats && stats_pool != vk::QueryPool::null() {
                self.device().cmd_begin_query(
                    cmd,
                    stats_pool,
                    index,
                    vk::QueryControlFlags::empty(),
                );
            }
        }
        let stats = stats && stats_pool != vk::QueryPool::null();
        self.passes.push(PassRecord { id, name, stats });
        self.open = true;
    }

    pub fn end_pass(&mut self) {
        assert!(self.open, "end_pass without begin_pass");
        let index = self.passes.len() as u32 - 1;
        let res = self.res.as_ref().unwrap();
        let (cmd, timestamps, stats_pool) = (res.cmd, res.timestamps, res.stats);
        // SAFETY: recording; the pass's queries were begun in this list.
        unsafe {
            if self.passes[index as usize].stats {
                self.device().cmd_end_query(cmd, stats_pool, index);
            }
            if timestamps != vk::QueryPool::null() {
                self.device().cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    timestamps,
                    2 * index + 1,
                );
            }
            let id = self.passes[index as usize].id;
            self.queue.shared.mark_pass(cmd, id, true);
            self.queue.shared.debug_fn.cmd_end_debug_utils_label(cmd);
        }
        self.open = false;
    }
}

impl Drop for CommandList<'_> {
    fn drop(&mut self) {
        if let Some(res) = self.res.take() {
            self.queue.recycle(res);
        }
        for item in self.keep.drain(..) {
            self.queue.shared.defer_after_submitted(item);
        }
    }
}

fn store_op(store: bool) -> vk::AttachmentStoreOp {
    if store {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    }
}

fn vk_stages(s: Stages, caps: &Caps) -> vk::PipelineStageFlags2 {
    use vk::PipelineStageFlags2 as P;
    if s.contains(Stages::ALL) {
        // ALL_COMMANDS covers every queue command but not host access.
        return if s.contains(Stages::HOST) {
            P::ALL_COMMANDS | P::HOST
        } else {
            P::ALL_COMMANDS
        };
    }
    let mut out = P::NONE;
    for (flag, stages) in [
        (
            Stages::TRANSFER,
            if caps.cooperative_vector.is_empty() {
                P::ALL_TRANSFER
            } else {
                P::ALL_TRANSFER | P::CONVERT_COOPERATIVE_VECTOR_MATRIX_NV
            },
        ),
        (Stages::COMPUTE, P::COMPUTE_SHADER),
        (Stages::TASK, P::TASK_SHADER_EXT),
        (Stages::MESH, P::MESH_SHADER_EXT),
        (Stages::PIXEL, P::FRAGMENT_SHADER),
        (
            Stages::RAY,
            if caps.ray_dispatch {
                P::RAY_TRACING_SHADER_KHR
            } else {
                P::NONE
            },
        ),
        (Stages::COLOR_OUT, P::COLOR_ATTACHMENT_OUTPUT),
        (
            Stages::DEPTH,
            P::EARLY_FRAGMENT_TESTS | P::LATE_FRAGMENT_TESTS,
        ),
        (Stages::AS_BUILD, P::ACCELERATION_STRUCTURE_BUILD_KHR),
        (Stages::HOST, P::HOST),
        (
            Stages::DECOMPRESS,
            if caps.memory_decompression {
                P::MEMORY_DECOMPRESSION_EXT
            } else {
                P::NONE
            },
        ),
    ] {
        if s.contains(flag) {
            out |= stages;
        }
    }
    out
}
