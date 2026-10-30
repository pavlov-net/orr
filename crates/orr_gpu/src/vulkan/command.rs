//! Command recording. Record calls read only their arguments and the device's fixed state.

use std::ffi::CString;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::mesh::{MeshPipeline, Rendering};
use super::pipeline::ComputePipeline;
use super::queue::{ListResources, MAX_PASSES, Queue};
use super::ray::RayDispatchPipeline;
use super::texture::Texture;
use crate::{
    ClearColor, ColorLoad, DepthLoad, Error, GpuPtr, Hazards, Stages, TextureRegion, TextureUsage,
};

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

pub(crate) struct PassRecord {
    pub name: &'static str,
    pub stats: bool,
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
        }
    }

    pub(crate) fn queue(&self) -> &'q Queue {
        self.queue
    }

    pub(crate) fn cmd(&self) -> vk::CommandBuffer {
        self.res
            .as_ref()
            .expect("command list already finished")
            .cmd
    }

    fn device(&self) -> &ash::Device {
        &self.queue.shared.device
    }

    /// Ends recording and hands the resources to the submit.
    pub(crate) fn finish(mut self) -> Result<(ListResources, Vec<PassRecord>), Error> {
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
        Ok((res, std::mem::take(&mut self.passes)))
    }

    /// Makes writes in `before` stages visible to `after` stages. [`Stages::RAY`] is
    /// ignored without `Caps::ray_dispatch`, since no ray-dispatch work can run then.
    pub fn barrier(&mut self, before: Stages, after: Stages, hazards: Hazards) {
        let ray = self.queue.shared.caps.ray_dispatch;
        let mut src = vk_stages(before, ray);
        let mut dst = vk_stages(after, ray);
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
        let (args_buffer, args_offset) = shared.locate(args.address(), stride * u64::from(max));
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
        let (buffer, offset) = self
            .queue
            .shared
            .locate(args.address(), size_of::<DispatchArgs>() as u64);
        // SAFETY: recording with a compute pipeline bound; the arguments lie in one buffer.
        unsafe {
            self.device()
                .cmd_dispatch_indirect(self.cmd(), buffer, offset);
        };
    }

    /// Copies `bytes` bytes between GPU allocations.
    pub fn copy(&mut self, dst: GpuPtr<u8>, src: GpuPtr<u8>, bytes: u64) {
        let shared = &self.queue.shared;
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

    fn texture_copy(
        &self,
        t: &Texture,
        region: &TextureRegion,
        buffer: GpuPtr<u8>,
    ) -> (vk::Buffer, vk::BufferImageCopy2<'static>) {
        let bytes = region.bytes(t.desc().format);
        let (b, offset) = self.queue.shared.locate(buffer.address(), bytes);
        let copy = vk::BufferImageCopy2::default()
            .buffer_offset(offset)
            .image_subresource(t.layers(region))
            .image_offset(vk::Offset3D {
                x: region.offset[0] as i32,
                y: region.offset[1] as i32,
                z: region.offset[2] as i32,
            })
            .image_extent(vk::Extent3D {
                width: region.extent[0],
                height: region.extent[1],
                depth: region.extent[2],
            });
        (b, copy)
    }

    /// Copies tightly packed texels from `src` into `region` of `dst`.
    pub fn copy_to_texture(&mut self, dst: &Texture, region: &TextureRegion, src: GpuPtr<u8>) {
        let (buffer, copy) = self.texture_copy(dst, region, src);
        let regions = [copy];
        let info = vk::CopyBufferToImageInfo2::default()
            .src_buffer(buffer)
            .dst_image(dst.image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        // SAFETY: recording; the texture is in GENERAL.
        unsafe { self.device().cmd_copy_buffer_to_image2(self.cmd(), &info) };
    }

    /// Copies `region` of `src` into `dst`, tightly packed.
    pub fn copy_from_texture(&mut self, dst: GpuPtr<u8>, src: &Texture, region: &TextureRegion) {
        let (buffer, copy) = self.texture_copy(src, region, dst);
        let regions = [copy];
        let info = vk::CopyImageToBufferInfo2::default()
            .src_image(src.image)
            .src_image_layout(vk::ImageLayout::GENERAL)
            .dst_buffer(buffer)
            .regions(&regions);
        // SAFETY: recording; the texture is in GENERAL.
        unsafe { self.device().cmd_copy_image_to_buffer2(self.cmd(), &info) };
    }

    /// Opens a named pass: a debug label and GPU timestamps, and with `stats` shader
    /// invocation counts. Passes do not nest; [`crate::Device::pass_results`] reads them.
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
        // SAFETY: recording; the queries were reset when the list was created.
        unsafe {
            self.queue.shared.debug_fn.cmd_begin_debug_utils_label(
                cmd,
                &vk::DebugUtilsLabelEXT::default().label_name(&label),
            );
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
        self.passes.push(PassRecord { name, stats });
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
    }
}

fn store_op(store: bool) -> vk::AttachmentStoreOp {
    if store {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    }
}

fn vk_stages(s: Stages, ray: bool) -> vk::PipelineStageFlags2 {
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
        (Stages::TRANSFER, P::ALL_TRANSFER),
        (Stages::COMPUTE, P::COMPUTE_SHADER),
        (Stages::TASK, P::TASK_SHADER_EXT),
        (Stages::MESH, P::MESH_SHADER_EXT),
        (Stages::PIXEL, P::FRAGMENT_SHADER),
        (
            Stages::RAY,
            if ray {
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
    ] {
        if s.contains(flag) {
            out |= stages;
        }
    }
    out
}
