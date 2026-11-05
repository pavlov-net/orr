use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Archive, MTL4ArgumentTable, MTL4BlendState, MTL4CommandBuffer, MTL4CommandEncoder,
    MTL4Compiler, MTL4LibraryFunctionDescriptor, MTL4MeshRenderPipelineDescriptor,
    MTL4PipelineDescriptor, MTL4RenderCommandEncoder, MTL4RenderPassDescriptor, MTLBlendFactor,
    MTLBlendOperation, MTLClearColor, MTLColorWriteMask, MTLComputePipelineState, MTLCullMode,
    MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLLoadAction,
    MTLRenderPipelineState, MTLRenderStages, MTLScissorRect, MTLStoreAction, MTLTriangleFillMode,
    MTLViewport, MTLWinding,
};

use super::Obj;
use super::command::{CommandList, RenderEncoder, size};
use super::device::{Device, metal_error};
use super::heap::compare;
use super::texture::pixel_format;
use crate::{
    BlendFactor, BlendOp, ClearColor, ColorLoad, ColorMask, Cull, DepthLoad, Error, Fill,
    FrontFace, GpuPtr, Hazards, MeshArgs, MeshPipelineDesc, RasterState, Rendering, Stages,
};

/// Object payload bytes a task stage may write: Metal's limit, since the build does not
/// record each kernel's payload size yet.
const PAYLOAD_BYTES: usize = 16 << 10;

/// A task/mesh/pixel pipeline drawn by [`CommandList::draw_mesh_tasks`] inside
/// [`CommandList::begin_rendering`]. Command lists keep the pipelines they draw with until
/// their submit completes.
pub struct MeshPipeline {
    raw: Obj<ProtocolObject<dyn MTLRenderPipelineState>>,
    /// Metal sets depth testing and raster state on the encoder, at each draw.
    depth: Option<Obj<ProtocolObject<dyn MTLDepthStencilState>>>,
    raster: RasterState,
    task_workgroup: [u32; 3],
    mesh_workgroup: [u32; 3],
}

impl Device {
    /// The mesh pipeline of `d`, labeled `name`.
    pub fn create_mesh_pipeline(
        &self,
        d: &MeshPipelineDesc<'_>,
        name: &str,
    ) -> Result<MeshPipeline, Error> {
        if d.raster.conservative {
            return Err(Error::NotImplemented("conservative rasterization"));
        }
        let shared = &self.shared;
        let desc = MTL4MeshRenderPipelineDescriptor::new();
        desc.setLabel(Some(&NSString::from_str(name)));
        if let Some(task) = &d.task {
            let f = task.function(shared)?;
            desc.setObjectFunctionDescriptor(Some(&f));
            desc.setRequiredThreadsPerObjectThreadgroup(size(task.entry.workgroup));
            desc.setPayloadMemoryLength(PAYLOAD_BYTES);
        }
        let mesh = d.mesh.function(shared)?;
        desc.setMeshFunctionDescriptor(Some(&mesh));
        desc.setRequiredThreadsPerMeshThreadgroup(size(d.mesh.entry.workgroup));
        if let Some(pixel) = &d.pixel {
            let f = pixel.function(shared)?;
            desc.setFragmentFunctionDescriptor(Some(&f));
        }
        let attachments = desc.colorAttachments();
        for (i, c) in d.colors.iter().enumerate() {
            // SAFETY: the index is below Metal's eight color attachments, which pixel_format
            // and the pipeline compile validate.
            let a = unsafe { attachments.objectAtIndexedSubscript(i) };
            a.setPixelFormat(pixel_format(c.format)?);
            a.setWriteMask(write_mask(c.write));
            if let Some(b) = c.blend {
                a.setBlendingState(MTL4BlendState::Enabled);
                a.setSourceRGBBlendFactor(blend_factor(b.color.src));
                a.setDestinationRGBBlendFactor(blend_factor(b.color.dst));
                a.setRgbBlendOperation(blend_op(b.color.op));
                a.setSourceAlphaBlendFactor(blend_factor(b.alpha.src));
                a.setDestinationAlphaBlendFactor(blend_factor(b.alpha.dst));
                a.setAlphaBlendOperation(blend_op(b.alpha.op));
            }
        }
        // SAFETY: a sample count the device validates.
        unsafe { desc.setRasterSampleCount(d.samples.max(1) as usize) };
        let base: &MTL4PipelineDescriptor = &desc;
        // The stages, then the state the descriptor takes from `d`.
        let mut key = blake3::Hasher::new();
        key.update(b"mesh");
        for (tag, stage) in [(b't', d.task), (b'm', Some(d.mesh)), (b'p', d.pixel)] {
            if let Some(s) = stage {
                key.update(&[tag]);
                s.key(&mut key);
            }
        }
        key.update(format!("{:?}", (d.colors, d.depth, d.raster, d.samples)).as_bytes());
        let raw = shared.cached_render(
            key.finalize(),
            self.cached_only,
            |archive| {
                archive
                    .newRenderPipelineStateWithDescriptor_error(base)
                    .ok()
            },
            |options| {
                shared
                    .compiler
                    .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base, options)
                    .map_err(|e| metal_error("newRenderPipelineState", &e))
            },
            Retained::into_super(desc.clone()),
        )?;
        let depth = match d.depth {
            Some(t) => {
                let ds = MTLDepthStencilDescriptor::new();
                ds.setDepthCompareFunction(compare(t.compare));
                ds.setDepthWriteEnabled(t.write);
                let state = shared
                    .device
                    .newDepthStencilStateWithDescriptor(&ds)
                    .ok_or_else(|| Error::Metal {
                        call: "newDepthStencilState",
                        message: name.to_owned(),
                    })?;
                Some(Obj(state))
            }
            None => None,
        };
        Ok(MeshPipeline {
            raw,
            depth,
            raster: d.raster,
            task_workgroup: d.task.map_or([1, 1, 1], |t| t.entry.workgroup),
            mesh_workgroup: d.mesh.entry.workgroup,
        })
    }
}

/// A render pass being recorded. Metal opens its encoder at `end_rendering`, after a kernel
/// has gated the arguments of the pass's counted draws, which no render command can do.
pub(super) struct PendingPass {
    desc: Retained<MTL4RenderPassDescriptor>,
    extent: [u32; 2],
    commands: Vec<RenderCommand>,
    /// Counted draws' arguments to gate, as (args, count, gated copy, max).
    gates: Vec<(u64, u64, u64, u32)>,
}

enum RenderCommand {
    Scissor(MTLScissorRect),
    Draw {
        pipeline: DrawState,
        root: u64,
        groups: Groups,
    },
}

enum Groups {
    Direct([u32; 3]),
    /// The address of Metal's indirect threadgroup counts, laid out as [`MeshArgs`].
    Indirect(u64),
}

/// What a draw sets from its [`MeshPipeline`].
#[derive(Clone)]
struct DrawState {
    raw: Obj<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth: Option<Obj<ProtocolObject<dyn MTLDepthStencilState>>>,
    raster: RasterState,
    task_workgroup: [u32; 3],
    mesh_workgroup: [u32; 3],
}

/// Copies `max` draws' threadgroup counts, zeroing those at or past the count.
const DRAW_GATE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Args { device const uint* src; device const uint* count; device uint* dst; uint max; uint pad; };
kernel void gate(constant Args& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.max) return;
    bool live = i < *a.count;
    for (uint c = 0; c < 3; c++) a.dst[3 * i + c] = live ? a.src[3 * i + c] : 0;
}
"#;

/// The arguments of [`DRAW_GATE`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GateArgs {
    src: u64,
    count: u64,
    dst: u64,
    max: u32,
    pad: u32,
}

impl CommandList<'_> {
    /// Starts drawing into `r`'s attachments.
    pub fn begin_rendering(&mut self, r: &Rendering<'_>) {
        assert!(self.pass.is_none(), "begin_rendering inside rendering");
        let pass = MTL4RenderPassDescriptor::new();
        pass.setRenderTargetWidth(r.extent[0] as usize);
        pass.setRenderTargetHeight(r.extent[1] as usize);
        // Passes without attachments, which only write memory, rasterize one sample a pixel.
        // SAFETY: one sample is always supported.
        unsafe { pass.setDefaultRasterSampleCount(1) };
        let colors = pass.colorAttachments();
        for (i, c) in r.colors.iter().enumerate() {
            // SAFETY: the index is below Metal's eight color attachments; Metal validates the
            // pass.
            let a = unsafe { colors.objectAtIndexedSubscript(i) };
            a.setTexture(Some(&c.texture.raw));
            a.setLoadAction(match c.load {
                ColorLoad::Keep => MTLLoadAction::Load,
                ColorLoad::Clear(_) => MTLLoadAction::Clear,
                ColorLoad::Discard => MTLLoadAction::DontCare,
            });
            if let ColorLoad::Clear(clear) = c.load {
                a.setClearColor(clear_color(clear));
            }
            a.setStoreAction(store_action(c.store));
            self.keep.push(Box::new(c.texture.raw.clone()));
        }
        if let Some(d) = &r.depth {
            let a = pass.depthAttachment();
            a.setTexture(Some(&d.texture.raw));
            a.setLoadAction(match d.load {
                DepthLoad::Keep => MTLLoadAction::Load,
                DepthLoad::Clear(_) => MTLLoadAction::Clear,
                DepthLoad::Discard => MTLLoadAction::DontCare,
            });
            if let DepthLoad::Clear(z) = d.load {
                a.setClearDepth(f64::from(z));
            }
            a.setStoreAction(store_action(d.store));
            if d.texture.desc().format.has_stencil() {
                let s = pass.stencilAttachment();
                s.setTexture(Some(&d.texture.raw));
                s.setLoadAction(match d.load {
                    DepthLoad::Keep => MTLLoadAction::Load,
                    DepthLoad::Clear(_) => MTLLoadAction::Clear,
                    DepthLoad::Discard => MTLLoadAction::DontCare,
                });
                s.setClearStencil(0);
                s.setStoreAction(store_action(d.store));
            }
            self.keep.push(Box::new(d.texture.raw.clone()));
        }
        self.pass = Some(PendingPass {
            desc: pass,
            extent: r.extent,
            commands: Vec::new(),
            gates: Vec::new(),
        });
    }

    /// Limits drawing to `extent` pixels from `offset`, clipped to the pass: Metal takes no
    /// rectangle reaching past it.
    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        let pass = self.pass();
        let [x, y] = [0, 1].map(|i| offset[i].min(pass.extent[i]));
        let [w, h] = [0, 1].map(|i| extent[i].min(pass.extent[i] - [x, y][i]));
        pass.commands.push(RenderCommand::Scissor(MTLScissorRect {
            x: x as usize,
            y: y as usize,
            width: w as usize,
            height: h as usize,
        }));
    }

    pub fn end_rendering(&mut self) {
        let pass = self
            .pass
            .take()
            .expect("end_rendering without begin_rendering");
        if !pass.gates.is_empty() {
            let state = self.queue.shared.draw_gate().expect("the draw gate kernel");
            for &(src, count, dst, max) in &pass.gates {
                let args = GateArgs {
                    src,
                    count,
                    dst,
                    max,
                    pad: 0,
                };
                self.dispatch_state(
                    &state,
                    bytemuck::bytes_of(&args),
                    [max.div_ceil(64), 1, 1],
                    [64, 1, 1],
                );
            }
            self.barrier(Stages::COMPUTE, Stages::ALL, Hazards::DRAW_ARGUMENTS);
        }
        self.end_encoder();
        let e = autoreleasepool(|_| self.buffer.renderCommandEncoderWithDescriptor(&pass.desc))
            .expect("renderCommandEncoderWithDescriptor");
        self.open_barrier(ProtocolObject::from_ref(&*e));
        // orr's clip space has +y pointing down the target, Metal's up: the viewport flips it.
        let (w, h) = (f64::from(pass.extent[0]), f64::from(pass.extent[1]));
        e.setViewport(MTLViewport {
            originX: 0.0,
            originY: h,
            width: w,
            height: -h,
            znear: 0.0,
            zfar: 1.0,
        });
        for c in pass.commands {
            match c {
                RenderCommand::Scissor(r) => e.setScissorRect(r),
                RenderCommand::Draw {
                    pipeline,
                    root,
                    groups,
                } => self.encode_draw(&e, &pipeline, root, groups),
            }
        }
        e.endEncoding();
    }

    fn pass(&mut self) -> &mut PendingPass {
        self.pass.as_mut().expect("drawing outside begin_rendering")
    }

    fn draw(&mut self, p: &MeshPipeline, root: u64, groups: Groups) {
        let pipeline = DrawState {
            raw: p.raw.clone(),
            depth: p.depth.clone(),
            raster: p.raster,
            task_workgroup: p.task_workgroup,
            mesh_workgroup: p.mesh_workgroup,
        };
        self.keep.push(Box::new(pipeline.clone()));
        self.pass().commands.push(RenderCommand::Draw {
            pipeline,
            root,
            groups,
        });
    }

    fn encode_draw(&self, e: &RenderEncoder, p: &DrawState, root: u64, groups: Groups) {
        e.setRenderPipelineState(&p.raw);
        let raster = p.raster;
        e.setCullMode(match raster.cull {
            Cull::None => MTLCullMode::None,
            Cull::Front => MTLCullMode::Front,
            Cull::Back => MTLCullMode::Back,
        });
        e.setFrontFacingWinding(match raster.front_face {
            FrontFace::CounterClockwise => MTLWinding::CounterClockwise,
            FrontFace::Clockwise => MTLWinding::Clockwise,
        });
        e.setTriangleFillMode(match raster.fill {
            Fill::Solid => MTLTriangleFillMode::Fill,
            Fill::Lines => MTLTriangleFillMode::Lines,
        });
        if let Some(b) = raster.depth_bias {
            e.setDepthBias_slopeScale_clamp(b.constant, b.slope, b.clamp);
        }
        e.setDepthStencilState(p.depth.as_deref());
        // SAFETY: the root's address is the caller's to keep valid until the submit
        // completes, as on every backend.
        unsafe { self.table.setAddress_atIndex(root, 0) };
        e.setArgumentTable_atStages(
            &self.table,
            MTLRenderStages::Object | MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
        let (task, mesh) = (size(p.task_workgroup), size(p.mesh_workgroup));
        match groups {
            Groups::Direct(g) => e
                .drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    size(g),
                    task,
                    mesh,
                ),
            Groups::Indirect(args) => e
                .drawMeshThreadgroupsWithIndirectBuffer_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                    args, task, mesh,
                ),
        }
    }

    /// Draws `groups` task workgroups of `p`, or mesh workgroups without a task stage, with
    /// the root at `root`.
    pub fn draw_mesh_tasks<R>(&mut self, p: &MeshPipeline, root: GpuPtr<R>, groups: [u32; 3]) {
        self.draw(p, root.address().0, Groups::Direct(groups));
    }

    /// Draws up to `max` times with the workgroup counts at `args`, as many times as the
    /// GPU wrote to `count` by the time the pass starts.
    pub fn draw_mesh_tasks_indirect_count<R>(
        &mut self,
        p: &MeshPipeline,
        root: GpuPtr<R>,
        args: GpuPtr<MeshArgs>,
        count: GpuPtr<u32>,
        max: u32,
    ) {
        let stride = size_of::<MeshArgs>() as u64;
        let gated = self.inline_bytes(&vec![0; (stride * u64::from(max)) as usize]);
        self.pass()
            .gates
            .push((args.address().0, count.address().0, gated, max));
        for i in 0..u64::from(max) {
            self.draw(p, root.address().0, Groups::Indirect(gated + i * stride));
        }
    }
}

impl super::device::Shared {
    /// The compute pipeline of [`DRAW_GATE`], compiled once.
    fn draw_gate(&self) -> Result<Obj<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        static STATE: OnceLock<Obj<ProtocolObject<dyn MTLComputePipelineState>>> = OnceLock::new();
        if let Some(s) = STATE.get() {
            return Ok(s.clone());
        }
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(DRAW_GATE), None)
            .map_err(|e| metal_error("newLibraryWithSource", &e))?;
        let f = MTL4LibraryFunctionDescriptor::new();
        f.setName(Some(&NSString::from_str("gate")));
        f.setLibrary(Some(&library));
        let state = self.compute_state(&f, [64, 1, 1], "draw gate")?;
        Ok(STATE.get_or_init(|| state).clone())
    }
}

fn store_action(store: bool) -> MTLStoreAction {
    if store {
        MTLStoreAction::Store
    } else {
        MTLStoreAction::DontCare
    }
}

fn clear_color(c: ClearColor) -> MTLClearColor {
    let [red, green, blue, alpha] = match c {
        ClearColor::Float(v) => v.map(f64::from),
        ClearColor::Uint(v) => v.map(f64::from),
        ClearColor::Sint(v) => v.map(f64::from),
    };
    MTLClearColor {
        red,
        green,
        blue,
        alpha,
    }
}

fn write_mask(m: ColorMask) -> MTLColorWriteMask {
    let mut out = MTLColorWriteMask::None;
    for (orr, metal) in [
        (ColorMask::R, MTLColorWriteMask::Red),
        (ColorMask::G, MTLColorWriteMask::Green),
        (ColorMask::B, MTLColorWriteMask::Blue),
        (ColorMask::A, MTLColorWriteMask::Alpha),
    ] {
        if m.contains(orr) {
            out |= metal;
        }
    }
    out
}

fn blend_factor(f: BlendFactor) -> MTLBlendFactor {
    match f {
        BlendFactor::Zero => MTLBlendFactor::Zero,
        BlendFactor::One => MTLBlendFactor::One,
        BlendFactor::SrcColor => MTLBlendFactor::SourceColor,
        BlendFactor::OneMinusSrcColor => MTLBlendFactor::OneMinusSourceColor,
        BlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        BlendFactor::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        BlendFactor::DstColor => MTLBlendFactor::DestinationColor,
        BlendFactor::OneMinusDstColor => MTLBlendFactor::OneMinusDestinationColor,
        BlendFactor::DstAlpha => MTLBlendFactor::DestinationAlpha,
        BlendFactor::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
    }
}

fn blend_op(o: BlendOp) -> MTLBlendOperation {
    match o {
        BlendOp::Add => MTLBlendOperation::Add,
        BlendOp::Subtract => MTLBlendOperation::Subtract,
        BlendOp::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        BlendOp::Min => MTLBlendOperation::Min,
        BlendOp::Max => MTLBlendOperation::Max,
    }
}
