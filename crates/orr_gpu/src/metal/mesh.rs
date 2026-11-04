use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ArgumentTable, MTL4BlendState, MTL4CommandBuffer, MTL4Compiler,
    MTL4MeshRenderPipelineDescriptor, MTL4PipelineDescriptor, MTL4RenderCommandEncoder,
    MTL4RenderPassDescriptor, MTLBlendFactor, MTLBlendOperation, MTLClearColor, MTLColorWriteMask,
    MTLCullMode, MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLLoadAction,
    MTLRenderPipelineState, MTLRenderStages, MTLScissorRect, MTLStoreAction, MTLTriangleFillMode,
    MTLViewport, MTLWinding,
};

use super::command::{CommandList, Encoder, size};
use super::device::{Device, metal_error};
use super::heap::compare;
use super::pipeline::ShaderStage;
use super::texture::{Texture, pixel_format};
use super::{Obj, not_implemented};
use crate::{
    BlendFactor, BlendOp, ClearColor, ColorLoad, ColorMask, ColorTarget, Cull, DepthLoad,
    DepthTarget, Error, Fill, FrontFace, GpuPtr, RasterState,
};

/// Object payload bytes a task stage may write: Metal's limit, since the build does not
/// record each kernel's payload size yet.
const PAYLOAD_BYTES: usize = 16 << 10;

/// How to build a mesh pipeline. Every stage reads the root pointer passed to the draw.
#[derive(Clone, Copy, Debug)]
pub struct MeshPipelineDesc<'a> {
    pub task: Option<ShaderStage<'a>>,
    pub mesh: ShaderStage<'a>,
    /// `None`: no pixel stage; only depth is written.
    pub pixel: Option<ShaderStage<'a>>,
    /// Color targets in attachment order.
    pub colors: &'a [ColorTarget],
    pub depth: Option<DepthTarget>,
    pub raster: RasterState,
    /// Rasterization samples per pixel: 1, or more for pipelines with no targets.
    pub samples: u32,
}

impl<'a> MeshPipelineDesc<'a> {
    /// A mesh stage alone: no targets, default raster state, one sample.
    pub fn new(mesh: ShaderStage<'a>) -> Self {
        Self {
            task: None,
            mesh,
            pixel: None,
            colors: &[],
            depth: None,
            raster: RasterState::default(),
            samples: 1,
        }
    }
}

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

/// A color attachment of [`Rendering`]: mip 0, layer 0 of a texture with `COLOR` usage.
#[derive(Clone, Copy)]
pub struct ColorAttachment<'a> {
    pub texture: &'a Texture,
    pub load: ColorLoad,
    /// Keep what rendering wrote; otherwise the contents are undefined afterwards.
    pub store: bool,
}

/// The depth attachment of [`Rendering`]: mip 0, layer 0 of a texture with `DEPTH` usage.
#[derive(Clone, Copy)]
pub struct DepthAttachment<'a> {
    pub texture: &'a Texture,
    pub load: DepthLoad,
    pub store: bool,
}

/// What [`CommandList::begin_rendering`] draws into. With no attachments the extent alone
/// sets the raster grid.
#[derive(Clone, Copy)]
pub struct Rendering<'a> {
    /// Width and height in pixels, from the origin.
    pub extent: [u32; 2],
    pub colors: &'a [ColorAttachment<'a>],
    pub depth: Option<DepthAttachment<'a>>,
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
        let raw = shared
            .compiler
            .newRenderPipelineStateWithDescriptor_compilerTaskOptions_error(base, None)
            .map_err(|e| metal_error("newRenderPipelineState", &e))?;
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
            raw: Obj(raw),
            depth,
            raster: d.raster,
            task_workgroup: d.task.map_or([1, 1, 1], |t| t.entry.workgroup),
            mesh_workgroup: d.mesh.entry.workgroup,
        })
    }
}

impl CommandList<'_> {
    /// Starts drawing into `r`'s attachments.
    pub fn begin_rendering(&mut self, r: &Rendering<'_>) {
        self.end_encoder();
        let pass = MTL4RenderPassDescriptor::new();
        pass.setRenderTargetWidth(r.extent[0] as usize);
        pass.setRenderTargetHeight(r.extent[1] as usize);
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
        let e = self
            .buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .expect("renderCommandEncoderWithDescriptor");
        self.open_barrier(ProtocolObject::from_ref(&*e));
        // orr's clip space has +y pointing down the target, Metal's up: the viewport flips it.
        let (w, h) = (f64::from(r.extent[0]), f64::from(r.extent[1]));
        e.setViewport(MTLViewport {
            originX: 0.0,
            originY: h,
            width: w,
            height: -h,
            znear: 0.0,
            zfar: 1.0,
        });
        self.encoder = Encoder::Render(e);
    }

    /// Limits drawing to `extent` pixels from `offset`.
    pub fn set_scissor(&mut self, offset: [u32; 2], extent: [u32; 2]) {
        self.render().setScissorRect(MTLScissorRect {
            x: offset[0] as usize,
            y: offset[1] as usize,
            width: extent[0] as usize,
            height: extent[1] as usize,
        });
    }

    pub fn end_rendering(&mut self) {
        assert!(
            matches!(self.encoder, Encoder::Render(_)),
            "end_rendering without begin_rendering"
        );
        self.end_encoder();
    }

    fn render(&self) -> Retained<ProtocolObject<dyn MTL4RenderCommandEncoder>> {
        match &self.encoder {
            Encoder::Render(e) => e.clone(),
            _ => panic!("drawing outside begin_rendering"),
        }
    }

    /// Draws `groups` task workgroups of `p`, or mesh workgroups without a task stage, with
    /// the root at `root`.
    pub fn draw_mesh_tasks<R>(&mut self, p: &MeshPipeline, root: GpuPtr<R>, groups: [u32; 3]) {
        let e = self.render();
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
        unsafe { self.table.setAddress_atIndex(root.address().0, 0) };
        e.setArgumentTable_atStages(
            &self.table,
            MTLRenderStages::Object | MTLRenderStages::Mesh | MTLRenderStages::Fragment,
        );
        e.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            size(groups),
            size(p.task_workgroup),
            size(p.mesh_workgroup),
        );
        self.keep.push(Box::new(p.raw.clone()));
        if let Some(d) = &p.depth {
            self.keep.push(Box::new(d.clone()));
        }
    }

    pub fn draw_mesh_tasks_indirect_count<R>(
        &mut self,
        p: &MeshPipeline,
        root: GpuPtr<R>,
        args: GpuPtr<super::MeshArgs>,
        count: GpuPtr<u32>,
        max: u32,
    ) {
        let _ = (p, root, args, count, max);
        not_implemented("CommandList::draw_mesh_tasks_indirect_count")
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
