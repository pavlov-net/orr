//! Mesh pipelines (optional task, mesh, optional pixel) and the attachments rendering writes.

use std::hash::Hash;
use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::cache::{KeyHasher, PipelineKind};
use super::device::{Device, Shared};
use super::garbage::Item;
use super::pipeline::{ShaderStage, StageParts};
use super::texture::{Texture, vk_compare, vk_format};
use crate::{
    Blend, BlendFactor, BlendOp, ColorLoad, ColorTarget, Cull, DepthLoad, DepthTarget, Error, Fill,
    FrontFace, RasterState, Stage,
};

/// How to build a mesh pipeline. Every stage reads the root pointer passed to the draw.
#[derive(Clone, Copy, Debug)]
pub struct MeshPipelineDesc<'a> {
    pub task: Option<ShaderStage<'a>>,
    pub mesh: ShaderStage<'a>,
    /// `None`: no pixel stage; only depth is written.
    pub pixel: Option<ShaderStage<'a>>,
    /// Colour targets in attachment order.
    pub colors: &'a [ColorTarget],
    pub depth: Option<DepthTarget>,
    pub raster: RasterState,
    /// Rasterisation samples per pixel: 1, or more for pipelines with no targets.
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

/// A task/mesh/pixel pipeline drawn by [`crate::CommandList::draw_mesh_tasks`] inside
/// [`crate::CommandList::begin_rendering`]. Dropping it destroys it once everything
/// submitted before the drop has completed.
pub struct MeshPipeline {
    pub(crate) raw: vk::Pipeline,
    shared: Arc<Shared>,
}

impl Drop for MeshPipeline {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Pipeline(self.raw));
    }
}

/// A colour attachment of [`Rendering`]: mip 0, layer 0 of a texture with `COLOR` usage.
#[derive(Clone, Copy)]
pub struct ColorAttachment<'a> {
    pub texture: &'a Texture,
    pub load: ColorLoad,
    /// Keep what rendering wrote; otherwise the contents are undefined afterwards.
    pub store: bool,
}

/// The depth attachment of [`Rendering`]: mip 0, layer 0 of a texture with `DEPTH` usage.
/// A depth-stencil format clears its stencil to 0 with `DepthLoad::Clear`.
#[derive(Clone, Copy)]
pub struct DepthAttachment<'a> {
    pub texture: &'a Texture,
    pub load: DepthLoad,
    pub store: bool,
}

/// What [`crate::CommandList::begin_rendering`] draws into. With no attachments the extent
/// alone sets the raster grid.
#[derive(Clone, Copy)]
pub struct Rendering<'a> {
    /// Width and height in pixels, from the origin.
    pub extent: [u32; 2],
    pub colors: &'a [ColorAttachment<'a>],
    pub depth: Option<DepthAttachment<'a>>,
}

fn vk_factor(f: BlendFactor) -> vk::BlendFactor {
    use BlendFactor as B;
    use vk::BlendFactor as V;
    match f {
        B::Zero => V::ZERO,
        B::One => V::ONE,
        B::SrcColor => V::SRC_COLOR,
        B::OneMinusSrcColor => V::ONE_MINUS_SRC_COLOR,
        B::SrcAlpha => V::SRC_ALPHA,
        B::OneMinusSrcAlpha => V::ONE_MINUS_SRC_ALPHA,
        B::DstColor => V::DST_COLOR,
        B::OneMinusDstColor => V::ONE_MINUS_DST_COLOR,
        B::DstAlpha => V::DST_ALPHA,
        B::OneMinusDstAlpha => V::ONE_MINUS_DST_ALPHA,
    }
}

fn vk_blend_op(op: BlendOp) -> vk::BlendOp {
    match op {
        BlendOp::Add => vk::BlendOp::ADD,
        BlendOp::Subtract => vk::BlendOp::SUBTRACT,
        BlendOp::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        BlendOp::Min => vk::BlendOp::MIN,
        BlendOp::Max => vk::BlendOp::MAX,
    }
}

fn blend_state(t: &ColorTarget) -> vk::PipelineColorBlendAttachmentState {
    let s = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::from_raw(t.write.bits()));
    let Some(Blend { color, alpha }) = t.blend else {
        return s;
    };
    s.blend_enable(true)
        .src_color_blend_factor(vk_factor(color.src))
        .dst_color_blend_factor(vk_factor(color.dst))
        .color_blend_op(vk_blend_op(color.op))
        .src_alpha_blend_factor(vk_factor(alpha.src))
        .dst_alpha_blend_factor(vk_factor(alpha.dst))
        .alpha_blend_op(vk_blend_op(alpha.op))
}

fn check(shared: &Shared, d: &MeshPipelineDesc<'_>, name: &str) -> Result<(), Error> {
    let bad = |m: String| Err(Error::Invalid(format!("mesh pipeline {name}: {m}")));
    let max = shared.caps.limits.max_color_targets;
    if d.colors.len() > max as usize {
        return bad(format!(
            "{} colour targets, the device has {max}",
            d.colors.len()
        ));
    }
    if let Some(t) = d.colors.iter().find(|t| t.format.is_depth()) {
        return bad(format!("{:?} is not a colour format", t.format));
    }
    if let Some(t) = d.depth.filter(|t| !t.format.is_depth()) {
        return bad(format!("{:?} is not a depth format", t.format));
    }
    if !d.samples.is_power_of_two() || d.samples > 16 {
        return bad(format!("{} samples", d.samples));
    }
    if d.samples > 1 && (!d.colors.is_empty() || d.depth.is_some()) {
        return bad("more than one sample needs a pipeline with no targets".into());
    }
    if d.raster.conservative && !shared.caps.conservative_raster {
        return Err(Error::Unsupported {
            device: shared.caps.device_name.clone(),
            missing: "conservative rasterisation (VK_EXT_conservative_rasterization)".into(),
        });
    }
    Ok(())
}

impl Device {
    /// Creates a mesh pipeline. Viewport and scissor are dynamic:
    /// [`crate::CommandList::begin_rendering`] sets both to the rendering extent and
    /// [`crate::CommandList::set_scissor`] narrows the scissor. Clip-space y points down the
    /// target: y = -1 is row 0.
    pub fn create_mesh_pipeline(
        &self,
        d: &MeshPipelineDesc<'_>,
        name: &str,
    ) -> Result<MeshPipeline, Error> {
        let shared = &self.shared;
        check(shared, d, name)?;
        let mut parts = Vec::with_capacity(3);
        if let Some(s) = &d.task {
            parts.push((
                vk::ShaderStageFlags::TASK_EXT,
                StageParts::new(shared, s, Stage::Task, name)?,
            ));
        }
        parts.push((
            vk::ShaderStageFlags::MESH_EXT,
            StageParts::new(shared, &d.mesh, Stage::Mesh, name)?,
        ));
        if let Some(s) = &d.pixel {
            parts.push((
                vk::ShaderStageFlags::FRAGMENT,
                StageParts::new(shared, s, Stage::Pixel, name)?,
            ));
        }
        let mut key = KeyHasher::new(PipelineKind::Mesh);
        for (stage, p) in &parts {
            p.hash(*stage, &mut key);
        }
        (d.colors, d.depth, d.raster, d.samples).hash(&mut key);

        let r = &d.raster;
        let colors: Vec<_> = d.colors.iter().map(|t| vk_format(t.format)).collect();
        let blends: Vec<_> = d.colors.iter().map(blend_state).collect();
        let depth_format = d
            .depth
            .map_or(vk::Format::UNDEFINED, |t| vk_format(t.format));
        let stencil_format = match d.depth {
            Some(t) if t.format.has_stencil() => depth_format,
            _ => vk::Format::UNDEFINED,
        };
        let bias = r.depth_bias.unwrap_or_default();
        let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let raw = shared.create_pipeline(
            PipelineKind::Mesh,
            key.finish_key(),
            name,
            "vkCreateGraphicsPipelines",
            |chain| {
                let mut stage_chains: Vec<_> = parts.iter().map(|(_, p)| p.chain()).collect();
                let stages: Vec<_> = stage_chains
                    .iter_mut()
                    .zip(&parts)
                    .map(|(c, (stage, _))| c.info(*stage))
                    .collect();
                let viewport = vk::PipelineViewportStateCreateInfo::default()
                    .viewport_count(1)
                    .scissor_count(1);
                let mut conservative =
                    vk::PipelineRasterizationConservativeStateCreateInfoEXT::default()
                        .conservative_rasterization_mode(
                            vk::ConservativeRasterizationModeEXT::OVERESTIMATE,
                        );
                let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
                    .polygon_mode(match r.fill {
                        Fill::Solid => vk::PolygonMode::FILL,
                        Fill::Lines => vk::PolygonMode::LINE,
                    })
                    .cull_mode(match r.cull {
                        Cull::None => vk::CullModeFlags::NONE,
                        Cull::Front => vk::CullModeFlags::FRONT,
                        Cull::Back => vk::CullModeFlags::BACK,
                    })
                    .front_face(match r.front_face {
                        FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
                        FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
                    })
                    .depth_bias_enable(r.depth_bias.is_some())
                    .depth_bias_constant_factor(bias.constant)
                    .depth_bias_slope_factor(bias.slope)
                    .depth_bias_clamp(bias.clamp)
                    .line_width(1.0);
                if r.conservative {
                    raster = raster.push(&mut conservative);
                }
                let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                    .rasterization_samples(vk::SampleCountFlags::from_raw(d.samples));
                let depth =
                    d.depth
                        .map_or_else(vk::PipelineDepthStencilStateCreateInfo::default, |t| {
                            vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_test_enable(true)
                                .depth_write_enable(t.write)
                                .depth_compare_op(vk_compare(t.compare))
                        });
                let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
                let dynamic =
                    vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
                let mut rendering = vk::PipelineRenderingCreateInfo::default()
                    .color_attachment_formats(&colors)
                    .depth_attachment_format(depth_format)
                    .stencil_attachment_format(stencil_format);
                rendering.p_next = chain.head();
                let mut info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .viewport_state(&viewport)
                    .rasterization_state(&raster)
                    .multisample_state(&multisample)
                    .depth_stencil_state(&depth)
                    .color_blend_state(&blend)
                    .dynamic_state(&dynamic);
                info.p_next = std::ptr::from_ref(&rendering).cast();
                // SAFETY: valid create info; the SPIR-V was validated by the shader build.
                unsafe {
                    shared
                        .device
                        .create_graphics_pipelines(chain.cache, &[info], None)
                }
                .map(|p| p[0])
                .map_err(|(_, e)| e)
            },
        )?;
        Ok(MeshPipeline {
            raw,
            shared: Arc::clone(shared),
        })
    }
}
