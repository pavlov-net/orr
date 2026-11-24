//! Graphics pipelines and the attachments rendering writes.

use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::device::{Device, Shared};
use super::driver_cpu::DriverCpu;
use super::garbage::Item;
use super::pipeline::StageParts;
use super::texture::{vk_compare, vk_format};
use crate::raster::RasterTargets;
use crate::{
    Blend, BlendFactor, BlendOp, ColorTarget, Cull, Error, Fill, FrontFace, MeshPipelineDesc,
    PipelineKind, PrimitiveTopology, Stage, VertexPipelineDesc,
};

/// A task/mesh/pixel pipeline drawn by [`crate::CommandList::draw_mesh_tasks`] inside
/// [`crate::CommandList::begin_rendering`]. Dropping it destroys it once the GPU is done
/// with the work submitted before the drop and with the command lists open at it.
pub struct MeshPipeline {
    pub(super) raw: vk::Pipeline,
    shared: Arc<Shared>,
}

impl Drop for MeshPipeline {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Pipeline(self.raw));
    }
}

/// A vertex/pixel pipeline drawn by [`crate::CommandList::draw`] inside rendering.
pub struct VertexPipeline {
    pub(super) raw: vk::Pipeline,
    shared: Arc<Shared>,
}

impl Drop for VertexPipeline {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Pipeline(self.raw));
    }
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

fn check(shared: &Shared, d: RasterTargets<'_>, name: &str) -> Result<(), Error> {
    d.validate(shared.caps.limits.max_color_targets, name)?;
    if d.raster.conservative && !shared.caps.conservative_raster {
        return Err(Error::Unsupported {
            device: shared.caps.device_name.clone(),
            missing: "conservative rasterization (VK_EXT_conservative_rasterization)".into(),
        });
    }
    Ok(())
}

impl Shared {
    /// A mesh pipeline with `extra` creation flags.
    pub(super) fn create_mesh(
        &self,
        d: &MeshPipelineDesc<'_>,
        extra: vk::PipelineCreateFlags2,
        name: &str,
    ) -> Result<vk::Pipeline, Error> {
        let targets = d.raster_targets();
        check(self, targets, name)?;
        let mut parts = Vec::with_capacity(3);
        if let Some(s) = &d.task {
            parts.push((
                vk::ShaderStageFlags::TASK_EXT,
                StageParts::new(self, s, Stage::Task, name)?,
            ));
        }
        parts.push((
            vk::ShaderStageFlags::MESH_EXT,
            StageParts::new(self, &d.mesh, Stage::Mesh, name)?,
        ));
        if let Some(s) = &d.pixel {
            parts.push((
                vk::ShaderStageFlags::FRAGMENT,
                StageParts::new(self, s, Stage::Pixel, name)?,
            ));
        }
        self.create_graphics(targets, &parts, None, extra, name)
    }

    fn create_vertex(
        &self,
        d: &VertexPipelineDesc<'_>,
        extra: vk::PipelineCreateFlags2,
        name: &str,
    ) -> Result<vk::Pipeline, Error> {
        let targets = d.raster_targets();
        check(self, targets, name)?;
        let mut parts = vec![(
            vk::ShaderStageFlags::VERTEX,
            StageParts::new(self, &d.vertex, Stage::Vertex, name)?,
        )];
        if let Some(pixel) = &d.pixel {
            parts.push((
                vk::ShaderStageFlags::FRAGMENT,
                StageParts::new(self, pixel, Stage::Pixel, name)?,
            ));
        }
        self.create_graphics(targets, &parts, Some(d.topology), extra, name)
    }

    fn create_graphics(
        &self,
        d: RasterTargets<'_>,
        parts: &[(vk::ShaderStageFlags, StageParts<'_>)],
        topology: Option<PrimitiveTopology>,
        extra: vk::PipelineCreateFlags2,
        name: &str,
    ) -> Result<vk::Pipeline, Error> {
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
        let mut state = format!("{:?}", (d.colors, d.depth, d.raster, d.samples));
        let kind = match topology {
            Some(topology) => {
                use std::fmt::Write as _;
                write!(state, " {topology:?}").unwrap();
                PipelineKind::Vertex
            }
            None => PipelineKind::Mesh,
        };
        self.create_pipeline(
            &parts.iter().map(|(_, p)| p).collect::<Vec<_>>(),
            state.as_bytes(),
            extra,
            kind,
            name,
            "vkCreateGraphicsPipelines",
            |cache, next| {
                let mut stage_chains: Vec<_> = parts.iter().map(|(_, p)| p.chain()).collect();
                let stages: Vec<_> = stage_chains
                    .iter_mut()
                    .zip(parts)
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
                rendering.p_next = next;
                let mut info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .viewport_state(&viewport)
                    .rasterization_state(&raster)
                    .multisample_state(&multisample)
                    .depth_stencil_state(&depth)
                    .color_blend_state(&blend)
                    .dynamic_state(&dynamic);
                let input = vk::PipelineVertexInputStateCreateInfo::default();
                let assembly;
                if let Some(topology) = topology {
                    assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(
                        match topology {
                            PrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
                            PrimitiveTopology::TriangleStrip => {
                                vk::PrimitiveTopology::TRIANGLE_STRIP
                            }
                        },
                    );
                    info = info
                        .vertex_input_state(&input)
                        .input_assembly_state(&assembly);
                }
                info.p_next = std::ptr::from_ref(&rendering).cast();
                let (created, cpu) = DriverCpu::serial(|| {
                    // SAFETY: valid create info; the SPIR-V was validated by the shader build.
                    unsafe { self.device.create_graphics_pipelines(cache, &[info], None) }
                });
                created.map(|p| (p[0], cpu)).map_err(|(_, e)| e)
            },
            |_| Vec::new(),
        )
        .map(|(raw, _)| raw)
    }
}

impl Device {
    /// Creates a mesh pipeline. Viewport and scissor are dynamic:
    /// [`crate::CommandList::begin_rendering`] sets both to the rendering extent and
    /// [`crate::CommandList::set_scissor`] narrows the scissor. Clip-space y points down the
    /// target: y = -1 is row 0. Threads asking for one pipeline at once compile it once: the
    /// others wait for it.
    pub fn create_mesh_pipeline(
        &self,
        d: &MeshPipelineDesc<'_>,
        name: &str,
    ) -> Result<MeshPipeline, Error> {
        let _zone = crate::zone!("create pipeline", name);
        Ok(MeshPipeline {
            raw: self.shared.create_mesh(d, self.creation_flags, name)?,
            shared: Arc::clone(&self.shared),
        })
    }

    /// Creates a vertex pipeline with the same dynamic viewport and scissor as a mesh pipeline.
    pub fn create_vertex_pipeline(
        &self,
        d: &VertexPipelineDesc<'_>,
        name: &str,
    ) -> Result<VertexPipeline, Error> {
        let _zone = crate::zone!("create pipeline", name);
        Ok(VertexPipeline {
            raw: self.shared.create_vertex(d, self.creation_flags, name)?,
            shared: Arc::clone(&self.shared),
        })
    }
}
