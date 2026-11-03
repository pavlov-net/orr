//! The stock draw of a view's particle producer entries at the transparency point.

use orr_scene::TransparencyPass;
use orr_scene::orr_gpu::{
    Blend, ColorAttachment, ColorLoad, ColorMask, ColorTarget, CompareOp, DepthAttachment,
    DepthLoad, DepthTarget, Device, Format, GpuPtr, MeshPipeline, MeshPipelineDesc, RasterState,
    Rendering, ShaderStage, SpecData,
};

use crate::shaders::media::particle_draw;
use crate::shaders::types::ParticleLighting;
use crate::{Error, KernelSet, module};

/// Draws the visible clusters of a view's particle producer entries into its transparency
/// overlay: each covered pixel in front of the opaque depth takes its surface's base color as
/// albedo, its emission and its material's opacity, lit and fogged by `particle_lighting`'s
/// `particle_shade`, blended premultiplied in the order the clusters arrive. Fragments above 0.1
/// coverage also enter the view's front layer ([`crate::ViewPass::overlay_front`]) where it is on.
pub struct ParticleDraw {
    pipeline: MeshPipeline,
}

impl ParticleDraw {
    /// `kernels` replaces the draw's module as [`crate::SceneLighting::new`] does, for builds
    /// that link another surface or medium.
    pub fn new(device: &Device, kernels: &KernelSet) -> Result<Self, Error> {
        let m = module(kernels, &particle_draw::MODULE)?;
        let pipeline = device.create_mesh_pipeline(
            &MeshPipelineDesc {
                task: None,
                mesh: ShaderStage::new(m, "particle_mesh", SpecData::default())?,
                pixel: Some(ShaderStage::new(m, "particle_pixel", SpecData::default())?),
                colors: &[ColorTarget {
                    format: Format::Rgba16Float,
                    blend: Some(Blend::PREMULTIPLIED),
                    write: ColorMask::ALL,
                }],
                depth: Some(DepthTarget {
                    format: Format::D32Float,
                    compare: CompareOp::GreaterOrEqual,
                    write: false,
                }),
                raster: RasterState::default(),
                samples: 1,
            },
            "particles",
        )?;
        Ok(Self { pipeline })
    }

    /// Records the draw into `pass` with `lighting`, the view's
    /// [`crate::LightingView::particle_lighting`] of the frame, and `anisotropy`, the
    /// Henyey-Greenstein parameter of the particles' response to directional light. Draws
    /// nothing when `lighting` is null.
    pub fn record(
        &self,
        pass: &mut TransparencyPass<'_, '_>,
        lighting: GpuPtr<ParticleLighting>,
        anisotropy: f32,
    ) -> Result<(), Error> {
        if lighting == GpuPtr::NULL {
            return Ok(());
        }
        let root = pass.put_root(&particle_draw::Root {
            scene: pass.scene,
            view: pass.view,
            raster: pass.raster,
            lighting,
            anisotropy,
            _pad0: 0,
            _pad1: 0,
        })?;
        pass.cmd.begin_rendering(&Rendering {
            extent: pass.extent,
            colors: &[ColorAttachment {
                texture: pass.overlay,
                load: ColorLoad::Keep,
                store: true,
            }],
            depth: Some(DepthAttachment {
                texture: pass.depth,
                load: DepthLoad::Keep,
                store: true,
            }),
        });
        pass.cmd.draw_mesh_tasks_indirect_count(
            &self.pipeline,
            root,
            pass.particle_args,
            pass.particle_draws,
            1,
        );
        pass.cmd.end_rendering();
        Ok(())
    }
}
