//! The transparency point: after a view's opaque lighting, the consumer's hook draws into a
//! premultiplied overlay at the render extent, which is then composited over the view's
//! radiance. Particle producer entries are drawn only here for primary visibility; the
//! view's culling lists their visible clusters for the hook's mesh draws (`particles.slang`).

use orr_gpu::{CommandList, Device, GpuPtr, MeshArgs, MeshPipeline, Texture, TextureIndex};

use crate::memory::UploadWindow;
use crate::{RasterGpu, Result, SceneRoot, ViewConstants};

/// What a transparency hook draws with.
pub struct TransparencyPass<'a, 'q> {
    pub cmd: &'a mut CommandList<'q>,
    /// RGBA16F at `extent`, cleared to 0, a color target: premultiplied pre-exposed radiance,
    /// alpha = coverage, blended over front to back or sorted back to front by the hook.
    pub overlay: &'a Texture,
    /// The view's D32 reverse-Z depth, for a read-only depth test against the opaque surface.
    pub depth: &'a Texture,
    /// Sampled view of `depth`, for soft fades.
    pub depth_index: TextureIndex,
    pub extent: [u32; 2],
    pub view: GpuPtr<ViewConstants>,
    pub scene: GpuPtr<SceneRoot>,
    pub raster: GpuPtr<RasterGpu>,
    pub(crate) particle_draws: Vec<ParticleDraws>,
    pub(crate) device: &'a Device,
    pub(crate) window: &'a UploadWindow,
}

/// One mesh draw over the view's particle list, with the raster record it takes.
pub(crate) struct ParticleDraws {
    pub(crate) raster: GpuPtr<RasterGpu>,
    pub(crate) args: GpuPtr<MeshArgs>,
    pub(crate) count: GpuPtr<u32>,
}

impl TransparencyPass<'_, '_> {
    /// `root` in this frame's upload window at an address a draw can take as its root.
    pub fn put_root<T: bytemuck::Pod>(&self, root: &T) -> Result<GpuPtr<T>> {
        self.window.put_root(self.device, root)
    }

    /// Draws one mesh workgroup of `pipeline` per visible particle cluster, inside rendering,
    /// in as many draws as the device bounds them to. Each draw's root is what `root` makes of
    /// the raster record that draw takes, to give `particle_draw` in place of
    /// [`Self::raster`].
    pub fn draw_particles<T: bytemuck::Pod>(
        &mut self,
        pipeline: &MeshPipeline,
        root: impl Fn(GpuPtr<RasterGpu>) -> T,
    ) -> Result<()> {
        for d in &self.particle_draws {
            let r = self.window.put_root(self.device, &root(d.raster))?;
            self.cmd
                .draw_mesh_tasks_indirect_count(pipeline, r, d.args, d.count, 1);
        }
        Ok(())
    }
}
