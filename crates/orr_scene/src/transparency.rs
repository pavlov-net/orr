//! The transparency point: after a view's opaque lighting, the consumer's hook draws into a
//! premultiplied overlay at the render extent, which is then composited over the view's
//! radiance. Particle producer entries are drawn only here for primary visibility; the
//! view's culling lists their visible clusters for the hook's mesh draws (`particles.slang`).

use orr_gpu::{CommandList, Device, GpuPtr, MeshArgs, Texture, TextureIndex};

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
    /// One mesh workgroup per visible particle cluster, for
    /// `draw_mesh_tasks_indirect_count` with `particle_draws` as the count.
    pub particle_args: GpuPtr<MeshArgs>,
    pub particle_draws: GpuPtr<u32>,
    pub(crate) device: &'a Device,
    pub(crate) window: &'a UploadWindow,
}

impl TransparencyPass<'_, '_> {
    /// `root` in this frame's upload window at an address a draw can take as its root.
    pub fn put_root<T: bytemuck::Pod>(&self, root: &T) -> Result<GpuPtr<T>> {
        self.window.put_root(self.device, root)
    }
}
