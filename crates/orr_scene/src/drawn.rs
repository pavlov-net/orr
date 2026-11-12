//! The opaque point: inside each pass of a view's raster, the consumer's hook draws its drawn
//! producer entries into the visibility buffer with its own task and mesh pipelines
//! (`drawn_raster.slang`).

use orr_gpu::{CommandList, Device, GpuPtr};

use crate::memory::UploadWindow;
use crate::{RasterGpu, Result, SceneRoot, ViewConstants};

/// What an opaque hook draws with: one pass of a view's raster.
pub struct OpaquePass<'a, 'q> {
    /// Inside rendering over `extent` with no targets, so the hook records only mesh draws,
    /// of pipelines made with no color target, no depth target and one sample
    /// ([`crate::Scenes::mesh_pipeline`], which hot reload reaches).
    pub cmd: &'a mut CommandList<'q>,
    pub extent: [u32; 2],
    pub view: GpuPtr<ViewConstants>,
    pub scene: GpuPtr<SceneRoot>,
    /// The pass's raster record: the view's visible list, visibility buffer and HZB.
    pub raster: GpuPtr<RasterGpu>,
    /// The second of the raster's two passes, whose HZB holds what the first drew. The first
    /// holds last frame's while the record's flags say so, and the view then runs both; a
    /// draw that culls against the HZB draws in the second pass only what its first rejected
    /// (`drawn_occluded`).
    pub second: bool,
    pub(crate) device: &'a Device,
    pub(crate) window: &'a UploadWindow,
}

impl OpaquePass<'_, '_> {
    /// `value` in this frame's upload window, 16-byte aligned; the GPU can read it until the
    /// frame completes.
    pub fn put<T: bytemuck::Pod>(&self, value: &T) -> Result<GpuPtr<T>> {
        self.window.put(self.device, value)
    }

    /// `root` in this frame's upload window at an address a draw can take as its root.
    pub fn put_root<T: bytemuck::Pod>(&self, root: &T) -> Result<GpuPtr<T>> {
        self.window.put_root(self.device, root)
    }
}

/// A view's opaque hook, run once per pass of its raster.
pub type OpaqueHook<'a> = dyn FnMut(&mut OpaquePass<'_, '_>) + 'a;
