//! Motion blur at output resolution, between adapt and the other camera effects.

use orr_scene::ViewConstants;
use orr_scene::orr_gpu::{
    CommandList, ComputePipeline, Device, Format, GpuPtr, Hazards, ShaderStage, SpecData, Stages,
    TextureIndex,
};

use crate::passes::groups;
use crate::shaders::post::motion_blur::{self, Root};
use crate::{ChainImages, Image, Result, ViewOutput};

/// Pixels across a velocity tile (`TILE` in `motion_blur.slang`).
const TILE: u32 = 20;

/// The longest half vector, as a fraction of the output width.
pub const MOTION_BLUR_MAX_RADIUS: f32 = 0.03;

/// Blurring one view's radiance along its screen motion over the open shutter.
#[derive(Clone, Copy, Debug)]
pub struct MotionBlurPass<'a> {
    /// Lighting's `rr_depth`: R32F reverse-Z at render resolution.
    pub rr_depth: TextureIndex,
    /// Lighting's `rr_motion`: RG16F previous minus current UV at render resolution.
    pub rr_motion: TextureIndex,
    /// The output extent.
    pub extent: [u32; 2],
    /// The render's view constants: the render extent and jitter.
    pub view: GpuPtr<ViewConstants>,
    /// `Lens::shutter`: the fraction of the rendered-frame interval the shutter is open.
    pub shutter: f32,
    /// Frames generated after this one; 0 with frame generation off.
    pub generated: u32,
    /// The view's history reset this frame; motion vectors do not hold.
    pub history_reset: bool,
    /// orr's frame id.
    pub frame: u64,
    pub output: &'a ViewOutput,
}

pub(crate) struct MotionBlurKernels {
    tile_rows: ComputePipeline,
    tile_max: ComputePipeline,
    neighbors: ComputePipeline,
    gather: ComputePipeline,
}

impl MotionBlurKernels {
    pub(crate) fn new(gpu: &Device) -> Result<Self> {
        let pipeline = |entry, name| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(&motion_blur::MODULE, entry, SpecData::default())?;
            Ok(gpu.create_compute_pipeline(&stage, name)?)
        };
        Ok(Self {
            tile_rows: pipeline("tile_rows", "motion_blur_tile_rows")?,
            tile_max: pipeline("tile_max", "motion_blur_tile_max")?,
            neighbors: pipeline("neighbors", "motion_blur_neighbors")?,
            gather: pipeline("gather", "motion_blur_gather")?,
        })
    }

    /// The pass's root, reading `chain.rr_output` and writing `chain.linear`.
    pub(crate) fn root(
        targets: &MotionBlurTargets,
        pass: &MotionBlurPass<'_>,
        chain: &ChainImages,
    ) -> Root {
        let [width, height] = pass.extent;
        let [tiles_x, tiles_y] = tiles(pass.extent);
        let max_radius = MOTION_BLUR_MAX_RADIUS * width as f32;
        Root {
            view: pass.view,
            rr_depth: pass.rr_depth,
            rr_motion: pass.rr_motion,
            source: chain.rr_output,
            target: chain.linear_store,
            tile_rows: targets.tile_rows.sampled,
            tile_rows_store: targets.tile_rows.storage,
            tiles: targets.tiles.sampled,
            tiles_store: targets.tiles.storage,
            neighbors: targets.neighbors.sampled,
            neighbors_store: targets.neighbors.storage,
            width,
            height,
            tiles_x,
            tiles_y,
            reach: 1 + (max_radius / TILE as f32) as i32,
            scale: 0.5 * pass.shutter / (pass.generated + 1) as f32,
            max_radius,
            pairs: (pass.output.post.motion_blur_samples / 2).max(1),
        }
    }

    /// Records the pass's four dispatches, then a barrier.
    pub(crate) fn record(&self, list: &mut CommandList<'_>, root: GpuPtr<Root>, extent: [u32; 2]) {
        let height = extent[1];
        let [tiles_x, tiles_y] = tiles(extent);
        let step = |list: &mut CommandList<'_>| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        };
        list.begin_pass("motion_blur", false);
        list.dispatch(&self.tile_rows, root, groups([tiles_x, height], 8));
        step(list);
        list.dispatch(&self.tile_max, root, groups([tiles_x, tiles_y], 8));
        step(list);
        list.dispatch(&self.neighbors, root, groups([tiles_x, tiles_y], 8));
        step(list);
        list.dispatch(&self.gather, root, groups(extent, 8));
        list.end_pass();
        step(list);
    }
}

fn tiles(extent: [u32; 2]) -> [u32; 2] {
    extent.map(|n| n.div_ceil(TILE))
}

/// A view's velocity tiles, allocated while it blurs.
pub(crate) struct MotionBlurTargets {
    tile_rows: Image,
    tiles: Image,
    neighbors: Image,
    extent: [u32; 2],
    pub(crate) fresh: bool,
}

impl MotionBlurTargets {
    pub(crate) fn new(gpu: &Device, extent: [u32; 2]) -> Result<Self> {
        let [tx, ty] = tiles(extent);
        Ok(Self {
            tile_rows: Image::new(gpu, Format::Rg16Float, [tx, extent[1]], "blur tile rows")?,
            tiles: Image::new(gpu, Format::Rg16Float, [tx, ty], "blur tiles")?,
            neighbors: Image::new(gpu, Format::Rgba16Float, [tx, ty], "blur neighbors")?,
            extent,
            fresh: true,
        })
    }

    pub(crate) fn extent(&self) -> [u32; 2] {
        self.extent
    }

    pub(crate) fn discard(&self, list: &mut CommandList<'_>) {
        for image in [&self.tile_rows, &self.tiles, &self.neighbors] {
            list.discard(image.texture());
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        [&self.tile_rows, &self.tiles, &self.neighbors]
            .iter()
            .map(|i| i.bytes())
            .sum()
    }
}
