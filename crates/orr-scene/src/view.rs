//! Views: what a camera renders, its history between frames and the constants kernels read.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec3, Mat3, Mat4, Quat, UVec2, Vec2, Vec4};
use orr_gpu::{Allocation, GpuPtr, Texture};

use crate::raster::ViewTargets;
use crate::world::WorldPos;
use crate::{GBufferGpu, SceneId};

/// Per-view constants (`view.slang`). View space has no translation: x right, y up, looking
/// down -z. Clip space is Vulkan's (y down) with reverse Z and an infinite far plane.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewConstants {
    pub clip_from_view: [[f32; 4]; 4],
    pub clip_from_view_jittered: [[f32; 4]; 4],
    /// Inverse of `clip_from_view_jittered`.
    pub view_from_clip: [[f32; 4]; 4],
    pub prev_clip_from_view: [[f32; 4]; 4],
    /// Rows of the rotation; the translation column is 0.
    pub view_from_world_rotation: [[f32; 4]; 3],
    pub prev_view_from_world_rotation: [[f32; 4]; 3],
    /// The camera in render space.
    pub eye_render: [f32; 4],
    /// This frame's camera minus last frame's, w 0.
    pub eye_motion: [f32; 4],
    /// w = cell_log2.
    pub eye_cell: [i32; 4],
    pub eye_offset: [f32; 4],
    /// Render pixels.
    pub jitter: [f32; 2],
    pub extent: [u32; 2],
    /// Two slots; this render reads `exposure_slot` and writes the other. The slots swap
    /// each time the view renders.
    pub exposure: GpuPtr<ExposureState>,
    pub exposure_slot: u32,
    pub frame_index: u32,
    pub delta_time: f32,
    /// Added to the level of detail of every primary-hit texture lookup.
    pub mip_bias: f32,
}

/// One exposure slot of a view (`view.slang`); orr-output writes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ExposureState {
    pub pre_exposure: f32,
    pub prev_pre_exposure: f32,
    pub adapted_ev100: f32,
    pub target_ev100: f32,
    pub long_term_ev100: f32,
    pub white_point_xy: [f32; 2],
    pub initialized: u32,
}

/// A persistent camera's state between frames: its previous camera and exposure slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewHistoryId(pub u32);

/// A perspective projection with reverse Z and an infinite far plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projection {
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Near plane distance, m.
    pub near: f32,
}

impl Projection {
    /// Clip from view for an image of `extent`, unjittered: Vulkan clip space with y down.
    pub fn clip_from_view(&self, extent: UVec2) -> Mat4 {
        clip_from_view(self, extent)
    }
}

/// The lens the camera effects read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    /// Focal length = sensor_width / (2 tan(fov_x / 2)).
    pub sensor_width_mm: f32,
    /// Fraction of the frame interval the shutter is open; 0 = no motion blur.
    pub shutter: f32,
    pub depth_of_field: Option<DepthOfField>,
    pub glare: Option<Glare>,
    pub distortion: Option<LensDistortion>,
}

impl Default for Lens {
    fn default() -> Self {
        Self {
            sensor_width_mm: 36.0,
            shutter: 0.5,
            depth_of_field: None,
            glare: Some(Glare { strength: 1.0 }),
            distortion: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthOfField {
    /// Depth of field only; exposure is metered.
    pub f_number: f32,
    pub focus: Focus,
    /// Aperture blades; 0 = round.
    pub blades: u32,
    /// The aperture is round at or below this f-number.
    pub round_at_f_number: f32,
    /// Largest circle of confusion, as a fraction of the output width.
    pub max_coc: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Focus {
    Manual {
        distance_m: f32,
    },
    /// Focuses on the depth at `uv`, following it at `speed`.
    Auto {
        uv: [f32; 2],
        speed: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glare {
    /// 1.0 = the fitted PSF energy.
    pub strength: f32,
}

/// Brown-Conrady radial terms on the radius normalised to the half-diagonal; lateral
/// chromatic aberration scales the red and blue radius relative to green.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensDistortion {
    pub k1: f32,
    pub k2: f32,
    pub chromatic_aberration: f32,
}

/// Where a view's image goes.
#[derive(Clone, Copy)]
pub enum ViewTarget<'a> {
    /// The frame's display.
    Display,
    Texture(&'a Texture),
    /// Headless: the view stops after ray reconstruction.
    AfterRr,
}

/// One camera rendering one scene this frame.
#[derive(Clone, Copy)]
pub struct ViewDesc<'a> {
    /// Must be listed in the frame.
    pub scene: SceneId,
    pub history: ViewHistoryId,
    pub position: DVec3,
    /// World from view.
    pub rotation: Quat,
    pub projection: Projection,
    pub lens: Lens,
    pub render_extent: UVec2,
    /// Render pixels: each sample sits at its pixel centre minus the jitter, y down.
    pub jitter: Vec2,
    /// A camera cut: the previous camera is taken to be this one.
    pub reset_history: bool,
    /// Texture level-of-detail bias for an upscaled image: log2(render width / output
    /// width), 0 without upscaling.
    pub mip_bias: f32,
    pub target: ViewTarget<'a>,
}

/// What a view's raster half hands to the passes after it. The targets stay valid until
/// the frame's GPU work completes.
#[derive(Clone)]
pub struct RasterView {
    pub view: GpuPtr<ViewConstants>,
    pub scene: GpuPtr<crate::SceneRoot>,
    /// The G-buffer as kernels read it, through `load_gsurface`.
    pub gbuffer: GpuPtr<GBufferGpu>,
    /// What `gbuffer` holds: the visibility buffer, the visible clusters and the targets'
    /// sampled views.
    pub targets: GBufferGpu,
    /// D32, reverse Z: 0 where nothing was drawn.
    pub depth: Arc<Texture>,
    /// RG16F: previous uv minus current uv.
    pub motion: Arc<Texture>,
    pub gbuf_a: Arc<Texture>,
    pub gbuf_b: Arc<Texture>,
    /// Present while a resident material may need it.
    pub gbuf_x: Option<Arc<Texture>>,
    /// R32F: the farthest reverse-Z depth per texel, mip 0 at the render extent.
    pub hzb: Arc<Texture>,
    /// Visible clusters written, then visible clusters and cluster work dropped for want
    /// of room (`Budgets::visible_clusters`, `Budgets::cluster_work`), 1, cluster work
    /// items asked for, mesh-shader opaque and alpha draws, software-rastered clusters, and
    /// visible particle clusters.
    pub counts: GpuPtr<u32>,
    /// The view's raster state: `gbuffer` and the culling lists, the particle list among
    /// them.
    pub raster: GpuPtr<crate::RasterGpu>,
    /// The view history it rendered with.
    pub history: ViewHistoryId,
    /// The scene's sky statistics, which lighting's scene pass writes and exposure reads.
    pub sky_stats: GpuPtr<crate::SkyStats>,
    /// The scene focus in render space; the view's camera while the scene has none.
    pub focus: [f32; 3],
}

#[derive(Clone, Copy)]
struct Camera {
    scene: SceneId,
    position: DVec3,
    rotation: Quat,
    clip_from_view: Mat4,
}

pub(crate) struct ViewHistory {
    pub exposure: Allocation<ExposureState>,
    /// The exposure slot the view's next render reads.
    slot: u32,
    previous: Option<Camera>,
    pub targets: Option<ViewTargets>,
}

impl ViewHistory {
    pub(crate) fn new(exposure: Allocation<ExposureState>) -> Self {
        Self {
            exposure,
            slot: 0,
            previous: None,
            targets: None,
        }
    }
}

fn rows4(m: Mat4) -> [[f32; 4]; 4] {
    m.transpose().to_cols_array_2d()
}

fn rotation_rows(m: Mat3) -> [[f32; 4]; 3] {
    let r = m.transpose().to_cols_array_2d();
    r.map(|row| [row[0], row[1], row[2], 0.0])
}

/// `clip_from_view` for Vulkan clip space: reverse Z, infinite far, y down.
pub(crate) fn clip_from_view(p: &Projection, extent: UVec2) -> Mat4 {
    let aspect = extent.x as f32 / extent.y.max(1) as f32;
    let f = 1.0 / (0.5 * p.fov_y).tan();
    let rows = [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, 0.0, p.near],
        [0.0, 0.0, -1.0, 0.0],
    ];
    Mat4::from_cols_array_2d(&rows).transpose()
}

/// Shifts the image by `jitter` render pixels, so each sample lands at its pixel centre
/// minus the jitter.
pub(crate) fn jittered(clip_from_view: Mat4, jitter: Vec2, extent: UVec2) -> Mat4 {
    let ndc = jitter * 2.0 / extent.as_vec2();
    let mut shift = Mat4::IDENTITY;
    shift.w_axis = Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
    shift * clip_from_view
}

/// Builds the view's constants and moves its history on to this frame.
pub(crate) fn view_constants(
    desc: &ViewDesc<'_>,
    history: &mut ViewHistory,
    origin_cell: IVec3,
    cell_log2: u32,
    frame: u64,
    delta_time: f32,
) -> ViewConstants {
    let clip = clip_from_view(&desc.projection, desc.render_extent);
    let clip_jittered = jittered(clip, desc.jitter, desc.render_extent);
    let current = Camera {
        scene: desc.scene,
        position: desc.position,
        rotation: desc.rotation,
        clip_from_view: clip,
    };
    let previous = match history.previous {
        Some(p) if p.scene == desc.scene && !desc.reset_history => p,
        _ => current,
    };
    history.previous = Some(current);
    let exposure_slot = history.slot;
    history.slot ^= 1;

    let eye = WorldPos::new(desc.position, cell_log2);
    let origin = WorldPos {
        cell: origin_cell,
        offset: glam::Vec3::ZERO,
    };
    let view_from_world = Mat3::from_quat(desc.rotation.inverse());
    let prev_view_from_world = Mat3::from_quat(previous.rotation.inverse());
    ViewConstants {
        clip_from_view: rows4(clip),
        clip_from_view_jittered: rows4(clip_jittered),
        view_from_clip: rows4(clip_jittered.inverse()),
        prev_clip_from_view: rows4(previous.clip_from_view),
        view_from_world_rotation: rotation_rows(view_from_world),
        prev_view_from_world_rotation: rotation_rows(prev_view_from_world),
        eye_render: eye.relative_to(origin, cell_log2).extend(0.0).to_array(),
        eye_motion: (desc.position - previous.position)
            .as_vec3()
            .extend(0.0)
            .to_array(),
        eye_cell: eye.cell.extend(cell_log2 as i32).to_array(),
        eye_offset: eye.offset.extend(0.0).to_array(),
        jitter: desc.jitter.to_array(),
        extent: desc.render_extent.to_array(),
        exposure: history.exposure.gpu,
        exposure_slot,
        frame_index: frame as u32,
        delta_time,
        mip_bias: desc.mip_bias,
    }
}
