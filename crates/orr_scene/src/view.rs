//! Views: what a camera renders, its history between frames and the constants kernels read.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use orr_gpu::{Allocation, DispatchArgs, GpuPtr, Texture};
use orr_math::camera::rh::proj::vulkan;
use orr_math::{Affine3, DVec3, IVec3, Mat4, Quat, UVec2, Vec2, Vec4};
use orr_pack::AffineRows;

use crate::arena::ViewArena;
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
    /// Last frame's projection at this frame's output aspect, unjittered: a change of aspect
    /// moves nothing, while a change of field of view does.
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
    /// [`TextureFilter`] as its `view.slang` constant.
    pub texture_filter: u32,
    pub _pad: u32,
}

/// How a view's material textures are filtered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureFilter {
    /// The samplers' trilinear and anisotropic filtering.
    #[default]
    Hardware,
    /// Under magnification, in the material resolve and at ray hits, one texel picked at
    /// random per lookup so its expected value is the bicubic (cubic B-spline) filtered
    /// texture; elsewhere the samplers' filtering. Its noise needs a temporal resolve such
    /// as DLSS ray reconstruction to average it.
    Stochastic,
}

/// One exposure slot of a view (`view.slang`); orr_output writes it.
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
    /// Clip from view for an output image of `extent`, unjittered: Vulkan clip space with y
    /// down. Only its aspect ratio matters.
    pub fn clip_from_view(&self, extent: UVec2) -> Mat4 {
        clip_from_view(self, extent)
    }

    /// Pixels per radian at the center of an image `height` pixels tall.
    pub fn pixels_per_radian(&self, height: u32) -> f32 {
        0.5 * height as f32 / (0.5 * self.fov_y).tan()
    }
}

/// The lens the camera effects read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    /// Focal length = sensor_width / (2 tan(fov_x / 2)).
    pub sensor_width_mm: f32,
    /// With the output height it gives the pixel pitch, which sets how many pixels glare
    /// reaches: the same share of the output's height at any aspect ratio. It need not follow
    /// `sensor_width_mm`, and is not meant to change with the view's shape.
    pub sensor_height_mm: f32,
    /// Fraction of the time between presented images the shutter is open; 0 = no motion blur.
    pub shutter: f32,
    pub depth_of_field: Option<DepthOfField>,
    pub glare: Option<Glare>,
    pub distortion: Option<LensDistortion>,
}

impl Default for Lens {
    fn default() -> Self {
        Self {
            sensor_width_mm: 36.0,
            sensor_height_mm: 20.25,
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
    /// 1.0 = the fitted PSF energy. Only light exposed past about SDR's white spreads, so a
    /// scene with none shows no glare. The image's total energy is the same at any strength.
    pub strength: f32,
}

/// Lens distortion applied in tonemap's gather. The output pixel at radius `r` from the
/// center, in half-diagonals, takes green from radius `s r (1 + k1 r^2 + k2 r^4)`: positive
/// `k1` gives barrel distortion, negative pincushion. `s` scales the image so every output
/// pixel samples inside the rendered frame, the border reaching its edge. The radius must
/// map monotonically over the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensDistortion {
    pub k1: f32,
    pub k2: f32,
    /// Lateral chromatic aberration: red samples at green's radius times
    /// `1 + chromatic_aberration`, blue times `1 - chromatic_aberration`.
    pub chromatic_aberration: f32,
}

/// One camera rendering one scene this frame.
#[derive(Clone, Copy)]
pub struct ViewDesc {
    /// Must be listed in the frame.
    pub scene: SceneId,
    pub history: ViewHistoryId,
    pub position: DVec3,
    /// World from view.
    pub rotation: Quat,
    pub projection: Projection,
    pub lens: Lens,
    pub render_extent: UVec2,
    /// The extent of the image the view ends up in, whose aspect ratio the projection takes.
    /// An upscaler's render extent only rounds to it.
    pub output_extent: UVec2,
    /// Render pixels: each sample sits at its pixel center minus the jitter, y down.
    pub jitter: Vec2,
    /// A camera cut: the previous camera is taken to be this one; occlusion culling, frame
    /// generation and lighting drop their history. A view's first render is a cut without it.
    pub camera_cut: bool,
    /// Added to the level of detail of primary-hit texture lookups, as an upscaler asks for
    /// its render size; 0 without one.
    pub mip_bias: f32,
    /// How the view filters material textures.
    pub texture_filter: TextureFilter,
}

/// What a view's raster half hands to the passes after it. Depth, motion and the HZB stay
/// valid until the frame's GPU work completes. The G-buffer textures, the visibility buffer
/// and the visible clusters live in the view's arena: when the view's render enters
/// [`crate::ViewStep::Rr`] or a later step, other targets may have taken their memory.
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
    /// RG16F: previous uv minus current uv, both at this frame's output aspect. An image kept
    /// from before a change of output extent is framed differently and needs a reset.
    pub motion: Arc<Texture>,
    pub gbuf_a: Arc<Texture>,
    pub gbuf_b: Arc<Texture>,
    /// Present while a resident material may need it.
    pub gbuf_x: Option<Arc<Texture>>,
    /// With `gbuf_x`.
    pub gbuf_y: Option<Arc<Texture>>,
    /// R32F: the farthest reverse-Z depth per texel, mip 0 at the render extent.
    pub hzb: Arc<Texture>,
    /// Visible clusters written, then visible clusters and cluster work dropped for want
    /// of room (`Budgets::visible_clusters`, `Budgets::cluster_work`), 1, cluster work
    /// items asked for, mesh-shader opaque and alpha draws, software-rastered clusters, and
    /// visible particle clusters; entry 16 counts the visible dielectric and presence
    /// clusters that drew into `targets.layer_tiles`, and entry 17 the visible smoke clusters.
    pub counts: GpuPtr<u32>,
    /// The view's raster state: `gbuffer` and the culling lists, the particle list among
    /// them.
    pub raster: GpuPtr<crate::RasterGpu>,
    /// Workgroups of one per cluster of `raster`'s smoke list, when the scene holds a smoke
    /// entry.
    pub smoke_args: Option<GpuPtr<DispatchArgs>>,
    /// Whether paths through any material the scene holds may take a random walk.
    pub walking_materials: bool,
    /// Whether a surface of any material the scene holds bounds an interior: a dielectric that
    /// is not thin-walled. Without one no point lies in an interior the view does not name.
    pub interior_materials: bool,
    /// Whether one of those interiors scatters, read without a surface point: its material's
    /// transmission depth and scatter are above 0.
    pub scattering_interiors: bool,
    /// Whether the scene holds a medium proxy.
    pub medium_proxies: bool,
    /// Whether any material the scene holds disperses: its transmission has a dispersion scale
    /// and an Abbe number above 0.
    pub dispersive_materials: bool,
    /// The view history it rendered with.
    pub history: ViewHistoryId,
    /// The scene's sky statistics, which lighting's scene pass writes and exposure reads.
    pub sky_stats: GpuPtr<crate::SkyStats>,
    /// The scene focus in render space; the view's camera while the scene has none.
    pub focus: [f32; 3],
}

pub(crate) struct ViewHistory {
    pub(crate) exposure: Allocation<ExposureState>,
    /// The exposure slot the view's next render reads.
    slot: u32,
    previous: Option<ViewDesc>,
    pub(crate) targets: Option<ViewTargets>,
    /// The transient targets of the view's render, orr_scene's and the later passes'.
    pub(crate) arena: ViewArena,
}

impl ViewHistory {
    pub(crate) fn new(exposure: Allocation<ExposureState>, arena: ViewArena) -> Self {
        Self {
            exposure,
            slot: 0,
            previous: None,
            targets: None,
            arena,
        }
    }
}

/// `clip_from_view` for Vulkan clip space: reverse Z, infinite far, y down.
fn clip_from_view(p: &Projection, extent: UVec2) -> Mat4 {
    let aspect = extent.x as f32 / extent.y.max(1) as f32;
    vulkan::perspective_infinite_reverse(p.fov_y, aspect, p.near)
}

/// Shifts the image by `jitter` render pixels, so each sample lands at its pixel center
/// minus the jitter.
fn jittered(clip_from_view: Mat4, jitter: Vec2, extent: UVec2) -> Mat4 {
    let ndc = jitter * 2.0 / extent.as_vec2();
    let mut shift = Mat4::IDENTITY;
    shift.w_axis = Vec4::new(ndc.x, ndc.y, 0.0, 1.0);
    shift * clip_from_view
}

/// Builds the view's constants and moves its history on to this frame.
pub(crate) fn view_constants(
    desc: &ViewDesc,
    history: &mut ViewHistory,
    origin_cell: IVec3,
    cell_log2: u32,
    frame: u64,
    delta_time: f32,
) -> ViewConstants {
    let clip = clip_from_view(&desc.projection, desc.output_extent);
    let clip_jittered = jittered(clip, desc.jitter, desc.render_extent);
    let previous = match history.previous {
        Some(p) if p.scene == desc.scene && !desc.camera_cut => p,
        _ => *desc,
    };
    history.previous = Some(*desc);
    let exposure_slot = history.slot;
    history.slot ^= 1;

    let eye = WorldPos::new(desc.position, cell_log2);
    let origin = WorldPos {
        cell: origin_cell,
        offset: orr_math::Vec3::ZERO,
    };
    let rows = |m: Mat4| m.transpose().to_cols_array_2d();
    let rotation_rows = |q: Quat| AffineRows::from(Affine3::from_quat(q.inverse())).rows;
    ViewConstants {
        clip_from_view: rows(clip),
        clip_from_view_jittered: rows(clip_jittered),
        view_from_clip: rows(clip_jittered.inverse()),
        prev_clip_from_view: rows(clip_from_view(&previous.projection, desc.output_extent)),
        view_from_world_rotation: rotation_rows(desc.rotation),
        prev_view_from_world_rotation: rotation_rows(previous.rotation),
        eye_render: eye.relative_to(origin, cell_log2).extend(0.0).to_array(),
        eye_motion: (desc.position - previous.position)
            .as_vec3()
            .extend(0.0)
            .to_array(),
        eye_cell: eye.cell.extend(cell_log2 as i32).to_array(),
        eye_offset: eye.offset.extend(0.0).to_array(),
        jitter: desc.jitter.to_array(),
        extent: desc.render_extent.to_array(),
        exposure: history.exposure.gpu(),
        exposure_slot,
        frame_index: frame as u32,
        delta_time,
        mip_bias: desc.mip_bias,
        texture_filter: match desc.texture_filter {
            TextureFilter::Hardware => 0,
            TextureFilter::Stochastic => 1,
        },
        _pad: 0,
    }
}
