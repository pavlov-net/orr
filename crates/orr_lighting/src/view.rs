//! Per-view lighting: the settings of one `render_view` call, the view's lighting targets and
//! history, the world cache's update paths, the opaque path dispatch, the dielectric dispatch
//! and the guide resolve.

use orr_math::{DVec3, Quat};
use orr_scene::orr_gpu::{
    self, Allocation, CommandList, ComputePipeline, Device, DispatchArgs, Format, GpuPtr, Hazards,
    MemoryKind, RayDispatchPipeline, ShaderModule, ShaderStage, SpecData, Stages,
    StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage,
};
use orr_scene::{
    ArenaRequest, ArenaTexture, GBufferGpu, MaterialId, SceneId, SceneRoot, Span, ViewArena,
    ViewConstants, ViewDesc, ViewStep,
};

use crate::media::{MediaKernels, MediaSettings, MediaStats, MediaView};
use crate::scene::{FRAME_SLOTS, Ring, device_local};
use crate::shaders::dielectric::{dispatch, tiles};
use crate::shaders::guides::{frost, resolve};
use crate::shaders::light_sampling::local;
use crate::shaders::path::opaque;
use crate::shaders::types::{
    FrostRoot, GuideRoot, LayerGuide, LayerTilesRoot, LocalLights, PathRoot,
};
use crate::shaders::world_cache::update;
use crate::subsurface::{CapTailTable, SSS_STEPS, SubsurfaceDepth};
use crate::{BsdfTables, Error, KernelSet, PathEntry, SceneLighting, module, pipeline};

/// Lighting settings of one view, passed with every render of it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ViewLighting {
    /// Light candidates resampled at a path's first vertex.
    pub primary_di_candidates: u32,
    /// Light candidates resampled at its later vertices.
    pub bounce_di_candidates: u32,
    /// Share of the first vertex's candidates, past the directional lights' share, drawn from
    /// the lights that its screen tile's pixels found in the view's last frame, at most
    /// [`MAX_LOCAL_LIGHT_SHARE`]; the rest come from the scene's light selection. 0 turns the
    /// local samplers off.
    pub local_light_share: f32,
    /// BSDF bounces after the first vertex's; the last bounce still adds the emission and
    /// sky its ray reaches. Dielectric vertices do not count: a path through dielectrics has
    /// this many bounces after its first opaque vertex or interior scatter.
    pub max_bounces: u32,
    /// Dielectric and presence vertices a path may have, its first included.
    pub dielectric_depth: u32,
    /// Whether dielectrics with a dispersion (an Abbe number and a dispersion scale above 0)
    /// split light by wavelength. Off, they refract at their IOR at 587.6 nm.
    pub dispersion: bool,
    /// Steps of a subsurface walk before it stops and keeps the share it would yet carry
    /// out, clamped to [`SSS_STEPS`](crate::SSS_STEPS).
    pub sss_max_steps: u32,
    /// Which subsurface vertices of a path walk.
    pub sss_depth: SubsurfaceDepth,
    /// Share of surface pixels whose opaque path runs, clamped to
    /// [[`MIN_PIXEL_FRACTION`], 1]. Each frame every surface pixel runs its path with this
    /// probability, chosen independently per pixel, and divides the path's estimate by it;
    /// the others keep their emission and add no path. Sky pixels and the dielectric, media
    /// and guide passes run on every pixel.
    pub pixel_fraction: f32,
    /// The dielectric whose interior holds the eye; each pixel flips it at every dielectric
    /// surface between the eye and its near-plane point, so a near plane straddling the
    /// surface works.
    pub camera_medium: Option<MaterialId>,
    pub media: MediaSettings,
    pub guides: GuideSettings,
    pub debug_view: LightingDebugView,
    /// Accumulation reference instead of the real-time path.
    pub reference: Option<ReferenceSettings>,
}

impl Default for ViewLighting {
    fn default() -> Self {
        Self {
            primary_di_candidates: 8,
            bounce_di_candidates: 4,
            local_light_share: 0.5,
            max_bounces: 3,
            dielectric_depth: 4,
            dispersion: true,
            sss_max_steps: 16,
            sss_depth: SubsurfaceDepth::FirstEvent,
            pixel_fraction: 1.0,
            camera_medium: None,
            media: MediaSettings::default(),
            guides: GuideSettings::default(),
            debug_view: LightingDebugView::None,
            reference: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GuideSettings {
    /// Pixels with a linear roughness below this take the motion of their mirror reflection
    /// in `specular_motion`; others take their surface's motion.
    pub specular_motion_roughness: f32,
    pub camera_jump: CameraJump,
}

impl Default for GuideSettings {
    fn default() -> Self {
        Self {
            specular_motion_roughness: 0.25,
            camera_jump: CameraJump::default(),
        }
    }
}

/// A camera move between consecutive frames of a view large enough to reset its history.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CameraJump {
    /// m.
    pub distance: f64,
    /// Radians of view rotation.
    pub angle: f32,
}

impl Default for CameraJump {
    fn default() -> Self {
        Self {
            distance: 2.0,
            angle: 30f32.to_radians(),
        }
    }
}

impl CameraJump {
    /// Whether a camera moving from `previous` to `current`, each a position and a
    /// world-from-view rotation, jumps.
    pub fn jumps(&self, previous: (DVec3, Quat), current: (DVec3, Quat)) -> bool {
        previous.0.distance(current.0) > self.distance
            || previous.1.angle_between(current.1) > self.angle
    }
}

/// What a view's radiance target shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LightingDebugView {
    #[default]
    None,
    /// Light reaching the first vertex directly: its emission, next-event estimation there
    /// and the emission and sky its BSDF ray reaches.
    DirectOnly,
    /// Everything `DirectOnly` leaves out.
    IndirectOnly,
    /// Light through next-event estimation alone, unweighted.
    NeeOnly,
    /// Light through BSDF rays alone, unweighted.
    BsdfOnly,
    /// Rays traced per pixel in red, and 1 in green where the path's radiance was not
    /// finite.
    RaysPerPixel,
    /// The first vertex's ray origin offset in m, in red.
    RayOriginBias,
    /// The world cache's radiance at the first vertex, as a path ending there would add it;
    /// 0 where no resolved cell covers it.
    WorldCache,
    /// The first vertex's world cache level in red, 1 in green where a resolved cell covers
    /// it, and a hash of its cell in blue.
    WorldCacheLod,
    /// A ray reconstruction guide as the guide resolve computed it, before its target's
    /// quantization; alpha is 1 on surfaces.
    Guide(GuideKind),
    /// On pixels whose first layer is a dielectric: its reflection's share of the layer's
    /// albedo in red, its transmission's in green, and the path's dielectric vertices over
    /// `dielectric_depth` in blue.
    DielectricLayer,
    /// The media's in-scatter over each pixel's camera segment alone.
    Media,
    /// The froxel volume: each pixel's media transmittance in red, its depth as a share of
    /// the froxel slices in green, and a checker of the froxel columns in blue.
    Froxels,
    /// Light leaving through subsurface walks: what each walk's exit and the path after it
    /// add.
    Subsurface,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GuideKind {
    DiffuseAlbedo,
    SpecularAlbedo,
    /// 0.5 n + 0.5.
    Normal,
    /// Linear roughness in every channel.
    Roughness,
    /// Previous uv minus current uv in red and green.
    SpecularMotion,
}

/// The accumulation reference: no world cache, no Russian roulette before 64 bounces, and
/// optionally a thin lens traced from the eye for depth-of-field references.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct ReferenceSettings {
    pub lens: Option<ThinLens>,
}

/// A thin lens at the eye. Each frame traces every pixel's first ray from a point of the
/// aperture disc through a jittered point of the pixel on the focal plane, which stays
/// fixed, instead of reading the G-buffer.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ThinLens {
    /// m.
    pub aperture_radius: f32,
    /// m along the view axis.
    pub focus_distance: f32,
}

/// Largest [`ViewLighting::local_light_share`]: the scene's selection keeps a share, so every
/// light stays reachable.
pub const MAX_LOCAL_LIGHT_SHARE: f32 = 0.9;

/// Smallest [`ViewLighting::pixel_fraction`].
pub const MIN_PIXEL_FRACTION: f32 = 1.0 / 16.0;

/// Pixels per side of the screen tiles that share a local light sampler (`light_sampling`'s
/// `LOCAL_TILE`).
const LOCAL_TILE: u32 = 16;

/// `path`'s `LIGHT_FEEDBACK_COUNT`.
const LIGHT_FEEDBACK_COUNT: u32 = 1;

/// Bounces from which Russian roulette may end a path: after the first, or 64 in the
/// reference.
const ROULETTE_DEPTH: u32 = 1;
const REFERENCE_ROULETTE_DEPTH: u32 = 64;

/// `path`'s `CACHE_TERMINATE`.
const CACHE_TERMINATE: u32 = 1;

/// `dielectric`'s `DIELECTRIC_REFERENCE`, `DIELECTRIC_DEBUG_LAYER`, `DIELECTRIC_DISPERSION`
/// and `DIELECTRIC_FROST`.
const DIELECTRIC_REFERENCE: u32 = 1;
const DIELECTRIC_DEBUG_LAYER: u32 = 2;
const DIELECTRIC_DISPERSION: u32 = 4;
const DIELECTRIC_FROST: u32 = 8;

/// `path`'s `SSS_PRIMARY_ONLY`.
const SSS_PRIMARY_ONLY: u32 = 1;

/// `MATERIAL_NONE`.
const NO_MATERIAL: u32 = u32::MAX;

/// A [`LightingView`] target and its storage view.
struct Target {
    texture: Texture,
    store: StorageImageIndex,
}

/// A frame's light feedback: written by its opaque dispatch, read by the next frame's local
/// samplers.
struct Feedback {
    target: Target,
    sampled: TextureIndex,
}

/// The background layer behind out-of-focus foreground and the frame the opaque dispatch
/// last wrote it.
struct DofLayer {
    target: Target,
    sampled: TextureIndex,
    frame: u64,
    device: Device,
}

impl DofLayer {
    fn new(device: &Device, extent: [u32; 2]) -> Result<Self, Error> {
        let desc = TextureDesc::d2(
            Format::Rgba16Float,
            extent[0],
            extent[1],
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        );
        let texture = device.create_texture(&desc, TextureMemory::Dedicated, "dof background")?;
        let view = orr_gpu::ViewDesc::default();
        let store = device.write_storage(device.alloc_image_slot()?, &texture, &view);
        let sampled = device.write_sampled(device.alloc_image_slot()?, &texture, &view);
        Ok(Self {
            target: Target { texture, store },
            sampled,
            frame: 0,
            device: device.clone(),
        })
    }
}

impl Drop for DofLayer {
    fn drop(&mut self) {
        self.device
            .free_image_slot_after_submitted(self.target.store.0);
        self.device.free_image_slot_after_submitted(self.sampled.0);
    }
}

/// Arena names of a [`LightingView`]'s targets.
const RADIANCE: &str = "radiance";
const DIFFUSE_ALBEDO: &str = "diffuse albedo";
const SPECULAR_ALBEDO: &str = "specular albedo";
const NORMAL_ROUGHNESS: &str = "normal roughness";
const SPECULAR_MOTION: &str = "specular motion";
const RR_DEPTH: &str = "rr depth";
const RR_MOTION: &str = "rr motion";

/// The radiance lives until the end of the frame: the passes after ray reconstruction and
/// the consumer read it.
const RADIANCE_SPAN: Span = Span::new(ViewStep::Lighting, ViewStep::After);
/// The guides and lighting's depth and motion are ray reconstruction's inputs.
const GUIDE_SPAN: Span = Span::new(ViewStep::Lighting, ViewStep::Rr);

/// The view's targets in its arena, taken at one layout.
struct Targets {
    generation: u64,
    radiance: ArenaTexture,
    diffuse_albedo: ArenaTexture,
    specular_albedo: ArenaTexture,
    normal_roughness: ArenaTexture,
    specular_motion: ArenaTexture,
    depth: ArenaTexture,
    motion: ArenaTexture,
}

impl Targets {
    fn take(arena: &ViewArena) -> Result<Self, Error> {
        let get = |name| {
            arena.texture(name).cloned().ok_or_else(|| {
                Error::Gpu(orr_gpu::Error::Invalid(format!(
                    "the view's arena holds no {name} target"
                )))
            })
        };
        Ok(Self {
            generation: arena.generation(),
            radiance: get(RADIANCE)?,
            diffuse_albedo: get(DIFFUSE_ALBEDO)?,
            specular_albedo: get(SPECULAR_ALBEDO)?,
            normal_roughness: get(NORMAL_ROUGHNESS)?,
            specular_motion: get(SPECULAR_MOTION)?,
            depth: get(RR_DEPTH)?,
            motion: get(RR_MOTION)?,
        })
    }
}

/// One view's lighting targets at its render extent, which are ray reconstruction's inputs
/// and live in the view's arena, and its history; kept with the view's history.
pub struct LightingView {
    device: Device,
    extent: [u32; 2],
    targets: Targets,
    feedback: [Feedback; 2],
    /// The feedback the next opaque dispatch writes; the other holds the last frame's.
    feedback_index: usize,
    local: Allocation<LocalLights>,
    local_tiles: [u32; 2],
    slots: Vec<u32>,
    /// The dielectric dispatch's layer memory: each pixel's first layer for the guide
    /// resolve, the tiles it traced and its arguments (`traced_tiles` in `path.slang`); the
    /// frame the dispatch last wrote it, and whether it kept the frosted-glass term's inputs.
    layers: Allocation<LayerGuide>,
    layers_frame: Option<u64>,
    layers_frost: bool,
    /// The frame and path settings of the last opaque dispatch.
    path: Option<(u64, PathRoot)>,
    roots: Ring<PathRoot>,
    dielectric_roots: Ring<PathRoot>,
    layer_tile_roots: Ring<LayerTilesRoot>,
    update_roots: Ring<PathRoot>,
    local_roots: Ring<local::Root>,
    guide_roots: Ring<GuideRoot>,
    frost_roots: Ring<FrostRoot>,
    previous: Option<(SceneId, DVec3, Quat)>,
    history_reset: bool,
    pub(crate) media: MediaView,
    dof: Option<DofLayer>,
}

impl LightingView {
    /// The targets a view's arena holds for its [`LightingView`] at `extent`.
    pub fn requests(extent: [u32; 2]) -> Vec<ArenaRequest> {
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        let d2 = |format| TextureDesc::d2(format, extent[0], extent[1], usage);
        let guide = |name, format| ArenaRequest::texture(name, d2(format), GUIDE_SPAN);
        vec![
            ArenaRequest::texture(RADIANCE, d2(Format::Rgba16Float), RADIANCE_SPAN),
            guide(DIFFUSE_ALBEDO, Format::Rgba8Unorm),
            guide(SPECULAR_ALBEDO, Format::Rgba8Unorm),
            guide(NORMAL_ROUGHNESS, Format::Rgba16Float),
            guide(SPECULAR_MOTION, Format::Rg16Float),
            guide(RR_DEPTH, Format::R32Float),
            guide(RR_MOTION, Format::Rg16Float),
        ]
    }

    /// A view over the targets `arena` holds from [`Self::requests`], at their extent.
    pub fn new(device: &Device, arena: &ViewArena) -> Result<Self, Error> {
        let targets = Targets::take(arena)?;
        let [w, h, _] = targets.radiance.texture().desc().extent;
        let extent = [w, h];
        let mut slots = Vec::new();
        let mut target = |format, name: &str| -> Result<Target, Error> {
            let desc = TextureDesc::d2(
                format,
                extent[0],
                extent[1],
                TextureUsage::SAMPLED | TextureUsage::STORAGE,
            );
            let texture = device.create_texture(&desc, TextureMemory::Dedicated, name)?;
            let slot = device.alloc_image_slot()?;
            slots.push(slot);
            let store = device.write_storage(slot, &texture, &orr_gpu::ViewDesc::default());
            Ok(Target { texture, store })
        };
        let feedback_targets = [
            target(Format::Rg32Uint, "light feedback")?,
            target(Format::Rg32Uint, "light feedback")?,
        ];
        let mut sampled_view = |texture: &Texture| -> Result<TextureIndex, Error> {
            let slot = device.alloc_image_slot()?;
            slots.push(slot);
            Ok(device.write_sampled(slot, texture, &orr_gpu::ViewDesc::default()))
        };
        let [a, b] = feedback_targets;
        let feedback = [
            Feedback {
                sampled: sampled_view(&a.texture)?,
                target: a,
            },
            Feedback {
                sampled: sampled_view(&b.texture)?,
                target: b,
            },
        ];
        let local_tiles = extent.map(|e| e.div_ceil(LOCAL_TILE));
        let local = device_local(
            device,
            u64::from(local_tiles[0]) * u64::from(local_tiles[1]),
            "local light samplers",
        )?;
        let layers = device_local(device, LayerMemory::new(extent).guides, "layer guides")?;
        Ok(Self {
            device: device.clone(),
            extent,
            targets,
            feedback,
            feedback_index: 0,
            local,
            local_tiles,
            slots,
            layers,
            layers_frame: None,
            layers_frost: false,
            path: None,
            roots: Ring::new(device, "path roots")?,
            dielectric_roots: Ring::new(device, "dielectric roots")?,
            layer_tile_roots: Ring::new(device, "layer tile roots")?,
            update_roots: Ring::new(device, "update path roots")?,
            local_roots: Ring::new(device, "local light sampler roots")?,
            guide_roots: Ring::new(device, "guide roots")?,
            frost_roots: Ring::new(device, "frosted glass roots")?,
            previous: None,
            history_reset: true,
            media: MediaView::new(device)?,
            dof: None,
        })
    }

    /// Takes the targets again after `arena` was laid out anew, at the view's extent.
    pub fn bind(&mut self, arena: &ViewArena) -> Result<(), Error> {
        if self.targets.generation != arena.generation() {
            self.targets = Targets::take(arena)?;
        }
        Ok(())
    }

    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Starts the view's frame `frame` and settles its history reset. A camera cut
    /// (`desc.reset_history`), output's exposure cut, a camera jump past
    /// `lighting.guides.camera_jump` since the view's last frame, a change of scene, the
    /// view's first frame and [`SceneLighting::reset_history`] before the scene's pass of
    /// `frame` each set it. Call once per frame, after the scene's [`SceneLighting::record`]
    /// and before the view's lighting records; returns [`Self::history_reset`].
    pub fn begin(
        &mut self,
        frame: u64,
        scene: &SceneLighting,
        desc: &ViewDesc<'_>,
        exposure_cut: bool,
        lighting: &ViewLighting,
    ) -> bool {
        let current = (desc.position, desc.rotation);
        let continues = self.previous.is_some_and(|(id, position, rotation)| {
            id == desc.scene
                && !lighting
                    .guides
                    .camera_jump
                    .jumps((position, rotation), current)
        });
        self.history_reset =
            !continues || desc.reset_history || exposure_cut || scene.resets_history(frame);
        self.previous = Some((desc.scene, desc.position, desc.rotation));
        self.history_reset
    }

    /// Whether ray reconstruction drops its history this frame (its `InReset`), as
    /// [`Self::begin`] settled it.
    pub fn history_reset(&self) -> bool {
        self.history_reset
    }

    /// RGBA16F, pre-exposed noisy radiance; alpha is 1 where the G-buffer holds a surface.
    pub fn radiance(&self) -> &Texture {
        self.targets.radiance.texture()
    }

    pub fn radiance_index(&self) -> TextureIndex {
        self.targets.radiance.sampled()
    }

    /// The radiance's storage view, for passes that draw over it after lighting.
    pub fn radiance_storage(&self) -> StorageImageIndex {
        self.targets.radiance.storage(0)
    }

    /// RGBA8 unorm, not sRGB: diffuse albedo, subsurface included; 0.5 on sky pixels.
    pub fn diffuse_albedo(&self) -> &Texture {
        self.targets.diffuse_albedo.texture()
    }

    /// RGBA8 unorm, not sRGB: specular albedo; 0 on sky pixels.
    pub fn specular_albedo(&self) -> &Texture {
        self.targets.specular_albedo.texture()
    }

    /// RGBA16F: world normal in RGB and linear roughness in A; 0 on sky pixels.
    pub fn normal_roughness(&self) -> &Texture {
        self.targets.normal_roughness.texture()
    }

    /// RG16F, previous uv minus current uv: the mirror reflection's motion on smooth pixels,
    /// the surface's elsewhere.
    pub fn specular_motion(&self) -> &Texture {
        self.targets.specular_motion.texture()
    }

    /// R32F reverse-Z device depth, 0 on sky pixels.
    pub fn depth(&self) -> &Texture {
        self.targets.depth.texture()
    }

    /// RG16F, previous uv minus current uv, unjittered.
    pub fn motion(&self) -> &Texture {
        self.targets.motion.texture()
    }

    /// RGBA16F, the background layer behind out-of-focus foreground as the last opaque
    /// dispatch given a [`DofBackgroundPass`] wrote it: on each marked pixel, the pre-exposed
    /// radiance its lens ray reaches past the mask's depth in RGB (with the camera segment's
    /// media where media apply ran) and that hit's reverse-Z device depth in A, 0 for sky;
    /// (0, 0, 0, -1) on unmarked pixels. `None` once a dispatch runs without one.
    pub fn dof_background(&self) -> Option<&Texture> {
        self.dof.as_ref().map(|l| &l.target.texture)
    }

    pub fn dof_background_index(&self) -> Option<TextureIndex> {
        self.dof.as_ref().map(|l| l.sampled)
    }

    /// RG32_UINT: per pixel, the light feedback of the last opaque dispatch (a light's
    /// feedback id or `u32::MAX`) and its first vertex's camera distance in m as `f32` bits, 0
    /// on sky pixels.
    pub fn light_feedback(&self) -> &Texture {
        &self.feedback[1 - self.feedback_index].target.texture
    }
}

impl LightingView {
    /// The layer guides the dielectric dispatch of `frame` wrote; null when it did not run.
    pub(crate) fn layer_guides(&self, frame: u64) -> GpuPtr<LayerGuide> {
        if self.layers_frame == Some(frame) {
            self.layers.gpu()
        } else {
            GpuPtr::NULL
        }
    }

    /// The background layer's storage view when the opaque dispatch of `frame` wrote it.
    pub(crate) fn dof_background_store(&self, frame: u64) -> StorageImageIndex {
        self.dof
            .as_ref()
            .filter(|l| l.frame == frame)
            .map_or(StorageImageIndex::NONE, |l| l.target.store)
    }

    /// The froxel counts of `frame`, once its submit has completed; `None` when that frame
    /// recorded no froxels or its slot holds a later frame.
    pub fn media_stats(&self, frame: u64) -> Option<MediaStats> {
        self.media.stats(frame)
    }

    /// RGBA16F 3D, the froxels' isotropic incident radiance (unexposed, nits) as the last
    /// [`LightingKernels::record_media`] wrote it, with the visibility of the first shadowed
    /// directional light in A; `None` before the first.
    pub fn froxel_incident(&self) -> Option<TextureIndex> {
        self.media.incident()
    }

    /// RGBA16F 3D, per froxel the pre-exposed in-scatter from the eye to its far boundary in
    /// RGB and the luminance of the transmittance in A; `None` before the first
    /// [`LightingKernels::record_media`].
    pub fn froxel_integrated(&self) -> Option<TextureIndex> {
        self.media.integrated()
    }
}

impl Drop for LightingView {
    fn drop(&mut self) {
        for &slot in &self.slots {
            self.device.free_image_slot_after_submitted(slot);
        }
    }
}

/// Where the parts of a view's layer memory lie (`traced_tiles` in `path.slang`).
struct LayerMemory {
    /// Guides of room the memory takes.
    guides: u64,
    /// Bytes from its start to the dispatch's tile count, then its compute workgroups and
    /// its ray extent.
    args: u64,
}

impl LayerMemory {
    fn new([w, h]: [u32; 2]) -> Self {
        let tiles = u64::from(w.div_ceil(LAYER_TILE)) * u64::from(h.div_ceil(LAYER_TILE));
        let guide = size_of::<LayerGuide>() as u64;
        let args = guide * u64::from(w) * u64::from(h) + 8 * tiles;
        Self {
            guides: (args + 4 * 7).div_ceil(guide),
            args,
        }
    }
}

/// Pixels per side of a tile of the dielectric dispatch (`LAYER_TILE`).
const LAYER_TILE: u32 = 8;

/// `LayerTilesRoot.flags` (`dielectric.slang`).
const LAYER_TILES_EVERYWHERE: u32 = 1 << 0;
const LAYER_TILES_FROST: u32 = 1 << 1;

/// What the opaque dispatch and the guide resolve read for one view and frame.
#[derive(Clone, Copy, Debug)]
pub struct ViewPass {
    pub frame: u64,
    pub scene: GpuPtr<SceneRoot>,
    pub view: GpuPtr<ViewConstants>,
    /// The view's G-buffer at the [`LightingView`]'s extent.
    pub gbuffer: GpuPtr<GBufferGpu>,
    /// The frame's index in every pixel's sample sequence; consecutive frames take
    /// consecutive indices.
    pub sample_index: u32,
    /// Separates sample sets of the same pixels and sample index.
    pub seed: u32,
    /// The opaque dispatch also traces the view's background layer
    /// ([`LightingView::dof_background`]); `None` frees it.
    pub dof_background: Option<DofBackgroundPass>,
    /// The dielectric dispatch keeps what [`LightingKernels::record_frosted_glass`] reads:
    /// set while output's depth-of-field guide is on.
    pub frosted_glass: bool,
}

/// What the background layer behind out-of-focus foreground reads in one frame. Each marked
/// pixel traces a second camera ray from a point of the lens's aperture disc through a
/// jittered point of the pixel on the focal plane, starting past the mask's depth, and the
/// path from its hit at the opaque path's settings. The ray sees opaque geometry only:
/// dielectric and presence layers past the mask are not traced.
#[derive(Clone, Copy, Debug)]
pub struct DofBackgroundPass {
    /// Float image at the view's extent, read from its red channel: per pixel, the reverse-Z
    /// device depth past which its ray starts; 0 leaves the pixel unmarked.
    pub mask: TextureIndex,
    /// m.
    pub aperture_radius: f32,
    /// The focus distance in m along the view axis, read on the GPU by the dispatch.
    pub focus_distance: GpuPtr<f32>,
}

/// A kernel with a raygen and a compute entry over one body, as one of them.
enum PathPipeline {
    Compute(ComputePipeline),
    Rays(RayDispatchPipeline),
}

impl PathPipeline {
    fn new(
        device: &Device,
        m: &ShaderModule,
        spec: SpecData,
        rays: bool,
        name: &str,
    ) -> Result<Self, Error> {
        Ok(if rays {
            let stage = ShaderStage::new(m, "raygen_main", spec)?;
            Self::Rays(device.create_ray_dispatch_pipeline(&stage, name)?)
        } else {
            let stage = ShaderStage::new(m, "compute_main", spec)?;
            Self::Compute(device.create_compute_pipeline(&stage, name)?)
        })
    }

    /// Runs the kernel over the workgroups `groups` holds, or as rays over the extent `rays`
    /// holds.
    fn record_indirect<R>(
        &self,
        list: &mut CommandList,
        root: GpuPtr<R>,
        groups: GpuPtr<DispatchArgs>,
        rays: GpuPtr<DispatchArgs>,
    ) {
        match self {
            Self::Compute(p) => list.dispatch_indirect(p, root, groups),
            Self::Rays(p) => list.dispatch_rays_indirect(p, root, rays),
        }
    }

    /// Runs the kernel over `extent` invocations.
    fn record<R>(&self, list: &mut CommandList, root: GpuPtr<R>, extent: [u32; 2]) {
        let [w, h] = extent;
        match self {
            Self::Compute(p) => {
                let [gx, gy, _] = p.workgroup();
                list.dispatch(p, root, [w.div_ceil(gx), h.div_ceil(gy), 1]);
            }
            Self::Rays(p) => list.dispatch_rays(p, root, [w, h, 1]),
        }
    }
}

/// The per-view lighting kernels of one device.
pub struct LightingKernels {
    opaque: PathPipeline,
    dielectric: PathPipeline,
    layer_tiles: ComputePipeline,
    layer_args: ComputePipeline,
    /// A zero word that starts each frame's tile count.
    zero: Allocation<u32>,
    update: PathPipeline,
    guides: ComputePipeline,
    frost: ComputePipeline,
    local: ComputePipeline,
    media: MediaKernels,
    cap_tail: CapTailTable,
}

/// The specialization constants of kernel module `$kernel`, which imports `bsdf`.
macro_rules! bsdf_spec {
    ($kernel:ident, $tables:expr) => {
        $kernel::Spec {
            openpbr_lut_slot: $tables.first_slot(),
            ..$kernel::Spec::default()
        }
        .bytes()
    };
}

impl LightingKernels {
    /// `kernels` replaces modules as [`SceneLighting::new`] does; `tables` must outlive the
    /// kernels' use.
    pub fn new(
        device: &Device,
        kernels: &KernelSet,
        tables: &BsdfTables,
        entry: PathEntry,
    ) -> Result<Self, Error> {
        let caps = device.caps();
        let rays = match entry {
            PathEntry::Fastest => caps.ray_dispatch && caps.reorder,
            PathEntry::Compute => false,
            PathEntry::RayDispatch => true,
        };
        let bytes = bsdf_spec!(opaque, tables);
        let spec = SpecData {
            entries: opaque::Spec::ENTRIES,
            bytes: &bytes,
        };
        let m = module(kernels, &opaque::MODULE)?;
        let opaque = PathPipeline::new(device, m, spec, rays, "opaque paths")?;
        let bytes = bsdf_spec!(dispatch, tables);
        let spec = SpecData {
            entries: dispatch::Spec::ENTRIES,
            bytes: &bytes,
        };
        let m = module(kernels, &dispatch::MODULE)?;
        let dielectric = PathPipeline::new(device, m, spec, rays, "dielectric paths")?;
        let bytes = bsdf_spec!(update, tables);
        let spec = SpecData {
            entries: update::Spec::ENTRIES,
            bytes: &bytes,
        };
        let m = module(kernels, &update::MODULE)?;
        let update = PathPipeline::new(device, m, spec, rays, "world cache update paths")?;
        let bytes = bsdf_spec!(resolve, tables);
        let spec = SpecData {
            entries: resolve::Spec::ENTRIES,
            bytes: &bytes,
        };
        let stage = ShaderStage::new(module(kernels, &resolve::MODULE)?, "main", spec)?;
        let guides = device.create_compute_pipeline(&stage, "guide resolve")?;
        let mut zero = device.gpu_malloc::<u32>(MemoryKind::Upload, 1, "zero")?;
        zero.as_mut_slice().expect("mapped")[0] = 0;
        Ok(Self {
            opaque,
            dielectric,
            layer_tiles: pipeline(device, kernels, &tiles::MODULE, "tiles")?,
            layer_args: pipeline(device, kernels, &tiles::MODULE, "args")?,
            zero,
            update,
            guides,
            frost: pipeline(device, kernels, &frost::MODULE, "main")?,
            local: pipeline(device, kernels, &local::MODULE, "main")?,
            media: MediaKernels::new(device, kernels)?,
            cap_tail: CapTailTable::new(device)?,
        })
    }

    /// Records the froxel passes of `pass` after the scene's [`SceneLighting::record`], the
    /// view's raster half and [`LightingView::begin`]: picks the shadowed directional lights,
    /// writes the linked medium into the froxels, the light reaching those in front of the
    /// view's surfaces (blended with their reprojected history) and each column's in-scatter
    /// and transmittance. The outputs are visible to later compute and ray stages. Makes the
    /// view's froxel targets at its first call and when the extent or the media settings
    /// that shape them change; a history reset, a new `near_range` and new targets restart
    /// the froxel history.
    pub fn record_media(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> Result<(), Error> {
        self.media.record(list, scene, view, pass, lighting)
    }

    /// Records media apply into `view`'s radiance after [`Self::record_media`] and the paths
    /// that wrote the radiance: each pixel's radiance x the transmittance over its camera
    /// segment, plus the in-scatter over it, to its surface or the scene's `near_range`,
    /// whichever is nearer (sky pixels take the whole range). The reference marches the
    /// segment per pixel instead. Orders nothing after it.
    pub fn record_media_apply(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) {
        self.media.record_apply(list, scene, view, pass, lighting);
    }

    /// Records the world cache's update paths, the view's local light samplers and the
    /// opaque dispatch of `pass` into `view`'s radiance and light feedback after the scene's
    /// [`SceneLighting::record`] and the view's raster half in the same or an earlier list;
    /// orders nothing after it. The scene's first view in a frame traces one update path per
    /// `update_block`^2 pixels, its later views one per four times as many. The reference
    /// traces none, never reads the cache, builds no local samplers and adds nothing to the
    /// scene's light feedback. With [`ViewPass::dof_background`] the dispatch also writes the
    /// view's background layer, made at the first such call; without it the layer is freed.
    pub fn record_opaque(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> Result<(), Error> {
        let (roulette_depth, lens) = match lighting.reference {
            Some(r) => (REFERENCE_ROULETTE_DEPTH, r.lens),
            None => (ROULETTE_DEPTH, None),
        };
        let mode = match lighting.debug_view {
            LightingDebugView::None
            | LightingDebugView::Guide(_)
            | LightingDebugView::DielectricLayer
            | LightingDebugView::Media
            | LightingDebugView::Froxels => 0,
            LightingDebugView::DirectOnly => 1,
            LightingDebugView::IndirectOnly => 2,
            LightingDebugView::NeeOnly => 3,
            LightingDebugView::BsdfOnly => 4,
            LightingDebugView::RaysPerPixel => 5,
            LightingDebugView::RayOriginBias => 6,
            LightingDebugView::WorldCache => 7,
            LightingDebugView::WorldCacheLod => 8,
            LightingDebugView::Subsurface => 9,
        };
        let reference = lighting.reference.is_some();
        let first = !reference && scene.first_view(pass.frame);
        let block = scene.world_cache_settings().update_block * if first { 1 } else { 2 };
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let local_share = lighting.local_light_share.clamp(0.0, MAX_LOCAL_LIGHT_SHARE);
        let local = !reference && local_share > 0.0;
        let written = view.feedback_index;
        view.feedback_index = 1 - written;
        if local {
            let root = view.local_roots.write(
                slot,
                local::Root {
                    view: pass.view,
                    gbuffer: pass.gbuffer,
                    set: scene.light_set(),
                    local: view.local.gpu(),
                    previous: view.feedback[1 - written].sampled,
                    columns: view.local_tiles[0],
                    share: local_share,
                    reset: u32::from(view.history_reset),
                },
            );
            let [w, h] = view.local_tiles;
            list.begin_pass("local light samplers", false);
            list.dispatch(&self.local, root, [w, h, 1]);
            list.end_pass();
            list.barrier(
                Stages::COMPUTE,
                Stages::COMPUTE | Stages::RAY,
                Hazards::empty(),
            );
        }
        if pass.dof_background.is_none() {
            view.dof = None;
        } else if view.dof.is_none() {
            let layer = DofLayer::new(&view.device, view.extent)?;
            list.discard(&layer.target.texture);
            view.dof = Some(layer);
        }
        let (dof_background, dof_mask, dof_focus, dof_aperture) =
            match (pass.dof_background, view.dof.as_mut()) {
                (Some(d), Some(layer)) => {
                    layer.frame = pass.frame;
                    (
                        layer.target.store,
                        d.mask,
                        d.focus_distance,
                        d.aperture_radius,
                    )
                }
                _ => (
                    StorageImageIndex::NONE,
                    TextureIndex::NONE,
                    GpuPtr::NULL,
                    0.0,
                ),
            };
        let path_root = PathRoot {
            scene: pass.scene,
            view: pass.view,
            gbuffer: pass.gbuffer,
            set: scene.light_set(),
            cache: scene.world_cache(pass.frame),
            local: if local {
                view.local.gpu()
            } else {
                GpuPtr::NULL
            },
            radiance: view.targets.radiance.storage(0),
            primary_candidates: lighting.primary_di_candidates,
            bounce_candidates: lighting.bounce_di_candidates,
            max_bounces: lighting.max_bounces,
            roulette_depth,
            mode,
            frame: pass.sample_index,
            seed: pass.seed,
            segment_steps: lighting.media.segment_steps,
            aperture_radius: lens.map_or(0.0, |l| l.aperture_radius),
            focus_distance: lens.map_or(0.0, |l| l.focus_distance),
            cache_flags: if reference { 0 } else { CACHE_TERMINATE },
            update_block: block,
            feedback: view.feedback[written].target.store,
            local_columns: view.local_tiles[0],
            light_flags: if reference { 0 } else { LIGHT_FEEDBACK_COUNT },
            layers: GpuPtr::NULL,
            froxels: GpuPtr::NULL,
            dielectric_depth: 0,
            camera_medium: NO_MATERIAL,
            dielectric_flags: 0,
            sss_max_steps: lighting
                .sss_max_steps
                .clamp(*SSS_STEPS.start(), *SSS_STEPS.end()),
            cap_tail: self.cap_tail.gpu(),
            sss_flags: match lighting.sss_depth {
                SubsurfaceDepth::FirstEvent => 0,
                SubsurfaceDepth::PrimaryOnly => SSS_PRIMARY_ONLY,
            },
            pixel_fraction: lighting.pixel_fraction.clamp(MIN_PIXEL_FRACTION, 1.0),
            dof_background,
            dof_mask,
            dof_focus,
            dof_aperture,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let root = view.roots.write(slot, path_root);
        view.path = Some((pass.frame, path_root));
        if !reference {
            let update_root = view.update_roots.write(
                slot,
                PathRoot {
                    mode: 0,
                    local: GpuPtr::NULL,
                    light_flags: 0,
                    dof_background: StorageImageIndex::NONE,
                    ..path_root
                },
            );
            list.begin_pass("world cache update paths", false);
            self.update
                .record(list, update_root, view.extent.map(|e| e.div_ceil(block)));
            list.end_pass();
        }
        list.begin_pass("opaque paths", false);
        self.opaque.record(list, root, view.extent);
        list.end_pass();
        Ok(())
    }

    /// Records the dielectric dispatch of `pass` after [`Self::record_opaque`] of the same
    /// view and frame, ordered after it, with its path settings: per pixel of the 8 x 8 tiles
    /// the G-buffer's layer tiles mark and of the tiles beside them, a camera ray against the
    /// dielectric mask up to the opaque depth finds the first dielectric or presence layer,
    /// whose path replaces the pixel's radiance, and leaves the layer's guide for
    /// [`Self::record_guides`]. Every tile is traced while a layer may lie before the near
    /// plane, while [`ViewLighting::camera_medium`] is set and when the G-buffer has no layer
    /// tiles. Pixels whose eye lies inside
    /// [`ViewLighting::camera_medium`] trace their path through it from the G-buffer's surface
    /// too. Outside the reference, segments between layers read the froxels of
    /// [`Self::record_media`] of the same frame where it ran. Camera rays are pinhole rays
    /// whatever the reference's lens. With [`ViewPass::frosted_glass`] it also keeps each
    /// pixel's background and rough-transmission blur for [`Self::record_frosted_glass`].
    /// Records nothing without the opaque dispatch of `pass.frame`. Orders nothing after it.
    pub fn record_dielectric(
        &self,
        list: &mut CommandList<'_>,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) {
        let Some((_, path)) = view.path.filter(|(frame, _)| *frame == pass.frame) else {
            return;
        };
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let mut flags = 0;
        let mut froxels = view.media.volume(pass.frame);
        if lighting.reference.is_some() {
            flags |= DIELECTRIC_REFERENCE;
            froxels = GpuPtr::NULL;
        }
        if lighting.debug_view == LightingDebugView::DielectricLayer {
            flags |= DIELECTRIC_DEBUG_LAYER;
        }
        if lighting.dispersion {
            flags |= DIELECTRIC_DISPERSION;
        }
        if pass.frosted_glass {
            flags |= DIELECTRIC_FROST;
        }
        let root = view.dielectric_roots.write(
            slot,
            PathRoot {
                light_flags: 0,
                dof_background: StorageImageIndex::NONE,
                layers: view.layers.gpu(),
                froxels,
                dielectric_depth: lighting.dielectric_depth,
                camera_medium: lighting.camera_medium.map_or(NO_MATERIAL, |m| m.0),
                dielectric_flags: flags,
                ..path
            },
        );
        let mut tile_flags = 0;
        if lighting.camera_medium.is_some() {
            tile_flags |= LAYER_TILES_EVERYWHERE;
        }
        if pass.frosted_glass {
            tile_flags |= LAYER_TILES_FROST;
        }
        let tiles_root = view.layer_tile_roots.write(
            slot,
            LayerTilesRoot {
                gbuffer: pass.gbuffer,
                layers: view.layers.gpu(),
                radiance: view.targets.radiance.storage(0),
                flags: tile_flags,
                _pad0: 0,
                _pad1: 0,
            },
        );
        view.layers_frame = Some(pass.frame);
        view.layers_frost = pass.frosted_glass;
        let args = view
            .layers
            .gpu()
            .cast::<u8>()
            .offset(LayerMemory::new(view.extent).args);
        list.begin_pass("dielectric paths", false);
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::TRANSFER,
            Hazards::empty(),
        );
        list.copy(args, self.zero.gpu().cast(), 4);
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        let [tx, ty] = view.extent.map(|e| e.div_ceil(LAYER_TILE));
        list.dispatch(&self.layer_tiles, tiles_root, [tx, ty, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.layer_args, tiles_root, [1, 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::RAY,
            Hazards::DRAW_ARGUMENTS,
        );
        self.dielectric
            .record_indirect(list, root, args.offset(4).cast(), args.offset(16).cast());
        list.end_pass();
    }

    /// Records the guide resolve of `pass` into `view`'s guide targets after the view's
    /// raster half, and after [`Self::record_dielectric`] where that runs in the frame, whose
    /// layer guides it blends in. Under a [`LightingDebugView::Guide`] view it also overwrites the
    /// radiance, so it must then be ordered after [`Self::record_opaque`]. Orders nothing
    /// after it.
    pub fn record_guides(
        &self,
        list: &mut CommandList<'_>,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) {
        let debug = match lighting.debug_view {
            LightingDebugView::Guide(GuideKind::DiffuseAlbedo) => 1,
            LightingDebugView::Guide(GuideKind::SpecularAlbedo) => 2,
            LightingDebugView::Guide(GuideKind::Normal) => 3,
            LightingDebugView::Guide(GuideKind::Roughness) => 4,
            LightingDebugView::Guide(GuideKind::SpecularMotion) => 5,
            _ => 0,
        };
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let root = view.guide_roots.write(
            slot,
            GuideRoot {
                scene: pass.scene,
                view: pass.view,
                gbuffer: pass.gbuffer,
                layers: view.layer_guides(pass.frame),
                radiance: view.targets.radiance.storage(0),
                diffuse_albedo: view.targets.diffuse_albedo.storage(0),
                specular_albedo: view.targets.specular_albedo.storage(0),
                normal_roughness: view.targets.normal_roughness.storage(0),
                specular_motion: view.targets.specular_motion.storage(0),
                depth: view.targets.depth.storage(0),
                motion: view.targets.motion.storage(0),
                motion_roughness: lighting.guides.specular_motion_roughness,
                debug,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        let [w, h] = view.extent;
        let [gx, gy, _] = self.guides.workgroup();
        list.begin_pass("guide resolve", false);
        list.dispatch(&self.guides, root, [w.div_ceil(gx), h.div_ceil(gy), 1]);
        list.end_pass();
    }

    /// Adds the frosted-glass term into `guide`, ray reconstruction's depth-of-field guide
    /// (R16F at the view's extent, `luminance(after - before)`), after output wrote it and
    /// after [`Self::record_dielectric`] of `pass` with [`ViewPass::frosted_glass`]. A pixel
    /// whose dominant layer is rough transmission adds its transmission's luminance times
    /// the change that blurring the background over its rough interfaces' disc makes. The
    /// background is the opaque radiance at the same pixels, unbent by refraction. Other
    /// pixels keep what output wrote. Records nothing without such a dispatch in
    /// `pass.frame`. Orders nothing after it.
    pub fn record_frosted_glass(
        &self,
        list: &mut CommandList<'_>,
        view: &mut LightingView,
        pass: &ViewPass,
        guide: StorageImageIndex,
    ) {
        if !view.layers_frost || view.layers_frame != Some(pass.frame) {
            return;
        }
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let root = view.frost_roots.write(
            slot,
            FrostRoot {
                layers: view.layers.gpu(),
                guide,
                width: view.extent[0],
                height: view.extent[1],
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        let [w, h] = view.extent;
        let [gx, gy, _] = self.frost.workgroup();
        list.begin_pass("frosted glass", false);
        list.dispatch(&self.frost, root, [w.div_ceil(gx), h.div_ceil(gy), 1]);
        list.end_pass();
    }
}
