//! Per-view lighting: the settings of one `render_view` call, the view's lighting targets and
//! history, the layer guides, the world cache's update paths, the view dispatch
//! and the guide resolve.

use std::sync::Arc;

use orr_math::{DVec3, Quat};
use orr_scene::orr_gpu::{
    self, Allocation, CommandList, ComputePipeline, Device, DispatchArgs, Format, GpuPtr, Hazards,
    MemoryKind, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
    TextureUsage,
};
use orr_scene::{
    ArenaRequest, ArenaTexture, GBufferGpu, JobResult, LightTables, Loading, MaterialId, RasterGpu,
    SceneFeatures, SceneRoot, Span, Urgency, ViewArena, ViewConstants, ViewDesc, ViewStep,
};

use crate::caustics::PhotonMap;
use crate::media::{
    MediaKernels, MediaSettings, MediaStats, MediaView, fogged, links_medium, readers, shows_media,
};
use crate::scene::{FRAME_SLOTS, Ring, device_local};
use crate::shaders::dielectric::tiles;
use crate::shaders::guides::{frost, overlay_front};
use crate::shaders::light_sampling::local;
use crate::shaders::types::{
    CausticsGpu, FrostRoot, GuideRoot, LayerGuide, LayerTilesRoot, LocalLights, OverlayFrontRoot,
    ParticleLighting, PathRoot,
};
use crate::staged::{
    FEATURE_DISPERSION, FEATURE_INTERIORS, FEATURE_STACK, FEATURE_WALK, MAX_ROUNDS, MAY_ALL,
    MAY_CACHE, MAY_CAUSTICS, MAY_COAT, MAY_COVERAGE, MAY_DEBUG, MAY_DIRECTIONAL, MAY_DISPERSION,
    MAY_EMISSIVE, MAY_FILM, MAY_FROST, MAY_FROXELS, MAY_FUZZ, MAY_INTERIORS, MAY_LAYERS, MAY_LENS,
    MAY_LOCAL, MAY_LOCAL_LIGHTS, MAY_PARTICLES, MAY_PROXIES, MAY_REFERENCE, MAY_SMOKE, MAY_STACK,
    MAY_SUBSURFACE, MAY_WALK, Source, SourcePaths, Stage, StageSet, StagedKernels, StagedMemory,
    StateParts,
};
use crate::subsurface::{SSS_STEPS, SubsurfaceDepth, SubsurfaceTable};
use crate::support::{Masked, MaskedFrame, MaskedSupport, Runs};
use crate::{BsdfTables, CompilePipeline, Error, KernelSet, PathEntry, SceneLighting, pipeline};

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
    /// Dielectric and presence vertices a path may have, its first included, at least 64 under
    /// the reference. Past them it ends in the world cache's light at that point, where a cell
    /// covers it.
    pub dielectric_depth: u32,
    /// Whether dielectrics with a dispersion (an Abbe number and a dispersion scale above 0)
    /// split light by wavelength. Off, they refract at their IOR at 587.6 nm.
    pub dispersion: bool,
    /// Steps of a walk through a subsurface material or a dense dielectric interior before it
    /// stops and keeps the share it would yet carry out, clamped to
    /// [`SSS_STEPS`](crate::SSS_STEPS). The reference walks up to 65,536 steps, past what the
    /// subsurface table covers, and ends a path whose walk reaches them, counting it in
    /// [`LightingStats::capped_walks`](crate::LightingStats::capped_walks).
    pub sss_max_steps: u32,
    /// Which subsurface vertices of a path walk; under the reference, every one.
    pub sss_depth: SubsurfaceDepth,
    /// Share of surface pixels with no layer in front whose path runs, clamped to
    /// [[`MIN_PIXEL_FRACTION`], 1]. Each frame every surface pixel runs its path with this
    /// probability, chosen independently per pixel, and divides the path's estimate by it;
    /// the others keep their emission and add no path. Sky pixels, pixels behind a layer or
    /// in an interior, and the media and guide passes run on every pixel.
    pub pixel_fraction: f32,
    /// The dielectric whose interior holds the eye; each pixel flips it at every dielectric
    /// surface between the eye and its near-plane point, so a near plane straddling the
    /// surface works. The froxels then hold its absorption and scattering in place of the
    /// scene's medium up to the surface leaving it.
    pub camera_medium: Option<MaterialId>,
    pub media: MediaSettings,
    pub guides: GuideSettings,
    pub debug_view: LightingDebugView,
    /// Accumulation reference instead of the real-time path.
    pub reference: Option<ReferenceSettings>,
    /// Whether the path kernels are compiled for what the scene and these settings may meet,
    /// which changes no result. Each new mix of features compiles its own pipelines, for
    /// seconds on a first run; `false` runs the pipelines that hold every feature, which a run
    /// over many small scenes compiles once.
    pub specialize: bool,
    /// Whether each stage of the staged path loop records in a pass of its own ("view trace",
    /// "view shade" and so on), so pass timings tell the stages apart.
    pub stage_passes: bool,
}

impl Default for ViewLighting {
    fn default() -> Self {
        Self {
            primary_di_candidates: 8,
            bounce_di_candidates: 4,
            local_light_share: 0.5,
            max_bounces: 3,
            dielectric_depth: 16,
            dispersion: true,
            sss_max_steps: 16,
            sss_depth: SubsurfaceDepth::FirstEvent,
            pixel_fraction: 1.0,
            camera_medium: None,
            media: MediaSettings::default(),
            guides: GuideSettings::default(),
            debug_view: LightingDebugView::None,
            reference: None,
            specialize: true,
            stage_passes: false,
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
    fn jumps(&self, previous: (DVec3, Quat), current: (DVec3, Quat)) -> bool {
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
    /// Light leaving through walks of subsurface materials and dielectric interiors: what each
    /// walk's exit and the path after it add.
    Subsurface,
    /// What the first vertex gathers of the caustic photons' light, from the scene's grid or a
    /// reference view's map.
    Caustics,
    /// Each pixel's camera ray traced against the scene's acceleration structure as the path
    /// rays meet it, in place of the radiance; the paths run as without a debug view.
    Traced(TracedKind),
}

/// What a [`LightingDebugView::Traced`] view shows of each pixel's traced camera ray, which
/// meets opaque, alpha-tested and dielectric geometry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TracedKind {
    /// Gray falling with the hit's distance, blue where the ray leaves the scene.
    Distance,
    /// The hit's geometric normal, `0.5 n + 0.5`.
    Normal,
    /// Red where the traced hit lies nearer than the drawn surface by more than the first
    /// vertex's ray origin offset, green where it lies farther, magenta where the ray leaves
    /// where the raster drew, cyan where it hits where the raster drew nothing; dark gray where
    /// they agree.
    Disagreement,
    /// The drawn first vertex's ray toward the scene's first directional light, from the origin
    /// the paths give it, against opaque and alpha-tested geometry: white by the cosine where it
    /// leaves the scene, dark red where something blocks it, dark blue on faces turned away.
    Sun,
}

/// Whether the paths run a debug view of their own.
fn path_debug(view: LightingDebugView) -> bool {
    !matches!(view, LightingDebugView::None | LightingDebugView::Traced(_))
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

/// The accumulation reference: no world cache, no Russian roulette before 64 bounces, a walk
/// at every subsurface vertex of up to the most steps, light leaving a dense interior by its
/// way out's ray alone, and optionally a thin lens traced from the eye for depth-of-field
/// references. Each frame the
/// view traces caustic photons of its own into a map that holds that frame's alone, gathered
/// within a radius that shrinks every frame as progressive photon mapping's.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ReferenceSettings {
    pub lens: Option<ThinLens>,
    /// Photons each frame, at most [`crate::MAX_PHOTONS`]; 0 leaves caustics to BSDF rays, as
    /// does a scene whose [`crate::CausticSettings::photons`] is 0, since its scene pass weighs
    /// where the photons start.
    pub photons: u32,
    /// The gather radius of the first frame after a history reset, m.
    pub photon_radius: f32,
}

impl Default for ReferenceSettings {
    fn default() -> Self {
        Self {
            lens: None,
            photons: 1 << 20,
            photon_radius: 0.008,
        }
    }
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

/// The layer vertices a reference path takes at least: a ray through a solid can take many.
const REFERENCE_DIELECTRIC_DEPTH: u32 = 64;

/// The steps a reference walk takes at most. The reference ends a path whose walk reaches them
/// instead of keeping the subsurface table's share, and its walks end by Russian roulette on
/// their throughput, so only a medium that absorbs nothing walks this far.
const REFERENCE_SSS_STEPS: u32 = 1 << 16;

/// The layer vertices `lighting`'s paths and dielectric guides take.
fn dielectric_depth(lighting: &ViewLighting) -> u32 {
    if lighting.reference.is_some() {
        lighting.dielectric_depth.max(REFERENCE_DIELECTRIC_DEPTH)
    } else {
        lighting.dielectric_depth
    }
}

/// `path`'s `DIELECTRIC_REFERENCE`, `DIELECTRIC_DEBUG_LAYER`, `DIELECTRIC_DISPERSION` and
/// `DIELECTRIC_FROST`.
const DIELECTRIC_REFERENCE: u32 = 1;
const DIELECTRIC_DEBUG_LAYER: u32 = 2;
const DIELECTRIC_DISPERSION: u32 = 4;
const DIELECTRIC_FROST: u32 = 8;

/// Opaque vertices a photon meets at most: the mirrors of its chain and the surface after them.
const PHOTON_BOUNCES: u32 = 16;

/// The reference photon map's cell offset in frame `index` of sample set `seed`, in [0, 1)^3 of
/// a cell.
fn photon_offset(index: u32, seed: u32) -> [f32; 3] {
    let mut h = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    [0u32; 3].map(|_| {
        h ^= h >> 16;
        h = h.wrapping_mul(0x7FEB_352D);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846C_A68B);
        h ^= h >> 16;
        (h >> 8) as f32 / (1u32 << 24) as f32
    })
}

/// `path`'s `SSS_PRIMARY_ONLY`.
const SSS_PRIMARY_ONLY: u32 = 1;

/// `MATERIAL_NONE`.
pub(crate) const NO_MATERIAL: u32 = u32::MAX;

/// A [`LightingView`] target and its storage view.
struct Target {
    texture: Texture,
    store: StorageImageIndex,
}

/// A frame's light feedback: written by its view dispatch, read by the next frame's local
/// samplers.
struct Feedback {
    target: Target,
    sampled: TextureIndex,
    /// The scene's light tables when the feedback was written, which its ids of emissive
    /// entries and of the environment follow.
    tables: LightTables,
}

/// A target the view holds only while a feature is on, with its own image slots.
struct OptionalTarget {
    texture: Texture,
    store: StorageImageIndex,
    sampled: TextureIndex,
    device: Device,
}

impl OptionalTarget {
    fn new(device: &Device, format: Format, extent: [u32; 2], name: &str) -> Result<Self, Error> {
        let desc = TextureDesc::d2(
            format,
            extent[0],
            extent[1],
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        );
        let texture = device.create_texture(&desc, TextureMemory::Dedicated, name)?;
        let view = orr_gpu::ViewDesc::default();
        let store = device.write_storage(device.alloc_image_slot()?, &texture, &view);
        let sampled = device.write_sampled(device.alloc_image_slot()?, &texture, &view);
        Ok(Self {
            texture,
            store,
            sampled,
            device: device.clone(),
        })
    }
}

impl Drop for OptionalTarget {
    fn drop(&mut self) {
        self.device.free_image_slot_after_submitted(self.store.0);
        self.device.free_image_slot_after_submitted(self.sampled.0);
    }
}

/// The background layer behind out-of-focus foreground and the frame the background dispatch
/// last wrote it.
struct DofLayer {
    target: OptionalTarget,
    frame: u64,
}

/// A pixel's front layer over the G-buffer: one 64-bit word per pixel, so that a fragment's
/// depth and motion enter together and overlapping fragments in any order leave the nearest,
/// and the targets the words resolve into.
struct OverlayFront {
    words: Allocation<u64>,
    depth: OptionalTarget,
    motion: OptionalTarget,
    weight: OptionalTarget,
    roots: Ring<OverlayFrontRoot>,
    frame: u64,
}

impl OverlayFront {
    fn new(device: &Device, extent: [u32; 2]) -> Result<Self, Error> {
        let words = device_local(
            device,
            u64::from(extent[0]) * u64::from(extent[1]),
            "overlay front words",
        )?;
        let depth = OptionalTarget::new(device, Format::R32Float, extent, "overlay depth")?;
        let motion = OptionalTarget::new(device, Format::Rg16Float, extent, "overlay motion")?;
        let weight = OptionalTarget::new(device, Format::R16Float, extent, "overlay weight")?;
        Ok(Self {
            words,
            depth,
            motion,
            weight,
            roots: Ring::new(device, "overlay front roots")?,
            frame: 0,
        })
    }

    fn dispatch(
        &mut self,
        list: &mut CommandList<'_>,
        pipeline: &ComputePipeline,
        name: &'static str,
        [w, h]: [u32; 2],
        overlay: TextureIndex,
    ) {
        // The frame's clear and resolve share a root: the clear reads nothing the resolve changes.
        let root = self.roots.write(
            (self.frame % FRAME_SLOTS as u64) as usize,
            OverlayFrontRoot {
                words: self.words.gpu(),
                depth: self.depth.store,
                motion: self.motion.store,
                weight: self.weight.store,
                overlay,
                width: w,
                height: h,
            },
        );
        let [gx, gy, _] = pipeline.workgroup();
        list.begin_pass(name, false);
        list.dispatch(pipeline, root, [w.div_ceil(gx), h.div_ceil(gy), 1]);
        list.end_pass();
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
const DOF_GUIDE: &str = "dof guide";

/// What the passes after ray reconstruction and the consumer read.
const TO_FRAME_END: Span = Span::new(ViewStep::Lighting, ViewStep::After);
/// What only ray reconstruction reads.
const TO_RECONSTRUCTION: Span = Span::new(ViewStep::Lighting, ViewStep::Rr);

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
    /// Held only for ray reconstruction.
    dof_guide: Option<ArenaTexture>,
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
            dof_guide: arena.texture(DOF_GUIDE).cloned(),
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
    /// The feedback the next view dispatch writes; the other holds the last frame's.
    feedback_index: usize,
    local: Allocation<LocalLights>,
    local_tiles: [u32; 2],
    slots: Vec<u32>,
    /// The layer memory: each pixel's first layer for the guide
    /// resolve, the traced tiles and their arguments (`traced_tiles` in `path.slang`); the
    /// frame that last wrote it, and whether it kept the frosted-glass term's inputs.
    layers: Allocation<LayerGuide>,
    layers_frame: Option<u64>,
    layers_frost: bool,
    layer_tile_roots: Ring<LayerTilesRoot>,
    local_roots: Ring<local::Root>,
    guide_roots: Ring<GuideRoot>,
    frost_roots: Ring<FrostRoot>,
    previous: Option<ViewDesc>,
    /// [`SceneLighting::resets`] at the last [`Self::begin`].
    scene_resets: u64,
    history_reset: bool,
    pub(crate) media: MediaView,
    dof: Option<DofLayer>,
    overlay_front: Option<OverlayFront>,
    /// The reference's photon map, while the view traces one.
    photon_map: Option<PhotonMap>,
    /// What the staged paths keep between stages, made at their first frame: the view's and
    /// its update paths'.
    staged: Option<StagedMemory>,
    /// The memory of the scene's photons while they run on the stages and one slice of
    /// `staged` holds fewer paths than the scene has photons: a slice costs its rounds of
    /// dispatches whatever it holds, so the photons take memory of their own number.
    staged_photons: Option<StagedMemory>,
}

impl LightingView {
    /// The targets a view's arena holds for its [`LightingView`] at `extent`. With
    /// `reconstructed`, ray reconstruction's output replaces the radiance after it, later
    /// passes read lighting's depth and motion, and the view holds a depth-of-field guide.
    pub fn requests(extent: [u32; 2], reconstructed: bool) -> Vec<ArenaRequest> {
        let (radiance, surface) = if reconstructed {
            (TO_RECONSTRUCTION, TO_FRAME_END)
        } else {
            (TO_FRAME_END, TO_RECONSTRUCTION)
        };
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        let d2 = |format| TextureDesc::d2(format, extent[0], extent[1], usage);
        let guide = |name, format| ArenaRequest::texture(name, d2(format), TO_RECONSTRUCTION);
        let mut requests = vec![
            ArenaRequest::texture(RADIANCE, d2(Format::Rgba16Float), radiance),
            guide(DIFFUSE_ALBEDO, Format::Rgba8Unorm),
            guide(SPECULAR_ALBEDO, Format::Rgba8Unorm),
            guide(NORMAL_ROUGHNESS, Format::Rgba16Float),
            guide(SPECULAR_MOTION, Format::Rg16Float),
            ArenaRequest::texture(RR_DEPTH, d2(Format::R32Float), surface),
            ArenaRequest::texture(RR_MOTION, d2(Format::Rg16Float), surface),
        ];
        if reconstructed {
            requests.push(guide(DOF_GUIDE, Format::R16Float));
        }
        requests
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
                tables: LightTables::default(),
            },
            Feedback {
                sampled: sampled_view(&b.texture)?,
                target: b,
                tables: LightTables::default(),
            },
        ];
        // The tile grid moves by up to a tile each frame, so it spans one tile more each way.
        let local_tiles = extent.map(|e| e.div_ceil(LOCAL_TILE) + 1);
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
            layer_tile_roots: Ring::new(device, "layer tile roots")?,
            local_roots: Ring::new(device, "local light sampler roots")?,
            guide_roots: Ring::new(device, "guide roots")?,
            frost_roots: Ring::new(device, "frosted glass roots")?,
            previous: None,
            scene_resets: 0,
            history_reset: true,
            staged: None,
            staged_photons: None,
            media: MediaView::new(device)?,
            dof: None,
            overlay_front: None,
            photon_map: None,
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

    /// Starts the view's frame and settles its history reset. A camera cut (`desc.camera_cut`),
    /// a camera jump past `lighting.guides.camera_jump` since the view's last frame, a change of
    /// scene or output extent, the view's first frame and [`SceneLighting::reset_history`] since
    /// the view's last frame each set it. A camera cut also starts exposure over, and ray
    /// reconstruction's history is in pre-exposed units. Call once per frame, after the scene's
    /// [`SceneLighting::record`] and before the view's lighting records; returns
    /// [`Self::history_reset`].
    pub fn begin(
        &mut self,
        scene: &SceneLighting,
        desc: &ViewDesc,
        lighting: &ViewLighting,
    ) -> bool {
        let continues = self.previous.is_some_and(|p| {
            p.scene == desc.scene
                && p.output_extent == desc.output_extent
                && !lighting
                    .guides
                    .camera_jump
                    .jumps((p.position, p.rotation), (desc.position, desc.rotation))
        });
        let scene_resets = scene.resets();
        self.history_reset = !continues || desc.camera_cut || scene_resets != self.scene_resets;
        self.scene_resets = scene_resets;
        self.previous = Some(*desc);
        self.history_reset
    }

    /// Whether ray reconstruction drops its history this frame (its `InReset`), as
    /// [`Self::begin`] settled it.
    pub fn history_reset(&self) -> bool {
        self.history_reset
    }

    /// The view's local light samplers as its last paths built them, one per screen tile row
    /// by row, and the tiles in a row and the rows.
    pub fn local_lights(&self) -> (GpuPtr<LocalLights>, [u32; 2]) {
        (self.local.gpu(), self.local_tiles)
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

    /// R16F, ray reconstruction's depth-of-field guide, held where [`Self::requests`] was
    /// `reconstructed`: per pixel `guide_luminance(after - before)` (`(r + 2g + b) / 4`) of
    /// what changed the radiance from what the other guides show. The guide resolve given the
    /// target clears it, and [`LightingKernels::record_frosted_glass`] adds the frosted-glass
    /// term.
    pub fn depth_of_field_guide(&self) -> Option<&Texture> {
        self.targets.dof_guide.as_ref().map(|t| &**t.texture())
    }

    /// The storage view of [`Self::depth_of_field_guide`].
    pub fn depth_of_field_guide_storage(&self) -> Option<StorageImageIndex> {
        self.targets.dof_guide.as_ref().map(|t| t.storage(0))
    }

    /// R32F reverse-Z device depth, 0 on sky pixels.
    pub fn depth(&self) -> &Arc<Texture> {
        self.targets.depth.texture()
    }

    pub fn depth_index(&self) -> TextureIndex {
        self.targets.depth.sampled()
    }

    /// RG16F, previous uv minus current uv, unjittered.
    pub fn motion(&self) -> &Arc<Texture> {
        self.targets.motion.texture()
    }

    pub fn motion_index(&self) -> TextureIndex {
        self.targets.motion.sampled()
    }

    /// The sampled view of the RGBA16F background layer behind out-of-focus foreground, as the
    /// last [`LightingKernels::record_paths`] given a [`DofBackgroundPass`] wrote it: on each
    /// marked pixel, the pre-exposed radiance its lens ray reaches past the mask's depth in RGB
    /// (with the camera segment's media where media apply ran) and that hit's reverse-Z device
    /// depth in A, 0 for sky; (0, 0, 0, -1) on unmarked pixels. `None` once a call runs without
    /// one.
    pub fn dof_background_index(&self) -> Option<TextureIndex> {
        self.dof.as_ref().map(|l| l.target.sampled)
    }

    /// R32F reverse-Z device depth of each pixel's front layer as
    /// [`LightingKernels::record_overlay_front`] last wrote it: the nearest of the first
    /// dielectric or presence layer where the dominant layer lies behind it, and overlay
    /// fragments above 0.1 coverage; 0 on pixels with no such layer. `None` once an opaque
    /// dispatch runs without [`ViewPass::overlay_front`].
    pub fn overlay_depth(&self) -> Option<&Texture> {
        self.overlay_front.as_ref().map(|f| &f.depth.texture)
    }

    /// RG16F, previous uv minus current uv, unjittered, of [`Self::overlay_depth`]'s layer; 0 on
    /// pixels without one.
    pub fn overlay_motion(&self) -> Option<&Texture> {
        self.overlay_front.as_ref().map(|f| &f.motion.texture)
    }

    /// R16F, the share of each pixel that [`Self::overlay_depth`]'s layer takes from the dominant
    /// layer behind it, from the overlay's coverage and a see-through layer's share of the
    /// pixel's light; 0 on pixels without one.
    pub fn overlay_weight(&self) -> Option<&Texture> {
        self.overlay_front.as_ref().map(|f| &f.weight.texture)
    }

    /// The sampled views of [`Self::overlay_depth`], [`Self::overlay_motion`] and
    /// [`Self::overlay_weight`].
    pub fn overlay_front_indices(&self) -> Option<[TextureIndex; 3]> {
        self.overlay_front
            .as_ref()
            .map(|f| [f.depth.sampled, f.motion.sampled, f.weight.sampled])
    }
}

impl LightingView {
    /// The layer guides of `frame`; null when its paths did not run.
    pub(crate) fn layer_guides(&self, frame: u64) -> GpuPtr<LayerGuide> {
        if self.layers_frame == Some(frame) {
            self.layers.gpu()
        } else {
            GpuPtr::NULL
        }
    }

    /// The background layer's storage view when the background dispatch of `frame` wrote it.
    pub(crate) fn dof_background_store(&self, frame: u64) -> StorageImageIndex {
        self.dof
            .as_ref()
            .filter(|l| l.frame == frame)
            .map_or(StorageImageIndex::NONE, |l| l.target.store)
    }

    /// The overlay front words when the paths of `frame` cleared them.
    pub(crate) fn overlay_front_words(&self, frame: u64) -> GpuPtr<u64> {
        self.overlay_front
            .as_ref()
            .filter(|f| f.frame == frame)
            .map_or(GpuPtr::NULL, |f| f.words.gpu())
    }

    /// The froxel counts of `frame`, once its submit has completed; `None` when
    /// [`LightingKernels::record_media`] did not run in that frame or its slot holds a later
    /// frame.
    pub fn media_stats(&self, frame: u64) -> Option<MediaStats> {
        self.media.stats(frame)
    }

    /// The path pipelines the view's last [`LightingKernels::record_paths`] ran, where the
    /// staged loop ran them.
    pub fn paths_ran(&self) -> Option<PathsRan> {
        self.staged.as_ref().map(|m| {
            let (asked, mut ran) = m.ran();
            // The photons ran the sets their own memory holds.
            if let Some(photons) = &self.staged_photons {
                ran.extend(photons.ran().1);
            }
            PathsRan { asked, ran }
        })
    }

    /// The light and fog transparency hooks of `frame` shade particles with, through
    /// [`crate::ParticleDraw`] or their own `particle_shade`; written by
    /// [`LightingKernels::record_media_apply`], null when that frame recorded none or its
    /// [`ViewPass::particle_lighting`] was not set.
    pub fn particle_lighting(&self, frame: u64) -> GpuPtr<ParticleLighting> {
        self.media.particle_lighting(frame)
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
    /// Bytes from its start to the tile count, then the workgroups of the layer guides'
    /// dispatch over the listed tiles.
    args: u64,
}

impl LayerMemory {
    fn new([w, h]: [u32; 2]) -> Self {
        let tiles = u64::from(w.div_ceil(LAYER_TILE)) * u64::from(h.div_ceil(LAYER_TILE));
        let guide = size_of::<LayerGuide>() as u64;
        let args = guide * u64::from(w) * u64::from(h) + 8 * tiles;
        Self {
            guides: (args + 4 * 4).div_ceil(guide),
            args,
        }
    }
}

/// Pixels per side of a tile of the layer guides (`LAYER_TILE`).
const LAYER_TILE: u32 = 8;

/// `LayerTilesRoot.flags` (`dielectric.slang`).
const LAYER_TILES_EVERYWHERE: u32 = 1 << 0;

/// What the view's paths and the guide resolve read for one view and frame.
#[derive(Clone, Copy, Debug)]
pub struct ViewPass {
    pub frame: u64,
    pub scene: GpuPtr<SceneRoot>,
    pub view: GpuPtr<ViewConstants>,
    /// The view's G-buffer at the [`LightingView`]'s extent.
    pub gbuffer: GpuPtr<GBufferGpu>,
    /// The view's raster state, whose smoke list the froxels read.
    pub raster: GpuPtr<RasterGpu>,
    /// [`orr_scene::RasterView::smoke_args`]; the view keeps the memory smoke takes only while
    /// this is set.
    pub smoke_args: Option<GpuPtr<DispatchArgs>>,
    /// The frame's index in every pixel's sample sequence; consecutive frames take
    /// consecutive indices.
    pub sample_index: u32,
    /// Separates sample sets of the same pixels and sample index.
    pub seed: u32,
    /// A dispatch beside the view's traces its background layer
    /// ([`LightingView::dof_background_index`]); `None` frees it.
    pub dof_background: Option<DofBackgroundPass>,
    /// The view dispatch keeps what [`LightingKernels::record_frosted_glass`] reads: set while
    /// ray reconstruction reads a depth-of-field guide.
    pub frosted_glass: bool,
    /// The view keeps a front layer for each pixel ([`LightingView::overlay_depth`]), which
    /// [`LightingKernels::record_overlay_front`] resolves: set while output's
    /// transparency-aware circle of confusion and velocity are on. `false` frees it.
    pub overlay_front: bool,
    /// [`orr_scene::RasterView::features`].
    pub scene_features: SceneFeatures,
    /// Something reads the frame's [`LightingView::particle_lighting`], as a transparency hook
    /// that shades particles does; `false` leaves it null.
    pub particle_lighting: bool,
}

/// What the background layer behind out-of-focus foreground reads in one frame. Each marked
/// pixel traces a second camera ray from a point of the lens's aperture disc through a
/// jittered point of the pixel on the focal plane, starting past the mask's depth, and the
/// path from its hit at the view path's settings, which meets dielectric and presence layers
/// as any path does.
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

/// The kernels beside the path kernels, each compiled as a job of its own, and the small tables
/// they use.
struct SupportKernels {
    layer_tiles: ComputePipeline,
    layer_args: ComputePipeline,
    /// A zero word that starts each frame's tile count.
    zero: Allocation<u32>,
    frost: ComputePipeline,
    overlay_clear: ComputePipeline,
    overlay_resolve: ComputePipeline,
    local: ComputePipeline,
    refine: ComputePipeline,
    media: MediaKernels,
    /// The guide resolve, the layer guides, the froxels' light pass and media apply, made for
    /// what a scene may hold when a view first runs them.
    masked: Arc<MaskedSupport>,
    subsurface_table: SubsurfaceTable,
}

impl SupportKernels {
    /// Its own pipelines that every scene runs alike, in the order [`Self::assemble`] takes
    /// them; [`MediaKernels`]' follow.
    const PIPELINES: [CompilePipeline; 7] = [
        |d, k| pipeline(d, k, &tiles::MODULE, "tiles"),
        |d, k| pipeline(d, k, &tiles::MODULE, "args"),
        |d, k| pipeline(d, k, &frost::MODULE, "main"),
        |d, k| pipeline(d, k, &overlay_front::MODULE, "clear"),
        |d, k| pipeline(d, k, &overlay_front::MODULE, "resolve"),
        |d, k| pipeline(d, k, &local::MODULE, "main"),
        |d, k| pipeline(d, k, &local::MODULE, "refine"),
    ];

    /// Every pipeline of the support kernels.
    fn pipelines() -> impl Iterator<Item = CompilePipeline> {
        Self::PIPELINES.into_iter().chain(MediaKernels::PIPELINES)
    }

    /// The kernels over `pipelines`, which [`Self::pipelines`] compiled from `kernels` in its
    /// order, and `masked`.
    fn assemble(
        device: &Device,
        kernels: &KernelSet,
        pipelines: Vec<ComputePipeline>,
        masked: Arc<MaskedSupport>,
    ) -> Result<Self, Error> {
        let mut pipelines = pipelines.into_iter();
        let mut next = || pipelines.next().expect("a pipeline per entry");
        let mut zero = device.gpu_malloc::<u32>(MemoryKind::Upload, 1, "zero")?;
        zero.as_mut_slice().expect("mapped")[0] = 0;
        Ok(Self {
            layer_tiles: next(),
            layer_args: next(),
            zero,
            frost: next(),
            overlay_clear: next(),
            overlay_resolve: next(),
            local: next(),
            refine: next(),
            media: MediaKernels::new(device, kernels, &mut pipelines)?,
            masked,
            subsurface_table: SubsurfaceTable::new(device)?,
        })
    }

    /// Compiles the pipelines every scene runs alike one after another on the calling thread.
    fn new(device: &Device, kernels: &KernelSet) -> Result<Self, Error> {
        let pipelines = Self::pipelines()
            .map(|compile| compile(device, kernels))
            .collect::<Result<Vec<_>, _>>()?;
        let masked = Arc::new(MaskedSupport::new(device, kernels));
        Self::assemble(device, kernels, pipelines, masked)
    }
}

/// The rounds of the staged loop a path of `max_bounces` bounces and `dielectric_depth` layers
/// may take: one per iteration of a path (`path_iterations`), which traces at most once,
/// and one for the vertex its start leaves to shade.
fn rounds(max_bounces: u32, dielectric_depth: u32, reference: bool) -> Result<u32, Error> {
    let walks = if reference {
        u64::from(max_bounces) + 1
    } else {
        1
    };
    let rounds = u64::from(max_bounces) + 3 + walks + u64::from(dielectric_depth) + 1;
    if rounds > u64::from(MAX_ROUNDS) {
        return Err(Error::Settings(
            "the bounces and dielectric depth give paths more rounds than their queues can hold",
        ));
    }
    Ok(rounds as u32)
}

/// [`rounds`] of a path under `root`.
fn path_rounds(root: &PathRoot) -> u32 {
    let reference = root.dielectric_flags & DIELECTRIC_REFERENCE != 0;
    rounds(root.max_bounces, root.dielectric_depth, reference)
        .expect("record_paths checked the view's and the photons' rounds")
}

/// Slots of the memory a view keeps for the scene's photons at most, where its own holds
/// fewer: the photons of a frame at [`crate::CausticSettings`]'s default count.
const PHOTON_SLOTS: u32 = 1 << 18;

/// Whether path kernels run as ray dispatches.
fn runs_rays(entry: PathEntry) -> bool {
    match entry {
        // The compute entries: a shade stage's reorder costs more than its sorted lanes save.
        PathEntry::Fastest | PathEntry::Compute => false,
        PathEntry::RayDispatch => true,
    }
}

const SUPPORT: &str = "lighting compute kernels";

/// Queues `compile` at `urgency`, which writes the device's pipeline cache file when it finishes.
pub(crate) fn compile_job<T: Send + 'static>(
    device: &Device,
    loading: &Loading,
    urgency: Urgency,
    compile: impl FnOnce(&Device) -> Result<T, Error> + Send + 'static,
) -> JobResult<Result<T, Error>> {
    let device = device.clone();
    loading.spawn_result_at(urgency, move || {
        let compiled = compile(&device);
        if let Err(e) = device.save_pipeline_cache() {
            log::warn!("pipeline cache not saved: {e}");
        }
        compiled
    })
}

/// A compile job's result, or how it failed. Call once it finished.
fn compiled<T>(job: &JobResult<Result<T, Error>>, kernel: &'static str) -> Result<T, Error> {
    match job.take().expect("the job has finished") {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(Error::Compile {
            kernel,
            source: Box::new(e),
        }),
        Err(message) => Err(Error::CompilePanicked { kernel, message }),
    }
}

/// [`LightingKernels`] compiling as jobs on the loading side, from
/// [`LightingKernels::spawn`]. Dropping it cancels the jobs no thread has started; one
/// already compiling finishes on its thread, since the driver cannot be interrupted.
pub struct CompilingKernels {
    device: Device,
    kernels: &'static KernelSet,
    loading: Arc<Loading>,
    /// The staged loop's pipelines compile when paths first ask for them, which a view may
    /// while the other kernels compile.
    paths: Arc<StagedPaths>,
    /// One job per support pipeline, in [`SupportKernels::pipelines`]' order.
    support: Vec<JobResult<Result<ComputePipeline, Error>>>,
    /// The support kernels made per mask, which a view asks for as it does its path pipelines.
    masked: Arc<MaskedSupport>,
    tables: BsdfTables,
}

impl CompilingKernels {
    fn jobs(&self) -> Vec<&dyn JobState> {
        self.support.iter().map(|j| j as &dyn JobState).collect()
    }

    /// How many jobs have not finished.
    pub fn outstanding(&self) -> u32 {
        self.jobs().iter().filter(|job| !job.finished()).count() as u32
    }

    /// Asks for the path pipelines a view's frame with `needs` takes under `lighting`, as
    /// [`LightingKernels::prepare_paths`] does once the kernels are ready: asked for here,
    /// they compile beside the kernels and not after them. The pipelines paths do without
    /// wait until a view renders. An error where one of them failed to compile.
    pub fn prepare_paths(
        &self,
        scene: &SceneLighting,
        needs: PathNeeds,
        lighting: &ViewLighting,
    ) -> Result<(), Error> {
        let (wanted, stages) = self.paths.wanted(scene, needs, lighting);
        self.paths.ask(wanted, &stages, lighting)?;
        ask_support(&self.masked, needs, lighting)?;
        Ok(())
    }

    /// How many pipelines asked for through [`Self::prepare_paths`] are still compiling.
    pub fn paths_outstanding(&self) -> u32 {
        self.paths.stages.outstanding() + self.masked.outstanding()
    }

    /// Whether a thread has started any job.
    pub fn started(&self) -> bool {
        self.jobs().iter().any(|job| job.started())
    }

    /// Blocks until every job has finished, running queued jobs on the calling thread meanwhile,
    /// so it finishes with no workers. `loading` is the side the jobs were queued on.
    pub fn wait(&self, loading: &Loading) {
        for job in self.jobs() {
            job.wait(loading);
        }
    }

    /// The kernels, or the first failure, once [`Self::outstanding`] is 0.
    pub fn finish(&self) -> Result<LightingKernels, Error> {
        assert_eq!(self.outstanding(), 0, "kernels are still compiling");
        let mut support = Vec::new();
        for job in &self.support {
            support.push(compiled(job, SUPPORT)?);
        }
        let out = LightingKernels {
            paths: Arc::clone(&self.paths),
            support: SupportKernels::assemble(
                &self.device,
                self.kernels,
                support,
                Arc::clone(&self.masked),
            )?,
            _tables: self.tables.clone(),
        };
        out.attach(&self.device, &self.loading);
        Ok(out)
    }
}

/// A compile job, whatever it makes.
trait JobState {
    fn finished(&self) -> bool;
    fn started(&self) -> bool;
    fn wait(&self, loading: &Loading);
}

impl<T: Send + 'static> JobState for JobResult<T> {
    fn finished(&self) -> bool {
        JobResult::finished(self)
    }
    fn started(&self) -> bool {
        JobResult::started(self)
    }
    fn wait(&self, loading: &Loading) {
        JobResult::wait(self, loading);
    }
}

/// The staged loop's pipelines, which run every source's paths.
struct StagedPaths {
    stages: StagedKernels,
    /// The kernels link a medium.
    linked_medium: bool,
}

/// The path pipelines a view's frame ran, by what they hold: `specialize.slang`'s bits in the
/// low word and `scene_may.slang`'s in the high one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathsRan {
    /// What the view asked its pipelines to hold: what its scene and settings say its paths
    /// may meet, or everything without [`ViewLighting::specialize`].
    pub asked: u64,
    /// What the pipelines each of its sources ran were made to hold.
    pub ran: Vec<u64>,
}

impl PathsRan {
    /// No source ran pipelines that hold more than the view asked for, as it does while the
    /// ones made for it compile. A source may run ones that hold less: those made for it alone,
    /// where the view asked for several sources' together.
    pub fn exact(&self) -> bool {
        self.ran.iter().all(|&made| made & !self.asked == 0)
    }
}

/// The path pipelines [`LightingKernels::wait_paths`] waited for, which stay while this does.
#[must_use = "the pipelines waited for stay only while this is held"]
pub struct WaitedPaths {
    _set: Arc<StageSet>,
}

/// What decides the path pipelines of a view's frame, known before the view is rastered.
#[derive(Clone, Copy, Debug)]
pub struct PathNeeds {
    /// [`orr_scene::Scenes::scene_features`]: [`ViewPass::scene_features`] to be.
    pub features: SceneFeatures,
    /// [`ViewPass::frosted_glass`] to be.
    pub frosted_glass: bool,
    /// Whether [`ViewPass::dof_background`] will be set.
    pub dof_background: bool,
    /// [`ViewPass::particle_lighting`] to be: the froxels' light pass is then among the kernels
    /// asked for. A caller that cannot tell says false; once a pass has declared it, it is
    /// asked for whatever this says.
    pub particle_lighting: bool,
}

/// What each source's paths may meet in a frame (`specialize.slang`'s bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathMay {
    view: u64,
    background: u64,
    update: u64,
    photons: u64,
}

impl PathMay {
    const ALL: Self = Self {
        view: MAY_ALL,
        background: MAY_ALL,
        update: MAY_ALL,
        photons: MAY_ALL,
    };
}

fn bit(on: bool, bit: u64) -> u64 {
    if on { bit } else { 0 }
}

/// `PathRoot.features` of a view's paths in a scene that may hold `features`: whether they
/// walk and split by wavelength, and the parts their state holds.
fn path_features(features: SceneFeatures, lighting: &ViewLighting) -> u32 {
    let bounded = features.contains(SceneFeatures::INTERIORS);
    // The eye's interior is one whatever the materials say.
    let interiors = bounded || lighting.camera_medium.is_some();
    // A stack face bounds an interior, and a stack bends directional lights alone.
    let stack = bounded && features.contains(SceneFeatures::DIRECTIONAL_LIGHTS);
    // The walk stage runs a subsurface walk an opaque vertex starts; an interior's walk is the
    // trace stage's.
    let walk =
        features.contains(SceneFeatures::WALKING) && features.contains(SceneFeatures::SUBSURFACE);
    let part = |on: bool, feature: u32| if on { feature } else { 0 };
    part(walk, FEATURE_WALK)
        | part(disperses(features, lighting), FEATURE_DISPERSION)
        | part(interiors, FEATURE_INTERIORS)
        | part(stack, FEATURE_STACK)
}

/// The features an update path keeps of its view's: it does not walk, nor split by wavelength.
const UPDATE_FEATURES: u32 = FEATURE_INTERIORS | FEATURE_STACK;

/// What paths whose root has `features` may meet.
fn features_may(features: u32) -> u64 {
    bit(features & FEATURE_WALK != 0, MAY_WALK)
        | bit(features & FEATURE_DISPERSION != 0, MAY_DISPERSION)
        | bit(features & FEATURE_INTERIORS != 0, MAY_INTERIORS)
        | bit(features & FEATURE_STACK != 0, MAY_STACK)
}

/// What the paths of a scene that may hold `features` may meet, beyond what their roots say:
/// the low word's bits the scene decides, and the high word, what its surface evaluations and
/// hits may meet.
fn scene_may(features: SceneFeatures) -> u64 {
    u64::from(features.surface_may()) << 32
        | bit(features.contains(SceneFeatures::COAT), MAY_COAT)
        | bit(features.contains(SceneFeatures::FUZZ), MAY_FUZZ)
        | bit(features.contains(SceneFeatures::THIN_FILM), MAY_FILM)
        | bit(features.contains(SceneFeatures::SUBSURFACE), MAY_SUBSURFACE)
        | bit(features.contains(SceneFeatures::LAYERS), MAY_LAYERS)
        | bit(features.contains(SceneFeatures::COVERAGE), MAY_COVERAGE)
        | bit(features.contains(SceneFeatures::PARTICLES), MAY_PARTICLES)
        | bit(
            features.contains(SceneFeatures::LOCAL_LIGHTS),
            MAY_LOCAL_LIGHTS,
        )
        | bit(
            features.contains(SceneFeatures::DIRECTIONAL_LIGHTS),
            MAY_DIRECTIONAL,
        )
        | bit(features.contains(SceneFeatures::EMISSIVE), MAY_EMISSIVE)
        | bit(
            features.contains(SceneFeatures::MEDIUM_PROXIES),
            MAY_PROXIES,
        )
        | bit(features.contains(SceneFeatures::SMOKE), MAY_SMOKE)
}

/// What `root` says its paths may meet: the bits of `specialize.slang` a root decides.
fn root_may(root: &PathRoot) -> u64 {
    let flag = |f: u32| root.dielectric_flags & f != 0;
    features_may(root.features)
        | bit(root.caustics != GpuPtr::NULL, MAY_CAUSTICS)
        | bit(root.mode != 0 || flag(DIELECTRIC_DEBUG_LAYER), MAY_DEBUG)
        | bit(flag(DIELECTRIC_REFERENCE), MAY_REFERENCE)
        | bit(flag(DIELECTRIC_FROST), MAY_FROST)
        | bit(root.aperture_radius > 0.0, MAY_LENS)
        | bit(root.local != GpuPtr::NULL, MAY_LOCAL)
        | bit(root.froxels != GpuPtr::NULL, MAY_FROXELS)
        | bit(root.cache_flags & CACHE_TERMINATE != 0, MAY_CACHE)
}

/// `PathRoot.mode` of a view under `lighting`.
fn path_mode(lighting: &ViewLighting) -> u32 {
    match lighting.debug_view {
        LightingDebugView::None
        | LightingDebugView::Guide(_)
        | LightingDebugView::DielectricLayer
        | LightingDebugView::Media
        | LightingDebugView::Froxels
        | LightingDebugView::Traced(_) => 0,
        LightingDebugView::DirectOnly => 1,
        LightingDebugView::IndirectOnly => 2,
        LightingDebugView::NeeOnly => 3,
        LightingDebugView::BsdfOnly => 4,
        LightingDebugView::RaysPerPixel => 5,
        LightingDebugView::RayOriginBias => 6,
        LightingDebugView::WorldCache => 7,
        LightingDebugView::WorldCacheLod => 8,
        LightingDebugView::Subsurface => 9,
        LightingDebugView::Caustics => 10,
    }
}

/// The share of a view's first-vertex candidates its local light samplers draw: 0 under the
/// reference, which has none.
fn local_share(lighting: &ViewLighting) -> f32 {
    if lighting.reference.is_some() {
        0.0
    } else {
        lighting.local_light_share.clamp(0.0, MAX_LOCAL_LIGHT_SHARE)
    }
}

/// Whether a view's paths may split by wavelength.
fn disperses(features: SceneFeatures, lighting: &ViewLighting) -> bool {
    lighting.dispersion && features.contains(SceneFeatures::DISPERSIVE)
}

/// Whether the paths of a view with `lighting` read `scene`'s caustics in a frame the scene may
/// hold `features`, once the scene's pass has recorded them: it records them where the scene
/// may hold a caster.
fn gathers_caustics(
    scene: &SceneLighting,
    features: SceneFeatures,
    lighting: &ViewLighting,
) -> bool {
    features.contains(SceneFeatures::CASTERS)
        && scene.caustics().photons() > 0
        && lighting.reference.is_none_or(|r| r.photons > 0)
}

/// What the masked support kernels of a view's frame may meet (a mask as the path pipelines'), under
/// kernels that link a medium where `linked_medium`: what the scene's features, the view's
/// settings and `frosted_glass` ([`ViewPass::frosted_glass`]) rule in. The layer guides run under
/// the view's path root, which asks nothing beyond it. The bits no input here decides stay set.
fn support_may(
    linked_medium: bool,
    features: SceneFeatures,
    frosted_glass: bool,
    lighting: &ViewLighting,
) -> u64 {
    if !lighting.specialize {
        return MAY_ALL;
    }
    const DECIDED: u64 = MAY_COAT
        | MAY_FUZZ
        | MAY_FILM
        | MAY_SUBSURFACE
        | MAY_LAYERS
        | MAY_COVERAGE
        | MAY_PARTICLES
        | MAY_LOCAL_LIGHTS
        | MAY_DIRECTIONAL
        | MAY_EMISSIVE
        | MAY_PROXIES
        | MAY_SMOKE
        | MAY_WALK
        | MAY_DISPERSION
        | MAY_INTERIORS
        | MAY_STACK
        | MAY_LENS
        | MAY_FROXELS
        | MAY_DEBUG
        | MAY_REFERENCE
        | MAY_FROST
        | MAY_LOCAL
        | MAY_CACHE;
    let reference = lighting.reference.is_some();
    let lens = traces_lens(lighting);
    let froxels = !reference && fogged(linked_medium, features, lighting);
    // The scene decides the whole high word.
    (u64::from(u32::MAX) & !DECIDED)
        | scene_may(features)
        | features_may(path_features(features, lighting))
        | bit(lens, MAY_LENS)
        | bit(froxels, MAY_FROXELS)
        | bit(path_debug(lighting.debug_view), MAY_DEBUG)
        | bit(reference, MAY_REFERENCE)
        | bit(frosted_glass, MAY_FROST)
        | bit(local_share(lighting) > 0.0, MAY_LOCAL)
        | bit(!reference, MAY_CACHE)
}

/// Whether `lighting`'s reference traces its camera rays through a lens: one with no aperture
/// is the pinhole the G-buffer was rastered through.
fn traces_lens(lighting: &ViewLighting) -> bool {
    lighting
        .reference
        .and_then(|r| r.lens)
        .is_some_and(|l| l.aperture_radius > 0.0)
}

/// Which masked support kernels a frame of a scene that may hold `features` runs under
/// `lighting`, with [`ViewPass::particle_lighting`] as `particle_lighting`: what a view asks for
/// ahead and what its frame's records take. The layer guides run where a tile can be listed:
/// the scene may hold a layer, or the view names the eye's interior or traces a lens, which list
/// every tile.
fn support_runs(
    masked: &MaskedSupport,
    features: SceneFeatures,
    particle_lighting: bool,
    lighting: &ViewLighting,
) -> Runs {
    let lens = traces_lens(lighting);
    let readers = readers(
        masked.linked_medium(),
        features,
        particle_lighting,
        lighting,
    );
    Runs {
        layer_guides: features.contains(SceneFeatures::LAYERS)
            || lighting.camera_medium.is_some()
            || lens,
        froxel_light: readers.lit,
        media_apply: readers.fogged && shows_media(lighting),
    }
}

/// Asks for the masked support kernels a view's frame with `needs` runs under `lighting`, and
/// says whether they are ready.
fn ask_support(
    masked: &MaskedSupport,
    needs: PathNeeds,
    lighting: &ViewLighting,
) -> Result<bool, Error> {
    let may = support_may(
        masked.linked_medium(),
        needs.features,
        needs.frosted_glass,
        lighting,
    );
    let particle_lighting = needs.particle_lighting || masked.particle_lighting();
    let runs = support_runs(masked, needs.features, particle_lighting, lighting);
    masked.ask(may, runs, lighting.specialize)
}

impl StagedPaths {
    fn new(stages: StagedKernels, linked_medium: bool) -> Self {
        Self {
            stages,
            linked_medium,
        }
    }

    /// Queues, behind the pipelines a view cannot render without and ahead of mounts, the
    /// stages that may meet everything: they run a scene that comes to hold more than its own
    /// were made for, which would wait in black for new ones otherwise.
    fn start_general(&self) {
        // Every source's stages: a scene that comes to hold more runs any of them.
        let mut stages = Source::View.stages(true);
        stages.extend([Stage::Background, Stage::Update, Stage::Photon]);
        self.stages.start_general(&stages);
    }

    /// What the paths of a frame with `needs` under `lighting` may meet: what a view asks
    /// pipelines for, and what [`LightingKernels::record_paths`] runs each of its sources with.
    fn predict(&self, scene: &SceneLighting, needs: PathNeeds, lighting: &ViewLighting) -> PathMay {
        if !lighting.specialize {
            return PathMay::ALL;
        }
        let features = needs.features;
        let reference = lighting.reference.is_some();
        let lens = traces_lens(lighting);
        let froxels = !reference && fogged(self.linked_medium, features, lighting);
        let path = path_features(features, lighting);
        let shared = scene_may(features)
            | bit(gathers_caustics(scene, features, lighting), MAY_CAUSTICS)
            | bit(lens, MAY_LENS)
            | bit(froxels, MAY_FROXELS);
        let layer = lighting.debug_view == LightingDebugView::DielectricLayer;
        let background = shared
            | features_may(path)
            | bit(layer, MAY_DEBUG)
            | bit(reference, MAY_REFERENCE)
            | bit(needs.frosted_glass, MAY_FROST)
            | bit(local_share(lighting) > 0.0, MAY_LOCAL)
            | bit(!reference, MAY_CACHE);
        PathMay {
            view: background | bit(path_mode(lighting) != 0, MAY_DEBUG),
            background,
            // Update paths run outside the reference and end in the cache.
            update: shared | features_may(path & UPDATE_FEATURES) | MAY_CACHE,
            photons: shared | features_may(path & !FEATURE_WALK) | bit(reference, MAY_REFERENCE),
        }
    }

    /// The path pipelines a view's frame with `needs` takes under `lighting`: what they may
    /// meet together and their stages.
    fn wanted(
        &self,
        scene: &SceneLighting,
        needs: PathNeeds,
        lighting: &ViewLighting,
    ) -> (u64, Vec<Stage>) {
        let caustics = gathers_caustics(scene, needs.features, lighting);
        let may = self.predict(scene, needs, lighting);
        let mut wanted = may.view | Source::View.may();
        let mut stages =
            Source::View.stages(path_features(needs.features, lighting) & FEATURE_WALK != 0);
        if needs.dof_background {
            wanted |= may.background | Source::Background.may();
            stages.push(Stage::Background);
        }
        // Asked for together, a frame's sources share their pipelines. Update paths run
        // outside the reference.
        if lighting.reference.is_none() {
            wanted |= may.update | Source::Update.may();
            stages.push(Stage::Update);
        }
        if caustics {
            wanted |= may.photons | Source::Photon.may();
            stages.push(Stage::Photon);
        }
        (wanted, stages)
    }

    /// Asks for pipelines that hold `wanted` over `stages`, as [`Self::wanted`] gives them, and
    /// says whether some are ready.
    fn ask(&self, wanted: u64, stages: &[Stage], lighting: &ViewLighting) -> Result<bool, Error> {
        let ready = self.stages.prepare(wanted, stages)?;
        if lighting.specialize {
            self.start_general();
        }
        Ok(ready)
    }
}

/// The per-view lighting kernels of one device.
pub struct LightingKernels {
    paths: Arc<StagedPaths>,
    support: SupportKernels,
    _tables: BsdfTables,
}

impl LightingKernels {
    /// Compiles the kernels every scene runs alike one after another on the calling thread.
    /// `kernels` replaces modules as [`SceneLighting::new`] does. The kernels read `tables` and
    /// keep them alive. The path pipelines, the guide resolve, the layer guides, the froxels'
    /// light pass and media apply compile for what a scene may hold when a view first asks for
    /// them ([`Self::prepare_paths`]).
    pub fn new(
        device: &Device,
        kernels: &KernelSet,
        tables: &BsdfTables,
        entry: PathEntry,
    ) -> Result<Self, Error> {
        let rays = runs_rays(entry);
        let paths = Arc::new(StagedPaths::new(
            StagedKernels::new(device, kernels, rays),
            links_medium(kernels),
        ));
        Ok(Self {
            paths,
            support: SupportKernels::new(device, kernels)?,
            _tables: tables.clone(),
        })
    }

    /// [`Self::new`] as jobs on `loading`, one per pipeline, each
    /// writing the device's pipeline cache file when it finishes, so an interrupted first run
    /// keeps what it compiled. Returns at once; a job that fails or panics reports it through
    /// [`CompilingKernels::finish`].
    pub fn spawn(
        device: &Device,
        kernels: &'static KernelSet,
        tables: &BsdfTables,
        entry: PathEntry,
        loading: &Arc<Loading>,
    ) -> CompilingKernels {
        let rays = runs_rays(entry);
        let paths = Arc::new(StagedPaths::new(
            StagedKernels::new(device, kernels, rays),
            links_medium(kernels),
        ));
        paths.stages.attach(device, loading);
        let masked = Arc::new(MaskedSupport::new(device, kernels));
        masked.attach(device, loading);
        CompilingKernels {
            device: device.clone(),
            kernels,
            loading: Arc::clone(loading),
            paths,
            masked,
            support: SupportKernels::pipelines()
                .map(|compile| {
                    compile_job(device, loading, Urgency::Needed, move |d| {
                        compile(d, kernels)
                    })
                })
                .collect(),
            tables: tables.clone(),
        }
    }

    /// Compiles the path kernels' later pipelines on `device` as jobs on `loading` from here on,
    /// whatever device the kernels were made on (one that only reads its caches, say).
    /// [`CompilingKernels::finish`] calls it; kernels from [`Self::new`] compile their later
    /// pipelines on the calling thread until this call.
    pub fn attach(&self, device: &Device, loading: &Arc<Loading>) {
        self.paths.stages.attach(device, loading);
        self.support.masked.attach(device, loading);
    }

    /// Queues every path pipeline no view has asked for yet: what [`Self::prepare_paths`] queues
    /// behind a view's own. A caller that compiles ahead of its first frame calls this
    /// and runs the loading side's jobs.
    pub fn compile_rest(&self) {
        self.paths.start_general();
        self.support.masked.start_general();
    }

    /// The masked support kernels with the mask of `pass`'s frame under `lighting`, and
    /// which of them the frame runs.
    fn masked_frame(&self, pass: &ViewPass, lighting: &ViewLighting) -> MaskedFrame<'_> {
        let kernels = &*self.support.masked;
        MaskedFrame {
            kernels,
            may: support_may(
                kernels.linked_medium(),
                pass.scene_features,
                pass.frosted_glass,
                lighting,
            ),
            runs: support_runs(
                kernels,
                pass.scene_features,
                pass.particle_lighting,
                lighting,
            ),
        }
    }

    /// Asks for the path pipelines and the masked support kernels a view's frame with `needs`
    /// takes under `lighting`, and says whether they are ready. The frame's record calls wait
    /// for them otherwise, so a caller that would rather show a frame without paths asks here
    /// first. An error where one of them failed to compile.
    pub fn prepare_paths(
        &self,
        scene: &SceneLighting,
        needs: PathNeeds,
        lighting: &ViewLighting,
    ) -> Result<bool, Error> {
        let support = ask_support(&self.support.masked, needs, lighting)?;
        let k = &self.paths;
        let (wanted, stages) = k.wanted(scene, needs, lighting);
        Ok(k.ask(wanted, &stages, lighting)? && support)
    }

    /// Waits for the kernels paths do without until they are ready and then for the pipelines
    /// made for exactly what a view's frame with `needs` takes under `lighting`, running queued
    /// loading jobs on this thread meanwhile: the frame then runs what it would once every
    /// compile had finished, whatever else is ready. The caller keeps what this returns until
    /// [`Self::record_paths`] has recorded that frame: the pipelines stay for it meanwhile.
    pub fn wait_paths(
        &self,
        scene: &SceneLighting,
        needs: PathNeeds,
        lighting: &ViewLighting,
    ) -> Result<WaitedPaths, Error> {
        // The masked support kernels compile beside the path pipelines waited for first.
        ask_support(&self.support.masked, needs, lighting)?;
        let (wanted, stages) = self.paths.wanted(scene, needs, lighting);
        let waited = self.paths.stages.wait_exact(wanted, &stages)?;
        let masked = &self.support.masked;
        let may = support_may(
            masked.linked_medium(),
            needs.features,
            needs.frosted_glass,
            lighting,
        );
        let particle_lighting = needs.particle_lighting || masked.particle_lighting();
        let runs = support_runs(masked, needs.features, particle_lighting, lighting);
        masked.wait_exact(may, runs)?;
        Ok(WaitedPaths { _set: waited })
    }

    /// How many pipelines asked for are still compiling.
    pub fn outstanding(&self) -> u32 {
        self.paths.stages.outstanding() + self.support.masked.outstanding()
    }

    /// Records the froxel passes of `pass` after the scene's [`SceneLighting::record`], the
    /// view's raster half and [`LightingView::begin`]: picks the shadowed directional lights,
    /// writes the linked medium into the froxels, the light reaching those in front of the
    /// view's surfaces (blended with their reprojected history) and each column's in-scatter
    /// and transmittance, then the aerial-perspective LUT from `near_range` to
    /// [`MediaSettings::ap_range`]. The outputs are visible to later compute and ray stages.
    /// Makes the view's froxel targets at its first call and when the extent or the media
    /// settings that shape them change; a history reset, a new `near_range` and new targets
    /// restart the froxel history.
    ///
    /// A pass runs only where something reads what it writes. A camera segment can hold a
    /// medium, which the froxels' medium, their integration, the aerial-perspective LUT and
    /// media apply serve, while the kernels link one, the scene holds a medium proxy, a smoke
    /// entry or a material that bounds an interior, or [`ViewLighting::camera_medium`] is set.
    /// The froxels' light has a reader while the kernels link a medium, the scene holds a
    /// medium proxy, a smoke entry or an interior that scatters, `camera_medium` is set or
    /// [`ViewPass::particle_lighting`] is. [`LightingDebugView::Media`] and
    /// [`LightingDebugView::Froxels`] read everything. With neither kind of reader the view
    /// frees its froxel targets and only the shadowed lights are picked. The froxel history
    /// starts over when the light gets a reader again.
    pub fn record_media(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> Result<(), Error> {
        let masked = self.masked_frame(pass, lighting);
        self.support
            .media
            .record(list, scene, view, pass, lighting, masked)
    }

    /// Records media apply into `view`'s radiance after [`Self::record_media`] and the paths
    /// that wrote the radiance: each pixel's radiance x the transmittance over its camera
    /// segment to its surface, plus the in-scatter over it, from the froxels to the scene's
    /// `near_range`, the aerial-perspective LUT to [`MediaSettings::ap_range`] and a march
    /// past it. Sky pixels take the froxels' whole range; the environment carries the medium
    /// beyond. Medium proxies the camera ray crosses, at most the nearest 4 per pixel, are
    /// marched per pixel at their voxel spacing, lit by the froxels' isotropic light and
    /// [`MediaSettings::proxy_shadow_rays`] shadow rays. Smoke producers come through the
    /// froxels inside `near_range` and as one lit layer per pixel past it. The reference marches
    /// the segment per pixel instead. Where no camera segment can hold a medium
    /// ([`Self::record_media`]) the radiance stays as the paths wrote it. Also writes the frame's
    /// [`LightingView::particle_lighting`] for the transparency hook after it. Orders nothing
    /// after it. Waits for media apply's pipeline where no view asked for it ahead
    /// ([`Self::prepare_paths`]); an error where it failed to compile.
    pub fn record_media_apply(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> Result<(), Error> {
        let masked = self.masked_frame(pass, lighting);
        self.support
            .media
            .record_apply(list, scene, view, pass, lighting, masked)
    }

    /// Records the view's paths of `pass` into `view`'s radiance and light feedback after the
    /// scene's [`SceneLighting::record`], the view's raster half in the same or an earlier list
    /// and [`Self::record_media`] where that runs in the frame: its local light samplers, its
    /// layer guides, the world cache's update paths and the view dispatch; orders nothing after
    /// it. The layer guides cover the 8 x 8 tiles the G-buffer's layer tiles mark and the tiles
    /// beside them, where a camera ray up to the opaque depth finds each pixel's first dielectric
    /// or presence layer and leaves its guide for [`Self::record_guides`]. Every tile counts while
    /// a layer may lie before the near plane, while [`ViewLighting::camera_medium`] is set and when
    /// the G-buffer has no layer tiles. The scene's first real-time view of a frame then traces the
    /// frame's caustic photons ([`crate::CausticSettings`]) and resolves the scene's grid, which
    /// every view's paths of the frame gather from; the reference traces photons of its own each
    /// frame into the view's map ([`ReferenceSettings::photons`]). One path per pixel follows, in
    /// one dispatch of the path loop, from the first layer on those tiles and elsewhere from the
    /// G-buffer's surface. Outside the reference, segments between layers read the froxels of
    /// [`Self::record_media`] of the same frame where it ran, and those froxels hold the camera
    /// segment through an interior, which media apply adds; without them the path samples it. The
    /// scene's first view in a frame traces one update path per `update_block`^2 pixels, its
    /// later views one per four times as many. The reference
    /// traces none, never reads the cache, builds no local samplers and adds nothing to the
    /// scene's light feedback; its thin lens rays go through the same paths. With
    /// [`ViewPass::dof_background`] a dispatch of its own writes the view's background layer, made
    /// at the first such call; without it the layer is freed. [`ViewPass::overlay_front`] clears
    /// the view's front layer the same way, and pixels whose dominant layer lies behind their
    /// first layer take that first layer as their front layer. With [`ViewPass::frosted_glass`]
    /// the view keeps each pixel's background and rough-transmission blur for
    /// [`Self::record_frosted_glass`].
    pub fn record_paths(
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
        let mode = path_mode(lighting);
        let reference = lighting.reference.is_some();
        let first = !reference && scene.first_view(pass.frame);
        let block = scene.world_cache_settings().update_block * if first { 1 } else { 2 };
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let local_share = local_share(lighting);
        let local = local_share > 0.0;
        let written = view.feedback_index;
        view.feedback_index = 1 - written;
        view.feedback[written].tables = scene.light_tables();
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
                    frame: pass.sample_index,
                    fed_lights: view.feedback[1 - written].tables.lights,
                    fed_entries: view.feedback[1 - written].tables.entries,
                    _pad1: 0,
                },
            );
            let [w, h] = view.local_tiles;
            list.begin_pass("local light samplers", false);
            list.dispatch(&self.support.local, root, [w, h, 1]);
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            list.dispatch(&self.support.refine, root, [w, h, 1]);
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
            let target = OptionalTarget::new(
                &view.device,
                Format::Rgba16Float,
                view.extent,
                "dof background",
            )?;
            list.discard(&target.texture);
            view.dof = Some(DofLayer { target, frame: 0 });
        }
        if !pass.overlay_front {
            view.overlay_front = None;
        } else {
            if view.overlay_front.is_none() {
                let front = OverlayFront::new(&view.device, view.extent)?;
                list.discard(&front.depth.texture);
                list.discard(&front.motion.texture);
                list.discard(&front.weight.texture);
                view.overlay_front = Some(front);
            }
            let front = view.overlay_front.as_mut().expect("made above");
            front.frame = pass.frame;
            front.dispatch(
                list,
                &self.support.overlay_clear,
                "overlay front clear",
                view.extent,
                TextureIndex::NONE,
            );
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
        let mut flags = 0;
        let mut froxels = view.media.volume(pass.frame);
        if reference {
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
        let features = path_features(pass.scene_features, lighting);
        // Asked for together, the sources that run in the stages share their pipelines.
        let needs = PathNeeds {
            features: pass.scene_features,
            frosted_glass: pass.frosted_glass,
            dof_background: pass.dof_background.is_some(),
            particle_lighting: pass.particle_lighting,
        };
        let k = &self.paths;
        let predicted = k.predict(scene, needs, lighting);
        let pixels = view.extent[0] * view.extent[1];
        let parts = StateParts::of(features);
        // The queues' rounds: a view's passes or a photon's bounces, and one more that the
        // last round's stages append to.
        let depth = dielectric_depth(lighting);
        let max_bounces = lighting.max_bounces;
        let passes = if pass.frosted_glass { 2 } else { 1 };
        let rows = (rounds(max_bounces, depth, reference)? * passes).max(rounds(
            PHOTON_BOUNCES,
            depth,
            reference,
        )?) + 1;
        // The update paths of a view's blocks ride with its own, shared out over its slices.
        let update = if reference {
            0
        } else {
            let blocks = view.extent.map(|e| e.div_ceil(block));
            blocks[0] * blocks[1]
        };
        if !view
            .staged
            .as_ref()
            .is_some_and(|m| m.fits(pixels, parts, rows, update))
        {
            view.staged = None;
            view.staged = Some(StagedMemory::new(
                &view.device,
                pixels,
                parts,
                rows,
                update,
            )?);
        }
        let (wanted, stages) = k.wanted(scene, needs, lighting);
        k.ask(wanted, &stages, lighting)?;
        view.staged
            .as_ref()
            .expect("made above")
            .begin_frame(wanted);

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
            layers: view.layers.gpu(),
            froxels,
            dielectric_depth: dielectric_depth(lighting),
            camera_medium: lighting.camera_medium.map_or(NO_MATERIAL, |m| m.0),
            dielectric_flags: flags,
            sss_max_steps: if reference {
                REFERENCE_SSS_STEPS
            } else {
                lighting
                    .sss_max_steps
                    .clamp(*SSS_STEPS.start(), *SSS_STEPS.end())
            },
            subsurface_table: self.support.subsurface_table.gpu(),
            sss_flags: match lighting.sss_depth {
                SubsurfaceDepth::FirstEvent => 0,
                SubsurfaceDepth::PrimaryOnly => SSS_PRIMARY_ONLY,
            },
            pixel_fraction: lighting.pixel_fraction.clamp(MIN_PIXEL_FRACTION, 1.0),
            dof_background,
            dof_mask,
            dof_focus,
            dof_aperture,
            source: 0,
            overlay_front: view.overlay_front_words(pass.frame),
            caustics: GpuPtr::NULL,
            stages: GpuPtr::NULL,
            slice_base: 0,
            slice_count: 0,
            features,
            round: 0,
            round_first: 0,
            round_next: 0,
            _pad0: 0,
            _pad1: 0,
        };
        // What a source's paths may meet is what the view asked pipelines for, which its root
        // asks nothing beyond: a pipeline leaves out what its mask rules out whatever a root
        // says.
        let may = |root: &PathRoot, predicted: u64| {
            let asks = scene_may(pass.scene_features) | root_may(root);
            assert!(
                asks & !predicted == 0,
                "a path root asks for {asks:#x}, beyond the {predicted:#x} its pipelines hold"
            );
            predicted
        };
        let photons = predicted.photons;
        let path_root = PathRoot {
            caustics: self
                .record_photons(list, scene, view, pass, lighting, &path_root, photons)?,
            ..path_root
        };
        self.record_layers(list, view, pass, lighting, &path_root)?;
        let update_root = (!reference).then_some(PathRoot {
            mode: 0,
            local: GpuPtr::NULL,
            light_flags: 0,
            dof_background: StorageImageIndex::NONE,
            dielectric_flags: path_root.dielectric_flags & DIELECTRIC_DISPERSION,
            features: path_root.features & UPDATE_FEATURES,
            pixel_fraction: 1.0,
            ..path_root
        });
        let blocks = view.extent.map(|e| e.div_ceil(block));
        let pixels = view.extent[0] * view.extent[1];
        // The update paths ride in the view's stages, under the root the start stage
        // makes of the view's as `update_root` is made here.
        let mut meets = may(&path_root, predicted.view);
        let mut update = 0;
        if let Some(update_root) = &update_root {
            assert_eq!(
                path_rounds(update_root),
                path_rounds(&path_root),
                "update paths take the rounds of their view's"
            );
            meets |= may(update_root, predicted.update) | Source::Update.may();
            update = blocks[0] * blocks[1];
        }
        list.begin_pass(
            if update == 0 {
                "view paths"
            } else {
                "view and update paths"
            },
            false,
        );
        k.stages.record(
            list,
            &SourcePaths {
                memory: view.staged.as_ref().expect("made above"),
                root: &path_root,
                source: Source::View,
                may: meets,
                items: pixels,
                rounds: path_rounds(&path_root),
                passes: if pass.frosted_glass { 2 } else { 1 },
                update,
            },
            lighting.stage_passes,
        )?;
        list.end_pass();

        if pass.dof_background.is_some() {
            list.begin_pass("background paths", false);
            // The background layer's path is the full one at the view path's other settings.
            let background_root = PathRoot {
                mode: 0,
                pixel_fraction: 1.0,
                ..path_root
            };
            k.stages.record(
                list,
                &SourcePaths {
                    memory: view.staged.as_ref().expect("made above"),
                    root: &background_root,
                    source: Source::Background,
                    may: may(&background_root, predicted.background),
                    items: pixels,
                    rounds: path_rounds(&background_root),
                    passes: 1,
                    update: 0,
                },
                lighting.stage_passes,
            )?;

            list.end_pass();
        }
        Ok(())
    }

    /// Records the frame's caustic photons where `view` traces them, and returns the caustics
    /// its paths gather from: the scene's grid, whose photons and resolve the scene's first
    /// real-time view of the frame records, or under the reference the view's own map, filled
    /// each frame; null with caustics off.
    #[allow(clippy::too_many_arguments)]
    fn record_photons(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
        path: &PathRoot,
        may: u64,
    ) -> Result<GpuPtr<CausticsGpu>, Error> {
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let caustics = scene.caustics();
        let grid = caustics.root(slot);
        let (root, photons) = match lighting.reference {
            Some(r) if r.photons > 0 && grid != GpuPtr::NULL => {
                if r.photons > crate::MAX_PHOTONS {
                    return Err(Error::Settings(
                        "reference photons must be at most MAX_PHOTONS",
                    ));
                }
                if !view.photon_map.as_ref().is_some_and(|m| m.holds(r.photons)) {
                    view.photon_map = None;
                    view.photon_map = Some(PhotonMap::new(&view.device, r.photons)?);
                }
                let map = view.photon_map.as_mut().expect("made above");
                map.advance(view.history_reset, r.photon_radius);
                let offset = photon_offset(pass.sample_index, pass.seed);
                list.begin_pass("caustic photon map", false);
                list.barrier(
                    Stages::COMPUTE | Stages::RAY,
                    Stages::COMPUTE,
                    Hazards::empty(),
                );
                let root = caustics.record_map(list, map, slot, r.photons, offset);
                list.barrier(
                    Stages::COMPUTE,
                    Stages::COMPUTE | Stages::RAY,
                    Hazards::empty(),
                );
                list.end_pass();
                (root, r.photons)
            }
            Some(_) => {
                view.photon_map = None;
                view.staged_photons = None;
                return Ok(GpuPtr::NULL);
            }
            None => {
                view.photon_map = None;
                if grid == GpuPtr::NULL || !scene.first_photons(pass.frame) {
                    // No photons of this frame run in the view: it keeps no memory for them,
                    // and the sets its last ones ran are no longer among those its frame ran.
                    view.staged_photons = None;
                    return Ok(grid);
                }
                (grid, caustics.photons())
            }
        };
        let photon_root = PathRoot {
            mode: 0,
            local: GpuPtr::NULL,
            light_flags: 0,
            cache_flags: 0,
            roulette_depth: REFERENCE_ROULETTE_DEPTH,
            max_bounces: PHOTON_BOUNCES,
            dof_background: StorageImageIndex::NONE,
            dielectric_flags: path.dielectric_flags
                & (DIELECTRIC_DISPERSION | DIELECTRIC_REFERENCE),
            // Photons do not walk.
            features: path.features & !FEATURE_WALK,
            pixel_fraction: 1.0,
            caustics: root,
            ..*path
        };
        list.begin_pass("caustic photons", false);
        let parts = StateParts::of(photon_root.features);
        let own = view.staged.as_ref().expect("made by record_paths");
        // As many slots as the photons of a frame at its default count: more photons
        // than that run in slices that each hold as many.
        let slots = photons.min(PHOTON_SLOTS);
        let queues = path_rounds(&photon_root) + 1;
        if own.holds(slots, parts, queues) {
            view.staged_photons = None;
        } else if !view
            .staged_photons
            .as_ref()
            .is_some_and(|m| m.fits(slots, parts, queues, 0))
        {
            view.staged_photons = None;
            view.staged_photons = Some(StagedMemory::new(&view.device, slots, parts, queues, 0)?);
        }
        // The memory holds the sets this frame's photons run until its next one, under
        // what the view asked its pipelines for.
        if let Some(memory) = &view.staged_photons {
            memory.begin_frame(own.ran().0);
        }
        let k = &self.paths;
        let asks = scene_may(pass.scene_features) | root_may(&photon_root);
        assert!(
            asks & !may == 0,
            "the photons' root asks for {asks:#x}, beyond the {may:#x} its pipelines hold"
        );
        k.stages.record(
            list,
            &SourcePaths {
                memory: view.staged_photons.as_ref().unwrap_or(own),
                root: &photon_root,
                source: Source::Photon,
                may,
                items: photons,
                rounds: path_rounds(&photon_root),
                passes: 1,
                update: 0,
            },
            lighting.stage_passes,
        )?;

        if lighting.reference.is_none() {
            list.barrier(
                Stages::COMPUTE | Stages::RAY,
                Stages::COMPUTE,
                Hazards::empty(),
            );
            caustics.record_resolve(list, slot);
        }
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::COMPUTE | Stages::RAY,
            Hazards::empty(),
        );
        list.end_pass();
        Ok(root)
    }

    /// Records the tiles of `pass` that hold layers and the layer guides over them, before any
    /// path of the frame reads them.
    fn record_layers(
        &self,
        list: &mut CommandList<'_>,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
        path: &PathRoot,
    ) -> Result<(), Error> {
        let masked = self.masked_frame(pass, lighting);
        let may = masked.may;
        let asks = scene_may(pass.scene_features) | root_may(path);
        assert!(
            asks & !may == 0,
            "the layer guides' root asks for {asks:#x}, beyond the {may:#x} their pipeline holds"
        );
        // No tile is listed where the scene holds no layer and the view lists none itself.
        let guides = if masked.runs.layer_guides {
            Some(masked.take(Masked::LayerGuides)?)
        } else {
            None
        };
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let mut tile_flags = 0;
        if lighting.camera_medium.is_some() || traces_lens(lighting) {
            tile_flags |= LAYER_TILES_EVERYWHERE;
        }
        let tiles_root = view.layer_tile_roots.write(
            slot,
            LayerTilesRoot {
                gbuffer: pass.gbuffer,
                layers: view.layers.gpu(),
                flags: tile_flags,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
        );
        view.layers_frame = Some(pass.frame);
        view.layers_frost = pass.frosted_glass;
        let args = view
            .layers
            .gpu()
            .cast::<u8>()
            .offset(LayerMemory::new(view.extent).args);
        list.begin_pass("layer guides", false);
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::TRANSFER,
            Hazards::empty(),
        );
        list.copy(args, self.support.zero.gpu().cast(), 4);
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        let [tx, ty] = view.extent.map(|e| e.div_ceil(LAYER_TILE));
        let [gx, gy, _] = self.support.layer_tiles.workgroup();
        list.dispatch(
            &self.support.layer_tiles,
            tiles_root,
            [tx.div_ceil(gx), ty.div_ceil(gy), 1],
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.support.layer_args, tiles_root, [1, 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::RAY,
            Hazards::DRAW_ARGUMENTS,
        );
        if let Some(set) = &guides {
            let guides = set.get(Masked::LayerGuides).expect("taken above");
            list.dispatch_indirect_inline(guides, path, args.offset(4).cast());
        }
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::RAY,
            Hazards::empty(),
        );
        list.end_pass();
        Ok(())
    }

    /// Records the guide resolve of `pass` into `view`'s guide targets after the view's
    /// raster half, and after [`Self::record_paths`] where that runs in the frame, whose
    /// layer guides it reads. It writes every pixel of the view's
    /// [`LightingView::depth_of_field_guide`] where it holds one. Under a
    /// [`LightingDebugView::Guide`] view it also overwrites the radiance, so it must then be
    /// ordered after [`Self::record_paths`]. Orders nothing after it. Waits for the guide
    /// resolve's pipeline where no view asked for it ahead ([`Self::prepare_paths`]); an error
    /// where it failed to compile.
    pub fn record_guides(
        &self,
        list: &mut CommandList<'_>,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> Result<(), Error> {
        let masked = self.masked_frame(pass, lighting);
        let set = masked.take(Masked::Guides)?;
        let guides = set.get(Masked::Guides).expect("taken above");
        let debug = match lighting.debug_view {
            LightingDebugView::Guide(GuideKind::DiffuseAlbedo) => 1,
            LightingDebugView::Guide(GuideKind::SpecularAlbedo) => 2,
            LightingDebugView::Guide(GuideKind::Normal) => 3,
            LightingDebugView::Guide(GuideKind::Roughness) => 4,
            LightingDebugView::Guide(GuideKind::SpecularMotion) => 5,
            _ => 0,
        };
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let dof_guide = view.depth_of_field_guide_storage();
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
                dof_guide: dof_guide.unwrap_or(StorageImageIndex::NONE),
                motion_roughness: lighting.guides.specular_motion_roughness,
                debug,
                dielectric_depth: dielectric_depth(lighting),
                _pad0: 0,
            },
        );
        let [w, h] = view.extent;
        let [gx, gy, _] = guides.workgroup();
        list.begin_pass("guide resolve", false);
        list.dispatch(guides, root, [w.div_ceil(gx), h.div_ceil(gy), 1]);
        list.end_pass();
        Ok(())
    }

    /// Resolves `view`'s front layer into [`LightingView::overlay_depth`],
    /// [`LightingView::overlay_motion`] and [`LightingView::overlay_weight`]. `overlay` is the
    /// frame's premultiplied overlay, where it drew one. The caller orders it after the frame's
    /// view dispatch and transparency hook, whose fragments enter the layer through
    /// `particle_front`, while the overlay still holds its contents. Records nothing without
    /// [`ViewPass::overlay_front`]. Orders nothing after it.
    pub fn record_overlay_front(
        &self,
        list: &mut CommandList<'_>,
        view: &mut LightingView,
        overlay: Option<TextureIndex>,
    ) {
        let extent = view.extent;
        if let Some(front) = &mut view.overlay_front {
            front.dispatch(
                list,
                &self.support.overlay_resolve,
                "overlay front",
                extent,
                overlay.unwrap_or(TextureIndex::NONE),
            );
        }
    }

    /// Adds the frosted-glass term into `guide`, ray reconstruction's depth-of-field guide
    /// (R16F at the view's extent, `guide_luminance(after - before)`), after the guide resolve
    /// or output's depth-of-field guide pass wrote it and after [`Self::record_paths`] of
    /// `pass` with [`ViewPass::frosted_glass`]. A pixel whose dominant layer is rough
    /// transmission adds its transmission's luminance times the change that blurring the
    /// background over its rough interfaces' disc makes. The background is the radiance of the
    /// paths from the G-buffer's surfaces at those pixels, unbent by refraction, and each other
    /// pixel's own radiance. Other pixels keep what that write left. Records nothing without
    /// such a dispatch in `pass.frame`. Orders nothing after it.
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
        let [gx, gy, _] = self.support.frost.workgroup();
        list.begin_pass("frosted glass", false);
        list.dispatch(
            &self.support.frost,
            root,
            [w.div_ceil(gx), h.div_ceil(gy), 1],
        );
        list.end_pass();
    }
}
