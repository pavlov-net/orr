//! Per-view participating media: the froxel volume to the scene's `near_range`, the
//! aerial-perspective LUT past it, and media apply.

use orr_scene::orr_gpu::{
    self, AddressMode, Allocation, CommandList, ComputePipeline, DEFAULT_MEDIUM, Device, Filter,
    Format, GpuPtr, Hazards, MemoryKind, SamplerDesc, SamplerIndex, Stages, StorageImageIndex,
    Texture, TextureDesc, TextureIndex, TextureKind, TextureMemory, TextureUsage,
};
use orr_scene::{MaterialId, SceneFeatures};

use crate::scene::{FRAME_SLOTS, Ring};
use crate::shaders::media::{aerial, apply, froxel, smoke};
use crate::shaders::types::{
    EyeColumn, FroxelLights, FroxelRoot, FroxelVolume, LayerGuide, MediaApplyRoot, MediaStatsGpu,
    ParticleLighting,
};
use crate::view::{LightingDebugView, LightingView, NO_MATERIAL, ViewLighting, ViewPass};
use crate::{CompilePipeline, Error, KernelSet, SceneLighting, pipeline};

/// Pixels per side of a froxel column's tile (`media`'s `FROXEL_TILE`).
const FROXEL_TILE: u32 = 8;

/// Columns, rows and shells of the aerial-perspective LUT (`media`'s `AERIAL_SIZE`).
pub const AERIAL_SIZE: u32 = 32;

/// Most directional lights the froxels trace shadow rays toward (`media`'s
/// `MAX_FROXEL_SHADOWED_LIGHTS`).
pub const MAX_FROXEL_SHADOWED_LIGHTS: u32 = 4;

/// Most shadow rays per light a camera ray traces over a stretch of medium proxies
/// (`camera_media`'s `MAX_PROXY_RAYS`).
pub const MAX_PROXY_RAYS: u32 = 8;

/// `froxels`' `SMOKE_WORDS`.
const SMOKE_WORDS: u32 = 7;

/// `camera_media`'s `MEDIA_*`.
const MEDIA_COMPOSE: u32 = 0;
const MEDIA_INSCATTER: u32 = 1;
const MEDIA_FROXELS: u32 = 2;
const MEDIA_OFF: u32 = 3;

/// Frames of history the froxels count up to; past 16 the blend stays at 1/16.
const MAX_HISTORY: u32 = 64;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MediaSettings {
    /// Froxel slices from the eye to the scene's `near_range`, at least 1.
    pub slices: u32,
    /// Far end of the aerial-perspective LUT, in m from the eye; surfaces past the scene's
    /// `near_range` take their camera segment's medium from the LUT up to it and from a
    /// march past it. At most `near_range` turns the LUT off.
    pub ap_range: f32,
    /// Steps of the in-scatter march on path segments of 500 m or more, on camera segments
    /// past `ap_range`, and of the reference's camera segment. Also the most steps a camera
    /// ray takes over a stretch through medium proxies, which it otherwise marches at their
    /// voxel spacing.
    pub segment_steps: u32,
    /// The froxels' transmittance per channel instead of its luminance.
    pub chromatic_transmittance: bool,
    /// Directional lights, the brightest by selection weight, toward which each froxel
    /// traces a shadow ray, at most [`MAX_FROXEL_SHADOWED_LIGHTS`]; the others light the
    /// froxels unshadowed and isotropically.
    pub froxel_shadowed_lights: u32,
    /// Shadow rays per shadowed light a camera ray traces over each stretch through medium
    /// proxies, at most [`MAX_PROXY_RAYS`]; its steps take visibility between the two nearest,
    /// so a proxy smaller than a froxel keeps its self-shadowing. 0 takes the froxels'
    /// visibility inside them, and one ray per stretch past them.
    pub proxy_shadow_rays: u32,
    /// Henyey-Greenstein parameter of smoke particles' response to the froxels' shadowed
    /// directional lights.
    pub smoke_anisotropy: f32,
}

impl Default for MediaSettings {
    fn default() -> Self {
        Self {
            slices: 64,
            ap_range: 128_000.0,
            segment_steps: 16,
            chromatic_transmittance: false,
            froxel_shadowed_lights: 1,
            proxy_shadow_rays: 8,
            smoke_anisotropy: 0.0,
        }
    }
}

/// One frame's froxel counts, read back after it completes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MediaStats {
    pub frame: u64,
    /// Froxels in front of the surfaces of their tiles and the tiles around them, which the light
    /// pass lit: 0 in a frame where nothing reads the froxels' light.
    pub active_froxels: u32,
    /// Rays the froxels traced.
    pub froxel_rays: u32,
    /// Medium proxies in the view's frustum.
    pub visible_proxies: u32,
    /// Froxel columns whose near-plane point lies in a dielectric's interior.
    pub eye_columns: u32,
}

/// One 3D target with its storage and sampled views.
struct Volume {
    _texture: Texture,
    store: StorageImageIndex,
    sampled: TextureIndex,
}

/// What a view's froxel targets were made for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shape {
    grid: [u32; 3],
    /// Shadowed lights past the first, which take `visibility` channels.
    extra: u32,
    chromatic: bool,
    /// The scene holds a smoke entry.
    smoke: bool,
    /// A column of this view has lain in a dielectric's interior, which takes rgb transmittance
    /// and each shadowed light's distance through that interior.
    eye: bool,
}

/// A view's froxel targets: `froxel_material`, `froxel_incident` and `froxel_visibility`
/// with their histories, `froxel_integrated`; `eye_depth` with its history; and its `aerial`
/// LUT.
struct Volumes {
    device: Device,
    shape: Shape,
    slots: Vec<u32>,
    sigma: Volume,
    words: Volume,
    incident: [Volume; 2],
    visibility: Option<[Volume; 2]>,
    integrated: Volume,
    integrated_rgb: Option<Volume>,
    aerial: Volume,
    aerial_rgb: Option<Volume>,
    /// `froxels`' `SMOKE_WORDS` per froxel.
    smoke: Option<Allocation<u32>>,
    eye: Allocation<EyeColumn>,
    eye_depth: Option<[Volume; 2]>,
}

impl Volumes {
    fn new(device: &Device, list: &mut CommandList<'_>, shape: Shape) -> Result<Self, Error> {
        let mut slots = Vec::new();
        let mut sized = |extent, format, name: &str| -> Result<Volume, Error> {
            let desc = TextureDesc {
                kind: TextureKind::D3,
                format,
                extent,
                mips: 1,
                layers: 1,
                usage: TextureUsage::SAMPLED | TextureUsage::STORAGE,
            };
            let texture = device.create_texture(&desc, TextureMemory::Dedicated, name)?;
            list.discard(&texture);
            let view = orr_gpu::ViewDesc::default();
            let store = device.alloc_image_slot()?;
            let sampled = device.alloc_image_slot()?;
            slots.extend([store, sampled]);
            Ok(Volume {
                store: device.write_storage(store, &texture, &view),
                sampled: device.write_sampled(sampled, &texture, &view),
                _texture: texture,
            })
        };
        let lut = [AERIAL_SIZE; 3];
        let aerial = sized(lut, Format::Rgba16Float, "aerial perspective")?;
        let aerial_rgb = if shape.chromatic {
            Some(sized(lut, Format::Rgba16Float, "aerial perspective rgb")?)
        } else {
            None
        };
        let mut volume = |format, name: &str| sized(shape.grid, format, name);
        let visibility_format = match shape.extra {
            1 => Format::R8Unorm,
            2 => Format::Rg8Unorm,
            _ => Format::Rgba8Unorm,
        };
        let eye_depth_format = match shape.extra {
            0 => Format::R16Float,
            1 => Format::Rg16Float,
            _ => Format::Rgba16Float,
        };
        // 8 bits run out of transmittance within meters of water.
        let integrated_rgb_format = if shape.eye {
            Format::Rgba16Float
        } else {
            Format::Rgba8Unorm
        };
        Ok(Self {
            sigma: volume(Format::Rgba16Float, "froxel material")?,
            words: volume(Format::Rgba32Uint, "froxel material words")?,
            incident: [
                volume(Format::Rgba16Float, "froxel incident")?,
                volume(Format::Rgba16Float, "froxel incident")?,
            ],
            visibility: if shape.extra > 0 {
                Some([
                    volume(visibility_format, "froxel visibility")?,
                    volume(visibility_format, "froxel visibility")?,
                ])
            } else {
                None
            },
            integrated: volume(Format::Rgba16Float, "froxel integrated")?,
            integrated_rgb: if shape.chromatic || shape.eye {
                Some(volume(integrated_rgb_format, "froxel integrated rgb")?)
            } else {
                None
            },
            eye_depth: if shape.eye {
                Some([
                    volume(eye_depth_format, "froxel eye depth")?,
                    volume(eye_depth_format, "froxel eye depth")?,
                ])
            } else {
                None
            },
            aerial,
            aerial_rgb,
            smoke: if shape.smoke {
                let [columns, rows, slices] = shape.grid;
                let words = u64::from(SMOKE_WORDS) * u64::from(columns * rows * slices);
                Some(device.gpu_malloc(MemoryKind::DeviceLocal, words, "froxel smoke")?)
            } else {
                None
            },
            eye: device.gpu_malloc(
                MemoryKind::DeviceLocal,
                u64::from(shape.grid[0] * shape.grid[1]),
                "froxel eye columns",
            )?,
            device: device.clone(),
            shape,
            slots,
        })
    }
}

impl Drop for Volumes {
    fn drop(&mut self) {
        for &slot in &self.slots {
            self.device.free_image_slot_after_submitted(slot);
        }
    }
}

/// What reads a frame's froxels, from what the kernels, the scene and the view hold. Each
/// names a reader exactly or errs toward one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Readers {
    /// A camera segment can hold a medium: the linked one, a proxy, smoke or an interior. Then
    /// the froxels' medium and integration, the aerial-perspective LUT and media apply are
    /// read, and paths read the volume; otherwise they would hold no medium and leave every
    /// radiance as it is.
    fogged: bool,
    /// The froxels' light is read: by a medium that scatters, which the linked one, a proxy,
    /// smoke and an interior with scatter can be, or by particle lighting.
    lit: bool,
}

impl Readers {
    const NONE: Self = Self {
        fogged: false,
        lit: false,
    };

    fn any(self) -> bool {
        self.fogged || self.lit
    }
}

/// What [`MediaKernels::record`] recorded in a frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Recorded {
    frame: u64,
    readers: Readers,
}

/// A view's media state: its froxel targets, made at the first [`MediaKernels::record`] whose
/// froxels have a reader, remade when the grid or the settings that shape it change and freed
/// when no reader is left, and their light's history.
pub(crate) struct MediaView {
    volumes: Option<Volumes>,
    /// The `incident` and `visibility` this frame writes; the other holds the last frame's.
    current: usize,
    /// Frames the froxel history holds.
    history: u32,
    near_range: f32,
    camera_medium: Option<MaterialId>,
    /// A column of this view has lain in a dielectric's interior.
    eye: bool,
    lights: Allocation<FroxelLights>,
    stats: Allocation<MediaStatsGpu>,
    recorded: [Option<Recorded>; FRAME_SLOTS],
    roots: Ring<FroxelRoot>,
    apply_roots: Ring<MediaApplyRoot>,
    volume_roots: Ring<FroxelVolume>,
    particles: Ring<ParticleLighting>,
    particle_frames: [Option<u64>; FRAME_SLOTS],
}

impl MediaView {
    pub(crate) fn new(device: &Device) -> Result<Self, Error> {
        let mut lights =
            device.gpu_malloc::<FroxelLights>(MemoryKind::Upload, 1, "froxel shadowed lights")?;
        lights.as_mut_slice().unwrap()[0] = FroxelLights {
            ids: [u32::MAX; 4],
            count: 0,
            changed: 0,
            proxies: 0,
            far_smoke: 0,
            eye: 0,
        };
        Ok(Self {
            volumes: None,
            current: 0,
            history: 0,
            near_range: 0.0,
            camera_medium: None,
            eye: false,
            lights,
            stats: device.gpu_malloc(MemoryKind::Readback, FRAME_SLOTS as u64, "media stats")?,
            recorded: [None; FRAME_SLOTS],
            roots: Ring::new(device, "froxel roots")?,
            apply_roots: Ring::new(device, "media apply roots")?,
            volume_roots: Ring::new(device, "froxel volumes")?,
            particles: Ring::new(device, "particle lighting")?,
            particle_frames: [None; FRAME_SLOTS],
        })
    }

    /// What `frame` recorded, while its slot holds it.
    fn recorded(&self, frame: u64) -> Option<Recorded> {
        self.recorded[(frame % FRAME_SLOTS as u64) as usize].filter(|r| r.frame == frame)
    }

    /// Froxel counts of `frame`, once its submit has completed.
    pub(crate) fn stats(&self, frame: u64) -> Option<MediaStats> {
        let slot = (frame % FRAME_SLOTS as u64) as usize;
        self.recorded(frame).map(|_| {
            let s = self.stats.as_slice().unwrap()[slot];
            MediaStats {
                frame,
                active_froxels: s.active,
                froxel_rays: s.rays,
                visible_proxies: s.proxies,
                eye_columns: s.eye_columns,
            }
        })
    }

    /// The froxel volume `frame` recorded, for kernels after its passes; null when it
    /// recorded none, as when no camera segment can hold a medium: a path's segments then read
    /// the linked medium, which is none, and no path starts in an interior.
    pub(crate) fn volume(&self, frame: u64) -> GpuPtr<FroxelVolume> {
        let slot = (frame % FRAME_SLOTS as u64) as usize;
        if self.recorded(frame).is_some_and(|r| r.readers.fogged) {
            self.volume_roots.at(slot)
        } else {
            GpuPtr::NULL
        }
    }

    /// The particle lighting media apply of `frame` wrote; null when it recorded none.
    pub(crate) fn particle_lighting(&self, frame: u64) -> GpuPtr<ParticleLighting> {
        let slot = (frame % FRAME_SLOTS as u64) as usize;
        if self.particle_frames[slot] == Some(frame) {
            self.particles.at(slot)
        } else {
            GpuPtr::NULL
        }
    }
}

/// Whether `kernels` link a medium.
pub(crate) fn links_medium(kernels: &KernelSet) -> bool {
    kernels.medium != DEFAULT_MEDIUM
}

/// Whether a medium that scatters may lie on a camera segment of a scene that may hold
/// `features`, under kernels that link a medium where `linked`.
fn scatters(linked: bool, features: SceneFeatures, lighting: &ViewLighting) -> bool {
    linked
        || features.contains(SceneFeatures::MEDIUM_PROXIES)
        || features.contains(SceneFeatures::SMOKE)
        || lighting.camera_medium.is_some()
        || matches!(
            lighting.debug_view,
            LightingDebugView::Media | LightingDebugView::Froxels
        )
}

/// Whether a camera segment of a scene that may hold `features` can hold a medium, under
/// kernels that link one where `linked`: then a frame's froxels hold it, and its paths read
/// them.
pub(crate) fn fogged(linked: bool, features: SceneFeatures, lighting: &ViewLighting) -> bool {
    scatters(linked, features, lighting) || features.contains(SceneFeatures::INTERIORS)
}

/// The media kernels of one device.
pub(crate) struct MediaKernels {
    order: ComputePipeline,
    eye_column: ComputePipeline,
    eye_reach: ComputePipeline,
    inject: ComputePipeline,
    smoke: ComputePipeline,
    light: ComputePipeline,
    integrate: ComputePipeline,
    aerial: ComputePipeline,
    apply: ComputePipeline,
    sampler: SamplerIndex,
    /// The kernels link a medium.
    linked_medium: bool,
}

impl MediaKernels {
    /// Its pipelines, in the order [`Self::new`] takes them.
    pub(crate) const PIPELINES: [CompilePipeline; 9] = [
        |d, k| pipeline(d, k, &froxel::MODULE, "order"),
        |d, k| pipeline(d, k, &froxel::MODULE, "eye_column"),
        |d, k| pipeline(d, k, &froxel::MODULE, "eye_reach"),
        |d, k| pipeline(d, k, &froxel::MODULE, "inject"),
        |d, k| pipeline(d, k, &smoke::MODULE, "main"),
        |d, k| pipeline(d, k, &froxel::MODULE, "light"),
        |d, k| pipeline(d, k, &froxel::MODULE, "integrate"),
        |d, k| pipeline(d, k, &aerial::MODULE, "main"),
        |d, k| pipeline(d, k, &apply::MODULE, "main"),
    ];

    /// The kernels over the next of `pipelines`, which [`Self::PIPELINES`] compiled from
    /// `kernels`.
    pub(crate) fn new(
        device: &Device,
        kernels: &KernelSet,
        pipelines: &mut impl Iterator<Item = ComputePipeline>,
    ) -> Result<Self, Error> {
        let mut next = || pipelines.next().expect("a pipeline per entry");
        Ok(Self {
            order: next(),
            eye_column: next(),
            eye_reach: next(),
            inject: next(),
            smoke: next(),
            light: next(),
            integrate: next(),
            aerial: next(),
            apply: next(),
            sampler: device.sampler(&SamplerDesc {
                mag: Filter::Linear,
                min: Filter::Linear,
                mip: Filter::Nearest,
                address: [AddressMode::ClampToEdge; 3],
                ..SamplerDesc::default()
            })?,
            linked_medium: links_medium(kernels),
        })
    }

    fn scatters(&self, features: SceneFeatures, lighting: &ViewLighting) -> bool {
        scatters(self.linked_medium, features, lighting)
    }

    fn fogged(&self, features: SceneFeatures, lighting: &ViewLighting) -> bool {
        fogged(self.linked_medium, features, lighting)
    }

    /// What reads the froxels of `pass`. A debug view of them reads all of them, and the
    /// interior a view names counts as one that scatters.
    fn readers(&self, pass: &ViewPass, lighting: &ViewLighting) -> Readers {
        let scatters = self.scatters(pass.scene_features, lighting);
        Readers {
            fogged: self.fogged(pass.scene_features, lighting),
            lit: scatters
                || pass
                    .scene_features
                    .contains(SceneFeatures::SCATTERING_INTERIORS)
                || pass.particle_lighting,
        }
    }

    /// The froxel passes' root of `pass`, with no targets while `media` holds none: what the
    /// order pass alone reads.
    fn root(
        &self,
        media: &MediaView,
        scene: &SceneLighting,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> FroxelRoot {
        let settings = &lighting.media;
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let v = media.volumes.as_ref();
        let [columns, rows, slices] = v.map_or([0; 3], |v| v.shape.grid);
        let (cur, prev) = (media.current, 1 - media.current);
        let pair = |t: Option<&Volume>| {
            t.map_or((StorageImageIndex::NONE, TextureIndex::NONE), |t| {
                (t.store, t.sampled)
            })
        };
        // This frame's storage view and the last frame's sampled view.
        let history = |p: Option<&[Volume; 2]>| {
            p.map_or((StorageImageIndex::NONE, TextureIndex::NONE), |p| {
                (p[cur].store, p[prev].sampled)
            })
        };
        let (sigma, _) = pair(v.map(|v| &v.sigma));
        let (words, _) = pair(v.map(|v| &v.words));
        let (incident, incident_history) = history(v.map(|v| &v.incident));
        let (visibility, visibility_history) = history(v.and_then(|v| v.visibility.as_ref()));
        let (integrated, integrated_sampled) = pair(v.map(|v| &v.integrated));
        let (integrated_rgb, integrated_rgb_sampled) =
            pair(v.and_then(|v| v.integrated_rgb.as_ref()));
        let (aerial, aerial_sampled) = pair(v.map(|v| &v.aerial));
        let (aerial_rgb, aerial_rgb_sampled) = pair(v.and_then(|v| v.aerial_rgb.as_ref()));
        let (eye_depth, eye_depth_history) = history(v.and_then(|v| v.eye_depth.as_ref()));
        FroxelRoot {
            scene: pass.scene,
            view: pass.view,
            gbuffer: pass.gbuffer,
            set: scene.light_set(),
            cache: scene.world_cache(pass.frame),
            lights: media.lights.gpu(),
            stats: media.stats.gpu().offset(slot as u64),
            raster: pass.raster,
            smoke: v
                .and_then(|v| v.smoke.as_ref())
                .map_or(GpuPtr::NULL, Allocation::gpu),
            eye: v.map_or(GpuPtr::NULL, |v| v.eye.gpu()),
            sigma,
            words,
            incident,
            incident_history,
            visibility,
            visibility_history,
            integrated,
            integrated_rgb,
            integrated_sampled,
            integrated_rgb_sampled,
            sampler: self.sampler,
            columns,
            rows,
            slices,
            shadowed: settings
                .froxel_shadowed_lights
                .min(MAX_FROXEL_SHADOWED_LIGHTS),
            frame: pass.sample_index,
            seed: pass.seed,
            history: media.history,
            chromatic: u32::from(v.is_some_and(|v| v.shape.chromatic)),
            aerial,
            aerial_rgb,
            aerial_sampled,
            aerial_rgb_sampled,
            ap_range: settings.ap_range,
            smoke_anisotropy: settings.smoke_anisotropy,
            camera_medium: lighting.camera_medium.map_or(NO_MATERIAL, |m| m.0),
            eye_depth,
            eye_depth_history,
        }
    }

    /// What transparency hooks light and fog particles with after media apply of `pass`: the
    /// frame's froxels where its light pass ran (`readers.lit`), and their fog and the
    /// aerial-perspective LUT where a camera segment can hold a medium (`readers.fogged`).
    /// Without those the fog is none.
    fn particle_lighting(
        &self,
        media: &MediaView,
        readers: Readers,
        scene: &SceneLighting,
        pass: &ViewPass,
        lighting: &ViewLighting,
        guides: (GpuPtr<LayerGuide>, GpuPtr<u64>),
    ) -> ParticleLighting {
        let (layers, overlay_front) = guides;
        let held = media.volumes.as_ref().filter(|_| readers.any());
        let v = held.filter(|_| readers.lit);
        let fog = held.filter(|_| readers.fogged);
        let [columns, rows, slices] = held.map_or([0; 3], |v| v.shape.grid);
        let current = |p: &[Volume; 2]| p[media.current].sampled;
        ParticleLighting {
            scene: pass.scene,
            view: pass.view,
            set: scene.light_set(),
            cache: scene.world_cache(pass.frame),
            lights: media.lights.gpu(),
            eye: held.map_or(GpuPtr::NULL, |v| v.eye.gpu()),
            layers,
            overlay_front,
            incident: v.map_or(TextureIndex::NONE, |v| current(&v.incident)),
            visibility: v
                .and_then(|v| v.visibility.as_ref())
                .map_or(TextureIndex::NONE, current),
            integrated: fog.map_or(TextureIndex::NONE, |v| v.integrated.sampled),
            integrated_rgb: fog
                .and_then(|v| v.integrated_rgb.as_ref())
                .map_or(TextureIndex::NONE, |t| t.sampled),
            sampler: self.sampler,
            columns,
            rows,
            slices,
            frame: pass.sample_index,
            seed: pass.seed,
            aerial: fog.map_or(TextureIndex::NONE, |v| v.aerial.sampled),
            aerial_rgb: fog
                .and_then(|v| v.aerial_rgb.as_ref())
                .map_or(TextureIndex::NONE, |t| t.sampled),
            ap_range: lighting.media.ap_range,
            eye_depth: v
                .and_then(|v| v.eye_depth.as_ref())
                .map_or(TextureIndex::NONE, current),
        }
    }

    /// Records the froxel passes and the aerial-perspective LUT of `pass` after the scene's
    /// [`SceneLighting::record`] and the view's raster half; their outputs are visible to
    /// later compute and ray stages. Each pass runs where something reads what it writes
    /// ([`Readers`]); the order pass always does, since the shadowed lights it picks depend on
    /// the frames before. The light's history starts over after a frame without its pass.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> Result<(), Error> {
        let extent = view.extent();
        let reset = view.history_reset();
        let media = &mut view.media;
        let settings = &lighting.media;
        let readers = self.readers(pass, lighting);
        let shadowed = settings
            .froxel_shadowed_lights
            .min(MAX_FROXEL_SHADOWED_LIGHTS);
        // This slot's last frame has completed; its stats say whether a near plane reached an
        // interior the view does not name.
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        media.eye = media.eye
            || lighting.camera_medium.is_some()
            || (media.recorded[slot].is_some()
                && media.stats.as_slice().unwrap()[slot].eye_columns > 0);
        let near_range = scene.settings().near_range;
        let mut grid = [0; 3];
        if readers.any() {
            let shape = Shape {
                grid: [
                    extent[0].div_ceil(FROXEL_TILE),
                    extent[1].div_ceil(FROXEL_TILE),
                    settings.slices.max(1),
                ],
                extra: shadowed.saturating_sub(1),
                chromatic: settings.chromatic_transmittance,
                smoke: pass.smoke_args.is_some(),
                eye: media.eye,
            };
            let mut fresh = false;
            if media.volumes.as_ref().is_none_or(|v| v.shape != shape) {
                let device = scene.device();
                media.volumes = Some(Volumes::new(device, list, shape)?);
                fresh = true;
            }
            if fresh
                || reset
                || media.near_range != near_range
                || media.camera_medium != lighting.camera_medium
            {
                media.history = 0;
            }
            media.current = 1 - media.current;
            grid = shape.grid;
        } else {
            media.volumes = None;
        }
        media.near_range = near_range;
        media.camera_medium = lighting.camera_medium;
        media.recorded[slot] = Some(Recorded {
            frame: pass.frame,
            readers,
        });
        let root = self.root(media, scene, pass, lighting);
        if readers.fogged {
            media.volume_roots.write(
                slot,
                FroxelVolume {
                    lights: root.lights,
                    eye: root.eye,
                    sigma: root.sigma,
                    words: root.words,
                    incident: root.incident,
                    visibility: root.visibility,
                    columns: root.columns,
                    rows: root.rows,
                    slices: root.slices,
                    chromatic: root.chromatic,
                },
            );
        }
        let root = media.roots.write(slot, root);
        let [columns, rows, slices] = grid;
        let step = |list: &mut CommandList<'_>| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        };
        list.begin_pass("froxels", false);
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        list.dispatch(&self.order, root, [1, 1, 1]);
        step(list);
        if readers.any() {
            // With no reader at all no column can lie in an interior, and nothing reads the
            // columns.
            let column_groups = [columns.div_ceil(8), rows.div_ceil(8), 1];
            list.dispatch(&self.eye_column, root, column_groups);
            step(list);
            list.dispatch(&self.eye_reach, root, column_groups);
            step(list);
        }
        if readers.fogged {
            list.dispatch(
                &self.inject,
                root,
                [columns.div_ceil(4), rows.div_ceil(4), slices.div_ceil(4)],
            );
            step(list);
            if let Some(args) = pass.smoke_args {
                list.dispatch_indirect(&self.smoke, root, args);
                step(list);
            }
        }
        if readers.lit {
            let [light_group, ..] = self.light.workgroup();
            list.dispatch(&self.light, root, [columns.div_ceil(light_group), rows, 1]);
            step(list);
            media.history = (media.history + 1).min(MAX_HISTORY);
        } else {
            media.history = 0;
        }
        if readers.fogged {
            list.dispatch(
                &self.integrate,
                root,
                [columns.div_ceil(8), rows.div_ceil(8), 1],
            );
        }
        list.end_pass();
        list.begin_pass("aerial perspective", false);
        if readers.fogged && lighting.media.ap_range > near_range {
            let groups = AERIAL_SIZE.div_ceil(8);
            list.dispatch(&self.aerial, root, [groups, groups, 1]);
        }
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::RAY | Stages::HOST,
            Hazards::empty(),
        );
        list.end_pass();
        Ok(())
    }

    /// Records media apply of `pass` into the view's radiance after the froxel passes and
    /// the paths that wrote it, where a camera segment can hold a medium and the debug view
    /// shows media apply; orders nothing after it.
    pub(crate) fn record_apply(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) {
        let extent = view.extent();
        let radiance = view.radiance_storage();
        let layers = view.layer_guides(pass.frame);
        let dof_background = view.dof_background_store(pass.frame);
        let overlay_front = view.overlay_front_words(pass.frame);
        let media = &mut view.media;
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let readers = media
            .recorded(pass.frame)
            .map_or(Readers::NONE, |r| r.readers);
        media.particle_frames[slot] = pass.particle_lighting.then_some(pass.frame);
        let guides = (layers, overlay_front);
        let particles = self.particle_lighting(media, readers, scene, pass, lighting, guides);
        let particles = media.particles.write(slot, particles);
        let mode = match lighting.debug_view {
            LightingDebugView::Media => MEDIA_INSCATTER,
            LightingDebugView::Froxels => MEDIA_FROXELS,
            LightingDebugView::RaysPerPixel
            | LightingDebugView::RayOriginBias
            | LightingDebugView::WorldCache
            | LightingDebugView::WorldCacheLod
            | LightingDebugView::Guide(_)
            | LightingDebugView::DielectricLayer
            | LightingDebugView::Subsurface => MEDIA_OFF,
            _ => MEDIA_COMPOSE,
        };
        if !readers.fogged || mode == MEDIA_OFF {
            return;
        }
        let root = media.apply_roots.write(
            slot,
            MediaApplyRoot {
                froxels: media.roots.at(slot),
                lighting: particles,
                layers,
                radiance,
                dof_background,
                mode,
                reference: u32::from(lighting.reference.is_some()),
                segment_steps: lighting.media.segment_steps,
                proxy_rays: lighting.media.proxy_shadow_rays.min(MAX_PROXY_RAYS),
            },
        );
        let [w, h] = extent;
        list.begin_pass("media apply", false);
        list.dispatch(&self.apply, root, [w.div_ceil(8), h.div_ceil(8), 1]);
        list.end_pass();
    }
}
