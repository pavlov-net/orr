//! Per-view participating media: the froxel volume to the scene's `near_range`, the
//! aerial-perspective LUT past it, and media apply.

use orr_scene::orr_gpu::{
    self, AddressMode, Allocation, CommandList, ComputePipeline, Device, Filter, Format, GpuPtr,
    Hazards, MemoryKind, SamplerDesc, SamplerIndex, Stages, StorageImageIndex, Texture,
    TextureDesc, TextureIndex, TextureKind, TextureMemory, TextureUsage,
};

use crate::scene::{FRAME_SLOTS, Ring};
use crate::shaders::media::{aerial, apply, froxel, smoke};
use crate::shaders::types::{
    FroxelLights, FroxelRoot, FroxelVolume, LayerGuide, MediaApplyRoot, MediaStatsGpu,
    ParticleLighting,
};
use crate::view::{LightingDebugView, LightingView, ViewLighting, ViewPass};
use crate::{Error, KernelSet, SceneLighting, pipeline};

/// Pixels per side of a froxel column's tile (`media`'s `FROXEL_TILE`).
pub const FROXEL_TILE: u32 = 8;

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
    /// Froxels in front of the surfaces of their tiles and the tiles around them.
    pub active_froxels: u32,
    /// Rays the froxels traced.
    pub froxel_rays: u32,
    /// Medium proxies in the view's frustum.
    pub visible_proxies: u32,
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
}

/// A view's froxel targets: `froxel_material`, `froxel_incident` and `froxel_visibility`
/// with their histories, `froxel_integrated`; and its `aerial` LUT.
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
            integrated_rgb: if shape.chromatic {
                Some(volume(Format::Rgba8Unorm, "froxel integrated rgb")?)
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

/// A view's media state: its froxel targets, made at the first [`MediaKernels::record`]
/// and remade when the grid or the settings that shape it change, and their history.
pub(crate) struct MediaView {
    volumes: Option<Volumes>,
    /// The `incident` and `visibility` this frame writes; the other holds the last frame's.
    current: usize,
    /// Frames the froxel history holds.
    history: u32,
    near_range: f32,
    lights: Allocation<FroxelLights>,
    stats: Allocation<MediaStatsGpu>,
    frames: [Option<u64>; FRAME_SLOTS],
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
            _pad0: 0,
        };
        Ok(Self {
            volumes: None,
            current: 0,
            history: 0,
            near_range: 0.0,
            lights,
            stats: device.gpu_malloc(MemoryKind::Readback, FRAME_SLOTS as u64, "media stats")?,
            frames: [None; FRAME_SLOTS],
            roots: Ring::new(device, "froxel roots")?,
            apply_roots: Ring::new(device, "media apply roots")?,
            volume_roots: Ring::new(device, "froxel volumes")?,
            particles: Ring::new(device, "particle lighting")?,
            particle_frames: [None; FRAME_SLOTS],
        })
    }

    /// Froxel counts of `frame`, once its submit has completed.
    pub(crate) fn stats(&self, frame: u64) -> Option<MediaStats> {
        let slot = (frame % FRAME_SLOTS as u64) as usize;
        (self.frames[slot] == Some(frame)).then(|| {
            let s = self.stats.as_slice().unwrap()[slot];
            MediaStats {
                frame,
                active_froxels: s.active,
                froxel_rays: s.rays,
                visible_proxies: s.proxies,
            }
        })
    }

    /// The froxel volume `frame` recorded, for kernels after its passes; null when it
    /// recorded none.
    pub(crate) fn volume(&self, frame: u64) -> GpuPtr<FroxelVolume> {
        let slot = (frame % FRAME_SLOTS as u64) as usize;
        if self.frames[slot] == Some(frame) {
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

/// The media kernels of one device.
pub(crate) struct MediaKernels {
    order: ComputePipeline,
    inject: ComputePipeline,
    smoke: ComputePipeline,
    light: ComputePipeline,
    integrate: ComputePipeline,
    aerial: ComputePipeline,
    apply: ComputePipeline,
    sampler: SamplerIndex,
}

impl MediaKernels {
    pub(crate) fn new(device: &Device, kernels: &KernelSet) -> Result<Self, Error> {
        Ok(Self {
            order: pipeline(device, kernels, &froxel::MODULE, "order")?,
            inject: pipeline(device, kernels, &froxel::MODULE, "inject")?,
            smoke: pipeline(device, kernels, &smoke::MODULE, "main")?,
            light: pipeline(device, kernels, &froxel::MODULE, "light")?,
            integrate: pipeline(device, kernels, &froxel::MODULE, "integrate")?,
            aerial: pipeline(device, kernels, &aerial::MODULE, "main")?,
            apply: pipeline(device, kernels, &apply::MODULE, "main")?,
            sampler: device.sampler(&SamplerDesc {
                mag: Filter::Linear,
                min: Filter::Linear,
                mip: Filter::Nearest,
                address: [AddressMode::ClampToEdge; 3],
                ..SamplerDesc::default()
            })?,
        })
    }

    fn root(
        &self,
        media: &MediaView,
        scene: &SceneLighting,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) -> FroxelRoot {
        let settings = &lighting.media;
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let v = media.volumes.as_ref().expect("froxel volumes");
        let [columns, rows, slices] = v.shape.grid;
        let (cur, prev) = (media.current, 1 - media.current);
        let (visibility, visibility_history) = v
            .visibility
            .as_ref()
            .map_or((StorageImageIndex::NONE, TextureIndex::NONE), |p| {
                (p[cur].store, p[prev].sampled)
            });
        let pair = |t: Option<&Volume>| {
            t.map_or((StorageImageIndex::NONE, TextureIndex::NONE), |t| {
                (t.store, t.sampled)
            })
        };
        let (integrated_rgb, integrated_rgb_sampled) = pair(v.integrated_rgb.as_ref());
        let (aerial_rgb, aerial_rgb_sampled) = pair(v.aerial_rgb.as_ref());
        FroxelRoot {
            scene: pass.scene,
            view: pass.view,
            gbuffer: pass.gbuffer,
            set: scene.light_set(),
            cache: scene.world_cache(pass.frame),
            lights: media.lights.gpu(),
            stats: media.stats.gpu().offset(slot as u64),
            raster: pass.raster,
            smoke: v.smoke.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
            sigma: v.sigma.store,
            words: v.words.store,
            incident: v.incident[cur].store,
            incident_history: v.incident[prev].sampled,
            visibility,
            visibility_history,
            integrated: v.integrated.store,
            integrated_rgb,
            integrated_sampled: v.integrated.sampled,
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
            chromatic: u32::from(v.shape.chromatic),
            aerial: v.aerial.store,
            aerial_rgb,
            aerial_sampled: v.aerial.sampled,
            aerial_rgb_sampled,
            ap_range: settings.ap_range,
            smoke_anisotropy: settings.smoke_anisotropy,
            _pad0: 0,
        }
    }

    /// What transparency hooks light and fog particles with after media apply of `pass`:
    /// this frame's froxels and aerial-perspective LUT where `media` holds them.
    fn particle_lighting(
        &self,
        media: &MediaView,
        scene: &SceneLighting,
        pass: &ViewPass,
        lighting: &ViewLighting,
        layers: GpuPtr<LayerGuide>,
    ) -> ParticleLighting {
        let v = media.volumes.as_ref();
        let [columns, rows, slices] = v.map_or([0; 3], |v| v.shape.grid);
        let current = |p: &[Volume; 2]| p[media.current].sampled;
        ParticleLighting {
            scene: pass.scene,
            view: pass.view,
            set: scene.light_set(),
            cache: scene.world_cache(pass.frame),
            lights: media.lights.gpu(),
            layers,
            incident: v.map_or(TextureIndex::NONE, |v| current(&v.incident)),
            visibility: v
                .and_then(|v| v.visibility.as_ref())
                .map_or(TextureIndex::NONE, current),
            integrated: v.map_or(TextureIndex::NONE, |v| v.integrated.sampled),
            integrated_rgb: v
                .and_then(|v| v.integrated_rgb.as_ref())
                .map_or(TextureIndex::NONE, |t| t.sampled),
            sampler: self.sampler,
            columns,
            rows,
            slices,
            frame: pass.sample_index,
            seed: pass.seed,
            aerial: v.map_or(TextureIndex::NONE, |v| v.aerial.sampled),
            aerial_rgb: v
                .and_then(|v| v.aerial_rgb.as_ref())
                .map_or(TextureIndex::NONE, |t| t.sampled),
            ap_range: lighting.media.ap_range,
            _pad0: 0,
        }
    }

    /// Records the froxel passes and the aerial-perspective LUT of `pass` after the scene's
    /// [`SceneLighting::record`] and the view's raster half; their outputs are visible to
    /// later compute and ray stages.
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
        let shadowed = settings
            .froxel_shadowed_lights
            .min(MAX_FROXEL_SHADOWED_LIGHTS);
        let shape = Shape {
            grid: [
                extent[0].div_ceil(FROXEL_TILE),
                extent[1].div_ceil(FROXEL_TILE),
                settings.slices.max(1),
            ],
            extra: shadowed.saturating_sub(1),
            chromatic: settings.chromatic_transmittance,
            smoke: pass.smoke_args.is_some(),
        };
        let mut fresh = false;
        if media.volumes.as_ref().is_none_or(|v| v.shape != shape) {
            let device = scene.device();
            media.volumes = Some(Volumes::new(device, list, shape)?);
            fresh = true;
        }
        let near_range = scene.settings().near_range;
        if fresh || reset || media.near_range != near_range {
            media.history = 0;
        }
        media.near_range = near_range;
        media.current = 1 - media.current;
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        media.frames[slot] = Some(pass.frame);
        let root = self.root(media, scene, pass, lighting);
        media.volume_roots.write(
            slot,
            FroxelVolume {
                lights: root.lights,
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
        let root = media.roots.write(slot, root);
        let [columns, rows, slices] = shape.grid;
        list.begin_pass("froxels", false);
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        list.dispatch(&self.order, root, [1, 1, 1]);
        list.dispatch(
            &self.inject,
            root,
            [columns.div_ceil(4), rows.div_ceil(4), slices.div_ceil(4)],
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        if let Some(args) = pass.smoke_args {
            list.dispatch_indirect(&self.smoke, root, args);
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        }
        list.dispatch(&self.light, root, [columns, rows, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(
            &self.integrate,
            root,
            [columns.div_ceil(8), rows.div_ceil(8), 1],
        );
        list.end_pass();
        list.begin_pass("aerial perspective", false);
        if lighting.media.ap_range > near_range {
            let groups = AERIAL_SIZE.div_ceil(8);
            list.dispatch(&self.aerial, root, [groups, groups, 1]);
        }
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::RAY | Stages::HOST,
            Hazards::empty(),
        );
        list.end_pass();
        media.history = (media.history + 1).min(MAX_HISTORY);
        Ok(())
    }

    /// Records media apply of `pass` into the view's radiance after the froxel passes and
    /// the paths that wrote it; orders nothing after it.
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
        let media = &mut view.media;
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        media.particle_frames[slot] = Some(pass.frame);
        let particles = self.particle_lighting(media, scene, pass, lighting, layers);
        let particles = media.particles.write(slot, particles);
        if media.frames[slot] != Some(pass.frame) {
            return;
        }
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
