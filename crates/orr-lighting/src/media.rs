//! Per-view participating media: the froxel volume to the scene's `near_range` and media
//! apply.

use orr_scene::orr_gpu::{
    self, AddressMode, Allocation, CommandList, ComputePipeline, Device, Filter, Format, GpuPtr,
    Hazards, MemoryKind, SamplerDesc, SamplerIndex, Stages, StorageImageIndex, Texture,
    TextureDesc, TextureIndex, TextureKind, TextureMemory, TextureUsage,
};

use crate::scene::{FRAME_SLOTS, Ring};
use crate::shaders::media::{apply, froxel};
use crate::shaders::types::{FroxelLights, FroxelRoot, FroxelVolume, MediaStatsGpu};
use crate::view::{LightingDebugView, LightingView, ViewLighting, ViewPass};
use crate::{Error, KernelSet, SceneLighting, pipeline};

/// Pixels per side of a froxel column's tile (`media`'s `FROXEL_TILE`).
pub const FROXEL_TILE: u32 = 8;

/// Most directional lights the froxels trace shadow rays toward (`media`'s
/// `MAX_FROXEL_SHADOWED_LIGHTS`).
pub const MAX_FROXEL_SHADOWED_LIGHTS: u32 = 4;

/// `froxels`' `MEDIA_*`.
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
    /// Steps of the in-scatter march on path segments of 500 m or more, and of the
    /// reference's camera segment.
    pub segment_steps: u32,
    /// The froxels' transmittance per channel instead of its luminance.
    pub chromatic_transmittance: bool,
    /// Directional lights, the brightest by selection weight, toward which each froxel
    /// traces a shadow ray, at most [`MAX_FROXEL_SHADOWED_LIGHTS`]; the others light the
    /// froxels unshadowed and isotropically.
    pub froxel_shadowed_lights: u32,
}

impl Default for MediaSettings {
    fn default() -> Self {
        Self {
            slices: 64,
            segment_steps: 16,
            chromatic_transmittance: false,
            froxel_shadowed_lights: 1,
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
}

/// A view's froxel targets: `froxel_material`, `froxel_incident` and `froxel_visibility`
/// with their histories, `froxel_integrated`.
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
}

impl Volumes {
    fn new(device: &Device, list: &mut CommandList<'_>, shape: Shape) -> Result<Self, Error> {
        let mut slots = Vec::new();
        let [w, h, d] = shape.grid;
        let mut volume = |format, name: &str| -> Result<Volume, Error> {
            let desc = TextureDesc {
                kind: TextureKind::D3,
                format,
                extent: [w, h, d],
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
    apply_roots: Ring<FroxelRoot>,
    volume_roots: Ring<FroxelVolume>,
}

impl MediaView {
    pub(crate) fn new(device: &Device) -> Result<Self, Error> {
        let mut lights =
            device.gpu_malloc::<FroxelLights>(MemoryKind::Upload, 1, "froxel shadowed lights")?;
        lights.as_mut_slice().unwrap()[0] = FroxelLights {
            ids: [u32::MAX; 4],
            count: 0,
            changed: 0,
            _pad0: 0,
            _pad1: 0,
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

    /// RGBA16F, this frame's isotropic incident radiance per froxel (unexposed, nits) and the
    /// visibility of the first shadowed light in A.
    pub(crate) fn incident(&self) -> Option<TextureIndex> {
        self.volumes
            .as_ref()
            .map(|v| v.incident[self.current].sampled)
    }

    /// RGBA16F, per froxel the pre-exposed in-scatter from the eye to its far boundary and
    /// the luminance of the transmittance over it.
    pub(crate) fn integrated(&self) -> Option<TextureIndex> {
        self.volumes.as_ref().map(|v| v.integrated.sampled)
    }
}

/// The media kernels of one device.
pub(crate) struct MediaKernels {
    order: ComputePipeline,
    inject: ComputePipeline,
    light: ComputePipeline,
    integrate: ComputePipeline,
    apply: ComputePipeline,
    sampler: SamplerIndex,
}

impl MediaKernels {
    pub(crate) fn new(device: &Device, kernels: &KernelSet) -> Result<Self, Error> {
        Ok(Self {
            order: pipeline(device, kernels, &froxel::MODULE, "order")?,
            inject: pipeline(device, kernels, &froxel::MODULE, "inject")?,
            light: pipeline(device, kernels, &froxel::MODULE, "light")?,
            integrate: pipeline(device, kernels, &froxel::MODULE, "integrate")?,
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
        radiance: StorageImageIndex,
        mode: u32,
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
        let (integrated_rgb, integrated_rgb_sampled) = v
            .integrated_rgb
            .as_ref()
            .map_or((StorageImageIndex::NONE, TextureIndex::NONE), |t| {
                (t.store, t.sampled)
            });
        FroxelRoot {
            scene: pass.scene,
            view: pass.view,
            gbuffer: pass.gbuffer,
            set: scene.light_set(),
            cache: scene.world_cache(pass.frame),
            lights: media.lights.gpu(),
            stats: media.stats.gpu().offset(slot as u64),
            layers: GpuPtr::NULL,
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
            radiance,
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
            mode,
            reference: u32::from(lighting.reference.is_some()),
            segment_steps: settings.segment_steps,
            _pad0: 0,
        }
    }

    /// Records the froxel passes of `pass` after the scene's [`SceneLighting::record`] and
    /// the view's raster half; their outputs are visible to later compute and ray stages.
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
        let root = self.root(
            media,
            scene,
            pass,
            lighting,
            StorageImageIndex::NONE,
            MEDIA_COMPOSE,
        );
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
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.order, root, [1, 1, 1]);
        list.dispatch(
            &self.inject,
            root,
            [columns.div_ceil(4), rows.div_ceil(4), slices.div_ceil(4)],
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.light, root, [columns, rows, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(
            &self.integrate,
            root,
            [columns.div_ceil(8), rows.div_ceil(8), 1],
        );
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
        let media = &mut view.media;
        if media.volumes.is_none() {
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
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let root = FroxelRoot {
            layers,
            ..self.root(media, scene, pass, lighting, radiance, mode)
        };
        let root = media.apply_roots.write(slot, root);
        let [w, h] = extent;
        list.begin_pass("media apply", false);
        list.dispatch(&self.apply, root, [w.div_ceil(8), h.div_ceil(8), 1]);
        list.end_pass();
    }
}
