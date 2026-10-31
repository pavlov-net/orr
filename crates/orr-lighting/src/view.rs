//! Per-view lighting: the settings of one `render_view` call, the view's lighting targets,
//! and the opaque path dispatch.

use orr_scene::orr_gpu::{
    CommandList, ComputePipeline, Device, Format, GpuPtr, RayDispatchPipeline, ShaderStage,
    SpecData, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage,
    ViewDesc,
};
use orr_scene::{GBufferGpu, SceneRoot, ViewConstants};

use crate::scene::{FRAME_SLOTS, Ring};
use crate::shaders::path::opaque;
use crate::shaders::types::PathRoot;
use crate::{BsdfTables, Error, KernelSet, SceneLighting, module};

/// Lighting settings of one view, passed with every render of it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ViewLighting {
    /// Light candidates resampled at a path's first vertex.
    pub primary_di_candidates: u32,
    /// Light candidates resampled at its later vertices.
    pub bounce_di_candidates: u32,
    /// BSDF bounces after the first vertex's; the last bounce still adds the emission and
    /// sky its ray reaches.
    pub max_bounces: u32,
    pub media: MediaSettings,
    pub debug_view: LightingDebugView,
    /// Accumulation reference instead of the real-time path.
    pub reference: Option<ReferenceSettings>,
}

impl Default for ViewLighting {
    fn default() -> Self {
        Self {
            primary_di_candidates: 8,
            bounce_di_candidates: 4,
            max_bounces: 3,
            media: MediaSettings::default(),
            debug_view: LightingDebugView::None,
            reference: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MediaSettings {
    /// Steps of the in-scatter march on path segments of 500 m or more.
    pub segment_steps: u32,
}

impl Default for MediaSettings {
    fn default() -> Self {
        Self { segment_steps: 16 }
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
}

/// The accumulation reference: no Russian roulette before 64 bounces, and optionally a thin
/// lens traced from the eye for depth-of-field references.
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

/// Bounces from which Russian roulette may end a path: after the first, or 64 in the
/// reference.
const ROULETTE_DEPTH: u32 = 1;
const REFERENCE_ROULETTE_DEPTH: u32 = 64;

/// One view's lighting targets at its render extent, kept with the view's history.
pub struct LightingView {
    device: Device,
    extent: [u32; 2],
    radiance: Texture,
    slots: [u32; 2],
    radiance_store: StorageImageIndex,
    radiance_sampled: TextureIndex,
    roots: Ring<PathRoot>,
}

impl LightingView {
    pub fn new(device: &Device, extent: [u32; 2]) -> Result<Self, Error> {
        let desc = TextureDesc::d2(
            Format::Rgba16Float,
            extent[0],
            extent[1],
            TextureUsage::SAMPLED | TextureUsage::STORAGE,
        );
        let radiance = device.create_texture(&desc, TextureMemory::Dedicated, "radiance")?;
        let slots = [device.alloc_image_slot()?, device.alloc_image_slot()?];
        let radiance_store = device.write_storage(slots[0], &radiance, &ViewDesc::default());
        let radiance_sampled = device.write_sampled(slots[1], &radiance, &ViewDesc::default());
        Ok(Self {
            device: device.clone(),
            extent,
            radiance,
            slots,
            radiance_store,
            radiance_sampled,
            roots: Ring::new(device, "path roots")?,
        })
    }

    pub fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// RGBA16F, pre-exposed noisy radiance; alpha is 1 where the G-buffer holds a surface.
    pub fn radiance(&self) -> &Texture {
        &self.radiance
    }

    pub fn radiance_index(&self) -> TextureIndex {
        self.radiance_sampled
    }

    /// The radiance's storage view, for passes that draw over it after lighting.
    pub fn radiance_storage(&self) -> StorageImageIndex {
        self.radiance_store
    }
}

impl Drop for LightingView {
    fn drop(&mut self) {
        for slot in self.slots {
            self.device.free_image_slot_after_submitted(slot);
        }
    }
}

/// What the opaque dispatch reads for one view and frame.
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
}

/// Which entry point of a kernel with both runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PathEntry {
    /// The raygen entry, which reorders at coherence points, where the device reorders in
    /// hardware; the compute entry otherwise.
    #[default]
    Fastest,
    Compute,
    /// Fails without `Caps::ray_dispatch`.
    RayDispatch,
}

enum Opaque {
    Compute(ComputePipeline),
    Rays(RayDispatchPipeline),
}

/// The per-view lighting kernels of one device.
pub struct LightingKernels {
    opaque: Opaque,
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
        let m = module(kernels, &opaque::MODULE)?;
        let bytes = opaque::Spec {
            openpbr_lut_slot: tables.first_slot(),
            ..opaque::Spec::default()
        }
        .bytes();
        let spec = SpecData {
            entries: opaque::Spec::ENTRIES,
            bytes: &bytes,
        };
        let caps = device.caps();
        let rays = match entry {
            PathEntry::Fastest => caps.ray_dispatch && caps.reorder,
            PathEntry::Compute => false,
            PathEntry::RayDispatch => true,
        };
        let opaque = if rays {
            let stage = ShaderStage::new(m, "raygen_main", spec)?;
            Opaque::Rays(device.create_ray_dispatch_pipeline(&stage, "opaque paths")?)
        } else {
            let stage = ShaderStage::new(m, "compute_main", spec)?;
            Opaque::Compute(device.create_compute_pipeline(&stage, "opaque paths")?)
        };
        Ok(Self { opaque })
    }

    /// Records the opaque dispatch of `pass` into `view`'s radiance after the scene's
    /// [`SceneLighting::record`] and the view's raster half in the same or an earlier list;
    /// orders nothing after it.
    pub fn record_opaque(
        &self,
        list: &mut CommandList<'_>,
        scene: &SceneLighting,
        view: &mut LightingView,
        pass: &ViewPass,
        lighting: &ViewLighting,
    ) {
        let (roulette_depth, lens) = match lighting.reference {
            Some(r) => (REFERENCE_ROULETTE_DEPTH, r.lens),
            None => (ROULETTE_DEPTH, None),
        };
        let mode = match lighting.debug_view {
            LightingDebugView::None => 0,
            LightingDebugView::DirectOnly => 1,
            LightingDebugView::IndirectOnly => 2,
            LightingDebugView::NeeOnly => 3,
            LightingDebugView::BsdfOnly => 4,
            LightingDebugView::RaysPerPixel => 5,
            LightingDebugView::RayOriginBias => 6,
        };
        let slot = (pass.frame % FRAME_SLOTS as u64) as usize;
        let root = view.roots.write(
            slot,
            PathRoot {
                scene: pass.scene,
                view: pass.view,
                gbuffer: pass.gbuffer,
                set: scene.light_set(),
                radiance: view.radiance_store,
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
                _pad0: 0,
            },
        );
        let [w, h] = view.extent;
        list.begin_pass("opaque paths", false);
        match &self.opaque {
            Opaque::Compute(p) => {
                let [gx, gy, _] = p.workgroup();
                list.dispatch(p, root, [w.div_ceil(gx), h.div_ceil(gy), 1]);
            }
            Opaque::Rays(p) => list.dispatch_rays(p, root, [w, h, 1]),
        }
        list.end_pass();
    }
}
