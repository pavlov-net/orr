//! Per-view lighting: the settings of one `render_view` call, the view's lighting targets and
//! history, the opaque path dispatch and the guide resolve.

use orr_math::{DVec3, Quat};
use orr_scene::orr_gpu::{
    self, CommandList, ComputePipeline, Device, Format, GpuPtr, RayDispatchPipeline, ShaderStage,
    SpecData, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage,
};
use orr_scene::{GBufferGpu, SceneId, SceneRoot, ViewConstants, ViewDesc};

use crate::scene::{FRAME_SLOTS, Ring};
use crate::shaders::guides::resolve;
use crate::shaders::path::opaque;
use crate::shaders::types::{GuideRoot, PathRoot};
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
            max_bounces: 3,
            media: MediaSettings::default(),
            guides: GuideSettings::default(),
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
    /// A ray reconstruction guide as the guide resolve computed it, before its target's
    /// quantization; alpha is 1 on surfaces.
    Guide(GuideKind),
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

/// A [`LightingView`] target and its storage view.
struct Target {
    texture: Texture,
    store: StorageImageIndex,
}

/// One view's lighting targets at its render extent, which are ray reconstruction's inputs,
/// and its history; kept with the view's history.
pub struct LightingView {
    device: Device,
    extent: [u32; 2],
    radiance: Target,
    diffuse_albedo: Target,
    specular_albedo: Target,
    normal_roughness: Target,
    specular_motion: Target,
    depth: Target,
    motion: Target,
    slots: Vec<u32>,
    radiance_sampled: TextureIndex,
    roots: Ring<PathRoot>,
    guide_roots: Ring<GuideRoot>,
    previous: Option<(SceneId, DVec3, Quat)>,
    history_reset: bool,
}

impl LightingView {
    pub fn new(device: &Device, extent: [u32; 2]) -> Result<Self, Error> {
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
        let radiance = target(Format::Rgba16Float, "radiance")?;
        let diffuse_albedo = target(Format::Rgba8Unorm, "diffuse albedo")?;
        let specular_albedo = target(Format::Rgba8Unorm, "specular albedo")?;
        let normal_roughness = target(Format::Rgba16Float, "normal roughness")?;
        let specular_motion = target(Format::Rg16Float, "specular motion")?;
        let depth = target(Format::R32Float, "rr depth")?;
        let motion = target(Format::Rg16Float, "rr motion")?;
        let sampled = device.alloc_image_slot()?;
        slots.push(sampled);
        let radiance_sampled =
            device.write_sampled(sampled, &radiance.texture, &orr_gpu::ViewDesc::default());
        Ok(Self {
            device: device.clone(),
            extent,
            radiance,
            diffuse_albedo,
            specular_albedo,
            normal_roughness,
            specular_motion,
            depth,
            motion,
            slots,
            radiance_sampled,
            roots: Ring::new(device, "path roots")?,
            guide_roots: Ring::new(device, "guide roots")?,
            previous: None,
            history_reset: true,
        })
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
        &self.radiance.texture
    }

    pub fn radiance_index(&self) -> TextureIndex {
        self.radiance_sampled
    }

    /// The radiance's storage view, for passes that draw over it after lighting.
    pub fn radiance_storage(&self) -> StorageImageIndex {
        self.radiance.store
    }

    /// RGBA8 unorm, not sRGB: diffuse albedo, subsurface included; 0.5 on sky pixels.
    pub fn diffuse_albedo(&self) -> &Texture {
        &self.diffuse_albedo.texture
    }

    /// RGBA8 unorm, not sRGB: specular albedo; 0 on sky pixels.
    pub fn specular_albedo(&self) -> &Texture {
        &self.specular_albedo.texture
    }

    /// RGBA16F: world normal in RGB and linear roughness in A; 0 on sky pixels.
    pub fn normal_roughness(&self) -> &Texture {
        &self.normal_roughness.texture
    }

    /// RG16F, previous uv minus current uv: the mirror reflection's motion on smooth pixels,
    /// the surface's elsewhere.
    pub fn specular_motion(&self) -> &Texture {
        &self.specular_motion.texture
    }

    /// R32F reverse-Z device depth, 0 on sky pixels.
    pub fn depth(&self) -> &Texture {
        &self.depth.texture
    }

    /// RG16F, previous uv minus current uv, unjittered.
    pub fn motion(&self) -> &Texture {
        &self.motion.texture
    }
}

impl Drop for LightingView {
    fn drop(&mut self) {
        for &slot in &self.slots {
            self.device.free_image_slot_after_submitted(slot);
        }
    }
}

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
    guides: ComputePipeline,
}

/// The specialisation constants of kernel module `$kernel`, which imports `bsdf`.
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
        let m = module(kernels, &opaque::MODULE)?;
        let bytes = bsdf_spec!(opaque, tables);
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
        let bytes = bsdf_spec!(resolve, tables);
        let spec = SpecData {
            entries: resolve::Spec::ENTRIES,
            bytes: &bytes,
        };
        let stage = ShaderStage::new(module(kernels, &resolve::MODULE)?, "main", spec)?;
        let guides = device.create_compute_pipeline(&stage, "guide resolve")?;
        Ok(Self { opaque, guides })
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
            LightingDebugView::None | LightingDebugView::Guide(_) => 0,
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
                radiance: view.radiance.store,
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

    /// Records the guide resolve of `pass` into `view`'s guide targets after the view's
    /// raster half. Under a [`LightingDebugView::Guide`] view it also overwrites the
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
                radiance: view.radiance.store,
                diffuse_albedo: view.diffuse_albedo.store,
                specular_albedo: view.specular_albedo.store,
                normal_roughness: view.normal_roughness.store,
                specular_motion: view.specular_motion.store,
                depth: view.depth.store,
                motion: view.motion.store,
                motion_roughness: lighting.guides.specular_motion_roughness,
                debug,
                _pad0: 0,
            },
        );
        let [w, h] = view.extent;
        let [gx, gy, _] = self.guides.workgroup();
        list.begin_pass("guide resolve", false);
        list.dispatch(&self.guides, root, [w.div_ceil(gx), h.div_ceil(gy), 1]);
        list.end_pass();
    }
}
