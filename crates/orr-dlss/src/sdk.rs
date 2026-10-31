//! NGX on an orr-gpu device: device hooks, initialization, and ray reconstruction and frame
//! generation recorded into orr command lists.

use std::ffi::CString;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use ash::vk;
use orr_gpu::vulkan::{self, Features};
use orr_gpu::{Allocation, CommandList, Device, DeviceHooks, QueueKind, Texture};

use crate::raw::FgCamera;
use crate::{DlssError, Quality, Support, Uuid, raw};

/// Serialises every NGX call this layer makes; NGX is not thread-safe.
static NGX: Mutex<()> = Mutex::new(());

fn ngx_lock() -> MutexGuard<'static, ()> {
    NGX.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// NGX state used only under [`NGX`].
struct Locked<T>(T);

// SAFETY: NGX handles have no thread affinity; every call on them holds the `NGX` lock.
unsafe impl<T> Send for Locked<T> {}
// SAFETY: as above.
unsafe impl<T> Sync for Locked<T> {}

/// How long feature creation waits for its setup commands.
const CREATE_TIMEOUT: Duration = Duration::from_secs(30);

/// The [`DeviceHooks`] that add NGX's instance and device extensions and features.
pub struct DlssDeviceHooks {
    project_id: Uuid,
}

impl DeviceHooks for DlssDeviceHooks {
    fn instance_extensions(&self, _entry: &ash::Entry) -> Vec<CString> {
        let _ngx = ngx_lock();
        raw::instance_extensions(self.project_id).unwrap_or_else(|e| {
            log::warn!("NGX instance extensions: {e}");
            Vec::new()
        })
    }

    fn device_extensions(
        &self,
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
    ) -> Vec<CString> {
        let _ngx = ngx_lock();
        // SAFETY: orr passes a physical device of `instance`.
        unsafe { raw::device_extensions(instance, adapter, self.project_id) }.unwrap_or_else(|e| {
            log::warn!("NGX device extensions: {e}");
            Vec::new()
        })
    }

    fn device_features(&self, _adapter: vk::PhysicalDevice, features: &mut Features<'_>) {
        raw::enable_features(features.vulkan_1_4);
    }
}

/// The NGX instance, released once the SDK and every feature made from it are gone.
struct NgxInstance(ManuallyDrop<Locked<raw::Ngx>>);

impl Drop for NgxInstance {
    fn drop(&mut self) {
        let _ngx = ngx_lock();
        // SAFETY: dropped once, here.
        unsafe { ManuallyDrop::drop(&mut self.0) };
    }
}

/// NGX initialised on an orr-gpu device. Every NGX call made through this API is serialised
/// across threads. Clones share the NGX instance.
#[derive(Clone)]
pub struct DlssSdk {
    ngx: Arc<NgxInstance>,
    gpu: Device,
}

impl DlssSdk {
    /// The hooks to create the device with; [`DlssSdk::new`] needs their extensions.
    pub fn device_hooks(project_id: Uuid) -> DlssDeviceHooks {
        DlssDeviceHooks { project_id }
    }

    /// Initialises NGX on `gpu`, which was created with [`DlssSdk::device_hooks`].
    pub fn new(gpu: &Device, project_id: Uuid) -> Result<Self, DlssError> {
        let device = vulkan::raw_device(gpu);
        let _ngx = ngx_lock();
        // SAFETY: the physical device belongs to the instance.
        let wanted =
            unsafe { raw::device_extensions(device.instance, device.physical_device, project_id)? };
        if let Some(missing) = wanted.iter().find(|e| !device.has_extension(e)) {
            return Err(DlssError::MissingExtension(
                missing.to_string_lossy().into_owned(),
            ));
        }
        let vulkan = raw::Vulkan {
            entry: device.entry,
            instance: device.instance,
            physical_device: device.physical_device,
            device: device.device,
        };
        // SAFETY: the device has NGX's extensions, and `NgxInstance` is released before it:
        // the SDK and every feature hold the device or release through its garbage list.
        let ngx = unsafe { raw::Ngx::new(vulkan, project_id)? };
        Ok(Self {
            ngx: Arc::new(NgxInstance(ManuallyDrop::new(Locked(ngx)))),
            gpu: gpu.clone(),
        })
    }

    pub fn support(&self) -> Support {
        let _ngx = ngx_lock();
        self.ngx.0.0.support()
    }

    /// Device memory NGX reports for the features alive now, in bytes.
    pub fn vram_bytes(&self) -> u64 {
        let _ngx = ngx_lock();
        self.ngx.0.0.vram_bytes()
    }
}

/// Ray reconstruction inputs at render resolution and the output at output resolution.
/// Every texture is 2D with one mip and one layer, and has sampled or storage usage.
#[derive(Clone, Copy)]
pub struct RrInputs<'a> {
    /// Pre-exposed HDR radiance.
    pub color: &'a Texture,
    /// Radiance before the transparency layer; not bound by default.
    pub color_before_transparency: Option<&'a Texture>,
    /// Reverse-Z device depth.
    pub depth: &'a Texture,
    /// Unjittered motion, previous UV minus current UV.
    pub motion: &'a Texture,
    pub diffuse_albedo: &'a Texture,
    pub specular_albedo: &'a Texture,
    /// World-space normal in RGB, linear roughness in A.
    pub normal_roughness: &'a Texture,
    /// Motion of reflected surfaces, in the same units as `motion`.
    pub specular_motion: &'a Texture,
    /// Premultiplied colour with coverage in alpha, composited after denoising.
    pub transparency: Option<&'a Texture>,
    pub depth_of_field_guide: Option<&'a Texture>,
    pub output: &'a Texture,
    /// Drops the history.
    pub reset: bool,
    /// From [`RayReconstruction::jitter`] for this frame.
    pub jitter: [f32; 2],
}

/// One ray reconstruction feature at a fixed render and output size.
///
/// Dropping it releases NGX's feature once everything submitted before the drop has
/// completed.
pub struct RayReconstruction {
    inner: Option<Locked<raw::RayReconstruction>>,
    ngx: Arc<NgxInstance>,
    gpu: Device,
}

impl RayReconstruction {
    /// Creates the feature. NGX's setup commands run on the Render queue, and this waits for
    /// them; call it on the Render queue's owner thread. `preset` is a
    /// `NVSDK_NGX_RayReconstruction_Hint_Render_Preset` value; `None` keeps NGX's default.
    pub fn new(
        sdk: &DlssSdk,
        output: [u32; 2],
        quality: Quality,
        preset: Option<u32>,
    ) -> Result<Self, DlssError> {
        let queue = sdk
            .gpu
            .queue(QueueKind::Render)
            .expect("the Render queue always exists");
        let mut list = queue.command_list()?;
        let inner = {
            let cmd = list.vulkan_raw();
            let _ngx = ngx_lock();
            // SAFETY: `cmd` records on the Render queue of NGX's device. The feature is
            // released through the device's garbage list, after the GPU has finished every
            // submit before the drop and before NGX shuts down (it holds the instance).
            unsafe {
                raw::RayReconstruction::new(&sdk.ngx.0.0, cmd.buffer, output, quality, preset)?
            }
        };
        let rr = Self {
            inner: Some(Locked(inner)),
            ngx: Arc::clone(&sdk.ngx),
            gpu: sdk.gpu.clone(),
        };
        let done = queue.submit(vec![list], &[])?;
        queue.wait(done, CREATE_TIMEOUT)?;
        Ok(rr)
    }

    fn raw(&self) -> &raw::RayReconstruction {
        &self.inner.as_ref().expect("released on drop only").0
    }

    /// Render resolution, fixed for the feature's lifetime.
    pub fn render_size(&self) -> [u32; 2] {
        self.raw().render_size()
    }

    pub fn output_size(&self) -> [u32; 2] {
        self.raw().output_size()
    }

    /// NGX's jitter offset for `frame` (see [`crate::jitter`]).
    pub fn jitter(&self, frame: u64) -> [f32; 2] {
        self.raw().jitter(frame)
    }

    /// Texture LOD bias for the render resolution (see [`crate::mip_bias`]).
    pub fn mip_bias(&self) -> f32 {
        self.raw().mip_bias()
    }

    /// Records the evaluation into `cmd`, a list on a compute-capable queue of the SDK's
    /// device. Inputs are at [`render_size`](Self::render_size), the output at
    /// [`output_size`](Self::output_size).
    pub fn record(
        &mut self,
        cmd: &mut CommandList<'_>,
        inputs: &RrInputs<'_>,
    ) -> Result<(), DlssError> {
        let opt = |t: Option<&Texture>, name| t.map(|t| image(t, name)).transpose();
        let color = image(inputs.color, "color")?;
        let depth = image(inputs.depth, "depth")?;
        let motion = image(inputs.motion, "motion")?;
        let diffuse = image(inputs.diffuse_albedo, "diffuse_albedo")?;
        let specular = image(inputs.specular_albedo, "specular_albedo")?;
        let normals = image(inputs.normal_roughness, "normal_roughness")?;
        let specular_motion = image(inputs.specular_motion, "specular_motion")?;
        let before = opt(
            inputs.color_before_transparency,
            "color_before_transparency",
        )?;
        let transparency = opt(inputs.transparency, "transparency")?;
        let dof = opt(inputs.depth_of_field_guide, "depth_of_field_guide")?;
        let output = image(inputs.output, "output")?;
        let raw_inputs = raw::RrInputs {
            color: &color,
            color_before_transparency: before.as_ref(),
            depth: &depth,
            motion: &motion,
            diffuse_albedo: &diffuse,
            specular_albedo: &specular,
            normal_roughness: &normals,
            specular_motion: &specular_motion,
            transparency: transparency.as_ref(),
            depth_of_field_guide: dof.as_ref(),
            output: &output,
            reset: inputs.reset,
            jitter: inputs.jitter,
        };
        let device = vulkan::raw_device(&self.gpu).device.handle();
        let raw_cmd = cmd.vulkan_raw();
        assert_eq!(
            raw_cmd.device.handle(),
            device,
            "RR records on its SDK's device"
        );
        let buffer = raw_cmd.buffer;
        let rr = &mut self.inner.as_mut().expect("released on drop only").0;
        let _ngx = ngx_lock();
        // SAFETY: `buffer` records on NGX's device, after the classic bind `vulkan_raw`
        // recorded. orr textures stay in GENERAL and are destroyed only after the work
        // submitted before their drop completes, so every image outlives this evaluation.
        unsafe { rr.record(buffer, &raw_inputs) }
    }
}

impl Drop for RayReconstruction {
    fn drop(&mut self) {
        let inner = self.inner.take();
        let ngx = Arc::clone(&self.ngx);
        vulkan::defer_after_submitted(&self.gpu, move || {
            {
                let _ngx = ngx_lock();
                // Locals drop in reverse order: the feature goes before the lock.
                let _feature = inner;
            }
            // Shutting NGX down takes the lock itself.
            drop(ngx);
        });
    }
}

/// One frame generation evaluate on orr textures. Images are 2D with one mip and layer, and
/// have sampled or storage usage.
#[derive(Clone, Copy)]
pub struct FgInputs<'a> {
    /// Display-encoded, without UI, in the format and size the feature was made for.
    pub backbuffer: &'a Texture,
    /// Reverse-Z device depth at render resolution.
    pub depth: &'a Texture,
    /// Unjittered motion at render resolution, previous UV minus current UV.
    pub motion: &'a Texture,
    /// RGBA16F bidirectional distortion field, while a lens or hook warps the image.
    pub distortion: Option<&'a Texture>,
    /// This index's output, in the backbuffer's format and size.
    pub interpolated: &'a Texture,
    /// Host-readable; read it with [`FrameGeneration::interpolation_disabled`] once the
    /// evaluate has completed.
    pub disable_interpolation: Option<&'a Allocation<u32>>,
    pub camera: FgCamera,
    /// NGX's jitter of the guides, render pixels.
    pub jitter: [f32; 2],
    /// Scales `motion` to render pixels: the render extent.
    pub mvec_scale: [f32; 2],
    pub reset: bool,
    /// Increases by one each rendered frame, with frame generation on or off.
    pub frame_id: u64,
    /// Menus and loading screens.
    pub not_rendering_scene: bool,
    /// Scales NGX's linearized depth, `1 / depth`; 1 is NGX's default.
    pub depth_scale: f32,
}

/// One frame generation feature for one backbuffer size and format.
///
/// Dropping it releases NGX's feature once everything submitted before the drop has
/// completed.
pub struct FrameGeneration {
    inner: Option<Locked<raw::FrameGeneration>>,
    hdr: bool,
    format: vk::Format,
    ngx: Arc<NgxInstance>,
    gpu: Device,
}

impl FrameGeneration {
    /// Creates the feature for backbuffers like `backbuffer`; `hdr` when they hold PQ. NGX's
    /// setup commands run on the Render queue, and this waits for them; NGX may finish
    /// creating its resources on its own thread. Call it on the Render queue's owner
    /// thread.
    pub fn new(sdk: &DlssSdk, backbuffer: &Texture, hdr: bool) -> Result<Self, DlssError> {
        let queue = sdk
            .gpu
            .queue(QueueKind::Render)
            .expect("the Render queue always exists");
        let h = vulkan::image_handles(backbuffer)?;
        let output = [h.extent.width, h.extent.height];
        let mut list = queue.command_list()?;
        let inner = {
            let cmd = list.vulkan_raw();
            let _ngx = ngx_lock();
            // SAFETY: `cmd` records on the Render queue of NGX's device. The feature is
            // released through the device's garbage list, after the GPU has finished every
            // submit before the drop and before NGX shuts down (it holds the instance).
            unsafe { raw::FrameGeneration::new(&sdk.ngx.0.0, cmd.buffer, output, h.format)? }
        };
        let fg = Self {
            inner: Some(Locked(inner)),
            hdr,
            format: h.format,
            ngx: Arc::clone(&sdk.ngx),
            gpu: sdk.gpu.clone(),
        };
        let done = queue.submit(vec![list], &[])?;
        queue.wait(done, CREATE_TIMEOUT)?;
        Ok(fg)
    }

    fn raw(&self) -> &raw::FrameGeneration {
        &self.inner.as_ref().expect("released on drop only").0
    }

    pub fn output_size(&self) -> [u32; 2] {
        self.raw().output_size()
    }

    /// The most images one rendered frame may generate (`MultiFrameCountMax`).
    pub fn max_generated(&self) -> u32 {
        self.raw().max_generated()
    }

    /// Records generated image `index` of `count` (1-based) into `cmd`, a list on a
    /// compute-capable queue of the SDK's device. Record every index in order each frame.
    pub fn record(
        &mut self,
        cmd: &mut CommandList<'_>,
        inputs: &FgInputs<'_>,
        index: u32,
        count: u32,
    ) -> Result<(), DlssError> {
        let backbuffer = image(inputs.backbuffer, "backbuffer")?;
        assert_eq!(
            backbuffer.format, self.format,
            "FG backbuffers keep the format the feature was made for"
        );
        let depth = image(inputs.depth, "depth")?;
        let motion = image(inputs.motion, "motion")?;
        let distortion = inputs
            .distortion
            .map(|t| image(t, "distortion"))
            .transpose()?;
        let interpolated = image(inputs.interpolated, "interpolated")?;
        let flag = inputs.disable_interpolation.map(|a| raw::Buffer {
            buffer: vulkan::buffer_handle(a),
            size: a.size,
        });
        let raw_inputs = raw::FgInputs {
            backbuffer: &backbuffer,
            depth: &depth,
            motion: &motion,
            distortion: distortion.as_ref(),
            interpolated: &interpolated,
            disable_interpolation: flag,
            camera: &inputs.camera,
            jitter: inputs.jitter,
            mvec_scale: inputs.mvec_scale,
            reset: inputs.reset,
            frame_id: inputs.frame_id,
            not_rendering_scene: inputs.not_rendering_scene,
            hdr: self.hdr,
            depth_scale: inputs.depth_scale,
        };
        let device = vulkan::raw_device(&self.gpu).device.handle();
        let raw_cmd = cmd.vulkan_raw();
        assert_eq!(
            raw_cmd.device.handle(),
            device,
            "FG records on its SDK's device"
        );
        let buffer = raw_cmd.buffer;
        let fg = &mut self.inner.as_mut().expect("released on drop only").0;
        let _ngx = ngx_lock();
        // SAFETY: `buffer` records on NGX's device, after the classic bind `vulkan_raw`
        // recorded. orr textures stay in GENERAL, and they and the flag buffer are destroyed
        // only after the work submitted before their drop completes.
        unsafe { fg.record(buffer, &raw_inputs, index, count) }
    }

    /// Whether NGX asked, through `flag`, that this frame's generated images not be shown.
    /// Read it once the evaluates that wrote it have completed; zero it before they run.
    pub fn interpolation_disabled(flag: &Allocation<u32>) -> bool {
        flag.as_slice()
            .and_then(|v| v.first())
            .is_some_and(|v| v & 0xff != 0)
    }
}

impl Drop for FrameGeneration {
    fn drop(&mut self) {
        let inner = self.inner.take();
        let ngx = Arc::clone(&self.ngx);
        vulkan::defer_after_submitted(&self.gpu, move || {
            {
                let _ngx = ngx_lock();
                let _feature = inner;
            }
            drop(ngx);
        });
    }
}

/// The raw image of a texture NGX reads or writes.
fn image(t: &Texture, name: &str) -> Result<raw::Image, DlssError> {
    let d = t.desc();
    assert!(
        d.mips == 1 && d.layers == 1 && d.extent[2] == 1,
        "NGX {name}: one 2D mip and layer, not {d:?}"
    );
    let h = vulkan::image_handles(t)?;
    Ok(raw::Image {
        image: h.image,
        view: h.view,
        format: h.format,
        extent: [h.extent.width, h.extent.height],
    })
}
