//! Frame interpolation between two displayed frames: `MTL4FXFrameInterpolator` on orr textures.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTL4FXFrameInterpolator, MTLFXFrameInterpolatorBase, MTLFXFrameInterpolatorDescriptor,
};
use orr_gpu::interop::{defer_after_submitted, raw_device, texture_handle};
use orr_gpu::{CommandList, Device, Texture};

use crate::{
    Error, Float4x4, Input, Keep, check_usage, enable_interpolator_distortion, encode_after_list,
    interpolator_has_matrices, interpolators_take_distortion, set_interpolator_distortion,
    set_interpolator_matrices,
};

/// One interpolated image's inputs, with orr's conventions.
#[derive(Clone, Copy)]
pub struct InterpolatorInputs<'a> {
    /// The frame to interpolate toward, display-encoded and without UI.
    pub color: &'a Texture,
    /// The frame before it, like `color`.
    pub previous_color: &'a Texture,
    /// Reverse-Z device depth and unjittered motion (previous UV minus current UV) of `color`,
    /// at its render extent.
    pub depth: &'a Texture,
    pub motion: &'a Texture,
    /// RGBA16F bidirectional distortion field, while the image is warped.
    pub distortion: Option<&'a Texture>,
    /// Like `color`.
    pub output: &'a Texture,
    pub reset: bool,
    /// orr's jitter of `depth` and `motion`, render pixels.
    pub jitter: [f32; 2],
    /// Seconds between `previous_color` and `color`.
    pub delta_time: f32,
    /// m.
    pub near: f32,
    pub far: f32,
    /// Radians.
    pub fov_y: f32,
    /// Width over height of `color`.
    pub aspect: f32,
    /// World from view and clip from view, with world relative to a point that stays put while
    /// the history lasts; used where the OS takes them (macOS 27).
    pub world_to_view: Float4x4,
    pub view_to_clip: Float4x4,
}

/// Frame interpolation for one display image format and size. The effect is made at the first
/// [`record`](Self::record), from its inputs, and made again when the render extent or the
/// distortion field comes or goes.
pub struct FrameInterpolator {
    device: Device,
    made: Option<Made>,
}

struct Made {
    raw: Retained<ProtocolObject<dyn MTL4FXFrameInterpolator>>,
    key: Key,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Key {
    color: MTLPixelFormat,
    depth: MTLPixelFormat,
    motion: MTLPixelFormat,
    input: [u32; 2],
    output: [u32; 2],
    distortion: bool,
}

impl FrameInterpolator {
    /// Whether `device` runs Metal 4 frame interpolation.
    pub fn supported(device: &Device) -> bool {
        crate::support(device).interpolation
    }

    pub fn new(device: &Device) -> Result<Self, Error> {
        if !Self::supported(device) {
            return Err(Error::Unsupported("frame interpolator"));
        }
        Ok(Self {
            device: device.clone(),
            made: None,
        })
    }

    /// Records one interpolated image into `list`, on a queue of the interpolator's device.
    pub fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &InterpolatorInputs<'_>,
    ) -> Result<(), Error> {
        let raw = texture_handle;
        let extent = |t: &Texture| {
            let e = t.desc().extent;
            [e[0], e[1]]
        };
        let key = Key {
            color: raw(inputs.color).pixelFormat(),
            depth: raw(inputs.depth).pixelFormat(),
            motion: raw(inputs.motion).pixelFormat(),
            input: extent(inputs.depth),
            output: extent(inputs.color),
            distortion: inputs.distortion.is_some(),
        };
        let mut reset = inputs.reset;
        if self.made.as_ref().is_some_and(|m| m.key != key) {
            self.release();
            reset = true;
        }
        let made = match &mut self.made {
            Some(m) => m,
            None => self.made.insert(Made::new(&self.device, key)?),
        };
        let f = &made.raw;
        let e = &**f;
        // SAFETY: usage queries of a live effect.
        let need = unsafe {
            [
                e.colorTextureUsage(),
                e.depthTextureUsage(),
                e.motionTextureUsage(),
                e.outputTextureUsage(),
            ]
        };
        check_usage(Input::Color, raw(inputs.color), need[0])?;
        check_usage(Input::Depth, raw(inputs.depth), need[1])?;
        check_usage(Input::Motion, raw(inputs.motion), need[2])?;
        check_usage(Input::Output, raw(inputs.output), need[3])?;
        // SAFETY: setters of a live effect, with textures of the formats and sizes it was made
        // for.
        unsafe {
            f.setColorTexture(Some(raw(inputs.color)));
            f.setPrevColorTexture(Some(raw(inputs.previous_color)));
            f.setDepthTexture(Some(raw(inputs.depth)));
            f.setMotionTexture(Some(raw(inputs.motion)));
            f.setOutputTexture(Some(raw(inputs.output)));
            f.setUITexture(None);
            let [mx, my] = crate::conventions::motion_scale(key.input);
            f.setMotionVectorScaleX(mx);
            f.setMotionVectorScaleY(my);
            let [x, y] = crate::conventions::jitter(inputs.jitter);
            f.setJitterOffsetX(x);
            f.setJitterOffsetY(y);
            f.setDeltaTime(inputs.delta_time);
            f.setNearPlane(inputs.near);
            f.setFarPlane(inputs.far);
            f.setFieldOfView(inputs.fov_y.to_degrees());
            f.setAspectRatio(inputs.aspect);
            f.setDepthReversed(true);
            f.setShouldResetHistory(reset);
        }
        let base = ProtocolObject::<dyn MTLFXFrameInterpolatorBase>::from_ref(e);
        if interpolator_has_matrices(base) {
            set_interpolator_matrices(base, inputs.world_to_view, inputs.view_to_clip);
        }
        if key.distortion {
            set_interpolator_distortion(base, inputs.distortion.map(raw));
        }
        encode_after_list(list, |buffer| {
            // SAFETY: a Metal 4 effect encoding into a Metal 4 command buffer of its device.
            unsafe { f.encodeToCommandBuffer(buffer) }
        });
        Ok(())
    }

    /// Releases the effect once the GPU is done with it.
    fn release(&mut self) {
        if let Some(m) = self.made.take() {
            let raw = Keep(m.raw);
            defer_after_submitted(&self.device, move || drop(raw));
        }
    }
}

impl Made {
    fn new(device: &Device, k: Key) -> Result<Self, Error> {
        let rd = raw_device(device);
        // SAFETY: descriptor setters and effect creation on a live device and compiler.
        let raw = unsafe {
            let d = MTLFXFrameInterpolatorDescriptor::new();
            d.setColorTextureFormat(k.color);
            d.setOutputTextureFormat(k.color);
            d.setDepthTextureFormat(k.depth);
            d.setMotionTextureFormat(k.motion);
            d.setInputWidth(k.input[0] as usize);
            d.setInputHeight(k.input[1] as usize);
            d.setOutputWidth(k.output[0] as usize);
            d.setOutputHeight(k.output[1] as usize);
            if k.distortion {
                if !interpolators_take_distortion(&d) {
                    return Err(Error::Unsupported(
                        "frame interpolator distortion field (macOS 27)",
                    ));
                }
                enable_interpolator_distortion(&d);
            }
            d.newFrameInterpolatorWithDevice_compiler(rd.device, rd.compiler)
        }
        .ok_or_else(|| Error::Create {
            effect: "frame interpolator",
            formats: format!("{k:?}"),
        })?;
        Ok(Self { raw, key: k })
    }
}

impl Drop for FrameInterpolator {
    fn drop(&mut self) {
        self.release();
    }
}

// SAFETY: one thread at a time configures and encodes the effect (`record` takes `&mut self`),
// and MetalFX effects, like the command buffers they encode into, may move between threads.
unsafe impl Send for FrameInterpolator {}
