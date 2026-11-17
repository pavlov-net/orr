//! Temporal upscaling and antialiasing: `MTL4FXTemporalScaler` on orr textures.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTL4FXTemporalScaler, MTLFXFrameInterpolatableScaler, MTLFXTemporalScalerBase,
    MTLFXTemporalScalerDescriptor,
};
use orr_gpu::interop::{defer_after_submitted, raw_device, texture_handle};
use orr_gpu::{CommandList, Device, Texture};

use crate::{Error, Input, Keep, ScaleRange, check_usage, encode_after_list, support};

/// Temporal upscaling's inputs at render resolution and its output at output resolution, with
/// orr's conventions; [`TemporalScaler::record`] converts them to MetalFX's.
#[derive(Clone, Copy)]
pub struct TemporalInputs<'a> {
    /// HDR color, multiplied by `pre_exposure`.
    pub color: &'a Texture,
    /// Reverse-Z device depth.
    pub depth: &'a Texture,
    /// Unjittered motion, previous UV minus current UV.
    pub motion: &'a Texture,
    /// 1x1: the exposure that brings `color / pre_exposure` to the tonemapper's range; `None`
    /// for color already exposed.
    pub exposure: Option<&'a Texture>,
    pub pre_exposure: f32,
    /// One channel in [0, 1]: how far to trust this frame over the history.
    pub reactive_mask: Option<&'a Texture>,
    pub output: &'a Texture,
    /// Drops the history.
    pub reset: bool,
    /// orr's jitter for this frame: the projection samples each pixel at its center minus this
    /// value, in render pixels, y down.
    pub jitter: [f32; 2],
}

/// One view's temporal upscaling at a fixed render and output size. The effect is made at the
/// first [`record`](Self::record), from its inputs' formats.
pub struct TemporalScaler {
    device: Device,
    render: [u32; 2],
    output: [u32; 2],
    made: Option<Made>,
}

struct Made {
    raw: Retained<ProtocolObject<dyn MTL4FXTemporalScaler>>,
    formats: Formats,
}

#[derive(Clone, Copy, PartialEq)]
struct Formats {
    color: MTLPixelFormat,
    depth: MTLPixelFormat,
    motion: MTLPixelFormat,
    output: MTLPixelFormat,
    reactive: Option<MTLPixelFormat>,
}

impl TemporalScaler {
    /// The device's input scale range, or why it has none.
    pub fn scale_range(device: &Device) -> Result<ScaleRange, Error> {
        support(device)
            .temporal
            .ok_or(Error::Unsupported("temporal scaler"))
    }

    /// A scaler from `render` to `output`, whose scale must be within
    /// [`scale_range`](Self::scale_range).
    pub fn new(device: &Device, render: [u32; 2], output: [u32; 2]) -> Result<Self, Error> {
        Self::scale_range(device)?.check(render, output)?;
        Ok(Self {
            device: device.clone(),
            render,
            output,
            made: None,
        })
    }

    pub fn render_size(&self) -> [u32; 2] {
        self.render
    }

    pub fn output_size(&self) -> [u32; 2] {
        self.output
    }

    /// The effect, for a frame interpolator to link to; `None` before the first record.
    pub fn interpolatable(&self) -> Option<&ProtocolObject<dyn MTLFXFrameInterpolatableScaler>> {
        self.made
            .as_ref()
            .map(|m| ProtocolObject::from_ref(&*m.raw))
    }

    /// Records the upscale into `list`, which must be on a queue of the scaler's device.
    pub fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &TemporalInputs<'_>,
    ) -> Result<(), Error> {
        let raw = texture_handle;
        let formats = Formats {
            color: raw(inputs.color).pixelFormat(),
            depth: raw(inputs.depth).pixelFormat(),
            motion: raw(inputs.motion).pixelFormat(),
            output: raw(inputs.output).pixelFormat(),
            reactive: inputs.reactive_mask.map(|t| raw(t).pixelFormat()),
        };
        // An input that comes or goes, such as the reactive mask, needs another effect, which
        // starts without history.
        let mut reset = inputs.reset;
        if self.made.as_ref().is_some_and(|m| m.formats != formats) {
            self.release();
            reset = true;
        }
        let made = match &mut self.made {
            Some(m) => m,
            None => self
                .made
                .insert(Made::new(&self.device, self.render, self.output, formats)?),
        };
        let s = &made.raw;
        let e = &**s;
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
        // for; MetalFX holds what it is given until the next set.
        unsafe {
            s.setColorTexture(Some(raw(inputs.color)));
            s.setDepthTexture(Some(raw(inputs.depth)));
            s.setMotionTexture(Some(raw(inputs.motion)));
            s.setOutputTexture(Some(raw(inputs.output)));
            s.setExposureTexture(inputs.exposure.map(raw));
            s.setReactiveMaskTexture(inputs.reactive_mask.map(raw));
            s.setPreExposure(inputs.pre_exposure);
            s.setInputContentWidth(self.render[0] as usize);
            s.setInputContentHeight(self.render[1] as usize);
            let [x, y] = crate::conventions::jitter(inputs.jitter);
            s.setJitterOffsetX(x);
            s.setJitterOffsetY(y);
            let [mx, my] = crate::conventions::motion_scale(self.render);
            s.setMotionVectorScaleX(mx);
            s.setMotionVectorScaleY(my);
            s.setDepthReversed(true);
            s.setReset(reset);
        }
        encode_after_list(list, |buffer| {
            // SAFETY: a Metal 4 effect encoding into a Metal 4 command buffer of its device.
            unsafe { s.encodeToCommandBuffer(buffer) }
        });
        Ok(())
    }
}

impl Made {
    fn new(device: &Device, render: [u32; 2], output: [u32; 2], f: Formats) -> Result<Self, Error> {
        let rd = raw_device(device);
        // SAFETY: descriptor setters and effect creation on a live device and compiler.
        let raw = unsafe {
            let d = MTLFXTemporalScalerDescriptor::new();
            d.setColorTextureFormat(f.color);
            d.setDepthTextureFormat(f.depth);
            d.setMotionTextureFormat(f.motion);
            d.setOutputTextureFormat(f.output);
            d.setInputWidth(render[0] as usize);
            d.setInputHeight(render[1] as usize);
            d.setOutputWidth(output[0] as usize);
            d.setOutputHeight(output[1] as usize);
            d.setAutoExposureEnabled(false);
            if let Some(r) = f.reactive {
                d.setReactiveMaskTextureEnabled(true);
                d.setReactiveMaskTextureFormat(r);
            }
            d.newTemporalScalerWithDevice_compiler(rd.device, rd.compiler)
        }
        .ok_or_else(|| Error::Create {
            effect: "temporal scaler",
            formats: format!(
                "color {:?}, depth {:?}, motion {:?}, output {:?}, reactive {:?}",
                f.color, f.depth, f.motion, f.output, f.reactive
            ),
        })?;
        Ok(Self { raw, formats: f })
    }
}

impl TemporalScaler {
    /// Releases the effect once the GPU is done with it.
    fn release(&mut self) {
        if let Some(m) = self.made.take() {
            let raw = Keep(m.raw);
            defer_after_submitted(&self.device, move || drop(raw));
        }
    }
}

impl Drop for TemporalScaler {
    fn drop(&mut self) {
        self.release();
    }
}

// SAFETY: one thread at a time configures and encodes the effect (`record` takes `&mut self`),
// and MetalFX effects, like the command buffers they encode into, may move between threads.
unsafe impl Send for TemporalScaler {}
