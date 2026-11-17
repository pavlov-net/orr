//! Denoising and upscaling of path-traced radiance: `MTL4FXTemporalDenoisedScaler` on orr
//! textures.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTL4FXTemporalDenoisedScaler, MTLFXFrameInterpolatableScaler, MTLFXTemporalDenoisedScalerBase,
    MTLFXTemporalDenoisedScalerDescriptor,
};
use orr_gpu::interop::{defer_after_submitted, raw_device, texture_handle};
use orr_gpu::{CommandList, Device, Texture};

use crate::{Error, Float4x4, Input, Keep, ScaleRange, check_usage, encode_after_list, support};

/// Ray reconstruction's inputs at render resolution and its output at output resolution, with
/// orr's conventions.
#[derive(Clone, Copy)]
pub struct DenoisedInputs<'a> {
    /// Pre-exposed HDR radiance.
    pub color: &'a Texture,
    /// Reverse-Z device depth.
    pub depth: &'a Texture,
    /// Unjittered motion, previous UV minus current UV.
    pub motion: &'a Texture,
    pub diffuse_albedo: &'a Texture,
    pub specular_albedo: &'a Texture,
    /// World-space normal in RGB.
    pub normal: &'a Texture,
    /// Linear roughness in R.
    pub roughness: &'a Texture,
    /// m from each surface to its mirror reflection's hit.
    pub specular_hit_distance: Option<&'a Texture>,
    /// Premultiplied linear color with coverage in alpha, composited after denoising.
    pub transparency: Option<&'a Texture>,
    pub output: &'a Texture,
    pub reset: bool,
    /// orr's jitter: each sample sits at its pixel center minus this, render pixels, y down.
    pub jitter: [f32; 2],
    /// World from view and clip from view, unjittered, with world relative to a point that
    /// stays put while the history lasts.
    pub world_to_view: Float4x4,
    pub view_to_clip: Float4x4,
}

/// One view's ray reconstruction at a fixed render and output size. The effect is made at the
/// first [`record`](Self::record), from its inputs' formats.
pub struct DenoisedScaler {
    device: Device,
    render: [u32; 2],
    output: [u32; 2],
    made: Option<Made>,
}

struct Made {
    raw: Retained<ProtocolObject<dyn MTL4FXTemporalDenoisedScaler>>,
    formats: Formats,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Formats {
    color: MTLPixelFormat,
    depth: MTLPixelFormat,
    motion: MTLPixelFormat,
    diffuse: MTLPixelFormat,
    specular: MTLPixelFormat,
    normal: MTLPixelFormat,
    roughness: MTLPixelFormat,
    hit_distance: Option<MTLPixelFormat>,
    transparency: Option<MTLPixelFormat>,
    output: MTLPixelFormat,
}

impl DenoisedScaler {
    /// The device's scale range, or why it has none.
    pub fn scale_range(device: &Device) -> Result<ScaleRange, Error> {
        support(device)
            .denoised
            .ok_or(Error::Unsupported("denoised scaler"))
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

    /// Records the reconstruction into `list`, on a queue of the scaler's device.
    pub fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &DenoisedInputs<'_>,
    ) -> Result<(), Error> {
        let raw = texture_handle;
        let fmt = |t: &Texture| raw(t).pixelFormat();
        let formats = Formats {
            color: fmt(inputs.color),
            depth: fmt(inputs.depth),
            motion: fmt(inputs.motion),
            diffuse: fmt(inputs.diffuse_albedo),
            specular: fmt(inputs.specular_albedo),
            normal: fmt(inputs.normal),
            roughness: fmt(inputs.roughness),
            hit_distance: inputs.specular_hit_distance.map(fmt),
            transparency: inputs.transparency.map(fmt),
            output: fmt(inputs.output),
        };
        // An input that comes or goes, such as the transparency overlay, needs another effect,
        // which starts without history.
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
        // for.
        unsafe {
            s.setColorTexture(Some(raw(inputs.color)));
            s.setDepthTexture(Some(raw(inputs.depth)));
            s.setMotionTexture(Some(raw(inputs.motion)));
            s.setDiffuseAlbedoTexture(Some(raw(inputs.diffuse_albedo)));
            s.setSpecularAlbedoTexture(Some(raw(inputs.specular_albedo)));
            s.setNormalTexture(Some(raw(inputs.normal)));
            s.setRoughnessTexture(Some(raw(inputs.roughness)));
            s.setSpecularHitDistanceTexture(inputs.specular_hit_distance.map(raw));
            s.setTransparencyOverlayTexture(inputs.transparency.map(raw));
            s.setOutputTexture(Some(raw(inputs.output)));
            // The color is pre-exposed, and the history drops at each step of the pre-exposure.
            s.setExposureTexture(None);
            s.setPreExposure(1.0);
            let [x, y] = crate::conventions::jitter(inputs.jitter);
            s.setJitterOffsetX(x);
            s.setJitterOffsetY(y);
            let [mx, my] = crate::conventions::motion_scale(self.render);
            s.setMotionVectorScaleX(mx);
            s.setMotionVectorScaleY(my);
            s.setDepthReversed(true);
            s.setShouldResetHistory(reset);
        }
        crate::set_scaler_matrices(
            ProtocolObject::from_ref(e),
            inputs.world_to_view,
            inputs.view_to_clip,
        );
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
        let compiler = crate::compiler(rd.device)?;
        // SAFETY: descriptor setters and effect creation on a live device and compiler.
        let raw = unsafe {
            let d = MTLFXTemporalDenoisedScalerDescriptor::new();
            d.setColorTextureFormat(f.color);
            d.setDepthTextureFormat(f.depth);
            d.setMotionTextureFormat(f.motion);
            d.setDiffuseAlbedoTextureFormat(f.diffuse);
            d.setSpecularAlbedoTextureFormat(f.specular);
            d.setNormalTextureFormat(f.normal);
            d.setRoughnessTextureFormat(f.roughness);
            if let Some(h) = f.hit_distance {
                d.setSpecularHitDistanceTextureEnabled(true);
                d.setSpecularHitDistanceTextureFormat(h);
            }
            if let Some(t) = f.transparency {
                d.setTransparencyOverlayTextureEnabled(true);
                d.setTransparencyOverlayTextureFormat(t);
            }
            d.setOutputTextureFormat(f.output);
            d.setInputWidth(render[0] as usize);
            d.setInputHeight(render[1] as usize);
            d.setOutputWidth(output[0] as usize);
            d.setOutputHeight(output[1] as usize);
            d.setAutoExposureEnabled(false);
            d.newTemporalDenoisedScalerWithDevice_compiler(rd.device, &compiler)
        }
        .ok_or_else(|| Error::Create {
            effect: "denoised scaler",
            formats: format!("{f:?}"),
        })?;
        Ok(Self { raw, formats: f })
    }
}

impl DenoisedScaler {
    /// Releases the effect once the GPU is done with it.
    fn release(&mut self) {
        if let Some(m) = self.made.take() {
            let raw = Keep(m.raw);
            defer_after_submitted(&self.device, move || drop(raw));
        }
    }
}

impl Drop for DenoisedScaler {
    fn drop(&mut self) {
        self.release();
    }
}

// SAFETY: one thread at a time configures and encodes the effect (`record` takes `&mut self`),
// and MetalFX effects, like the command buffers they encode into, may move between threads.
unsafe impl Send for DenoisedScaler {}
