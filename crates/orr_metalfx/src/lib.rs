//! MetalFX on orr's Metal 4 backend: which effects the device supports, and the parts of the
//! MetalFX API that `objc2-metal-fx` does not bind, the `simd_float4x4` camera matrices and
//! the frame interpolator's newer properties.
//!
//! The effects run through their Metal 4 variants, built on orr's `MTL4Compiler` and encoded
//! into orr's command buffers through [`orr_gpu::interop`].

#![cfg(target_vendor = "apple")]

mod ext;

pub use ext::{
    Float4x4, interpolator_has_matrices, set_interpolator_matrices, set_scaler_matrices,
};
pub use objc2_metal_fx;

use objc2_metal_fx::{
    MTLFXFrameInterpolatorDescriptor, MTLFXTemporalDenoisedScalerDescriptor,
    MTLFXTemporalScalerDescriptor,
};
use orr_gpu::Device;
use orr_gpu::interop::raw_device;

/// The input scales an upscaler accepts on a device: input size over output size, per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleRange {
    pub min: f32,
    pub max: f32,
}

impl ScaleRange {
    pub fn contains(self, scale: f32) -> bool {
        (self.min..=self.max).contains(&scale)
    }
}

/// The MetalFX effects a device runs on Metal 4 command buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Support {
    /// Denoise and upscale, with its scale range.
    pub denoised: Option<ScaleRange>,
    /// Temporal upscale, with its scale range.
    pub temporal: Option<ScaleRange>,
    /// Frame interpolation.
    pub interpolation: bool,
}

/// What `device` supports.
pub fn support(device: &Device) -> Support {
    let d = raw_device(device).device;
    // SAFETY: class queries on a live device.
    unsafe {
        let denoised =
            MTLFXTemporalDenoisedScalerDescriptor::supportsMetal4FX(d).then(|| ScaleRange {
                min: MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMinScaleForDevice(
                    d,
                ),
                max: MTLFXTemporalDenoisedScalerDescriptor::supportedInputContentMaxScaleForDevice(
                    d,
                ),
            });
        let temporal = MTLFXTemporalScalerDescriptor::supportsMetal4FX(d).then(|| ScaleRange {
            min: MTLFXTemporalScalerDescriptor::supportedInputContentMinScaleForDevice(d),
            max: MTLFXTemporalScalerDescriptor::supportedInputContentMaxScaleForDevice(d),
        });
        Support {
            denoised,
            temporal,
            interpolation: MTLFXFrameInterpolatorDescriptor::supportsMetal4FX(d),
        }
    }
}
