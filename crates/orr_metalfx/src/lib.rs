//! MetalFX on orr's Metal 4 backend: which effects the device supports, and the parts of the
//! MetalFX API that `objc2-metal-fx` does not bind, the `simd_float4x4` camera matrices and
//! the frame interpolator's newer properties.
//!
//! The effects run through their Metal 4 variants, built on orr's `MTL4Compiler` and encoded
//! into orr's command buffers through [`orr_gpu::interop`].

#![cfg(target_vendor = "apple")]

pub mod conventions;
mod denoised;
mod error;
mod ext;
mod temporal;

pub use denoised::{DenoisedInputs, DenoisedScaler};
pub use error::Error;
pub use ext::{
    Float4x4, interpolator_has_matrices, set_interpolator_matrices, set_scaler_matrices,
};
pub use objc2_metal_fx;
pub use temporal::{TemporalInputs, TemporalScaler};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTL4CommandBuffer, MTLTexture, MTLTextureUsage};
use objc2_metal_fx::{
    MTLFXFrameInterpolatorDescriptor, MTLFXTemporalDenoisedScalerDescriptor,
    MTLFXTemporalScalerDescriptor,
};
use orr_gpu::interop::raw_device;
use orr_gpu::{CommandList, Device};

/// The scales an upscaler accepts on a device: output size over input size, per axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScaleRange {
    pub min: f32,
    pub max: f32,
}

impl ScaleRange {
    pub fn contains(self, scale: f32) -> bool {
        (self.min..=self.max).contains(&scale)
    }

    /// Fails unless rendering at `render` for `output` is in range on both axes.
    pub fn check(self, render: [u32; 2], output: [u32; 2]) -> Result<(), Error> {
        let fits = (0..2).all(|i| self.contains(output[i] as f32 / render[i] as f32));
        if fits {
            Ok(())
        } else {
            Err(Error::Scale {
                render,
                output,
                min: self.min,
                max: self.max,
            })
        }
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

/// An effect's input or output, for errors.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Input {
    Color,
    Depth,
    Motion,
    Output,
}

impl Input {
    fn name(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::Depth => "depth",
            Self::Motion => "motion",
            Self::Output => "output",
        }
    }
}

/// Fails if `texture` lacks usage of `need`.
pub(crate) fn check_usage(
    input: Input,
    texture: &ProtocolObject<dyn MTLTexture>,
    need: MTLTextureUsage,
) -> Result<(), Error> {
    let have = texture.usage();
    if have.contains(need) {
        Ok(())
    } else {
        Err(Error::Usage {
            input: input.name(),
            have,
            need,
        })
    }
}

/// Has `encode` encode an effect into `list`'s command buffer, after everything recorded on the
/// list before. Work recorded on the list afterwards waits for the effect through a barrier, as
/// for orr's own work.
pub(crate) fn encode_after_list(
    list: &mut CommandList<'_>,
    encode: impl FnOnce(&ProtocolObject<dyn MTL4CommandBuffer>),
) {
    encode(list.metal_raw());
}

/// An effect handed to the release queue: dropped on whichever thread collects garbage, once the
/// GPU is done with it.
pub(crate) struct Keep<T: ?Sized>(pub(crate) Retained<T>);

// SAFETY: effects are only released through this, after their last encode; releasing an
// Objective-C object is thread-safe.
unsafe impl<T: ?Sized> Send for Keep<T> {}
