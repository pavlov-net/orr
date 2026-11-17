//! Selectors `objc2-metal-fx` does not bind: the camera matrices, which take `simd_float4x4`,
//! a type objc2 has no encoding for, and the frame interpolator's matrices, which macOS 27
//! adds.

use std::arch::aarch64::float32x4_t;

use objc2::encode::{Encode, Encoding};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::{msg_send, sel};
use objc2_metal_fx::{MTLFXFrameInterpolatorBase, MTLFXTemporalDenoisedScalerBase};

/// A column-major 4x4 matrix, laid out and passed as C's `simd_float4x4`: four 16-byte vector
/// columns, which arm64 passes in vector registers. An array of floats of the same size would
/// go through memory instead.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Float4x4 {
    columns: [float32x4_t; 4],
}

impl Float4x4 {
    /// The matrix whose column `i` is `columns[i]`.
    pub fn from_cols_array_2d(columns: &[[f32; 4]; 4]) -> Self {
        // SAFETY: `float32x4_t` is four `f32` lanes in order, with no invalid bit patterns.
        let columns = unsafe { std::mem::transmute::<[[f32; 4]; 4], [float32x4_t; 4]>(*columns) };
        Self { columns }
    }

    pub fn to_cols_array_2d(self) -> [[f32; 4]; 4] {
        // SAFETY: as in `from_cols_array_2d`.
        unsafe { std::mem::transmute::<[float32x4_t; 4], [[f32; 4]; 4]>(self.columns) }
    }
}

// SAFETY: the encoding the Objective-C runtime reports for `simd_float4x4`, `{?=[4]}`: clang
// writes no element encoding for vector types.
unsafe impl Encode for Float4x4 {
    const ENCODING: Encoding = Encoding::Struct("?", &[Encoding::Array(4, &Encoding::None)]);
}

/// Sets the world-to-view and view-to-clip matrices a denoised scaler reads its normals and
/// depth through.
pub fn set_scaler_matrices(
    scaler: &ProtocolObject<dyn MTLFXTemporalDenoisedScalerBase>,
    world_to_view: Float4x4,
    view_to_clip: Float4x4,
) {
    // SAFETY: both selectors are required by the protocol and take one `simd_float4x4`.
    unsafe {
        let _: () = msg_send![scaler, setWorldToViewMatrix: world_to_view];
        let _: () = msg_send![scaler, setViewToClipMatrix: view_to_clip];
    }
}

/// Whether the OS's frame interpolators take camera matrices (macOS 27).
pub fn interpolator_has_matrices(
    interpolator: &ProtocolObject<dyn MTLFXFrameInterpolatorBase>,
) -> bool {
    interpolator.respondsToSelector(sel!(setWorldToViewMatrix:))
}

/// Sets the matrices an interpolator derives camera motion from. Needs
/// [`interpolator_has_matrices`].
pub fn set_interpolator_matrices(
    interpolator: &ProtocolObject<dyn MTLFXFrameInterpolatorBase>,
    world_to_view: Float4x4,
    view_to_clip: Float4x4,
) {
    assert!(
        interpolator_has_matrices(interpolator),
        "frame interpolator matrices need macOS 27"
    );
    // SAFETY: the interpolator implements both selectors (checked), which take one
    // `simd_float4x4`.
    unsafe {
        let _: () = msg_send![interpolator, setWorldToViewMatrix: world_to_view];
        let _: () = msg_send![interpolator, setViewToClipMatrix: view_to_clip];
    }
}

#[cfg(test)]
mod tests {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLPixelFormat};
    use objc2_metal_fx::MTLFXTemporalDenoisedScalerDescriptor;

    use super::*;

    /// The matrices reach a scaler column for column: a wrong argument layout for
    /// `simd_float4x4` reads other registers or memory.
    #[test]
    fn matrices_reach_the_scaler_unchanged() {
        let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
        // SAFETY: plain descriptor creation, setters and scaler creation on a live device.
        let scaler = unsafe {
            let desc = MTLFXTemporalDenoisedScalerDescriptor::new();
            desc.setColorTextureFormat(MTLPixelFormat::RGBA16Float);
            desc.setDepthTextureFormat(MTLPixelFormat::R32Float);
            desc.setMotionTextureFormat(MTLPixelFormat::RG16Float);
            desc.setDiffuseAlbedoTextureFormat(MTLPixelFormat::RGBA8Unorm);
            desc.setSpecularAlbedoTextureFormat(MTLPixelFormat::RGBA8Unorm);
            desc.setNormalTextureFormat(MTLPixelFormat::RGBA16Float);
            desc.setRoughnessTextureFormat(MTLPixelFormat::R16Float);
            desc.setOutputTextureFormat(MTLPixelFormat::RGBA16Float);
            desc.setInputWidth(64);
            desc.setInputHeight(64);
            desc.setOutputWidth(128);
            desc.setOutputHeight(128);
            desc.newTemporalDenoisedScalerWithDevice(&device)
        }
        .expect("a denoised scaler");
        let cols =
            |base: f32| std::array::from_fn(|c| std::array::from_fn(|r| base + (4 * c + r) as f32));
        let (a, b) = (cols(1.0), cols(100.0));
        let scaler = ProtocolObject::from_ref(&*scaler);
        set_scaler_matrices(
            scaler,
            Float4x4::from_cols_array_2d(&a),
            Float4x4::from_cols_array_2d(&b),
        );
        // SAFETY: required protocol getters returning `simd_float4x4`.
        let (wv, vc): (Float4x4, Float4x4) = unsafe {
            (
                msg_send![scaler, worldToViewMatrix],
                msg_send![scaler, viewToClipMatrix],
            )
        };
        assert_eq!(wv.to_cols_array_2d(), a);
        assert_eq!(vc.to_cols_array_2d(), b);
    }
}
