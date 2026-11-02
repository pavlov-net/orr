use ash::vk;

/// An orr_dlss failure.
#[derive(Debug, thiserror::Error)]
pub enum DlssError {
    /// The crate was built without `DLSS_SDK` and without the `mock` feature.
    #[error("orr_dlss was built without the DLSS SDK")]
    NoSdk,
    /// NGX does not offer the feature on this device.
    #[error("{0} is not supported on this device")]
    Unsupported(&'static str),
    /// NGX does not offer this mode at this output size.
    #[error("NGX does not offer {mode:?} at an output of {output:?}")]
    ModeUnavailable { mode: crate::Mode, output: [u32; 2] },
    /// An NGX call failed.
    #[error("{call} failed: NGX result {result:#x}")]
    Ngx { call: &'static str, result: u32 },
    /// The device was created without an extension NGX needs; create it with
    /// [`crate::DlssSdk::device_hooks`].
    #[error("the device lacks {0}, which NGX needs")]
    MissingExtension(String),
    /// An orr_gpu call failed.
    #[error(transparent)]
    Gpu(#[from] orr_gpu::Error),
    /// A Vulkan call failed.
    #[error("{call} failed: {result}")]
    Vulkan {
        call: &'static str,
        result: vk::Result,
    },
}
