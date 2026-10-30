use ash::vk;

/// An orr-dlss failure.
#[derive(Debug, thiserror::Error)]
pub enum DlssError {
    /// The crate was built without `DLSS_SDK` and without the `mock` feature.
    #[error("orr-dlss was built without the DLSS SDK")]
    NoSdk,
    /// NGX does not offer the feature on this device.
    #[error("{0} is not supported on this device")]
    Unsupported(&'static str),
    /// An NGX call failed.
    #[error("{call} failed: NGX result {result:#x}")]
    Ngx { call: &'static str, result: u32 },
    /// A Vulkan call failed.
    #[error("{call} failed: {result}")]
    Vulkan {
        call: &'static str,
        result: vk::Result,
    },
}
