#[cfg(not(target_vendor = "apple"))]
use std::fmt;

/// An orr_gpu failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The Vulkan loader could not be found or loaded.
    #[cfg(not(target_vendor = "apple"))]
    #[error("Vulkan loader: {0}")]
    Loader(String),
    /// No device offers everything orr requires. Names the first missing item of the best
    /// candidate.
    #[error("no usable GPU: {device} lacks {missing}")]
    Unsupported { device: String, missing: String },
    /// No Vulkan 1.4 device was found, or on Apple targets no Metal device.
    #[cfg_attr(not(target_vendor = "apple"), error("no Vulkan 1.4 device"))]
    #[cfg_attr(target_vendor = "apple", error("no Metal device"))]
    NoDevice,
    /// A Vulkan call failed.
    #[cfg(not(target_vendor = "apple"))]
    #[error("{call} failed: {result}")]
    Vulkan {
        call: &'static str,
        result: VkResult,
    },
    /// A wait ran out of time.
    #[error("timed out")]
    Timeout,
    /// The device was lost.
    #[error("device lost")]
    DeviceLost,
    /// A handle from [`crate::Device::cached_pipelines_only`] asked for a pipeline no cache holds.
    #[error("the pipeline is not in a cache")]
    CompileRequired,
    /// A shader stage does not fit the call.
    #[error("shader: {0}")]
    Shader(String),
    /// An argument breaks a documented rule.
    #[error("invalid argument: {0}")]
    Invalid(String),
    /// The heap has no free slot.
    #[error("{0} heap is full")]
    HeapFull(&'static str),
    /// Aftermath crash dumps were asked for and cannot be enabled.
    #[cfg(not(target_vendor = "apple"))]
    #[error("Aftermath: {0}")]
    Aftermath(String),
    /// A Metal call failed.
    #[cfg(target_vendor = "apple")]
    #[error("{call} failed: {message}")]
    Metal { call: &'static str, message: String },
    /// The Metal backend does not implement the call yet.
    #[cfg(target_vendor = "apple")]
    #[error("the Metal backend does not implement {0} yet")]
    NotImplemented(&'static str),
    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
}

/// A Vulkan result code.
#[cfg(not(target_vendor = "apple"))]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VkResult(pub i32);

#[cfg(not(target_vendor = "apple"))]
impl fmt::Display for VkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&ash::vk::Result::from_raw(self.0), f)
    }
}

#[cfg(not(target_vendor = "apple"))]
impl fmt::Debug for VkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(not(target_vendor = "apple"))]
impl Error {
    pub(crate) fn vk(call: &'static str) -> impl FnOnce(ash::vk::Result) -> Error {
        move |r| match r {
            ash::vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
            ash::vk::Result::TIMEOUT => Error::Timeout,
            ash::vk::Result::PIPELINE_COMPILE_REQUIRED => Error::CompileRequired,
            r => Error::Vulkan {
                call,
                result: VkResult(r.as_raw()),
            },
        }
    }
}
