use std::fmt;

/// An orr-gpu failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The Vulkan loader could not be found or loaded.
    #[error("Vulkan loader: {0}")]
    Loader(String),
    /// No device offers everything orr requires. Names the first missing item of the best
    /// candidate.
    #[error("no usable GPU: {device} lacks {missing}")]
    Unsupported { device: String, missing: String },
    /// No Vulkan 1.4 device was found.
    #[error("no Vulkan 1.4 device")]
    NoDevice,
    /// A Vulkan call failed.
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
    /// A shader stage does not fit the call.
    #[error("shader: {0}")]
    Shader(String),
    /// An argument breaks a documented rule.
    #[error("invalid argument: {0}")]
    Invalid(String),
    /// The heap has no free slot.
    #[error("{0} heap is full")]
    HeapFull(&'static str),
    /// A transient arena has no room left for an allocation.
    #[error("transient arena full: {0}")]
    ArenaFull(String),
    /// Aftermath crash dumps were asked for and cannot be enabled.
    #[error("Aftermath: {0}")]
    Aftermath(String),
    /// A file could not be read or written.
    #[error("{}: {source}", path.display())]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
}

/// A Vulkan result code.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VkResult(pub i32);

impl fmt::Display for VkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&ash::vk::Result::from_raw(self.0), f)
    }
}

impl fmt::Debug for VkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Error {
    pub(crate) fn vk(call: &'static str) -> impl FnOnce(ash::vk::Result) -> Error {
        move |r| match r {
            ash::vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
            ash::vk::Result::TIMEOUT => Error::Timeout,
            r => Error::Vulkan {
                call,
                result: VkResult(r.as_raw()),
            },
        }
    }
}
