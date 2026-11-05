//! A window for swapchain tests, on the platforms that have one. Each has the same calls.

#[cfg(target_os = "macos")]
mod appkit;
#[cfg(windows)]
mod win32;

#[cfg(target_os = "macos")]
pub use appkit::Window;
#[cfg(windows)]
pub use win32::Window;
