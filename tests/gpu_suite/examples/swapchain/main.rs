//! Presents a test card to a window, cycling through the encodings the surface offers (SDR,
//! PQ, scRGB) every two seconds (with HDR metadata) and the present mode FIFO <-> IMMEDIATE every second. Logs
//! each switch and whether it recreated the swapchain. Runs for the given seconds (default
//! 12) or until the window closes.
//!
//! On Windows and macOS, the platforms with a `gpu_suite::window`. Elsewhere the example says
//! so and exits.

#[cfg(any(windows, target_os = "macos"))]
mod windowed;

#[cfg(any(windows, target_os = "macos"))]
fn main() {
    windowed::main();
}

#[cfg(not(any(windows, target_os = "macos")))]
#[allow(clippy::print_stderr)]
fn main() {
    eprintln!("the swapchain example needs gpu_suite's window, which Windows and macOS have");
}
