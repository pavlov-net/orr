//! Presents a test card to a window, cycling the encoding SDR -> PQ -> scRGB every two
//! seconds (with HDR metadata) and the present mode FIFO <-> IMMEDIATE every second. Logs
//! each switch and whether it recreated the swapchain. Runs for the given seconds (default
//! 12) or until the window closes.
//!
//! Windows only: `gpu_suite::window` is a Win32 window. Elsewhere the example says so and
//! exits.

#[cfg(windows)]
mod windowed;

#[cfg(windows)]
fn main() {
    windowed::main();
}

#[cfg(not(windows))]
#[allow(clippy::print_stderr)]
fn main() {
    eprintln!("the swapchain example needs gpu_suite's window, which is Windows-only");
}
