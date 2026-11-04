//! A window presented through `Display`: a row of gray steps
//! from 1/64 of paper white up to the resolved peak, a calibration pattern that cycles every
//! four seconds (peak patch on a 10,000-nit surround, paper white on 18% gray, black patch
//! on black) and egui text naming the resolved display. Toggling Windows HDR, moving the SDR
//! content brightness slider or moving the window to another monitor re-resolves on the
//! next frame and is logged. Runs for the given seconds (default 60) or until closed.
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
    eprintln!("the display example needs gpu_suite's window, which is Windows-only");
}
