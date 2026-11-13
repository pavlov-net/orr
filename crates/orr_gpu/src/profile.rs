//! Tracy zones, plots, frame marks and thread names for orr's crates. Without the `tracy` feature
//! none of it emits anything.
//!
//! orr never starts the client. Under tracy-client's default lifetime it runs from process start;
//! under `manual-lifetime` orr records nothing until the program starts it.

#[cfg(feature = "tracy")]
#[doc(hidden)]
pub use tracy_client;

/// A zone named `$name` (a literal) that lasts until the value drops, with `$text` (a `&str`,
/// evaluated only while the client runs) shown on it when given.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! zone {
    ($name:literal) => {{
        let location = $crate::profile::tracy_client::span_location!($name);
        $crate::profile::tracy_client::Client::running().map(|c| c.span(location, 0))
    }};
    ($name:literal, $text:expr) => {{
        let location = $crate::profile::tracy_client::span_location!($name);
        $crate::profile::tracy_client::Client::running().map(|c| {
            let span = c.span(location, 0);
            span.emit_text($text);
            span
        })
    }};
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! zone {
    ($name:literal $(, $text:expr)?) => {
        $crate::profile::Off
    };
}

/// Records `$value` (an `f64`, evaluated only while the client runs) on the plot named `$name`.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! plot {
    ($name:literal, $value:expr) => {
        if let Some(c) = $crate::profile::tracy_client::Client::running() {
            c.plot($crate::profile::tracy_client::plot_name!($name), $value);
        }
    };
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! plot {
    ($name:literal, $value:expr) => {};
}

/// Ends a frame: the profiler's frames without a name, or the secondary frames named `$name`.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! frame_mark {
    () => {
        if let Some(c) = $crate::profile::tracy_client::Client::running() {
            c.frame_mark();
        }
    };
    ($name:literal) => {
        if let Some(c) = $crate::profile::tracy_client::Client::running() {
            c.secondary_frame_mark($crate::profile::tracy_client::frame_name!($name));
        }
    };
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! frame_mark {
    ($($name:literal)?) => {};
}

/// What [`zone!`] gives without the `tracy` feature.
#[cfg(not(feature = "tracy"))]
#[doc(hidden)]
pub struct Off;

// A zone ends early with `drop`, with the feature and without.
#[cfg(not(feature = "tracy"))]
impl Drop for Off {
    fn drop(&mut self) {}
}

/// Names the calling thread `$name` (a `&str`, evaluated only while the client runs) in the
/// profiler.
#[cfg(feature = "tracy")]
#[macro_export]
macro_rules! name_thread {
    ($name:expr) => {
        if let Some(c) = $crate::profile::tracy_client::Client::running() {
            c.set_thread_name($name);
        }
    };
}

#[cfg(not(feature = "tracy"))]
#[macro_export]
macro_rules! name_thread {
    ($name:expr) => {};
}
