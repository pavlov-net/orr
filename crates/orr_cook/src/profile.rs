//! Tracy zones, plots and thread names, compiled to nothing without the `tracy` feature.

/// A zone named `$name` that lasts until the value drops, with `$text` (a `&str`) shown on
/// it when given.
#[cfg(feature = "tracy")]
macro_rules! zone {
    ($name:literal $(, $text:expr)?) => {
        ::tracy_client::Client::running().map(|c| {
            let span = c.span(::tracy_client::span_location!($name), 0);
            $(span.emit_text($text);)?
            span
        })
    };
}

#[cfg(not(feature = "tracy"))]
macro_rules! zone {
    ($name:literal $(, $text:expr)?) => {
        $crate::profile::Off
    };
}

/// Records `$value` (an `f64`) on the plot named `$name`.
#[cfg(feature = "tracy")]
macro_rules! plot {
    ($name:literal, $value:expr) => {
        if let Some(c) = ::tracy_client::Client::running() {
            c.plot(::tracy_client::plot_name!($name), $value);
        }
    };
}

#[cfg(not(feature = "tracy"))]
macro_rules! plot {
    ($name:literal, $value:expr) => {};
}

pub(crate) use {plot, zone};

/// What [`zone!`] gives without the `tracy` feature.
#[cfg(not(feature = "tracy"))]
pub(crate) struct Off;

/// Starts the profiler client; later calls do nothing.
pub(crate) fn start() {
    #[cfg(feature = "tracy")]
    tracy_client::Client::start();
}

/// Names the calling thread in the profiler.
pub(crate) fn name_thread(name: &str) {
    #[cfg(feature = "tracy")]
    if let Some(c) = tracy_client::Client::running() {
        c.set_thread_name(name);
    }
    #[cfg(not(feature = "tracy"))]
    let _ = name;
}
