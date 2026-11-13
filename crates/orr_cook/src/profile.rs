//! Tracy zones, plots and thread names, compiled to nothing without the `tracy` feature.

cfg_select! {
    feature = "tracy" => {
        /// A zone named `$name` that lasts until the value drops, with `$text` (a `&str`) shown
        /// on it when given, and otherwise the name of the thread's cook job when it has one.
        macro_rules! zone {
            ($name:literal) => {
                ::tracy_client::Client::running().map(|c| {
                    let span = c.span(::tracy_client::span_location!($name), 0);
                    $crate::job::with_name(|job| span.emit_text(job));
                    span
                })
            };
            ($name:literal, $text:expr) => {
                ::tracy_client::Client::running().map(|c| {
                    let span = c.span(::tracy_client::span_location!($name), 0);
                    span.emit_text($text);
                    span
                })
            };
        }

        /// Records `$value` (an `f64`) on the plot named `$name`.
        macro_rules! plot {
            ($name:literal, $value:expr) => {
                if let Some(c) = ::tracy_client::Client::running() {
                    c.plot(::tracy_client::plot_name!($name), $value);
                }
            };
        }
    }
    _ => {
        macro_rules! zone {
            ($name:literal $(, $text:expr)?) => {
                $crate::profile::Off
            };
        }

        macro_rules! plot {
            ($name:literal, $value:expr) => {};
        }

        /// What [`zone!`] gives without the `tracy` feature.
        pub(crate) struct Off;

        // A zone ends early with `drop`, with the feature and without.
        impl Drop for Off {
            fn drop(&mut self) {}
        }
    }
}

pub(crate) use {plot, zone};

/// Starts the profiler client; later calls do nothing.
pub(crate) fn start() {
    #[cfg(feature = "tracy")]
    tracy_client::Client::start();
}

/// Names the calling thread in the profiler.
pub(crate) fn name_thread(name: &str) {
    cfg_select! {
        feature = "tracy" => {
            if let Some(c) = tracy_client::Client::running() {
                c.set_thread_name(name);
            }
        }
        _ => {
            let _ = name;
        }
    }
}
