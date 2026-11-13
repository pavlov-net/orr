//! Tracy zones, plots, frame marks and thread names for orr's crates, and a GPU zone per pass
//! from the passes' timestamps. Without the `tracy` feature none of it emits anything.
//!
//! orr never starts the client. Under tracy-client's default lifetime it runs from process start;
//! under `manual-lifetime` orr records nothing until the program starts it.

#[cfg(feature = "tracy")]
#[doc(hidden)]
pub use tracy_client;

#[cfg(not(target_vendor = "apple"))]
use crate::{PassResult, QueueKind};

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

/// A device's GPU contexts in the profiler, one per queue, each made when its queue's first passes
/// with CPU times arrive. A context's zones must arrive in time order, so devices alive at once
/// keep to their own; a dropped device's contexts go to later devices, since the profiler allows
/// 255 in a process's life. Metal's passes have no CPU times, so Metal has none.
#[cfg(not(target_vendor = "apple"))]
#[derive(Default)]
pub(crate) struct GpuZones {
    #[cfg(feature = "tracy")]
    contexts: std::sync::Mutex<[gpu::Context; QueueKind::ALL.len()]>,
}

#[cfg(not(target_vendor = "apple"))]
impl GpuZones {
    /// Sends the passes of one completed submit on `kind`'s queue as zones on that queue's
    /// context. A pass without CPU times is left out: its ticks have no place on the profiler's
    /// clock.
    pub(crate) fn passes(&self, kind: QueueKind, passes: &[PassResult]) {
        cfg_select! {
            feature = "tracy" => {
                gpu::passes(&mut self.contexts.lock().unwrap()[kind.index()], kind, passes);
            }
            _ => {
                let _ = (kind, passes);
            }
        }
    }
}

#[cfg(all(feature = "tracy", not(target_vendor = "apple")))]
impl Drop for GpuZones {
    fn drop(&mut self) {
        let contexts = self.contexts.get_mut().unwrap();
        let mut pool = gpu::POOL.lock().unwrap();
        for (free, slot) in pool.free.iter_mut().zip(contexts) {
            if let gpu::Context::Made(c) = std::mem::take(slot) {
                free.push(c);
            }
        }
    }
}

#[cfg(all(feature = "tracy", not(target_vendor = "apple")))]
mod gpu {
    use std::sync::Mutex;
    use std::time::Instant;

    use tracy_client::{Client, GpuContext, GpuContextType};

    use crate::{PassResult, QueueKind};

    /// A queue's context: not yet made, made, or refused because the process has made all the
    /// contexts the profiler allows.
    #[derive(Default)]
    pub(super) enum Context {
        #[default]
        Unmade,
        Made(GpuContext),
        Refused,
    }

    /// The contexts of dropped devices by queue kind, and how many each kind has made.
    pub(super) struct Pool {
        pub(super) free: [Vec<GpuContext>; QueueKind::ALL.len()],
        made: [u32; QueueKind::ALL.len()],
    }

    pub(super) static POOL: Mutex<Pool> = Mutex::new(Pool {
        free: [const { Vec::new() }; QueueKind::ALL.len()],
        made: [0; QueueKind::ALL.len()],
    });

    /// The context clock's reading at each batch's sync: a batch's times are nanoseconds from
    /// the CPU instant of its sync, offset to stay positive for passes that ran before it.
    const SYNCED: i64 = 1 << 40;

    /// Sends `passes` on `slot`, `kind`'s context, which the caller holds for the whole batch.
    pub(super) fn passes(slot: &mut Context, kind: QueueKind, passes: &[PassResult]) {
        if passes.iter().all(|p| p.cpu.is_none()) {
            return;
        }
        let Some(client) = Client::running() else {
            return;
        };
        let name = kind.profile_name();
        if matches!(slot, Context::Unmade) {
            let mut pool = POOL.lock().unwrap();
            *slot = match pool.free[kind.index()].pop() {
                Some(c) => Context::Made(c),
                None => {
                    let made = &mut pool.made[kind.index()];
                    let track = match *made {
                        0 => name.to_owned(),
                        n => format!("{name} {}", n + 1),
                    };
                    // Each batch syncs the clock, so the starting reading places nothing.
                    match client.new_gpu_context(Some(&track), GpuContextType::Vulkan, SYNCED, 1.0)
                    {
                        Ok(c) => {
                            *made += 1;
                            Context::Made(c)
                        }
                        Err(e) => {
                            log::warn!("no GPU zones for the {kind:?} queue: {e}");
                            Context::Refused
                        }
                    }
                }
            };
        }
        let Context::Made(context) = slot else {
            return;
        };
        // The pass times are on the CPU clock already, so the sync ties the context's clock to
        // it at this instant, and drift between the two never builds up.
        let now = Instant::now();
        context.sync_gpu_time(SYNCED);
        let at = |t: Instant| match t.checked_duration_since(now) {
            Some(after) => SYNCED + after.as_nanos() as i64,
            None => SYNCED - now.duration_since(t).as_nanos() as i64,
        };
        for p in passes {
            let Some((begin, end)) = p.cpu else {
                continue;
            };
            let mut span = context
                .span_alloc(p.name, name, "", 0)
                .expect("each span frees its queries before the next is made");
            span.end_zone();
            span.upload_timestamp_start(at(begin));
            span.upload_timestamp_end(at(end));
        }
    }
}
