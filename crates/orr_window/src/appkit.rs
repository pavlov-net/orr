//! The display watch on macOS: reads the `NSScreen` the window's view is on.

use std::ptr::NonNull;

use block2::RcBlock;
use dispatch2::DispatchQueue;
use objc2::MainThreadMarker;
use objc2::rc::{Retained, Weak};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_app_kit::{
    NSApplicationDidChangeScreenParametersNotification, NSScreen, NSView,
    NSWindowDidChangeBackingPropertiesNotification, NSWindowDidChangeScreenNotification,
};
use objc2_foundation::{
    NSNotification, NSNotificationCenter, NSObjectProtocol, NSOperationQueue, NSRunLoop,
    NSRunLoopCommonModes, NSTimer,
};
use raw_window_handle::RawWindowHandle;

use crate::{DisplayFeed, DisplayInfo, Error, Result, steady};

/// macOS reports a screen's range as EDR values relative to its SDR white, which moves with
/// the brightness setting. This is the absolute luminance macOS gives EDR 1.0 when it maps PQ
/// content, so that the feed's nits mean what PQ encodes; macOS doesn't document it.
const EDR_REFERENCE_NITS: f32 = 100.0;

/// The headroom changes with brightness and when content asks for EDR, and neither raises an
/// event, so the watch also reads this often.
const READ_INTERVAL_S: f64 = 1.0;

/// Reads once a second, on moves to another screen, on backing scale changes and when the
/// screen configuration changes. Its timer and observers run on the main run loop.
pub(crate) struct Watch {
    observers: Option<Observers>,
}

/// The read timer and notification observers. Made, run and torn down on the main thread.
struct Observers {
    timer: Retained<NSTimer>,
    tokens: Vec<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
}

// SAFETY: `Observers` is made on the main thread and only touched again by `stop`, which
// `Watch`'s drop runs on the main thread.
unsafe impl Send for Observers {}
// SAFETY: no shared reference reaches its contents.
unsafe impl Sync for Observers {}

impl Observers {
    /// Stops the timer and removes the observers; on the main thread.
    fn stop(self) {
        self.timer.invalidate();
        let center = NSNotificationCenter::defaultCenter();
        for token in &self.tokens {
            // SAFETY: a token `addObserverForName` returned, removed once.
            unsafe { center.removeObserver(token.as_ref()) };
        }
    }
}

impl Drop for Watch {
    /// Stops at once on the main thread; from another thread it queues the stop there and
    /// does not wait, since the main thread may be waiting for this one.
    fn drop(&mut self) {
        if let Some(observers) = self.observers.take() {
            if MainThreadMarker::new().is_some() {
                observers.stop();
            } else {
                DispatchQueue::main().exec_async(move || observers.stop());
            }
        }
    }
}

/// Publishes what the window's screen reports. A view that has gone or has no screen keeps
/// the last values.
#[derive(Clone)]
struct Reader {
    view: Weak<NSView>,
    feed: DisplayFeed,
}

impl Reader {
    fn update(&self) {
        let screen = self
            .view
            .load()
            .and_then(|v| v.window())
            .and_then(|w| w.screen());
        if let Some(screen) = screen {
            self.feed.set(steady(self.feed.latest().1, read(&screen)));
        }
    }
}

impl Watch {
    pub(crate) fn new(
        window: RawWindowHandle,
        hook: Option<fn(u32)>,
    ) -> Result<(Self, DisplayFeed)> {
        let _ = hook;
        let RawWindowHandle::AppKit(handle) = window else {
            return Err(Error::UnsupportedWindow);
        };
        if MainThreadMarker::new().is_none() {
            return Err(Error::NotWindowThread);
        }
        // SAFETY: the caller's handle names a live `NSView`, used here on the main thread.
        let view: &NSView = unsafe { handle.ns_view.cast().as_ref() };
        let feed = DisplayFeed::hdr_off();
        let reader = Reader {
            view: Weak::new(view),
            feed: feed.clone(),
        };
        reader.update();

        let tick = reader.clone();
        let block = RcBlock::new(move |_: NonNull<NSTimer>| tick.update());
        // SAFETY: a repeating timer with a positive interval, which runs `block` on the run
        // loop it is added to.
        let timer =
            unsafe { NSTimer::timerWithTimeInterval_repeats_block(READ_INTERVAL_S, true, &block) };
        // SAFETY: the main run loop, from the main thread; common modes keep the reads going
        // while the user drags or resizes the window.
        unsafe { NSRunLoop::mainRunLoop().addTimer_forMode(&timer, NSRunLoopCommonModes) };

        let center = NSNotificationCenter::defaultCenter();
        let queue = NSOperationQueue::mainQueue();
        let window = view.window();
        let window: Option<&AnyObject> = window.as_deref().map(AsRef::as_ref);
        // SAFETY: AppKit's notification names are constants, set before any code runs.
        let names = unsafe {
            [
                NSWindowDidChangeScreenNotification,
                NSWindowDidChangeBackingPropertiesNotification,
                NSApplicationDidChangeScreenParametersNotification,
            ]
        };
        let observed = [(names[0], window), (names[1], window), (names[2], None)];
        let tokens = observed
            .into_iter()
            .map(|(name, object)| {
                let reader = reader.clone();
                let block = RcBlock::new(move |_: NonNull<NSNotification>| reader.update());
                // SAFETY: the block runs on the main queue only, and is made and released on
                // the main thread, so what it captures never leaves it.
                unsafe {
                    center.addObserverForName_object_queue_usingBlock(
                        Some(name),
                        object,
                        Some(&queue),
                        &block,
                    )
                }
            })
            .collect();
        Ok((
            Self {
                observers: Some(Observers { timer, tokens }),
            },
            feed,
        ))
    }
}

/// What `screen` reports. A screen whose potential headroom is above 1 can show HDR; macOS has
/// no HDR switch to read, and composites HDR content on such screens at any time.
fn read(screen: &NSScreen) -> DisplayInfo {
    let potential = screen.maximumPotentialExtendedDynamicRangeColorComponentValue() as f32;
    let current = screen.maximumExtendedDynamicRangeColorComponentValue() as f32;
    let fps = screen.maximumFramesPerSecond();
    let hdr = potential.is_finite() && potential > 1.0;
    DisplayInfo {
        os_hdr_on: Some(hdr),
        max_nits: (hdr && current.is_finite()).then(|| EDR_REFERENCE_NITS * current.max(1.0)),
        sdr_white_nits: Some(EDR_REFERENCE_NITS),
        refresh_hz: (fps > 0).then_some(fps as f32),
        ..DisplayInfo::default()
    }
}
