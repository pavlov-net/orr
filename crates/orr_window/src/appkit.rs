//! The display watch on macOS: reads the `NSScreen` the window's view is on.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use block2::RcBlock;
use dispatch2::DispatchQueue;
use objc2::MainThreadMarker;
use objc2::rc::{Retained, Weak};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2_app_kit::{
    NSApplicationDidChangeScreenParametersNotification, NSScreen, NSView,
    NSWindowDidBecomeKeyNotification, NSWindowDidChangeBackingPropertiesNotification,
    NSWindowDidChangeScreenNotification,
};
use objc2_foundation::{
    NSNotification, NSNotificationCenter, NSObjectProtocol, NSOperationQueue, NSRunLoop,
    NSRunLoopCommonModes, NSTimer,
};
use raw_window_handle::RawWindowHandle;

use crate::{DisplayFeed, DisplayHeadroom, DisplayInfo, Error, Result, steady};

/// The headroom changes with brightness and when content asks for EDR, without an event for
/// each step, so the watch also reads this often. Frames on an HDR surface read for themselves
/// through [`DisplayFeed::refresh`].
const READ_INTERVAL_S: f64 = 1.0;

/// Reads once a second, on moves to another screen, on backing scale changes, on gaining focus,
/// when the screen configuration changes and when a frame asks. Its timer and observers run on
/// the main run loop.
pub(crate) struct Watch {
    observers: Option<Observers>,
    _frame_read: Arc<dyn Fn() + Send + Sync>,
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

/// A read for a frame, which may be asked for on any thread.
struct FrameRead {
    reader: Reader,
    queued: AtomicBool,
}

// SAFETY: `reader`'s view is only loaded on the main thread; other threads only drop the weak
// reference, which the runtime makes atomic. Its feed is `Send + Sync`.
unsafe impl Send for FrameRead {}
// SAFETY: as above.
unsafe impl Sync for FrameRead {}

impl FrameRead {
    /// Reads now on the main thread. Elsewhere it queues one read there and doesn't wait, since
    /// the main thread may be waiting for this one.
    fn request(this: &Arc<Self>) {
        if MainThreadMarker::new().is_some() {
            this.reader.update();
        } else if !this.queued.swap(true, Ordering::AcqRel) {
            let this = Arc::clone(this);
            DispatchQueue::main().exec_async(move || {
                this.queued.store(false, Ordering::Release);
                this.reader.update();
            });
        }
    }
}

impl Watch {
    pub(crate) fn new(window: RawWindowHandle) -> Result<(Self, DisplayFeed)> {
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

        let frame_read = Arc::new(FrameRead {
            reader: reader.clone(),
            queued: AtomicBool::new(false),
        });
        let read: Arc<dyn Fn() + Send + Sync> = Arc::new(move || FrameRead::request(&frame_read));
        feed.set_refresh(Arc::downgrade(&read));

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
                NSWindowDidBecomeKeyNotification,
                NSApplicationDidChangeScreenParametersNotification,
            ]
        };
        let observed = [
            (names[0], window),
            (names[1], window),
            (names[2], window),
            (names[3], None),
        ];
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
                _frame_read: read,
            },
            feed,
        ))
    }
}

/// What `screen` reports. HDR is on when the potential headroom is above 1: built-in XDR
/// screens always, external HDR screens with their High Dynamic Range setting on.
fn read(screen: &NSScreen) -> DisplayInfo {
    // Anything but a number is no headroom: an infinity would reach the tone curve as its peak.
    let finite = |v: f64| Some(v as f32).filter(|v| v.is_finite());
    let potential =
        finite(screen.maximumPotentialExtendedDynamicRangeColorComponentValue()).unwrap_or(1.0);
    let current = finite(screen.maximumExtendedDynamicRangeColorComponentValue()).unwrap_or(1.0);
    let reference = finite(screen.maximumReferenceExtendedDynamicRangeColorComponentValue());
    let fps = screen.maximumFramesPerSecond();
    DisplayInfo {
        os_hdr_on: Some(potential > 1.0),
        headroom: Some(DisplayHeadroom {
            current,
            potential,
            // 0 on screens without one.
            reference: reference.filter(|r| *r > 0.0),
        }),
        refresh_hz: (fps > 0).then_some(fps as f32),
        ..DisplayInfo::default()
    }
}
