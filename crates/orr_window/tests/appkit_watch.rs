//! The display watch on a macOS window. AppKit windows live on the main thread, which libtest
//! never runs tests on, so this target has no harness. Needs a logged-in desktop session.

#[cfg(target_os = "macos")]
fn main() {
    use std::ptr::NonNull;
    use std::time::Duration;

    use objc2::MainThreadMarker;
    use objc2_app_kit::{
        NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSWindow,
        NSWindowStyleMask,
    };
    use objc2_foundation::{NSDate, NSPoint, NSRect, NSRunLoop, NSSize};
    use orr_window::DisplayWatch;
    use raw_window_handle::{AppKitWindowHandle, RawWindowHandle};

    /// Runs the main run loop for `time`, as a window's thread does.
    fn pump_for(time: Duration) {
        let until = NSDate::dateWithTimeIntervalSinceNow(time.as_secs_f64());
        NSRunLoop::currentRunLoop().runUntilDate(&until);
    }

    let mtm = MainThreadMarker::new().expect("the test runs on the main thread");
    let app = NSApplication::sharedApplication(mtm);
    app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
    let rect = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(32.0, 32.0));
    // SAFETY: a plain borderless window, made and used on the main thread.
    let window = unsafe {
        NSWindow::initWithContentRect_styleMask_backing_defer(
            mtm.alloc(),
            rect,
            NSWindowStyleMask::Borderless,
            NSBackingStoreType::Buffered,
            false,
        )
    };
    // SAFETY: the window stays alive until this test ends, and is closed explicitly.
    unsafe { window.setReleasedWhenClosed(false) };
    window.orderFrontRegardless();
    let view = window.contentView().expect("a window has a content view");
    let handle = RawWindowHandle::AppKit(AppKitWindowHandle::new(NonNull::from(&*view).cast()));

    let (watching, feed) = DisplayWatch::new(handle).unwrap();
    let (_, info) = feed.latest();
    assert!(info.os_hdr_on.is_some());
    assert!(info.headroom.is_some());
    assert_eq!((info.sdr_white_nits, info.max_nits), (None, None));
    assert!(info.refresh_hz.is_some_and(|hz| hz >= 23.0));

    /// A window handle sent to another thread, which only tries to watch it.
    struct Sent(RawWindowHandle);
    // SAFETY: the other thread passes it to `DisplayWatch::new`, which checks its thread
    // before touching the view.
    unsafe impl Send for Sent {}

    // Off the main thread there is no watch; dropped there, it stops on the main thread.
    let sent = Sent(handle);
    let made_elsewhere = std::thread::spawn(move || {
        let sent = sent;
        DisplayWatch::new(sent.0).err()
    })
    .join();
    assert!(matches!(
        made_elsewhere.unwrap(),
        Some(orr_window::Error::NotWindowThread)
    ));
    pump_for(Duration::from_millis(1200));
    std::thread::spawn(move || drop(watching)).join().unwrap();
    pump_for(Duration::from_millis(100));

    // The window may go first.
    let (watching, _feed) = DisplayWatch::new(handle).unwrap();
    window.close();
    pump_for(Duration::from_millis(1200));
    drop(watching);
}

#[cfg(not(target_os = "macos"))]
fn main() {}
