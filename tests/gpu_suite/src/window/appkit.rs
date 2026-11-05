//! A plain AppKit window for swapchain tests. AppKit windows live on the main thread, which
//! creates it and pumps its events; libtest runs tests on other threads, so the targets that
//! use it have no harness.

use std::ptr::NonNull;

use objc2::MainThreadMarker;
use objc2::rc::Retained;
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEventMask, NSView,
    NSWindow, NSWindowStyleMask,
};
use objc2_foundation::{NSDate, NSDefaultRunLoopMode, NSPoint, NSRect, NSSize, NSString};
use raw_window_handle::{
    AppKitDisplayHandle, AppKitWindowHandle, RawDisplayHandle, RawWindowHandle,
};

/// A titled, resizable window, ordered behind the app's other windows, from an app without a
/// Dock icon that never activates, so a test run stays out of the way.
pub struct Window {
    window: Retained<NSWindow>,
    view: Retained<NSView>,
    mtm: MainThreadMarker,
}

impl Window {
    /// Opens a window whose content area is `width` x `height` points.
    pub fn new(title: &str, width: i32, height: i32) -> Window {
        let mtm = MainThreadMarker::new().expect(
            "AppKit windows are made on the main thread; run this target without a harness",
        );
        let app = NSApplication::sharedApplication(mtm);
        if app.activationPolicy() != NSApplicationActivationPolicy::Accessory {
            app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
            app.finishLaunching();
        }
        let rect = NSRect::new(
            NSPoint::new(100.0, 100.0),
            NSSize::new(f64::from(width), f64::from(height)),
        );
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable;
        // SAFETY: a plain window, made and used on the main thread.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                mtm.alloc(),
                rect,
                style,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // SAFETY: `Window` holds it until it drops, then closes it.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setTitle(&NSString::from_str(title));
        window.orderBack(None);
        let view = window.contentView().expect("a window has a content view");
        let mut w = Window { window, view, mtm };
        w.pump();
        w
    }

    pub fn window_handle(&self) -> RawWindowHandle {
        RawWindowHandle::AppKit(AppKitWindowHandle::new(NonNull::from(&*self.view).cast()))
    }

    pub fn display_handle(&self) -> RawDisplayHandle {
        RawDisplayHandle::AppKit(AppKitDisplayHandle::new())
    }

    /// Handles pending events; false once the window was closed.
    pub fn pump(&mut self) -> bool {
        let app = NSApplication::sharedApplication(self.mtm);
        let past = NSDate::distantPast();
        loop {
            // SAFETY: the default run loop mode is a constant.
            let mode = unsafe { NSDefaultRunLoopMode };
            let Some(event) = app.nextEventMatchingMask_untilDate_inMode_dequeue(
                NSEventMask::Any,
                Some(&past),
                mode,
                true,
            ) else {
                break;
            };
            app.sendEvent(&event);
        }
        app.updateWindows();
        self.window.isVisible()
    }

    /// Resizes the content area to `width` x `height` points.
    pub fn resize(&mut self, width: i32, height: i32) {
        self.window
            .setContentSize(NSSize::new(f64::from(width), f64::from(height)));
        self.pump();
    }

    /// The content area in pixels.
    pub fn client_size(&self) -> [u32; 2] {
        let size = self.view.bounds().size;
        let scale = self.window.backingScaleFactor();
        [
            (size.width * scale).round() as u32,
            (size.height * scale).round() as u32,
        ]
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        self.window.close();
        self.pump();
    }
}
