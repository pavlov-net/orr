//! Needs an interactive desktop session.
#![cfg(windows)]

use std::io::Error;
use std::num::NonZeroIsize;
use std::time::{Duration, Instant};

use orr_window::{DisplayFeed, DisplayWatch};
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};

mod bindings;
use bindings::*;

extern "system" fn window_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    assert_ne!(msg, WM_TIMER as u32, "the watch's timer reached the window");
    // SAFETY: default handling of this window's messages.
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

/// Runs the thread's messages for `time`, as a window's thread does; how many were timers.
fn pump_for(time: Duration) -> u32 {
    let start = Instant::now();
    let mut timers = 0;
    while start.elapsed() < time {
        let mut msg = MSG::default();
        // SAFETY: plain message pumping on this thread with a live `msg`.
        unsafe {
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE as u32) != 0 {
                timers += u32::from(msg.message == WM_TIMER as u32);
                DispatchMessageW(&msg);
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    timers
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain([0]).collect()
}

fn window() -> HWND {
    let class = wide("orr_display_watch_test");
    let title = wide("display watch");
    // SAFETY: plain window creation on this thread.
    let hwnd = unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        assert!(!instance.is_null(), "{}", Error::last_os_error());
        RegisterClassW(&WNDCLASSW {
            lpfnWndProc: Some(window_proc),
            hInstance: instance,
            lpszClassName: class.as_ptr(),
            ..Default::default()
        });
        CreateWindowExW(
            (WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE) as u32,
            class.as_ptr(),
            title.as_ptr(),
            WS_POPUP,
            0,
            0,
            32,
            32,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            instance,
            std::ptr::null(),
        )
    };
    assert!(!hwnd.is_null(), "{}", Error::last_os_error());
    hwnd
}

fn watch(hwnd: HWND) -> (DisplayWatch, DisplayFeed) {
    let handle = Win32WindowHandle::new(NonZeroIsize::new(hwnd as isize).unwrap());
    DisplayWatch::new(RawWindowHandle::Win32(handle)).unwrap()
}

/// Whether the window's read timer fires over a little more than its second.
fn timer_fires() -> bool {
    pump_for(Duration::from_millis(1200)) > 0
}

#[test]
fn reads_the_window_monitor() {
    let hwnd = window();
    let (watching, feed) = watch(hwnd);

    let (count, info) = feed.latest();
    assert!(info.os_hdr_on.is_some());
    assert!(info.max_nits.is_some_and(|n| n > 0.0));
    assert!(info.refresh_hz.is_some_and(|hz| hz >= 23.0));

    // A display mode change re-reads through the subclass; an unchanged display keeps the count.
    // SAFETY: the window belongs to this thread.
    unsafe { SendMessageW(hwnd, WM_DISPLAYCHANGE as u32, 0, 0) };
    assert_eq!(feed.latest(), (count, info));

    // The once-a-second read runs through the window's messages and ends with the watch.
    assert!(timer_fires(), "no read timer");
    assert_eq!(feed.latest(), (count, info));
    drop(watching);
    assert!(!timer_fires(), "the read timer outlives the watch");

    // Dropped on another thread, it stops when this thread next runs its messages.
    let (watching, _feed) = watch(hwnd);
    std::thread::spawn(move || drop(watching)).join().unwrap();
    assert!(!timer_fires(), "the read timer outlives the watch");

    // The window may go first.
    let (watching, _feed) = watch(hwnd);
    // SAFETY: the window is ours.
    let destroyed = unsafe { DestroyWindow(hwnd) };
    assert_ne!(destroyed, 0, "{}", Error::last_os_error());
    drop(watching);
}
