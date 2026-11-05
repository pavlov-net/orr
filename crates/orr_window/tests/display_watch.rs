//! Needs an interactive desktop session.
#![cfg(windows)]

use std::num::NonZeroIsize;
use std::time::{Duration, Instant};

use orr_window::{DisplayFeed, DisplayWatch};
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, MSG, PM_REMOVE, PeekMessageW,
    RegisterClassW, SendMessageW, WM_DISPLAYCHANGE, WM_TIMER, WNDCLASSW, WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW, WS_POPUP,
};
use windows::core::w;

extern "system" fn window_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    assert_ne!(msg, WM_TIMER, "the watch's timer reached the window");
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
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                timers += u32::from(msg.message == WM_TIMER);
                DispatchMessageW(&msg);
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    timers
}

fn window() -> HWND {
    // SAFETY: plain window creation on this thread.
    unsafe {
        let instance = GetModuleHandleW(None).unwrap();
        let class = w!("orr_display_watch_test");
        RegisterClassW(&WNDCLASSW {
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: class,
            ..Default::default()
        });
        CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            class,
            w!("display watch"),
            WS_POPUP,
            0,
            0,
            32,
            32,
            None,
            None,
            Some(instance.into()),
            None,
        )
        .unwrap()
    }
}

fn watch(hwnd: HWND) -> (DisplayWatch, DisplayFeed) {
    let handle = Win32WindowHandle::new(NonZeroIsize::new(hwnd.0 as isize).unwrap());
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
    unsafe { SendMessageW(hwnd, WM_DISPLAYCHANGE, Some(WPARAM(0)), Some(LPARAM(0))) };
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
    unsafe { DestroyWindow(hwnd).unwrap() };
    drop(watching);
}
