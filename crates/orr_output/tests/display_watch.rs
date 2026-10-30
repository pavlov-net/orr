//! Needs an interactive desktop session.
#![cfg(windows)]

use std::num::NonZeroIsize;

use orr_output::DisplayWatch;
use raw_window_handle::{RawWindowHandle, Win32WindowHandle};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, SendMessageW, WM_DISPLAYCHANGE,
    WNDCLASSW, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_POPUP,
};
use windows::core::w;

extern "system" fn window_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // SAFETY: default handling of this window's messages.
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

#[test]
fn reads_the_window_monitor() {
    // SAFETY: plain window creation on this thread.
    let hwnd = unsafe {
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
    };
    let handle = Win32WindowHandle::new(NonZeroIsize::new(hwnd.0 as isize).unwrap());
    let (watch, feed) = DisplayWatch::new(RawWindowHandle::Win32(handle)).unwrap();

    let (count, info) = feed.latest();
    assert!(info.os_hdr_on.is_some());
    assert!(info.max_nits.is_some_and(|n| n > 0.0));
    assert!(info.refresh_hz.is_some_and(|hz| hz >= 23.0));

    // A display mode change re-reads through the subclass; an unchanged display keeps the count.
    // SAFETY: the window belongs to this thread.
    unsafe { SendMessageW(hwnd, WM_DISPLAYCHANGE, Some(WPARAM(0)), Some(LPARAM(0))) };
    assert_eq!(feed.latest(), (count, info));

    drop(watch);
    // SAFETY: the watch is gone; the window is ours.
    unsafe { DestroyWindow(HWND(hwnd.0)).unwrap() };
}
