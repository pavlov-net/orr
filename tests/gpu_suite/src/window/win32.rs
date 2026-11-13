//! A plain Win32 window for swapchain tests. The thread that creates it pumps its messages.

use std::io::Error;
use std::num::NonZeroIsize;
use std::sync::atomic::{AtomicBool, Ordering};

use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
};

mod bindings;
use bindings::*;

static CLOSED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    if msg == WM_CLOSE as u32 {
        CLOSED.store(true, Ordering::Relaxed);
        return 0;
    }
    // SAFETY: forwarding the message the system passed in.
    unsafe { DefWindowProcW(hwnd, msg, w, l) }
}

/// A captioned, resizable window, shown behind every other window and without taking focus, so
/// a test run stays out of the way of whatever else is on the screen.
pub struct Window {
    hwnd: HWND,
    hinstance: isize,
}

impl Window {
    /// Opens a window whose client area is about `width` x `height`.
    pub fn new(title: &str, width: i32, height: i32) -> Window {
        let wide = |text: &str| text.encode_utf16().chain([0]).collect::<Vec<u16>>();
        let class = wide("orr_gpu_suite");
        let title = wide(title);
        // SAFETY: plain Win32 calls with valid arguments.
        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            assert!(
                !hinstance.is_null(),
                "GetModuleHandleW: {}",
                Error::last_os_error()
            );
            let wc = WNDCLASSW {
                style: (CS_HREDRAW | CS_VREDRAW) as u32,
                lpfnWndProc: Some(window_proc),
                hInstance: hinstance,
                lpszClassName: class.as_ptr(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                ..Default::default()
            };
            // A second registration fails harmlessly.
            RegisterClassW(&raw const wc);
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                title.as_ptr(),
                WS_OVERLAPPEDWINDOW as u32,
                100,
                100,
                width,
                height,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinstance,
                std::ptr::null(),
            );
            assert!(
                !hwnd.is_null(),
                "CreateWindowExW: {}",
                Error::last_os_error()
            );
            SetWindowPos(
                hwnd,
                HWND_BOTTOM,
                0,
                0,
                0,
                0,
                (SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW) as u32,
            );
            let mut w = Window {
                hwnd,
                hinstance: hinstance as isize,
            };
            w.pump();
            w
        }
    }

    pub fn window_handle(&self) -> RawWindowHandle {
        let mut h = Win32WindowHandle::new(NonZeroIsize::new(self.hwnd as isize).unwrap());
        h.hinstance = NonZeroIsize::new(self.hinstance);
        RawWindowHandle::Win32(h)
    }

    pub fn display_handle(&self) -> RawDisplayHandle {
        RawDisplayHandle::Windows(WindowsDisplayHandle::new())
    }

    /// Handles pending messages; false once the window was asked to close.
    pub fn pump(&mut self) -> bool {
        // SAFETY: the message loop of the thread that owns the window.
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&raw mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE as u32) != 0 {
                TranslateMessage(&raw const msg);
                DispatchMessageW(&raw const msg);
            }
        }
        !CLOSED.load(Ordering::Relaxed)
    }

    /// Resizes the window's outer frame.
    pub fn resize(&mut self, width: i32, height: i32) {
        // SAFETY: the window is live and owned by this thread.
        unsafe {
            SetWindowPos(
                self.hwnd,
                std::ptr::null_mut(),
                0,
                0,
                width,
                height,
                (SWP_NOMOVE | SWP_NOZORDER) as u32,
            );
        }
        self.pump();
    }

    /// The client area in pixels.
    pub fn client_size(&self) -> [u32; 2] {
        let mut rc = RECT::default();
        // SAFETY: the window is live.
        unsafe { GetClientRect(self.hwnd, &raw mut rc) };
        [(rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32]
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // SAFETY: the window is live and owned by this thread.
        unsafe { DestroyWindow(self.hwnd) };
        self.pump();
        CLOSED.store(false, Ordering::Relaxed);
    }
}
