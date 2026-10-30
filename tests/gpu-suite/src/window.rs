//! A plain Win32 window for swapchain tests. The thread that creates it pumps its messages.

use std::num::NonZeroIsize;
use std::sync::atomic::{AtomicBool, Ordering};

use raw_window_handle::{
    RawDisplayHandle, RawWindowHandle, Win32WindowHandle, WindowsDisplayHandle,
};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CS_HREDRAW, CS_VREDRAW, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    GetClientRect, IDC_ARROW, LoadCursorW, MSG, PM_REMOVE, PeekMessageW, RegisterClassW,
    SW_SHOWNOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, SetWindowPos, ShowWindow, TranslateMessage,
    WINDOW_EX_STYLE, WM_CLOSE, WNDCLASSW, WS_OVERLAPPEDWINDOW,
};
use windows::core::w;

static CLOSED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn window_proc(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    if msg == WM_CLOSE {
        CLOSED.store(true, Ordering::Relaxed);
        return LRESULT(0);
    }
    // SAFETY: forwarding the message the system passed in.
    unsafe { DefWindowProcW(hwnd, msg, w, l) }
}

/// A captioned, resizable window, shown without taking focus.
pub struct Window {
    hwnd: HWND,
    hinstance: isize,
}

impl Window {
    /// Opens a window whose client area is about `width` x `height`.
    pub fn new(title: &str, width: i32, height: i32) -> Window {
        // SAFETY: plain Win32 calls with valid arguments; the class name is static.
        unsafe {
            let hinstance = GetModuleHandleW(None).expect("GetModuleHandleW");
            let class = w!("orr-gpu-suite");
            let wc = WNDCLASSW {
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(window_proc),
                hInstance: hinstance.into(),
                lpszClassName: class,
                hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
                ..Default::default()
            };
            // A second registration fails harmlessly.
            RegisterClassW(&wc);
            let title: Vec<u16> = title.encode_utf16().chain([0]).collect();
            let hwnd = CreateWindowExW(
                WINDOW_EX_STYLE(0),
                class,
                windows::core::PCWSTR(title.as_ptr()),
                WS_OVERLAPPEDWINDOW,
                100,
                100,
                width,
                height,
                None,
                None,
                Some(hinstance.into()),
                None,
            )
            .expect("CreateWindowExW");
            let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
            let mut w = Window {
                hwnd,
                hinstance: hinstance.0 as isize,
            };
            w.pump();
            w
        }
    }

    pub fn window_handle(&self) -> RawWindowHandle {
        let mut h = Win32WindowHandle::new(NonZeroIsize::new(self.hwnd.0 as isize).unwrap());
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
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        !CLOSED.load(Ordering::Relaxed)
    }

    /// Resizes the window's outer frame.
    pub fn resize(&mut self, width: i32, height: i32) {
        // SAFETY: the window is live and owned by this thread.
        unsafe {
            let _ = SetWindowPos(
                self.hwnd,
                None,
                0,
                0,
                width,
                height,
                SWP_NOMOVE | SWP_NOZORDER,
            );
        }
        self.pump();
    }

    /// The client area in pixels.
    pub fn client_size(&self) -> [u32; 2] {
        let mut rc = RECT::default();
        // SAFETY: the window is live.
        let _ = unsafe { GetClientRect(self.hwnd, &mut rc) };
        [(rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32]
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        // SAFETY: the window is live and owned by this thread.
        let _ = unsafe { DestroyWindow(self.hwnd) };
        self.pump();
        CLOSED.store(false, Ordering::Relaxed);
    }
}
