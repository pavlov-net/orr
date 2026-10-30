//! A plain Win32 window: a borderless popup covering the primary monitor, or a 1280x720
//! captioned window. The thread that creates it pumps its messages.

use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{GetMonitorInfoW, MONITOR_DEFAULTTOPRIMARY, MONITORINFO, MonitorFromWindow};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::HiDpi::{DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub struct Window {
    pub hwnd: HWND,
    pub hinstance: isize,
}

static CLOSED: AtomicBool = AtomicBool::new(false);

unsafe extern "system" fn proc_(hwnd: HWND, msg: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    unsafe {
        match msg {
            WM_CLOSE | WM_DESTROY => {
                CLOSED.store(true, Ordering::SeqCst);
                0
            }
            WM_KEYDOWN if w == 0x1B => {
                CLOSED.store(true, Ordering::SeqCst);
                0
            }
            _ => DefWindowProcW(hwnd, msg, w, l),
        }
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

impl Window {
    pub fn new(fullscreen: bool) -> Window {
        unsafe {
            SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            let hinstance = GetModuleHandleW(std::ptr::null());
            let class = wide("orr-present-probe");
            let wc = WNDCLASSW {
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(proc_),
                hInstance: hinstance,
                lpszClassName: class.as_ptr(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                ..std::mem::zeroed()
            };
            RegisterClassW(&wc);
            let title = wide("present probe");
            let (style, x, y, w, h) = if fullscreen { (WS_POPUP | WS_VISIBLE, 0, 0, 640, 480) } else { (WS_OVERLAPPEDWINDOW | WS_VISIBLE, 100, 100, 1280, 720) };
            let ex = if fullscreen { WS_EX_TOPMOST } else { 0 };
            let hwnd = CreateWindowExW(ex, class.as_ptr(), title.as_ptr(), style, x, y, w, h, std::ptr::null_mut(), std::ptr::null_mut(), hinstance, std::ptr::null());
            assert!(!hwnd.is_null(), "CreateWindowExW");
            let mut extent = [1280, 720];
            if fullscreen {
                let mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);
                let mut mi = MONITORINFO { cbSize: std::mem::size_of::<MONITORINFO>() as u32, ..std::mem::zeroed() };
                GetMonitorInfoW(mon, &mut mi);
                let r = mi.rcMonitor;
                SetWindowPos(hwnd, HWND_TOP, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_SHOWWINDOW | SWP_FRAMECHANGED);
            }
            ShowWindow(hwnd, SW_SHOW);
            // Focus-stealing prevention blocks a background process; attaching to the foreground
            // thread's input lets the window take the foreground.
            let fg = GetForegroundWindow();
            let fg_thread = GetWindowThreadProcessId(fg, std::ptr::null_mut());
            let me = windows_sys::Win32::System::Threading::GetCurrentThreadId();
            windows_sys::Win32::System::Threading::AttachThreadInput(me, fg_thread, 1);
            BringWindowToTop(hwnd);
            SetForegroundWindow(hwnd);
            windows_sys::Win32::System::Threading::AttachThreadInput(me, fg_thread, 0);
            let mut rc = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            GetClientRect(hwnd, &mut rc);
            if rc.right > 0 {
                extent = [(rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32];
            }
            let mut win = Window { hwnd, hinstance: hinstance as isize };
            win.pump();
            crate::row("window", format!("{}x{} fullscreen {fullscreen}, foreground {}", extent[0], extent[1], GetForegroundWindow() == hwnd));
            win
        }
    }

    /// Handles pending messages; false once the window is closed or Escape is pressed.
    pub fn pump(&mut self) -> bool {
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        !CLOSED.load(Ordering::SeqCst)
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            DestroyWindow(self.hwnd);
        }
        self.pump();
        CLOSED.store(false, Ordering::SeqCst);
    }
}

/// QueryPerformanceCounter in seconds.
pub fn qpc() -> f64 {
    use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    let (mut c, mut f) = (0i64, 0i64);
    unsafe {
        QueryPerformanceCounter(&mut c);
        QueryPerformanceFrequency(&mut f);
    }
    c as f64 / f as f64
}

/// Raw QueryPerformanceCounter ticks and frequency.
pub fn qpc_raw() -> (i64, i64) {
    use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    let (mut c, mut f) = (0i64, 0i64);
    unsafe {
        QueryPerformanceCounter(&mut c);
        QueryPerformanceFrequency(&mut f);
    }
    (c, f)
}

/// CPU cycles the calling thread has run (QueryThreadCycleTime).
pub fn cycles() -> u64 {
    let mut c = 0u64;
    unsafe {
        windows_sys::Win32::System::WindowsProgramming::QueryThreadCycleTime(windows_sys::Win32::System::Threading::GetCurrentThread(), &mut c);
    }
    c
}

/// Thread cycles per millisecond, from a 50 ms spin.
pub fn cycles_per_ms() -> f64 {
    let t = std::time::Instant::now();
    let c = cycles();
    while t.elapsed().as_secs_f64() < 0.05 {
        std::hint::spin_loop();
    }
    (cycles() - c) as f64 / (t.elapsed().as_secs_f64() * 1e3)
}

pub fn set_priority_highest(on: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST, THREAD_PRIORITY_NORMAL};
    unsafe {
        SetThreadPriority(GetCurrentThread(), if on { THREAD_PRIORITY_HIGHEST } else { THREAD_PRIORITY_NORMAL });
    }
}
