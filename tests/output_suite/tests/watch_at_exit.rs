//! Ends a process with a `DisplayWatch`'s thread setup alive on its main thread, whose thread
//! destructors run after the system DLLs have shut down. The loader often hides a fault at that
//! point, so a handler turns one into a failing exit code. Has no test harness, since a test
//! thread is not the main thread. Needs an interactive desktop session.

#[cfg(windows)]
fn main() {
    use windows::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    use windows::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};

    const FAULTED: u32 = 0xfa17;

    unsafe extern "system" fn on_exception(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes a valid record.
        let code: NTSTATUS = unsafe { (*(*info).ExceptionRecord).ExceptionCode };
        if code == EXCEPTION_ACCESS_VIOLATION {
            // SAFETY: ends this process.
            let _ = unsafe { TerminateProcess(GetCurrentProcess(), FAULTED) };
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    if std::env::args().any(|a| a == "--list") {
        return;
    }
    let window = gpu_suite::window::Window::new("orr watch at exit", 320, 200);
    let (watch, _feed) = orr_output::DisplayWatch::new(window.window_handle()).unwrap();
    drop(watch);
    drop(window);
    // The handler sees faults that something goes on to handle, so it starts only here.
    // SAFETY: the handler touches only what the system passes it.
    let handler = unsafe { AddVectoredExceptionHandler(1, Some(on_exception)) };
    assert!(!handler.is_null());
}

#[cfg(not(windows))]
fn main() {}
