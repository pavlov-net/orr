//! A winit window driven by the example's own frame loop: [`Window::pump`] handles pending
//! events between frames. One per process, since winit allows one event loop.

use std::time::Duration;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use winit::window::{WindowAttributes, WindowId};

struct App {
    /// Taken when the event loop first resumes, which is when winit allows a window.
    attributes: Option<WindowAttributes>,
    window: Option<winit::window::Window>,
    closed: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(a) = self.attributes.take() {
            self.window = Some(event_loop.create_window(a).expect("window"));
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested | WindowEvent::Destroyed) {
            self.closed = true;
        }
    }
}

/// A resizable window, shown without taking focus and closed when dropped.
pub struct Window {
    app: App,
    event_loop: EventLoop<()>,
}

impl Window {
    /// Opens a window whose client area is `width` x `height` physical pixels.
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        let event_loop = EventLoop::new().expect("event loop");
        let attributes = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(PhysicalSize::new(width, height))
            .with_active(false);
        let mut w = Window {
            app: App {
                attributes: Some(attributes),
                window: None,
                closed: false,
            },
            event_loop,
        };
        while w.app.window.is_none() {
            assert!(w.pump(), "the window closed while opening");
        }
        w
    }

    /// Handles pending events without waiting; false once the window was closed.
    pub fn pump(&mut self) -> bool {
        let status = self
            .event_loop
            .pump_app_events(Some(Duration::ZERO), &mut self.app);
        !self.app.closed && matches!(status, PumpStatus::Continue)
    }

    /// Asks for a client area of `width` x `height` physical pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let _ = self
            .window()
            .request_inner_size(PhysicalSize::new(width, height));
    }

    pub fn window_handle(&self) -> RawWindowHandle {
        self.window()
            .window_handle()
            .expect("window handle")
            .as_raw()
    }

    pub fn display_handle(&self) -> RawDisplayHandle {
        self.window()
            .display_handle()
            .expect("display handle")
            .as_raw()
    }

    fn window(&self) -> &winit::window::Window {
        self.app.window.as_ref().expect("window")
    }
}
