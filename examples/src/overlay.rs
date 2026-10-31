//! A text readout drawn over the view through egui, on Windows; nothing elsewhere.

use orr::{FrameTime, UiPass};

#[cfg(windows)]
pub struct Overlay {
    ctx: egui::Context,
    painter: orr_egui::Painter,
    pending: egui::TexturesDelta,
    primitives: Vec<egui::ClippedPrimitive>,
    pixels_per_point: f32,
}

#[cfg(windows)]
impl Overlay {
    pub fn new(frames_in_flight: u32) -> Self {
        Self {
            ctx: egui::Context::default(),
            painter: orr_egui::Painter::new(crate::device(), frames_in_flight).expect("painter"),
            pending: egui::TexturesDelta::default(),
            primitives: Vec::new(),
            pixels_per_point: 1.0,
        }
    }

    /// Lays `text` out in the top-left corner of a `w` x `h` window.
    pub fn layout(&mut self, [w, h]: [u32; 2], time: FrameTime, text: &str) {
        use egui::{Color32, Pos2, RawInput, Rect, RichText, vec2};
        let out = self.ctx.run_ui(
            RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(w as f32, h as f32))),
                time: Some(time.absolute),
                predicted_dt: time.delta,
                ..RawInput::default()
            },
            |ui| {
                egui::Frame::new()
                    .fill(Color32::from_black_alpha(200))
                    .inner_margin(8)
                    .show(ui, |ui| {
                        ui.label(RichText::new(text).size(16.0).color(Color32::WHITE));
                    });
            },
        );
        self.primitives = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        self.pixels_per_point = out.pixels_per_point;
        self.pending.append(out.textures_delta);
    }

    /// Draws a white dot centered at `x`, 24 px above the bottom of the layer.
    pub fn paint_cursor(&mut self, pass: &mut UiPass<'_, '_>, frame: u64, x: f32) {
        use egui::epaint::ClippedShape;
        use egui::{Color32, Pos2, Rect, Shape};
        let [_, h, _] = pass.layer.desc().extent;
        let center = Pos2::new(x, h as f32 / self.pixels_per_point - 24.0);
        let dot = self.ctx.tessellate(
            vec![ClippedShape::new(
                Rect::EVERYTHING,
                Shape::circle_filled(center, 10.0, Color32::WHITE),
            )],
            self.pixels_per_point,
        );
        self.painter
            .paint(pass.cmd, frame, pass.layer, &dot, self.pixels_per_point)
            .expect("egui paint");
    }

    /// Draws the last layout into the frame's UI layer.
    pub fn paint(&mut self, pass: &mut UiPass<'_, '_>, frame: u64) {
        self.painter
            .update_textures(pass.cmd, frame, std::mem::take(&mut self.pending))
            .expect("egui textures");
        self.painter
            .paint(
                pass.cmd,
                frame,
                pass.layer,
                &self.primitives,
                self.pixels_per_point,
            )
            .expect("egui paint");
    }
}

#[cfg(not(windows))]
pub struct Overlay;

#[cfg(not(windows))]
impl Overlay {
    pub fn new(_: u32) -> Self {
        Self
    }

    pub fn layout(&mut self, _: [u32; 2], _: FrameTime, _: &str) {}

    pub fn paint(&mut self, _: &mut UiPass<'_, '_>, _: u64) {}

    pub fn paint_cursor(&mut self, _: &mut UiPass<'_, '_>, _: u64, _: f32) {}
}
