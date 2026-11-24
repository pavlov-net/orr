//! A GPUI scene composited through orr's UI hook over a display calibration pattern.

fn main() {
    use std::time::Instant;

    use gpui::{
        AtlasKey, Bounds, ContentMask, Corners, DevicePixels, Edges, ImageId, PaintedPath,
        PathBuilder, PolychromeSprite, Quad, RenderImageParams, ScaledPixels, Scene,
        linear_color_stop, linear_gradient, point, px, rgba, size,
    };
    use orr::{
        CalibrationPattern, Display, DisplayRequest, FrameDesc, FrameTime, Renderer, RendererDesc,
    };
    use orr_examples::{device, window::Window};

    if !cfg!(any(windows, target_os = "macos")) {
        orr_examples::init_logging();
        log::error!("this example needs Win32 or AppKit presentation");
        return;
    }

    let gpu = device();
    let mut window = Window::new("orr GPUI overlay", 720, 480);
    let desc = RendererDesc::default();
    let mut display = Display::new(
        gpu,
        window.window_handle(),
        window.display_handle(),
        &DisplayRequest::default(),
        desc.frames_in_flight,
    )
    .expect("display");
    let mut renderer = Renderer::new(gpu, desc).expect("renderer");
    let mut painter = orr_gpui::Painter::new(gpu).expect("GPUI painter");

    let pixel = ScaledPixels;
    let clip = ContentMask {
        bounds: Bounds {
            origin: point(pixel(24.0), pixel(24.0)),
            size: size(pixel(272.0), pixel(156.0)),
        },
    };
    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        bounds: clip.bounds,
        content_mask: clip,
        background: linear_gradient(
            120.0,
            linear_color_stop(rgba(0x182948e8), 0.0),
            linear_color_stop(rgba(0x435c8a60), 1.0),
        ),
        corner_radii: Corners::all(pixel(16.0)),
        border_widths: Edges::all(pixel(1.0)),
        border_color: rgba(0xaacaffb0).into(),
        ..Quad::default()
    });

    let mut path = PathBuilder::fill();
    path.move_to(point(px(126.0), px(56.0)));
    path.line_to(point(px(184.0), px(104.0)));
    path.line_to(point(px(126.0), px(152.0)));
    path.close();
    scene.insert_primitive(PaintedPath::new(
        path.build().expect("path"),
        1.0,
        ContentMask {
            bounds: Bounds {
                origin: point(px(24.0), px(24.0)),
                size: size(px(272.0), px(156.0)),
            },
        },
        rgba(0x60e5f4bc).into(),
    ));

    let atlas = painter.sprite_atlas();
    let tile = atlas
        .get_or_insert_with(
            AtlasKey::Image(RenderImageParams {
                image_id: ImageId(0),
                frame_index: 0,
            }),
            &mut || {
                Ok(Some((
                    size(DevicePixels(2), DevicePixels(2)),
                    [
                        180, 235, 255, 240, 255, 160, 80, 190, 255, 160, 80, 190, 180, 235, 255,
                        240,
                    ]
                    .as_slice()
                    .into(),
                )))
            },
        )
        .expect("atlas")
        .expect("sprite");
    scene.insert_primitive(PolychromeSprite {
        order: 0,
        pad: 0,
        tile,
        grayscale: false.into(),
        opacity: 1.0,
        bounds: Bounds {
            origin: point(pixel(48.0), pixel(80.0)),
            size: size(pixel(50.0), pixel(50.0)),
        },
        content_mask: clip,
        corner_radii: Corners::all(pixel(8.0)),
    });
    scene.finish();
    let ui = painter.prepare(&scene).expect("prepare GPUI scene");

    let start = Instant::now();
    let mut previous = 0.0;
    while window.pump() {
        let absolute = start.elapsed().as_secs_f64();
        let mut frame = renderer
            .begin_frame(FrameDesc {
                time: FrameTime {
                    absolute,
                    delta: (absolute - previous) as f32,
                },
                scenes: &[],
                display: Some(&mut display),
            })
            .expect("frame");
        previous = absolute;
        if let Some((output, _)) = frame.display() {
            frame
                .pattern(CalibrationPattern::PaperWhite {
                    nits: output.paper_white_nits,
                })
                .expect("pattern");
            frame
                .ui_only(&mut |pass| {
                    painter
                        .paint(pass.cmd, pass.layer, &ui)
                        .expect("paint GPUI")
                })
                .expect("UI");
        }
        frame.end().expect("end frame");
    }
    display.wait_idle();
}
