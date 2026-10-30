//! A window presented through `Display` with the display watch's feed: a row of grey steps
//! from 1/64 of paper white up to the resolved peak, a calibration pattern that cycles every
//! four seconds (peak patch on a 10,000-nit surround, paper white on 18% grey, black patch
//! on black) and egui text naming the resolved display. Toggling Windows HDR, moving the SDR
//! content brightness slider or moving the window to another monitor re-resolves on the
//! next frame and is logged. Runs for the given seconds (default 60) or until closed.

use std::time::{Duration, Instant};

use egui::{Color32, Context, FullOutput, Pos2, RawInput, Rect, RichText, vec2};
use gpu_suite::window::Window;
use half::f16;
use orr_egui::Painter;
use orr_output::{
    CalibrationPattern, ComposePass, ComposeRoot, ComposeSource, Display, DisplayRequest,
    DisplayWatch, OutputKernels, ResolvedDisplay,
};
use orr_scene::orr_gpu::{
    Allocation, Format, Hazards, MemoryKind, QueueKind, Stages, Texture, TextureDesc, TextureIndex,
    TextureMemory, TextureRegion, TextureUsage, TimelinePoint, ViewDesc,
};
use orr_scene::testing::{assert_clean, device};

const FRAMES_IN_FLIGHT: u32 = 2;

/// The display-linear image (linear Rec.709, 1.0 = paper white) for one display, pattern
/// and extent.
fn image(
    display: &ResolvedDisplay,
    pattern: CalibrationPattern,
    [w, h]: [u32; 2],
) -> Vec<[u16; 4]> {
    let steps = 16;
    let top = ((display.peak_nits / display.paper_white_nits).log2() + 6.0).max(1.0);
    let levels = pattern.levels(display);
    let mut texels = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let v = if y < h / 4 {
                let step = (x * steps / w) as f32;
                (top * step / (steps - 1) as f32 - 6.0).exp2()
            } else {
                let (cx, cy) = (
                    x as f32 / w as f32 - 0.5,
                    (y - h / 4) as f32 / h as f32 - 0.375,
                );
                if cx.abs() < 0.1 && cy.abs() < 0.15 {
                    levels.patch
                } else {
                    levels.surround
                }
            };
            let c = f16::from_f32(v).to_bits();
            texels.push([c, c, c, f16::ONE.to_bits()]);
        }
    }
    texels
}

/// The display-linear image and the upload it was copied from.
struct Linear {
    _texture: Texture,
    index: TextureIndex,
    _upload: Allocation<[u16; 4]>,
    key: (ResolvedDisplay, usize, [u32; 2]),
}

fn main() {
    let seconds: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(60);
    let gpu = device();
    let render = gpu.queue(QueueKind::Render).unwrap();
    let mut window = Window::new("orr display", 1280, 720);
    let (_watch, feed) = DisplayWatch::new(window.window_handle()).unwrap();
    let mut display = Display::new(
        gpu,
        window.window_handle(),
        window.display_handle(),
        &DisplayRequest::default(),
        Some(feed.clone()),
        FRAMES_IN_FLIGHT,
    )
    .unwrap();
    let kernels = OutputKernels::new(gpu).unwrap();
    let mut roots: Vec<ComposeRoot> = (0..FRAMES_IN_FLIGHT)
        .map(|_| ComposeRoot::new(gpu).unwrap())
        .collect();
    let mut painter = Painter::new(gpu, FRAMES_IN_FLIGHT).unwrap();
    let ctx = Context::default();
    let mut linear: Option<Linear> = None;
    let mut last: Option<TimelinePoint> = None;
    let mut logged: Option<ResolvedDisplay> = None;
    let start = Instant::now();
    let mut frame = 0u64;
    while window.pump() && start.elapsed() < Duration::from_secs(seconds) {
        frame += 1;
        let Some(mut f) = display.begin_frame(frame).unwrap() else {
            std::thread::sleep(Duration::from_millis(10));
            continue;
        };
        let mut list = render.command_list().unwrap();
        let (resolved, extent, wait_for) = (f.display, f.extent, f.wait);
        if logged != Some(resolved) {
            log::info!("{resolved:?}; feed {:?}", feed.latest().1);
            logged = Some(resolved);
        }
        let which = (start.elapsed().as_secs() / 4 % 3) as usize;
        let pattern = [
            CalibrationPattern::Peak {
                nits: resolved.peak_nits,
            },
            CalibrationPattern::PaperWhite {
                nits: resolved.paper_white_nits,
            },
            CalibrationPattern::Black {
                nits: resolved.black_nits + 0.05,
            },
        ][which];
        let key = (resolved, which, extent);
        if linear.as_ref().is_none_or(|l| l.key != key) {
            // The old image may still be read by frames in flight.
            if let Some(p) = last {
                render.wait(p, Duration::from_secs(10)).unwrap();
            }
            let texels = image(&resolved, pattern, extent);
            let mut upload = gpu
                .gpu_malloc::<[u16; 4]>(MemoryKind::Upload, texels.len() as u64, "linear upload")
                .unwrap();
            upload.as_mut_slice().unwrap().copy_from_slice(&texels);
            let desc = TextureDesc::d2(
                Format::Rgba16Float,
                extent[0],
                extent[1],
                TextureUsage::SAMPLED,
            );
            let texture = gpu
                .create_texture(&desc, TextureMemory::Dedicated, "display linear")
                .unwrap();
            let index = gpu.write_sampled(
                gpu.alloc_image_slot().unwrap(),
                &texture,
                &ViewDesc::default(),
            );
            list.discard(&texture);
            list.copy_to_texture(
                &texture,
                &TextureRegion::whole_mip(texture.desc(), 0),
                upload.gpu.cast(),
            );
            list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
            if let Some(old) = linear.take() {
                gpu.free_image_slot_after_submitted(old.index.0);
            }
            linear = Some(Linear {
                _texture: texture,
                index,
                _upload: upload,
                key,
            });
        }
        let target = f.backbuffer(&mut list);
        kernels.compose(
            &mut list,
            &mut roots[(frame % u64::from(FRAMES_IN_FLIGHT)) as usize],
            &ComposePass {
                source: ComposeSource::Linear(linear.as_ref().unwrap().index),
                ui: None,
                target,
                extent,
                display: &resolved,
            },
        );

        let info = feed.latest().1;
        let FullOutput {
            textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = ctx.run_ui(
            RawInput {
                screen_rect: Some(Rect::from_min_size(
                    Pos2::ZERO,
                    vec2(extent[0] as f32, extent[1] as f32),
                )),
                ..RawInput::default()
            },
            |ui| {
                ui.add_space(extent[1] as f32 / 4.0 + 8.0);
                let text = format!(
                    "{:?}  paper white {:.0}  peak {:.0}  black {:.3} nits\n\
                     OS: HDR {:?}, SDR white {:?}, max {:?}, min {:?}, {:?} Hz\n\
                     pattern {pattern:?}",
                    resolved.encoding,
                    resolved.paper_white_nits,
                    resolved.peak_nits,
                    resolved.black_nits,
                    info.os_hdr_on,
                    info.sdr_white_nits,
                    info.max_nits,
                    info.min_nits,
                    info.refresh_hz,
                );
                ui.label(RichText::new(text).size(18.0).color(Color32::WHITE));
            },
        );
        let primitives = ctx.tessellate(shapes, pixels_per_point);
        painter
            .update_textures(&mut list, frame, textures_delta)
            .unwrap();
        let layer = f.ui_layer(&mut list).texture();
        painter
            .paint(&mut list, frame, layer, &primitives, pixels_per_point)
            .unwrap();

        let point = render.submit(vec![list], wait_for.as_slice()).unwrap();
        display.present(point).unwrap();
        last = Some(point);
        gpu.collect_garbage();
    }
    drop(display);
    if let Some(p) = last {
        render.wait(p, Duration::from_secs(10)).unwrap();
    }
    if let Some(l) = linear {
        gpu.free_image_slot_after_submitted(l.index.0);
    }
    drop(painter);
    gpu.collect_garbage();
    assert_clean();
    log::info!("{frame} frames, validation clean");
}
