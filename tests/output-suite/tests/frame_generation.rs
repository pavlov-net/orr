//! Frame generation through `orr::Renderer` and a `Display` on a real window: a bar crossing
//! the window, shaded by a test kernel and shown with `Frame::compose`, with the egui demo
//! windows over it. The display runs 2x, then 4x where NGX allows it, then off, `Dynamic`
//! on a PQ request, 2x on a scRGB request and off again. Each frame presents its generated
//! images before itself, scRGB presents as PQ while frame generation is on, every frame id
//! keeps one complete latency report, and the validation layer stays silent.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use egui::{Context, FullOutput, Pos2, RawInput, Rect, TexturesDelta, vec2};
use gpu_suite::check_latency_reports;
use gpu_suite::window::Window;
use orr::orr_gpu::{
    Device, DeviceDesc, Format, MemoryKind, QueueKind, QueueRequest, SpecData, StorageImageIndex,
    Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage, ViewDesc as ImageView,
};
use orr::{
    ConsumerMarker, Display, DisplayRequest, DlssSdk, FrameDesc, FrameGenerationMode, FrameTime,
    Lens, Projection, Renderer, RendererDesc, SceneId, ViewDesc, ViewHistoryId, ViewTarget,
};
use orr_egui::Painter;
use orr_math::{DVec3, Quat, UVec2, Vec2};
use orr_output::OutputFormat;
use orr_output::orr_dlss::Uuid;
use output_suite::shaders::moving_bar;

const PROJECT_ID: Uuid = Uuid::from_u128(0x2f6b_91c4_5a0e_4d3b_b7e2_0c18_d94a_6e53);
/// Frames per phase.
const PHASE: u64 = 160;
const PHASES: u64 = 6;
/// Frames into a phase before its generated count is checked: the feature and its history
/// settle first.
const SETTLE: u64 = 16;
const BAR_WIDTH: f32 = 64.0;
/// Pixels per rendered frame.
const SPEED: f32 = 6.0;

/// The bar's images at one extent, as `Arc`s frame generation can keep for the frame.
struct Targets {
    extent: [u32; 2],
    color: Texture,
    color_sampled: TextureIndex,
    color_store: StorageImageIndex,
    depth: Arc<Texture>,
    depth_store: StorageImageIndex,
    motion: Arc<Texture>,
    motion_store: StorageImageIndex,
}

impl Targets {
    fn new(gpu: &Device, extent: [u32; 2]) -> Self {
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        let texture = |format, name| {
            gpu.create_texture(
                &TextureDesc::d2(format, extent[0], extent[1], usage),
                TextureMemory::Dedicated,
                name,
            )
            .unwrap()
        };
        let view = ImageView::default();
        let store = |t: &Texture| gpu.write_storage(gpu.alloc_image_slot().unwrap(), t, &view);
        let color = texture(Format::Rgba16Float, "bar color");
        let depth = Arc::new(texture(Format::R32Float, "bar depth"));
        let motion = Arc::new(texture(Format::Rg16Float, "bar motion"));
        Self {
            extent,
            color_sampled: gpu.write_sampled(gpu.alloc_image_slot().unwrap(), &color, &view),
            color_store: store(&color),
            depth_store: store(&depth),
            motion_store: store(&motion),
            color,
            depth,
            motion,
        }
    }
}

/// The mode, request and expected encoding of each phase; `None` for the request's own.
fn phase(
    p: u64,
    most: u32,
    recommended: FrameGenerationMode,
) -> (FrameGenerationMode, OutputFormat, OutputFormat) {
    use FrameGenerationMode::{Fixed, Off};
    use OutputFormat::{Hdr10Pq, ScRgb, Sdr};
    match p {
        0 => (Fixed(1), Sdr, Sdr),
        1 => (Fixed(most), Sdr, Sdr),
        2 => (Off, Sdr, Sdr),
        3 => (recommended, Hdr10Pq, Hdr10Pq),
        4 => (Fixed(1), ScRgb, Hdr10Pq),
        _ => (Off, ScRgb, ScRgb),
    }
}

#[test]
fn frame_generation_presents_generated_images_before_each_frame() {
    gpu_suite::log_to_stderr();
    let hooks = DlssSdk::device_hooks(PROJECT_ID);
    let gpu = Device::new(&DeviceDesc {
        name: "output-suite frame generation",
        hooks: Some(&hooks),
        validation: true,
        queues: QueueRequest {
            render_compute: true,
            async_compute: true,
            copy: true,
            present: true,
        },
        ..DeviceDesc::default()
    })
    .unwrap();
    let sdk = DlssSdk::new(&gpu, PROJECT_ID).unwrap();
    let support = sdk.support();
    if !support.fg {
        log::warn!("skipped: NGX offers no frame generation here");
        return;
    }
    let mut window = Window::new("orr frame generation", 1280, 720);
    let desc = RendererDesc::default();
    let request = |encoding| DisplayRequest {
        encoding: Some(encoding),
        vsync: false,
        ..DisplayRequest::default()
    };
    let mut display = Display::new(
        &gpu,
        window.window_handle(),
        window.display_handle(),
        &request(OutputFormat::Sdr),
        None,
        desc.frames_in_flight,
    )
    .unwrap();
    if let Err(e) = display.set_frame_generation(&sdk, FrameGenerationMode::Fixed(1)) {
        log::warn!("skipped: {e}");
        return;
    }
    let most = support.max_generated.min(3);
    let recommended = FrameGenerationMode::recommended(&support);

    let mut renderer = Renderer::new(&gpu, desc).unwrap();
    let bar = renderer
        .compute_pipeline(&moving_bar::MODULE, "main", SpecData::default())
        .unwrap();
    let mut roots = gpu
        .gpu_malloc::<moving_bar::Root>(
            MemoryKind::Upload,
            u64::from(desc.frames_in_flight),
            "bar roots",
        )
        .unwrap();
    let mut painter = Painter::new(&gpu, desc.frames_in_flight).unwrap();
    let ctx = Context::default();
    let mut demo = egui_demo_lib::DemoWindows::default();
    let mut pending = TexturesDelta::default();
    let mut targets: Option<Targets> = None;
    let mut reports = BTreeMap::new();
    let mut timings = Vec::new();
    let mut last_present_id = 0;
    let mut last = None;
    let mut checked = 0;

    for i in 0..PHASE * PHASES {
        window.pump();
        let (mode, encoding, expected) = phase(i / PHASE, most, recommended);
        if i % PHASE == 0 {
            display.set_request(&request(encoding));
            display.set_frame_generation(&sdk, mode).unwrap();
        }
        let time = FrameTime {
            absolute: i as f64 / 60.0,
            delta: 1.0 / 60.0,
        };
        let mut f = renderer
            .begin_frame(FrameDesc {
                time,
                scenes: &[],
                display: Some(&mut display),
            })
            .unwrap();
        let id = f.id();
        f.latency_marker(ConsumerMarker::SimulationStart);
        f.latency_marker(ConsumerMarker::SimulationEnd);
        let Some((_, extent)) = f.display() else {
            last = Some((id, f.end().unwrap()));
            continue;
        };
        if targets.as_ref().is_none_or(|t| t.extent != extent) {
            let t = Targets::new(&gpu, extent);
            let list = f.list().unwrap();
            for tex in [&t.color, &*t.depth, &*t.motion] {
                list.discard(tex);
            }
            targets = Some(t);
        }
        let t = targets.as_ref().unwrap();
        let travel = extent[0] as f32 - BAR_WIDTH;
        let slot = (id % u64::from(desc.frames_in_flight)) as usize;
        roots.as_mut_slice().unwrap()[slot] = moving_bar::Root {
            color: t.color_store,
            depth: t.depth_store,
            motion: t.motion_store,
            width: extent[0],
            height: extent[1],
            bar_left: (i as f32 * SPEED) % travel,
            bar_width: BAR_WIDTH,
            motion_u: -SPEED / extent[0] as f32,
        };
        let pipeline = f.pipeline(bar);
        let list = f.list().unwrap();
        list.begin_pass("moving_bar", false);
        list.dispatch(
            pipeline,
            roots.gpu.offset(slot as u64),
            [extent[0].div_ceil(8), extent[1].div_ceil(8), 1],
        );
        list.end_pass();
        f.compose(t.color_sampled).unwrap();
        let view = ViewDesc {
            scene: SceneId(0),
            history: ViewHistoryId(0),
            position: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            projection: Projection {
                fov_y: 1.0,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::from(extent),
            jitter: Vec2::ZERO,
            // The bar jumps back when it wraps.
            reset_history: i % PHASE == 0 || (i as f32 * SPEED) % travel < SPEED,
            mip_bias: 0.0,
            target: ViewTarget::Display,
        };
        f.frame_generation(&view, &t.depth, &t.motion, false);

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
                time: Some(time.absolute),
                predicted_dt: time.delta,
                ..RawInput::default()
            },
            |ui| demo.ui(ui),
        );
        let primitives = ctx.tessellate(shapes, pixels_per_point);
        pending.append(textures_delta);
        f.ui_only(&mut |pass| {
            painter
                .update_textures(pass.cmd, id, std::mem::take(&mut pending))
                .unwrap();
            painter
                .paint(pass.cmd, id, pass.layer, &primitives, pixels_per_point)
                .unwrap();
        })
        .unwrap();
        last = Some((id, f.end().unwrap()));

        if id.is_multiple_of(16) {
            timings.clear();
            display.latency().unwrap().timings(&mut timings);
            for r in &timings {
                reports.insert(r.frame_id, *r);
            }
        }
        if i % 8 == 7 && i % PHASE >= SETTLE {
            display.wait_idle();
            let p = display.last_presented().unwrap();
            assert_eq!(p.frame, id);
            let present_id = p.present_id.expect("presented");
            assert!(
                present_id > last_present_id + u64::from(p.generated),
                "the frame follows its generated images"
            );
            last_present_id = present_id;
            assert_eq!(p.encoding, expected, "frame {id}");
            match mode {
                FrameGenerationMode::Off => assert_eq!(p.generated, 0, "frame {id}"),
                FrameGenerationMode::Fixed(n) => assert_eq!(p.generated, n, "frame {id}"),
                FrameGenerationMode::Dynamic { max, .. } => {
                    assert!(p.generated <= max, "frame {id}");
                }
            }
            checked += 1;
        }
    }
    let (last_id, point) = last.unwrap();
    display.wait_idle();
    gpu.queue(QueueKind::Render)
        .unwrap()
        .wait(point, Duration::from_secs(10))
        .unwrap();
    assert!(checked > 0);
    check_latency_reports(&reports, last_id);
    drop(display);
    drop(painter);
    drop(roots);
    drop(targets);
    drop(renderer);
    gpu.collect_garbage();
    assert_eq!(
        gpu.validation_messages(),
        0,
        "validation messages (see the log)"
    );
}
