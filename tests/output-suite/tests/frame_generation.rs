//! Frame generation through `orr::Renderer` and a `Display` on a real window: a bar crossing
//! the window, shaded by a test kernel and shown with `Frame::compose`, with the egui demo
//! windows and a cursor moving at constant speed over it. The display runs 2x, then 4x
//! where NGX allows it, then off, `Dynamic` on a PQ request, 2x on a scRGB request, off
//! again, and 4x with the UI drawn per presented image. Each frame presents its generated
//! images before itself, scRGB presents as PQ while frame generation is on, the UI hook runs
//! once per frame except with the UI per presented image, where it runs once per presented
//! image and the cursor advances by equal steps from image to image, every frame id keeps
//! one complete latency report, and the validation layer stays silent. Headless, a
//! lens-distorted pan keeps its generated images aligned through the distortion field.
#![cfg(windows)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use egui::epaint::ClippedShape;
use egui::{Color32, Context, FullOutput, Pos2, RawInput, Rect, Shape, TexturesDelta, vec2};
use gpu_suite::check_latency_reports;
use gpu_suite::window::Window;
use orr::orr_gpu::{
    Device, DeviceDesc, Format, MemoryKind, QueueKind, QueueRequest, SpecData, StorageImageIndex,
    Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage, ViewDesc as ImageView,
};
use orr::{
    ConsumerMarker, Display, DisplayRequest, DlssSdk, FrameDesc, FrameGenerationMode, FrameTime,
    Lens, Projection, Renderer, RendererDesc, SceneId, TextureFilter, ViewDesc, ViewHistoryId,
    ViewTarget,
};
use orr_egui::Painter;
use orr_math::{DVec3, Quat, UVec2, Vec2};
use orr_output::OutputFormat;
use orr_output::orr_dlss::Uuid;
use output_suite::shaders::moving_bar;

const PROJECT_ID: Uuid = Uuid::from_u128(0x2f6b_91c4_5a0e_4d3b_b7e2_0c18_d94a_6e53);

/// NGX frame generation and the driver's latency reports are per process: a second test's
/// frames break the first one's report order, so one test runs at a time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}
/// Frames per phase.
const PHASE: u64 = 160;
const PHASES: u64 = 7;
/// The phase that draws the UI per presented image.
const UI_PER_PRESENT: u64 = 6;
/// Frames into a phase before its generated count is checked: the feature and its history
/// settle first.
const SETTLE: u64 = 16;
const BAR_WIDTH: f32 = 64.0;
/// Pixels per rendered frame.
const SPEED: f32 = 6.0;
/// The cursor's speed, pixels per second.
const CURSOR_SPEED: f64 = 480.0;

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
        5 => (Off, ScRgb, ScRgb),
        _ => (Fixed(most), Sdr, Sdr),
    }
}

#[test]
fn frame_generation_presents_generated_images_before_each_frame() {
    let _serial = serial();
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
    let mut last_cursor: Option<f64> = None;
    let mut cursor_steps = 0;

    for i in 0..PHASE * PHASES {
        window.pump();
        let (mode, encoding, expected) = phase(i / PHASE, most, recommended);
        let per_present = i / PHASE == UI_PER_PRESENT;
        if i % PHASE == 0 {
            display.set_request(&request(encoding));
            display.set_frame_generation(&sdk, mode).unwrap();
            display.set_ui_per_present(per_present);
            last_cursor = None;
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
            texture_filter: TextureFilter::Hardware,
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
        // (presented, time offset, cursor position unwrapped) per run of the UI hook.
        let mut paints = Vec::new();
        f.ui_only(&mut |pass| {
            painter
                .update_textures(pass.cmd, id, std::mem::take(&mut pending))
                .unwrap();
            painter
                .paint(pass.cmd, id, pass.layer, &primitives, pixels_per_point)
                .unwrap();
            let x = (time.absolute + f64::from(pass.time_offset)) * CURSOR_SPEED;
            let center = Pos2::new((x % f64::from(extent[0])) as f32, 24.0);
            let cursor = ctx.tessellate(
                vec![ClippedShape::new(
                    Rect::EVERYTHING,
                    Shape::circle_filled(center, 12.0, Color32::WHITE),
                )],
                pixels_per_point,
            );
            painter
                .paint(pass.cmd, id, pass.layer, &cursor, pixels_per_point)
                .unwrap();
            paints.push((pass.presented, pass.time_offset, x));
        })
        .unwrap();
        last = Some((id, f.end().unwrap()));

        if per_present {
            let FrameGenerationMode::Fixed(n) = mode else {
                unreachable!()
            };
            let order: Vec<_> = paints.iter().map(|&(k, _, _)| k).collect();
            assert_eq!(order, (0..=n).collect::<Vec<_>>(), "frame {id}");
            let step = CURSOR_SPEED * f64::from(time.delta) / f64::from(n + 1);
            for &(_, _, x) in &paints {
                if let Some(prev) = last_cursor {
                    assert!(
                        ((x - prev) - step).abs() <= 0.1 * step,
                        "frame {id}: the cursor moved {} px, not {step} px",
                        x - prev
                    );
                    cursor_steps += 1;
                }
                last_cursor = Some(x);
            }
        } else {
            assert_eq!(paints, [(0, 0.0, paints[0].2)], "frame {id}");
        }

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
    assert!(cursor_steps > 0);
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

/// Frame generation of a lens-distorted pan: tonemap writes the lens into the distortion
/// field, and with the field passed the generated images keep their FLIP to the pan rendered
/// at their own time, relative to the FLIP of showing the previous frame, within 10% of the
/// same pan without distortion. Relative, because the lens changes the content's own
/// FLIP. Without the field they do markedly worse.
mod lens_distortion {
    use std::time::Duration;

    use orr_output::orr_dlss::{DlssSdk, FgCamera, FgInputs, FrameGeneration};
    use orr_output::{
        ComposePass, ComposeRoot, ComposeSource, DisplayRequest, DistortionField, Exposure,
        ExposurePass, OutputFormat, OutputKernels, OutputView, ResolvedDisplay, TonemapPass,
        ViewOutput,
    };
    use orr_scene::orr_gpu::{
        Allocation, ComputePipeline, Device, DeviceDesc, Format, Hazards, MemoryKind, QueueKind,
        ShaderStage, SpecData, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex,
        TextureMemory, TextureRegion, TextureUsage, ViewDesc,
    };
    use orr_scene::testing::{Image, Pixels, flip, read_texture_on};
    use orr_scene::{ExposureState, LensDistortion, SkyStats, ViewConstants};
    use output_suite::shaders::blur_scene;

    use super::PROJECT_ID;

    const OUT: [u32; 2] = [1280, 720];
    const FRAMES: u32 = 12;
    /// Scene time per frame: the pan moves a fraction of a pixel past whole pixels, so
    /// generated images resample with the lens off too.
    const STEP: f32 = 0.6875;
    /// The coarse pan in `blur_scene.slang`, whose textures stay well above the pixel pitch
    /// through the lens.
    const PAN: u32 = 2;
    const LENS: LensDistortion = LensDistortion {
        k1: 0.15,
        k2: 0.04,
        chromatic_aberration: 0.003,
    };

    /// A storage texture with its sampled and storage views.
    struct Target {
        texture: Texture,
        sampled: TextureIndex,
        storage: StorageImageIndex,
    }

    impl Target {
        fn new(gpu: &Device, format: Format, name: &str) -> Self {
            let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
            let desc = TextureDesc::d2(format, OUT[0], OUT[1], usage);
            let texture = gpu
                .create_texture(&desc, TextureMemory::Dedicated, name)
                .unwrap();
            let view = ViewDesc::default();
            Self {
                sampled: gpu.write_sampled(gpu.alloc_image_slot().unwrap(), &texture, &view),
                storage: gpu.write_storage(gpu.alloc_image_slot().unwrap(), &texture, &view),
                texture,
            }
        }
    }

    /// The pan rendered, tonemapped and composed for an SDR display, frame by frame, each
    /// frame waited on.
    struct Pan {
        gpu: Device,
        kernels: OutputKernels,
        output: OutputView,
        scene: ComputePipeline,
        display: ResolvedDisplay,
        scene_root: Allocation<blur_scene::Root>,
        exposure: Allocation<ExposureState>,
        constants: Allocation<ViewConstants>,
        sky: Allocation<SkyStats>,
        compose: ComposeRoot,
        radiance: Target,
        depth: Target,
        motion: Target,
        linear: Target,
        frame: u64,
    }

    impl Pan {
        fn new(gpu: &Device) -> Self {
            let stage =
                ShaderStage::new(&blur_scene::MODULE, "frame", SpecData::default()).unwrap();
            let mut exposure = gpu.gpu_malloc(MemoryKind::Upload, 2, "exposure").unwrap();
            exposure.as_mut_slice().unwrap().fill(ExposureState {
                pre_exposure: 1.0,
                prev_pre_exposure: 1.0,
                ..ExposureState::default()
            });
            let mut sky = gpu.gpu_malloc(MemoryKind::Upload, 1, "sky").unwrap();
            sky.as_mut_slice().unwrap()[0] = bytemuck::Zeroable::zeroed();
            let pan = Self {
                gpu: gpu.clone(),
                kernels: OutputKernels::new(gpu).unwrap(),
                output: OutputView::new(gpu, 2).unwrap(),
                scene: gpu.create_compute_pipeline(&stage, "pan").unwrap(),
                display: DisplayRequest::default().resolve(&[OutputFormat::Sdr], None, false),
                scene_root: gpu.gpu_malloc(MemoryKind::Upload, 1, "pan root").unwrap(),
                exposure,
                constants: gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap(),
                sky,
                compose: ComposeRoot::new(gpu).unwrap(),
                radiance: Target::new(gpu, Format::Rgba16Float, "pan radiance"),
                depth: Target::new(gpu, Format::R32Float, "pan depth"),
                motion: Target::new(gpu, Format::Rg16Float, "pan motion"),
                linear: Target::new(gpu, Format::Rgba16Float, "pan linear"),
                frame: 0,
            };
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            for t in [&pan.radiance, &pan.depth, &pan.motion, &pan.linear] {
                list.discard(&t.texture);
            }
            let done = queue.submit(vec![list], &[]).unwrap();
            queue.wait(done, Duration::from_secs(10)).unwrap();
            pan
        }

        /// Records the pan at `time` through tonemap with `lens` (writing `field`) and
        /// compose into `target`.
        fn record(
            &mut self,
            list: &mut orr_scene::orr_gpu::CommandList<'_>,
            time: f32,
            lens: Option<LensDistortion>,
            field: Option<&mut DistortionField>,
            target: &Target,
        ) {
            self.frame += 1;
            self.scene_root.as_mut_slice().unwrap()[0] = blur_scene::Root {
                reference: orr_scene::orr_gpu::GpuPtr::NULL,
                radiance: self.radiance.storage,
                depth: self.depth.storage,
                motion: self.motion.storage,
                width: OUT[0],
                height: OUT[1],
                scene: PAN,
                subframes: 1,
                shutter: 0.0,
                motion_scale: STEP,
                time,
            };
            let mut c: ViewConstants = bytemuck::Zeroable::zeroed();
            c.exposure = self.exposure.gpu;
            c.exposure_slot = (self.frame % 2) as u32;
            c.extent = OUT;
            c.delta_time = 1.0 / 60.0;
            self.constants.as_mut_slice().unwrap()[0] = c;
            let output = ViewOutput {
                exposure: Exposure::Manual {
                    ev100: -(1.2f32.log2()),
                },
                ..ViewOutput::default()
            };
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            list.dispatch(
                &self.scene,
                self.scene_root.gpu,
                [OUT[0].div_ceil(8), OUT[1].div_ceil(8), 1],
            );
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            let roots = self.output.roots(self.frame);
            self.kernels.exposure(
                list,
                roots,
                &ExposurePass {
                    source: self.radiance.sampled,
                    extent: OUT,
                    view: self.constants.gpu,
                    sky: self.sky.gpu,
                    frame: self.frame,
                    output: &output,
                },
            );
            self.kernels.tonemap(
                list,
                roots,
                &TonemapPass {
                    source: self.radiance.sampled,
                    target: self.linear.storage,
                    extent: OUT,
                    view: self.constants.gpu,
                    display: &self.display,
                    output: &output,
                    hook_source: None,
                    glare: None,
                    distortion: lens,
                    frame: self.frame,
                },
                field,
            );
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            self.kernels.compose(
                list,
                &mut self.compose,
                &ComposePass {
                    source: ComposeSource::Linear(self.linear.sampled),
                    ui: None,
                    target: target.storage,
                    extent: OUT,
                    display: &self.display,
                },
            );
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
        }

        /// The pan at `time` with `lens`, as the display shows it.
        fn truth(&mut self, time: f32, lens: Option<LensDistortion>, target: &Target) -> Image {
            let gpu = self.gpu.clone();
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            self.record(&mut list, time, lens, None, target);
            let done = queue.submit(vec![list], &[]).unwrap();
            queue.wait(done, Duration::from_secs(10)).unwrap();
            rgba8(&self.gpu, &target.texture)
        }
    }

    fn rgba8(gpu: &Device, t: &Texture) -> Image {
        Image {
            width: OUT[0],
            height: OUT[1],
            pixels: Pixels::Rgba8(read_texture_on(
                gpu,
                t,
                &TextureRegion::whole_mip(t.desc(), 0),
            )),
        }
    }

    /// A static camera: 60 degrees vertical, reverse Z with no far plane, 0.1 m near.
    fn camera() -> FgCamera {
        let f = 1.0 / (30f32.to_radians()).tan();
        let aspect = OUT[0] as f32 / OUT[1] as f32;
        let near = 0.1;
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        FgCamera {
            view_to_clip: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, -f, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, near, 0.0],
            ],
            clip_to_view: [
                [aspect / f, 0.0, 0.0, 0.0],
                [0.0, -1.0 / f, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0 / near],
                [0.0, 0.0, -1.0, 0.0],
            ],
            clip_to_prev_clip: identity,
            prev_clip_to_clip: identity,
            position: [0.0; 3],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            forward: [0.0, 0.0, -1.0],
            near,
            far: 100_000.0,
            fov_y: 60f32.to_radians(),
            aspect,
        }
    }

    /// Runs `FRAMES` frames of the pan generating `count` images each, the lens and field as
    /// given. Returns, for each of the last frame's generated images, its FLIP to the pan
    /// rendered at its time over the FLIP of showing the previous frame instead: the error
    /// frame generation leaves, relative to the content's own.
    fn run(
        gpu: &Device,
        sdk: &DlssSdk,
        count: u32,
        lens: Option<LensDistortion>,
        with_field: bool,
    ) -> Vec<f32> {
        let mut pan = Pan::new(gpu);
        let backbuffer = Target::new(gpu, Format::Rgba8Unorm, "pan backbuffer");
        let truth = Target::new(gpu, Format::Rgba8Unorm, "pan truth");
        let generated: Vec<Target> = (0..count)
            .map(|_| Target::new(gpu, Format::Rgba8Unorm, "pan generated"))
            .collect();
        let mut field = DistortionField::new(gpu, OUT).unwrap();
        let mut fg = FrameGeneration::new(sdk, &backbuffer.texture, false).unwrap();
        let mut flag: Allocation<u32> = gpu
            .gpu_malloc(MemoryKind::Readback, 1, "interpolation disabled")
            .unwrap();
        let queue = gpu.queue(QueueKind::Render).unwrap();
        for f in 0..FRAMES {
            let mut list = queue.command_list().unwrap();
            if f == 0 {
                for t in [&backbuffer, &truth].into_iter().chain(&generated) {
                    list.discard(&t.texture);
                }
            }
            flag.as_mut_slice().unwrap()[0] = 0;
            let written = with_field.then_some(&mut field);
            pan.record(&mut list, f as f32 * STEP, lens, written, &backbuffer);
            for (k, out) in generated.iter().enumerate() {
                let inputs = FgInputs {
                    backbuffer: &backbuffer.texture,
                    depth: &pan.depth.texture,
                    motion: &pan.motion.texture,
                    distortion: with_field.then(|| field.current_texture()),
                    interpolated: &out.texture,
                    disable_interpolation: Some(&flag),
                    camera: camera(),
                    jitter: [0.0; 2],
                    mvec_scale: OUT.map(|v| v as f32),
                    reset: f == 0,
                    frame_id: u64::from(f) + 1,
                    not_rendering_scene: false,
                    depth_scale: 1.0,
                };
                fg.record(&mut list, &inputs, k as u32 + 1, count).unwrap();
            }
            let done = queue.submit(vec![list], &[]).unwrap();
            queue.wait(done, Duration::from_secs(30)).unwrap();
        }
        assert!(
            !FrameGeneration::interpolation_disabled(&flag),
            "NGX disabled interpolation on a steady frame"
        );
        let images: Vec<Image> = generated.iter().map(|g| rgba8(gpu, &g.texture)).collect();
        let last = (FRAMES - 1) as f32;
        let previous = pan.truth((last - 1.0) * STEP, lens, &truth);
        let errors = images
            .iter()
            .enumerate()
            .map(|(k, image)| {
                let time = (last - 1.0 + (k as f32 + 1.0) / (count as f32 + 1.0)) * STEP;
                let want = pan.truth(time, lens, &truth);
                flip(&want, image).mean / flip(&want, &previous).mean
            })
            .collect();
        drop(fg);
        gpu.collect_garbage();
        errors
    }

    #[test]
    fn distorted_pan_keeps_mid_frame_flip() {
        let _serial = super::serial();
        gpu_suite::log_to_stderr();
        let hooks = DlssSdk::device_hooks(PROJECT_ID);
        let gpu = Device::new(&DeviceDesc {
            name: "output-suite lens frame generation",
            hooks: Some(&hooks),
            validation: true,
            ..DeviceDesc::default()
        })
        .unwrap();
        let sdk = DlssSdk::new(&gpu, PROJECT_ID).unwrap();
        let support = sdk.support();
        if !support.fg {
            log::warn!("skipped: NGX offers no frame generation here");
            return;
        }
        let mut counts = vec![1];
        if support.max_generated >= 3 {
            counts.push(3);
        }
        for count in counts {
            let mean = |e: &[f32]| e.iter().sum::<f32>() / e.len() as f32;
            let plain = mean(&run(&gpu, &sdk, count, None, false));
            let distorted = mean(&run(&gpu, &sdk, count, Some(LENS), true));
            let unaligned = mean(&run(&gpu, &sdk, count, Some(LENS), false));
            log::info!(
                "{}x relative FLIP: no lens {plain:.4}, lens with field {distorted:.4}, \
                 lens without field {unaligned:.4}",
                count + 1
            );
            assert!(
                distorted <= 1.1 * plain,
                "{}x: relative FLIP {distorted:.4} with the lens, {plain:.4} without",
                count + 1
            );
            assert!(
                unaligned > 1.3 * distorted,
                "{}x: the field does not align the lens: {unaligned:.4} without it, \
                 {distorted:.4} with",
                count + 1
            );
        }
        assert_eq!(
            gpu.validation_messages(),
            0,
            "validation messages (see the log)"
        );
    }
}
