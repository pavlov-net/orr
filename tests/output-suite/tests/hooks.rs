//! Hook points: depth and motion at output resolution, the read-only sources, the distortion
//! field, and what a view without hooks records and holds.

use half::f16;
use orr_output::{
    ChainImages, DisplayRequest, DistortionField, Exposure, ExposurePass, HookInputsPass, HookPass,
    HookPoint, Hooks, OutputFormat, OutputKernels, OutputView, ResolvedDisplay, TonemapPass,
    ViewOutput,
};
use orr_scene::orr_gpu::{
    Allocation, ComputePipeline, Format, GpuPtr, Hazards, MemoryKind, QueueKind, ShaderStage,
    SpecData, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
    TextureRegion, TextureUsage, ViewDesc,
};
use orr_scene::testing::{assert_clean, device, read_buffer, read_texture, wait};
use orr_scene::{ExposureState, SkyStats, ViewConstants};
use output_suite::shaders::{dump, field_offset, hook_line, swirl};

const OUT: [u32; 2] = [240, 136];
const RENDER: [u32; 2] = [160, 68];
/// No output pixel centre lands within 0.15 render pixels of a sample boundary.
const JITTER: [f32; 2] = [0.3125, -0.4375];

/// Reverse-Z depth of the render image: far geometry, with a near box.
const FAR: f32 = 0.01;
const NEAR: f32 = 0.5;
const LINE_DEPTH: f32 = 0.1;
const LINE_COLOUR: [f32; 4] = [1.0, 0.0, 1.0, 1.0];

fn render_depth(x: u32, y: u32) -> f32 {
    if (50..110).contains(&x) && (20..45).contains(&y) {
        NEAR
    } else {
        FAR
    }
}

/// Exact in RG16F.
fn render_motion(x: u32, y: u32) -> [f32; 2] {
    [x as f32 / 256.0, -(y as f32) / 512.0]
}

/// The render pixel whose jittered sample is nearest the centre of output pixel `p`.
fn render_pixel(p: [u32; 2]) -> [u32; 2] {
    std::array::from_fn(|a| {
        let at = (p[a] as f32 + 0.5) * RENDER[a] as f32 / OUT[a] as f32 + JITTER[a];
        (at.floor() as i64).clamp(0, i64::from(RENDER[a]) - 1) as u32
    })
}

/// RR's output before any hook: smooth, inside 0..1.
fn radiance(x: u32, y: u32) -> [f32; 4] {
    let u = (x as f32 + 0.5) / OUT[0] as f32;
    let v = (y as f32 + 0.5) / OUT[1] as f32;
    let c = [u, v, 0.5 + 0.4 * (6.0 * u + 3.0 * v).sin(), 1.0];
    c.map(|x| f16::from_f32(x).to_f32())
}

fn pixels(extent: [u32; 2]) -> impl Iterator<Item = [u32; 2]> {
    (0..extent[1]).flat_map(move |y| (0..extent[0]).map(move |x| [x, y]))
}

fn rgba16f_bytes(c: &[[f32; 4]]) -> Vec<u8> {
    c.iter()
        .flatten()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect()
}

fn read_rgba16f(t: &Texture) -> Vec<[f32; 4]> {
    read_texture(t, &TextureRegion::whole_mip(t.desc(), 0))
        .chunks_exact(8)
        .map(|p| std::array::from_fn(|c| f16::from_le_bytes([p[2 * c], p[2 * c + 1]]).to_f32()))
        .collect()
}

fn pipeline(module: &'static orr_shaders::ShaderModule, name: &str) -> ComputePipeline {
    let stage = ShaderStage::new(module, "main", SpecData::default()).unwrap();
    device().create_compute_pipeline(&stage, name).unwrap()
}

fn groups(extent: [u32; 2]) -> [u32; 3] {
    [extent[0].div_ceil(8), extent[1].div_ceil(8), 1]
}

/// A root in its own upload memory, rewritten each frame the hook runs; every frame is
/// waited on.
struct Root<T: 'static>(Allocation<T>);

impl<T: bytemuck::Pod> Root<T> {
    fn new() -> Self {
        Self(
            device()
                .gpu_malloc(MemoryKind::Upload, 1, "hook root")
                .unwrap(),
        )
    }

    fn put(&mut self, value: T) -> GpuPtr<T> {
        self.0.as_mut_slice().unwrap()[0] = value;
        self.0.gpu
    }
}

type Hook<'a> = &'a mut dyn FnMut(&mut HookPass<'_, '_>);

/// One view with lighting's targets filled from the functions above, RR's output reset to
/// [`radiance`] each frame, manual exposure and tonemap into the display-linear image.
struct View {
    kernels: OutputKernels,
    output: OutputView,
    display: ResolvedDisplay,
    exposure: Allocation<ExposureState>,
    constants: Allocation<ViewConstants>,
    sky: Allocation<SkyStats>,
    upload: Allocation<u8>,
    rr_depth: Texture,
    rr_motion: Texture,
    rr_output: Texture,
    linear: Texture,
    rr_depth_index: TextureIndex,
    rr_motion_index: TextureIndex,
    chain: ChainImages,
    frame: u64,
}

impl View {
    fn new() -> Self {
        let gpu = device();
        let texture = |format, extent: [u32; 2], name| {
            let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
            let desc = TextureDesc::d2(format, extent[0], extent[1], usage);
            gpu.create_texture(&desc, TextureMemory::Dedicated, name)
                .unwrap()
        };
        let rr_depth = texture(Format::R32Float, RENDER, "rr depth");
        let rr_motion = texture(Format::Rg16Float, RENDER, "rr motion");
        let rr_output = texture(Format::Rgba16Float, OUT, "rr output");
        let linear = texture(Format::Rgba16Float, OUT, "display linear");
        let view = ViewDesc::default();
        let slot = || gpu.alloc_image_slot().unwrap();
        let chain = ChainImages {
            rr_output: gpu.write_sampled(slot(), &rr_output, &view),
            rr_output_store: gpu.write_storage(slot(), &rr_output, &view),
            linear: gpu.write_sampled(slot(), &linear, &view),
            linear_store: gpu.write_storage(slot(), &linear, &view),
        };

        let depth: Vec<u8> = pixels(RENDER)
            .flat_map(|[x, y]| render_depth(x, y).to_le_bytes())
            .collect();
        let motion: Vec<u8> = pixels(RENDER)
            .flat_map(|[x, y]| render_motion(x, y))
            .flat_map(|v| f16::from_f32(v).to_le_bytes())
            .collect();
        let image: Vec<[f32; 4]> = pixels(OUT).map(|[x, y]| radiance(x, y)).collect();
        let image = rgba16f_bytes(&image);
        let mut upload = gpu
            .gpu_malloc::<u8>(
                MemoryKind::Upload,
                (depth.len() + motion.len() + image.len()) as u64,
                "view upload",
            )
            .unwrap();
        let bytes = upload.as_mut_slice().unwrap();
        bytes[..image.len()].copy_from_slice(&image);
        bytes[image.len()..image.len() + depth.len()].copy_from_slice(&depth);
        bytes[image.len() + depth.len()..].copy_from_slice(&motion);

        let exposure = gpu
            .gpu_malloc::<ExposureState>(MemoryKind::Upload, 2, "exposure slots")
            .unwrap();
        let mut sky = gpu.gpu_malloc(MemoryKind::Upload, 1, "sky").unwrap();
        sky.as_mut_slice().unwrap()[0] = bytemuck::Zeroable::zeroed();
        let mut v = Self {
            kernels: OutputKernels::new(gpu).unwrap(),
            output: OutputView::new(gpu, 2).unwrap(),
            display: DisplayRequest::default().resolve(&[OutputFormat::Sdr], None, false),
            exposure,
            constants: gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap(),
            sky,
            upload,
            rr_depth_index: gpu.write_sampled(slot(), &rr_depth, &view),
            rr_motion_index: gpu.write_sampled(slot(), &rr_motion, &view),
            rr_depth,
            rr_motion,
            rr_output,
            linear,
            chain,
            frame: 0,
        };
        v.exposure.as_mut_slice().unwrap().fill(ExposureState {
            pre_exposure: 1.0,
            prev_pre_exposure: 1.0,
            ..ExposureState::default()
        });
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        for t in [&v.rr_depth, &v.rr_motion, &v.rr_output, &v.linear] {
            list.discard(t);
        }
        let at = image.len() as u64;
        let region = |t: &Texture| TextureRegion::whole_mip(t.desc(), 0);
        list.copy_to_texture(&v.rr_depth, &region(&v.rr_depth), v.upload.gpu.offset(at));
        let at = at + depth.len() as u64;
        list.copy_to_texture(&v.rr_motion, &region(&v.rr_motion), v.upload.gpu.offset(at));
        wait(queue.submit(vec![list], &[]).unwrap());
        v
    }

    /// Renders one frame: RR's output reset, hook inputs, the `scene_referred` hook,
    /// exposure, tonemap and the `display_linear` hook. Returns the frame's pass names.
    fn frame(
        &mut self,
        mut field: Option<&mut DistortionField>,
        scene_referred: Option<Hook<'_>>,
        display_linear: Option<Hook<'_>>,
    ) -> Vec<&'static str> {
        let gpu = device();
        self.frame += 1;
        let mut constants: ViewConstants = bytemuck::Zeroable::zeroed();
        constants.exposure = self.exposure.gpu;
        constants.exposure_slot = (self.frame % 2) as u32;
        constants.extent = RENDER;
        constants.jitter = JITTER;
        constants.frame_index = self.frame as u32;
        constants.delta_time = 1.0 / 60.0;
        self.constants.as_mut_slice().unwrap()[0] = constants;
        let view = self.constants.gpu;
        let output = ViewOutput {
            exposure: Exposure::Manual { ev100: 0.0 },
            distortion_field: field.is_some(),
            ..ViewOutput::default()
        };
        let hooks = Hooks {
            scene_referred: scene_referred.is_some(),
            display_linear: display_linear.is_some(),
        };

        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        let whole = TextureRegion::whole_mip(self.rr_output.desc(), 0);
        list.copy_to_texture(&self.rr_output, &whole, self.upload.gpu);
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        let inputs = self
            .kernels
            .hook_inputs(
                &mut list,
                &mut self.output,
                &HookInputsPass {
                    rr_depth: self.rr_depth_index,
                    rr_motion: self.rr_motion_index,
                    chain: self.chain,
                    extent: OUT,
                    view,
                    jitter: JITTER,
                    frame: self.frame,
                    hooks,
                },
                field.as_deref_mut(),
            )
            .unwrap();
        assert_eq!(inputs.is_some(), hooks.any());
        if let (Some(inputs), Some(hook)) = (&inputs, scene_referred) {
            inputs.run(
                &mut list,
                HookPoint::SceneReferred,
                &self.chain,
                field.as_deref_mut(),
                hook,
            );
        }
        let roots = self.output.roots(self.frame);
        self.kernels.exposure(
            &mut list,
            roots,
            &ExposurePass {
                source: self.chain.rr_output,
                extent: OUT,
                view,
                sky: self.sky.gpu,
                frame: self.frame,
                output: &output,
            },
        );
        self.kernels.tonemap(
            &mut list,
            roots,
            &TonemapPass {
                source: self.chain.rr_output,
                target: self.chain.linear_store,
                extent: OUT,
                view,
                display: &self.display,
                output: &output,
                hook_source: hooks.display_linear.then_some(self.chain.rr_output_store),
            },
        );
        if let (Some(inputs), Some(hook)) = (&inputs, display_linear) {
            inputs.run(
                &mut list,
                HookPoint::DisplayLinear,
                &self.chain,
                field,
                hook,
            );
        }
        let point = queue.submit(vec![list], &[]).unwrap();
        wait(point);
        gpu.pass_results(point)
            .unwrap()
            .iter()
            .map(|p| p.name)
            .collect()
    }
}

/// Copies `t` (any float format, read as four channels) back.
fn dump_texture(t: TextureIndex, extent: [u32; 2]) -> Vec<[f32; 4]> {
    let gpu = device();
    let n = u64::from(extent[0] * extent[1]);
    let out = gpu
        .gpu_malloc::<f32>(MemoryKind::DeviceLocal, 4 * n, "dump")
        .unwrap();
    let mut root = Root::<dump::Root>::new();
    let root = root.put(dump::Root {
        out: out.gpu,
        source: t,
        width: extent[0],
        height: extent[1],
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    });
    let p = pipeline(&dump::MODULE, "dump");
    let queue = gpu.queue(QueueKind::Render).unwrap();
    let mut list = queue.command_list().unwrap();
    list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
    list.dispatch(&p, root, groups(extent));
    wait(queue.submit(vec![list], &[]).unwrap());
    read_buffer(out.gpu, 4 * n)
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

/// A hook drawing the debug line, depth-tested against the hook's depth. It records the
/// hook's depth and motion views for the caller.
fn line_hook<'a>(
    p: &'a ComputePipeline,
    root: &'a mut Root<hook_line::Root>,
    seen: &'a mut Option<(TextureIndex, TextureIndex)>,
) -> impl FnMut(&mut HookPass<'_, '_>) + 'a {
    move |pass: &mut HookPass<'_, '_>| {
        *seen = Some((pass.depth, pass.motion));
        let r = root.put(hook_line::Root {
            depth: pass.depth,
            image: pass.image,
            width: pass.extent[0],
            height: pass.extent[1],
            row: 40,
            column: 120,
            line_depth: LINE_DEPTH,
            _pad: 0,
        });
        pass.cmd.begin_pass("line", false);
        pass.cmd.dispatch(p, r, groups(pass.extent));
        pass.cmd.end_pass();
    }
}

fn on_line([x, y]: [u32; 2]) -> bool {
    y == 40 || x == 120
}

/// Whether the line shows at `p`: in front of the far geometry, behind the box.
fn line_visible(p: [u32; 2]) -> bool {
    let [x, y] = render_pixel(p);
    LINE_DEPTH >= render_depth(x, y)
}

#[test]
fn debug_line_is_hidden_by_nearer_geometry() {
    let mut v = View::new();
    let p = pipeline(&hook_line::MODULE, "line");
    let mut root = Root::new();
    let mut seen = None;

    v.frame(None, Some(&mut line_hook(&p, &mut root, &mut seen)), None);
    let scene = read_rgba16f(&v.rr_output);
    let (depth, motion) = seen.expect("hook ran");
    let depth = dump_texture(depth, OUT);
    let motion = dump_texture(motion, OUT);
    let mut hidden = 0;
    for (i, p) in pixels(OUT).enumerate() {
        let [rx, ry] = render_pixel(p);
        assert_eq!(depth[i][0], render_depth(rx, ry), "depth at {p:?}");
        assert_eq!(
            [motion[i][0], motion[i][1]],
            render_motion(rx, ry),
            "motion at {p:?}"
        );
        let expected = if on_line(p) && line_visible(p) {
            LINE_COLOUR
        } else {
            radiance(p[0], p[1])
        };
        hidden += usize::from(on_line(p) && !line_visible(p));
        assert_eq!(scene[i], expected, "scene_referred line at {p:?}");
    }
    assert!(hidden > 0, "the box hides no line pixel");

    let mut seen = None;
    v.frame(None, None, Some(&mut line_hook(&p, &mut root, &mut seen)));
    let linear = read_rgba16f(&v.linear);
    for (i, p) in pixels(OUT).enumerate() {
        if on_line(p) {
            assert_eq!(
                linear[i] == LINE_COLOUR,
                line_visible(p),
                "display_linear line at {p:?}"
            );
        }
    }
    assert_clean();
}

fn swirl_hook<'a>(
    p: &'a ComputePipeline,
    root: &'a mut Root<swirl::Root>,
) -> impl FnMut(&mut HookPass<'_, '_>) + 'a {
    move |pass: &mut HookPass<'_, '_>| {
        let r = root.put(swirl::Root {
            source: pass.source,
            image: pass.image,
            width: pass.extent[0],
            height: pass.extent[1],
            strength: STRENGTH,
            radius: RADIUS,
            _pad0: 0,
            _pad1: 0,
        });
        pass.cmd.begin_pass("swirl", false);
        pass.cmd.dispatch(p, r, groups(pass.extent));
        pass.cmd.end_pass();
    }
}

const STRENGTH: f32 = 2.5;
const RADIUS: f32 = 60.0;

/// The swirl kernel in f64.
fn swirl_reference(source: &[[f32; 4]]) -> Vec<[f64; 4]> {
    let [w, h] = OUT.map(i64::from);
    let texel = |x: i64, y: i64| source[(y.clamp(0, h - 1) * w + x.clamp(0, w - 1)) as usize];
    pixels(OUT)
        .map(|[x, y]| {
            let (cx, cy) = (0.5 * w as f64, 0.5 * h as f64);
            let (qx, qy) = (f64::from(x) + 0.5 - cx, f64::from(y) + 0.5 - cy);
            let t = (1.0 - qx.hypot(qy) / f64::from(RADIUS)).clamp(0.0, 1.0);
            let (s, c) = (f64::from(STRENGTH) * t * t).sin_cos();
            let ax = cx + c * qx - s * qy - 0.5;
            let ay = cy + s * qx + c * qy - 0.5;
            let (ix, iy) = (ax.floor(), ay.floor());
            let (fx, fy) = (ax - ix, ay - iy);
            let (ix, iy) = (ix as i64, iy as i64);
            std::array::from_fn(|ch| {
                let at = |dx, dy| f64::from(texel(ix + dx, iy + dy)[ch]);
                let top = at(0, 0) + (at(1, 0) - at(0, 0)) * fx;
                let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * fx;
                top + (bottom - top) * fy
            })
        })
        .collect()
}

fn assert_within_10_bit(image: &[[f32; 4]], reference: &[[f64; 4]], what: &str) {
    let mut worst = 0.0f64;
    for (a, b) in image.iter().zip(reference) {
        for c in 0..3 {
            worst = worst.max((f64::from(a[c]) - b[c]).abs() * 1023.0);
        }
    }
    assert!(worst <= 1.0, "{what}: {worst:.3} code values at 10 bits");
}

#[test]
fn swirl_warp_reads_its_source() {
    let mut v = View::new();
    let p = pipeline(&swirl::MODULE, "swirl");
    let mut root = Root::new();

    v.frame(None, Some(&mut swirl_hook(&p, &mut root)), None);
    let original: Vec<[f32; 4]> = pixels(OUT).map(|[x, y]| radiance(x, y)).collect();
    assert_within_10_bit(
        &read_rgba16f(&v.rr_output),
        &swirl_reference(&original),
        "scene_referred swirl",
    );

    // The source of `display_linear` is tonemap's output.
    v.frame(None, None, Some(&mut |_: &mut HookPass<'_, '_>| {}));
    assert_eq!(read_rgba16f(&v.rr_output), read_rgba16f(&v.linear));

    v.frame(None, None, Some(&mut swirl_hook(&p, &mut root)));
    let source = read_rgba16f(&v.rr_output);
    assert_within_10_bit(
        &read_rgba16f(&v.linear),
        &swirl_reference(&source),
        "display_linear swirl",
    );
    assert_clean();
}

#[test]
fn no_hooks_records_and_holds_nothing() {
    let mut v = View::new();
    let base = v.frame(None, None, None);
    assert_eq!(base, ["adapt", "tonemap"]);
    assert_eq!(v.output.hook_input_bytes(), 0);

    let p = pipeline(&hook_line::MODULE, "line");
    let mut root = Root::new();
    let mut seen = None;
    let mut hook = line_hook(&p, &mut root, &mut seen);
    let mut other_root = Root::new();
    let mut other_seen = None;
    let mut other = line_hook(&p, &mut other_root, &mut other_seen);
    let with_hooks = [
        v.frame(None, Some(&mut hook), None),
        v.frame(None, None, Some(&mut hook)),
        v.frame(None, Some(&mut hook), Some(&mut other)),
    ];
    let bytes = u64::from(OUT[0] * OUT[1]) * (4 + 4);
    assert!(v.output.hook_input_bytes() >= bytes);
    for passes in with_hooks {
        let added: Vec<_> = passes
            .iter()
            .filter(|&&n| n != "line" && !base.contains(&n))
            .collect();
        assert_eq!(added, [&"hook_inputs"], "{passes:?}");
    }

    assert_eq!(v.frame(None, None, None), base);
    assert_eq!(v.output.hook_input_bytes(), 0);
    assert_clean();
}

fn offset_hook<'a>(
    p: &'a ComputePipeline,
    root: &'a mut Root<field_offset::Root>,
    shift: Option<[f32; 2]>,
) -> impl FnMut(&mut HookPass<'_, '_>) + 'a {
    move |pass: &mut HookPass<'_, '_>| {
        assert_ne!(pass.distortion, StorageImageIndex::NONE);
        let Some([dx, dy]) = shift else { return };
        let extent = [pass.extent[0].div_ceil(2), pass.extent[1].div_ceil(2)];
        let r = root.put(field_offset::Root {
            source: pass.distortion_source,
            target: pass.distortion,
            width: extent[0],
            height: extent[1],
            dx,
            dy,
            _pad0: 0,
            _pad1: 0,
        });
        pass.cmd.dispatch(p, r, groups(extent));
        pass.distortion_written = true;
    }
}

#[test]
fn distortion_field_clears_and_composes() {
    let mut v = View::new();
    let mut field = DistortionField::new(device(), OUT).unwrap();
    assert_eq!(field.extent(), [120, 68]);
    let p = pipeline(&field_offset::MODULE, "field offset");
    let (mut a, mut b) = (Root::new(), Root::new());
    let near =
        |got: [f32; 4], want: [f32; 4]| got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-3);
    let check = |field: &DistortionField, want: [f32; 4], what: &str| {
        let got = dump_texture(field.current(), field.extent());
        assert!(got.iter().all(|&g| near(g, want)), "{what}: {:?}", got[0]);
    };

    // Each stage composes into the field the stage before wrote.
    v.frame(
        Some(&mut field),
        Some(&mut offset_hook(&p, &mut a, Some([0.01, 0.02]))),
        Some(&mut offset_hook(&p, &mut b, Some([0.03, -0.01]))),
    );
    check(&field, [0.04, 0.01, -0.04, -0.01], "two stages");

    // Hook inputs clear the field to identity; hooks that write nothing leave it there.
    v.frame(
        Some(&mut field),
        Some(&mut offset_hook(&p, &mut a, None)),
        Some(&mut offset_hook(&p, &mut b, None)),
    );
    check(&field, [0.0; 4], "no stage");

    v.frame(
        Some(&mut field),
        Some(&mut offset_hook(&p, &mut a, None)),
        Some(&mut offset_hook(&p, &mut b, Some([0.03, -0.01]))),
    );
    check(&field, [0.03, -0.01, -0.03, 0.01], "second stage only");
    assert_clean();
}
