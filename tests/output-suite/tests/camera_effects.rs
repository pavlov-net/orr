//! Camera effects after RR: motion blur against shutter-accumulated references and depth of
//! field against lens-disc references.

use half::f16;
use orr_output::{ChainImages, MotionBlurPass, OutputKernels, OutputView, ViewOutput, srgb_encode};
use orr_scene::ViewConstants;
use orr_scene::orr_gpu::{
    Allocation, ComputePipeline, Format, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData,
    Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};
use orr_scene::testing::{
    FlipError, Image, Pixels, assert_clean, device, flip, read_buffer, read_texture, wait,
};
use output_suite::shaders::blur_scene;

/// Not a multiple of the 20-pixel tile either way.
const OUT: [u32; 2] = [336, 190];
const SHUTTER: f32 = 0.5;
const SUBFRAMES: u32 = 64;

#[derive(Clone, Copy, Debug)]
enum Scene {
    Wheel,
    Pan,
}

/// One view's chain, depth and motion at `OUT` (render extent equal to output, no jitter),
/// filled by `blur_scene.slang`.
struct Blur {
    kernels: OutputKernels,
    output: OutputView,
    scene: ComputePipeline,
    reference: ComputePipeline,
    root: Allocation<blur_scene::Root>,
    constants: Allocation<ViewConstants>,
    rr_output: Texture,
    linear: Texture,
    depth: TextureIndex,
    motion: TextureIndex,
    radiance_store: StorageImageIndex,
    depth_store: StorageImageIndex,
    motion_store: StorageImageIndex,
    chain: ChainImages,
    reference_out: Allocation<f32>,
    frame: u64,
    _targets: [Texture; 2],
}

impl Blur {
    fn new() -> Self {
        let gpu = device();
        let texture = |format, name| {
            let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
            let desc = TextureDesc::d2(format, OUT[0], OUT[1], usage);
            gpu.create_texture(&desc, TextureMemory::Dedicated, name)
                .unwrap()
        };
        let rr_output = texture(Format::Rgba16Float, "rr output");
        let linear = texture(Format::Rgba16Float, "display linear");
        let depth = texture(Format::R32Float, "rr depth");
        let motion = texture(Format::Rg16Float, "rr motion");
        let view = ViewDesc::default();
        let slot = || gpu.alloc_image_slot().unwrap();
        let chain = ChainImages {
            rr_output: gpu.write_sampled(slot(), &rr_output, &view),
            rr_output_store: gpu.write_storage(slot(), &rr_output, &view),
            linear: gpu.write_sampled(slot(), &linear, &view),
            linear_store: gpu.write_storage(slot(), &linear, &view),
        };
        let pipeline = |entry, name| {
            let stage = ShaderStage::new(&blur_scene::MODULE, entry, SpecData::default()).unwrap();
            gpu.create_compute_pipeline(&stage, name).unwrap()
        };
        let mut constants = gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap();
        let mut c: ViewConstants = bytemuck::Zeroable::zeroed();
        c.extent = OUT;
        constants.as_mut_slice().unwrap()[0] = c;
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        for t in [&rr_output, &linear, &depth, &motion] {
            list.discard(t);
        }
        wait(queue.submit(vec![list], &[]).unwrap());
        Self {
            kernels: OutputKernels::new(gpu).unwrap(),
            output: OutputView::new(gpu, 2).unwrap(),
            scene: pipeline("frame", "blur scene"),
            reference: pipeline("reference", "blur reference"),
            root: gpu.gpu_malloc(MemoryKind::Upload, 1, "scene root").unwrap(),
            constants,
            radiance_store: chain.rr_output_store,
            depth: gpu.write_sampled(slot(), &depth, &view),
            motion: gpu.write_sampled(slot(), &motion, &view),
            depth_store: gpu.write_storage(slot(), &depth, &view),
            motion_store: gpu.write_storage(slot(), &motion, &view),
            chain,
            reference_out: gpu
                .gpu_malloc(
                    MemoryKind::DeviceLocal,
                    4 * u64::from(OUT[0] * OUT[1]),
                    "reference",
                )
                .unwrap(),
            frame: 0,
            rr_output,
            linear,
            _targets: [depth, motion],
        }
    }

    fn put_root(&mut self, scene: Scene, motion_scale: f32) {
        self.root.as_mut_slice().unwrap()[0] = blur_scene::Root {
            reference: self.reference_out.gpu,
            radiance: self.radiance_store,
            depth: self.depth_store,
            motion: self.motion_store,
            width: OUT[0],
            height: OUT[1],
            scene: scene as u32,
            subframes: SUBFRAMES,
            shutter: SHUTTER,
            motion_scale,
            _pad: 0,
        };
    }

    /// The scene at time 0 into RR's output, blurred with `shutter` and `generated`. Returns
    /// the image the chain ends on and the frame's pass names.
    fn render(
        &mut self,
        scene: Scene,
        motion_scale: f32,
        pass: impl FnOnce(&mut MotionBlurPass<'_>),
    ) -> (Vec<[f32; 4]>, Vec<&'static str>) {
        let gpu = device();
        self.frame += 1;
        self.put_root(scene, motion_scale);
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        list.dispatch(&self.scene, self.root.gpu, groups(OUT));
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        let output = ViewOutput::default();
        let mut p = MotionBlurPass {
            rr_depth: self.depth,
            rr_motion: self.motion,
            extent: OUT,
            view: self.constants.gpu,
            shutter: SHUTTER,
            generated: 0,
            history_reset: false,
            frame: self.frame,
            output: &output,
        };
        pass(&mut p);
        let mut chain = self.chain;
        let blurred = self
            .kernels
            .motion_blur(&mut list, &mut self.output, &p, &mut chain)
            .unwrap();
        assert_eq!(blurred, chain != self.chain);
        let point = queue.submit(vec![list], &[]).unwrap();
        wait(point);
        let names = gpu
            .pass_results(point)
            .unwrap()
            .iter()
            .map(|p| p.name)
            .collect();
        let image = if blurred {
            &self.linear
        } else {
            &self.rr_output
        };
        (read_rgba16f(image), names)
    }

    /// The scene averaged over the open shutter.
    fn reference(&mut self, scene: Scene) -> Vec<[f32; 4]> {
        let gpu = device();
        self.put_root(scene, 1.0);
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        list.dispatch(&self.reference, self.root.gpu, groups(OUT));
        wait(queue.submit(vec![list], &[]).unwrap());
        read_buffer(self.reference_out.gpu, 4 * u64::from(OUT[0] * OUT[1]))
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }
}

fn groups(extent: [u32; 2]) -> [u32; 3] {
    [extent[0].div_ceil(8), extent[1].div_ceil(8), 1]
}

fn read_rgba16f(t: &Texture) -> Vec<[f32; 4]> {
    read_texture(t, &TextureRegion::whole_mip(t.desc(), 0))
        .chunks_exact(8)
        .map(|p| std::array::from_fn(|c| f16::from_le_bytes([p[2 * c], p[2 * c + 1]]).to_f32()))
        .collect()
}

fn srgb8(linear: &[[f32; 4]]) -> Image {
    let pixels = linear
        .iter()
        .flat_map(|p| {
            let c = |v: f32| (srgb_encode(f64::from(v).clamp(0.0, 1.0)) * 255.0).round() as u8;
            [c(p[0]), c(p[1]), c(p[2]), 255]
        })
        .collect();
    Image {
        width: OUT[0],
        height: OUT[1],
        pixels: Pixels::Rgba8(pixels),
    }
}

fn flip_against(reference: &[[f32; 4]], image: &[[f32; 4]]) -> FlipError {
    flip(&srgb8(reference), &srgb8(image))
}

/// Blurred FLIP against the reference at most `mean`, and well under the unblurred frame's.
fn check_scene(scene: Scene, mean: f32) {
    let mut b = Blur::new();
    let reference = b.reference(scene);
    let (sharp, _) = b.render(scene, 1.0, |p| p.shutter = 0.0);
    let (blurred, names) = b.render(scene, 1.0, |_| {});
    assert_eq!(names, ["motion_blur"]);
    let sharp = flip_against(&reference, &sharp);
    let blurred = flip_against(&reference, &blurred);
    log::info!("{scene:?}: FLIP blurred {blurred:?}, unblurred {sharp:?}");
    assert!(blurred.mean <= mean, "{scene:?}: {blurred:?}");
    assert!(
        blurred.mean < 0.6 * sharp.mean,
        "{scene:?}: {blurred:?} against {sharp:?}"
    );
    assert_clean();
}

#[test]
fn spinning_wheel_matches_shutter_reference() {
    check_scene(Scene::Wheel, 0.05);
}

#[test]
fn camera_pan_matches_shutter_reference() {
    check_scene(Scene::Pan, 0.05);
}

/// At one display rate, frame generation renders 2x or 4x the motion per frame; the streak
/// per presented frame stays the same.
#[test]
fn frame_generation_keeps_streak_per_presented_frame() {
    let mut b = Blur::new();
    let (base, _) = b.render(Scene::Wheel, 1.0, |_| {});
    for generated in [1, 3] {
        let scale = (generated + 1) as f32;
        let (image, _) = b.render(Scene::Wheel, scale, |p| p.generated = generated);
        assert!(image == base, "{generated} generated frames");
    }
    assert_clean();
}

#[test]
fn closed_shutter_and_history_reset_record_nothing() {
    let mut b = Blur::new();
    let (sharp, names) = b.render(Scene::Pan, 1.0, |p| p.shutter = 0.0);
    assert!(names.is_empty(), "{names:?}");
    assert_eq!(b.output.motion_blur_bytes(), 0);
    let (reset, names) = b.render(Scene::Pan, 1.0, |p| p.history_reset = true);
    assert!(names.is_empty(), "{names:?}");
    assert!(reset == sharp);
    b.render(Scene::Pan, 1.0, |_| {});
    assert!(b.output.motion_blur_bytes() > 0);
    b.render(Scene::Pan, 1.0, |p| p.shutter = 0.0);
    assert_eq!(b.output.motion_blur_bytes(), 0);
    assert_clean();
}

/// Depth of field on both paths against lens-disc accumulation references.
mod depth_of_field {
    use orr_output::{
        ChainImages, DepthOfFieldPass, DofPath, OutputKernels, OutputView, ViewOutput, srgb_encode,
    };
    use orr_scene::orr_gpu::{
        Allocation, ComputePipeline, Format, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData,
        Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
        TextureRegion, TextureUsage, ViewDesc,
    };
    use orr_scene::testing::{
        FlipError, Image, Pixels, assert_clean, device, flip, read_buffer, read_texture, wait,
    };
    use orr_scene::{DepthOfField, Focus, ViewConstants};
    use output_suite::shaders::dof_scene;

    use super::{groups, read_rgba16f};

    const OUT: [u32; 2] = [320, 180];
    const LENS_SAMPLES: u32 = 256;
    const SENSOR_MM: f32 = 36.0;
    /// tan(20 degrees): a 40-degree horizontal field of view.
    const TAN_X: f32 = 0.363_970_23;
    const NEAR: f32 = 0.1;
    const FOCUS: f32 = 1.0;

    #[derive(Clone, Copy, Debug)]
    enum Scene {
        Foreground,
        Background,
        Highlights,
        Water,
    }

    fn lens(blades: u32, f_number: f32) -> DepthOfField {
        DepthOfField {
            f_number,
            focus: Focus::Manual { distance_m: FOCUS },
            blades,
            round_at_f_number: 0.7,
            max_coc: 0.12,
        }
    }

    fn tan_y() -> f32 {
        TAN_X * OUT[1] as f32 / OUT[0] as f32
    }

    /// The aperture radius, m, of the pinhole camera whose lens-disc rays blur by the
    /// thin-lens CoC.
    fn lens_radius(dof: &DepthOfField) -> f32 {
        let focal = 0.5 * SENSOR_MM / TAN_X;
        let aperture = focal / dof.f_number;
        let focus = 1000.0 * FOCUS;
        let radius_px = 0.5 * aperture * focal / (focus - focal) / SENSOR_MM * OUT[0] as f32;
        radius_px * FOCUS * 2.0 * TAN_X / OUT[0] as f32
    }

    /// One view's chain and render targets at `OUT` (render extent equal to output, no
    /// jitter, no motion), filled by `dof_scene.slang`.
    struct Bed {
        kernels: OutputKernels,
        output: OutputView,
        scene: ComputePipeline,
        reference: ComputePipeline,
        root: Allocation<dof_scene::Root>,
        constants: Allocation<ViewConstants>,
        rr_output: Texture,
        linear: Texture,
        depth: TextureIndex,
        depth_store: StorageImageIndex,
        motion: TextureIndex,
        motion_store: StorageImageIndex,
        chain: ChainImages,
        reference_out: Allocation<f32>,
        frame: u64,
        _targets: [Texture; 2],
    }

    impl Bed {
        fn new() -> Self {
            let gpu = device();
            let texture = |format, name| {
                let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
                let desc = TextureDesc::d2(format, OUT[0], OUT[1], usage);
                gpu.create_texture(&desc, TextureMemory::Dedicated, name)
                    .unwrap()
            };
            let rr_output = texture(Format::Rgba16Float, "rr output");
            let linear = texture(Format::Rgba16Float, "display linear");
            let depth = texture(Format::R32Float, "rr depth");
            let motion = texture(Format::Rg16Float, "rr motion");
            let view = ViewDesc::default();
            let slot = || gpu.alloc_image_slot().unwrap();
            let chain = ChainImages {
                rr_output: gpu.write_sampled(slot(), &rr_output, &view),
                rr_output_store: gpu.write_storage(slot(), &rr_output, &view),
                linear: gpu.write_sampled(slot(), &linear, &view),
                linear_store: gpu.write_storage(slot(), &linear, &view),
            };
            let pipeline = |entry, name| {
                let stage =
                    ShaderStage::new(&dof_scene::MODULE, entry, SpecData::default()).unwrap();
                gpu.create_compute_pipeline(&stage, name).unwrap()
            };
            let mut constants = gpu.gpu_malloc(MemoryKind::Upload, 1, "view").unwrap();
            let mut c: ViewConstants = bytemuck::Zeroable::zeroed();
            c.clip_from_view[0][0] = 1.0 / TAN_X;
            c.clip_from_view[1][1] = -1.0 / tan_y();
            c.clip_from_view[3][2] = -1.0;
            c.clip_from_view[2][3] = NEAR;
            c.extent = OUT;
            c.delta_time = 1.0 / 60.0;
            constants.as_mut_slice().unwrap()[0] = c;
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            for t in [&rr_output, &linear, &depth, &motion] {
                list.discard(t);
            }
            wait(queue.submit(vec![list], &[]).unwrap());
            Self {
                kernels: OutputKernels::new(gpu).unwrap(),
                output: OutputView::new(gpu, 2).unwrap(),
                scene: pipeline("frame", "dof scene"),
                reference: pipeline("reference", "dof reference"),
                root: gpu.gpu_malloc(MemoryKind::Upload, 1, "scene root").unwrap(),
                constants,
                depth: gpu.write_sampled(slot(), &depth, &view),
                depth_store: gpu.write_storage(slot(), &depth, &view),
                motion: gpu.write_sampled(slot(), &motion, &view),
                motion_store: gpu.write_storage(slot(), &motion, &view),
                chain,
                reference_out: gpu
                    .gpu_malloc(
                        MemoryKind::DeviceLocal,
                        4 * u64::from(OUT[0] * OUT[1]),
                        "reference",
                    )
                    .unwrap(),
                frame: 0,
                rr_output,
                linear,
                _targets: [depth, motion],
            }
        }

        fn put_root(&mut self, scene: Scene, dof: &DepthOfField) {
            let polygon = dof.blades >= 3 && dof.f_number > dof.round_at_f_number;
            self.root.as_mut_slice().unwrap()[0] = dof_scene::Root {
                reference: self.reference_out.gpu,
                radiance: self.chain.rr_output_store,
                depth: self.depth_store,
                motion: self.motion_store,
                width: OUT[0],
                height: OUT[1],
                scene: scene as u32,
                lens_samples: LENS_SAMPLES,
                tan_x: TAN_X,
                tan_y: tan_y(),
                near: NEAR,
                focus: FOCUS,
                lens_radius: lens_radius(dof),
                blades: if polygon { dof.blades } else { 0 },
                roundness: (dof.round_at_f_number / dof.f_number).min(1.0),
            };
        }

        /// The scene through `dof` on `path`: the image the chain ends on (on the guide
        /// path the blurred color for RR), the frame's pass names, and on the guide path the
        /// guide.
        fn render(
            &mut self,
            scene: Scene,
            dof: Option<DepthOfField>,
            path: DofPath,
        ) -> (Vec<[f32; 4]>, Vec<&'static str>, Option<Vec<f32>>) {
            let gpu = device();
            self.frame += 1;
            self.put_root(scene, &dof.unwrap_or(lens(0, 1.4)));
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            list.dispatch(&self.scene, self.root.gpu, groups(OUT));
            list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
            let mut output = ViewOutput::default();
            output.post.dof_path = path;
            let pass = DepthOfFieldPass {
                rr_depth: self.depth,
                rr_motion: self.motion,
                extent: OUT,
                view: self.constants.gpu,
                sensor_width_mm: SENSOR_MM,
                depth_of_field: dof,
                history_reset: false,
                frame: self.frame,
                output: &output,
            };
            let mut chain = self.chain;
            let recorded = match path {
                DofPath::AfterRr => self
                    .kernels
                    .depth_of_field(&mut list, &mut self.output, &pass, &mut chain)
                    .unwrap(),
                DofPath::Guide => self
                    .kernels
                    .depth_of_field_guide(&mut list, &mut self.output, &pass, chain.rr_output)
                    .unwrap(),
            };
            let point = queue.submit(vec![list], &[]).unwrap();
            wait(point);
            let names = gpu
                .pass_results(point)
                .unwrap()
                .iter()
                .map(|p| p.name)
                .collect();
            if path == DofPath::Guide && recorded {
                let images = self.output.depth_of_field_guide().unwrap();
                let guide = read_r16f(images.guide);
                return (read_rgba16f(images.color), names, Some(guide));
            }
            let image = if chain.rr_output == self.chain.linear {
                &self.linear
            } else {
                &self.rr_output
            };
            (read_rgba16f(image), names, None)
        }

        fn reference(&mut self, scene: Scene, dof: &DepthOfField) -> Vec<[f32; 4]> {
            let gpu = device();
            self.put_root(scene, dof);
            let queue = gpu.queue(QueueKind::Render).unwrap();
            let mut list = queue.command_list().unwrap();
            list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
            list.dispatch(&self.reference, self.root.gpu, groups(OUT));
            wait(queue.submit(vec![list], &[]).unwrap());
            read_buffer(self.reference_out.gpu, 4 * u64::from(OUT[0] * OUT[1]))
                .chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect()
        }
    }

    fn read_r16f(t: &Texture) -> Vec<f32> {
        read_texture(t, &TextureRegion::whole_mip(t.desc(), 0))
            .chunks_exact(2)
            .map(|p| half::f16::from_le_bytes([p[0], p[1]]).to_f32())
            .collect()
    }

    /// Reinhard per channel, then sRGB, so highlights keep their spread.
    fn display(linear: &[[f32; 4]]) -> Image {
        let pixels = linear
            .iter()
            .flat_map(|p| {
                let c = |v: f32| {
                    let v = f64::from(v.max(0.0));
                    (srgb_encode(v / (1.0 + v)) * 255.0).round() as u8
                };
                [c(p[0]), c(p[1]), c(p[2]), 255]
            })
            .collect();
        Image {
            width: OUT[0],
            height: OUT[1],
            pixels: Pixels::Rgba8(pixels),
        }
    }

    fn flip_against(reference: &[[f32; 4]], image: &[[f32; 4]]) -> FlipError {
        flip(&display(reference), &display(image))
    }

    /// FLIP of the blurred frame against the reference at most `mean` on both paths, and
    /// well under the sharp frame's.
    fn check(scene: Scene, dof: DepthOfField, mean: f32) {
        for path in [DofPath::AfterRr, DofPath::Guide] {
            let mut b = Bed::new();
            let reference = b.reference(scene, &dof);
            let (sharp, names, _) = b.render(scene, None, path);
            assert!(names.is_empty(), "{names:?}");
            let (blurred, names, _) = b.render(scene, Some(dof), path);
            let name = match path {
                DofPath::AfterRr => "depth_of_field",
                DofPath::Guide => "depth_of_field_guide",
            };
            assert_eq!(names, [name]);
            let sharp = flip_against(&reference, &sharp);
            let blurred = flip_against(&reference, &blurred);
            log::info!("{scene:?} {path:?}: FLIP blurred {blurred:?}, sharp {sharp:?}");
            assert!(blurred.mean <= mean, "{scene:?} {path:?}: {blurred:?}");
            assert!(
                blurred.mean < 0.6 * sharp.mean,
                "{scene:?} {path:?}: {blurred:?} against {sharp:?}"
            );
        }
        assert_clean();
    }

    #[test]
    fn foreground_over_focus_matches_lens_reference() {
        check(Scene::Foreground, lens(0, 1.4), 0.05);
    }

    #[test]
    fn focus_over_background_matches_lens_reference() {
        check(Scene::Background, lens(0, 1.4), 0.06);
    }

    #[test]
    fn highlights_through_foliage_match_lens_reference() {
        // Highlights hidden behind leaves in the pinhole image still reach the lens around
        // them; post-process depth of field cannot recover them.
        check(Scene::Highlights, lens(6, 1.0), 0.22);
    }

    /// The water plane and particles are blurred at the opaque depth behind them.
    #[test]
    fn water_and_particles_match_lens_reference() {
        check(Scene::Water, lens(0, 1.4), 0.06);
    }

    #[test]
    fn static_camera_holds_still_and_off_frees_images() {
        let mut b = Bed::new();
        let dof = Some(lens(6, 1.0));
        let (first, _, _) = b.render(Scene::Highlights, dof, DofPath::AfterRr);
        assert!(b.output.depth_of_field_bytes() > 0);
        // Sprites are added in whatever order they were found, so frames match to rounding.
        let still = |a: &[[f32; 4]], b: &[[f32; 4]]| {
            a.iter()
                .zip(b)
                .all(|(a, b)| (0..3).all(|c| (a[c] - b[c]).abs() <= 2e-3 * a[c].abs().max(1.0)))
        };
        for _ in 0..2 {
            let (next, _, _) = b.render(Scene::Highlights, dof, DofPath::AfterRr);
            assert!(still(&first, &next));
        }
        let (_, names, _) = b.render(Scene::Highlights, None, DofPath::AfterRr);
        assert!(names.is_empty(), "{names:?}");
        assert_eq!(b.output.depth_of_field_bytes(), 0);
        assert_clean();
    }

    #[test]
    fn autofocus_finds_the_surface_under_its_point() {
        let mut b = Bed::new();
        let (manual, _, _) = b.render(Scene::Foreground, Some(lens(0, 1.4)), DofPath::AfterRr);
        let auto = DepthOfField {
            focus: Focus::Auto {
                uv: [0.85, 0.8],
                speed: 4.0,
            },
            ..lens(0, 1.4)
        };
        let (focused, _, _) = b.render(Scene::Foreground, Some(auto), DofPath::AfterRr);
        let largest = manual
            .iter()
            .zip(&focused)
            .flat_map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).abs()))
            .fold(0.0, f32::max);
        assert!(largest < 0.01, "{largest}");
        assert_clean();
    }

    /// The guide is `luminance(after - before)` with luminance `(r + 2g + b) / 4`, 0 where
    /// nothing blurs.
    #[test]
    fn guide_holds_the_luminance_change() {
        let mut b = Bed::new();
        let (before, _, _) = b.render(Scene::Foreground, None, DofPath::Guide);
        let (after, _, guide) = b.render(Scene::Foreground, Some(lens(0, 1.4)), DofPath::Guide);
        let guide = guide.unwrap();
        let luminance = |p: &[f32; 4]| (p[0] + 2.0 * p[1] + p[2]) / 4.0;
        let mut unchanged = 0;
        for ((a, b), g) in after.iter().zip(&before).zip(&guide) {
            let expected = luminance(a) - luminance(b);
            assert!(
                (g - expected).abs() < 4e-3,
                "guide {g}, expected {expected}"
            );
            unchanged += usize::from(*g == 0.0);
        }
        assert!(
            unchanged > guide.len() / 4,
            "{unchanged} of {}",
            guide.len()
        );
        assert_clean();
    }
}
