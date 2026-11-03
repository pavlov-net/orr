//! The frame through `orr::Renderer`: frames with no scene and no view, a scene frame with
//! a view, a consumer pipeline that hot reload replaces, views whose transient targets share
//! memory, the glare a view's lens asks for, and views while the path kernels compile.

use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use orr::{
    ArenaReport, Budgets, CompilingKernels, DirectionalLight, DiscProfile, Exposure, FrameDesc,
    FrameTime, Glare, GlareMethod, GroupWriter, HookPass, InstanceFlags, InstancesDesc, Lens,
    Light, LightId, LitView, LoadState, MountId, OutputDebug, PackSource, PathEntry, PipelineId,
    PostQuality, Projection, ReloadReport, Renderer, RendererDesc, SceneDesc, SceneId,
    TextureFilter, TransformGroupDesc, TransparencyPass, ViewDesc, ViewHooks, ViewLighting,
    ViewOutput, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Source};
use orr_gpu::{
    Format, GpuPtr, MemoryKind, SpecData, TextureDesc, TextureMemory, TextureRegion, TextureUsage,
};
use orr_math::{Affine3, DVec3, Quat, UVec2, Vec2, Vec3};
use orr_pack::material::Material;
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::{assert_clean, device, read_buffer, read_texture, wait};
use scene_suite::hot::fill;

#[test]
fn empty_frames_and_scene_frames_submit_in_order() {
    let desc = RendererDesc {
        path_entry: PathEntry::Compute,
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(device(), desc).unwrap();
    let mut ids = Vec::new();
    let mut last = None;
    for i in 0..4 {
        let f = r
            .begin_frame(FrameDesc {
                time: FrameTime {
                    absolute: f64::from(i) / 60.0,
                    delta: 1.0 / 60.0,
                },
                scenes: &[],
                display: None,
            })
            .unwrap();
        ids.push(f.id());
        last = Some(f.end().unwrap());
    }
    assert_eq!(ids, [1, 2, 3, 4]);

    let s = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let history = r.create_raster_view_history().unwrap();
    let list = [s];
    for i in 0..3 {
        let mut f = r
            .begin_frame(FrameDesc {
                time: FrameTime::default(),
                scenes: &list,
                display: None,
            })
            .unwrap();
        f.scene(s)
            .unwrap()
            .set_focus(DVec3::new(f64::from(i) * 700.0, 0.0, 0.0));
        f.update_scenes(&mut |_| {}).unwrap();
        f.raster_view(&ViewDesc {
            scene: s,
            history,
            position: DVec3::new(f64::from(i) * 700.0, 2.0, 0.0),
            rotation: Quat::IDENTITY,
            projection: Projection {
                fov_y: 1.0,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::new(1280, 720),
            output_extent: UVec2::new(1280, 720),
            jitter: Vec2::ZERO,
            camera_cut: false,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
        })
        .unwrap();
        last = Some(f.end().unwrap());
    }
    wait(last.unwrap());
    let f = r
        .begin_frame(FrameDesc {
            time: FrameTime::default(),
            scenes: &[],
            display: None,
        })
        .unwrap();
    wait(f.end().unwrap());
    let stats = r.stats();
    assert_eq!(stats.frame, 7);
    let steps: Vec<&str> = stats.cpu.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        steps,
        [
            "begin_frame",
            "write",
            "update_scenes",
            "raster_view",
            "end"
        ]
    );
    assert!(stats.gpu.iter().any(|p| p.name == "rebase"));
    assert_clean();
}

/// Writes a file's original text back when dropped.
struct Restore {
    path: PathBuf,
    text: String,
}

impl Drop for Restore {
    fn drop(&mut self) {
        fs::write(&self.path, &self.text).unwrap();
    }
}

const FILLED: u64 = 64;

/// One frame that dispatches `fill` into `out`: what hot reload applied at its step 1 and
/// the values it wrote.
fn fill_frame(
    r: &mut Renderer,
    s: SceneId,
    pipeline: PipelineId,
    out: GpuPtr<u32>,
) -> (Vec<ReloadReport>, Vec<u32>) {
    let list = [s];
    let mut f = r
        .begin_frame(FrameDesc {
            time: FrameTime::default(),
            scenes: &list,
            display: None,
        })
        .unwrap();
    let reloads = f.reloads().to_vec();
    f.update_scenes(&mut |c| {
        let root = c
            .put_root(&fill::Root {
                out,
                count: FILLED as u32,
                pad: 0,
            })
            .unwrap();
        c.cmd.dispatch(c.pipeline(pipeline), root, [1, 1, 1]);
    })
    .unwrap();
    wait(f.end().unwrap());
    (reloads, read_buffer(out, FILLED))
}

/// Writes `text` over the kernel and runs frames until hot reload reports on it.
fn edit(
    path: &Path,
    text: &str,
    r: &mut Renderer,
    s: SceneId,
    pipeline: PipelineId,
    out: GpuPtr<u32>,
) -> (ReloadReport, Vec<u32>) {
    fs::write(path, text).unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        let (reports, values) = fill_frame(r, s, pipeline, out);
        let report = reports.into_iter().find(|rep| match rep {
            ReloadReport::Swapped { kernel, .. } | ReloadReport::Rejected { kernel, .. } => {
                *kernel == fill::MODULE.name
            }
        });
        if let Some(report) = report {
            return (report, values);
        }
    }
    panic!("no reload of {} within 60 s", fill::MODULE.name);
}

#[test]
fn editing_a_kernel_body_swaps_its_pipeline_and_bad_edits_keep_the_old_one() {
    let path = Path::new(scene_suite::HOT_SRC).join("fill.slang");
    let original = fs::read_to_string(&path).unwrap();
    let _restore = Restore {
        path: path.clone(),
        text: original.clone(),
    };
    let mut r = Renderer::new(device(), RendererDesc::default()).unwrap();
    let loading = r.loading();
    let worker = std::thread::spawn({
        let loading = Arc::clone(&loading);
        move || loading.run_worker()
    });
    let pipeline = r
        .compute_pipeline(&fill::MODULE, "main", SpecData::default())
        .unwrap();
    let s = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let out = device()
        .gpu_malloc::<u32>(MemoryKind::DeviceLocal, FILLED, "fill")
        .unwrap();
    let (_, values) = fill_frame(&mut r, s, pipeline, out.gpu());
    assert!(values.iter().all(|&v| v == 1), "{values:?}");

    let two = original.replace("root.out[id.x] = 1;", "root.out[id.x] = 2;");
    assert_ne!(two, original);
    let (report, values) = edit(&path, &two, &mut r, s, pipeline, out.gpu());
    assert_eq!(
        report,
        ReloadReport::Swapped {
            kernel: "fill",
            pipelines: 1
        }
    );
    assert!(values.iter().all(|&v| v == 2), "{values:?}");

    let broken = two.replace("= 2;", "= 2 +;");
    let (report, values) = edit(&path, &broken, &mut r, s, pipeline, out.gpu());
    let ReloadReport::Rejected { reason, .. } = report else {
        panic!("a syntax error swapped the pipeline");
    };
    assert!(reason.contains("fill.slang:19:"), "{reason}");
    assert!(values.iter().all(|&v| v == 2), "{values:?}");

    let wider = two.replace(
        "uint pad;",
        "uint pad;
    float4 tint;",
    );
    let (report, values) = edit(&path, &wider, &mut r, s, pipeline, out.gpu());
    let ReloadReport::Rejected { reason, .. } = report else {
        panic!("a root edit swapped the pipeline");
    };
    assert!(reason.contains("rebuild required"), "{reason}");
    assert!(values.iter().all(|&v| v == 2), "{values:?}");

    loading.stop_workers();
    worker.join().unwrap();
    drop(r);
    assert_clean();
}

/// Triangles of the quad over `corners`, in order around its edge, facing `normal`.
fn quad(m: &mut MeshInput, corners: [Vec3; 4], normal: Vec3) {
    let first = m.positions.len() as u32;
    m.positions.extend(corners);
    m.normals.extend([normal; 4]);
    let [a, b, c, _] = corners;
    let order = if (b - a).cross(c - a).dot(normal) >= 0.0 {
        [0, 1, 2, 0, 2, 3]
    } else {
        [0, 2, 1, 0, 3, 2]
    };
    m.indices.extend(order.map(|i| first + i));
}

/// A 1 m cube centered on the origin.
fn cube() -> MeshInput {
    let mut cube = MeshInput::default();
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let n = Vec3::AXES[axis] * sign;
            let (u, v) = (Vec3::AXES[(axis + 1) % 3], Vec3::AXES[(axis + 2) % 3]);
            let (c, u, v) = (0.5 * n, 0.5 * u, 0.5 * v);
            quad(&mut cube, [c - u - v, c + u - v, c + u + v, c - u + v], n);
        }
    }
    cube
}

/// A pack of `models`, each a mesh with one material.
fn pack_of<const N: usize>(models: [(&str, MeshInput, Material); N]) -> Arc<[u8]> {
    let model = |(name, mut mesh, material): (&str, MeshInput, Material)| {
        mesh.sections = vec![Span {
            first: 0,
            count: mesh.indices.len() as u32,
        }];
        AssetSpec {
            name: name.into(),
            source: Source::Mesh {
                mesh,
                materials: vec![material],
                settings: MeshSettings::default(),
            },
        }
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            models.into_iter().map(model).collect(),
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0)
}

/// A pack of an 8 m floor and a 1 m box.
fn floor_and_box() -> Arc<[u8]> {
    let mut floor = MeshInput::default();
    let corners = [(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)];
    quad(
        &mut floor,
        corners.map(|(x, z)| Vec3::new(x, 0.0, z)),
        Vec3::Y,
    );
    let diffuse = |base_color| Material {
        base_color,
        specular_roughness: 0.5,
        ..Material::default()
    };
    pack_of([
        ("floor", floor, diffuse([0.5; 3])),
        ("box", cube(), diffuse([0.8, 0.2, 0.1])),
    ])
}

/// The EV100 at which [`lamp`] exposes to [`LAMP_EXPOSED`].
const LAMP_EV100: f32 = 14.0;
/// Past the display's white, so glare spreads most of it; the headless image clips it at 1.
const LAMP_EXPOSED: f32 = 8.0;

/// A pack of a 1 m cube that only emits.
fn lamp() -> Arc<[u8]> {
    pack_of([(
        "lamp",
        cube(),
        Material {
            base_color: [0.0; 3],
            emission_color: [1.0; 3],
            emission_luminance: LAMP_EXPOSED / orr::orr_output::exposure_from_ev100(LAMP_EV100),
            ..Material::default()
        },
    )])
}

fn time(frame: u32) -> FrameTime {
    FrameTime {
        absolute: f64::from(frame) / 60.0,
        delta: 1.0 / 60.0,
    }
}

/// Mounts `pack` and runs empty frames from `frame` on until it has loaded; something must
/// be running the loading jobs.
fn mount_pack(r: &mut Renderer, pack: &Arc<[u8]>, frame: &mut u32) -> MountId {
    let mount = r.mount(PackSource::Memory(Arc::clone(pack)));
    let start = Instant::now();
    while matches!(r.mount_state(mount), LoadState::Pending) {
        *frame += 1;
        let f = r
            .begin_frame(FrameDesc {
                time: time(*frame),
                scenes: &[],
                display: None,
            })
            .unwrap();
        wait(f.end().unwrap());
        assert!(start.elapsed() < Duration::from_secs(60), "mount pending");
    }
    if let LoadState::Failed(e) = r.mount_state(mount) {
        panic!("mount failed: {e}");
    }
    mount
}

/// Places `pack`'s floor and box, a sun and the focus in `scene`.
fn place_floor_and_box(f: &orr::Frame<'_>, scene: SceneId, models: [orr::ModelId; 2]) {
    let mut s = f.scene(scene).unwrap();
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: 2,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    let raised = Affine3::from_translation(Vec3::new(0.0, 0.5, 0.0));
    s.transforms(group, 0, 2)
        .unwrap()
        .copy_from_slice(&[Affine3::IDENTITY.into(), raised.into()]);
    for (first, model) in models.into_iter().enumerate() {
        s.add_instances(&InstancesDesc {
            model,
            copies: 1,
            transforms: group,
            first: first as u32,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 1000.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
    }
    let sun = DirectionalLight {
        direction_to_light: Vec3::new(0.3, 0.8, 0.5).normalize(),
        illuminance: Vec3::splat(50_000.0),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    };
    s.set_light(LightId(0), Some(&Light::Directional(sun)))
        .unwrap();
    s.set_focus(DVec3::ZERO);
}

/// The view of `scene` from `eye`, looking at the origin, rendered into `image`.
fn view_of<'a>(
    scene: SceneId,
    history: orr::ViewHistoryId,
    eye: DVec3,
    image: &'a orr_gpu::Texture,
) -> LitView<'a> {
    LitView {
        scene,
        history,
        position: eye,
        rotation: orr_math::camera::rh::view::look_at_quat(eye.as_vec3(), Vec3::ZERO, Vec3::Y)
            .inverse(),
        projection: Projection {
            fov_y: 1.0,
            near: 0.05,
        },
        lens: Lens::default(),
        camera_cut: false,
        not_rendering_scene: false,
        screenshot: None,
        texture_filter: TextureFilter::Hardware,
        target: ViewTarget::Texture(image),
    }
}

/// Four frames of `pack`'s floor and box through `render_view` into an RGBA16F texture, the
/// camera moving: the first and last without hooks, the others with every hook. Returns
/// each frame's image and the view's arena after a frame with hooks.
fn hooked_frames(pack: &Arc<[u8]>, alias: bool) -> (Vec<Vec<u8>>, ArenaReport) {
    let d = device();
    // Small cluster lists, so the per-pixel targets take most of the arena and share bytes.
    let desc = RendererDesc {
        alias_view_targets: alias,
        path_entry: PathEntry::Compute,
        budgets: Budgets {
            visible_clusters: 1024,
            cluster_work: 1024,
            ..Budgets::default()
        },
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(d, desc).unwrap();
    let loading = r.loading();
    let worker = {
        let l = Arc::clone(&loading);
        std::thread::spawn(move || l.run_worker())
    };
    let mut frame = 0;
    let mount = mount_pack(&mut r, pack, &mut frame);
    let models = ["floor", "box"].map(|name| r.model(mount, AssetId::from_name(name)).unwrap());
    let scene = r
        .create_scene(
            &SceneDesc {
                light_capacity: 1,
                ..SceneDesc::default()
            },
            &Default::default(),
        )
        .unwrap();
    let history = r.create_view_history().unwrap();
    r.wait_paths().unwrap();
    let [w, h] = [320, 240];
    let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
    let image = d
        .create_texture(
            &TextureDesc::d2(Format::Rgba16Float, w, h, usage),
            TextureMemory::Dedicated,
            "view image",
        )
        .unwrap();

    let mut images = Vec::new();
    let mut report = None;
    for i in 0..4 {
        frame += 1;
        let listed = [scene];
        let mut f = r
            .begin_frame(FrameDesc {
                time: time(frame),
                scenes: &listed,
                display: None,
            })
            .unwrap();
        if i == 0 {
            place_floor_and_box(&f, scene, models);
        }
        f.update_scenes(&mut |_| {}).unwrap();
        let hooked = i == 1 || i == 2;
        let mut transparency = |_: &mut TransparencyPass<'_, '_>, _| {};
        let mut scene_referred = |_: &mut HookPass<'_, '_>| {};
        let mut display_linear = |_: &mut HookPass<'_, '_>| {};
        let mut hooks = if hooked {
            ViewHooks {
                transparency: Some(&mut transparency),
                scene_referred: Some(&mut scene_referred),
                display_linear: Some(&mut display_linear),
                ui: None,
            }
        } else {
            ViewHooks::default()
        };
        let eye = DVec3::new(0.2 * f64::from(i), 2.0, 4.0);
        f.render_view(
            &view_of(scene, history, eye, &image),
            &ViewLighting::default(),
            &ViewOutput::default(),
            &mut hooks,
        )
        .unwrap();
        wait(f.end().unwrap());
        images.push(read_texture(
            &image,
            &TextureRegion::whole_mip(image.desc(), 0),
        ));
        if hooked {
            report = r.view_arena_report(history);
        }
    }
    loading.stop_workers();
    worker.join().unwrap();
    (images, report.unwrap())
}

/// Renders `pack`'s floor and box into an RGBA16F texture: `waiting` frames before the path kernels
/// have compiled, with the camera elsewhere and every hook set, then `lit` frames with the camera
/// moving. With `black`, the waiting frames render the view; without, they render nothing, which
/// any view may do. Returns the images of each kind.
fn frames_around_the_compile(
    pack: &Arc<[u8]>,
    waiting: u32,
    black: bool,
    lit: u32,
) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let d = device();
    let desc = RendererDesc {
        path_entry: PathEntry::Compute,
        try_cached_paths: false,
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(d, desc).unwrap();
    let loading = r.loading();
    let worker = {
        let l = Arc::clone(&loading);
        std::thread::spawn(move || l.run_worker())
    };
    let mut frame = 0;
    let mount = mount_pack(&mut r, pack, &mut frame);
    let models = ["floor", "box"].map(|name| r.model(mount, AssetId::from_name(name)).unwrap());
    loading.stop_workers();
    worker.join().unwrap();
    let scene = r
        .create_scene(
            &SceneDesc {
                light_capacity: 1,
                ..SceneDesc::default()
            },
            &Default::default(),
        )
        .unwrap();
    assert!(!r.paths_ready());
    let history = r.create_view_history().unwrap();
    assert_eq!(r.paths_outstanding() as usize, CompilingKernels::JOBS);
    let [w, h] = [320, 240];
    let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
    let image = d
        .create_texture(
            &TextureDesc::d2(Format::Rgba16Float, w, h, usage),
            TextureMemory::Dedicated,
            "view image",
        )
        .unwrap();
    let listed = [scene];
    let (mut before, mut after) = (Vec::new(), Vec::new());
    for i in 0..waiting + lit {
        if i == waiting {
            r.wait_paths().unwrap();
            assert!(r.paths_ready());
            assert_eq!(r.paths_outstanding(), 0);
        }
        frame += 1;
        let mut f = r
            .begin_frame(FrameDesc {
                time: time(frame),
                scenes: &listed,
                display: None,
            })
            .unwrap();
        assert_eq!(f.paths_ready(), i >= waiting);
        if i == 0 {
            place_floor_and_box(&f, scene, models);
        }
        f.update_scenes(&mut |_| {}).unwrap();
        let waited = i < waiting;
        if !waited || black {
            let (image_hooks, display_hooks) = (Cell::new(0), Cell::new(0));
            let mut transparency =
                |_: &mut TransparencyPass<'_, '_>, _| image_hooks.set(image_hooks.get() + 1);
            let mut scene_referred =
                |_: &mut HookPass<'_, '_>| image_hooks.set(image_hooks.get() + 1);
            let mut display_linear =
                |_: &mut HookPass<'_, '_>| display_hooks.set(display_hooks.get() + 1);
            let mut hooks = if waited {
                ViewHooks {
                    transparency: Some(&mut transparency),
                    scene_referred: Some(&mut scene_referred),
                    display_linear: Some(&mut display_linear),
                    ui: None,
                }
            } else {
                ViewHooks::default()
            };
            let eye = if waited {
                DVec3::new(-3.0, 1.0, 2.0)
            } else {
                DVec3::new(0.2 * f64::from(i - waiting), 2.0, 4.0)
            };
            let rendered = f
                .render_view(
                    &view_of(scene, history, eye, &image),
                    &ViewLighting::default(),
                    &ViewOutput::default(),
                    &mut hooks,
                )
                .unwrap();
            assert_eq!(rendered.compiling, waited);
            if waited {
                assert_eq!(image_hooks.get(), 0);
                assert_eq!(display_hooks.get(), 1);
            }
        }
        wait(f.end().unwrap());
        if !waited || black {
            let bytes = read_texture(&image, &TextureRegion::whole_mip(image.desc(), 0));
            if waited {
                before.push(bytes);
            } else {
                after.push(bytes);
            }
        }
    }
    (before, after)
}

#[test]
fn a_view_is_black_while_the_path_kernels_compile_and_then_renders_as_if_it_never_was() {
    let pack = floor_and_box();
    let (black, lit) = frames_around_the_compile(&pack, 3, true, 3);
    let (_, unwaited) = frames_around_the_compile(&pack, 3, false, 3);
    assert_eq!(black.len(), 3);
    for image in &black {
        // RGBA16F half floats: zero color and 1.0 alpha, as compose writes it.
        assert!(
            image.chunks(8).all(|t| t == [0, 0, 0, 0, 0, 0, 0x00, 0x3c]),
            "a view is not black while its kernels compile"
        );
    }
    let texel = &lit[0][..8];
    assert!(
        lit[0].chunks(8).any(|t| t != texel),
        "the view is one color"
    );
    for (i, (a, b)) in lit.iter().zip(&unwaited).enumerate() {
        assert!(a == b, "lit frame {i} differs after black frames");
    }
    assert_clean();
}

/// A path kernel that cannot compile: the device has no ray dispatch and the renderer asks for it.
#[test]
fn a_failed_path_compile_comes_back_from_wait_paths_and_render_view() {
    let d = orr_gpu::Device::new(&orr_gpu::DeviceDesc {
        name: "no ray dispatch",
        skip_extensions: &[c"VK_KHR_ray_tracing_pipeline"],
        ..orr_gpu::DeviceDesc::default()
    })
    .unwrap();
    let desc = RendererDesc {
        path_entry: PathEntry::RayDispatch,
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(&d, desc).unwrap();
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let history = r.create_view_history().unwrap();
    let failed = |e: orr::Error| match e {
        orr::Error::PathKernels(e) => assert!(
            matches!(&*e, orr::LightingError::Compile { kernel, .. } if *kernel == "opaque paths"),
            "{e}"
        ),
        e => panic!("{e}"),
    };
    failed(r.wait_paths().unwrap_err());
    assert!(!r.paths_ready());
    assert_eq!(r.paths_outstanding(), 0);
    failed(r.wait_paths().unwrap_err());
    let image = d
        .create_texture(
            &TextureDesc::d2(
                Format::Rgba16Float,
                64,
                64,
                TextureUsage::SAMPLED | TextureUsage::STORAGE,
            ),
            TextureMemory::Dedicated,
            "view image",
        )
        .unwrap();
    let listed = [scene];
    let mut f = r
        .begin_frame(FrameDesc {
            time: time(1),
            scenes: &listed,
            display: None,
        })
        .unwrap();
    assert!(!f.paths_ready());
    let eye = DVec3::new(0.0, 2.0, 4.0);
    let error = f
        .render_view(
            &view_of(scene, history, eye, &image),
            &ViewLighting::default(),
            &ViewOutput::default(),
            &mut ViewHooks::default(),
        )
        .err()
        .expect("render_view with no kernels");
    failed(error);
    f.end().unwrap();
    d.wait_submitted(Duration::from_secs(10)).unwrap();
}

#[test]
fn aliased_view_targets_render_the_same_images_in_less_memory() {
    let pack = floor_and_box();
    let (aliased, report) = hooked_frames(&pack, true);
    let (separate, unaliased) = hooked_frames(&pack, false);
    let texel = &aliased[0][..8];
    assert!(
        aliased[0].chunks(8).any(|t| t != texel),
        "the view renders one color"
    );
    for (i, (a, b)) in aliased.iter().zip(&separate).enumerate() {
        assert!(a == b, "frame {i} differs with aliasing");
    }
    assert_eq!(unaliased.bytes, unaliased.unaliased_bytes);
    assert_eq!(report.unaliased_bytes, unaliased.unaliased_bytes);
    assert!(
        report.bytes < report.unaliased_bytes,
        "aliasing saves nothing: {report:?}"
    );
    assert_clean();
}

#[test]
fn render_view_runs_the_glare_its_lens_asks_for() {
    let d = device();
    let desc = RendererDesc {
        path_entry: PathEntry::Compute,
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(d, desc).unwrap();
    let loading = r.loading();
    let worker = {
        let l = Arc::clone(&loading);
        std::thread::spawn(move || l.run_worker())
    };
    let mut frame = 0;
    let mount = mount_pack(&mut r, &lamp(), &mut frame);
    let lamp = r.model(mount, AssetId::from_name("lamp")).unwrap();
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let history = r.create_view_history().unwrap();
    r.wait_paths().unwrap();
    let [w, h] = [320u32, 240];
    let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
    let image = d
        .create_texture(
            &TextureDesc::d2(Format::Rgba16Float, w, h, usage),
            TextureMemory::Dedicated,
            "view image",
        )
        .unwrap();

    frame += 1;
    let listed = [scene];
    let mut f = r
        .begin_frame(FrameDesc {
            time: time(frame),
            scenes: &listed,
            display: None,
        })
        .unwrap();
    let mut s = f.scene(scene).unwrap();
    let group = s
        .create_transform_group(&TransformGroupDesc {
            count: 1,
            writer: GroupWriter::Cpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    s.transforms(group, 0, 1)
        .unwrap()
        .copy_from_slice(&[Affine3::IDENTITY.into()]);
    s.add_instances(&InstancesDesc {
        model: lamp,
        copies: 1,
        transforms: group,
        first: 0,
        pose: None,
        material_set: None,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 1000.0,
        animation_range: 0.0,
        user_tag: 0,
    })
    .unwrap();
    s.set_focus(DVec3::ZERO);
    drop(s);
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end().unwrap());

    // A 1 m lamp seen from 30 m on a black background, without GT7.
    let mut render = |lens, post, ev100| {
        frame += 1;
        let mut f = r
            .begin_frame(FrameDesc {
                time: time(frame),
                scenes: &listed,
                display: None,
            })
            .unwrap();
        f.update_scenes(&mut |_| {}).unwrap();
        let eye = DVec3::new(0.0, 0.0, 30.0);
        f.render_view(
            &LitView {
                lens,
                ..view_of(scene, history, eye, &image)
            },
            &ViewLighting::default(),
            &ViewOutput {
                exposure: Exposure::Manual { ev100 },
                post,
                debug: OutputDebug::Linear,
                ..ViewOutput::default()
            },
            &mut ViewHooks::default(),
        )
        .unwrap();
        f.end().unwrap()
    };
    // The image's linear green channel, row by row, once `point` has completed.
    let read = |point| -> Vec<f64> {
        wait(point);
        let texels = read_texture(&image, &TextureRegion::whole_mip(image.desc(), 0));
        bytemuck::pod_collect_to_vec::<u8, [u16; 4]>(&texels)
            .iter()
            .map(|t| orr_color::srgb_decode(f64::from(half::f16::from_bits(t[1]).to_f32())))
            .collect()
    };
    // 2.25 mm over 240 pixels is the pixel pitch glare's fit takes as its reference.
    let bare = Lens {
        sensor_height_mm: 2.25,
        glare: None,
        ..Lens::default()
    };
    let on = Lens {
        glare: Some(Glare { strength: 1.0 }),
        ..bare
    };
    let bands = PostQuality::default();
    let fft = PostQuality {
        glare: GlareMethod::Fft,
        ..bands
    };
    // The first frames settle the pre-exposure, which the radiance's rounding depends on.
    for _ in 0..3 {
        render(bare, bands, LAMP_EV100);
    }
    let off = read(render(bare, bands, LAMP_EV100));
    let glared = read(render(on, bands, LAMP_EV100));
    let double = Lens {
        glare: Some(Glare { strength: 2.0 }),
        ..on
    };
    let doubled = read(render(double, bands, LAMP_EV100));
    let coarse_lens = Lens {
        sensor_height_mm: 27.0,
        ..on
    };
    let coarse = read(render(coarse_lens, bands, LAMP_EV100));
    let convolved = read(render(on, fft, LAMP_EV100));
    // One stop darker this frame: glare must read the exposure this frame's adapt wrote.
    let darker = read(render(on, bands, LAMP_EV100 + 1.0));
    // Glare turned on and off with frames in flight: each off frame frees a chain they use.
    for i in 0..4 {
        let lens = if i % 2 == 0 { on } else { bare };
        render(lens, bands, LAMP_EV100);
    }
    let off_again = read(render(bare, bands, LAMP_EV100));
    loading.stop_workers();
    worker.join().unwrap();

    let halo = |image: &[f64]| -> f64 {
        let unlit = image.iter().zip(&off).filter(|(_, off)| **off == 0.0);
        unlit.map(|(v, _)| v).sum()
    };
    let near = |a: f64, b: f64, tolerance: f64| (a / b - 1.0).abs() < tolerance;
    let peak = off.iter().copied().fold(0.0, f64::max);
    assert!(near(peak, 1.0, 2e-3), "the lamp clips at {peak}");
    // The lamp's near face, 29.5 m away, is 7.4 pixels across, centered on a pixel corner.
    let lit = off.iter().filter(|&&v| v > 0.0).count();
    assert_eq!(lit, 8 * 8, "light outside the lamp without glare");
    assert!(off == off_again, "turning glare off leaves it in the image");

    // The near face's area in pixels times what it exposes to.
    let energy = 7.4 * 7.4 * f64::from(LAMP_EXPOSED);
    let base = halo(&glared);
    assert!(base > 0.05 * energy, "halo of {base} from energy {energy}");
    // A stop darker the lamp gives up 2.5 of 4 instead of 6.5 of 8, through a wider F-number; on
    // the frame before's exposure the halo would grow.
    let dimmer = halo(&darker);
    assert!(dimmer < 0.8 * base, "a stop darker: {dimmer} of {base}");
    let (twice, far, fft_halo) = (halo(&doubled), halo(&coarse), halo(&convolved));
    assert!(
        near(twice, 2.0 * base, 0.01),
        "strength 2: {twice} of {base}"
    );
    assert!(far < 0.5 * base, "pixels 12 times coarser: {far} of {base}");
    assert!(near(fft_halo, base, 0.05), "FFT: {fft_halo} of {base}");
    assert_clean();
}
