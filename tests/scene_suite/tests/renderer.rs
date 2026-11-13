//! The frame through `orr::Renderer`: frames with no scene and no view, a scene frame with
//! a view, a consumer pipeline that hot reload replaces, views whose transient targets share
//! memory, the glare a view's lens asks for, exposure and ray reconstruction's resets across a
//! camera cut, and views while the path kernels compile.

use std::assert_matches;
use std::cell::Cell;
use std::collections::HashMap;
use std::fs;
use std::mem::DropGuard;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use orr::orr_output::{
    ReconstructionBackend, ReconstructionFeature, ReconstructionInputs, ReconstructionMode,
    ReconstructionSupport, UpscaleQuality,
};
use orr::{
    ArenaReport, AutoExposure, DirectionalLight, DiscProfile, Exposure, FrameDesc, FrameTime,
    Glare, GlareMethod, GroupWriter, HookPass, InstanceFlags, InstancesDesc, Lens, Light, LightId,
    LitView, LoadState, MountId, OutputDebug, PackSource, PathEntry, PipelineId, PostQuality,
    Projection, ReloadReport, Renderer, RendererDesc, SceneDesc, SceneId, TextureFilter,
    TransformGroupDesc, TransparencyPass, ViewDesc, ViewHooks, ViewLighting, ViewOutput,
    ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Source};
use orr_gpu::{
    Allocation, CommandList, Format, GpuPtr, Hazards, MemoryKind, SpecData, Stages, TextureDesc,
    TextureMemory, TextureRegion, TextureUsage,
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
    let _restore = DropGuard::new((path.clone(), original.clone()), |(path, text)| {
        fs::write(path, text).unwrap();
    });
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
    s.set_light(LightId(0), Some(&sun())).unwrap();
    s.set_focus(DVec3::ZERO);
}

/// The light of [`place_floor_and_box`], the scene's only one.
fn sun() -> Light {
    Light::Directional(DirectionalLight {
        direction_to_light: Vec3::new(0.3, 0.8, 0.5).normalize(),
        illuminance: Vec3::splat(50_000.0),
        angular_radius: 4.65e-3,
        disc: DiscProfile::Uniform,
    })
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
    let desc = RendererDesc {
        alias_view_targets: alias,
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
    let mount = mount_pack(&mut r, pack, &mut frame);
    let models = ["floor", "box"].map(|name| r.model(mount, AssetId::from_name(name)).unwrap());
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
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
                opaque: None,
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
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    assert!(!r.kernels_ready());
    let history = r.create_view_history().unwrap();
    let kernels = r.paths_outstanding();
    assert!(kernels > 0);
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
            // A view that rendered black asked for its own path pipelines beside the kernels,
            // which the wait for the kernels leaves to the view's first lit render.
            let asked = black && waiting > 0;
            assert_eq!(r.paths_outstanding() > kernels, asked);
            r.wait_paths().unwrap();
            assert!(r.kernels_ready());
            assert_eq!(r.paths_outstanding() > 0, asked);
        }
        frame += 1;
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
                    opaque: None,
                }
            } else {
                ViewHooks::default()
            };
            let eye = if waited {
                DVec3::new(-3.0, 1.0, 2.0)
            } else {
                DVec3::new(0.2 * f64::from(i - waiting), 2.0, 4.0)
            };
            let view = view_of(scene, history, eye, &image);
            let lit = f
                .view_lit(
                    &view,
                    &ViewLighting::default(),
                    hooks.transparency.is_some(),
                )
                .unwrap();
            assert_eq!(lit, !waited);
            let rendered = f
                .render_view(
                    &view,
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

/// A copy fading between two steps of its chain is written with every triangle a candidate of
/// the rays that meet it. A scene of opaque materials says so from the moment a model has a
/// chain, which keeps candidate handling in the path pipelines made for it.
#[test]
fn a_chain_marks_its_scenes_for_ray_candidates() {
    let d = device();
    let mut r = Renderer::new(d, RendererDesc::default()).unwrap();
    let loading = r.loading();
    let worker = {
        let l = Arc::clone(&loading);
        std::thread::spawn(move || l.run_worker())
    };
    let mut frame = 0;
    let mount = mount_pack(&mut r, &floor_and_box(), &mut frame);
    let models = ["floor", "box"].map(|name| r.model(mount, AssetId::from_name(name)).unwrap());
    loading.stop_workers();
    worker.join().unwrap();
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let listed = [scene];
    let coverage = orr_scene::SceneFeatures::COVERAGE;
    let mut features = |r: &mut Renderer, place: bool| {
        frame += 1;
        let mut f = r
            .begin_frame(FrameDesc {
                time: time(frame),
                scenes: &listed,
                display: None,
            })
            .unwrap();
        if place {
            place_floor_and_box(&f, scene, models);
        }
        f.update_scenes(&mut |_| {}).unwrap();
        let features = f.scene_features(scene).unwrap();
        wait(f.end().unwrap());
        features
    };
    assert!(!features(&mut r, true).contains(coverage));
    r.set_chain(models[1], &[(models[0], 10.0)]).unwrap();
    assert!(features(&mut r, false).contains(coverage));
    assert_clean();
}

/// A copy fading between two steps of its chain is written with every triangle a candidate of
/// the rays that meet it. Path pipelines made for a scene of plain materials take them too: a
/// view on them renders each frame of the box's fade as a view of the same frames on the
/// pipelines that hold everything does, and runs the pipelines it asked for.
#[test]
fn a_fading_copy_stops_rays_under_the_pipelines_made_for_its_scene() {
    let pack = floor_and_box();
    let d = device();
    let desc = RendererDesc {
        path_entry: PathEntry::Compute,
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(d, desc).unwrap();
    let loading = r.loading();
    let workers: Vec<_> = Vec::from_fn(8, |_| {
        let l = Arc::clone(&loading);
        std::thread::spawn(move || l.run_worker())
    });
    let mut frame = 0;
    let mount = mount_pack(&mut r, &pack, &mut frame);
    let models = ["floor", "box"].map(|name| r.model(mount, AssetId::from_name(name)).unwrap());
    // A copy takes its step at once when it is first seen, so the box has its chain from the
    // start and is seen near before the focus moves away.
    r.set_chain(models[1], &[(models[0], 10.0)]).unwrap();
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    // One view on the pipelines that hold everything and one on those made for the scene.
    let views = [false, true].map(|specialize| {
        let history = r.create_view_history().unwrap();
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        let image = d
            .create_texture(
                &TextureDesc::d2(Format::Rgba16Float, 160, 120, usage),
                TextureMemory::Dedicated,
                "view image",
            )
            .unwrap();
        let lighting = ViewLighting {
            specialize,
            ..ViewLighting::default()
        };
        (history, image, lighting)
    });
    r.wait_paths().unwrap();
    let listed = [scene];
    let mut frames = Vec::new();
    for i in 0..14 {
        frame += 1;
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
        if i >= 2 {
            f.scene(scene)
                .unwrap()
                .set_focus(DVec3::new(0.0, 0.0, 100.0));
        }
        f.update_scenes(&mut |_| {}).unwrap();
        let eye = DVec3::new(0.0, 2.0, 4.0);
        for (history, image, lighting) in &views {
            let rendered = f
                .render_view(
                    &view_of(scene, *history, eye, image),
                    lighting,
                    &ViewOutput::default(),
                    &mut ViewHooks::default(),
                )
                .unwrap();
            assert!(
                rendered.exact_paths,
                "frame {i} ran path pipelines that hold more than its view asked for"
            );
        }
        wait(f.end().unwrap());
        frames.push(
            views.each_ref().map(|(_, image, _)| {
                read_texture(image, &TextureRegion::whole_mip(image.desc(), 0))
            }),
        );
    }
    drop(r);
    loading.stop_workers();
    for worker in workers {
        worker.join().unwrap();
    }
    // Half floats of the red channel.
    let red = |image: &[u8]| -> Vec<f32> {
        image
            .chunks(8)
            .map(|t| half::f16::from_le_bytes([t[0], t[1]]).to_f32())
            .collect()
    };
    let apart = |a: &[u8], b: &[u8]| -> f64 {
        let (a, b) = (red(a), red(b));
        let sum: f64 = a.iter().map(|&v| f64::from(v)).sum();
        let diff: f64 = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| f64::from((x - y).abs()))
            .sum();
        diff / sum
    };
    assert!(
        apart(&frames[1][0], &frames[13][0]) > 0.02,
        "the box took no farther step"
    );
    for (i, [general, exact]) in frames.iter().enumerate() {
        let d = apart(general, exact);
        assert!(d < 0.005, "frame {i} differs by {d} of its light");
    }
    assert_clean();
}

/// A path pipeline that cannot compile: the device has no ray dispatch and the renderer asks for
/// it. The pipelines compile when a view first asks for them, so the view's render reports it,
/// and every wait and render after it.
#[test]
fn a_failed_path_compile_comes_back_from_render_view_and_wait_paths() {
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
        orr::Error::PathKernels(e) => {
            assert_matches!(
                &*e,
                orr::LightingError::LateCompile { kernel, .. } if *kernel == "staged opaque shade",
                "{e}"
            );
        }
        e => panic!("{e}"),
    };
    r.wait_paths().unwrap();
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
    let eye = DVec3::new(0.0, 2.0, 4.0);
    let error = f
        .render_view(
            &view_of(scene, history, eye, &image),
            &ViewLighting::default(),
            &ViewOutput::default(),
            &mut ViewHooks::default(),
        )
        .err()
        .expect("render_view with a pipeline that failed");
    failed(error);
    f.end().unwrap();
    d.wait_submitted(Duration::from_secs(10)).unwrap();
    failed(r.wait_paths().unwrap_err());
    assert!(!r.kernels_ready());
    assert_eq!(r.paths_outstanding(), 0);
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

/// Places `pack`'s lamp at the origin of `scene`, alone, with the focus on it.
fn place_lamp(f: &orr::Frame<'_>, scene: SceneId, lamp: orr::ModelId) {
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
}

/// A view starts, and starts over at a camera cut, lit and shown at its exposure's reference,
/// and holds that until a frame has light to meter. Its pre-exposure then waits at the
/// reference for the frame that has read that back.
#[test]
fn a_camera_cut_starts_exposure_over_at_its_reference() {
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
    let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
    let image = d
        .create_texture(
            &TextureDesc::d2(Format::Rgba16Float, 320, 240, usage),
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
    place_lamp(&f, scene, lamp);
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end().unwrap());

    // One frame of the view from 30 m, facing the lamp or away from it into the black, and
    // the telemetry it leaves.
    let mut render = |facing: bool, camera_cut: bool, reference_ev100: f32| {
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
        let toward = if facing {
            Vec3::ZERO
        } else {
            2.0 * eye.as_vec3()
        };
        f.render_view(
            &LitView {
                rotation: orr_math::camera::rh::view::look_at_quat(eye.as_vec3(), toward, Vec3::Y)
                    .inverse(),
                camera_cut,
                ..view_of(scene, history, eye, &image)
            },
            &ViewLighting::default(),
            &ViewOutput {
                exposure: Exposure::Auto(AutoExposure {
                    reference_ev100,
                    ..AutoExposure::default()
                }),
                ..ViewOutput::default()
            },
            &mut ViewHooks::default(),
        )
        .unwrap();
        wait(f.end().unwrap());
        frame += 1;
        let f = r
            .begin_frame(FrameDesc {
                time: time(frame),
                scenes: &[],
                display: None,
            })
            .unwrap();
        let telemetry = f
            .exposure_telemetry(history)
            .expect("the frame's telemetry");
        wait(f.end().unwrap());
        telemetry
    };
    let exposure = orr::orr_output::exposure_from_ev100;
    // The view's first render, with nothing to meter.
    let unset = render(false, false, 6.0);
    assert_eq!(unset.metered_ev100, None);
    assert_eq!(
        (unset.adapted_ev100, unset.pre_exposure),
        (6.0, exposure(6.0))
    );
    let lit = render(true, false, 6.0);
    assert!(lit.metered_ev100.is_some());
    assert!(lit.adapted_ev100 > 10.0, "the lamp exposes at {lit:?}");
    assert_eq!(lit.pre_exposure, exposure(6.0));
    let held = render(false, false, 9.0);
    assert_eq!(held.adapted_ev100, lit.adapted_ev100);
    let follows = held.pre_exposure / exposure(held.adapted_ev100) - 1.0;
    assert!(
        follows.abs() < 1e-5,
        "pre-exposure {follows} from the exposure"
    );
    let cut = render(false, true, 9.0);
    assert_eq!(cut.metered_ev100, None);
    assert_eq!((cut.adapted_ev100, cut.pre_exposure), (9.0, exposure(9.0)));
    loading.stop_workers();
    worker.join().unwrap();
    assert_clean();
}

/// Ray reconstruction's stand-in at the output size: it passes the radiance through and
/// notes each evaluation's reset.
struct Passthrough(Arc<Mutex<Vec<bool>>>);

struct PassthroughFeature {
    size: [u32; 2],
    texels: Allocation<[u16; 4]>,
    resets: Arc<Mutex<Vec<bool>>>,
}

impl ReconstructionBackend for Passthrough {
    fn support(&self) -> ReconstructionSupport {
        ReconstructionSupport {
            available: true,
            dlaa: false,
            min_output: [1, 1],
            scale: None,
        }
    }

    fn feature(
        &self,
        output: [u32; 2],
        _: UpscaleQuality,
    ) -> orr::orr_output::Result<Box<dyn ReconstructionFeature>> {
        self.feature_at(output, output)
    }

    fn feature_at(
        &self,
        _: [u32; 2],
        output: [u32; 2],
    ) -> orr::orr_output::Result<Box<dyn ReconstructionFeature>> {
        let texels = u64::from(output[0] * output[1]);
        Ok(Box::new(PassthroughFeature {
            size: output,
            texels: device().gpu_malloc(MemoryKind::DeviceLocal, texels, "passthrough")?,
            resets: Arc::clone(&self.0),
        }))
    }
}

impl ReconstructionFeature for PassthroughFeature {
    fn mode(&self) -> ReconstructionMode {
        ReconstructionMode::RenderExtent
    }

    fn render_size(&self) -> [u32; 2] {
        self.size
    }

    fn output_size(&self) -> [u32; 2] {
        self.size
    }

    fn jitter(&self, _: u64) -> [f32; 2] {
        [0.0; 2]
    }

    fn mip_bias(&self) -> f32 {
        0.0
    }

    fn record(
        &mut self,
        list: &mut CommandList<'_>,
        inputs: &ReconstructionInputs<'_>,
    ) -> orr::orr_output::Result<()> {
        self.resets.lock().unwrap().push(inputs.reset);
        let whole = TextureRegion::whole_mip(inputs.color.desc(), 0);
        list.copy_from_texture(self.texels.gpu().cast(), inputs.color, &whole);
        list.barrier(Stages::TRANSFER, Stages::TRANSFER, Hazards::empty());
        list.copy_to_texture(inputs.output, &whole, self.texels.gpu().cast());
        Ok(())
    }
}

/// One frame of a sequence: whether the scene's light is on, whether the frame is a camera
/// cut, and the manual exposure in place of auto exposure.
#[derive(Clone, Copy)]
struct Shot {
    lit: bool,
    cut: bool,
    manual: Option<f32>,
}

/// Ray reconstruction drops its history on a camera cut's frame and on the frame lit with
/// the step from the reference to the exposure the view sets, and on no other, with every
/// frame waited on and with frames left in flight. Each case gives its frames, then per
/// frame whether ray reconstruction resets (`?`: either, the frame after a cut whose
/// telemetry may not be back; `a`: on exactly one of them, by when the readback lands), then
/// the frames lit with a step when each frame's telemetry is back before the next.
#[test]
fn ray_reconstruction_resets_on_the_frame_pre_exposure_steps() {
    const MANUAL_EV100: f32 = 10.0;
    let auto = |lit| Shot {
        lit,
        cut: false,
        manual: None,
    };
    let manual = Shot {
        lit: true,
        cut: false,
        manual: Some(MANUAL_EV100),
    };
    let cut = |shot: Shot| Shot { cut: true, ..shot };
    let cases: [(&str, Vec<Shot>, &str, &str); 6] = [
        (
            "a cut with light",
            vec![cut(auto(true)), auto(true), auto(true), auto(true)],
            "TTFF",
            "FTFF",
        ),
        (
            "a cut into black, then light",
            vec![
                cut(auto(false)),
                auto(false),
                auto(true),
                auto(true),
                auto(true),
                auto(true),
                auto(true),
            ],
            "T?FFaaF",
            "FFFFTFF",
        ),
        (
            "a manual cut",
            vec![cut(manual), manual, manual],
            "TFF",
            "FFF",
        ),
        (
            "manual exposure over an unset view",
            vec![cut(auto(false)), manual, manual, manual],
            "T?TF",
            "FFTF",
        ),
        (
            "manual exposure over a view whose pre-exposure waits",
            vec![cut(auto(false)), auto(true), manual, manual, manual],
            "T?FTF",
            "FFFTF",
        ),
        // The step to the manual exposure is no cut's: nothing resets on it.
        (
            "manual exposure over an adapting view",
            vec![
                cut(auto(true)),
                auto(true),
                auto(true),
                auto(true),
                manual,
                manual,
            ],
            "TTFFFF",
            "FTFFFT",
        ),
    ];

    let d = device();
    let desc = RendererDesc {
        path_entry: PathEntry::Compute,
        ..RendererDesc::default()
    };
    let mut r = Renderer::new(d, desc).unwrap();
    let resets = Arc::new(Mutex::new(Vec::new()));
    assert!(r.set_reconstruction(Passthrough(Arc::clone(&resets))));
    let loading = r.loading();
    let worker = {
        let l = Arc::clone(&loading);
        std::thread::spawn(move || l.run_worker())
    };
    let mut frame = 0;
    let mount = mount_pack(&mut r, &floor_and_box(), &mut frame);
    let models = ["floor", "box"].map(|name| r.model(mount, AssetId::from_name(name)).unwrap());
    let scene = r
        .create_scene(&SceneDesc::default(), &Default::default())
        .unwrap();
    let history = r.create_view_history().unwrap();
    r.wait_paths().unwrap();
    let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
    let image = d
        .create_texture(
            &TextureDesc::d2(Format::Rgba16Float, 320, 240, usage),
            TextureMemory::Dedicated,
            "view image",
        )
        .unwrap();
    frame += 1;
    let listed = [scene];
    let f = r
        .begin_frame(FrameDesc {
            time: time(frame),
            scenes: &listed,
            display: None,
        })
        .unwrap();
    place_floor_and_box(&f, scene, models);
    wait(f.end().unwrap());

    let exposure = |shot: &Shot| match shot.manual {
        Some(ev100) => Exposure::Manual { ev100 },
        None => Exposure::default(),
    };
    let mut last = None;
    for waited in [true, false] {
        for (name, shots, reset, stepped) in &cases {
            resets.lock().unwrap().clear();
            // Each frame's id, and by frame id the pre-exposure its telemetry leaves for the
            // next.
            let mut ids = Vec::new();
            let mut next_pre_exposure = HashMap::new();
            for shot in shots {
                frame += 1;
                let mut f = r
                    .begin_frame(FrameDesc {
                        time: time(frame),
                        scenes: &listed,
                        display: None,
                    })
                    .unwrap();
                if let Some(t) = f.exposure_telemetry(history) {
                    next_pre_exposure.insert(t.frame, t.pre_exposure);
                }
                ids.push(f.id());
                let light = shot.lit.then(sun);
                f.scene(scene)
                    .unwrap()
                    .set_light(LightId(0), light.as_ref())
                    .unwrap();
                f.update_scenes(&mut |_| {}).unwrap();
                f.render_view(
                    &LitView {
                        camera_cut: shot.cut,
                        ..view_of(scene, history, DVec3::new(3.0, 2.5, 5.0), &image)
                    },
                    &ViewLighting::default(),
                    &ViewOutput {
                        exposure: exposure(shot),
                        ..ViewOutput::default()
                    },
                    &mut ViewHooks::default(),
                )
                .unwrap();
                let point = f.end().unwrap();
                if waited {
                    wait(point);
                }
                last = Some(point);
            }
            let got: String = resets
                .lock()
                .unwrap()
                .iter()
                .map(|&r| if r { 'T' } else { 'F' })
                .collect();
            let either = got
                .chars()
                .zip(reset.chars())
                .filter(|(_, want)| *want == 'a')
                .filter(|(got, _)| *got == 'T')
                .count();
            let fits = got.len() == reset.len()
                && got
                    .chars()
                    .zip(reset.chars())
                    .all(|(got, want)| matches!(want, '?' | 'a') || got == want)
                && either == usize::from(reset.contains('a'));
            assert!(
                fits,
                "{name}, waited {waited}: resets {got} against {reset}"
            );
            if !waited {
                continue;
            }
            // Waited on, a frame's telemetry is back before the next, which is lit with it;
            // a cut's frame is lit as its exposure starts.
            let lit: Vec<f32> = Vec::from_fn(shots.len(), |i| match shots[i].cut {
                true => exposure(&shots[i]).start().pre_exposure,
                false => next_pre_exposure[&ids[i - 1]],
            });
            let steps: String = (0..shots.len())
                .map(|i| i > 0 && (lit[i] / lit[i - 1]).log2().abs() > 0.5)
                .map(|step| if step { 'T' } else { 'F' })
                .collect();
            assert_eq!(steps, *stepped, "{name}: steps");
            // Every reset but a cut's lands on a step, the reads of `?` and `a` included.
            for (i, shot) in shots.iter().enumerate() {
                let reset = got.as_bytes()[i] == b'T';
                let step = steps.as_bytes()[i] == b'T';
                assert!(
                    !reset || shot.cut || step,
                    "{name}: frame {i} resets with no step"
                );
            }
        }
    }
    wait(last.expect("a frame"));
    loading.stop_workers();
    worker.join().unwrap();
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
    place_lamp(&f, scene, lamp);
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
