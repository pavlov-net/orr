//! The frame through `orr::Renderer`: frames with no scene and no view, a scene frame with
//! a view, a consumer pipeline that hot reload replaces, and views whose transient targets
//! share memory.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{DVec3, Mat3, Quat, UVec2, Vec2, Vec3};
use orr::{
    Affine3, ArenaReport, Budgets, DirectionalLight, DiscProfile, FrameDesc, FrameTime,
    GroupWriter, HookPass, InstanceFlags, InstancesDesc, Lens, Light, LightId, LoadState,
    PackSource, PipelineId, Projection, ReloadReport, Renderer, RendererDesc, SceneDesc, SceneId,
    TextureFilter, TransformGroupDesc, TransparencyPass, ViewDesc, ViewHooks, ViewLighting,
    ViewOutput, ViewTarget,
};
use orr_cook::{ArchiveSpec, AssetSpec, Manifest, MeshInput, MeshSettings, Source};
use orr_gpu::{
    Format, GpuPtr, MemoryKind, SpecData, TextureDesc, TextureMemory, TextureRegion, TextureUsage,
};
use orr_pack::material::Material;
use orr_pack::{AssetId, Span, Target};
use orr_scene::testing::{assert_clean, device, read_buffer, read_texture, wait};
use scene_suite::hot::fill;

#[test]
fn empty_frames_and_scene_frames_submit_in_order() {
    let mut r = Renderer::new(device(), RendererDesc::default()).unwrap();
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
    let history = r.create_view_history().unwrap();
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
            .set_focus(DVec3::new(f64::from(i) * 700.0, 0.0, 0.0), 1000.0);
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
            jitter: Vec2::ZERO,
            reset_history: false,
            mip_bias: 0.0,
            texture_filter: TextureFilter::Hardware,
            target: ViewTarget::AfterRr,
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
    let (_, values) = fill_frame(&mut r, s, pipeline, out.gpu);
    assert!(values.iter().all(|&v| v == 1), "{values:?}");

    let two = original.replace("root.out[id.x] = 1;", "root.out[id.x] = 2;");
    assert_ne!(two, original);
    let (report, values) = edit(&path, &two, &mut r, s, pipeline, out.gpu);
    assert_eq!(
        report,
        ReloadReport::Swapped {
            kernel: "fill",
            pipelines: 1
        }
    );
    assert!(values.iter().all(|&v| v == 2), "{values:?}");

    let broken = two.replace("= 2;", "= 2 +;");
    let (report, values) = edit(&path, &broken, &mut r, s, pipeline, out.gpu);
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
    let (report, values) = edit(&path, &wider, &mut r, s, pipeline, out.gpu);
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

/// A pack of an 8 m floor and a 1 m box.
fn floor_and_box() -> Arc<[u8]> {
    let mut floor = MeshInput::default();
    let corners = [(-4.0, -4.0), (4.0, -4.0), (4.0, 4.0), (-4.0, 4.0)];
    quad(
        &mut floor,
        corners.map(|(x, z)| Vec3::new(x, 0.0, z)),
        Vec3::Y,
    );
    let mut cube = MeshInput::default();
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let n = Vec3::AXES[axis] * sign;
            let (u, v) = (Vec3::AXES[(axis + 1) % 3], Vec3::AXES[(axis + 2) % 3]);
            let (c, u, v) = (0.5 * n, 0.5 * u, 0.5 * v);
            quad(&mut cube, [c - u - v, c + u - v, c + u + v, c - u + v], n);
        }
    }
    let model = |name: &str, mut mesh: MeshInput, base_color| {
        mesh.sections = vec![Span {
            first: 0,
            count: mesh.indices.len() as u32,
        }];
        AssetSpec {
            name: name.into(),
            source: Source::Mesh {
                mesh,
                materials: vec![Material {
                    base_color,
                    specular_roughness: 0.5,
                    ..Material::default()
                }],
                settings: MeshSettings::default(),
            },
        }
    };
    let manifest = Manifest {
        archives: vec![ArchiveSpec::new(
            Target::Memory,
            vec![
                model("floor", floor, [0.5; 3]),
                model("box", cube, [0.8, 0.2, 0.1]),
            ],
        )],
    };
    orr_cook::cook(&manifest, &Default::default())
        .unwrap()
        .memory
        .remove(0)
}

/// World from view for a camera at `eye` looking at the origin, y up.
fn look_at_origin(eye: DVec3) -> Quat {
    let back = eye.as_vec3().normalize();
    let right = Vec3::Y.cross(back).normalize();
    Quat::from_mat3(&Mat3::from_cols(right, back.cross(right), back))
}

fn time(frame: u32) -> FrameTime {
    FrameTime {
        absolute: f64::from(frame) / 60.0,
        delta: 1.0 / 60.0,
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
    let mount = r.mount(PackSource::Memory(Arc::clone(pack)));
    let start = Instant::now();
    let mut frame = 0;
    while matches!(r.mount_state(mount), LoadState::Pending) {
        frame += 1;
        let f = r
            .begin_frame(FrameDesc {
                time: time(frame),
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
            let mut s = f.scene(scene).unwrap();
            let group = s
                .create_transform_group(&TransformGroupDesc {
                    count: 2,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .unwrap();
            let raised = Affine3 {
                rows: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            };
            s.transforms(group, 0, 2)
                .unwrap()
                .copy_from_slice(&[Affine3::IDENTITY, raised]);
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
            s.set_focus(DVec3::ZERO, 100.0);
        }
        f.update_scenes(&mut |_| {}).unwrap();
        if i == 0 {
            f.list().unwrap().discard(&image);
        }
        let hooked = i == 1 || i == 2;
        let mut transparency = |_: &mut TransparencyPass<'_, '_>| {};
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
            &ViewDesc {
                scene,
                history,
                position: eye,
                rotation: look_at_origin(eye),
                projection: Projection {
                    fov_y: 1.0,
                    near: 0.05,
                },
                lens: Lens::default(),
                render_extent: UVec2::new(w, h),
                jitter: Vec2::ZERO,
                reset_history: false,
                mip_bias: 0.0,
                texture_filter: TextureFilter::Hardware,
                target: ViewTarget::Texture(&image),
            },
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
