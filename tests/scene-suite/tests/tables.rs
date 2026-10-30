//! Scene tables, uploads, groups, lights and views against what was written.

use glam::{DQuat, DVec3, Quat, UVec2, Vec2, Vec3, Vec4};
use orr_gpu::{GpuPtr, Hazards, MemoryKind, Stages};
use orr_scene::testing::{assert_clean, device, read_buffer, wait};
use orr_scene::{
    Affine3, CopyVariant, FrameDesc, FrameTime, GroupWriter, InstanceFlags, InstanceGpu,
    InstanceRange, InstancesDesc, Lens, Light, LightGpu, MaterialGpu, MaterialId, ModelId, Pose,
    Projection, RendererDesc, SceneDesc, SceneId, SceneRoot, Scenes, TransformGroupDesc,
    TransformGroupId, ViewConstants, ViewDesc, ViewTarget, WorldPos,
};
use scene_suite::pipeline;
use scene_suite::shaders::{placement, reconstruct};

fn frame<'a>(scenes: &'a [SceneId]) -> FrameDesc<'a> {
    FrameDesc {
        time: FrameTime {
            absolute: 0.0,
            delta: 1.0 / 60.0,
        },
        scenes,
        low_latency: None,
    }
}

fn transform(i: u32, salt: f32) -> Affine3 {
    let f = i as f32;
    Affine3 {
        rows: [
            [1.0, 0.0, salt, f * 0.5],
            [0.0, 1.0, 0.0, -f],
            [0.0, 0.0, 1.0, salt + f * 0.25],
        ],
    }
}

/// Runs one frame of `scene` with `write`, waits for it and returns the scene root.
fn run_frame(
    scenes: &mut Scenes,
    scene: SceneId,
    write: impl FnOnce(&mut orr_scene::SceneWriter<'_>),
) -> SceneRoot {
    let list = [scene];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    write(&mut f.scene(scene).unwrap());
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());
    read_buffer(root, 1)[0]
}

#[test]
fn instance_and_material_tables_read_back_as_written() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut mats = f.materials();
    let materials: Vec<(MaterialId, MaterialGpu)> = (0..3)
        .map(|i| {
            let mut m: MaterialGpu = bytemuck::Zeroable::zeroed();
            m.base_color = [i as f32, 0.5, 0.25];
            m.specular_ior = 1.5;
            m.surface = i;
            (mats.create(&m).unwrap(), m)
        })
        .collect();
    let mut w = f.scene(s).unwrap();
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 1000,
            writer: GroupWriter::Cpu,
            anchor: DVec3::new(100.0, 0.0, -50.0),
        })
        .unwrap();
    let pose_group = w
        .create_transform_group(&TransformGroupDesc {
            count: 64,
            writer: GroupWriter::Gpu,
            anchor: DVec3::ZERO,
        })
        .unwrap();
    for (i, t) in w.transforms(g, 0, 1000).unwrap().iter_mut().enumerate() {
        *t = transform(i as u32, 0.0);
    }
    let a = w
        .add_instances(&InstancesDesc {
            model: ModelId(5),
            copies: 600,
            transforms: g,
            first: 0,
            pose: Some(Pose {
                group: pose_group,
                index: 3,
                variants: 4,
                stride: 4,
            }),
            material_set: Some(12),
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            rt_range: 300.0,
            animation_range: 50.0,
            user_tag: 77,
        })
        .unwrap();
    let b = w
        .add_instances(&InstancesDesc {
            model: ModelId(9),
            copies: 400,
            transforms: g,
            first: 600,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER,
            rt_range: 0.0,
            animation_range: 0.0,
            user_tag: 3,
        })
        .unwrap();
    assert_eq!((a.first, b.first), (0, 600));
    // A later flags write over part of a range, and an earlier variant write overlapped
    // by a later one.
    w.set_instance_flags(
        InstanceRange {
            first: 100,
            count: 50,
        },
        InstanceFlags::CAST_ONLY,
    );
    for v in w.copy_variants(b, 10, 20).unwrap() {
        *v = CopyVariant {
            pose: 1,
            material_set: 2,
            section_mask: 0xf0f0,
        };
    }
    for v in w.copy_variants(b, 25, 10).unwrap() {
        *v = CopyVariant {
            pose: 7,
            material_set: 0,
            section_mask: 1,
        };
    }
    drop(w);
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());

    let root: SceneRoot = read_buffer(root, 1)[0];
    let got = read_buffer(root.instances, 1000);
    for (slot, inst) in got.iter().enumerate() {
        let slot = slot as u32;
        let variant = match slot {
            635.. => 0,
            625.. => u32::from_le_bytes([7, 0, 1, 0]),
            610.. => u32::from_le_bytes([1, 2, 0xf0, 0xf0]),
            _ => 0,
        };
        let expected = if slot < 600 {
            InstanceGpu {
                group: g.0,
                group_index: slot,
                model: 5,
                material_base: 12,
                pose_group: pose_group.0,
                pose_index: 3,
                first_part: u32::MAX,
                rt_range: 0x5cb0, // 300 as f16
                animation_range: 0x5240,
                flags: if (100..150).contains(&slot) { 4 } else { 3 },
                candidate_class: 0,
                pose_variants: 4,
                pose_stride: 4,
                _pad: 0,
                user_tag: 77,
                variant,
            }
        } else {
            InstanceGpu {
                group: g.0,
                group_index: slot,
                model: 9,
                material_base: u32::MAX,
                pose_group: 0,
                pose_index: 0,
                first_part: u32::MAX,
                rt_range: 0,
                animation_range: 0,
                flags: 1,
                candidate_class: 0,
                pose_variants: 0,
                pose_stride: 0,
                _pad: 0,
                user_tag: 3,
                variant,
            }
        };
        assert_eq!(
            bytemuck::bytes_of(inst),
            bytemuck::bytes_of(&expected),
            "slot {slot}: {inst:?}"
        );
    }
    let state = read_buffer(root.slot_state, 1000);
    assert!(state.iter().all(|s| bytemuck::bytes_of(s) == [0; 8]));
    let table = read_buffer(root.materials, 3);
    for ((id, m), got) in materials.iter().zip(&table) {
        assert_eq!(bytemuck::bytes_of(m), bytemuck::bytes_of(got), "{id:?}");
    }
    assert_clean();
}

#[test]
fn transforms_land_in_one_scatter_from_one_or_four_threads() {
    const N: u32 = 100_000;
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let mut groups = [TransformGroupId(0); 2];
    run_frame(&mut scenes, s, |w| {
        for g in &mut groups {
            *g = w
                .create_transform_group(&TransformGroupDesc {
                    count: N,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::ZERO,
                })
                .unwrap();
        }
    });
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    {
        let mut w = f.scene(s).unwrap();
        for (i, t) in w
            .transforms(groups[0], 0, N)
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *t = transform(i as u32, 1.0);
        }
        let all = w.transforms(groups[1], 0, N).unwrap();
        std::thread::scope(|scope| {
            for (q, quarter) in all.chunks_mut((N / 4) as usize).enumerate() {
                scope.spawn(move || {
                    for (i, t) in quarter.iter_mut().enumerate() {
                        *t = transform((q * (N / 4) as usize + i) as u32, 1.0);
                    }
                });
            }
        });
    }
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    let point = f.end(&[]).unwrap();
    wait(point);

    let root: SceneRoot = read_buffer(root, 1)[0];
    let records = read_buffer(root.groups, 3);
    let one = read_buffer::<Affine3>(records[groups[0].0 as usize].current.cast(), u64::from(N));
    let four = read_buffer::<Affine3>(records[groups[1].0 as usize].current.cast(), u64::from(N));
    assert!(
        one.iter()
            .enumerate()
            .all(|(i, t)| *t == transform(i as u32, 1.0))
    );
    assert_eq!(one, four, "four threads wrote a different table");

    // The frame's figures arrive once a later frame starts after it completed.
    let f = scenes.begin_frame(&frame(&list)).unwrap();
    f.end(&[]).unwrap();
    let stats = scenes.stats();
    assert_eq!(stats.scatter_dispatches, 1);
    let words = 2 * u64::from(N) * 12;
    assert!(stats.scatter_words >= words && stats.scatter_words < words + 1024);
    let upload = stats.gpu.iter().find(|p| p.name == "upload").unwrap();
    assert!(upload.stats.unwrap().compute >= words);
    assert!(stats.upload_bytes >= 2 * u64::from(N) * 48);
    assert_clean();
}

#[test]
fn previous_transforms_follow_writes() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let mut g = TransformGroupId(0);
    let write = |w: &mut orr_scene::SceneWriter<'_>, g: TransformGroupId, salt: f32| {
        for (i, t) in w.transforms(g, 0, 4).unwrap().iter_mut().enumerate() {
            *t = transform(i as u32, salt);
        }
    };
    let read = |root: &SceneRoot, g: TransformGroupId| {
        let r = read_buffer(root.groups.offset(u64::from(g.0)), 1)[0];
        let cur = read_buffer::<Affine3>(r.current.cast(), 4);
        let prev = read_buffer::<Affine3>(r.previous.cast(), 4);
        (r.current == r.previous, cur, prev)
    };
    let expect = |salt: f32| (0..4).map(|i| transform(i, salt)).collect::<Vec<_>>();

    let root = run_frame(&mut scenes, s, |w| {
        g = w
            .create_transform_group(&TransformGroupDesc {
                count: 4,
                writer: GroupWriter::Cpu,
                anchor: DVec3::ZERO,
            })
            .unwrap();
        write(w, g, 1.0);
    });
    let (same, cur, _) = read(&root, g);
    assert!(same, "a new group's previous is its current");
    assert_eq!(cur, expect(1.0));

    let root = run_frame(&mut scenes, s, |w| write(w, g, 2.0));
    let (same, cur, prev) = read(&root, g);
    assert!(!same);
    assert_eq!((cur, prev), (expect(2.0), expect(1.0)));

    let root = run_frame(&mut scenes, s, |_| {});
    let (same, cur, _) = read(&root, g);
    assert!(same, "an unwritten group keeps previous == current");
    assert_eq!(cur, expect(2.0));

    let root = run_frame(&mut scenes, s, |w| write(w, g, 3.0));
    let (_, cur, prev) = read(&root, g);
    assert_eq!((cur, prev), (expect(3.0), expect(2.0)));

    // Entries 1 and 2 teleport: their previous is their new transform, the others move.
    let root = run_frame(&mut scenes, s, |w| {
        w.transforms(g, 0, 1).unwrap()[0] = transform(0, 4.0);
        for (i, t) in w
            .transforms_without_motion(g, 1, 2)
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *t = transform(i as u32 + 1, 4.0);
        }
    });
    let (_, cur, prev) = read(&root, g);
    let mut moved = expect(4.0);
    moved[3] = transform(3, 3.0);
    assert_eq!(cur, moved);
    assert_eq!(prev, [transform(0, 3.0), moved[1], moved[2], moved[3]]);
    assert_clean();
}

#[test]
fn an_unlisted_scene_records_nothing() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let a = scenes.create_scene(&SceneDesc::default()).unwrap();
    let b = scenes.create_scene(&SceneDesc::default()).unwrap();
    let both = [a, b];
    let mut f = scenes.begin_frame(&frame(&both)).unwrap();
    let mut roots = Vec::new();
    f.update_scenes(&mut |c| roots.push(c.root)).unwrap();
    wait(f.end(&[]).unwrap());

    let only_b = [b];
    let mut f = scenes.begin_frame(&frame(&only_b)).unwrap();
    let frame_b = f.id();
    assert!(matches!(f.scene(a), Err(orr_scene::Error::NotListed(_))));
    let mut computed = Vec::new();
    f.update_scenes(&mut |c| computed.push(c.scene)).unwrap();
    wait(f.end(&[]).unwrap());
    assert_eq!(computed, [b]);
    let ra: SceneRoot = read_buffer(roots[0], 1)[0];
    let rb: SceneRoot = read_buffer(roots[1], 1)[0];
    assert_eq!(
        ra.frame_index,
        frame_b as u32 - 1,
        "the unlisted scene's root was written"
    );
    assert_eq!(rb.frame_index, frame_b as u32);

    let f = scenes.begin_frame(&frame(&[])).unwrap();
    f.end(&[]).unwrap();
    assert_eq!(scenes.stats().scenes_updated, 1);
    assert_clean();
}

#[test]
fn lights_move_with_the_render_origin() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let far = DVec3::new(20_000.25, 30.5, -9_000.125);
    let light = Light::Sphere {
        position: far,
        radius: 0.1,
        intensity: Vec3::new(1.0, 0.0, 0.0),
        cone: None,
    };
    let origin_of = |root: &SceneRoot| {
        DVec3::new(
            f64::from(root.origin_cell[0]),
            f64::from(root.origin_cell[1]),
            f64::from(root.origin_cell[2]),
        ) * 256.0
    };
    let rect_center = DVec3::new(-31_000.5, 12.0, 4_000.75);
    let rect = Light::Rect {
        center: rect_center,
        edge_x: Vec3::X,
        edge_y: Vec3::Z,
        luminance: Vec3::ONE,
    };
    let sun = Light::Directional(orr_scene::DirectionalLight {
        direction_to_light: Vec3::Y,
        illuminance: Vec3::splat(1e5),
        angular_radius: 0.0047,
        disc: orr_scene::DiscProfile::LimbDarkened,
    });
    let check = |root: &SceneRoot| {
        let l: Vec<LightGpu> = read_buffer(root.lights, 3);
        // Render positions equal the CPU's from the f64 inputs.
        assert_eq!(Vec3::from(l[0].position), (far - origin_of(root)).as_vec3());
        assert_eq!(
            Vec3::from(l[1].position),
            (rect_center - origin_of(root)).as_vec3()
        );
        assert_eq!((l[0].kind, l[1].kind, l[2].kind), (2, 3, 1));
        assert_eq!(l[2].position, [0.0; 3]);
        // Rec.709 red in Rec.2020.
        assert!(
            (l[0].color[0] - 0.627_403_9).abs() < 1e-6
                && (l[0].color[1] - 0.069_097_29).abs() < 1e-6
        );
    };
    // The lights are written against the old origin and the focus moves in the same frame.
    let root = run_frame(&mut scenes, s, |w| {
        w.set_light(orr_scene::LightId(0), Some(&light)).unwrap();
        w.set_light(orr_scene::LightId(1), Some(&rect)).unwrap();
        w.set_light(orr_scene::LightId(2), Some(&sun)).unwrap();
        w.set_focus(far + DVec3::new(10.0, 0.0, 0.0), 1000.0);
    });
    assert_eq!(origin_of(&root), DVec3::new(19_968.0, 0.0, -9_216.0));
    check(&root);
    // A focus 2 km away moves the origin and the lights without rewriting them.
    let root = run_frame(&mut scenes, s, |w| {
        w.set_focus(far + DVec3::new(2000.0, 0.0, 0.0), 1000.0);
    });
    assert_eq!(origin_of(&root).x, 22_016.0);
    check(&root);
    assert_clean();
}

#[test]
fn view_constants_project_and_reconstruct_render_positions() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = scenes.create_view_history().unwrap();
    let camera = DVec3::new(-12_345.5, 210.0, 8_000.75);
    let rotation = (DQuat::from_rotation_y(0.7) * DQuat::from_rotation_x(-0.2)).as_quat();
    let extent = UVec2::new(1920, 1080);
    let jitter = Vec2::new(0.3, -0.4);
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    f.scene(s).unwrap().set_focus(camera, 1000.0);
    f.update_scenes(&mut |_| {}).unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: s,
            history,
            position: camera,
            rotation,
            projection: Projection {
                fov_y: 1.0,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: extent,
            jitter,
            reset_history: true,
            target: ViewTarget::AfterRr,
        })
        .unwrap();

    // Render-space points in front of the camera, projected on the CPU with the
    // constants' own matrices.
    let vc: ViewConstants = {
        let point = f.end(&[]).unwrap();
        wait(point);
        read_buffer(view.view, 1)[0]
    };
    let root: SceneRoot = read_buffer(view.scene, 1)[0];
    let origin = WorldPos {
        cell: glam::IVec3::from_slice(&root.origin_cell[..3]),
        offset: Vec3::ZERO,
    };
    let eye = WorldPos::new(camera, 8).relative_to(origin, 8);
    assert_eq!(Vec3::from_slice(&vc.eye_render[..3]), eye);
    let rows = |m: [[f32; 4]; 4]| glam::Mat4::from_cols_array_2d(&m).transpose();
    let clip = rows(vc.clip_from_view_jittered);

    // A point on the view axis lands at the image centre plus the jitter, so each sample
    // sits at its pixel centre minus the jitter.
    let axis = clip * Vec4::new(0.0, 0.0, -10.0, 1.0);
    let pixel = (axis.truncate().truncate() / axis.w * 0.5 + 0.5) * extent.as_vec2();
    let centre = extent.as_vec2() * 0.5 + jitter;
    assert!(
        (pixel - centre).abs().max_element() < 1e-3,
        "{pixel} vs {centre}"
    );
    // Reverse Z, infinite far: depth 1 at the near plane.
    let near = clip * Vec4::new(0.0, 0.0, -0.1, 1.0);
    assert!((near.z / near.w - 1.0).abs() < 1e-6);

    let mut points = Vec::new();
    let mut samples = Vec::new();
    for i in 0..256 {
        let t = i as f32 / 256.0;
        let local = Vec3::new(
            (t * 37.0).sin() * 3.0,
            (t * 11.0).cos() * 2.0,
            -1.0 - 400.0 * t * t,
        );
        let render = rotation * local + eye;
        let c = clip * local.extend(1.0);
        let ndc = c.truncate() / c.w;
        samples.push([ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5, ndc.z, 0.0]);
        points.push(render);
    }
    let mut samples_buf = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Upload, 256, "samples")
        .unwrap();
    samples_buf
        .as_mut_slice()
        .unwrap()
        .copy_from_slice(&samples);
    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, 256, "out")
        .unwrap();
    let mut root_buf = d
        .gpu_malloc::<reconstruct::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root_buf.as_mut_slice().unwrap()[0] = reconstruct::Root {
        view: view.view,
        samples: samples_buf.gpu,
        out: out.gpu,
        count: 256,
        _pad: 0,
    };
    // The constants stay in the frame's upload window until a later frame reuses it.
    dispatch_once(&reconstruct::MODULE, root_buf.gpu, 4);
    for (got, want) in out.as_slice().unwrap().iter().zip(&points) {
        let got = Vec3::from_slice(&got[..3]);
        let err = (got - *want).length() / (*want - eye).length();
        assert!(err < 1e-4, "{got} vs {want}");
    }
    assert_clean();
}

#[test]
fn exposure_slots_alternate_per_render_of_a_view() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = scenes.create_view_history().unwrap();
    let list = [s];
    let mut slots = Vec::new();
    // The view renders on frames 1, 2 and 4 only.
    for frame_id in 1..=4 {
        let mut f = scenes.begin_frame(&frame(&list)).unwrap();
        if frame_id != 3 {
            let view = f
                .raster_view(&ViewDesc {
                    scene: s,
                    history,
                    position: DVec3::ZERO,
                    rotation: Quat::IDENTITY,
                    projection: Projection {
                        fov_y: 1.0,
                        near: 0.1,
                    },
                    lens: Lens::default(),
                    render_extent: UVec2::new(64, 64),
                    jitter: Vec2::ZERO,
                    reset_history: false,
                    target: ViewTarget::AfterRr,
                })
                .unwrap();
            wait(f.end(&[]).unwrap());
            let vc: ViewConstants = read_buffer(view.view, 1)[0];
            slots.push(vc.exposure_slot);
        } else {
            wait(f.end(&[]).unwrap());
        }
    }
    assert_eq!(slots, [0, 1, 0]);
    assert_clean();
}

fn dispatch_once<R>(module: &'static orr_gpu::ShaderModule, root: GpuPtr<R>, groups: u32) {
    let p = pipeline(module, "main");
    let render = device().queue(orr_gpu::QueueKind::Render).unwrap();
    let mut cmd = render.command_list().unwrap();
    cmd.dispatch(&p, root, [groups, 1, 1]);
    cmd.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![cmd], &[]).unwrap());
}

#[test]
fn copies_are_placed_relative_to_the_eye_from_cells() {
    const N: u32 = 64;
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes.create_scene(&SceneDesc::default()).unwrap();
    let history = scenes.create_view_history().unwrap();
    let anchor = DVec3::new(20_480.75, 12.0, -31_999.5);
    let camera = DVec3::new(20_300.125, 40.0, -31_500.0);
    let list = [s];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    {
        let mut w = f.scene(s).unwrap();
        let g = w
            .create_transform_group(&TransformGroupDesc {
                count: N,
                writer: GroupWriter::Cpu,
                anchor,
            })
            .unwrap();
        for (i, t) in w.transforms(g, 0, N).unwrap().iter_mut().enumerate() {
            *t = transform(i as u32, 0.125);
        }
        w.add_instances(&InstancesDesc {
            model: ModelId(0),
            copies: N,
            transforms: g,
            first: 0,
            pose: None,
            material_set: None,
            flags: InstanceFlags::RASTER,
            rt_range: 0.0,
            animation_range: 0.0,
            user_tag: 0,
        })
        .unwrap();
        w.set_focus(camera, 1000.0);
    }
    f.update_scenes(&mut |_| {}).unwrap();
    let view = f
        .raster_view(&ViewDesc {
            scene: s,
            history,
            position: camera,
            rotation: Quat::IDENTITY,
            projection: Projection {
                fov_y: 1.0,
                near: 0.1,
            },
            lens: Lens::default(),
            render_extent: UVec2::new(64, 64),
            jitter: Vec2::ZERO,
            reset_history: false,
            target: ViewTarget::AfterRr,
        })
        .unwrap();
    wait(f.end(&[]).unwrap());

    let out = d
        .gpu_malloc::<[f32; 4]>(MemoryKind::Readback, u64::from(2 * N), "placements")
        .unwrap();
    let mut root = d
        .gpu_malloc::<placement::Root>(MemoryKind::Upload, 1, "root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = placement::Root {
        scene: view.scene,
        view: view.view,
        out: out.gpu,
        count: N,
        _pad: 0,
    };
    dispatch_once(&placement::MODULE, root.gpu, 1);
    let eye = WorldPos::new(camera, 8);
    let base = WorldPos::new(anchor, 8).relative_to(eye, 8);
    for (i, got) in out.as_slice().unwrap().chunks(2).enumerate() {
        let t = transform(i as u32, 0.125).rows;
        let want = base + Vec3::new(t[0][3], t[1][3], t[2][3]);
        assert_eq!(Vec3::from_slice(&got[0][..3]), want, "copy {i}");
        assert_eq!(got[0][..3], got[1][..3], "a new group's previous transform");
        let exact = anchor - camera + DVec3::new(t[0][3].into(), t[1][3].into(), t[2][3].into());
        assert!((want.as_dvec3() - exact).abs().max_element() < 1e-4);
    }
    assert_clean();
}
