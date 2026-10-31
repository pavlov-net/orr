//! The emissive list: instance spans and producer entries, their power summed on the GPU.

use glam::{DMat3, DVec3, Mat3, Vec3};
use orr_gpu::{GpuPtr, MemoryKind};
use orr_scene::testing::{assert_clean, device, read_buffer, set_model_emission, wait};
use orr_scene::{
    Affine3, ClusterTopology, CopyVariant, EmissiveGpu, EmissiveSpan, FrameDesc, FrameTime,
    GroupWriter, InstanceFlags, InstanceRange, InstancesDesc, MaterialGpu, MaterialId,
    ModelEmission, ModelId, ProducerDesc, ProducerId, ProducerUpdate, RendererDesc, SceneDesc,
    SceneId, SceneRoot, Scenes, TransformGroupDesc, TransformGroupId, material_flags,
};
use orr_shaders::bytemuck::Zeroable;
use scene_suite::pipeline;
use scene_suite::shaders::emit;

const PRODUCERS: u32 = 4;

fn frame(scenes: &[SceneId]) -> FrameDesc<'_> {
    FrameDesc {
        time: FrameTime {
            absolute: 0.0,
            delta: 1.0 / 60.0,
        },
        scenes,
        low_latency: None,
    }
}

fn emitting(color: [f32; 3], luminance: f32) -> MaterialGpu {
    let mut m = MaterialGpu::zeroed();
    m.emission_color = color;
    m.emission_luminance = luminance;
    m.flags = material_flags::EMISSIVE;
    m
}

fn affine(m: Mat3, t: Vec3) -> Affine3 {
    let r = |i: usize| {
        let row = m.row(i);
        [row.x, row.y, row.z, t[i]]
    };
    Affine3 {
        rows: [r(0), r(1), r(2)],
    }
}

fn entries(root: &SceneRoot) -> Vec<EmissiveGpu> {
    read_buffer(
        root.emissive,
        u64::from(root.producer_capacity + root.emissive_capacity),
    )
}

fn close(got: f32, expected: f64, what: &str) {
    let tolerance = 1e-3 * expected.abs().max(1e-6);
    assert!(
        (f64::from(got) - expected).abs() <= tolerance,
        "{what}: {got} against {expected}"
    );
}

fn close3(got: [f32; 3], expected: DVec3, what: &str) {
    for i in 0..3 {
        close(got[i], expected[i], what);
    }
}

#[test]
fn instance_spans_follow_scale_material_sets_hidden_sections_and_removal() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes
        .create_scene(&SceneDesc {
            producer_capacity: PRODUCERS,
            emissive_capacity: 12,
            ..SceneDesc::default()
        })
        .unwrap();
    let spans = [
        EmissiveSpan {
            section: 0,
            first_cluster: 4,
            cluster_count: 3,
            area: 1.5,
        },
        EmissiveSpan {
            section: 1,
            first_cluster: 9,
            cluster_count: 2,
            area: 2.0,
        },
    ];
    let (red, blue, green) = ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    let list = [s];
    let f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut mats = f.materials();
    let lamp = mats.create(&emitting(red, 100.0)).unwrap();
    let shade = mats.create(&emitting(blue, 10.0)).unwrap();
    let other = mats.create(&emitting(green, 1000.0)).unwrap();
    let dark = mats.create(&MaterialGpu::zeroed()).unwrap();
    wait(f.end(&[]).unwrap());
    set_model_emission(
        &mut scenes,
        ModelId(3),
        Some(ModelEmission {
            sections: 2,
            materials: vec![lamp, shade],
            spans: spans.to_vec(),
        }),
    );

    let scales = [
        Mat3::IDENTITY,
        Mat3::from_diagonal(Vec3::splat(2.0)),
        Mat3::from_diagonal(Vec3::new(2.0, 1.0, 0.5)),
        Mat3::from_rotation_y(0.7) * Mat3::from_diagonal(Vec3::splat(3.0)),
        Mat3::IDENTITY,
    ];
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut w = f.scene(s).unwrap();
    let g = w
        .create_transform_group(&TransformGroupDesc {
            count: 8,
            writer: GroupWriter::Cpu,
            anchor: DVec3::new(5000.0, 0.0, 0.0),
        })
        .unwrap();
    for (t, m) in w.transforms(g, 0, 5).unwrap().iter_mut().zip(scales) {
        *t = affine(m, Vec3::new(1.0, 2.0, 3.0));
    }
    let desc = |model, copies, material_set| InstancesDesc {
        model: ModelId(model),
        copies,
        transforms: g,
        first: 0,
        pose: None,
        material_set,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        rt_range: 100.0,
        animation_range: 0.0,
        user_tag: 0,
    };
    // A model without emission adds no entries.
    w.add_instances(&desc(9, 2, None)).unwrap();
    let lamps = w.add_instances(&desc(3, 4, None)).unwrap();
    let mut set = desc(3, 1, Some(10));
    set.first = 4;
    let with_set = w.add_instances(&set).unwrap();
    // Set 1 of the copy's material sets holds (other, dark).
    w.material_sets(10, 4)
        .unwrap()
        .copy_from_slice(&[lamp, lamp, other, dark]);
    let set_one = CopyVariant {
        pose: 0,
        material_set: 1,
        section_mask: 0,
    };
    w.copy_variants(with_set, 0, &[set_one]).unwrap();
    // Copy 3 hides section 1.
    let hidden = CopyVariant {
        pose: 0,
        material_set: 0,
        section_mask: 0b10,
    };
    w.copy_variants(lamps, 3, &[hidden]).unwrap();
    // Twelve entries hold five copies of two spans; a sixth and seventh copy do not fit,
    // and the failed call leaves its slots free.
    let full = w.add_instances(&desc(3, 2, None));
    assert!(full.is_err());
    let next = w.add_instances(&desc(9, 1, None)).unwrap();
    assert_eq!(next.first, with_set.first + 1);
    drop(w);
    let mut root = GpuPtr::NULL;
    f.update_scenes(&mut |c| root = c.root).unwrap();
    wait(f.end(&[]).unwrap());
    let root: SceneRoot = read_buffer(root, 1)[0];

    let luminance = |m: [f32; 3], l: f64| DVec3::from(m.map(f64::from)) * l;
    let det23 = |m: Mat3| f64::from(m.determinant().abs()).powf(2.0 / 3.0);
    let expect = |copy: usize, span: usize, material: DVec3| {
        let area = f64::from(spans[span].area) * det23(scales[copy]);
        (area, material * area)
    };
    // (slot, copy, span, luminance), None when the section is hidden.
    let check = |all: &[EmissiveGpu], live: &[(u32, usize, usize, Option<DVec3>)]| {
        let instances = &all[PRODUCERS as usize..];
        let mut seen = 0;
        for e in instances.iter().filter(|e| e.kind != 0) {
            let &(_, copy, span, material) = live
                .iter()
                .find(|(slot, _, span, _)| {
                    *slot == e.source && spans[*span].first_cluster == e.first_cluster
                })
                .unwrap_or_else(|| panic!("unexpected entry {e:?}"));
            assert_eq!(e.kind, 1);
            assert_eq!(e.cluster_count, spans[span].cluster_count);
            assert_eq!((e.section, e.sections), (u32::from(spans[span].section), 2));
            let (area, power) = material.map_or((0.0, DVec3::ZERO), |l| expect(copy, span, l));
            close(e.area, area, "area");
            close3(e.power, power, "power");
            seen += 1;
        }
        assert_eq!(seen, live.len());
    };
    let red100 = luminance(red, 100.0);
    let blue10 = luminance(blue, 10.0);
    let mut live = Vec::new();
    for copy in 0..4 {
        live.push((lamps.first + copy as u32, copy, 0, Some(red100)));
        let hidden = copy == 3;
        live.push((
            lamps.first + copy as u32,
            copy,
            1,
            (!hidden).then_some(blue10),
        ));
    }
    live.push((with_set.first, 4, 0, Some(luminance(green, 1000.0))));
    live.push((with_set.first, 4, 1, Some(DVec3::ZERO)));
    check(&entries(&root), &live);

    // Removing copies 1 and 2 clears their entries; the rest keep their power.
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    f.scene(s).unwrap().remove(InstanceRange {
        first: lamps.first + 1,
        count: 2,
    });
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    live.retain(|&(slot, ..)| slot != lamps.first + 1 && slot != lamps.first + 2);
    check(&entries(&root), &live);

    // Once that frame has completed, the freed entries take two new copies.
    for _ in 0..2 {
        let mut f = scenes.begin_frame(&frame(&list)).unwrap();
        f.update_scenes(&mut |_| {}).unwrap();
        wait(f.end(&[]).unwrap());
    }
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let again = f
        .scene(s)
        .unwrap()
        .add_instances(&desc(3, 2, None))
        .unwrap();
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());
    for copy in 0..2 {
        live.push((again.first + copy as u32, copy, 0, Some(red100)));
        live.push((again.first + copy as u32, copy, 1, Some(blue10)));
    }
    check(&entries(&root), &live);
    assert_clean();
}

/// Area of the triangle `a b c` after `m`.
fn area(m: DMat3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    0.5 * (m * (b - a)).cross(m * (c - a)).length()
}

/// What `emit.slang` emits: per material index, the transformed area of `count` quads.
fn emitted(m: DMat3, count: u32, size: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    for c in 0..count {
        let s = size * f64::from(1 + c % 3);
        let o = DVec3::new(2.0 * f64::from(c), 0.0, 0.0);
        let p = [
            o,
            o + DVec3::new(s, 0.0, 0.0),
            o + DVec3::new(s, 0.5 * s, s),
            o + DVec3::new(0.0, 0.0, s),
        ];
        out[(c % 3) as usize] += area(m, p[0], p[1], p[2]) + area(m, p[0], p[2], p[3]);
    }
    out
}

#[test]
fn producer_power_follows_emitted_clusters_every_frame() {
    let d = device();
    let mut scenes = Scenes::new(d, &RendererDesc::default()).unwrap();
    let s = scenes
        .create_scene(&SceneDesc {
            producer_capacity: PRODUCERS,
            emissive_capacity: 0,
            ..SceneDesc::default()
        })
        .unwrap();
    let p = pipeline(&emit::MODULE, "main");
    let mut quad = d
        .gpu_malloc::<u8>(MemoryKind::Upload, 8, "quad triangles")
        .unwrap();
    quad.as_mut_slice()
        .unwrap()
        .copy_from_slice(&[0, 1, 2, 0, 2, 3, 0, 0]);
    let topology = [ClusterTopology {
        vertex_count: 4,
        triangles: &[[0, 1, 2], [0, 2, 3]],
    }];
    let linear = Mat3::from_rotation_z(0.4)
        * Mat3::from_cols(
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(0.3, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.75),
        );
    let m = linear.as_dmat3();
    let (warm, cool) = ([1.0, 0.5, 0.25], [0.25, 0.5, 1.0]);
    let luminance = [
        DVec3::new(1.0, 0.5, 0.25) * 50.0,
        DVec3::ZERO,
        DVec3::new(0.25, 0.5, 1.0) * 4.0,
    ];
    let list = [s];
    let run = |scenes: &mut Scenes,
               write: &mut dyn FnMut(&mut orr_scene::SceneWriter<'_>),
               emits: &[(ProducerId, u32, f32)]|
     -> SceneRoot {
        let mut f = scenes.begin_frame(&frame(&list)).unwrap();
        write(&mut f.scene(s).unwrap());
        let mut root = GpuPtr::NULL;
        f.update_scenes(&mut |c| {
            root = c.root;
            for &(producer, count, size) in emits {
                let r = c
                    .put_root(&emit::Root {
                        scene: c.root,
                        triangles: quad.gpu,
                        producer: producer.0,
                        count,
                        size,
                        _pad: 0,
                    })
                    .unwrap();
                c.cmd.dispatch(&p, r, [count.div_ceil(64).max(1), 1, 1]);
            }
        })
        .unwrap();
        wait(f.end(&[]).unwrap());
        read_buffer(root, 1)[0]
    };

    let mut materials = [MaterialId(0); 3];
    let mut group = TransformGroupId(0);
    run(
        &mut scenes,
        &mut |w| {
            group = w
                .create_transform_group(&TransformGroupDesc {
                    count: 2,
                    writer: GroupWriter::Cpu,
                    anchor: DVec3::new(-3000.0, 10.0, 7000.0),
                })
                .unwrap();
            w.transforms(group, 0, 2).unwrap().copy_from_slice(&[
                affine(linear, Vec3::new(4.0, 0.0, -2.0)),
                affine(Mat3::IDENTITY, Vec3::ZERO),
            ]);
        },
        &[],
    );
    let mut f = scenes.begin_frame(&frame(&list)).unwrap();
    let mut mats = f.materials();
    materials[0] = mats.create(&emitting(warm, 50.0)).unwrap();
    materials[1] = mats.create(&MaterialGpu::zeroed()).unwrap();
    materials[2] = mats.create(&emitting(cool, 4.0)).unwrap();
    let mut w = f.scene(s).unwrap();
    let producer = |update, max_clusters, transform| ProducerDesc {
        max_clusters,
        max_vertices: 4 * max_clusters,
        update,
        topologies: &topology,
        materials: &materials,
        transform: (group, transform),
        previous_positions: false,
        vertex_user: false,
        flags: InstanceFlags::RASTER | InstanceFlags::RAY,
        user_tag: 0,
        ..ProducerDesc::default()
    };
    w.set_producer(
        ProducerId(1),
        Some(&producer(ProducerUpdate::EveryFrame, 100, 0)),
    )
    .unwrap();
    w.set_producer(
        ProducerId(2),
        Some(&producer(ProducerUpdate::Static, 16, 1)),
    )
    .unwrap();
    drop(w);
    f.update_scenes(&mut |_| {}).unwrap();
    wait(f.end(&[]).unwrap());

    let expected = |m: DMat3, count: u32, size: f64| {
        let a = emitted(m, count, size);
        let power = (0..3).map(|i| luminance[i] * a[i]).sum::<DVec3>();
        let clusters = (0..count).filter(|c| c % 3 != 1).count() as u32;
        (a[0] + a[2], power, clusters)
    };
    let check = |e: &EmissiveGpu, (area, power, clusters): (f64, DVec3, u32), what: &str| {
        assert_eq!(e.kind, 2, "{what}");
        assert_eq!(e.cluster_count, clusters, "{what}");
        close(e.area, area, what);
        close3(e.power, power, what);
    };
    // The static entry emits once; the other changes count and size every frame with no
    // write to the list.
    let statics = expected(DMat3::IDENTITY, 11, 0.5);
    for (frame, &(count, size)) in [
        (11, 0.5),
        (7, 1.0),
        (64, 0.25),
        (99, 2.0),
        (3, 0.125),
        (0, 1.0),
    ]
    .iter()
    .enumerate()
    {
        let mut emits = vec![(ProducerId(1), count, size)];
        if frame == 0 {
            emits.push((ProducerId(2), 11, 0.5));
        }
        let root = run(&mut scenes, &mut |_| {}, &emits);
        let all = entries(&root);
        check(
            &all[1],
            expected(m, count, f64::from(size)),
            "every-frame entry",
        );
        check(&all[2], statics, "static entry");
        assert_eq!((all[0].kind, all[3].kind), (0, 0));
    }

    // Past capacity the batch keeps 100 clusters and drops the rest.
    let root = run(&mut scenes, &mut |_| {}, &[(ProducerId(1), 150, 1.0)]);
    let all = entries(&root);
    let full = expected(m, 150, 1.0);
    assert!(all[1].cluster_count <= 100 && all[1].cluster_count >= 50);
    assert!(f64::from(all[1].area) < full.0 && all[1].area > 0.0);

    // Moving the entry's transform changes its power with no write to the list.
    let root = run(
        &mut scenes,
        &mut |w| {
            w.transforms(group, 0, 1).unwrap()[0] = affine(Mat3::IDENTITY, Vec3::ZERO);
        },
        &[(ProducerId(1), 20, 1.0)],
    );
    check(
        &entries(&root)[1],
        expected(DMat3::IDENTITY, 20, 1.0),
        "moved entry",
    );

    // A cleared entry leaves the list.
    let root = run(
        &mut scenes,
        &mut |w| w.set_producer(ProducerId(1), None).unwrap(),
        &[],
    );
    let all = entries(&root);
    assert_eq!((all[1].kind, all[1].power), (0, [0.0; 3]));
    check(&all[2], statics, "static entry");
    assert_clean();
}
