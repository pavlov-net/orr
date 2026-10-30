//! Device, memory, heaps, compute, timestamps and the empty TLAS.

use gpu_suite::shaders::{atomic_max, heap_views, pattern, rays, root_layout};
use gpu_suite::{
    CUSTOM_INDEX, assert_clean, device, pipeline, readback, render, run, triangle_scene, upload,
};
use orr_gpu::{
    Filter, Format, GpuPtr, Hazards, SamplerDesc, SamplerIndex, Stages, TextureDesc, TextureMemory,
    TextureRegion, TextureUsage, Tlas, ViewDesc,
};

#[test]
fn root_layout_reads_back_exact() {
    let rec = root_layout::Record {
        tag: 0xdead_beef,
        inner: root_layout::Inner {
            a: 17,
            b: [1.5, -2.25, 1e-3],
        },
        pad: 0,
        big: 0x0123_4567_89ab_cdef,
        tail: [-0.0, 3.0e9],
    };
    let rec = upload(&[rec], "root layout record");
    let out = readback::<u32>(30, "root layout out");
    let m = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [-1.0, -2.0, -3.0, -4.0],
    ];
    let v = [1.0, 0.5, -2.0, 4.0];
    let root = upload(
        &[root_layout::Root {
            m,
            v,
            rec: rec.gpu,
            out: out.gpu,
            seed: 99,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }],
        "root layout root",
    );
    let p = pipeline(&root_layout::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.dispatch(&p, root.gpu, [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);

    let got = out.as_slice().unwrap();
    let floats = |range: std::ops::Range<usize>| {
        got[range]
            .iter()
            .map(|&w| f32::from_bits(w))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        floats(0..16),
        m.iter().flatten().copied().collect::<Vec<_>>(),
        "matrix rows"
    );
    let mv: Vec<f32> = m
        .iter()
        .map(|r| r.iter().zip(&v).map(|(a, b)| a * b).sum())
        .collect();
    assert_eq!(floats(16..20), mv, "mul(m, v) uses rows");
    assert_eq!(&got[20..22], &[0xdead_beef, 17]);
    assert_eq!(floats(22..25), [1.5, -2.25, 1e-3]);
    assert_eq!(
        u64::from(got[25]) | u64::from(got[26]) << 32,
        0x0123_4567_89ab_cdef
    );
    assert_eq!(got[27], (-0.0f32).to_bits());
    assert_eq!(f32::from_bits(got[28]), 3.0e9);
    assert_eq!(got[29], 99);
    assert_clean();
}

#[test]
fn sampled_and_storage_views_in_adjacent_slots() {
    let d = device();
    let (w, h) = (1024u32, 1024u32);
    let desc = TextureDesc::d2(
        Format::Rgba32Float,
        w,
        h,
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
    );
    let tex = d
        .create_texture(&desc, TextureMemory::Dedicated, "heap views")
        .unwrap();
    let storage_slot = d.alloc_image_slot().unwrap();
    let sampled_slot = d.alloc_image_slot().unwrap();
    assert_eq!(sampled_slot, storage_slot + 1, "fresh slots are adjacent");
    let target = d.write_storage(storage_slot, &tex, &ViewDesc::default());
    let source = d.write_sampled(sampled_slot, &tex, &ViewDesc::default());
    let nearest = SamplerIndex(3);
    d.write_sampler(
        nearest,
        &SamplerDesc {
            mag: Filter::Nearest,
            min: Filter::Nearest,
            mip: Filter::Nearest,
            ..SamplerDesc::default()
        },
    );

    let out = readback::<[f32; 4]>(u64::from(w * h), "heap views out");
    let texels = readback::<[f32; 4]>(u64::from(w * h), "heap views texels");
    let root = upload(
        &[heap_views::Root {
            out: out.gpu,
            source,
            nearest,
            target,
            width: w,
            height: h,
            pad: 0,
        }],
        "heap views root",
    );
    let store = pipeline(&heap_views::MODULE, "store_main");
    let sample = pipeline(&heap_views::MODULE, "sample_main");
    let groups = [w.div_ceil(8), h.div_ceil(4), 1];
    let mut list = render().command_list().unwrap();
    list.discard(&tex);
    list.dispatch(&store, root.gpu, groups);
    list.barrier(
        Stages::COMPUTE,
        Stages::COMPUTE | Stages::TRANSFER,
        Hazards::empty(),
    );
    list.dispatch(&sample, root.gpu, groups);
    list.copy_from_texture(texels.gpu.cast(), &tex, &TextureRegion::whole_mip(&desc, 0));
    list.barrier(
        Stages::COMPUTE | Stages::TRANSFER,
        Stages::HOST,
        Hazards::empty(),
    );
    let point = run(list);

    let expected = |x: u32, y: u32| {
        [
            (x as f32 + 0.5) / w as f32,
            (y as f32 + 0.5) / h as f32,
            (x ^ y) as f32,
            1.0,
        ]
    };
    let sampled = out.as_slice().unwrap();
    let copied = texels.as_slice().unwrap();
    let mut wrong = 0;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            wrong += usize::from(sampled[i] != expected(x, y) || copied[i] != expected(x, y));
        }
    }
    assert_eq!(wrong, 0, "texels differing from the pattern");
    d.free_image_slot(storage_slot, point);
    d.free_image_slot(sampled_slot, point);
    assert_clean();
}

#[test]
fn atomic_max_u64_matches_cpu() {
    let count = 1 << 20;
    let slots = 1000u64;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let values: Vec<u64> = (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        })
        .collect();
    let mut expected = vec![0u64; slots as usize];
    for (i, &v) in values.iter().enumerate() {
        let s = &mut expected[i % slots as usize];
        *s = (*s).max(v);
    }
    let values = upload(&values, "atomic values");
    let out = readback::<u64>(slots, "atomic slots");
    let zero = upload(&vec![0u64; slots as usize], "atomic zero");
    let root = upload(
        &[atomic_max::Root {
            slots: out.gpu,
            values: values.gpu,
            count: count as u32,
            slot_count: slots as u32,
            pad0: 0,
            pad1: 0,
        }],
        "atomic root",
    );
    let p = pipeline(&atomic_max::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.copy(out.gpu.cast(), zero.gpu.cast(), out.size);
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&p, root.gpu, [(count as u32).div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);
    assert_eq!(out.as_slice().unwrap(), &expected[..]);
    assert_clean();
}

/// Traces 1024x1024 rays at `scene` through `entry`; returns the per-pixel results.
fn trace(scene: Tlas, entry: &str) -> Vec<u32> {
    let (w, h) = (1024u32, 1024u32);
    let out = readback::<u32>(u64::from(w * h), "rays out");
    let scene_ptr = upload(&[scene], "scene pointer");
    let root = upload(
        &[rays::Root {
            out: out.gpu,
            scene,
            scene_ptr: scene_ptr.gpu,
            width: w,
            height: h,
        }],
        "rays root",
    );
    let p = pipeline(&rays::MODULE, entry);
    let mut list = render().command_list().unwrap();
    list.dispatch(&p, root.gpu, [w.div_ceil(8), h.div_ceil(4), 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);
    out.as_slice().unwrap().to_vec()
}

/// Pixels whose centre is inside the triangle hit; those within 1e-3 of an edge are skipped.
fn check_triangle(got: &[u32]) {
    let (w, h) = (1024usize, 1024usize);
    let mut wrong = 0;
    let mut hits = 0;
    for y in 0..h {
        for x in 0..w {
            let (px, py) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let edges = [px - 0.1, py - 0.1, 1.0 - px - py];
            if edges.iter().any(|e| e.abs() < 1e-3) {
                continue;
            }
            let inside = edges.iter().all(|&e| e > 0.0);
            let expected = if inside { CUSTOM_INDEX + 1 } else { 0 };
            hits += usize::from(inside);
            wrong += usize::from(got[y * w + x] != expected);
        }
    }
    assert!(hits > 100_000);
    assert_eq!(wrong, 0, "rays disagreeing with the CPU reference");
}

#[test]
fn tlas_address_in_root_and_through_pointer() {
    let scene = triangle_scene();
    check_triangle(&trace(scene.tlas.tlas(), "main"));
    check_triangle(&trace(scene.tlas.tlas(), "main_ptr"));
    assert_clean();
}

#[test]
fn empty_tlas_misses_every_ray() {
    let empty = device().empty_tlas();
    assert_ne!(empty, Tlas(0));
    for entry in ["main", "main_ptr"] {
        let got = trace(empty, entry);
        assert!(
            got.iter().all(|&v| v == 0),
            "{entry}: a ray hit the empty TLAS"
        );
    }
    assert_clean();
}

#[test]
fn lists_recorded_on_two_threads() {
    let count = 1 << 16;
    let a = readback::<u32>(count, "thread a");
    let b = readback::<u32>(count, "thread b");
    let root_a = upload(
        &[pattern::Root {
            out: a.gpu,
            count: count as u32,
            seed: 1,
        }],
        "root a",
    );
    let root_b = upload(
        &[pattern::Root {
            out: b.gpu,
            count: count as u32,
            seed: 2,
        }],
        "root b",
    );
    let p = pipeline(&pattern::MODULE, "main");
    let groups = [(count as u32).div_ceil(64), 1, 1];
    let queue = render();
    let lists = std::thread::scope(|s| {
        let record = |root: GpuPtr<pattern::Root>, stats: bool| {
            let p = &p;
            move || {
                let mut list = queue.command_list().unwrap();
                list.begin_pass("pattern", stats);
                list.dispatch(p, root, groups);
                list.end_pass();
                list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
                list
            }
        };
        let ta = s.spawn(record(root_a.gpu, true));
        let tb = s.spawn(record(root_b.gpu, false));
        vec![ta.join().unwrap(), tb.join().unwrap()]
    });
    let point = queue.submit(lists, &[]).unwrap();
    queue
        .wait(point, std::time::Duration::from_secs(10))
        .unwrap();
    for (out, seed) in [(&a, 1u32), (&b, 2)] {
        let got = out.as_slice().unwrap();
        assert!(
            got.iter()
                .enumerate()
                .all(|(i, &v)| v == (i as u32).wrapping_mul(2_654_435_761) ^ seed)
        );
    }

    let results = device().pass_results(point).expect("completed");
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|r| r.name == "pattern" && r.gpu_ms > 0.0)
    );
    let stats = results[0].stats.expect("statistics were asked for");
    assert_eq!(stats.compute, u64::from(groups[0]) * 64);
    assert!(results[1].stats.is_none());
    if device().caps().calibrated_timestamps {
        let (begin, end) = results[0].cpu.expect("calibrated span");
        assert!(end >= begin);
    }
    assert_clean();
}
