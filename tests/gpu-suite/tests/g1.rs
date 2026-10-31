//! Device, memory, heaps, compute, timestamps and the empty TLAS.

use gpu_suite::shaders::{atomic_max, coopvec, heap_views, pattern, rays, root_layout};
use gpu_suite::{
    CUSTOM_INDEX, assert_clean, device, noise, pipeline, readback, render, run, triangle_scene,
    upload,
};
use orr_gpu::{
    Device, DeviceDesc, Filter, Format, GpuPtr, Hazards, SamplerDesc, SamplerIndex, Stages,
    TextureDesc, TextureMemory, TextureRegion, TextureUsage, Tlas, ViewDesc,
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

/// 16x16 weights converted into the inference layout, on the GPU and on the host, multiply
/// FP16 vectors through `orr.coopvec` as a CPU matmul does, at every 64-byte placement of
/// the matrix and with a bias. FP16 weights always; E4M3 and E5M2 weights where the device
/// has 8-bit floats.
#[test]
fn cooperative_vector_matmul_matches_cpu() {
    use half::f16;
    use orr_gpu::ComponentType;
    let d = device();
    let supports = |w: ComponentType| {
        d.caps().cooperative_vector.iter().any(|c| {
            c.input == ComponentType::F16
                && c.input_interpretation == w
                && c.matrix_interpretation == w
                && c.bias_interpretation == ComponentType::F16
                && c.result == ComponentType::F16
        })
    };
    if !gpu_suite::require(supports(ComponentType::F16), "no FP16 cooperative vectors") {
        return;
    }
    const VECTORS: usize = 4096;
    let mut rnd = noise(7);
    let mut random = |scale: f32| f16::from_f32((rnd() * 2.0 - 1.0) * scale);
    let weights: Vec<f16> = (0..16 * 16).map(|_| random(1.0 / 8.0)).collect();
    let inputs: Vec<f16> = (0..16 * VECTORS).map(|_| random(1.0)).collect();
    let bias: Vec<f16> = (0..16).map(|_| random(0.5)).collect();
    coopvec_case(ComponentType::F16, "f16", &weights, &inputs, &bias);

    if !gpu_suite::require(d.caps().shader_float8, "no 8-bit floats") {
        return;
    }
    // Values with two mantissa bits, so FP16, E4M3 and E5M2 hold them exactly and the CPU
    // matmul needs no FP8 rounding.
    let mut rnd = noise(11);
    let mut exact = |min_exp: i32| {
        let r = (rnd() * 32.0) as u32;
        let sign = if r & 1 == 0 { 1.0 } else { -1.0 };
        let mantissa = 1.0 + ((r >> 1) & 3) as f32 / 4.0;
        f16::from_f32(sign * mantissa * 2f32.powi(-min_exp - ((r >> 3) & 3) as i32))
    };
    let weights: Vec<f16> = (0..16 * 16).map(|_| exact(3)).collect();
    let inputs: Vec<f16> = (0..16 * VECTORS).map(|_| exact(0)).collect();
    for (t, entry) in [(ComponentType::E4m3, "e4m3"), (ComponentType::E5m2, "e5m2")] {
        if gpu_suite::require(supports(t), "no cooperative vectors with this FP8 type") {
            coopvec_case(t, entry, &weights, &inputs, &bias);
        }
    }
}

/// Converts row-major FP16 `weights` into `component` in the inference layout and multiplies
/// `inputs` with them through the `entry` kernel, at each matrix placement and once with
/// `bias`, checking each result against a CPU matmul.
fn coopvec_case(
    component: orr_gpu::ComponentType,
    entry: &str,
    weights: &[half::f16],
    inputs: &[half::f16],
    bias: &[half::f16],
) {
    use orr_gpu::{ComponentType, MatrixConversion, MatrixFormat, MatrixLayout, MemoryKind};
    const N: usize = 16;
    let vectors = inputs.len() / N;
    let d = device();
    let conversion = MatrixConversion {
        rows: N as u32,
        columns: N as u32,
        src_format: MatrixFormat::new(ComponentType::F16, MatrixLayout::RowMajor),
        dst_format: MatrixFormat::new(component, MatrixLayout::InferencingOptimal),
    };
    let bytes = d
        .matrix_bytes(N as u32, N as u32, conversion.dst_format)
        .unwrap();

    // The GPU converts into device memory at offset 64; the host into upload memory at 128.
    let row_major = upload(weights, "row-major weights");
    let on_gpu = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, 64 + bytes, "gpu-converted weights")
        .unwrap();
    let mut on_host = d
        .gpu_malloc::<u8>(MemoryKind::Upload, 128 + bytes, "host-converted weights")
        .unwrap();
    let written = d
        .convert_matrix(
            &conversion,
            orr_shaders::bytemuck::cast_slice(weights),
            &mut on_host,
            128,
        )
        .unwrap();
    assert_eq!(written, bytes);
    // The bias starts 16 bytes into its allocation.
    let mut biased = vec![half::f16::ZERO; 8];
    biased.extend(bias);
    let bias_buf = upload(&biased, "coopvec bias");

    let as_f32: Vec<f32> = inputs.iter().map(|v| v.to_f32()).collect();
    let input = upload(&as_f32, "coopvec input");
    let gpu = on_gpu.gpu.address().0;
    let host = on_host.gpu.address().0;
    // (base, matrix address, matrix offset, bias offset): the matrix's start in the offset
    // operand, in the address, or split between them.
    let runs = [
        (gpu, gpu, 64, None),
        (gpu, gpu + 64, 0, None),
        (host, host, 128, None),
        (host, host + 64, 64, None),
        (host, host + 128, 0, None),
        (gpu, gpu + 64, 0, Some(16)),
    ];
    let outputs: Vec<_> = runs
        .iter()
        .map(|_| readback::<f32>((N * vectors) as u64, "coopvec output"))
        .collect();
    let roots: Vec<_> = runs
        .iter()
        .zip(&outputs)
        .map(|(&(_, weights, offset, bias), out)| {
            upload(
                &[coopvec::Root {
                    input: input.gpu,
                    output: out.gpu,
                    weights,
                    bias: bias.map_or(0, |_| bias_buf.gpu.address().0),
                    offset,
                    bias_offset: bias.unwrap_or(0),
                    count: vectors as u32,
                    pad: 0,
                }],
                "coopvec root",
            )
        })
        .collect();
    let p = pipeline(&coopvec::MODULE, entry);
    let mut list = render().command_list().unwrap();
    list.convert_matrix(
        &conversion,
        row_major.gpu.address(),
        on_gpu.gpu.offset(64).address(),
    );
    list.barrier(
        Stages::TRANSFER | Stages::HOST,
        Stages::COMPUTE,
        Hazards::empty(),
    );
    for root in &roots {
        list.dispatch(&p, root.gpu, [(vectors as u32).div_ceil(32), 1, 1]);
    }
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);

    for (&(base, weights_at, offset, with_bias), out) in runs.iter().zip(&outputs) {
        let got = out.as_slice().unwrap();
        let (mut outside, mut worst) = (0, 0.0f32);
        for v in 0..vectors {
            for row in 0..N {
                let b = with_bias.map_or(0.0, |_| bias[row].to_f32());
                let want = b
                    + (0..N)
                        .map(|k| weights[row * N + k].to_f32() * as_f32[v * N + k])
                        .sum::<f32>();
                let err = (got[v * N + row] - want).abs();
                worst = worst.max(err);
                outside += usize::from(err > 2e-3 + 1e-2 * want.abs());
            }
        }
        let side = if base == gpu { "gpu" } else { "host" };
        let bias_note = if with_bias.is_some() {
            " with bias"
        } else {
            ""
        };
        let place = format!(
            "{entry}, {side}-converted, matrix at +{} offset {offset}{bias_note}",
            weights_at - base
        );
        log::info!("cooperative vectors {place}: max error {worst:.2e}");
        assert_eq!(outside, 0, "{place}: max error {worst}");
    }
    assert_clean();
}

/// Buffer copies, a texture upload with padded rows and its readback, an indirect dispatch
/// and a compacted-size query on `d`, each checked against the CPU. Returns the compacted
/// size of a small BLAS.
fn address_command_workload(d: &Device) -> u64 {
    use orr_gpu::{
        ClassicBuild, ClassicDesc, ClassicGeometry, ClassicKind, DispatchArgs, MemoryKind,
        QueueKind, RtasFlags, ShaderStage, SpecData,
    };
    let queue = d.queue(QueueKind::Render).unwrap();
    let upload = |words: &[u32]| {
        let mut a = d
            .gpu_malloc::<u32>(MemoryKind::Upload, words.len() as u64, "workload upload")
            .unwrap();
        a.as_mut_slice().unwrap().copy_from_slice(words);
        a
    };
    let words: Vec<u32> = (0..4096u32).map(|i| i.wrapping_mul(0x9e37_79b9)).collect();
    let src = upload(&words);
    let middle = d
        .gpu_malloc::<u32>(MemoryKind::DeviceLocal, 4096, "workload middle")
        .unwrap();
    let copied = d
        .gpu_malloc::<u32>(MemoryKind::Readback, 4096, "workload copied")
        .unwrap();

    // 40x24 texels taken from rows 48 texels apart.
    let desc = TextureDesc::d2(Format::Rgba8Unorm, 40, 24, TextureUsage::SAMPLED);
    let region = TextureRegion::whole_mip(&desc, 0);
    let t = d
        .create_texture(&desc, TextureMemory::Dedicated, "workload texture")
        .unwrap();
    let texels = d
        .gpu_malloc::<u32>(MemoryKind::Readback, 40 * 24, "workload texels")
        .unwrap();

    let stage = ShaderStage::new(&pattern::MODULE, "main", SpecData::default()).unwrap();
    let p = d
        .create_compute_pipeline(&stage, "workload pattern")
        .unwrap();
    let n = 1000u32;
    let pattern_out = d
        .gpu_malloc::<u32>(MemoryKind::Readback, u64::from(n), "workload pattern")
        .unwrap();
    let mut root = d
        .gpu_malloc::<pattern::Root>(MemoryKind::Upload, 1, "workload root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = pattern::Root {
        out: pattern_out.gpu,
        count: n,
        seed: 0x5eed,
    };
    let mut args = d
        .gpu_malloc::<DispatchArgs>(MemoryKind::Upload, 1, "workload args")
        .unwrap();
    args.as_mut_slice().unwrap()[0] = DispatchArgs {
        x: n.div_ceil(64),
        y: 1,
        z: 1,
    };

    let mut positions = d
        .gpu_malloc::<[f32; 3]>(MemoryKind::Upload, 3, "workload triangle")
        .unwrap();
    positions.as_mut_slice().unwrap().copy_from_slice(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let blas_desc = ClassicDesc {
        kind: ClassicKind::Blas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Triangles {
            positions: positions.gpu,
            vertex_count: 3,
            indices: GpuPtr::NULL,
            triangle_count: 1,
            opaque: true,
        },
    };
    let sizes = d.classic_sizes(&blas_desc);
    let blas_block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "workload blas")
        .unwrap();
    let scratch = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.scratch, "workload scratch")
        .unwrap();
    let blas = d
        .create_classic_as(ClassicKind::Blas, sizes.result_max, &blas_block, 0)
        .unwrap();
    let compacted = d
        .gpu_malloc::<u64>(MemoryKind::Readback, 1, "workload compacted")
        .unwrap();

    let mut list = queue.command_list().unwrap();
    list.copy(middle.gpu.cast(), src.gpu.cast(), 4 * 4096);
    list.discard(&t);
    list.barrier(Stages::TRANSFER, Stages::TRANSFER, Hazards::empty());
    list.copy(copied.gpu.cast(), middle.gpu.cast(), 4 * 4096);
    list.copy_rows_to_texture(&t, &region, src.gpu.cast(), 48);
    list.barrier(Stages::TRANSFER, Stages::TRANSFER, Hazards::empty());
    list.copy_from_texture(texels.gpu.cast(), &t, &region);
    list.dispatch_indirect(&p, root.gpu, args.gpu);
    list.build_classic(&ClassicBuild {
        desc: &blas_desc,
        dst: &blas,
        scratch: scratch.gpu.address(),
    });
    list.classic_compacted_size(&blas, compacted.gpu);
    list.barrier(
        Stages::TRANSFER | Stages::COMPUTE,
        Stages::HOST,
        Hazards::empty(),
    );
    let point = queue.submit(vec![list], &[]).unwrap();
    queue
        .wait(point, std::time::Duration::from_secs(10))
        .unwrap();

    assert_eq!(copied.as_slice().unwrap(), &words[..], "buffer copies");
    let rows: Vec<u32> = (0..24)
        .flat_map(|y| words[y * 48..y * 48 + 40].iter().copied())
        .collect();
    assert_eq!(texels.as_slice().unwrap(), &rows[..], "texture rows");
    let want: Vec<u32> = (0..n)
        .map(|i| i.wrapping_mul(2_654_435_761) ^ 0x5eed)
        .collect();
    assert_eq!(
        pattern_out.as_slice().unwrap(),
        &want[..],
        "indirect dispatch"
    );
    let size = compacted.as_slice().unwrap()[0];
    assert!(
        size > 0 && size <= sizes.result_max,
        "compacted to {size} B"
    );
    size
}

/// The commands that take device addresses directly give the same results as their
/// buffer-handle forms on a device without `VK_KHR_device_address_commands`.
#[test]
fn address_commands_match_buffer_commands() {
    let d = device();
    if !gpu_suite::require(
        d.caps().device_address_commands,
        "no device address commands",
    ) {
        return;
    }
    let with = address_command_workload(d);
    let without = Device::new(&DeviceDesc {
        name: "gpu-suite buffer commands",
        validation: true,
        skip_extensions: &[ash::khr::device_address_commands::NAME],
        ..DeviceDesc::default()
    })
    .unwrap();
    assert!(!without.caps().device_address_commands);
    assert_eq!(address_command_workload(&without), with, "compacted sizes");
    assert_eq!(
        without.validation_messages(),
        0,
        "validation messages (see the log)"
    );
    assert_clean();
}

#[test]
fn priorities_allocate_and_the_report_carries_the_pressure() {
    use orr_gpu::{MemoryKind, MemoryPriority};
    let d = device();
    for priority in [
        MemoryPriority::Low,
        MemoryPriority::Normal,
        MemoryPriority::High,
    ] {
        for kind in [MemoryKind::Upload, MemoryKind::DeviceLocal] {
            let a = d
                .gpu_malloc_with_priority::<u32>(kind, priority, 1 << 16, "priority")
                .unwrap();
            assert!(
                a.gpu.address().0.is_multiple_of(256),
                "{kind:?} {priority:?}"
            );
        }
    }
    d.collect_garbage();
    let pressure = d.memory_report().pressure();
    if d.caps().memory_budget {
        let p = pressure.expect("a device-local heap with an OS budget");
        assert!(p > 0.0 && p.is_finite(), "pressure {p}");
    } else {
        assert_eq!(pressure, None);
    }
    assert_clean();
}
