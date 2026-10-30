//! The shader-abi kernels on the GPU, each against a CPU reference: writes through a
//! pointer, heap views, the 64-bit max, alpha-tested traces of a classic TLAS and of the
//! opacity-micromap cluster scene, one body behind raygen and compute entries over a PTLAS,
//! candidate traces and a task/mesh/pixel draw into two targets.

use gpu_suite::clusters::{self as cl, Builder, Cluster};
use gpu_suite::{assert_clean, device, pipeline, readback, render, run, upload};
use orr_gpu::rtas::consts::GEOMETRY_CULL_DISABLE;
use orr_gpu::{
    Allocation, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicKind, ClearColor,
    ColorAttachment, ColorLoad, ColorTarget, ComputePipeline, Filter, Format, Hazards, MemoryKind,
    MeshPipelineDesc, MicromapDesc, MicromapTriangle, MicromapUsage, OmmFormat, Rendering,
    RtasFlags, SamplerDesc, ShaderStage, SpecData, Stages, TextureDesc, TextureMemory,
    TextureRegion, TextureUsage, Tlas,
};
use shader_abi::shaders::{alpha, atomic, candidates, clusters, mesh, pattern, texture};

const EDGE: f32 = 2e-4;
/// `RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE` of `orr.trace`.
const FORCE_OPACITY_MICROMAP_2_STATE: u32 = 0x400;
const SIZE: u32 = 256;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn ray_at(x: u32, y: u32, size: u32) -> [f32; 2] {
    [
        (x as f32 + 0.5) / size as f32,
        (y as f32 + 0.5) / size as f32,
    ]
}

/// The smallest signed distance from `p` to the edges of `v` (positive inside).
fn edge_distance(v: [[f32; 2]; 3], p: [f32; 2]) -> f32 {
    let area =
        (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) - (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]);
    (0..3)
        .map(|e| {
            let (a, b) = (v[e], v[(e + 1) % 3]);
            let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
            cross * area.signum() / (b[0] - a[0]).hypot(b[1] - a[1])
        })
        .fold(f32::INFINITY, f32::min)
}

/// A ray along +z from z = -1 at `p` against triangle `v` placed at `offset`: the hit t,
/// no hit, or `Err` when the ray passes within EDGE of an edge.
fn hit_t(v: [[f32; 3]; 3], offset: [f32; 3], p: [f32; 2]) -> Result<Option<f32>, ()> {
    let q = [p[0] - offset[0], p[1] - offset[1]];
    let xy = v.map(|c| [c[0], c[1]]);
    let lo = |k: usize| xy.iter().map(|c| c[k]).fold(f32::INFINITY, f32::min);
    let hi = |k: usize| xy.iter().map(|c| c[k]).fold(f32::NEG_INFINITY, f32::max);
    if q[0] < lo(0) - EDGE || q[0] > hi(0) + EDGE || q[1] < lo(1) - EDGE || q[1] > hi(1) + EDGE {
        return Ok(None);
    }
    let d = edge_distance(xy, q);
    if d.abs() < EDGE {
        return Err(());
    }
    if d < 0.0 {
        return Ok(None);
    }
    let cross =
        |a: [f32; 2], b: [f32; 2]| (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]);
    let area = cross(xy[0], xy[1]) + cross(xy[1], xy[2]) + cross(xy[2], xy[0]);
    let w = [
        cross(xy[1], xy[2]),
        cross(xy[2], xy[0]),
        cross(xy[0], xy[1]),
    ]
    .map(|c| c / area);
    Ok(Some(
        1.0 + offset[2] + w[0] * v[0][2] + w[1] * v[1][2] + w[2] * v[2][2],
    ))
}

fn corners(c: &Cluster, t: usize) -> [[f32; 3]; 3] {
    c.tris[t].map(|i| c.verts[i as usize])
}

/// A classic BLAS over `c`'s triangles, kept alive by `b`; returns its address.
fn classic_blas(b: &mut Builder, c: &Cluster, opaque: bool) -> u64 {
    let d = device();
    let desc = ClassicDesc {
        kind: ClassicKind::Blas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Triangles {
            positions: b.upload(&c.verts),
            vertex_count: c.verts.len() as u32,
            indices: b.upload(&c.tris),
            triangle_count: c.tris.len() as u32,
            opaque,
        },
    };
    let sizes = d.classic_sizes(&desc);
    let block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "classic blas")
        .unwrap();
    let scratch = b.block(MemoryKind::DeviceLocal, sizes.scratch);
    let blas = d
        .create_classic_as(ClassicKind::Blas, sizes.result_max, &block, 0)
        .unwrap();
    b.list().build_classic(&ClassicBuild {
        desc: &desc,
        dst: &blas,
        scratch: scratch.address(),
    });
    let address = blas.address().0;
    b.keep_any(blas);
    b.keep(block);
    address
}

/// Records a trace barrier, `dispatch` and a host barrier into `b`'s list.
fn traced(b: &mut Builder, p: &ComputePipeline, root: orr_gpu::GpuPtr<impl Sized>, size: u32) {
    let [gx, gy, _] = p.workgroup();
    let list = b.list();
    list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
    list.dispatch(p, root, [size.div_ceil(gx), size.div_ceil(gy), 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
}

#[test]
fn pattern_through_pointer() {
    let count = 100_003u32;
    let out = readback::<u32>(u64::from(count), "pattern out");
    let root = upload(
        &[pattern::Root {
            out: out.gpu,
            count,
            seed: 0x51ed_270b,
        }],
        "pattern root",
    );
    let p = pipeline(&pattern::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.dispatch(&p, root.gpu, [count.div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);
    let wrong = (0..count)
        .zip(out.as_slice().unwrap())
        .filter(|&(i, &v)| v != i.wrapping_mul(2_654_435_761) ^ 0x51ed_270b)
        .count();
    assert_eq!(wrong, 0, "words differing from the pattern");
    assert_clean();
}

#[test]
fn heap_texture_sample_and_storage_write() {
    let d = device();
    let (w, h) = (256u32, 128u32);
    let desc = TextureDesc::d2(
        Format::Rgba32Float,
        w,
        h,
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
    );
    let tex = d
        .create_texture(&desc, TextureMemory::Dedicated, "abi texture")
        .unwrap();
    let mut state = 0x853c_49e6_748f_ea9bu64;
    let texels: Vec<[f32; 4]> = (0..w * h)
        .map(|_| [0; 4].map(|_| (xorshift(&mut state) >> 40) as f32 / 4096.0 - 2048.0))
        .collect();
    let src = upload(&texels, "abi texels");
    let source_slot = d.alloc_image_slot().unwrap();
    let target_slot = d.alloc_image_slot().unwrap();
    let source = d.write_sampled(source_slot, &tex, &Default::default());
    let target = d.write_storage(target_slot, &tex, &Default::default());
    let nearest = d
        .sampler(&SamplerDesc {
            mag: Filter::Nearest,
            min: Filter::Nearest,
            mip: Filter::Nearest,
            ..SamplerDesc::default()
        })
        .unwrap();
    let sampled = readback::<[f32; 4]>(u64::from(w * h), "abi sampled");
    let stored = readback::<[f32; 4]>(u64::from(w * h), "abi stored");
    let root = upload(
        &[texture::Root {
            out: sampled.gpu,
            source,
            nearest,
            target,
            width: w,
            height: h,
            pad: 0,
        }],
        "abi texture root",
    );
    let sample = pipeline(&texture::MODULE, "sample_main");
    let store = pipeline(&texture::MODULE, "store_main");
    let groups = [w.div_ceil(8), h.div_ceil(4), 1];
    let whole = TextureRegion::whole_mip(&desc, 0);
    let mut list = render().command_list().unwrap();
    list.discard(&tex);
    list.copy_to_texture(&tex, &whole, src.gpu.cast());
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&sample, root.gpu, groups);
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&store, root.gpu, groups);
    list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
    list.copy_from_texture(stored.gpu.cast(), &tex, &whole);
    list.barrier(
        Stages::COMPUTE | Stages::TRANSFER,
        Stages::HOST,
        Hazards::empty(),
    );
    let point = run(list);

    let bits = |v: &[f32; 4]| v.map(f32::to_bits);
    let sampled = sampled.as_slice().unwrap();
    let wrong = (0..texels.len())
        .filter(|&i| bits(&sampled[i]) != bits(&texels[i]))
        .count();
    assert_eq!(wrong, 0, "point samples differing from the uploaded texels");
    let stored = stored.as_slice().unwrap();
    let mut wrong = 0;
    for y in 0..h {
        for x in 0..w {
            let want = [
                (x as f32 + 0.5) / w as f32,
                (y as f32 + 0.5) / h as f32,
                0.25,
                1.0,
            ];
            wrong += usize::from(stored[(y * w + x) as usize] != want);
        }
    }
    assert_eq!(wrong, 0, "texels differing from the storage writes");
    d.free_image_slot(source_slot, point);
    d.free_image_slot(target_slot, point);
    assert_clean();
}

#[test]
fn atomic_max_u64_through_pointer() {
    let count = 1u32 << 18;
    let slots = 257u32;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let values: Vec<u64> = (0..count).map(|_| xorshift(&mut state)).collect();
    let mut expected = vec![0u64; slots as usize];
    for (i, &v) in values.iter().enumerate() {
        let s = &mut expected[i % slots as usize];
        *s = (*s).max(v);
    }
    let values = upload(&values, "abi atomic values");
    let zero = upload(&vec![0u64; slots as usize], "abi atomic zero");
    let out = readback::<u64>(u64::from(slots), "abi atomic slots");
    let root = upload(
        &[atomic::Root {
            slots: out.gpu,
            values: values.gpu,
            count,
            slot_count: slots,
            pad0: 0,
            pad1: 0,
        }],
        "abi atomic root",
    );
    let p = pipeline(&atomic::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.copy(out.gpu.cast(), zero.gpu.cast(), out.size);
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&p, root.gpu, [count.div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);
    assert_eq!(out.as_slice().unwrap(), &expected[..]);
    assert_clean();
}

/// The cluster probe's alpha rule, which `alpha.slang` applies.
fn alpha_keeps(cluster: u32, primitive: u32) -> bool {
    !cluster.wrapping_add(primitive).is_multiple_of(3)
}

/// A classic instance as the alpha reference sees it.
struct Placed<'a> {
    cluster: &'a Cluster,
    offset: [f32; 3],
    opaque: bool,
}

/// The alpha kernel's record for ray (x, y), or `None` near an edge.
fn alpha_expect(scene: &[Placed<'_>], x: u32, y: u32) -> Option<[u32; 4]> {
    let p = ray_at(x, y, SIZE);
    let mut hits = Vec::new();
    for (i, inst) in scene.iter().enumerate() {
        for t in 0..inst.cluster.tris.len() {
            if let Some(dist) = hit_t(corners(inst.cluster, t), inst.offset, p).ok()? {
                hits.push((dist, i, t));
            }
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let first = hits
        .iter()
        .find(|&&(_, i, t)| scene[i].opaque || alpha_keeps(u32::MAX, t as u32));
    Some(first.map_or([0; 4], |&(_, i, t)| [1, i as u32, t as u32, u32::MAX]))
}

fn alpha_root(
    b: &mut Builder,
    scene: Tlas,
    size: u32,
    flags: u32,
) -> (Allocation<[u32; 4]>, orr_gpu::GpuPtr<alpha::Root>) {
    let out = readback::<[u32; 4]>(u64::from(size * size), "alpha out");
    let root = b.upload(&[alpha::Root {
        out: out.gpu,
        scene,
        width: size,
        height: size,
        flags,
        pad: 0,
    }]);
    (out, root)
}

/// Three alpha-tested copies of a grid and an opaque back quad in a classic TLAS, traced
/// through the module and through its opacity-micromap variant.
#[test]
fn alpha_test_on_classic_tlas() {
    let mut front = cl::grid(
        0,
        8,
        8,
        [0.1, 0.1, 0.8, 0.8],
        |x, y| 0.2 + 0.1 * x - 0.05 * y,
        4,
    );
    front.geom_flags = GEOMETRY_CULL_DISABLE;
    let back = cl::back_cluster();
    let mut b = Builder::new();
    let blas = [
        classic_blas(&mut b, &front, false),
        classic_blas(&mut b, &back, true),
    ];
    let table = b.upload(&blas);
    let offsets = [[0.0, 0.0, 0.0], [0.13, 0.07, 0.1], [-0.08, 0.11, 0.2]];
    let mut instances: Vec<_> = offsets.iter().map(|&o| cl::instance(o, 5, 0)).collect();
    instances.push(cl::instance([0.0; 3], 6, 1));
    let tlas = b.build_tlas(table, &instances);

    let module = ShaderStage::new(&alpha::MODULE, "main", SpecData::default()).unwrap();
    let plain = ShaderStage {
        spirv_omm: &[],
        ..module
    };
    let d = device();
    let pipelines = [
        d.create_compute_pipeline(&plain, "alpha").unwrap(),
        d.create_compute_pipeline(&module, "alpha omm").unwrap(),
    ];
    let outs: Vec<_> = pipelines
        .iter()
        .map(|p| {
            let (out, root) = alpha_root(&mut b, tlas, SIZE, 0);
            traced(&mut b, p, root, SIZE);
            out
        })
        .collect();
    b.flush();

    let mut scene: Vec<Placed<'_>> = offsets
        .iter()
        .map(|&offset| Placed {
            cluster: &front,
            offset,
            opaque: false,
        })
        .collect();
    scene.push(Placed {
        cluster: &back,
        offset: [0.0; 3],
        opaque: true,
    });
    for (out, what) in outs.iter().zip(["module", "omm variant"]) {
        let got = out.as_slice().unwrap();
        let (mut checked, mut alpha_hits, mut wrong) = (0, 0, Vec::new());
        for y in 0..SIZE {
            for x in 0..SIZE {
                let Some(want) = alpha_expect(&scene, x, y) else {
                    continue;
                };
                checked += 1;
                alpha_hits += usize::from(want[0] == 1 && want[1] < 3);
                let g = got[(y * SIZE + x) as usize];
                if g != want && wrong.len() < 8 {
                    wrong.push((x, y, g, want));
                }
            }
        }
        assert!(
            checked > (SIZE * SIZE) as usize * 9 / 10,
            "{what}: {checked} rays checked"
        );
        assert!(
            alpha_hits > checked / 4,
            "{what}: {alpha_hits} alpha-tested hits"
        );
        assert!(
            wrong.is_empty(),
            "{what}: rays differing from the reference: {wrong:?}"
        );
    }
    assert_clean();
}

/// The opacity-micromap variant over the cluster probe's micromap scene, in 4-state and
/// forced 2-state rays, ray for ray equal to the probe kernel that is checked against the
/// CPU reference in the same submit.
#[test]
fn alpha_test_omm_variant_on_micromap_clusters() {
    let d = device();
    if !d.caps().omm || !d.caps().cluster.omm_in_clusters {
        log::warn!(
            "skipped: {} has no opacity micromaps in clusters",
            d.caps().device_name
        );
        return;
    }
    assert!(
        !alpha::MODULE.spirv_omm.is_empty(),
        "alpha.slang has an OMM variant"
    );
    let mut b = Builder::new();
    let data = b.upload(&cl::OMM_STATES);
    let triangles: Vec<_> = (0..4)
        .map(|i| MicromapTriangle::new(i, 0, OmmFormat::FourState))
        .collect();
    let usage = [MicromapUsage {
        count: 4,
        level: 0,
        format: OmmFormat::FourState,
    }];
    let desc = MicromapDesc {
        flags: RtasFlags::FAST_TRACE,
        usage: &usage,
        data: data.address(),
        triangles: b.upload(&triangles),
    };
    let sizes = d.micromap_sizes(&desc);
    let block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "micromap")
        .unwrap();
    let scratch = b.block(MemoryKind::DeviceLocal, sizes.scratch);
    let micromap = d.create_micromap(sizes.result_max, &block, 0).unwrap();
    b.list().build_micromap(&desc, &micromap, scratch.address());

    let (nx, ny) = cl::clamp_grid(10, 10);
    let mut front = cl::grid(
        3003,
        nx,
        ny,
        [0.05, 0.05, 0.95, 0.95],
        |x, y| 0.3 + 0.04 * x + 0.02 * y,
        1,
    );
    front.geom_flags = GEOMETRY_CULL_DISABLE;
    front.base_geom = 2;
    let table: [i32; 8] = [-1, -2, -3, 0, 1, 3, -5, 2];
    front.omm_index = Some((0..front.tris.len()).map(|t| table[t % 8]).collect());
    let clusters = vec![front, cl::back_cluster()];
    let flags = cl::data_flags() | RtasFlags::ALLOW_OMM;
    let clas = b.build_clas(&clusters, flags, micromap.address().0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0], vec![1]], None);
    let tlas = b.build_tlas(
        blas.addrs.gpu,
        &[
            cl::instance([0.0; 3], cl::FRONT_CUSTOM, 0),
            cl::instance([0.0; 3], cl::BACK_CUSTOM, 1),
        ],
    );
    let p = pipeline(&alpha::MODULE, "main");
    let modes = [cl::MODE_ALPHA, cl::MODE_ALPHA | cl::MODE_OMM_2_STATE];
    let runs: Vec<_> = modes
        .iter()
        .map(|&mode| {
            let probe = b.trace(tlas, mode);
            let flags = if mode & cl::MODE_OMM_2_STATE != 0 {
                FORCE_OPACITY_MICROMAP_2_STATE
            } else {
                0
            };
            let (out, root) = alpha_root(&mut b, tlas, cl::W, flags);
            traced(&mut b, &p, root, cl::W);
            (mode, probe, out)
        })
        .collect();
    b.flush();

    let refs = [
        cl::ref_inst(cl::FRONT_CUSTOM, 0, [0.0; 3], &[&clusters[0]]),
        cl::ref_inst(cl::BACK_CUSTOM, 1, [0.0; 3], &[&clusters[1]]),
    ];
    let mut results = Vec::new();
    for (mode, probe, out) in &runs {
        let probe = probe.as_slice().unwrap();
        cl::check(probe, &refs, *mode).assert_ok(&format!("probe kernel, mode {mode}"));
        let got = out.as_slice().unwrap();
        let mut wrong = Vec::new();
        for (i, g) in got.iter().enumerate() {
            let o = &probe[i * cl::REC..][..cl::REC];
            let want = if o[0] == 1 {
                [1, o[2], o[4], o[3]]
            } else {
                [0; 4]
            };
            if *g != want && wrong.len() < 8 {
                wrong.push((i as u32 % cl::W, i as u32 / cl::W, *g, want));
            }
        }
        assert!(
            wrong.is_empty(),
            "mode {mode}: rays differing from the probe kernel: {wrong:?}"
        );
        results.push(got.to_vec());
    }
    assert_ne!(
        results[0], results[1],
        "forcing two states changes the result"
    );
    drop((micromap, block));
    assert_clean();
}

/// The object-space corners of triangle `t` of `c` as position fetch returns them.
fn cluster_positions(c: &Cluster, t: u32) -> [[f32; 3]; 3] {
    let truncate = |v: f32| {
        if c.truncate == 0 {
            v
        } else {
            f32::from_bits(v.to_bits() & !((1u32 << c.truncate) - 1))
        }
    };
    corners(c, t as usize).map(|v| v.map(truncate))
}

/// One body behind a raygen and a compute entry over a PTLAS: the entries agree on every
/// ray, agree with the probe kernel (checked against the CPU reference) on hits, and the
/// fetched positions equal the CPU positions bit for bit.
#[test]
fn raygen_and_compute_entries_trace_one_ptlas() {
    let d = device();
    if !d.caps().ray_dispatch {
        log::warn!(
            "skipped: {} has no ray-dispatch pipelines",
            d.caps().device_name
        );
        return;
    }
    let base = cl::base_clusters();
    let mut b = Builder::new();
    let clas = b.build_clas(&base, cl::data_flags(), 0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0, 1, 2, 3], vec![4]], None);
    let limits = cl::ptlas_limits(2);
    let (dst, scratch) = b.ptlas_blocks(limits);
    let ops = cl::PtlasOps {
        writes: vec![
            cl::ptlas_write(0, 0, cl::FRONT_CUSTOM, [0.0, 0.0, cl::FRONT_Z], 0, true),
            cl::ptlas_write(1, 1, cl::BACK_CUSTOM, [0.0; 3], 1, false),
        ],
        ..Default::default()
    };
    b.build_ptlas(limits, blas.addrs.gpu, &ops, dst, scratch, false);
    let tlas = Tlas(dst.0);
    let probe = b.trace(tlas, 0);

    let stage = |entry| ShaderStage::new(&clusters::MODULE, entry, SpecData::default()).unwrap();
    let compute = d
        .create_compute_pipeline(&stage("main_compute"), "abi clusters compute")
        .unwrap();
    let rays = d
        .create_ray_dispatch_pipeline(&stage("main_rays"), "abi clusters raygen")
        .unwrap();
    let (w, h) = (cl::W, cl::H);
    let outs =
        [0, 1].map(|_| readback::<clusters::ClusterHit>(u64::from(w * h), "abi cluster hits"));
    let roots = outs.each_ref().map(|out| {
        b.upload(&[clusters::Root {
            out: out.gpu,
            scene: tlas,
            width: w,
            height: h,
            pad0: 0,
            pad1: 0,
        }])
    });
    let [gx, gy, _] = compute.workgroup();
    let list = b.list();
    list.barrier(
        Stages::ALL,
        Stages::COMPUTE | Stages::RAY,
        Hazards::ACCEL_STRUCT,
    );
    list.dispatch(&compute, roots[0], [w.div_ceil(gx), h.div_ceil(gy), 1]);
    list.dispatch_rays(&rays, roots[1], [w, h, 1]);
    list.barrier(
        Stages::COMPUTE | Stages::RAY,
        Stages::HOST,
        Hazards::empty(),
    );
    b.flush();

    let insts = [
        cl::ref_inst(
            cl::FRONT_CUSTOM,
            0,
            [0.0, 0.0, cl::FRONT_Z],
            &base[..4].iter().collect::<Vec<_>>(),
        ),
        cl::ref_inst(cl::BACK_CUSTOM, 1, [0.0; 3], &[&base[4]]),
    ];
    let probe = probe.as_slice().unwrap();
    cl::check(probe, &insts, 0).assert_ok("probe kernel");
    let (from_compute, from_rays) = (outs[0].as_slice().unwrap(), outs[1].as_slice().unwrap());
    let bits = |r: &clusters::ClusterHit| {
        let mut v = vec![r.hit, r.instance, r.cluster as u32, r.primitive];
        v.extend(r.positions.iter().flatten().map(|f| f.to_bits()));
        v
    };
    let (mut hits, mut wrong) = (0, Vec::new());
    for i in 0..(w * h) as usize {
        let (c, r) = (&from_compute[i], &from_rays[i]);
        let o = &probe[i * cl::REC..][..cl::REC];
        let mut bad = |what: &str| {
            if wrong.len() < 8 {
                wrong.push(format!("ray {i}: {what}"));
            }
        };
        if bits(c) != bits(r) {
            bad("raygen and compute differ");
        }
        if c.hit != o[0] {
            bad("hit differs from the probe kernel");
            continue;
        }
        if c.hit == 0 {
            continue;
        }
        hits += 1;
        if [c.instance, c.cluster as u32, c.primitive] != [o[2], o[3], o[4]] {
            bad("ids differ from the probe kernel");
        }
        let Some(cluster) = base.iter().find(|k| k.id == c.cluster as u32) else {
            bad("unknown cluster id");
            continue;
        };
        let want = cluster_positions(cluster, c.primitive);
        if c.positions.map(|v| v.map(f32::to_bits)) != want.map(|v| v.map(f32::to_bits)) {
            bad("positions differ from the CPU positions");
        }
    }
    assert!(hits > (w * h) as usize / 2, "only {hits} hits");
    assert!(wrong.is_empty(), "{}", wrong.join("; "));
    assert_clean();
}

/// Even layers of a stack are half-transparent grey, odd ones pass; an opaque triangle sits
/// in the stack's left half behind at most four layers.
fn candidate_scene() -> (Cluster, Cluster) {
    let n = 11;
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    for k in 0..n {
        let a = 0.4 * k as f32;
        let z = 0.1 + 0.07 * k as f32;
        for c in 0..3 {
            let t = a + c as f32 * std::f32::consts::TAU / 3.0;
            verts.push([0.5 + 0.6 * t.cos(), 0.5 + 0.6 * t.sin(), z]);
        }
        tris.push([3 * k, 3 * k + 1, 3 * k + 2]);
    }
    let layers = Cluster {
        verts,
        tris,
        geom_flags: 0,
        ..cl::grid(0, 1, 1, [0.0; 4], |_, _| 0.0, 4)
    };
    (
        layers,
        cl::grid(0, 1, 1, [0.0, 0.0, 0.5, 1.0], |_, _| 0.6, 4),
    )
}

#[test]
fn transmittance_and_translucent_resolve() {
    let (layers, opaque) = candidate_scene();
    let mut b = Builder::new();
    let blas = [
        classic_blas(&mut b, &layers, false),
        classic_blas(&mut b, &opaque, true),
    ];
    let table = b.upload(&blas);
    let tlas = b.build_tlas(
        table,
        &[cl::instance([0.0; 3], 0, 0), cl::instance([0.0; 3], 0, 1)],
    );
    let p = pipeline(&candidates::MODULE, "main");
    let limits = [64u32, 3];
    let outs = limits.map(|max_candidates| {
        let out = readback::<candidates::CandidateResult>(u64::from(SIZE * SIZE), "candidates out");
        let root = b.upload(&[candidates::Root {
            out: out.gpu,
            scene: tlas,
            width: SIZE,
            height: SIZE,
            max_candidates,
            pad: 0,
        }]);
        traced(&mut b, &p, root, SIZE);
        out
    });
    b.flush();

    let (full, capped) = (outs[0].as_slice().unwrap(), outs[1].as_slice().unwrap());
    let (mut checked, mut overflows, mut wrong) = (0, 0, Vec::new());
    for y in 0..SIZE {
        'rays: for x in 0..SIZE {
            let p = ray_at(x, y, SIZE);
            let mut on_ray = Vec::new();
            for t in 0..layers.tris.len() {
                match hit_t(corners(&layers, t), [0.0; 3], p) {
                    Err(()) => continue 'rays,
                    Ok(Some(d)) => on_ray.push((d, t % 2 == 0)),
                    Ok(None) => {}
                }
            }
            let mut wall = None;
            for t in 0..opaque.tris.len() {
                match hit_t(corners(&opaque, t), [0.0; 3], p) {
                    Err(()) => continue 'rays,
                    Ok(Some(d)) => wall = Some(d),
                    Ok(None) => {}
                }
            }
            checked += 1;
            let evens = on_ray.iter().filter(|c| c.1).count() as i32;
            let front = on_ray
                .iter()
                .filter(|c| c.1 && wall.is_none_or(|w| c.0 < w))
                .count() as i32;
            let i = (y * SIZE + x) as usize;
            let mut bad = |what: String| {
                if wrong.len() < 8 {
                    wrong.push(format!("({x},{y}) {what}"));
                }
            };
            let r = &full[i];
            let product = if wall.is_some() {
                0.0
            } else {
                0.5f32.powi(evens)
            };
            if r.transmittance != [product; 3] || r.overflow != 0 {
                bad(format!(
                    "transmittance {:?} overflow {} want {product}",
                    r.transmittance, r.overflow
                ));
            }
            if r.hit != u32::from(wall.is_some()) || (wall.is_some() && r.instance != 1) {
                bad(format!("hit {} instance {}", r.hit, r.instance));
            }
            let cover = 1.0 - 0.5f32.powi(front);
            let want = [0.5 * cover, 0.5 * cover, 0.5 * cover, cover];
            if r.layers != want {
                bad(format!("layers {:?} want {want:?}", r.layers));
            }
            if wall.is_none() {
                let c = &capped[i];
                let n = on_ray.len() as i32;
                let overflow = n > 3;
                overflows += usize::from(overflow);
                let lowest = 0.5f32.powi(evens.min(3));
                let highest = 0.5f32.powi((3 - (n - evens)).max(0).min(evens));
                let product = c.transmittance[0];
                let in_range = if overflow {
                    product >= lowest && product <= highest
                } else {
                    product == 0.5f32.powi(evens)
                };
                if c.overflow != u32::from(overflow) || !in_range {
                    bad(format!(
                        "capped: transmittance {product} overflow {} with {n} candidates",
                        c.overflow
                    ));
                }
            }
        }
    }
    assert!(
        checked > (SIZE * SIZE) as usize / 2,
        "{checked} rays checked"
    );
    assert!(overflows > 1000, "{overflows} capped traces overflow");
    assert!(wrong.is_empty(), "{}", wrong.join("; "));
    assert_clean();
}

#[test]
fn task_mesh_pixel_two_targets() {
    const W: u32 = 256;
    const H: u32 = 192;
    let d = device();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || (xorshift(&mut state) >> 40) as f32 / (1u64 << 24) as f32;
    // One triangle in each cell of an 8x4 grid over clip space.
    let mut vertices = Vec::new();
    for j in 0..4 {
        for i in 0..8 {
            let (x0, y0) = (-1.0 + 0.25 * i as f32, -1.0 + 0.5 * j as f32);
            for (cx, cy) in [(0.1, 0.1), (0.9, 0.2), (0.3, 0.9)] {
                let x = x0 + 0.25 * (cx + 0.08 * (next() - 0.5));
                let y = y0 + 0.5 * (cy + 0.08 * (next() - 0.5));
                vertices.push([x, y, 0.5, 1.0]);
            }
        }
    }
    let count = (vertices.len() / 3) as u32;
    let flat = [0.25, 0.5, 1.0, 2.0];
    let targets = [0, 1].map(|_| {
        d.create_texture(
            &TextureDesc::d2(Format::Rgba32Float, W, H, TextureUsage::COLOR),
            TextureMemory::Dedicated,
            "abi mesh target",
        )
        .unwrap()
    });
    let stage = |e| ShaderStage::new(&mesh::MODULE, e, SpecData::default()).unwrap();
    let colors = [ColorTarget::new(Format::Rgba32Float); 2];
    let p = d
        .create_mesh_pipeline(
            &MeshPipelineDesc {
                task: Some(stage("task_main")),
                pixel: Some(stage("pixel_main")),
                colors: &colors,
                ..MeshPipelineDesc::new(stage("mesh_main"))
            },
            "abi mesh",
        )
        .unwrap();
    let verts = upload(&vertices, "abi mesh vertices");
    let root = upload(
        &[mesh::Root {
            flat,
            vertices: verts.gpu,
            triangle_count: count,
            pad: 0,
        }],
        "abi mesh root",
    );
    let outs = [0, 1].map(|_| readback::<[f32; 4]>(u64::from(W * H), "abi mesh out"));
    let mut list = render().command_list().unwrap();
    for t in &targets {
        list.discard(t);
    }
    let attachments = targets.each_ref().map(|texture| ColorAttachment {
        texture,
        load: ColorLoad::Clear(ClearColor::Float([0.0; 4])),
        store: true,
    });
    list.begin_rendering(&Rendering {
        extent: [W, H],
        colors: &attachments,
        depth: None,
    });
    list.draw_mesh_tasks(&p, root.gpu, [1, 1, 1]);
    list.end_rendering();
    list.barrier(Stages::COLOR_OUT, Stages::TRANSFER, Hazards::empty());
    for (out, t) in outs.iter().zip(&targets) {
        list.copy_from_texture(out.gpu.cast(), t, &TextureRegion::whole_mip(t.desc(), 0));
    }
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run(list);

    let (coverage, value) = (outs[0].as_slice().unwrap(), outs[1].as_slice().unwrap());
    let to_px = |v: [f32; 4]| [(v[0] + 1.0) * 0.5 * W as f32, (v[1] + 1.0) * 0.5 * H as f32];
    let (mut checked, mut covered, mut wrong) = (0, 0, Vec::new());
    for y in 0..H {
        'px: for x in 0..W {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let mut inside = None;
            for t in 0..count as usize {
                let v = [0, 1, 2].map(|k| to_px(vertices[3 * t + k]));
                let dist = edge_distance(v, p);
                if dist.abs() < 0.01 {
                    continue 'px;
                }
                if dist > 0.0 {
                    inside = Some(t);
                }
            }
            checked += 1;
            let i = (y * W + x) as usize;
            let (want_c, want_v) = match inside {
                Some(t) => {
                    covered += 1;
                    ([1.0; 4], flat.map(|f| f * (t + 1) as f32))
                }
                None => ([0.0; 4], [0.0; 4]),
            };
            if (coverage[i] != want_c || value[i] != want_v) && wrong.len() < 8 {
                wrong.push((x, y, inside, coverage[i], value[i]));
            }
        }
    }
    assert!(
        checked > (W * H) as usize * 9 / 10,
        "{checked} pixels away from edges"
    );
    assert!(covered > checked / 8, "{covered} covered pixels");
    assert!(
        wrong.is_empty(),
        "pixels differing from the reference: {wrong:?}"
    );
    assert_clean();
}
