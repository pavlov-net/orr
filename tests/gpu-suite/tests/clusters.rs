//! Cluster acceleration structures and partitioned TLAS on orr-gpu: the cluster ray-query
//! probe's cases as regression tests, checked ray by ray against the CPU reference.

use std::time::Duration;

use gpu_suite::clusters::*;
use gpu_suite::{assert_clean, device, readback, render, upload};
use orr_gpu::rtas::consts::{
    GEOMETRY_CULL_DISABLE, GEOMETRY_NO_DUPLICATE_ANY_HIT, GEOMETRY_OPAQUE, PTLAS_GLOBAL_PARTITION,
};
use orr_gpu::rtas::{PtlasTranslate, PtlasUpdate};
use orr_gpu::{
    BlasLimits, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance, ClassicKind,
    ClusterVertexFormat, GpuAddress, Hazards, MemoryKind, MicromapDesc, MicromapTriangle,
    MicromapUsage, OmmFormat, PageBind, RtasFlags, RtasLimits, RtasMode, RtasOp, RtasOpDesc,
    Stages, Tlas,
};

/// The reference of a front instance over every cluster but the last and a back instance
/// over the last.
fn front_back(clusters: &[Cluster], front_offset: [f32; 3]) -> [RefInst; 2] {
    let (back, front) = clusters.split_last().unwrap();
    let front: Vec<&Cluster> = front.iter().collect();
    [
        ref_inst(FRONT_CUSTOM, 0, front_offset, &front),
        ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[back]),
    ]
}

fn two_instances(front_z: f32) -> [ClassicInstance; 2] {
    [
        instance([0.0, 0.0, front_z], FRONT_CUSTOM, 0),
        instance([0.0; 3], BACK_CUSTOM, 1),
    ]
}

/// CLAS of the base clusters at `clas_at` and cluster BLAS at `blas_at` (fresh blocks when
/// `None`) under a classic TLAS, traced opaque and forced non-opaque.
fn trace_base(b: &mut Builder, clas_at: Option<GpuAddress>, blas_at: Option<GpuAddress>) {
    let clusters = base_clusters();
    let clas = b.build_clas(&clusters, data_flags(), 0, clas_at, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0, 1, 2, 3], vec![4]], blas_at);
    let tlas = b.build_tlas(blas.addrs.gpu, &two_instances(FRONT_Z));
    let out = b.trace(tlas, 0);
    let out_fno = b.trace(tlas, MODE_FORCE_NON_OPAQUE);
    b.flush();
    let insts = front_back(&clusters, [0.0, 0.0, FRONT_Z]);
    let r = check(out.as_slice().unwrap(), &insts, 0);
    r.assert_ok("opaque");
    assert_eq!(r.clusters.len(), 5, "every cluster is hit");
    check(out_fno.as_slice().unwrap(), &insts, MODE_FORCE_NON_OPAQUE)
        .assert_ok("forced non-opaque");
}

#[test]
fn clas_tlas() {
    trace_base(&mut Builder::new(), None, None);
    assert_clean();
}

/// A cluster BLAS and a compacted ordinary BLAS in one TLAS; ordinary hits have no cluster
/// id.
#[test]
fn clas_mixed() {
    let d = device();
    let clusters = base_clusters();
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters[..4], data_flags(), 0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0, 1, 2, 3]], None);

    let back = &clusters[4];
    let desc = ClassicDesc {
        kind: ClassicKind::Blas,
        flags: data_flags(),
        geometry: ClassicGeometry::Triangles {
            positions: b.upload(&back.verts),
            vertex_count: back.verts.len() as u32,
            indices: b.upload(&back.tris),
            triangle_count: back.tris.len() as u32,
            opaque: true,
        },
    };
    let sizes = d.classic_sizes(&desc);
    let full_block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "full blas")
        .unwrap();
    let scratch = b.block(MemoryKind::DeviceLocal, sizes.scratch);
    let full = d
        .create_classic_as(ClassicKind::Blas, sizes.result_max, &full_block, 0)
        .unwrap();
    let compacted = readback::<u64>(1, "compacted size");
    b.list().build_classic(&ClassicBuild {
        desc: &desc,
        dst: &full,
        scratch: scratch.address(),
    });
    b.list().classic_compacted_size(&full, compacted.gpu);
    b.list()
        .barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    b.flush();
    let size = compacted.as_slice().unwrap()[0];
    assert!(
        size > 0 && size <= sizes.result_max,
        "compacted to {size} of {} bytes",
        sizes.result_max
    );
    let small_block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, size, "compacted blas")
        .unwrap();
    let small = d
        .create_classic_as(ClassicKind::Blas, size, &small_block, 0)
        .unwrap();
    b.list().compact_classic(&small, &full);
    let table = upload(
        &[blas.addrs.as_slice().unwrap()[0], small.address().0],
        "blas table",
    );
    let tlas = b.build_tlas(table.gpu, &two_instances(FRONT_Z));
    let out = b.trace(tlas, 0);
    let out_fno = b.trace(tlas, MODE_FORCE_NON_OPAQUE);
    b.flush();

    let mut insts = front_back(&clusters, [0.0, 0.0, FRONT_Z]);
    for t in &mut insts[1].tris {
        t.cluster = u32::MAX;
        t.geom = 0;
    }
    let r = check(out.as_slice().unwrap(), &insts, 0);
    r.assert_ok("opaque");
    assert!(r.clusters.contains(&u32::MAX), "ordinary hits seen");
    check(out_fno.as_slice().unwrap(), &insts, MODE_FORCE_NON_OPAQUE)
        .assert_ok("forced non-opaque");
    drop((full, small, full_block, small_block));
    assert_clean();
}

/// Writes into two partitions, then in place a BLAS update, a global-partition write and a
/// translation, then a second translation that replaces the first.
#[test]
fn clas_ptlas() {
    let clusters = base_clusters();
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters, data_flags(), 0, None, None);
    let blas = b.build_blas(
        clas.addrs.gpu,
        &[vec![0, 1, 2, 3], vec![4], vec![0, 2]],
        None,
    );
    let limits = ptlas_limits(2);
    let (dst, scratch) = b.ptlas_blocks(limits);
    let tlas = Tlas(dst.0);
    let back = [&clusters[4]];

    let ops = PtlasOps {
        writes: vec![
            ptlas_write(0, 0, FRONT_CUSTOM, [0.0, 0.0, FRONT_Z], 0, true),
            ptlas_write(1, 1, BACK_CUSTOM, [0.0; 3], 1, false),
        ],
        ..Default::default()
    };
    b.build_ptlas(limits, blas.addrs.gpu, &ops, dst, scratch, false);
    let out = b.trace(tlas, 0);
    b.flush();
    let insts = front_back(&clusters, [0.0, 0.0, FRONT_Z]);
    check(out.as_slice().unwrap(), &insts, 0).assert_ok("writes");

    let ops = PtlasOps {
        writes: vec![ptlas_write(
            1,
            PTLAS_GLOBAL_PARTITION,
            BACK_CUSTOM,
            [0.0, 0.125, 0.0],
            1,
            false,
        )],
        updates: vec![PtlasUpdate {
            blas: 2,
            instance_index: 0,
            pad: 0,
        }],
        translates: vec![PtlasTranslate {
            translation: [0.125, -0.0625, 0.0],
            partition: 0,
        }],
        ..Default::default()
    };
    b.build_ptlas(limits, blas.addrs.gpu, &ops, dst, scratch, true);
    let out = b.trace(tlas, 0);
    b.flush();
    let subset = [&clusters[0], &clusters[2]];
    let insts = [
        ref_inst(FRONT_CUSTOM, 0, [0.125, -0.0625, FRONT_Z], &subset),
        ref_inst(BACK_CUSTOM, 1, [0.0, 0.125, 0.0], &back),
    ];
    check(out.as_slice().unwrap(), &insts, 0).assert_ok("update, global write, translation");

    let ops = PtlasOps {
        translates: vec![PtlasTranslate {
            translation: [0.0625, 0.0, 0.0],
            partition: 0,
        }],
        ..Default::default()
    };
    b.build_ptlas(limits, blas.addrs.gpu, &ops, dst, scratch, true);
    let out = b.trace(tlas, 0);
    b.flush();
    let insts = [
        ref_inst(FRONT_CUSTOM, 0, [0.0625, 0.0, FRONT_Z], &subset),
        ref_inst(BACK_CUSTOM, 1, [0.0, 0.125, 0.0], &back),
    ];
    check(out.as_slice().unwrap(), &insts, 0).assert_ok("translation replaces");
    assert_clean();
}

/// Non-opaque clusters with per-triangle flags and geometry indices, a no-duplicate cluster
/// and an alpha test in the candidate handler.
#[test]
fn clas_alpha() {
    let (nx, ny) = clamp_grid(8, 8);
    let rect = [0.05, 0.05, 0.95, 0.48];
    let mut mixed = grid(2001, nx, ny, rect, |x, y| 0.3 + 0.05 * x - 0.03 * y, 1);
    mixed.geom_flags = GEOMETRY_CULL_DISABLE;
    mixed.base_geom = 20;
    mixed.tri_gflags = Some(
        (0..mixed.tris.len() as u32)
            .map(|t| (t % 4) | if t % 4 == 0 { GEOMETRY_OPAQUE << 29 } else { 0 })
            .collect(),
    );
    let rect = [0.05, 0.52, 0.95, 0.95];
    let mut nodup = grid(2002, 6, 6, rect, |x, y| 0.35 - 0.05 * x + 0.02 * y, 2);
    nodup.geom_flags = GEOMETRY_CULL_DISABLE | GEOMETRY_NO_DUPLICATE_ANY_HIT;
    nodup.base_geom = 30;
    let clusters = vec![mixed, nodup, back_cluster()];
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters, data_flags(), 0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0, 1], vec![2]], None);
    let tlas = b.build_tlas(blas.addrs.gpu, &two_instances(0.0));
    let out = b.trace(tlas, MODE_ALPHA);
    let out_fno = b.trace(tlas, MODE_ALPHA | MODE_FORCE_NON_OPAQUE);
    b.flush();
    let insts = front_back(&clusters, [0.0; 3]);
    let r = check(out.as_slice().unwrap(), &insts, MODE_ALPHA);
    r.assert_ok("alpha");
    assert!(r.candidates > 0, "candidates arrive");
    check(
        out_fno.as_slice().unwrap(),
        &insts,
        MODE_ALPHA | MODE_FORCE_NON_OPAQUE,
    )
    .assert_ok("alpha, forced non-opaque");
    assert_clean();
}

/// A 4-state level-0 micromap referenced from a CLAS, with special indices -1..-3 and -5,
/// traced in 4-state and forced 2-state mode.
#[test]
fn clas_omm() {
    let d = device();
    assert!(
        d.caps().cluster.omm_in_clusters,
        "the dev GPU has OMM in clusters"
    );
    let mut b = Builder::new();
    let data = b.upload(&OMM_STATES);
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

    let (nx, ny) = clamp_grid(10, 10);
    let rect = [0.05, 0.05, 0.95, 0.95];
    let mut front = grid(3003, nx, ny, rect, |x, y| 0.3 + 0.04 * x + 0.02 * y, 1);
    front.geom_flags = GEOMETRY_CULL_DISABLE;
    front.base_geom = 2;
    let table: [i32; 8] = [-1, -2, -3, 0, 1, 3, -5, 2];
    front.omm_index = Some((0..front.tris.len()).map(|t| table[t % 8]).collect());
    let clusters = vec![front, back_cluster()];
    let flags = data_flags() | RtasFlags::ALLOW_OMM;
    let clas = b.build_clas(&clusters, flags, micromap.address().0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0], vec![1]], None);
    let tlas = b.build_tlas(blas.addrs.gpu, &two_instances(0.0));
    let out = b.trace(tlas, MODE_ALPHA);
    let out2 = b.trace(tlas, MODE_ALPHA | MODE_OMM_2_STATE);
    b.flush();
    let insts = front_back(&clusters, [0.0; 3]);
    check(out.as_slice().unwrap(), &insts, MODE_ALPHA).assert_ok("4-state");
    check(
        out2.as_slice().unwrap(),
        &insts,
        MODE_ALPHA | MODE_OMM_2_STATE,
    )
    .assert_ok("2-state");
    let mut plain = clusters[0].clone();
    plain.omm_index = None;
    let ignored = check(
        out.as_slice().unwrap(),
        &front_back(&[plain, back_cluster()], [0.0; 3]),
        MODE_ALPHA,
    );
    assert!(!ignored.bad.is_empty(), "the micromap changes the result");
    drop((micromap, block));
    assert_clean();
}

/// Two templates, one with instantiation bounds, instantiated with displaced vertices and
/// cluster id and geometry index offsets.
#[test]
fn clas_template() {
    let d = device();
    let base = base_clusters();
    let back = [base[4].clone()];
    let templates = [base[1].clone(), base[3].clone()];
    let displace = |c: &Cluster, s: f32| -> Vec<[f32; 3]> {
        c.verts
            .iter()
            .map(|v| {
                [
                    v[0] + 0.01 * (31.0 * v[1]).sin(),
                    v[1] + 0.01 * (29.0 * v[0]).cos(),
                    v[2] + s * (13.0 * v[0] + 7.0 * v[1]).sin(),
                ]
            })
            .collect()
    };
    let d3 = displace(&templates[1], 0.04);
    let d1 = displace(&templates[0], -0.03);
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for v in &templates[1].verts {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k] - 0.1);
            hi[k] = hi[k].max(v[k] + 0.1);
        }
    }
    let mut b = Builder::new();
    let bounds = b.upload(&[lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], 0.0, 0.0]);
    let align = u64::from(d.caps().cluster.align_bounds);
    assert!(bounds.address().0.is_multiple_of(align));
    let records = template_records(&mut b, &templates, &[0, bounds.address().0]);
    let mut root = zeroed_root();
    root.templates = b.upload(&records);
    root.op = OP_TEMPLATE;
    let desc = clas_desc(
        RtasOp::BuildTemplates,
        RtasMode::Implicit,
        data_flags(),
        &templates,
    );
    let tmpl = b.cluster_op(&desc, root, 2, None, None);

    let v3 = b.upload(&d3);
    let v1 = b.upload(&d1);
    // CLAS addresses: the back cluster, then the two instantiations.
    let table = readback::<u64>(3, "clas table");
    let mut root = zeroed_root();
    root.table = tmpl.addrs.gpu;
    root.instantiate = b.upload(&[
        instantiate(1, v3.address(), 100, 5),
        instantiate(0, v1.address(), 200, 0),
    ]);
    root.op = OP_INSTANTIATE;
    let desc = clas_desc(
        RtasOp::InstantiateTemplates,
        RtasMode::Implicit,
        data_flags(),
        &templates,
    );
    let inst = b.cluster_op(&desc, root, 2, None, Some(table.gpu.offset(1)));
    b.build_clas(&back, data_flags(), 0, None, Some(table.gpu));
    let blas = b.build_blas(table.gpu, &[vec![1, 2], vec![0]], None);
    let tlas = b.build_tlas(blas.addrs.gpu, &two_instances(0.0));
    let out = b.trace(tlas, 0);
    b.flush();
    let insts = [
        RefInst {
            custom: FRONT_CUSTOM,
            index: 0,
            offset: [0.0; 3],
            tris: [
                ref_tris(&templates[1], Some(&d3), 100, 5),
                ref_tris(&templates[0], Some(&d1), 200, 0),
            ]
            .concat(),
        },
        ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&back[0]]),
    ];
    let r = check(out.as_slice().unwrap(), &insts, 0);
    r.assert_ok("instantiated");
    assert_eq!(
        r.clusters.len(),
        3,
        "both instantiations and the back are hit"
    );
    assert!(inst.sizes.as_slice().unwrap().iter().all(|&s| s > 0));
    assert_clean();
}

/// CLAS and cluster BLAS built into a sparse range whose pages are bound in reverse order
/// from two blocks, with results straddling pages.
#[test]
fn clas_sparse() {
    let d = device();
    let clusters = base_clusters();
    let clas_desc = clas_desc(
        RtasOp::BuildClas,
        RtasMode::Implicit,
        data_flags(),
        &clusters,
    );
    let clas_bytes = d.rtas_sizes(&clas_desc).result_max;
    let blas_bytes = d
        .rtas_sizes(&RtasOpDesc {
            op: RtasOp::BuildClusterBlas,
            mode: RtasMode::Implicit,
            flags: RtasFlags::FAST_TRACE,
            vertex_format: ClusterVertexFormat::F32x3,
            limits: RtasLimits::Blas(BlasLimits {
                max_args: 2,
                max_clas_per_blas: 4,
                max_total_clas: 5,
            }),
        })
        .result_max;
    let page = d.gpu_malloc_sparse::<u8>(1, "page size").unwrap().page_size;
    let clas_off = page - 512;
    let blas_off = (clas_off + clas_bytes).next_multiple_of(page / 2) + page / 2 - 256;
    let bound = (blas_off + blas_bytes).div_ceil(page);
    let range = d
        .gpu_malloc_sparse::<u8>((bound + 2) * page, "sparse clas")
        .unwrap();
    let half = bound.div_ceil(2);
    let pages_a = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, half * page, "pages a")
        .unwrap();
    let pages_b = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, half * page, "pages b")
        .unwrap();
    // Page i takes slot bound - 1 - i: odd slots from block a, even ones from block b.
    let binds: Vec<PageBind<'_>> = (0..bound)
        .map(|i| {
            let slot = bound - 1 - i;
            let block = if slot % 2 == 1 { &pages_a } else { &pages_b };
            PageBind {
                first_page: i,
                count: 1,
                memory: Some((block, slot / 2 * page)),
            }
        })
        .collect();
    let bound_at = d.bind_sparse_pages(&range, &binds).unwrap();
    let mut b = Builder::new();
    b.waits.push(bound_at);
    let base = range.gpu.address().0;
    trace_base(
        &mut b,
        Some(GpuAddress(base + clas_off)),
        Some(GpuAddress(base + blas_off)),
    );
    drop(b);
    let unbind = [PageBind {
        first_page: 0,
        count: bound,
        memory: None,
    }];
    let unbound = d.bind_sparse_pages(&range, &unbind).unwrap();
    render().wait(unbound, Duration::from_secs(10)).unwrap();
    drop((range, pages_a, pages_b));
    assert_clean();
}

/// Implicit builds write each CLAS's built size; a GPU pool allocator takes them at
/// 128-byte granularity and the CLAS move there, where they trace as before.
#[test]
fn clas_pool_sizes_and_moves() {
    let clusters = base_clusters();
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters, data_flags(), 0, None, None);
    let n = clusters.len() as u32;
    let pool = b.block(MemoryKind::DeviceLocal, clas.reserved.result_max);
    let mut used = readback::<u32>(1, "pool used");
    used.as_mut_slice().unwrap()[0] = 0;
    let moved = readback::<u64>(u64::from(n), "pool addresses");
    let mut root = zeroed_root();
    root.sizes = clas.sizes.gpu;
    root.pool_used = used.gpu;
    root.pool = pool.address().0;
    root.refs = moved.gpu;
    root.n = n;
    root.op = OP_POOL;
    b.args(root);
    let mut root = zeroed_root();
    root.table = clas.addrs.gpu;
    root.moves = b.upload(&(0..u64::from(n)).map(move_record).collect::<Vec<_>>());
    root.op = OP_MOVE;
    let desc = move_desc(n, clas.reserved.result_max);
    b.cluster_op(&desc, root, n, None, Some(moved.gpu));
    let blas = b.build_blas(moved.gpu, &[vec![0, 1, 2, 3], vec![4]], None);
    let tlas = b.build_tlas(blas.addrs.gpu, &two_instances(FRONT_Z));
    let out = b.trace(tlas, 0);
    b.flush();

    let sizes = clas.sizes.as_slice().unwrap();
    assert!(sizes.iter().all(|&s| s > 0), "built sizes {sizes:?}");
    let sum: u32 = sizes.iter().map(|s| s.next_multiple_of(128)).sum();
    assert_eq!(
        used.as_slice().unwrap()[0],
        sum,
        "pool bytes against built sizes {sizes:?}"
    );
    assert!(u64::from(sum) <= clas.reserved.result_max);
    let start = pool.address().0;
    let end = start + u64::from(sum);
    assert!(
        moved
            .as_slice()
            .unwrap()
            .iter()
            .all(|&a| a >= start && a < end && a.is_multiple_of(128))
    );
    let insts = front_back(&clusters, [0.0, 0.0, FRONT_Z]);
    check(out.as_slice().unwrap(), &insts, 0).assert_ok("moved CLAS");
    assert_clean();
}

/// A PTLAS rewritten in place every frame with two frames in flight: each frame's trace
/// sees that frame's instances.
#[test]
fn ptlas_every_frame_two_in_flight() {
    const FRAMES: usize = 6;
    let clusters = base_clusters();
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters, data_flags(), 0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0, 1, 2, 3], vec![4]], None);
    b.flush();
    let limits = ptlas_limits(2);
    let (dst, scratch) = b.ptlas_blocks(limits);
    let slots = [0, 1].map(|_| readback::<u32>(u64::from(W * H) * REC as u64, "frame rays"));
    let offset = |f: usize| [0.02 * f as f32, 0.01 * f as f32, FRONT_Z];
    let check_frame = |f: usize| {
        let insts = front_back(&clusters, offset(f));
        check(slots[f % 2].as_slice().unwrap(), &insts, 0).assert_ok(&format!("frame {f}"));
    };
    let mut points = Vec::new();
    for f in 0..FRAMES {
        if f >= 2 {
            render()
                .wait(points[f - 2], Duration::from_secs(10))
                .unwrap();
            check_frame(f - 2);
        }
        let ops = PtlasOps {
            writes: vec![
                ptlas_write(0, 0, FRONT_CUSTOM, offset(f), 0, false),
                ptlas_write(1, 1, BACK_CUSTOM, [0.0; 3], 1, false),
            ],
            ..Default::default()
        };
        b.build_ptlas(limits, blas.addrs.gpu, &ops, dst, scratch, f > 0);
        b.trace_into(Tlas(dst.0), 0, slots[f % 2].gpu);
        points.push(b.submit());
    }
    for (f, &point) in points.iter().enumerate().skip(FRAMES - 2) {
        render().wait(point, Duration::from_secs(10)).unwrap();
        check_frame(f);
    }
    assert_clean();
}
