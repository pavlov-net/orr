//! Cluster acceleration structures and partitioned TLAS on orr-gpu: the cluster ray-query
//! probe's cases as regression tests, checked ray by ray against the CPU reference.

use std::collections::BTreeMap;
use std::time::Duration;

use gpu_suite::clusters::*;
use gpu_suite::shaders::cluster_trace;
use gpu_suite::{assert_clean, device, noise, readback, render, upload};
use orr_gpu::rtas::consts::{
    GEOMETRY_CULL_DISABLE, GEOMETRY_NO_DUPLICATE_ANY_HIT, GEOMETRY_OPAQUE, PTLAS_GLOBAL_PARTITION,
};
use orr_gpu::rtas::{PtlasTranslate, PtlasUpdate};
use orr_gpu::{
    BlasLimits, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance, ClassicKind,
    ClusterVertexFormat, GpuAddress, Hazards, LssEndCaps, LssIndexing, MemoryKind, MicromapDesc,
    MicromapTriangle, MicromapUsage, OmmFormat, PageBind, RtasFlags, RtasLimits, RtasMode, RtasOp,
    RtasOpDesc, ShaderStage, SpecData, Stages, Tlas,
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

const LSS_CUSTOM: u32 = 55;
/// Rays this close to a segment's silhouette or end are not checked.
const LSS_EDGE: f32 = 1e-3;
const LSS_T_EPS: f32 = 2e-4;
const CLUSTER_T_EPS: f32 = 1e-4;

/// One segment over the unit square at depth `z`, radius `r` at both ends.
#[derive(Clone, Copy)]
struct Segment {
    a: [f32; 2],
    b: [f32; 2],
    z: f32,
    r: f32,
}

/// 24 segments alternating in front of the cluster instance (z 0.12) and behind it (z 0.6).
fn segments() -> Vec<Segment> {
    let mut rnd = noise(0x2545_f491);
    (0..24)
        .map(|i| Segment {
            a: [0.05 + 0.9 * rnd(), 0.05 + 0.9 * rnd()],
            b: [0.05 + 0.9 * rnd(), 0.05 + 0.9 * rnd()],
            z: if i % 2 == 0 { 0.12 } else { 0.6 },
            r: 0.008 + 0.022 * rnd(),
        })
        .collect()
}

enum Want {
    Miss,
    Cluster(usize, f32),
    Lss(usize, f32),
}

/// The nearest hit of ray (x, y) among the cluster instance `front` and `segs`; `None`
/// near any triangle edge, segment silhouette or segment end.
fn lss_reference(front: &RefInst, segs: &[Segment], x: u32, y: u32) -> Option<Want> {
    let mut best = match reference_hit(std::slice::from_ref(front), x, y)? {
        Some((_, tri, t)) => Want::Cluster(tri, t),
        None => Want::Miss,
    };
    let p = [(x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32];
    for (i, s) in segs.iter().enumerate() {
        let d = [s.b[0] - s.a[0], s.b[1] - s.a[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let u = ((p[0] - s.a[0]) * d[0] + (p[1] - s.a[1]) * d[1]) / (len * len);
        let q = [s.a[0] + u * d[0] - p[0], s.a[1] + u * d[1] - p[1]];
        let dist = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let near_end = (u * len).abs() < LSS_EDGE || ((u - 1.0) * len).abs() < LSS_EDGE;
        let inside = (0.0..=1.0).contains(&u);
        if (near_end && dist < s.r + LSS_EDGE) || (inside && (dist - s.r).abs() < LSS_EDGE) {
            return None;
        }
        let best_t = match best {
            Want::Miss => f32::INFINITY,
            Want::Cluster(_, t) | Want::Lss(_, t) => t,
        };
        if inside && dist < s.r {
            let t = 1.0 + s.z - (s.r * s.r - dist * dist).sqrt();
            if t < best_t {
                best = Want::Lss(i, t);
            }
        }
    }
    Some(best)
}

/// Checks every ray record of `out` against the reference; returns the LSS and cluster hit
/// counts.
fn check_lss(out: &[u32], front: &RefInst, segs: &[Segment], what: &str) -> (u32, u32) {
    let (mut lss, mut cluster) = (0, 0);
    let mut bad: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut first = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let o = &out[(y * W + x) as usize * REC..][..REC];
            let Some(want) = lss_reference(front, segs, x, y) else {
                continue;
            };
            let mut fail = |kind: &'static str, detail: String| {
                *bad.entry(kind).or_insert(0) += 1;
                if first.len() < 4 {
                    first.push(format!("({x},{y}) {kind}: {detail}"));
                }
            };
            if o[19] != 0x600d {
                fail("unwritten", format!("{:#x}", o[19]));
                continue;
            }
            let t = f32::from_bits(o[6]);
            match want {
                Want::Miss if o[0] != 0 => fail("hit_on_miss", format!("instance {}", o[1])),
                Want::Miss => {}
                Want::Cluster(tri, want_t) => {
                    let rt = &front.tris[tri];
                    if o[0] != 1 {
                        fail("miss_on_cluster", String::new());
                    } else if o[1..6] != [FRONT_CUSTOM, 0, rt.cluster, rt.prim, rt.geom] {
                        fail("cluster_ids", format!("{:?}", &o[1..6]));
                    } else if (t - want_t).abs() > CLUSTER_T_EPS {
                        fail("cluster_t", format!("{t} want {want_t}"));
                    } else {
                        cluster += 1;
                    }
                }
                Want::Lss(seg, want_t) => {
                    if o[0] != 1 {
                        fail("miss_on_lss", format!("segment {seg}"));
                    } else if o[1..6] != [LSS_CUSTOM, 1, u32::MAX, seg as u32, 0] {
                        fail("lss_ids", format!("{:?} want segment {seg}", &o[1..6]));
                    } else if (t - want_t).abs() > LSS_T_EPS {
                        fail("lss_t", format!("{t} want {want_t}"));
                    } else {
                        lss += 1;
                    }
                }
            }
        }
    }
    assert!(
        bad.is_empty() && lss > 0 && cluster > 0,
        "{what}: {lss} LSS and {cluster} cluster hits, mismatches {bad:?}: {}",
        first.join("; ")
    );
    (lss, cluster)
}

/// An LSS BLAS over `segs`, as list segments or as successive indices over the same
/// vertices.
fn lss_blas(b: &mut Builder, segs: &[Segment], indexing: LssIndexing) -> GpuAddress {
    let d = device();
    let verts: Vec<[f32; 3]> = segs
        .iter()
        .flat_map(|s| [[s.a[0], s.a[1], s.z], [s.b[0], s.b[1], s.z]])
        .collect();
    let radii: Vec<f32> = segs.iter().flat_map(|s| [s.r, s.r]).collect();
    let indices: Vec<u32> = match indexing {
        LssIndexing::List => (0..verts.len() as u32).collect(),
        LssIndexing::Successive => (0..segs.len() as u32).map(|i| 2 * i).collect(),
    };
    let desc = ClassicDesc {
        kind: ClassicKind::Blas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Lss {
            positions: b.upload(&verts),
            radii: b.upload(&radii),
            indices: b.upload(&indices),
            segment_count: segs.len() as u32,
            indexing,
            end_caps: LssEndCaps::None,
            opaque: true,
        },
    };
    let sizes = d.classic_sizes(&desc);
    let block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "lss blas")
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
    let address = blas.address();
    b.keep_any(blas);
    b.keep(block);
    address
}

/// A classic TLAS and a PTLAS that mix a cluster BLAS with an LSS BLAS (list indexing in
/// one, successive in the other) give the reference hits from the compute entry and from
/// the raygen entry of one kernel. LSS hits report their segment and cluster id -1.
#[test]
fn lss_beside_clusters() {
    let d = device();
    if !gpu_suite::require(d.caps().lss, "no linear swept spheres") {
        return;
    }
    let raygen = d.caps().ray_dispatch.then(|| {
        let stage =
            ShaderStage::new(&cluster_trace::MODULE, "raygen", SpecData::default()).unwrap();
        d.create_ray_dispatch_pipeline(&stage, "lss raygen")
            .unwrap()
    });
    let clusters = base_clusters();
    let segs = segments();
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters[..4], data_flags(), 0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0, 1, 2, 3]], None);
    let list_blas = lss_blas(&mut b, &segs, LssIndexing::List);
    let successive_blas = lss_blas(&mut b, &segs, LssIndexing::Successive);
    b.flush();
    let cluster_blas = blas.addrs.as_slice().unwrap()[0];
    let tlas_table = b.upload(&[cluster_blas, list_blas.0]);
    let ptlas_table = b.upload(&[cluster_blas, successive_blas.0]);
    let tlas = b.build_tlas(
        tlas_table,
        &[
            instance([0.0, 0.0, FRONT_Z], FRONT_CUSTOM, 0),
            instance([0.0; 3], LSS_CUSTOM, 1),
        ],
    );
    let limits = ptlas_limits(2);
    let (dst, scratch) = b.ptlas_blocks(limits);
    let ops = PtlasOps {
        writes: vec![
            ptlas_write(0, 0, FRONT_CUSTOM, [0.0, 0.0, FRONT_Z], 0, false),
            ptlas_write(1, 1, LSS_CUSTOM, [0.0; 3], 1, false),
        ],
        ..Default::default()
    };
    b.build_ptlas(limits, ptlas_table, &ops, dst, scratch, false);

    let front = ref_inst(
        FRONT_CUSTOM,
        0,
        [0.0, 0.0, FRONT_Z],
        &clusters[..4].iter().collect::<Vec<_>>(),
    );
    for (label, scene) in [("tlas", tlas), ("ptlas", Tlas(dst.0))] {
        let compute = b.trace(scene, 0);
        let rays = raygen.as_ref().map(|p| b.trace_rays(p, scene, 0));
        b.flush();
        let compute = compute.as_slice().unwrap();
        let hits = check_lss(compute, &front, &segs, &format!("{label} compute"));
        if let Some(rays) = rays {
            let rays = rays.as_slice().unwrap();
            assert_eq!(
                check_lss(rays, &front, &segs, &format!("{label} raygen")),
                hits,
                "{label}: raygen and compute hit counts"
            );
            let differ = compute
                .chunks(REC)
                .zip(rays.chunks(REC))
                .filter(|(c, r)| c[..7] != r[..7])
                .count();
            assert_eq!(
                differ, 0,
                "{label}: rays whose raygen and compute hits differ"
            );
        }
        log::info!("{label}: {} LSS and {} cluster hits", hits.0, hits.1);
    }
    assert_clean();
}
