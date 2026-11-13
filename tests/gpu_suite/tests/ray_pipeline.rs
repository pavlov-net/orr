//! Ray tracing pipelines on cluster scenes, checked ray by ray against the CPU reference of
//! the cluster ray-query tests: the hit each shader reports, the record each one reads, ray
//! types by offset and stride, hit objects, stores across reorders, invokes and calls,
//! procedural groups, indirect launches, deferred creation and compute stand-ins.

use std::time::SystemTime;

use gpu_suite::clusters::*;
use gpu_suite::shaders::ray_pipeline::{self, Dumped, HitData, Root};
use gpu_suite::{assert_clean, device, pipeline, readback, require, upload};
use orr_gpu::rtas::consts::{GEOMETRY_CULL_DISABLE, GEOMETRY_OPAQUE};
use orr_gpu::{
    Allocation, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicKind, CompileThreads,
    DispatchArgs, GpuPtr, Hazards, HitGroup, HitTable, MemoryKind, RayPipeline, RayPipelineDesc,
    RayPipelineFlags, RtasFlags, ShaderRecord, ShaderStage, SpecData, Stages, Tlas,
};
use orr_shaders::bytemuck::{self, Zeroable};

const SEEN: usize = 8;
const NONE: u32 = u32::MAX;
const MODE_FORCE_NON_OPAQUE: u32 = 1;
const MODE_ALPHA: u32 = 2;
const MODE_BOX: u32 = 4;
const MODE_REDIRECT: u32 = 8;
const SHADER_HIT_A: u32 = 1;
const SHADER_HIT_B: u32 = 2;
const SHADER_MISS: u32 = 3;
const SHADER_BOX: u32 = 4;
const RAY_TYPES: u32 = 2;
const MISS_VALUES: [u32; 2] = [0x111, 0x222];
const CALL_ADD: u32 = 5;
const CALL_SCALE: u32 = 3;

// Stages, raygens, misses, hit groups and callables of `desc`.
const STAGES: [&str; 13] = [
    "closest",
    "traced",
    "visibility",
    "hit_a",
    "hit_b",
    "cover",
    "miss",
    "scale",
    "box",
    "box_hit",
    "vis_hit",
    "vis_miss",
    "vis_call",
];
const RAYGEN_CLOSEST: u32 = 0;
const RAYGEN_TRACED: u32 = 1;
const RAYGEN_VISIBILITY: u32 = 2;
const GROUP_A: u32 = 0;
const GROUP_B: u32 = 1;
const GROUP_BOX: u32 = 2;
const GROUP_VIS: u32 = 3;
const MISS_VIS: u32 = 2;
const CALL_SCALE_INDEX: u32 = 0;
const CALL_VIS: u32 = 1;

/// `MissData`: the value and a pad word.
type MissData = [u32; 2];
/// `CallData`: what to add and what to scale by.
type CallData = [u32; 2];

fn stages(salt: &[u8; 4]) -> Vec<ShaderStage<'_>> {
    let spec = SpecData {
        entries: ray_pipeline::Spec::ENTRIES,
        bytes: salt,
    };
    STAGES
        .iter()
        .map(|e| ShaderStage::new(&ray_pipeline::MODULE, e, spec).unwrap())
        .collect()
}

/// The suite's pipeline, its closest-hit shaders salted with `salt`.
fn create(salt: u32, threads: CompileThreads<'_>, flags: RayPipelineFlags) -> RayPipeline {
    let salt = salt.to_le_bytes();
    let stages = stages(&salt);
    let misses: Vec<MissData> = MISS_VALUES.iter().map(|&value| [value, 0]).collect();
    let call: CallData = [CALL_ADD, CALL_SCALE];
    let desc = RayPipelineDesc {
        stages: &stages,
        raygens: &[
            ShaderRecord::new(0),
            ShaderRecord::new(1),
            ShaderRecord::new(2),
        ],
        misses: &[
            ShaderRecord {
                stage: 6,
                data: bytemuck::bytes_of(&misses[0]),
            },
            ShaderRecord {
                stage: 6,
                data: bytemuck::bytes_of(&misses[1]),
            },
            ShaderRecord::new(11),
        ],
        hit_groups: &[
            HitGroup {
                closest_hit: Some(3),
                any_hit: Some(5),
                intersection: None,
            },
            HitGroup {
                closest_hit: Some(4),
                any_hit: Some(5),
                intersection: None,
            },
            HitGroup {
                closest_hit: Some(9),
                any_hit: None,
                intersection: Some(8),
            },
            HitGroup {
                closest_hit: Some(10),
                ..HitGroup::default()
            },
        ],
        callables: &[
            ShaderRecord {
                stage: 7,
                data: bytemuck::bytes_of(&call),
            },
            ShaderRecord::new(12),
        ],
        max_recursion: 1,
        flags: flags | RayPipelineFlags::CLUSTERS,
        threads,
    };
    device()
        .create_ray_pipeline(&desc, "ray pipeline suite")
        .unwrap()
}

/// A hit table of `count` records, each `fill` names the group and data of, or zeros.
struct Table {
    _records: Allocation<u8>,
    table: HitTable,
}

fn hit_table(p: &RayPipeline, count: u64, fill: impl Fn(u64) -> Option<(u32, Vec<u8>)>) -> Table {
    let stride = p.hit_stride();
    let mut records = device()
        .gpu_malloc::<u8>(MemoryKind::Upload, stride * count, "hit records")
        .unwrap();
    let bytes = records.as_mut_slice().unwrap();
    bytes.fill(0);
    for i in 0..count {
        if let Some((group, data)) = fill(i) {
            let at = (i * stride) as usize;
            p.write_hit_record(&mut bytes[at..at + stride as usize], group, &data);
        }
    }
    Table {
        table: HitTable {
            address: records.gpu().address(),
            stride,
            count,
        },
        _records: records,
    }
}

fn hit_data(custom: u32, geometry: u32, ray_type: u32) -> Vec<u8> {
    bytemuck::bytes_of(&HitData {
        custom,
        geometry,
        tag: u64::from(ray_type) << 32 | 0x5eed,
    })
    .to_vec()
}

/// Records of the triangle groups for instances whose (custom index, contribution, geometry
/// indices) `instances` gives: one per ray type, A for type 0 and B for type 1.
fn triangle_table(p: &RayPipeline, instances: &[(u32, u32, Vec<u32>)]) -> Table {
    let count = instances
        .iter()
        .map(|(_, hit_group, geoms)| {
            u64::from(hit_group + geoms.iter().max().unwrap() * RAY_TYPES + RAY_TYPES)
        })
        .max()
        .unwrap();
    hit_table(p, count, |i| {
        instances.iter().find_map(|(custom, hit_group, geoms)| {
            let local = (i as u32).checked_sub(*hit_group)?;
            let geom = local / RAY_TYPES;
            let ray_type = local % RAY_TYPES;
            geoms.contains(&geom).then(|| {
                (
                    [GROUP_A, GROUP_B][ray_type as usize],
                    hit_data(*custom, geom, ray_type),
                )
            })
        })
    })
}

struct Launch {
    out: Allocation<u32>,
    seen: Allocation<u32>,
}

fn root(scene: Tlas, launch: &Launch, mode: u32, ray_type: u32, callable: u32) -> Root {
    Root {
        out: launch.out.gpu(),
        seen: launch.seen.gpu(),
        scene,
        dumped: GpuPtr::NULL,
        width: W,
        height: H,
        mode,
        ray_type,
        ray_types: RAY_TYPES,
        callable,
        pad0: 0,
        pad1: 0,
    }
}

fn launch_buffers() -> Launch {
    Launch {
        out: readback::<u32>(u64::from(W * H) * REC as u64, "rays"),
        seen: readback::<u32>(u64::from(W * H) * SEEN as u64, "seen"),
    }
}

/// Traces every ray of the unit square from `raygen` into fresh buffers.
fn trace(
    b: &mut Builder,
    p: &RayPipeline,
    raygen: u32,
    hits: &HitTable,
    root_of: impl Fn(&Launch) -> Root,
) -> Launch {
    let l = launch_buffers();
    let r = root_of(&l);
    let list = b.list();
    list.barrier(Stages::ALL, Stages::RAY, Hazards::ACCEL_STRUCT);
    list.trace_rays(p, raygen, hits, &r, [W, H, 1]);
    list.barrier(Stages::RAY, Stages::HOST, Hazards::empty());
    l
}

/// Where a hit's record is: (custom index, contribution) per reference instance.
struct Placed<'a> {
    insts: &'a [RefInst],
    hit_groups: &'a [u32],
}

/// Checks what the shaders saw against the reference: the record the closest-hit or miss
/// shader read, the shader the ray type selects (the other one's with MODE_REDIRECT), the
/// callable's result, any-hit records, and for `traced` the hit object's geometry index,
/// record index and record data.
fn check_seen(
    seen: &[u32],
    placed: &Placed<'_>,
    mode: u32,
    ray_type: u32,
    traced: bool,
    what: &str,
) {
    let ran = if mode & MODE_REDIRECT != 0 {
        1 - ray_type
    } else {
        ray_type
    };
    let mode = mode & !MODE_REDIRECT;
    let reference = Reference::new(placed.insts);
    let mut bad = 0;
    let mut first = Vec::new();
    let mut hits = 0;
    for y in 0..H {
        for x in 0..W {
            let Some(hit) = reference.committed_hit(x, y, mode) else {
                continue;
            };
            let s = &seen[(y * W + x) as usize * SEEN..][..SEEN];
            let want: [u32; 8] = match hit {
                Some((ii, ti, _)) => {
                    hits += 1;
                    let inst = &placed.insts[ii];
                    let geom = inst.tris[ti].geom;
                    let value = inst.custom << 16 | geom;
                    let record = placed.hit_groups[ii] + geom * RAY_TYPES + ray_type;
                    [
                        value,
                        [SHADER_HIT_A, SHADER_HIT_B][ran as usize],
                        ran,
                        value * CALL_SCALE + CALL_ADD,
                        0,
                        if traced { geom } else { 0 },
                        if traced { record } else { 0 },
                        if traced { inst.custom } else { 0 },
                    ]
                }
                None => {
                    let value = MISS_VALUES[ray_type as usize];
                    [
                        value,
                        SHADER_MISS,
                        NONE,
                        value * CALL_SCALE + CALL_ADD,
                        0,
                        if traced { NONE } else { 0 },
                        if traced { ray_type } else { 0 },
                        if traced { value } else { 0 },
                    ]
                }
            };
            if s != want {
                bad += 1;
                if first.len() < 4 {
                    first.push(format!("({x},{y}) {s:x?} want {want:x?}"));
                }
            }
        }
    }
    assert!(
        bad == 0 && hits > 0,
        "{what}: {bad} rays saw the wrong records ({hits} hits): {}",
        first.join("; ")
    );
}

/// Traces `scene` from both raygens, with each ray type, opaque, forced non-opaque and alpha
/// tested, and from `traced` running the other ray type's record, and checks every ray against
/// the reference.
fn trace_and_check(
    b: &mut Builder,
    p: &RayPipeline,
    table: &HitTable,
    scene: Tlas,
    placed: &Placed<'_>,
    what: &str,
) {
    let modes = [0, MODE_FORCE_NON_OPAQUE, MODE_FORCE_NON_OPAQUE | MODE_ALPHA];
    let mut runs = Vec::new();
    for raygen in [RAYGEN_CLOSEST, RAYGEN_TRACED] {
        for ray_type in 0..RAY_TYPES {
            let redirect = (raygen == RAYGEN_TRACED).then_some(MODE_REDIRECT);
            for mode in modes.into_iter().chain(redirect) {
                let l = trace(b, p, raygen, table, |l| {
                    root(scene, l, mode, ray_type, CALL_SCALE_INDEX)
                });
                runs.push((raygen, ray_type, mode, l));
            }
        }
    }
    b.flush();
    for (raygen, ray_type, mode, l) in &runs {
        let what = format!("{what}, raygen {raygen}, ray type {ray_type}, mode {mode}");
        let r = check(
            l.out.as_slice().unwrap(),
            placed.insts,
            *mode & !MODE_REDIRECT,
        );
        r.assert_ok(&what);
        if *mode == MODE_FORCE_NON_OPAQUE {
            assert!(r.candidates > 0, "{what}: no any-hit ran");
        }
        check_seen(
            l.seen.as_slice().unwrap(),
            placed,
            *mode,
            *ray_type,
            *raygen == RAYGEN_TRACED,
            &what,
        );
    }
}

fn base_scene(b: &mut Builder, hit_groups: [u32; 2]) -> (Tlas, Vec<Cluster>) {
    let clusters = base_clusters();
    let clas = b.build_clas(&clusters, data_flags(), 0, None, None);
    let blas = b.build_blas(clas.addrs.gpu(), &[vec![0, 1, 2, 3], vec![4]], None);
    let tlas = b.build_tlas(
        blas.addrs.gpu(),
        &[
            instance([0.0, 0.0, FRONT_Z], FRONT_CUSTOM, 0).with_hit_group(hit_groups[0]),
            instance([0.0; 3], BACK_CUSTOM, 1).with_hit_group(hit_groups[1]),
        ],
    );
    (tlas, clusters)
}

fn front_back(clusters: &[Cluster]) -> [RefInst; 2] {
    let (back, front) = clusters.split_last().unwrap();
    let front: Vec<&Cluster> = front.iter().collect();
    [
        ref_inst(FRONT_CUSTOM, 0, [0.0, 0.0, FRONT_Z], &front),
        ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[back]),
    ]
}

fn geometries(inst: &RefInst) -> Vec<u32> {
    let mut g: Vec<u32> = inst.tris.iter().map(|t| t.geom).collect();
    g.sort_unstable();
    g.dedup();
    g
}

fn skipped() -> bool {
    !require(device().caps().ray_dispatch, "no ray tracing pipelines")
}

/// Closest-hit, any-hit (accepting, ignoring and counting candidates), miss and callable
/// shaders on the cluster base scene; records chosen by instance contribution, geometry index
/// and ray type, read by the shaders and through hit objects.
#[test]
fn stages_read_their_records() {
    if skipped() {
        return;
    }
    let p = create(0, CompileThreads::default(), RayPipelineFlags::empty());
    let mut b = Builder::new();
    let hit_groups = [3, 41];
    let (tlas, clusters) = base_scene(&mut b, hit_groups);
    let insts = front_back(&clusters);
    let t = triangle_table(
        &p,
        &[
            (FRONT_CUSTOM, hit_groups[0], geometries(&insts[0])),
            (BACK_CUSTOM, hit_groups[1], geometries(&insts[1])),
        ],
    );
    let placed = Placed {
        insts: &insts,
        hit_groups: &hit_groups,
    };
    trace_and_check(&mut b, &p, &t.table, tlas, &placed, "base clusters");
    assert_clean();
}

/// One quad per cluster over the unit square: 4225 clusters whose triangles have geometry
/// indices up to 8449, past bit 12, which NVIDIA driver 621.01 reads wrong in hit shaders.
fn many_clusters() -> Vec<Cluster> {
    const N: u32 = 65;
    let cell = 1.0 / N as f32;
    let mut clusters: Vec<Cluster> = (0..N * N)
        .map(|c| {
            let (i, j) = ((c % N) as f32, (c / N) as f32);
            let rect = [
                i * cell + 0.1 * cell,
                j * cell + 0.1 * cell,
                (i + 0.9) * cell,
                (j + 0.9) * cell,
            ];
            let mut cl = grid(
                10_000 + c,
                1,
                1,
                rect,
                move |x, y| 0.2 + 0.1 * x + 0.05 * y,
                1,
            );
            cl.base_geom = 2 * c;
            cl.geom_flags = GEOMETRY_OPAQUE | GEOMETRY_CULL_DISABLE;
            cl.tri_gflags = Some(vec![0, 1]);
            cl
        })
        .collect();
    clusters.push(back_cluster());
    clusters
}

/// Hit records chosen by cluster geometry indices past 4096, under a classic TLAS and a
/// partitioned TLAS, each instance with its own contribution.
#[test]
fn cluster_geometry_indices_select_records() {
    if skipped() {
        return;
    }
    let caps = device().caps().cluster;
    let clusters = many_clusters();
    let max_geom = 2 * 65 * 65 - 1;
    if !require(
        caps.max_geometry_index >= max_geom,
        "the device's cluster geometry indices stop short of the test's",
    ) {
        return;
    }
    let p = create(0, CompileThreads::default(), RayPipelineFlags::empty());
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters, data_flags(), 0, None, None);
    let front: Vec<u32> = (0..clusters.len() as u32 - 1).collect();
    let back = clusters.len() as u32 - 1;
    let blas = b.build_blas(clas.addrs.gpu(), &[front, vec![back]], None);
    let hit_groups = [7, 2 * max_geom + 100];
    let tlas = b.build_tlas(
        blas.addrs.gpu(),
        &[
            instance([0.0, 0.0, FRONT_Z], FRONT_CUSTOM, 0).with_hit_group(hit_groups[0]),
            instance([0.0; 3], BACK_CUSTOM, 1).with_hit_group(hit_groups[1]),
        ],
    );
    let limits = ptlas_limits(2);
    let (dst, scratch) = b.ptlas_blocks(limits);
    let mut writes = vec![
        ptlas_write(0, 0, FRONT_CUSTOM, [0.0, 0.0, FRONT_Z], 0, false),
        ptlas_write(1, 1, BACK_CUSTOM, [0.0; 3], 1, false),
    ];
    for (w, g) in writes.iter_mut().zip(hit_groups) {
        w.hit_group = g;
    }
    let ops = PtlasOps {
        writes,
        ..PtlasOps::default()
    };
    b.build_ptlas(limits, blas.addrs.gpu(), &ops, dst, scratch, false);
    let insts = front_back(&clusters);
    let t = triangle_table(
        &p,
        &[
            (FRONT_CUSTOM, hit_groups[0], geometries(&insts[0])),
            (BACK_CUSTOM, hit_groups[1], geometries(&insts[1])),
        ],
    );
    let placed = Placed {
        insts: &insts,
        hit_groups: &hit_groups,
    };
    trace_and_check(&mut b, &p, &t.table, tlas, &placed, "classic TLAS");
    trace_and_check(
        &mut b,
        &p,
        &t.table,
        Tlas(dst.0),
        &placed,
        "partitioned TLAS",
    );
    assert_clean();
}

/// Each thread's stores reach the shaders after it: the closest-hit or miss shader an invoke
/// runs after a reorder, the raygen after the invoke, a callable, and the raygen after the
/// call and after a reorder that follows it, through `load_coherent` and `store_coherent`.
/// An atomic after the reorders counts every thread. Plain accesses and accesses coherent at
/// shader call scope run too, and their stale reads are logged: on NVIDIA's driver both can
/// miss a callable's store after a reorder.
#[test]
fn stores_reach_the_shaders_that_follow() {
    if skipped() {
        return;
    }
    const VIS: usize = 4;
    const ALL: u32 = 31;
    const MODE_SHADER_CALL: u32 = 16;
    const MODE_READ_AFTER_CALL: u32 = 32;
    const MODE_DEVICE: u32 = 64;
    let p = create(0, CompileThreads::default(), RayPipelineFlags::empty());
    let mut b = Builder::new();
    let (tlas, _) = base_scene(&mut b, [0, 16]);
    let t = hit_table(&p, 32, |_| Some((GROUP_VIS, Vec::new())));
    let (w, h) = (1024u32, 1024u32);
    let mut failed = Vec::new();
    for access in [0, MODE_SHADER_CALL, MODE_DEVICE] {
        for read in [0, MODE_READ_AFTER_CALL] {
            let mode = access | read;
            let words = readback::<u32>(u64::from(w * h) * VIS as u64, "visibility");
            let mut count = readback::<u32>(1, "count");
            count.as_mut_slice().unwrap()[0] = 0;
            let r = Root {
                out: count.gpu(),
                seen: words.gpu(),
                scene: tlas,
                dumped: GpuPtr::NULL,
                width: w,
                height: h,
                mode,
                ray_type: MISS_VIS,
                ray_types: 1,
                callable: CALL_VIS,
                pad0: 0,
                pad1: 0,
            };
            let list = b.list();
            list.barrier(
                Stages::ALL | Stages::HOST,
                Stages::RAY,
                Hazards::ACCEL_STRUCT,
            );
            list.trace_rays(&p, RAYGEN_VISIBILITY, &t.table, &r, [w, h, 1]);
            list.barrier(Stages::RAY, Stages::HOST, Hazards::empty());
            b.flush();
            let words = words.as_slice().unwrap();
            let stale: Vec<(usize, u32)> = (0..(w * h) as usize)
                .map(|i| (i, words[i * VIS + 3]))
                .filter(|&(_, f)| f != ALL)
                .collect();
            let what = format!(
                "mode {mode}: {} of {} threads read a stale value (flags of the first: {:?})",
                stale.len(),
                w * h,
                &stale[..stale.len().min(8)]
            );
            log::info!("{what}");
            if access == MODE_DEVICE && !stale.is_empty() {
                failed.push(what);
            }
            assert_eq!(
                count.as_slice().unwrap()[0],
                w * h,
                "mode {mode}: atomics after the reorders"
            );
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
    assert_clean();
}

/// A procedural hit group over one box, through an indirect launch; a zero-width indirect
/// launch before it runs nothing.
#[test]
fn procedural_groups_and_indirect_launches() {
    if skipped() {
        return;
    }
    const BOX: [f32; 6] = [0.25, 0.25, 0.3, 0.75, 0.75, 0.6];
    const BOX_CUSTOM: u32 = 5;
    let d = device();
    let p = create(0, CompileThreads::default(), RayPipelineFlags::empty());
    let mut b = Builder::new();
    let boxes = b.upload(&[BOX]);
    let desc = ClassicDesc {
        kind: ClassicKind::Blas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Aabbs {
            boxes,
            count: 1,
            opaque: true,
        },
    };
    let sizes = d.classic_sizes(&desc);
    let block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "box blas")
        .unwrap();
    let scratch = b.block(MemoryKind::DeviceLocal, sizes.scratch);
    let blas = d
        .create_classic_as(ClassicKind::Blas, sizes.result_max, &block, 0)
        .unwrap();
    b.keep_any(block);
    b.list().build_classic(&ClassicBuild {
        desc: &desc,
        dst: &blas,
        scratch: scratch.address(),
    });
    b.list()
        .barrier(Stages::AS_BUILD, Stages::AS_BUILD, Hazards::ACCEL_STRUCT);
    let table = b.upload(&[blas.address().0]);
    let tlas = b.build_tlas(table, &[instance([0.0; 3], BOX_CUSTOM, 0)]);
    b.keep_any(blas);
    let entry: MissData = [BOX[2].to_bits(), 0];
    let t = hit_table(&p, 1, |_| {
        Some((GROUP_BOX, bytemuck::bytes_of(&entry).to_vec()))
    });
    let args = upload(
        &[
            DispatchArgs { x: 0, y: H, z: 1 },
            DispatchArgs { x: W, y: H, z: 1 },
        ],
        "trace args",
    );
    let mut l = launch_buffers();
    l.out.as_mut_slice().unwrap().fill(NONE);
    let r = root(tlas, &l, MODE_BOX, 0, CALL_SCALE_INDEX);
    let list = b.list();
    list.barrier(
        Stages::ALL | Stages::HOST,
        Stages::RAY,
        Hazards::ACCEL_STRUCT | Hazards::DRAW_ARGUMENTS,
    );
    list.trace_rays_indirect(&p, RAYGEN_CLOSEST, &t.table, &r, args.gpu());
    list.barrier(Stages::RAY, Stages::HOST, Hazards::empty());
    b.flush();
    let out = l.out.as_slice().unwrap();
    assert!(
        out.iter().all(|&v| v == NONE),
        "a zero-width launch wrote something"
    );
    let r = root(tlas, &l, MODE_BOX, 0, CALL_SCALE_INDEX);
    let list = b.list();
    list.barrier(
        Stages::ALL,
        Stages::RAY,
        Hazards::ACCEL_STRUCT | Hazards::DRAW_ARGUMENTS,
    );
    list.trace_rays_indirect(&p, RAYGEN_CLOSEST, &t.table, &r, args.gpu().offset(1));
    list.barrier(Stages::RAY, Stages::HOST, Hazards::empty());
    b.flush();
    let (out, seen) = (l.out.as_slice().unwrap(), l.seen.as_slice().unwrap());
    let mut hits = 0;
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            let (o, s) = (&out[i * REC..][..REC], &seen[i * SEEN..][..SEEN]);
            let (px, py) = ((x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32);
            let inside = px > BOX[0] && px < BOX[3] && py > BOX[1] && py < BOX[4];
            let edge = [px - BOX[0], BOX[3] - px, py - BOX[1], BOX[4] - py]
                .iter()
                .any(|e| e.abs() < 1e-4);
            if edge {
                continue;
            }
            let want_seen = if inside {
                hits += 1;
                [BOX[2].to_bits(), SHADER_BOX]
            } else {
                [MISS_VALUES[0], SHADER_MISS]
            };
            assert_eq!([s[0], s[1]], want_seen, "({x},{y}) seen");
            assert_eq!(o[0], u32::from(inside), "({x},{y}) hit");
            if inside {
                assert_eq!([o[1], o[2], o[4], o[5]], [BOX_CUSTOM, 0, 0, 0], "({x},{y})");
                assert_eq!(f32::from_bits(o[6]), 1.0 + BOX[2], "({x},{y}) t");
            }
        }
    }
    assert!(hits > 0);
    assert_clean();
}

/// The closest-hit's body compiled as a compute stand-in, over the inputs the closest-hit
/// dumped, writes what the closest-hit wrote; the stand-in reports its code statistics.
#[test]
fn a_stand_in_computes_what_its_shader_does() {
    if skipped() {
        return;
    }
    let p = create(0, CompileThreads::default(), RayPipelineFlags::empty());
    let mut b = Builder::new();
    let hit_groups = [0, 20];
    let (tlas, clusters) = base_scene(&mut b, hit_groups);
    let insts = front_back(&clusters);
    let t = triangle_table(
        &p,
        &[
            (FRONT_CUSTOM, hit_groups[0], geometries(&insts[0])),
            (BACK_CUSTOM, hit_groups[1], geometries(&insts[1])),
        ],
    );
    let mut dumped = device()
        .gpu_malloc::<Dumped>(MemoryKind::Readback, u64::from(W * H), "dumped")
        .unwrap();
    dumped.as_mut_slice().unwrap().fill(Dumped::zeroed());
    let shader = trace(&mut b, &p, RAYGEN_CLOSEST, &t.table, |l| Root {
        dumped: dumped.gpu(),
        ..root(tlas, l, 0, 0, CALL_SCALE_INDEX)
    });
    let standin = launch_buffers();
    let compute = pipeline(&ray_pipeline::MODULE, "standin");
    let r = Root {
        dumped: dumped.gpu(),
        ..root(tlas, &standin, 0, 0, CALL_SCALE_INDEX)
    };
    let list = b.list();
    list.barrier(Stages::RAY, Stages::COMPUTE, Hazards::empty());
    list.dispatch_inline(&compute, &r, [(W * H).div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    b.flush();
    let mut compared = 0;
    for (i, d) in dumped.as_slice().unwrap().iter().enumerate() {
        if d.shader == 0 {
            continue;
        }
        compared += 1;
        let words = |a: &Allocation<u32>, n: usize, at: &[usize]| -> Vec<u32> {
            let s = &a.as_slice().unwrap()[i * n..][..n];
            at.iter().map(|&k| s[k]).collect()
        };
        let rec: Vec<usize> = (0..7).chain(10..19).collect();
        assert_eq!(
            words(&standin.out, REC, &rec),
            words(&shader.out, REC, &rec),
            "ray {i}: ray record"
        );
        assert_eq!(
            words(&standin.seen, SEEN, &[0, 1, 2]),
            words(&shader.seen, SEEN, &[0, 1, 2]),
            "ray {i}: payload"
        );
    }
    assert!(compared > 0);
    if let Some(stats) = compute.statistics() {
        log::info!("stand-in of the closest-hit body: {stats:?}");
        assert!(stats.registers > 0, "{stats:?}");
    }
    assert_clean();
}

/// Pipelines created on one thread, on spawned threads and on a caller's pool, each compiling
/// its closest-hit shaders anew, trace alike.
#[test]
fn deferred_creation_on_any_threads_traces_alike() {
    if skipped() {
        return;
    }
    let salt = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos()
        | 1;
    let spawn = |task: Box<dyn FnOnce() + Send>| {
        std::thread::spawn(task);
    };
    let pipelines = [
        create(salt, CompileThreads::Spawned(1), RayPipelineFlags::empty()),
        create(
            salt + 1,
            CompileThreads::Spawned(8),
            RayPipelineFlags::empty(),
        ),
        create(
            salt + 2,
            CompileThreads::Pool {
                spawn: &spawn,
                max: 8,
            },
            RayPipelineFlags::empty(),
        ),
    ];
    let mut b = Builder::new();
    let hit_groups = [0, 20];
    let (tlas, clusters) = base_scene(&mut b, hit_groups);
    let insts = front_back(&clusters);
    let placed = Placed {
        insts: &insts,
        hit_groups: &hit_groups,
    };
    for (n, p) in pipelines.iter().enumerate() {
        let t = triangle_table(
            p,
            &[
                (FRONT_CUSTOM, hit_groups[0], geometries(&insts[0])),
                (BACK_CUSTOM, hit_groups[1], geometries(&insts[1])),
            ],
        );
        let l = trace(&mut b, p, RAYGEN_TRACED, &t.table, |l| {
            root(tlas, l, MODE_FORCE_NON_OPAQUE, 1, CALL_SCALE_INDEX)
        });
        b.flush();
        let what = format!("pipeline {n}");
        check(l.out.as_slice().unwrap(), &insts, MODE_FORCE_NON_OPAQUE).assert_ok(&what);
        check_seen(
            l.seen.as_slice().unwrap(),
            &placed,
            MODE_FORCE_NON_OPAQUE,
            1,
            true,
            &what,
        );
    }
    assert_clean();
}
