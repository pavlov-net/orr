//! Cluster scenes, their CPU reference and builders over `rtas_execute`, ported from the
//! cluster ray-query probe. Every ray traces +z over the unit square; the reference checks
//! hit, ids, cluster id, primitive, geometry index, t, positions and the nearest candidate.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use orr_gpu::rtas::consts::{
    GEOMETRY_CULL_DISABLE, GEOMETRY_NO_DUPLICATE_ANY_HIT, GEOMETRY_OPAQUE,
};
use orr_gpu::rtas::{
    ClasBuild, ClusterBlas, Instantiate, Move, PtlasCommand, PtlasTranslate, PtlasUpdate,
    PtlasWrite, TemplateBuild,
};
use orr_gpu::{
    Allocation, BlasLimits, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance,
    ClassicKind, ClusterLimits, ClusterVertexFormat, CommandList, ComputePipeline, GpuAddress,
    GpuPtr, Hazards, InstanceFlags, MemoryKind, MoveKind, MoveLimits, PtlasLimits,
    RayDispatchPipeline, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, RtasSizes,
    Stages, TimelinePoint, Tlas,
};
use orr_shaders::bytemuck::{Pod, Zeroable};

use crate::shaders::{cluster_args, cluster_trace};
use crate::{device, pipeline, readback, render, upload};

pub const W: u32 = 512;
pub const H: u32 = 512;
pub const REC: usize = 20;
const EDGE_EPS: f32 = 2e-4;
const T_EPS: f32 = 1e-4;

pub const MODE_FORCE_NON_OPAQUE: u32 = 1;
pub const MODE_ALPHA: u32 = 2;
pub const MODE_OMM_2_STATE: u32 = 4;

pub const OP_CLAS: u32 = 0;
pub const OP_TEMPLATE: u32 = 1;
pub const OP_INSTANTIATE: u32 = 2;
pub const OP_GATHER: u32 = 3;
pub const OP_BLAS: u32 = 4;
pub const OP_CLASSIC: u32 = 5;
pub const OP_WRITE: u32 = 6;
pub const OP_UPDATE: u32 = 7;
pub const OP_TRANSLATE: u32 = 8;
pub const OP_COMMAND: u32 = 9;
pub const OP_MOVE: u32 = 10;
pub const OP_POOL: u32 = 11;
/// A `blas` table index meaning "no BLAS".
pub const NO_BLAS: u64 = 0xffff_ffff;

pub const FRONT_CUSTOM: u32 = 41;
pub const BACK_CUSTOM: u32 = 77;
pub const FRONT_Z: f32 = 0.05;

/// The 4-state values of the four micromap entries the OMM case builds (level 0).
pub const OMM_STATES: [u8; 4] = [0, 1, 2, 3];

pub fn translate(t: [f32; 3]) -> [[f32; 4]; 3] {
    [
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
    ]
}

/// One cluster as the CPU describes it.
#[derive(Clone)]
pub struct Cluster {
    pub id: u32,
    pub verts: Vec<[f32; 3]>,
    pub tris: Vec<[u32; 3]>,
    /// 1, 2 or 4.
    pub index_bytes: u32,
    pub truncate: u32,
    pub base_geom: u32,
    pub geom_flags: u32,
    /// Per-triangle geometry index (bits 0-23) and flags (bits 29-31).
    pub tri_gflags: Option<Vec<u32>>,
    /// Per-triangle micromap index or special index.
    pub omm_index: Option<Vec<i32>>,
}

/// Opacity of a triangle as traversal sees it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Opaque,
    NonOpaque,
    Transparent,
}

#[derive(Clone)]
pub struct RefTri {
    v: [[f32; 3]; 3],
    pub cluster: u32,
    pub prim: u32,
    pub geom: u32,
    kind: Kind,
    kind_2state: Kind,
    no_dup: bool,
}

/// An instance as the reference sees it.
#[derive(Clone)]
pub struct RefInst {
    pub custom: u32,
    pub index: u32,
    pub offset: [f32; 3],
    pub tris: Vec<RefTri>,
}

fn alpha_keep(cluster: u32, prim: u32) -> bool {
    !cluster.wrapping_add(prim).is_multiple_of(3)
}

fn truncate(v: f32, bits: u32) -> f32 {
    if bits == 0 {
        v
    } else {
        f32::from_bits(v.to_bits() & !((1u32 << bits) - 1))
    }
}

/// A grid cluster over `rect` with `nx` x `ny` quads and alternating diagonals.
pub fn grid(
    id: u32,
    nx: u32,
    ny: u32,
    rect: [f32; 4],
    z: impl Fn(f32, f32) -> f32,
    index_bytes: u32,
) -> Cluster {
    let mut verts = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            let x = rect[0] + (rect[2] - rect[0]) * i as f32 / nx as f32;
            let y = rect[1] + (rect[3] - rect[1]) * j as f32 / ny as f32;
            verts.push([x, y, z(x, y)]);
        }
    }
    let mut tris = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let a = j * (nx + 1) + i;
            let (b, c) = (a + 1, a + nx + 1);
            let d = c + 1;
            if (i + j) % 2 == 0 {
                tris.push([a, b, d]);
                tris.push([a, d, c]);
            } else {
                tris.push([a, b, c]);
                tris.push([b, d, c]);
            }
        }
    }
    Cluster {
        id,
        verts,
        tris,
        index_bytes,
        truncate: 0,
        base_geom: 0,
        geom_flags: GEOMETRY_OPAQUE | GEOMETRY_CULL_DISABLE,
        tri_gflags: None,
        omm_index: None,
    }
}

/// The largest `nx` up to the given one whose grid fits the device's cluster limits.
pub fn clamp_grid(mut nx: u32, ny: u32) -> (u32, u32) {
    let c = device().caps().cluster;
    while nx > 1 && (2 * nx * ny > c.max_tris || (nx + 1) * (ny + 1) > c.max_verts) {
        nx -= 1;
    }
    (nx, ny)
}

fn quadrant(q: u32) -> [f32; 4] {
    let (qx, qy) = ((q % 2) as f32 * 0.5, (q / 2) as f32 * 0.5);
    [qx + 0.03, qy + 0.03, qx + 0.47, qy + 0.47]
}

/// Four opaque grid clusters, one per quadrant (8, 24, 72 and up to 256 triangles; 8-, 16-,
/// 32- and 8-bit indices; cluster 2 truncates 8 mantissa bits), then one back quad.
pub fn base_clusters() -> Vec<Cluster> {
    let grids = [(2, 2, 1), (4, 3, 2), (6, 6, 4), (16, 8, 1)];
    let mut clusters: Vec<Cluster> = grids
        .iter()
        .enumerate()
        .map(|(c, &(nx, ny, ib))| {
            let (nx, ny) = clamp_grid(nx, ny);
            let cf = c as f32;
            let mut cl = grid(
                1000 + 17 * c as u32,
                nx,
                ny,
                quadrant(c as u32),
                move |x, y| {
                    0.2 + 0.1 * x + 0.05 * y + 0.01 * cf + 0.02 * (9.0 * x).sin() * (7.0 * y).cos()
                },
                ib,
            );
            cl.base_geom = c as u32;
            cl
        })
        .collect();
    clusters[2].truncate = 8;
    clusters.push(back_cluster());
    clusters
}

/// The full-square back cluster at z 0.8, geometry index 9.
pub fn back_cluster() -> Cluster {
    let mut back = grid(5000, 1, 1, [0.0, 0.0, 1.0, 1.0], |_, _| 0.8, 1);
    back.base_geom = 9;
    back
}

fn omm_kind(index: i32, geometry: Kind) -> (Kind, Kind) {
    let state = |s: u8| match s {
        0 => (Kind::Transparent, Kind::Transparent),
        1 => (Kind::Opaque, Kind::Opaque),
        2 => (Kind::NonOpaque, Kind::Transparent),
        _ => (Kind::NonOpaque, Kind::Opaque),
    };
    match index {
        -1 => state(0),
        -2 => state(1),
        -3 => state(2),
        -4 => state(3),
        -5 => (geometry, geometry),
        i => state(OMM_STATES[i as usize]),
    }
}

/// Reference triangles of `c`, with replacement vertices for instantiated templates.
pub fn ref_tris(
    c: &Cluster,
    verts: Option<&[[f32; 3]]>,
    id_offset: u32,
    geom_offset: u32,
) -> Vec<RefTri> {
    let verts = verts.unwrap_or(&c.verts);
    let tv = |i: u32| verts[i as usize].map(|x| truncate(x, c.truncate));
    c.tris
        .iter()
        .enumerate()
        .map(|(t, tri)| {
            let word = c.tri_gflags.as_ref().map_or(0, |g| g[t]);
            let flags = c.geom_flags | (word >> 29);
            let geometry = if flags & GEOMETRY_OPAQUE != 0 {
                Kind::Opaque
            } else {
                Kind::NonOpaque
            };
            let (kind, kind_2state) = match &c.omm_index {
                Some(omm) => omm_kind(omm[t], geometry),
                None => (geometry, geometry),
            };
            RefTri {
                v: [tv(tri[0]), tv(tri[1]), tv(tri[2])],
                cluster: c.id + id_offset,
                prim: t as u32,
                geom: c.base_geom + (word & 0xff_ffff) + geom_offset,
                kind,
                kind_2state,
                no_dup: flags & GEOMETRY_NO_DUPLICATE_ANY_HIT != 0,
            }
        })
        .collect()
}

pub fn ref_inst(custom: u32, index: u32, offset: [f32; 3], clusters: &[&Cluster]) -> RefInst {
    RefInst {
        custom,
        index,
        offset,
        tris: clusters
            .iter()
            .flat_map(|c| ref_tris(c, None, 0, 0))
            .collect(),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Cand {
    /// No non-opaque triangle on the ray.
    Zero,
    /// The nearest candidate is (cluster, prim), with an exact count when known.
    Nearest(u32, u32, Option<u32>),
    /// Candidates may or may not arrive.
    Any,
}

struct Expect {
    /// (instance, triangle, t)
    hit: Option<(usize, usize, f32)>,
    cand: Cand,
}

fn cross(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn seg_dist(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
    let t = (((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / (ex * ex + ey * ey)).clamp(0.0, 1.0);
    let (dx, dy) = (p[0] - a[0] - t * ex, p[1] - a[1] - t * ey);
    (dx * dx + dy * dy).sqrt()
}

/// The reference for ray (x, y); `None` when it passes within EDGE_EPS of an edge.
fn expect(insts: &[RefInst], x: u32, y: u32, mode: u32) -> Option<Expect> {
    let p = [(x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32];
    let mut hits = Vec::new();
    for (ii, inst) in insts.iter().enumerate() {
        let q = [p[0] - inst.offset[0], p[1] - inst.offset[1]];
        for (ti, tri) in inst.tris.iter().enumerate() {
            let v = tri.v.map(|v| [v[0], v[1]]);
            let minx = v[0][0].min(v[1][0]).min(v[2][0]);
            let maxx = v[0][0].max(v[1][0]).max(v[2][0]);
            let miny = v[0][1].min(v[1][1]).min(v[2][1]);
            let maxy = v[0][1].max(v[1][1]).max(v[2][1]);
            if q[0] < minx - EDGE_EPS
                || q[0] > maxx + EDGE_EPS
                || q[1] < miny - EDGE_EPS
                || q[1] > maxy + EDGE_EPS
            {
                continue;
            }
            if seg_dist(v[0], v[1], q) < EDGE_EPS
                || seg_dist(v[1], v[2], q) < EDGE_EPS
                || seg_dist(v[2], v[0], q) < EDGE_EPS
            {
                return None;
            }
            let area = cross(v[0], v[1], v[2]);
            let b = [
                cross(v[1], v[2], q) / area,
                cross(v[2], v[0], q) / area,
                cross(v[0], v[1], q) / area,
            ];
            if b.iter().all(|&w| w > 0.0) {
                let t = 1.0
                    + inst.offset[2]
                    + b[0] * tri.v[0][2]
                    + b[1] * tri.v[1][2]
                    + b[2] * tri.v[2][2];
                let mut kind = if mode & MODE_OMM_2_STATE != 0 {
                    tri.kind_2state
                } else {
                    tri.kind
                };
                if mode & MODE_FORCE_NON_OPAQUE != 0 && kind == Kind::Opaque {
                    kind = Kind::NonOpaque;
                }
                if kind != Kind::Transparent {
                    hits.push((t, ii, ti, kind));
                }
            }
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    let tri = |h: &(f32, usize, usize, Kind)| &insts[h.1].tris[h.2];
    let prim = |h: &(f32, usize, usize, Kind)| tri(h).prim;
    let committed = hits
        .iter()
        .find(|h| {
            h.3 == Kind::Opaque || mode & MODE_ALPHA == 0 || alpha_keep(tri(h).cluster, prim(h))
        })
        .map(|h| (h.1, h.2, h.0));
    let opaque_t = hits
        .iter()
        .find(|h| h.3 == Kind::Opaque)
        .map_or(f32::INFINITY, |h| h.0);
    let before: Vec<_> = hits
        .iter()
        .filter(|h| h.3 == Kind::NonOpaque && h.0 < opaque_t)
        .collect();
    let any_non_opaque = hits.iter().any(|h| h.3 == Kind::NonOpaque);
    let cand = match before.first() {
        Some(h) => {
            let all = hits.iter().filter(|h| h.3 == Kind::NonOpaque).count();
            let exact = (before.len() == all && before.iter().all(|h| tri(h).no_dup))
                .then_some(before.len() as u32);
            Cand::Nearest(tri(h).cluster, prim(h), exact)
        }
        None if !any_non_opaque => Cand::Zero,
        None => Cand::Any,
    };
    Some(Expect {
        hit: committed,
        cand,
    })
}

/// The opaque hit (instance, triangle, t) the reference expects for ray (x, y), or `None`
/// when the ray passes within the edge tolerance.
pub fn reference_hit(insts: &[RefInst], x: u32, y: u32) -> Option<Option<(usize, usize, f32)>> {
    expect(insts, x, y, 0).map(|e| e.hit)
}

/// Mismatches of one checked trace.
#[derive(Default)]
pub struct Report {
    pub checked: u32,
    pub hits: u32,
    pub candidates: u32,
    pub clusters: BTreeSet<u32>,
    pub bad: BTreeMap<&'static str, u32>,
    first_bad: Vec<String>,
}

impl Report {
    /// Fails the test with the mismatches, or when nothing was hit.
    pub fn assert_ok(&self, what: &str) {
        assert!(
            self.bad.is_empty() && self.hits > 0,
            "{what}: {} rays checked, {} hits, mismatches {:?}: {}",
            self.checked,
            self.hits,
            self.bad,
            self.first_bad.join("; ")
        );
    }
}

/// Compares every ray record of `out` with the reference.
pub fn check(out: &[u32], insts: &[RefInst], mode: u32) -> Report {
    let mut r = Report::default();
    for y in 0..H {
        for x in 0..W {
            let o = &out[(y * W + x) as usize * REC..][..REC];
            let Some(e) = expect(insts, x, y, mode) else {
                continue;
            };
            r.checked += 1;
            let mut bad = |what: &'static str, detail: String| {
                *r.bad.entry(what).or_insert(0) += 1;
                if r.first_bad.len() < 4 {
                    r.first_bad.push(format!("({x},{y}) {what}: {detail}"));
                }
            };
            if o[19] != 0x600d {
                bad("unwritten", format!("{:#x}", o[19]));
                continue;
            }
            if o[7] > 0 {
                r.candidates += 1;
            }
            match e.cand {
                Cand::Zero if o[7] != 0 => bad("candidate_count", format!("{} want 0", o[7])),
                Cand::Nearest(c, p, exact) => {
                    if o[7] == 0 {
                        bad("candidate_missing", format!("want cluster {c} prim {p}"));
                    } else if o[8] != c || o[9] != p {
                        bad(
                            "candidate_id",
                            format!("cluster {} prim {} want {c} {p}", o[8], o[9]),
                        );
                    } else if let Some(n) = exact
                        && o[7] != n
                    {
                        bad("candidate_dup", format!("{} want {n}", o[7]));
                    }
                }
                _ => {}
            }
            let Some((ii, ti, t)) = e.hit else {
                if o[0] != 0 {
                    bad("hit_on_miss", format!("instance {} cluster {}", o[1], o[3]));
                }
                continue;
            };
            if o[0] != 1 {
                bad("miss_on_hit", format!("want instance {}", insts[ii].custom));
                continue;
            }
            r.hits += 1;
            let inst = &insts[ii];
            let tri = &inst.tris[ti];
            let prim = tri.prim;
            r.clusters.insert(o[3]);
            if o[1] != inst.custom || o[2] != inst.index {
                bad(
                    "instance",
                    format!("{} {} want {} {}", o[1], o[2], inst.custom, inst.index),
                );
            }
            if o[3] != tri.cluster {
                bad("cluster_id", format!("{} want {}", o[3], tri.cluster));
            }
            if o[4] != prim {
                bad("primitive", format!("{} want {prim}", o[4]));
            }
            if o[5] != tri.geom {
                bad("geometry_index", format!("{} want {}", o[5], tri.geom));
            }
            let got_t = f32::from_bits(o[6]);
            if (got_t - t).abs() > T_EPS {
                bad("t", format!("{got_t} want {t}"));
            }
            let got: Vec<f32> = o[10..19].iter().map(|&b| f32::from_bits(b)).collect();
            let want: Vec<f32> = tri.v.iter().flatten().copied().collect();
            if got != want {
                bad("position_fetch", format!("{got:?} want {want:?}"));
            }
        }
    }
    r
}

/// The result of one cluster op: an address and a size per record, read back.
pub struct Out {
    pub addrs: Allocation<u64>,
    pub sizes: Allocation<u32>,
    pub reserved: RtasSizes,
}

/// Records cluster scenes into one render-queue list at a time, keeping every block alive.
pub struct Builder {
    pub list: Option<CommandList<'static>>,
    keep: Vec<Box<dyn Any>>,
    args: ComputePipeline,
    trace: ComputePipeline,
    /// Points the next submit waits for.
    pub waits: Vec<TimelinePoint>,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Self {
            list: Some(render().command_list().unwrap()),
            keep: Vec::new(),
            args: pipeline(&cluster_args::MODULE, "main"),
            trace: pipeline(&cluster_trace::MODULE, "main"),
            waits: Vec::new(),
        }
    }

    pub fn list(&mut self) -> &mut CommandList<'static> {
        self.list.as_mut().unwrap()
    }

    /// Keeps `a` alive with the builder; returns its address.
    pub fn keep<T: 'static>(&mut self, a: Allocation<T>) -> GpuPtr<T> {
        let gpu = a.gpu;
        self.keep.push(Box::new(a));
        gpu
    }

    pub fn keep_any(&mut self, a: impl Any) {
        self.keep.push(Box::new(a));
    }

    pub fn upload<T: Pod>(&mut self, values: &[T]) -> GpuPtr<T> {
        let a = upload(values, "cluster input");
        self.keep(a)
    }

    pub fn block(&mut self, kind: MemoryKind, bytes: u64) -> GpuPtr<u8> {
        let a = device()
            .gpu_malloc::<u8>(kind, bytes.max(4), "cluster block")
            .unwrap();
        self.keep(a)
    }

    /// Submits the list, waits for it and starts the next one.
    pub fn flush(&mut self) -> TimelinePoint {
        let point = self.submit();
        render().wait(point, Duration::from_secs(10)).unwrap();
        point
    }

    /// Submits the list without waiting and starts the next one.
    pub fn submit(&mut self) -> TimelinePoint {
        let list = self.list.take().unwrap();
        let queue = render();
        let point = queue
            .submit(vec![list], &std::mem::take(&mut self.waits))
            .unwrap();
        self.list = Some(queue.command_list().unwrap());
        point
    }

    /// Runs the args kernel over `n` records.
    pub fn args(&mut self, root: cluster_args::Root) {
        let n = root.n;
        let root = self.upload(&[root]);
        let list = self.list.as_mut().unwrap();
        list.dispatch(&self.args, root, [n.div_ceil(64).max(1), 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::AS_BUILD,
            Hazards::empty(),
        );
    }

    /// One cluster op over `n` records the args kernel writes from `root` (`dst` and
    /// `count` filled in here), into `implicit` or a fresh block.
    pub fn cluster_op(
        &mut self,
        desc: &RtasOpDesc,
        mut root: cluster_args::Root,
        n: u32,
        implicit: Option<GpuAddress>,
        addrs: Option<GpuPtr<u64>>,
    ) -> Out {
        let d = device();
        let caps = d.caps().cluster;
        let stride = u64::from(d.rtas_args_stride(desc.op));
        let records = self.block(MemoryKind::DeviceLocal, stride * u64::from(n));
        let count = self.block(MemoryKind::DeviceLocal, 4).cast::<u32>();
        root.dst = records.address().0;
        root.count = count;
        root.n = n;
        self.args(root);
        let reserved = d.rtas_sizes(desc);
        let implicit = implicit.unwrap_or_else(|| {
            self.block(MemoryKind::DeviceLocal, reserved.result_max)
                .address()
        });
        let scratch = self.block(MemoryKind::DeviceLocal, reserved.scratch);
        assert!(
            scratch
                .address()
                .0
                .is_multiple_of(u64::from(caps.align_scratch))
        );
        let out_addrs = d
            .gpu_malloc::<u64>(MemoryKind::Readback, u64::from(n), "addresses")
            .unwrap();
        let sizes = d
            .gpu_malloc::<u32>(MemoryKind::Readback, u64::from(n), "sizes")
            .unwrap();
        let dst_addresses = addrs.unwrap_or(out_addrs.gpu);
        self.list().rtas_execute(
            desc,
            &RtasIo {
                args: records.address(),
                count,
                dst_implicit: implicit,
                dst_addresses,
                dst_sizes: sizes.gpu,
                scratch: scratch.address(),
                src: GpuAddress(0),
            },
        );
        Out {
            addrs: out_addrs,
            sizes,
            reserved,
        }
    }

    /// Uploads the clusters' geometry and returns their `ClasBuild` records.
    pub fn clas_records(&mut self, clusters: &[Cluster], omm_array: u64) -> Vec<ClasBuild> {
        clusters
            .iter()
            .map(|c| {
                let verts = self.upload(&c.verts);
                let index_bytes: Vec<u8> = c
                    .tris
                    .iter()
                    .flatten()
                    .flat_map(|&i| match c.index_bytes {
                        1 => vec![u8::try_from(i).unwrap()],
                        2 => u16::try_from(i).unwrap().to_le_bytes().to_vec(),
                        _ => i.to_le_bytes().to_vec(),
                    })
                    .collect();
                let indices = self.upload(&index_bytes);
                let gflags = c.tri_gflags.as_ref().map(|g| self.upload(g));
                let omm = c.omm_index.as_ref().map(|g| self.upload(g));
                ClasBuild {
                    cluster_id: c.id,
                    flags: 0,
                    tri_count: c.tris.len() as u32,
                    vertex_count: c.verts.len() as u32,
                    truncate_bits: c.truncate,
                    index_format: c.index_bytes,
                    base_geometry_and_flags: orr_gpu::rtas::geometry_index_and_flags(
                        c.base_geom,
                        c.geom_flags,
                    ),
                    omm_index_format: 4,
                    vertices: verts.address().0,
                    indices: indices.address().0,
                    triangle_flags: gflags.map_or(0, |g| g.address().0),
                    omm_array: if omm.is_some() { omm_array } else { 0 },
                    omm_indices: omm.map_or(0, |g| g.address().0),
                }
            })
            .collect()
    }

    /// CLAS from `clusters`.
    pub fn build_clas(
        &mut self,
        clusters: &[Cluster],
        flags: RtasFlags,
        omm_array: u64,
        implicit: Option<GpuAddress>,
        addrs: Option<GpuPtr<u64>>,
    ) -> Out {
        let records = self.clas_records(clusters, omm_array);
        let mut root = zeroed_root();
        root.clas = self.upload(&records);
        let desc = clas_desc(RtasOp::BuildClas, RtasMode::Implicit, flags, clusters);
        self.cluster_op(&desc, root, clusters.len() as u32, implicit, addrs)
    }

    /// Cluster BLAS, one per group of indices into the CLAS address `table`.
    pub fn build_blas(
        &mut self,
        table: GpuPtr<u64>,
        groups: &[Vec<u32>],
        implicit: Option<GpuAddress>,
    ) -> Out {
        let flat: Vec<u32> = groups.iter().flatten().copied().collect();
        let mut first = 0u64;
        let blas: Vec<ClusterBlas> = groups
            .iter()
            .map(|g| {
                let b = ClusterBlas {
                    clas_addresses: first,
                    clas_count: g.len() as u32,
                    pad: 0,
                };
                first += g.len() as u64;
                b
            })
            .collect();
        let refs = self
            .block(MemoryKind::DeviceLocal, 8 * flat.len() as u64)
            .cast::<u64>();
        let mut root = zeroed_root();
        root.table = table;
        root.gather = self.upload(&flat);
        root.refs = refs;
        root.n = flat.len() as u32;
        root.op = OP_GATHER;
        self.args(root);
        let mut root = zeroed_root();
        root.refs = refs;
        root.blas = self.upload(&blas);
        root.op = OP_BLAS;
        let desc = RtasOpDesc {
            op: RtasOp::BuildClusterBlas,
            mode: RtasMode::Implicit,
            flags: RtasFlags::FAST_TRACE,
            vertex_format: ClusterVertexFormat::F32x3,
            limits: RtasLimits::Blas(BlasLimits {
                max_args: groups.len() as u32,
                max_clas_per_blas: groups.iter().map(|g| g.len() as u32).max().unwrap(),
                max_total_clas: flat.len() as u32,
            }),
        };
        self.cluster_op(&desc, root, groups.len() as u32, implicit, None)
    }

    /// A classic TLAS over `instances`, whose `blas` fields are indices into `table`.
    pub fn build_tlas(&mut self, table: GpuPtr<u64>, instances: &[ClassicInstance]) -> Tlas {
        let d = device();
        let n = instances.len() as u32;
        let records = self.block(MemoryKind::DeviceLocal, 64 * u64::from(n));
        let mut root = zeroed_root();
        root.dst = records.address().0;
        root.table = table;
        root.classic = self.upload(instances);
        root.n = n;
        root.op = OP_CLASSIC;
        self.args(root);
        let desc = ClassicDesc {
            kind: ClassicKind::Tlas,
            flags: RtasFlags::FAST_TRACE,
            geometry: ClassicGeometry::Instances {
                records: records.address(),
                capacity: n,
            },
        };
        let sizes = d.classic_sizes(&desc);
        let block = d
            .gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "tlas")
            .unwrap();
        let scratch = self.block(MemoryKind::DeviceLocal, sizes.scratch);
        let tlas = d
            .create_classic_as(ClassicKind::Tlas, sizes.result_max, &block, 0)
            .unwrap();
        self.list().build_classic(&ClassicBuild {
            desc: &desc,
            dst: &tlas,
            scratch: scratch.address(),
        });
        let t = tlas.tlas();
        self.keep_any(tlas);
        self.keep(block);
        t
    }

    /// Traces every ray of the unit square into a fresh readback buffer.
    pub fn trace(&mut self, tlas: Tlas, mode: u32) -> Allocation<u32> {
        let out = readback::<u32>(u64::from(W * H) * REC as u64, "rays");
        self.trace_into(tlas, mode, out.gpu);
        out
    }

    /// Traces `tlas` from the raygen entry of `p`, the ray-dispatch pipeline of
    /// `cluster_trace`.
    pub fn trace_rays(
        &mut self,
        p: &RayDispatchPipeline,
        tlas: Tlas,
        mode: u32,
    ) -> Allocation<u32> {
        let out = readback::<u32>(u64::from(W * H) * REC as u64, "rays");
        let root = self.trace_root(tlas, mode, out.gpu);
        let list = self.list.as_mut().unwrap();
        list.barrier(Stages::ALL, Stages::RAY, Hazards::ACCEL_STRUCT);
        list.dispatch_rays(p, root, [W, H, 1]);
        list.barrier(Stages::RAY, Stages::HOST, Hazards::empty());
        out
    }

    fn trace_root(
        &mut self,
        tlas: Tlas,
        mode: u32,
        out: GpuPtr<u32>,
    ) -> GpuPtr<cluster_trace::Root> {
        self.upload(&[cluster_trace::Root {
            out,
            scene: tlas,
            width: W,
            height: H,
            mode,
            pad: 0,
        }])
    }

    pub fn trace_into(&mut self, tlas: Tlas, mode: u32, out: GpuPtr<u32>) {
        let root = self.trace_root(tlas, mode, out);
        let list = self.list.as_mut().unwrap();
        list.barrier(Stages::ALL, Stages::COMPUTE, Hazards::ACCEL_STRUCT);
        list.dispatch(&self.trace, root, [W / 8, H / 8, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::HOST | Stages::AS_BUILD,
            Hazards::empty(),
        );
    }
}

/// An args root with every pointer null.
pub fn zeroed_root() -> cluster_args::Root {
    cluster_args::Root::zeroed()
}

/// The desc of a cluster, template or instantiation batch over `clusters`.
pub fn clas_desc(op: RtasOp, mode: RtasMode, flags: RtasFlags, clusters: &[Cluster]) -> RtasOpDesc {
    let geometries = clusters
        .iter()
        .map(|c| {
            c.tri_gflags.as_ref().map_or(1, |g| {
                g.iter()
                    .map(|w| w & 0xff_ffff)
                    .collect::<BTreeSet<_>>()
                    .len() as u32
            })
        })
        .max()
        .unwrap_or(1);
    RtasOpDesc {
        op,
        mode,
        flags,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Cluster(ClusterLimits {
            max_args: clusters.len() as u32,
            max_tris_per_cluster: clusters.iter().map(|c| c.tris.len() as u32).max().unwrap(),
            max_verts_per_cluster: clusters.iter().map(|c| c.verts.len() as u32).max().unwrap(),
            max_total_tris: clusters.iter().map(|c| c.tris.len() as u32).sum(),
            max_total_verts: clusters.iter().map(|c| c.verts.len() as u32).sum(),
            max_geometry_index: 64,
            max_geometries_per_cluster: geometries,
            min_truncate_bits: clusters.iter().map(|c| c.truncate).min().unwrap(),
        }),
    }
}

/// A classic instance with a translation, cull disabled, over BLAS `table` index `blas`.
pub fn instance(t: [f32; 3], custom: u32, blas: u64) -> ClassicInstance {
    ClassicInstance::new(
        translate(t),
        custom,
        0xff,
        InstanceFlags::CULL_DISABLE,
        GpuAddress(blas),
    )
}

/// Data access and fast trace: what every traced CLAS here carries.
pub fn data_flags() -> RtasFlags {
    RtasFlags::FAST_TRACE | RtasFlags::ALLOW_DATA_ACCESS
}

/// PTLAS limits of the small scenes here: `instances` instances in two partitions.
pub fn ptlas_limits(instances: u32) -> PtlasLimits {
    PtlasLimits {
        instances,
        partitions: 2,
        max_per_partition: instances,
        global_instances: instances,
        max_ops: 3,
        translation: true,
    }
}

/// A PTLAS write over BLAS `table` index `blas` (or [`NO_BLAS`]).
pub fn ptlas_write(
    index: u32,
    partition: u32,
    custom: u32,
    t: [f32; 3],
    blas: u64,
    aabb: bool,
) -> PtlasWrite {
    use orr_gpu::rtas::consts::{INSTANCE_CULL_DISABLE, INSTANCE_EXPLICIT_AABB};
    PtlasWrite {
        transform: translate(t),
        aabb_min: [-0.1, -0.1, 0.0],
        aabb_max: [1.1, 1.1, 1.0],
        blas,
        custom_index: custom,
        mask: 0xff,
        flags: INSTANCE_CULL_DISABLE | if aabb { INSTANCE_EXPLICIT_AABB } else { 0 },
        instance_index: index,
        partition,
        pad: 0,
    }
}

/// The record arrays and commands of one PTLAS build.
#[derive(Default)]
pub struct PtlasOps {
    pub writes: Vec<PtlasWrite>,
    pub updates: Vec<PtlasUpdate>,
    pub translates: Vec<PtlasTranslate>,
    /// Repeats the first command, which the debug check rejects.
    pub repeat_first: bool,
}

impl Builder {
    /// Writes `ops` through the args kernel (BLAS addresses from `table`) and builds the
    /// PTLAS at `dst`, in place when `update`.
    pub fn build_ptlas(
        &mut self,
        limits: PtlasLimits,
        table: GpuPtr<u64>,
        ops: &PtlasOps,
        dst: GpuAddress,
        scratch: GpuAddress,
        update: bool,
    ) {
        use orr_gpu::PtlasArgs;
        use orr_gpu::rtas::consts::{PTLAS_TRANSLATE, PTLAS_UPDATE, PTLAS_WRITE};
        let d = device();
        let mut commands = Vec::new();
        let mut array =
            |b: &mut Self,
             kind: PtlasArgs,
             op: u32,
             n: usize,
             fill: &dyn Fn(&mut Self, &mut cluster_args::Root)| {
                if n == 0 {
                    return;
                }
                let stride = u64::from(d.ptlas_args_stride(kind));
                let records = b.block(MemoryKind::DeviceLocal, stride * n as u64);
                let mut root = zeroed_root();
                root.dst = records.address().0;
                root.table = table;
                root.n = n as u32;
                fill(b, &mut root);
                b.args(root);
                commands.push(PtlasCommand {
                    args: records.address().0,
                    op,
                    count: n as u32,
                });
            };
        array(
            self,
            PtlasArgs::Write,
            PTLAS_WRITE,
            ops.writes.len(),
            &|b, r| {
                r.writes = b.upload(&ops.writes);
                r.op = OP_WRITE;
            },
        );
        array(
            self,
            PtlasArgs::Update,
            PTLAS_UPDATE,
            ops.updates.len(),
            &|b, r| {
                r.updates = b.upload(&ops.updates);
                r.op = OP_UPDATE;
            },
        );
        array(
            self,
            PtlasArgs::Translate,
            PTLAS_TRANSLATE,
            ops.translates.len(),
            &|b, r| {
                r.translates = b.upload(&ops.translates);
                r.op = OP_TRANSLATE;
            },
        );
        if ops.repeat_first {
            commands.push(commands[0]);
        }
        let records = self.block(
            MemoryKind::DeviceLocal,
            u64::from(d.rtas_args_stride(RtasOp::Ptlas)) * commands.len() as u64,
        );
        let count = self.block(MemoryKind::DeviceLocal, 4).cast::<u32>();
        let mut root = zeroed_root();
        root.dst = records.address().0;
        root.count = count;
        root.commands = self.upload(&commands);
        root.n = commands.len() as u32;
        root.op = OP_COMMAND;
        self.args(root);
        let desc = ptlas_desc(limits);
        self.list().rtas_execute(
            &desc,
            &RtasIo {
                args: records.address(),
                count,
                dst_implicit: dst,
                dst_addresses: GpuPtr::NULL,
                dst_sizes: GpuPtr::NULL,
                scratch,
                src: if update { dst } else { GpuAddress(0) },
            },
        );
    }

    /// PTLAS storage and scratch for `limits`.
    pub fn ptlas_blocks(&mut self, limits: PtlasLimits) -> (GpuAddress, GpuAddress) {
        let sizes = device().rtas_sizes(&ptlas_desc(limits));
        (
            self.block(MemoryKind::DeviceLocal, sizes.result_max)
                .address(),
            self.block(MemoryKind::DeviceLocal, sizes.scratch).address(),
        )
    }
}

pub fn ptlas_desc(limits: PtlasLimits) -> RtasOpDesc {
    RtasOpDesc {
        op: RtasOp::Ptlas,
        mode: RtasMode::Implicit,
        flags: RtasFlags::FAST_TRACE,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Ptlas(limits),
    }
}

/// Template and instantiation records of `templates`.
pub fn template_records(
    b: &mut Builder,
    templates: &[Cluster],
    bounds: &[u64],
) -> Vec<TemplateBuild> {
    b.clas_records(templates, 0)
        .into_iter()
        .enumerate()
        .map(|(i, cluster)| TemplateBuild {
            cluster,
            bounds: bounds.get(i).copied().unwrap_or(0),
        })
        .collect()
}

pub fn instantiate(
    template_index: u64,
    vertices: GpuAddress,
    id_offset: u32,
    geom_offset: u32,
) -> Instantiate {
    Instantiate {
        cluster_id_offset: id_offset,
        geometry_index_offset: geom_offset,
        template_address: template_index,
        vertices: vertices.0,
    }
}

pub fn move_record(table_index: u64) -> Move {
    Move { src: table_index }
}

pub fn move_desc(n: u32, max_bytes: u64) -> RtasOpDesc {
    RtasOpDesc {
        op: RtasOp::Move,
        mode: RtasMode::Explicit,
        flags: RtasFlags::empty(),
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Move(MoveLimits {
            max_args: n,
            max_bytes,
            kind: MoveKind::Clas,
            no_overlap: true,
        }),
    }
}
