//! Cluster acceleration structures of mounted meshes, built on the loading queue.
//!
//! A mesh's clusters build into CLAS from their resident pages: the `clas` kernel expands
//! each cluster's grid positions to f32 and writes its per-triangle flags and micromap
//! indices, the batch builds implicitly into scratch, and once the built sizes are read back
//! one move puts the batch into a block of exactly that size. Clusters with micromaps build
//! in their own batches, the only ones that allow them. Each cut a mesh caches then builds
//! one cluster BLAS over the moved CLAS, sized and moved the same way; a cut with animated
//! part clusters also builds one over its body and one per animated part. Section-mask
//! BLAS build on the frame's list the first frame a copy names the mask.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use orr_gpu::{
    Allocation, BlasLimits, ClusterLimits, ClusterVertexFormat, CommandList, ComputePipeline,
    Device, GpuAddress, GpuPtr, Hazards, KernelSet, MemoryKind, MemoryPriority, Micromap,
    MicromapDesc, MicromapTriangle, MicromapUsage, MoveKind, MoveLimits, OmmFormat, QueueKind,
    RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, ShaderStage, SpecData, Stages,
    TimelinePoint,
};
use orr_pack::mesh::{Cluster, MeshHeader, Part};

use crate::memory::{RangeAllocator, UploadWindow};
use crate::shaders::accel::{self, AccelRoot, AddressJob, ClasJob};
use crate::{ClusterRef, PartGpu, Result, RtCutGpu};

/// `ClasJob.flags` (`accel.slang`).
const CLAS_MICROMAP: u32 = 1;
const CLAS_NON_OPAQUE: u32 = 2;
/// `AccelRoot.op` of `records`.
const ACCEL_BLAS: u32 = 0;
const ACCEL_MOVE: u32 = 1;
/// Cuts a DAG mesh caches, at most.
const DAG_CUTS: usize = 12;
/// Clusters in one CLAS batch, at most.
const BATCH_CLUSTERS: usize = 1 << 16;
const WAIT: Duration = Duration::from_secs(60);

/// The loading side's acceleration-structure kernels.
pub(crate) struct AccelKernels {
    clas: ComputePipeline,
    records: ComputePipeline,
}

impl AccelKernels {
    pub(crate) fn new(device: &Device, kernels: &KernelSet) -> Result<Self> {
        let module = kernels.module(&accel::MODULE)?;
        let pipeline = |entry| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(module, entry, SpecData::default())?;
            Ok(device.create_compute_pipeline(&stage, module.name)?)
        };
        Ok(Self {
            clas: pipeline("clas")?,
            records: pipeline("records")?,
        })
    }
}

/// Per-frame and per-batch limits of loading-queue work.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AccelLimits {
    /// Triangles one CLAS batch builds.
    pub clas_triangles: u32,
    /// Micromaps one submit builds.
    pub micromaps: u32,
}

/// What the loading side keeps of a cluster as its page decodes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClusterInfo {
    pub page: u32,
    pub local: u32,
    pub vertex_count: u32,
    pub tri_count: u32,
    pub section: u8,
    pub part: u8,
    pub flags: u8,
    pub micromap: bool,
    pub self_error: f32,
    pub parent_error: f32,
}

impl ClusterInfo {
    pub(crate) fn new(page: u32, local: u32, c: &Cluster) -> Self {
        Self {
            page,
            local,
            vertex_count: u32::from(c.vertex_count),
            tri_count: u32::from(c.tri_count),
            section: c.section,
            part: c.part,
            flags: c.flags,
            micromap: c.omm_first != Cluster::NO_OMM && c.alpha_first < c.tri_count,
            self_error: c.self_error,
            parent_error: c.parent_error,
        }
    }
}

/// A page's micromap region: its offset in the page and the entries per subdivision level.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PageMicromap {
    pub offset: u32,
    pub usage: [u32; 13],
}

/// One resident mesh as the build sees it.
pub(crate) struct MeshInput<'a> {
    pub header: &'a MeshHeader,
    /// Resident page addresses.
    pub pages: &'a [u64],
    /// By mesh cluster index.
    pub clusters: &'a [ClusterInfo],
    /// By page; `None` where the page has no micromap region.
    pub micromaps: &'a [Option<PageMicromap>],
    /// Discrete levels, finest first: error and mesh cluster range. Empty in DAG mode.
    pub levels: &'a [(f32, u32, u32)],
    /// Sections whose every triangle is non-opaque (linked coverage).
    pub non_opaque: &'a [bool],
    /// Cluster id of mesh cluster 0.
    pub cluster_base: u32,
    /// The mesh's part table.
    pub parts: &'a [Part],
}

/// A mesh's animated parts: those of its clusters with `ANIMATED_PART`, from the lowest
/// such part to the last of the part table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Animated {
    pub first: u32,
    pub count: u32,
}

impl Animated {
    pub(crate) fn of(clusters: &[ClusterInfo], part_count: u32) -> Self {
        let parts = part_count.max(1);
        let first = clusters
            .iter()
            .filter(|c| c.flags & Cluster::ANIMATED_PART != 0)
            .map(|c| u32::from(c.part))
            .min();
        match first {
            Some(first) => Self {
                first,
                count: parts.saturating_sub(first),
            },
            None => Self {
                first: parts,
                count: 0,
            },
        }
    }

    fn index(&self, c: &ClusterInfo) -> Option<usize> {
        let animated = c.flags & Cluster::ANIMATED_PART != 0 && u32::from(c.part) >= self.first;
        animated.then(|| (u32::from(c.part) - self.first) as usize)
    }
}

/// What a mesh keeps for its section-mask BLAS: the CLAS of each cluster and the clusters
/// of each cut.
#[derive(Clone, Debug, Default)]
pub(crate) struct MeshRt {
    pub clas: Vec<u64>,
    pub sections: Vec<u8>,
    pub animated: Vec<bool>,
    pub cuts: Vec<Vec<u32>>,
    /// Per cut: it has animated part clusters, so its body BLAS differs from its rest one.
    pub parted: Vec<bool>,
}

impl MeshRt {
    /// Per table slot (rest of each cut, then body of each cut): the clusters left by
    /// `mask`, or None where the slot repeats the rest slot of its cut.
    fn mask_lists(&self, mask: u16) -> Vec<Option<Vec<u64>>> {
        let shown = |c: u32| {
            let s = self.sections[c as usize];
            s >= 16 || mask >> s & 1 == 0
        };
        let rest = self.cuts.iter().map(|cut| {
            Some(
                cut.iter()
                    .filter(|&&c| shown(c))
                    .map(|&c| self.clas[c as usize])
                    .collect(),
            )
        });
        let body = self.cuts.iter().zip(&self.parted).map(|(cut, &parted)| {
            parted.then(|| {
                cut.iter()
                    .filter(|&&c| shown(c) && !self.animated[c as usize])
                    .map(|&c| self.clas[c as usize])
                    .collect()
            })
        });
        rest.chain(body).collect()
    }
}

/// A mesh's traced cuts and animated parts: their records' place in the mount's tables.
#[derive(Clone, Debug, Default)]
pub(crate) struct MeshCuts {
    pub table: GpuPtr<RtCutGpu>,
    pub count: u32,
    pub parts: GpuPtr<PartGpu>,
    pub animated: Animated,
    /// The sections of every animated part.
    pub animated_sections: u32,
    pub rt: MeshRt,
}

/// The acceleration structures of a mount, kept alive while it is resident.
#[derive(Default)]
pub(crate) struct MountAccel {
    blocks: Vec<Allocation<u8>>,
    micromaps: Vec<(Micromap, Allocation<u8>)>,
    cuts: Option<Allocation<RtCutGpu>>,
    part_blas: Option<Allocation<u64>>,
    parts: Option<Allocation<PartGpu>>,
}

/// What mounts on the loading side share with the render thread: the kernels, the batch
/// limits and the resident cluster ids, which the loading side takes and the render thread
/// frees with their mount.
pub(crate) struct AccelShared {
    pub kernels: AccelKernels,
    pub limits: AccelLimits,
    pub cluster_ids: Mutex<RangeAllocator>,
    pub cluster_capacity: u32,
}

/// The resident cluster table entries of a mesh's clusters; their CLAS once built.
pub(crate) fn cluster_refs(m: &MeshInput<'_>) -> Vec<ClusterRef> {
    m.clusters
        .iter()
        .zip(0..)
        .map(|(c, mesh_cluster)| ClusterRef {
            // SAFETY: the page is resident in the mount's block.
            page: unsafe { GpuPtr::from_address(GpuAddress(m.pages[c.page as usize])) },
            local: c.local,
            mesh_cluster,
            clas: 0,
        })
        .collect()
}

/// `tables.slang` `cluster_hash` of one mesh cluster.
fn cluster_hash(mesh_cluster: u32) -> [u32; 2] {
    let mut a = mesh_cluster
        .wrapping_mul(0x9e37_79b1)
        .wrapping_add(0x7f4a_7c15);
    a ^= a >> 16;
    a = a.wrapping_mul(0x85eb_ca6b);
    a ^= a >> 13;
    a = a.wrapping_mul(0xc2b2_ae35);
    a ^= a >> 16;
    let mut b = (mesh_cluster ^ 0x5bd1_e995).wrapping_mul(0x27d4_eb2f);
    b ^= b >> 15;
    b = b.wrapping_mul(0x1656_67b1);
    b ^= b >> 13;
    [a, b]
}

/// The order-free hash the traversal gives a set of mesh clusters.
pub(crate) fn cut_hash(clusters: &[u32]) -> [u32; 2] {
    clusters.iter().fold([0, 0], |h, &c| {
        let k = cluster_hash(c);
        [h[0].wrapping_add(k[0]), h[1].wrapping_add(k[1])]
    })
}

/// The clusters of each cut a mesh caches, finest first, with the largest error in it.
/// Discrete meshes cache their levels. DAG meshes cache cuts at fixed thresholds: a cut
/// takes every cluster whose own error is at or under the threshold and whose parent's is
/// over it, from the merged stage once the threshold reaches the merge error. Copies whose
/// traversed cut equals one trace it, as do distant copies and posed ones.
pub(crate) fn cuts(m: &MeshInput<'_>) -> Vec<(f32, Vec<u32>)> {
    if !m.levels.is_empty() {
        return m
            .levels
            .iter()
            .map(|&(error, first, count)| (error, (first..first + count).collect()))
            .collect();
    }
    let merge = m.header.merge_error;
    let finite = |e: f32| e.is_finite() && e > 0.0;
    let smallest = m
        .clusters
        .iter()
        .map(|c| c.self_error)
        .filter(|&e| finite(e))
        .fold(f32::INFINITY, f32::min);
    let largest = m
        .clusters
        .iter()
        .map(|c| c.self_error)
        .filter(|&e| finite(e))
        .fold(0.0, f32::max);
    let mut thresholds = vec![0.0f32];
    if smallest.is_finite() {
        let mut t = smallest;
        while thresholds.len() < DAG_CUTS {
            thresholds.push(t);
            if t >= largest {
                break;
            }
            t *= 4.0;
        }
    }
    let mut out: Vec<(f32, Vec<u32>)> = Vec::new();
    for t in thresholds {
        let merged = t >= merge;
        let selected: Vec<u32> = (0..m.clusters.len() as u32)
            .filter(|&i| {
                let c = &m.clusters[i as usize];
                ((c.flags & Cluster::MERGED) != 0) == merged
                    && c.self_error <= t
                    && c.parent_error > t
            })
            .collect();
        let error = selected
            .iter()
            .map(|&i| m.clusters[i as usize].self_error)
            .fold(0.0, f32::max);
        if selected.is_empty() || out.last().is_some_and(|(_, s)| *s == selected) {
            continue;
        }
        out.push((error, selected));
    }
    out
}

/// One list on the loading queue, the memory it reads and the point to wait first.
struct Step<'d> {
    device: &'d Device,
    kind: QueueKind,
    list: Option<CommandList<'d>>,
    keep: Vec<Allocation<u8>>,
    /// The last submitted list's memory, which the next list may still read (a move reads
    /// the implicit build's results); dropped when that next list is submitted.
    previous: Vec<Allocation<u8>>,
    wait: Vec<TimelinePoint>,
}

impl<'d> Step<'d> {
    fn new(device: &'d Device, kind: QueueKind, wait: Option<TimelinePoint>) -> Result<Self> {
        let mut s = Self {
            device,
            kind,
            list: None,
            keep: Vec::new(),
            previous: Vec::new(),
            wait: wait.into_iter().collect(),
        };
        s.open()?;
        Ok(s)
    }

    fn open(&mut self) -> Result<()> {
        let queue = self.device.queue(self.kind).expect("loading queue");
        let mut list = queue.command_list()?;
        list.barrier(Stages::ALL, Stages::ALL, Hazards::ACCEL_STRUCT);
        self.list = Some(list);
        Ok(())
    }

    fn list(&mut self) -> &mut CommandList<'d> {
        self.list.as_mut().expect("open list")
    }

    fn block(&mut self, kind: MemoryKind, bytes: u64, name: &str) -> Result<GpuPtr<u8>> {
        let a = self.device.gpu_malloc::<u8>(kind, bytes.max(16), name)?;
        let gpu = a.gpu;
        self.keep.push(a);
        Ok(gpu)
    }

    fn upload<T: bytemuck::Pod>(&mut self, values: &[T], name: &str) -> Result<GpuPtr<T>> {
        let bytes = size_of_val(values) as u64;
        let mut a = self
            .device
            .gpu_malloc::<u8>(MemoryKind::Upload, bytes.max(16), name)?;
        a.as_mut_slice().expect("upload is mapped")[..bytes as usize]
            .copy_from_slice(bytemuck::cast_slice(values));
        let gpu = a.gpu.cast();
        self.keep.push(a);
        Ok(gpu)
    }

    fn dispatch(&mut self, p: &ComputePipeline, root: &AccelRoot, groups: u32) -> Result<()> {
        let root = self.upload(std::slice::from_ref(root), "accel root")?;
        let list = self.list();
        list.dispatch(p, root, [groups.max(1), 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::AS_BUILD,
            Hazards::empty(),
        );
        Ok(())
    }

    /// Submits the list and opens the next.
    fn submit(&mut self) -> Result<TimelinePoint> {
        let queue = self.device.queue(self.kind).expect("loading queue");
        let list = self.list.take().expect("open list");
        let point = queue.submit(vec![list], &std::mem::take(&mut self.wait))?;
        self.previous = std::mem::take(&mut self.keep);
        self.open()?;
        Ok(point)
    }

    fn submit_and_wait(&mut self) -> Result<()> {
        let point = self.submit()?;
        let queue = self.device.queue(self.kind).expect("loading queue");
        Ok(queue.wait(point, WAIT)?)
    }
}

/// Moves the implicit results of one batch (addresses and sizes read back) into one block
/// of exactly their aligned size; returns their new addresses.
fn pack(
    step: &mut Step<'_>,
    k: &AccelKernels,
    kind: MoveKind,
    align: u64,
    addresses: &[u64],
    sizes: &[u32],
    keep: &mut Vec<Allocation<u8>>,
) -> Result<Vec<u64>> {
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut total = 0u64;
    for &s in sizes {
        offsets.push(total);
        total += u64::from(s).next_multiple_of(align);
    }
    let block = step.device.gpu_malloc_with_priority::<u8>(
        MemoryKind::DeviceLocal,
        MemoryPriority::High,
        total.max(align),
        "cluster AS",
    )?;
    let base = block.gpu.address().0;
    keep.push(block);
    let dst: Vec<u64> = offsets.iter().map(|o| base + o).collect();
    let jobs: Vec<AddressJob> = addresses
        .iter()
        .map(|&a| AddressJob {
            address: a,
            count: 0,
            _pad: 0,
        })
        .collect();
    let n = jobs.len() as u32;
    let stride = u64::from(step.device.rtas_args_stride(RtasOp::Move));
    let records = step.block(MemoryKind::DeviceLocal, stride * u64::from(n), "moves")?;
    let count = step
        .block(MemoryKind::DeviceLocal, 4, "move count")?
        .cast::<u32>();
    let root = AccelRoot {
        addresses: step.upload(&jobs, "move jobs")?,
        records,
        record_count: count,
        count: n,
        op: ACCEL_MOVE,
        ..bytemuck::Zeroable::zeroed()
    };
    step.dispatch(&k.records, &root, n.div_ceil(64))?;
    let dst_addresses = step.upload(&dst, "move destinations")?;
    let desc = RtasOpDesc {
        op: RtasOp::Move,
        mode: RtasMode::Explicit,
        flags: RtasFlags::empty(),
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Move(MoveLimits {
            max_args: n,
            max_bytes: total,
            kind,
            no_overlap: true,
        }),
    };
    let sizes = step.device.rtas_sizes(&desc);
    let scratch = step.block(MemoryKind::DeviceLocal, sizes.scratch, "move scratch")?;
    step.list().rtas_execute(
        &desc,
        &RtasIo {
            args: records.address(),
            count,
            dst_implicit: GpuAddress(0),
            dst_addresses,
            dst_sizes: GpuPtr::NULL,
            scratch: scratch.address(),
            src: GpuAddress(0),
        },
    );
    // Later builds in the same list read the moved structures.
    step.list()
        .barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    Ok(dst)
}

/// Runs one implicit cluster op over `n` records the `dispatch` closure writes, waits for
/// it and returns each result's address and size.
fn implicit(
    step: &mut Step<'_>,
    desc: &RtasOpDesc,
    n: u32,
    write: impl FnOnce(&mut Step<'_>, GpuPtr<u8>, GpuPtr<u32>) -> Result<()>,
) -> Result<(Vec<u64>, Vec<u32>)> {
    let d = step.device;
    let stride = u64::from(d.rtas_args_stride(desc.op));
    let records = step.block(MemoryKind::DeviceLocal, stride * u64::from(n), "records")?;
    let count = step
        .block(MemoryKind::DeviceLocal, 4, "record count")?
        .cast::<u32>();
    write(step, records, count)?;
    let sizes = d.rtas_sizes(desc);
    let dst = step.block(
        MemoryKind::DeviceLocal,
        sizes.result_max,
        "implicit results",
    )?;
    let scratch = step.block(MemoryKind::DeviceLocal, sizes.scratch, "build scratch")?;
    let addresses = d.gpu_malloc::<u64>(MemoryKind::Readback, u64::from(n), "addresses")?;
    let built = d.gpu_malloc::<u32>(MemoryKind::Readback, u64::from(n), "built sizes")?;
    step.list().rtas_execute(
        desc,
        &RtasIo {
            args: records.address(),
            count,
            dst_implicit: dst.address(),
            dst_addresses: addresses.gpu,
            dst_sizes: built.gpu,
            scratch: scratch.address(),
            src: GpuAddress(0),
        },
    );
    step.list()
        .barrier(Stages::AS_BUILD, Stages::HOST, Hazards::empty());
    step.submit_and_wait()?;
    Ok((
        addresses.as_slice().expect("readback is mapped").to_vec(),
        built.as_slice().expect("readback is mapped").to_vec(),
    ))
}

/// What [`build`] made.
pub(crate) struct Built {
    pub accel: MountAccel,
    /// By mesh.
    pub cuts: Vec<MeshCuts>,
    /// The point the last move signals; none when nothing was built.
    pub point: Option<(QueueKind, TimelinePoint)>,
}

/// Builds the micromaps, CLAS and cut BLAS of `meshes` on the loading queue after `wait`.
pub(crate) fn build(
    device: &Device,
    k: &AccelKernels,
    limits: AccelLimits,
    meshes: &[MeshInput<'_>],
    wait: Option<TimelinePoint>,
) -> Result<Built> {
    let caps = device.caps();
    let mut accel = MountAccel::default();
    let supported = caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0;
    if !supported || meshes.iter().all(|m| m.clusters.is_empty()) {
        return Ok(Built {
            accel,
            cuts: meshes
                .iter()
                .map(|m| MeshCuts {
                    animated: Animated::of(m.clusters, m.header.parts.count),
                    ..MeshCuts::default()
                })
                .collect(),
            point: None,
        });
    }
    let kind = [QueueKind::Loading, QueueKind::Render]
        .into_iter()
        .find(|&q| device.queue(q).is_some())
        .expect("render queue");
    let mut step = Step::new(device, kind, wait)?;

    // Micromaps, one per page with a micromap region, at most `limits.micromaps` a submit.
    let omm = caps.omm && caps.cluster.omm_in_clusters;
    let mut micromap_of: Vec<Vec<u64>> = meshes.iter().map(|m| vec![0; m.pages.len()]).collect();
    if omm {
        let mut built = 0;
        for (mi, m) in meshes.iter().enumerate() {
            for (p, region) in m.micromaps.iter().enumerate() {
                let Some(region) = region else { continue };
                let usage: Vec<MicromapUsage> = region
                    .usage
                    .iter()
                    .enumerate()
                    .filter(|&(_, &n)| n > 0)
                    .map(|(level, &count)| MicromapUsage {
                        count,
                        level: level as u32,
                        format: OmmFormat::FourState,
                    })
                    .collect();
                if usage.is_empty() {
                    continue;
                }
                let at = GpuAddress(m.pages[p] + u64::from(region.offset));
                let desc = MicromapDesc {
                    flags: RtasFlags::FAST_TRACE,
                    usage: &usage,
                    data: at,
                    // SAFETY: the entries start the page's micromap region.
                    triangles: unsafe { GpuPtr::<MicromapTriangle>::from_address(at) },
                };
                let sizes = device.micromap_sizes(&desc);
                let block = device.gpu_malloc_with_priority::<u8>(
                    MemoryKind::DeviceLocal,
                    MemoryPriority::High,
                    sizes.result_max,
                    "micromap",
                )?;
                let micromap = device.create_micromap(sizes.result_max, &block, 0)?;
                let scratch = step.block(MemoryKind::DeviceLocal, sizes.scratch, "omm scratch")?;
                step.list()
                    .build_micromap(&desc, &micromap, scratch.address());
                micromap_of[mi][p] = micromap.address().0;
                accel.micromaps.push((micromap, block));
                built += 1;
                if built % limits.micromaps.max(1) == 0 {
                    step.submit()?;
                }
            }
        }
        // The CLAS builds below read the micromaps.
        step.list()
            .barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    // CLAS: clusters without micromaps first, then those with, in batches of at most
    // `limits.clas_triangles` triangles.
    let mut clas: Vec<Vec<u64>> = meshes.iter().map(|m| vec![0; m.clusters.len()]).collect();
    for with_omm in [false, true] {
        let mut members: Vec<(usize, usize)> = Vec::new();
        for (mi, m) in meshes.iter().enumerate() {
            for (ci, c) in m.clusters.iter().enumerate() {
                let uses = omm
                    && c.micromap
                    && !m.non_opaque[c.section as usize]
                    && micromap_of[mi][c.page as usize] != 0;
                if uses == with_omm {
                    members.push((mi, ci));
                }
            }
        }
        let mut start = 0;
        while start < members.len() {
            let mut end = start;
            let mut tris = 0;
            while end < members.len() && end - start < BATCH_CLUSTERS {
                let (mi, ci) = members[end];
                let t = meshes[mi].clusters[ci].tri_count;
                if end > start && tris + t > limits.clas_triangles {
                    break;
                }
                tris += t;
                end += 1;
            }
            let batch = &members[start..end];
            build_clas(
                &mut step,
                k,
                meshes,
                &micromap_of,
                batch,
                with_omm,
                &mut clas,
                &mut accel.blocks,
            )?;
            start = end;
        }
    }

    // One cluster BLAS per cut over the moved CLAS; a cut with animated part clusters also
    // one over its body (when not empty) and one per animated part with clusters in it.
    let mut jobs = Vec::new();
    let mut lists: Vec<u64> = Vec::new();
    let mut job = |clusters: &[u32], clas: &[u64]| {
        jobs.push((lists.len(), clusters.len() as u32));
        lists.extend(clusters.iter().map(|&c| clas[c as usize]));
        jobs.len() - 1
    };
    let mut plans = Vec::new();
    let mut per_mesh = Vec::with_capacity(meshes.len());
    for (mi, m) in meshes.iter().enumerate() {
        let animated = Animated::of(m.clusters, m.header.parts.count);
        let cs = cuts(m);
        per_mesh.push((cs.len() as u32, animated));
        for (error, clusters) in cs {
            let rest = job(&clusters, &clas[mi]);
            let mut parts = vec![Vec::new(); animated.count as usize];
            let mut body = Vec::new();
            for &c in &clusters {
                match animated.index(&m.clusters[c as usize]) {
                    Some(j) => parts[j].push(c),
                    None => body.push(c),
                }
            }
            let parted = parts.iter().any(|p| !p.is_empty());
            let (body, parts) = if parted {
                let body = (!body.is_empty()).then(|| job(&body, &clas[mi]));
                let parts: Vec<Option<usize>> = parts
                    .iter()
                    .map(|p| (!p.is_empty()).then(|| job(p, &clas[mi])))
                    .collect();
                (body, Some(parts))
            } else {
                (Some(rest), None)
            };
            plans.push(CutPlan {
                error,
                clusters,
                rest,
                body,
                parts,
            });
        }
    }
    let address_list = step.upload(&lists, "BLAS CLAS addresses")?;
    let blas_jobs: Vec<AddressJob> = jobs
        .iter()
        .map(|&(first, count)| AddressJob {
            address: address_list.address().0 + 8 * first as u64,
            count,
            _pad: 0,
        })
        .collect();
    let n = blas_jobs.len() as u32;
    let desc = RtasOpDesc {
        op: RtasOp::BuildClusterBlas,
        mode: RtasMode::Implicit,
        flags: RtasFlags::FAST_TRACE,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Blas(BlasLimits {
            max_args: n,
            max_clas_per_blas: jobs.iter().map(|j| j.1).max().unwrap_or(1),
            max_total_clas: lists.len() as u32,
        }),
    };
    let (addresses, sizes) = implicit(&mut step, &desc, n, |step, records, count| {
        let root = AccelRoot {
            addresses: step.upload(&blas_jobs, "BLAS jobs")?,
            records,
            record_count: count,
            count: n,
            op: ACCEL_BLAS,
            ..bytemuck::Zeroable::zeroed()
        };
        step.dispatch(&k.records, &root, n.div_ceil(64))
    })?;
    let align = u64::from(caps.cluster.align_blas);
    let blas = pack(
        &mut step,
        k,
        MoveKind::ClusterBlas,
        align,
        &addresses,
        &sizes,
        &mut accel.blocks,
    )?;
    let point = step.submit()?;

    let part_rows: usize = plans
        .iter()
        .filter_map(|p| p.parts.as_ref().map(Vec::len))
        .sum();
    let mut part_blas =
        device.gpu_malloc::<u64>(MemoryKind::Upload, part_rows.max(1) as u64, "part BLAS")?;
    let at = |j: Option<usize>| j.map_or(0, |j| blas[j]);
    let mut row = 0;
    let records: Vec<RtCutGpu> = plans
        .iter()
        .map(|p| {
            let parts = match &p.parts {
                None => GpuPtr::NULL,
                Some(parts) => {
                    let out = &mut part_blas.as_mut_slice().expect("upload is mapped")
                        [row..row + parts.len()];
                    for (o, &j) in out.iter_mut().zip(parts) {
                        *o = at(j);
                    }
                    row += parts.len();
                    part_blas.gpu.offset((row - parts.len()) as u64)
                }
            };
            RtCutGpu {
                blas: blas[p.rest],
                error: p.error,
                clusters: p.clusters.len() as u32,
                body: at(p.body),
                parts,
                hash: cut_hash(&p.clusters),
            }
        })
        .collect();
    let mut table = device.gpu_malloc::<RtCutGpu>(
        MemoryKind::Upload,
        records.len().max(1) as u64,
        "mesh cuts",
    )?;
    table.as_mut_slice().expect("upload is mapped")[..records.len()].copy_from_slice(&records);

    let per_part: Vec<Vec<PartGpu>> = meshes
        .iter()
        .zip(&per_mesh)
        .map(|(m, &(_, animated))| part_records(m, animated))
        .collect();
    let part_records: Vec<PartGpu> = per_part.iter().flatten().copied().collect();
    let mut parts = device.gpu_malloc::<PartGpu>(
        MemoryKind::Upload,
        part_records.len().max(1) as u64,
        "animated parts",
    )?;
    parts.as_mut_slice().expect("upload is mapped")[..part_records.len()]
        .copy_from_slice(&part_records);

    let mut first = 0u64;
    let mut first_part = 0u64;
    let mut plans = plans.into_iter();
    let cuts = meshes
        .iter()
        .zip(&clas)
        .zip(&per_mesh)
        .zip(&per_part)
        .map(|(((m, clas), &(count, animated)), parts_of)| {
            let mine: Vec<CutPlan> = plans.by_ref().take(count as usize).collect();
            let c = MeshCuts {
                table: table.gpu.offset(first),
                count,
                parts: parts.gpu.offset(first_part),
                animated,
                animated_sections: parts_of.iter().fold(0, |s, p| s | p.sections),
                rt: MeshRt {
                    clas: clas.clone(),
                    sections: m.clusters.iter().map(|c| c.section).collect(),
                    animated: m
                        .clusters
                        .iter()
                        .map(|c| animated.index(c).is_some())
                        .collect(),
                    parted: mine.iter().map(|p| p.parts.is_some()).collect(),
                    cuts: mine.into_iter().map(|p| p.clusters).collect(),
                },
            };
            first += u64::from(count);
            first_part += u64::from(animated.count);
            c
        })
        .collect();
    accel.cuts = Some(table);
    accel.part_blas = Some(part_blas);
    accel.parts = Some(parts);
    Ok(Built {
        accel,
        cuts,
        point: Some((kind, point)),
    })
}

/// One traced cut's BLAS jobs.
struct CutPlan {
    error: f32,
    clusters: Vec<u32>,
    rest: usize,
    /// None: an empty body.
    body: Option<usize>,
    /// Per animated part, None where it has no clusters; None without animated part clusters.
    parts: Option<Vec<Option<usize>>>,
}

/// Each animated part's rest bounds and the sections its clusters use.
fn part_records(m: &MeshInput<'_>, animated: Animated) -> Vec<PartGpu> {
    let mut out: Vec<PartGpu> = (0..animated.count)
        .map(|j| {
            let bounds = m
                .parts
                .get((animated.first + j) as usize)
                .map_or(orr_pack::Aabb::EMPTY, |p| p.bounds);
            PartGpu {
                bounds_min: bounds.min,
                sections: 0,
                bounds_max: bounds.max,
                _pad: 0,
            }
        })
        .collect();
    for c in m.clusters {
        if let Some(j) = animated.index(c)
            && c.section < 32
        {
            out[j].sections |= 1 << c.section;
        }
    }
    out
}

/// The BLAS of one (mesh, section mask): a block of their implicit results and the table
/// `MaskGpu::blas` points at.
pub(crate) struct MaskBlas {
    pub table: Allocation<u64>,
    /// Shared by the masks of one build; none when every slot is empty.
    _block: Option<Arc<Allocation<u8>>>,
}

/// Records on `list` the BLAS of each `(mesh, mask)` over the clusters the mask leaves,
/// one implicit build for all, and the copies of their addresses into each table; `keep`
/// takes the build's scratch, to free once the list has run.
pub(crate) fn build_masks(
    device: &Device,
    k: &AccelKernels,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    masks: &[(&MeshRt, u16)],
    keep: &mut Vec<Allocation<u8>>,
) -> Result<Vec<MaskBlas>> {
    let dl = MemoryKind::DeviceLocal;
    let mut lists: Vec<u64> = Vec::new();
    let mut jobs: Vec<(usize, u32)> = Vec::new();
    // Per job: the table and slots it fills.
    let mut targets: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut tables = Vec::with_capacity(masks.len());
    for (mi, &(rt, mask)) in masks.iter().enumerate() {
        let slots = mask_slots(rt.mask_lists(mask));
        tables.push(device.gpu_malloc::<u64>(dl, 2 * rt.cuts.len().max(1) as u64, "mask BLAS")?);
        for (clusters, filled) in slots {
            jobs.push((lists.len(), clusters.len() as u32));
            lists.extend(clusters);
            targets.push((mi, filled));
        }
    }
    let n = jobs.len() as u32;
    let longest = tables.iter().map(|t| t.size).max().unwrap_or(8);
    let (zeros, out) = window.slice::<u64>(device, (longest / 8) as u32)?;
    out.fill(0);
    for t in &tables {
        list.copy(t.gpu.cast(), zeros.cast(), t.size);
    }
    if n == 0 {
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        return Ok(tables
            .into_iter()
            .map(|table| MaskBlas {
                table,
                _block: None,
            })
            .collect());
    }
    let (src, out) = window.slice::<u64>(device, lists.len() as u32)?;
    out.copy_from_slice(&lists);
    let (job_src, job_out) = window.slice::<AddressJob>(device, n)?;
    for (o, &(first, count)) in job_out.iter_mut().zip(&jobs) {
        *o = AddressJob {
            address: src.address().0 + 8 * first as u64,
            count,
            _pad: 0,
        };
    }
    let desc = RtasOpDesc {
        op: RtasOp::BuildClusterBlas,
        mode: RtasMode::Implicit,
        flags: RtasFlags::FAST_TRACE,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Blas(BlasLimits {
            max_args: n,
            max_clas_per_blas: jobs.iter().map(|j| j.1).max().unwrap_or(1),
            max_total_clas: lists.len() as u32,
        }),
    };
    let sizes = device.rtas_sizes(&desc);
    let stride = u64::from(device.rtas_args_stride(RtasOp::BuildClusterBlas));
    let (records, _) = window.slice::<u8>(device, (stride * u64::from(n)) as u32)?;
    let (count, _) = window.slice::<u32>(device, 1)?;
    let (addresses, _) = window.slice::<u64>(device, n)?;
    let (built, _) = window.slice::<u32>(device, n)?;
    let scratch = device.gpu_malloc::<u8>(dl, sizes.scratch.max(16), "mask BLAS scratch")?;
    let scratch_at = scratch.gpu.address();
    keep.push(scratch);
    let block = Arc::new(device.gpu_malloc_with_priority::<u8>(
        dl,
        MemoryPriority::High,
        sizes.result_max.max(16),
        "mask BLAS block",
    )?);
    let root = AccelRoot {
        addresses: job_src,
        records,
        record_count: count,
        count: n,
        op: ACCEL_BLAS,
        ..bytemuck::Zeroable::zeroed()
    };
    let root = window.put_root(device, &root)?;
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    list.dispatch(&k.records, root, [n.div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
    list.rtas_execute(
        &desc,
        &RtasIo {
            args: records.address(),
            count,
            dst_implicit: block.gpu.address(),
            dst_addresses: addresses,
            dst_sizes: built,
            scratch: scratch_at,
            src: GpuAddress(0),
        },
    );
    list.barrier(Stages::AS_BUILD, Stages::TRANSFER, Hazards::empty());
    for (j, (mi, filled)) in targets.iter().enumerate() {
        for &slot in filled {
            list.copy(
                tables[*mi].gpu.offset(u64::from(slot)).cast(),
                addresses.offset(j as u64).cast(),
                8,
            );
        }
    }
    list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
    Ok(tables
        .into_iter()
        .map(|table| MaskBlas {
            table,
            _block: Some(Arc::clone(&block)),
        })
        .collect())
}

/// The non-empty cluster lists of a mask's table slots and the slots each fills: a slot
/// that repeats its cut's rest slot takes the rest slot's BLAS.
fn mask_slots(lists: Vec<Option<Vec<u64>>>) -> Vec<(Vec<u64>, Vec<u32>)> {
    let cuts = lists.len() / 2;
    let mut out: Vec<(Vec<u64>, Vec<u32>)> = Vec::new();
    let mut rest_job = vec![None; cuts];
    for (slot, l) in lists.into_iter().enumerate() {
        match l {
            Some(l) if !l.is_empty() => {
                if slot < cuts {
                    rest_job[slot] = Some(out.len());
                }
                out.push((l, vec![slot as u32]));
            }
            Some(_) => {}
            None => {
                if let Some(j) = rest_job[slot - cuts] {
                    out[j].1.push(slot as u32);
                }
            }
        }
    }
    out
}

/// One CLAS batch: build, read the sizes back and move into a block of its own.
#[allow(clippy::too_many_arguments)]
fn build_clas(
    step: &mut Step<'_>,
    k: &AccelKernels,
    meshes: &[MeshInput<'_>],
    micromaps: &[Vec<u64>],
    batch: &[(usize, usize)],
    with_omm: bool,
    clas: &mut [Vec<u64>],
    keep: &mut Vec<Allocation<u8>>,
) -> Result<()> {
    let caps = step.device.caps();
    let mut jobs = Vec::with_capacity(batch.len());
    let (mut vertices, mut triangles) = (0u32, 0u32);
    let (mut max_tris, mut max_verts, mut min_truncate) = (1, 1, u32::MAX);
    for &(mi, ci) in batch {
        let m = &meshes[mi];
        let c = &m.clusters[ci];
        let mut flags = 0;
        if m.non_opaque[c.section as usize] {
            flags |= CLAS_NON_OPAQUE;
        }
        if with_omm {
            flags |= CLAS_MICROMAP;
        }
        let truncate = u32::from(m.header.clas_truncate_bits);
        jobs.push(ClasJob {
            // SAFETY: the page is resident in the mount's block.
            page: unsafe { GpuPtr::from_address(GpuAddress(m.pages[c.page as usize])) },
            micromap: micromaps[mi][c.page as usize],
            local: c.local,
            cluster_id: m.cluster_base + ci as u32,
            vertex_first: vertices,
            triangle_first: triangles,
            step: m.header.step(),
            truncate_bits: truncate,
            flags,
            _pad: 0,
        });
        vertices += c.vertex_count;
        triangles += c.tri_count;
        max_tris = max_tris.max(c.tri_count);
        max_verts = max_verts.max(c.vertex_count);
        min_truncate = min_truncate.min(truncate);
    }
    let n = jobs.len() as u32;
    let mut flags = RtasFlags::FAST_TRACE | RtasFlags::ALLOW_DATA_ACCESS;
    if with_omm {
        flags |= RtasFlags::ALLOW_OMM;
    }
    let desc = RtasOpDesc {
        op: RtasOp::BuildClas,
        mode: RtasMode::Implicit,
        flags,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Cluster(ClusterLimits {
            max_args: n,
            max_tris_per_cluster: max_tris,
            max_verts_per_cluster: max_verts,
            max_total_tris: triangles,
            max_total_verts: vertices,
            max_geometry_index: 0,
            max_geometries_per_cluster: 1,
            min_truncate_bits: min_truncate,
        }),
    };
    let (addresses, sizes) = implicit(step, &desc, n, |step, records, count| {
        let root = AccelRoot {
            clas: step.upload(&jobs, "CLAS jobs")?,
            records,
            record_count: count,
            vertices: step
                .block(
                    MemoryKind::DeviceLocal,
                    12 * u64::from(vertices),
                    "positions",
                )?
                .cast(),
            triangle_flags: step
                .block(MemoryKind::DeviceLocal, 4 * u64::from(triangles), "flags")?
                .cast(),
            omm_indices: step
                .block(MemoryKind::DeviceLocal, 4 * u64::from(triangles), "omm")?
                .cast(),
            count: n,
            ..bytemuck::Zeroable::zeroed()
        };
        step.dispatch(&k.clas, &root, n)
    })?;
    let align = u64::from(caps.cluster.align_clas);
    let moved = pack(step, k, MoveKind::Clas, align, &addresses, &sizes, keep)?;
    for (&(mi, ci), a) in batch.iter().zip(moved) {
        clas[mi][ci] = a;
    }
    Ok(())
}
