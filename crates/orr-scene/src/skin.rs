//! Skinned copies: a skinned mesh's skin table at mount, and each frame the skinning of every
//! skinned copy of a scene into its own vertex arrays, the instantiation of its cut's cluster
//! templates and one cluster BLAS per traced copy, on the frame's list after scene compute.
//!
//! A copy takes one cut a frame by its distance from the scene focus, the same for raster
//! and ray tracing. Its vertex arrays hold every vertex of its mesh, by mesh cluster, so a
//! cut change needs nothing moved; previous positions come from the palette group's previous
//! transforms, so motion follows the palette. The frame's instantiated CLAS and BLAS land in
//! blocks sized for `SceneDesc::skinned_clusters`; frames run in order on the render queue,
//! so a frame's builds never overlap the traces of the frame before.

use orr_gpu::{
    Allocation, BlasLimits, ClusterLimits, ClusterVertexFormat, CommandList, Device, DispatchArgs,
    GpuAddress, GpuPtr, Hazards, MemoryKind, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp,
    RtasOpDesc, Stages,
};

use crate::accel::{self, MeshInput};
use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::scene::{InstanceFlags, ModelId, TransformGroupId};
use crate::shaders::skin::{self, SkinFrameGpu, SkinRoot};
use crate::{Result, SceneRoot, SkinMeshGpu};

/// `skin.slang` `SKIN_*`.
const SKIN_WORK: u64 = 0;
const SKIN_JOBS: u64 = 1;
const SKIN_REFUSED: u64 = 2;
const SKIN_ARGS: u64 = 4;
const SKIN_COUNTERS: u64 = 8;

/// `copies` skinned copies of `model`, one instance slot each. Copy `i` is placed by
/// transform `first + i` of `transforms` and posed by palette entries
/// `palette_first + i x joints..` of `palette`, one skinning matrix per skeleton joint
/// (`orr_pack::skeleton`), which a consumer's pose kernel or the CPU writes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinnedDesc {
    pub model: ModelId,
    pub copies: u32,
    pub transforms: TransformGroupId,
    pub first: u32,
    pub palette: TransformGroupId,
    pub palette_first: u32,
    /// Base in the scene's material-set table; None: the model's own materials.
    pub material_set: Option<u32>,
    pub flags: InstanceFlags,
    /// m from the scene focus within which the copies are traced.
    pub rt_range: f32,
    /// Never read by orr.
    pub user_tag: u32,
}

/// Appends `values` at the next 16-byte boundary; returns their offset.
fn put<T: bytemuck::Pod>(bytes: &mut Vec<u8>, values: &[T]) -> u64 {
    bytes.resize(bytes.len().next_multiple_of(16), 0);
    let at = bytes.len() as u64;
    bytes.extend_from_slice(bytemuck::cast_slice(values));
    at
}

/// The skin table of a skinned mesh: [`SkinMeshGpu`] and its arrays, in one block.
pub(crate) fn mesh_table(
    device: &Device,
    m: &MeshInput<'_>,
    templates: &[u64],
) -> Result<Allocation<u8>> {
    let mut first = Vec::with_capacity(m.clusters.len() + 1);
    let mut total = 0u32;
    for c in m.clusters {
        first.push(total);
        total += c.vertex_count;
    }
    first.push(total);
    let cuts = accel::cuts(m);
    let mut clusters = Vec::new();
    let mut cut_first = Vec::with_capacity(cuts.len() + 1);
    for (_, c) in &cuts {
        cut_first.push(clusters.len() as u32);
        clusters.extend_from_slice(c);
    }
    cut_first.push(clusters.len() as u32);
    let errors: Vec<f32> = cuts.iter().map(|&(e, _)| e).collect();
    let mut templates = templates.to_vec();
    templates.resize(m.clusters.len(), 0);

    let mut bytes = vec![0u8; size_of::<SkinMeshGpu>()];
    let at = [
        put(&mut bytes, &first),
        put(&mut bytes, &templates),
        put(&mut bytes, &clusters),
        put(&mut bytes, &cut_first),
        put(&mut bytes, &errors),
    ];
    let mut block =
        device.gpu_malloc::<u8>(MemoryKind::Upload, bytes.len() as u64, "skin table")?;
    let base = block.gpu;
    let record = SkinMeshGpu {
        vertex_first: base.offset(at[0]).cast(),
        templates: base.offset(at[1]).cast(),
        cut_clusters: base.offset(at[2]).cast(),
        cut_first: base.offset(at[3]).cast(),
        cut_error: base.offset(at[4]).cast(),
        cut_count: cuts.len() as u32,
        vertex_count: total,
        bound: m
            .clusters
            .iter()
            .map(|c| c.deformation_bound)
            .fold(0.0, f32::max),
        _pad: 0,
    };
    bytes[..size_of::<SkinMeshGpu>()].copy_from_slice(bytemuck::bytes_of(&record));
    block.as_mut_slice().expect("upload is mapped")[..bytes.len()].copy_from_slice(&bytes);
    Ok(block)
}

/// The vertex arrays of `copies` skinned copies of a mesh of `vertices` vertices in
/// `clusters` clusters: per copy, positions, previous positions, normals, tangents and
/// cluster boxes.
pub(crate) struct CopyArrays {
    pub memory: Allocation<u8>,
    stride: u64,
    vertices: u64,
}

impl CopyArrays {
    pub(crate) fn new(device: &Device, copies: u32, vertices: u32, clusters: u32) -> Result<Self> {
        let v = u64::from(vertices.max(1));
        let stride = (32 * v + 24 * u64::from(clusters.max(1))).next_multiple_of(256);
        let memory = device.gpu_malloc::<u8>(
            MemoryKind::DeviceLocal,
            stride * u64::from(copies.max(1)),
            "skinned vertices",
        )?;
        Ok(Self {
            memory,
            stride,
            vertices: v,
        })
    }

    /// Copy `i`'s positions, previous positions, normals, tangents and boxes.
    pub(crate) fn of(&self, i: u32) -> [GpuPtr<u8>; 5] {
        let base = self.memory.gpu.offset(self.stride * u64::from(i));
        let v = self.vertices;
        [
            base,
            base.offset(12 * v),
            base.offset(24 * v),
            base.offset(28 * v),
            base.offset(32 * v),
        ]
    }
}

/// The skinning pipelines.
pub(crate) struct SkinKernels {
    begin: PipelineId,
    select: PipelineId,
    args: PipelineId,
    skin: PipelineId,
    finish: PipelineId,
}

impl SkinKernels {
    pub(crate) fn new(
        device: &Device,
        kernels: &orr_gpu::KernelSet,
        p: &mut Pipelines,
    ) -> Result<Self> {
        let m = &skin::MODULE;
        let mut c = |entry| p.create(device, kernels, m, entry, Default::default());
        Ok(Self {
            begin: c("begin")?,
            select: c("select")?,
            args: c("args")?,
            skin: c("skin")?,
            finish: c("finish")?,
        })
    }
}

/// One implicit build of the frame: its desc and result block.
struct Op {
    desc: RtasOpDesc,
    results: Allocation<u8>,
}

impl Op {
    fn new(device: &Device, desc: RtasOpDesc, name: &str) -> Result<Self> {
        let sizes = device.rtas_sizes(&desc);
        Ok(Self {
            desc,
            results: device.gpu_malloc(MemoryKind::DeviceLocal, sizes.result_max.max(256), name)?,
        })
    }
}

/// What the skinning of a frame reports, read back once it completes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkinStats {
    /// Clusters skinned.
    pub clusters: u32,
    /// Copies traced: cluster BLAS built.
    pub traced: u32,
    /// Copies whose cut found no room left in `SceneDesc::skinned_clusters`; they are
    /// neither drawn nor traced that frame.
    pub refused: u32,
}

impl SkinStats {
    pub(crate) fn add(&mut self, o: &Self) {
        self.clusters += o.clusters;
        self.traced += o.traced;
        self.refused += o.refused;
    }
}

/// A scene's skinning memory, made the first frame it has skinned copies.
pub(crate) struct SkinFrame {
    record: Allocation<SkinFrameGpu>,
    pub work: GpuPtr<[u32; 2]>,
    counters: GpuPtr<u32>,
    instantiate: Option<Op>,
    blas: Option<Op>,
    scratch: Option<Allocation<u8>>,
    clas: GpuPtr<u64>,
    blas_addresses: GpuPtr<u64>,
    instantiate_records: GpuAddress,
    blas_records: GpuAddress,
    readback: Allocation<u32>,
    _memory: Allocation<u8>,
}

impl SkinFrame {
    pub(crate) fn new(device: &Device, clusters: u32, copies: u32, fif: u32) -> Result<Self> {
        let caps = device.caps();
        let traced = caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0;
        let n = u64::from(clusters.max(1));
        let jobs = u64::from(copies.max(1));
        let stride = |op| u64::from(device.rtas_args_stride(op));
        let mut end = 0u64;
        let mut take = |bytes: u64| {
            let start = end;
            end = (end + bytes).next_multiple_of(256);
            start
        };
        let work_at = take(8 * n);
        let counters_at = take(4 * SKIN_COUNTERS);
        let instantiate_at = take(stride(RtasOp::InstantiateTemplates) * n);
        let blas_records_at = take(stride(RtasOp::BuildClusterBlas) * jobs);
        let clas_at = take(8 * n);
        let blas_at = take(8 * jobs);
        let memory = device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, end, "skinning")?;
        let at = |o: u64| memory.gpu.offset(o);
        let mut record = device.gpu_malloc::<SkinFrameGpu>(MemoryKind::Upload, 1, "skin frame")?;
        record.as_mut_slice().expect("upload is mapped")[0] = SkinFrameGpu {
            work: at(work_at).cast(),
            instantiate: at(instantiate_at),
            blas_records: at(blas_records_at),
            clas: at(clas_at).cast(),
            blas: at(blas_at).cast(),
            counters: at(counters_at).cast(),
            work_capacity: clusters,
            traced: u32::from(traced),
        };
        let (instantiate, blas, scratch) = if traced {
            let instantiate = Op::new(
                device,
                RtasOpDesc {
                    op: RtasOp::InstantiateTemplates,
                    mode: RtasMode::Implicit,
                    flags: RtasFlags::FAST_TRACE | RtasFlags::ALLOW_DATA_ACCESS,
                    vertex_format: ClusterVertexFormat::F32x3,
                    limits: RtasLimits::Cluster(ClusterLimits {
                        max_args: clusters.max(1),
                        max_tris_per_cluster: 128,
                        max_verts_per_cluster: 256,
                        max_total_tris: clusters.max(1) * 128,
                        max_total_verts: clusters.max(1) * 256,
                        max_geometry_index: 0,
                        max_geometries_per_cluster: 1,
                        min_truncate_bits: 0,
                    }),
                },
                "skinned CLAS",
            )?;
            let blas = Op::new(
                device,
                RtasOpDesc {
                    op: RtasOp::BuildClusterBlas,
                    mode: RtasMode::Implicit,
                    flags: RtasFlags::FAST_TRACE,
                    vertex_format: ClusterVertexFormat::F32x3,
                    limits: RtasLimits::Blas(BlasLimits {
                        max_args: copies.max(1),
                        max_clas_per_blas: clusters.max(1),
                        max_total_clas: clusters.max(1),
                    }),
                },
                "skinned BLAS",
            )?;
            let bytes = device
                .rtas_sizes(&instantiate.desc)
                .scratch
                .max(device.rtas_sizes(&blas.desc).scratch);
            let scratch = device.gpu_malloc::<u8>(
                MemoryKind::DeviceLocal,
                bytes.max(256),
                "skinning scratch",
            )?;
            (Some(instantiate), Some(blas), Some(scratch))
        } else {
            (None, None, None)
        };
        Ok(Self {
            work: at(work_at).cast(),
            counters: at(counters_at).cast(),
            clas: at(clas_at).cast(),
            blas_addresses: at(blas_at).cast(),
            instantiate_records: at(instantiate_at).address(),
            blas_records: at(blas_records_at).address(),
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                SKIN_COUNTERS * u64::from(fif.max(1)),
                "skin counts readback",
            )?,
            record,
            instantiate,
            blas,
            scratch,
            _memory: memory,
        })
    }

    /// Records the frame's skinning of `entries` skinned copies of the scene at `scene`,
    /// and the copy of its counters into readback slot `slot`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &self,
        device: &Device,
        window: &UploadWindow,
        list: &mut CommandList<'_>,
        p: &Pipelines,
        k: &SkinKernels,
        scene: GpuPtr<SceneRoot>,
        focus: Option<([i32; 4], [f32; 4])>,
        error_pixels: f32,
        entries: u32,
        slot: usize,
    ) -> Result<()> {
        let (focus_cell, focus_offset) = focus.unwrap_or(([0; 4], [0.0; 4]));
        let root = window.put_root(
            device,
            &SkinRoot {
                scene,
                frame: self.record.gpu,
                focus_cell,
                focus_offset,
                error_pixels,
                entries,
                _pad: 0,
            },
        )?;
        let groups = entries.div_ceil(64).max(1);
        list.begin_pass("skin", false);
        list.dispatch(p.get(k.begin), root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(p.get(k.select), root, [groups, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(p.get(k.args), root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        let args: GpuPtr<DispatchArgs> = self.counters.offset(SKIN_ARGS).cast();
        list.dispatch_indirect(p.get(k.skin), root, args);
        if let (Some(inst), Some(blas), Some(scratch)) =
            (&self.instantiate, &self.blas, &self.scratch)
        {
            let scratch = scratch.gpu.address();
            list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
            list.rtas_execute(
                &inst.desc,
                &RtasIo {
                    args: self.instantiate_records,
                    count: self.counters.offset(SKIN_WORK),
                    dst_implicit: inst.results.gpu.address(),
                    dst_addresses: self.clas,
                    dst_sizes: GpuPtr::NULL,
                    scratch,
                    src: GpuAddress(0),
                },
            );
            list.barrier(Stages::AS_BUILD, Stages::AS_BUILD, Hazards::ACCEL_STRUCT);
            list.rtas_execute(
                &blas.desc,
                &RtasIo {
                    args: self.blas_records,
                    count: self.counters.offset(SKIN_JOBS),
                    dst_implicit: blas.results.gpu.address(),
                    dst_addresses: self.blas_addresses,
                    dst_sizes: GpuPtr::NULL,
                    scratch,
                    src: GpuAddress(0),
                },
            );
            list.barrier(
                Stages::AS_BUILD,
                Stages::COMPUTE | Stages::AS_BUILD,
                Hazards::ACCEL_STRUCT,
            );
        } else {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        }
        list.dispatch(p.get(k.finish), root, [groups, 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::COMPUTE | Stages::TRANSFER,
            Hazards::empty(),
        );
        list.copy(
            self.readback.gpu.offset(SKIN_COUNTERS * slot as u64).cast(),
            self.counters.cast(),
            4 * SKIN_COUNTERS,
        );
        list.end_pass();
        list.barrier(
            Stages::COMPUTE | Stages::TRANSFER,
            Stages::ALL,
            Hazards::empty(),
        );
        Ok(())
    }

    /// The figures of readback slot `slot`.
    pub(crate) fn figures(&self, slot: usize) -> SkinStats {
        let Some(c) = self.readback.as_slice() else {
            return SkinStats::default();
        };
        let c = &c[slot * SKIN_COUNTERS as usize..];
        SkinStats {
            clusters: c[SKIN_WORK as usize],
            traced: c[SKIN_JOBS as usize],
            refused: c[SKIN_REFUSED as usize],
        }
    }
}
