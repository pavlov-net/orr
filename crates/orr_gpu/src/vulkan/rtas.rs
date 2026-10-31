//! Cluster acceleration structures (CLAS, templates, cluster BLAS, moves) and partitioned
//! TLAS builds on `VK_NV_cluster_acceleration_structure` and
//! `VK_NV_partitioned_acceleration_structure`.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::accel::vk_build_flags;
use super::command::CommandList;
use super::device::{Device, Shared};
use super::garbage::RawBuffer;
use super::memory::MemoryKind;
use super::pipeline::ShaderStage;
use crate::{
    ClusterCaps, ClusterVertexFormat, Error, GpuAddress, GpuPtr, Hazards, MoveKind, PtlasArgs,
    RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, RtasSizes, SpecData, Stages,
};

const PTLAS_COMMAND_STRIDE: u32 =
    size_of::<vk::BuildPartitionedAccelerationStructureIndirectCommandNV>() as u32;

/// Alignment of a PTLAS.
const PTLAS_ALIGN: u32 = 256;

/// Scratch bytes a move takes per object at least: 621.01 reports none but writes 24 per
/// moved object plus 16, past the end of a smaller block into whatever follows it.
const MOVE_SCRATCH_PER_OBJECT: u64 = 32;

pub(crate) fn cluster_caps(
    p: &vk::PhysicalDeviceClusterAccelerationStructurePropertiesNV<'_>,
    pt: &vk::PhysicalDevicePartitionedAccelerationStructurePropertiesNV<'_>,
    omm: bool,
) -> ClusterCaps {
    ClusterCaps {
        formats: &[ClusterVertexFormat::F32x3],
        max_tris: p.max_triangles_per_cluster,
        max_verts: p.max_vertices_per_cluster,
        align_clas: p.cluster_byte_alignment,
        align_template: p.cluster_template_byte_alignment,
        align_blas: p.cluster_bottom_level_byte_alignment,
        align_scratch: p.cluster_scratch_byte_alignment,
        align_bounds: p.cluster_template_bounds_byte_alignment,
        max_geometry_index: p.max_cluster_geometry_index,
        omm_in_clusters: omm,
        max_partitions: pt.max_partition_count,
    }
}

fn vk_format(f: ClusterVertexFormat) -> vk::Format {
    match f {
        ClusterVertexFormat::F32x3 => vk::Format::R32G32B32_SFLOAT,
    }
}

fn vk_mode(m: RtasMode) -> vk::ClusterAccelerationStructureOpModeNV {
    match m {
        RtasMode::Implicit => vk::ClusterAccelerationStructureOpModeNV::IMPLICIT_DESTINATIONS,
        RtasMode::Explicit => vk::ClusterAccelerationStructureOpModeNV::EXPLICIT_DESTINATIONS,
        RtasMode::GetSizes => vk::ClusterAccelerationStructureOpModeNV::COMPUTE_SIZES,
    }
}

fn vk_op(op: RtasOp) -> vk::ClusterAccelerationStructureOpTypeNV {
    use vk::ClusterAccelerationStructureOpTypeNV as T;
    match op {
        RtasOp::BuildClas => T::BUILD_TRIANGLE_CLUSTER,
        RtasOp::BuildTemplates => T::BUILD_TRIANGLE_CLUSTER_TEMPLATE,
        RtasOp::InstantiateTemplates => T::INSTANTIATE_TRIANGLE_CLUSTER,
        RtasOp::BuildClusterBlas => T::BUILD_CLUSTERS_BOTTOM_LEVEL,
        RtasOp::Move => T::MOVE_OBJECTS,
        RtasOp::Ptlas => unreachable!("a PTLAS build is not a cluster op"),
    }
}

fn args_stride(op: RtasOp) -> u32 {
    (match op {
        RtasOp::BuildClas => {
            size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV>()
        }
        RtasOp::BuildTemplates => {
            size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV>()
        }
        RtasOp::InstantiateTemplates => {
            size_of::<vk::ClusterAccelerationStructureInstantiateClusterInfoNV>()
        }
        RtasOp::BuildClusterBlas => {
            size_of::<vk::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV>()
        }
        RtasOp::Move => size_of::<vk::ClusterAccelerationStructureMoveObjectsInfoNV>(),
        RtasOp::Ptlas => PTLAS_COMMAND_STRIDE as usize,
    }) as u32
}

/// The capacity limits checked against the op; panics on a mismatch.
fn max_args(d: &RtasOpDesc) -> u32 {
    match (d.op, d.limits) {
        (
            RtasOp::BuildClas | RtasOp::BuildTemplates | RtasOp::InstantiateTemplates,
            RtasLimits::Cluster(l),
        ) => l.max_args,
        (RtasOp::BuildClusterBlas, RtasLimits::Blas(l)) => l.max_args,
        (RtasOp::Move, RtasLimits::Move(l)) => l.max_args,
        (RtasOp::Ptlas, RtasLimits::Ptlas(l)) => l.max_ops,
        (op, limits) => panic!("{op:?} takes other limits than {limits:?}"),
    }
}

/// Calls `f` with the Vulkan input of a cluster op.
fn with_cluster_input<R>(
    d: &RtasOpDesc,
    f: impl FnOnce(&vk::ClusterAccelerationStructureInputInfoNV<'_>) -> R,
) -> R {
    let mut triangles;
    let mut blas;
    let mut moves;
    let input = match d.limits {
        RtasLimits::Cluster(l) => {
            triangles = vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
                .vertex_format(vk_format(d.vertex_format))
                .max_geometry_index_value(l.max_geometry_index)
                .max_cluster_unique_geometry_count(l.max_geometries_per_cluster)
                .max_cluster_triangle_count(l.max_tris_per_cluster)
                .max_cluster_vertex_count(l.max_verts_per_cluster)
                .max_total_triangle_count(l.max_total_tris)
                .max_total_vertex_count(l.max_total_verts)
                .min_position_truncate_bit_count(l.min_truncate_bits);
            vk::ClusterAccelerationStructureOpInputNV {
                p_triangle_clusters: &mut triangles,
            }
        }
        RtasLimits::Blas(l) => {
            blas = vk::ClusterAccelerationStructureClustersBottomLevelInputNV::default()
                .max_total_cluster_count(l.max_total_clas)
                .max_cluster_count_per_acceleration_structure(l.max_clas_per_blas);
            vk::ClusterAccelerationStructureOpInputNV {
                p_clusters_bottom_level: &mut blas,
            }
        }
        RtasLimits::Move(l) => {
            moves = vk::ClusterAccelerationStructureMoveObjectsInputNV::default()
                .ty(match l.kind {
                    MoveKind::Clas => vk::ClusterAccelerationStructureTypeNV::TRIANGLE_CLUSTER,
                    MoveKind::Template => {
                        vk::ClusterAccelerationStructureTypeNV::TRIANGLE_CLUSTER_TEMPLATE
                    }
                    MoveKind::ClusterBlas => {
                        vk::ClusterAccelerationStructureTypeNV::CLUSTERS_BOTTOM_LEVEL
                    }
                })
                .no_move_overlap(l.no_overlap)
                .max_moved_bytes(l.max_bytes);
            vk::ClusterAccelerationStructureOpInputNV {
                p_move_objects: &mut moves,
            }
        }
        RtasLimits::Ptlas(_) => unreachable!("a PTLAS build is not a cluster op"),
    };
    let info = vk::ClusterAccelerationStructureInputInfoNV::default()
        .max_acceleration_structure_count(max_args(d))
        .flags(vk_build_flags(d.flags))
        .op_type(vk_op(d.op))
        .op_mode(vk_mode(d.mode))
        .op_input(input);
    f(&info)
}

/// Calls `f` with the Vulkan input of a PTLAS build.
fn with_ptlas_input<R>(
    d: &RtasOpDesc,
    f: impl FnOnce(&vk::PartitionedAccelerationStructureInstancesInputNV<'_>) -> R,
) -> R {
    let RtasLimits::Ptlas(l) = d.limits else {
        panic!("{:?} takes other limits than {:?}", d.op, d.limits);
    };
    let mut translation =
        vk::PartitionedAccelerationStructureFlagsNV::default().enable_partition_translation(true);
    let mut input = vk::PartitionedAccelerationStructureInstancesInputNV::default()
        .flags(vk_build_flags(d.flags))
        .instance_count(l.instances)
        .max_instance_per_partition_count(l.max_per_partition)
        .partition_count(l.partitions)
        .max_instance_in_global_partition_count(l.global_instances);
    if l.translation {
        input = input.push(&mut translation);
    }
    f(&input)
}

impl Shared {
    fn check_desc(&self, d: &RtasOpDesc) {
        let c = &self.caps.cluster;
        max_args(d);
        assert!(
            !d.flags.contains(RtasFlags::ALLOW_OMM) || c.omm_in_clusters,
            "RtasFlags::ALLOW_OMM needs ClusterCaps::omm_in_clusters"
        );
        match d.limits {
            RtasLimits::Cluster(l) => assert!(
                l.max_tris_per_cluster <= c.max_tris
                    && l.max_verts_per_cluster <= c.max_verts
                    && l.max_geometry_index <= c.max_geometry_index,
                "cluster limits {l:?} beyond the device's {c:?}"
            ),
            RtasLimits::Ptlas(l) => assert!(
                l.partitions <= c.max_partitions && l.max_ops <= 3,
                "PTLAS limits {l:?}: at most {} partitions and 3 commands",
                c.max_partitions
            ),
            _ => {}
        }
    }

    /// Alignment of the implicit results of `op`.
    fn result_align(&self, op: RtasOp) -> u32 {
        let c = &self.caps.cluster;
        match op {
            RtasOp::BuildTemplates => c.align_template,
            RtasOp::BuildClusterBlas => c.align_blas,
            RtasOp::Ptlas => PTLAS_ALIGN,
            RtasOp::Move => c.align_clas.max(c.align_template).max(c.align_blas),
            RtasOp::BuildClas | RtasOp::InstantiateTemplates => c.align_clas,
        }
    }
}

impl Device {
    /// The memory `d` needs at most: the implicit result (or the PTLAS) and scratch. One
    /// query sizes a whole batch.
    pub fn rtas_sizes(&self, d: &RtasOpDesc) -> RtasSizes {
        let shared = &self.shared;
        shared.check_desc(d);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: valid inputs; both extensions are required at device creation.
        unsafe {
            if d.op == RtasOp::Ptlas {
                with_ptlas_input(d, |input| {
                    shared
                        .ptlas_fn
                        .get_partitioned_acceleration_structures_build_sizes(input, &mut sizes);
                });
            } else {
                with_cluster_input(d, |input| {
                    shared
                        .cluster_fn
                        .get_cluster_acceleration_structure_build_sizes(input, &mut sizes);
                });
            }
        }
        let mut scratch = sizes.build_scratch_size;
        if let RtasLimits::Move(l) = d.limits {
            scratch = scratch.max(MOVE_SCRATCH_PER_OBJECT * u64::from(l.max_args) + 256);
        }
        RtasSizes {
            result_max: sizes.acceleration_structure_size,
            scratch,
        }
    }

    /// Bytes per argument record of `op` in the backend's layout: the stride `rtas` `store`
    /// writes at. For [`RtasOp::Ptlas`], the command record.
    pub fn rtas_args_stride(&self, op: RtasOp) -> u32 {
        args_stride(op)
    }

    /// Bytes per record of the arrays PTLAS commands point at.
    pub fn ptlas_args_stride(&self, args: PtlasArgs) -> u32 {
        match args {
            PtlasArgs::Write => {
                size_of::<vk::PartitionedAccelerationStructureWriteInstanceDataNV>() as u32
            }
            PtlasArgs::Update => {
                size_of::<vk::PartitionedAccelerationStructureUpdateInstanceDataNV>() as u32
            }
            PtlasArgs::Translate => size_of::<
                vk::PartitionedAccelerationStructureWritePartitionTranslationDataNV,
            >() as u32,
        }
    }
}

impl CommandList<'_> {
    /// Records `d` over the records and count kernels wrote at `io`, then makes the results
    /// visible to traces, builds and kernels. Order the argument writes before it with a
    /// barrier into [`Stages::AS_BUILD`]. In debug builds a PTLAS build whose commands
    /// exceed `max_ops`, repeat an op type or name an unknown op is rejected on the GPU: it
    /// builds nothing and counts in [`Device::validation_messages`].
    pub fn rtas_execute(&mut self, d: &RtasOpDesc, io: &RtasIo) {
        let shared = std::sync::Arc::clone(&self.queue().shared);
        shared.check_desc(d);
        let scratch_align = u64::from(shared.caps.cluster.align_scratch);
        assert!(
            io.scratch.0.is_multiple_of(scratch_align),
            "scratch {:#x} is not aligned to {scratch_align}",
            io.scratch.0
        );
        let align = u64::from(shared.result_align(d.op));
        let builds_implicit = d.op == RtasOp::Ptlas || d.mode == RtasMode::Implicit;
        assert!(
            !builds_implicit || io.dst_implicit.0.is_multiple_of(align),
            "{:?} result {:#x} is not aligned to {align}",
            d.op,
            io.dst_implicit.0
        );
        if d.op == RtasOp::Ptlas {
            if cfg!(debug_assertions) {
                self.check_ptlas_commands(&shared, d, io);
            }
            with_ptlas_input(d, |input| {
                let info = vk::BuildPartitionedAccelerationStructureInfoNV {
                    input: *input,
                    src_acceleration_structure_data: io.src.0,
                    dst_acceleration_structure_data: io.dst_implicit.0,
                    scratch_data: io.scratch.0,
                    src_infos: io.args.0,
                    src_infos_count: io.count.address().0,
                    ..Default::default()
                };
                // SAFETY: recording; every address is live device memory the caller owns.
                unsafe {
                    shared
                        .ptlas_fn
                        .cmd_build_partitioned_acceleration_structures(self.cmd(), &info);
                }
            });
        } else {
            let n = u64::from(max_args(d));
            let stride = u64::from(args_stride(d.op));
            let region = |address: u64, stride: u64| {
                if address == 0 {
                    vk::StridedDeviceAddressRegionKHR::default()
                } else {
                    vk::StridedDeviceAddressRegionKHR {
                        device_address: address,
                        stride,
                        size: stride * n,
                    }
                }
            };
            with_cluster_input(d, |input| {
                let commands = vk::ClusterAccelerationStructureCommandsInfoNV::default()
                    .input(*input)
                    .dst_implicit_data(if d.mode == RtasMode::Implicit {
                        io.dst_implicit.0
                    } else {
                        0
                    })
                    .scratch_data(io.scratch.0)
                    .dst_addresses_array(region(io.dst_addresses.address().0, 8))
                    .dst_sizes_array(region(io.dst_sizes.address().0, 4))
                    .src_infos_array(region(io.args.0, stride))
                    .src_infos_count(io.count.address().0);
                // SAFETY: recording; every address is live device memory the caller owns.
                unsafe {
                    shared
                        .cluster_fn
                        .cmd_build_cluster_acceleration_structure_indirect(self.cmd(), &commands);
                }
            });
        }
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    /// Runs the debug check kernel over a PTLAS build's commands.
    fn check_ptlas_commands(&mut self, shared: &Shared, d: &RtasOpDesc, io: &RtasIo) {
        let RtasLimits::Ptlas(l) = d.limits else {
            unreachable!()
        };
        let check = RtasCheck::get(shared);
        let root = crate::shaders::vulkan::rtas_check::Root {
            // SAFETY: the caller's command records start at `args`.
            commands: unsafe { GpuPtr::from_address(io.args) },
            count: io.count,
            failures: check.failures_ptr(),
            max_ops: l.max_ops,
            stride_words: PTLAS_COMMAND_STRIDE / 4,
        };
        let root = self.keep_upload(bytemuck::bytes_of(&root), "orr PTLAS check root");
        self.barrier(Stages::ALL, Stages::COMPUTE, Hazards::empty());
        self.dispatch_raw(check.pipeline, root, [1, 1, 1]);
        self.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
    }
}

/// The debug PTLAS check: its pipeline and the count of rejected builds the GPU adds to.
/// Destroyed with the device.
pub(crate) struct RtasCheck {
    pub pipeline: vk::Pipeline,
    pub failures: RawBuffer,
    counter: NonNull<u32>,
    reported: AtomicU32,
}

// SAFETY: the counter is mapped memory only read here; the GPU writes it with atomics.
unsafe impl Send for RtasCheck {}
// SAFETY: as above.
unsafe impl Sync for RtasCheck {}

impl RtasCheck {
    fn new(shared: &Shared) -> Result<Self, Error> {
        let stage = ShaderStage::new(
            &crate::shaders::vulkan::rtas_check::MODULE,
            "main",
            SpecData::default(),
        )?;
        let pipeline = shared.create_compute(&stage, "orr PTLAS check")?;
        let made = shared.create_buffer(
            MemoryKind::Readback,
            4,
            shared.buffer_usage,
            4,
            "orr PTLAS check failures",
        );
        let (failures, mapped) = match made {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: the pipeline was never used.
                unsafe { shared.device.destroy_pipeline(pipeline, None) };
                return Err(e);
            }
        };
        let counter = mapped.expect("Readback memory is mapped").cast::<u32>();
        // SAFETY: the mapping holds four bytes nothing else uses yet.
        unsafe { counter.as_ptr().write_volatile(0) };
        Ok(Self {
            pipeline,
            failures,
            counter,
            reported: AtomicU32::new(0),
        })
    }

    /// The device's check, created on first use.
    fn get(shared: &Shared) -> &Self {
        shared.rtas_check.get_or_init(|| {
            Self::new(shared).unwrap_or_else(|e| panic!("creating the debug PTLAS check: {e}"))
        })
    }

    fn failures_ptr(&self) -> GpuPtr<u32> {
        // SAFETY: the buffer holds one u32.
        unsafe { GpuPtr::from_address(GpuAddress(self.failures.address)) }
    }

    /// Rejected builds so far; logs the ones not reported before.
    pub(crate) fn failures(&self) -> u32 {
        // SAFETY: the mapping lives as long as `self`; the GPU only adds to it.
        let n = unsafe { self.counter.as_ptr().read_volatile() };
        let before = self.reported.swap(n, Ordering::Relaxed);
        if n > before {
            log::error!(
                "{} PTLAS build(s) rejected: more commands than PtlasLimits::max_ops, an unknown op or two commands of one op type",
                n - before
            );
        }
        n
    }
}
