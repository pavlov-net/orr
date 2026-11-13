//! The `rtas` records: every store writes exactly the Vulkan struct ash packs from the same
//! values, and partition translations are exact across a 32 km world.

use ash::vk;
use gpu_suite::shaders::rtas_store::{self, Translation};
use gpu_suite::{assert_clean, device, pipeline, readback, render, run, upload};
use orr_gpu::rtas::consts::{INSTANCE_CULL_DISABLE, INSTANCE_EXPLICIT_AABB, INSTANCE_FORCE_OPAQUE};
use orr_gpu::rtas::{
    ClasBuild, ClusterBlas, Instantiate, Move, PtlasCommand, PtlasTranslate, PtlasUpdate,
    PtlasWrite, TemplateBuild, geometry_index_and_flags,
};
use orr_gpu::{
    ClassicInstance, GpuAddress, Hazards, InstanceFlags, PtlasArgs, RtasOp, Stages,
    partition_translation,
};

const RECORDS: usize = 3;

fn bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data Vulkan structs viewed as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) }
}

/// A distinct 64-bit address per (record, field).
fn addr(i: usize, field: u64) -> u64 {
    0x0000_7f00_0000_0000 | (i as u64) << 24 | field << 8
}

fn clas_record(i: usize) -> ClasBuild {
    ClasBuild {
        cluster_id: 1000 + i as u32 * 77,
        flags: (i % 2) as u32,
        tri_count: [1, 128, 256][i],
        vertex_count: [3, 200, 256][i],
        truncate_bits: [0, 8, 23][i],
        index_format: [1, 2, 4][i],
        base_geometry_and_flags: geometry_index_and_flags(0xab_cdef - i as u32, [1, 6, 7][i]),
        omm_index_format: [4, 2, 1][i],
        vertices: addr(i, 1),
        indices: addr(i, 2),
        triangle_flags: if i == 0 { 0 } else { addr(i, 3) },
        omm_array: if i == 1 { 0 } else { addr(i, 4) },
        omm_indices: if i == 1 { 0 } else { addr(i, 5) },
    }
}

fn vk_clas(c: &ClasBuild) -> vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV {
    let omm = c.omm_indices != 0;
    vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV {
        cluster_id: c.cluster_id,
        cluster_flags: vk::ClusterAccelerationStructureClusterFlagsNV::from_raw(c.flags),
        triangle_cluster_info_packed: vk::Packed9_9_6_4_4::new(
            c.tri_count,
            c.vertex_count,
            c.truncate_bits,
            c.index_format,
            if omm { c.omm_index_format } else { 1 },
        ),
        base_geometry_index_and_geometry_flags:
            vk::ClusterAccelerationStructureGeometryIndexAndGeometryFlagsNV {
                geometry_index_and_geometry_flags: vk::Packed24_5_3::new(
                    c.base_geometry_and_flags & 0xff_ffff,
                    (c.base_geometry_and_flags >> 29) as u8,
                ),
            },
        index_buffer_stride: c.index_format as u16,
        vertex_buffer_stride: 12,
        geometry_index_and_flags_buffer_stride: if c.triangle_flags != 0 { 4 } else { 0 },
        opacity_micromap_index_buffer_stride: if omm { c.omm_index_format as u16 } else { 0 },
        index_buffer: c.indices,
        vertex_buffer: c.vertices,
        geometry_index_and_flags_buffer: c.triangle_flags,
        opacity_micromap_array: c.omm_array,
        opacity_micromap_index_buffer: c.omm_indices,
    }
}

fn transform(i: usize) -> [[f32; 4]; 3] {
    let f = i as f32;
    [
        [1.0 + f, -0.5, 0.25, 1e6 + f],
        [0.125, -2.0, 3.5, -7.75],
        [f32::MIN_POSITIVE, -0.0, 1e-7, 42.0 - f],
    ]
}

fn flat(m: [[f32; 4]; 3]) -> [f32; 12] {
    orr_shaders::bytemuck::cast(m)
}

fn write(i: usize) -> PtlasWrite {
    PtlasWrite {
        transform: transform(i),
        aabb_min: [-1.0 - i as f32, -2.0, -3.0],
        aabb_max: [4.0, 5.0 + i as f32, 6.0],
        blas: addr(i, 6),
        custom_index: 0xff_fff0 + i as u32,
        mask: [0xff, 0x01, 0x80][i],
        flags: [
            INSTANCE_CULL_DISABLE,
            INSTANCE_EXPLICIT_AABB | INSTANCE_FORCE_OPAQUE,
            0,
        ][i],
        instance_index: 500_000 + i as u32,
        partition: [0, 2047, u32::MAX][i],
        hit_group: [0, 0xab_cdef, 0xff_ffff][i],
    }
}

#[test]
fn stores_match_vulkan_structs() {
    let d = device();
    let clas: Vec<ClasBuild> = Vec::from_fn(RECORDS, clas_record);
    let templates: Vec<TemplateBuild> = Vec::from_fn(RECORDS, |i| TemplateBuild {
        cluster: clas_record(i),
        bounds: if i == 0 { 0 } else { addr(i, 7) },
    });
    let instantiate: Vec<Instantiate> = Vec::from_fn(RECORDS, |i| Instantiate {
        cluster_id_offset: 0xffff_0000 + i as u32,
        geometry_index_offset: 0xff_ff00 + i as u32,
        template_address: addr(i, 8),
        vertices: addr(i, 9),
    });
    let blas: Vec<ClusterBlas> = Vec::from_fn(RECORDS, |i| ClusterBlas {
        clas_addresses: addr(i, 10),
        clas_count: [1, 4096, 65535][i],
        pad: 0,
    });
    let moves: Vec<Move> = Vec::from_fn(RECORDS, |i| Move { src: addr(i, 11) });
    let writes: Vec<PtlasWrite> = Vec::from_fn(RECORDS, write);
    let updates: Vec<PtlasUpdate> = Vec::from_fn(RECORDS, |i| PtlasUpdate {
        blas: if i == 2 { 0 } else { addr(i, 12) },
        instance_index: 7 * i as u32,
        hit_group: [0xff_ffff, 0, 0x12_3456][i],
    });
    let translates: Vec<PtlasTranslate> = Vec::from_fn(RECORDS, |i| PtlasTranslate {
        translation: [-32768.0 * i as f32, 0.5, 16384.0],
        partition: [0, 1, 2047][i],
    });
    let commands: Vec<PtlasCommand> = Vec::from_fn(RECORDS, |i| PtlasCommand {
        args: addr(i, 13),
        op: i as u32,
        count: [1, 1000, 4096][i],
    });
    let classic: Vec<ClassicInstance> = Vec::from_fn(RECORDS, |i| {
        ClassicInstance::new(
            transform(i),
            0xff_ff00 + i as u32,
            [0xff, 1, 0x80][i],
            [
                InstanceFlags::CULL_DISABLE,
                InstanceFlags::FORCE_OPAQUE | InstanceFlags::FLIP_FACING,
                InstanceFlags::FORCE_NO_OPAQUE,
            ][i],
            GpuAddress(addr(i, 14)),
        )
        .with_hit_group([0x80_0001, 0, 0xff_ffff][i])
    });

    let cells: Vec<Translation> = (0..64)
        .map(|i: i32| Translation {
            cell: [i * 977 - 31_000, -i * 13, (i * i) % 4097],
            cell_log2: (i % 11) as u32,
            origin_cell: [-(i * 31) % 513, i * 7, -i],
            pad: 0,
        })
        .map(|mut t| {
            // Keep every offset within the 32 km range.
            for k in 0..3 {
                let limit = 32_768 >> t.cell_log2;
                t.cell[k] = t.cell[k].clamp(-limit, limit);
                t.origin_cell[k] = t.origin_cell[k].clamp(-limit, limit);
            }
            t
        })
        .collect();

    // The output arrays, checking each stride orr_gpu reports against the struct size.
    let out = |op: Option<RtasOp>, args: Option<PtlasArgs>, size: usize, name: &str| {
        let stride = match (op, args) {
            (Some(op), _) => d.rtas_args_stride(op) as usize,
            (_, Some(a)) => d.ptlas_args_stride(a) as usize,
            _ => size,
        };
        assert_eq!(stride, size, "{name} stride");
        readback::<u8>((stride * RECORDS) as u64, name)
    };
    let o_clas = out(
        Some(RtasOp::BuildClas),
        None,
        size_of::<vk::ClusterAccelerationStructureBuildTriangleClusterInfoNV>(),
        "clas",
    );
    let o_templates = out(Some(RtasOp::BuildTemplates), None, 72, "templates");
    let o_inst = out(Some(RtasOp::InstantiateTemplates), None, 32, "instantiate");
    let o_blas = out(Some(RtasOp::BuildClusterBlas), None, 16, "blas");
    let o_moves = out(Some(RtasOp::Move), None, 8, "moves");
    let o_writes = out(None, Some(PtlasArgs::Write), 104, "writes");
    let o_updates = out(None, Some(PtlasArgs::Update), 16, "updates");
    let o_translates = out(None, Some(PtlasArgs::Translate), 16, "translates");
    let o_commands = out(Some(RtasOp::Ptlas), None, 24, "commands");
    let o_classic = out(None, None, 64, "classic");
    let translations = readback::<[f32; 3]>(cells.len() as u64, "translations");

    let inputs = (
        upload(&clas, "clas in"),
        upload(&templates, "templates in"),
        upload(&instantiate, "instantiate in"),
        upload(&blas, "blas in"),
        upload(&moves, "moves in"),
        upload(&writes, "writes in"),
        upload(&updates, "updates in"),
        upload(&translates, "translates in"),
        upload(&commands, "commands in"),
        upload(&classic, "classic in"),
        upload(&cells, "cells in"),
    );
    let root = upload(
        &[rtas_store::Root {
            clas: inputs.0.gpu(),
            templates: inputs.1.gpu(),
            instantiate: inputs.2.gpu(),
            blas: inputs.3.gpu(),
            moves: inputs.4.gpu(),
            writes: inputs.5.gpu(),
            updates: inputs.6.gpu(),
            translates: inputs.7.gpu(),
            commands: inputs.8.gpu(),
            classic: inputs.9.gpu(),
            cells: inputs.10.gpu(),
            translations: translations.gpu(),
            out_clas: o_clas.gpu().address().0,
            out_templates: o_templates.gpu().address().0,
            out_instantiate: o_inst.gpu().address().0,
            out_blas: o_blas.gpu().address().0,
            out_moves: o_moves.gpu().address().0,
            out_writes: o_writes.gpu().address().0,
            out_updates: o_updates.gpu().address().0,
            out_translates: o_translates.gpu().address().0,
            out_commands: o_commands.gpu().address().0,
            out_classic: o_classic.gpu().address().0,
            records: RECORDS as u32,
            cell_count: cells.len() as u32,
            pad: 0,
        }],
        "store root",
    );
    let p = pipeline(&rtas_store::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.dispatch(&p, root.gpu(), [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    run(list);

    let vk_clas_all: Vec<_> = clas.iter().map(vk_clas).collect();
    assert_eq!(o_clas.as_slice().unwrap(), bytes(&vk_clas_all), "ClasBuild");

    let vk_templates: Vec<_> = templates
        .iter()
        .map(|t| {
            let c = vk_clas(&t.cluster);
            vk::ClusterAccelerationStructureBuildTriangleClusterTemplateInfoNV {
                cluster_id: c.cluster_id,
                cluster_flags: c.cluster_flags,
                triangle_cluster_info_packed: c.triangle_cluster_info_packed,
                base_geometry_index_and_geometry_flags: c.base_geometry_index_and_geometry_flags,
                index_buffer_stride: c.index_buffer_stride,
                vertex_buffer_stride: c.vertex_buffer_stride,
                geometry_index_and_flags_buffer_stride: c.geometry_index_and_flags_buffer_stride,
                opacity_micromap_index_buffer_stride: c.opacity_micromap_index_buffer_stride,
                index_buffer: c.index_buffer,
                vertex_buffer: c.vertex_buffer,
                geometry_index_and_flags_buffer: c.geometry_index_and_flags_buffer,
                opacity_micromap_array: c.opacity_micromap_array,
                opacity_micromap_index_buffer: c.opacity_micromap_index_buffer,
                instantiation_bounding_box_limit: t.bounds,
            }
        })
        .collect();
    assert_eq!(
        o_templates.as_slice().unwrap(),
        bytes(&vk_templates),
        "TemplateBuild"
    );

    let vk_inst: Vec<_> = instantiate
        .iter()
        .map(
            |v| vk::ClusterAccelerationStructureInstantiateClusterInfoNV {
                cluster_id_offset: v.cluster_id_offset,
                geometry_index_offset_and_reserved: vk::Packed24_8::new(v.geometry_index_offset, 0),
                cluster_template_address: v.template_address,
                vertex_buffer: vk::StridedDeviceAddressNV {
                    start_address: v.vertices,
                    stride_in_bytes: 12,
                },
            },
        )
        .collect();
    assert_eq!(o_inst.as_slice().unwrap(), bytes(&vk_inst), "Instantiate");

    let vk_blas: Vec<_> = blas
        .iter()
        .map(
            |v| vk::ClusterAccelerationStructureBuildClustersBottomLevelInfoNV {
                cluster_references_count: v.clas_count,
                cluster_references_stride: 8,
                cluster_references: v.clas_addresses,
            },
        )
        .collect();
    assert_eq!(o_blas.as_slice().unwrap(), bytes(&vk_blas), "ClusterBlas");

    let vk_moves: Vec<_> = moves
        .iter()
        .map(|v| vk::ClusterAccelerationStructureMoveObjectsInfoNV {
            src_acceleration_structure: v.src,
        })
        .collect();
    assert_eq!(o_moves.as_slice().unwrap(), bytes(&vk_moves), "Move");

    let vk_writes: Vec<_> = writes
        .iter()
        .map(
            |v| vk::PartitionedAccelerationStructureWriteInstanceDataNV {
                transform: vk::TransformMatrixKHR {
                    matrix: flat(v.transform),
                },
                explicit_aabb: [
                    v.aabb_min[0],
                    v.aabb_min[1],
                    v.aabb_min[2],
                    v.aabb_max[0],
                    v.aabb_max[1],
                    v.aabb_max[2],
                ],
                instance_id: v.custom_index,
                instance_mask: v.mask,
                instance_contribution_to_hit_group_index: v.hit_group,
                instance_flags: vk::PartitionedAccelerationStructureInstanceFlagsNV::from_raw(
                    v.flags,
                ),
                instance_index: v.instance_index,
                partition_index: v.partition,
                acceleration_structure: v.blas,
            },
        )
        .collect();
    assert_eq!(
        o_writes.as_slice().unwrap(),
        bytes(&vk_writes),
        "PtlasWrite"
    );

    let vk_updates: Vec<_> = updates
        .iter()
        .map(
            |v| vk::PartitionedAccelerationStructureUpdateInstanceDataNV {
                instance_index: v.instance_index,
                instance_contribution_to_hit_group_index: v.hit_group,
                acceleration_structure: v.blas,
            },
        )
        .collect();
    assert_eq!(
        o_updates.as_slice().unwrap(),
        bytes(&vk_updates),
        "PtlasUpdate"
    );

    let vk_translates: Vec<_> = translates
        .iter()
        .map(
            |v| vk::PartitionedAccelerationStructureWritePartitionTranslationDataNV {
                partition_index: v.partition,
                partition_translation: v.translation,
            },
        )
        .collect();
    assert_eq!(
        o_translates.as_slice().unwrap(),
        bytes(&vk_translates),
        "PtlasTranslate"
    );

    let vk_commands: Vec<_> = commands
        .iter()
        .map(
            |v| vk::BuildPartitionedAccelerationStructureIndirectCommandNV {
                op_type: vk::PartitionedAccelerationStructureOpTypeNV::from_raw(v.op as i32),
                arg_count: v.count,
                arg_data: vk::StridedDeviceAddressNV {
                    start_address: v.args,
                    stride_in_bytes: u64::from(d.ptlas_args_stride(
                        [PtlasArgs::Write, PtlasArgs::Update, PtlasArgs::Translate][v.op as usize],
                    )),
                },
            },
        )
        .collect();
    assert_eq!(
        o_commands.as_slice().unwrap(),
        bytes(&vk_commands),
        "PtlasCommand"
    );

    let vk_classic: Vec<_> = classic
        .iter()
        .map(|v| vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: v.transform,
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(
                v.custom_index_and_mask & 0xff_ffff,
                (v.custom_index_and_mask >> 24) as u8,
            ),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                v.sbt_offset_and_flags & 0xff_ffff,
                (v.sbt_offset_and_flags >> 24) as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: v.blas,
            },
        })
        .collect();
    assert_eq!(
        o_classic.as_slice().unwrap(),
        bytes(&vk_classic),
        "ClassicInstance"
    );
    assert_eq!(
        classic[1].sbt_offset_and_flags >> 24,
        u32::from((InstanceFlags::FORCE_OPAQUE | InstanceFlags::FLIP_FACING).bits())
    );

    for (t, got) in cells.iter().zip(translations.as_slice().unwrap()) {
        let want = partition_translation(t.cell, t.origin_cell, t.cell_log2);
        assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits), "{t:?}");
    }
    assert_clean();
}
