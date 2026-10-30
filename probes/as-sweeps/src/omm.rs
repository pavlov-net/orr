//! Opacity micromaps referenced from CLAS, built through whichever OMM extension the device
//! enabled. With VK_KHR_opacity_micromap the micromap is a `VkAccelerationStructureKHR` of type
//! OPACITY_MICROMAP created at a device address (VK_KHR_device_address_commands) and built
//! with vkCmdBuildAccelerationStructuresKHR; the CLAS takes that address.

use super::*;

pub(crate) struct Micromap {
    /// Address a CLAS passes as `opacityMicromapArray`.
    pub addr: u64,
    pub size: u64,
    pub scratch: u64,
    /// KHR only: the address vkGetAccelerationStructureDeviceAddressKHR returns.
    pub queried_addr: Option<u64>,
}

/// Records a micromap build into `cmd` (EXT or KHR, as the device enabled) from 4-state or
/// 2-state data. `tris` index `data`; `usage` lists (count, level, format).
pub(crate) unsafe fn build_micromap(
    ctx: &Ctx,
    cmd: vk::CommandBuffer,
    case: &str,
    data: &[u8],
    tris: &[vk::MicromapTriangleKHR],
    usage: &[vk::MicromapUsageKHR],
) -> Micromap {
    let gpu = &ctx.gpu;
    let data = ctx.upload(data);
    let tri_array = ctx.upload(tris);
    match gpu.omm_api {
        OmmApi::Ext => {
            let omm = gpu.omm.as_ref().unwrap();
            let mut info = vk::MicromapBuildInfoEXT {
                ty: vk::MicromapTypeEXT::OPACITY_MICROMAP,
                flags: vk::BuildMicromapFlagsEXT::PREFER_FAST_TRACE,
                mode: vk::BuildMicromapModeEXT::BUILD,
                usage_counts_count: usage.len() as u32,
                p_usage_counts: usage.as_ptr().cast(),
                data: vk::DeviceOrHostAddressConstKHR { device_address: data.addr },
                triangle_array: vk::DeviceOrHostAddressConstKHR { device_address: tri_array.addr },
                triangle_array_stride: 8,
                ..Default::default()
            };
            let mut sizes = vk::MicromapBuildSizesInfoEXT::default();
            (omm.fp().get_micromap_build_sizes_ext)(gpu.device.handle(), vk::AccelerationStructureBuildTypeKHR::DEVICE, &info, &mut sizes);
            let backing = ctx.buf(sizes.micromap_size, Mem::Device, 256);
            let base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(backing.buffer));
            let create = vk::MicromapCreateInfoEXT {
                buffer: backing.buffer,
                offset: backing.addr - base,
                size: sizes.micromap_size,
                ty: vk::MicromapTypeEXT::OPACITY_MICROMAP,
                ..Default::default()
            };
            let mut micromap = vk::MicromapEXT::null();
            let res = (omm.fp().create_micromap_ext)(gpu.device.handle(), &create, std::ptr::null(), &mut micromap);
            if res != vk::Result::SUCCESS {
                finish(case, "WRONG", &format!("vkCreateMicromapEXT {res:?}"));
            }
            let scratch = ctx.buf(sizes.build_scratch_size, Mem::Device, 256);
            info.dst_micromap = micromap;
            info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch.addr };
            (omm.fp().cmd_build_micromaps_ext)(cmd, 1, &info);
            gpu.full_barrier(cmd, false);
            Micromap { addr: backing.addr, size: sizes.micromap_size, scratch: sizes.build_scratch_size, queried_addr: None }
        }
        OmmApi::Khr => {
            let dac = gpu.dac.as_ref().unwrap();
            let mm_data = vk::AccelerationStructureGeometryMicromapDataKHR {
                usage_counts_count: usage.len() as u32,
                p_usage_counts: usage.as_ptr(),
                data: data.addr,
                triangle_array: tri_array.addr,
                triangle_array_stride: 8,
                ..Default::default()
            };
            // The micromap data rides in the geometry's pNext; the geometry union is unused.
            let geometry = [vk::AccelerationStructureGeometryKHR {
                p_next: (&raw const mm_data).cast(),
                geometry_type: vk::GeometryTypeKHR::MICROMAP,
                ..Default::default()
            }];
            let mut build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::OPACITY_MICROMAP)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geometry);
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            // pMaxPrimitiveCounts must be null for micromaps.
            (gpu.accel.fp().get_acceleration_structure_build_sizes_khr)(
                gpu.device.handle(),
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build,
                std::ptr::null(),
                &mut sizes,
            );
            let backing = ctx.buf(sizes.acceleration_structure_size, Mem::Device, 256);
            let create = vk::AccelerationStructureCreateInfo2KHR {
                address_range: vk::DeviceAddressRangeKHR { address: backing.addr, size: sizes.acceleration_structure_size },
                // The backing buffer has storage-buffer usage, which the address flags must declare.
                address_flags: vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE,
                ty: vk::AccelerationStructureTypeKHR::OPACITY_MICROMAP,
                ..Default::default()
            };
            let micromap = match dac.create_acceleration_structure2(&create, None) {
                Ok(m) => m,
                Err(e) => finish(case, "WRONG", &format!("vkCreateAccelerationStructure2KHR {e:?}")),
            };
            let queried = gpu
                .accel
                .get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(micromap));
            let scratch = ctx.buf(sizes.build_scratch_size, Mem::Device, u64::from(gpu.as_props.min_acceleration_structure_scratch_offset_alignment));
            build = build.dst_acceleration_structure(micromap).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch.addr });
            // A micromap build takes a null range-info pointer.
            let ranges: [*const vk::AccelerationStructureBuildRangeInfoKHR; 1] = [std::ptr::null()];
            (gpu.accel.fp().cmd_build_acceleration_structures_khr)(cmd, 1, &build, ranges.as_ptr());
            gpu.full_barrier(cmd, false);
            Micromap {
                addr: backing.addr,
                size: sizes.acceleration_structure_size,
                scratch: sizes.build_scratch_size,
                queried_addr: Some(queried),
            }
        }
        OmmApi::None => finish(case, "UNSUPPORTED", "no opacity micromap extension"),
    }
}

/// A 4-state level-0 OMM referenced from a CLAS: special indices -1..-3 and -5 plus four
/// entries (transparent, opaque, unknown-transparent, unknown-opaque), traced in 4-state and
/// forced 2-state mode against the CPU reference.
pub(crate) unsafe fn case_omm(ctx: &Ctx, case: &str, pipeline: vk::Pipeline) -> (bool, String) {
    let gpu = &ctx.gpu;
    let rev = ext_rev(&gpu.exts, ash::nv::cluster_acceleration_structure::NAME).unwrap_or(0);
    if rev < 4 {
        finish(case, "UNSUPPORTED", &format!("VK_NV_cluster_acceleration_structure rev {rev} < 4 (no ALLOW_CLUSTER_OPACITY_MICROMAPS)"));
    }
    let format = vk::OpacityMicromapFormatKHR::TYPE_4_STATE.as_raw();
    let tris: Vec<vk::MicromapTriangleKHR> =
        (0..4u32).map(|i| vk::MicromapTriangleKHR { data_offset: i, subdivision_level: 0, format: format as u16 }).collect();
    let usage = [vk::MicromapUsageKHR { count: 4, subdivision_level: 0, format: vk::OpacityMicromapFormatKHR::from_raw(format) }];

    let (nx, ny) = ctx.clamp_grid(10, 10);
    let mut front = grid(3003, nx, ny, [0.05, 0.05, 0.95, 0.95], |x, y| 0.3 + 0.04 * x + 0.02 * y, 1);
    front.geom_flags = GF_CULL_DISABLE;
    front.base_geom = 2;
    let table: [i32; 8] = [-1, -2, -3, 0, 1, 3, -5, 2];
    front.omm_index = Some((0..front.tris.len()).map(|t| table[t % 8]).collect());
    let mut back = grid(5000, 1, 1, [0.0, 0.0, 1.0, 1.0], |_, _| 0.8, 1);
    back.base_geom = 9;
    let clusters = vec![front, back];

    let cmd = gpu.begin();
    gpu.device.cmd_reset_query_pool(cmd, gpu.timestamps, 0, 32);
    gpu.timestamp(cmd, 0);
    let mm = build_micromap(ctx, cmd, case, &OMM_STATES, &tris, &usage);
    gpu.timestamp(cmd, 1);
    let up = ctx.upload_clusters(&clusters, mm.addr, &[]);
    let flags = data_flags() | vk::BuildAccelerationStructureFlagsKHR::ALLOW_CLUSTER_OPACITY_MICROMAPS_NV;
    let clas = ctx.build_clas(cmd, &up, &clusters, flags, None, None);
    gpu.timestamp(cmd, 2);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0], vec![1]], None);
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &[inst_desc([0.0; 3], FRONT_CUSTOM, 0), inst_desc([0.0; 3], BACK_CUSTOM, 1)]);
    gpu.timestamp(cmd, 3);
    let out = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out, tlas, MODE_ALPHA);
    let out2 = ctx.out_buffer();
    ctx.trace(cmd, pipeline, &out2, tlas, MODE_ALPHA | MODE_OMM_2_STATE);
    gpu.timestamp(cmd, 4);
    gpu.submit_wait(cmd, case, "omm+build+trace");
    let t = gpu.read_timestamps(5);
    let insts = [ref_inst(FRONT_CUSTOM, 0, [0.0; 3], &[&clusters[0]]), ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&clusters[1]])];
    let r = check(&out, &insts, MODE_ALPHA, true);
    let r2 = check(&out2, &insts, MODE_ALPHA | MODE_OMM_2_STATE, true);
    // Control: the same rays as if the OMM were ignored (all front triangles non-opaque).
    let mut plain = clusters[0].clone();
    plain.omm_index = None;
    let r_ignored = check(&out, &[ref_inst(FRONT_CUSTOM, 0, [0.0; 3], &[&plain]), insts[1].clone()], MODE_ALPHA, true);
    let addr = match mm.queried_addr {
        Some(q) => format!("micromap_addr={:#x} queried_addr={q:#x}", mm.addr),
        None => format!("micromap_addr={:#x} (backing buffer)", mm.addr),
    };
    let detail = format!(
        "omm_api={:?} cluster_rev={rev} {addr} micromap_bytes={} scratch={} 4state[{}] 2state[{}] if_omm_ignored_mismatches={:?} {} gpu_us(omm/clas/blas+tlas/traces)={}",
        gpu.omm_api,
        mm.size,
        mm.scratch,
        r.summary(),
        r2.summary(),
        r_ignored.bad,
        sizes_detail(gpu, &clas, &blas, 2, 2, &clusters),
        us(&t)
    );
    (r.ok() && r2.ok(), detail)
}
