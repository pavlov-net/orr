//! A TLAS (classic and partitioned) over one cluster BLAS and one LSS BLAS
//! (VK_NV_ray_tracing_linear_swept_spheres), traced by inline ray query from compute. No
//! ray-tracing pipeline exists, so there is no pipeline flag to set. Checked against a CPU
//! reference with the LSS-aware shader and with the plain cluster shader (no LSS capability).

use super::*;

const LSS_CUSTOM: u32 = 55;
const LSS_EPS: f32 = 1e-3;
const LSS_T_EPS: f32 = 2e-4;

#[derive(Clone, Copy)]
struct Segment {
    a: [f32; 2],
    b: [f32; 2],
    z: f32,
    r: f32,
}

/// Deterministic segments over the unit square, alternating in front of the cluster
/// instance (z 0.12) and behind it (z 0.6).
fn segments() -> Vec<Segment> {
    let mut s = 0x2545_f491u32;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
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

/// Reference for ray (x, y): None when it passes near any edge, silhouette or segment end.
fn reference(front: &RefInst, segs: &[Segment], x: u32, y: u32) -> Option<Want> {
    let e = expect(std::slice::from_ref(front), x, y, 0)?;
    let p = [(x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32];
    let mut best = match e.hit {
        Some((_, tri, t)) => Want::Cluster(tri, t),
        None => Want::Miss,
    };
    let best_t = |w: &Want| match w {
        Want::Miss => f32::INFINITY,
        Want::Cluster(_, t) | Want::Lss(_, t) => *t,
    };
    for (i, s) in segs.iter().enumerate() {
        let d = [s.b[0] - s.a[0], s.b[1] - s.a[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let u = ((p[0] - s.a[0]) * d[0] + (p[1] - s.a[1]) * d[1]) / (len * len);
        let q = [s.a[0] + u * d[0] - p[0], s.a[1] + u * d[1] - p[1]];
        let dist = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let near_end = (u * len).abs() < LSS_EPS || ((u - 1.0) * len).abs() < LSS_EPS;
        let inside = (0.0..=1.0).contains(&u);
        if (near_end && dist < s.r + LSS_EPS) || (inside && (dist - s.r).abs() < LSS_EPS) {
            return None;
        }
        if inside && dist < s.r {
            let t = 1.0 + s.z - (s.r * s.r - dist * dist).sqrt();
            if t < best_t(&best) {
                best = Want::Lss(i, t);
            }
        }
    }
    Some(best)
}

/// Compares a record buffer with the reference. `lss_fields` is the LSS-aware layout.
unsafe fn check_lss(out: &Buf, front: &RefInst, segs: &[Segment], lss_fields: bool) -> (Report, BTreeMap<u32, u32>) {
    let recs = std::slice::from_raw_parts(out.ptr.cast::<u32>(), (W * H) as usize * REC);
    let mut r = Report::default();
    let mut statuses = BTreeMap::new();
    for y in 0..H {
        for x in 0..W {
            let o = &recs[(y * W + x) as usize * REC..][..REC];
            let Some(want) = reference(front, segs, x, y) else {
                r.skipped += 1;
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
            let got_hit = o[0] != 0;
            if got_hit {
                *statuses.entry(o[0]).or_insert(0u32) += 1;
            }
            let t = f32::from_bits(o[6]);
            match want {
                Want::Miss => {
                    if got_hit {
                        bad("hit_on_miss", format!("instance {} prim {}", o[2], o[4]));
                    }
                }
                Want::Cluster(tri, want_t) => {
                    if !got_hit {
                        bad("miss_on_cluster", String::new());
                        continue;
                    }
                    r.hits += 1;
                    let rt = &front.tris[tri];
                    r.clusters.insert(o[3]);
                    if o[1] != FRONT_CUSTOM || o[2] != 0 || o[4] != rt.prim || o[3] != rt.cluster || o[5] != rt.geom {
                        bad("cluster_ids", format!("got custom {} index {} cluster {} prim {} geom {}", o[1], o[2], o[3], o[4], o[5]));
                    }
                    if (t - want_t).abs() > T_EPS {
                        bad("cluster_t", format!("got {t} want {want_t}"));
                    }
                    if lss_fields && o[7] != 0 {
                        bad("cluster_is_lss", String::new());
                    }
                }
                Want::Lss(seg, want_t) => {
                    if !got_hit {
                        bad("miss_on_lss", format!("segment {seg}"));
                        continue;
                    }
                    r.hits += 1;
                    r.candidates += 1;
                    r.clusters.insert(o[3]);
                    if o[1] != LSS_CUSTOM || o[2] != 1 || o[4] != seg as u32 || o[5] != 0 {
                        bad("lss_ids", format!("got custom {} index {} prim {} geom {} want segment {seg}", o[1], o[2], o[4], o[5]));
                    }
                    if o[3] != u32::MAX {
                        bad("lss_cluster_id", format!("got {}", o[3]));
                    }
                    if (t - want_t).abs() > LSS_T_EPS {
                        bad("lss_t", format!("got {t} want {want_t}"));
                    }
                    if lss_fields {
                        let s = segs[seg];
                        let want_pr = [s.a[0], s.a[1], s.z, s.r, s.b[0], s.b[1], s.z, s.r];
                        let got_pr: Vec<f32> = o[8..16].iter().map(|&b| f32::from_bits(b)).collect();
                        if o[7] != 1 {
                            bad("lss_not_flagged", String::new());
                        } else if got_pr != want_pr {
                            bad("lss_positions", format!("got {got_pr:?} want {want_pr:?}"));
                        }
                    }
                }
            }
        }
    }
    (r, statuses)
}

unsafe fn build_lss_blas(ctx: &Ctx, cmd: vk::CommandBuffer, segs: &[Segment]) -> (u64, u64) {
    let gpu = &ctx.gpu;
    let verts: Vec<[f32; 3]> = segs.iter().flat_map(|s| [[s.a[0], s.a[1], s.z], [s.b[0], s.b[1], s.z]]).collect();
    let radii: Vec<f32> = segs.iter().flat_map(|s| [s.r, s.r]).collect();
    let indices: Vec<u32> = (0..2 * segs.len() as u32).collect();
    let (v, rad, idx) = (ctx.upload(&verts), ctx.upload(&radii), ctx.upload(&indices));
    let lss = vk::AccelerationStructureGeometryLinearSweptSpheresDataNV::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: v.addr })
        .vertex_stride(12)
        .radius_format(vk::Format::R32_SFLOAT)
        .radius_data(vk::DeviceOrHostAddressConstKHR { device_address: rad.addr })
        .radius_stride(4)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: idx.addr })
        .index_stride(4)
        .indexing_mode(vk::RayTracingLssIndexingModeNV::LIST)
        .end_caps_mode(vk::RayTracingLssPrimitiveEndCapsModeNV::NONE);
    let geometry = [vk::AccelerationStructureGeometryKHR {
        p_next: (&raw const lss).cast(),
        geometry_type: vk::GeometryTypeKHR::LINEAR_SWEPT_SPHERES_NV,
        flags: vk::GeometryFlagsKHR::OPAQUE,
        ..Default::default()
    }];
    let n = segs.len() as u32;
    let mut build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometry);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    gpu.accel.get_acceleration_structure_build_sizes(vk::AccelerationStructureBuildTypeKHR::DEVICE, &build, Some(&[n]), &mut sizes);
    let backing = ctx.buf(sizes.acceleration_structure_size, Mem::Device, 256);
    let base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(backing.buffer));
    let blas = gpu
        .accel
        .create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(backing.buffer)
                .offset(backing.addr - base)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )
        .unwrap();
    let scratch = ctx.buf(sizes.build_scratch_size, Mem::Device, u64::from(gpu.as_props.min_acceleration_structure_scratch_offset_alignment));
    build = build.dst_acceleration_structure(blas).scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch.addr });
    let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n)];
    gpu.accel.cmd_build_acceleration_structures(cmd, &[build], &[Some(&ranges)]);
    gpu.full_barrier(cmd, false);
    let addr = gpu.accel.get_acceleration_structure_device_address(&vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(blas));
    (addr, sizes.acceleration_structure_size)
}

pub(crate) unsafe fn case_lss_mixed(ctx: &Ctx, case: &str, plain: vk::Pipeline) -> (bool, String) {
    let gpu = &ctx.gpu;
    if !gpu.has_lss {
        finish(case, "UNSUPPORTED", "VK_NV_ray_tracing_linear_swept_spheres absent");
    }
    let lss_pipe = gpu.pipeline(&spv!("trace_lss"));
    let clusters = base_clusters(ctx);
    let front = &clusters[..4];
    let segs = segments();
    let up = ctx.upload_clusters(front, 0, &[]);

    let cmd = gpu.begin();
    let clas = ctx.build_clas(cmd, &up, front, data_flags(), None, None);
    let blas = ctx.build_blas(cmd, clas.addrs.addr, &[vec![0, 1, 2, 3]], None);
    let (lss_blas, lss_bytes) = build_lss_blas(ctx, cmd, &segs);
    let base = gpu.device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(blas.addrs.buffer));
    gpu.device.cmd_update_buffer(cmd, blas.addrs.buffer, blas.addrs.addr - base + 8, bytes(&[lss_blas]));
    gpu.full_barrier(cmd, false);
    let tlas = ctx.build_tlas(cmd, blas.addrs.addr, &[inst_desc([0.0, 0.0, FRONT_Z], FRONT_CUSTOM, 0), inst_desc([0.0; 3], LSS_CUSTOM, 1)]);
    gpu.submit_wait(cmd, case, "clas+blas+lss+tlas");
    let cluster_blas: u64 = blas.addrs.read::<u64>(1)[0];

    // The same two instances in a PTLAS, one partition each.
    let d = ptlas::PtlasDesc { instances: 2, per_partition: 1, partitions: 2, global: 0 };
    let (size, scratch_size) = ptlas::sizes(gpu, &d);
    let dst = ctx.buf(size, Mem::Device, 256);
    let scratch = ctx.buf(scratch_size, Mem::Device, 256);
    let writes = ctx.upload(&[
        ptlas::write_rec(0, 0, FRONT_CUSTOM, translate([0.0, 0.0, FRONT_Z]), cluster_blas, None),
        ptlas::write_rec(1, 1, LSS_CUSTOM, translate([0.0; 3]), lss_blas, None),
    ]);
    let ops = ptlas::OpList::new(ctx, &[ptlas::Op::write(writes.addr, 2)]);
    let cmd = gpu.begin();
    ptlas::record(gpu, cmd, &d, dst.addr, scratch.addr, true, &ops);
    gpu.submit_wait(cmd, case, "ptlas");

    let front_ref = ref_inst(FRONT_CUSTOM, 0, [0.0, 0.0, FRONT_Z], &front.iter().collect::<Vec<_>>());
    let mut parts = Vec::new();
    let mut ok = true;
    for (label, as_addr) in [("tlas", tlas), ("ptlas", dst.addr)] {
        for (shader, pipe, lss_fields, mode) in
            [("lss_shader", lss_pipe, true, 0), ("plain_shader_triangle_status", plain, false, 0), ("plain_shader_any_status", plain, false, 8)]
        {
            let cmd = gpu.begin();
            let out = ctx.out_buffer();
            ctx.trace(cmd, pipe, &out, as_addr, mode);
            gpu.submit_wait(cmd, case, &format!("trace {label} {shader}"));
            let (r, statuses) = check_lss(&out, &front_ref, &segs, lss_fields);
            // The triangle-status shader is expected to drop LSS hits; it is reported, not gated.
            if mode != 0 || lss_fields {
                ok &= r.bad.is_empty() && r.candidates > 0 && r.hits > r.candidates;
            }
            parts.push(format!("{label}/{shader}[{} lss_hits={} committed_status_counts={statuses:?}]", r.summary(), r.candidates));
        }
    }
    (ok, format!("lss_blas_bytes={lss_bytes} segments={} {}", segs.len(), parts.join(" ")))
}
