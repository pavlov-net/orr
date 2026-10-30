//! CLAS, template and cluster BLAS built sizes at orr's cluster shape: meshoptimizer spatial
//! clusters of at most 128 triangles (128 vertices on solid sections, 256 on foliage),
//! positions on the per-mesh integer grid with `clas_truncate_bits`, 8-bit indices,
//! per-triangle flag words, `ALLOW_DATA_ACCESS`, OMM for alpha-tested sections, implicit
//! builds. Arguments: `role=path.glb#mesh#primitive` (primitive `*` merges all of the mesh),
//! role `solid`, `foliage` (opaque leaves) or `alpha` (alpha-tested foliage with a 4-state
//! level-4 OMM per triangle); `role@N=` sets the vertex cap to N.

use super::*;

/// Minimal GLB reader for positions and indices, with `KHR_meshopt_compression` buffer views
/// decoded by meshoptimizer and `KHR_mesh_quantization` integer positions dequantized by the
/// mesh node's transform (skinned nodes keep bind-space positions).
mod glb {
    use serde_json::Value;

    struct File {
        json: Value,
        bin: Vec<u8>,
    }

    impl File {
        /// Raw bytes of a buffer view, decoded when it is meshopt-compressed.
        fn view(&self, index: usize) -> (Vec<u8>, usize) {
            let v = &self.json["bufferViews"][index];
            let ext = v["extensions"].get("KHR_meshopt_compression").or_else(|| v["extensions"].get("EXT_meshopt_compression"));
            let stride = v["byteStride"].as_u64().unwrap_or(0) as usize;
            let Some(e) = ext else {
                let off = v["byteOffset"].as_u64().unwrap_or(0) as usize;
                let len = v["byteLength"].as_u64().unwrap() as usize;
                return (self.bin[off..off + len].to_vec(), stride);
            };
            assert_eq!(e["buffer"].as_u64().unwrap(), 0, "compressed data outside the GLB chunk");
            let off = e["byteOffset"].as_u64().unwrap_or(0) as usize;
            let len = e["byteLength"].as_u64().unwrap() as usize;
            let size = e["byteStride"].as_u64().unwrap() as usize;
            let count = e["count"].as_u64().unwrap() as usize;
            let src = &self.bin[off..off + len];
            let mut dst = vec![0u8; size * count];
            let rc = unsafe {
                let d = dst.as_mut_ptr().cast();
                match e["mode"].as_str().unwrap() {
                    "ATTRIBUTES" => meshopt::ffi::meshopt_decodeVertexBuffer(d, count, size, src.as_ptr(), len),
                    "TRIANGLES" => meshopt::ffi::meshopt_decodeIndexBuffer(d, count, size, src.as_ptr(), len),
                    "INDICES" => meshopt::ffi::meshopt_decodeIndexSequence(d, count, size, src.as_ptr(), len),
                    m => panic!("meshopt mode {m}"),
                }
            };
            assert_eq!(rc, 0, "meshopt decode failed");
            match e["filter"].as_str().unwrap_or("NONE") {
                "NONE" => {}
                "EXPONENTIAL" => unsafe { meshopt::ffi::meshopt_decodeFilterExp(dst.as_mut_ptr().cast(), count, size) },
                "OCTAHEDRAL" => unsafe { meshopt::ffi::meshopt_decodeFilterOct(dst.as_mut_ptr().cast(), count, size) },
                f => panic!("meshopt filter {f} on a position or index view"),
            }
            (dst, size)
        }

        /// An accessor's components as f64 (normalized integers mapped per glTF).
        fn accessor(&self, index: usize) -> (Vec<f64>, usize) {
            let a = &self.json["accessors"][index];
            let comps = match a["type"].as_str().unwrap() {
                "SCALAR" => 1,
                "VEC2" => 2,
                "VEC3" => 3,
                "VEC4" => 4,
                t => panic!("accessor type {t}"),
            };
            let ct = a["componentType"].as_u64().unwrap();
            let csize = match ct {
                5120 | 5121 => 1,
                5122 | 5123 => 2,
                _ => 4,
            };
            let norm = a["normalized"].as_bool().unwrap_or(false);
            let count = a["count"].as_u64().unwrap() as usize;
            let (data, stride) = self.view(a["bufferView"].as_u64().unwrap() as usize);
            let stride = if stride == 0 { comps * csize } else { stride };
            let base = a["byteOffset"].as_u64().unwrap_or(0) as usize;
            let mut out = Vec::with_capacity(count * comps);
            for i in 0..count {
                for c in 0..comps {
                    let o = base + i * stride + c * csize;
                    let b = &data[o..o + csize];
                    let v = match ct {
                        5120 => (f64::from(b[0] as i8), 127.0),
                        5121 => (f64::from(b[0]), 255.0),
                        5122 => (f64::from(i16::from_le_bytes([b[0], b[1]])), 32767.0),
                        5123 => (f64::from(u16::from_le_bytes([b[0], b[1]])), 65535.0),
                        5125 => (f64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]])), 1.0),
                        _ => (f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])), 1.0),
                    };
                    out.push(if norm { (v.0 / v.1).max(-1.0) } else { v.0 });
                }
            }
            (out, comps)
        }
    }

    pub(super) fn read(path: &str, mesh: usize, prim: &str) -> (Vec<[f32; 3]>, Vec<u32>) {
        let raw = std::fs::read(path).expect("glb");
        let word = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
        assert_eq!(&raw[0..4], b"glTF");
        let jlen = word(12);
        let json: Value = serde_json::from_slice(&raw[20..20 + jlen]).unwrap();
        let bo = 20 + jlen;
        let bin = raw[bo + 8..bo + 8 + word(bo)].to_vec();
        let f = File { json, bin };
        // The node that instances the mesh: translation, rotation and scale dequantize positions.
        let node = f.json["nodes"].as_array().unwrap().iter().find(|n| n["mesh"].as_u64() == Some(mesh as u64));
        let get = |n: Option<&Value>, k: &str, d: &[f64]| -> Vec<f64> {
            n.filter(|n| n.get("skin").is_none())
                .and_then(|n| n[k].as_array())
                .map_or(d.to_vec(), |a| a.iter().map(|v| v.as_f64().unwrap()).collect())
        };
        assert!(node.is_none_or(|n| n.get("matrix").is_none()), "node matrices not handled");
        let t = get(node, "translation", &[0.0; 3]);
        let r = get(node, "rotation", &[0.0, 0.0, 0.0, 1.0]);
        let sc = get(node, "scale", &[1.0; 3]);
        // Quaternion rotation: p + 2w (q x p) + 2 q x (q x p).
        let rot = |p: [f64; 3]| {
            let q = [r[0], r[1], r[2]];
            let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
            let c1 = cross(q, p);
            let c2 = cross(q, c1);
            [0, 1, 2].map(|k| p[k] + 2.0 * r[3] * c1[k] + 2.0 * c2[k])
        };
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        for (pi, p) in f.json["meshes"][mesh]["primitives"].as_array().unwrap().iter().enumerate() {
            if prim != "*" && prim.parse::<usize>().unwrap() != pi {
                continue;
            }
            let first = pos.len() as u32;
            let (v, comps) = f.accessor(p["attributes"]["POSITION"].as_u64().unwrap() as usize);
            assert_eq!(comps, 3);
            for c in v.chunks(3) {
                let s = rot([c[0] * sc[0], c[1] * sc[1], c[2] * sc[2]]);
                pos.push([(s[0] + t[0]) as f32, (s[1] + t[1]) as f32, (s[2] + t[2]) as f32]);
            }
            let (i, _) = f.accessor(p["indices"].as_u64().unwrap() as usize);
            idx.extend(i.iter().map(|&x| x as u32 + first));
        }
        assert!(!idx.is_empty(), "no primitive selected");
        (pos, idx)
    }
}

struct Section {
    role: String,
    label: String,
    /// Grid positions as f32, cluster-contiguous.
    verts: Vec<[f32; 3]>,
    /// u8x3 cluster-local triangles, cluster-contiguous.
    tris: Vec<u8>,
    /// (first vertex, vertex count, first triangle, triangle count) per cluster.
    clusters: Vec<(u32, u32, u32, u32)>,
    truncate: u32,
    step_log2: i32,
    max_verts: u32,
}

fn load(spec: &str, max_verts_override: Option<u32>) -> Section {
    let (role, rest) = spec.split_once('=').expect("role=path#mesh#primitive");
    let mut it = rest.split('#');
    let path = it.next().unwrap();
    let mesh: usize = it.next().expect("mesh index").parse().unwrap();
    let prim = it.next().expect("primitive index or *");
    let (pos, idx) = glb::read(path, mesh, prim);

    // Per-mesh grid (contracts.md 17): step 2^k with k = ceil(log2(extent / 65535)).
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for v in &pos {
        for k in 0..3 {
            lo[k] = lo[k].min(v[k]);
            hi[k] = hi[k].max(v[k]);
        }
    }
    let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
    let step_log2 = (f64::from(extent) / 65535.0).log2().ceil() as i32;
    let step = 2f64.powi(step_log2);
    let q: Vec<[i32; 3]> = pos.iter().map(|v| v.map(|x| (f64::from(x) / step).round() as i32)).collect();
    let max_abs = q.iter().flatten().map(|v| v.unsigned_abs()).max().unwrap();
    let truncate = 24u32.saturating_sub(32 - max_abs.leading_zeros());
    let max_verts = max_verts_override.unwrap_or(if role == "solid" { 128 } else { 256 });

    let adapter = meshopt::VertexDataAdapter::new(bytes(&pos), 12, 0).unwrap();
    let fill = std::env::var("PRB5_FILL").ok().and_then(|f| f.parse().ok()).unwrap_or(0.5);
    let m = meshopt::build_meshlets_spatial(&idx, &adapter, max_verts as usize, 32, 128, fill);
    let mut s = Section {
        role: role.to_owned(),
        label: format!("{}#{mesh}#{prim}", std::path::Path::new(path).file_stem().unwrap().to_string_lossy()),
        verts: Vec::new(),
        tris: Vec::new(),
        clusters: Vec::new(),
        truncate,
        step_log2,
        max_verts,
    };
    for ml in &m.meshlets {
        let first_v = s.verts.len() as u32;
        let first_t = (s.tris.len() / 3) as u32;
        for &vi in &m.vertices[ml.vertex_offset as usize..][..ml.vertex_count as usize] {
            s.verts.push(q[vi as usize].map(|g| (f64::from(g) * step) as f32));
        }
        s.tris.extend_from_slice(&m.triangles[ml.triangle_offset as usize..][..ml.triangle_count as usize * 3]);
        s.clusters.push((first_v, ml.vertex_count, first_t, ml.triangle_count));
    }
    s
}

struct Variant {
    name: &'static str,
    truncate: bool,
    data_access: bool,
    omm_flag: bool,
    omm: bool,
    alpha: bool,
}

struct Built {
    sizes: Vec<u32>,
    host: u64,
    scratch: u64,
}

impl Built {
    fn sum(&self) -> u64 {
        self.sizes.iter().map(|&s| u64::from(s)).sum()
    }
    fn pool(&self, align: u64) -> u64 {
        self.sizes.iter().map(|&s| align_up(u64::from(s), align)).sum()
    }
    fn pct(&self, p: f64) -> u32 {
        let mut v = self.sizes.clone();
        v.sort_unstable();
        v[((v.len() - 1) as f64 * p) as usize]
    }
}

struct Shared {
    vbuf: Buf,
    ibuf: Buf,
    gflags_opaque: Buf,
    gflags_alpha: Buf,
    omm_index: Buf,
    bbox: Buf,
}

pub(crate) unsafe fn case_clas_bytes(ctx: &Ctx, case: &str, args: &[String]) -> (bool, String) {
    let gpu = &ctx.gpu;
    let specs: Vec<String> = if args.is_empty() {
        std::env::var("PRB5_MESHES").map(|s| s.split_whitespace().map(str::to_owned).collect()).unwrap_or_default()
    } else {
        args.to_vec()
    };
    if specs.is_empty() {
        finish(case, "UNSUPPORTED", "no meshes: pass role=path.glb#mesh#primitive or set PRB5_MESHES");
    }
    let align = u64::from(gpu.cl_props.cluster_byte_alignment);
    let mut detail = Vec::new();
    let mut ok = true;
    for spec in &specs {
        // `role@N=...` overrides the vertex cap.
        let (spec, cap) = match spec.split_once('=') {
            Some((r, rest)) if r.contains('@') => {
                let (role, n) = r.split_once('@').unwrap();
                (format!("{role}={rest}"), Some(n.parse().unwrap()))
            }
            _ => (spec.clone(), None),
        };
        let s = load(&spec, cap);
        let n = s.clusters.len() as u32;
        let tri_total: u32 = s.clusters.iter().map(|c| c.3).sum();
        let vert_total: u32 = s.clusters.iter().map(|c| c.1).sum();
        let max_t = s.clusters.iter().map(|c| c.3).max().unwrap();
        let max_v = s.clusters.iter().map(|c| c.1).max().unwrap();

        // Shared buffers: vertices, indices, flag words, OMM indices, template bounds.
        let mut lo_hi = Vec::with_capacity(n as usize);
        for &(fv, vc, _, _) in &s.clusters {
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for v in &s.verts[fv as usize..][..vc as usize] {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            let grow = 0.02 * (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
            lo_hi.push([lo[0] - grow, lo[1] - grow, lo[2] - grow, hi[0] + grow, hi[1] + grow, hi[2] + grow, 0.0, 0.0]);
        }
        let omm_entries = tri_total.min(8192);
        let b = Shared {
            vbuf: ctx.upload(&s.verts),
            ibuf: ctx.upload(&s.tris),
            gflags_opaque: ctx.upload(&vec![GF_OPAQUE << 29; tri_total as usize]),
            gflags_alpha: ctx.upload(&vec![0u32; tri_total as usize]),
            omm_index: ctx.upload(&(0..tri_total).map(|t| (t % omm_entries) as i16).collect::<Vec<_>>()),
            bbox: ctx.upload(&lo_hi),
        };

        // 4-state level-4 micromap: 256 micro-triangles x 2 bits = 64 B per entry.
        let mut r = 0x1234_5678u32;
        let data: Vec<u8> = (0..omm_entries * 64)
            .map(|_| {
                r ^= r << 13;
                r ^= r >> 17;
                r ^= r << 5;
                r as u8
            })
            .collect();
        let format = vk::OpacityMicromapFormatKHR::TYPE_4_STATE.as_raw();
        let mm_tris: Vec<vk::MicromapTriangleKHR> =
            (0..omm_entries).map(|i| vk::MicromapTriangleKHR { data_offset: i * 64, subdivision_level: 4, format: format as u16 }).collect();
        let usage = [vk::MicromapUsageKHR { count: omm_entries, subdivision_level: 4, format: vk::OpacityMicromapFormatKHR::from_raw(format) }];
        let cmd = gpu.begin();
        let mm = omm::build_micromap(ctx, cmd, case, &data, &mm_tris, &usage);
        gpu.submit_wait(cmd, case, "micromap");

        let descs = |v: &Variant, bbox: bool| -> Buf {
            let d: Vec<ClusterDesc> = s
                .clusters
                .iter()
                .enumerate()
                .map(|(ci, &(fv, vc, ft, tc))| ClusterDesc {
                    index_addr: b.ibuf.addr + u64::from(ft) * 3,
                    vertex_addr: b.vbuf.addr + u64::from(fv) * 12,
                    gflags_addr: if v.alpha { b.gflags_alpha.addr } else { b.gflags_opaque.addr } + u64::from(ft) * 4,
                    omm_array: if v.omm { mm.addr } else { 0 },
                    omm_index: if v.omm { b.omm_index.addr + u64::from(ft) * 2 } else { 0 },
                    bbox_addr: if bbox { b.bbox.addr + ci as u64 * 32 } else { 0 },
                    id: ci as u32,
                    tris: tc,
                    verts: vc,
                    index_type: 1,
                    truncate: if v.truncate { s.truncate } else { 0 },
                    base_geom: 0,
                    geom_flags: 0,
                    cluster_flags: 0,
                    index_stride: 1,
                    vertex_stride: 12,
                    gflags_stride: 4,
                    omm_index_type: if v.omm { 2 } else { 1 },
                    omm_index_stride: if v.omm { 2 } else { 0 },
                    pad: [0; 3],
                })
                .collect();
            ctx.upload(&d)
        };
        let input = |v: &Variant| {
            vk::ClusterAccelerationStructureTriangleClusterInputNV::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .max_geometry_index_value(1)
                .max_cluster_unique_geometry_count(1)
                .max_cluster_triangle_count(max_t)
                .max_cluster_vertex_count(max_v)
                .max_total_triangle_count(tri_total)
                .max_total_vertex_count(vert_total)
                .min_position_truncate_bit_count(if v.truncate { s.truncate } else { 0 })
        };
        let flags = |v: &Variant| {
            let mut f = vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE;
            if v.data_access {
                f |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS;
            }
            if v.omm_flag {
                f |= vk::BuildAccelerationStructureFlagsKHR::ALLOW_CLUSTER_OPACITY_MICROMAPS_NV;
            }
            f
        };
        let op = |cmd, ty, args_op, src: &Buf, v: &Variant, aux: u64| -> ClusterOut {
            let mut ti = input(v);
            let mut inp = vk::ClusterAccelerationStructureOpInputNV { p_triangle_clusters: &mut ti };
            ctx.cluster_op(cmd, ty, args_op, src.addr, aux, n, &mut inp, flags(v), None, None)
        };
        let read = |o: &ClusterOut| Built { sizes: o.sizes.read(n as usize), host: o.implicit_size, scratch: o.scratch_size };

        let alpha = s.role == "alpha";
        // The OMM flag is batch-wide: orr sets it only on alpha-tested batches.
        let base = Variant { name: "orr", truncate: true, data_access: true, omm_flag: alpha, omm: alpha, alpha };
        let mut variants = vec![
            Variant { name: "orr", ..base },
            Variant { name: "truncate_0", truncate: false, ..base },
            Variant { name: "no_data_access", data_access: false, ..base },
        ];
        if alpha {
            variants.push(Variant { name: "alpha_no_omm", omm_flag: false, omm: false, ..base });
        } else {
            variants.push(Variant { name: "omm_flag_no_array", omm_flag: true, ..base });
        }
        let head = format!(
            "mesh={} role={} fill_weight={} clusters={n} triangles={tri_total} vertices={vert_total} tris_per_cluster={:.1} verts_per_cluster={:.1} vertex_cap={} step_log2={} truncate_bits={}",
            s.label,
            s.role,
            std::env::var("PRB5_FILL").unwrap_or("0.5".into()),
            f64::from(tri_total) / f64::from(n),
            f64::from(vert_total) / f64::from(n),
            s.max_verts,
            s.step_log2,
            s.truncate
        );
        let mut orr_clas = None;
        for v in &variants {
            let src = descs(v, false);
            let cmd = gpu.begin();
            let o = op(cmd, vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER, OP_CLUSTER, &src, v, 0);
            gpu.submit_wait(cmd, case, v.name);
            let built = read(&o);
            ok &= built.sizes.iter().all(|&x| x > 0);
            println!(
                "ROW clas {head} variant={} built_sum={} B_per_tri={:.2} pool_bytes_128={} pool_B_per_tri={:.2} cluster_bytes_p50={} p95={} max={} host_implicit={} host_B_per_tri={:.2} host_over_built={:.2} scratch={}",
                v.name,
                built.sum(),
                built.sum() as f64 / f64::from(tri_total),
                built.pool(align),
                built.pool(align) as f64 / f64::from(tri_total),
                built.pct(0.5),
                built.pct(0.95),
                built.pct(1.0),
                built.host,
                built.host as f64 / f64::from(tri_total),
                built.host as f64 / built.sum() as f64,
                built.scratch
            );
            if v.name == "orr" {
                detail.push(format!("{}:{}:{:.2}B/tri", s.label, s.role, built.pool(align) as f64 / f64::from(tri_total)));
                orr_clas = Some(o);
            }
        }

        // One cluster BLAS over every CLAS of the section.
        let clas = orr_clas.unwrap();
        let cmd = gpu.begin();
        let all: Vec<u32> = (0..n).collect();
        let blas = ctx.build_blas(cmd, clas.addrs.addr, &[all], None);
        gpu.submit_wait(cmd, case, "cluster blas");
        let blas_bytes = blas.sizes.read::<u32>(1)[0];
        println!(
            "ROW blas {head} blas_bytes={blas_bytes} B_per_clas={:.1} B_per_tri={:.2} host_implicit={}",
            f64::from(blas_bytes) / f64::from(n),
            f64::from(blas_bytes) / f64::from(tri_total),
            blas.implicit_size
        );

        // Templates with and without bounds, then instantiation from the rest positions.
        let v = &variants[0];
        for bounds in [true, false] {
            let src = descs(v, bounds);
            let cmd = gpu.begin();
            let t = op(cmd, vk::ClusterAccelerationStructureOpTypeNV::BUILD_TRIANGLE_CLUSTER_TEMPLATE, OP_TEMPLATE, &src, v, 0);
            gpu.submit_wait(cmd, case, "templates");
            let tb = read(&t);
            let inst: Vec<InstantiateDesc> = s
                .clusters
                .iter()
                .enumerate()
                .map(|(ci, &(fv, _, _, _))| InstantiateDesc {
                    template_addr_ptr: t.addrs.addr + 8 * ci as u64,
                    vertex_addr: b.vbuf.addr + u64::from(fv) * 12,
                    vertex_stride: 12,
                    cluster_id_offset: 0,
                    geometry_index_offset: 0,
                })
                .collect();
            let isrc = ctx.upload(&inst);
            let cmd = gpu.begin();
            let i = op(cmd, vk::ClusterAccelerationStructureOpTypeNV::INSTANTIATE_TRIANGLE_CLUSTER, OP_INSTANTIATE, &isrc, v, 0);
            gpu.submit_wait(cmd, case, "instantiate");
            let ib = read(&i);
            println!(
                "ROW template {head} bounds={bounds} template_sum={} template_B_per_tri={:.2} template_host={} instantiated_sum={} instantiated_B_per_tri={:.2} instantiated_pool_128={} instantiate_host={} instantiate_host_over_built={:.2}",
                tb.sum(),
                tb.sum() as f64 / f64::from(tri_total),
                tb.host,
                ib.sum(),
                ib.sum() as f64 / f64::from(tri_total),
                ib.pool(align),
                ib.host,
                ib.host as f64 / ib.sum() as f64
            );
        }
        if alpha {
            println!("ROW omm {head} omm_entries={omm_entries} level=4 micromap_bytes={} B_per_entry={:.1}", mm.size, mm.size as f64 / f64::from(omm_entries));
        }
    }
    (ok, detail.join(" "))
}
