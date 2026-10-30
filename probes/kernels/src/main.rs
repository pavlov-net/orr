//! Kernel microbenchmarks and crash tooling on the descriptor-heap device.
//!
//! `kernels vbuf`: 64-bit atomic max into an R64_UINT storage image from the heap vs a buffer by
//! address (linear and 8x8 tiles), software-raster micro-triangles at 1920x1080, checked against
//! a CPU raster. `kernels root`: root as a uniform buffer (`PUSH_ADDRESS`) vs a pointer in push
//! data, compute and mesh. `kernels coopvec`: FP16 16x16 matmul through cooperative vectors,
//! inline SPIR-V over an address vs Slang over an `INDIRECT_ADDRESS` buffer, at matrix offsets
//! 0, 16 and 64, against a CPU matmul. `kernels sky`: atmosphere LUT passes at 1, 2, 4, 8 folded
//! lights. `kernels crash`: Aftermath dump of a forced out-of-bounds write (see crash.rs).
//!
//! Build from this directory with `cargo build --release --target-dir ../../target/probes`.
//! Knobs: PROBE_ROUNDS (31), PROBE_GAP_MS (6), PROBE_WARMUP_MS (1500), PROBE_VBUF (cell x layers
//! list, `1x4,1x8,2x8,4x8`). Validation: `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; time
//! without it.
#![allow(clippy::too_many_arguments)]

mod crash;
mod gpu;

use ash::vk;
use gpu::*;

const W: u32 = 1920;
const H: u32 = 1080;

fn pcg(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn report(names: &[&str], t: &[(f64, f64, f64)]) {
    let base = t[0].0;
    for (n, (m, lo, hi)) in names.iter().zip(t) {
        println!("  {n:<30} median {m:>9.1} us  p10 {lo:>9.1}  p90 {hi:>9.1}  ratio {:.3}", m / base);
    }
}

unsafe fn copy_image(gpu: &Gpu, cmd: vk::CommandBuffer, img: &Image, dst: &Buf) {
    let region = vk::BufferImageCopy::default()
        .buffer_offset(dst.offset)
        .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
        .image_extent(vk::Extent3D { width: img.width, height: img.height, depth: 1 });
    unsafe { gpu.device.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, dst.buffer, &[region]) };
}

// ---------------------------------------------------------------------------------------------
// Visibility buffer.

#[repr(C)]
#[derive(Clone, Copy)]
struct VbufRoot {
    vis: u64,
    img: u32,
    width: u32,
    height: u32,
    cell: u32,
    cells_x: u32,
    cells_y: u32,
    layers: u32,
    ntri: u32,
    pad: [u32; 2],
}

/// CPU raster identical to `raster` in vbuf.slang; returns the visibility buffer and the number
/// of covered samples.
fn raster_ref(r: &VbufRoot) -> (Vec<u64>, u64) {
    let mut vis = vec![0u64; (W * H) as usize];
    let mut covered = 0u64;
    let edge = |a: [i32; 2], b: [i32; 2], p: [i32; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    let per_layer = r.cells_x * r.cells_y * 2;
    let s = r.cell as i32 * 256;
    for t in 0..r.ntri {
        let layer = t / per_layer;
        let c = (t % per_layer) >> 1;
        let mut base = [(c % r.cells_x) as i32 * s, (c / r.cells_x) as i32 * s];
        base[0] += (pcg(layer.wrapping_mul(7919)) & 255) as i32 * s / 256;
        base[1] += (pcg(layer.wrapping_mul(104729)) & 255) as i32 * s / 256;
        let corner = if t & 1 == 1 { [[s, 0], [s, s], [0, s]] } else { [[0, 0], [s, 0], [0, s]] };
        let mut v = [[0i32; 2]; 3];
        for k in 0..3 {
            let h = pcg(t.wrapping_mul(3).wrapping_add(k as u32));
            let j = [(h & 0xffff) as i32 - 32768, (h >> 16) as i32 - 32768];
            v[k] = [base[0] + corner[k][0] + j[0] * s / 131072, base[1] + corner[k][1] + j[1] * s / 131072];
        }
        if edge(v[0], v[1], v[2]) < 0 {
            v.swap(1, 2);
        }
        let lo = [v[0][0].min(v[1][0]).min(v[2][0]), v[0][1].min(v[1][1]).min(v[2][1])];
        let hi = [v[0][0].max(v[1][0]).max(v[2][0]), v[0][1].max(v[1][1]).max(v[2][1])];
        let x0 = ((lo[0] - 128 + 255) >> 8).max(0);
        let y0 = ((lo[1] - 128 + 255) >> 8).max(0);
        let x1 = ((hi[0] - 128) >> 8).min(W as i32 - 1);
        let y1 = ((hi[1] - 128) >> 8).min(H as i32 - 1);
        let depth = 0x3f80_0000u32 - (pcg(t ^ 0xa511_e9b3) >> 9);
        let value = (u64::from(depth) << 32) | u64::from(t);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let p = [x * 256 + 128, y * 256 + 128];
                if edge(v[0], v[1], p) >= 0 && edge(v[1], v[2], p) >= 0 && edge(v[2], v[0], p) >= 0 {
                    let i = (y as u32 * W + x as u32) as usize;
                    vis[i] = vis[i].max(value);
                    covered += 1;
                }
            }
        }
    }
    (vis, covered)
}

unsafe fn vbuf() {
    unsafe {
        let gpu = &Gpu::new(&Options::default());
        println!("device: {} driver {}", gpu.name, gpu.driver);
        let image = gpu.image(vk::Format::R64_UINT, W, H, vk::ImageUsageFlags::STORAGE);
        gpu.write_image_slot(0, &image, true);
        let pixels = u64::from(W * H);
        let lin = gpu.buffer(pixels * 8, Mem::Upload);
        let tiled = gpu.buffer(pixels * 8, Mem::Upload);
        let readback = gpu.buffer(pixels * 8, Mem::Readback);
        let roots = gpu.buffer(4096, Mem::Upload);
        let pipes = [
            gpu.compute(&spv!("vbuf_image"), c"raster", &[]),
            gpu.compute(&spv!("vbuf_linear"), c"raster", &[]),
            gpu.compute(&spv!("vbuf_tiled"), c"raster", &[]),
        ];
        for (p, n) in pipes.iter().zip(["image", "linear", "tiled"]) {
            println!("stats {n}:{}", gpu.stats(p.pipeline));
        }
        // PROBE_IMAGE_CLEAR=copy clears the image by a copy from a zero buffer instead of
        // vkCmdClearColorImage (which may leave a fast-clear state the atomics must resolve).
        let zero = gpu.buffer(pixels * 8, Mem::Device);
        let copy_clear = std::env::var("PROBE_IMAGE_CLEAR").is_ok_and(|v| v == "copy");
        let clear = |cmd: vk::CommandBuffer| {
            let range = [color_range()];
            if copy_clear {
                gpu.device.cmd_fill_buffer(cmd, zero.buffer, zero.offset, pixels * 8, 0);
                gpu.full_barrier(cmd);
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(zero.offset)
                    .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                    .image_extent(vk::Extent3D { width: W, height: H, depth: 1 });
                gpu.device.cmd_copy_buffer_to_image(cmd, zero.buffer, image.image, vk::ImageLayout::GENERAL, &[region]);
            } else {
                gpu.device.cmd_clear_color_image(cmd, image.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue { uint32: [0; 4] }, &range);
            }
            gpu.device.cmd_fill_buffer(cmd, lin.buffer, lin.offset, pixels * 8, 0);
            gpu.device.cmd_fill_buffer(cmd, tiled.buffer, tiled.offset, pixels * 8, 0);
        };
        let configs: Vec<(u32, u32)> = std::env::var("PROBE_VBUF")
            .ok()
            .map(|s| s.split(',').map(|c| c.split_once('x').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap()).collect())
            .unwrap_or_else(|| vec![(1, 4), (1, 8), (2, 8), (4, 8)]);
        for (ci, &(cell, layers)) in configs.iter().enumerate() {
            let cells_x = W.div_ceil(cell);
            let cells_y = H.div_ceil(cell);
            let ntri = cells_x * cells_y * 2 * layers;
            let root = VbufRoot { vis: 0, img: 0, width: W, height: H, cell, cells_x, cells_y, layers, ntri, pad: [0; 2] };
            let slot = |k: usize, vis: u64| {
                let off = (ci * 4 + k) * 64;
                std::ptr::write(roots.ptr.add(off).cast::<VbufRoot>(), VbufRoot { vis, ..root });
                roots.addr + off as u64
            };
            let addrs = [slot(0, 0), slot(1, lin.addr), slot(2, tiled.addr)];
            let rec = |k: usize| {
                let p = &pipes[k];
                let a = addrs[k];
                move |cmd: vk::CommandBuffer| {
                    gpu.device.cmd_bind_pipeline(cmd, p.point, p.pipeline);
                    gpu.push_root(cmd, a);
                    gpu.device.cmd_dispatch(cmd, ntri.div_ceil(64), 1, 1);
                }
            };
            let (r0, r1, r2) = (rec(0), rec(1), rec(2));
            let cmd = gpu.begin();
            gpu.bind_heaps(cmd);
            clear(cmd);
            gpu.full_barrier(cmd);
            r0(cmd);
            r1(cmd);
            r2(cmd);
            gpu.full_barrier(cmd);
            copy_image(gpu, cmd, &image, &readback);
            gpu.submit_wait(cmd, "vbuf check");
            let (reference, covered) = raster_ref(&root);
            let img: Vec<u64> = readback.read(pixels as usize);
            let l: Vec<u64> = lin.read(pixels as usize);
            let t: Vec<u64> = tiled.read(pixels as usize);
            let detile = |x: u32, y: u32| t[(((y >> 3) * (W >> 3) + (x >> 3)) * 64 + (y & 7) * 8 + (x & 7)) as usize];
            let mut bad = [0usize; 3];
            for y in 0..H {
                for x in 0..W {
                    let i = (y * W + x) as usize;
                    bad[0] += usize::from(img[i] != reference[i]);
                    bad[1] += usize::from(l[i] != reference[i]);
                    bad[2] += usize::from(detile(x, y) != reference[i]);
                }
            }
            let empty = reference.iter().filter(|&&v| v == 0).count();
            println!(
                "vbuf cell={cell}px layers={layers}: {:.2}M triangles, {:.2} atomics/pixel, {empty} empty pixels; mismatches vs CPU: image={} linear={} tiled={}",
                f64::from(ntri) / 1e6,
                covered as f64 / pixels as f64,
                bad[0],
                bad[1],
                bad[2]
            );
            let times = gpu.bench(&[&r0, &r1, &r2], Some(&clear));
            report(&["image (heap, R64_UINT)", "buffer linear", "buffer 8x8 tiles"], &times);
        }
        let (e, w) = dump_messages();
        println!("validation errors={e} warnings={w}");
    }
}

// ---------------------------------------------------------------------------------------------
// Root as UBO vs pointer.

#[repr(C)]
#[derive(Clone, Copy)]
struct BenchRoot {
    m: [f32; 16],
    p: [[f32; 4]; 16],
    src: [u64; 8],
    out: u64,
    n: u32,
    iters: u32,
    patches_x: u32,
    pad: [u32; 3],
}
const _: () = assert!(size_of::<BenchRoot>() == 416);

unsafe fn root() {
    unsafe {
        let gpu = &Gpu::new(&Options::default());
        println!("device: {} driver {}", gpu.name, gpu.driver);
        let n = W * H;
        let srcs: Vec<Buf> = (0..8u32)
            .map(|k| {
                let b = gpu.buffer(65536 * 4, Mem::Upload);
                let v: Vec<f32> = (0..65536u32).map(|i| (pcg(i + k * 65536) & 0xffff) as f32 / 65535.0).collect();
                b.write(0, &v);
                b
            })
            .collect();
        let outs = [gpu.buffer(u64::from(n) * 4, Mem::Upload), gpu.buffer(u64::from(n) * 4, Mem::Upload)];
        let roots = gpu.buffer(4096, Mem::Upload);
        let a = 0.3f32;
        let (c, s) = (a.cos(), a.sin());
        let patches_x = 256u32;
        let patches = patches_x * 256;
        let scale = 2.0 / (patches_x * 7) as f32;
        let mut p = [[0.0f32; 4]; 16];
        for (i, v) in p.iter_mut().enumerate() {
            *v = [0.1 + i as f32 * 0.01, 0.2, 0.3, 0.4];
        }
        p[0] = [scale, scale, 0.0, 0.0];
        p[1] = [-1.0, -1.0, 0.0, 0.0];
        p[2] = [0.5, 0.0, 0.0, 0.0];
        let base = BenchRoot {
            m: [c, -s * 0.1, 0.0, 0.0, s * 0.1, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            p,
            src: std::array::from_fn(|k| srcs[k].addr),
            out: 0,
            n,
            iters: 64,
            patches_x,
            pad: [0; 3],
        };
        let root_at = |k: usize, out: u64| {
            std::ptr::write(roots.ptr.add(k * 512).cast::<BenchRoot>(), BenchRoot { out, ..base });
            roots.addr + k as u64 * 512
        };
        let r_ubo = root_at(0, outs[0].addr);
        let r_ptr = root_at(1, outs[1].addr);
        for kernel in ["params", "pointers"] {
            let (ubo, ptr) = if kernel == "params" {
                (gpu.compute(&spv!("root_params_ubo"), c"params", &[]), gpu.compute(&spv!("root_params_ptr"), c"params", &[]))
            } else {
                (gpu.compute(&spv!("root_pointers_ubo"), c"pointers", &[]), gpu.compute(&spv!("root_pointers_ptr"), c"pointers", &[]))
            };
            println!("stats {kernel} ubo:{}", gpu.stats(ubo.pipeline));
            println!("stats {kernel} ptr:{}", gpu.stats(ptr.pipeline));
            let r0 = |cmd: vk::CommandBuffer| {
                gpu.device.cmd_bind_pipeline(cmd, ubo.point, ubo.pipeline);
                gpu.push_root(cmd, r_ubo);
                gpu.device.cmd_dispatch(cmd, n.div_ceil(32), 1, 1);
            };
            let r1 = |cmd: vk::CommandBuffer| {
                gpu.device.cmd_bind_pipeline(cmd, ptr.point, ptr.pipeline);
                gpu.push_root(cmd, r_ptr);
                gpu.device.cmd_dispatch(cmd, n.div_ceil(32), 1, 1);
            };
            let cmd = gpu.begin();
            gpu.bind_heaps(cmd);
            r0(cmd);
            r1(cmd);
            gpu.submit_wait(cmd, "root check");
            let a: Vec<u32> = outs[0].read(n as usize);
            let b: Vec<u32> = outs[1].read(n as usize);
            let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let zero = a.iter().filter(|&&x| x == 0).count();
            println!("root {kernel}: {n} threads x 64 iterations; ubo vs ptr mismatches={diff}, zero outputs={zero}");
            report(&["root UBO (PUSH_ADDRESS)", "pointer in push data"], &gpu.bench(&[&r0, &r1], None));
        }
        let target = gpu.image(vk::Format::R32_UINT, W, H, vk::ImageUsageFlags::COLOR_ATTACHMENT);
        let view = gpu.image_view(&target);
        let readback = gpu.buffer(u64::from(n) * 4, Mem::Readback);
        let mesh = [
            gpu.mesh_pipeline(&spv!("root_mesh_ubo"), &spv!("root_pixel_ubo"), vk::Format::R32_UINT),
            gpu.mesh_pipeline(&spv!("root_mesh_ptr"), &spv!("root_pixel_ptr"), vk::Format::R32_UINT),
        ];
        let draw = |k: usize| {
            let p = &mesh[k];
            let r = [r_ubo, r_ptr][k];
            move |cmd: vk::CommandBuffer| {
                let color = [vk::RenderingAttachmentInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue::default())];
                let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: W, height: H } };
                gpu.device.cmd_begin_rendering(cmd, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color));
                gpu.device.cmd_set_viewport(cmd, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: W as f32, height: H as f32, min_depth: 0.0, max_depth: 1.0 }]);
                gpu.device.cmd_set_scissor(cmd, 0, &[area]);
                gpu.device.cmd_bind_pipeline(cmd, p.point, p.pipeline);
                gpu.push_root(cmd, r);
                gpu.mesh.cmd_draw_mesh_tasks(cmd, patches, 1, 1);
                gpu.device.cmd_end_rendering(cmd);
            }
        };
        let (d0, d1) = (draw(0), draw(1));
        let mut images = Vec::new();
        for d in [&d0 as &dyn Fn(vk::CommandBuffer), &d1] {
            let cmd = gpu.begin();
            gpu.bind_heaps(cmd);
            d(cmd);
            gpu.full_barrier(cmd);
            copy_image(gpu, cmd, &target, &readback);
            gpu.submit_wait(cmd, "mesh check");
            images.push(readback.read::<u32>(n as usize));
        }
        let diff = images[0].iter().zip(&images[1]).filter(|(x, y)| x != y).count();
        let lit = images[0].iter().filter(|&&v| v != 0).count();
        println!("root mesh: {patches} groups x 98 triangles, {lit} pixels written; ubo vs ptr mismatches={diff}");
        report(&["root UBO (PUSH_ADDRESS)", "pointer in push data"], &gpu.bench(&[&d0, &d1], None));
        let (e, w) = dump_messages();
        println!("validation errors={e} warnings={w}");
    }
}

// ---------------------------------------------------------------------------------------------
// Cooperative vectors.

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = 1u32 << (shift - 1);
        let r = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | r as u16;
    }
    let r = man + 0xfff + ((man >> 13) & 1);
    let (exp, r) = if r & 0x80_0000 != 0 { (exp + 1, 0) } else { (exp, r) };
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | (r >> 13) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            let mut e = 113u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CvRoot {
    input: u64,
    output: u64,
    weights: u64,
    offset: u32,
    count: u32,
    iters: u32,
    pad: [u32; 3],
}
const _: () = assert!(size_of::<CvRoot>() == 48);

unsafe fn coopvec() {
    unsafe {
        let gpu = &Gpu::new(&Options::default());
        println!("device: {} driver {}", gpu.name, gpu.driver);
        let n = gpu.coopvec_instance.get_physical_device_cooperative_vector_properties_len(gpu.pdev).unwrap();
        let mut props = vec![vk::CooperativeVectorPropertiesNV::default(); n];
        gpu.coopvec_instance.get_physical_device_cooperative_vector_properties(gpu.pdev, &mut props).unwrap();
        println!("cooperative vector combinations: {n}");
        for p in &props {
            println!(
                "  input {:?}/{:?} matrix {:?} bias {:?} result {:?} transpose {}",
                p.input_type, p.input_interpretation, p.matrix_interpretation, p.bias_interpretation, p.result_type, p.transpose
            );
        }
        // Weights: row = output, column = input, row-major FP16.
        let weights: Vec<u16> = (0..256u32).map(|i| f32_to_f16(((pcg(i ^ 0x5bd1_e995) & 0xffff) as f32 / 65535.0 - 0.5) / 8.0)).collect();
        let mut dst_size = 0usize;
        let mut info = vk::ConvertCooperativeVectorMatrixInfoNV {
            src_size: 512,
            src_data: vk::DeviceOrHostAddressConstKHR { host_address: weights.as_ptr().cast() },
            p_dst_size: &mut dst_size,
            dst_data: vk::DeviceOrHostAddressKHR { host_address: std::ptr::null_mut() },
            src_component_type: vk::ComponentTypeKHR::FLOAT16,
            dst_component_type: vk::ComponentTypeKHR::FLOAT16,
            num_rows: 16,
            num_columns: 16,
            src_layout: vk::CooperativeVectorMatrixLayoutNV::ROW_MAJOR,
            src_stride: 32,
            dst_layout: vk::CooperativeVectorMatrixLayoutNV::INFERENCING_OPTIMAL,
            dst_stride: 0,
            ..Default::default()
        };
        gpu.coopvec.convert_cooperative_vector_matrix(&info).expect("size query");
        let mut optimal = vec![0u8; dst_size];
        info.dst_data = vk::DeviceOrHostAddressKHR { host_address: optimal.as_mut_ptr().cast() };
        gpu.coopvec.convert_cooperative_vector_matrix(&info).expect("convert");
        let same_as_row_major = dst_size == 512 && optimal == bytes(&weights);
        println!("InferencingOptimal 16x16 FP16: {dst_size} bytes, identical to row-major: {same_as_row_major}");

        let count = 32 * 1024u32;
        let inputs: Vec<u16> = (0..count * 16).map(|i| f32_to_f16((pcg(i) & 0xffff) as f32 / 65535.0 * 2.0 - 1.0)).collect();
        let input = gpu.buffer(u64::from(count) * 32, Mem::Upload);
        input.write(0, &inputs);
        let output = gpu.buffer(u64::from(count) * 64, Mem::Upload);
        let wbuf = gpu.buffer(4096, Mem::Upload);
        let roots = gpu.buffer(4096, Mem::Upload);
        let mut reference = Vec::with_capacity((count * 16) as usize);
        for v in 0..count as usize {
            for row in 0..16 {
                reference.push((0..16).map(|k| f16_to_f32(weights[row * 16 + k]) * f16_to_f32(inputs[v * 16 + k])).sum::<f32>());
            }
        }
        let pipes = [
            gpu.compute(&spv!("coopvec_asm"), c"main", &[]),
            gpu.compute(&spv!("coopvec_bab"), c"main", &[indirect_mapping(1, 16)]),
        ];
        let forms = ["inline SPIR-V, address in root", "coopVecMatMul, INDIRECT_ADDRESS"];
        // (matrix written at block + place, MatrixOffset operand, root `weights` = block + shift)
        let placements: [(usize, u32, u64, &str); 5] = [
            (0, 0, 0, "address +0, MatrixOffset 0"),
            (16, 16, 0, "address +0, MatrixOffset 16"),
            (64, 64, 0, "address +0, MatrixOffset 64"),
            (1024 + 16, 0, 1024 + 16, "address +16, MatrixOffset 0"),
            (2048 + 64, 0, 2048 + 64, "address +64, MatrixOffset 0"),
        ];
        let mut slot = 0usize;
        for (pi, pipe) in pipes.iter().enumerate() {
            for &(place, offset, shift, label) in &placements {
                std::ptr::write_bytes(wbuf.ptr, 0, 4096);
                wbuf.write(place, &optimal);
                std::ptr::write_bytes(output.ptr, 0xff, (count * 64) as usize);
                let r = CvRoot { input: input.addr, output: output.addr, weights: wbuf.addr + shift, offset, count, iters: 1, pad: [0; 3] };
                std::ptr::write(roots.ptr.add(slot * 64).cast::<CvRoot>(), r);
                let addr = roots.addr + slot as u64 * 64;
                slot += 1;
                let cmd = gpu.begin();
                gpu.bind_heaps(cmd);
                gpu.device.cmd_bind_pipeline(cmd, pipe.point, pipe.pipeline);
                gpu.push_root(cmd, addr);
                gpu.device.cmd_dispatch(cmd, count / 32, 1, 1);
                let res = gpu.try_submit_wait(cmd);
                let got: Vec<f32> = output.read((count * 16) as usize);
                let mut bad = 0usize;
                let mut max_err = 0.0f32;
                for (g, r) in got.iter().zip(&reference) {
                    let err = (g - r).abs();
                    max_err = max_err.max(if err.is_nan() { f32::INFINITY } else { err });
                    if !(err <= 2e-3 + 1e-2 * r.abs()) {
                        bad += 1;
                    }
                }
                println!("coopvec [{}] {label}: submit={res:?} max_abs_err={max_err:.2e} outside FP16 tolerance={bad}/{}", forms[pi], count * 16);
            }
        }
        let big = 1u32 << 20;
        let input2 = gpu.buffer(u64::from(big) * 32, Mem::Device);
        let output2 = gpu.buffer(u64::from(big) * 64, Mem::Device);
        std::ptr::write_bytes(wbuf.ptr, 0, 4096);
        wbuf.write(0, &optimal);
        let r = CvRoot { input: input2.addr, output: output2.addr, weights: wbuf.addr, offset: 0, count: big, iters: 16, pad: [0; 3] };
        std::ptr::write(roots.ptr.add(2048).cast::<CvRoot>(), r);
        let rec = |k: usize| {
            let p = &pipes[k];
            move |cmd: vk::CommandBuffer| {
                gpu.device.cmd_bind_pipeline(cmd, p.point, p.pipeline);
                gpu.push_root(cmd, roots.addr + 2048);
                gpu.device.cmd_dispatch(cmd, big / 32, 1, 1);
            }
        };
        let (r0, r1) = (rec(0), rec(1));
        println!("coopvec timing: {big} vectors x 16 chained matmuls");
        report(&forms, &gpu.bench(&[&r0, &r1], None));
        let (e, w) = dump_messages();
        println!("validation errors={e} warnings={w}");
    }
}

// ---------------------------------------------------------------------------------------------
// Atmosphere LUTs.

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SkyLight {
    dir_lux: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SkyRoot {
    lights: [SkyLight; 8],
    light_count: u32,
    trans_img: u32,
    ms_img: u32,
    sky_img: u32,
    trans_tex: u32,
    ms_tex: u32,
    sampler: u32,
    sky_w: u32,
    sky_h: u32,
    steps: u32,
    altitude: f32,
    pad: u32,
}

unsafe fn sky() {
    unsafe {
        let gpu = &Gpu::new(&Options::default());
        println!("device: {} driver {}", gpu.name, gpu.driver);
        let f = vk::Format::R32G32B32A32_SFLOAT;
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        let trans = gpu.image(f, 256, 128, usage);
        let ms = gpu.image(f, 64, 32, usage);
        let sky_img = gpu.image(f, 400, 200, usage);
        gpu.write_image_slot(0, &trans, true);
        gpu.write_image_slot(1, &ms, true);
        gpu.write_image_slot(2, &sky_img, true);
        gpu.write_image_slot(3, &trans, false);
        gpu.write_image_slot(4, &ms, false);
        let p_t = gpu.compute(&spv!("sky_transmittance"), c"transmittance", &[]);
        let p_m = gpu.compute(&spv!("sky_multiscatter"), c"multiscatter", &[]);
        let p_s = gpu.compute(&spv!("sky_skyview"), c"skyview", &[]);
        for (n, p) in [("transmittance", &p_t), ("multiscatter", &p_m), ("skyview", &p_s)] {
            println!("stats {n}:{}", gpu.stats(p.pipeline));
        }
        let roots = gpu.buffer(16384, Mem::Upload);
        let readback = gpu.buffer(400 * 200 * 16, Mem::Readback);
        let light = |el: f32, az: f32, lux: f32| {
            let (el, az) = (el.to_radians(), az.to_radians());
            SkyLight { dir_lux: [el.cos() * az.cos(), el.sin(), el.cos() * az.sin(), lux], color: [1.0, 0.98, 0.95, 1.0] }
        };
        let all = [
            light(30.0, 0.0, 120000.0),
            light(8.0, 140.0, 60000.0),
            light(45.0, 250.0, 0.3),
            light(15.0, 300.0, 0.1),
            light(60.0, 60.0, 0.05),
            light(3.0, 200.0, 0.05),
            light(20.0, 100.0, 0.02),
            light(70.0, 330.0, 0.01),
        ];
        let make = |slot: usize, count: u32, w: u32, h: u32, only: Option<usize>| {
            let mut r = SkyRoot {
                light_count: count,
                trans_img: 0,
                ms_img: 1,
                sky_img: 2,
                trans_tex: 3,
                ms_tex: 4,
                sampler: 0,
                sky_w: w,
                sky_h: h,
                steps: 32,
                altitude: 200.0,
                ..Default::default()
            };
            match only {
                Some(i) => r.lights[0] = all[i],
                None => r.lights = all,
            }
            std::ptr::write(roots.ptr.add(slot * 512).cast::<SkyRoot>(), r);
            roots.addr + slot as u64 * 512
        };
        let dispatch = |cmd: vk::CommandBuffer, p: &Pipeline, root: u64, x: u32, y: u32| {
            gpu.device.cmd_bind_pipeline(cmd, p.point, p.pipeline);
            gpu.push_root(cmd, root);
            gpu.device.cmd_dispatch(cmd, x, y, 1);
        };
        let base_root = make(0, 4, 400, 200, None);
        let luts = |cmd: vk::CommandBuffer| {
            dispatch(cmd, &p_t, base_root, 256 / 8, 128 / 4);
            gpu.full_barrier(cmd);
            dispatch(cmd, &p_m, base_root, 64, 32);
            gpu.full_barrier(cmd);
        };
        let read_sky = |root: u64| -> Vec<[f32; 4]> {
            let cmd = gpu.begin();
            gpu.bind_heaps(cmd);
            luts(cmd);
            dispatch(cmd, &p_s, root, 400 / 8, 200 / 4);
            gpu.full_barrier(cmd);
            copy_image(gpu, cmd, &sky_img, &readback);
            gpu.submit_wait(cmd, "sky check");
            readback.read(400 * 200)
        };
        let both = read_sky(make(1, 2, 400, 200, None));
        let a = read_sky(make(2, 1, 400, 200, Some(0)));
        let b = read_sky(make(3, 1, 400, 200, Some(1)));
        let mut max_rel = 0.0f32;
        let mut finite = true;
        for i in 0..both.len() {
            for c in 0..3 {
                let s = a[i][c] + b[i][c];
                finite &= both[i][c].is_finite();
                if s > 1e-3 {
                    max_rel = max_rel.max((both[i][c] - s).abs() / s);
                }
            }
        }
        let zenith = both[199 * 400];
        let horizon = both[100 * 400 + 200];
        println!(
            "sky check: finite={finite}; two-light LUT vs sum of one-light LUTs max_rel={max_rel:.2e}; zenith {:.0} {:.0} {:.0} nits, horizon {:.0} {:.0} {:.0}",
            zenith[0], zenith[1], zenith[2], horizon[0], horizon[1], horizon[2]
        );
        let p_s = &p_s;
        let dispatch = &dispatch;
        let sv = |root: u64, w: u32, h: u32| move |cmd: vk::CommandBuffer| dispatch(cmd, p_s, root, w.div_ceil(8), h.div_ceil(4));
        let r1 = sv(make(10, 1, 400, 200, None), 400, 200);
        let r2 = sv(make(11, 2, 400, 200, None), 400, 200);
        let r4 = sv(make(12, 4, 400, 200, None), 400, 200);
        let r8 = sv(make(13, 8, 400, 200, None), 400, 200);
        let r4s = sv(make(14, 4, 200, 100, None), 200, 100);
        let r2s = sv(make(15, 2, 200, 100, None), 200, 100);
        let r_t = |cmd: vk::CommandBuffer| dispatch(cmd, &p_t, base_root, 256 / 8, 128 / 4);
        let r_m = |cmd: vk::CommandBuffer| dispatch(cmd, &p_m, base_root, 64, 32);
        let r_all = |cmd: vk::CommandBuffer| {
            luts(cmd);
            r4(cmd);
        };
        let names = [
            "sky-view 400x200, 1 light",
            "sky-view 400x200, 2 lights",
            "sky-view 400x200, 4 lights",
            "sky-view 400x200, 8 lights",
            "sky-view 200x100, 4 lights",
            "sky-view 200x100, 2 lights",
            "transmittance 256x128",
            "multiple scattering 64x32",
            "all three, 4 lights",
        ];
        let cmd = gpu.begin();
        gpu.bind_heaps(cmd);
        luts(cmd);
        gpu.submit_wait(cmd, "prime");
        report(&names, &gpu.bench(&[&r1, &r2, &r4, &r8, &r4s, &r2s, &r_t, &r_m, &r_all], None));
        let (e, w) = dump_messages();
        println!("validation errors={e} warnings={w}");
    }
}

/// Repeats the debug-info sky passes for PROBE_CAPTURE_S seconds (8), one frame per submission
/// with a VK_EXT_frame_boundary marker, for an Nsight Graphics capture of a heap device.
unsafe fn capture() {
    unsafe {
        let gpu = &Gpu::new(&Options { frame_boundary: true, ..Default::default() });
        println!("device: {} driver {}; VK_EXT_frame_boundary: {}", gpu.name, gpu.driver, gpu.frame_boundary);
        let f = vk::Format::R32G32B32A32_SFLOAT;
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        let trans = gpu.image(f, 256, 128, usage);
        let ms = gpu.image(f, 64, 32, usage);
        let sky_img = gpu.image(f, 400, 200, usage);
        for (slot, img, storage) in [(0, &trans, true), (1, &ms, true), (2, &sky_img, true), (3, &trans, false), (4, &ms, false)] {
            gpu.write_image_slot(slot, img, storage);
        }
        let pipes = [
            (gpu.compute(&spv!("sky_g_transmittance"), c"transmittance", &[]), 256 / 8, 128 / 4),
            (gpu.compute(&spv!("sky_g_multiscatter"), c"multiscatter", &[]), 64, 32),
            (gpu.compute(&spv!("sky_g_skyview"), c"skyview", &[]), 400 / 8, 200 / 4),
        ];
        let roots = gpu.buffer(512, Mem::Upload);
        let mut r = SkyRoot { light_count: 1, trans_img: 0, ms_img: 1, sky_img: 2, trans_tex: 3, ms_tex: 4, sky_w: 400, sky_h: 200, steps: 32, altitude: 200.0, ..Default::default() };
        r.lights[0] = SkyLight { dir_lux: [0.0, 0.5, 0.866, 120000.0], color: [1.0; 4] };
        std::ptr::write(roots.ptr.cast::<SkyRoot>(), r);
        let until = std::time::Instant::now() + std::time::Duration::from_secs(u64::from(env_u32("PROBE_CAPTURE_S", 8)));
        let mut frame = 0u64;
        while std::time::Instant::now() < until {
            let cmd = gpu.begin();
            gpu.bind_heaps(cmd);
            for (p, x, y) in &pipes {
                gpu.device.cmd_bind_pipeline(cmd, p.point, p.pipeline);
                gpu.push_root(cmd, roots.addr);
                gpu.device.cmd_dispatch(cmd, *x, *y, 1);
                gpu.full_barrier(cmd);
            }
            gpu.submit_frame(cmd, frame);
            frame += 1;
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
        println!("frames {frame}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("vbuf") => unsafe { vbuf() },
        Some("root") => unsafe { root() },
        Some("coopvec") => unsafe { coopvec() },
        Some("sky") => unsafe { sky() },
        Some("capture") => unsafe { capture() },
        Some("crash") => unsafe { crash::run(args.get(2).map(String::as_str).unwrap_or("stripped")) },
        _ => println!("usage: kernels vbuf|root|coopvec|sky|capture|crash [full|stripped]"),
    }
}
