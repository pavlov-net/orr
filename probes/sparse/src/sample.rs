//! Case 3a: sampling cost of sparse BC7 images against the same images non-sparse. Sixteen
//! 4096^2 BC7 images per variant, the same texels in every variant; a resolve-style kernel
//! (four anisotropic SampleGrad taps per pixel at 2560x1440, over 4 images or over 16 images
//! as four materials) and PRB-1's random-walk kernel.
//! Sparse images take no host-transfer usage (621.01 reports that combination unsupported), so
//! they are filled by a queue copy; "whole-mip, staged" is the same image without sparse.

use crate::gpu::{Gpu, Mem, RENDER, SET_IMAGES, Submitter};
use crate::tex::{self, BC7, Image, Kind, color, layers, mip_size};
use crate::{Rng, row};
use ash::vk;
use std::time::Duration;

const SIZE: u32 = 4096;
const MIPS: u32 = 13;

#[derive(Clone, Copy)]
struct Var {
    name: &'static str,
    kind: Kind,
    /// Host image copy (else a staged queue copy).
    host_copy: bool,
    /// Tile slots shuffled in the backing block (sparse only).
    shuffle: bool,
    /// The tile path's LOD clamp from a residency map.
    clamp: bool,
    layout: vk::ImageLayout,
}

const VARS: [Var; 7] = [
    Var { name: "plain (exclusive, device-local, staged, READ_ONLY)", kind: Kind { sparse: false, host: false, concurrent: false }, host_copy: false, shuffle: false, clamp: false, layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL },
    Var { name: "whole-mip, host copy (host-transfer, concurrent, GENERAL)", kind: Kind { sparse: false, host: true, concurrent: true }, host_copy: true, shuffle: false, clamp: false, layout: vk::ImageLayout::GENERAL },
    Var { name: "whole-mip, staged (concurrent, device-local, GENERAL)", kind: Kind { sparse: false, host: false, concurrent: true }, host_copy: false, shuffle: false, clamp: false, layout: vk::ImageLayout::GENERAL },
    Var { name: "sparse tiles (concurrent, device-local, staged, shuffled slots, GENERAL)", kind: Kind { sparse: true, host: false, concurrent: true }, host_copy: false, shuffle: true, clamp: false, layout: vk::ImageLayout::GENERAL },
    Var { name: "sparse tiles + residency-map LOD clamp", kind: Kind { sparse: true, host: false, concurrent: true }, host_copy: false, shuffle: true, clamp: true, layout: vk::ImageLayout::GENERAL },
    Var { name: "sparse tiles, contiguous slots", kind: Kind { sparse: true, host: false, concurrent: true }, host_copy: false, shuffle: false, clamp: false, layout: vk::ImageLayout::GENERAL },
    Var { name: "plain, second copy (placement spread)", kind: Kind { sparse: false, host: false, concurrent: false }, host_copy: false, shuffle: false, clamp: false, layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL },
];

#[repr(C)]
#[derive(Clone, Copy)]
struct Root {
    dst: u64,
    residency: u64,
    width: u32,
    height: u32,
    /// `clamp` for the resolve kernel, `taps` for the walk kernel.
    arg: u32,
    pad: u32,
}

/// BC7 mode 6 blocks with random endpoints and indices, per mip.
fn texels(seed: u64) -> Vec<Vec<u8>> {
    let mut r = Rng(seed);
    (0..MIPS)
        .map(|l| {
            let (w, h) = mip_size(SIZE, SIZE, l);
            let n = BC7.mip_bytes(w, h);
            let mut v = Vec::with_capacity(n);
            while v.len() < n {
                let mut b = r.next().to_le_bytes();
                b[0] = 0x40;
                v.extend_from_slice(&b);
                v.extend_from_slice(&r.next().to_le_bytes());
            }
            v
        })
        .collect()
}

unsafe fn upload(gpu: &Gpu, sub: &mut Submitter, img: &Image, v: Var, data: &[Vec<u8>], staging: &crate::gpu::Buf) {
    if v.host_copy {
        tex::host_to_general(gpu, img.image, MIPS);
        for (l, d) in data.iter().enumerate() {
            let (w, h) = mip_size(SIZE, SIZE, l as u32);
            let r = [vk::MemoryToImageCopy::default().host_pointer(d.as_ptr().cast()).image_subresource(layers(l as u32)).image_extent(vk::Extent3D { width: w, height: h, depth: 1 })];
            gpu.device
                .copy_memory_to_image(&vk::CopyMemoryToImageInfo::default().dst_image(img.image).dst_image_layout(vk::ImageLayout::GENERAL).regions(&r))
                .unwrap();
        }
        return;
    }
    sub.run(|cmd| {
        let to_dst = [vk::ImageMemoryBarrier2::default()
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img.image)
            .subresource_range(color(MIPS))];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_dst));
        let mut off = 0u64;
        let regions: Vec<_> = (0..MIPS)
            .map(|l| {
                let (w, h) = mip_size(SIZE, SIZE, l);
                let r = vk::BufferImageCopy { buffer_offset: off, image_subresource: layers(l), image_extent: vk::Extent3D { width: w, height: h, depth: 1 }, ..Default::default() };
                off += data[l as usize].len() as u64;
                r
            })
            .collect();
        gpu.device.cmd_copy_buffer_to_image(cmd, staging.buffer, img.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
        let to_read = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(v.layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img.image)
            .subresource_range(color(MIPS))];
        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_read));
    });
}

pub unsafe fn run(gpu: &Gpu) {
    let resolve = gpu.pipeline(&crate::spv!("resolve"), gpu.tex_layout);
    let walk = gpu.pipeline(&crate::spv!("walk"), gpu.tex_layout);
    let sampler = gpu
        .device
        .create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(8.0)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )
        .unwrap();
    let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 128 }];
    let pool = gpu.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(16).pool_sizes(&sizes), None).unwrap();
    let mut sub = Submitter::new(gpu, RENDER, 2, 4);
    let data: Vec<Vec<Vec<u8>>> = (0..SET_IMAGES as u64).map(|i| texels(7 + i)).collect();
    let staging: Vec<_> = data
        .iter()
        .map(|d| {
            let b = gpu.buffer(d.iter().map(|m| m.len() as u64).sum(), Mem::Upload);
            let mut off = 0;
            for m in d {
                b.write(off, m);
                off += m.len();
            }
            b
        })
        .collect();
    // Residency map: per texture, 16x16 tiles of mip 0, min resident LOD 0 (all resident).
    let residency = gpu.buffer(u64::from(SET_IMAGES) * 256 * 4, Mem::Device);
    sub.run(|cmd| {
        gpu.device.cmd_fill_buffer(cmd, residency.buffer, 0, vk::WHOLE_SIZE, 0);
        gpu.barrier(cmd);
    });
    let mut sets = Vec::new();
    let mut imgs = Vec::new();
    for (vi, v) in VARS.iter().enumerate() {
        let mut infos = Vec::new();
        for (i, d) in data.iter().enumerate() {
            let mut img = tex::create(gpu, BC7, SIZE, SIZE, MIPS, 1, v.kind);
            if v.kind.sparse {
                tex::bind_all(gpu, &mut img, v.shuffle.then_some(31 + i as u64));
            } else {
                tex::bind_dedicated(gpu, &mut img);
            }
            upload(gpu, &mut sub, &img, *v, d, &staging[i]);
            if i == 0 {
                let tail = img.sreq.map_or("-".into(), |s| format!("tail at mip {} ({} B)", s.image_mip_tail_first_lod, s.image_mip_tail_size));
                row(
                    &format!("variant {vi}: {}", v.name),
                    format!("memory type {}, size {} align {} typeBits {:#x}, {tail}", tex::image_type(gpu, img.req.memory_type_bits), img.req.size, img.req.alignment, img.req.memory_type_bits),
                );
            }
            let view = gpu
                .device
                .create_image_view(&vk::ImageViewCreateInfo::default().image(img.image).view_type(vk::ImageViewType::TYPE_2D).format(BC7.fmt).subresource_range(color(MIPS)), None)
                .unwrap();
            infos.push(vk::DescriptorImageInfo { sampler, image_view: view, image_layout: v.layout });
            imgs.push(img);
        }
        let lay = [gpu.tex_set_layout];
        let set = gpu.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&lay)).unwrap()[0];
        let wr = [vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&infos)];
        gpu.device.update_descriptor_sets(&wr, &[]);
        sets.push(set);
    }
    let kernels = [
        ("resolve 2560x1440, 4 images, 4 aniso SampleGrad taps", resolve, (2560u32, 1440u32), 8u32, 0u32),
        ("resolve 2560x1440, 16 images as 4 materials per 32 px block, 4 aniso SampleGrad taps", resolve, (2560, 1440), 8, 2),
        ("walk 1920x1080, 1 image, 32 random taps", walk, (1920, 1080), 16, 0),
        ("control: resolve 2560x1440 without samples", resolve, (2560, 1440), 8, 4),
    ];
    for (kname, pipe, (w, h), groups, flags) in kernels {
        // The warm dispatch writes host memory for the comparison; the timed one device memory.
        let outs: Vec<_> = VARS.iter().map(|_| gpu.buffer(u64::from(w * h) * 4, Mem::Readback)).collect();
        let timed = gpu.buffer(u64::from(w * h) * 4, Mem::Device);
        let mut t: Vec<Vec<f64>> = vec![Vec::new(); VARS.len()];
        for it in 0..crate::iters(62) {
            for (k, v) in VARS.iter().enumerate() {
                let arg = if pipe == walk { 32 } else { flags | u32::from(v.clamp) };
                let root = Root { dst: outs[k].addr, residency: residency.addr, width: w, height: h, arg, pad: 0 };
                let cmd = sub.begin();
                gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                gpu.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, gpu.tex_layout, 0, &[sets[k]], &[]);
                gpu.push(cmd, gpu.tex_layout, &root);
                gpu.device.cmd_dispatch(cmd, w.div_ceil(groups), h.div_ceil(groups), 1);
                gpu.barrier(cmd);
                gpu.push(cmd, gpu.tex_layout, &Root { dst: timed.addr, ..root });
                sub.stamp(cmd, 0);
                gpu.device.cmd_dispatch(cmd, w.div_ceil(groups), h.div_ceil(groups), 1);
                sub.stamp(cmd, 1);
                gpu.barrier(cmd);
                let val = sub.submit(cmd);
                let ms = sub.ms(val, 0, 1);
                if it > 0 {
                    t[k].push(ms);
                }
                std::thread::sleep(Duration::from_millis(6));
            }
        }
        let reference = std::slice::from_raw_parts(outs[0].ptr, (w * h * 4) as usize).to_vec();
        let base = crate::median(&mut t[0].clone());
        let whole = crate::median(&mut t[2].clone());
        for (k, v) in VARS.iter().enumerate() {
            if (pipe == walk && v.clamp) || (flags == 4 && k > 0) {
                continue;
            }
            let m = crate::median(&mut t[k]);
            let same = std::slice::from_raw_parts(outs[k].ptr, (w * h * 4) as usize) == &reference[..];
            row(
                &format!("{kname}: variant {k}"),
                format!("{m:.4} ms ({:+.2}% vs plain, {:+.2}% vs whole-mip staged); output equals plain: {same}", (m / base - 1.0) * 100.0, (m / whole - 1.0) * 100.0),
            );
            if !same && flags != 4 {
                crate::fail("sampled output differs between variants");
            }
        }
    }
}
