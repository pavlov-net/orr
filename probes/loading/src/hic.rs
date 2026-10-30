//! Case 2: host image copy. Worker throughput into BC7 images (whole mips and 64 KiB tiles),
//! the per-format host copy performance query, and whether HOST_TRANSFER usage or concurrent
//! sharing changes an image's memory requirements or its sampling cost.

use crate::gpu::{Gpu, Mem, RENDER, Submitter};
use crate::{Rng, Work, row, run_workers, with_frame};
use ash::vk::{self, TaggedStructure};
use std::time::Duration;

const SIZE: u32 = 4096;
const MIPS: u32 = 13;

fn mip_bytes(fmt: vk::Format, size: u32, level: u32) -> usize {
    let s = (size >> level).max(1);
    match fmt {
        vk::Format::R8G8B8A8_UNORM => (s * s * 4) as usize,
        _ => (s.div_ceil(4) * s.div_ceil(4) * 16) as usize,
    }
}

pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub req: vk::MemoryRequirements,
    pub ty: u32,
}

pub unsafe fn image(gpu: &Gpu, fmt: vk::Format, size: u32, mips: u32, host: bool, concurrent: bool) -> Image {
    let mut usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    if host {
        usage |= vk::ImageUsageFlags::HOST_TRANSFER;
    }
    let fam = gpu.families;
    let mut ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(fmt)
        .extent(vk::Extent3D { width: size, height: size, depth: 1 })
        .mip_levels(mips)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    ci = if concurrent { ci.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&fam) } else { ci.sharing_mode(vk::SharingMode::EXCLUSIVE) };
    let image = gpu.device.create_image(&ci, None).unwrap();
    let req = gpu.device.get_image_memory_requirements(image);
    // HOST_TRANSFER images may exclude the plain device-local type; take Upload then.
    let ty = gpu
        .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::HOST_VISIBLE)
        .unwrap_or_else(|| gpu.kind_type(req.memory_type_bits, Mem::Upload));
    let memory = gpu.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None).unwrap();
    gpu.device.bind_image_memory(image, memory, 0).unwrap();
    Image { image, memory, req, ty }
}

fn color(mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: mips, base_array_layer: 0, layer_count: 1 }
}

fn layers(level: u32) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: level, base_array_layer: 0, layer_count: 1 }
}

pub unsafe fn host_to_general(gpu: &Gpu, img: vk::Image, mips: u32) {
    let t = [vk::HostImageLayoutTransitionInfo::default()
        .image(img)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(color(mips))];
    gpu.device.transition_image_layout(&t).unwrap();
}

pub unsafe fn host_copy(gpu: &Gpu, img: vk::Image, level: u32, offset: (i32, i32), extent: (u32, u32), data: &[u8]) {
    let r = [vk::MemoryToImageCopy::default()
        .host_pointer(data.as_ptr().cast())
        .image_subresource(layers(level))
        .image_offset(vk::Offset3D { x: offset.0, y: offset.1, z: 0 })
        .image_extent(vk::Extent3D { width: extent.0, height: extent.1, depth: 1 })];
    gpu.device
        .copy_memory_to_image(&vk::CopyMemoryToImageInfo::default().dst_image(img).dst_image_layout(vk::ImageLayout::GENERAL).regions(&r))
        .unwrap();
}

fn texels(fmt: vk::Format, seed: u64) -> Vec<Vec<u8>> {
    let mut r = Rng(seed);
    (0..MIPS)
        .map(|l| {
            let n = mip_bytes(fmt, SIZE, l);
            let mut v = Vec::with_capacity(n);
            while v.len() < n {
                if fmt == vk::Format::R8G8B8A8_UNORM {
                    v.extend_from_slice(&(r.next() as u32 | 0xff00_0000).to_le_bytes());
                } else {
                    // BC7 mode 6 blocks with random endpoints and indices.
                    let mut b = r.next().to_le_bytes();
                    b[0] = 0x40;
                    v.extend_from_slice(&b);
                    v.extend_from_slice(&r.next().to_le_bytes());
                }
            }
            v.truncate(n);
            v
        })
        .collect()
}

pub unsafe fn run(gpu: &'static Gpu) {
    perf_query(gpu);
    sampling(gpu);
    throughput(gpu);
}

unsafe fn perf_query(gpu: &Gpu) {
    let formats = [
        vk::Format::BC7_SRGB_BLOCK,
        vk::Format::BC7_UNORM_BLOCK,
        vk::Format::BC6H_UFLOAT_BLOCK,
        vk::Format::BC5_UNORM_BLOCK,
        vk::Format::BC4_UNORM_BLOCK,
        vk::Format::R16_UNORM,
        vk::Format::R8_UNORM,
        vk::Format::R8G8B8A8_SRGB,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::Format::R32_SFLOAT,
    ];
    for fmt in formats {
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::HOST_TRANSFER;
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(fmt)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage);
        let mut q = vk::HostImageCopyDevicePerformanceQuery::default();
        let mut out = vk::ImageFormatProperties2::default().push(&mut q);
        let r = gpu.instance.get_physical_device_image_format_properties2(gpu.pdev, &info, &mut out);
        let fp = gpu.instance.get_physical_device_format_properties(gpu.pdev, fmt);
        let host_feature = {
            let mut f3 = vk::FormatProperties3::default();
            let mut f2 = vk::FormatProperties2::default().push(&mut f3);
            gpu.instance.get_physical_device_format_properties2(gpu.pdev, fmt, &mut f2);
            f3.optimal_tiling_features.contains(vk::FormatFeatureFlags2::HOST_IMAGE_TRANSFER)
        };
        let _ = fp;
        row(
            &format!("perf query {fmt:?}"),
            match r {
                Ok(()) => format!(
                    "optimalDeviceAccess={} identicalMemoryLayout={} HOST_IMAGE_TRANSFER feature={host_feature}",
                    q.optimal_device_access == vk::TRUE,
                    q.identical_memory_layout == vk::TRUE
                ),
                Err(e) => format!("{e:?}"),
            },
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SampleRoot {
    dst: u64,
    width: u32,
    height: u32,
    taps: u32,
    pad: u32,
}

#[derive(Clone, Copy, Debug)]
struct Var {
    host: bool,
    concurrent: bool,
    layout: vk::ImageLayout,
}

unsafe fn sampling(gpu: &Gpu) {
    let vars = [
        Var { host: false, concurrent: false, layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL },
        Var { host: false, concurrent: false, layout: vk::ImageLayout::GENERAL },
        Var { host: false, concurrent: true, layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL },
        Var { host: true, concurrent: false, layout: vk::ImageLayout::GENERAL },
        Var { host: true, concurrent: true, layout: vk::ImageLayout::GENERAL },
    ];
    let pipe = gpu.pipeline(&crate::spv!("sample"), gpu.tex_layout);
    let sampler = gpu
        .device
        .create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )
        .unwrap();
    let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 16 }];
    let pool = gpu.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(16).pool_sizes(&sizes), None).unwrap();
    let (w, h, taps) = (1920u32, 1080u32, 32u32);
    let mut sub = Submitter::new(gpu, RENDER, 2, 4);
    for fmt in [vk::Format::BC7_UNORM_BLOCK, vk::Format::R8G8B8A8_UNORM] {
        let data = texels(fmt, 7);
        let staging_len: usize = data.iter().map(|d| d.len()).sum();
        let staging = gpu.buffer(staging_len as u64, Mem::Upload);
        let mut off = 0;
        for d in &data {
            staging.write(off, d);
            off += d.len();
        }
        let mut sets = Vec::new();
        let mut imgs = Vec::new();
        let mut outs = Vec::new();
        for v in vars {
            let img = image(gpu, fmt, SIZE, MIPS, v.host, v.concurrent);
            if v.host {
                host_to_general(gpu, img.image, MIPS);
                for (l, d) in data.iter().enumerate() {
                    let s = (SIZE >> l).max(1);
                    host_copy(gpu, img.image, l as u32, (0, 0), (s, s), d);
                }
            } else {
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
                            let s = (SIZE >> l).max(1);
                            let r = vk::BufferImageCopy { buffer_offset: off, image_subresource: layers(l), image_extent: vk::Extent3D { width: s, height: s, depth: 1 }, ..Default::default() };
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
            row(
                &format!("requirements {fmt:?} host={} concurrent={}", v.host, v.concurrent),
                format!("size {} align {} typeBits {:#x} (type {})", img.req.size, img.req.alignment, img.req.memory_type_bits, img.ty),
            );
            let view = gpu
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default().image(img.image).view_type(vk::ImageViewType::TYPE_2D).format(fmt).subresource_range(color(MIPS)),
                    None,
                )
                .unwrap();
            let lay = [gpu.tex_set_layout];
            let set = gpu.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&lay)).unwrap()[0];
            let ii = [vk::DescriptorImageInfo { sampler, image_view: view, image_layout: v.layout }];
            let wr = [vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ii)];
            gpu.device.update_descriptor_sets(&wr, &[]);
            sets.push(set);
            imgs.push(img);
            outs.push(gpu.buffer(u64::from(w * h) * 4, Mem::Readback));
        }
        let mut t: Vec<Vec<f64>> = vec![Vec::new(); vars.len()];
        for it in 0..crate::iters(42) {
            for k in 0..vars.len() {
                let root = SampleRoot { dst: outs[k].addr, width: w, height: h, taps, pad: 0 };
                let cmd = sub.begin();
                gpu.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                gpu.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, gpu.tex_layout, 0, &[sets[k]], &[]);
                gpu.push(cmd, gpu.tex_layout, &root);
                gpu.device.cmd_dispatch(cmd, w.div_ceil(16), h.div_ceil(16), 1);
                gpu.barrier(cmd);
                sub.stamp(cmd, 0);
                gpu.device.cmd_dispatch(cmd, w.div_ceil(16), h.div_ceil(16), 1);
                sub.stamp(cmd, 1);
                gpu.barrier(cmd);
                let v = sub.submit(cmd);
                let ms = sub.ms(v, 0, 1);
                if it > 0 {
                    t[k].push(ms);
                }
                std::thread::sleep(Duration::from_millis(6));
            }
        }
        let reference = std::slice::from_raw_parts(outs[0].ptr, (w * h * 4) as usize).to_vec();
        let base = crate::median(&mut t[0].clone());
        for (k, v) in vars.iter().enumerate() {
            let m = crate::median(&mut t[k]);
            let same = std::slice::from_raw_parts(outs[k].ptr, (w * h * 4) as usize) == &reference[..];
            row(
                &format!("sampling {fmt:?} host={} concurrent={} layout={:?}", v.host, v.concurrent, v.layout),
                format!("{m:.4} ms ({:+.2}% vs plain exclusive READ_ONLY); output equals reference: {same}", (m / base - 1.0) * 100.0),
            );
            if !same {
                crate::fail("sampled texels differ between upload paths");
            }
        }
        for img in imgs {
            gpu.device.destroy_image(img.image, None);
            gpu.device.free_memory(img.memory, None);
        }
        for o in outs {
            gpu.free(o);
        }
        gpu.free(staging);
        gpu.device.reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
    }
}

unsafe fn throughput(gpu: &'static Gpu) {
    let fmt = vk::Format::BC7_UNORM_BLOCK;
    let data = texels(fmt, 11);
    let tile: Vec<u8> = data[0][..64 << 10].to_vec();
    let imgs: Vec<Image> = (0..8)
        .map(|_| {
            let i = image(gpu, fmt, SIZE, MIPS, true, true);
            host_to_general(gpu, i.image, MIPS);
            i
        })
        .collect();
    let handles: Vec<u64> = imgs.iter().map(|i| ash::vk::Handle::as_raw(i.image)).collect();
    let data = &data;
    let tile = &tile;
    let handles = &handles;
    with_frame(gpu, |frame| {
        for tiles in [false, true] {
            for workers in [1usize, 4, 8] {
                let (delta, works) = frame.ab(2, Duration::from_millis(1500), |win| {
                    let step = |w: usize, i: u64| -> u64 {
                        let img: vk::Image = ash::vk::Handle::from_raw(handles[w]);
                        if tiles {
                            // 256x256 BC7 texels = one 64 KiB tile of mip 0.
                            let t = (i * 2654435761) % 256;
                            let (x, y) = ((t % 16) as i32 * 256, (t / 16) as i32 * 256);
                            host_copy(gpu, img, 0, (x, y), (256, 256), tile);
                            64 << 10
                        } else {
                            let l = (i % 3) as u32; // mips 0-2 carry 99% of the bytes
                            let s = SIZE >> l;
                            host_copy(gpu, img, l, (0, 0), (s, s), &data[l as usize]);
                            data[l as usize].len() as u64
                        }
                    };
                    run_workers(workers, win, &step)
                });
                let w = Work::sum(&works);
                row(&format!("host copy BC7 {} x{workers}", if tiles { "64 KiB tiles" } else { "whole mips 0-2" }), format!("{}; {}", w.text(), delta.text()));
            }
        }
    });
    // Read back mip 1 of image 0 and a tile region; both must equal the source.
    let img = imgs[0].image;
    host_copy(gpu, img, 1, (0, 0), (SIZE / 2, SIZE / 2), &data[1]);
    host_copy(gpu, img, 0, (512, 768), (256, 256), tile);
    let mut back = vec![0u8; data[1].len()];
    let read = |level: u32, off: (i32, i32), ext: (u32, u32), out: &mut [u8]| {
        let r = [vk::ImageToMemoryCopy::default()
            .host_pointer(out.as_mut_ptr().cast())
            .image_subresource(layers(level))
            .image_offset(vk::Offset3D { x: off.0, y: off.1, z: 0 })
            .image_extent(vk::Extent3D { width: ext.0, height: ext.1, depth: 1 })];
        gpu.device
            .copy_image_to_memory(&vk::CopyImageToMemoryInfo::default().src_image(img).src_image_layout(vk::ImageLayout::GENERAL).regions(&r))
            .unwrap();
    };
    read(1, (0, 0), (SIZE / 2, SIZE / 2), &mut back);
    let mip_ok = back == data[1];
    let mut tb = vec![0u8; 64 << 10];
    read(0, (512, 768), (256, 256), &mut tb);
    let tile_ok = &tb == tile;
    row("host copy readback equals source (mip 1, tile)", format!("{mip_ok}, {tile_ok}"));
    if !(mip_ok && tile_ok) {
        crate::fail("host copy readback differs");
    }
    for i in imgs {
        gpu.device.destroy_image(i.image, None);
        gpu.device.free_memory(i.memory, None);
    }
}
