//! Case 4: the two NVIDIA 617.14 issues, rerun on the installed driver.
//! - Stale address-based image copy: upload by vkCmdCopyMemoryToImageKHR, transfer barrier,
//!   readback by vkCmdCopyImageToMemoryKHR in the same submission, no timeline wait between.
//!   Buffer-handle commands are the control.
//! - Three queues (render, loading, copy) driven by three threads, each copying
//!   upload -> scratch -> readback 256 times per launch; one fresh process per launch.

use crate::gpu::{Buf, Gpu, Mem, RENDER, Submitter};
use crate::row;
use ash::vk;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Rgba2d,
    Bc7,
    Rgba3d,
    Array,
    Cube,
}

struct Target {
    image: vk::Image,
    memory: vk::DeviceMemory,
    extent: vk::Extent3D,
    layers: u32,
    bytes: u64,
}

unsafe fn target(gpu: &Gpu, kind: Kind) -> Target {
    let (fmt, ty, extent, layers, flags) = match kind {
        Kind::Rgba2d => (vk::Format::R8G8B8A8_UNORM, vk::ImageType::TYPE_2D, vk::Extent3D { width: 256, height: 256, depth: 1 }, 1, vk::ImageCreateFlags::empty()),
        Kind::Bc7 => (vk::Format::BC7_UNORM_BLOCK, vk::ImageType::TYPE_2D, vk::Extent3D { width: 512, height: 512, depth: 1 }, 1, vk::ImageCreateFlags::empty()),
        Kind::Rgba3d => (vk::Format::R8G8B8A8_UNORM, vk::ImageType::TYPE_3D, vk::Extent3D { width: 64, height: 64, depth: 16 }, 1, vk::ImageCreateFlags::empty()),
        Kind::Array => (vk::Format::R8G8B8A8_UNORM, vk::ImageType::TYPE_2D, vk::Extent3D { width: 128, height: 128, depth: 1 }, 4, vk::ImageCreateFlags::empty()),
        Kind::Cube => (vk::Format::R8G8B8A8_UNORM, vk::ImageType::TYPE_2D, vk::Extent3D { width: 128, height: 128, depth: 1 }, 6, vk::ImageCreateFlags::CUBE_COMPATIBLE),
    };
    let bytes = if fmt == vk::Format::BC7_UNORM_BLOCK {
        u64::from(extent.width / 4 * extent.height / 4 * 16)
    } else {
        u64::from(extent.width * extent.height * extent.depth * 4 * layers)
    };
    let fam = gpu.families;
    let image = gpu
        .device
        .create_image(
            &vk::ImageCreateInfo::default()
                .flags(flags)
                .image_type(ty)
                .format(fmt)
                .extent(extent)
                .mip_levels(1)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&fam),
            None,
        )
        .unwrap();
    let req = gpu.device.get_image_memory_requirements(image);
    let memory = gpu
        .device
        .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(gpu.kind_type(req.memory_type_bits, Mem::Device)), None)
        .unwrap();
    gpu.device.bind_image_memory(image, memory, 0).unwrap();
    Target { image, memory, extent, layers, bytes }
}

pub unsafe fn stale(gpu: &Gpu) {
    let runs = 1000u32;
    let mut sub = Submitter::new(gpu, RENDER, 1, 2);
    for kind in [Kind::Rgba2d, Kind::Bc7, Kind::Rgba3d, Kind::Array, Kind::Cube] {
        let t = target(gpu, kind);
        let up = gpu.buffer(t.bytes, Mem::Upload);
        let rb = gpu.buffer(t.bytes, Mem::Readback);
        let sub_layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: t.layers };
        for address in [true, false] {
            let mut fails = 0u32;
            let mut first = String::new();
            for run in 0..runs {
                let words = (t.bytes / 4) as usize;
                let pat: Vec<u32> = (0..words as u32).map(|w| run.wrapping_mul(0x9e37_79b9) ^ w.wrapping_mul(0x85eb_ca6b) ^ (kind as u32) << 28).collect();
                up.write(0, &pat);
                std::ptr::write_bytes(rb.ptr, 0xcd, t.bytes as usize);
                let cmd = sub.begin();
                let to_dst = [vk::ImageMemoryBarrier2::default()
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(t.image)
                    .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: t.layers })];
                gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&to_dst));
                if address {
                    let r = [vk::DeviceMemoryImageCopyKHR::default()
                        .address_range(vk::DeviceAddressRangeKHR { address: up.addr, size: t.bytes })
                        .address_flags(vk::AddressCommandFlagsKHR::FULLY_BOUND | vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE)
                        .image_subresource(sub_layers)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_extent(t.extent)];
                    gpu.dac.cmd_copy_memory_to_image(cmd, &vk::CopyDeviceMemoryImageInfoKHR::default().image(t.image).regions(&r));
                } else {
                    let r = [vk::BufferImageCopy { image_subresource: sub_layers, image_extent: t.extent, ..Default::default() }];
                    gpu.device.cmd_copy_buffer_to_image(cmd, up.buffer, t.image, vk::ImageLayout::GENERAL, &r);
                }
                let b = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
                if address {
                    let r = [vk::DeviceMemoryImageCopyKHR::default()
                        .address_range(vk::DeviceAddressRangeKHR { address: rb.addr, size: t.bytes })
                        .address_flags(vk::AddressCommandFlagsKHR::FULLY_BOUND | vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE)
                        .image_subresource(sub_layers)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_extent(t.extent)];
                    gpu.dac.cmd_copy_image_to_memory(cmd, &vk::CopyDeviceMemoryImageInfoKHR::default().image(t.image).regions(&r));
                } else {
                    let r = [vk::BufferImageCopy { image_subresource: sub_layers, image_extent: t.extent, ..Default::default() }];
                    gpu.device.cmd_copy_image_to_buffer(cmd, t.image, vk::ImageLayout::GENERAL, rb.buffer, &r);
                }
                let h = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&h));
                let v = sub.submit(cmd);
                sub.wait(v);
                let got = std::slice::from_raw_parts(rb.ptr.cast::<u32>(), words);
                if got != &pat[..] {
                    let bad = got.iter().zip(&pat).filter(|(a, b)| a != b).count();
                    if fails == 0 {
                        let i = got.iter().zip(&pat).position(|(a, b)| a != b).unwrap();
                        first = format!("run {run}: {bad} of {words} words wrong, word {i} = {:08x} expected {:08x}", got[i], pat[i]);
                    }
                    fails += 1;
                }
            }
            row(
                &format!("stale copy {kind:?} {}", if address { "address commands" } else { "buffer-handle control" }),
                format!("{fails} of {runs} runs wrong{}", if fails > 0 { format!(" (first: {first})") } else { String::new() }),
            );
        }
        gpu.free(up);
        gpu.free(rb);
        gpu.device.destroy_image(t.image, None);
        gpu.device.free_memory(t.memory, None);
    }
}

const REPRO_BYTES: u64 = 256;

/// One launch of the three-queue repro. Exit 0 pass, 1 Vulkan error or device loss, 3 stale
/// readback.
pub fn queues_child(args: &[String]) {
    let buffers = args.iter().any(|a| a == "--buffers");
    let iterations: u32 = args.iter().position(|a| a == "--iterations").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(256);
    unsafe {
        let gpu: &'static Gpu = Box::leak(Box::new(Gpu::new(!buffers)));
        std::thread::scope(|s| {
            for q in 0..3 {
                s.spawn(move || {
                    let alloc = |kind| if buffers { gpu.buffer_core(REPRO_BYTES, kind) } else { gpu.buffer(REPRO_BYTES, kind) };
                    let (up, scratch, rb) = (alloc(Mem::Upload), alloc(Mem::Device), alloc(Mem::Upload));
                    let mut sub = Submitter::new(gpu, q, 1, 1);
                    let fam = gpu.families[q];
                    for it in 1..=iterations {
                        let words: Vec<u32> = (0..(REPRO_BYTES / 4) as u32).map(|w| fam * 0x0100_0000 + it * 0x100 + w).collect();
                        up.write(0, &words);
                        std::ptr::write_bytes(rb.ptr, 0xcd, REPRO_BYTES as usize);
                        let cmd = sub.begin();
                        copy(gpu, cmd, buffers, &up, &scratch);
                        let b = [vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&b));
                        copy(gpu, cmd, buffers, &scratch, &rb);
                        let h = [vk::MemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                        gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(&h));
                        let v = sub.submit(cmd);
                        sub.wait(v);
                        let got = std::slice::from_raw_parts(rb.ptr.cast::<u32>(), words.len());
                        if got != &words[..] {
                            let i = got.iter().zip(&words).position(|(a, b)| a != b).unwrap();
                            let counter = gpu.device.get_semaphore_counter_value(sub.timeline);
                            eprintln!("queue {q} iteration {it} word {i}: expected {:08x} received {:08x}; counter {counter:?}", words[i], got[i]);
                            std::process::exit(3);
                        }
                    }
                    std::mem::forget(sub);
                });
            }
        });
    }
    std::process::exit(0);
}

unsafe fn copy(gpu: &Gpu, cmd: vk::CommandBuffer, buffers: bool, src: &Buf, dst: &Buf) {
    if buffers {
        let r = [vk::BufferCopy2::default().size(REPRO_BYTES)];
        gpu.device.cmd_copy_buffer2(cmd, &vk::CopyBufferInfo2::default().src_buffer(src.buffer).dst_buffer(dst.buffer).regions(&r));
    } else {
        let r = [vk::DeviceMemoryCopyKHR::default()
            .src_range(vk::DeviceAddressRangeKHR { address: src.addr, size: REPRO_BYTES })
            .src_flags(vk::AddressCommandFlagsKHR::FULLY_BOUND | vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE)
            .dst_range(vk::DeviceAddressRangeKHR { address: dst.addr, size: REPRO_BYTES })
            .dst_flags(vk::AddressCommandFlagsKHR::FULLY_BOUND | vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE)];
        gpu.dac.cmd_copy_memory(cmd, &vk::CopyDeviceMemoryInfoKHR::default().regions(&r));
    }
}

/// `queues [launches]`: fresh-process launches of the repro, core copies then address copies.
pub fn queues_parent(args: &[String]) {
    let launches: u32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let exe = std::env::current_exe().unwrap();
    for (name, flag) in [("core vkCmdCopyBuffer2, no extensions", "--buffers"), ("vkCmdCopyMemoryKHR", "--address")] {
        let mut fails = Vec::new();
        let t0 = std::time::Instant::now();
        for l in 0..launches {
            let out = std::process::Command::new(&exe).args(["queues-child", flag]).env_remove("VK_INSTANCE_LAYERS").output().unwrap();
            if !out.status.success() {
                let err = String::from_utf8_lossy(&out.stderr).lines().chain(String::from_utf8_lossy(&out.stdout).lines()).take(2).collect::<Vec<_>>().join(" / ");
                fails.push(format!("launch {l}: exit {:?} {err}", out.status.code()));
            }
        }
        row(
            &format!("three queues, three threads, {name}"),
            format!(
                "{} of {launches} launches failed ({:.0} s){}",
                fails.len(),
                t0.elapsed().as_secs_f64(),
                fails.iter().take(3).map(|f| format!("; {f}")).collect::<String>()
            ),
        );
    }
}
