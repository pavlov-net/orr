//! Sparse images: creation, tile and mip-tail binds, cooked tile layout. Cases 4 (`hic`: host
//! image copy of cooked tiles into bound tiles) and 5 (`tail`: mip tail start and block shape
//! per format against the cook rule).

use crate::gpu::{Buf, Gpu, LOADING, Mem, RENDER, Submitter};
use crate::{Rng, row};
use ash::vk;

#[derive(Clone, Copy)]
pub struct Fmt {
    pub fmt: vk::Format,
    pub name: &'static str,
    /// Texels per block edge (4 for BCn, 1 otherwise).
    pub block: u32,
    /// Bytes per block (BCn) or texel.
    pub bytes: u32,
}

pub const BC7: Fmt = Fmt { fmt: vk::Format::BC7_UNORM_BLOCK, name: "BC7", block: 4, bytes: 16 };
pub const FORMATS: [Fmt; 8] = [
    BC7,
    Fmt { fmt: vk::Format::BC7_SRGB_BLOCK, name: "BC7_SRGB", block: 4, bytes: 16 },
    Fmt { fmt: vk::Format::BC4_UNORM_BLOCK, name: "BC4", block: 4, bytes: 8 },
    Fmt { fmt: vk::Format::BC5_UNORM_BLOCK, name: "BC5", block: 4, bytes: 16 },
    Fmt { fmt: vk::Format::BC6H_UFLOAT_BLOCK, name: "BC6H", block: 4, bytes: 16 },
    Fmt { fmt: vk::Format::R16_UNORM, name: "R16", block: 1, bytes: 2 },
    Fmt { fmt: vk::Format::R8_UNORM, name: "R8", block: 1, bytes: 1 },
    Fmt { fmt: vk::Format::R8G8B8A8_UNORM, name: "RGBA8", block: 1, bytes: 4 },
];

impl Fmt {
    /// The cook's tile shape in texels: the Vulkan standard 2D sparse block shape.
    pub fn tile(&self) -> (u32, u32) {
        let (w, h) = match self.bytes * 8 {
            8 => (256, 256),
            16 => (256, 128),
            32 => (128, 128),
            64 => (128, 64),
            _ => (64, 64),
        };
        (w * self.block, h * self.block)
    }

    /// The cook's tail rule: the first mip smaller than one tile in either axis.
    pub fn cook_tail(&self, w: u32, h: u32, mips: u32) -> u32 {
        let (tw, th) = self.tile();
        (0..mips).find(|&m| (w >> m).max(1) < tw || (h >> m).max(1) < th).unwrap_or(mips)
    }

    /// Bytes of a tightly packed mip.
    pub fn mip_bytes(&self, w: u32, h: u32) -> usize {
        (w.div_ceil(self.block) * h.div_ceil(self.block) * self.bytes) as usize
    }
}

pub fn full_mips(w: u32, h: u32) -> u32 {
    32 - w.max(h).leading_zeros()
}

pub fn mip_size(w: u32, h: u32, m: u32) -> (u32, u32) {
    ((w >> m).max(1), (h >> m).max(1))
}

pub fn color(mips: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: mips, base_array_layer: 0, layer_count: 1 }
}

pub fn layers(level: u32) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: level, base_array_layer: 0, layer_count: 1 }
}

pub struct Image {
    pub image: vk::Image,
    pub f: Fmt,
    pub w: u32,
    pub h: u32,
    pub mips: u32,
    pub req: vk::MemoryRequirements,
    /// Sparse images only.
    pub sreq: Option<vk::SparseImageMemoryRequirements>,
    pub memory: Vec<vk::DeviceMemory>,
}

#[derive(Clone, Copy)]
pub struct Kind {
    pub sparse: bool,
    pub host: bool,
    pub concurrent: bool,
}

pub fn usage(host: bool) -> vk::ImageUsageFlags {
    let u = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    if host { u | vk::ImageUsageFlags::HOST_TRANSFER } else { u }
}

pub unsafe fn create(gpu: &Gpu, f: Fmt, w: u32, h: u32, mips: u32, array: u32, k: Kind) -> Image {
    let fam = gpu.families;
    let mut ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(f.fmt)
        .extent(vk::Extent3D { width: w, height: h, depth: 1 })
        .mip_levels(mips)
        .array_layers(array)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage(k.host))
        .initial_layout(vk::ImageLayout::UNDEFINED);
    if k.sparse {
        ci = ci.flags(vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY);
    }
    ci = if k.concurrent { ci.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&fam) } else { ci.sharing_mode(vk::SharingMode::EXCLUSIVE) };
    let image = match gpu.device.create_image(&ci, None) {
        Ok(i) => i,
        Err(e) => crate::fail(&format!("create image {} {w}x{h} sparse={} host={}: {e:?}", f.name, k.sparse, k.host)),
    };
    let req = gpu.device.get_image_memory_requirements(image);
    let sreq = k.sparse.then(|| {
        let v = gpu.device.get_image_sparse_memory_requirements(image);
        if v.len() != 1 {
            crate::fail(&format!("expected one sparse requirement (colour aspect), got {}", v.len()));
        }
        v[0]
    });
    Image { image, f, w, h, mips, req, sreq, memory: Vec::new() }
}

/// Memory type for an image: plain device-local when its type bits allow it, else Upload
/// (host-transfer images accept host-visible types only on 621.01).
pub fn image_type(gpu: &Gpu, bits: u32) -> u32 {
    gpu.memory_type(bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::HOST_VISIBLE)
        .or_else(|| gpu.memory_type(bits, vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE, vk::MemoryPropertyFlags::empty()))
        .unwrap_or_else(|| crate::fail(&format!("no device-local memory type in {bits:#x}")))
}

/// Binds a non-sparse image to its own allocation.
pub unsafe fn bind_dedicated(gpu: &Gpu, img: &mut Image) {
    let ty = image_type(gpu, img.req.memory_type_bits);
    let m = gpu.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(img.req.size).memory_type_index(ty), None).unwrap();
    gpu.device.bind_image_memory(img.image, m, 0).unwrap();
    img.memory.push(m);
}

impl Image {
    pub fn tail(&self) -> u32 {
        self.sreq.map_or(self.mips, |s| s.image_mip_tail_first_lod.min(self.mips))
    }
    pub fn granularity(&self) -> (u32, u32) {
        let g = self.sreq.unwrap().format_properties.image_granularity;
        (g.width, g.height)
    }
    pub fn tile_bytes(&self) -> u64 {
        self.req.alignment
    }
    /// Tiles per axis of mip `m`.
    pub fn grid(&self, m: u32) -> (u32, u32) {
        let (tw, th) = self.granularity();
        let (mw, mh) = mip_size(self.w, self.h, m);
        (mw.div_ceil(tw), mh.div_ceil(th))
    }
    /// Every tile above the tail as (mip, tx, ty).
    pub fn tiles(&self) -> Vec<(u32, u32, u32)> {
        let mut v = Vec::new();
        for m in 0..self.tail() {
            let (gx, gy) = self.grid(m);
            for ty in 0..gy {
                for tx in 0..gx {
                    v.push((m, tx, ty));
                }
            }
        }
        v
    }
    /// Region of a tile, clipped to its mip.
    pub fn tile_region(&self, t: (u32, u32, u32)) -> (vk::Offset3D, vk::Extent3D) {
        let (tw, th) = self.granularity();
        let (mw, mh) = mip_size(self.w, self.h, t.0);
        let (x, y) = (t.1 * tw, t.2 * th);
        (vk::Offset3D { x: x as i32, y: y as i32, z: 0 }, vk::Extent3D { width: tw.min(mw - x), height: th.min(mh - y), depth: 1 })
    }
    pub fn tile_bind(&self, t: (u32, u32, u32), memory: vk::DeviceMemory, offset: u64) -> vk::SparseImageMemoryBind {
        let (o, e) = self.tile_region(t);
        vk::SparseImageMemoryBind {
            subresource: vk::ImageSubresource { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: t.0, array_layer: 0 },
            offset: o,
            extent: e,
            memory,
            memory_offset: offset,
            flags: vk::SparseMemoryBindFlags::empty(),
        }
    }
    pub fn tail_bytes(&self) -> u64 {
        let s = self.sreq.unwrap();
        if self.tail() < self.mips { s.image_mip_tail_size.next_multiple_of(self.tile_bytes()) } else { 0 }
    }
}

/// Allocates one block for every tile plus the mip tail of a sparse image and binds it (tile
/// slots shuffled with `shuffle`'s seed when given). Waits for the bind.
pub unsafe fn bind_all(gpu: &Gpu, img: &mut Image, shuffle: Option<u64>) {
    let tiles = img.tiles();
    let tb = img.tile_bytes();
    let tail = img.tail_bytes();
    let ty = image_type(gpu, img.req.memory_type_bits);
    let mem = gpu.alloc(tiles.len() as u64 * tb + tail, ty);
    let mut slots: Vec<u64> = (0..tiles.len() as u64).collect();
    if let Some(seed) = shuffle {
        Rng(seed).shuffle(&mut slots);
    }
    let binds: Vec<_> = tiles.iter().zip(&slots).map(|(&t, &s)| img.tile_bind(t, mem, s * tb)).collect();
    let tail_bind = [vk::SparseMemoryBind {
        resource_offset: img.sreq.unwrap().image_mip_tail_offset,
        size: img.sreq.unwrap().image_mip_tail_size,
        memory: mem,
        memory_offset: tiles.len() as u64 * tb,
        flags: vk::SparseMemoryBindFlags::empty(),
    }];
    let ib = [vk::SparseImageMemoryBindInfo::default().image(img.image).binds(&binds)];
    let ob = [vk::SparseImageOpaqueMemoryBindInfo::default().image(img.image).binds(&tail_bind)];
    let mut info = vk::BindSparseInfo::default();
    if !binds.is_empty() {
        info = info.image_binds(&ib);
    }
    if tail > 0 {
        info = info.image_opaque_binds(&ob);
    }
    let sem = gpu.timeline();
    gpu.bind_sparse(LOADING, info, &[], (sem, 1));
    gpu.wait_value(sem, 1).unwrap();
    gpu.device.destroy_semaphore(sem, None);
    img.memory.push(mem);
}

pub unsafe fn host_to_general(gpu: &Gpu, img: vk::Image, mips: u32) {
    let t = [vk::HostImageLayoutTransitionInfo::default()
        .image(img)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .subresource_range(color(mips))];
    gpu.device.transition_image_layout(&t).unwrap();
}

pub unsafe fn destroy(gpu: &Gpu, img: Image) {
    gpu.device.destroy_image(img.image, None);
    for m in img.memory {
        gpu.device.free_memory(m, None);
    }
}

/// Cooked texture: per mip above the tail, 64 KiB tiles in the standard block shape,
/// row-major, padded at the right and bottom edges, block rows tightly packed; tail mips
/// tightly packed. Also the tight reference of every mip.
pub struct Cooked {
    pub tiles: Vec<Vec<Vec<u8>>>,
    pub tail: Vec<Vec<u8>>,
    pub tight: Vec<Vec<u8>>,
}

pub fn cook(f: Fmt, w: u32, h: u32, mips: u32, tail: u32, seed: u64) -> Cooked {
    let mut r = Rng(seed);
    let (tw, th) = f.tile();
    let (tbw, tbh) = (tw / f.block, th / f.block);
    let tile_bytes = (tbw * tbh * f.bytes) as usize;
    let mut tiles = Vec::new();
    let mut tight = Vec::new();
    let mut tail_mips = Vec::new();
    for m in 0..mips {
        let (mw, mh) = mip_size(w, h, m);
        if m >= tail {
            let mut t = vec![0u8; f.mip_bytes(mw, mh)];
            r.fill(&mut t);
            tight.push(t.clone());
            tail_mips.push(t);
            continue;
        }
        let (gx, gy) = (mw.div_ceil(tw), mh.div_ceil(th));
        let mut mt = Vec::new();
        for _ in 0..gx * gy {
            let mut t = vec![0u8; tile_bytes];
            r.fill(&mut t);
            mt.push(t);
        }
        let (bw, bh) = (mw.div_ceil(f.block), mh.div_ceil(f.block));
        let bpb = f.bytes as usize;
        let mut t = vec![0u8; f.mip_bytes(mw, mh)];
        for by in 0..bh {
            for bx in 0..bw {
                let tile = &mt[((by / tbh) * gx + bx / tbw) as usize];
                let src = (((by % tbh) * tbw + bx % tbw) as usize) * bpb;
                let dst = ((by * bw + bx) as usize) * bpb;
                t[dst..dst + bpb].copy_from_slice(&tile[src..src + bpb]);
            }
        }
        tight.push(t);
        tiles.push(mt);
    }
    Cooked { tiles, tail: tail_mips, tight }
}

/// Host-copies a cooked texture: one region per tile with the full tile as row length and
/// image height and the extent clipped to the mip; tail mips whole with tight rows.
pub unsafe fn host_upload(gpu: &Gpu, img: &Image, c: &Cooked) {
    let (tw, th) = img.f.tile();
    let tail = img.tail();
    for (m, mt) in c.tiles.iter().enumerate() {
        let (gx, _) = img.grid(m as u32);
        let regions: Vec<_> = mt
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let (o, e) = img.tile_region((m as u32, i as u32 % gx, i as u32 / gx));
                vk::MemoryToImageCopy::default()
                    .host_pointer(t.as_ptr().cast())
                    .memory_row_length(tw)
                    .memory_image_height(th)
                    .image_subresource(layers(m as u32))
                    .image_offset(o)
                    .image_extent(e)
            })
            .collect();
        gpu.device
            .copy_memory_to_image(&vk::CopyMemoryToImageInfo::default().dst_image(img.image).dst_image_layout(vk::ImageLayout::GENERAL).regions(&regions))
            .unwrap();
    }
    for (i, t) in c.tail.iter().enumerate() {
        let m = tail + i as u32;
        let (mw, mh) = mip_size(img.w, img.h, m);
        let r = [vk::MemoryToImageCopy::default()
            .host_pointer(t.as_ptr().cast())
            .image_subresource(layers(m))
            .image_extent(vk::Extent3D { width: mw, height: mh, depth: 1 })];
        gpu.device
            .copy_memory_to_image(&vk::CopyMemoryToImageInfo::default().dst_image(img.image).dst_image_layout(vk::ImageLayout::GENERAL).regions(&r))
            .unwrap();
    }
}

/// Queue-copies a cooked texture on the loading queue from a staging buffer, with the same
/// regions as `host_upload` (full-tile row length and image height, extent clipped).
pub unsafe fn staged_upload(gpu: &Gpu, sub: &mut Submitter, img: &Image, c: &Cooked, staging: &Buf) {
    let (tw, th) = img.f.tile();
    let tail = img.tail();
    let mut regions = Vec::new();
    let mut off = 0u64;
    for (m, mt) in c.tiles.iter().enumerate() {
        let (gx, _) = img.grid(m as u32);
        for (i, t) in mt.iter().enumerate() {
            let (o, e) = img.tile_region((m as u32, i as u32 % gx, i as u32 / gx));
            staging.write(off as usize, t);
            regions.push(vk::BufferImageCopy { buffer_offset: off, buffer_row_length: tw, buffer_image_height: th, image_subresource: layers(m as u32), image_offset: o, image_extent: e });
            off += t.len() as u64;
        }
    }
    for (i, t) in c.tail.iter().enumerate() {
        let m = tail + i as u32;
        let (mw, mh) = mip_size(img.w, img.h, m);
        staging.write(off as usize, t);
        regions.push(vk::BufferImageCopy { buffer_offset: off, image_subresource: layers(m), image_extent: vk::Extent3D { width: mw, height: mh, depth: 1 }, ..Default::default() });
        off = (off + t.len() as u64).next_multiple_of(16);
    }
    assert!(off <= staging.size);
    sub.run(|cmd| {
        to_general(gpu, cmd, img.image, img.mips);
        gpu.device.cmd_copy_buffer_to_image(cmd, staging.buffer, img.image, vk::ImageLayout::GENERAL, &regions);
        gpu.barrier(cmd);
    });
}

/// UNDEFINED to GENERAL on the queue (contents discarded).
pub unsafe fn to_general(gpu: &Gpu, cmd: vk::CommandBuffer, image: vk::Image, mips: u32) {
    let b = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color(mips))];
    gpu.device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&b));
}

/// Reads every mip back with a GPU copy and, for host-transfer images, on the host; returns
/// (host equal, GPU equal).
unsafe fn verify(gpu: &Gpu, sub: &mut Submitter, img: &Image, c: &Cooked, out: &Buf, host: bool) -> (bool, bool) {
    let mut host_ok = true;
    let mut offs = Vec::new();
    let mut off = 0u64;
    for m in 0..img.mips {
        let (mw, mh) = mip_size(img.w, img.h, m);
        let mut back = vec![0u8; c.tight[m as usize].len()];
        if !host {
            offs.push(off);
            off = (off + back.len() as u64).next_multiple_of(16);
            continue;
        }
        let r = [vk::ImageToMemoryCopy::default()
            .host_pointer(back.as_mut_ptr().cast())
            .image_subresource(layers(m))
            .image_extent(vk::Extent3D { width: mw, height: mh, depth: 1 })];
        gpu.device
            .copy_image_to_memory(&vk::CopyImageToMemoryInfo::default().src_image(img.image).src_image_layout(vk::ImageLayout::GENERAL).regions(&r))
            .unwrap();
        host_ok &= back == c.tight[m as usize];
        offs.push(off);
        off = (off + back.len() as u64).next_multiple_of(16);
    }
    assert!(off <= out.size);
    std::ptr::write_bytes(out.ptr, 0xcd, off as usize);
    sub.run(|cmd| {
        let regions: Vec<_> = (0..img.mips)
            .map(|m| {
                let (mw, mh) = mip_size(img.w, img.h, m);
                vk::BufferImageCopy { buffer_offset: offs[m as usize], image_subresource: layers(m), image_extent: vk::Extent3D { width: mw, height: mh, depth: 1 }, ..Default::default() }
            })
            .collect();
        gpu.device.cmd_copy_image_to_buffer(cmd, img.image, vk::ImageLayout::GENERAL, out.buffer, &regions);
        gpu.barrier(cmd);
    });
    let gpu_ok = (0..img.mips as usize).all(|m| {
        let t = &c.tight[m];
        std::slice::from_raw_parts(out.ptr.add(offs[m] as usize), t.len()) == &t[..]
    });
    (host_ok, gpu_ok)
}

/// Case 4: cooked tiles, padded edge tiles included, copied into bound tiles of sparse images:
/// by host image copy (`host`, on host-transfer sparse images, which 621.01 reports
/// unsupported) or by a loading-queue copy from staging (spec-valid). Verified through a GPU
/// copy (and on the host for host copies); then every tile above the tail is unbound, rebound
/// to shuffled memory, copied again and verified again.
pub unsafe fn hic(gpu: &Gpu, host: bool) {
    let mut sub = Submitter::new(gpu, RENDER, 2, 2);
    let mut lsub = Submitter::new(gpu, LOADING, 2, 2);
    let out = gpu.buffer(96 << 20, Mem::Readback);
    let staging = gpu.buffer(96 << 20, Mem::Upload);
    let sizes = [(4096u32, 4096u32), (1300, 700), (700, 1300), (2600, 300)];
    let mut all = true;
    for f in FORMATS {
        for &(w, h) in &sizes {
            let mips = full_mips(w, h);
            let mut img = create(gpu, f, w, h, mips, 1, Kind { sparse: true, host, concurrent: true });
            let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            if gpu.memory_type(img.req.memory_type_bits, dl, vk::MemoryPropertyFlags::empty()).is_none() {
                row(&format!("{} {w}x{h}", f.name), format!("no device-local memory type: typeBits {:#x}", img.req.memory_type_bits));
                all = false;
                destroy(gpu, img);
                continue;
            }
            let tail = img.tail();
            let c = cook(f, w, h, mips, tail, u64::from(w * 31 + h) ^ f.fmt.as_raw() as u64);
            bind_all(gpu, &mut img, None);
            let upload = |img: &Image, lsub: &mut Submitter| {
                if host {
                    host_to_general(gpu, img.image, mips);
                    host_upload(gpu, img, &c);
                } else {
                    staged_upload(gpu, lsub, img, &c, &staging);
                }
            };
            upload(&img, &mut lsub);
            let (h1, g1) = verify(gpu, &mut sub, &img, &c, &out, host);
            // Rebind every tile to shuffled slots of a new block, copy and verify again.
            let tiles = img.tiles();
            let tb = img.tile_bytes();
            let ty = image_type(gpu, img.req.memory_type_bits);
            let mem = gpu.alloc(tiles.len().max(1) as u64 * tb, ty);
            let mut slots: Vec<u64> = (0..tiles.len() as u64).collect();
            Rng(9).shuffle(&mut slots);
            let unbind: Vec<_> = tiles.iter().map(|&t| img.tile_bind(t, vk::DeviceMemory::null(), 0)).collect();
            let rebind: Vec<_> = tiles.iter().zip(&slots).map(|(&t, &s)| img.tile_bind(t, mem, s * tb)).collect();
            let sem = gpu.timeline();
            if !tiles.is_empty() {
                for (v, b) in [(1u64, &unbind), (2, &rebind)] {
                    let ib = [vk::SparseImageMemoryBindInfo::default().image(img.image).binds(b)];
                    gpu.bind_sparse(LOADING, vk::BindSparseInfo::default().image_binds(&ib), &[], (sem, v));
                }
                gpu.wait_value(sem, 2).unwrap();
            }
            gpu.device.destroy_semaphore(sem, None);
            img.memory.push(mem);
            upload(&img, &mut lsub);
            let (h2, g2) = verify(gpu, &mut sub, &img, &c, &out, host);
            let edge = (0..tail).any(|m| {
                let (mw, mh) = mip_size(w, h, m);
                let (tw, th) = f.tile();
                mw % tw != 0 || mh % th != 0
            });
            row(
                &format!("{} {w}x{h}", f.name),
                format!(
                    "{mips} mips, tail at {tail}, {} tiles ({}padded edge tiles), {} B tail; {} copy; host readback equal {}, GPU copy equal {g1}/{g2} (first bind / after shuffled rebind)",
                    tiles.len(),
                    if edge { "" } else { "no " },
                    img.tail_bytes(),
                    if host { "host image" } else { "loading-queue staged" },
                    if host { format!("{h1}/{h2}") } else { "n/a".into() }
                ),
            );
            all &= h1 && g1 && h2 && g2;
            destroy(gpu, img);
        }
    }
    row("all copies into tiles read back equal", all);
    if !all {
        crate::fail("copy into sparse tiles read back different");
    }
}

/// Case 5: mip tail start, block shape and flags per format and size against the cook rule.
pub unsafe fn tail(gpu: &Gpu) {
    row("residencyAlignedMipSize", gpu.sparse_props.residency_aligned_mip_size == vk::TRUE);
    row("residencyStandard2DBlockShape", gpu.sparse_props.residency_standard2_d_block_shape == vk::TRUE);
    let sizes = [
        (4096u32, 4096u32),
        (2048, 1024),
        (1024, 2048),
        (4096, 256),
        (256, 4096),
        (1300, 700),
        (700, 1300),
        (300, 5000),
        (512, 512),
        (512, 256),
        (256, 256),
        (256, 128),
        (128, 128),
        (100, 60),
        (16384, 16384),
    ];
    let mut mismatches = Vec::new();
    for f in FORMATS {
        let props = gpu.instance.get_physical_device_sparse_image_format_properties(
            gpu.pdev,
            f.fmt,
            vk::ImageType::TYPE_2D,
            vk::SampleCountFlags::TYPE_1,
            usage(false),
            vk::ImageTiling::OPTIMAL,
        );
        let g = props.first().map(|p| (p.image_granularity.width, p.image_granularity.height));
        let mut limits = Vec::new();
        for (sparse, host) in [(true, true), (true, false), (false, true)] {
            let flags = if sparse { vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY } else { vk::ImageCreateFlags::empty() };
            let r = gpu.instance.get_physical_device_image_format_properties(gpu.pdev, f.fmt, vk::ImageType::TYPE_2D, vk::ImageTiling::OPTIMAL, usage(host), flags);
            limits.push(match r {
                Ok(p) => format!("sparse={sparse} host={host}: max {}x{} {} mips, {} MiB", p.max_extent.width, p.max_extent.height, p.max_mip_levels, p.max_resource_size >> 20),
                Err(e) => format!("sparse={sparse} host={host}: {e:?}"),
            });
        }
        let host_sparse = gpu
            .instance
            .get_physical_device_sparse_image_format_properties(gpu.pdev, f.fmt, vk::ImageType::TYPE_2D, vk::SampleCountFlags::TYPE_1, usage(true), vk::ImageTiling::OPTIMAL)
            .len();
        limits.push(format!("sparse format properties with host-transfer usage: {host_sparse} entries"));
        row(&format!("{} image format limits", f.name), limits.join("; "));
        let mut text = Vec::new();
        let mut ok = 0;
        for &(w, h) in &sizes {
            let mips = full_mips(w, h);
            let mut lods = Vec::new();
            for host in [false] {
                let img = create(gpu, f, w, h, mips, 1, Kind { sparse: true, host, concurrent: true });
                let s = img.sreq.unwrap();
                lods.push((s.image_mip_tail_first_lod, s.image_mip_tail_size, s.format_properties.flags));
                destroy(gpu, img);
            }
            let want = f.cook_tail(w, h, mips);
            let (lod, size, flags) = lods[0];
            if lod == want {
                ok += 1;
            } else {
                mismatches.push(format!("{} {w}x{h}: device {lod} cook {want}", f.name));
            }
            text.push(format!("{w}x{h}:{lod}/{size}B{}", if flags.is_empty() { String::new() } else { format!(" {flags:?}") }));
        }
        row(
            &format!("{} tail first lod / tail bytes", f.name),
            format!("granularity {:?} (cook {:?}); {}/{} match the cook rule; {}", g, f.tile(), ok, sizes.len(), text.join(", ")),
        );
        if g != Some(f.tile()) {
            mismatches.push(format!("{} granularity {:?} vs cook {:?}", f.name, g, f.tile()));
        }
        // Array of 6 layers: one tail per layer or one for all.
        let img = create(gpu, f, 1024, 1024, full_mips(1024, 1024), 6, Kind { sparse: true, host: false, concurrent: true });
        let s = img.sreq.unwrap();
        row(
            &format!("{} 1024^2 x6 layers", f.name),
            format!(
                "flags {:?}, tail first lod {}, size {}, offset {}, stride {}",
                s.format_properties.flags, s.image_mip_tail_first_lod, s.image_mip_tail_size, s.image_mip_tail_offset, s.image_mip_tail_stride
            ),
        );
        destroy(gpu, img);
    }
    row("mismatches", if mismatches.is_empty() { "none".to_string() } else { mismatches.join("; ") });
}
