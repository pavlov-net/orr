//! Textures: images that stay in `GENERAL`, with memory from orr-gpu's image pools, placed
//! or sparse memory, and host copies.

use std::sync::{Arc, OnceLock};

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use super::memory::{Allocation, MemoryKind, memory_error, sharing};
use crate::{
    AddressMode, CompareOp, Error, Filter, Format, SparseLayout, TextureDesc, TextureKind,
    TextureRegion, TextureUsage, TimelinePoint, ViewDesc, ViewKind,
};

/// Where a texture's memory comes from.
pub enum TextureMemory<'a> {
    /// Memory orr-gpu manages, from blocks that hold only images. Render targets (`COLOR` or
    /// `DEPTH` usage, or single-mip 2D `STORAGE` textures of 1 MiB and more), textures of
    /// 16 MiB and more and textures the driver wants alone get their own `VkDeviceMemory`;
    /// render targets sit at the highest memory priority.
    Dedicated,
    /// Inside a block, at an offset aligned as [`Device::texture_requirements`] says: a
    /// [`MemoryKind::DeviceLocal`] block, or an Upload block for
    /// [`TextureUsage::HOST_COPY`]. The block must outlive the texture.
    Placed {
        block: &'a Allocation<u8>,
        offset: u64,
    },
    /// No memory until [`Device::bind_sparse`] binds its tiles and mip tails
    /// ([`Texture::sparse_layout`]) to DeviceLocal blocks. 2D and cube colour textures.
    Sparse,
}

/// Where a sparse texture's mip tails sit in its opaque range.
pub(crate) struct SparseTail {
    pub offset: u64,
    pub stride: u64,
    pub size: u64,
}

/// An image and its memory. It has no layout state: [`crate::CommandList::discard`] puts it
/// in `GENERAL` before first use, and it stays there.
///
/// Dropping it destroys it once everything submitted before the drop has completed;
/// [`Device::free_texture`] names the point instead.
pub struct Texture {
    pub(crate) image: vk::Image,
    memory: Option<orr_memory::Allocation>,
    /// The view rendering writes through: mip 0, layer 0. Null without `COLOR` or `DEPTH`
    /// usage.
    attachment: vk::ImageView,
    /// Every mip and layer; made by the first `vulkan::image_handles`.
    pub(crate) full_view: OnceLock<vk::ImageView>,
    desc: TextureDesc,
    /// False for a swapchain image: the swapchain owns it and drops this texture once its
    /// presents have completed.
    owns_image: bool,
    /// With [`TextureMemory::Sparse`].
    pub(crate) sparse: Option<(SparseLayout, SparseTail)>,
    shared: Arc<Shared>,
}

impl Texture {
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }

    pub(crate) fn shared(&self) -> &Shared {
        &self.shared
    }

    fn views(&self) -> [vk::ImageView; 2] {
        [
            self.attachment,
            self.full_view.get().copied().unwrap_or_default(),
        ]
    }

    /// How tiles and mip tails bind, for a texture created with [`TextureMemory::Sparse`].
    pub fn sparse_layout(&self) -> Option<&SparseLayout> {
        self.sparse.as_ref().map(|(l, _)| l)
    }

    /// The view rendering writes through; panics unless the texture has `usage`.
    pub(crate) fn attachment(&self, usage: TextureUsage) -> vk::ImageView {
        assert!(
            self.desc.usage.contains(usage),
            "a {:?} attachment needs a texture with {usage:?} usage",
            self.desc.format
        );
        self.attachment
    }

    pub(crate) fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange::default()
            .aspect_mask(aspect(self.desc.format))
            .level_count(self.desc.mips)
            .layer_count(self.desc.layers)
    }

    pub(crate) fn layers(&self, region: &TextureRegion) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(data_aspect(self.desc.format))
            .mip_level(region.mip)
            .base_array_layer(region.base_layer)
            .layer_count(region.layers)
    }

    pub(crate) fn view_info(&self, v: &ViewDesc) -> vk::ImageViewCreateInfo<'static> {
        let d = &self.desc;
        let mips = if v.mips == 0 {
            d.mips - v.base_mip
        } else {
            v.mips
        };
        let layers = if v.layers == 0 {
            d.layers - v.base_layer
        } else {
            v.layers
        };
        let kind = v.kind.unwrap_or(match d.kind {
            TextureKind::D2 if layers == 1 => ViewKind::D2,
            TextureKind::D2 => ViewKind::D2Array,
            TextureKind::D3 => ViewKind::D3,
            TextureKind::Cube if layers == 6 => ViewKind::Cube,
            TextureKind::Cube => ViewKind::CubeArray,
        });
        let view_type = match kind {
            ViewKind::D2 => vk::ImageViewType::TYPE_2D,
            ViewKind::D2Array => vk::ImageViewType::TYPE_2D_ARRAY,
            ViewKind::D3 => vk::ImageViewType::TYPE_3D,
            ViewKind::Cube => vk::ImageViewType::CUBE,
            ViewKind::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        };
        vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(view_type)
            .format(vk_format(d.format))
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(data_aspect(d.format))
                    .base_mip_level(v.base_mip)
                    .level_count(mips)
                    .base_array_layer(v.base_layer)
                    .layer_count(layers),
            )
    }

    /// A swapchain image as a texture; the swapchain keeps the image.
    pub(crate) fn swapchain_image(
        shared: &Arc<Shared>,
        image: vk::Image,
        desc: &TextureDesc,
        name: &str,
    ) -> Result<Texture, Error> {
        let mut t = Texture {
            image,
            memory: None,
            attachment: vk::ImageView::null(),
            full_view: OnceLock::new(),
            desc: *desc,
            owns_image: false,
            sparse: None,
            shared: Arc::clone(shared),
        };
        t.create_attachment(name)?;
        Ok(t)
    }

    /// Creates the view rendering writes through, for colour and depth targets.
    fn create_attachment(&mut self, name: &str) -> Result<(), Error> {
        if !self
            .desc
            .usage
            .intersects(TextureUsage::COLOR | TextureUsage::DEPTH)
        {
            return Ok(());
        }
        let mut info = self.view_info(&ViewDesc {
            kind: Some(ViewKind::D2),
            mips: 1,
            layers: 1,
            ..ViewDesc::default()
        });
        info.subresource_range.aspect_mask = aspect(self.desc.format);
        // SAFETY: the image is live and bound to memory. On failure dropping the texture
        // frees it.
        self.attachment = unsafe { self.shared.device.create_image_view(&info, None) }
            .map_err(Error::vk("vkCreateImageView"))?;
        name_object(&self.shared.debug_fn, self.attachment, name);
        Ok(())
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        if !self.owns_image {
            for v in self.views() {
                // SAFETY: the swapchain drops its textures once no work uses them.
                unsafe { self.shared.device.destroy_image_view(v, None) };
            }
            return;
        }
        self.shared.defer_after_submitted(Item::Image {
            image: self.image,
            memory: self.memory.take(),
            views: self.views(),
        });
    }
}

/// Render targets, and single-mip 2D storage textures of 1 MiB and more, which are
/// resolution-sized targets.
fn is_target(desc: &TextureDesc, bytes: u64) -> bool {
    desc.usage
        .intersects(TextureUsage::COLOR | TextureUsage::DEPTH)
        || (desc.usage.contains(TextureUsage::STORAGE)
            && desc.kind == TextureKind::D2
            && desc.mips == 1
            && desc.layers == 1
            && bytes >= 1 << 20)
}

fn aspect(format: Format) -> vk::ImageAspectFlags {
    if format.has_stencil() {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        data_aspect(format)
    }
}

/// The aspect views read and copies move: depth for depth-stencil formats.
fn data_aspect(format: Format) -> vk::ImageAspectFlags {
    if format.is_depth() {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

pub(crate) fn image_info<'a>(
    desc: &TextureDesc,
    sparse: bool,
    families: &'a [u32],
) -> vk::ImageCreateInfo<'a> {
    let mut usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    for (flag, vk_flag) in [
        (TextureUsage::SAMPLED, vk::ImageUsageFlags::SAMPLED),
        (TextureUsage::STORAGE, vk::ImageUsageFlags::STORAGE),
        (TextureUsage::COLOR, vk::ImageUsageFlags::COLOR_ATTACHMENT),
        (
            TextureUsage::DEPTH,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ),
        (TextureUsage::HOST_COPY, vk::ImageUsageFlags::HOST_TRANSFER),
    ] {
        if desc.usage.contains(flag) {
            usage |= vk_flag;
        }
    }
    let (image_type, mut flags) = match desc.kind {
        TextureKind::D2 => (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty()),
        TextureKind::D3 => (vk::ImageType::TYPE_3D, vk::ImageCreateFlags::empty()),
        TextureKind::Cube => (
            vk::ImageType::TYPE_2D,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
        ),
    };
    if sparse {
        flags |= vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY;
    }
    let (sharing, families) = sharing(families);
    vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(image_type)
        .format(vk_format(desc.format))
        .extent(vk::Extent3D {
            width: desc.extent[0],
            height: desc.extent[1],
            depth: desc.extent[2],
        })
        .mip_levels(desc.mips)
        .array_layers(desc.layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(sharing)
        .queue_family_indices(families)
        .initial_layout(vk::ImageLayout::UNDEFINED)
}

fn check(desc: &TextureDesc, sparse: bool, caps: &crate::Caps) -> Result<(), Error> {
    let bad = |m: &str| Err(Error::Invalid(format!("texture {desc:?}: {m}")));
    if desc.extent.contains(&0) || desc.mips == 0 || desc.layers == 0 {
        return bad("zero extent, mips or layers");
    }
    if desc.kind != TextureKind::D3 && desc.extent[2] != 1 {
        return bad("depth must be 1 unless the kind is D3");
    }
    if desc.kind == TextureKind::D3 && desc.layers != 1 {
        return bad("3D textures have one layer");
    }
    if desc.kind == TextureKind::D3
        && desc
            .usage
            .intersects(TextureUsage::COLOR | TextureUsage::DEPTH)
    {
        return bad("3D textures cannot be colour or depth targets");
    }
    if desc.kind == TextureKind::Cube
        && (!desc.layers.is_multiple_of(6) || desc.extent[0] != desc.extent[1])
    {
        return bad("cube textures are square with a multiple of 6 layers");
    }
    let largest = desc.extent.iter().copied().max().unwrap_or(1);
    if desc.mips > 32 - largest.leading_zeros() {
        return bad("more mips than the extent allows");
    }
    let host = desc.usage.contains(TextureUsage::HOST_COPY);
    if sparse && (desc.kind == TextureKind::D3 || desc.format.is_depth() || host) {
        return bad("sparse textures are 2D or cube colour textures without host copy");
    }
    if host && !caps.host_image_copy {
        return bad("host copy needs Caps::host_image_copy");
    }
    Ok(())
}

/// Fails when the device cannot create the image `info` describes.
fn check_format(
    shared: &Shared,
    info: &vk::ImageCreateInfo<'_>,
    desc: &TextureDesc,
) -> Result<(), Error> {
    let query = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(info.format)
        .ty(info.image_type)
        .tiling(info.tiling)
        .usage(info.usage)
        .flags(info.flags);
    let mut out = vk::ImageFormatProperties2::default();
    // SAFETY: a plain query on the device's physical device.
    let r = unsafe {
        shared
            .instance
            .get_physical_device_image_format_properties2(shared.physical, &query, &mut out)
    };
    let l = out.image_format_properties;
    let fits = r.is_ok()
        && desc.extent[0] <= l.max_extent.width
        && desc.extent[1] <= l.max_extent.height
        && desc.extent[2] <= l.max_extent.depth
        && desc.mips <= l.max_mip_levels
        && desc.layers <= l.max_array_layers;
    if fits {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "texture {desc:?}: the device cannot create it with usage {:?} and flags {:?}",
            info.usage, info.flags
        )))
    }
}

/// The tile shape and mip tail placement of the sparse image `image`.
fn sparse_layout(
    shared: &Shared,
    image: vk::Image,
    desc: &TextureDesc,
    tile_bytes: u64,
) -> Result<(SparseLayout, SparseTail), Error> {
    let d = &shared.device;
    let info = vk::ImageSparseMemoryRequirementsInfo2::default().image(image);
    // SAFETY: `image` is a live sparse image.
    let n = unsafe { d.get_image_sparse_memory_requirements2_len(&info) };
    let mut reqs = vec![vk::SparseImageMemoryRequirements2::default(); n];
    // SAFETY: as above, with room for `n` entries.
    unsafe { d.get_image_sparse_memory_requirements2(&info, &mut reqs) };
    let colour_only = |r: &vk::SparseImageMemoryRequirements2<'_>| {
        r.memory_requirements.format_properties.aspect_mask == vk::ImageAspectFlags::COLOR
    };
    let r = match reqs.as_slice() {
        [r] if colour_only(r) => r.memory_requirements,
        _ => {
            return Err(Error::Invalid(format!(
                "texture {desc:?}: sparse aspects other than colour alone"
            )));
        }
    };
    let g = r.format_properties.image_granularity;
    let tail_first_mip = r.image_mip_tail_first_lod.min(desc.mips);
    let has_tail = tail_first_mip < desc.mips;
    let layout = SparseLayout {
        tile_extent: [g.width, g.height, g.depth],
        tile_bytes,
        tail_first_mip,
        tail_bytes: if has_tail {
            r.image_mip_tail_size.next_multiple_of(tile_bytes)
        } else {
            0
        },
        single_tail: r
            .format_properties
            .flags
            .contains(vk::SparseImageFormatFlags::SINGLE_MIPTAIL),
    };
    let tail = SparseTail {
        offset: r.image_mip_tail_offset,
        stride: r.image_mip_tail_stride,
        size: if has_tail { r.image_mip_tail_size } else { 0 },
    };
    Ok((layout, tail))
}

fn host_check(t: &Texture, region: &TextureRegion, len: usize) -> Result<(), Error> {
    let bytes = region.bytes(t.desc.format);
    if !t.desc.usage.contains(TextureUsage::HOST_COPY) || (len as u64) < bytes {
        return Err(Error::Invalid(format!(
            "a host copy of {bytes} bytes through {len} with a {:?} texture",
            t.desc.usage
        )));
    }
    Ok(())
}

pub(crate) fn vk_offset(r: &TextureRegion) -> vk::Offset3D {
    vk::Offset3D {
        x: r.offset[0] as i32,
        y: r.offset[1] as i32,
        z: r.offset[2] as i32,
    }
}

pub(crate) fn vk_extent(r: &TextureRegion) -> vk::Extent3D {
    vk::Extent3D {
        width: r.extent[0],
        height: r.extent[1],
        depth: r.extent[2],
    }
}

pub(crate) fn indirect_copy_format(shared: &Shared, format: Format) -> bool {
    if !shared.caps.indirect_copy || format.is_depth() {
        return false;
    }
    let mut p3 = vk::FormatProperties3::default();
    {
        let mut p = vk::FormatProperties2::default().push(&mut p3);
        // SAFETY: a plain query on the device's physical device.
        unsafe {
            shared.instance.get_physical_device_format_properties2(
                shared.physical,
                vk_format(format),
                &mut p,
            );
        }
    }
    p3.optimal_tiling_features
        .contains(vk::FormatFeatureFlags2::COPY_IMAGE_INDIRECT_DST_KHR)
}

impl Device {
    /// [`crate::CommandList::copy_to_texture_indirect`] can copy into textures of `format`:
    /// `Caps::indirect_copy` and a color format the device copies into indirectly.
    pub fn indirect_copy_format(&self, format: Format) -> bool {
        indirect_copy_format(&self.shared, format)
    }

    /// Size and alignment of a texture's memory, for placing it in a block.
    pub fn texture_requirements(&self, desc: &TextureDesc) -> (u64, u64) {
        let info = image_info(desc, false, &self.shared.families);
        let req = vk::DeviceImageMemoryRequirements::default().create_info(&info);
        let mut out = vk::MemoryRequirements2::default();
        // SAFETY: valid create info.
        unsafe {
            self.shared
                .device
                .get_device_image_memory_requirements(&req, &mut out);
        };
        (
            out.memory_requirements.size,
            out.memory_requirements.alignment,
        )
    }

    /// Creates a texture. Its contents are undefined until written; record
    /// [`crate::CommandList::discard`] before its first GPU use. A
    /// [`TextureUsage::HOST_COPY`] texture starts in `GENERAL`, ready for host copies.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        memory: TextureMemory<'_>,
        name: &str,
    ) -> Result<Texture, Error> {
        let is_sparse = matches!(memory, TextureMemory::Sparse);
        let shared = &self.shared;
        check(desc, is_sparse, &shared.caps)?;
        let host = desc.usage.contains(TextureUsage::HOST_COPY);
        let d = &shared.device;
        let info = image_info(desc, is_sparse, &shared.families);
        if is_sparse || host {
            check_format(shared, &info, desc)?;
        }
        let (image, own) = if let TextureMemory::Dedicated = memory {
            let bytes = self.texture_requirements(desc).0;
            let (pool, flags) = shared.pools.image(host, is_target(desc, bytes), bytes);
            // SAFETY: valid create info for this device.
            let made = unsafe {
                shared
                    .pools
                    .allocator
                    .create_image(pool, &info, flags, name)
            };
            let (image, allocation) = made.map_err(|e| match e {
                orr_memory::Error::Vulkan {
                    result: vk::Result::ERROR_FEATURE_NOT_PRESENT,
                    ..
                } => Error::Invalid(format!(
                    "{name}: memory type {} cannot hold this texture",
                    pool.memory_type()
                )),
                e => memory_error(e),
            })?;
            (image, Some(allocation))
        } else {
            // SAFETY: valid create info for this device.
            let image =
                unsafe { d.create_image(&info, None) }.map_err(Error::vk("vkCreateImage"))?;
            (image, None)
        };
        // SAFETY: `image` is live.
        let req = unsafe { d.get_image_memory_requirements(image) };
        let destroy = || {
            // SAFETY: the image is not used anywhere else and owns no memory.
            unsafe { d.destroy_image(image, None) };
        };
        let mut sparse = None;
        match memory {
            TextureMemory::Sparse => {
                let ty = shared.memory.of(MemoryKind::DeviceLocal);
                let layout = if req.memory_type_bits & (1 << ty) == 0 {
                    Err(Error::Invalid(format!(
                        "{name}: sparse textures cannot take DeviceLocal memory (type {ty})"
                    )))
                } else {
                    sparse_layout(shared, image, desc, req.alignment)
                };
                sparse = Some(layout.inspect_err(|_| destroy())?);
            }
            TextureMemory::Dedicated => {}
            TextureMemory::Placed { block, offset } => {
                let raw = block.raw();
                let ty = shared.memory.of(raw.kind);
                let (_, memory_offset) = raw.memory_at(offset);
                let fits = (!host || raw.kind == MemoryKind::Upload)
                    && req.memory_type_bits & (1 << ty) != 0
                    && memory_offset.is_multiple_of(req.alignment)
                    && offset + req.size <= raw.size;
                if !fits {
                    destroy();
                    return Err(Error::Invalid(format!(
                        "{name}: {} bytes aligned to {} do not fit a {:?} block of {} bytes at offset {offset} (memory offset {memory_offset})",
                        req.size, req.alignment, raw.kind, raw.size
                    )));
                }
                // SAFETY: the range lies inside the block's memory, whose type the image
                // accepts; the allocator serializes binds with its own use of the memory.
                unsafe {
                    shared
                        .pools
                        .allocator
                        .bind_image(&raw.allocation, offset, image)
                }
                .map_err(|e| {
                    destroy();
                    memory_error(e)
                })?;
            }
        }
        name_object(&shared.debug_fn, image, name);
        let mut t = Texture {
            image,
            memory: own,
            attachment: vk::ImageView::null(),
            full_view: OnceLock::new(),
            desc: *desc,
            owns_image: true,
            sparse,
            shared: Arc::clone(shared),
        };
        if host {
            let transition = [vk::HostImageLayoutTransitionInfo::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .subresource_range(t.full_range())];
            // SAFETY: the image is bound, has host-transfer usage and nothing uses it yet.
            unsafe { d.transition_image_layout(&transition) }
                .map_err(Error::vk("vkTransitionImageLayout"))?;
        }
        t.create_attachment(name)?;
        Ok(t)
    }

    /// Writes tightly packed texels from `data` into `region` of `t` on the calling thread,
    /// with no command list. `t` needs [`TextureUsage::HOST_COPY`]. Threads may copy into
    /// disjoint regions at once; no GPU work may use the region during the copy. Submits made
    /// after it returns see the texels.
    pub fn host_copy_to_texture(
        &self,
        t: &Texture,
        region: &TextureRegion,
        data: &[u8],
    ) -> Result<(), Error> {
        host_check(t, region, data.len())?;
        let regions = [vk::MemoryToImageCopy::default()
            .host_pointer(data.as_ptr().cast())
            .image_subresource(t.layers(region))
            .image_offset(vk_offset(region))
            .image_extent(vk_extent(region))];
        let info = vk::CopyMemoryToImageInfo::default()
            .dst_image(t.image)
            .dst_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        // SAFETY: the texture is in GENERAL with host-transfer usage; `data` holds the region.
        unsafe { self.shared.device.copy_memory_to_image(&info) }
            .map_err(Error::vk("vkCopyMemoryToImage"))
    }

    /// Reads `region` of `t` into `out`, tightly packed, on the calling thread. `t` needs
    /// [`TextureUsage::HOST_COPY`], and GPU writes to the region must have completed.
    pub fn host_copy_from_texture(
        &self,
        out: &mut [u8],
        t: &Texture,
        region: &TextureRegion,
    ) -> Result<(), Error> {
        host_check(t, region, out.len())?;
        let regions = [vk::ImageToMemoryCopy::default()
            .host_pointer(out.as_mut_ptr().cast())
            .image_subresource(t.layers(region))
            .image_offset(vk_offset(region))
            .image_extent(vk_extent(region))];
        let info = vk::CopyImageToMemoryInfo::default()
            .src_image(t.image)
            .src_image_layout(vk::ImageLayout::GENERAL)
            .regions(&regions);
        // SAFETY: the texture is in GENERAL with host-transfer usage; `out` holds the region.
        unsafe { self.shared.device.copy_image_to_memory(&info) }
            .map_err(Error::vk("vkCopyImageToMemory"))
    }

    /// Destroys `t` once `after` has completed.
    pub fn free_texture(&self, t: Texture, after: TimelinePoint) {
        assert!(t.owns_image, "a swapchain image is not freed by its user");
        let mut t = std::mem::ManuallyDrop::new(t);
        let item = Item::Image {
            image: t.image,
            memory: t.memory.take(),
            views: t.views(),
        };
        // SAFETY: `t` is never used or dropped again; its Arc is released exactly once.
        unsafe { std::ptr::drop_in_place(&mut t.shared) };
        self.shared.defer(self.shared.after_point(after), item);
    }
}

pub(crate) fn vk_format(f: Format) -> vk::Format {
    use Format::*;
    use vk::Format as V;
    match f {
        R8Unorm => V::R8_UNORM,
        R8Snorm => V::R8_SNORM,
        R8Uint => V::R8_UINT,
        Rg8Unorm => V::R8G8_UNORM,
        Rg8Snorm => V::R8G8_SNORM,
        Rgba8Unorm => V::R8G8B8A8_UNORM,
        Rgba8Srgb => V::R8G8B8A8_SRGB,
        Rgba8Snorm => V::R8G8B8A8_SNORM,
        Rgba8Uint => V::R8G8B8A8_UINT,
        Bgra8Unorm => V::B8G8R8A8_UNORM,
        Bgra8Srgb => V::B8G8R8A8_SRGB,
        R16Unorm => V::R16_UNORM,
        R16Float => V::R16_SFLOAT,
        R16Uint => V::R16_UINT,
        Rg16Unorm => V::R16G16_UNORM,
        Rg16Float => V::R16G16_SFLOAT,
        Rg16Uint => V::R16G16_UINT,
        Rgba16Unorm => V::R16G16B16A16_UNORM,
        Rgba16Float => V::R16G16B16A16_SFLOAT,
        Rgba16Uint => V::R16G16B16A16_UINT,
        R32Float => V::R32_SFLOAT,
        R32Uint => V::R32_UINT,
        R32Sint => V::R32_SINT,
        Rg32Float => V::R32G32_SFLOAT,
        Rg32Uint => V::R32G32_UINT,
        Rgba32Float => V::R32G32B32A32_SFLOAT,
        Rgba32Uint => V::R32G32B32A32_UINT,
        R64Uint => V::R64_UINT,
        A2b10g10r10Unorm => V::A2B10G10R10_UNORM_PACK32,
        B10g11r11Ufloat => V::B10G11R11_UFLOAT_PACK32,
        E5b9g9r9Ufloat => V::E5B9G9R9_UFLOAT_PACK32,
        D16Unorm => V::D16_UNORM,
        D32Float => V::D32_SFLOAT,
        D32FloatS8Uint => V::D32_SFLOAT_S8_UINT,
        Bc1Unorm => V::BC1_RGBA_UNORM_BLOCK,
        Bc1Srgb => V::BC1_RGBA_SRGB_BLOCK,
        Bc3Unorm => V::BC3_UNORM_BLOCK,
        Bc3Srgb => V::BC3_SRGB_BLOCK,
        Bc4Unorm => V::BC4_UNORM_BLOCK,
        Bc4Snorm => V::BC4_SNORM_BLOCK,
        Bc5Unorm => V::BC5_UNORM_BLOCK,
        Bc5Snorm => V::BC5_SNORM_BLOCK,
        Bc6hUfloat => V::BC6H_UFLOAT_BLOCK,
        Bc6hSfloat => V::BC6H_SFLOAT_BLOCK,
        Bc7Unorm => V::BC7_UNORM_BLOCK,
        Bc7Srgb => V::BC7_SRGB_BLOCK,
    }
}

pub(crate) fn vk_filter(f: Filter) -> vk::Filter {
    match f {
        Filter::Nearest => vk::Filter::NEAREST,
        Filter::Linear => vk::Filter::LINEAR,
    }
}

pub(crate) fn vk_mipmap_mode(f: Filter) -> vk::SamplerMipmapMode {
    match f {
        Filter::Nearest => vk::SamplerMipmapMode::NEAREST,
        Filter::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

pub(crate) fn vk_address_mode(a: AddressMode) -> vk::SamplerAddressMode {
    match a {
        AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        AddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

pub(crate) fn vk_compare(c: CompareOp) -> vk::CompareOp {
    match c {
        CompareOp::Never => vk::CompareOp::NEVER,
        CompareOp::Less => vk::CompareOp::LESS,
        CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
        CompareOp::Equal => vk::CompareOp::EQUAL,
        CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
        CompareOp::Greater => vk::CompareOp::GREATER,
        CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
        CompareOp::Always => vk::CompareOp::ALWAYS,
    }
}
