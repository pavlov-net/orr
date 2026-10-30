//! Textures: images that stay in `GENERAL`, with dedicated or placed memory.

use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use super::memory::{Allocation, MemoryKind, sharing};
use crate::{
    AddressMode, CompareOp, Error, Filter, Format, TextureDesc, TextureKind, TextureRegion,
    TextureUsage, TimelinePoint, ViewDesc, ViewKind,
};

/// Where a texture's memory comes from.
pub enum TextureMemory<'a> {
    /// Its own allocation.
    Dedicated,
    /// Inside a [`MemoryKind::DeviceLocal`] block, at an offset aligned as
    /// [`Device::texture_requirements`] says. The block must outlive the texture.
    Placed {
        block: &'a Allocation<u8>,
        offset: u64,
    },
}

/// An image and its memory. It has no layout state: [`crate::CommandList::discard`] puts it
/// in `GENERAL` before first use, and it stays there.
///
/// Dropping it destroys it once everything submitted before the drop has completed;
/// [`Device::free_texture`] names the point instead.
pub struct Texture {
    pub(crate) image: vk::Image,
    memory: Option<vk::DeviceMemory>,
    /// The view rendering writes through: mip 0, layer 0. Null without `COLOR` or `DEPTH`
    /// usage.
    attachment: vk::ImageView,
    desc: TextureDesc,
    /// False for a swapchain image: the swapchain owns it and drops this texture once its
    /// presents have completed.
    owns_image: bool,
    shared: Arc<Shared>,
}

impl Texture {
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
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
            desc: *desc,
            owns_image: false,
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
            // SAFETY: the swapchain drops its textures once no work uses them.
            unsafe { self.shared.device.destroy_image_view(self.attachment, None) };
            return;
        }
        self.shared.defer_after_submitted(Item::Image {
            image: self.image,
            memory: self.memory,
            view: self.attachment,
        });
    }
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

pub(crate) fn image_info<'a>(desc: &TextureDesc, families: &'a [u32]) -> vk::ImageCreateInfo<'a> {
    let mut usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    for (flag, vk_flag) in [
        (TextureUsage::SAMPLED, vk::ImageUsageFlags::SAMPLED),
        (TextureUsage::STORAGE, vk::ImageUsageFlags::STORAGE),
        (TextureUsage::COLOR, vk::ImageUsageFlags::COLOR_ATTACHMENT),
        (
            TextureUsage::DEPTH,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ),
    ] {
        if desc.usage.contains(flag) {
            usage |= vk_flag;
        }
    }
    let (image_type, flags) = match desc.kind {
        TextureKind::D2 => (vk::ImageType::TYPE_2D, vk::ImageCreateFlags::empty()),
        TextureKind::D3 => (vk::ImageType::TYPE_3D, vk::ImageCreateFlags::empty()),
        TextureKind::Cube => (
            vk::ImageType::TYPE_2D,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
        ),
    };
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

fn check(desc: &TextureDesc) -> Result<(), Error> {
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
    Ok(())
}

impl Device {
    /// Size and alignment of a texture's memory, for placing it in a block.
    pub fn texture_requirements(&self, desc: &TextureDesc) -> (u64, u64) {
        let info = image_info(desc, &self.shared.families);
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
    /// [`crate::CommandList::discard`] before its first use.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        memory: TextureMemory<'_>,
        name: &str,
    ) -> Result<Texture, Error> {
        check(desc)?;
        let shared = &self.shared;
        let d = &shared.device;
        let info = image_info(desc, &shared.families);
        // SAFETY: valid create info for this device.
        let image = unsafe { d.create_image(&info, None) }.map_err(Error::vk("vkCreateImage"))?;
        // SAFETY: `image` is live.
        let req = unsafe { d.get_image_memory_requirements(image) };
        let destroy = || {
            // SAFETY: the image is not used anywhere else.
            unsafe { d.destroy_image(image, None) };
        };
        let own = match memory {
            TextureMemory::Dedicated => {
                let ty = shared.memory.of(MemoryKind::DeviceLocal);
                let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
                let alloc = vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty)
                    .push(&mut dedicated);
                // SAFETY: valid allocate info; the device-local type was chosen to hold images.
                let mem = unsafe { d.allocate_memory(&alloc, None) };
                let mem = mem.map_err(|e| {
                    destroy();
                    Error::vk("vkAllocateMemory")(e)
                })?;
                // SAFETY: fresh image and dedicated memory.
                if let Err(e) = unsafe { d.bind_image_memory(image, mem, 0) } {
                    destroy();
                    // SAFETY: the memory is not used anywhere else.
                    unsafe { d.free_memory(mem, None) };
                    return Err(Error::vk("vkBindImageMemory")(e));
                }
                name_object(&shared.debug_fn, mem, name);
                Some(mem)
            }
            TextureMemory::Placed { block, offset } => {
                let raw = block.raw();
                let ty = shared.memory.of(raw.kind);
                let fits = req.memory_type_bits & (1 << ty) != 0
                    && offset % req.alignment == 0
                    && offset + req.size <= raw.size;
                if !fits {
                    destroy();
                    return Err(Error::Invalid(format!(
                        "{name}: {} bytes aligned to {} do not fit a {:?} block of {} bytes at offset {offset}",
                        req.size, req.alignment, raw.kind, raw.size
                    )));
                }
                // SAFETY: the range lies inside the block's memory, whose type the image accepts.
                unsafe { d.bind_image_memory(image, raw.memory, offset) }.map_err(|e| {
                    destroy();
                    Error::vk("vkBindImageMemory")(e)
                })?;
                None
            }
        };
        name_object(&shared.debug_fn, image, name);
        let mut t = Texture {
            image,
            memory: own,
            attachment: vk::ImageView::null(),
            desc: *desc,
            owns_image: true,
            shared: Arc::clone(shared),
        };
        t.create_attachment(name)?;
        Ok(t)
    }

    /// Destroys `t` once `after` has completed.
    pub fn free_texture(&self, t: Texture, after: TimelinePoint) {
        assert!(t.owns_image, "a swapchain image is not freed by its user");
        let mut t = std::mem::ManuallyDrop::new(t);
        let item = Item::Image {
            image: t.image,
            memory: t.memory,
            view: t.attachment,
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
