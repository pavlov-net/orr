use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4ComputeCommandEncoder, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLOrigin,
    MTLPixelFormat, MTLResidencySet, MTLResource, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use super::command::CommandList;
use super::device::{Device, Shared};
use super::memory::Allocation;
use super::sparse::SPARSE_PAGE;
use super::{Obj, lock};
use crate::{
    Error, Format, GpuPtr, SparseLayout, TextureDesc, TextureKind, TextureRegion, TextureUsage,
};

/// Where a texture's memory comes from.
pub enum TextureMemory<'a> {
    /// Its own memory.
    Dedicated,
    /// Inside a [`MemoryKind::DeviceLocal`](super::MemoryKind::DeviceLocal) block, at an
    /// offset aligned as [`Device::texture_requirements`] says. The block must outlive the
    /// texture.
    Placed {
        block: &'a Allocation<u8>,
        offset: u64,
    },
    /// No memory until [`Device::bind_sparse`] binds its tiles and mip tails
    /// ([`Texture::sparse_layout`]) to DeviceLocal blocks. 2D and cube color textures.
    Sparse,
}

/// A texture and its memory. It has no layout state.
///
/// Dropping it releases it once everything submitted before the drop has completed.
pub struct Texture {
    pub(super) raw: Obj<ProtocolObject<dyn MTLTexture>>,
    desc: TextureDesc,
    /// The texture is in the residency set itself, not through a heap.
    dedicated: bool,
    /// With [`TextureMemory::Sparse`].
    sparse: Option<(SparseLayout, MetalTail)>,
    shared: Arc<Shared>,
}

impl Texture {
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }

    /// How tiles and mip tails bind, for a texture created with [`TextureMemory::Sparse`].
    pub fn sparse_layout(&self) -> Option<&SparseLayout> {
        self.sparse.as_ref().map(|(l, _)| l)
    }

    pub(super) fn metal_tail(&self) -> Option<&MetalTail> {
        self.sparse.as_ref().map(|(_, t)| t)
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        let raw = self.raw.clone();
        let shared = Arc::clone(&self.shared);
        let dedicated = self.dedicated;
        self.shared.defer(Box::new(move || {
            if dedicated {
                let mut r = lock(&shared.residency);
                r.set.removeAllocation(ProtocolObject::from_ref(&*raw));
                r.dirty = true;
            }
            drop(raw);
        }));
    }
}

impl Device {
    /// Whether [`CommandList::copy_to_texture_indirect`] takes `format`; never on Metal yet.
    pub fn indirect_copy_format(&self, format: Format) -> bool {
        let _ = format;
        false
    }

    /// The size and alignment of a texture of `desc` placed in a block.
    pub fn texture_requirements(&self, desc: &TextureDesc) -> (u64, u64) {
        let d = descriptor(desc).expect("texture_requirements");
        let sa = self.shared.device.heapTextureSizeAndAlignWithDescriptor(&d);
        (sa.size as u64, sa.align as u64)
    }

    /// A texture of `desc` in `memory`, labeled `name`.
    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        memory: TextureMemory<'_>,
        name: &str,
    ) -> Result<Texture, Error> {
        if desc.usage.contains(TextureUsage::HOST_COPY) {
            return Err(Error::NotImplemented("host-copy textures"));
        }
        if desc.usage.contains(TextureUsage::DEPTH) != desc.format.is_depth() {
            return Err(Error::Invalid(format!(
                "{name}: DEPTH usage goes with depth formats only, not {:?}",
                desc.format
            )));
        }
        let d = descriptor(desc)?;
        let failed = || Error::Metal {
            call: "newTexture",
            message: format!("{name}: {desc:?}"),
        };
        let sparse = matches!(memory, TextureMemory::Sparse);
        if sparse && (desc.kind == TextureKind::D3 || desc.format.is_depth()) {
            return Err(Error::Invalid(format!(
                "{name}: sparse textures are 2D or cube color textures"
            )));
        }
        let (raw, dedicated) = match memory {
            TextureMemory::Dedicated | TextureMemory::Sparse => {
                if sparse {
                    d.setPlacementSparsePageSize(SPARSE_PAGE);
                }
                let t = self
                    .shared
                    .device
                    .newTextureWithDescriptor(&d)
                    .ok_or_else(failed)?;
                let mut r = lock(&self.shared.residency);
                r.set.addAllocation(ProtocolObject::from_ref(&*t));
                r.dirty = true;
                (t, true)
            }
            TextureMemory::Placed { block, offset } => {
                let heap = block.raw().heap.as_ref().ok_or_else(|| {
                    Error::Invalid(format!("{name}: placed in a block that is not DeviceLocal"))
                })?;
                // SAFETY: the offset and the texture's size lie within the heap (the caller
                // aligned the offset as `texture_requirements` says, and the block outlives
                // the texture).
                let t = unsafe { heap.newTextureWithDescriptor_offset(&d, offset as usize) }
                    .ok_or_else(failed)?;
                (t, false)
            }
        };
        raw.setLabel(Some(&NSString::from_str(name)));
        let sparse = sparse.then(|| self.sparse_layout(&raw, &d, desc));
        Ok(Texture {
            raw: Obj(raw),
            desc: *desc,
            dedicated,
            sparse,
            shared: Arc::clone(&self.shared),
        })
    }

    /// The tile shape and mip tail of the sparse texture `raw`. Every layer has its own tail.
    ///
    /// The tail starts at the first mip smaller than a tile along either axis, as on Vulkan
    /// with standard tile shapes and as orr_pack cooks textures. Metal's own tail can start
    /// later (it rounds the extent up to powers of two), so the mips between are tiled in
    /// Metal but bind as part of the tail here.
    fn sparse_layout(
        &self,
        raw: &ProtocolObject<dyn MTLTexture>,
        d: &MTLTextureDescriptor,
        desc: &TextureDesc,
    ) -> (SparseLayout, MetalTail) {
        let device = &self.shared.device;
        let tile_bytes = device.sparseTileSizeInBytesForSparsePageSize(SPARSE_PAGE) as u64;
        // SAFETY: one sample, the type and format of a texture Metal created.
        let g = unsafe {
            device.sparseTileSizeWithTextureType_pixelFormat_sampleCount_sparsePageSize(
                d.textureType(),
                d.pixelFormat(),
                1,
                SPARSE_PAGE,
            )
        };
        let tile_extent = [g.width as u32, g.height as u32, g.depth as u32];
        let metal_first = (raw.firstMipmapInTail() as u32).min(desc.mips);
        let metal = MetalTail {
            first_mip: metal_first,
            bytes: if metal_first < desc.mips {
                (raw.tailSizeInBytes() as u64).next_multiple_of(tile_bytes)
            } else {
                0
            },
        };
        let tail_first_mip = (0..metal_first)
            .find(|&m| {
                let size = |e: u32, t: u32| (e >> m).max(1) < t;
                size(desc.extent[0], tile_extent[0]) || size(desc.extent[1], tile_extent[1])
            })
            .unwrap_or(metal_first);
        let mut layout = SparseLayout {
            tile_extent,
            tile_bytes,
            tail_first_mip,
            tail_bytes: 0,
            single_tail: false,
        };
        let tiled: u64 = (tail_first_mip..metal_first)
            .map(|m| {
                let [x, y] = layout.tiles(desc, m);
                u64::from(x * y)
            })
            .sum();
        layout.tail_bytes = tiled * tile_bytes + metal.bytes;
        (layout, metal)
    }
}

/// Metal's own mip tail of a sparse texture: its first mip and the bytes one layer's maps.
pub(super) struct MetalTail {
    pub(super) first_mip: u32,
    pub(super) bytes: u64,
}

impl Device {
    pub fn host_copy_to_texture(
        &self,
        t: &Texture,
        region: &TextureRegion,
        data: &[u8],
    ) -> Result<(), Error> {
        let _ = (t, region, data);
        Err(Error::NotImplemented("Device::host_copy_to_texture"))
    }

    pub fn host_copy_from_texture(
        &self,
        out: &mut [u8],
        t: &Texture,
        region: &TextureRegion,
    ) -> Result<(), Error> {
        let _ = (out, t, region);
        Err(Error::NotImplemented("Device::host_copy_from_texture"))
    }
}

/// The Metal descriptor of `desc`: private storage, untracked.
fn descriptor(desc: &TextureDesc) -> Result<objc2::rc::Retained<MTLTextureDescriptor>, Error> {
    let d = MTLTextureDescriptor::new();
    let (kind, array) = match desc.kind {
        TextureKind::D2 if desc.layers > 1 => (MTLTextureType::Type2DArray, desc.layers),
        TextureKind::D2 => (MTLTextureType::Type2D, 1),
        TextureKind::D3 => (MTLTextureType::Type3D, 1),
        TextureKind::Cube if desc.layers > 6 => (MTLTextureType::TypeCubeArray, desc.layers / 6),
        TextureKind::Cube => (MTLTextureType::TypeCube, 1),
    };
    d.setTextureType(kind);
    d.setPixelFormat(pixel_format(desc.format)?);
    let mut usage = MTLTextureUsage::empty();
    if desc.usage.contains(TextureUsage::SAMPLED) {
        usage |= MTLTextureUsage::ShaderRead;
    }
    if desc.usage.contains(TextureUsage::STORAGE) {
        usage |= MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
        if matches!(
            desc.format,
            Format::R32Uint | Format::R32Sint | Format::R64Uint
        ) {
            usage |= MTLTextureUsage::ShaderAtomic;
        }
    }
    if desc
        .usage
        .intersects(TextureUsage::COLOR | TextureUsage::DEPTH)
    {
        usage |= MTLTextureUsage::RenderTarget;
    }
    d.setUsage(usage);
    d.setStorageMode(MTLStorageMode::Private);
    d.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
    // SAFETY: the sizes are the caller's texture description; Metal validates them.
    unsafe {
        d.setWidth(desc.extent[0] as usize);
        d.setHeight(desc.extent[1] as usize);
        d.setDepth(desc.extent[2].max(1) as usize);
        d.setMipmapLevelCount(desc.mips.max(1) as usize);
        d.setArrayLength(array.max(1) as usize);
    }
    Ok(d)
}

pub(super) fn pixel_format(f: Format) -> Result<MTLPixelFormat, Error> {
    use Format::*;
    Ok(match f {
        R8Unorm => MTLPixelFormat::R8Unorm,
        R8Snorm => MTLPixelFormat::R8Snorm,
        R8Uint => MTLPixelFormat::R8Uint,
        Rg8Unorm => MTLPixelFormat::RG8Unorm,
        Rg8Snorm => MTLPixelFormat::RG8Snorm,
        Rgba8Unorm => MTLPixelFormat::RGBA8Unorm,
        Rgba8Srgb => MTLPixelFormat::RGBA8Unorm_sRGB,
        Rgba8Snorm => MTLPixelFormat::RGBA8Snorm,
        Rgba8Uint => MTLPixelFormat::RGBA8Uint,
        Bgra8Unorm => MTLPixelFormat::BGRA8Unorm,
        Bgra8Srgb => MTLPixelFormat::BGRA8Unorm_sRGB,
        R16Unorm => MTLPixelFormat::R16Unorm,
        R16Float => MTLPixelFormat::R16Float,
        R16Uint => MTLPixelFormat::R16Uint,
        Rg16Unorm => MTLPixelFormat::RG16Unorm,
        Rg16Float => MTLPixelFormat::RG16Float,
        Rg16Uint => MTLPixelFormat::RG16Uint,
        Rgba16Unorm => MTLPixelFormat::RGBA16Unorm,
        Rgba16Float => MTLPixelFormat::RGBA16Float,
        Rgba16Uint => MTLPixelFormat::RGBA16Uint,
        R32Float => MTLPixelFormat::R32Float,
        R32Uint => MTLPixelFormat::R32Uint,
        R32Sint => MTLPixelFormat::R32Sint,
        Rg32Float => MTLPixelFormat::RG32Float,
        Rg32Uint => MTLPixelFormat::RG32Uint,
        Rgba32Float => MTLPixelFormat::RGBA32Float,
        Rgba32Uint => MTLPixelFormat::RGBA32Uint,
        // Metal's 64-bit texture atomics work on two-channel 32-bit texels.
        R64Uint => MTLPixelFormat::RG32Uint,
        A2b10g10r10Unorm => MTLPixelFormat::RGB10A2Unorm,
        B10g11r11Ufloat => MTLPixelFormat::RG11B10Float,
        E5b9g9r9Ufloat => MTLPixelFormat::RGB9E5Float,
        D16Unorm => MTLPixelFormat::Depth16Unorm,
        D32Float => MTLPixelFormat::Depth32Float,
        D32FloatS8Uint => MTLPixelFormat::Depth32Float_Stencil8,
        Bc1Unorm => MTLPixelFormat::BC1_RGBA,
        Bc1Srgb => MTLPixelFormat::BC1_RGBA_sRGB,
        Bc3Unorm => MTLPixelFormat::BC3_RGBA,
        Bc3Srgb => MTLPixelFormat::BC3_RGBA_sRGB,
        Bc4Unorm => MTLPixelFormat::BC4_RUnorm,
        Bc4Snorm => MTLPixelFormat::BC4_RSnorm,
        Bc5Unorm => MTLPixelFormat::BC5_RGUnorm,
        Bc5Snorm => MTLPixelFormat::BC5_RGSnorm,
        Bc6hUfloat => MTLPixelFormat::BC6H_RGBUfloat,
        Bc6hSfloat => MTLPixelFormat::BC6H_RGBFloat,
        Bc7Unorm => MTLPixelFormat::BC7_RGBAUnorm,
        Bc7Srgb => MTLPixelFormat::BC7_RGBAUnorm_sRGB,
        A4r4g4b4Unorm => return Err(Error::NotImplemented("A4r4g4b4Unorm textures")),
    })
}

/// The row and image pitches of tightly packed data for `region` of `format`, with rows
/// `row_texels` texels apart (0: the region's width).
fn pitches(format: Format, region: &TextureRegion, row_texels: u32) -> (u64, u64) {
    let b = format.block_extent();
    let row_texels = if row_texels == 0 {
        region.extent[0]
    } else {
        row_texels
    };
    let row = u64::from(row_texels.div_ceil(b)) * u64::from(format.block_bytes());
    (row, row * u64::from(region.extent[1].div_ceil(b)))
}

fn origin(o: [u32; 3]) -> MTLOrigin {
    MTLOrigin {
        x: o[0] as usize,
        y: o[1] as usize,
        z: o[2] as usize,
    }
}

fn extent(e: [u32; 3]) -> MTLSize {
    MTLSize {
        width: e[0] as usize,
        height: e[1] as usize,
        depth: e[2].max(1) as usize,
    }
}

impl CommandList<'_> {
    /// Copies tightly packed texels from `src` into `region` of `dst`.
    pub fn copy_to_texture(&mut self, dst: &Texture, region: &TextureRegion, src: GpuPtr<u8>) {
        self.copy_rows_to_texture(dst, region, src, 0);
    }

    /// Copies texels from `src`, whose rows start `row_texels` texels apart, into `region` of
    /// `dst`.
    pub fn copy_rows_to_texture(
        &mut self,
        dst: &Texture,
        region: &TextureRegion,
        src: GpuPtr<u8>,
        row_texels: u32,
    ) {
        let (row, image) = pitches(dst.desc.format, region, row_texels);
        let layer = image * u64::from(region.extent[2].max(1));
        let bytes = layer * u64::from(region.layers);
        let (buffer, offset) = self.buffer_at(src.address().0, bytes, "texture copy source");
        let e = self.compute();
        for l in 0..region.layers {
            // SAFETY: the source range lies within a live buffer (checked above); Metal
            // validates the texture region.
            unsafe {
                e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &buffer,
                    (offset + layer * u64::from(l)) as usize,
                    row as usize,
                    image as usize,
                    extent(region.extent),
                    &dst.raw,
                    (region.base_layer + l) as usize,
                    region.mip as usize,
                    origin(region.offset),
                );
            }
        }
        self.keep.push(Box::new(dst.raw.clone()));
    }

    /// Copies `region` of `src` into `dst`, tightly packed.
    pub fn copy_from_texture(&mut self, dst: GpuPtr<u8>, src: &Texture, region: &TextureRegion) {
        let (row, image) = pitches(src.desc.format, region, 0);
        let layer = image * u64::from(region.extent[2].max(1));
        let bytes = layer * u64::from(region.layers);
        let (buffer, offset) = self.buffer_at(dst.address().0, bytes, "texture copy destination");
        let e = self.compute();
        for l in 0..region.layers {
            // SAFETY: as in `copy_rows_to_texture`.
            unsafe {
                e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    &src.raw,
                    (region.base_layer + l) as usize,
                    region.mip as usize,
                    origin(region.offset),
                    extent(region.extent),
                    &buffer,
                    (offset + layer * u64::from(l)) as usize,
                    row as usize,
                    image as usize,
                );
            }
        }
        self.keep.push(Box::new(src.raw.clone()));
    }
}
