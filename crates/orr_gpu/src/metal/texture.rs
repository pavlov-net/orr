use super::command::CommandList;
use super::device::Device;
use super::memory::Allocation;
use super::not_implemented;
use crate::{Error, Format, GpuPtr, SparseLayout, TextureDesc, TextureRegion};

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
    /// No memory until bound; not supported on Metal yet.
    Sparse,
}

/// A texture and its memory. Not implemented on Metal yet: [`Device::create_texture`] fails.
pub struct Texture {
    desc: TextureDesc,
    never: std::convert::Infallible,
}

impl Texture {
    pub fn desc(&self) -> &TextureDesc {
        let _ = &self.desc;
        match self.never {}
    }

    pub fn sparse_layout(&self) -> Option<&SparseLayout> {
        match self.never {}
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
        let _ = desc;
        not_implemented("Device::texture_requirements")
    }

    pub fn create_texture(
        &self,
        desc: &TextureDesc,
        memory: TextureMemory<'_>,
        name: &str,
    ) -> Result<Texture, Error> {
        let _ = (desc, memory, name);
        Err(Error::NotImplemented("Device::create_texture"))
    }

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

impl CommandList<'_> {
    pub fn copy_to_texture(&mut self, dst: &Texture, region: &TextureRegion, src: GpuPtr<u8>) {
        let _ = (dst, region, src);
        not_implemented("CommandList::copy_to_texture")
    }

    pub fn copy_rows_to_texture(
        &mut self,
        dst: &Texture,
        region: &TextureRegion,
        src: GpuPtr<u8>,
        row_texels: u32,
    ) {
        let _ = (dst, region, src, row_texels);
        not_implemented("CommandList::copy_rows_to_texture")
    }

    pub fn copy_from_texture(&mut self, dst: GpuPtr<u8>, src: &Texture, region: &TextureRegion) {
        let _ = (dst, src, region);
        not_implemented("CommandList::copy_from_texture")
    }
}
