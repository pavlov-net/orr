/// A texel format.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Format {
    R8Unorm,
    R8Snorm,
    R8Uint,
    Rg8Unorm,
    Rg8Snorm,
    Rgba8Unorm,
    Rgba8Srgb,
    Rgba8Snorm,
    Rgba8Uint,
    Bgra8Unorm,
    Bgra8Srgb,
    R16Unorm,
    R16Float,
    R16Uint,
    Rg16Unorm,
    Rg16Float,
    Rg16Uint,
    Rgba16Unorm,
    Rgba16Float,
    Rgba16Uint,
    R32Float,
    R32Uint,
    R32Sint,
    Rg32Float,
    Rg32Uint,
    Rgba32Float,
    Rgba32Uint,
    /// 64-bit integer texels for image atomics; shaders name it `RWTexture2D<uint64_t, 0, 40>`.
    R64Uint,
    A2b10g10r10Unorm,
    B10g11r11Ufloat,
    E5b9g9r9Ufloat,
    D16Unorm,
    D32Float,
    D32FloatS8Uint,
    Bc1Unorm,
    Bc1Srgb,
    Bc3Unorm,
    Bc3Srgb,
    Bc4Unorm,
    Bc4Snorm,
    Bc5Unorm,
    Bc5Snorm,
    Bc6hUfloat,
    Bc6hSfloat,
    Bc7Unorm,
    Bc7Srgb,
}

impl Format {
    /// Bytes per texel, or per 4x4 block for block-compressed formats. Depth-stencil
    /// formats count the depth aspect, the one copies move.
    pub const fn block_bytes(self) -> u32 {
        use Format::*;
        match self {
            R8Unorm | R8Snorm | R8Uint => 1,
            Rg8Unorm | Rg8Snorm | R16Unorm | R16Float | R16Uint | D16Unorm => 2,
            Rgba8Unorm | Rgba8Srgb | Rgba8Snorm | Rgba8Uint | Bgra8Unorm | Bgra8Srgb
            | Rg16Unorm | Rg16Float | Rg16Uint | R32Float | R32Uint | R32Sint
            | A2b10g10r10Unorm | B10g11r11Ufloat | E5b9g9r9Ufloat | D32Float | D32FloatS8Uint => 4,
            Rgba16Unorm | Rgba16Float | Rgba16Uint | Rg32Float | Rg32Uint | R64Uint => 8,
            Rgba32Float | Rgba32Uint => 16,
            Bc1Unorm | Bc1Srgb | Bc4Unorm | Bc4Snorm => 8,
            Bc3Unorm | Bc3Srgb | Bc5Unorm | Bc5Snorm | Bc6hUfloat | Bc6hSfloat | Bc7Unorm
            | Bc7Srgb => 16,
        }
    }

    /// Texels per block side: 4 for block-compressed formats, 1 otherwise.
    pub const fn block_extent(self) -> u32 {
        if self.is_compressed() { 4 } else { 1 }
    }

    pub const fn is_compressed(self) -> bool {
        use Format::*;
        matches!(
            self,
            Bc1Unorm
                | Bc1Srgb
                | Bc3Unorm
                | Bc3Srgb
                | Bc4Unorm
                | Bc4Snorm
                | Bc5Unorm
                | Bc5Snorm
                | Bc6hUfloat
                | Bc6hSfloat
                | Bc7Unorm
                | Bc7Srgb
        )
    }

    pub const fn is_depth(self) -> bool {
        matches!(
            self,
            Format::D16Unorm | Format::D32Float | Format::D32FloatS8Uint
        )
    }

    pub const fn has_stencil(self) -> bool {
        matches!(self, Format::D32FloatS8Uint)
    }

    /// Bytes a tightly packed `width` x `height` x `depth` region takes.
    pub const fn region_bytes(self, extent: [u32; 3]) -> u64 {
        let b = self.block_extent();
        extent[0].div_ceil(b) as u64
            * extent[1].div_ceil(b) as u64
            * extent[2] as u64
            * self.block_bytes() as u64
    }
}
