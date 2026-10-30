use crate::Format;

/// The shape of a texture.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TextureKind {
    /// 2D, or a 2D array when `layers` > 1.
    D2,
    D3,
    /// Cube faces as layers; `layers` is a multiple of 6.
    Cube,
}

bitflags::bitflags! {
    /// How a texture is used. Copies to and from it are always allowed.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct TextureUsage: u32 {
        /// Sampled views ([`crate::TextureIndex`]).
        const SAMPLED = 1 << 0;
        /// Storage views ([`crate::StorageImageIndex`]).
        const STORAGE = 1 << 1;
        /// Colour target of a mesh pipeline.
        const COLOR = 1 << 2;
        /// Depth target of a mesh pipeline.
        const DEPTH = 1 << 3;
    }
}

/// A texture's shape, format and usage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureDesc {
    pub kind: TextureKind,
    pub format: Format,
    /// Width, height, depth; depth is 1 unless `kind` is [`TextureKind::D3`].
    pub extent: [u32; 3],
    pub mips: u32,
    pub layers: u32,
    pub usage: TextureUsage,
}

impl TextureDesc {
    /// A single-mip 2D texture.
    pub fn d2(format: Format, width: u32, height: u32, usage: TextureUsage) -> Self {
        Self {
            kind: TextureKind::D2,
            format,
            extent: [width, height, 1],
            mips: 1,
            layers: 1,
            usage,
        }
    }
}

/// The shape of a heap view.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ViewKind {
    D2,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// The part of a texture a heap view covers. The default covers every mip and layer with
/// the texture's natural kind.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct ViewDesc {
    /// `None`: 2D or 2D array by layer count, 3D, cube or cube array.
    pub kind: Option<ViewKind>,
    pub base_mip: u32,
    /// 0: every mip from `base_mip`.
    pub mips: u32,
    pub base_layer: u32,
    /// 0: every layer from `base_layer`.
    pub layers: u32,
}

/// A box of texels in one mip of a range of layers. Buffer data for it is tightly packed:
/// rows of whole blocks, then slices, then layers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureRegion {
    pub mip: u32,
    pub base_layer: u32,
    pub layers: u32,
    pub offset: [u32; 3],
    pub extent: [u32; 3],
}

impl TextureRegion {
    /// Every texel of `mip` in every layer of a texture described by `desc`.
    pub fn whole_mip(desc: &TextureDesc, mip: u32) -> Self {
        let size = |e: u32| (e >> mip).max(1);
        Self {
            mip,
            base_layer: 0,
            layers: desc.layers,
            offset: [0; 3],
            extent: [
                size(desc.extent[0]),
                size(desc.extent[1]),
                size(desc.extent[2]),
            ],
        }
    }

    /// Bytes of tightly packed data for the region in `format`.
    pub fn bytes(&self, format: Format) -> u64 {
        format.region_bytes(self.extent) * u64::from(self.layers)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    /// Clamps to transparent black.
    ClampToBorder,
}

/// Comparison for depth-compare samplers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CompareOp {
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

/// A sampler written into the sampler heap.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SamplerDesc {
    pub mag: Filter,
    pub min: Filter,
    pub mip: Filter,
    /// U, V, W.
    pub address: [AddressMode; 3],
    /// 1 or less: off.
    pub max_anisotropy: f32,
    pub compare: Option<CompareOp>,
    pub mip_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
}

impl Default for SamplerDesc {
    /// Trilinear, repeating, no anisotropy.
    fn default() -> Self {
        Self {
            mag: Filter::Linear,
            min: Filter::Linear,
            mip: Filter::Linear,
            address: [AddressMode::Repeat; 3],
            max_anisotropy: 1.0,
            compare: None,
            mip_bias: 0.0,
            min_lod: 0.0,
            max_lod: 1000.0,
        }
    }
}
