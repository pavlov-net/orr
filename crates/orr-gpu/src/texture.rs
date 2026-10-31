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
        /// Colour target of a mesh pipeline: rendering writes mip 0, layer 0. Not for 3D
        /// textures.
        const COLOR = 1 << 2;
        /// Depth target of a mesh pipeline, like `COLOR`.
        const DEPTH = 1 << 3;
        /// Written and read by the host ([`crate::Device::host_copy_to_texture`]) with no
        /// command list. Needs `Caps::host_image_copy`; the texture takes Upload memory and
        /// cannot be sparse.
        const HOST_COPY = 1 << 4;
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

/// One mip of a range of layers: what a texture copy region names besides its box.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureLayers {
    pub mip: u32,
    pub base_layer: u32,
    pub layers: u32,
}

impl TextureRegion {
    /// The mip and layers the region covers.
    pub fn layers(&self) -> TextureLayers {
        TextureLayers {
            mip: self.mip,
            base_layer: self.base_layer,
            layers: self.layers,
        }
    }

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

/// How a sparse texture's memory is bound: tiles above the mip tail, then the tail.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SparseLayout {
    /// Texels per tile per axis.
    pub tile_extent: [u32; 3],
    /// Bytes per tile, which is also the alignment of bound memory offsets.
    pub tile_bytes: u64,
    /// The first mip of the mip tail; the texture's mip count when it has no tail.
    pub tail_first_mip: u32,
    /// Bytes one mip tail binds, a whole number of tiles; 0 without a tail.
    pub tail_bytes: u64,
    /// One tail covers every layer; otherwise each layer binds its own.
    pub single_tail: bool,
}

impl SparseLayout {
    /// Tiles per axis of `mip` of a texture described by `desc`.
    pub fn tiles(&self, desc: &TextureDesc, mip: u32) -> [u32; 2] {
        let size = |e: u32, t: u32| (e >> mip).max(1).div_ceil(t);
        [
            size(desc.extent[0], self.tile_extent[0]),
            size(desc.extent[1], self.tile_extent[1]),
        ]
    }

    /// Mip tails a texture described by `desc` binds: 0 without a tail, 1 with
    /// `single_tail`, otherwise one per layer.
    pub fn tails(&self, desc: &TextureDesc) -> u32 {
        match (self.tail_bytes, self.single_tail) {
            (0, _) => 0,
            (_, true) => 1,
            (_, false) => desc.layers,
        }
    }

    /// The texels tile `tile` of `mip` covers, clipped to the mip, in every layer from
    /// `layer`.
    pub fn tile_region(
        &self,
        desc: &TextureDesc,
        mip: u32,
        layer: u32,
        tile: [u32; 2],
    ) -> TextureRegion {
        let size = |e: u32| (e >> mip).max(1);
        let x = tile[0] * self.tile_extent[0];
        let y = tile[1] * self.tile_extent[1];
        TextureRegion {
            mip,
            base_layer: layer,
            layers: 1,
            offset: [x, y, 0],
            extent: [
                self.tile_extent[0].min(size(desc.extent[0]) - x),
                self.tile_extent[1].min(size(desc.extent[1]) - y),
                1,
            ],
        }
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

/// A comparison, for depth tests and depth-compare samplers: `op(new, stored)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CompareOp {
    Never,
    Less,
    LessOrEqual,
    Equal,
    NotEqual,
    Greater,
    GreaterOrEqual,
    Always,
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
