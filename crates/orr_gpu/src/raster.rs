use crate::{CompareOp, Format, ShaderStage, Texture};

/// Which triangles the rasterizer drops.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Cull {
    #[default]
    None,
    Front,
    Back,
}

/// The winding of front-facing triangles in framebuffer space.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum FrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}

/// How triangles are filled.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Fill {
    #[default]
    Solid,
    /// Edges only.
    Lines,
}

/// A depth offset added to every fragment: `constant` in units of the depth format's
/// resolution plus `slope` times the triangle's depth slope, limited to `clamp` when it is
/// not 0.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct DepthBias {
    pub constant: f32,
    pub slope: f32,
    pub clamp: f32,
}

/// Rasterizer state of a mesh pipeline.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct RasterState {
    pub cull: Cull,
    pub front_face: FrontFace,
    pub fill: Fill,
    pub depth_bias: Option<DepthBias>,
    /// Covers every pixel a triangle touches; needs `Caps::conservative_raster`.
    pub conservative: bool,
}

/// A blend factor.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
}

/// How a blend combines the weighted source and destination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BlendOp {
    Add,
    /// Source minus destination.
    Subtract,
    /// Destination minus source.
    ReverseSubtract,
    Min,
    Max,
}

/// One blend equation: `op(src * src_factor, dst * dst_factor)`. `Min` and `Max` ignore the
/// factors.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlendEquation {
    pub src: BlendFactor,
    pub dst: BlendFactor,
    pub op: BlendOp,
}

/// Blending of one color target.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Blend {
    pub color: BlendEquation,
    pub alpha: BlendEquation,
}

impl Blend {
    /// Premultiplied-alpha over: `src + dst * (1 - src.a)`.
    pub const PREMULTIPLIED: Blend = Blend {
        color: BlendEquation {
            src: BlendFactor::One,
            dst: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        },
        alpha: BlendEquation {
            src: BlendFactor::One,
            dst: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        },
    };

    /// `src + dst`.
    pub const ADDITIVE: Blend = Blend {
        color: BlendEquation {
            src: BlendFactor::One,
            dst: BlendFactor::One,
            op: BlendOp::Add,
        },
        alpha: BlendEquation {
            src: BlendFactor::One,
            dst: BlendFactor::One,
            op: BlendOp::Add,
        },
    };
}

bitflags::bitflags! {
    /// Color channels a target writes.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct ColorMask: u32 {
        const R = 1 << 0;
        const G = 1 << 1;
        const B = 1 << 2;
        const A = 1 << 3;
        const ALL = 0b1111;
    }
}

/// A color target of a mesh pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ColorTarget {
    pub format: Format,
    /// `None` writes the pixel output as it is.
    pub blend: Option<Blend>,
    pub write: ColorMask,
}

impl ColorTarget {
    /// A target of `format` that writes every channel without blending.
    pub fn new(format: Format) -> Self {
        Self {
            format,
            blend: None,
            write: ColorMask::ALL,
        }
    }
}

/// The depth target of a mesh pipeline: a fragment passes when `compare(fragment, stored)`
/// holds, and writes its depth when `write` is set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DepthTarget {
    pub format: Format,
    pub compare: CompareOp,
    pub write: bool,
}

/// The value a color attachment is cleared to, in the target format's class.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ClearColor {
    /// Float, unorm and snorm formats.
    Float([f32; 4]),
    /// Unsigned integer formats.
    Uint([u32; 4]),
    /// Signed integer formats.
    Sint([i32; 4]),
}

/// What a color attachment holds when rendering begins.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ColorLoad {
    /// Its current contents.
    Keep,
    Clear(ClearColor),
    /// Undefined contents; every pixel is about to be written.
    Discard,
}

/// What a depth attachment holds when rendering begins.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DepthLoad {
    Keep,
    Clear(f32),
    Discard,
}

/// A color attachment of [`Rendering`]: mip 0, layer 0 of a texture with `COLOR` usage.
#[derive(Clone, Copy)]
pub struct ColorAttachment<'a> {
    pub texture: &'a Texture,
    pub load: ColorLoad,
    /// Keep what rendering wrote; otherwise the contents are undefined afterwards.
    pub store: bool,
}

/// The depth attachment of [`Rendering`]: mip 0, layer 0 of a texture with `DEPTH` usage.
/// A depth-stencil format clears its stencil to 0 with `DepthLoad::Clear`.
#[derive(Clone, Copy)]
pub struct DepthAttachment<'a> {
    pub texture: &'a Texture,
    pub load: DepthLoad,
    pub store: bool,
}

/// What [`CommandList::begin_rendering`](crate::CommandList::begin_rendering) draws into. With no attachments the extent
/// alone sets the raster grid.
#[derive(Clone, Copy)]
pub struct Rendering<'a> {
    /// Width and height in pixels, from the origin.
    pub extent: [u32; 2],
    pub colors: &'a [ColorAttachment<'a>],
    pub depth: Option<DepthAttachment<'a>>,
}

/// How to build a mesh pipeline. Every stage reads the root pointer passed to the draw.
#[derive(Clone, Copy, Debug)]
pub struct MeshPipelineDesc<'a> {
    pub task: Option<ShaderStage<'a>>,
    pub mesh: ShaderStage<'a>,
    /// `None`: no pixel stage; only depth is written.
    pub pixel: Option<ShaderStage<'a>>,
    /// Color targets in attachment order.
    pub colors: &'a [ColorTarget],
    pub depth: Option<DepthTarget>,
    pub raster: RasterState,
    /// Rasterization samples per pixel: 1, or more for pipelines with no targets.
    pub samples: u32,
}

impl<'a> MeshPipelineDesc<'a> {
    /// A mesh stage alone: no targets, default raster state, one sample.
    pub fn new(mesh: ShaderStage<'a>) -> Self {
        Self {
            task: None,
            mesh,
            pixel: None,
            colors: &[],
            depth: None,
            raster: RasterState::default(),
            samples: 1,
        }
    }
}
