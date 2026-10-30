use std::hash::{Hash, Hasher};

use crate::{CompareOp, Format};

/// Which triangles the rasteriser drops.
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

impl Hash for DepthBias {
    fn hash<H: Hasher>(&self, h: &mut H) {
        for v in [self.constant, self.slope, self.clamp] {
            v.to_bits().hash(h);
        }
    }
}

/// Rasteriser state of a mesh pipeline.
#[derive(Clone, Copy, PartialEq, Debug, Default, Hash)]
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

/// Blending of one colour target.
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
    /// Colour channels a target writes.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct ColorMask: u32 {
        const R = 1 << 0;
        const G = 1 << 1;
        const B = 1 << 2;
        const A = 1 << 3;
        const ALL = 0b1111;
    }
}

/// A colour target of a mesh pipeline.
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

/// The value a colour attachment is cleared to, in the target format's class.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ClearColor {
    /// Float, unorm and snorm formats.
    Float([f32; 4]),
    /// Unsigned integer formats.
    Uint([u32; 4]),
    /// Signed integer formats.
    Sint([i32; 4]),
}

/// What a colour attachment holds when rendering begins.
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
