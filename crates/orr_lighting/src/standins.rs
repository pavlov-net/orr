//! The shader bodies of the ray tracing pipeline path tracer as compute stand-ins
//! (`path/standins.slang`). A launch of a ray tracing pipeline runs every thread at the register
//! count of its largest linked shader, capped at 128, and the driver reports nothing of those
//! pipelines, so each shader's body, the one the pipeline ships, is compiled as a compute kernel
//! that takes its inputs from memory and stores what it hands on, and the driver's statistics of
//! that kernel say whether the shader fits its kind's budget.

use std::time::Instant;

use orr_scene::orr_gpu::{CodeStatistics, Device, ShaderStage, SpecData};

use crate::shaders::path::standins;
use crate::staged::SPEC_ENTRIES;
use crate::{Error, KernelSet, module};

/// The kind of shader a body runs as, which sets its budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderKind {
    Raygen,
    ClosestHit,
    Miss,
    Callable,
    AnyHit,
    Intersection,
}

impl ShaderKind {
    /// The most registers its body may take as a compute kernel, with no spill. A raygen costs
    /// what its body does; a closest-hit shader about 16 more, which callables and miss shaders
    /// are taken to share; an any-hit or intersection shader has less room than either. Each
    /// leaves the launch three allocation units under the cap of 128.
    pub const fn budget(self) -> u32 {
        match self {
            Self::Raygen => 96,
            Self::ClosestHit | Self::Miss | Self::Callable => 80,
            Self::AnyHit | Self::Intersection => 64,
        }
    }
}

/// A shader body of the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    /// A plain opaque hit's whole vertex: the closest-hit shader.
    Vertex,
    /// A path's first vertex, on the G-buffer's surface: its callable.
    First,
    /// A path ray's any-hit at a non-opaque triangle.
    CoverPath,
    /// A shadow ray's any-hit at a non-opaque triangle.
    CoverShadow,
    /// A medium proxy's intersection.
    BoxMedium,
    /// A shadow ray's any-hit at a medium proxy.
    ProxyShadow,
}

impl Body {
    pub const ALL: [Self; 6] = [
        Self::Vertex,
        Self::First,
        Self::CoverPath,
        Self::CoverShadow,
        Self::BoxMedium,
        Self::ProxyShadow,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::First => "first vertex",
            Self::CoverPath => "cover path",
            Self::CoverShadow => "cover shadow",
            Self::BoxMedium => "box medium",
            Self::ProxyShadow => "proxy shadow",
        }
    }

    pub fn kind(self) -> ShaderKind {
        match self {
            Self::Vertex => ShaderKind::ClosestHit,
            Self::CoverPath | Self::CoverShadow | Self::ProxyShadow => ShaderKind::AnyHit,
            Self::BoxMedium => ShaderKind::Intersection,
            Self::First => ShaderKind::Callable,
        }
    }

    fn entry(self) -> &'static str {
        match self {
            Self::Vertex => "vertex_hit",
            Self::First => "first_vertex",
            Self::CoverPath => "cover_path_hit",
            Self::CoverShadow => "cover_shadow_hit",
            Self::BoxMedium => "box_medium_hit",
            Self::ProxyShadow => "proxy_shadow_hit",
        }
    }
}

/// What the driver reports of a stand-in, and how long its pipeline took to create.
#[derive(Clone, Copy, Debug)]
pub struct BodyStatistics {
    pub code: CodeStatistics,
    pub seconds: f64,
}

impl BodyStatistics {
    /// Whether the body fits its kind's budget with nothing in local or shared memory: the
    /// stand-ins hold no arrays and no shared data, so either is spill.
    pub fn fits(&self, kind: ShaderKind) -> bool {
        self.code.registers <= kind.budget()
            && self.code.local_bytes == 0
            && self.code.shared_bytes == 0
    }
}

/// Compiles `body`'s stand-in for paths that may meet `may` (`path_may` in the low word,
/// `path_may2` in the high one), as the pipeline made for that mask folds its shader; `None`
/// where the device reports no statistics.
pub fn compile_body(
    device: &Device,
    kernels: &KernelSet,
    body: Body,
    may: u64,
) -> Result<Option<BodyStatistics>, Error> {
    let m = module(kernels, &standins::MODULE)?;
    let bytes = may.to_le_bytes();
    let spec = SpecData {
        entries: &SPEC_ENTRIES,
        bytes: &bytes,
    };
    let stage = ShaderStage::new(m, body.entry(), spec)?;
    let started = Instant::now();
    let pipeline =
        device.create_compute_pipeline(&stage, &format!("stand-in {} {may:#x}", body.name()))?;
    let seconds = started.elapsed().as_secs_f64();
    Ok(pipeline
        .statistics()
        .map(|code| BodyStatistics { code, seconds }))
}
