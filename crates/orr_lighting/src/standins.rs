//! The shader bodies of the ray tracing pipeline path tracer as compute stand-ins
//! (`path/standins.slang`). A launch of a ray tracing pipeline runs every thread at the register
//! count of its largest linked shader, capped at 128, and the driver reports nothing of those
//! pipelines, so each body is compiled as a compute kernel that loads what its shader loads and
//! stores what it hands on, and the driver's statistics of that kernel say whether the shader
//! fits its kind's budget.

use std::time::Instant;

use orr_scene::orr_gpu::{CodeStatistics, Device, ShaderStage, SpecData};

use crate::shaders::path::standins;
use crate::staged::{MAY_BACKGROUND, MAY_PHOTON, MAY_RICH, MAY_UPDATE, MAY_VIEW, SPEC_ENTRIES};
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

/// A shader body of an opaque vertex.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Body {
    /// A hit into the packed surface: the closest-hit shader.
    Resolve,
    /// The light the vertex sends along the path.
    Emission,
    /// Next-event estimation of a surface with no lobe a plain one lacks.
    DirectPlain,
    /// Next-event estimation of a surface with every lobe.
    DirectRich,
    /// The next ray of a plain surface.
    ScatterPlain,
    /// The next ray of a rich surface.
    ScatterRich,
}

impl Body {
    pub const ALL: [Self; 6] = [
        Self::Resolve,
        Self::Emission,
        Self::DirectPlain,
        Self::DirectRich,
        Self::ScatterPlain,
        Self::ScatterRich,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Resolve => "resolve",
            Self::Emission => "emission",
            Self::DirectPlain => "direct plain",
            Self::DirectRich => "direct rich",
            Self::ScatterPlain => "scatter plain",
            Self::ScatterRich => "scatter rich",
        }
    }

    pub fn kind(self) -> ShaderKind {
        match self {
            Self::Resolve => ShaderKind::ClosestHit,
            _ => ShaderKind::Callable,
        }
    }

    fn entry(self) -> &'static str {
        match self {
            Self::Resolve => "resolve",
            Self::Emission => "emission",
            Self::DirectPlain | Self::DirectRich => "direct",
            Self::ScatterPlain | Self::ScatterRich => "scatter",
        }
    }

    /// What its stand-in for paths that may meet `may` may meet. The bodies are a view path's:
    /// the other sources' paths run their own start, record and finish shaders. A plain class's
    /// surfaces have none of the lobes a rich one may.
    fn may(self, may: u64) -> u64 {
        let view = may & !(MAY_BACKGROUND | MAY_UPDATE | MAY_PHOTON) | MAY_VIEW;
        match self {
            Self::DirectPlain | Self::ScatterPlain => view & !MAY_RICH,
            _ => view,
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
/// `path_may2` in the high one); `None` where the device reports no statistics.
pub fn compile_body(
    device: &Device,
    kernels: &KernelSet,
    body: Body,
    may: u64,
) -> Result<Option<BodyStatistics>, Error> {
    let m = module(kernels, &standins::MODULE)?;
    let bytes = body.may(may).to_le_bytes();
    let spec = SpecData {
        entries: &SPEC_ENTRIES,
        bytes: &bytes,
    };
    let stage = ShaderStage::new(m, body.entry(), spec)?;
    let started = Instant::now();
    let pipeline = device.create_compute_pipeline(
        &stage,
        &format!("stand-in {} {:#x}", body.name(), body.may(may)),
    )?;
    let seconds = started.elapsed().as_secs_f64();
    Ok(pipeline
        .statistics()
        .map(|code| BodyStatistics { code, seconds }))
}
