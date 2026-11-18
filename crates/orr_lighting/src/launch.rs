//! The path tracer (`path/plain.slang`): ray tracing pipelines made per mask, whose
//! raygen runs a source's paths from their start to their end, a launch per slice of them,
//! through the scene's hit records; and the memory a view's paths keep in them.

mod memory;
mod record;
mod stages;

pub(crate) use memory::{LaunchMemory, SlotLayout, SlotParts, SourceMemory};
pub(crate) use stages::{PathPipeline, PathPipelines};
pub use stages::{PathStageSizes, PipelineSize, Role, StageSize, path_stage_sizes};
