//! The pipeline path tracer (`path/launch.slang`): ray tracing pipelines made per mask, whose
//! raygen runs a source's paths from their start to their end, a launch per slice of them,
//! through the scene's hit records; and the memory a view's paths keep in them.

mod memory;
mod record;
mod stages;

pub(crate) use memory::{LaunchMemory, SlotLayout, SlotParts, SourceMemory};
pub use stages::{PIPELINE_CAP, Role, StageSize, over_cap, path_stage_sizes};
pub(crate) use stages::{PathPipeline, PathPipelines};
