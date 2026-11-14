//! The pipeline path tracer (`path/launch.slang`): one ray tracing pipeline whose raygens run a
//! source's paths from their start to their end, a launch per slice of them, through the scene's
//! hit records; and the memory a view's paths keep in it.

mod memory;
mod record;
mod stages;

pub(crate) use memory::{LaunchMemory, SlotLayout};
pub(crate) use stages::{PathPipeline, Raygen};
