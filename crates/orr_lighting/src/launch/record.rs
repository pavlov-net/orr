//! A source's paths recorded through the pipeline, in slices of its memory's slots.

use orr_scene::orr_gpu::{CommandList, Hazards, HitTable, Stages};

use super::memory::SourceMemory;
use super::stages::PathPipeline;
use crate::shaders::types::PathRoot;

impl PathPipeline {
    /// Records `items` paths of `root`'s source, over `hits`, in slices of `memory`'s slots, each
    /// after the one before it. Orders nothing after them.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        hits: &HitTable,
        root: &PathRoot,
        memory: &SourceMemory,
        items: u32,
    ) {
        let mut root = PathRoot {
            launch: memory.header.gpu(),
            ..*root
        };
        let raygen =
            self.raygens[root.source as usize].expect("the pipeline runs the root's source");
        let mut base = 0;
        while base < items {
            if base > 0 {
                list.barrier(Stages::RAY, Stages::RAY, Hazards::empty());
            }
            root.slice_base = base;
            root.slice_count = (items - base).min(memory.slots);
            list.trace_rays(
                &self.pipeline,
                raygen,
                hits,
                &root,
                [root.slice_count, 1, 1],
            );
            base += root.slice_count;
        }
    }
}
