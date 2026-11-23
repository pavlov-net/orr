//! A source's paths recorded through the pipeline, in slices of its memory's slots.

use orr_scene::orr_gpu::{CommandList, GpuPtr, Hazards, HitTable, Stages};

use super::memory::SourceMemory;
use super::stages::PathPipeline;
use crate::shaders::types::{LaunchHeader, PathRoot, RayList};

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

impl PathPipeline {
    /// Records `launches` launches of `root`'s source over `hits`, each after the one before it,
    /// launch k over the slots of `header`, `slots` of them, from item k x `slots` on, as wide as
    /// the first word of `extents[k]` on the GPU says, which a barrier with
    /// [`Hazards::DRAW_ARGUMENTS`] orders before them. Orders nothing after them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_indirect(
        &self,
        list: &mut CommandList<'_>,
        hits: &HitTable,
        root: &PathRoot,
        header: GpuPtr<LaunchHeader>,
        slots: u32,
        extents: GpuPtr<RayList>,
        launches: u32,
    ) {
        let raygen =
            self.raygens[root.source as usize].expect("the pipeline runs the root's source");
        for k in 0..launches {
            if k > 0 {
                list.barrier(Stages::RAY, Stages::RAY, Hazards::empty());
            }
            let root = PathRoot {
                launch: header,
                slice_base: k * slots,
                slice_count: slots,
                ..*root
            };
            list.trace_rays_indirect(
                &self.pipeline,
                raygen,
                hits,
                &root,
                extents.offset(u64::from(k)).cast(),
            );
        }
    }
}
