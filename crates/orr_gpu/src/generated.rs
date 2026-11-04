//! Device-generated command sequences, alike on every backend.

use bytemuck::{Pod, Zeroable};

use crate::{ExecutionSet, GpuPtr};

/// One sequence of [`CommandList::execute_generated`](crate::CommandList::execute_generated): an index into the execution set, the
/// root pointer pushed for that pipeline and the workgroup counts of its dispatch (task
/// workgroups for mesh sets, mesh workgroups without a task stage). Kernels write sequences
/// with `store` of the `sequence` Slang module.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Sequence {
    pub pipeline: u32,
    /// The root's address, low word first.
    pub root: [u32; 2],
    pub groups: [u32; 3],
}

impl Sequence {
    pub fn new<R>(pipeline: u32, root: GpuPtr<R>, groups: [u32; 3]) -> Self {
        let a = root.address().0;
        Self {
            pipeline,
            root: [a as u32, (a >> 32) as u32],
            groups,
        }
    }
}

/// The kind of pipelines an execution set holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SetKind {
    Compute,
    Mesh,
}

/// One run of an execution set: `min(*count, max)` sequences read from `sequences`, an
/// allocation with room for `max`, with preprocess memory from
/// [`ExecutionSet::preprocess_layout`](crate::ExecutionSet::preprocess_layout) for `max`. Each execute takes preprocess memory no
/// earlier execute used.
#[derive(Clone, Copy)]
pub struct Generated<'a> {
    pub set: &'a ExecutionSet,
    pub sequences: GpuPtr<Sequence>,
    pub count: GpuPtr<u32>,
    pub max: u32,
    pub preprocess: GpuPtr<u8>,
}
