bitflags::bitflags! {
    /// Pipeline stages a barrier orders.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct Stages: u32 {
        /// Copies and matrix conversions.
        const TRANSFER = 1 << 0;
        const COMPUTE = 1 << 1;
        const TASK = 1 << 2;
        const MESH = 1 << 3;
        const PIXEL = 1 << 4;
        /// Ray-dispatch pipelines; ignored without `Caps::ray_dispatch`.
        const RAY = 1 << 5;
        /// Colour attachment writes and blending.
        const COLOR_OUT = 1 << 6;
        /// Depth tests and writes.
        const DEPTH = 1 << 7;
        /// Acceleration-structure builds and copies.
        const AS_BUILD = 1 << 8;
        /// Host reads and writes of mapped memory.
        const HOST = 1 << 9;
        const ALL = 1 << 10;
        /// Memory decompression; ignored without `Caps::memory_decompression`.
        const DECOMPRESS = 1 << 11;
        /// Preprocessing of device-generated commands
        /// ([`crate::CommandList::preprocess_generated`]); ignored without
        /// `Caps::generated_commands`.
        const PREPROCESS = 1 << 12;
    }
}

bitflags::bitflags! {
    /// Reads a barrier also makes visible beyond shader reads in the `after` stages.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct Hazards: u32 {
        /// Indirect dispatch and draw arguments and indirect copy regions.
        const DRAW_ARGUMENTS = 1 << 0;
        /// Heap descriptors written on the GPU.
        const DESCRIPTORS = 1 << 1;
        /// Acceleration structures read by traces or later builds.
        const ACCEL_STRUCT = 1 << 2;
    }
}

/// A value on one queue's timeline. A submit returns the point its work signals when done.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TimelinePoint {
    pub(crate) timeline: u32,
    pub(crate) value: u64,
}

impl TimelinePoint {
    /// The timeline value.
    pub fn value(self) -> u64 {
        self.value
    }
}
