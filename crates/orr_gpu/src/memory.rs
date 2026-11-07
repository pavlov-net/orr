//! Memory kinds, priorities and the memory report, alike on every backend.

/// Where an allocation lives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemoryKind {
    /// Host-visible and mapped, written by the CPU sequentially and read by the GPU:
    /// device-local with resizable BAR (`Caps::rebar`), host memory without, shared storage
    /// on Metal's unified memory. Possibly write-combined: the CPU should not read it.
    Upload,
    /// Device-local, not mapped. Textures can be placed in these blocks.
    DeviceLocal,
    /// Host-visible and cached, mapped, for the CPU to read back GPU results.
    Readback,
}

impl MemoryKind {
    /// Every kind, in [`MemoryReport::kinds`] order.
    pub const ALL: [MemoryKind; 3] = [
        MemoryKind::Upload,
        MemoryKind::DeviceLocal,
        MemoryKind::Readback,
    ];

    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// How firmly the OS keeps an allocation in VRAM when the process is over its budget, with
/// `Caps::memory_priority`. Allocations of different priorities never share a block. Metal
/// has no such control: every priority behaves alike there.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MemoryPriority {
    /// Paged out first: streaming pools, which shrink under pressure anyway.
    Low,
    /// Resident data; what [`Device::gpu_malloc`](crate::Device::gpu_malloc) gives.
    #[default]
    Normal,
    /// Paged out last: acceleration structures and tables every frame reads. Render
    /// targets and transient arenas take it too.
    High,
}

/// Memory use per [`MemoryKind`] and per heap, and the OS budget per heap.
#[derive(Clone, Debug)]
pub struct MemoryReport {
    /// Indexed like [`MemoryKind`]: Upload, DeviceLocal, Readback.
    pub kinds: [MemoryUsage; 3],
    pub heaps: Vec<HeapReport>,
    /// Indexed like `kinds`: the index in `heaps` of the heap each kind's memory comes from.
    pub kind_heaps: [usize; 3],
    /// Device memory allocations (`VkDeviceMemory` or `MTLHeap`) orr_gpu has made so far.
    pub device_allocations: u64,
}

impl MemoryReport {
    pub fn kind(&self, kind: MemoryKind) -> MemoryUsage {
        self.kinds[kind.index()]
    }

    /// The heap `kind`'s memory comes from.
    pub fn heap_of(&self, kind: MemoryKind) -> &HeapReport {
        &self.heaps[self.kind_heaps[kind.index()]]
    }

    /// The largest share of its OS budget a device-local heap uses; above 1 the OS pages
    /// memory out.
    pub fn pressure(&self) -> f64 {
        self.heaps
            .iter()
            .filter(|h| h.device_local)
            .map(|h| h.usage as f64 / h.budget as f64)
            .fold(0.0, f64::max)
    }
}

/// Live `gpu_malloc` bytes and allocations of one kind, transient arena blocks included.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct MemoryUsage {
    pub bytes: u64,
    pub allocations: u64,
}

/// One memory heap. Metal reports its unified memory as one device-local heap.
#[derive(Clone, Copy, Debug)]
pub struct HeapReport {
    pub size: u64,
    pub device_local: bool,
    /// Bytes of the device memory blocks orr_gpu holds on the heap.
    pub block_bytes: u64,
    /// What the OS lets this process use.
    pub budget: u64,
    /// What this process uses, as the OS counts it.
    pub usage: u64,
}
