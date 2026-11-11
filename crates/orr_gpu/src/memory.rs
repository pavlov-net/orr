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
    /// Host-visible, coherent and cached system memory, mapped, for the CPU to fill while
    /// reading what it writes, as decoders do, and the GPU to copy or read from: host memory
    /// on a discrete GPU, shared storage on Metal. Where a device has no cached memory it is
    /// uncached, and where the host has no memory of its own, as on unified memory, it is on
    /// the GPU's heap, which `Caps::staging_device_local` reports. Allocations start at a
    /// host page of their memory.
    Staging,
}

impl MemoryKind {
    /// Every kind, in [`MemoryReport::kinds`] order.
    pub const ALL: [MemoryKind; 4] = [
        MemoryKind::Upload,
        MemoryKind::DeviceLocal,
        MemoryKind::Readback,
        MemoryKind::Staging,
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
    /// Indexed like [`MemoryKind::ALL`].
    pub kinds: [MemoryUsage; MemoryKind::ALL.len()],
    pub heaps: Vec<HeapReport>,
    /// Indexed like `kinds`: the index in `heaps` of the heap each kind's memory comes from.
    pub kind_heaps: [usize; MemoryKind::ALL.len()],
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

/// Live bytes and allocations of one kind that orr_gpu's allocator holds: `gpu_malloc`
/// buffers, transient arena blocks and textures with memory of their own. Bytes are what each
/// allocation takes of its memory.
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
    /// Bytes of freed device memory blocks of `block_bytes` that go back to the system once
    /// the GPU is done with them and [`Device::collect_garbage`] reaches them.
    ///
    /// [`Device::collect_garbage`]: crate::Device::collect_garbage
    pub freeing: u64,
}

impl HeapReport {
    /// What the process uses on the heap beyond orr_gpu's blocks: swapchains, SDKs and the
    /// driver's own memory. Negative while the OS count lags the blocks.
    pub fn outside(&self) -> i64 {
        self.usage as i64 - self.block_bytes as i64
    }
}

/// Where device memory goes, gathered on demand: [`Device::memory_report`]'s figures by
/// allocation label and by allocator pool.
///
/// [`Device::memory_report`]: crate::Device::memory_report
#[derive(Clone, Debug)]
pub struct MemoryBreakdown {
    /// Per label and kind with live allocations, by bytes, the largest first.
    pub labels: Vec<LabelUsage>,
    /// Per allocator pool: on Vulkan the VMA pools, on Metal the heap pools.
    pub pools: Vec<PoolUsage>,
    pub heaps: Vec<HeapReport>,
    /// Live device memory allocations (`VkDeviceMemory` or `MTLHeap`) orr_gpu holds.
    pub device_allocations: u64,
    /// The most the device allows at once, [`Limits::max_memory_allocations`].
    ///
    /// [`Limits::max_memory_allocations`]: crate::Limits::max_memory_allocations
    pub max_device_allocations: u64,
}

/// Live allocations of one label and kind: the name `gpu_malloc` or `create_texture` took,
/// without the index a ring of arenas adds to each block's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelUsage {
    pub label: Box<str>,
    pub kind: MemoryKind,
    pub bytes: u64,
    pub allocations: u64,
    /// Of `bytes`, those in device memory of their own.
    pub dedicated_bytes: u64,
    pub dedicated: u64,
}

/// What one allocator pool holds: its device memory blocks, dedicated ones among them, and
/// the allocations in them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolUsage {
    pub name: Box<str>,
    /// The index of its heap in [`MemoryBreakdown::heaps`].
    pub heap: usize,
    pub blocks: u64,
    pub block_bytes: u64,
    pub allocations: u64,
    pub allocation_bytes: u64,
}

impl PoolUsage {
    /// Bytes of its blocks no allocation takes.
    pub fn free_in_blocks(&self) -> u64 {
        self.block_bytes - self.allocation_bytes
    }
}

/// One live allocation's entry in a [`Ledger`], kept with the allocation so its release
/// needs no lookup.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Charge {
    label: u32,
    kind: MemoryKind,
    bytes: u64,
    dedicated: bool,
}

#[derive(Clone, Copy, Default)]
struct Count {
    bytes: u64,
    allocations: u64,
    dedicated_bytes: u64,
    dedicated: u64,
}

impl Count {
    fn add(&mut self, c: &Charge) {
        self.bytes += c.bytes;
        self.allocations += 1;
        if c.dedicated {
            self.dedicated_bytes += c.bytes;
            self.dedicated += 1;
        }
    }

    fn sub(&mut self, c: &Charge) {
        self.bytes -= c.bytes;
        self.allocations -= 1;
        if c.dedicated {
            self.dedicated_bytes -= c.bytes;
            self.dedicated -= 1;
        }
    }
}

/// Live allocations by label and kind, labels interned on first use.
#[derive(Default)]
pub(crate) struct Ledger(std::sync::Mutex<LedgerState>);

#[derive(Default)]
struct LedgerState {
    index: std::collections::HashMap<Box<str>, u32>,
    labels: Vec<(Box<str>, [Count; MemoryKind::ALL.len()])>,
    kinds: [Count; MemoryKind::ALL.len()],
}

impl Ledger {
    /// Counts an allocation of `bytes` under `label`.
    pub(crate) fn charge(
        &self,
        label: &str,
        kind: MemoryKind,
        bytes: u64,
        dedicated: bool,
    ) -> Charge {
        let mut s = self.0.lock().unwrap();
        let label = match s.index.get(label) {
            Some(&i) => i,
            None => {
                let i = s.labels.len() as u32;
                s.index.insert(label.into(), i);
                s.labels.push((label.into(), Default::default()));
                i
            }
        };
        let c = Charge {
            label,
            kind,
            bytes,
            dedicated,
        };
        s.labels[label as usize].1[kind.index()].add(&c);
        s.kinds[kind.index()].add(&c);
        c
    }

    /// Takes a freed allocation out.
    pub(crate) fn release(&self, c: Charge) {
        let mut s = self.0.lock().unwrap();
        s.labels[c.label as usize].1[c.kind.index()].sub(&c);
        s.kinds[c.kind.index()].sub(&c);
    }

    /// Live bytes and allocations per kind.
    pub(crate) fn kinds(&self) -> [MemoryUsage; MemoryKind::ALL.len()] {
        let s = self.0.lock().unwrap();
        s.kinds.map(|c| MemoryUsage {
            bytes: c.bytes,
            allocations: c.allocations,
        })
    }

    /// Per label and kind with live allocations, the largest first.
    pub(crate) fn labels(&self) -> Vec<LabelUsage> {
        let s = self.0.lock().unwrap();
        let mut out: Vec<LabelUsage> = s
            .labels
            .iter()
            .flat_map(|(label, kinds)| {
                MemoryKind::ALL
                    .iter()
                    .zip(kinds)
                    .filter(|(_, c)| c.allocations > 0)
                    .map(|(&kind, c)| LabelUsage {
                        label: label.clone(),
                        kind,
                        bytes: c.bytes,
                        allocations: c.allocations,
                        dedicated_bytes: c.dedicated_bytes,
                        dedicated: c.dedicated,
                    })
            })
            .collect();
        out.sort_by(|a, b| b.bytes.cmp(&a.bytes).then_with(|| a.label.cmp(&b.label)));
        out
    }
}
