//! The slots of a source's paths in the pipeline and their records, as `path/lean/state.slang`
//! lays them out: [`LEAN_HOT_WORDS`] words of every slot and a record by the source, which the
//! launch's header (`LaunchHeader`) places.

use orr_scene::orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::Error;
use crate::shaders::consts::{
    LEAN_HOT_WORDS, LEAN_UPDATE_RECORD_WORDS, LEAN_VIEW_RECORD_WORDS, PART_NONE,
};
use crate::shaders::types::LaunchHeader;

/// Paths of a source one launch runs at most, which bounds the memory a view's paths keep: a
/// source with more runs them in slices.
const MAX_SLOTS: u32 = 1 << 20;

/// What a slot of a source's paths keeps beside its hot words: its record's words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotLayout {
    record_words: u32,
}

impl SlotLayout {
    /// The slots of a view's paths.
    pub(crate) fn view() -> Self {
        Self {
            record_words: LEAN_VIEW_RECORD_WORDS,
        }
    }

    /// The slots of its update paths.
    pub(crate) fn update() -> Self {
        Self {
            record_words: LEAN_UPDATE_RECORD_WORDS,
        }
    }

    /// The words of a slot: its hot words and its record.
    fn slot_words(self) -> u64 {
        u64::from(LEAN_HOT_WORDS) + u64::from(self.record_words)
    }
}

/// The slots of a slice of one source's paths in the pipeline and their records.
pub(crate) struct SourceMemory {
    pub(super) slots: u32,
    layout: SlotLayout,
    pub(super) header: Allocation<LaunchHeader>,
    _hot: Allocation<[u32; 4]>,
    _records: Allocation<[u32; 4]>,
}

impl SourceMemory {
    /// The slots of `items` paths, at most [`MAX_SLOTS`], laid out as `layout`.
    fn new(device: &Device, items: u32, layout: SlotLayout, name: &str) -> Result<Self, Error> {
        let slots = slots(items);
        let n = u64::from(slots);
        let words = |count: u32, what: &str| {
            device.gpu_malloc::<[u32; 4]>(
                MemoryKind::DeviceLocal,
                u64::from(count) * n,
                &format!("{name} {what}"),
            )
        };
        let hot = words(LEAN_HOT_WORDS, "paths")?;
        let records = words(layout.record_words, "path records")?;
        let mut header = device.gpu_malloc::<LaunchHeader>(MemoryKind::Upload, 1, name)?;
        header.as_mut_slice().expect("mapped")[0] = LaunchHeader {
            hot: hot.gpu(),
            cold: GpuPtr::NULL,
            scratch: GpuPtr::NULL,
            records: records.gpu(),
            cold_words: 0,
            scratch_words: 0,
            walk_at: PART_NONE,
            layer_at: PART_NONE,
            interior_at: PART_NONE,
            stack_at: PART_NONE,
            light_at: PART_NONE,
            spectral_at: PART_NONE,
        };
        Ok(Self {
            slots,
            layout,
            header,
            _hot: hot,
            _records: records,
        })
    }
}

fn slots(items: u32) -> u32 {
    items.clamp(1, MAX_SLOTS)
}

/// What a view's paths keep in the pipeline: its own paths' slots and its update paths'.
pub(crate) struct LaunchMemory {
    pub(crate) view: SourceMemory,
    pub(crate) update: SourceMemory,
}

impl LaunchMemory {
    /// Slots for a slice of a view of `pixels` pixels and one of its `update` update paths.
    pub(crate) fn new(device: &Device, pixels: u32, update: u32) -> Result<Self, Error> {
        let memory = Self {
            view: SourceMemory::new(device, pixels, SlotLayout::view(), "launched view")?,
            update: SourceMemory::new(device, update, SlotLayout::update(), "launched update")?,
        };
        log::debug!(
            "launched paths: {} view slots of {} bytes and {} update slots of {} bytes",
            memory.view.slots,
            16 * memory.view.layout.slot_words(),
            memory.update.slots,
            16 * memory.update.layout.slot_words()
        );
        Ok(memory)
    }

    /// Whether it holds the slots [`Self::new`] makes for its arguments.
    pub(crate) fn fits(&self, pixels: u32, update: u32) -> bool {
        self.view.slots == slots(pixels) && self.update.slots == slots(update)
    }
}
