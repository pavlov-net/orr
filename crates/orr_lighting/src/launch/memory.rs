//! The slots of a source's paths in the pipeline and their records.

use orr_scene::orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::Error;
use crate::shaders::consts::{
    SLOT_INTERIOR_PART, SLOT_LAYER_PART, SLOT_LIGHT_PART, SLOT_SPECTRAL_PART, SLOT_STACK_PART,
    SLOT_WORDS, WALK_WORDS,
};
use crate::shaders::types::PathStages;
use crate::staged::{
    FEATURE_COATED_EMITTERS, FEATURE_DISPERSION, FEATURE_INTERIORS, FEATURE_LAYERS, FEATURE_STACK,
    FEATURE_WALK, RECORD_WORDS, UPDATE_RECORD_WORDS,
};

/// The words of a slot of paths whose root has `features`, of a view's or its background layer's
/// (`view`) or not, as `path/launch/slot.slang`'s `slot_stride` lays it out.
fn slot_words(features: u32, view: bool) -> u64 {
    let walk = view && features & (FEATURE_WALK | FEATURE_INTERIORS) != 0;
    let part = |on: bool, words: u32| if on { u64::from(words) } else { 0 };
    u64::from(SLOT_WORDS)
        + part(walk, WALK_WORDS)
        + part(features & FEATURE_LAYERS != 0, SLOT_LAYER_PART)
        + part(features & FEATURE_INTERIORS != 0, SLOT_INTERIOR_PART)
        + part(features & FEATURE_STACK != 0, SLOT_STACK_PART)
        + part(
            features & (FEATURE_STACK | FEATURE_COATED_EMITTERS) != 0,
            SLOT_LIGHT_PART,
        )
        + part(features & FEATURE_DISPERSION != 0, SLOT_SPECTRAL_PART)
}

/// Paths of a source one launch runs at most, which bounds the memory a view's paths keep: a
/// source with more runs them in slices.
const MAX_SLOTS: u32 = 1 << 20;

/// The slots of a slice of one source's paths in the pipeline and their records.
pub(crate) struct SourceMemory {
    pub(super) slots: u32,
    slot_words: u64,
    pub(super) header: Allocation<PathStages>,
    _words: Allocation<[u32; 4]>,
    _records: Allocation<[u32; 4]>,
}

impl SourceMemory {
    /// The slots of `items` paths, at most [`MAX_SLOTS`], of `slot_words` words, with records of
    /// `record_words` words.
    fn new(
        device: &Device,
        items: u32,
        slot_words: u64,
        record_words: u64,
        name: &str,
    ) -> Result<Self, Error> {
        let slots = slots(items);
        let n = u64::from(slots);
        let words = |count: u64, what: &str| {
            device.gpu_malloc::<[u32; 4]>(MemoryKind::DeviceLocal, count, &format!("{name} {what}"))
        };
        let state = words(slot_words * n, "paths")?;
        let records = words(record_words * n, "path records")?;
        let mut header = device.gpu_malloc::<PathStages>(MemoryKind::Upload, 1, name)?;
        header.as_mut_slice().expect("mapped")[0] = PathStages {
            capacity: slots,
            rounds: 0,
            queues: GpuPtr::NULL,
            lists: GpuPtr::NULL,
            words: state.gpu(),
            shadows: GpuPtr::NULL,
            records: records.gpu(),
            update_records: GpuPtr::NULL,
            update_first: slots,
            _pad0: 0,
        };
        Ok(Self {
            slots,
            slot_words,
            header,
            _words: state,
            _records: records,
        })
    }
}

fn slots(items: u32) -> u32 {
    items.clamp(1, MAX_SLOTS)
}

/// What a view's paths keep in the pipeline: its own paths' slots, which its background layer's
/// paths take after them, its update paths' and, where the view's are fewer than a frame's
/// photons take, its photons'.
pub(crate) struct LaunchMemory {
    pub(crate) view: SourceMemory,
    pub(crate) update: SourceMemory,
    photons: Option<SourceMemory>,
}

impl LaunchMemory {
    /// Slots for a slice of a view of `pixels` pixels whose paths' root has `features`, and one
    /// of its `update` update paths, whose root has `update_features`.
    pub(crate) fn new(
        device: &Device,
        pixels: u32,
        features: u32,
        update: u32,
        update_features: u32,
    ) -> Result<Self, Error> {
        let memory = Self {
            view: SourceMemory::new(
                device,
                pixels,
                slot_words(features, true),
                RECORD_WORDS,
                "launched view",
            )?,
            update: SourceMemory::new(
                device,
                update,
                slot_words(update_features, false),
                UPDATE_RECORD_WORDS,
                "launched update",
            )?,
            photons: None,
        };
        log::debug!(
            "launched paths: {} view slots of {} bytes and {} update slots of {} bytes",
            memory.view.slots,
            16 * memory.view.slot_words,
            memory.update.slots,
            16 * memory.update.slot_words
        );
        Ok(memory)
    }

    /// Whether it holds the slots [`Self::new`] makes for its arguments.
    pub(crate) fn fits(
        &self,
        pixels: u32,
        features: u32,
        update: u32,
        update_features: u32,
    ) -> bool {
        self.view.slots == slots(pixels)
            && self.view.slot_words == slot_words(features, true)
            && self.update.slots == slots(update)
            && self.update.slot_words == slot_words(update_features, false)
    }

    /// The slots photons whose root has `features` run in, `photons` of them a launch at least:
    /// the view's where it holds as many, else memory of their own.
    pub(crate) fn photons(
        &mut self,
        device: &Device,
        photons: u32,
        features: u32,
    ) -> Result<&SourceMemory, Error> {
        let words = slot_words(features, false);
        if self.view.slots >= photons && self.view.slot_words >= words {
            self.photons = None;
            return Ok(&self.view);
        }
        if !self
            .photons
            .as_ref()
            .is_some_and(|m| m.slots == slots(photons) && m.slot_words == words)
        {
            self.photons = None;
            self.photons = Some(SourceMemory::new(
                device,
                photons,
                words,
                RECORD_WORDS,
                "launched photons",
            )?);
        }
        Ok(self.photons.as_ref().expect("made above"))
    }
}
