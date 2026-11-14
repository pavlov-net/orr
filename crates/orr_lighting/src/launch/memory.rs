//! The slots of a source's paths in the pipeline and their records, as `path/launch/slot.slang`
//! lays them out: [`HOT_WORDS`] words of every slot at a fixed stride, the cold parts the root's
//! source and features keep, and a scratch, which the launch's header (`LaunchHeader`) places.

use orr_scene::orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::Error;
use crate::shaders::consts::{
    CANDIDATE_WORDS, FRONT_WORDS, HOT_WORDS, PART_NONE, SLOT_INTERIOR_PART, SLOT_LAYER_PART,
    SLOT_LIGHT_PART, SLOT_SPECTRAL_PART, SLOT_STACK_PART, WALK_WORDS,
};
use crate::shaders::types::LaunchHeader;
use crate::staged::{
    FEATURE_COATED_EMITTERS, FEATURE_DISPERSION, FEATURE_INTERIORS, FEATURE_LAYERS, FEATURE_STACK,
    FEATURE_WALK, RECORD_WORDS, UPDATE_RECORD_WORDS,
};

/// Paths of a source one launch runs at most, which bounds the memory a view's paths keep: a
/// source with more runs them in slices.
const MAX_SLOTS: u32 = 1 << 20;

/// What a slot of a root's paths keeps beside its hot words: its cold parts, each at its first
/// word or [`PART_NONE`], and its scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotLayout {
    cold_words: u32,
    scratch_words: u32,
    walk_at: u32,
    layer_at: u32,
    interior_at: u32,
    stack_at: u32,
    light_at: u32,
    spectral_at: u32,
}

impl SlotLayout {
    /// The slots of paths whose root has `features`, of a view's or its background layer's
    /// (`view`) or not, whose vertices draw at most `candidates` light candidates.
    pub(crate) fn new(features: u32, view: bool, candidates: u32) -> Self {
        let mut cold_words = 0;
        let mut part = |kept: bool, words: u32| {
            if !kept {
                return PART_NONE;
            }
            let at = cold_words;
            cold_words += words;
            at
        };
        let walk_at = part(
            view && features & (FEATURE_WALK | FEATURE_INTERIORS) != 0,
            WALK_WORDS,
        );
        let layer_at = part(features & FEATURE_LAYERS != 0, SLOT_LAYER_PART);
        let interior_at = part(features & FEATURE_INTERIORS != 0, SLOT_INTERIOR_PART);
        let stack_at = part(features & FEATURE_STACK != 0, SLOT_STACK_PART);
        let light_at = part(
            features & (FEATURE_STACK | FEATURE_COATED_EMITTERS) != 0,
            SLOT_LIGHT_PART,
        );
        let spectral_at = part(features & FEATURE_DISPERSION != 0, SLOT_SPECTRAL_PART);
        Self {
            cold_words,
            scratch_words: FRONT_WORDS.max(candidates * CANDIDATE_WORDS),
            walk_at,
            layer_at,
            interior_at,
            stack_at,
            light_at,
            spectral_at,
        }
    }

    /// The words of a slot: its hot words, its cold parts and its scratch.
    fn slot_words(&self) -> u64 {
        u64::from(HOT_WORDS) + u64::from(self.cold_words) + u64::from(self.scratch_words)
    }

    /// Whether slots laid out as `self` hold those of `other` in the same memory.
    fn holds(&self, other: &Self) -> bool {
        self.cold_words >= other.cold_words && self.scratch_words >= other.scratch_words
    }
}

/// The words a [`SourceMemory`] owns.
struct Words {
    _hot: Allocation<[u32; 4]>,
    _cold: Option<Allocation<[u32; 4]>>,
    _scratch: Allocation<[u32; 4]>,
    _records: Allocation<[u32; 4]>,
}

/// The slots of a slice of one source's paths in the pipeline and their records.
pub(crate) struct SourceMemory {
    pub(super) slots: u32,
    layout: SlotLayout,
    pub(super) header: Allocation<LaunchHeader>,
    /// `None` for paths that run in another source's words.
    _words: Option<Words>,
}

impl SourceMemory {
    /// The slots of `items` paths, at most [`MAX_SLOTS`], laid out as `layout`, with records of
    /// `record_words` words.
    fn new(
        device: &Device,
        items: u32,
        layout: SlotLayout,
        record_words: u64,
        name: &str,
    ) -> Result<Self, Error> {
        let slots = slots(items);
        let n = u64::from(slots);
        let words = |count: u64, what: &str| {
            device.gpu_malloc::<[u32; 4]>(
                MemoryKind::DeviceLocal,
                count * n,
                &format!("{name} {what}"),
            )
        };
        let hot = words(u64::from(HOT_WORDS), "paths")?;
        let cold = if layout.cold_words == 0 {
            None
        } else {
            Some(words(u64::from(layout.cold_words), "path parts")?)
        };
        let scratch = words(u64::from(layout.scratch_words), "path scratch")?;
        let records = words(record_words, "path records")?;
        let header = header(
            device,
            name,
            &layout,
            [
                hot.gpu(),
                cold.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
                scratch.gpu(),
                records.gpu(),
            ],
        )?;
        Ok(Self {
            slots,
            layout,
            header,
            _words: Some(Words {
                _hot: hot,
                _cold: cold,
                _scratch: scratch,
                _records: records,
            }),
        })
    }

    /// The slots of `items` paths laid out as `layout` in `words`' memory, which holds them.
    fn within(
        device: &Device,
        words: &SourceMemory,
        items: u32,
        layout: SlotLayout,
        name: &str,
    ) -> Result<Self, Error> {
        let theirs: &LaunchHeader = &words.header.as_slice().expect("mapped")[0];
        let header = header(
            device,
            name,
            &layout,
            [theirs.hot, theirs.cold, theirs.scratch, theirs.records],
        )?;
        Ok(Self {
            slots: slots(items),
            layout,
            header,
            _words: None,
        })
    }
}

/// The header of slots laid out as `layout` over the hot, cold, scratch and record words of
/// `words`.
fn header(
    device: &Device,
    name: &str,
    layout: &SlotLayout,
    words: [GpuPtr<[u32; 4]>; 4],
) -> Result<Allocation<LaunchHeader>, Error> {
    let [hot, cold, scratch, records] = words;
    let mut header = device.gpu_malloc::<LaunchHeader>(MemoryKind::Upload, 1, name)?;
    header.as_mut_slice().expect("mapped")[0] = LaunchHeader {
        hot,
        cold,
        scratch,
        records,
        cold_words: layout.cold_words,
        scratch_words: layout.scratch_words,
        walk_at: layout.walk_at,
        layer_at: layout.layer_at,
        interior_at: layout.interior_at,
        stack_at: layout.stack_at,
        light_at: layout.light_at,
        spectral_at: layout.spectral_at,
    };
    Ok(header)
}

fn slots(items: u32) -> u32 {
    items.clamp(1, MAX_SLOTS)
}

/// What a view's paths keep in the pipeline: its own paths' slots, which its background layer's
/// paths take after them, its update paths' and its photons', in the view's words where they hold
/// a frame's photons.
pub(crate) struct LaunchMemory {
    pub(crate) view: SourceMemory,
    pub(crate) update: SourceMemory,
    photons: Option<SourceMemory>,
}

impl LaunchMemory {
    /// Slots for a slice of a view of `pixels` pixels laid out as `view`, and one of its `update`
    /// update paths laid out as `update_layout`.
    pub(crate) fn new(
        device: &Device,
        pixels: u32,
        view: SlotLayout,
        update: u32,
        update_layout: SlotLayout,
    ) -> Result<Self, Error> {
        let memory = Self {
            view: SourceMemory::new(device, pixels, view, RECORD_WORDS, "launched view")?,
            update: SourceMemory::new(
                device,
                update,
                update_layout,
                UPDATE_RECORD_WORDS,
                "launched update",
            )?,
            photons: None,
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
    pub(crate) fn fits(
        &self,
        pixels: u32,
        view: SlotLayout,
        update: u32,
        update_layout: SlotLayout,
    ) -> bool {
        self.view.slots == slots(pixels)
            && self.view.layout == view
            && self.update.slots == slots(update)
            && self.update.layout == update_layout
    }

    /// The slots photons laid out as `layout` run in, `photons` of them a launch at least: the
    /// view's words where they hold as many, else memory of their own.
    pub(crate) fn photons(
        &mut self,
        device: &Device,
        photons: u32,
        layout: SlotLayout,
    ) -> Result<&SourceMemory, Error> {
        let within = self.view.slots >= photons && self.view.layout.holds(&layout);
        let kept = self.photons.as_ref().is_some_and(|m| {
            m.layout == layout && m.slots == slots(photons) && m._words.is_none() == within
        });
        if !kept {
            self.photons = None;
            self.photons = Some(if within {
                SourceMemory::within(device, &self.view, photons, layout, "launched photons")?
            } else {
                SourceMemory::new(device, photons, layout, RECORD_WORDS, "launched photons")?
            });
        }
        Ok(self.photons.as_ref().expect("made above"))
    }
}
