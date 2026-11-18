//! The slots of a source's paths in the pipeline and their records, as `path/state.slang`
//! lays them out: [`PATH_HOT_WORDS`] words of every slot and a record by the source, which the
//! launch's header (`LaunchHeader`) places.

use orr_scene::orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::Error;
use crate::shaders::consts::{
    PATH_COLD_WORDS, PATH_HOT_WORDS, PATH_PHOTON_RECORD_WORDS, PATH_SPECTRAL_WORDS,
    PATH_STACK_WORDS, PATH_UPDATE_RECORD_WORDS, PATH_VIEW_RECORD_WORDS,
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
            record_words: PATH_VIEW_RECORD_WORDS,
        }
    }

    /// The slots of its update paths.
    pub(crate) fn update() -> Self {
        Self {
            record_words: PATH_UPDATE_RECORD_WORDS,
        }
    }

    /// The slots of a frame's caustic photons.
    pub(crate) fn photon() -> Self {
        Self {
            record_words: PATH_PHOTON_RECORD_WORDS,
        }
    }

    /// The words of a slot: its hot words and its record.
    fn slot_words(self) -> u64 {
        u64::from(PATH_HOT_WORDS) + u64::from(self.record_words)
    }
}

/// The slots of a slice of one source's paths in the pipeline and their records.
pub(crate) struct SourceMemory {
    pub(super) slots: u32,
    layout: SlotLayout,
    pub(super) header: Allocation<LaunchHeader>,
    _hot: Allocation<[u32; 4]>,
    cold: Option<Allocation<[u32; 4]>>,
    stack: Option<Allocation<[u32; 4]>>,
    chain: Option<Allocation<[u32; 4]>>,
    spectral: Option<Allocation<[u32; 4]>>,
    debug: Option<Allocation<[u32; 4]>>,
    _records: Allocation<[u32; 4]>,
}

impl SourceMemory {
    /// The slots of `items` paths, at most [`MAX_SLOTS`], laid out as `layout`, with their cold
    /// words where `cold` holds, their plane stack's where `stack` does, their chain word
    /// (`PathChain`) where `chain` does, their spectral words (`PathSpectral`) where `spectral`
    /// does and their debug word where `debug` does.
    pub(crate) fn new(
        device: &Device,
        items: u32,
        layout: SlotLayout,
        parts: SlotParts,
        name: &str,
    ) -> Result<Self, Error> {
        let SlotParts {
            cold,
            stack,
            chain,
            spectral,
            debug,
        } = parts;
        let slots = slots(items);
        let n = u64::from(slots);
        let words = |count: u32, what: &str| {
            device.gpu_malloc::<[u32; 4]>(
                MemoryKind::DeviceLocal,
                u64::from(count) * n,
                &format!("{name} {what}"),
            )
        };
        let hot = words(PATH_HOT_WORDS, "paths")?;
        let cold = cold
            .then(|| words(PATH_COLD_WORDS, "path interiors"))
            .transpose()?;
        let stack = stack
            .then(|| words(PATH_STACK_WORDS, "path plane stacks"))
            .transpose()?;
        let chain = chain.then(|| words(1, "path chains")).transpose()?;
        let spectral = spectral
            .then(|| words(PATH_SPECTRAL_WORDS, "path wavelengths"))
            .transpose()?;
        let debug = debug.then(|| words(1, "path debug views")).transpose()?;
        let records = words(layout.record_words, "path records")?;
        let mut header = device.gpu_malloc::<LaunchHeader>(MemoryKind::Upload, 1, name)?;
        header.as_mut_slice().expect("mapped")[0] = LaunchHeader {
            hot: hot.gpu(),
            cold: cold.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
            stack: stack.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
            records: records.gpu(),
            chain: chain.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
            spectral: spectral.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
            debug: debug.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
        };
        Ok(Self {
            slots,
            layout,
            header,
            _hot: hot,
            cold,
            stack,
            chain,
            spectral,
            debug,
            _records: records,
        })
    }

    /// Whether it holds the slots [`Self::new`] makes of `items` paths laid out as `layout` with
    /// `parts`.
    pub(crate) fn fits(&self, items: u32, layout: SlotLayout, parts: SlotParts) -> bool {
        self.slots == slots(items) && self.layout == layout && self.parts() == parts
    }

    fn parts(&self) -> SlotParts {
        SlotParts {
            cold: self.cold.is_some(),
            stack: self.stack.is_some(),
            chain: self.chain.is_some(),
            spectral: self.spectral.is_some(),
            debug: self.debug.is_some(),
        }
    }
}

/// The parts a slot holds beside its hot words and its record: its cold words, which paths that
/// may meet a layer, an interior or a walk keep; the words of a plane stack, which paths that may
/// meet one keep; the chain word, which paths that may gather caustic photons keep; the spectral
/// words, which paths that may split by wavelength keep; and the debug word, which paths whose
/// pipeline may show a debug view keep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SlotParts {
    pub(crate) cold: bool,
    pub(crate) stack: bool,
    pub(crate) chain: bool,
    pub(crate) spectral: bool,
    pub(crate) debug: bool,
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
    /// Slots for a slice of a view of `pixels` pixels and one of its `update` update paths, with
    /// `parts`.
    pub(crate) fn new(
        device: &Device,
        pixels: u32,
        update: u32,
        parts: SlotParts,
    ) -> Result<Self, Error> {
        let view = SourceMemory::new(device, pixels, SlotLayout::view(), parts, "launched view")?;
        let update = SourceMemory::new(
            device,
            update,
            SlotLayout::update(),
            parts,
            "launched update",
        )?;
        let memory = Self { view, update };
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
    pub(crate) fn fits(&self, pixels: u32, update: u32, parts: SlotParts) -> bool {
        self.view.fits(pixels, SlotLayout::view(), parts)
            && self.update.fits(update, SlotLayout::update(), parts)
    }
}
