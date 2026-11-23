//! The slots of a source's paths in the pipeline and their records, as `path/state.slang`
//! lays them out: [`PATH_HOT_WORDS`] words of every slot and a record by the source, which the
//! launch's header (`LaunchHeader`) places.

use orr_scene::orr_gpu::{Allocation, Device, GpuPtr, MemoryKind};

use crate::Error;
use crate::shaders::consts::{
    PATH_COLD_WORDS, PATH_HOT_WORDS, PATH_MANIFOLD_WORDS, PATH_PHOTON_RECORD_WORDS,
    PATH_SPECTRAL_WORDS, PATH_UPDATE_RECORD_WORDS, PATH_VIEW_RECORD_WORDS,
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
    manifold: Option<Allocation<[u32; 4]>>,
    chain: Option<Allocation<[u32; 4]>>,
    spectral: Option<Allocation<[u32; 4]>>,
    debug: Option<Allocation<[u32; 4]>>,
    _records: Allocation<[u32; 4]>,
}

impl SourceMemory {
    /// The slots of `items` paths, at most [`MAX_SLOTS`], laid out as `layout`, with their cold
    /// words where `cold` holds, their manifold words where `manifold` does, their chain word
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
            manifold,
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
        let manifold = manifold
            .then(|| words(PATH_MANIFOLD_WORDS, "path manifold walks"))
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
            manifold: manifold.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
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
            manifold,
            chain,
            spectral,
            debug,
            _records: records,
        })
    }

    /// Its launch header, which places its slots.
    pub(crate) fn header(&self) -> GpuPtr<LaunchHeader> {
        self.header.gpu()
    }

    /// The paths a launch of its slots runs at most.
    pub(crate) fn slots(&self) -> u32 {
        self.slots
    }

    /// Whether it holds the slots [`Self::new`] makes of `items` paths laid out as `layout` with
    /// `parts`.
    pub(crate) fn fits(&self, items: u32, layout: SlotLayout, parts: SlotParts) -> bool {
        self.slots == slots(items) && self.layout == layout && self.parts() == parts
    }

    fn parts(&self) -> SlotParts {
        SlotParts {
            cold: self.cold.is_some(),
            manifold: self.manifold.is_some(),
            chain: self.chain.is_some(),
            spectral: self.spectral.is_some(),
            debug: self.debug.is_some(),
        }
    }
}

/// The parts a slot holds beside its hot words and its record: its cold words, which paths that
/// may meet a layer, an interior or a walk keep; the words of manifold next-event estimation, which
/// paths that a directional light may reach through the surfaces of solid glass keep; the chain
/// word, which paths that may gather caustic photons keep; the spectral words, which paths that
/// may split by wavelength keep; and the debug word, which paths whose pipeline may show a debug
/// view keep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SlotParts {
    pub(crate) cold: bool,
    pub(crate) manifold: bool,
    pub(crate) chain: bool,
    pub(crate) spectral: bool,
    pub(crate) debug: bool,
}

fn slots(items: u32) -> u32 {
    items.clamp(1, MAX_SLOTS)
}

/// What a view's paths keep in the pipeline: its own paths' slots and its update paths', each made
/// at the first frame that runs them.
#[derive(Default)]
pub(crate) struct LaunchMemory {
    view: Option<SourceMemory>,
    update: Option<SourceMemory>,
}

impl LaunchMemory {
    /// Slots for a slice of a view of `pixels` pixels with `parts`, made again where those it holds
    /// do not fit.
    pub(crate) fn view(
        &mut self,
        device: &Device,
        pixels: u32,
        parts: SlotParts,
    ) -> Result<&SourceMemory, Error> {
        fitted(
            &mut self.view,
            device,
            pixels,
            SlotLayout::view(),
            parts,
            "launched view",
        )
    }

    /// Slots for a slice of a view's `update` update paths with `parts`, made again where those it
    /// holds do not fit.
    pub(crate) fn update(
        &mut self,
        device: &Device,
        update: u32,
        parts: SlotParts,
    ) -> Result<&SourceMemory, Error> {
        fitted(
            &mut self.update,
            device,
            update,
            SlotLayout::update(),
            parts,
            "launched update",
        )
    }

    /// Whether a frame launched the view's paths.
    pub(crate) fn launched(&self) -> bool {
        self.view.is_some()
    }
}

/// The slots `memory` holds, made again for `items` paths laid out as `layout` with `parts` where
/// they do not fit.
fn fitted<'a>(
    memory: &'a mut Option<SourceMemory>,
    device: &Device,
    items: u32,
    layout: SlotLayout,
    parts: SlotParts,
    name: &str,
) -> Result<&'a SourceMemory, Error> {
    if !memory
        .as_ref()
        .is_some_and(|m| m.fits(items, layout, parts))
    {
        *memory = None;
        let made = SourceMemory::new(device, items, layout, parts, name)?;
        log::debug!(
            "{name}: {} slots of {} bytes",
            made.slots,
            16 * made.layout.slot_words()
        );
        *memory = Some(made);
    }
    Ok(memory.as_ref().expect("made above"))
}
