//! A view's transient targets: textures and buffers whose contents live within one render of
//! the view, placed so that targets whose spans in the fixed frame order do not overlap share
//! memory.

use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use orr_gpu::{
    Allocation, CommandList, Device, GpuPtr, Hazards, MemoryKind, MemoryPriority, Stages,
    StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage, ViewDesc,
};

use crate::Result;

/// The steps of one view's render in the fixed frame order. A render of a view enters them
/// in this order; steps it does not run are skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ViewStep {
    /// Culling, raster and the G-buffer resolve.
    Raster,
    /// The scene's lighting pass and the view's paths and guides.
    Lighting,
    /// The transparency point.
    Transparency,
    /// Ray reconstruction.
    Rr,
    /// Hook inputs, the scene-referred hook and the scene-referred screenshot.
    SceneReferred,
    /// Metering and adaptation.
    Exposure,
    Tonemap,
    /// The display-linear hook.
    DisplayLinear,
    /// UI, compose and the display screenshot.
    Compose,
    /// Passes the consumer records after the render, up to the end of the frame.
    After,
}

#[cfg(debug_assertions)]
impl ViewStep {
    const ALL: [Self; 10] = [
        Self::Raster,
        Self::Lighting,
        Self::Transparency,
        Self::Rr,
        Self::SceneReferred,
        Self::Exposure,
        Self::Tonemap,
        Self::DisplayLinear,
        Self::Compose,
        Self::After,
    ];
}

/// The steps a target's contents live through, both included.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    pub first: ViewStep,
    pub last: ViewStep,
}

impl Span {
    pub const fn new(first: ViewStep, last: ViewStep) -> Self {
        Self { first, last }
    }

    pub fn contains(&self, step: ViewStep) -> bool {
        self.first <= step && step <= self.last
    }

    pub fn overlaps(&self, other: &Span) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// What a target is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaKind {
    /// Sampled through one view of every mip and, with `STORAGE` usage, written through a
    /// storage view per mip.
    Texture(TextureDesc),
    /// Device-local bytes at a 256-byte aligned address.
    Buffer { bytes: u64 },
}

/// One target of a view's arena. Names are unique within an arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaRequest {
    pub name: &'static str,
    pub kind: ArenaKind,
    pub span: Span,
}

impl ArenaRequest {
    pub const fn texture(name: &'static str, desc: TextureDesc, span: Span) -> Self {
        Self {
            name,
            kind: ArenaKind::Texture(desc),
            span,
        }
    }

    pub const fn buffer(name: &'static str, bytes: u64, span: Span) -> Self {
        Self {
            name,
            kind: ArenaKind::Buffer { bytes },
            span,
        }
    }
}

/// Offsets within the shared block are multiples of this, so a buffer and an image never
/// share a `bufferImageGranularity` page unless they alias on purpose.
const PLACEMENT: u64 = 64 << 10;

/// The step and layout the arena is at, which debug builds check every target access
/// against.
#[cfg(debug_assertions)]
#[derive(Default)]
struct Shared {
    /// 1 + the `ViewStep` index; 0 before the first step is entered.
    step: AtomicU8,
    generation: AtomicU64,
}

/// Where a target sits, for access checks in debug builds.
#[derive(Clone)]
struct Place {
    #[cfg(debug_assertions)]
    name: &'static str,
    #[cfg(debug_assertions)]
    span: Span,
    #[cfg(debug_assertions)]
    generation: u64,
    #[cfg(debug_assertions)]
    shared: Arc<Shared>,
}

impl Place {
    #[inline]
    fn check(&self) {
        #[cfg(debug_assertions)]
        {
            let generation = self.shared.generation.load(Ordering::Relaxed);
            assert_eq!(
                self.generation, generation,
                "view target {} used after its arena was laid out again",
                self.name
            );
            let step = self.shared.step.load(Ordering::Relaxed);
            if let Some(step) = step.checked_sub(1).map(|i| ViewStep::ALL[usize::from(i)]) {
                assert!(
                    self.span.contains(step),
                    "view target {} used at {step:?}, outside its span {:?}",
                    self.name,
                    self.span
                );
            }
        }
    }
}

/// A texture of a [`ViewArena`]. Its accessors check, in debug builds, that the arena's
/// current step lies in its span.
#[derive(Clone)]
pub struct ArenaTexture {
    texture: Arc<Texture>,
    sampled: TextureIndex,
    storage: Vec<StorageImageIndex>,
    place: Place,
}

impl ArenaTexture {
    pub fn texture(&self) -> &Arc<Texture> {
        self.place.check();
        &self.texture
    }

    /// Every mip.
    pub fn sampled(&self) -> TextureIndex {
        self.place.check();
        self.sampled
    }

    /// `mip` alone; [`StorageImageIndex::NONE`] without `STORAGE` usage.
    pub fn storage(&self, mip: u32) -> StorageImageIndex {
        self.place.check();
        self.storage
            .get(mip as usize)
            .copied()
            .unwrap_or(StorageImageIndex::NONE)
    }
}

/// A buffer of a [`ViewArena`], checked like [`ArenaTexture`].
#[derive(Clone)]
pub struct ArenaBuffer {
    gpu: GpuPtr<u8>,
    place: Place,
}

impl ArenaBuffer {
    pub fn gpu<T>(&self) -> GpuPtr<T> {
        self.place.check();
        self.gpu.cast()
    }
}

enum Made {
    Texture(ArenaTexture),
    Buffer(ArenaBuffer),
}

struct Entry {
    request: ArenaRequest,
    /// Offset in the shared block; 0 without aliasing.
    offset: u64,
    bytes: u64,
    /// Shares bytes with another target, so it is discarded at its first step each render.
    aliased: bool,
    made: Made,
    /// Its own memory without aliasing.
    _own: Option<Allocation<u8>>,
}

/// One target's place in [`ArenaReport`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaTarget {
    pub name: &'static str,
    pub span: Span,
    /// Offset in the shared block; 0 without aliasing.
    pub offset: u64,
    pub bytes: u64,
    pub aliased: bool,
}

/// What a view's arena holds.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArenaReport {
    /// Device memory the targets take: the shared block with aliasing, else the sum.
    pub bytes: u64,
    /// The targets' sizes summed.
    pub unaliased_bytes: u64,
    pub targets: Vec<ArenaTarget>,
}

/// One view's transient targets. With aliasing they sit in one device-local block, and
/// targets whose spans do not overlap share bytes; without it each has its own memory.
/// A render lays the arena out once, before its first step ([`Self::layout`]), and then
/// enters each step it runs in order ([`Self::enter`]), which discards the aliased targets
/// that start there after a barrier from everything before. A target's contents are
/// undefined outside its span, and debug builds panic when a target is accessed outside it.
pub struct ViewArena {
    device: Device,
    aliasing: bool,
    entries: Vec<Entry>,
    block: Option<Allocation<u8>>,
    generation: u64,
    step: Option<ViewStep>,
    #[cfg(debug_assertions)]
    shared: Arc<Shared>,
}

impl ViewArena {
    pub fn new(device: &Device, aliasing: bool) -> Self {
        Self {
            device: device.clone(),
            aliasing,
            entries: Vec::new(),
            block: None,
            generation: 0,
            step: None,
            #[cfg(debug_assertions)]
            shared: Arc::default(),
        }
    }

    /// Changes on every new layout; targets taken before it are stale.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Makes the targets `requests` name, in place of the arena's, unless it holds exactly
    /// those already. New textures are discarded into `list`. Ends the arena's last render:
    /// no step is current until [`Self::enter`]. Returns whether it made new targets.
    pub fn layout(
        &mut self,
        requests: &[ArenaRequest],
        list: &mut CommandList<'_>,
    ) -> Result<bool> {
        self.set_step(None);
        let same = self.entries.len() == requests.len()
            && self
                .entries
                .iter()
                .zip(requests)
                .all(|(e, r)| e.request == *r);
        if same {
            return Ok(false);
        }
        for (i, r) in requests.iter().enumerate() {
            assert!(
                requests[..i].iter().all(|o| o.name != r.name),
                "view target {} requested twice",
                r.name
            );
        }
        self.release();
        self.generation += 1;
        #[cfg(debug_assertions)]
        self.shared
            .generation
            .store(self.generation, Ordering::Relaxed);

        let sizes: Vec<(u64, u64)> = requests.iter().map(|r| self.requirements(r)).collect();
        let offsets = if self.aliasing {
            place(requests, &sizes)
        } else {
            vec![0; requests.len()]
        };
        let end = offsets
            .iter()
            .zip(&sizes)
            .map(|(o, (bytes, _))| o + bytes)
            .max()
            .unwrap_or(0);
        if self.aliasing && end > 0 {
            self.block = Some(self.device.gpu_malloc_with_priority(
                MemoryKind::DeviceLocal,
                MemoryPriority::High,
                end,
                "view targets",
            )?);
        }
        for (i, r) in requests.iter().enumerate() {
            let (bytes, _) = sizes[i];
            let aliased = self.aliasing
                && (0..requests.len()).any(|j| {
                    j != i
                        && offsets[j] < offsets[i] + bytes
                        && offsets[i] < offsets[j] + sizes[j].0
                });
            let place = Place {
                #[cfg(debug_assertions)]
                name: r.name,
                #[cfg(debug_assertions)]
                span: r.span,
                #[cfg(debug_assertions)]
                generation: self.generation,
                #[cfg(debug_assertions)]
                shared: Arc::clone(&self.shared),
            };
            let (made, own) = match r.kind {
                ArenaKind::Texture(desc) => {
                    let memory = match &self.block {
                        Some(block) => TextureMemory::Placed {
                            block,
                            offset: offsets[i],
                        },
                        None => TextureMemory::Dedicated,
                    };
                    let texture = self.device.create_texture(&desc, memory, r.name)?;
                    list.discard(&texture);
                    (Made::Texture(self.views(texture, &desc, place)?), None)
                }
                ArenaKind::Buffer { bytes } => {
                    let (gpu, own) = match &self.block {
                        Some(block) => (block.gpu.offset(offsets[i]), None),
                        None => {
                            let a = self.device.gpu_malloc::<u8>(
                                MemoryKind::DeviceLocal,
                                bytes.max(1),
                                r.name,
                            )?;
                            (a.gpu, Some(a))
                        }
                    };
                    (Made::Buffer(ArenaBuffer { gpu, place }), own)
                }
            };
            self.entries.push(Entry {
                request: *r,
                offset: offsets[i],
                bytes,
                aliased,
                made,
                _own: own,
            });
        }
        Ok(true)
    }

    /// Enters `step` of a render: after one barrier from all earlier work, discards the
    /// aliased textures whose span starts there. Steps are entered in order.
    pub fn enter(&mut self, step: ViewStep, list: &mut CommandList<'_>) {
        debug_assert!(
            self.step.is_none_or(|s| s < step),
            "view step {step:?} entered after {:?}",
            self.step
        );
        self.set_step(Some(step));
        let mut starts = self
            .entries
            .iter()
            .filter(|e| e.aliased && e.request.span.first == step)
            .peekable();
        if starts.peek().is_none() {
            return;
        }
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        for e in starts {
            if let Made::Texture(t) = &e.made {
                list.discard(&t.texture);
            }
        }
    }

    /// The texture `name`, if the layout holds one.
    pub fn texture(&self, name: &str) -> Option<&ArenaTexture> {
        self.entries.iter().find_map(|e| match &e.made {
            Made::Texture(t) if e.request.name == name => Some(t),
            _ => None,
        })
    }

    /// The buffer `name`, if the layout holds one.
    pub fn buffer(&self, name: &str) -> Option<&ArenaBuffer> {
        self.entries.iter().find_map(|e| match &e.made {
            Made::Buffer(b) if e.request.name == name => Some(b),
            _ => None,
        })
    }

    pub fn report(&self) -> ArenaReport {
        let unaliased_bytes = self.entries.iter().map(|e| e.bytes).sum();
        ArenaReport {
            bytes: self.block.as_ref().map_or(unaliased_bytes, |b| b.size),
            unaliased_bytes,
            targets: self
                .entries
                .iter()
                .map(|e| ArenaTarget {
                    name: e.request.name,
                    span: e.request.span,
                    offset: e.offset,
                    bytes: e.bytes,
                    aliased: e.aliased,
                })
                .collect(),
        }
    }

    /// Drops every target, freeing their views once submitted work is done with them.
    fn release(&mut self) {
        for e in self.entries.drain(..) {
            if let Made::Texture(t) = &e.made {
                let d = &self.device;
                d.free_image_slot_after_submitted(t.sampled.0);
                for s in &t.storage {
                    d.free_image_slot_after_submitted(s.0);
                }
            }
        }
        self.block = None;
    }

    fn set_step(&mut self, step: Option<ViewStep>) {
        self.step = step;
        #[cfg(debug_assertions)]
        self.shared
            .step
            .store(step.map_or(0, |s| s as u8 + 1), Ordering::Relaxed);
    }

    /// Size and alignment of a target's memory.
    fn requirements(&self, r: &ArenaRequest) -> (u64, u64) {
        match r.kind {
            ArenaKind::Texture(desc) => self.device.texture_requirements(&desc),
            ArenaKind::Buffer { bytes } => (bytes.max(1), 256),
        }
    }

    fn views(&self, texture: Texture, desc: &TextureDesc, place: Place) -> Result<ArenaTexture> {
        let d = &self.device;
        let sampled = d.write_sampled(d.alloc_image_slot()?, &texture, &ViewDesc::default());
        let mut storage = Vec::new();
        if desc.usage.contains(TextureUsage::STORAGE) {
            for mip in 0..desc.mips {
                let view = ViewDesc {
                    base_mip: mip,
                    mips: 1,
                    ..ViewDesc::default()
                };
                storage.push(d.write_storage(d.alloc_image_slot()?, &texture, &view));
            }
        }
        Ok(ArenaTexture {
            texture: Arc::new(texture),
            sampled,
            storage,
            place,
        })
    }
}

impl Drop for ViewArena {
    fn drop(&mut self) {
        self.release();
    }
}

/// Offsets for `requests` (sizes and alignments in `sizes`): largest first, each at the
/// lowest offset clear of every target placed before it whose span overlaps its own.
fn place(requests: &[ArenaRequest], sizes: &[(u64, u64)]) -> Vec<u64> {
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(sizes[i].0), i));
    let mut offsets = vec![0u64; requests.len()];
    let mut placed: Vec<usize> = Vec::new();
    for &i in &order {
        let (bytes, align) = sizes[i];
        let align = align.max(PLACEMENT);
        let mut busy: Vec<(u64, u64)> = placed
            .iter()
            .filter(|&&j| requests[j].span.overlaps(&requests[i].span))
            .map(|&j| (offsets[j], offsets[j] + sizes[j].0))
            .collect();
        busy.sort_unstable();
        let mut at = 0;
        for (start, end) in busy {
            if at + bytes <= start {
                break;
            }
            at = at.max(end.next_multiple_of(align));
        }
        offsets[i] = at;
        placed.push(i);
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(name: &'static str, bytes: u64, first: ViewStep, last: ViewStep) -> ArenaRequest {
        ArenaRequest::buffer(name, bytes, Span::new(first, last))
    }

    #[test]
    fn disjoint_spans_share_and_overlapping_spans_do_not() {
        use ViewStep::*;
        let requests = [
            buffer("gbuffer", 4 * PLACEMENT, Raster, Transparency),
            buffer("radiance", 2 * PLACEMENT, Lighting, After),
            buffer("linear", 2 * PLACEMENT, SceneReferred, After),
            buffer("cull", PLACEMENT, Raster, Raster),
        ];
        let sizes: Vec<_> = requests
            .iter()
            .map(|r| match r.kind {
                ArenaKind::Buffer { bytes } => (bytes, 256),
                ArenaKind::Texture(_) => unreachable!(),
            })
            .collect();
        let offsets = place(&requests, &sizes);
        for i in 0..requests.len() {
            for j in 0..i {
                let disjoint =
                    offsets[i] + sizes[i].0 <= offsets[j] || offsets[j] + sizes[j].0 <= offsets[i];
                if requests[i].span.overlaps(&requests[j].span) {
                    assert!(
                        disjoint,
                        "{} and {} overlap",
                        requests[i].name, requests[j].name
                    );
                }
            }
        }
        // linear fits in the G-buffer's bytes, after which nothing reads them.
        assert!(offsets[2] + sizes[2].0 <= sizes[0].0);
        let end = (0..4).map(|i| offsets[i] + sizes[i].0).max().unwrap();
        assert_eq!(end, 6 * PLACEMENT);
    }
}
