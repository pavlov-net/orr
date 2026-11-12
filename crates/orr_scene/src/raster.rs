//! A view's raster half: instance and cluster culling over the scene's raster candidate
//! cut, the mesh-shader and software raster into the visibility buffer, both in two passes
//! around an HZB of the first, the opaque hook's draws in each, the dielectric and presence
//! clusters into the layer tiles, the material resolve per surface bin into the G-buffer and
//! motion, the final HZB, and the depth target.

use std::alloc::Layout;
use std::sync::Arc;

use bytemuck::Zeroable;
use orr_gpu::{
    Allocation, ColorTarget, CommandList, CompareOp, DepthAttachment, DepthLoad, DepthTarget,
    Device, DispatchArgs, Format, Generated, GpuPtr, Hazards, KernelSet, MemoryKind,
    MemoryPriority, MeshArgs, MeshPipeline, MeshPipelineDesc, ROOT_ALIGN, RasterState, Rendering,
    Sequence, ShaderStage, SpecData, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex,
    TextureMemory, TextureUsage, ViewDesc,
};

use crate::arena::{ArenaBuffer, ArenaRequest, ArenaTexture, Span, ViewArena, ViewStep};
use crate::drawn::{OpaqueHook, OpaquePass};
use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::shaders::{cull, hzb, raster, resolve};
use crate::transparency::ParticleDraws;
use crate::{
    ClusterWork, GBufferGpu, RasterGpu, RasterRoot, Result, SceneId, SceneRoot, ViewConstants,
    VisibleCluster,
};

/// `RasterGpu.flags` (`visibility.slang`).
const RASTER_OCCLUSION: u32 = 1;

/// `RasterGpu.counts` entries and `RasterGpu.args` offsets (`visibility.slang`).
const COUNTS: u64 = 18;
const COUNT_DRAWS: u64 = 3;
const ARGS: u64 = 9;
const ARGS_WORK: u64 = 0;
const ARGS_SOFTWARE: u64 = 3;
const ARGS_SMOKE: u64 = 6;
/// The lists mesh draws take, in `RasterGpu.draw_args` (`DRAW_*` in `visibility.slang`).
const DRAW_OPAQUE: u32 = 0;
const DRAW_ALPHA: u32 = 1;
const DRAW_COVERAGE: u32 = 2;
const DRAW_PARTICLES: u32 = 3;
const DRAW_LISTS: u32 = 4;
/// Workgroups per row of a workgroup grid (`GRID_X` in `visibility.slang`).
pub(crate) const GRID_X: u32 = 65535;
const MAX_HZB_MIPS: u32 = 16;
/// Pixels per side of a resolve tile and of a layer tile (`LAYER_TILE` in `gbuffer.slang`).
const TILE: u32 = 8;

/// How the material resolve runs its bins.
enum Resolve {
    /// One indirect dispatch per bin, empty or not.
    Recorded(PipelineId),
    /// `sequences` writes a sequence per bin with tiles, which one device-generated command
    /// runs on `set`, the resolve pipeline's execution set.
    Generated {
        sequences: PipelineId,
        set: PipelineId,
    },
}

/// The compute kernels are hot-reloadable [`Pipelines`]; the mesh pipelines are not.
pub(crate) struct RasterPipelines {
    clear: PipelineId,
    instances: PipelineId,
    cap: PipelineId,
    clusters: PipelineId,
    producers: PipelineId,
    raster_args: PipelineId,
    opaque: MeshPipeline,
    alpha: MeshPipeline,
    coverage: MeshPipeline,
    software: PipelineId,
    classify: PipelineId,
    resolve: Resolve,
    seed: PipelineId,
    reduce: PipelineId,
    depth: MeshPipeline,
    /// Surface bins: linked surfaces plus `StandardSurface`.
    pub(crate) bins: u32,
}

impl RasterPipelines {
    /// With `generated` the resolve runs as device-generated commands; the device must have
    /// `Caps::generated_commands`.
    pub(crate) fn new(
        device: &Device,
        kernels: &KernelSet,
        pipelines: &mut Pipelines,
        generated: bool,
    ) -> Result<Self> {
        let module = kernels.module(&raster::MODULE)?;
        let stage = |entry| ShaderStage::new(module, entry, SpecData::default());
        let mesh = |mesh: &'static str, pixel: &'static str, depth: Option<DepthTarget>| {
            let desc = MeshPipelineDesc {
                task: None,
                mesh: stage(mesh)?,
                pixel: Some(stage(pixel)?),
                colors: &[] as &[ColorTarget],
                depth,
                raster: RasterState::default(),
                samples: 1,
            };
            Ok::<_, crate::Error>(device.create_mesh_pipeline(&desc, mesh)?)
        };
        let none = SpecData::default;
        let resolve = if generated {
            Resolve::Generated {
                sequences: pipelines.create(
                    device,
                    kernels,
                    &resolve::MODULE,
                    "sequences",
                    none(),
                )?,
                set: pipelines.create_set(device, kernels, &resolve::MODULE, "resolve")?,
            }
        } else {
            Resolve::Recorded(pipelines.create(
                device,
                kernels,
                &resolve::MODULE,
                "resolve",
                none(),
            )?)
        };
        let mut compute = |own, entry| pipelines.create(device, kernels, own, entry, none());
        Ok(Self {
            clear: compute(&cull::MODULE, "clear")?,
            instances: compute(&cull::MODULE, "instances")?,
            cap: compute(&cull::MODULE, "cap")?,
            clusters: compute(&cull::MODULE, "clusters")?,
            producers: compute(&cull::MODULE, "producers")?,
            raster_args: compute(&cull::MODULE, "raster_args")?,
            opaque: mesh("cluster_mesh", "opaque_pixel", None)?,
            alpha: mesh("cluster_mesh", "alpha_pixel", None)?,
            coverage: mesh("coverage_mesh", "coverage_pixel", None)?,
            software: compute(&raster::MODULE, "software")?,
            classify: compute(&resolve::MODULE, "classify")?,
            resolve,
            seed: compute(&hzb::MODULE, "seed")?,
            reduce: compute(&hzb::MODULE, "reduce")?,
            depth: mesh(
                "fullscreen",
                "depth",
                Some(DepthTarget {
                    format: Format::D32Float,
                    compare: CompareOp::Always,
                    write: true,
                }),
            )?,
            bins: kernels.surfaces + 1,
        })
    }

    /// Preprocess memory one view's resolve takes from [`orr_gpu::DeviceArenas`], when it
    /// runs as device-generated commands.
    pub(crate) fn preprocess_layout(&self, pipelines: &Pipelines) -> Option<Layout> {
        match self.resolve {
            Resolve::Recorded(_) => None,
            Resolve::Generated { set, .. } => Some(pipelines.set(set).preprocess_layout(self.bins)),
        }
    }
}

/// A per-view texture kept between frames, with a sampled view and, unless it is the depth
/// target, storage views of each mip.
pub(crate) struct Target {
    pub(crate) texture: Arc<Texture>,
    pub(crate) sampled: TextureIndex,
    storage: Vec<StorageImageIndex>,
    device: Device,
}

impl Target {
    fn new(device: &Device, desc: &TextureDesc, name: &str) -> Result<Self> {
        let texture = device.create_texture(desc, TextureMemory::Dedicated, name)?;
        let sampled =
            device.write_sampled(device.alloc_image_slot()?, &texture, &ViewDesc::default());
        let mut storage = Vec::new();
        if desc.usage.contains(TextureUsage::STORAGE) {
            for mip in 0..desc.mips {
                let view = ViewDesc {
                    base_mip: mip,
                    mips: 1,
                    ..ViewDesc::default()
                };
                storage.push(device.write_storage(device.alloc_image_slot()?, &texture, &view));
            }
        }
        Ok(Self {
            texture: Arc::new(texture),
            sampled,
            storage,
            device: device.clone(),
        })
    }

    fn d2(device: &Device, format: Format, extent: [u32; 2], name: &str) -> Result<Self> {
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        Self::new(
            device,
            &TextureDesc::d2(format, extent[0], extent[1], usage),
            name,
        )
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        self.device.free_image_slot_after_submitted(self.sampled.0);
        for s in &self.storage {
            self.device.free_image_slot_after_submitted(s.0);
        }
    }
}

/// The G-buffer's textures, the extended targets only while a material needs them.
type GBufferTextures = (
    Arc<Texture>,
    Arc<Texture>,
    Option<Arc<Texture>>,
    Option<Arc<Texture>>,
);

/// Sizes of a view's raster targets past its lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TargetSizes {
    pub(crate) extent: [u32; 2],
    pub(crate) extended: bool,
    pub(crate) bins: u32,
}

/// Capacities of a view's cluster lists, which follow what its frames want.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ListSizes {
    /// Visible clusters, and so the entries of each raster list.
    pub(crate) visible_clusters: u32,
    /// Work the first culling pass leaves to the second.
    pub(crate) work: u32,
}

/// Visible clusters a view's list holds at most: a visibility buffer word names one in 25 bits
/// (`VISIBLE_CLUSTER_MAX`, `gbuffer.slang`).
pub(crate) const VISIBLE_CLUSTER_LIMIT: u32 = 1 << 25;

/// A view's cluster lists, each in memory of its own, so that a change of capacity leaves the
/// view's other targets as they are. The lists a frame's visible clusters size and the work
/// list are sized apart: a change of one keeps the other's memory.
pub(crate) struct ViewLists {
    visible: VisibleLists,
    work: Allocation<ClusterWork>,
    work_capacity: u32,
}

/// The lists that hold an entry per visible cluster, and the draws over them.
struct VisibleLists {
    capacity: u32,
    /// Mesh draws over each list they take, and the entries one covers at most.
    draws: u32,
    draw_groups: u32,
    clusters: Allocation<VisibleCluster>,
    opaque: Allocation<u32>,
    alpha: Allocation<u32>,
    software: Allocation<u32>,
    particles: Allocation<u32>,
    coverage: Allocation<u32>,
    smoke: Allocation<u32>,
    /// Per `DRAW_*` list, its draws' workgroups.
    draw_args: Allocation<MeshArgs>,
}

const LIST_MEMORY: (MemoryKind, MemoryPriority) = (MemoryKind::DeviceLocal, MemoryPriority::High);

impl VisibleLists {
    fn new(device: &Device, capacity: u32) -> Result<Self> {
        let (dl, high) = LIST_MEMORY;
        let visible = u64::from(capacity.max(1));
        let list = |name| device.gpu_malloc_with_priority::<u32>(dl, high, visible, name);
        // A grid fills whole rows, so a draw covers whole rows within the device's limit.
        let rows = device.caps().limits.max_mesh_draw_groups / GRID_X;
        let draw_groups = rows * GRID_X;
        let draws = capacity.div_ceil(draw_groups).max(1);
        Ok(Self {
            capacity,
            draws,
            draw_groups,
            clusters: device.gpu_malloc_with_priority(dl, high, visible, "visible clusters")?,
            opaque: list("opaque clusters")?,
            alpha: list("alpha clusters")?,
            software: list("software clusters")?,
            particles: list("particle clusters")?,
            coverage: list("coverage clusters")?,
            smoke: list("smoke clusters")?,
            draw_args: device.gpu_malloc_with_priority(
                dl,
                high,
                u64::from(DRAW_LISTS * draws),
                "cluster draw args",
            )?,
        })
    }
}

fn work_list(device: &Device, capacity: u32) -> Result<Allocation<ClusterWork>> {
    let (dl, high) = LIST_MEMORY;
    let n = u64::from(capacity.max(1));
    Ok(device.gpu_malloc_with_priority(dl, high, n, "cluster work")?)
}

impl ViewLists {
    pub(crate) fn new(device: &Device, sizes: ListSizes) -> Result<Self> {
        Ok(Self {
            visible: VisibleLists::new(device, sizes.visible_clusters)?,
            work: work_list(device, sizes.work)?,
            work_capacity: sizes.work,
        })
    }

    pub(crate) fn sizes(&self) -> ListSizes {
        ListSizes {
            visible_clusters: self.visible.capacity,
            work: self.work_capacity,
        }
    }

    /// Takes the capacities of `sizes`, replacing only the lists whose capacity changes; the
    /// frames in flight keep the replaced ones. A list that finds no memory stays as it is,
    /// with the error, beside the lists that took theirs.
    pub(crate) fn resize(&mut self, device: &Device, sizes: ListSizes) -> Result<()> {
        if sizes.work != self.work_capacity {
            self.work = work_list(device, sizes.work)?;
            self.work_capacity = sizes.work;
        }
        if sizes.visible_clusters != self.visible.capacity {
            self.visible = VisibleLists::new(device, sizes.visible_clusters)?;
        }
        Ok(())
    }

    /// The workgroups of draw `i` over list `list` (`DRAW_*`).
    fn draw_args(&self, list: u32, i: u32) -> GpuPtr<MeshArgs> {
        let v = &self.visible;
        v.draw_args.gpu().offset(u64::from(list * v.draws + i))
    }
}

/// Arena names of the raster's transient targets.
const VISIBILITY: &str = "visibility buffer";
const LAYER_TILES: &str = "layer tiles";
const BIN_ARGS: &str = "bin args";
const BIN_TILES: &str = "bin tiles";
const BIN_SEQUENCES: &str = "bin sequences";
const DEPTH_VALUES: &str = "resolved depth";
const GBUF_A: &str = "gbuf_a";
const GBUF_B: &str = "gbuf_b";
const GBUF_X: &str = "gbuf_x";
const GBUF_Y: &str = "gbuf_y";
const OVERLAY: &str = "transparency overlay";

/// Raster's bins and the depth it resolves, which only the raster half reads.
const CULL: Span = Span::new(ViewStep::Raster, ViewStep::Raster);
/// The G-buffer and what `load_gsurface` and the transparency point read beside it.
const GBUFFER: Span = Span::new(ViewStep::Raster, ViewStep::Transparency);
/// Ray reconstruction may composite the overlay itself.
const OVERLAY_SPAN: Span = Span::new(ViewStep::Transparency, ViewStep::Rr);

/// The raster's targets in the view's arena, taken at one layout.
struct Transient {
    generation: u64,
    visibility: ArenaBuffer,
    layer_tiles: ArenaBuffer,
    bin_args: ArenaBuffer,
    bin_tiles: ArenaBuffer,
    bin_sequences: ArenaBuffer,
    depth_values: ArenaBuffer,
    a: ArenaTexture,
    b: ArenaTexture,
    x: Option<ArenaTexture>,
    y: Option<ArenaTexture>,
    overlay: ArenaTexture,
}

impl Transient {
    fn take(arena: &ViewArena) -> Self {
        let buffer = |name| arena.buffer(name).expect("raster target").clone();
        let texture = |name| arena.texture(name).cloned();
        Self {
            generation: arena.generation(),
            visibility: buffer(VISIBILITY),
            layer_tiles: buffer(LAYER_TILES),
            bin_args: buffer(BIN_ARGS),
            bin_tiles: buffer(BIN_TILES),
            bin_sequences: buffer(BIN_SEQUENCES),
            depth_values: buffer(DEPTH_VALUES),
            a: texture(GBUF_A).expect("raster target"),
            b: texture(GBUF_B).expect("raster target"),
            x: texture(GBUF_X),
            y: texture(GBUF_Y),
            overlay: texture(OVERLAY).expect("raster target"),
        }
    }
}

/// One view history's raster targets, kept while its extent and sizes hold: the ones read
/// after the view's render or by its next frame, and the transient ones of its arena.
pub(crate) struct ViewTargets {
    pub(crate) sizes: TargetSizes,
    /// Per instance slot of the scene rendered, written by every render before it is read.
    slot_visibility: Allocation<u32>,
    slots: u32,
    /// The slot visibility a render of this frame may read, replaced by a larger one.
    replaced: Vec<(u64, Allocation<u32>)>,
    /// Workgroup args (`ARGS`), then counts.
    args: Allocation<u32>,
    pub(crate) depth: Target,
    pub(crate) motion: Target,
    pub(crate) hzb: Target,
    transient: Transient,
    /// The scene and output extent the HZB holds the last render of.
    pub(crate) hzb_view: Option<(SceneId, [u32; 2])>,
}

fn tiles(extent: [u32; 2]) -> [u32; 2] {
    [extent[0].div_ceil(TILE), extent[1].div_ceil(TILE)]
}

fn hzb_mips(extent: [u32; 2]) -> u32 {
    (32 - extent[0].max(extent[1]).max(1).leading_zeros()).min(MAX_HZB_MIPS)
}

/// The visibility of `slots` instance slots.
fn slot_visibility(device: &Device, slots: u32) -> Result<Allocation<u32>> {
    Ok(device.gpu_malloc(
        MemoryKind::DeviceLocal,
        u64::from(slots.max(1)),
        "slot visibility",
    )?)
}

impl ViewTargets {
    /// The transient targets the raster half takes from the view's arena.
    pub(crate) fn requests(s: &TargetSizes) -> Vec<ArenaRequest> {
        let [w, h] = s.extent;
        let pixels = u64::from(w) * u64::from(h);
        let [tx, ty] = tiles(s.extent);
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        let d2 = |format| TextureDesc::d2(format, w, h, usage);
        let mut r = vec![
            ArenaRequest::buffer(VISIBILITY, 8 * pixels, GBUFFER),
            // Then the word marking every tile.
            ArenaRequest::buffer(
                LAYER_TILES,
                4 * (u64::from(tx) * u64::from(ty) + 1),
                GBUFFER,
            ),
            ArenaRequest::texture(GBUF_A, d2(Format::Rgba32Uint), GBUFFER),
            ArenaRequest::texture(GBUF_B, d2(Format::Rg32Uint), GBUFFER),
            // Then the sequence count.
            ArenaRequest::buffer(BIN_ARGS, 12 * u64::from(s.bins) + 4, CULL),
            ArenaRequest::buffer(
                BIN_SEQUENCES,
                size_of::<Sequence>() as u64 * u64::from(s.bins),
                CULL,
            ),
            ArenaRequest::buffer(
                BIN_TILES,
                4 * u64::from(s.bins) * u64::from(tx) * u64::from(ty),
                CULL,
            ),
            ArenaRequest::buffer(DEPTH_VALUES, 4 * pixels, CULL),
            ArenaRequest::texture(
                OVERLAY,
                TextureDesc::d2(
                    Format::Rgba16Float,
                    w,
                    h,
                    TextureUsage::SAMPLED | TextureUsage::COLOR,
                ),
                OVERLAY_SPAN,
            ),
        ];
        if s.extended {
            for name in [GBUF_X, GBUF_Y] {
                r.push(ArenaRequest::texture(name, d2(Format::Rgba32Uint), GBUFFER));
            }
        }
        r
    }

    /// New targets beside the transient ones of `arena`, which [`Self::requests`] laid out,
    /// with the visibility of `slots` instance slots; discarded into `list` so their first use
    /// sees `GENERAL`.
    pub(crate) fn new(
        device: &Device,
        s: TargetSizes,
        slots: u32,
        arena: &ViewArena,
        list: &mut CommandList<'_>,
    ) -> Result<Self> {
        let [w, h] = s.extent;
        let dl = MemoryKind::DeviceLocal;
        let depth = Target::new(
            device,
            &TextureDesc::d2(
                Format::D32Float,
                w,
                h,
                TextureUsage::DEPTH | TextureUsage::SAMPLED,
            ),
            "depth",
        )?;
        let hzb = Target::new(
            device,
            &TextureDesc {
                mips: hzb_mips(s.extent),
                ..TextureDesc::d2(
                    Format::R32Float,
                    w,
                    h,
                    TextureUsage::SAMPLED | TextureUsage::STORAGE,
                )
            },
            "hzb",
        )?;
        let t = Self {
            sizes: s,
            slot_visibility: slot_visibility(device, slots)?,
            slots,
            replaced: Vec::new(),
            args: device.gpu_malloc(dl, ARGS + COUNTS, "raster args")?,
            depth,
            motion: Target::d2(device, Format::Rg16Float, s.extent, "motion")?,
            hzb,
            transient: Transient::take(arena),
            hzb_view: None,
        };
        for target in [&t.depth, &t.motion, &t.hzb] {
            list.discard(&target.texture);
        }
        Ok(t)
    }

    /// Gives the slot visibility room for the `slots` instance slots of the scene `frame`
    /// renders next.
    pub(crate) fn follow_slots(&mut self, device: &Device, slots: u32, frame: u64) -> Result<()> {
        // The lists of the frames before are submitted, so their memory outlives them.
        self.replaced.retain(|&(replaced, _)| replaced == frame);
        if slots > self.slots {
            let old = std::mem::replace(&mut self.slot_visibility, slot_visibility(device, slots)?);
            self.replaced.push((frame, old));
            self.slots = slots;
        }
        Ok(())
    }

    /// Takes the transient targets again after `arena` was laid out anew.
    pub(crate) fn bind(&mut self, arena: &ViewArena) {
        if self.transient.generation != arena.generation() {
            self.transient = Transient::take(arena);
        }
    }

    pub(crate) fn gbuffer(&self, lists: &ViewLists) -> GBufferGpu {
        let t = &self.transient;
        GBufferGpu {
            visibility: t.visibility.gpu(),
            clusters: lists.visible.clusters.gpu(),
            depth: self.depth.sampled,
            a: t.a.sampled(),
            b: t.b.sampled(),
            x: t.x
                .as_ref()
                .map_or(TextureIndex::NONE, ArenaTexture::sampled),
            y: t.y
                .as_ref()
                .map_or(TextureIndex::NONE, ArenaTexture::sampled),
            motion: self.motion.sampled,
            hzb: self.hzb.sampled,
            extent: self.sizes.extent,
            hzb_mips: self.hzb.storage.len() as u32,
            layer_tiles: t.layer_tiles.gpu(),
        }
    }

    /// The G-buffer's textures: a, b, x and y.
    pub(crate) fn gbuffer_textures(&self) -> GBufferTextures {
        let t = &self.transient;
        let extended = |x: &Option<ArenaTexture>| x.as_ref().map(|x| Arc::clone(x.texture()));
        (
            Arc::clone(t.a.texture()),
            Arc::clone(t.b.texture()),
            extended(&t.x),
            extended(&t.y),
        )
    }

    fn record(
        &self,
        lists: &ViewLists,
        software_pixels: f32,
        occlusion: bool,
        slots: u32,
        bin_roots: GpuPtr<RasterRoot>,
    ) -> RasterGpu {
        let [tx, ty] = tiles(self.sizes.extent);
        let mut hzb = [StorageImageIndex::NONE; MAX_HZB_MIPS as usize];
        hzb[..self.hzb.storage.len()].copy_from_slice(&self.hzb.storage);
        let t = &self.transient;
        RasterGpu {
            gbuffer: self.gbuffer(lists),
            work: lists.work.gpu(),
            args: self.args.gpu(),
            draw_args: lists.visible.draw_args.gpu().cast(),
            opaque: lists.visible.opaque.gpu(),
            alpha: lists.visible.alpha.gpu(),
            software: lists.visible.software.gpu(),
            particles: lists.visible.particles.gpu(),
            coverage: lists.visible.coverage.gpu(),
            smoke: lists.visible.smoke.gpu(),
            counts: self.counts(),
            slot_visibility: self.slot_visibility.gpu(),
            bin_args: t.bin_args.gpu(),
            bin_tiles: t.bin_tiles.gpu(),
            sequences: t.bin_sequences.gpu::<u8>().address().0,
            bin_roots: bin_roots.address().0,
            depth: t.depth_values.gpu(),
            a: t.a.storage(0),
            b: t.b.storage(0),
            x: t.x
                .as_ref()
                .map_or(StorageImageIndex::NONE, |x| x.storage(0)),
            y: t.y
                .as_ref()
                .map_or(StorageImageIndex::NONE, |y| y.storage(0)),
            motion: self.motion.storage[0],
            hzb,
            tiles_x: tx,
            tile_capacity: tx * ty,
            work_capacity: lists.work_capacity,
            visible_capacity: lists.visible.capacity,
            bins: self.sizes.bins,
            software_pixels,
            flags: if occlusion { RASTER_OCCLUSION } else { 0 },
            slots,
            hzb_mips: self.hzb.storage.len() as u32,
            draws: lists.visible.draws,
            draw_groups: lists.visible.draw_groups,
            draw_first: 0,
            _pad: 0,
        }
    }

    /// A raster record holding only the extent and the HZB, for kernels outside the
    /// view's raster that test what its last render saw.
    pub(crate) fn hzb_record(&self) -> RasterGpu {
        let mut hzb = [StorageImageIndex::NONE; MAX_HZB_MIPS as usize];
        hzb[..self.hzb.storage.len()].copy_from_slice(&self.hzb.storage);
        let mut r = RasterGpu::zeroed();
        r.gbuffer.extent = self.sizes.extent;
        r.hzb = hzb;
        r.hzb_mips = self.hzb.storage.len() as u32;
        r
    }

    /// The transparency point's overlay.
    pub(crate) fn overlay(&self) -> &ArenaTexture {
        &self.transient.overlay
    }

    /// The mesh draws over the particle list: each takes the record of `draws` at its
    /// index, those [`record`] gave.
    pub(crate) fn particle_draws(
        &self,
        lists: &ViewLists,
        draws: &[GpuPtr<RasterGpu>],
    ) -> Vec<ParticleDraws> {
        (0..lists.visible.draws)
            .zip(draws)
            .map(|(i, &raster)| ParticleDraws {
                raster,
                args: lists.draw_args(DRAW_PARTICLES, i),
                count: self.counts().offset(COUNT_DRAWS),
            })
            .collect()
    }

    /// Workgroups of one per visible smoke cluster.
    pub(crate) fn smoke_args(&self) -> GpuPtr<DispatchArgs> {
        self.args.gpu().offset(ARGS_SMOKE).cast()
    }

    /// `RasterGpu.counts` ([`crate::RasterView::counts`]).
    pub(crate) fn counts(&self) -> GpuPtr<u32> {
        self.args.gpu().offset(ARGS)
    }

    /// `bin`'s resolve dispatch args.
    fn bin_args(&self, bin: u32) -> GpuPtr<DispatchArgs> {
        self.transient
            .bin_args
            .gpu::<u32>()
            .offset(3 * u64::from(bin))
            .cast()
    }

    /// The resolve's device-generated sequences, room for one per bin, and their count.
    fn sequences(&self) -> (GpuPtr<Sequence>, GpuPtr<u32>) {
        let t = &self.transient;
        let count = t
            .bin_args
            .gpu::<u32>()
            .offset(3 * u64::from(self.sizes.bins));
        (t.bin_sequences.gpu(), count)
    }
}

/// Bin `bin`'s root among roots [`ROOT_ALIGN`] bytes apart from `roots`.
fn bin_root(roots: GpuPtr<RasterRoot>, bin: u32) -> GpuPtr<RasterRoot> {
    roots
        .cast::<u8>()
        .offset(ROOT_ALIGN * u64::from(bin))
        .cast()
}

/// Records one view's raster half into `list`. With `occlusion` the first pass tests the
/// previous frame's HZB, which the view's previous render left. `preprocess` is memory of
/// [`RasterPipelines::preprocess_layout`], when it gives one. `producers` are the entries
/// whose clusters the raster culls, each with the most it holds. `opaque` draws in each pass
/// after the lists' draws. Returns the raster record of each mesh draw over a list, the
/// first of which every other pass takes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    device: &Device,
    p: &RasterPipelines,
    pipelines: &Pipelines,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    targets: &ViewTargets,
    lists: &ViewLists,
    scene: GpuPtr<SceneRoot>,
    slots: u32,
    producers: &[(u32, u32)],
    view: GpuPtr<ViewConstants>,
    software_pixels: f32,
    occlusion: bool,
    preprocess: Option<GpuPtr<u8>>,
    mut opaque: Option<&mut OpaqueHook<'_>>,
) -> Result<Vec<GpuPtr<RasterGpu>>> {
    let s = targets.sizes;
    let (raster, record) = window.slice::<RasterGpu>(device, 1)?;
    let root_of = |value| RasterRoot {
        scene,
        view,
        raster,
        value,
        _pad: 0,
    };
    // The bins' roots; bin 0's is also the root of every pass with value 0.
    let roots = window.put_roots(device, s.bins, root_of)?;
    record[0] = targets.record(lists, software_pixels, occlusion, slots, roots);
    let first = record[0];
    let mut draws = vec![raster];
    for i in 1..lists.visible.draws {
        let (at, r) = window.slice::<RasterGpu>(device, 1)?;
        r[0] = RasterGpu {
            draw_first: i * lists.visible.draw_groups,
            ..first
        };
        draws.push(at);
    }
    // Per draw over a list, the roots of its opaque or coverage draw and of its alpha draw.
    let draw_roots = draws
        .iter()
        .map(|&raster| {
            let at = |value| RasterRoot {
                raster,
                ..root_of(value)
            };
            Ok([
                window.put_root(device, &at(0))?,
                window.put_root(device, &at(1))?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;
    let root = |value| window.put_root(device, &root_of(value));
    let r0 = roots;
    let r1 = root(1)?;
    let pixels = s.extent[0] * s.extent[1];
    let [tx, ty] = tiles(s.extent);
    let args = targets.args.gpu();
    let grid = |at: u64| args.offset(at).cast::<DispatchArgs>();
    let one = targets.counts().offset(COUNT_DRAWS);
    let rendering = Rendering {
        extent: s.extent,
        colors: &[],
        depth: None,
    };

    let raster_stages = Stages::TASK | Stages::MESH | Stages::PIXEL;
    let culled = |list: &mut CommandList<'_>, pass: GpuPtr<RasterRoot>| {
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(pipelines.get(p.cap), pass, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        list.dispatch_indirect(pipelines.get(p.clusters), pass, grid(ARGS_WORK));
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(pipelines.get(p.raster_args), pass, [1, 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            raster_stages | Stages::COMPUTE,
            Hazards::DRAW_ARGUMENTS,
        );
    };
    // The software dispatch and the hardware draws share no barrier, so they overlap; all
    // write the visibility buffer by atomic max.
    let mut draw = |list: &mut CommandList<'_>, second: bool| {
        list.dispatch_indirect(pipelines.get(p.software), r0, grid(ARGS_SOFTWARE));
        list.begin_rendering(&rendering);
        for (i, [opaque, alpha]) in (0..).zip(&draw_roots) {
            let (o, a) = (
                lists.draw_args(DRAW_OPAQUE, i),
                lists.draw_args(DRAW_ALPHA, i),
            );
            list.draw_mesh_tasks_indirect_count(&p.opaque, *opaque, o, one, 1);
            list.draw_mesh_tasks_indirect_count(&p.alpha, *alpha, a, one, 1);
        }
        if let Some(hook) = opaque.as_mut() {
            hook(&mut OpaquePass {
                cmd: list,
                extent: s.extent,
                view,
                scene,
                raster,
                second,
                device,
                window,
            });
        }
        list.end_rendering();
        list.barrier(
            raster_stages | Stages::COMPUTE,
            Stages::COMPUTE,
            Hazards::empty(),
        );
    };
    let slot_groups = [slots.div_ceil(64).max(1), 1, 1];

    list.begin_pass("cull", false);
    let cleared = pixels
        .max(3 * s.bins + 1)
        .max(COUNTS as u32)
        .max(tx * ty + 1);
    list.dispatch(pipelines.get(p.clear), r0, [cleared.div_ceil(64), 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    list.dispatch(pipelines.get(p.instances), r0, slot_groups);
    for &(entry, clusters) in producers {
        // A grid of whole rows: its threads past the entry's clusters find none.
        let groups = clusters.div_ceil(64);
        let grid = [groups.min(GRID_X), groups.div_ceil(GRID_X), 1];
        list.dispatch(pipelines.get(p.producers), root(entry)?, grid);
    }
    culled(list, r0);
    list.end_pass();

    list.begin_pass("raster", false);
    draw(list, false);
    list.end_pass();

    // The second pass: what the first found occluded, against an HZB of what it drew.
    if occlusion {
        list.begin_pass("occlusion", false);
        reduce_hzb(list, pipelines, p, targets, &root, Some(r0))?;
        list.dispatch(pipelines.get(p.instances), r1, slot_groups);
        culled(list, r1);
        draw(list, true);
        list.end_pass();
    }

    // Dielectric and presence clusters against the final visibility buffer.
    list.begin_pass("layer tiles", false);
    list.barrier(
        raster_stages | Stages::COMPUTE,
        raster_stages,
        Hazards::DRAW_ARGUMENTS,
    );
    list.begin_rendering(&rendering);
    for (i, [coverage, _]) in (0..).zip(&draw_roots) {
        let args = lists.draw_args(DRAW_COVERAGE, i);
        list.draw_mesh_tasks_indirect_count(&p.coverage, *coverage, args, one, 1);
    }
    list.end_rendering();
    list.end_pass();
    list.barrier(raster_stages, Stages::COMPUTE, Hazards::empty());

    list.begin_pass("resolve", false);
    list.dispatch(pipelines.get(p.classify), r0, [tx * ty, 1, 1]);
    match p.resolve {
        Resolve::Recorded(resolve) => {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
            for bin in 0..s.bins {
                let args = targets.bin_args(bin);
                list.dispatch_indirect(pipelines.get(resolve), bin_root(roots, bin), args);
            }
        }
        Resolve::Generated { sequences, set } => {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
            list.dispatch(pipelines.get(sequences), r0, [s.bins.div_ceil(64), 1, 1]);
            list.barrier(
                Stages::COMPUTE,
                Stages::PREPROCESS | Stages::COMPUTE,
                Hazards::DRAW_ARGUMENTS,
            );
            let (sequences, count) = targets.sequences();
            let g = Generated {
                set: pipelines.set(set),
                sequences,
                count,
                max: s.bins,
                preprocess: preprocess.expect("preprocess memory for the generated resolve"),
            };
            list.execute_generated(&g, false);
        }
    }
    list.end_pass();
    list.barrier(
        Stages::COMPUTE,
        Stages::COMPUTE | Stages::PIXEL,
        Hazards::empty(),
    );

    list.begin_pass("hzb", false);
    reduce_hzb(list, pipelines, p, targets, &root, None)?;
    list.end_pass();

    list.begin_pass("depth", false);
    list.begin_rendering(&Rendering {
        extent: s.extent,
        colors: &[],
        depth: Some(DepthAttachment {
            texture: &targets.depth.texture,
            load: DepthLoad::Discard,
            store: true,
        }),
    });
    list.draw_mesh_tasks(&p.depth, r0, [1, 1, 1]);
    list.end_rendering();
    list.end_pass();
    list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
    Ok(draws)
}

/// Builds the HZB's mips from mip 0, after `seed` writes mip 0 from the visibility buffer
/// when given.
fn reduce_hzb(
    list: &mut CommandList<'_>,
    pipelines: &Pipelines,
    p: &RasterPipelines,
    targets: &ViewTargets,
    root: &dyn Fn(u32) -> Result<GpuPtr<RasterRoot>>,
    seed: Option<GpuPtr<RasterRoot>>,
) -> Result<()> {
    let [w, h] = targets.sizes.extent;
    if let Some(r) = seed {
        list.dispatch(pipelines.get(p.seed), r, [w.div_ceil(8), h.div_ceil(8), 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    }
    for mip in 1..targets.hzb.storage.len() as u32 {
        let size = [(w >> mip).max(1), (h >> mip).max(1)];
        list.dispatch(
            pipelines.get(p.reduce),
            root(mip)?,
            [size[0].div_ceil(8), size[1].div_ceil(8), 1],
        );
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    }
    Ok(())
}
