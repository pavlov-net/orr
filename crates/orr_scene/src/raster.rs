//! A view's raster half: instance and cluster culling over the scene's raster candidate
//! cut, the mesh-shader and software raster into the visibility buffer, both in two passes
//! around an HZB of the first, the dielectric and presence clusters into the layer tiles, the
//! material resolve per surface bin into the G-buffer and motion, the final HZB, and the
//! depth target.

use std::alloc::Layout;
use std::sync::Arc;

use bytemuck::Zeroable;
use orr_gpu::{
    Allocation, ColorTarget, CommandList, CompareOp, DepthAttachment, DepthLoad, DepthTarget,
    Device, DispatchArgs, Format, Generated, GpuPtr, Hazards, KernelSet, MemoryKind, MeshArgs,
    MeshPipeline, MeshPipelineDesc, ROOT_ALIGN, RasterState, Rendering, Sequence, ShaderStage,
    SpecData, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
    TextureUsage, ViewDesc,
};

use crate::arena::{ArenaBuffer, ArenaRequest, ArenaTexture, Span, ViewArena, ViewStep};
use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::shaders::{cull, hzb, raster, resolve};
use crate::{
    ClusterWork, GBufferGpu, RasterGpu, RasterRoot, Result, SceneId, SceneRoot, ViewConstants,
    VisibleCluster,
};

/// `RasterGpu.flags` (`visibility.slang`).
const RASTER_OCCLUSION: u32 = 1;

/// `RasterGpu.counts` entries and `RasterGpu.args` offsets (`visibility.slang`).
const COUNTS: u64 = 18;
const COUNT_DRAWS: u64 = 3;
const ARGS: u64 = 21;
const ARGS_WORK: u64 = 0;
const ARGS_OPAQUE: u64 = 3;
const ARGS_ALPHA: u64 = 6;
const ARGS_SOFTWARE: u64 = 9;
const ARGS_PARTICLES: u64 = 12;
const ARGS_COVERAGE: u64 = 15;
const ARGS_SMOKE: u64 = 18;
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

/// Sizes of a view's raster buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TargetSizes {
    pub(crate) extent: [u32; 2],
    pub(crate) extended: bool,
    pub(crate) visible_clusters: u32,
    pub(crate) work: u32,
    pub(crate) bins: u32,
    /// The scene's instance slots.
    pub(crate) slots: u32,
}

/// Arena names of the raster's transient targets.
const VISIBILITY: &str = "visibility buffer";
const CLUSTERS: &str = "visible clusters";
const WORK: &str = "cluster work";
const OPAQUE: &str = "opaque clusters";
const ALPHA: &str = "alpha clusters";
const SOFTWARE: &str = "software clusters";
const PARTICLES: &str = "particle clusters";
const COVERAGE: &str = "coverage clusters";
const SMOKE: &str = "smoke clusters";
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

/// Raster's working lists and the depth it resolves, which only the raster half reads.
const CULL: Span = Span::new(ViewStep::Raster, ViewStep::Raster);
/// The G-buffer and what `load_gsurface` and the transparency point read beside it.
const GBUFFER: Span = Span::new(ViewStep::Raster, ViewStep::Transparency);
/// Ray reconstruction may composite the overlay itself.
const OVERLAY_SPAN: Span = Span::new(ViewStep::Transparency, ViewStep::Rr);

/// The raster's targets in the view's arena, taken at one layout.
struct Transient {
    generation: u64,
    visibility: ArenaBuffer,
    clusters: ArenaBuffer,
    work: ArenaBuffer,
    opaque: ArenaBuffer,
    alpha: ArenaBuffer,
    software: ArenaBuffer,
    particles: ArenaBuffer,
    smoke: ArenaBuffer,
    coverage: ArenaBuffer,
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
            clusters: buffer(CLUSTERS),
            work: buffer(WORK),
            opaque: buffer(OPAQUE),
            alpha: buffer(ALPHA),
            software: buffer(SOFTWARE),
            particles: buffer(PARTICLES),
            smoke: buffer(SMOKE),
            coverage: buffer(COVERAGE),
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
    slot_visibility: Allocation<u32>,
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

impl ViewTargets {
    /// The transient targets the raster half takes from the view's arena.
    pub(crate) fn requests(s: &TargetSizes) -> Vec<ArenaRequest> {
        let [w, h] = s.extent;
        let pixels = u64::from(w) * u64::from(h);
        let [tx, ty] = tiles(s.extent);
        let clusters = u64::from(s.visible_clusters);
        let list = 4 * clusters;
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        let d2 = |format| TextureDesc::d2(format, w, h, usage);
        let mut r = vec![
            ArenaRequest::buffer(VISIBILITY, 8 * pixels, GBUFFER),
            ArenaRequest::buffer(
                CLUSTERS,
                size_of::<VisibleCluster>() as u64 * clusters,
                GBUFFER,
            ),
            ArenaRequest::buffer(PARTICLES, list, GBUFFER),
            ArenaRequest::buffer(SMOKE, list, GBUFFER),
            // Then the word marking every tile.
            ArenaRequest::buffer(
                LAYER_TILES,
                4 * (u64::from(tx) * u64::from(ty) + 1),
                GBUFFER,
            ),
            ArenaRequest::texture(GBUF_A, d2(Format::Rgba32Uint), GBUFFER),
            ArenaRequest::texture(GBUF_B, d2(Format::Rg32Uint), GBUFFER),
            ArenaRequest::buffer(
                WORK,
                size_of::<ClusterWork>() as u64 * u64::from(s.work),
                CULL,
            ),
            ArenaRequest::buffer(OPAQUE, list, CULL),
            ArenaRequest::buffer(ALPHA, list, CULL),
            ArenaRequest::buffer(SOFTWARE, list, CULL),
            ArenaRequest::buffer(COVERAGE, list, CULL),
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

    /// New targets beside the transient ones of `arena`, which [`Self::requests`] laid out;
    /// discarded into `list` so their first use sees `GENERAL`.
    pub(crate) fn new(
        device: &Device,
        s: TargetSizes,
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
            slot_visibility: device.gpu_malloc(dl, u64::from(s.slots.max(1)), "slot visibility")?,
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

    /// Takes the transient targets again after `arena` was laid out anew.
    pub(crate) fn bind(&mut self, arena: &ViewArena) {
        if self.transient.generation != arena.generation() {
            self.transient = Transient::take(arena);
        }
    }

    pub(crate) fn gbuffer(&self) -> GBufferGpu {
        let t = &self.transient;
        GBufferGpu {
            visibility: t.visibility.gpu(),
            clusters: t.clusters.gpu(),
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
            gbuffer: self.gbuffer(),
            work: t.work.gpu(),
            args: self.args.gpu(),
            opaque: t.opaque.gpu(),
            alpha: t.alpha.gpu(),
            software: t.software.gpu(),
            particles: t.particles.gpu(),
            coverage: t.coverage.gpu(),
            smoke: t.smoke.gpu(),
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
            work_capacity: self.sizes.work,
            visible_capacity: self.sizes.visible_clusters,
            bins: self.sizes.bins,
            software_pixels,
            flags: if occlusion { RASTER_OCCLUSION } else { 0 },
            slots,
            hzb_mips: self.hzb.storage.len() as u32,
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

    /// Mesh args of the particle draws, and the draw count beside them.
    pub(crate) fn particle_draws(&self) -> (GpuPtr<MeshArgs>, GpuPtr<u32>) {
        (
            self.args.gpu().offset(ARGS_PARTICLES).cast(),
            self.counts().offset(COUNT_DRAWS),
        )
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
/// [`RasterPipelines::preprocess_layout`], when it gives one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    device: &Device,
    p: &RasterPipelines,
    pipelines: &Pipelines,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    targets: &ViewTargets,
    scene: GpuPtr<SceneRoot>,
    slots: u32,
    producer_capacity: u32,
    view: GpuPtr<ViewConstants>,
    software_pixels: f32,
    occlusion: bool,
    preprocess: Option<GpuPtr<u8>>,
) -> Result<GpuPtr<RasterGpu>> {
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
    record[0] = targets.record(software_pixels, occlusion, slots, roots);
    let root = |value| window.put_root(device, &root_of(value));
    let r0 = roots;
    let r1 = root(1)?;
    let pixels = s.extent[0] * s.extent[1];
    let [tx, ty] = tiles(s.extent);
    let args = targets.args.gpu();
    let grid = |at: u64| args.offset(at).cast::<DispatchArgs>();
    let mesh_args = |at: u64| args.offset(at).cast::<MeshArgs>();
    let draws = targets.counts().offset(COUNT_DRAWS);
    let rendering = Rendering {
        extent: s.extent,
        colors: &[],
        depth: None,
    };

    let raster_stages = Stages::MESH | Stages::PIXEL;
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
    // The software dispatch and the hardware draws share no barrier, so they overlap; both
    // write the visibility buffer by atomic max.
    let draw = |list: &mut CommandList<'_>| {
        list.dispatch_indirect(pipelines.get(p.software), r0, grid(ARGS_SOFTWARE));
        list.begin_rendering(&rendering);
        list.draw_mesh_tasks_indirect_count(&p.opaque, r0, mesh_args(ARGS_OPAQUE), draws, 1);
        list.draw_mesh_tasks_indirect_count(&p.alpha, r1, mesh_args(ARGS_ALPHA), draws, 1);
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
    if producer_capacity > 0 {
        list.dispatch(pipelines.get(p.producers), r0, [producer_capacity, 1, 1]);
    }
    culled(list, r0);
    list.end_pass();

    list.begin_pass("raster", false);
    draw(list);
    list.end_pass();

    // The second pass: what the first found occluded, against an HZB of what it drew.
    if occlusion {
        list.begin_pass("occlusion", false);
        reduce_hzb(list, pipelines, p, targets, &root, Some(r0))?;
        list.dispatch(pipelines.get(p.instances), r1, slot_groups);
        culled(list, r1);
        draw(list);
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
    list.draw_mesh_tasks_indirect_count(&p.coverage, r0, mesh_args(ARGS_COVERAGE), draws, 1);
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
    Ok(raster)
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
