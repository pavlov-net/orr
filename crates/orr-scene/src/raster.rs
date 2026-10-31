//! A view's raster half: instance and cluster culling over the scene's raster candidate
//! cut, the mesh-shader and software raster into the visibility buffer, both in two passes
//! around an HZB of the first, the material resolve per surface bin into the G-buffer and
//! motion, the final HZB, and the depth target.

use std::sync::Arc;

use orr_gpu::{
    Allocation, ColorTarget, CommandList, CompareOp, DepthAttachment, DepthLoad, DepthTarget,
    Device, DispatchArgs, Format, GpuPtr, Hazards, KernelSet, MemoryKind, MeshArgs, MeshPipeline,
    MeshPipelineDesc, RasterState, Rendering, ShaderStage, SpecData, Stages, StorageImageIndex,
    Texture, TextureDesc, TextureIndex, TextureMemory, TextureUsage, ViewDesc,
};

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
const COUNTS: u64 = 16;
const COUNT_DRAWS: u64 = 3;
const ARGS: u64 = 15;
const ARGS_WORK: u64 = 0;
const ARGS_OPAQUE: u64 = 3;
const ARGS_ALPHA: u64 = 6;
const ARGS_SOFTWARE: u64 = 9;
const ARGS_PARTICLES: u64 = 12;
const MAX_HZB_MIPS: u32 = 16;
const TILE: u32 = 8;

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
    software: PipelineId,
    classify: PipelineId,
    resolve: PipelineId,
    seed: PipelineId,
    reduce: PipelineId,
    depth: MeshPipeline,
    /// Surface bins: linked surfaces plus `StandardSurface`.
    pub bins: u32,
}

impl RasterPipelines {
    pub(crate) fn new(
        device: &Device,
        kernels: &KernelSet,
        pipelines: &mut Pipelines,
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
            software: compute(&raster::MODULE, "software")?,
            classify: compute(&resolve::MODULE, "classify")?,
            resolve: compute(&resolve::MODULE, "resolve")?,
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
}

/// A per-view texture with a sampled view and, unless it is the depth target, storage
/// views of each mip.
pub(crate) struct Target {
    pub texture: Arc<Texture>,
    pub sampled: TextureIndex,
    pub storage: Vec<StorageImageIndex>,
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

/// Sizes of a view's raster buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TargetSizes {
    pub extent: [u32; 2],
    pub extended: bool,
    pub visible_clusters: u32,
    pub work: u32,
    pub bins: u32,
    /// The scene's instance slots.
    pub slots: u32,
}

/// One view history's raster targets, kept while its extent and sizes hold.
pub(crate) struct ViewTargets {
    pub sizes: TargetSizes,
    pub visibility: Allocation<u64>,
    pub clusters: Allocation<VisibleCluster>,
    work: Allocation<ClusterWork>,
    opaque: Allocation<u32>,
    alpha: Allocation<u32>,
    software: Allocation<u32>,
    particles: Allocation<u32>,
    slot_visibility: Allocation<u32>,
    /// Workgroup args (`ARGS`), then counts.
    args: Allocation<u32>,
    bin_args: Allocation<u32>,
    bin_tiles: Allocation<u32>,
    depth_values: Allocation<f32>,
    pub depth: Target,
    pub a: Target,
    pub b: Target,
    pub x: Option<Target>,
    pub motion: Target,
    pub hzb: Target,
    /// The transparency point's overlay, made at the first transparency pass.
    pub overlay: Option<Target>,
    /// The scene the HZB holds the last render of.
    pub hzb_scene: Option<SceneId>,
}

fn tiles(extent: [u32; 2]) -> [u32; 2] {
    [extent[0].div_ceil(TILE), extent[1].div_ceil(TILE)]
}

fn hzb_mips(extent: [u32; 2]) -> u32 {
    (32 - extent[0].max(extent[1]).max(1).leading_zeros()).min(MAX_HZB_MIPS)
}

impl ViewTargets {
    /// New targets, discarded into `list` so their first use sees `GENERAL`.
    pub(crate) fn new(device: &Device, s: TargetSizes, list: &mut CommandList<'_>) -> Result<Self> {
        let [w, h] = s.extent;
        let pixels = u64::from(w) * u64::from(h);
        let [tx, ty] = tiles(s.extent);
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
            visibility: device.gpu_malloc(dl, pixels, "visibility buffer")?,
            clusters: device.gpu_malloc(dl, u64::from(s.visible_clusters), "visible clusters")?,
            work: device.gpu_malloc(dl, u64::from(s.work), "cluster work")?,
            opaque: device.gpu_malloc(dl, u64::from(s.visible_clusters), "opaque clusters")?,
            alpha: device.gpu_malloc(dl, u64::from(s.visible_clusters), "alpha clusters")?,
            software: device.gpu_malloc(dl, u64::from(s.visible_clusters), "software clusters")?,
            particles: device.gpu_malloc(dl, u64::from(s.visible_clusters), "particle clusters")?,
            slot_visibility: device.gpu_malloc(dl, u64::from(s.slots.max(1)), "slot visibility")?,
            args: device.gpu_malloc(dl, ARGS + COUNTS, "raster args")?,
            bin_args: device.gpu_malloc(dl, 3 * u64::from(s.bins), "bin args")?,
            bin_tiles: device.gpu_malloc(
                dl,
                u64::from(s.bins) * u64::from(tx) * u64::from(ty),
                "bin tiles",
            )?,
            depth_values: device.gpu_malloc(dl, pixels, "resolved depth")?,
            depth,
            a: Target::d2(device, Format::Rgba32Uint, s.extent, "gbuf_a")?,
            b: Target::d2(device, Format::Rg32Uint, s.extent, "gbuf_b")?,
            x: s.extended
                .then(|| Target::d2(device, Format::Rgba32Uint, s.extent, "gbuf_x"))
                .transpose()?,
            motion: Target::d2(device, Format::Rg16Float, s.extent, "motion")?,
            hzb,
            overlay: None,
            hzb_scene: None,
        };
        for target in [&t.depth, &t.a, &t.b, &t.motion, &t.hzb]
            .into_iter()
            .chain(&t.x)
        {
            list.discard(&target.texture);
        }
        Ok(t)
    }

    pub(crate) fn gbuffer(&self) -> GBufferGpu {
        GBufferGpu {
            visibility: self.visibility.gpu,
            clusters: self.clusters.gpu,
            depth: self.depth.sampled,
            a: self.a.sampled,
            b: self.b.sampled,
            x: self.x.as_ref().map_or(TextureIndex::NONE, |x| x.sampled),
            motion: self.motion.sampled,
            hzb: self.hzb.sampled,
            extent: self.sizes.extent,
            hzb_mips: self.hzb.storage.len() as u32,
            _pad: 0,
        }
    }

    fn record(&self, software_pixels: f32, occlusion: bool, slots: u32) -> RasterGpu {
        let [tx, ty] = tiles(self.sizes.extent);
        let mut hzb = [StorageImageIndex::NONE; MAX_HZB_MIPS as usize];
        hzb[..self.hzb.storage.len()].copy_from_slice(&self.hzb.storage);
        RasterGpu {
            gbuffer: self.gbuffer(),
            work: self.work.gpu,
            args: self.args.gpu,
            opaque: self.opaque.gpu,
            alpha: self.alpha.gpu,
            software: self.software.gpu,
            particles: self.particles.gpu,
            counts: self.counts(),
            slot_visibility: self.slot_visibility.gpu,
            bin_args: self.bin_args.gpu,
            bin_tiles: self.bin_tiles.gpu,
            depth: self.depth_values.gpu,
            a: self.a.storage[0],
            b: self.b.storage[0],
            x: self
                .x
                .as_ref()
                .map_or(StorageImageIndex::NONE, |x| x.storage[0]),
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
            _pad: 0,
        }
    }

    /// The overlay, made on first use.
    pub(crate) fn overlay(
        &mut self,
        device: &Device,
        list: &mut CommandList<'_>,
    ) -> Result<&Target> {
        if self.overlay.is_none() {
            let [w, h] = self.sizes.extent;
            let usage = TextureUsage::SAMPLED | TextureUsage::COLOR;
            let desc = TextureDesc::d2(Format::Rgba16Float, w, h, usage);
            let overlay = Target::new(device, &desc, "transparency overlay")?;
            list.discard(&overlay.texture);
            self.overlay = Some(overlay);
        }
        Ok(self.overlay.as_ref().expect("made above"))
    }

    /// Mesh args of the particle draws, and the draw count beside them.
    pub(crate) fn particle_draws(&self) -> (GpuPtr<MeshArgs>, GpuPtr<u32>) {
        (
            self.args.gpu.offset(ARGS_PARTICLES).cast(),
            self.counts().offset(COUNT_DRAWS),
        )
    }

    /// `RasterGpu.counts` ([`crate::RasterView::counts`]).
    pub(crate) fn counts(&self) -> GpuPtr<u32> {
        self.args.gpu.offset(ARGS)
    }
}

/// Records one view's raster half into `list`. With `occlusion` the first pass tests the
/// previous frame's HZB, which the view's previous render left.
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
) -> Result<GpuPtr<RasterGpu>> {
    let raster = window.put(device, &targets.record(software_pixels, occlusion, slots))?;
    let root = |value: u32| {
        window.put_root(
            device,
            &RasterRoot {
                scene,
                view,
                raster,
                value,
                _pad: 0,
            },
        )
    };
    let r0 = root(0)?;
    let r1 = root(1)?;
    let s = targets.sizes;
    let pixels = s.extent[0] * s.extent[1];
    let [tx, ty] = tiles(s.extent);
    let args = targets.args.gpu;
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
    let cleared = pixels.max(3 * s.bins).max(COUNTS as u32);
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

    list.begin_pass("resolve", false);
    list.dispatch(pipelines.get(p.classify), r0, [tx * ty, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
    for bin in 0..s.bins {
        let bin_args = targets.bin_args.gpu.offset(3 * u64::from(bin));
        list.dispatch_indirect(
            pipelines.get(p.resolve),
            root(bin)?,
            bin_args.cast::<DispatchArgs>(),
        );
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
