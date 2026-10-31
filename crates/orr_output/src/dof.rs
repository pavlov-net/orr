//! Depth of field, after RR at output resolution or before it at render resolution with the
//! RR guide.

use orr_scene::orr_gpu::{
    AddressMode, Allocation, Blend, ColorAttachment, ColorLoad, ColorMask, ColorTarget,
    CommandList, ComputePipeline, Device, Filter, Format, GpuPtr, Hazards, MemoryKind, MeshArgs,
    MeshPipeline, MeshPipelineDesc, Rendering, SamplerDesc, SamplerIndex, ShaderStage, SpecData,
    Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureUsage, ViewDesc,
};
use orr_scene::{DepthOfField, Focus, ViewConstants};

use crate::passes::groups;
use crate::shaders::post::dof::{self, LensState, Root, Sprite};
use crate::{Image, Result, ViewOutput};

/// Half-resolution pixels across a CoC tile (`TILE` in `dof.slang`).
const TILE: u32 = 8;
/// Entries of the diaphragm table (`DIAPHRAGM_ENTRIES`).
const DIAPHRAGM_ENTRIES: u64 = 256;
/// The gather source's mips.
const MIPS: u32 = 5;

/// Where a view's depth of field runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DofPath {
    /// On RR's output at output resolution.
    #[default]
    AfterRr,
    /// On lighting's radiance at render resolution before RR, which reads the blurred
    /// radiance as its color and the change as `pInDepthOfFieldGuide`.
    Guide,
}

/// Blurring one view's radiance by its lens's circle of confusion.
#[derive(Clone, Copy, Debug)]
pub struct DepthOfFieldPass<'a> {
    /// Lighting's `rr_depth`: R32F reverse-Z at render resolution.
    pub rr_depth: TextureIndex,
    /// Lighting's `rr_motion`: RG16F previous minus current UV at render resolution.
    pub rr_motion: TextureIndex,
    /// The extent the effect runs at: the output extent after RR, the render extent on the
    /// guide path.
    pub extent: [u32; 2],
    /// The render's view constants: projection, render extent, jitter, exposure slots and
    /// `delta_time`.
    pub view: GpuPtr<ViewConstants>,
    /// `Lens::sensor_width_mm`, which with the projection gives the focal length.
    pub sensor_width_mm: f32,
    /// `Lens::depth_of_field`; `None` records nothing and frees the view's targets.
    pub depth_of_field: Option<DepthOfField>,
    /// The view's history reset this frame: the stabilizer restarts and autofocus snaps.
    pub history_reset: bool,
    /// orr's frame id.
    pub frame: u64,
    pub output: &'a ViewOutput,
}

/// The guide path's images, which RR reads as `color` and `depth_of_field_guide`.
#[derive(Clone, Copy)]
pub struct DofGuideImages<'a> {
    /// Blurred radiance, RGBA16F at render resolution.
    pub color: &'a Texture,
    /// R16F `luminance(after - before)` with luminance `(r + 2g + b) / 4`.
    pub guide: &'a Texture,
}

pub(crate) struct DofKernels {
    lens: ComputePipeline,
    setup: ComputePipeline,
    stabilize: ComputePipeline,
    select: ComputePipeline,
    reduce: ComputePipeline,
    dilate: ComputePipeline,
    gather: ComputePipeline,
    median_background: ComputePipeline,
    median_foreground: ComputePipeline,
    composite: ComputePipeline,
    sprites: MeshPipeline,
    point: SamplerIndex,
}

impl DofKernels {
    pub(crate) fn new(gpu: &Device) -> Result<Self> {
        let stage = |entry| ShaderStage::new(&dof::MODULE, entry, SpecData::default());
        let pipeline = |entry, name| -> Result<ComputePipeline> {
            Ok(gpu.create_compute_pipeline(&stage(entry)?, name)?)
        };
        let target = ColorTarget {
            blend: Some(Blend::ADDITIVE),
            write: ColorMask::R | ColorMask::G | ColorMask::B,
            ..ColorTarget::new(Format::Rgba16Float)
        };
        let sprites = gpu.create_mesh_pipeline(
            &MeshPipelineDesc {
                pixel: Some(stage("sprite_pixel")?),
                colors: &[target, target],
                ..MeshPipelineDesc::new(stage("sprite_mesh")?)
            },
            "dof_sprites",
        )?;
        Ok(Self {
            lens: pipeline("lens", "dof_lens")?,
            setup: pipeline("setup", "dof_setup")?,
            stabilize: pipeline("stabilize", "dof_stabilize")?,
            select: pipeline("select", "dof_select")?,
            reduce: pipeline("reduce", "dof_reduce")?,
            dilate: pipeline("dilate", "dof_dilate")?,
            gather: pipeline("gather", "dof_gather")?,
            median_background: pipeline("median_background", "dof_median_background")?,
            median_foreground: pipeline("median_foreground", "dof_median_foreground")?,
            composite: pipeline("composite", "dof_composite")?,
            sprites,
            point: gpu.sampler(&SamplerDesc {
                mag: Filter::Nearest,
                min: Filter::Nearest,
                mip: Filter::Nearest,
                address: [AddressMode::ClampToEdge; 3],
                ..SamplerDesc::default()
            })?,
        })
    }

    /// The pass's root, reading `source` and writing `target` (and `guide` on the guide
    /// path). Flips the targets' history.
    pub(crate) fn root(
        &self,
        targets: &mut DofTargets,
        pass: &DepthOfFieldPass<'_>,
        dof: &DepthOfField,
        source: TextureIndex,
        target: StorageImageIndex,
    ) -> Root {
        let [width, height] = pass.extent;
        let [half_width, half_height] = targets.half;
        let [tiles_x, tiles_y] = tiles(targets.half);
        let reset = pass.history_reset || std::mem::take(&mut targets.fresh);
        let (read, write) = targets.flip();
        let (history, stable) = (&targets.histories[read], &targets.histories[write]);
        let guide = targets
            .guide
            .as_ref()
            .map_or(StorageImageIndex::NONE, |g| g.guide.storage);
        let max_radius = 0.25 * dof.max_coc * width as f32;
        let (auto_focus, focus_distance, [focus_u, focus_v], focus_speed) = match dof.focus {
            Focus::Manual { distance_m } => (0, distance_m, [0.5, 0.5], 0.0),
            Focus::Auto { uv, speed } => (1, 0.0, uv, speed),
        };
        Root {
            view: pass.view,
            lens: targets.lens.gpu,
            diaphragm: targets.diaphragm.gpu,
            sprites: targets.sprites.gpu,
            scatter: targets.scatter.gpu,
            rr_depth: pass.rr_depth,
            rr_motion: pass.rr_motion,
            source,
            target,
            guide,
            layer: targets.layer.sampled,
            layer_store: targets.layer.storage,
            history: history.sampled,
            history_store: history.storage,
            stable: stable.sampled,
            stable_store: stable.storage,
            chain: targets.chain.sampled,
            chain0: targets.chain.storage,
            chain1: targets.mips[0],
            chain2: targets.mips[1],
            chain3: targets.mips[2],
            chain4: targets.mips[3],
            tiles: targets.tiles.sampled,
            tiles_store: targets.tiles.storage,
            dilated: targets.dilated.sampled,
            dilated_store: targets.dilated.storage,
            point: self.point,
            width,
            height,
            half_width,
            half_height,
            tiles_x,
            tiles_y,
            mips: targets.mip_count,
            rings: pass.output.post.dof_rings.max(1),
            max_sprites: targets.max_sprites,
            reach: 1 + (max_radius / TILE as f32) as i32,
            sensor_width_mm: pass.sensor_width_mm,
            f_number: dof.f_number,
            focus_distance,
            auto_focus,
            focus_u,
            focus_v,
            focus_speed,
            reset: u32::from(reset),
            blades: dof.blades,
            round_at_f_number: dof.round_at_f_number,
            max_coc: dof.max_coc,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Records the pass, then a barrier.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        name: &'static str,
        targets: &DofTargets,
        root: GpuPtr<Root>,
        extent: [u32; 2],
    ) {
        let half = targets.half;
        let step = |list: &mut CommandList<'_>| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        };
        list.begin_pass(name, false);
        list.dispatch(&self.lens, root, [1, 1, 1]);
        step(list);
        list.dispatch(&self.setup, root, groups(half, 8));
        step(list);
        list.dispatch(&self.stabilize, root, groups(half, 8));
        step(list);
        list.dispatch(&self.select, root, groups(half, TILE));
        step(list);
        list.dispatch(&self.reduce, root, groups(half.map(|n| n.div_ceil(2)), 16));
        list.dispatch(&self.dilate, root, groups(tiles(half), 8));
        step(list);
        list.dispatch(&self.gather, root, groups(half, 8));
        step(list);
        list.dispatch(&self.median_background, root, groups(half, 8));
        step(list);
        list.dispatch(&self.median_foreground, root, groups(half, 8));
        list.barrier(
            Stages::COMPUTE,
            Stages::MESH | Stages::PIXEL | Stages::COLOR_OUT,
            Hazards::DRAW_ARGUMENTS,
        );
        let keep = |texture| ColorAttachment {
            texture,
            load: ColorLoad::Keep,
            store: true,
        };
        list.begin_rendering(&Rendering {
            extent: half,
            colors: &[keep(targets.chain.texture()), keep(targets.layer.texture())],
            depth: None,
        });
        let args = targets.scatter.gpu.offset(1).cast::<MeshArgs>();
        let count = targets.scatter.gpu.offset(4);
        list.draw_mesh_tasks_indirect_count(&self.sprites, root, args, count, 1);
        list.end_rendering();
        list.barrier(Stages::COLOR_OUT, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.composite, root, groups(extent, 8));
        list.end_pass();
        step(list);
    }
}

fn tiles(half: [u32; 2]) -> [u32; 2] {
    half.map(|n| n.div_ceil(TILE))
}

/// The guide path's render-resolution images.
pub(crate) struct GuideTargets {
    pub(crate) color: Image,
    pub(crate) guide: Image,
}

/// A view's depth-of-field images, allocated while its lens has depth of field.
pub(crate) struct DofTargets {
    layer: Image,
    histories: [Image; 2],
    /// The image the next frame reads as its history.
    current: usize,
    chain: Image,
    /// Storage views of the chain's mips 1-4; mips past the chain's last alias its last.
    mips: [StorageImageIndex; 4],
    mip_count: u32,
    tiles: Image,
    dilated: Image,
    lens: Allocation<LensState>,
    diaphragm: Allocation<f32>,
    sprites: Allocation<Sprite>,
    scatter: Allocation<u32>,
    max_sprites: u32,
    pub(crate) guide: Option<GuideTargets>,
    path: DofPath,
    extent: [u32; 2],
    half: [u32; 2],
    pub(crate) fresh: bool,
    device: Device,
}

impl DofTargets {
    pub(crate) fn new(gpu: &Device, extent: [u32; 2], path: DofPath) -> Result<Self> {
        let half = extent.map(|n| n.div_ceil(2));
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE | TextureUsage::COLOR;
        let half_image = |name| {
            let desc = TextureDesc::d2(Format::Rgba16Float, half[0], half[1], usage);
            Image::with_desc(gpu, &desc, name)
        };
        let mip_count = MIPS.min(32 - half[0].min(half[1]).leading_zeros());
        let chain = Image::with_desc(
            gpu,
            &TextureDesc {
                mips: mip_count,
                ..TextureDesc::d2(Format::Rgba16Float, half[0], half[1], usage)
            },
            "dof chain",
        )?;
        let mut mips = [StorageImageIndex::NONE; 4];
        for (i, m) in mips.iter_mut().enumerate() {
            let mip = (i as u32 + 1).min(mip_count - 1);
            let view = ViewDesc {
                base_mip: mip,
                mips: 1,
                ..ViewDesc::default()
            };
            *m = gpu.write_storage(gpu.alloc_image_slot()?, chain.texture(), &view);
        }
        let [tx, ty] = tiles(half);
        let max_sprites = (half[0] * half[1] / 16).max(64);
        let guide = match path {
            DofPath::AfterRr => None,
            DofPath::Guide => Some(GuideTargets {
                color: Image::new(gpu, Format::Rgba16Float, extent, "dof guide color")?,
                guide: Image::new(gpu, Format::R16Float, extent, "dof guide")?,
            }),
        };
        Ok(Self {
            layer: half_image("dof layer")?,
            histories: [half_image("dof history")?, half_image("dof history")?],
            current: 0,
            chain,
            mips,
            mip_count,
            tiles: Image::new(gpu, Format::R16Float, [tx, ty], "dof tiles")?,
            dilated: Image::new(gpu, Format::R16Float, [tx, ty], "dof dilated tiles")?,
            lens: gpu.gpu_malloc(MemoryKind::DeviceLocal, 1, "dof lens")?,
            diaphragm: gpu.gpu_malloc(MemoryKind::DeviceLocal, DIAPHRAGM_ENTRIES, "diaphragm")?,
            sprites: gpu.gpu_malloc(
                MemoryKind::DeviceLocal,
                u64::from(max_sprites),
                "dof sprites",
            )?,
            scatter: gpu.gpu_malloc(MemoryKind::DeviceLocal, 5, "dof scatter")?,
            max_sprites,
            guide,
            path,
            extent,
            half,
            fresh: true,
            device: gpu.clone(),
        })
    }

    pub(crate) fn fits(&self, extent: [u32; 2], path: DofPath) -> bool {
        self.extent == extent && self.path == path
    }

    /// The history this frame reads and the image it writes, which the next frame reads.
    fn flip(&mut self) -> (usize, usize) {
        let read = self.current;
        self.current ^= 1;
        (read, self.current)
    }

    fn images(&self) -> impl Iterator<Item = &Image> {
        let guide = self.guide.iter().flat_map(|g| [&g.color, &g.guide]);
        [
            &self.layer,
            &self.histories[0],
            &self.histories[1],
            &self.chain,
            &self.tiles,
            &self.dilated,
        ]
        .into_iter()
        .chain(guide)
    }

    pub(crate) fn discard(&self, list: &mut CommandList<'_>) {
        for image in self.images() {
            list.discard(image.texture());
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.images().map(Image::bytes).sum::<u64>()
            + self.lens.size
            + self.diaphragm.size
            + self.sprites.size
            + self.scatter.size
    }

    pub(crate) fn guide_images(&self) -> Option<DofGuideImages<'_>> {
        self.guide.as_ref().map(|g| DofGuideImages {
            color: g.color.texture(),
            guide: g.guide.texture(),
        })
    }
}

impl Drop for DofTargets {
    fn drop(&mut self) {
        for m in self.mips {
            self.device.free_image_slot_after_submitted(m.0);
        }
    }
}
