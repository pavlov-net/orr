//! Local exposure's bilateral grid: the meter fills it, `local_exposure` blurs it and tonemap
//! slices it.

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, GpuPtr, Hazards, MemoryKind, ShaderStage,
    SpecData, Stages,
};

use crate::Result;
use crate::shaders::exposure::local_grid::{self, Root};

/// Pixels across a grid cell (`LOCAL_CELL` in `exposure.slang`).
const LOCAL_CELL: u32 = 32;
/// Bins per cell (`LOCAL_BINS` in `exposure.slang`).
pub(crate) const LOCAL_BINS: u32 = 16;
/// Threads per group of the grid's kernels (`GROUP` in `local_grid.slang`).
const GROUP: u32 = 64;

pub(crate) struct LocalKernels {
    clear: ComputePipeline,
    blur_bins: ComputePipeline,
    blur_rows: ComputePipeline,
    blur_columns: ComputePipeline,
}

impl LocalKernels {
    pub(crate) fn new(gpu: &Device) -> Result<Self> {
        let pipeline = |entry, name| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(&local_grid::MODULE, entry, SpecData::default())?;
            Ok(gpu.create_compute_pipeline(&stage, name)?)
        };
        Ok(Self {
            clear: pipeline("clear", "local_exposure_clear")?,
            blur_bins: pipeline("blur_bins", "local_exposure_bins")?,
            blur_rows: pipeline("blur_rows", "local_exposure_rows")?,
            blur_columns: pipeline("blur_columns", "local_exposure_columns")?,
        })
    }

    /// Records the `clear` pass, which zeroes a new grid's sums and pivot, then a barrier.
    pub(crate) fn clear(&self, list: &mut CommandList<'_>, root: GpuPtr<Root>, cells: [u32; 2]) {
        list.begin_pass("local_exposure_clear", false);
        list.dispatch(&self.clear, root, [(2 * bins(cells)).div_ceil(GROUP), 1, 1]);
        list.end_pass();
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    }

    /// Records the `local_exposure` pass: the blur over the bins, the cells' rows and their
    /// columns, which also clears the meter's sums. Order the meter's writes before it.
    pub(crate) fn blur(&self, list: &mut CommandList<'_>, root: GpuPtr<Root>, cells: [u32; 2]) {
        let groups = [bins(cells).div_ceil(GROUP), 1, 1];
        list.begin_pass("local_exposure", false);
        list.dispatch(&self.blur_bins, root, groups);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.blur_rows, root, groups);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.blur_columns, root, groups);
        list.end_pass();
    }
}

fn bins(cells: [u32; 2]) -> u32 {
    cells[0] * cells[1] * LOCAL_BINS
}

/// A view's grid, allocated while it has local exposure.
pub(crate) struct LocalTargets {
    /// The meter's sums: a count and an EV sum per bin.
    pub(crate) sums: Allocation<u32>,
    /// The blurred grid tonemap slices.
    pub(crate) grid: Allocation<[f32; 2]>,
    scratch: Allocation<[f32; 2]>,
    /// The EV100 the base is pulled toward, which adapt writes.
    pub(crate) pivot: Allocation<[f32; 2]>,
    /// Cells across and down.
    pub(crate) cells: [u32; 2],
    /// The output extent the grid covers.
    pub(crate) extent: [u32; 2],
}

impl LocalTargets {
    pub(crate) fn new(gpu: &Device, extent: [u32; 2]) -> Result<Self> {
        let cells = extent.map(|n| n.div_ceil(LOCAL_CELL));
        let n = u64::from(bins(cells));
        Ok(Self {
            sums: gpu.gpu_malloc(MemoryKind::DeviceLocal, 2 * n, "local exposure sums")?,
            grid: gpu.gpu_malloc(MemoryKind::DeviceLocal, n, "local exposure grid")?,
            scratch: gpu.gpu_malloc(MemoryKind::DeviceLocal, n, "local exposure scratch")?,
            pivot: gpu.gpu_malloc(MemoryKind::DeviceLocal, 1, "local exposure pivot")?,
            cells,
            extent,
        })
    }

    /// The root of the grid's kernels.
    pub(crate) fn root(&self) -> Root {
        Root {
            sums: self.sums.gpu(),
            grid: self.grid.gpu(),
            scratch: self.scratch.gpu(),
            pivot: self.pivot.gpu(),
            width: self.cells[0],
            height: self.cells[1],
            _pad0: 0,
            _pad1: 0,
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.sums.size() + self.grid.size() + self.scratch.size() + self.pivot.size()
    }
}
