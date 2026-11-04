//! The parts of orr_gpu the Metal backend does not implement yet, or that Metal has no
//! counterpart for: crash diagnostics, device-generated commands, cooperative vectors,
//! counter frames, ray-dispatch pipelines and swapchains. The types mirror the Vulkan
//! backend's; the calls fail, panic, or take handles no call creates.

use std::alloc::Layout;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::command::CommandList;
use super::device::Device;
use super::mesh::MeshPipelineDesc;
use super::not_implemented;
use super::pipeline::ShaderStage;
use super::queue::{Queue, QueueKind};
use super::texture::Texture;
use crate::{
    ComponentType, Error, GpuAddress, GpuPtr, HdrMetadata, LatencyMarker, LatencyTimings,
    OutputFormat, PresentMode, Refresh, StorageImageIndex, SwapchainDesc, TimelinePoint,
};

/// What a device records for crash diagnosis. Metal records nothing yet, whatever the mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Diagnostics<'a> {
    #[default]
    Off,
    Checkpoints,
    Aftermath {
        dump_dir: &'a Path,
    },
    Capture,
}

/// What a device was doing, and on a lost device why it was lost.
#[derive(Clone, Debug, Default)]
pub struct DiagnosticsReport {
    pub lost: bool,
    pub queues: Vec<QueueProgress>,
    pub faults: Vec<DeviceFault>,
    pub abort: Option<String>,
    pub crash_dump: Option<PathBuf>,
    pub memory: Option<String>,
}

impl std::fmt::Display for DiagnosticsReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.lost {
            writeln!(f, "device lost")?;
        }
        if let Some(abort) = &self.abort {
            writeln!(f, "abort: {abort}")?;
        }
        f.write_str("Metal records no further diagnostics yet")
    }
}

/// One queue's progress through its recent passes.
#[derive(Clone, Debug)]
pub struct QueueProgress {
    pub queue: QueueKind,
    pub submitted: u64,
    pub completed: Option<u64>,
    pub last_completed: Option<&'static str>,
    pub in_flight: Vec<&'static str>,
    pub checkpoints: Vec<Checkpoint>,
}

/// The last pass marker one pipeline stage passed.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub stage: String,
    pub pass: Option<&'static str>,
    pub end: bool,
}

/// One device fault report.
#[derive(Clone, Debug)]
pub struct DeviceFault {
    pub description: String,
    pub watchdog: bool,
    pub addresses: Vec<FaultAddress>,
    pub vendor: Vec<String>,
}

/// An address a fault reported.
#[derive(Clone, Debug)]
pub struct FaultAddress {
    pub access: FaultAccess,
    pub address: u64,
    pub precision: u64,
    pub allocation: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultAccess {
    Read,
    Write,
    Execute,
    Instruction,
    Other,
}

impl Device {
    /// An empty report: Metal records no diagnostics yet.
    pub fn diagnostics_report(&self) -> DiagnosticsReport {
        DiagnosticsReport::default()
    }
}

/// One sequence of [`CommandList::execute_generated`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Sequence {
    pub pipeline: u32,
    /// The root's address, low word first.
    pub root: [u32; 2],
    pub groups: [u32; 3],
}

impl Sequence {
    pub fn new<R>(pipeline: u32, root: GpuPtr<R>, groups: [u32; 3]) -> Self {
        let a = root.address().0;
        Self {
            pipeline,
            root: [a as u32, (a >> 32) as u32],
            groups,
        }
    }
}

/// The kind of pipelines an execution set holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SetKind {
    Compute,
    Mesh,
}

/// Pipelines that GPU-written sequences pick by index; not on Metal yet.
pub struct ExecutionSet {
    never: Infallible,
}

impl ExecutionSet {
    pub fn len(&self) -> u32 {
        match self.never {}
    }

    pub fn is_empty(&self) -> bool {
        match self.never {}
    }

    pub fn preprocess_layout(&self, max: u32) -> Layout {
        let _ = max;
        match self.never {}
    }
}

/// One run of an execution set.
#[derive(Clone, Copy)]
pub struct Generated<'a> {
    pub set: &'a ExecutionSet,
    pub sequences: GpuPtr<Sequence>,
    pub count: GpuPtr<u32>,
    pub max: u32,
    pub preprocess: GpuPtr<u8>,
}

impl Device {
    pub fn create_compute_set(
        &self,
        stages: &[ShaderStage<'_>],
        name: &str,
    ) -> Result<ExecutionSet, Error> {
        let _ = (stages, name);
        Err(Error::NotImplemented("Device::create_compute_set"))
    }

    pub fn create_mesh_set(
        &self,
        descs: &[MeshPipelineDesc<'_>],
        name: &str,
    ) -> Result<ExecutionSet, Error> {
        let _ = (descs, name);
        Err(Error::NotImplemented("Device::create_mesh_set"))
    }
}

impl CommandList<'_> {
    pub fn preprocess_generated(&mut self, g: &Generated<'_>) {
        match g.set.never {}
    }

    pub fn execute_generated(&mut self, g: &Generated<'_>, preprocessed: bool) {
        let _ = preprocessed;
        match g.set.never {}
    }
}

/// Matrices start at multiples of this many bytes.
pub const MATRIX_ALIGN: u64 = 64;

/// How a cooperative-vector matrix lays out its elements.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
    InferencingOptimal,
    TrainingOptimal,
}

/// The element type and layout of a matrix.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MatrixFormat {
    pub component: ComponentType,
    pub layout: MatrixLayout,
    pub stride: u32,
}

impl MatrixFormat {
    /// Tightly packed in `layout`.
    pub fn new(component: ComponentType, layout: MatrixLayout) -> Self {
        Self {
            component,
            layout,
            stride: 0,
        }
    }
}

/// A `rows` x `columns` matrix converted from `src_format` to `dst_format`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MatrixConversion {
    pub rows: u32,
    pub columns: u32,
    pub src_format: MatrixFormat,
    pub dst_format: MatrixFormat,
}

impl Device {
    pub fn matrix_bytes(
        &self,
        rows: u32,
        columns: u32,
        format: MatrixFormat,
    ) -> Result<u64, Error> {
        let _ = (rows, columns, format);
        Err(Error::NotImplemented("cooperative vectors"))
    }

    pub fn convert_matrix(
        &self,
        c: &MatrixConversion,
        src: &[u8],
        dst: &mut [u8],
    ) -> Result<u64, Error> {
        let _ = (c, src, dst);
        Err(Error::NotImplemented("cooperative vectors"))
    }
}

impl CommandList<'_> {
    pub fn convert_matrix(&mut self, c: &MatrixConversion, src: GpuAddress, dst: GpuAddress) {
        let _ = (c, src, dst);
        not_implemented("cooperative vectors")
    }
}

/// The metrics a counter frame collects for every pass.
pub const COUNTER_METRICS: [&str; 6] = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
];

/// The counters of one counter frame.
#[derive(Clone, Debug, Default)]
pub struct CounterReport {
    pub chip: String,
    pub ranges: Vec<CounterRange>,
}

/// One pass's counters.
#[derive(Clone, Debug)]
pub struct CounterRange {
    pub name: String,
    pub values: Vec<f64>,
}

impl Queue {
    pub fn begin_counter_frame(&self) -> Result<(), Error> {
        Err(Error::NotImplemented("counter frames"))
    }

    pub fn end_counter_frame(&self, last: TimelinePoint) -> Result<Option<CounterReport>, Error> {
        let _ = last;
        Err(Error::NotImplemented("counter frames"))
    }
}

/// A ray-dispatch pipeline; Metal has none.
pub struct RayDispatchPipeline {
    never: Infallible,
}

impl RayDispatchPipeline {
    pub(super) fn unreachable(&self) -> ! {
        match self.never {}
    }
}

impl Device {
    pub fn create_ray_dispatch_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<RayDispatchPipeline, Error> {
        let _ = (s, name);
        Err(Error::NotImplemented("ray-dispatch pipelines"))
    }
}

/// Low latency for one swapchain; Metal has no swapchains yet.
#[derive(Clone)]
pub struct LowLatency {
    never: Infallible,
}

impl LowLatency {
    pub fn set_mode(&self, low_latency: bool, boost: bool, min_interval_us: u32) {
        let _ = (low_latency, boost, min_interval_us);
        match self.never {}
    }

    pub fn sleep(&self) {
        match self.never {}
    }

    pub fn marker(&self, frame_id: u64, marker: LatencyMarker) {
        let _ = (frame_id, marker);
        match self.never {}
    }

    pub fn set_marker_hook(&self, hook: Option<fn(u64, LatencyMarker)>) {
        let _ = hook;
        match self.never {}
    }

    pub fn timings(&self, out: &mut Vec<LatencyTimings>) {
        let _ = out;
        match self.never {}
    }
}

/// An acquired swapchain image; Metal has no swapchains yet.
pub struct SwapchainFrame {
    never: Infallible,
}

impl SwapchainFrame {
    pub fn texture(&self) -> &Texture {
        match self.never {}
    }

    pub fn storage(&self) -> StorageImageIndex {
        match self.never {}
    }

    pub fn format(&self) -> OutputFormat {
        match self.never {}
    }

    pub fn extent(&self) -> [u32; 2] {
        match self.never {}
    }
}

/// A window's swapchain; [`Swapchain::new`] fails on Metal until presenting is implemented.
pub struct Swapchain {
    never: Infallible,
}

impl Swapchain {
    pub fn new(
        device: &Device,
        window: RawWindowHandle,
        _display: RawDisplayHandle,
        desc: &SwapchainDesc,
    ) -> Result<Swapchain, Error> {
        let _ = (device, window, desc);
        Err(Error::NotImplemented("swapchains"))
    }

    pub fn supported_formats(&self) -> &[OutputFormat] {
        match self.never {}
    }

    pub fn set_format(&mut self, f: OutputFormat, hdr: Option<HdrMetadata>) -> Result<(), Error> {
        let _ = (f, hdr);
        match self.never {}
    }

    pub fn set_min_images(&mut self, n: u32) {
        let _ = n;
        match self.never {}
    }

    pub fn set_present_mode(&mut self, m: PresentMode) {
        let _ = m;
        match self.never {}
    }

    pub fn latency(&self) -> Option<LowLatency> {
        match self.never {}
    }

    pub fn refresh(&self) -> Refresh {
        match self.never {}
    }

    pub fn wait_presented(&self, id: u64, timeout: Duration) -> Result<bool, Error> {
        let _ = (id, timeout);
        match self.never {}
    }

    pub fn acquire(&mut self) -> Result<Option<SwapchainFrame>, Error> {
        match self.never {}
    }

    pub fn present(&mut self, frame: SwapchainFrame, batch: Option<u32>) -> Result<u64, Error> {
        let _ = (frame, batch);
        match self.never {}
    }

    pub fn discard(&mut self, frame: SwapchainFrame) -> Result<(), Error> {
        let _ = frame;
        match self.never {}
    }
}
