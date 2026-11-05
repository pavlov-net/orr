//! The parts of orr_gpu the Metal backend does not implement yet, or that Metal has no
//! counterpart for: crash diagnostics, device-generated commands, cooperative vectors,
//! counter frames and ray-dispatch pipelines. The types mirror the Vulkan
//! backend's; the calls fail, panic, or take handles no call creates.

use std::alloc::Layout;
use std::convert::Infallible;
use std::path::Path;

use super::command::CommandList;
use super::device::Device;
use super::not_implemented;
use super::pipeline::ShaderStage;
use super::queue::Queue;
use crate::{
    CounterReport, DiagnosticsReport, Error, Generated, GpuAddress, MatrixConversion, MatrixFormat,
    MeshPipelineDesc, TimelinePoint,
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

impl Device {
    /// An empty report: Metal records no diagnostics yet.
    pub fn diagnostics_report(&self) -> DiagnosticsReport {
        DiagnosticsReport::default()
    }
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
