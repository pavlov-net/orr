//! The parts of orr_gpu the Metal backend does not implement yet, or that Metal has no
//! counterpart for: crash diagnostics, device-generated commands, cooperative vectors,
//! counter frames, ray tracing pipelines and host memory imports. The types mirror the
//! Vulkan backend's; the calls fail, panic, or take handles no call creates.

use std::alloc::Layout;
use std::path::Path;

use super::command::CommandList;
use super::device::Device;
use super::not_implemented;
use super::pipeline::ShaderStage;
use crate::{
    DiagnosticsReport, Error, Generated, GpuAddress, MatrixConversion, MatrixFormat,
    MeshPipelineDesc,
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

/// Host memory imported for the GPU to read in place; not on Metal yet, where
/// `newBufferWithBytesNoCopy` would wrap the pages.
pub struct ImportedMemory {
    never: !,
}

impl ImportedMemory {
    pub fn gpu(&self) -> GpuAddress {
        match self.never {}
    }

    pub fn size(&self) -> u64 {
        match self.never {}
    }

    pub fn flush(&self, offset: u64, len: u64) -> Result<(), Error> {
        let _ = (offset, len);
        match self.never {}
    }
}

impl Device {
    /// Fails: Metal reports `Caps::host_memory_import` false.
    ///
    /// # Safety
    ///
    /// As on Vulkan: the memory stays allocated, readable and writable, until `keep` is
    /// dropped.
    pub unsafe fn import_host_memory(
        &self,
        ptr: std::ptr::NonNull<u8>,
        len: u64,
        keep: Box<dyn std::any::Any + Send + Sync>,
        name: &str,
    ) -> Result<ImportedMemory, Error> {
        let _ = (ptr, len, keep, name);
        Err(Error::NotImplemented("Device::import_host_memory"))
    }
}

/// Pipelines that GPU-written sequences pick by index; not on Metal yet.
pub struct ExecutionSet {
    never: !,
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

/// A ray-dispatch pipeline; Metal has none.
pub struct RayDispatchPipeline {
    never: !,
}

/// A ray tracing pipeline; Metal has none.
pub struct RayPipeline {
    never: !,
}

impl RayPipeline {
    pub(super) fn unreachable(&self) -> ! {
        match self.never {}
    }

    pub fn hit_handle(&self, group: u32) -> &[u8] {
        let _ = group;
        match self.never {}
    }

    pub fn hit_handles(&self) -> &[u8] {
        match self.never {}
    }

    pub fn hit_record_bytes(&self, group: u32) -> u32 {
        let _ = group;
        match self.never {}
    }

    pub fn hit_stride(&self) -> u64 {
        match self.never {}
    }

    pub fn write_hit_record(&self, record: &mut [u8], group: u32, data: &[u8]) {
        let _ = (record, group, data);
        match self.never {}
    }
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

    pub fn create_ray_pipeline(
        &self,
        desc: &crate::RayPipelineDesc<'_>,
        name: &str,
    ) -> Result<RayPipeline, Error> {
        let _ = (desc, name);
        Err(Error::NotImplemented("ray tracing pipelines"))
    }
}
