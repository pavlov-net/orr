//! Ray-dispatch pipelines: one raygen stage with a raygen-only shader table, for kernels that
//! reorder invocations at coherence points (contracts.md 3).

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use super::heap::SlotAllocator;
use super::memory::MemoryKind;
use super::pipeline::{ShaderStage, StageParts};
use crate::{Error, Stage};

/// Raygen records the block holds: one per live ray-dispatch pipeline.
const RECORDS: u32 = 4096;

/// The block of raygen shader-table records, one per ray-dispatch pipeline.
pub(crate) struct RecordBlock {
    pub raw: RawBuffer,
    ptr: NonNull<u8>,
    handle_bytes: usize,
    /// Record size and stride: the handle rounded up to the base alignment.
    stride: u64,
    slots: Mutex<SlotAllocator>,
}

// SAFETY: the mapped pointer is plain memory; a record is written only by the pipeline that
// took its slot.
unsafe impl Send for RecordBlock {}
// SAFETY: as above.
unsafe impl Sync for RecordBlock {}

impl RecordBlock {
    pub(crate) fn new(
        shared: &Shared,
        props: &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'_>,
    ) -> Result<RecordBlock, Error> {
        let base = u64::from(props.shader_group_base_alignment);
        let stride = u64::from(props.shader_group_handle_size).next_multiple_of(base);
        let (raw, ptr) = shared.create_buffer(
            MemoryKind::Upload,
            stride * u64::from(RECORDS),
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            base,
            "orr raygen records",
        )?;
        let Some(ptr) = ptr else {
            shared.destroy(Item::Buffer(raw));
            return Err(Error::Invalid(
                "raygen records: upload memory is not mapped".into(),
            ));
        };
        Ok(RecordBlock {
            raw,
            ptr,
            handle_bytes: props.shader_group_handle_size as usize,
            stride,
            slots: Mutex::default(),
        })
    }

    pub(crate) fn release(&self, slot: u32) {
        self.slots.lock().unwrap().release(slot);
    }
}

/// A ray-dispatch pipeline: one raygen entry run by [`crate::CommandList::dispatch_rays`].
/// Dropping it destroys it once everything submitted before the drop has completed.
pub struct RayDispatchPipeline {
    pub(crate) raw: vk::Pipeline,
    /// The raygen region of the shader table: this pipeline's record.
    pub(crate) raygen: vk::StridedDeviceAddressRegionKHR,
    slot: u32,
    shared: Arc<Shared>,
}

impl Drop for RayDispatchPipeline {
    fn drop(&mut self) {
        let after = self.shared.submitted();
        self.shared.defer(after, Item::Pipeline(self.raw));
        self.shared.defer(after, Item::RaygenRecord(self.slot));
    }
}

impl Device {
    /// Creates a ray-dispatch pipeline from a raygen entry point: one raygen stage, a
    /// raygen-only shader table (no miss, hit or callable records), recursion depth 0 and
    /// the root mapped as for compute. The entry traces with ray query; it cannot call
    /// `TraceRay` or reorder a hit object. Fails unless `Caps::ray_dispatch`.
    pub fn create_ray_dispatch_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<RayDispatchPipeline, Error> {
        let shared = &self.shared;
        let (Some(rt), Some(records)) = (&shared.rt_fn, shared.records.get()) else {
            return Err(Error::Unsupported {
                device: shared.caps.device_name.clone(),
                missing: "ray-dispatch pipelines (VK_KHR_ray_tracing_pipeline with \
                          VK_EXT_ray_tracing_invocation_reorder)"
                    .into(),
            });
        };
        let start = Instant::now();
        let parts = StageParts::new(shared, s, Stage::RayGen, name)?;
        let mut chain = parts.chain();
        let stages = [chain.info(vk::ShaderStageFlags::RAYGEN_KHR)];
        let groups = [vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)];
        let mut flags =
            vk::PipelineCreateFlags2CreateInfo::default().flags(shared.pipeline_flags());
        let info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(0)
            .push(&mut flags);
        // SAFETY: valid create info; the SPIR-V was validated by the shader build.
        let raw = unsafe {
            rt.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                vk::PipelineCache::null(),
                &[info],
                None,
            )
        }
        .map_err(|(_, e)| Error::vk("vkCreateRayTracingPipelinesKHR")(e))?[0];
        let destroy = |e: Error| {
            // SAFETY: the pipeline was never used.
            unsafe { shared.device.destroy_pipeline(raw, None) };
            e
        };
        // SAFETY: the pipeline has one group.
        let handle =
            unsafe { rt.get_ray_tracing_shader_group_handles(raw, 0, 1, records.handle_bytes) }
                .map_err(|e| destroy(Error::vk("vkGetRayTracingShaderGroupHandlesKHR")(e)))?;
        let slot = records
            .slots
            .lock()
            .unwrap()
            .take(RECORDS)
            .ok_or_else(|| destroy(Error::HeapFull("raygen record")))?;
        let offset = u64::from(slot) * records.stride;
        // SAFETY: the record lies inside the mapped block, and its slot is this pipeline's
        // alone; the GPU stopped reading it before the slot was freed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                handle.as_ptr(),
                records.ptr.as_ptr().add(offset as usize),
                handle.len(),
            );
        }
        shared.created(raw, name, start);
        Ok(RayDispatchPipeline {
            raw,
            raygen: vk::StridedDeviceAddressRegionKHR {
                device_address: records.raw.address + offset,
                stride: records.stride,
                size: records.stride,
            },
            slot,
            shared: Arc::clone(shared),
        })
    }
}
