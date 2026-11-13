//! Ray-dispatch pipelines: one raygen stage with a raygen-only shader table, for kernels that
//! reorder invocations at coherence points.

use std::mem::DropGuard;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use ash::vk;

use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use super::heap::SlotAllocator;
use super::pipeline::{ShaderStage, StageParts};
use crate::{Error, MemoryKind, Stage};

/// Raygen records the block holds: one per live ray-dispatch pipeline.
const RECORDS: u32 = 4096;

/// The block of raygen shader-table records, one per ray-dispatch pipeline.
pub(super) struct RecordBlock {
    pub(super) raw: RawBuffer,
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

    pub(super) fn release(&self, slot: u32) {
        self.slots.lock().unwrap().release(slot);
    }
}

/// A ray-dispatch pipeline: one raygen entry run by [`crate::CommandList::dispatch_rays`].
/// Dropping it destroys it once the GPU is done with the work submitted before the drop and
/// with the command lists open at it.
pub struct RayDispatchPipeline {
    pub(super) raw: vk::Pipeline,
    /// The size of the root the kernel holds in the push data.
    pub(super) inline_root_bytes: u32,
    /// The raygen region of the shader table: this pipeline's record.
    pub(super) raygen: vk::StridedDeviceAddressRegionKHR,
    slot: u32,
    shared: Arc<Shared>,
}

impl Drop for RayDispatchPipeline {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Pipeline(self.raw));
        self.shared
            .defer_after_submitted(Item::RaygenRecord(self.slot));
    }
}

impl Device {
    /// Creates a ray-dispatch pipeline from a raygen entry point: one raygen stage, a
    /// raygen-only shader table (no miss, hit or callable records), recursion depth 0 and
    /// the root held in the push data, so the kernel declares `ORR_ROOT_INLINE`. The entry traces
    /// with ray query; it cannot call `TraceRay` or reorder a hit object. Threads asking for one
    /// pipeline at once compile it once: the others wait for it. Fails unless
    /// `Caps::ray_dispatch`.
    pub fn create_ray_dispatch_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<RayDispatchPipeline, Error> {
        let _zone = crate::zone!("create pipeline", name);
        let shared = &self.shared;
        let (Some(rt), Some(records)) = (&shared.rt_fn, shared.records.get()) else {
            return Err(Error::Unsupported {
                device: shared.caps.device_name.clone(),
                missing: "ray-dispatch pipelines (VK_KHR_ray_tracing_pipeline with \
                          VK_EXT_ray_tracing_invocation_reorder)"
                    .into(),
            });
        };
        let parts = StageParts::new(shared, s, Stage::RayGen, name)?;
        let groups = [vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
            .general_shader(0)
            .closest_hit_shader(vk::SHADER_UNUSED_KHR)
            .any_hit_shader(vk::SHADER_UNUSED_KHR)
            .intersection_shader(vk::SHADER_UNUSED_KHR)];
        let raw = shared.create_pipeline(
            &[&parts],
            &[],
            self.creation_flags,
            name,
            "vkCreateRayTracingPipelinesKHR",
            |cache, next| {
                let mut stage = parts.chain();
                let stages = [stage.info(vk::ShaderStageFlags::RAYGEN_KHR)];
                let mut info = vk::RayTracingPipelineCreateInfoKHR::default()
                    .stages(&stages)
                    .groups(&groups)
                    .max_pipeline_ray_recursion_depth(0);
                info.p_next = next;
                // SAFETY: valid create info; the SPIR-V was validated by the shader build.
                unsafe {
                    rt.create_ray_tracing_pipelines(
                        vk::DeferredOperationKHR::null(),
                        cache,
                        &[info],
                        None,
                    )
                }
                .map(|p| p[0])
                .map_err(|(_, e)| e)
            },
        )?;
        // SAFETY: nothing uses the pipeline until it is returned.
        let raw = DropGuard::new(raw, |p| unsafe { shared.device.destroy_pipeline(p, None) });
        // SAFETY: the pipeline has one group.
        let handle =
            unsafe { rt.get_ray_tracing_shader_group_handles(*raw, 0, 1, records.handle_bytes) }
                .map_err(Error::vk("vkGetRayTracingShaderGroupHandlesKHR"))?;
        let slot = records
            .slots
            .lock()
            .unwrap()
            .take(RECORDS)
            .ok_or(Error::HeapFull("raygen record"))?;
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
        Ok(RayDispatchPipeline {
            inline_root_bytes: s.inline_root_bytes,
            raw: DropGuard::dismiss(raw),
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
