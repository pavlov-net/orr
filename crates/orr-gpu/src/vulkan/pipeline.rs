//! Compute pipelines in descriptor-heap mode with the root at binding (0, 0).

use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use crate::{EntryPoint, Error, ShaderModule, SpecData, Stage};

/// One shader stage of a pipeline: SPIR-V words, the entry point and specialisation data.
#[derive(Clone, Copy, Debug)]
pub struct ShaderStage<'a> {
    pub spirv: &'a [u32],
    /// The opacity-micromap variant; used instead of `spirv` when `Caps::omm` is set and it
    /// is not empty.
    pub spirv_omm: &'a [u32],
    pub entry: &'a EntryPoint,
    pub spec: SpecData<'a>,
}

impl<'a> ShaderStage<'a> {
    /// The entry point `entry` of `module`.
    pub fn new(module: &'a ShaderModule, entry: &str, spec: SpecData<'a>) -> Result<Self, Error> {
        let entry = module
            .entry_point(entry)
            .ok_or_else(|| Error::Shader(format!("{} has no entry point {entry}", module.name)))?;
        Ok(Self {
            spirv: module.spirv,
            spirv_omm: module.spirv_omm,
            entry,
            spec,
        })
    }
}

/// The root binding: (0, 0) read from the 8-byte address at push-data offset 0.
fn root_mapping() -> vk::DescriptorSetAndBindingMappingEXT<'static> {
    vk::DescriptorSetAndBindingMappingEXT {
        descriptor_set: 0,
        first_binding: 0,
        binding_count: 1,
        resource_mask: vk::SpirvResourceTypeFlagsEXT::UNIFORM_BUFFER
            | vk::SpirvResourceTypeFlagsEXT::READ_ONLY_STORAGE_BUFFER,
        source: vk::DescriptorMappingSourceEXT::PUSH_ADDRESS,
        source_data: vk::DescriptorMappingSourceDataEXT {
            push_address_offset: 0,
        },
        ..Default::default()
    }
}

/// A compute pipeline. Dropping it destroys it once everything submitted before the drop
/// has completed.
pub struct ComputePipeline {
    pub(crate) raw: vk::Pipeline,
    workgroup: [u32; 3],
    shared: Arc<Shared>,
}

impl ComputePipeline {
    /// The entry point's workgroup size.
    pub fn workgroup(&self) -> [u32; 3] {
        self.workgroup
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Pipeline(self.raw));
    }
}

impl Shared {
    pub(crate) fn code<'a>(&self, stage: &ShaderStage<'a>) -> &'a [u32] {
        if self.caps.omm && !stage.spirv_omm.is_empty() {
            stage.spirv_omm
        } else {
            stage.spirv
        }
    }

    /// Logs registers, shared memory and spills per executable.
    fn log_statistics(&self, pipeline: vk::Pipeline, name: &str) {
        let Some(exec) = &self.exec_fn else { return };
        let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
        // SAFETY: the pipeline was created with CAPTURE_STATISTICS.
        let count = unsafe { exec.get_pipeline_executable_properties_len(&info) }.unwrap_or(0);
        for i in 0..count as u32 {
            let einfo = vk::PipelineExecutableInfoKHR::default()
                .pipeline(pipeline)
                .executable_index(i);
            // SAFETY: as above.
            let n = unsafe { exec.get_pipeline_executable_statistics_len(&einfo) }.unwrap_or(0);
            let mut stats = vec![vk::PipelineExecutableStatisticKHR::default(); n];
            // SAFETY: as above, with room for `n` statistics.
            if unsafe { exec.get_pipeline_executable_statistics(&einfo, &mut stats) }.is_err() {
                continue;
            }
            let mut line = String::new();
            for s in &stats {
                let label = s
                    .name_as_c_str()
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if !(label.contains("Register")
                    || label.contains("Spill")
                    || label.contains("Shared"))
                {
                    continue;
                }
                // SAFETY: the format says which union member is set.
                let value = unsafe {
                    match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
                        vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => {
                            s.value.f64.to_string()
                        }
                        _ => continue,
                    }
                };
                line += &format!(" {}={value}", label.replace(' ', "_"));
            }
            log::debug!("pipeline {name} executable {i}:{line}");
        }
    }
}

impl Device {
    /// Creates a compute pipeline from a compute entry point.
    pub fn create_compute_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<ComputePipeline, Error> {
        if s.entry.stage != Stage::Compute {
            return Err(Error::Shader(format!(
                "{name}: {:?} is a {:?} entry",
                s.entry.name, s.entry.stage
            )));
        }
        let shared = &self.shared;
        let start = Instant::now();
        let map_entries: Vec<_> = s
            .spec
            .entries
            .iter()
            .map(|e| vk::SpecializationMapEntry {
                constant_id: e.constant_id,
                offset: e.offset,
                size: e.size as usize,
            })
            .collect();
        let spec = vk::SpecializationInfo::default()
            .map_entries(&map_entries)
            .data(s.spec.bytes);
        let mut module = vk::ShaderModuleCreateInfo::default().code(shared.code(s));
        let mappings = [root_mapping()];
        let mut mapping =
            vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default().mappings(&mappings);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .name(s.entry.name)
            .push(&mut module)
            .push(&mut mapping);
        if !map_entries.is_empty() {
            stage = stage.specialization_info(&spec);
        }
        let mut flags = vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT;
        if shared.exec_fn.is_some() {
            flags |= vk::PipelineCreateFlags2::CAPTURE_STATISTICS_KHR;
        }
        let mut flags = vk::PipelineCreateFlags2CreateInfo::default().flags(flags);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .push(&mut flags);
        // SAFETY: valid create info; the SPIR-V was validated by the shader build.
        let raw = unsafe {
            shared
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, e)| Error::vk("vkCreateComputePipelines")(e))?[0];
        name_object(&shared.debug_fn, raw, name);
        log::debug!(
            "pipeline {name}: created in {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        shared.log_statistics(raw, name);
        Ok(ComputePipeline {
            raw,
            workgroup: s.entry.workgroup,
            shared: Arc::clone(shared),
        })
    }
}
