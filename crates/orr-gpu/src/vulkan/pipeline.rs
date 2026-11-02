//! Shader stages and compute pipelines in descriptor-heap mode with the root at binding
//! (0, 0).

use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use super::spirv;
use crate::{EntryPoint, Error, ShaderModule, SpecData, Stage};

/// One shader stage of a pipeline: SPIR-V words, the entry point and specialisation data.
#[derive(Clone, Copy, Debug)]
pub struct ShaderStage<'a> {
    pub spirv: &'a [u32],
    /// `spirv` declares the opacity-micromap execution mode, which needs `Caps::omm`.
    pub omm: bool,
    /// The path of `spirv`'s separated debug info, handed to Aftermath; empty for none.
    pub debug_info: &'a str,
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
            omm: module.omm,
            debug_info: module.debug_info,
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

/// What a stage's create info points to: code, root mapping and specialisation entries.
pub(super) struct StageParts<'a> {
    code: Cow<'a, [u32]>,
    entry: &'a CStr,
    mappings: [vk::DescriptorSetAndBindingMappingEXT<'static>; 1],
    map_entries: Vec<vk::SpecializationMapEntry>,
    spec: &'a [u8],
}

/// The structs chained into a stage's create info.
pub(super) struct StageChain<'p> {
    module: vk::ShaderModuleCreateInfo<'p>,
    mapping: vk::ShaderDescriptorSetAndBindingMappingInfoEXT<'p>,
    spec: vk::SpecializationInfo<'p>,
    entry: &'p CStr,
}

impl<'a> StageParts<'a> {
    /// The parts of `s`, after checking that its entry is a `want` entry.
    pub(super) fn new(
        shared: &Shared,
        s: &ShaderStage<'a>,
        want: Stage,
        name: &str,
    ) -> Result<Self, Error> {
        if s.entry.stage != want {
            return Err(Error::Shader(format!(
                "{name}: {:?} is a {:?} entry, not {want:?}",
                s.entry.name, s.entry.stage
            )));
        }
        if s.omm && !shared.caps.omm {
            return Err(Error::Shader(format!(
                "{name}: the module was built with opacity micromaps, which the device has off; build the shaders with ORR_SHADER_OMM=0, or ORR_SHADER_CAPTURE=1 for a capture session"
            )));
        }
        let code = if matches!(want, Stage::Task | Stage::Mesh) {
            spirv::payload_first(s.spirv, s.entry.name.to_bytes())
        } else {
            Cow::Borrowed(s.spirv)
        };
        if shared.caps.cooperative_vector.is_empty() && spirv::cooperative_vectors(&code) {
            return Err(Error::Shader(format!(
                "{name}: the module uses cooperative vectors, which the device has none of; build the shaders with ORR_SHADER_NEURAL=0"
            )));
        }
        if !shared.caps.shader_abort && spirv::aborts(&code) {
            return Err(Error::Shader(format!(
                "{name}: the module was built with asserts, which need shaderAbort, and the device has none; build the shaders with ORR_SHADER_ASSERTS=0, or ORR_SHADER_CAPTURE=1 for a capture session"
            )));
        }
        if shared.caps.aftermath {
            // A sidecar that is not on this machine leaves the dump without source lines.
            if !s.debug_info.is_empty()
                && let Ok(sidecar) = std::fs::read(s.debug_info)
            {
                super::aftermath::add_sidecar(&code, &sidecar);
            } else {
                super::aftermath::add_shader(&code);
            }
        }
        Ok(Self {
            code,
            entry: s.entry.name,
            mappings: [root_mapping()],
            map_entries: s
                .spec
                .entries
                .iter()
                .map(|e| vk::SpecializationMapEntry {
                    constant_id: e.constant_id,
                    offset: e.offset,
                    size: e.size as usize,
                })
                .collect(),
            spec: s.spec.bytes,
        })
    }

    pub(super) fn chain(&self) -> StageChain<'_> {
        StageChain {
            module: vk::ShaderModuleCreateInfo::default().code(&self.code),
            mapping: vk::ShaderDescriptorSetAndBindingMappingInfoEXT::default()
                .mappings(&self.mappings),
            spec: vk::SpecializationInfo::default()
                .map_entries(&self.map_entries)
                .data(self.spec),
            entry: self.entry,
        }
    }
}

impl StageChain<'_> {
    pub(super) fn info(
        &mut self,
        stage: vk::ShaderStageFlags,
    ) -> vk::PipelineShaderStageCreateInfo<'_> {
        let info = vk::PipelineShaderStageCreateInfo::default()
            .stage(stage)
            .name(self.entry);
        let info = if self.spec.map_entry_count == 0 {
            info
        } else {
            info.specialization_info(&self.spec)
        };
        info.push(&mut self.module).push(&mut self.mapping)
    }
}

impl Shared {
    pub(super) fn pipeline_flags(&self) -> vk::PipelineCreateFlags2 {
        let mut flags = vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT;
        if self.exec_fn.is_some() {
            flags |= vk::PipelineCreateFlags2::CAPTURE_STATISTICS_KHR;
        }
        flags
    }

    pub(super) fn created(&self, raw: vk::Pipeline, name: &str, start: Instant) {
        name_object(&self.debug_fn, raw, name);
        log::debug!(
            "pipeline {name}: created in {:.1} ms",
            start.elapsed().as_secs_f64() * 1e3
        );
        self.log_statistics(raw, name);
    }

    /// A compute pipeline with `extra` creation flags.
    pub(crate) fn create_compute(
        &self,
        s: &ShaderStage<'_>,
        extra: vk::PipelineCreateFlags2,
        name: &str,
    ) -> Result<vk::Pipeline, Error> {
        let parts = StageParts::new(self, s, Stage::Compute, name)?;
        self.create_pipeline(extra, name, "vkCreateComputePipelines", |cache, next| {
            let mut stage = parts.chain();
            let mut info = vk::ComputePipelineCreateInfo::default()
                .stage(stage.info(vk::ShaderStageFlags::COMPUTE));
            info.p_next = next;
            // SAFETY: valid create info; the SPIR-V was validated by the shader build.
            unsafe { self.device.create_compute_pipelines(cache, &[info], None) }
                .map(|p| p[0])
                .map_err(|(_, e)| e)
        })
    }
}

impl Device {
    /// Creates a compute pipeline from a compute entry point.
    pub fn create_compute_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<ComputePipeline, Error> {
        Ok(ComputePipeline {
            raw: self
                .shared
                .create_compute(s, vk::PipelineCreateFlags2::empty(), name)?,
            workgroup: s.entry.workgroup,
            shared: Arc::clone(&self.shared),
        })
    }
}
