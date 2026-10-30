//! Shader stages and compute pipelines in descriptor-heap mode with the root at binding
//! (0, 0).

use std::borrow::Cow;
use std::ffi::CStr;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::cache::{KeyHasher, PipelineKind};
use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use super::spirv;
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
        let code = shared.code(s);
        let code = if matches!(want, Stage::Task | Stage::Mesh) {
            spirv::payload_first(code, s.entry.name.to_bytes())
        } else {
            Cow::Borrowed(code)
        };
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

    /// Feeds everything that shapes the compiled stage into a pipeline key.
    pub(super) fn hash(&self, stage: vk::ShaderStageFlags, h: &mut impl Hasher) {
        stage.as_raw().hash(h);
        bytemuck::cast_slice::<u32, u8>(&self.code).hash(h);
        self.entry.to_bytes().hash(h);
        self.map_entries.len().hash(h);
        for e in &self.map_entries {
            (e.constant_id, e.offset, e.size).hash(h);
        }
        self.spec.hash(h);
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

    pub(crate) fn create_compute(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<vk::Pipeline, Error> {
        let parts = StageParts::new(self, s, Stage::Compute, name)?;
        let mut key = KeyHasher::new(PipelineKind::Compute);
        parts.hash(vk::ShaderStageFlags::COMPUTE, &mut key);
        self.create_pipeline(
            PipelineKind::Compute,
            key.finish_key(),
            name,
            "vkCreateComputePipelines",
            |chain| {
                let mut stage = parts.chain();
                let mut info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage.info(vk::ShaderStageFlags::COMPUTE));
                info.p_next = chain.head();
                // SAFETY: valid create info; the SPIR-V was validated by the shader build.
                unsafe {
                    self.device
                        .create_compute_pipelines(chain.cache, &[info], None)
                }
                .map(|p| p[0])
                .map_err(|(_, e)| e)
            },
        )
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
            raw: self.shared.create_compute(s, name)?,
            workgroup: s.entry.workgroup,
            shared: Arc::clone(&self.shared),
        })
    }
}
