//! Shader stages and compute pipelines in descriptor-heap mode, with the root at binding (0, 0)
//! or in the push data.

use std::borrow::Cow;
use std::ffi::CStr;
use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::debug::name_object;
use super::device::{Device, Shared};
use super::driver_cpu::DriverCpu;
use super::garbage::Item;
use super::spirv;
use crate::specialize;
use crate::{
    CodeStatistics, CompileCost, EntryPoint, Error, PipelineKind, RayShader, ShaderModule,
    SpecData, Stage, StageEntry,
};

/// One shader stage of a pipeline: SPIR-V words, the entry point and specialization data.
#[derive(Clone, Copy, Debug)]
pub struct ShaderStage<'a> {
    pub spirv: &'a [u32],
    /// The module's Metal libraries, or a hot reload's. Unused here; they mirror the Metal
    /// backend's.
    pub metallib: &'a [u8],
    pub metallib_fragment: &'a [u8],
    pub metallib_stages: &'a [u8],
    /// `spirv` declares the opacity-micromap execution mode, which needs `Caps::omm`.
    pub omm: bool,
    /// The size of the root the kernel holds in the push data; 0 for a root in GPU memory.
    pub inline_root_bytes: u32,
    /// The path of `spirv`'s separated debug info, handed to Aftermath; empty for none.
    pub debug_info: &'a str,
    pub entry: &'a EntryPoint,
    pub spec: SpecData<'a>,
    /// The name the stage's entry point takes in the code the driver is handed, which its tools
    /// show the stage by; empty for the entry's own.
    pub label: &'a CStr,
    /// `spirv` is already the stage's code as [`Self::fold`] makes it, which the driver is
    /// handed as it is.
    pub folded: bool,
}

impl<'a> ShaderStage<'a> {
    /// The stage named `label` for the driver's tools: stages made of one entry with different
    /// constants are told apart by it.
    pub fn labeled(self, label: &'a CStr) -> Self {
        Self { label, ..self }
    }

    /// The stage with `code`, what [`Self::fold`] made of it, in place of its module's.
    pub fn folded(self, code: &'a [u32]) -> Self {
        Self {
            spirv: code,
            folded: true,
            ..self
        }
    }

    /// The code the driver is handed for the stage: its entry point alone, with its constants
    /// folded in (and a task or mesh entry's payload declared first). `None` where that is
    /// `spirv` as it is, as for a stage already folded.
    pub fn fold(&self) -> Option<Vec<u32>> {
        if self.folded {
            return None;
        }
        let _zone = crate::zone!("specialize");
        let started = Instant::now();
        let code = if matches!(self.entry.stage, Stage::Task | Stage::Mesh) {
            spirv::payload_first(self.spirv, self.entry.spirv_name.to_bytes())
        } else {
            Cow::Borrowed(self.spirv)
        };
        let folded = specialize::specialize(
            &code,
            self.entry.spirv_name.to_bytes(),
            self.spec.entries,
            self.spec.bytes,
        );
        if let Some(folded) = &folded {
            let name = if self.label.is_empty() {
                self.entry.spirv_name
            } else {
                self.label
            };
            log::debug!(
                "{}: {} of {} words left by its specialization, in {:.1} ms",
                name.to_string_lossy(),
                folded.len(),
                code.len(),
                started.elapsed().as_secs_f64() * 1e3
            );
        }
        folded.or(match code {
            Cow::Owned(code) => Some(code),
            Cow::Borrowed(_) => None,
        })
    }

    /// The entry point `entry` of `module`.
    pub fn new(module: &'a ShaderModule, entry: &str, spec: SpecData<'a>) -> Result<Self, Error> {
        let entry = module
            .entry_point(entry)
            .ok_or_else(|| Error::Shader(format!("{} has no entry point {entry}", module.name)))?;
        Ok(Self {
            spirv: module.spirv,
            metallib: module.metallib,
            metallib_fragment: module.metallib_fragment,
            metallib_stages: module.metallib_stages,
            omm: module.omm,
            inline_root_bytes: module.inline_root_bytes,
            debug_info: module.debug_info,
            entry,
            spec,
            label: c"",
            folded: false,
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

/// A compute pipeline. Dropping it destroys it once the GPU is done with the work submitted
/// before the drop and with the command lists open at it.
pub struct ComputePipeline {
    pub(super) raw: vk::Pipeline,
    /// The size of the root the kernel holds in the push data; 0 for a root in GPU memory.
    pub(super) inline_root_bytes: u32,
    workgroup: [u32; 3],
    compiled: Option<CompileCost>,
    shared: Arc<Shared>,
}

impl ComputePipeline {
    /// The entry point's workgroup size.
    pub fn workgroup(&self) -> [u32; 3] {
        self.workgroup
    }

    /// What compiling the pipeline cost the driver; `None` where it came from a cache or the
    /// pipeline store.
    pub fn compile_cost(&self) -> Option<&CompileCost> {
        self.compiled.as_ref()
    }

    /// What the driver reports of the pipeline's code; `None` where it reports nothing
    /// (without `Caps::pipeline_executable_info`) or not these figures.
    pub fn statistics(&self) -> Option<CodeStatistics> {
        self.shared.code_statistics(self.raw)
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Pipeline(self.raw));
    }
}

impl Shared {
    /// What the driver reports of the code of `pipeline`, which was created with
    /// CAPTURE_STATISTICS: of its executables the most registers, local and shared memory any
    /// one takes, and the code of all of them. `None` where it reports no executables, or names
    /// no register count or local memory among their statistics, as only NVIDIA's do.
    pub(super) fn code_statistics(&self, pipeline: vk::Pipeline) -> Option<CodeStatistics> {
        let exec = self.exec_fn.as_ref()?;
        let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
        // SAFETY: the pipeline was created with CAPTURE_STATISTICS.
        let count = unsafe { exec.get_pipeline_executable_properties_len(&info) }.ok()?;
        if count == 0 {
            return None;
        }
        let mut out = CodeStatistics::default();
        let (mut registers, mut local) = (false, false);
        for i in 0..count as u32 {
            let stats = self.executable_statistics(pipeline, i)?;
            for s in &stats {
                // SAFETY: the format says which union member is set.
                let value = unsafe {
                    match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64,
                        vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64 as u64,
                        _ => continue,
                    }
                };
                match s.name_as_c_str().ok().and_then(|c| c.to_str().ok()) {
                    Some("Register Count") => {
                        registers = true;
                        out.registers = out.registers.max(value as u32);
                    }
                    // NVIDIA's drivers set bit 36 beside the bytes.
                    Some("Local Memory Size") => {
                        local = true;
                        out.local_bytes = out.local_bytes.max(value & 0xffff_ffff);
                    }
                    Some("Shared Memory Size") => out.shared_bytes = out.shared_bytes.max(value),
                    Some("Binary Size") => out.binary_bytes += value,
                    _ => {}
                }
            }
        }
        (registers && local).then_some(out)
    }

    /// The statistics the driver reports for executable `index` of `pipeline`, which was
    /// created with CAPTURE_STATISTICS; `None` without `Caps::pipeline_executable_info`.
    fn executable_statistics(
        &self,
        pipeline: vk::Pipeline,
        index: u32,
    ) -> Option<Vec<vk::PipelineExecutableStatisticKHR<'static>>> {
        let exec = self.exec_fn.as_ref()?;
        let einfo = vk::PipelineExecutableInfoKHR::default()
            .pipeline(pipeline)
            .executable_index(index);
        // SAFETY: the pipeline was created with CAPTURE_STATISTICS.
        let n = unsafe { exec.get_pipeline_executable_statistics_len(&einfo) }.ok()?;
        let mut stats = vec![vk::PipelineExecutableStatisticKHR::default(); n];
        // SAFETY: as above, with room for `n` statistics.
        unsafe { exec.get_pipeline_executable_statistics(&einfo, &mut stats) }.ok()?;
        Some(stats)
    }

    /// Logs every statistic the driver reports for each of the pipeline's executables.
    fn log_statistics(&self, pipeline: vk::Pipeline, name: &str) {
        let Some(exec) = &self.exec_fn else { return };
        if !log::log_enabled!(log::Level::Debug) {
            return;
        }
        let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
        // SAFETY: the pipeline was created with CAPTURE_STATISTICS.
        let count = unsafe { exec.get_pipeline_executable_properties_len(&info) }.unwrap_or(0);
        for i in 0..count as u32 {
            let Some(stats) = self.executable_statistics(pipeline, i) else {
                continue;
            };
            let mut line = String::new();
            for s in &stats {
                let label = s
                    .name_as_c_str()
                    .map_or_default(|c| c.to_string_lossy().into_owned());
                // SAFETY: the format says which union member is set.
                let value = unsafe {
                    match s.format {
                        vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                            (s.value.b32 != vk::FALSE).to_string()
                        }
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

/// What a stage's create info points to: code, root mapping and specialization entries.
pub(super) struct StageParts<'a> {
    code: Cow<'a, [u32]>,
    entry: &'a CStr,
    /// Empty for a root in the push data.
    mappings: Vec<vk::DescriptorSetAndBindingMappingEXT<'static>>,
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
        if matches!(
            want,
            Stage::Task | Stage::Mesh | Stage::Vertex | Stage::Pixel
        ) && s.inline_root_bytes > 0
        {
            return Err(Error::Shader(format!(
                "{name}: a {want:?} entry cannot use ORR_ROOT_INLINE"
            )));
        }
        if s.omm && !shared.caps.omm {
            return Err(Error::Shader(format!(
                "{name}: the module was built with opacity micromaps, which the device has off; build the shaders with ORR_SHADER_OMM=0, or ORR_SHADER_CAPTURE=1 for a capture session"
            )));
        }
        // A stage with no constants is still pruned to its entry: the driver and the validation
        // layer otherwise read every other entry point's code once per stage.
        let code = s.fold().map_or(Cow::Borrowed(s.spirv), Cow::Owned);
        let (code, entry) = match specialize::rename_entry(
            &code,
            s.entry.spirv_name.to_bytes(),
            s.label.to_bytes(),
        ) {
            Some(renamed) if !s.label.is_empty() => (Cow::Owned(renamed), s.label),
            _ => (code, s.entry.spirv_name),
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
            entry,
            // A ray tracing stage's root, if any, is inline: the build checks it.
            mappings: if s.inline_root_bytes > 0 || want.ray_tracing() {
                Vec::new()
            } else {
                vec![root_mapping()]
            },
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

    /// The stage as the driver is handed it.
    pub(super) fn stage_entry(&self) -> StageEntry {
        StageEntry {
            name: self.entry.to_string_lossy().into_owned(),
            spec: self.spec.to_vec(),
        }
    }

    /// The instructions the stage's functions hand the driver.
    pub(super) fn instructions(&self) -> u32 {
        specialize::handed_instructions(&self.code)
    }

    /// Adds the stage's code, entry and specialization to `hasher`.
    pub(super) fn key(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&(self.code.len() as u64).to_le_bytes());
        hasher.update(bytemuck::cast_slice(&self.code));
        hasher.update(self.entry.to_bytes_with_nul());
        hasher.update(self.spec);
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
        let info = info.push(&mut self.module);
        // Ray tracing pipelines with this struct on a stage miss every pipeline cache.
        if self.mapping.mapping_count == 0 {
            info
        } else {
            info.push(&mut self.mapping)
        }
    }
}

impl Shared {
    pub(super) fn pipeline_flags(&self, statistics: bool) -> vk::PipelineCreateFlags2 {
        let mut flags = vk::PipelineCreateFlags2::DESCRIPTOR_HEAP_EXT;
        if statistics && self.exec_fn.is_some() {
            flags |= vk::PipelineCreateFlags2::CAPTURE_STATISTICS_KHR;
        }
        flags
    }

    /// Names a new pipeline and logs its creation, what it cost the driver where it `compiled`,
    /// its statistics when it was created with `statistics`, and its ray `shaders`' stacks: at info
    /// for a compile, at debug for a pipeline from a cache, which a run loads again and again.
    pub(super) fn created(
        &self,
        raw: vk::Pipeline,
        name: &str,
        start: Instant,
        compiled: Option<&CompileCost>,
        statistics: bool,
        shaders: &[RayShader],
    ) {
        name_object(&self.debug_fn, raw, name);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        match compiled {
            Some(c) => log::debug!(
                "pipeline {name}: created in {:.1} ms, compiled in {:.1} ms with {:.1} ms of \
                 driver CPU, {:.1} ms of it serial and {:.1} ms the longest of {} joins",
                ms(start.elapsed()),
                ms(c.wall),
                ms(c.cpu),
                ms(c.serial),
                ms(c.longest_join),
                c.joins
            ),
            None => log::debug!(
                "pipeline {name}: created in {:.1} ms from a cache",
                ms(start.elapsed())
            ),
        }
        if statistics {
            self.log_statistics(raw, name);
        }
        if !shaders.is_empty() {
            let mut kept: Vec<_> = shaders.iter().filter(|s| s.stack_bytes > 0).collect();
            kept.sort_by_key(|s| std::cmp::Reverse(s.stack_bytes));
            let listed: Vec<_> = kept
                .iter()
                .map(|s| format!("{} {} B", s.shader, s.stack_bytes))
                .collect();
            let level = if compiled.is_some() {
                log::Level::Info
            } else {
                log::Level::Debug
            };
            log::log!(
                level,
                "pipeline {name}: shader stacks {}; {} of {} shaders keep none",
                if listed.is_empty() {
                    "none".into()
                } else {
                    listed.join(", ")
                },
                shaders.len() - kept.len(),
                shaders.len()
            );
        }
    }

    /// A compute pipeline with `extra` creation flags.
    pub(super) fn create_compute(
        &self,
        s: &ShaderStage<'_>,
        extra: vk::PipelineCreateFlags2,
        name: &str,
    ) -> Result<(vk::Pipeline, Option<CompileCost>), Error> {
        let parts = StageParts::new(self, s, Stage::Compute, name)?;
        self.create_pipeline(
            &[&parts],
            &[],
            extra,
            PipelineKind::Compute,
            name,
            "vkCreateComputePipelines",
            |cache, next| {
                let mut stage = parts.chain();
                let mut info = vk::ComputePipelineCreateInfo::default()
                    .stage(stage.info(vk::ShaderStageFlags::COMPUTE));
                info.p_next = next;
                let (created, cpu) = DriverCpu::serial(|| {
                    // SAFETY: valid create info; the SPIR-V was validated by the shader build.
                    unsafe { self.device.create_compute_pipelines(cache, &[info], None) }
                });
                created.map(|p| (p[0], cpu)).map_err(|(_, e)| e)
            },
            |_| Vec::new(),
        )
    }
}

impl Device {
    /// Creates a compute pipeline from a compute entry point. Threads asking for one pipeline at
    /// once compile it once: the others wait for it.
    pub fn create_compute_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<ComputePipeline, Error> {
        let _zone = crate::zone!("create pipeline", name);
        let (raw, compiled) = self.shared.create_compute(s, self.creation_flags, name)?;
        Ok(ComputePipeline {
            raw,
            inline_root_bytes: s.inline_root_bytes,
            workgroup: s.entry.workgroup,
            compiled,
            shared: Arc::clone(&self.shared),
        })
    }
}
