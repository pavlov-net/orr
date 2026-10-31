//! Compute pipelines created through `Scenes`, which hot reload replaces when their kernels
//! change.

use std::ptr;
use std::sync::Arc;

use orr_gpu::{ComputePipeline, Device, KernelSet, ShaderModule, ShaderStage, SpecData, SpecEntry};

use crate::Result;
use crate::shaders::{animation, emissive, upload, world};

/// A compute pipeline of [`crate::Scenes::compute_pipeline`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PipelineId(u32);

/// What hot reload did to a kernel, reported at the step 1 that applied it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReloadReport {
    /// The kernel's pipelines run its new SPIR-V from this frame on.
    Swapped {
        kernel: &'static str,
        pipelines: u32,
    },
    /// The kernel failed to compile, with the file and line, or needs a rebuild ("rebuild
    /// required"); its pipelines keep running the SPIR-V they had.
    Rejected {
        kernel: &'static str,
        reason: String,
    },
}

/// A module's SPIR-V from a reload.
pub(crate) struct Code {
    pub(crate) spirv: Arc<[u32]>,
    pub(crate) spirv_omm: Arc<[u32]>,
}

/// How a pipeline was made, so a reload can make it again from new SPIR-V.
pub(crate) struct Recipe {
    pub(crate) module: &'static ShaderModule,
    entry: String,
    spec_entries: Vec<SpecEntry>,
    spec_bytes: Vec<u8>,
}

impl Recipe {
    /// The pipeline, from `code` in place of the module's SPIR-V when given.
    pub(crate) fn create(&self, device: &Device, code: Option<&Code>) -> Result<ComputePipeline> {
        let spec = SpecData {
            entries: &self.spec_entries,
            bytes: &self.spec_bytes,
        };
        let mut stage = ShaderStage::new(self.module, &self.entry, spec)?;
        if let Some(code) = code {
            stage.spirv = &code.spirv;
            stage.spirv_omm = &code.spirv_omm;
            // The module's sidecar describes the build's SPIR-V; a reload embeds its own.
            stage.debug_info = "";
        }
        Ok(device.create_compute_pipeline(&stage, self.module.name)?)
    }
}

/// orr-scene's own pipelines.
pub(crate) struct Own {
    pub(crate) scatter: PipelineId,
    pub(crate) rebase: PipelineId,
    pub(crate) emissive_instances: PipelineId,
    pub(crate) emissive_producers: PipelineId,
    pub(crate) animate: PipelineId,
}

pub(crate) struct Pipelines {
    pub(crate) recipes: Vec<Arc<Recipe>>,
    list: Vec<ComputePipeline>,
    /// The latest reloaded SPIR-V of each module that has reloaded.
    code: Vec<(&'static ShaderModule, Arc<Code>)>,
    pub(crate) own: Own,
}

impl Pipelines {
    pub(crate) fn new(device: &Device, kernels: &KernelSet) -> Result<Self> {
        let mut p = Self {
            recipes: Vec::new(),
            list: Vec::new(),
            code: Vec::new(),
            own: Own {
                scatter: PipelineId(0),
                rebase: PipelineId(0),
                emissive_instances: PipelineId(0),
                emissive_producers: PipelineId(0),
                animate: PipelineId(0),
            },
        };
        let none = SpecData::default();
        p.own = Own {
            scatter: p.create(device, kernels, &upload::MODULE, "scatter", none)?,
            rebase: p.create(device, kernels, &world::MODULE, "rebase_lights", none)?,
            emissive_instances: p.create(device, kernels, &emissive::MODULE, "instances", none)?,
            emissive_producers: p.create(device, kernels, &emissive::MODULE, "producers", none)?,
            animate: p.create(device, kernels, &animation::MODULE, "animate", none)?,
        };
        Ok(p)
    }

    /// The pipeline of `own`'s `entry`, from `kernels`' module of the same kernel file when
    /// it holds one, and from the module's latest reload when it has reloaded.
    pub(crate) fn create(
        &mut self,
        device: &Device,
        kernels: &KernelSet,
        own: &'static ShaderModule,
        entry: &str,
        spec: SpecData<'_>,
    ) -> Result<PipelineId> {
        let recipe = Recipe {
            module: kernels.module(own)?,
            entry: entry.to_owned(),
            spec_entries: spec.entries.to_vec(),
            spec_bytes: spec.bytes.to_vec(),
        };
        let pipeline = recipe.create(device, self.code_of(recipe.module))?;
        self.recipes.push(Arc::new(recipe));
        self.list.push(pipeline);
        Ok(PipelineId(self.list.len() as u32 - 1))
    }

    pub(crate) fn get(&self, id: PipelineId) -> &ComputePipeline {
        &self.list[id.0 as usize]
    }

    /// Puts `pipeline` in recipe `index`'s place. The old pipeline is destroyed once the
    /// work submitted before it was replaced has completed.
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    pub(crate) fn replace(&mut self, index: usize, pipeline: ComputePipeline) {
        self.list[index] = pipeline;
    }

    /// Records the SPIR-V `module` reloaded to, for pipelines created after the reload.
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    pub(crate) fn reloaded(&mut self, module: &'static ShaderModule, code: Arc<Code>) {
        match self.code.iter_mut().find(|(m, _)| ptr::eq(*m, module)) {
            Some((_, c)) => *c = code,
            None => self.code.push((module, code)),
        }
    }

    fn code_of(&self, module: &'static ShaderModule) -> Option<&Code> {
        self.code
            .iter()
            .find(|(m, _)| ptr::eq(*m, module))
            .map(|(_, c)| &**c)
    }
}
