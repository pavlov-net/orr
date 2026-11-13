//! Compute and mesh pipelines created through `Scenes`, which hot reload replaces when their
//! kernels change.

use std::ptr;
use std::sync::Arc;

use orr_gpu::{
    ColorTarget, ComputePipeline, DepthTarget, Device, ExecutionSet, KernelSet, MeshPipeline,
    MeshPipelineDesc, RasterState, ShaderModule, ShaderStage, SpecData, SpecEntry,
};

use crate::Result;
use crate::shaders::{
    animation, emissive, instance_page, page_motion, transparency, upload, world,
};

/// A compute pipeline of [`crate::Scenes::compute_pipeline`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PipelineId(u32);

/// A mesh pipeline of [`crate::Scenes::mesh_pipeline`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshPipelineId(u32);

/// A mesh pipeline of one module: its entry points and what it draws into.
#[derive(Clone, Copy, Debug)]
pub struct MeshStages<'a> {
    pub task: Option<&'a str>,
    pub mesh: &'a str,
    /// `None`: no pixel stage; only depth is written.
    pub pixel: Option<&'a str>,
    /// Color targets in attachment order.
    pub colors: &'a [ColorTarget],
    pub depth: Option<DepthTarget>,
    pub raster: RasterState,
    /// Rasterization samples per pixel: 1, or more for pipelines with no targets.
    pub samples: u32,
}

impl<'a> MeshStages<'a> {
    /// A mesh stage alone: no targets, default raster state, one sample.
    pub fn new(mesh: &'a str) -> Self {
        Self {
            task: None,
            mesh,
            pixel: None,
            colors: &[],
            depth: None,
            raster: RasterState::default(),
            samples: 1,
        }
    }
}

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

/// A module's SPIR-V and Metal libraries from a reload.
pub(crate) struct Code {
    pub(crate) spirv: Arc<[u32]>,
    pub(crate) metallib: Arc<[u8]>,
    pub(crate) metallib_fragment: Arc<[u8]>,
}

/// What a recipe makes: a compute pipeline, an execution set holding the pipeline alone, which
/// device-generated sequences run, or a mesh pipeline.
pub(crate) enum Made {
    Pipeline(ComputePipeline),
    Set(ExecutionSet),
    Mesh(MeshPipeline),
}

/// What a recipe makes of its module.
enum Kind {
    Compute {
        entry: String,
        spec_entries: Vec<SpecEntry>,
        spec_bytes: Vec<u8>,
        set: bool,
    },
    Mesh {
        task: Option<String>,
        mesh: String,
        pixel: Option<String>,
        colors: Vec<ColorTarget>,
        depth: Option<DepthTarget>,
        raster: RasterState,
        samples: u32,
    },
}

/// How a pipeline was made, so a reload can make it again from new SPIR-V.
pub(crate) struct Recipe {
    pub(crate) module: &'static ShaderModule,
    kind: Kind,
}

impl Recipe {
    /// `module`'s compute `entry` specialized by `spec`, or with `set` an execution set holding
    /// it alone.
    pub(crate) fn compute(
        module: &'static ShaderModule,
        entry: &str,
        spec: SpecData<'_>,
        set: bool,
    ) -> Self {
        Self {
            module,
            kind: Kind::Compute {
                entry: entry.to_owned(),
                spec_entries: spec.entries.to_vec(),
                spec_bytes: spec.bytes.to_vec(),
                set,
            },
        }
    }

    /// The pipeline or set, from `code` in place of the module's SPIR-V when given.
    pub(crate) fn create(&self, device: &Device, code: Option<&Code>) -> Result<Made> {
        let stage = |entry: &str, spec| -> Result<ShaderStage<'_>> {
            let mut stage = ShaderStage::new(self.module, entry, spec)?;
            if let Some(code) = code {
                stage.spirv = &code.spirv;
                stage.metallib = &code.metallib;
                stage.metallib_fragment = &code.metallib_fragment;
                // The module's sidecar describes the build's SPIR-V; a reload embeds its own.
                stage.debug_info = "";
            }
            Ok(stage)
        };
        let name = self.module.name;
        Ok(match &self.kind {
            Kind::Compute {
                entry,
                spec_entries,
                spec_bytes,
                set,
            } => {
                let spec = SpecData {
                    entries: spec_entries,
                    bytes: spec_bytes,
                };
                let stage = stage(entry, spec)?;
                if *set {
                    Made::Set(device.create_compute_set(&[stage], name)?)
                } else {
                    Made::Pipeline(device.create_compute_pipeline(&stage, name)?)
                }
            }
            Kind::Mesh {
                task,
                mesh,
                pixel,
                colors,
                depth,
                raster,
                samples,
            } => {
                let optional = |entry: &Option<String>| {
                    entry
                        .as_deref()
                        .map(|e| stage(e, SpecData::default()))
                        .transpose()
                };
                let desc = MeshPipelineDesc {
                    task: optional(task)?,
                    mesh: stage(mesh, SpecData::default())?,
                    pixel: optional(pixel)?,
                    colors,
                    depth: *depth,
                    raster: *raster,
                    samples: *samples,
                };
                Made::Mesh(device.create_mesh_pipeline(&desc, name)?)
            }
        })
    }
}

/// orr_scene's own pipelines.
pub(crate) struct Own {
    pub(crate) scatter: PipelineId,
    pub(crate) rebase: PipelineId,
    pub(crate) emissive_instances: PipelineId,
    pub(crate) emissive_layout: PipelineId,
    pub(crate) emissive_blocks: PipelineId,
    pub(crate) emissive_producers: PipelineId,
    pub(crate) emissive_entries: PipelineId,
    pub(crate) animate: PipelineId,
    pub(crate) composite: PipelineId,
    pub(crate) expand_page: PipelineId,
    pub(crate) page_motion: PipelineId,
}

pub(crate) struct Pipelines {
    pub(crate) recipes: Vec<Arc<Recipe>>,
    list: Vec<Made>,
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
                expand_page: PipelineId(0),
                page_motion: PipelineId(0),
                emissive_instances: PipelineId(0),
                emissive_layout: PipelineId(0),
                emissive_blocks: PipelineId(0),
                emissive_producers: PipelineId(0),
                emissive_entries: PipelineId(0),
                animate: PipelineId(0),
                composite: PipelineId(0),
            },
        };
        let none = SpecData::default();
        p.own = Own {
            scatter: p.create(device, kernels, &upload::MODULE, "scatter", none)?,
            rebase: p.create(device, kernels, &world::MODULE, "rebase_lights", none)?,
            expand_page: p.create(device, kernels, &instance_page::MODULE, "expand", none)?,
            page_motion: p.create(device, kernels, &page_motion::MODULE, "motion", none)?,
            emissive_instances: p.create(device, kernels, &emissive::MODULE, "instances", none)?,
            emissive_layout: p.create(device, kernels, &emissive::MODULE, "layout_blocks", none)?,
            emissive_blocks: p.create(device, kernels, &emissive::MODULE, "sum_blocks", none)?,
            emissive_producers: p.create(device, kernels, &emissive::MODULE, "producers", none)?,
            emissive_entries: p.create(device, kernels, &emissive::MODULE, "entries", none)?,
            animate: p.create(device, kernels, &animation::MODULE, "animate", none)?,
            composite: p.create(device, kernels, &transparency::MODULE, "composite", none)?,
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
        self.add(device, kernels, own, entry, spec, false)
    }

    /// As [`Pipelines::create`], an execution set holding the pipeline alone; needs
    /// `Caps::generated_commands`.
    pub(crate) fn create_set(
        &mut self,
        device: &Device,
        kernels: &KernelSet,
        own: &'static ShaderModule,
        entry: &str,
        spec: SpecData<'_>,
    ) -> Result<PipelineId> {
        self.add(device, kernels, own, entry, spec, true)
    }

    fn add(
        &mut self,
        device: &Device,
        kernels: &KernelSet,
        own: &'static ShaderModule,
        entry: &str,
        spec: SpecData<'_>,
        set: bool,
    ) -> Result<PipelineId> {
        let recipe = Recipe::compute(kernels.module(own)?, entry, spec, set);
        Ok(PipelineId(self.make(device, recipe)?))
    }

    /// The mesh pipeline of `own`'s `stages`, from `kernels`' module of the same kernel file
    /// when it holds one, and from the module's latest reload when it has reloaded.
    pub(crate) fn create_mesh(
        &mut self,
        device: &Device,
        kernels: &KernelSet,
        own: &'static ShaderModule,
        stages: &MeshStages<'_>,
    ) -> Result<MeshPipelineId> {
        let kind = Kind::Mesh {
            task: stages.task.map(str::to_owned),
            mesh: stages.mesh.to_owned(),
            pixel: stages.pixel.map(str::to_owned),
            colors: stages.colors.to_vec(),
            depth: stages.depth,
            raster: stages.raster,
            samples: stages.samples,
        };
        let recipe = Recipe {
            module: kernels.module(own)?,
            kind,
        };
        Ok(MeshPipelineId(self.make(device, recipe)?))
    }

    /// Makes `recipe` and returns its place in the list.
    fn make(&mut self, device: &Device, recipe: Recipe) -> Result<u32> {
        let made = recipe.create(device, self.code_of(recipe.module).as_deref())?;
        self.recipes.push(Arc::new(recipe));
        self.list.push(made);
        Ok(self.list.len() as u32 - 1)
    }

    pub(crate) fn get(&self, id: PipelineId) -> &ComputePipeline {
        match &self.list[id.0 as usize] {
            Made::Pipeline(p) => p,
            _ => panic!("{id:?} is not a compute pipeline"),
        }
    }

    pub(crate) fn set(&self, id: PipelineId) -> &ExecutionSet {
        match &self.list[id.0 as usize] {
            Made::Set(s) => s,
            _ => panic!("{id:?} is not an execution set"),
        }
    }

    pub(crate) fn mesh(&self, id: MeshPipelineId) -> &MeshPipeline {
        match &self.list[id.0 as usize] {
            Made::Mesh(p) => p,
            _ => panic!("{id:?} is not a mesh pipeline"),
        }
    }

    /// Puts `made` in recipe `index`'s place. The old pipeline or set is destroyed once the
    /// work submitted before it was replaced has completed.
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    pub(crate) fn replace(&mut self, index: usize, made: Made) {
        self.list[index] = made;
    }

    /// Records the SPIR-V `module` reloaded to, for pipelines created after the reload.
    #[cfg_attr(not(feature = "hot-reload"), allow(dead_code))]
    pub(crate) fn reloaded(&mut self, module: &'static ShaderModule, code: Arc<Code>) {
        match self.code.iter_mut().find(|(m, _)| ptr::eq(*m, module)) {
            Some((_, c)) => *c = code,
            None => self.code.push((module, code)),
        }
    }

    /// The SPIR-V `module` last reloaded to; none while it runs the build's.
    pub(crate) fn code_of(&self, module: &'static ShaderModule) -> Option<Arc<Code>> {
        self.code
            .iter()
            .find(|(m, _)| ptr::eq(*m, module))
            .map(|(_, c)| Arc::clone(c))
    }
}
