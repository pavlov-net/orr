//! Device-generated commands: execution sets of compute or mesh pipelines, run once per
//! GPU-written sequence that picks a pipeline, pushes its root and dispatches or draws.

use std::alloc::Layout;
use std::sync::Arc;

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::command::CommandList;
use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use super::memory::MemoryKind;
use super::mesh::MeshPipelineDesc;
use super::pipeline::ShaderStage;
use crate::{Error, GpuPtr};

/// One sequence of [`CommandList::execute_generated`]: an index into the execution set, the
/// root pointer pushed for that pipeline and the workgroup counts of its dispatch (task
/// workgroups for mesh sets, mesh workgroups without a task stage). Kernels write sequences
/// with `store` of the `sequence` Slang module.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Sequence {
    pub pipeline: u32,
    /// The root's address, low word first.
    pub root: [u32; 2],
    pub groups: [u32; 3],
}

impl Sequence {
    pub fn new<R>(pipeline: u32, root: GpuPtr<R>, groups: [u32; 3]) -> Self {
        let a = root.address().0;
        Self {
            pipeline,
            root: [a as u32, (a >> 32) as u32],
            groups,
        }
    }
}

/// Byte offsets of the tokens in a [`Sequence`], matching the `sequence` Slang module.
const ROOT_OFFSET: u32 = 4;
const GROUPS_OFFSET: u32 = 12;
const STRIDE: u32 = size_of::<Sequence>() as u32;

/// The kind of pipelines an execution set holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SetKind {
    Compute,
    Mesh,
}

/// Pipelines of one kind that GPU-written sequences pick by index. Dropping it destroys it
/// and its pipelines once everything submitted before the drop has completed.
pub struct ExecutionSet {
    raw: vk::IndirectExecutionSetEXT,
    /// The commands layout for explicit preprocessing; null for mesh sets.
    explicit: vk::IndirectCommandsLayoutEXT,
    /// The commands layout that preprocesses as it executes.
    implicit: vk::IndirectCommandsLayoutEXT,
    pipelines: Vec<vk::Pipeline>,
    stages: vk::ShaderStageFlags,
    kind: SetKind,
    shared: Arc<Shared>,
}

impl ExecutionSet {
    pub fn kind(&self) -> SetKind {
        self.kind
    }

    /// Pipelines in the set.
    pub fn len(&self) -> u32 {
        self.pipelines.len() as u32
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }

    /// Preprocess memory for up to `max` sequences of this set, explicit or not: allocate it
    /// from [`crate::DeviceArenas`], the only memory with the preprocess usage.
    pub fn preprocess_layout(&self, max: u32) -> Layout {
        let (size, align) = [self.explicit, self.implicit]
            .into_iter()
            .filter(|l| *l != vk::IndirectCommandsLayoutEXT::null())
            .map(|l| {
                let r = self.requirements(l, max);
                (r.size, r.alignment)
            })
            .fold((0, 1), |(s, a), (rs, ra)| (s.max(rs), a.max(ra)));
        Layout::from_size_align(size.max(16) as usize, align.max(16) as usize)
            .expect("preprocess size and alignment")
    }

    fn requirements(
        &self,
        layout: vk::IndirectCommandsLayoutEXT,
        max: u32,
    ) -> vk::MemoryRequirements {
        let info = vk::GeneratedCommandsMemoryRequirementsInfoEXT::default()
            .indirect_execution_set(self.raw)
            .indirect_commands_layout(layout)
            .max_sequence_count(max);
        let mut out = vk::MemoryRequirements2::default();
        // SAFETY: the set and the layout are live objects of this device.
        unsafe {
            self.shared
                .generated_fn
                .as_ref()
                .expect("an execution set exists")
                .get_generated_commands_memory_requirements(&info, &mut out);
        }
        out.memory_requirements
    }
}

impl Drop for ExecutionSet {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::ExecutionSet {
            set: self.raw,
            layouts: [self.explicit, self.implicit]
                .into_iter()
                .filter(|l| *l != vk::IndirectCommandsLayoutEXT::null())
                .collect(),
            pipelines: std::mem::take(&mut self.pipelines),
        });
    }
}

/// One run of an execution set: `min(*count, max)` sequences read from `sequences`, an
/// allocation with room for `max`, with preprocess memory from
/// [`ExecutionSet::preprocess_layout`] for `max`. Each execute takes preprocess memory no
/// earlier execute used.
#[derive(Clone, Copy)]
pub struct Generated<'a> {
    pub set: &'a ExecutionSet,
    pub sequences: GpuPtr<Sequence>,
    pub count: GpuPtr<u32>,
    pub max: u32,
    pub preprocess: GpuPtr<u8>,
}

impl Shared {
    /// Indirect-bindable pipelines, one per item; on failure the ones already created are
    /// destroyed.
    fn indirect_pipelines<T>(
        &self,
        items: &[T],
        name: &str,
        create: impl Fn(&T, vk::PipelineCreateFlags2, &str) -> Result<vk::Pipeline, Error>,
    ) -> Result<Vec<vk::Pipeline>, Error> {
        let mut pipelines = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let flags = vk::PipelineCreateFlags2::INDIRECT_BINDABLE_EXT;
            match create(item, flags, &format!("{name} {i}")) {
                Ok(p) => pipelines.push(p),
                Err(e) => {
                    self.defer_after_submitted(Item::ExecutionSet {
                        set: vk::IndirectExecutionSetEXT::null(),
                        layouts: Vec::new(),
                        pipelines,
                    });
                    return Err(e);
                }
            }
        }
        Ok(pipelines)
    }

    fn check_set(&self, count: usize, name: &str) -> Result<(), Error> {
        if !self.caps.generated_commands {
            return Err(Error::Unsupported {
                device: self.caps.device_name.clone(),
                missing: "device-generated commands (VK_EXT_device_generated_commands)".into(),
            });
        }
        let max = self.caps.limits.max_generated_pipelines;
        if count == 0 || count > max as usize {
            return Err(Error::Invalid(format!(
                "execution set {name}: {count} pipelines, the device takes 1 to {max}"
            )));
        }
        Ok(())
    }

    /// The set over `pipelines`, which it owns from here on, success or not.
    fn create_set(
        self: &Arc<Self>,
        kind: SetKind,
        stages: vk::ShaderStageFlags,
        pipelines: Vec<vk::Pipeline>,
        name: &str,
    ) -> Result<ExecutionSet, Error> {
        let f = self
            .generated_fn
            .as_ref()
            .expect("Caps::generated_commands");
        let mut set = ExecutionSet {
            raw: vk::IndirectExecutionSetEXT::null(),
            explicit: vk::IndirectCommandsLayoutEXT::null(),
            implicit: vk::IndirectCommandsLayoutEXT::null(),
            pipelines,
            stages,
            kind,
            shared: Arc::clone(self),
        };
        let info = vk::IndirectExecutionSetPipelineInfoEXT::default()
            .initial_pipeline(set.pipelines[0])
            .max_pipeline_count(set.len());
        let create = vk::IndirectExecutionSetCreateInfoEXT::default()
            .ty(vk::IndirectExecutionSetInfoTypeEXT::PIPELINES)
            .info(vk::IndirectExecutionSetInfoEXT {
                p_pipeline_info: &info,
            });
        // SAFETY: valid create info; every pipeline was created indirect-bindable.
        set.raw = unsafe { f.create_indirect_execution_set(&create, None) }
            .map_err(Error::vk("vkCreateIndirectExecutionSetEXT"))?;
        name_object(&self.debug_fn, set.raw, name);
        let writes: Vec<_> = set.pipelines[1..]
            .iter()
            .zip(1..)
            .map(|(&p, i)| {
                vk::WriteIndirectExecutionSetPipelineEXT::default()
                    .index(i)
                    .pipeline(p)
            })
            .collect();
        if !writes.is_empty() {
            // SAFETY: the set is new, so no submitted work uses it; the indices are distinct
            // and below the set's size.
            unsafe { f.update_indirect_execution_set_pipeline(set.raw, &writes) };
        }
        if kind == SetKind::Compute {
            set.explicit = self.commands_layout(kind, stages, true, name)?;
        }
        set.implicit = self.commands_layout(kind, stages, false, name)?;

        let bits = set.requirements(set.implicit, 1).memory_type_bits;
        let device_local = self.memory.of(MemoryKind::DeviceLocal);
        if bits & (1 << device_local) == 0 {
            return Err(Error::Unsupported {
                device: self.caps.device_name.clone(),
                missing: format!(
                    "preprocess memory in the DeviceLocal type {device_local} (types {bits:#x})"
                ),
            });
        }
        Ok(set)
    }

    /// The sequence layout: execution-set index, 8-byte push data at offset 0, then the
    /// dispatch or draw.
    fn commands_layout(
        &self,
        kind: SetKind,
        stages: vk::ShaderStageFlags,
        explicit: bool,
        name: &str,
    ) -> Result<vk::IndirectCommandsLayoutEXT, Error> {
        let set_token = vk::IndirectCommandsExecutionSetTokenEXT::default()
            .ty(vk::IndirectExecutionSetInfoTypeEXT::PIPELINES)
            .shader_stages(stages);
        let push = vk::IndirectCommandsPushConstantTokenEXT::default().update_range(
            vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL,
                offset: 0,
                size: 8,
            },
        );
        let action = match kind {
            SetKind::Compute => vk::IndirectCommandsTokenTypeEXT::DISPATCH,
            SetKind::Mesh => vk::IndirectCommandsTokenTypeEXT::DRAW_MESH_TASKS,
        };
        let tokens = [
            vk::IndirectCommandsLayoutTokenEXT::default()
                .ty(vk::IndirectCommandsTokenTypeEXT::EXECUTION_SET)
                .data(vk::IndirectCommandsTokenDataEXT {
                    p_execution_set: &set_token,
                })
                .offset(0),
            vk::IndirectCommandsLayoutTokenEXT::default()
                .ty(vk::IndirectCommandsTokenTypeEXT::PUSH_DATA)
                .data(vk::IndirectCommandsTokenDataEXT {
                    p_push_constant: &push,
                })
                .offset(ROOT_OFFSET),
            vk::IndirectCommandsLayoutTokenEXT::default()
                .ty(action)
                .offset(GROUPS_OFFSET),
        ];
        let flags = if explicit {
            vk::IndirectCommandsLayoutUsageFlagsEXT::EXPLICIT_PREPROCESS
        } else {
            vk::IndirectCommandsLayoutUsageFlagsEXT::empty()
        };
        let info = vk::IndirectCommandsLayoutCreateInfoEXT::default()
            .flags(flags)
            .shader_stages(stages)
            .indirect_stride(STRIDE)
            .tokens(&tokens);
        let f = self
            .generated_fn
            .as_ref()
            .expect("Caps::generated_commands");
        // SAFETY: valid create info; descriptor heaps leave the pipeline layout null.
        let layout = unsafe { f.create_indirect_commands_layout(&info, None) }
            .map_err(Error::vk("vkCreateIndirectCommandsLayoutEXT"))?;
        name_object(&self.debug_fn, layout, name);
        Ok(layout)
    }
}

impl Device {
    /// An execution set of compute pipelines, one per stage, picked by their index in
    /// `stages`. Needs `Caps::generated_commands`; at most `Limits::max_generated_pipelines`.
    pub fn create_compute_set(
        &self,
        stages: &[ShaderStage<'_>],
        name: &str,
    ) -> Result<ExecutionSet, Error> {
        let shared = &self.shared;
        shared.check_set(stages.len(), name)?;
        let pipelines = shared.indirect_pipelines(stages, name, |s, flags, name| {
            shared.create_compute(s, flags, name)
        })?;
        let stages = vk::ShaderStageFlags::COMPUTE;
        shared.create_set(SetKind::Compute, stages, pipelines, name)
    }

    /// An execution set of mesh pipelines, one per description, picked by their index in
    /// `descs`. Every description has the same stages, targets, raster state and samples,
    /// as the pipelines of one set must; only the shaders and their specialization differ.
    /// Needs `Caps::generated_commands`; at most `Limits::max_generated_pipelines`.
    pub fn create_mesh_set(
        &self,
        descs: &[MeshPipelineDesc<'_>],
        name: &str,
    ) -> Result<ExecutionSet, Error> {
        let shared = &self.shared;
        shared.check_set(descs.len(), name)?;
        let first = &descs[0];
        let same = |d: &MeshPipelineDesc<'_>| {
            d.task.is_some() == first.task.is_some()
                && d.pixel.is_some() == first.pixel.is_some()
                && d.colors == first.colors
                && d.depth == first.depth
                && d.raster == first.raster
                && d.samples == first.samples
        };
        if let Some(i) = descs.iter().position(|d| !same(d)) {
            return Err(Error::Invalid(format!(
                "execution set {name}: pipeline {i} differs from pipeline 0 in stages, targets, raster state or samples"
            )));
        }
        let pipelines = shared.indirect_pipelines(descs, name, |d, flags, name| {
            shared.create_mesh(d, flags, name)
        })?;
        let mut stages = vk::ShaderStageFlags::MESH_EXT;
        if first.task.is_some() {
            stages |= vk::ShaderStageFlags::TASK_EXT;
        }
        if first.pixel.is_some() {
            stages |= vk::ShaderStageFlags::FRAGMENT;
        }
        shared.create_set(SetKind::Mesh, stages, pipelines, name)
    }
}

impl CommandList<'_> {
    fn generated_info(
        &self,
        g: &Generated<'_>,
        explicit: bool,
    ) -> vk::GeneratedCommandsInfoEXT<'static> {
        assert!(
            self.queue().shaders,
            "generated commands run on a graphics or compute queue"
        );
        let set = g.set;
        let shared = &self.queue().shared;
        assert!(
            Arc::ptr_eq(&set.shared, shared),
            "execution set of another device"
        );
        let limit = shared.caps.limits.max_generated_sequences;
        assert!(
            (1..=limit).contains(&g.max),
            "{} generated sequences, the device takes 1 to {limit}",
            g.max
        );
        let bytes = u64::from(g.max) * u64::from(STRIDE);
        let layout = if explicit { set.explicit } else { set.implicit };
        let need = set.requirements(layout, g.max);
        // The sequences, the count and the preprocess memory each lie inside one allocation.
        shared.locate(g.sequences.address(), bytes);
        shared.locate(g.count.address(), 4);
        shared.locate(g.preprocess.address(), need.size);
        assert!(
            g.preprocess
                .address()
                .0
                .is_multiple_of(need.alignment.max(1)),
            "preprocess memory at {:#x}, not a multiple of {}",
            g.preprocess.address().0,
            need.alignment
        );
        vk::GeneratedCommandsInfoEXT::default()
            .shader_stages(set.stages)
            .indirect_execution_set(set.raw)
            .indirect_commands_layout(layout)
            .indirect_address(g.sequences.address().0)
            .indirect_address_size(bytes)
            .preprocess_address(g.preprocess.address().0)
            .preprocess_size(need.size)
            .max_sequence_count(g.max)
            .sequence_count_address(g.count.address().0)
    }

    /// Binds the set's first pipeline, the state preprocessing and execution read.
    fn bind_initial(&mut self, set: &ExecutionSet) {
        let point = match set.kind {
            SetKind::Compute => vk::PipelineBindPoint::COMPUTE,
            SetKind::Mesh => vk::PipelineBindPoint::GRAPHICS,
        };
        self.bind(point, set.pipelines[0], 0);
    }

    /// Turns `g`'s sequences into commands ahead of `execute_generated(g, true)`, which may
    /// come much later in this list or in a later list of the same queue family. Compute
    /// sets only, outside rendering. Order the writes of the sequences and the count with a
    /// barrier to [`crate::Stages::PREPROCESS`] carrying [`crate::Hazards::DRAW_ARGUMENTS`],
    /// and order this before the execute with a barrier from [`crate::Stages::PREPROCESS`]
    /// carrying [`crate::Hazards::DRAW_ARGUMENTS`].
    pub fn preprocess_generated(&mut self, g: &Generated<'_>) {
        assert!(
            g.set.kind == SetKind::Compute,
            "only compute sets preprocess ahead of their execute"
        );
        assert!(
            !self.in_rendering(),
            "preprocess_generated inside rendering"
        );
        let info = self.generated_info(g, true);
        self.bind_initial(g.set);
        let f = self.queue().shared.generated_fn.as_ref().unwrap();
        let cmd = self.cmd();
        // SAFETY: recording with the set's first pipeline, the heaps and push data bound in
        // this list, which is also the state list.
        unsafe { f.cmd_preprocess_generated_commands(cmd, &info, cmd) };
        self.pipeline_lost();
    }

    /// Runs `min(*count, max)` of `g`'s sequences, each binding its pipeline, pushing its
    /// root and dispatching (compute sets, outside rendering) or drawing (mesh sets, inside
    /// rendering). With `preprocessed` the sequences were turned into commands by an earlier
    /// [`CommandList::preprocess_generated`] of the same `g` whose inputs have not changed
    /// since; mesh sets always preprocess here. Order the writes of the sequences and the
    /// count with a barrier carrying [`crate::Hazards::DRAW_ARGUMENTS`]. Bound state is
    /// rebound by the next command.
    pub fn execute_generated(&mut self, g: &Generated<'_>, preprocessed: bool) {
        assert!(
            !(preprocessed && g.set.kind == SetKind::Mesh),
            "mesh sets preprocess as they execute"
        );
        assert_eq!(
            self.in_rendering(),
            g.set.kind == SetKind::Mesh,
            "mesh sets execute inside rendering, compute sets outside"
        );
        let info = self.generated_info(g, preprocessed);
        self.bind_initial(g.set);
        let f = self.queue().shared.generated_fn.as_ref().unwrap();
        // SAFETY: recording with the set's first pipeline, the heaps and push data bound as
        // at preprocessing.
        unsafe { f.cmd_execute_generated_commands(self.cmd(), preprocessed, &info) };
        self.pipeline_lost();
    }
}
