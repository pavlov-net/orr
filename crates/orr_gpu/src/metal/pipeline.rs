use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4ComputePipelineDescriptor, MTL4FunctionDescriptor,
    MTL4LibraryFunctionDescriptor, MTLComputePipelineState, MTLDevice, MTLLibrary,
};

use super::device::{Device, Shared, metal_error};
use super::{Obj, lock};
use crate::{EntryPoint, Error, ShaderModule, SpecData};

/// One shader stage of a pipeline: the module's Metal library, the entry point and
/// specialization data. The SPIR-V fields mirror the Vulkan backend's and go unused.
#[derive(Clone, Copy, Debug)]
pub struct ShaderStage<'a> {
    pub spirv: &'a [u32],
    pub omm: bool,
    /// The size of the root the kernel holds inline; 0 for a root in GPU memory.
    pub inline_root_bytes: u32,
    pub debug_info: &'a str,
    pub entry: &'a EntryPoint,
    pub spec: SpecData<'a>,
    pub(super) module: &'a ShaderModule,
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
            inline_root_bytes: module.inline_root_bytes,
            debug_info: module.debug_info,
            entry,
            spec,
            module,
        })
    }

    /// The Metal function descriptor of the entry: its name in the module's library, the
    /// fragment library for pixel entries.
    pub(super) fn function(
        &self,
        shared: &Shared,
    ) -> Result<objc2::rc::Retained<MTL4LibraryFunctionDescriptor>, Error> {
        if !self.spec.bytes.is_empty() {
            return Err(Error::NotImplemented("specialization constants"));
        }
        let name = self.entry.metal_name.to_str().unwrap_or("");
        if name.is_empty() {
            return Err(Error::Shader(format!(
                "{} entry {:?} has no Metal function",
                self.module.name, self.entry.name
            )));
        }
        let fragment = self.entry.stage == crate::Stage::Pixel;
        let library = shared.library(self.module, fragment)?;
        let f = MTL4LibraryFunctionDescriptor::new();
        f.setName(Some(&NSString::from_str(name)));
        f.setLibrary(Some(&library));
        Ok(f)
    }
}

impl Shared {
    /// `module`'s Metal library: the pixel entries' when `fragment`, the others' otherwise.
    /// Loaded once per module.
    pub(super) fn library(
        &self,
        module: &ShaderModule,
        fragment: bool,
    ) -> Result<Obj<ProtocolObject<dyn MTLLibrary>>, Error> {
        let mut libraries = lock(&self.libraries);
        if let Some(l) = libraries.get(&(module.hash, fragment)) {
            return Ok(l.clone());
        }
        let bytes = if fragment {
            module.metallib_fragment
        } else {
            module.metallib
        };
        if bytes.is_empty() {
            return Err(Error::Shader(format!(
                "{} has no Metal library: build it for an Apple target with the MSL check on",
                module.name
            )));
        }
        let data = dispatch2::DispatchData::from_static_bytes(bytes);
        let library = self
            .device
            .newLibraryWithData_error(&data)
            .map_err(|e| metal_error("newLibraryWithData", &e))?;
        library.setLabel(Some(&NSString::from_str(module.name)));
        let library = Obj(library);
        libraries.insert((module.hash, fragment), library.clone());
        Ok(library)
    }
}

/// A compute pipeline. Command lists keep the pipelines they bind until their submit
/// completes, so dropping it is safe at any time.
pub struct ComputePipeline {
    pub(super) raw: Obj<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The size of the root the kernel holds inline; 0 for a root in GPU memory.
    pub(super) inline_root_bytes: u32,
    workgroup: [u32; 3],
}

impl ComputePipeline {
    /// The kernel's workgroup size.
    pub fn workgroup(&self) -> [u32; 3] {
        self.workgroup
    }
}

impl Device {
    /// The compute pipeline of `s`, labeled `name`.
    pub fn create_compute_pipeline(
        &self,
        s: &ShaderStage<'_>,
        name: &str,
    ) -> Result<ComputePipeline, Error> {
        if s.entry.stage != crate::Stage::Compute {
            return Err(Error::Shader(format!(
                "{name}: {:?} is a {:?} entry",
                s.entry.name, s.entry.stage
            )));
        }
        let function = s.function(&self.shared)?;
        let desc = MTL4ComputePipelineDescriptor::new();
        desc.setLabel(Some(&NSString::from_str(name)));
        let f: &MTL4FunctionDescriptor = &function;
        desc.setComputeFunctionDescriptor(Some(f));
        desc.setRequiredThreadsPerThreadgroup(super::command::size(s.entry.workgroup));
        let raw = self
            .shared
            .compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&desc, None)
            .map_err(|e| metal_error("newComputePipelineState", &e))?;
        Ok(ComputePipeline {
            raw: Obj(raw),
            inline_root_bytes: s.inline_root_bytes,
            workgroup: s.entry.workgroup,
        })
    }
}
