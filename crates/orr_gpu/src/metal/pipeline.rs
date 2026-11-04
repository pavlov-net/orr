use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Archive, MTL4Compiler, MTL4CompilerTaskOptions, MTL4ComputePipelineDescriptor,
    MTL4FunctionDescriptor, MTL4LibraryFunctionDescriptor, MTL4SpecializedFunctionDescriptor,
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunction, MTLFunctionConstantValues,
    MTLLibrary,
};

use super::device::{Device, Shared, metal_error};
use super::{Obj, lock};
use crate::{EntryPoint, Error, ShaderModule, SpecData};

/// One shader stage of a pipeline: the module's Metal library, the entry point and
/// specialization data. The SPIR-V fields mirror the Vulkan backend's and go unused.
#[derive(Clone, Copy, Debug)]
pub struct ShaderStage<'a> {
    pub spirv: &'a [u32],
    /// The Metal library of the compute, task and mesh entries: the module's, or a hot
    /// reload's.
    pub metallib: &'a [u8],
    /// The Metal library of the pixel entries.
    pub metallib_fragment: &'a [u8],
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
            metallib: module.metallib,
            metallib_fragment: module.metallib_fragment,
            omm: module.omm,
            inline_root_bytes: module.inline_root_bytes,
            debug_info: module.debug_info,
            entry,
            spec,
            module,
        })
    }

    /// The stage's library for pixel entries when `fragment`, else for the others: the
    /// module's own, named by the module's hash, or a reload's, named by its contents.
    fn library(&self, fragment: bool) -> Library<'a> {
        let (bytes, own) = if fragment {
            (self.metallib_fragment, self.module.metallib_fragment)
        } else {
            (self.metallib, self.module.metallib)
        };
        if std::ptr::eq(bytes, own) {
            return Library::Module(self.module.hash, own);
        }
        let digest = blake3::hash(bytes);
        Library::Reload(
            u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap()),
            bytes,
        )
    }

    /// The Metal function descriptor of the entry: its name in the module's library, the
    /// fragment library for pixel entries, specialized when the stage has constants.
    pub(super) fn function(
        &self,
        shared: &Shared,
    ) -> Result<Retained<MTL4FunctionDescriptor>, Error> {
        let name = self.entry.metal_name.to_str().unwrap_or("");
        if name.is_empty() {
            return Err(Error::Shader(format!(
                "{} entry {:?} has no Metal function",
                self.module.name, self.entry.name
            )));
        }
        let fragment = self.entry.stage == crate::Stage::Pixel;
        let library = shared.library(self.module.name, fragment, self.library(fragment))?;
        let f = MTL4LibraryFunctionDescriptor::new();
        f.setName(Some(&NSString::from_str(name)));
        f.setLibrary(Some(&library));
        if self.spec.entries.is_empty() {
            return Ok(Retained::into_super(f));
        }
        let values = self.constant_values(&library, name)?;
        let specialized = MTL4SpecializedFunctionDescriptor::new();
        specialized.setFunctionDescriptor(Some(&f));
        specialized.setConstantValues(Some(&values));
        Ok(Retained::into_super(specialized))
    }

    /// Adds what tells this stage apart from others to a pipeline cache key: the module's
    /// code, the entry and the specialization data.
    pub(super) fn key(&self, hasher: &mut blake3::Hasher) {
        let fragment = self.entry.stage == crate::Stage::Pixel;
        hasher.update(&self.library(fragment).hash().to_le_bytes());
        hasher.update(self.entry.metal_name.to_bytes_with_nul());
        hasher.update(&(self.spec.bytes.len() as u64).to_le_bytes());
        hasher.update(self.spec.bytes);
        for e in self.spec.entries {
            for v in [e.constant_id, e.offset, e.size] {
                hasher.update(&v.to_le_bytes());
            }
        }
    }

    /// The stage's specialization constants as Metal function constants: Slang gives each
    /// the index of its `constant_id`. Values are laid out as for Vulkan, so a bool is four
    /// bytes here and one byte in Metal; constants the function lacks are skipped.
    fn constant_values(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        name: &str,
    ) -> Result<Retained<MTLFunctionConstantValues>, Error> {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| {
                Error::Shader(format!("{}: no Metal function {name}", self.module.name))
            })?;
        let dictionary = function.functionConstantsDictionary();
        let types: HashMap<usize, MTLDataType> = dictionary
            .allValues()
            .iter()
            .map(|c| (c.index(), c.r#type()))
            .collect();
        let values = MTLFunctionConstantValues::new();
        for e in self.spec.entries {
            let Some(&ty) = types.get(&(e.constant_id as usize)) else {
                continue;
            };
            let bytes = &self.spec.bytes[e.offset as usize..(e.offset + e.size) as usize];
            let word = match bytes {
                &[a, b, c, d] => u32::from_le_bytes([a, b, c, d]),
                _ => {
                    return Err(Error::Shader(format!(
                        "{}: constant {} is {} bytes, not 4",
                        self.module.name, e.constant_id, e.size
                    )));
                }
            };
            let flag = u8::from(word != 0);
            let value = match ty {
                MTLDataType::Bool => NonNull::from(&flag).cast(),
                MTLDataType::UInt | MTLDataType::Int | MTLDataType::Float => {
                    NonNull::from(&word).cast()
                }
                _ => {
                    return Err(Error::Shader(format!(
                        "{}: constant {} has Metal type {ty:?}",
                        self.module.name, e.constant_id
                    )));
                }
            };
            // SAFETY: `value` points at a value of `ty`, which Metal copies.
            unsafe { values.setConstantValue_type_atIndex(value, ty, e.constant_id as usize) };
        }
        Ok(values)
    }
}

impl Shared {
    /// The compute pipeline state of `function`, whose threadgroups are `workgroup`.
    pub(super) fn compute_state(
        &self,
        function: &MTL4FunctionDescriptor,
        workgroup: [u32; 3],
        name: &str,
    ) -> Result<Obj<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        let desc = compute_descriptor(function, workgroup, name);
        Ok(Obj(self.compile_compute(&desc, None, name)?))
    }

    /// Compiles the compute pipeline `desc`, named `name` in errors.
    fn compile_compute(
        &self,
        desc: &MTL4ComputePipelineDescriptor,
        options: Option<&MTL4CompilerTaskOptions>,
        name: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, Error> {
        self.compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(desc, options)
            .map_err(|e| {
                let mut e = metal_error("newComputePipelineState", &e);
                if let Error::Metal { message, .. } = &mut e {
                    *message = format!("{name}: {message}");
                }
                e
            })
    }

    /// The Metal library `library` of the module `module`, of its pixel entries when
    /// `fragment`. Loaded once per hash and kind.
    fn library(
        &self,
        module: &str,
        fragment: bool,
        library: Library<'_>,
    ) -> Result<Obj<ProtocolObject<dyn MTLLibrary>>, Error> {
        let key = (library.hash(), fragment);
        let mut libraries = lock(&self.libraries);
        if let Some(l) = libraries.get(&key) {
            return Ok(l.clone());
        }
        let (Library::Module(_, bytes) | Library::Reload(_, bytes)) = library;
        if bytes.is_empty() {
            return Err(Error::Shader(format!(
                "{module} has no Metal library: build it for an Apple target with the MSL check on"
            )));
        }
        let data = match library {
            Library::Module(_, bytes) => dispatch2::DispatchData::from_static_bytes(bytes),
            Library::Reload(_, bytes) => dispatch2::DispatchData::from_bytes(bytes),
        };
        let library = self
            .device
            .newLibraryWithData_error(&data)
            .map_err(|e| metal_error("newLibraryWithData", &e))?;
        library.setLabel(Some(&NSString::from_str(module)));
        let library = Obj(library);
        libraries.insert(key, library.clone());
        Ok(library)
    }
}

/// A stage's Metal library and the hash that names it.
#[derive(Clone, Copy)]
enum Library<'a> {
    /// The module's own, embedded in the binary.
    Module(u64, &'static [u8]),
    /// A hot reload's.
    Reload(u64, &'a [u8]),
}

impl Library<'_> {
    fn hash(self) -> u64 {
        match self {
            Library::Module(hash, _) | Library::Reload(hash, _) => hash,
        }
    }
}

/// The descriptor of a compute pipeline over `function` with threadgroups of `workgroup`.
fn compute_descriptor(
    function: &MTL4FunctionDescriptor,
    workgroup: [u32; 3],
    name: &str,
) -> Retained<MTL4ComputePipelineDescriptor> {
    let desc = MTL4ComputePipelineDescriptor::new();
    desc.setLabel(Some(&NSString::from_str(name)));
    desc.setComputeFunctionDescriptor(Some(function));
    desc.setRequiredThreadsPerThreadgroup(super::command::size(workgroup));
    desc
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
        let desc = compute_descriptor(&function, s.entry.workgroup, name);
        let mut key = blake3::Hasher::new();
        key.update(b"compute");
        s.key(&mut key);
        for v in s.entry.workgroup {
            key.update(&v.to_le_bytes());
        }
        let raw = self.shared.cached_compute(
            key.finalize(),
            self.cached_only,
            |archive| {
                archive
                    .newComputePipelineStateWithDescriptor_error(&desc)
                    .ok()
            },
            |options| self.shared.compile_compute(&desc, options, name),
            &desc,
        )?;
        Ok(ComputePipeline {
            raw,
            inline_root_bytes: s.inline_root_bytes,
            workgroup: s.entry.workgroup,
        })
    }
}
