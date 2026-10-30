//! The Slang library in process: global sessions, sessions, modules and linked programs.
//!
//! bindgen generates no vtables for interfaces that derive from another, so the methods the
//! build calls are listed here in `slang.h` declaration order. Unused slots are padding.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::Arc;

use super::sys;

type Result_ = sys::SlangResult;

#[repr(C)]
struct UnknownVtbl {
    query_interface: usize,
    add_ref: unsafe extern "system" fn(*mut c_void) -> u32,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}

#[repr(C)]
struct BlobVtbl {
    unknown: UnknownVtbl,
    get_buffer_pointer: unsafe extern "system" fn(*mut c_void) -> *const c_void,
    get_buffer_size: unsafe extern "system" fn(*mut c_void) -> usize,
}

#[repr(C)]
struct GlobalSessionVtbl {
    unknown: UnknownVtbl,
    create_session: unsafe extern "system" fn(
        *mut c_void,
        *const sys::slang_SessionDesc,
        *mut *mut c_void,
    ) -> Result_,
    // findProfile
    _unused0: usize,
    set_downstream_compiler_path:
        unsafe extern "system" fn(*mut c_void, sys::SlangPassThrough, *const c_char),
    // setDownstreamCompilerPrelude .. setSPIRVCoreGrammar
    _unused1: [usize; 21],
    parse_command_line_arguments: unsafe extern "system" fn(
        *mut c_void,
        c_int,
        *const *const c_char,
        *mut sys::slang_SessionDesc,
        *mut *mut c_void,
    ) -> Result_,
}

#[repr(C)]
struct SessionVtbl {
    unknown: UnknownVtbl,
    // getGlobalSession, loadModule, loadModuleFromSource
    _unused0: [usize; 3],
    create_composite_component_type: unsafe extern "system" fn(
        *mut c_void,
        *const *mut c_void,
        sys::SlangInt,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> Result_,
    // specializeType .. loadModuleFromIRBlob
    _unused1: [usize; 10],
    get_loaded_module_count: unsafe extern "system" fn(*mut c_void) -> sys::SlangInt,
    get_loaded_module: unsafe extern "system" fn(*mut c_void, sys::SlangInt) -> *mut c_void,
    // isBinaryModuleUpToDate
    _unused2: usize,
    load_module_from_source_string: unsafe extern "system" fn(
        *mut c_void,
        *const c_char,
        *const c_char,
        *const c_char,
        *mut *mut c_void,
    ) -> *mut c_void,
    // getDynamicObjectRTTIBytes, loadModuleInfoFromIRBlob
    _unused3: [usize; 2],
    get_decl_source_location: unsafe extern "system" fn(
        *mut c_void,
        *mut sys::SlangReflectionDecl,
        *mut sys::slang_SourceLocation,
    ) -> Result_,
}

#[repr(C)]
struct ComponentTypeVtbl {
    unknown: UnknownVtbl,
    // getSession
    _unused0: usize,
    get_layout:
        unsafe extern "system" fn(*mut c_void, sys::SlangInt, *mut *mut c_void) -> *mut c_void,
    // getSpecializationParamCount .. specialize
    _unused1: [usize; 5],
    link: unsafe extern "system" fn(*mut c_void, *mut *mut c_void, *mut *mut c_void) -> Result_,
    // getEntryPointHostCallable, renameEntryPoint, linkWithOptions
    _unused2: [usize; 3],
    get_target_code: unsafe extern "system" fn(
        *mut c_void,
        sys::SlangInt,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> Result_,
    // getTargetMetadata, getEntryPointMetadata
    _unused3: [usize; 2],
}

#[repr(C)]
struct EntryPointVtbl {
    component: ComponentTypeVtbl,
    get_function_reflection:
        unsafe extern "system" fn(*mut c_void) -> *mut sys::SlangReflectionFunction,
}

#[repr(C)]
struct ModuleVtbl {
    component: ComponentTypeVtbl,
    // findEntryPointByName
    _unused0: usize,
    get_defined_entry_point_count: unsafe extern "system" fn(*mut c_void) -> i32,
    get_defined_entry_point:
        unsafe extern "system" fn(*mut c_void, i32, *mut *mut c_void) -> Result_,
    // serialize, writeToFile, getName, getFilePath, getUniqueIdentity, findAndCheckEntryPoint
    _unused1: [usize; 6],
    get_dependency_file_count: unsafe extern "system" fn(*mut c_void) -> i32,
    get_dependency_file_path: unsafe extern "system" fn(*mut c_void, i32) -> *const c_char,
    get_module_reflection: unsafe extern "system" fn(*mut c_void) -> *mut sys::SlangReflectionDecl,
}

/// An owned COM reference.
struct Com(NonNull<c_void>);

impl Com {
    /// Takes ownership of a reference returned through an out parameter.
    fn from_raw(p: *mut c_void) -> Option<Self> {
        NonNull::new(p).map(Self)
    }

    /// Adds a reference to an object returned without one.
    fn borrowed(p: *mut c_void) -> Option<Self> {
        let com = NonNull::new(p).map(Self)?;
        // SAFETY: every Slang interface starts with the ISlangUnknown vtable.
        unsafe { (com.vtbl::<UnknownVtbl>().add_ref)(com.as_ptr()) };
        Some(com)
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    /// # Safety
    /// `V` must be a prefix of the object's vtable.
    unsafe fn vtbl<V>(&self) -> &V {
        // SAFETY: a COM object's first word points at its vtable; the caller picks `V`.
        unsafe { &**self.0.as_ptr().cast::<*const V>() }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        // SAFETY: every Slang interface starts with the ISlangUnknown vtable.
        unsafe { (self.vtbl::<UnknownVtbl>().release)(self.as_ptr()) };
    }
}

fn blob_bytes(blob: &Com) -> &[u8] {
    // SAFETY: `blob` is an ISlangBlob, which owns its buffer for its lifetime.
    unsafe {
        let v = blob.vtbl::<BlobVtbl>();
        let ptr = (v.get_buffer_pointer)(blob.as_ptr());
        let len = (v.get_buffer_size)(blob.as_ptr());
        if ptr.is_null() || len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(ptr.cast(), len)
        }
    }
}

/// The text of an owned blob (diagnostics, names), which this releases.
pub(crate) fn blob_text(blob: *mut c_void) -> String {
    Com::from_raw(blob)
        .map(|b| {
            String::from_utf8_lossy(blob_bytes(&b))
                .trim_end_matches('\0')
                .trim()
                .to_owned()
        })
        .unwrap_or_default()
}

fn failed(r: Result_) -> bool {
    r < 0
}

unsafe fn c_str<'a>(p: *const c_char) -> Option<&'a str> {
    // SAFETY: the caller passes a null or nul-terminated string that outlives 'a.
    (!p.is_null())
        .then(|| unsafe { CStr::from_ptr(p) }.to_str().ok())
        .flatten()
}

/// The loaded library and its exported functions.
pub(crate) struct Library {
    pub(crate) api: sys::SlangLib,
    /// The directory holding the library and the downstream libraries it loads by name.
    dir: CString,
}

impl Library {
    pub(crate) fn load(path: &Path) -> Result<Arc<Self>, libloading::Error> {
        #[cfg(windows)]
        // SAFETY: loading the Slang library runs no initialisers with preconditions;
        // altered search path makes its sibling DLLs resolve from its own directory.
        let library: libloading::Library = unsafe {
            libloading::os::windows::Library::load_with_flags(
                path,
                libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
            )?
        }
        .into();
        #[cfg(not(windows))]
        // SAFETY: as above.
        let library = unsafe { libloading::Library::new(path)? };
        // SAFETY: the symbols are declared with the signatures of the release's slang.h.
        let api = unsafe { sys::SlangLib::from_library(library)? };
        let dir = path.parent().unwrap_or(Path::new("."));
        let dir = CString::new(dir.to_string_lossy().as_bytes()).expect("path has no nul");
        Ok(Arc::new(Self { api, dir }))
    }

    pub(crate) fn build_tag(&self) -> String {
        // SAFETY: returns a static string.
        unsafe { c_str(self.api.spGetBuildTagString()) }
            .unwrap_or_default()
            .to_owned()
    }
}

/// One global session. Not thread-safe; each build thread creates its own.
pub(crate) struct GlobalSession {
    com: Com,
    pub(crate) lib: Arc<Library>,
}

impl GlobalSession {
    pub(crate) fn new(lib: Arc<Library>) -> Result<Self, String> {
        let desc = sys::SlangGlobalSessionDesc {
            structureSize: size_of::<sys::SlangGlobalSessionDesc>() as u32,
            apiVersion: 0,
            ..Default::default()
        };
        let mut out: *mut c_void = ptr::null_mut();
        // SAFETY: desc is a valid SlangGlobalSessionDesc; out receives one reference.
        let r = unsafe {
            lib.api
                .slang_createGlobalSession2(&desc, (&raw mut out).cast())
        };
        let com = match Com::from_raw(out) {
            Some(com) if !failed(r) => com,
            _ => return Err(format!("slang_createGlobalSession2 failed: {r:#x}")),
        };
        // Slang loads its SPIR-V tools by file name, which can find another release's copy
        // on the search path; pin them to the release beside the library.
        use sys::SlangPassThrough as P;
        for tool in [
            P::SLANG_PASS_THROUGH_GLSLANG,
            P::SLANG_PASS_THROUGH_SPIRV_OPT,
            P::SLANG_PASS_THROUGH_SPIRV_DIS,
            P::SLANG_PASS_THROUGH_SPIRV_LINK,
        ] {
            // SAFETY: the vtable is IGlobalSession's; the path outlives the call.
            unsafe {
                (com.vtbl::<GlobalSessionVtbl>().set_downstream_compiler_path)(
                    com.as_ptr(),
                    tool,
                    lib.dir.as_ptr(),
                );
            }
        }
        Ok(Self { com, lib })
    }

    /// A session configured by slangc command-line options.
    pub(crate) fn session(&self, args: &[String]) -> Result<Session, String> {
        self.session_with(args, false)
    }

    /// A session whose targets emit every `export`ed function, entry point or not: what
    /// slangc does for a source target given no entry points.
    pub(crate) fn whole_program_session(&self, args: &[String]) -> Result<Session, String> {
        self.session_with(args, true)
    }

    fn session_with(&self, args: &[String], whole_program: bool) -> Result<Session, String> {
        let args: Vec<CString> = args
            .iter()
            .map(|a| CString::new(a.as_str()).expect("no nul in compiler options"))
            .collect();
        let argv: Vec<*const c_char> = args.iter().map(|a| a.as_ptr()).collect();
        // SAFETY: the vtable is IGlobalSession's; argv outlives both calls, and the aux
        // allocation keeps the parsed desc's data alive until createSession returns.
        unsafe {
            let v = self.com.vtbl::<GlobalSessionVtbl>();
            let mut desc = sys::slang_SessionDesc {
                structureSize: size_of::<sys::slang_SessionDesc>(),
                ..Default::default()
            };
            let mut aux = ptr::null_mut();
            let r = (v.parse_command_line_arguments)(
                self.com.as_ptr(),
                argv.len() as c_int,
                argv.as_ptr(),
                &mut desc,
                &mut aux,
            );
            let _aux = Com::from_raw(aux);
            if failed(r) {
                return Err(format!("Slang rejected the options {args:?}: {r:#x}"));
            }
            let mut targets = Vec::new();
            if whole_program && !desc.targets.is_null() {
                targets.extend_from_slice(std::slice::from_raw_parts(
                    desc.targets,
                    desc.targetCount as usize,
                ));
                for t in &mut targets {
                    t.flags |=
                        sys::SLANG_TARGET_FLAG_GENERATE_WHOLE_PROGRAM.0 as sys::SlangTargetFlags;
                }
                desc.targets = targets.as_ptr();
            }
            let mut out = ptr::null_mut();
            let r = (v.create_session)(self.com.as_ptr(), &desc, &mut out);
            match Com::from_raw(out) {
                Some(com) if !failed(r) => Ok(Session {
                    com,
                    lib: Arc::clone(&self.lib),
                }),
                _ => Err(format!("createSession failed: {r:#x}")),
            }
        }
    }
}

pub(crate) struct Session {
    com: Com,
    pub(crate) lib: Arc<Library>,
}

/// A module loaded into a session.
pub(crate) struct Module {
    com: Com,
}

/// A linked program: one module and all its entry points.
pub(crate) struct Program {
    com: Com,
}

impl Session {
    /// Loads a module from source. Errors carry Slang's diagnostics.
    pub(crate) fn load(&self, name: &str, path: &Path, source: &str) -> Result<Module, String> {
        let name = CString::new(name).expect("module names have no nul");
        let path_c = CString::new(path.to_string_lossy().as_bytes()).expect("path has no nul");
        let source = CString::new(source).map_err(|_| "source contains a nul byte".to_owned())?;
        let mut diag = ptr::null_mut();
        // SAFETY: the vtable is ISession's; the strings outlive the call.
        let module = unsafe {
            (self
                .com
                .vtbl::<SessionVtbl>()
                .load_module_from_source_string)(
                self.com.as_ptr(),
                name.as_ptr(),
                path_c.as_ptr(),
                source.as_ptr(),
                &mut diag,
            )
        };
        let text = blob_text(diag);
        // The session owns loaded modules; the build keeps its own reference.
        match Com::borrowed(module) {
            Some(com) => Ok(Module { com }),
            None => Err(text),
        }
    }

    /// Composes the module, the entry points it defines whose names `keep` accepts, and
    /// `extra` modules, and links the result.
    pub(crate) fn link(
        &self,
        module: &Module,
        extra: &[Module],
        keep: impl Fn(&str) -> bool,
    ) -> Result<Program, String> {
        let mut parts = vec![module.com.as_ptr()];
        parts.extend(extra.iter().map(|m| m.com.as_ptr()));
        let mut entries = Vec::new();
        // SAFETY: the vtables are IModule's, IEntryPoint's and ISession's; out parameters
        // receive owned references.
        unsafe {
            let mv = module.com.vtbl::<ModuleVtbl>();
            for i in 0..(mv.get_defined_entry_point_count)(module.com.as_ptr()) {
                let mut ep = ptr::null_mut();
                let r = (mv.get_defined_entry_point)(module.com.as_ptr(), i, &mut ep);
                let ep = Com::from_raw(ep)
                    .filter(|_| !failed(r))
                    .ok_or("getDefinedEntryPoint failed")?;
                let function = (ep.vtbl::<EntryPointVtbl>().get_function_reflection)(ep.as_ptr());
                let name = c_str(self.lib.api.spReflectionFunction_GetName(function)).unwrap_or("");
                if !keep(name) {
                    continue;
                }
                parts.push(ep.as_ptr());
                entries.push(ep);
            }
            let sv = self.com.vtbl::<SessionVtbl>();
            let mut composite = ptr::null_mut();
            let mut diag = ptr::null_mut();
            let r = (sv.create_composite_component_type)(
                self.com.as_ptr(),
                parts.as_ptr(),
                parts.len() as sys::SlangInt,
                &mut composite,
                &mut diag,
            );
            let text = blob_text(diag);
            let composite = Com::from_raw(composite)
                .filter(|_| !failed(r))
                .ok_or(text)?;
            let mut linked = ptr::null_mut();
            let mut diag = ptr::null_mut();
            let r = (composite.vtbl::<ComponentTypeVtbl>().link)(
                composite.as_ptr(),
                &mut linked,
                &mut diag,
            );
            let text = blob_text(diag);
            let com = Com::from_raw(linked).filter(|_| !failed(r)).ok_or(text)?;
            Ok(Program { com })
        }
    }

    /// Every module loaded into the session, the kernel and everything it imports.
    pub(crate) fn loaded_modules(&self) -> Vec<Module> {
        // SAFETY: the vtable is ISession's; returned modules are owned by the session.
        unsafe {
            let v = self.com.vtbl::<SessionVtbl>();
            (0..(v.get_loaded_module_count)(self.com.as_ptr()))
                .filter_map(|i| Com::borrowed((v.get_loaded_module)(self.com.as_ptr(), i)))
                .map(|com| Module { com })
                .collect()
        }
    }

    /// The file and line that declare `decl`.
    pub(crate) fn decl_location(
        &self,
        decl: *mut sys::SlangReflectionDecl,
    ) -> Option<(PathBuf, u32)> {
        let mut loc = sys::slang_SourceLocation::default();
        // SAFETY: the vtable is ISession's; `decl` comes from a module of this session.
        unsafe {
            let r = (self.com.vtbl::<SessionVtbl>().get_decl_source_location)(
                self.com.as_ptr(),
                decl,
                &mut loc,
            );
            if failed(r) {
                return None;
            }
            c_str(loc.filePath).map(|p| (PathBuf::from(p), loc.line.max(0) as u32))
        }
    }
}

impl Module {
    /// The files this module was built from, its includes among them.
    pub(crate) fn dependencies(&self) -> Vec<PathBuf> {
        // SAFETY: the vtable is IModule's; paths live as long as the module.
        unsafe {
            let v = self.com.vtbl::<ModuleVtbl>();
            (0..(v.get_dependency_file_count)(self.com.as_ptr()))
                .filter_map(|i| c_str((v.get_dependency_file_path)(self.com.as_ptr(), i)))
                .map(PathBuf::from)
                .collect()
        }
    }

    pub(crate) fn reflection(&self) -> *mut sys::SlangReflectionDecl {
        // SAFETY: the vtable is IModule's.
        unsafe { (self.com.vtbl::<ModuleVtbl>().get_module_reflection)(self.com.as_ptr()) }
    }
}

impl Program {
    /// The code of the session's one target.
    fn code(&self) -> Result<Com, String> {
        let mut code = ptr::null_mut();
        let mut diag = ptr::null_mut();
        // SAFETY: the vtable is IComponentType's.
        let r = unsafe {
            (self.com.vtbl::<ComponentTypeVtbl>().get_target_code)(
                self.com.as_ptr(),
                0,
                &mut code,
                &mut diag,
            )
        };
        let text = blob_text(diag);
        Com::from_raw(code).filter(|_| !failed(r)).ok_or(text)
    }

    /// The target code as text, for source targets.
    pub(crate) fn text(&self) -> Result<String, String> {
        Ok(String::from_utf8_lossy(blob_bytes(&self.code()?))
            .trim_end_matches(' ')
            .to_owned())
    }

    /// The SPIR-V of the session's one target.
    pub(crate) fn spirv(&self) -> Result<Vec<u32>, String> {
        let code = self.code()?;
        let bytes = blob_bytes(&code);
        if !bytes.len().is_multiple_of(4) {
            return Err(format!(
                "SPIR-V of {} bytes is not whole words",
                bytes.len()
            ));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect())
    }

    /// The program layout of the session's one target; valid while the program lives.
    pub(crate) fn layout(&self) -> Result<*mut sys::SlangReflection, String> {
        let mut diag = ptr::null_mut();
        // SAFETY: the vtable is IComponentType's.
        let layout = unsafe {
            (self.com.vtbl::<ComponentTypeVtbl>().get_layout)(self.com.as_ptr(), 0, &mut diag)
        };
        let text = blob_text(diag);
        if layout.is_null() {
            Err(text)
        } else {
            Ok(layout.cast())
        }
    }
}
