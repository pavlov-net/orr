//! Nsight Aftermath GPU crash dumps: the library loaded from its SDK, crash dumps written to
//! the dump directory, and decoding with shader lookups by hash and debug name.

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::{Duration, Instant};

use super::device::Shared;
use super::diagnostics::unmarker;

const WATCH_VULKAN: u32 = 0x2;
const DEFER_DEBUG_INFO: u32 = 0x1;
const DESCRIPTION_APPLICATION_NAME: u32 = 0x1;
const DECODER_ALL_INFO: u32 = 0x3FFF;
const STATUS_NOT_STARTED: u32 = 0;
const STATUS_COLLECTING_FAILED: u32 = 2;
const STATUS_FINISHED: u32 = 4;
const RESULT_SUCCESS: u32 = 0x1;

/// How long a lost device waits for Aftermath to deliver its dump, and how long it waits for
/// Aftermath to notice the crash at all.
const DUMP_TIMEOUT: Duration = Duration::from_secs(20);
const START_TIMEOUT: Duration = Duration::from_secs(3);

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DebugInfoId([u64; 2]);

#[repr(C)]
struct ShaderHash(u64);

#[repr(C)]
struct DebugName([u8; 128]);

#[repr(C)]
struct SpirvCode {
    data: *const c_void,
    size: u32,
}

type SetData = unsafe extern "C" fn(*const c_void, u32);
type DumpCb = unsafe extern "C" fn(*const c_void, u32, *mut c_void);
type AddDescription = unsafe extern "C" fn(u32, *const c_char);
type DescriptionCb = unsafe extern "C" fn(AddDescription, *mut c_void);
type ResolveMarker = unsafe extern "C" fn(*const c_void, u32);
type ResolveMarkerCb = unsafe extern "C" fn(*const c_void, u32, *mut c_void, ResolveMarker);
type DebugInfoLookup = unsafe extern "C" fn(*const DebugInfoId, SetData, *mut c_void);
type ShaderLookup = unsafe extern "C" fn(*const ShaderHash, SetData, *mut c_void);
type SourceLookup = unsafe extern "C" fn(*const DebugName, SetData, *mut c_void);

struct Library {
    _lib: libloading::Library,
    version: u32,
    enable: unsafe extern "C" fn(
        u32,
        u32,
        u32,
        DumpCb,
        DumpCb,
        DescriptionCb,
        ResolveMarkerCb,
        *mut c_void,
    ) -> u32,
    status: unsafe extern "C" fn(*mut u32) -> u32,
    create_decoder: unsafe extern "C" fn(u32, *const c_void, u32, *mut *mut c_void) -> u32,
    destroy_decoder: unsafe extern "C" fn(*mut c_void) -> u32,
    generate_json: unsafe extern "C" fn(
        *mut c_void,
        u32,
        u32,
        DebugInfoLookup,
        ShaderLookup,
        SourceLookup,
        *mut c_void,
        *mut u32,
    ) -> u32,
    get_json: unsafe extern "C" fn(*mut c_void, u32, *mut u8) -> u32,
    debug_info_id: unsafe extern "C" fn(u32, *const c_void, u32, *mut DebugInfoId) -> u32,
    hash_spirv: unsafe extern "C" fn(u32, *const SpirvCode, *mut ShaderHash) -> u32,
    debug_name_spirv:
        unsafe extern "C" fn(u32, *const SpirvCode, *const SpirvCode, *mut DebugName) -> u32,
}

/// What the callbacks and the decoder share.
#[derive(Default)]
struct State {
    dump_dir: PathBuf,
    app: CString,
    /// Dumps written, not yet decoded.
    dumps: Vec<(PathBuf, Vec<u8>)>,
    debug_info: HashMap<DebugInfoId, Vec<u8>>,
    shaders: HashMap<u64, Vec<u8>>,
    sidecars: HashMap<Vec<u8>, Vec<u8>>,
    devices: Vec<Weak<Shared>>,
    /// The last resolved marker; Aftermath reads it until the next resolve.
    resolved: Vec<u8>,
}

static LIBRARY: OnceLock<Result<Library, String>> = OnceLock::new();
static STATE: Mutex<Option<State>> = Mutex::new(None);

fn state() -> MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

fn library() -> Option<&'static Library> {
    LIBRARY.get().and_then(|l| l.as_ref().ok())
}

/// The newest directory in `dir` whose name starts with `prefix`, by the numbers in its
/// name.
fn newest(dir: &Path, prefix: &str) -> Option<PathBuf> {
    let name = |p: &PathBuf| {
        p.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_owned()
    };
    let version = |p: &PathBuf| -> Vec<u64> {
        name(p)
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|n| n.parse().ok())
            .collect()
    };
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && name(p).starts_with(prefix))
        .max_by_key(version)
}

/// `AFTERMATH_SDK`, or the Aftermath SDK of the newest Nsight Graphics install.
fn sdk_root() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("AFTERMATH_SDK") {
        return Ok(root.into());
    }
    newest(
        Path::new(r"C:\Program Files\NVIDIA Corporation"),
        "Nsight Graphics",
    )
    .and_then(|n| newest(&n.join("SDKs").join("NsightAftermathSDK"), ""))
    .ok_or_else(|| "AFTERMATH_SDK is not set and no Nsight Graphics install has the SDK".into())
}

/// The API version the SDK's header declares, which its library expects.
fn header_version(root: &Path) -> Result<u32, String> {
    let header = root.join("include").join("GFSDK_Aftermath_Defines.h");
    let text =
        std::fs::read_to_string(&header).map_err(|e| format!("{}: {e}", header.display()))?;
    text.lines()
        .find_map(|l| {
            let (_, rest) = l.split_once("GFSDK_Aftermath_Version_API")?;
            let hex = rest.split_once("0x")?.1;
            let digits: String = hex.chars().take_while(char::is_ascii_hexdigit).collect();
            u32::from_str_radix(&digits, 16).ok()
        })
        .ok_or_else(|| format!("{}: no GFSDK_Aftermath_Version_API", header.display()))
}

fn load() -> Result<Library, String> {
    let root = sdk_root()?;
    let version = header_version(&root)?;
    let file = if cfg!(windows) {
        "GFSDK_Aftermath_Lib.x64.dll"
    } else {
        "libGFSDK_Aftermath_Lib.x64.so"
    };
    let path = root.join("lib").join("x64").join(file);
    // SAFETY: loading the Aftermath library, whose initialisers have no preconditions.
    let lib = unsafe { libloading::Library::new(&path) }
        .map_err(|e| format!("{}: {e}", path.display()))?;
    /// SAFETY: `T` must be the symbol's signature in the Aftermath headers.
    unsafe fn sym<T: Copy>(lib: &libloading::Library, name: &str) -> Result<T, String> {
        let c = format!("{name}\0");
        // SAFETY: the caller's type is the symbol's.
        unsafe { lib.get::<T>(c.as_bytes()) }
            .map(|s| *s)
            .map_err(|e| format!("{name}: {e}"))
    }
    // SAFETY: every field's type is its symbol's signature.
    unsafe {
        Ok(Library {
            version,
            enable: sym(&lib, "GFSDK_Aftermath_EnableGpuCrashDumps")?,
            status: sym(&lib, "GFSDK_Aftermath_GetCrashDumpStatus")?,
            create_decoder: sym(&lib, "GFSDK_Aftermath_GpuCrashDump_CreateDecoder")?,
            destroy_decoder: sym(&lib, "GFSDK_Aftermath_GpuCrashDump_DestroyDecoder")?,
            generate_json: sym(&lib, "GFSDK_Aftermath_GpuCrashDump_GenerateJSON")?,
            get_json: sym(&lib, "GFSDK_Aftermath_GpuCrashDump_GetJSON")?,
            debug_info_id: sym(&lib, "GFSDK_Aftermath_GetShaderDebugInfoIdentifier")?,
            hash_spirv: sym(&lib, "GFSDK_Aftermath_GetShaderHashSpirv")?,
            debug_name_spirv: sym(&lib, "GFSDK_Aftermath_GetShaderDebugNameSpirv")?,
            _lib: lib,
        })
    }
}

/// Enables crash dumps into `dump_dir` before the device is created. The library is
/// enabled once per process; later devices move the dump directory.
pub(crate) fn enable(dump_dir: &Path, app: &str) -> Result<(), String> {
    std::fs::create_dir_all(dump_dir).map_err(|e| format!("{}: {e}", dump_dir.display()))?;
    let mut first = false;
    let lib = LIBRARY
        .get_or_init(|| {
            first = true;
            load()
        })
        .as_ref()
        .map_err(Clone::clone)?;
    {
        let mut s = state();
        let s = s.get_or_insert_with(State::default);
        s.dump_dir = dump_dir.to_path_buf();
        s.app = CString::new(app.replace('\0', "")).unwrap_or_default();
    }
    if first {
        // SAFETY: the callbacks match Aftermath's signatures and live for the process.
        let r = unsafe {
            (lib.enable)(
                lib.version,
                WATCH_VULKAN,
                DEFER_DEBUG_INFO,
                on_dump,
                on_debug_info,
                on_description,
                on_resolve_marker,
                std::ptr::null_mut(),
            )
        };
        if r != RESULT_SUCCESS {
            return Err(format!(
                "GFSDK_Aftermath_EnableGpuCrashDumps returned {r:#x}"
            ));
        }
    }
    Ok(())
}

/// Lets crash dumps resolve the device's pass markers.
pub(crate) fn add_device(shared: Weak<Shared>) {
    if let Some(s) = state().as_mut() {
        s.devices.retain(|d| d.strong_count() > 0);
        s.devices.push(shared);
    }
}

fn spirv(words: &[u32]) -> SpirvCode {
    SpirvCode {
        data: words.as_ptr().cast(),
        size: (words.len() * 4) as u32,
    }
}

/// Keeps a module pipelines are created from, for the decoder's shader lookup.
pub(crate) fn add_shader(code: &[u32]) {
    let Some(lib) = library() else { return };
    let mut hash = ShaderHash(0);
    // SAFETY: `code` is a SPIR-V module of the stated size.
    if unsafe { (lib.hash_spirv)(lib.version, &spirv(code), &mut hash) } != RESULT_SUCCESS {
        return;
    }
    if let Some(s) = state().as_mut() {
        s.shaders
            .entry(hash.0)
            .or_insert_with(|| bytemuck::cast_slice(code).to_vec());
    }
}

/// Keeps a stripped module's separated debug info under the debug name Aftermath asks for.
pub(crate) fn add_sidecar(stripped: &[u32], sidecar: &[u8]) {
    let Some(lib) = library() else { return };
    add_shader(stripped);
    let full = SpirvCode {
        data: sidecar.as_ptr().cast(),
        size: sidecar.len() as u32,
    };
    let mut name = DebugName([0; 128]);
    // SAFETY: both are SPIR-V modules of the stated sizes.
    let r = unsafe { (lib.debug_name_spirv)(lib.version, &full, &spirv(stripped), &mut name) };
    if r != RESULT_SUCCESS {
        log::warn!("GFSDK_Aftermath_GetShaderDebugNameSpirv returned {r:#x}");
        return;
    }
    if let Some(s) = state().as_mut() {
        s.sidecars
            .insert(until_nul(&name.0).to_vec(), sidecar.to_vec());
    }
}

/// After device loss: waits for Aftermath's dump, decodes it to JSON beside it with the
/// driver's shader debug info and the shaders and sidecars it names, and returns the dump's
/// path.
pub(crate) fn wait_for_dump() -> Option<PathBuf> {
    let lib = library()?;
    let start = Instant::now();
    let mut status = STATUS_NOT_STARTED;
    while start.elapsed() < DUMP_TIMEOUT {
        // SAFETY: plain status query.
        unsafe { (lib.status)(&mut status) };
        if status == STATUS_FINISHED || status == STATUS_COLLECTING_FAILED {
            break;
        }
        if status == STATUS_NOT_STARTED && start.elapsed() > START_TIMEOUT {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let (path, dump) = {
        let mut s = state();
        let s = s.as_mut()?;
        for (id, blob) in &s.debug_info {
            write_beside(
                s,
                &format!("shader-{:016x}{:016x}.nvdbg", id.0[0], id.0[1]),
                blob,
            );
        }
        s.dumps.pop()?
    };
    match decode(lib, &dump) {
        Ok(json) => {
            let file = path.with_extension("json");
            if let Err(e) = std::fs::write(&file, json) {
                log::warn!("{}: {e}", file.display());
            }
        }
        Err(e) => log::warn!("Aftermath decode: {e}"),
    }
    Some(path)
}

fn decode(lib: &Library, dump: &[u8]) -> Result<String, String> {
    let mut decoder = std::ptr::null_mut();
    // SAFETY: `dump` is a complete crash dump.
    let r = unsafe {
        (lib.create_decoder)(
            lib.version,
            dump.as_ptr().cast(),
            dump.len() as u32,
            &mut decoder,
        )
    };
    if r != RESULT_SUCCESS {
        return Err(format!("CreateDecoder returned {r:#x}"));
    }
    let mut size = 0u32;
    // SAFETY: a live decoder; the lookups take the state lock, which is not held here.
    let r = unsafe {
        (lib.generate_json)(
            decoder,
            DECODER_ALL_INFO,
            0,
            lookup_debug_info,
            lookup_shader,
            lookup_source,
            std::ptr::null_mut(),
            &mut size,
        )
    };
    let mut json = vec![0u8; size as usize];
    let got = if r == RESULT_SUCCESS {
        // SAFETY: the buffer holds the size GenerateJSON reported.
        unsafe { (lib.get_json)(decoder, size, json.as_mut_ptr()) }
    } else {
        r
    };
    // SAFETY: the decoder is live and not used again.
    unsafe { (lib.destroy_decoder)(decoder) };
    if got != RESULT_SUCCESS {
        return Err(format!("GenerateJSON returned {got:#x}"));
    }
    Ok(String::from_utf8_lossy(until_nul(&json)).into_owned())
}

unsafe extern "C" fn on_dump(data: *const c_void, size: u32, _: *mut c_void) {
    // SAFETY: Aftermath passes a dump of `size` bytes for the duration of the call.
    let dump = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size as usize) }.to_vec();
    let mut s = state();
    let Some(s) = s.as_mut() else { return };
    let app = s.app.to_string_lossy();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let path = s
        .dump_dir
        .join(format!("{app}-{}-{stamp}.nv-gpudmp", std::process::id()));
    if let Err(e) = std::fs::write(&path, &dump) {
        log::error!("{}: {e}", path.display());
    }
    s.dumps.push((path, dump));
}

unsafe extern "C" fn on_debug_info(data: *const c_void, size: u32, _: *mut c_void) {
    let Some(lib) = library() else { return };
    let mut id = DebugInfoId([0; 2]);
    // SAFETY: Aftermath passes `size` bytes of debug info for the duration of the call.
    unsafe { (lib.debug_info_id)(lib.version, data, size, &mut id) };
    // SAFETY: as above.
    let blob = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size as usize) }.to_vec();
    if let Some(s) = state().as_mut() {
        s.debug_info.insert(id, blob);
    }
}

unsafe extern "C" fn on_description(add: AddDescription, _: *mut c_void) {
    let app = state().as_ref().map(|s| s.app.clone());
    if let Some(app) = app.filter(|a| !a.is_empty()) {
        // SAFETY: a NUL-terminated string Aftermath copies during the call.
        unsafe { add(DESCRIPTION_APPLICATION_NAME, app.as_ptr()) };
    }
}

unsafe extern "C" fn on_resolve_marker(
    data: *const c_void,
    _size: u32,
    _: *mut c_void,
    resolve: ResolveMarker,
) {
    let (id, end) = unmarker(data as usize);
    let mut s = state();
    let Some(s) = s.as_mut() else { return };
    let Some(name) = s
        .devices
        .iter()
        .filter_map(Weak::upgrade)
        .find_map(|d| d.pass_name(id))
    else {
        return;
    };
    s.resolved = format!("{} {name}", if end { "end" } else { "begin" }).into_bytes();
    // SAFETY: the bytes stay put until the next resolve, as Aftermath requires.
    unsafe { resolve(s.resolved.as_ptr().cast(), s.resolved.len() as u32) };
}

unsafe extern "C" fn lookup_debug_info(id: *const DebugInfoId, set: SetData, _: *mut c_void) {
    let s = state();
    // SAFETY: Aftermath passes a valid identifier.
    let id = unsafe { *id };
    if let Some(blob) = s.as_ref().and_then(|s| s.debug_info.get(&id)) {
        // SAFETY: Aftermath copies the data during the call.
        unsafe { set(blob.as_ptr().cast(), blob.len() as u32) };
    }
}

unsafe extern "C" fn lookup_shader(hash: *const ShaderHash, set: SetData, _: *mut c_void) {
    let s = state();
    // SAFETY: Aftermath passes a valid hash.
    let hash = unsafe { (*hash).0 };
    let Some(s) = s.as_ref() else { return };
    if let Some(code) = s.shaders.get(&hash) {
        write_beside(s, &format!("shader-{hash:016x}.spv"), code);
        // SAFETY: Aftermath copies the data during the call.
        unsafe { set(code.as_ptr().cast(), code.len() as u32) };
    }
}

unsafe extern "C" fn lookup_source(name: *const DebugName, set: SetData, _: *mut c_void) {
    let s = state();
    // SAFETY: Aftermath passes a valid name.
    let name = until_nul(unsafe { &(*name).0 });
    let Some(s) = s.as_ref() else { return };
    if let Some(sidecar) = s.sidecars.get(name) {
        write_beside(s, &String::from_utf8_lossy(name), sidecar);
        // SAFETY: Aftermath copies the data during the call.
        unsafe { set(sidecar.as_ptr().cast(), sidecar.len() as u32) };
    }
}

fn until_nul(bytes: &[u8]) -> &[u8] {
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    &bytes[..len]
}

/// Writes a file a dump refers to into the dump directory, where Nsight Graphics finds it.
fn write_beside(s: &State, name: &str, bytes: &[u8]) {
    let file = s.dump_dir.join(name);
    if let Err(e) = std::fs::write(&file, bytes) {
        log::warn!("{}: {e}", file.display());
    }
}
