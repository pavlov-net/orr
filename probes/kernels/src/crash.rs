//! Aftermath on a descriptor-heap device: a heap compute pipeline writes through an address no
//! allocation covers; the crash dump is decoded to JSON with the SPIR-V and its debug info
//! supplied through the lookup callbacks, and the JSON is searched for the source file and line.
//!
//! `kernels crash stripped` loads SPIR-V built with `-g2 -separate-debug-info` and serves the
//! `.dbg.spv` sidecar to the source lookup; `kernels crash full` loads SPIR-V with embedded `-g2`
//! debug info. The Aftermath library comes from `AFTERMATH_SDK` (the SDK root) or the Nsight
//! Graphics 2026.2 install. The dump and JSON land next to the executable. PROBE_NO_SIDECAR=1
//! withholds the sidecar, to show it is what maps addresses to lines.

use crate::gpu::*;
use std::ffi::c_void;
use std::sync::Mutex;

const VERSION_API: u32 = 0x21A;
const WATCH_VULKAN: u32 = 0x2;
const FEATURE_DEFER_DEBUG_INFO: u32 = 0x1;
const DECODER_ALL_INFO: u32 = 0x3FFF;
const STATUS_COLLECTING_FAILED: u32 = 2;
const STATUS_FINISHED: u32 = 4;
const RESULT_SUCCESS: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
struct DebugInfoId {
    id: [u64; 2],
}

#[repr(C)]
struct ShaderHash {
    hash: u64,
}

#[repr(C)]
struct DebugName {
    name: [u8; 128],
}

#[repr(C)]
struct SpirvCode {
    data: *const c_void,
    size: u32,
}

type SetData = extern "C" fn(*const c_void, u32);
type DumpCb = extern "C" fn(*const c_void, u32, *mut c_void);
type InfoCb = extern "C" fn(*const c_void, u32, *mut c_void);
type DebugInfoLookup = extern "C" fn(*const DebugInfoId, SetData, *mut c_void);
type ShaderLookup = extern "C" fn(*const ShaderHash, SetData, *mut c_void);
type SourceLookup = extern "C" fn(*const DebugName, SetData, *mut c_void);

struct Api {
    enable: extern "C" fn(u32, u32, u32, Option<DumpCb>, Option<InfoCb>, *const c_void, *const c_void, *mut c_void) -> u32,
    status: extern "C" fn(*mut u32) -> u32,
    create_decoder: extern "C" fn(u32, *const c_void, u32, *mut *mut c_void) -> u32,
    generate_json: extern "C" fn(*mut c_void, u32, u32, DebugInfoLookup, ShaderLookup, SourceLookup, *mut c_void, *mut u32) -> u32,
    get_json: extern "C" fn(*mut c_void, u32, *mut u8) -> u32,
    debug_info_id: extern "C" fn(u32, *const c_void, u32, *mut DebugInfoId) -> u32,
    hash_spirv: extern "C" fn(u32, *const SpirvCode, *mut ShaderHash) -> u32,
    debug_name_spirv: extern "C" fn(u32, *const SpirvCode, *const SpirvCode, *mut DebugName) -> u32,
}

unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}

unsafe fn load() -> (Api, String) {
    let root = std::env::var("AFTERMATH_SDK").unwrap_or_else(|_| {
        r"C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.2.0\SDKs\NsightAftermathSDK\2025.5.0.251114".into()
    });
    let dll = format!(r"{root}\lib\x64\GFSDK_Aftermath_Lib.x64.dll");
    let wide: Vec<u16> = dll.encode_utf16().chain([0]).collect();
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    assert!(!module.is_null(), "cannot load {dll}");
    let sym = |name: &str| {
        let c = format!("{name}\0");
        let p = unsafe { GetProcAddress(module, c.as_ptr()) };
        assert!(!p.is_null(), "missing {name}");
        p
    };
    let api = unsafe {
        Api {
            enable: std::mem::transmute(sym("GFSDK_Aftermath_EnableGpuCrashDumps")),
            status: std::mem::transmute(sym("GFSDK_Aftermath_GetCrashDumpStatus")),
            create_decoder: std::mem::transmute(sym("GFSDK_Aftermath_GpuCrashDump_CreateDecoder")),
            generate_json: std::mem::transmute(sym("GFSDK_Aftermath_GpuCrashDump_GenerateJSON")),
            get_json: std::mem::transmute(sym("GFSDK_Aftermath_GpuCrashDump_GetJSON")),
            debug_info_id: std::mem::transmute(sym("GFSDK_Aftermath_GetShaderDebugInfoIdentifier")),
            hash_spirv: std::mem::transmute(sym("GFSDK_Aftermath_GetShaderHashSpirv")),
            debug_name_spirv: std::mem::transmute(sym("GFSDK_Aftermath_GetShaderDebugNameSpirv")),
        }
    };
    (api, dll)
}

static DUMP: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static DEBUG_INFO: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static SPIRV: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static SIDECAR: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static LOOKUPS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static mut API: Option<Api> = None;

extern "C" fn on_dump(data: *const c_void, size: u32, _user: *mut c_void) {
    *DUMP.lock().unwrap() = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size as usize) }.to_vec();
}

extern "C" fn on_debug_info(data: *const c_void, size: u32, _user: *mut c_void) {
    DEBUG_INFO.lock().unwrap().push(unsafe { std::slice::from_raw_parts(data.cast::<u8>(), size as usize) }.to_vec());
}

#[allow(static_mut_refs)]
fn api() -> &'static Api {
    unsafe { API.as_ref().unwrap() }
}

extern "C" fn lookup_debug_info(id: *const DebugInfoId, set: SetData, _user: *mut c_void) {
    let want = unsafe { *id };
    LOOKUPS.lock().unwrap().push(format!("debug info {:016x}-{:016x}", want.id[0], want.id[1]));
    for blob in DEBUG_INFO.lock().unwrap().iter() {
        let mut got = DebugInfoId { id: [0; 2] };
        (api().debug_info_id)(VERSION_API, blob.as_ptr().cast(), blob.len() as u32, &mut got);
        if got == want {
            set(blob.as_ptr().cast(), blob.len() as u32);
        }
    }
}

extern "C" fn lookup_shader(hash: *const ShaderHash, set: SetData, _user: *mut c_void) {
    let want = unsafe { (*hash).hash };
    let spirv = SPIRV.lock().unwrap();
    let code = SpirvCode { data: spirv.as_ptr().cast(), size: spirv.len() as u32 };
    let mut got = ShaderHash { hash: 0 };
    (api().hash_spirv)(VERSION_API, &code, &mut got);
    LOOKUPS.lock().unwrap().push(format!("shader {want:016x} (ours {:016x})", got.hash));
    if got.hash == want {
        set(spirv.as_ptr().cast(), spirv.len() as u32);
    }
}

extern "C" fn lookup_source(name: *const DebugName, set: SetData, _user: *mut c_void) {
    let raw = unsafe { &(*name).name };
    let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    LOOKUPS.lock().unwrap().push(format!("source debug name {:?}", String::from_utf8_lossy(&raw[..len])));
    let sidecar = SIDECAR.lock().unwrap();
    if !sidecar.is_empty() {
        set(sidecar.as_ptr().cast(), sidecar.len() as u32);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Root {
    out: u64,
    bad: u64,
    count: u32,
    pad: [u32; 3],
}

pub unsafe fn run(mode: &str) {
    unsafe {
        let (a, dll) = load();
        API = Some(a);
        println!("aftermath: {dll}");
        let r = (api().enable)(VERSION_API, WATCH_VULKAN, FEATURE_DEFER_DEBUG_INFO, Some(on_dump), Some(on_debug_info), std::ptr::null(), std::ptr::null(), std::ptr::null_mut());
        println!("EnableGpuCrashDumps -> {r:#x}");
        let code: Vec<u32> = if mode == "full" { crate::spv!("crash_full") } else { crate::spv!("crash_stripped") };
        *SPIRV.lock().unwrap() = bytes(&code).to_vec();
        // PROBE_NO_SIDECAR=1 withholds the sidecar from the source lookup.
        if mode != "full" && std::env::var("PROBE_NO_SIDECAR").is_err() {
            *SIDECAR.lock().unwrap() = std::fs::read(env!("CRASH_SIDECAR")).expect("sidecar");
        }
        if mode != "full" {
            // The DebugName the source lookup will ask for, computed offline from the pair.
            let sidecar = std::fs::read(env!("CRASH_SIDECAR")).unwrap();
            let full = SpirvCode { data: sidecar.as_ptr().cast(), size: sidecar.len() as u32 };
            let stripped = SpirvCode { data: code.as_ptr().cast(), size: (code.len() * 4) as u32 };
            let mut name = DebugName { name: [0; 128] };
            let r = (api().debug_name_spirv)(VERSION_API, &full, &stripped, &mut name);
            let len = name.name.iter().position(|&b| b == 0).unwrap_or(128);
            println!("GetShaderDebugNameSpirv(sidecar, stripped) -> {r:#x} {:?}", String::from_utf8_lossy(&name.name[..len]));
        }
        let gpu = Gpu::new(&Options { diagnostics: true, ..Default::default() });
        println!("device: {} driver {}; mode {mode}; SPIR-V {} bytes, sidecar {} bytes", gpu.name, gpu.driver, code.len() * 4, SIDECAR.lock().unwrap().len());
        let pipe = gpu.compute(&code, c"main", &[]);
        let out = gpu.buffer(65536 * 4, Mem::Upload);
        let roots = gpu.buffer(256, Mem::Upload);
        let root = Root { out: out.addr, bad: out.addr + (1u64 << 40), count: 65536, pad: [0; 3] };
        std::ptr::write(roots.ptr.cast::<Root>(), root);
        let cmd = gpu.begin();
        gpu.bind_heaps(cmd);
        gpu.device.cmd_bind_pipeline(cmd, pipe.point, pipe.pipeline);
        gpu.push_root(cmd, roots.addr);
        gpu.device.cmd_dispatch(cmd, 65536 / 64, 1, 1);
        let res = gpu.try_submit_wait(cmd);
        println!("submit -> {res:?}");
        let start = std::time::Instant::now();
        let mut status = 0u32;
        while start.elapsed().as_secs() < 20 {
            (api().status)(&mut status);
            if status == STATUS_FINISHED || status == STATUS_COLLECTING_FAILED {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        println!("crash dump status {status} after {:.2} s", start.elapsed().as_secs_f64());
        let dump = DUMP.lock().unwrap().clone();
        let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
        if dump.is_empty() {
            println!("RESULT no crash dump");
            return;
        }
        std::fs::write(dir.join(format!("crash_{mode}.nv-gpudmp")), &dump).unwrap();
        println!("dump {} bytes; debug info blobs {}", dump.len(), DEBUG_INFO.lock().unwrap().len());
        let mut decoder = std::ptr::null_mut();
        let r = (api().create_decoder)(VERSION_API, dump.as_ptr().cast(), dump.len() as u32, &mut decoder);
        assert_eq!(r, RESULT_SUCCESS, "CreateDecoder {r:#x}");
        let mut size = 0u32;
        let r = (api().generate_json)(decoder, DECODER_ALL_INFO, 0, lookup_debug_info, lookup_shader, lookup_source, std::ptr::null_mut(), &mut size);
        println!("GenerateJSON -> {r:#x}, {size} bytes");
        let mut json = vec![0u8; size as usize];
        (api().get_json)(decoder, size, json.as_mut_ptr());
        let json = String::from_utf8_lossy(&json[..json.iter().position(|&b| b == 0).unwrap_or(json.len())]).into_owned();
        let path = dir.join(format!("crash_{mode}.json"));
        std::fs::write(&path, &json).unwrap();
        for l in LOOKUPS.lock().unwrap().iter() {
            println!("lookup: {l}");
        }
        let fault_line = include_str!("../shaders/crash.slang").lines().position(|l| l.contains("// FAULT")).unwrap() + 1;
        let mentions_file = json.contains("crash.slang");
        let mentions_line = json.contains(&format!("crash.slang:{fault_line}"));
        println!("json: {}; mentions crash.slang: {mentions_file}; fault line {fault_line} found: {mentions_line}", path.display());
        println!("RESULT {}", if mentions_file && mentions_line { "RESOLVED" } else { "UNRESOLVED" });
    }
}
