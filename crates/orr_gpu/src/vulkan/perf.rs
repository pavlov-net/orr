//! Counter sessions: per-pass hardware counters through the range profiler of the Nsight Perf
//! SDK's C API.
//!
//! Every pass of a counter frame is a range named after the pass and its occurrence in the
//! frame (`name#k`), so the SDK keeps one range per pass and a replay's ranges line up with
//! the frame it repeats.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::zeroed;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{JoinHandle, ThreadId};
use std::time::{Duration, Instant};

use ash::vk::{self, Handle as _};

use super::load_reason;
use super::queue::Queue;
use crate::{
    CounterDesc, CounterFrame, CounterMetric, Error, MAX_COUNTER_PASSES, PassCounters, QueueKind,
    TimelinePoint,
};

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    unreachable_pub,
    unsafe_op_in_unsafe_fn,
    clippy::all,
    clippy::pedantic
)]
mod sys {
    /// The struct's `NVPA_STRUCT_SIZE`, which its `structSize` field carries.
    pub(crate) trait StructSize {
        const SIZE: usize;
    }

    include!(concat!(env!("OUT_DIR"), "/nvperf.rs"));
    include!(concat!(env!("OUT_DIR"), "/nvperf_sizes.rs"));
}

use sys::StructSize;

/// How long a counter frame waits for its submits and for the SDK to collect them.
const TIMEOUT: Duration = Duration::from_secs(10);
/// The longest range name, with its NUL.
const MAX_RANGE_NAME: u32 = 256;
const SUCCESS: sys::NVPA_Status = sys::NVPA_Status_NVPA_STATUS_SUCCESS;

#[cfg(windows)]
const LIBRARY: &str = "nvperf_grfx_host.dll";
#[cfg(not(windows))]
const LIBRARY: &str = "libnvperf_grfx_host.so";

/// A struct with its `structSize` set and every other field zero unless named.
macro_rules! params {
    ($ty:ident { $($field:ident: $value:expr),* $(,)? }) => {
        sys::$ty {
            structSize: <sys::$ty as StructSize>::SIZE,
            $($field: $value,)*
            // SAFETY: the SDK's structs are plain C; zero is every field's default.
            ..unsafe { zeroed() }
        }
    };
}

/// Where a counted pass's counters land: unset while its frame is open, `None` once the frame
/// ended without them (it needs a replay, failed or was abandoned, or the pass's list wasn't
/// submitted in it).
pub(super) type CounterSlot = OnceLock<Option<PassCounters>>;

/// A pass recorded in a counter frame.
pub(super) struct Counted {
    pub(super) slot: Arc<CounterSlot>,
    /// The pass opened a range, which it closes; false past [`MAX_COUNTER_PASSES`].
    pub(super) ranged: bool,
    /// Its frame is open, so the list may still be submitted; changed under the submit lock.
    pub(super) open: Arc<AtomicBool>,
}

type GetProcAddress = unsafe extern "C" fn(*const c_char) -> sys::NVPA_GenericFn;
type Entry<P> = unsafe extern "C" fn(*mut P) -> sys::NVPA_Status;

/// The SDK's host library.
pub(crate) struct Library {
    _lib: libloading::Library,
    get: GetProcAddress,
}

/// The host library inside the SDK at `sdk`, or its `NvPerf` folder, for this target.
fn library_path(sdk: &Path) -> Result<PathBuf, String> {
    let os = if cfg!(windows) { "bin" } else { "lib" };
    let arch = if cfg!(target_arch = "aarch64") {
        "a64"
    } else {
        "x64"
    };
    [sdk.to_path_buf(), sdk.join("NvPerf")]
        .into_iter()
        .map(|root| root.join(os).join(arch).join(LIBRARY))
        .find(|p| p.is_file())
        .ok_or_else(|| {
            format!(
                "{} holds no {os}/{arch}/{LIBRARY}, nor does its NvPerf folder",
                sdk.display()
            )
        })
}

/// A failed call's status, with the remedy for the ones a user can fix.
fn status_text(status: sys::NVPA_Status) -> String {
    match status {
        sys::NVPA_Status_NVPA_STATUS_INSUFFICIENT_PRIVILEGE => {
            "profiling permissions are not enabled; see https://developer.nvidia.com/ERR_NVGPUCTRPERM"
                .into()
        }
        sys::NVPA_Status_NVPA_STATUS_INSUFFICIENT_DRIVER_VERSION => {
            "the driver is too old for this SDK".into()
        }
        sys::NVPA_Status_NVPA_STATUS_RESOURCE_UNAVAILABLE => {
            "another profiler holds the GPU's counters".into()
        }
        s => format!("status {s}"),
    }
}

fn sdk_error(e: String) -> Error {
    Error::Invalid(format!("Nsight Perf SDK: {e}"))
}

impl Library {
    /// The host library and the instance extensions its profiler needs; `None` (with a
    /// warning) when the SDK or the Vulkan loader cannot provide them.
    pub(crate) fn load(
        desc: &CounterDesc<'_>,
        entry: &ash::Entry,
    ) -> Option<(Library, Vec<CString>)> {
        Library::open(desc, entry)
            .inspect_err(|e| log::warn!("Nsight Perf SDK unavailable: {e}"))
            .ok()
    }

    fn open(desc: &CounterDesc<'_>, entry: &ash::Entry) -> Result<(Library, Vec<CString>), String> {
        let path = match desc.sdk {
            Some(sdk) => library_path(sdk)?,
            None => PathBuf::from(LIBRARY),
        };
        // SAFETY: the SDK's host library runs nothing at load beyond its own setup.
        let lib = unsafe { libloading::Library::new(&path) }
            .map_err(|e| format!("{}: {}", path.display(), load_reason(&e)))?;
        // SAFETY: the export has this signature (nvperf_common.h).
        let get = unsafe { lib.get::<GetProcAddress>(b"NVPA_GetProcAddress\0") }
            .map(|s| *s)
            .map_err(|e| format!("NVPA_GetProcAddress: {}", load_reason(&e)))?;
        let l = Library { _lib: lib, get };
        l.call(
            c"NVPW_InitializeHost",
            &mut params!(NVPW_InitializeHost_Params {}),
        )?;
        l.call(
            c"NVPW_InitializeTarget",
            &mut params!(NVPW_InitializeTarget_Params {}),
        )?;
        let mut p = params!(NVPW_VK_Profiler_GetRequiredInstanceExtensions_Params {
            apiVersion: vk::API_VERSION_1_4,
        });
        l.call(c"NVPW_VK_Profiler_GetRequiredInstanceExtensions", &mut p)?;
        // SAFETY: the SDK returned this many NUL-terminated names.
        let exts = unsafe { names(p.ppInstanceExtensionNames, p.numInstanceExtensionNames) };
        // SAFETY: plain query.
        let offered =
            unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        if let Some(e) = exts.iter().find(|e| {
            !offered
                .iter()
                .any(|o| o.extension_name_as_c_str() == Ok(e.as_c_str()))
        }) {
            return Err(format!("the Vulkan loader lacks {}", e.to_string_lossy()));
        }
        Ok((l, exts))
    }

    fn entry<P>(&self, name: &CStr) -> Result<Entry<P>, String> {
        // SAFETY: a NUL-terminated name; the library stays loaded while `self` lives.
        match unsafe { (self.get)(name.as_ptr()) } {
            // SAFETY: every NVPW entry point takes its params struct and returns a status;
            // `P` is the params struct named after `name`.
            Some(f) => Ok(unsafe {
                std::mem::transmute::<unsafe extern "C" fn() -> sys::NVPA_Status, Entry<P>>(f)
            }),
            None => Err(format!("{} not found", name.to_string_lossy())),
        }
    }

    fn call<P>(&self, name: &CStr, params: &mut P) -> Result<(), String> {
        let f = self.entry::<P>(name)?;
        // SAFETY: `params` is the entry point's params struct with its size set.
        let status = unsafe { f(params) };
        if status == SUCCESS {
            Ok(())
        } else {
            Err(format!(
                "{}: {}",
                name.to_string_lossy(),
                status_text(status)
            ))
        }
    }

    /// The device extensions the profiler needs on `physical`, which `offers` must all
    /// accept; `None` (with a warning) otherwise.
    pub(crate) fn device_extensions(
        self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        offers: impl Fn(&CStr) -> bool,
    ) -> Option<(Library, Vec<CString>)> {
        let run = || -> Result<Vec<CString>, String> {
            self.call(
                c"NVPW_VK_LoadDriver",
                &mut params!(NVPW_VK_LoadDriver_Params {
                    instance: instance.handle().as_raw() as _,
                }),
            )?;
            let mut p = params!(NVPW_VK_Profiler_GetRequiredDeviceExtensions_Params {
                apiVersion: vk::API_VERSION_1_4,
                instance: instance.handle().as_raw() as _,
                physicalDevice: physical.as_raw() as _,
                pfnGetInstanceProcAddr: entry.static_fn().get_instance_proc_addr as *mut c_void,
            });
            self.call(c"NVPW_VK_Profiler_GetRequiredDeviceExtensions", &mut p)?;
            // SAFETY: the SDK returned this many NUL-terminated names.
            let exts = unsafe { names(p.ppDeviceExtensionNames, p.numDeviceExtensionNames) };
            match exts.iter().find(|e| !offers(e)) {
                Some(e) => Err(format!("the GPU lacks {}", e.to_string_lossy())),
                None => Ok(exts),
            }
        };
        match run() {
            Ok(exts) => Some((self, exts)),
            Err(e) => {
                log::warn!("Nsight Perf SDK unavailable: {e}");
                None
            }
        }
    }
}

/// Closes `frame` to submits and returns the point its submits signal.
fn close(queue: &Queue, frame: &Frame) -> TimelinePoint {
    let _submits = queue.lock_submits();
    frame.open.store(false, Ordering::Release);
    queue.submitted_point()
}

/// # Safety
/// `list` points at `n` NUL-terminated strings, or is null.
unsafe fn names(list: *const *const c_char, n: usize) -> Vec<CString> {
    if list.is_null() {
        return Vec::new();
    }
    // SAFETY: as the caller promises.
    unsafe { std::slice::from_raw_parts(list, n) }
        .iter()
        // SAFETY: as the caller promises.
        .map(|&s| unsafe { CStr::from_ptr(s) }.to_owned())
        .collect()
}

/// Bytes at an 8-byte boundary, as the SDK's images need.
struct Aligned {
    words: Vec<u64>,
    len: usize,
}

impl Aligned {
    fn zeroed(len: usize) -> Aligned {
        Aligned {
            words: vec![0; len.div_ceil(8)],
            len,
        }
    }

    fn ptr(&self) -> *const u8 {
        self.words.as_ptr().cast()
    }

    fn mut_ptr(&mut self) -> *mut u8 {
        self.words.as_mut_ptr().cast()
    }
}

/// The device's link to the SDK, and the counter session open on one of its queues.
pub(crate) struct Perf {
    library: Library,
    chip: CString,
    evaluator: *mut sys::NVPW_MetricsEvaluator,
    _evaluator_scratch: Vec<u8>,
    push: Entry<sys::NVPW_VK_Profiler_CommandBuffer_PushRange_Params>,
    pop: Entry<sys::NVPW_VK_Profiler_CommandBuffer_PopRange_Params>,
    instance: vk::Instance,
    physical: vk::PhysicalDevice,
    device: vk::Device,
    get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
    /// The kind index plus one of the queue whose lists take ranges; 0 outside a frame.
    recording: AtomicUsize,
    counting: Mutex<Option<Counting>>,
}

// SAFETY: the evaluator is used under `counting`'s lock, and at creation and shutdown, and the
// SDK's entry points may be called from any thread.
unsafe impl Send for Perf {}
// SAFETY: as above.
unsafe impl Sync for Perf {}

/// The metrics of a counter session, scheduled into the config the SDK collects.
struct Plan {
    /// The evaluation requests of the available metrics, with each one's index among all.
    requests: Vec<sys::NVPW_MetricEvalRequest>,
    available: Vec<usize>,
    metrics: usize,
    config: Aligned,
    prefix: Aligned,
}

/// The open counter session: its queue, its plan and the SDK session collecting it.
struct Counting {
    kind: QueueKind,
    queue: vk::Queue,
    plan: Plan,
    replays: usize,
    /// `None` after a frame failed, until the next frame begins another.
    live: Option<Live>,
    frame: Frame,
}

/// An SDK session on the counting queue, and the collection under way in it.
struct Live {
    image: Aligned,
    scratch: Aligned,
    /// Runs the SDK's GPU work for the begin and end of each frame until the session ends.
    service: JoinHandle<()>,
    /// The SDK has the config for the collection under way; false before its first frame.
    configured: bool,
    /// The range names of the collection's first frame, which its replays repeat.
    first: Option<Vec<CString>>,
    /// Ranges and trace bytes the SDK dropped in the collection so far.
    dropped: usize,
}

/// The open frame's counted passes.
#[derive(Default)]
struct Frame {
    /// What its [`Counted`]s share.
    open: Arc<AtomicBool>,
    /// By range name.
    ranges: Vec<(CString, Arc<CounterSlot>)>,
    passes: usize,
    occurrences: HashMap<&'static str, u32>,
    /// The threads that recorded the passes.
    threads: HashSet<ThreadId>,
    /// The first range command that failed.
    failed: Option<String>,
}

/// Per-pass hardware counters on one queue, from [`Queue::start_counters`] until it drops.
///
/// Each frame to count goes between [`CounterSession::begin_frame`] and
/// [`CounterSession::end_frame`]; outside them nothing is counted and nothing costs more. The
/// session holds the GPU's profiling hardware: another session on this device, and other
/// profilers, can't start until it drops. Dropping it inside a frame waits for the queue and
/// discards the frame.
///
/// When the metrics fit in one frame ([`CounterSession::replays`] is 1), every counted frame
/// carries its own counters. Otherwise a collection takes that many frames, each counting a
/// different part of the metrics, and its counters land on the last one: the frames must be
/// the same work, so a renderer with temporal state (accumulation, history, animation, random
/// sequences) holds that state still across them, or each metric describes a slightly
/// different frame.
///
/// Every pass a counter frame counts is timed exactly
/// ([`CommandList::set_exact_timing`](crate::CommandList::set_exact_timing)): the SDK
/// serializes them anyway.
pub struct CounterSession<'q> {
    queue: &'q Queue,
    metrics: Vec<CounterMetric>,
    replays: usize,
}

impl Queue {
    /// Opens a counter session on this queue for `metrics`, by the SDK's names
    /// (`unit__counter.rollup.submetric`, such as [`crate::DEFAULT_COUNTER_METRICS`]). Needs
    /// [`crate::Caps::perf_counters`] and a render or compute queue; one session per device
    /// at a time.
    ///
    /// The SDK collects what the hardware can count at once in one frame; metrics that don't
    /// fit together take more ([`CounterSession::replays`]). A metric the GPU or driver does
    /// not expose is collected as `None`; when none is exposed, the session fails.
    pub fn start_counters(&self, metrics: &[&str]) -> Result<CounterSession<'_>, Error> {
        let perf =
            self.shared.perf.as_ref().ok_or_else(|| {
                Error::Invalid("counter sessions need Caps::perf_counters".into())
            })?;
        let (metrics, replays) = perf.start(self, metrics)?;
        Ok(CounterSession {
            queue: self,
            metrics,
            replays,
        })
    }

    fn counting(&self) -> &Perf {
        self.shared
            .perf
            .as_ref()
            .expect("a counter session's device has Caps::perf_counters")
    }
}

impl CounterSession<'_> {
    /// The session's metrics, in the order of [`PassCounters::Values`].
    pub fn metrics(&self) -> &[CounterMetric] {
        &self.metrics
    }

    /// The frames one collection takes: [`CounterSession::end_frame`] returns
    /// [`CounterFrame::Replay`] this many times less one, then [`CounterFrame::Complete`].
    pub fn replays(&self) -> usize {
        self.replays
    }

    /// Opens a frame: every pass recorded for this session's queue, on any thread, from now
    /// until [`CounterSession::end_frame`] is a counter range. The frame's lists are submitted
    /// before the end.
    ///
    /// The SDK's profiler isolates each range: a pass starts once the one before has drained,
    /// so passes that overlap in other frames don't in this one, and each is timed exactly.
    /// Counters are the whole GPU's while a range is open, other queues' and processes' work
    /// included.
    pub fn begin_frame(&mut self) -> Result<(), Error> {
        self.queue.counting().begin_frame(self.queue)
    }

    /// Closes the frame, waits for everything submitted to the queue and collects its
    /// counters. Until the collection has all its metrics the SDK asks for the frame again
    /// ([`CounterFrame::Replay`]): the same passes recorded in the same order on one thread,
    /// from the same scene, view and settings. Then the counters are on the
    /// [`crate::PassResult`]s of this, the collection's last frame ([`CounterFrame::Complete`]),
    /// and the next frame begins a new collection.
    ///
    /// A failed frame, such as a replay whose passes differ from its collection's first frame,
    /// discards its collection; the next frame begins a new one.
    pub fn end_frame(&mut self) -> Result<CounterFrame, Error> {
        self.queue.counting().end_frame(self.queue)
    }
}

impl Drop for CounterSession<'_> {
    fn drop(&mut self) {
        self.queue.counting().stop(self.queue);
    }
}

impl Perf {
    pub(crate) fn new(
        library: Library,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Option<Perf> {
        Perf::create(library, entry, instance, physical, device)
            .inspect_err(|e| log::warn!("Nsight Perf SDK unavailable: {e}"))
            .ok()
    }

    fn create(
        library: Library,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Result<Perf, String> {
        let lib = &library;
        let get_instance_proc_addr = entry.static_fn().get_instance_proc_addr;
        let get_device_proc_addr = instance.fp_v1_0().get_device_proc_addr;
        let mut index = params!(NVPW_VK_Device_GetDeviceIndex_Params {
            physicalDevice: physical.as_raw() as _,
            instance: instance.handle().as_raw() as _,
            device: device.handle().as_raw() as _,
            pfnGetInstanceProcAddr: get_instance_proc_addr as *mut c_void,
            pfnGetDeviceProcAddr: get_device_proc_addr as *mut c_void,
        });
        lib.call(c"NVPW_VK_Device_GetDeviceIndex", &mut index)?;
        let mut supported = params!(NVPW_VK_Profiler_IsGpuSupported_Params {
            deviceIndex: index.deviceIndex,
        });
        lib.call(c"NVPW_VK_Profiler_IsGpuSupported", &mut supported)?;
        if supported.isSupported == 0 {
            return Err(format!(
                "the range profiler does not support this GPU (architecture {}, SKU {}, SLI {}, \
                 CMP {}, WSL {})",
                supported.gpuArchitectureSupportLevel,
                supported.skuSupportLevel,
                supported.sliSupportLevel,
                supported.cmpSupportLevel,
                supported.wslSupportLevel
            ));
        }
        let mut device_names = params!(NVPW_Device_GetNames_Params {
            deviceIndex: index.deviceIndex,
        });
        lib.call(c"NVPW_Device_GetNames", &mut device_names)?;
        // SAFETY: the SDK returns a NUL-terminated chip name that lives as long as it does.
        let chip = unsafe { CStr::from_ptr(device_names.pChipName) }.to_owned();
        let push = lib.entry(c"NVPW_VK_Profiler_CommandBuffer_PushRange")?;
        let pop = lib.entry(c"NVPW_VK_Profiler_CommandBuffer_PopRange")?;

        let mut size = params!(NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize_Params {
            pChipName: chip.as_ptr(),
        });
        lib.call(
            c"NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize",
            &mut size,
        )?;
        let mut evaluator_scratch = vec![0u8; size.scratchBufferSize];
        let mut init = params!(NVPW_VK_MetricsEvaluator_Initialize_Params {
            pScratchBuffer: evaluator_scratch.as_mut_ptr(),
            scratchBufferSize: evaluator_scratch.len(),
            pChipName: chip.as_ptr(),
        });
        lib.call(c"NVPW_VK_MetricsEvaluator_Initialize", &mut init)?;
        Ok(Perf {
            library,
            chip,
            evaluator: init.pMetricsEvaluator,
            _evaluator_scratch: evaluator_scratch,
            push,
            pop,
            instance: instance.handle(),
            physical,
            device: device.handle(),
            get_instance_proc_addr,
            get_device_proc_addr,
            recording: AtomicUsize::new(0),
            counting: Mutex::new(None),
        })
    }

    /// Opens a range for the pass `name` in `cmd` when a counter frame is open on `kind`'s
    /// queue, and returns where its counters land. A pass past [`MAX_COUNTER_PASSES`] opens
    /// none and is [`PassCounters::Dropped`].
    pub(super) fn push_range(
        &self,
        kind: QueueKind,
        cmd: vk::CommandBuffer,
        name: &'static str,
    ) -> Option<Counted> {
        if self.recording.load(Ordering::Acquire) != kind.index() + 1 {
            return None;
        }
        let mut guard = self.counting.lock().unwrap();
        // The frame may have ended since the first look.
        if self.recording.load(Ordering::Acquire) != kind.index() + 1 {
            return None;
        }
        let f = &mut guard
            .as_mut()
            .expect("an open counter frame has a session")
            .frame;
        f.threads.insert(std::thread::current().id());
        let slot = Arc::new(CounterSlot::new());
        f.passes += 1;
        if f.passes > MAX_COUNTER_PASSES {
            let _ = slot.set(Some(PassCounters::Dropped));
            return Some(Counted {
                slot,
                ranged: false,
                open: Arc::clone(&f.open),
            });
        }
        let k = f.occurrences.entry(name).or_default();
        let range = CString::new(format!("{name}#{k}")).expect("pass names hold no NUL");
        *k += 1;
        assert!(
            range.as_bytes().len() < MAX_RANGE_NAME as usize,
            "pass {name} is too long for a counter range"
        );
        let mut p = params!(NVPW_VK_Profiler_CommandBuffer_PushRange_Params {
            commandBuffer: cmd.as_raw() as _,
            pRangeName: range.as_ptr(),
            rangeNameLength: range.as_bytes().len(),
        });
        // SAFETY: recording `cmd`; valid params.
        let status = unsafe { (self.push)(&mut p) };
        if status != SUCCESS {
            f.failed
                .get_or_insert_with(|| format!("PushRange({name}): {}", status_text(status)));
            return None;
        }
        f.ranges.push((range, Arc::clone(&slot)));
        Some(Counted {
            slot,
            ranged: true,
            open: Arc::clone(&f.open),
        })
    }

    pub(super) fn pop_range(&self, cmd: vk::CommandBuffer) {
        let mut p = params!(NVPW_VK_Profiler_CommandBuffer_PopRange_Params {
            commandBuffer: cmd.as_raw() as _,
        });
        // SAFETY: recording `cmd`, which opened the range.
        let status = unsafe { (self.pop)(&mut p) };
        if status != SUCCESS
            && let Some(c) = self.counting.lock().unwrap().as_mut()
        {
            c.frame
                .failed
                .get_or_insert_with(|| format!("PopRange: {}", status_text(status)));
        }
    }

    /// Schedules `names` for `queue` and begins the SDK session; returns the metrics and the
    /// frames a collection takes.
    fn start(&self, queue: &Queue, names: &[&str]) -> Result<(Vec<CounterMetric>, usize), Error> {
        let mut guard = self.counting.lock().unwrap();
        if let Some(c) = &*guard {
            return Err(Error::Invalid(format!(
                "a counter session is open on the {:?} queue",
                c.kind
            )));
        }
        let availability = self.availability(queue).map_err(sdk_error)?;
        let (metrics, plan, replays) = self.configure(&availability, names).map_err(sdk_error)?;
        if plan.requests.is_empty() {
            return Err(sdk_error(format!(
                "the {} exposes none of the metrics asked for",
                self.chip.to_string_lossy()
            )));
        }
        let live = self.begin_live(queue, &plan).map_err(sdk_error)?;
        *guard = Some(Counting {
            kind: queue.kind(),
            queue: queue.raw(),
            plan,
            replays,
            live: Some(live),
            frame: Frame::default(),
        });
        Ok((metrics, replays))
    }

    /// Which raw counters the driver exposes on `queue`.
    fn availability(&self, queue: &Queue) -> Result<Aligned, String> {
        let mut p = params!(NVPW_VK_Profiler_Queue_GetCounterAvailability_Params {
            instance: self.instance.as_raw() as _,
            physicalDevice: self.physical.as_raw() as _,
            device: self.device.as_raw() as _,
            queue: queue.raw().as_raw() as _,
            pfnGetInstanceProcAddr: self.get_instance_proc_addr as *mut c_void,
            pfnGetDeviceProcAddr: self.get_device_proc_addr as *mut c_void,
        });
        let _submits = queue.lock_submits();
        self.library
            .call(c"NVPW_VK_Profiler_Queue_GetCounterAvailability", &mut p)?;
        let mut image = Aligned::zeroed(p.counterAvailabilityImageSize);
        p.pCounterAvailabilityImage = image.mut_ptr();
        self.library
            .call(c"NVPW_VK_Profiler_Queue_GetCounterAvailability", &mut p)?;
        Ok(image)
    }

    fn begin_frame(&self, queue: &Queue) -> Result<(), Error> {
        let mut guard = self.counting.lock().unwrap();
        let c = guard.as_mut().expect("a counter session is open");
        if self.recording.load(Ordering::Acquire) != 0 {
            return Err(Error::Invalid("a counter frame is already open".into()));
        }
        if c.live.is_none() {
            c.live = Some(self.begin_live(queue, &c.plan).map_err(sdk_error)?);
        }
        let live = c.live.as_mut().unwrap();
        let begun = (|| {
            if !live.configured {
                let _submits = queue.lock_submits();
                self.library.call(
                    c"NVPW_VK_Profiler_Queue_SetConfig",
                    &mut params!(NVPW_VK_Profiler_Queue_SetConfig_Params {
                        queue: c.queue.as_raw() as _,
                        pConfig: c.plan.config.ptr(),
                        configSize: c.plan.config.len,
                        minNestingLevel: 1,
                        numNestingLevels: 1,
                        passIndex: 0,
                        targetNestingLevel: 1,
                    }),
                )?;
                live.configured = true;
            }
            let _submits = queue.lock_submits();
            self.library.call(
                c"NVPW_VK_Profiler_Queue_BeginPass",
                &mut params!(NVPW_VK_Profiler_Queue_BeginPass_Params {
                    queue: c.queue.as_raw() as _,
                }),
            )
        })();
        if let Err(e) = begun {
            let live = c.live.take().unwrap();
            let _submits = queue.lock_submits();
            self.end_live(live, c.queue);
            return Err(sdk_error(e));
        }
        c.frame = Frame {
            open: Arc::new(AtomicBool::new(true)),
            ..Frame::default()
        };
        self.recording
            .store(queue.kind().index() + 1, Ordering::Release);
        Ok(())
    }

    fn end_frame(&self, queue: &Queue) -> Result<CounterFrame, Error> {
        let mut guard = self.counting.lock().unwrap();
        let c = guard.as_mut().expect("a counter session is open");
        if self.recording.load(Ordering::Acquire) != queue.kind().index() + 1 {
            return Err(Error::Invalid(
                "no counter frame is open on this queue".into(),
            ));
        }
        self.recording.store(0, Ordering::Release);
        let frame = std::mem::take(&mut c.frame);
        let last = close(queue, &frame);
        let result = self.collect(queue, c, &frame, last);
        for (_, slot) in &frame.ranges {
            let _ = slot.set(None);
        }
        if result.is_err() {
            let live = c.live.take().unwrap();
            let _submits = queue.lock_submits();
            self.end_live(live, c.queue);
        }
        result
    }

    /// Waits for the frame, ends the pass it was and decodes it; with every replay collected,
    /// sets the frame's counters.
    fn collect(
        &self,
        queue: &Queue,
        c: &mut Counting,
        f: &Frame,
        last: TimelinePoint,
    ) -> Result<CounterFrame, Error> {
        let live = c.live.as_mut().unwrap();
        // The pass ends even when the wait fails, so the session can end after it.
        let waited = queue.wait(last, TIMEOUT);
        {
            let _submits = queue.lock_submits();
            self.library
                .call(
                    c"NVPW_VK_Profiler_Queue_EndPass",
                    &mut params!(NVPW_VK_Profiler_Queue_EndPass_Params {
                        queue: c.queue.as_raw() as _,
                    }),
                )
                .map_err(sdk_error)?;
        }
        waited?;
        if let Some(e) = &f.failed {
            return Err(sdk_error(e.clone()));
        }
        if c.replays > 1 {
            // Replays line ranges up by name and occurrence, which only recording on one
            // thread keeps in order.
            if f.threads.len() > 1 {
                return Err(sdk_error(format!(
                    "the frame's passes were recorded on {} threads; with {} replays they \
                     must be recorded on one",
                    f.threads.len(),
                    c.replays
                )));
            }
            // A range missing from a replay would evaluate from the replays that have it.
            let names: Vec<CString> = f.ranges.iter().map(|(n, _)| n.clone()).collect();
            match &live.first {
                None => live.first = Some(names),
                Some(first) if *first != names => {
                    return Err(sdk_error(format!(
                        "a replay recorded passes {:?}, its collection's first frame {:?}",
                        names, first
                    )));
                }
                Some(_) => {}
            }
        }
        let start = Instant::now();
        let all = loop {
            let mut p = params!(NVPW_VK_Profiler_Queue_DecodeCounters_Params {
                queue: c.queue.as_raw() as _,
                counterDataImageSize: live.image.len,
                pCounterDataImage: live.image.mut_ptr(),
                counterDataScratchBufferSize: live.scratch.len,
                pCounterDataScratchBuffer: live.scratch.mut_ptr(),
            });
            self.library
                .call(c"NVPW_VK_Profiler_Queue_DecodeCounters", &mut p)
                .map_err(sdk_error)?;
            live.dropped += p.numRangesDropped + p.numTraceBytesDropped;
            if p.onePassCollected != 0 {
                break p.allPassesCollected != 0;
            }
            if start.elapsed() > TIMEOUT {
                return Err(Error::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        if !all {
            return Ok(CounterFrame::Replay);
        }
        if live.dropped != 0 {
            return Err(sdk_error(format!(
                "the SDK dropped {} ranges or trace bytes",
                live.dropped
            )));
        }
        let mut found = self.evaluate(&c.plan, &live.image).map_err(sdk_error)?;
        // A range is missing when its list wasn't submitted in the frame; `end_frame` sets
        // those to `None`.
        for (range, slot) in &f.ranges {
            if let Some(values) = found.remove(range) {
                let _ = slot.set(Some(PassCounters::Values(values)));
            }
        }
        self.initialize(&c.plan, &mut live.image, &mut live.scratch)
            .map_err(sdk_error)?;
        live.configured = false;
        live.first = None;
        Ok(CounterFrame::Complete)
    }

    /// Ends the session: an open frame is waited for and discarded.
    fn stop(&self, queue: &Queue) {
        let mut guard = self.counting.lock().unwrap();
        let Some(c) = guard.take() else {
            return;
        };
        let open = self.recording.swap(0, Ordering::AcqRel) != 0;
        let Some(live) = c.live else {
            return;
        };
        if open {
            if let Err(e) = queue.wait(close(queue, &c.frame), TIMEOUT) {
                log::warn!("Nsight Perf SDK: the abandoned counter frame's submits: {e}");
            }
            let _submits = queue.lock_submits();
            if let Err(e) = self.library.call(
                c"NVPW_VK_Profiler_Queue_EndPass",
                &mut params!(NVPW_VK_Profiler_Queue_EndPass_Params {
                    queue: c.queue.as_raw() as _,
                }),
            ) {
                log::warn!("Nsight Perf SDK: {e}");
            }
            for (_, slot) in &c.frame.ranges {
                let _ = slot.set(None);
            }
        }
        let _submits = queue.lock_submits();
        self.end_live(live, c.queue);
    }

    fn image_options(plan: &Plan) -> sys::NVPW_VK_Profiler_CounterDataImageOptions {
        sys::NVPW_VK_Profiler_CounterDataImageOptions {
            structSize: <sys::NVPW_VK_Profiler_CounterDataImageOptions as StructSize>::SIZE,
            pCounterDataPrefix: plan.prefix.ptr(),
            counterDataPrefixSize: plan.prefix.len,
            maxNumRanges: MAX_COUNTER_PASSES as u32,
            // As NVIDIA's own range profiler sizes it.
            maxNumRangeTreeNodes: 2 * MAX_COUNTER_PASSES as u32,
            maxRangeNameLength: MAX_RANGE_NAME,
        }
    }

    /// An empty counter data image for `plan`, and its scratch buffer.
    fn counter_data(&self, plan: &Plan) -> Result<(Aligned, Aligned), String> {
        let lib = &self.library;
        let options = Perf::image_options(plan);
        let mut size = params!(NVPW_VK_Profiler_CounterDataImage_CalculateSize_Params {
            counterDataImageOptionsSize: options.structSize,
            pOptions: &options,
        });
        lib.call(
            c"NVPW_VK_Profiler_CounterDataImage_CalculateSize",
            &mut size,
        )?;
        let mut image = Aligned::zeroed(size.counterDataImageSize);
        self.initialize_image(plan, &mut image)?;
        let mut scratch_size = params!(
            NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize_Params {
                counterDataImageSize: image.len,
                pCounterDataImage: image.mut_ptr(),
            }
        );
        lib.call(
            c"NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize",
            &mut scratch_size,
        )?;
        let mut scratch = Aligned::zeroed(scratch_size.counterDataScratchBufferSize);
        self.initialize_scratch(&mut image, &mut scratch)?;
        Ok((image, scratch))
    }

    /// Empties `image` and `scratch` for the next collection.
    fn initialize(
        &self,
        plan: &Plan,
        image: &mut Aligned,
        scratch: &mut Aligned,
    ) -> Result<(), String> {
        self.initialize_image(plan, image)?;
        self.initialize_scratch(image, scratch)
    }

    fn initialize_image(&self, plan: &Plan, image: &mut Aligned) -> Result<(), String> {
        let options = Perf::image_options(plan);
        self.library.call(
            c"NVPW_VK_Profiler_CounterDataImage_Initialize",
            &mut params!(NVPW_VK_Profiler_CounterDataImage_Initialize_Params {
                counterDataImageOptionsSize: options.structSize,
                pOptions: &options,
                counterDataImageSize: image.len,
                pCounterDataImage: image.mut_ptr(),
            }),
        )
    }

    fn initialize_scratch(&self, image: &mut Aligned, scratch: &mut Aligned) -> Result<(), String> {
        self.library.call(
            c"NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer",
            &mut params!(
                NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params {
                    counterDataImageSize: image.len,
                    pCounterDataImage: image.mut_ptr(),
                    counterDataScratchBufferSize: scratch.len,
                    pCounterDataScratchBuffer: scratch.mut_ptr(),
                }
            ),
        )
    }

    /// Begins an SDK session on `queue` and its service thread.
    fn begin_live(&self, queue: &Queue, plan: &Plan) -> Result<Live, String> {
        let lib = &self.library;
        let (image, scratch) = self.counter_data(plan)?;
        let mut trace = params!(NVPW_VK_Profiler_CalcTraceBufferSize_Params {
            maxRangesPerPass: MAX_COUNTER_PASSES,
            avgRangeNameLength: MAX_RANGE_NAME as usize,
        });
        lib.call(c"NVPW_VK_Profiler_CalcTraceBufferSize", &mut trace)?;
        let service_fn = lib.entry::<sys::NVPW_VK_Queue_ServicePendingGpuOperations_Params>(
            c"NVPW_VK_Queue_ServicePendingGpuOperations",
        )?;
        let raw = queue.raw();
        {
            let _submits = queue.lock_submits();
            lib.call(
                c"NVPW_VK_Profiler_Queue_BeginSession",
                &mut params!(NVPW_VK_Profiler_Queue_BeginSession_Params {
                    device: self.device.as_raw() as _,
                    queue: raw.as_raw() as _,
                    // Each frame is waited for before the next begins.
                    numTraceBuffers: 1,
                    traceBufferSize: trace.traceBufferSize,
                    maxRangesPerPass: MAX_COUNTER_PASSES,
                    maxLaunchesPerPass: MAX_COUNTER_PASSES,
                    instance: self.instance.as_raw() as _,
                    physicalDevice: self.physical.as_raw() as _,
                    pfnGetInstanceProcAddr: self.get_instance_proc_addr as *mut c_void,
                    pfnGetDeviceProcAddr: self.get_device_proc_addr as *mut c_void,
                }),
            )?;
        }
        let queue_raw = raw.as_raw();
        let service = std::thread::Builder::new()
            .name("nvperf service".into())
            .spawn(move || {
                crate::name_thread!("nvperf service");
                let mut p = params!(NVPW_VK_Queue_ServicePendingGpuOperations_Params {
                    queue: queue_raw as _,
                    // Until the session ends.
                    numOperations: 0,
                    timeout: u32::MAX,
                });
                // SAFETY: valid params for the session's queue.
                let status = unsafe { service_fn(&mut p) };
                if status != SUCCESS {
                    log::warn!(
                        "Nsight Perf SDK: ServicePendingGpuOperations: {}",
                        status_text(status)
                    );
                }
            });
        match service {
            Ok(service) => Ok(Live {
                image,
                scratch,
                service,
                configured: false,
                first: None,
                dropped: 0,
            }),
            Err(e) => {
                let _submits = queue.lock_submits();
                self.end_session(raw);
                Err(e.to_string())
            }
        }
    }

    /// Ends `live`'s session on `queue`, whose submits the caller holds.
    fn end_live(&self, live: Live, queue: vk::Queue) {
        // The service thread returns once the session has ended; when it hasn't, joining
        // would wait forever, so the thread is left to the process.
        if self.end_session(queue) && live.service.join().is_err() {
            log::warn!("Nsight Perf SDK: the service thread panicked");
        }
    }

    /// Whether the session ended.
    fn end_session(&self, queue: vk::Queue) -> bool {
        let mut p = params!(NVPW_VK_Profiler_Queue_EndSession_Params {
            queue: queue.as_raw() as _,
            timeout: TIMEOUT.as_millis() as u32,
        });
        match self
            .library
            .call(c"NVPW_VK_Profiler_Queue_EndSession", &mut p)
        {
            Err(e) => {
                log::warn!("Nsight Perf SDK: {e}");
                false
            }
            Ok(()) if p.timeoutExpired != 0 => {
                log::warn!("Nsight Perf SDK: EndSession timed out");
                false
            }
            Ok(()) => true,
        }
    }

    /// Each range's values, by range name.
    fn evaluate(
        &self,
        plan: &Plan,
        image: &Aligned,
    ) -> Result<HashMap<CString, Vec<Option<f64>>>, String> {
        let lib = &self.library;
        lib.call(
            c"NVPW_MetricsEvaluator_SetDeviceAttributes",
            &mut params!(NVPW_MetricsEvaluator_SetDeviceAttributes_Params {
                pMetricsEvaluator: self.evaluator,
                pCounterDataImage: image.ptr(),
                counterDataImageSize: image.len,
            }),
        )?;
        let mut count = params!(NVPW_CounterData_GetNumRanges_Params {
            pCounterDataImage: image.ptr(),
        });
        lib.call(c"NVPW_CounterData_GetNumRanges", &mut count)?;
        let mut found = HashMap::with_capacity(count.numRanges);
        let mut values = vec![0.0f64; plan.requests.len()];
        for range in 0..count.numRanges {
            let mut desc = params!(NVPW_Profiler_CounterData_GetRangeDescriptions_Params {
                pCounterDataImage: image.ptr(),
                rangeIndex: range,
            });
            lib.call(c"NVPW_Profiler_CounterData_GetRangeDescriptions", &mut desc)?;
            let mut parts = vec![ptr::null(); desc.numDescriptions];
            desc.ppDescriptions = parts.as_mut_ptr();
            lib.call(c"NVPW_Profiler_CounterData_GetRangeDescriptions", &mut desc)?;
            // Passes don't nest, so each range is one name.
            let [name] = parts[..] else {
                return Err(format!("range {range} has {} names", parts.len()));
            };
            // SAFETY: the SDK returns NUL-terminated names inside `image`.
            let name = unsafe { CStr::from_ptr(name) }.to_owned();
            lib.call(
                c"NVPW_MetricsEvaluator_EvaluateToGpuValues",
                &mut params!(NVPW_MetricsEvaluator_EvaluateToGpuValues_Params {
                    pMetricsEvaluator: self.evaluator,
                    pMetricEvalRequests: plan.requests.as_ptr(),
                    numMetricEvalRequests: plan.requests.len(),
                    metricEvalRequestStructSize: <sys::NVPW_MetricEvalRequest as StructSize>::SIZE,
                    metricEvalRequestStrideSize: size_of::<sys::NVPW_MetricEvalRequest>(),
                    pCounterDataImage: image.ptr(),
                    counterDataImageSize: image.len,
                    rangeIndex: range,
                    pMetricValues: values.as_mut_ptr(),
                }),
            )?;
            let mut all = vec![None; plan.metrics];
            for (&at, &v) in plan.available.iter().zip(&values) {
                all[at] = Some(v);
            }
            found.insert(name, all);
        }
        Ok(found)
    }

    /// Ends a session its handle never closed and frees the evaluator; the device is idle.
    pub(crate) fn shut_down(&self) {
        if let Some(c) = self.counting.lock().unwrap().take()
            && let Some(live) = c.live
        {
            if self.recording.load(Ordering::Acquire) != 0 {
                let _ = self.library.call(
                    c"NVPW_VK_Profiler_Queue_EndPass",
                    &mut params!(NVPW_VK_Profiler_Queue_EndPass_Params {
                        queue: c.queue.as_raw() as _,
                    }),
                );
            }
            self.end_live(live, c.queue);
        }
        let _ = self.library.call(
            c"NVPW_MetricsEvaluator_Destroy",
            &mut params!(NVPW_MetricsEvaluator_Destroy_Params {
                pMetricsEvaluator: self.evaluator,
            }),
        );
    }

    /// Schedules the raw counters of every metric in `names` the chip and driver expose (by
    /// `availability`) and makes the config image and counter data prefix for them; returns
    /// every metric, the plan and its replays.
    fn configure(
        &self,
        availability: &Aligned,
        names: &[&str],
    ) -> Result<(Vec<CounterMetric>, Plan, usize), String> {
        let lib = &self.library;
        let mut create = params!(NVPW_VK_RawCounterConfig_Create_Params {
            pChipName: self.chip.as_ptr(),
            activityKind: sys::NVPA_ActivityKind_NVPA_ACTIVITY_KIND_PROFILER as u32,
        });
        lib.call(c"NVPW_VK_RawCounterConfig_Create", &mut create)?;
        let config = create.pRawCounterConfig;
        let mut builder = params!(NVPW_CounterDataBuilder_Create_Params {
            pChipName: self.chip.as_ptr(),
        });
        let made = lib
            .call(c"NVPW_CounterDataBuilder_Create", &mut builder)
            .and_then(|()| {
                let s = Scheduler {
                    lib,
                    evaluator: self.evaluator,
                    config,
                    builder: builder.pCounterDataBuilder,
                };
                let made = s.schedule(availability, names);
                let _ = lib.call(
                    c"NVPW_CounterDataBuilder_Destroy",
                    &mut params!(NVPW_CounterDataBuilder_Destroy_Params {
                        pCounterDataBuilder: builder.pCounterDataBuilder,
                    }),
                );
                made
            });
        let _ = lib.call(
            c"NVPW_RawCounterConfig_Destroy",
            &mut params!(NVPW_RawCounterConfig_Destroy_Params {
                pRawCounterConfig: config,
            }),
        );
        made
    }
}

/// Builds a session's config image and counter data prefix.
struct Scheduler<'a> {
    lib: &'a Library,
    evaluator: *mut sys::NVPW_MetricsEvaluator,
    config: *mut sys::NVPW_RawCounterConfig,
    builder: *mut sys::NVPA_CounterDataBuilder,
}

/// A raw counter in the SDK's default domain for it, with its value per unit instance when
/// `instances`.
fn raw_request(name: *const c_char, instances: bool) -> sys::NVPW_RawCounterRequest {
    sys::NVPW_RawCounterRequest {
        pRawCounterName: name,
        domain: sys::NVPW_RawCounterDomain_NVPW_RAW_COUNTER_DOMAIN_INVALID as u32,
        keepInstances: u8::from(instances),
        pPriv: ptr::null_mut(),
    }
}

const RAW_REQUEST_SIZE: usize = <sys::NVPW_RawCounterRequest as StructSize>::SIZE;

impl Scheduler<'_> {
    fn schedule(
        &self,
        availability: &Aligned,
        names: &[&str],
    ) -> Result<(Vec<CounterMetric>, Plan, usize), String> {
        let lib = self.lib;
        let config = self.config;
        lib.call(
            c"NVPW_RawCounterConfig_SetCounterAvailability",
            &mut params!(NVPW_RawCounterConfig_SetCounterAvailability_Params {
                pRawCounterConfig: config,
                pCounterAvailabilityImage: availability.ptr(),
            }),
        )?;
        let mut domains = params!(
            NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains_Params {
                pRawCounterConfig: config,
            }
        );
        lib.call(
            c"NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains",
            &mut domains,
        )?;
        let mut all = vec![0; domains.numAvailableDomains];
        domains.pAvailableDomains = all.as_mut_ptr();
        lib.call(
            c"NVPW_RawCounterConfig_GetAllAvailableRawCounterDomains",
            &mut domains,
        )?;
        lib.call(
            c"NVPW_RawCounterConfig_BeginPassGroup",
            &mut params!(NVPW_RawCounterConfig_BeginPassGroup_Params {
                pRawCounterConfig: config,
                numDomains: all.len(),
                pDomains: all.as_ptr(),
            }),
        )?;
        let mut metrics = Vec::with_capacity(names.len());
        let mut requests = Vec::new();
        let mut available = Vec::new();
        for (at, &name) in names.iter().enumerate() {
            let c_name = CString::new(name).map_err(|e| format!("metric {name:?}: {e}"))?;
            let scheduled = self.add_metric(&c_name);
            metrics.push(CounterMetric {
                name: name.to_owned(),
                available: scheduled.is_some(),
                unit: scheduled
                    .as_ref()
                    .map(|(_, unit)| unit.clone())
                    .unwrap_or_default(),
            });
            if let Some((request, _)) = scheduled {
                requests.push(request);
                available.push(at);
            }
        }
        lib.call(
            c"NVPW_RawCounterConfig_EndPassGroup",
            &mut params!(NVPW_RawCounterConfig_EndPassGroup_Params {
                pRawCounterConfig: config,
                numDomains: all.len(),
                pDomains: all.as_ptr(),
            }),
        )?;
        lib.call(
            c"NVPW_RawCounterConfig_GenerateConfigImage",
            &mut params!(NVPW_RawCounterConfig_GenerateConfigImage_Params {
                pRawCounterConfig: config,
            }),
        )?;
        let mut get = params!(NVPW_RawCounterConfig_GetConfigImage_Params {
            pRawCounterConfig: config,
        });
        lib.call(c"NVPW_RawCounterConfig_GetConfigImage", &mut get)?;
        let mut image = Aligned::zeroed(get.bytesCopied);
        get.bytesAllocated = image.len;
        get.pBuffer = image.mut_ptr();
        lib.call(c"NVPW_RawCounterConfig_GetConfigImage", &mut get)?;
        let mut passes = params!(NVPW_RawCounterConfig_GetNumPasses_Params {
            pRawCounterConfig: config,
        });
        lib.call(c"NVPW_RawCounterConfig_GetNumPasses", &mut passes)?;
        log::info!(
            "Nsight Perf SDK: {} of {} metrics in {} replays",
            requests.len(),
            names.len(),
            passes.numPasses
        );
        let mut get = params!(NVPW_CounterDataBuilder_GetCounterDataPrefix_Params {
            pCounterDataBuilder: self.builder,
        });
        lib.call(c"NVPW_CounterDataBuilder_GetCounterDataPrefix", &mut get)?;
        let mut prefix = Aligned::zeroed(get.bytesCopied);
        get.bytesAllocated = prefix.len;
        get.pBuffer = prefix.mut_ptr();
        lib.call(c"NVPW_CounterDataBuilder_GetCounterDataPrefix", &mut get)?;
        let plan = Plan {
            requests,
            available,
            metrics: names.len(),
            config: image,
            prefix,
        };
        Ok((metrics, plan, passes.numPasses))
    }

    /// Adds the raw counters `name` needs, with its evaluation request and unit; `None` (with
    /// a log line) when the chip or driver doesn't expose them.
    fn add_metric(&self, name: &CStr) -> Option<(sys::NVPW_MetricEvalRequest, String)> {
        let lib = self.lib;
        let unavailable = |why: String| {
            log::info!(
                "Nsight Perf SDK: {} is not available: {why}",
                name.to_string_lossy()
            );
        };
        // SAFETY: a plain C struct the call fills.
        let mut request: sys::NVPW_MetricEvalRequest = unsafe { zeroed() };
        let request_size = <sys::NVPW_MetricEvalRequest as StructSize>::SIZE;
        if let Err(e) = lib.call(
            c"NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest",
            &mut params!(
                NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params {
                    pMetricsEvaluator: self.evaluator,
                    pMetricName: name.as_ptr(),
                    pMetricEvalRequest: &mut request,
                    metricEvalRequestStructSize: request_size,
                }
            ),
        ) {
            unavailable(e);
            return None;
        }
        let mut deps = params!(NVPW_MetricsEvaluator_GetMetricRawDependencies_Params {
            pMetricsEvaluator: self.evaluator,
            pMetricEvalRequests: &request,
            numMetricEvalRequests: 1,
            metricEvalRequestStructSize: request_size,
            metricEvalRequestStrideSize: size_of::<sys::NVPW_MetricEvalRequest>(),
        });
        let raw_names = lib
            .call(c"NVPW_MetricsEvaluator_GetMetricRawDependencies", &mut deps)
            .and_then(|()| {
                let mut required = vec![ptr::null(); deps.numRawDependencies];
                let mut optional = vec![ptr::null(); deps.numOptionalRawDependencies];
                deps.ppRawDependencies = required.as_mut_ptr();
                deps.ppOptionalRawDependencies = optional.as_mut_ptr();
                lib.call(c"NVPW_MetricsEvaluator_GetMetricRawDependencies", &mut deps)?;
                Ok((required, optional))
            });
        let (required, optional) = match raw_names {
            Ok(r) => r,
            Err(e) => {
                unavailable(e);
                return None;
            }
        };
        // A maximum or minimum over a unit's instances needs each instance's value; every
        // other rollup reads the sum, and instances multiply the counter data each frame
        // initializes.
        let instances = request.metricType != sys::NVPW_MetricType_NVPW_METRIC_TYPE_RATIO as u8
            && [
                sys::NVPW_RollupOp_NVPW_ROLLUP_OP_MAX,
                sys::NVPW_RollupOp_NVPW_ROLLUP_OP_MIN,
            ]
            .contains(&i32::from(request.rollupOp));
        let required: Vec<_> = required
            .iter()
            .map(|&n| raw_request(n, instances))
            .collect();
        match self.schedulable(&required) {
            Ok(true) => {}
            Ok(false) => {
                unavailable("the driver exposes too few of its raw counters".into());
                return None;
            }
            Err(e) => {
                unavailable(e);
                return None;
            }
        }
        if let Err(e) = self.add(&required) {
            unavailable(e);
            return None;
        }
        // An optional counter refines the metric where the driver exposes it (the SDK's own
        // config builder treats a failure to add one as no failure).
        for &o in &optional {
            let _ = self.add(&[raw_request(o, instances)]);
        }
        match unit(lib, self.evaluator, &request) {
            Ok(unit) => Some((request, unit)),
            Err(e) => {
                unavailable(e);
                None
            }
        }
    }

    /// Whether the driver exposes the raw counters `requests` needs.
    fn schedulable(&self, requests: &[sys::NVPW_RawCounterRequest]) -> Result<bool, String> {
        let mut p = params!(NVPW_RawCounterConfig_AreRawCountersSchedulable_Params {
            pRawCounterConfig: self.config,
            rawCounterRequestStructSize: RAW_REQUEST_SIZE,
            numRawCounterRequests: requests.len(),
            pRawCounterRequests: requests.as_ptr(),
        });
        self.lib
            .call(c"NVPW_RawCounterConfig_AreRawCountersSchedulable", &mut p)?;
        Ok(p.schedulable != 0)
    }

    fn add(&self, requests: &[sys::NVPW_RawCounterRequest]) -> Result<(), String> {
        self.lib.call(
            c"NVPW_RawCounterConfig_AddRawCounters",
            &mut params!(NVPW_RawCounterConfig_AddRawCounters_Params {
                pRawCounterConfig: self.config,
                rawCounterRequestStructSize: RAW_REQUEST_SIZE,
                numRawCounterRequests: requests.len(),
                pRawCounterRequests: requests.as_ptr(),
            }),
        )?;
        self.lib.call(
            c"NVPW_CounterDataBuilder_AddRawCounters",
            &mut params!(NVPW_CounterDataBuilder_AddRawCounters_Params {
                pCounterDataBuilder: self.builder,
                rawCounterRequestStructSize: RAW_REQUEST_SIZE,
                numRawCounterRequests: requests.len(),
                pRawCounterRequests: requests.as_ptr(),
            }),
        )
    }
}

/// The metric's dimensional units, such as `nanosecond` or `percent`, joined as a product with
/// negative powers after a `/`.
fn unit(
    lib: &Library,
    evaluator: *mut sys::NVPW_MetricsEvaluator,
    request: &sys::NVPW_MetricEvalRequest,
) -> Result<String, String> {
    let mut p = params!(NVPW_MetricsEvaluator_GetMetricDimUnits_Params {
        pMetricsEvaluator: evaluator,
        pMetricEvalRequest: request,
        metricEvalRequestStructSize: <sys::NVPW_MetricEvalRequest as StructSize>::SIZE,
        dimUnitFactorStructSize: <sys::NVPW_DimUnitFactor as StructSize>::SIZE,
    });
    lib.call(c"NVPW_MetricsEvaluator_GetMetricDimUnits", &mut p)?;
    // SAFETY: plain C structs the call fills.
    let mut factors = vec![unsafe { zeroed::<sys::NVPW_DimUnitFactor>() }; p.numDimUnits];
    p.pDimUnits = factors.as_mut_ptr();
    lib.call(c"NVPW_MetricsEvaluator_GetMetricDimUnits", &mut p)?;
    let name = |f: &sys::NVPW_DimUnitFactor| -> Result<String, String> {
        let mut s = params!(NVPW_MetricsEvaluator_DimUnitToString_Params {
            pMetricsEvaluator: evaluator,
            dimUnit: f.dimUnit,
        });
        lib.call(c"NVPW_MetricsEvaluator_DimUnitToString", &mut s)?;
        // SAFETY: the SDK returns a NUL-terminated name it owns.
        let text = unsafe { CStr::from_ptr(s.pSingularName) }
            .to_string_lossy()
            .into_owned();
        Ok(match f.exponent.abs() {
            1 => text,
            e => format!("{text}^{e}"),
        })
    };
    let over = factors
        .iter()
        .filter(|f| f.exponent < 0)
        .map(name)
        .collect::<Result<Vec<_>, _>>()?;
    let mut text = factors
        .iter()
        .filter(|f| f.exponent > 0)
        .map(name)
        .collect::<Result<Vec<_>, _>>()?;
    if text.is_empty() && !over.is_empty() {
        text.push("1".into());
    }
    let mut unit = text.join(" * ");
    if !over.is_empty() {
        unit += " / ";
        unit += &over.join(" / ");
    }
    Ok(unit)
}
