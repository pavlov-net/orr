//! Nsight Perf SDK counter frames: per-pass hardware counters for a frame a caller asks for.
//!
//! The SDK is compiled in when `NSIGHT_PERF_SDK` names it at build time
//! ([`crate::DeviceDesc::perf_counters`] then loads it). Without it, or when the GPU or driver
//! refuses, [`crate::Caps::perf_counters`] is false and counter frames are refused.

use std::time::Duration;

use super::queue::Queue;
use crate::{Error, TimelinePoint};

/// The metrics a counter frame collects for every pass, in the order of
/// [`CounterRange::values`].
pub const COUNTER_METRICS: [&str; 6] = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
];

/// How long [`Queue::end_counter_frame`] waits for the frame's submits and their counters.
const DECODE_TIMEOUT: Duration = Duration::from_secs(10);

/// The counters of one counter frame: a range per pass, in submission order.
#[derive(Clone, Debug, Default)]
pub struct CounterReport {
    pub chip: String,
    pub ranges: Vec<CounterRange>,
}

/// One pass's counters.
#[derive(Clone, Debug)]
pub struct CounterRange {
    pub name: String,
    /// Values of [`COUNTER_METRICS`], in its order.
    pub values: Vec<f64>,
}

impl Queue {
    /// Opens a counter frame on this queue: the passes of every list recorded for it from now
    /// until [`Queue::end_counter_frame`] carry Nsight Perf SDK ranges. The frame's lists are
    /// submitted to this queue before the end call. Needs [`crate::Caps::perf_counters`].
    pub fn begin_counter_frame(&self) -> Result<(), Error> {
        let perf = self.perf()?;
        let _guard = self.lock_submits();
        perf.begin_frame(self.kind(), self.raw())
    }

    /// Closes the counter frame whose last submit signals `last`, waits for it and decodes
    /// its counters. The SDK collects the metrics over several replays of the same frame:
    /// `None` asks for the frame to be recorded and submitted again between another
    /// begin and end; `Some` is the finished report.
    pub fn end_counter_frame(&self, last: TimelinePoint) -> Result<Option<CounterReport>, Error> {
        let perf = self.perf()?;
        {
            let _guard = self.lock_submits();
            perf.end_frame(self.kind(), self.raw())?;
        }
        self.wait(last, DECODE_TIMEOUT)?;
        let _guard = self.lock_submits();
        perf.decode(self.raw(), DECODE_TIMEOUT)
    }

    fn perf(&self) -> Result<&Perf, Error> {
        self.shared
            .perf
            .as_ref()
            .ok_or_else(|| Error::Invalid("counter frames need Caps::perf_counters".into()))
    }
}

pub(crate) use imp::{Library, Perf};

#[cfg(not(nvperf))]
mod imp {
    use std::ffi::{CStr, CString};

    use ash::vk;

    use super::CounterReport;
    use crate::Error;
    use crate::vulkan::queue::QueueKind;

    /// Never made: orr-gpu was built without the Nsight Perf SDK.
    pub(crate) enum Library {}

    impl Library {
        pub(crate) fn load(wanted: bool, _: &ash::Entry) -> Option<(Library, Vec<CString>)> {
            if wanted {
                log::warn!(
                    "counter frames asked for, but orr-gpu was built without NSIGHT_PERF_SDK"
                );
            }
            None
        }

        pub(crate) fn device_extensions(
            self,
            _: &ash::Entry,
            _: &ash::Instance,
            _: vk::PhysicalDevice,
            _: impl Fn(&CStr) -> bool,
        ) -> Option<(Library, Vec<CString>)> {
            match self {}
        }
    }

    /// Never made: orr-gpu was built without the Nsight Perf SDK.
    pub(crate) enum Perf {}

    impl Perf {
        pub(crate) fn new(
            library: Library,
            _: &ash::Entry,
            _: &ash::Instance,
            _: vk::PhysicalDevice,
            _: &ash::Device,
        ) -> Option<Perf> {
            match library {}
        }

        pub(crate) fn push_range(&self, _: QueueKind, _: vk::CommandBuffer, _: &CStr) -> bool {
            match *self {}
        }

        pub(crate) fn pop_range(&self, _: vk::CommandBuffer) {
            match *self {}
        }

        pub(crate) fn begin_frame(&self, _: QueueKind, _: vk::Queue) -> Result<(), Error> {
            match *self {}
        }

        pub(crate) fn end_frame(&self, _: QueueKind, _: vk::Queue) -> Result<(), Error> {
            match *self {}
        }

        pub(crate) fn decode(
            &self,
            _: vk::Queue,
            _: std::time::Duration,
        ) -> Result<Option<CounterReport>, Error> {
            match *self {}
        }

        pub(crate) fn shut_down(&self) {
            match *self {}
        }
    }
}

#[cfg(nvperf)]
mod imp {
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::mem::{size_of, zeroed};
    use std::ptr;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use ash::vk::{self, Handle as _};

    use super::{COUNTER_METRICS, CounterRange, CounterReport};
    use crate::Error;
    use crate::vulkan::queue::QueueKind;

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
        include!(concat!(env!("OUT_DIR"), "/nvperf.rs"));
    }

    /// Ranges one replay of a counter frame holds at most, and their longest name.
    const MAX_RANGES: u32 = 512;
    const MAX_RANGE_NAME: u32 = 64;
    /// Submits one replay holds at most.
    const MAX_LAUNCHES: usize = 256;
    const TRACE_BUFFERS: usize = 2;
    const SUCCESS: sys::NVPA_Status = sys::NVPA_Status_NVPA_STATUS_SUCCESS;

    /// A params struct with `structSize` set and every other field zeroed unless named.
    macro_rules! params {
        ($ty:ident { $($field:ident: $value:expr),* $(,)? }) => {
            sys::$ty {
                structSize: size_of::<sys::$ty>(),
                $($field: $value,)*
                // SAFETY: Perf SDK params are plain C structs; zero is every field's default.
                ..unsafe { zeroed() }
            }
        };
    }

    type GetProcAddress =
        unsafe extern "C" fn(*mut sys::NVPA_GetProcAddress_Params) -> sys::NVPA_Status;
    type Entry<P> = unsafe extern "C" fn(*mut P) -> sys::NVPA_Status;

    /// The SDK's host library, loaded from the SDK that orr-gpu was built against.
    pub(crate) struct Library {
        _lib: libloading::Library,
        get: GetProcAddress,
    }

    impl Library {
        /// The SDK's host library and the instance extensions its profiler needs; `None`
        /// (with a warning) when the SDK or the Vulkan loader cannot provide them.
        pub(crate) fn load(wanted: bool, entry: &ash::Entry) -> Option<(Library, Vec<CString>)> {
            if !wanted {
                return None;
            }
            Library::open(entry)
                .inspect_err(|e| log::warn!("Nsight Perf SDK unavailable: {e}"))
                .ok()
        }

        fn open(entry: &ash::Entry) -> Result<(Library, Vec<CString>), String> {
            let path = env!("ORR_NVPERF_LIBRARY");
            // SAFETY: the SDK's host library runs no code at load beyond its own setup. Its
            // directory goes on the search path so it finds the target library beside it.
            let lib = unsafe {
                libloading::os::windows::Library::load_with_flags(
                    path,
                    libloading::os::windows::LOAD_WITH_ALTERED_SEARCH_PATH,
                )
            }
            .map_err(|e| format!("{path}: {e}"))?;
            let lib = libloading::Library::from(lib);
            // SAFETY: the export has this signature in every SDK release.
            let get = unsafe { lib.get::<GetProcAddress>(b"NVPA_GetProcAddress\0") }
                .map(|s| *s)
                .map_err(|e| format!("NVPA_GetProcAddress: {e}"))?;
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
            let mut p = params!(NVPA_GetProcAddress_Params {
                funcName: name.as_ptr(),
            });
            // SAFETY: valid params; the library stays loaded while `self` lives.
            let status = unsafe { (self.get)(&mut p) };
            match p.funcPtr {
                Some(f) if status == SUCCESS => {
                    // SAFETY: every NVPW entry point takes one params pointer and returns a
                    // status; `P` is the params struct named after `name`.
                    Ok(unsafe { std::mem::transmute::<unsafe extern "C" fn(), Entry<P>>(f) })
                }
                _ => Err(format!(
                    "{} not found (status {status})",
                    name.to_string_lossy()
                )),
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
                    "{} failed (status {status})",
                    name.to_string_lossy()
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

    /// # Safety
    /// `list` points at `n` NUL-terminated strings.
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

    /// The configuration every counter frame uses, built once per device.
    pub(crate) struct Perf {
        library: Library,
        chip: String,
        config: Vec<u8>,
        prefix: Vec<u8>,
        evaluator: *mut sys::NVPW_MetricsEvaluator,
        _evaluator_scratch: Vec<u8>,
        requests: Vec<sys::NVPW_MetricEvalRequest>,
        push: Entry<sys::NVPW_VK_Profiler_CommandBuffer_PushRange_Params>,
        pop: Entry<sys::NVPW_VK_Profiler_CommandBuffer_PopRange_Params>,
        instance: vk::Instance,
        device: vk::Device,
        get_instance_proc_addr: vk::PFN_vkGetInstanceProcAddr,
        get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
        /// The kind index plus one of the queue whose lists take ranges; 0 outside a frame.
        recording: AtomicUsize,
        session: Mutex<Option<Session>>,
    }

    // SAFETY: the evaluator is used under `session`'s lock or during creation only, and the
    // SDK's entry points may be called from any thread.
    unsafe impl Send for Perf {}
    // SAFETY: as above.
    unsafe impl Sync for Perf {}

    /// A profiler session on one queue, from the first counter frame until its report.
    struct Session {
        queue: vk::Queue,
        image: Vec<u8>,
        scratch: Vec<u8>,
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
                .inspect_err(|e| log::warn!("Nsight Perf SDK: {e}"))
                .ok()
        }

        fn create(
            library: Library,
            entry: &ash::Entry,
            instance: &ash::Instance,
            physical: vk::PhysicalDevice,
            device: &ash::Device,
        ) -> Result<Perf, String> {
            let get_instance_proc_addr = entry.static_fn().get_instance_proc_addr;
            let get_device_proc_addr = instance.fp_v1_0().get_device_proc_addr;
            let mut index = params!(NVPW_VK_Device_GetDeviceIndex_Params {
                physicalDevice: physical.as_raw() as _,
                instance: instance.handle().as_raw() as _,
                pfnGetInstanceProcAddr: get_instance_proc_addr as *mut c_void,
                pfnGetDeviceProcAddr: get_device_proc_addr as *mut c_void,
            });
            library.call(c"NVPW_VK_Device_GetDeviceIndex", &mut index)?;
            let mut supported = params!(NVPW_VK_Profiler_IsGpuSupported_Params {
                deviceIndex: index.deviceIndex,
            });
            library.call(c"NVPW_VK_Profiler_IsGpuSupported", &mut supported)?;
            if supported.isSupported == 0 {
                return Err("the GPU or driver does not support the range profiler".into());
            }
            let mut names = params!(NVPW_Device_GetNames_Params {
                deviceIndex: index.deviceIndex,
            });
            library.call(c"NVPW_Device_GetNames", &mut names)?;
            // SAFETY: the SDK returns a NUL-terminated chip name that lives as long as it.
            let chip_c = unsafe { CStr::from_ptr(names.pChipName) }.to_owned();

            let mut size = params!(NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize_Params {
                pChipName: chip_c.as_ptr(),
            });
            library.call(
                c"NVPW_VK_MetricsEvaluator_CalculateScratchBufferSize",
                &mut size,
            )?;
            let mut evaluator_scratch = vec![0u8; size.scratchBufferSize];
            let mut init = params!(NVPW_VK_MetricsEvaluator_Initialize_Params {
                pScratchBuffer: evaluator_scratch.as_mut_ptr(),
                scratchBufferSize: evaluator_scratch.len(),
                pChipName: chip_c.as_ptr(),
            });
            library.call(c"NVPW_VK_MetricsEvaluator_Initialize", &mut init)?;
            let evaluator = init.pMetricsEvaluator;

            let mut requests = Vec::new();
            let mut raw: Vec<*const c_char> = Vec::new();
            for metric in COUNTER_METRICS {
                let name = CString::new(metric).unwrap();
                // SAFETY: a plain C struct the call fills.
                let mut request: sys::NVPW_MetricEvalRequest = unsafe { zeroed() };
                library
                    .call(
                        c"NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest",
                        &mut params!(
                            NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params {
                                pMetricsEvaluator: evaluator,
                                pMetricName: name.as_ptr(),
                                pMetricEvalRequest: &mut request,
                                metricEvalRequestStructSize: size_of::<sys::NVPW_MetricEvalRequest>(
                                ),
                            }
                        ),
                    )
                    .map_err(|e| format!("metric {metric}: {e}"))?;
                let mut deps = params!(NVPW_MetricsEvaluator_GetMetricRawDependencies_Params {
                    pMetricsEvaluator: evaluator,
                    pMetricEvalRequests: &request,
                    numMetricEvalRequests: 1,
                    metricEvalRequestStructSize: size_of::<sys::NVPW_MetricEvalRequest>(),
                    metricEvalRequestStrideSize: size_of::<sys::NVPW_MetricEvalRequest>(),
                });
                library.call(c"NVPW_MetricsEvaluator_GetMetricRawDependencies", &mut deps)?;
                let mut list = vec![ptr::null(); deps.numRawDependencies];
                deps.ppRawDependencies = list.as_mut_ptr();
                library.call(c"NVPW_MetricsEvaluator_GetMetricRawDependencies", &mut deps)?;
                raw.extend(list);
                requests.push(request);
            }
            raw.sort_unstable();
            raw.dedup();
            let raw_requests: Vec<sys::NVPA_RawMetricRequest> = raw
                .iter()
                .map(|&name| {
                    params!(NVPA_RawMetricRequest {
                        pMetricName: name,
                        isolated: 1,
                        keepInstances: 1,
                    })
                })
                .collect();

            let config = config_image(&library, &chip_c, &raw_requests)?;
            let mut builder = params!(NVPW_CounterDataBuilder_Create_Params {
                pChipName: chip_c.as_ptr(),
            });
            library.call(c"NVPW_CounterDataBuilder_Create", &mut builder)?;
            let b = builder.pCounterDataBuilder;
            let prefix = (|| {
                library.call(
                    c"NVPW_CounterDataBuilder_AddMetrics",
                    &mut params!(NVPW_CounterDataBuilder_AddMetrics_Params {
                        pCounterDataBuilder: b,
                        pRawMetricRequests: raw_requests.as_ptr(),
                        numMetricRequests: raw_requests.len(),
                    }),
                )?;
                let mut get = params!(NVPW_CounterDataBuilder_GetCounterDataPrefix_Params {
                    pCounterDataBuilder: b,
                });
                library.call(c"NVPW_CounterDataBuilder_GetCounterDataPrefix", &mut get)?;
                let mut prefix = vec![0u8; get.bytesCopied];
                get.bytesAllocated = prefix.len();
                get.pBuffer = prefix.as_mut_ptr();
                library.call(c"NVPW_CounterDataBuilder_GetCounterDataPrefix", &mut get)?;
                Ok::<_, String>(prefix)
            })();
            let _ = library.call(
                c"NVPW_CounterDataBuilder_Destroy",
                &mut params!(NVPW_CounterDataBuilder_Destroy_Params {
                    pCounterDataBuilder: b,
                }),
            );
            let prefix = prefix?;

            Ok(Perf {
                push: library.entry(c"NVPW_VK_Profiler_CommandBuffer_PushRange")?,
                pop: library.entry(c"NVPW_VK_Profiler_CommandBuffer_PopRange")?,
                library,
                chip: chip_c.to_string_lossy().into_owned(),
                config,
                prefix,
                evaluator,
                _evaluator_scratch: evaluator_scratch,
                requests,
                instance: instance.handle(),
                device: device.handle(),
                get_instance_proc_addr,
                get_device_proc_addr,
                recording: AtomicUsize::new(0),
                session: Mutex::new(None),
            })
        }

        /// Opens a range named `name` in `cmd` when a counter frame is open on `kind`'s
        /// queue; true when it did.
        pub(crate) fn push_range(
            &self,
            kind: QueueKind,
            cmd: vk::CommandBuffer,
            name: &CStr,
        ) -> bool {
            if self.recording.load(Ordering::Relaxed) != kind.index() + 1 {
                return false;
            }
            let bytes = name.to_bytes();
            let mut p = params!(NVPW_VK_Profiler_CommandBuffer_PushRange_Params {
                commandBuffer: cmd.as_raw() as _,
                pRangeName: name.as_ptr(),
                rangeNameLength: bytes.len().min(MAX_RANGE_NAME as usize - 1),
            });
            // SAFETY: recording `cmd`; valid params.
            unsafe { (self.push)(&mut p) };
            true
        }

        pub(crate) fn pop_range(&self, cmd: vk::CommandBuffer) {
            let mut p = params!(NVPW_VK_Profiler_CommandBuffer_PopRange_Params {
                commandBuffer: cmd.as_raw() as _,
            });
            // SAFETY: recording `cmd`, which opened the range.
            unsafe { (self.pop)(&mut p) };
        }

        /// Called with the queue's submit lock held.
        pub(crate) fn begin_frame(&self, kind: QueueKind, raw: vk::Queue) -> Result<(), Error> {
            let fail = |e: String| Error::Invalid(format!("Nsight Perf SDK: {e}"));
            let mut session = self.session.lock().unwrap();
            match &*session {
                Some(s) if s.queue != raw => {
                    return Err(Error::Invalid(
                        "a counter frame is still collecting on another queue".into(),
                    ));
                }
                Some(_) => {}
                None => *session = Some(self.begin_session(raw).map_err(fail)?),
            }
            if self.recording.load(Ordering::Relaxed) != 0 {
                return Err(Error::Invalid("counter frame begun twice".into()));
            }
            self.library
                .call(
                    c"NVPW_VK_Profiler_Queue_BeginPass",
                    &mut params!(NVPW_VK_Profiler_Queue_BeginPass_Params {
                        queue: raw.as_raw() as _,
                    }),
                )
                .map_err(fail)?;
            self.recording.store(kind.index() + 1, Ordering::Relaxed);
            Ok(())
        }

        /// Called with the queue's submit lock held.
        pub(crate) fn end_frame(&self, kind: QueueKind, raw: vk::Queue) -> Result<(), Error> {
            if self.recording.swap(0, Ordering::Relaxed) != kind.index() + 1 {
                return Err(Error::Invalid(
                    "no counter frame is open on this queue".into(),
                ));
            }
            self.library
                .call(
                    c"NVPW_VK_Profiler_Queue_EndPass",
                    &mut params!(NVPW_VK_Profiler_Queue_EndPass_Params {
                        queue: raw.as_raw() as _,
                    }),
                )
                .map_err(|e| Error::Invalid(format!("Nsight Perf SDK: {e}")))
        }

        fn begin_session(&self, queue: vk::Queue) -> Result<Session, String> {
            let lib = &self.library;
            let options = params!(NVPW_VK_Profiler_CounterDataImageOptions {
                pCounterDataPrefix: self.prefix.as_ptr(),
                counterDataPrefixSize: self.prefix.len(),
                maxNumRanges: MAX_RANGES,
                maxNumRangeTreeNodes: MAX_RANGES,
                maxRangeNameLength: MAX_RANGE_NAME,
            });
            let mut size = params!(NVPW_VK_Profiler_CounterDataImage_CalculateSize_Params {
                counterDataImageOptionsSize: size_of::<sys::NVPW_VK_Profiler_CounterDataImageOptions>(
                ),
                pOptions: &options,
            });
            lib.call(
                c"NVPW_VK_Profiler_CounterDataImage_CalculateSize",
                &mut size,
            )?;
            let mut image = vec![0u8; size.counterDataImageSize];
            lib.call(
                c"NVPW_VK_Profiler_CounterDataImage_Initialize",
                &mut params!(NVPW_VK_Profiler_CounterDataImage_Initialize_Params {
                    counterDataImageOptionsSize: size_of::<
                        sys::NVPW_VK_Profiler_CounterDataImageOptions,
                    >(),
                    pOptions: &options,
                    counterDataImageSize: image.len(),
                    pCounterDataImage: image.as_mut_ptr(),
                }),
            )?;
            let mut scratch_size = params!(
                NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize_Params {
                    counterDataImageSize: image.len(),
                    pCounterDataImage: image.as_mut_ptr(),
                }
            );
            lib.call(
                c"NVPW_VK_Profiler_CounterDataImage_CalculateScratchBufferSize",
                &mut scratch_size,
            )?;
            let mut scratch = vec![0u8; scratch_size.counterDataScratchBufferSize];
            lib.call(
                c"NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer",
                &mut params!(
                    NVPW_VK_Profiler_CounterDataImage_InitializeScratchBuffer_Params {
                        counterDataImageSize: image.len(),
                        pCounterDataImage: image.as_mut_ptr(),
                        counterDataScratchBufferSize: scratch.len(),
                        pCounterDataScratchBuffer: scratch.as_mut_ptr(),
                    }
                ),
            )?;
            let mut trace = params!(NVPW_VK_Profiler_CalcTraceBufferSize_Params {
                maxRangesPerPass: MAX_RANGES as usize,
                avgRangeNameLength: 32,
            });
            lib.call(c"NVPW_VK_Profiler_CalcTraceBufferSize", &mut trace)?;
            lib.call(
                c"NVPW_VK_Profiler_Queue_BeginSession",
                &mut params!(NVPW_VK_Profiler_Queue_BeginSession_Params {
                    device: self.device.as_raw() as _,
                    queue: queue.as_raw() as _,
                    numTraceBuffers: TRACE_BUFFERS,
                    traceBufferSize: trace.traceBufferSize,
                    maxRangesPerPass: MAX_RANGES as usize,
                    maxLaunchesPerPass: MAX_LAUNCHES,
                    instance: self.instance.as_raw() as _,
                    pfnGetInstanceProcAddr: self.get_instance_proc_addr as *mut c_void,
                    pfnGetDeviceProcAddr: self.get_device_proc_addr as *mut c_void,
                }),
            )?;
            let set = lib.call(
                c"NVPW_VK_Profiler_Queue_SetConfig",
                &mut params!(NVPW_VK_Profiler_Queue_SetConfig_Params {
                    queue: queue.as_raw() as _,
                    pConfig: self.config.as_ptr(),
                    configSize: self.config.len(),
                    minNestingLevel: 1,
                    numNestingLevels: 1,
                    passIndex: 0,
                    targetNestingLevel: 1,
                }),
            );
            if let Err(e) = set {
                self.end_session(queue);
                return Err(e);
            }
            Ok(Session {
                queue,
                image,
                scratch,
            })
        }

        fn end_session(&self, queue: vk::Queue) {
            let _ = self.library.call(
                c"NVPW_VK_Profiler_Queue_EndSession",
                &mut params!(NVPW_VK_Profiler_Queue_EndSession_Params {
                    queue: queue.as_raw() as _,
                }),
            );
        }

        /// Decodes the frame just ended; with every replay collected, evaluates the report
        /// and ends the session. Called with the queue's submit lock held, once its submits
        /// have completed.
        pub(crate) fn decode(
            &self,
            queue: vk::Queue,
            timeout: Duration,
        ) -> Result<Option<CounterReport>, Error> {
            let fail = |e: String| Error::Invalid(format!("Nsight Perf SDK: {e}"));
            let mut guard = self.session.lock().unwrap();
            let s = guard
                .as_mut()
                .ok_or_else(|| Error::Invalid("no counter frame to decode".into()))?;
            let start = Instant::now();
            let all = loop {
                let mut p = params!(NVPW_VK_Profiler_Queue_DecodeCounters_Params {
                    queue: queue.as_raw() as _,
                    counterDataImageSize: s.image.len(),
                    pCounterDataImage: s.image.as_mut_ptr(),
                    counterDataScratchBufferSize: s.scratch.len(),
                    pCounterDataScratchBuffer: s.scratch.as_mut_ptr(),
                });
                self.library
                    .call(c"NVPW_VK_Profiler_Queue_DecodeCounters", &mut p)
                    .map_err(fail)?;
                if p.numRangesDropped != 0 {
                    log::warn!(
                        "Nsight Perf SDK dropped {} ranges of a counter frame",
                        p.numRangesDropped
                    );
                }
                if p.onePassCollected != 0 {
                    break p.allPassesCollected != 0;
                }
                if start.elapsed() > timeout {
                    return Err(Error::Timeout);
                }
                std::thread::sleep(Duration::from_millis(1));
            };
            if !all {
                return Ok(None);
            }
            let s = guard.take().unwrap();
            self.end_session(s.queue);
            self.evaluate(&s.image).map(Some).map_err(fail)
        }

        fn evaluate(&self, image: &[u8]) -> Result<CounterReport, String> {
            let lib = &self.library;
            lib.call(
                c"NVPW_MetricsEvaluator_SetDeviceAttributes",
                &mut params!(NVPW_MetricsEvaluator_SetDeviceAttributes_Params {
                    pMetricsEvaluator: self.evaluator,
                    pCounterDataImage: image.as_ptr(),
                    counterDataImageSize: image.len(),
                }),
            )?;
            let mut count = params!(NVPW_CounterData_GetNumRanges_Params {
                pCounterDataImage: image.as_ptr(),
            });
            lib.call(c"NVPW_CounterData_GetNumRanges", &mut count)?;
            let mut ranges = Vec::with_capacity(count.numRanges);
            for range in 0..count.numRanges {
                let mut desc = params!(NVPW_Profiler_CounterData_GetRangeDescriptions_Params {
                    pCounterDataImage: image.as_ptr(),
                    rangeIndex: range,
                });
                lib.call(c"NVPW_Profiler_CounterData_GetRangeDescriptions", &mut desc)?;
                let mut parts = vec![ptr::null(); desc.numDescriptions];
                desc.ppDescriptions = parts.as_mut_ptr();
                lib.call(c"NVPW_Profiler_CounterData_GetRangeDescriptions", &mut desc)?;
                let name = parts
                    .iter()
                    // SAFETY: the SDK returns NUL-terminated names inside `image`.
                    .map(|&p| unsafe { CStr::from_ptr(p) }.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                let mut values = vec![0.0f64; self.requests.len()];
                lib.call(
                    c"NVPW_MetricsEvaluator_EvaluateToGpuValues",
                    &mut params!(NVPW_MetricsEvaluator_EvaluateToGpuValues_Params {
                        pMetricsEvaluator: self.evaluator,
                        pMetricEvalRequests: self.requests.as_ptr(),
                        numMetricEvalRequests: self.requests.len(),
                        metricEvalRequestStructSize: size_of::<sys::NVPW_MetricEvalRequest>(),
                        metricEvalRequestStrideSize: size_of::<sys::NVPW_MetricEvalRequest>(),
                        pCounterDataImage: image.as_ptr(),
                        counterDataImageSize: image.len(),
                        rangeIndex: range,
                        isolated: 1,
                        pMetricValues: values.as_mut_ptr(),
                    }),
                )?;
                ranges.push(CounterRange { name, values });
            }
            Ok(CounterReport {
                chip: self.chip.clone(),
                ranges,
            })
        }

        /// Ends an unfinished session and frees the evaluator; before the device goes.
        pub(crate) fn shut_down(&self) {
            if let Some(s) = self.session.lock().unwrap().take() {
                self.end_session(s.queue);
            }
            let _ = self.library.call(
                c"NVPW_MetricsEvaluator_Destroy",
                &mut params!(NVPW_MetricsEvaluator_Destroy_Params {
                    pMetricsEvaluator: self.evaluator,
                }),
            );
        }
    }

    fn config_image(
        lib: &Library,
        chip: &CStr,
        requests: &[sys::NVPA_RawMetricRequest],
    ) -> Result<Vec<u8>, String> {
        let mut create = params!(NVPW_VK_RawMetricsConfig_Create_Params {
            activityKind: sys::NVPA_ActivityKind_NVPA_ACTIVITY_KIND_PROFILER,
            pChipName: chip.as_ptr(),
        });
        lib.call(c"NVPW_VK_RawMetricsConfig_Create", &mut create)?;
        let config = create.pRawMetricsConfig;
        let image = (|| {
            lib.call(
                c"NVPW_RawMetricsConfig_BeginPassGroup",
                &mut params!(NVPW_RawMetricsConfig_BeginPassGroup_Params {
                    pRawMetricsConfig: config,
                }),
            )?;
            lib.call(
                c"NVPW_RawMetricsConfig_AddMetrics",
                &mut params!(NVPW_RawMetricsConfig_AddMetrics_Params {
                    pRawMetricsConfig: config,
                    pRawMetricRequests: requests.as_ptr(),
                    numMetricRequests: requests.len(),
                }),
            )?;
            lib.call(
                c"NVPW_RawMetricsConfig_EndPassGroup",
                &mut params!(NVPW_RawMetricsConfig_EndPassGroup_Params {
                    pRawMetricsConfig: config,
                }),
            )?;
            lib.call(
                c"NVPW_RawMetricsConfig_GenerateConfigImage",
                &mut params!(NVPW_RawMetricsConfig_GenerateConfigImage_Params {
                    pRawMetricsConfig: config,
                }),
            )?;
            let mut get = params!(NVPW_RawMetricsConfig_GetConfigImage_Params {
                pRawMetricsConfig: config,
            });
            lib.call(c"NVPW_RawMetricsConfig_GetConfigImage", &mut get)?;
            let mut image = vec![0u8; get.bytesCopied];
            get.bytesAllocated = image.len();
            get.pBuffer = image.as_mut_ptr();
            lib.call(c"NVPW_RawMetricsConfig_GetConfigImage", &mut get)?;
            Ok(image)
        })();
        let _ = lib.call(
            c"NVPW_RawMetricsConfig_Destroy",
            &mut params!(NVPW_RawMetricsConfig_Destroy_Params {
                pRawMetricsConfig: config,
            }),
        );
        image
    }
}
