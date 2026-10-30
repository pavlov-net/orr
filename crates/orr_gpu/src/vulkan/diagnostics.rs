//! Crash diagnostics: pass breadcrumbs and checkpoints, device fault data, and the report
//! that names the passes a lost device was running.

use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, c_char, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;

use super::device::{Device, Shared};
use super::garbage::{MAX_TIMELINES, RawBuffer};
use super::memory::{MemoryKind, Range};
use super::queue::QueueKind;
use crate::Error;

/// What a device records for crash diagnosis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Diagnostics<'a> {
    /// Device-lost reports from submit records and device fault data.
    #[default]
    Off,
    /// Also writes a GPU breadcrumb and a checkpoint at every pass begin and end, so reports
    /// name the passes that had begun and not ended. A pass's end waits for its work, so
    /// passes do not overlap.
    Checkpoints,
    /// Checkpoints, and Nsight Aftermath crash dumps with shader debug info written to
    /// `dump_dir`. The Aftermath library comes from the SDK at `AFTERMATH_SDK`, or else the
    /// newest Nsight Graphics install. On a hang the driver also writes `ucode.elf` into the
    /// working directory.
    Aftermath { dump_dir: &'a Path },
    /// For Nsight Graphics captures: leaves off the optional extensions captures do not
    /// support (device address commands, KHR device fault, opacity micromaps, pipeline
    /// binaries, shader abort, shader constant data).
    Capture,
}

/// Pass slots in the breadcrumb buffer; the trail keeps fewer passes than this, so a slot
/// is never shared by two passes a report names.
const SLOTS: u32 = 4096;
const TRAIL_PASSES: usize = SLOTS as usize / 2;

/// Pass ids are unique in the process, so markers in a crash dump resolve to one pass.
static NEXT_PASS: AtomicU32 = AtomicU32::new(1);

pub(crate) fn next_pass_id() -> u32 {
    NEXT_PASS.fetch_add(1, Ordering::Relaxed)
}

/// The checkpoint marker of a pass begin or end.
fn marker(id: u32, end: bool) -> usize {
    (id as usize) << 1 | usize::from(end)
}

pub(crate) fn unmarker(token: usize) -> (u32, bool) {
    ((token >> 1) as u32, token & 1 == 1)
}

/// GPU-written pass progress: per slot, the id of the last pass that began there and the
/// last that ended there.
pub(crate) struct Breadcrumbs {
    pub raw: RawBuffer,
    words: NonNull<u32>,
    marker_fn: ash::amd::buffer_marker::Device,
}

// SAFETY: the mapping is plain memory the GPU writes and the CPU only reads.
unsafe impl Send for Breadcrumbs {}
// SAFETY: as above.
unsafe impl Sync for Breadcrumbs {}

impl Breadcrumbs {
    pub(crate) fn new(
        shared: &Shared,
        marker_fn: ash::amd::buffer_marker::Device,
    ) -> Result<Self, Error> {
        let (raw, mapped) = shared.create_buffer(
            MemoryKind::Readback,
            u64::from(SLOTS) * 8,
            shared.buffer_usage,
            4,
            "orr breadcrumbs",
        )?;
        let words = mapped.expect("Readback memory is mapped").cast::<u32>();
        // SAFETY: the mapping holds 2 * SLOTS words and nothing else writes it yet.
        unsafe { std::ptr::write_bytes(words.as_ptr(), 0, 2 * SLOTS as usize) };
        Ok(Self {
            raw,
            words,
            marker_fn,
        })
    }

    /// (began, ended) for pass `id`.
    fn state(&self, id: u32) -> (bool, bool) {
        let slot = (id % SLOTS) as usize;
        // SAFETY: both words lie in the mapping; the GPU may write them concurrently, which
        // a volatile read tolerates.
        let (begin, end) = unsafe {
            (
                self.words.as_ptr().add(2 * slot).read_volatile(),
                self.words.as_ptr().add(2 * slot + 1).read_volatile(),
            )
        };
        (begin == id, end == id)
    }
}

/// The passes of one submit.
pub(crate) struct SubmitRecord {
    queue: QueueKind,
    value: u64,
    passes: Vec<(u32, &'static str)>,
}

/// The recent submits with passes, oldest first.
#[derive(Default)]
pub(crate) struct Trail {
    records: VecDeque<SubmitRecord>,
    passes: usize,
}

impl Trail {
    fn name(&self, id: u32) -> Option<&'static str> {
        self.records
            .iter()
            .rev()
            .flat_map(|r| &r.passes)
            .find(|p| p.0 == id)
            .map(|p| p.1)
    }
}

/// Device fault entry points, KHR where offered.
pub(crate) enum FaultFns {
    Khr(ash::khr::device_fault::Device),
    Ext(ash::ext::device_fault::Device),
}

/// What a device was doing, and on a lost device why it was lost.
#[derive(Clone, Debug, Default)]
pub struct DiagnosticsReport {
    /// The device was lost; the report is the one made when that was first seen.
    pub lost: bool,
    pub queues: Vec<QueueProgress>,
    /// Device fault reports; lost devices only.
    pub faults: Vec<DeviceFault>,
    /// The message of the shader `abort` that lost the device.
    pub abort: Option<String>,
    /// The Aftermath crash dump, which Nsight Graphics opens; its decoded JSON sits beside
    /// it with a `.json` extension.
    pub crash_dump: Option<PathBuf>,
}

/// One queue's progress through its recent passes.
#[derive(Clone, Debug)]
pub struct QueueProgress {
    pub queue: QueueKind,
    /// The last timeline value submitted.
    pub submitted: u64,
    /// The last value completed; `None` once the device can no longer report it.
    pub completed: Option<u64>,
    /// The last pass, in submission order, known to have ended.
    pub last_completed: Option<&'static str>,
    /// Passes that began and had not ended, oldest first. Without breadcrumbs these are the
    /// passes of submits that had not completed.
    pub in_flight: Vec<&'static str>,
    /// The queue's last checkpoint per pipeline stage; lost devices with checkpoints only.
    pub checkpoints: Vec<Checkpoint>,
}

/// The last pass marker one pipeline stage passed.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub stage: String,
    /// `None` when the pass is older than the submits the device keeps.
    pub pass: Option<&'static str>,
    /// The marker is the pass's end rather than its begin.
    pub end: bool,
}

/// One device fault report.
#[derive(Clone, Debug)]
pub struct DeviceFault {
    pub description: String,
    /// The device stopped responding.
    pub watchdog: bool,
    pub addresses: Vec<FaultAddress>,
    /// Vendor descriptions with their codes and data.
    pub vendor: Vec<String>,
}

/// An address a fault reported.
#[derive(Clone, Debug)]
pub struct FaultAddress {
    pub access: FaultAccess,
    pub address: u64,
    /// The fault lies within this power-of-two span around `address`; 0 or 1 when exact.
    pub precision: u64,
    /// The allocation, live or recently freed, the address falls in, with the offset.
    pub allocation: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultAccess {
    Read,
    Write,
    Execute,
    /// The address of the faulting instruction.
    Instruction,
    Other,
}

impl From<vk::DeviceFaultAddressTypeKHR> for FaultAccess {
    fn from(t: vk::DeviceFaultAddressTypeKHR) -> Self {
        use vk::DeviceFaultAddressTypeKHR as T;
        match t {
            T::READ_INVALID => FaultAccess::Read,
            T::WRITE_INVALID => FaultAccess::Write,
            T::EXECUTE_INVALID => FaultAccess::Execute,
            T::INSTRUCTION_POINTER_UNKNOWN
            | T::INSTRUCTION_POINTER_INVALID
            | T::INSTRUCTION_POINTER_FAULT => FaultAccess::Instruction,
            _ => FaultAccess::Other,
        }
    }
}

/// Names the allocation holding `address`: a live one first, then the most recent freed
/// one, then any within `precision` of it.
pub(crate) fn describe_address(
    live: &BTreeMap<u64, Range>,
    freed: &VecDeque<(u64, Range)>,
    address: u64,
    precision: u64,
) -> Option<String> {
    let inside = |start: u64, r: &Range| (start..r.end).contains(&address);
    if let Some((&start, r)) = live.range(..=address).next_back()
        && inside(start, r)
    {
        return Some(format!("{:?} + {:#x}", r.name, address - start));
    }
    if let Some((start, r)) = freed.iter().rev().find(|(s, r)| inside(*s, r)) {
        return Some(format!("freed {:?} + {:#x}", r.name, address - start));
    }
    if precision > 1 {
        let lo = address & !(precision - 1);
        let hi = address | (precision - 1);
        let overlaps = |start: u64, r: &Range| start <= hi && lo < r.end;
        if let Some((_, r)) = live.iter().find(|(s, r)| overlaps(**s, r)) {
            return Some(format!("near {:?}", r.name));
        }
        if let Some((_, r)) = freed.iter().rev().find(|(s, r)| overlaps(*s, r)) {
            return Some(format!("near freed {:?}", r.name));
        }
    }
    None
}

fn c_text(chars: &[c_char]) -> String {
    // SAFETY: Vulkan writes NUL-terminated strings within these arrays.
    unsafe { CStr::from_ptr(chars.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// Wait for the driver to finish its fault reports.
const FAULT_TIMEOUT_NS: u64 = 2_000_000_000;

impl Shared {
    /// Records a pass begin or end in the breadcrumb buffer and as a checkpoint.
    pub(crate) fn mark_pass(&self, cmd: vk::CommandBuffer, id: u32, end: bool) {
        if let Some(b) = self.breadcrumbs.get() {
            let (stage, word) = if end {
                (vk::PipelineStageFlags2::ALL_COMMANDS, 1)
            } else {
                (vk::PipelineStageFlags2::TOP_OF_PIPE, 0)
            };
            let offset = (u64::from(id % SLOTS) * 2 + word) * 4;
            // SAFETY: recording; the offset lies inside the breadcrumb buffer, which has
            // transfer-destination usage.
            unsafe {
                if end {
                    // 621.01 writes the marker before earlier work has passed `stage`
                    // unless a barrier makes it wait.
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER);
                    self.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&[barrier]),
                    );
                }
                b.marker_fn
                    .cmd_write_buffer_marker2(cmd, stage, b.raw.buffer, offset, id);
            }
        }
        if let Some(c) = &self.checkpoint_fn {
            // SAFETY: recording; the marker is a token, never dereferenced.
            unsafe { c.cmd_set_checkpoint(cmd, marker(id, end) as *const c_void) };
        }
    }

    /// Keeps the passes of a successful submit for reports.
    pub(crate) fn record_submit(
        &self,
        queue: QueueKind,
        value: u64,
        passes: Vec<(u32, &'static str)>,
    ) {
        if passes.is_empty() {
            return;
        }
        let mut trail = self.trail.lock().unwrap();
        trail.passes += passes.len();
        trail.records.push_back(SubmitRecord {
            queue,
            value,
            passes,
        });
        while trail.passes > TRAIL_PASSES {
            let old = trail.records.pop_front().unwrap();
            trail.passes -= old.passes.len();
        }
    }

    /// The name of pass `id`, if it is still in the trail.
    pub(crate) fn pass_name(&self, id: u32) -> Option<&'static str> {
        self.trail.lock().unwrap().name(id)
    }

    /// Maps a failed Vulkan call to an error; a lost device is reported once.
    pub(crate) fn vk_error(&self, call: &'static str, e: vk::Result) -> Error {
        if e == vk::Result::ERROR_DEVICE_LOST {
            self.device_lost()
        } else {
            Error::vk(call)(e)
        }
    }

    /// Makes and logs the device-lost report the first time loss is seen.
    pub(crate) fn device_lost(&self) -> Error {
        self.lost.get_or_init(|| {
            let report = self.diagnostics(true);
            log::error!("{report}");
            report
        });
        Error::DeviceLost
    }

    fn diagnostics(&self, lost: bool) -> DiagnosticsReport {
        let crash_dump = if lost && self.caps.aftermath {
            super::aftermath::wait_for_dump()
        } else {
            None
        };
        let queues = (0..MAX_TIMELINES)
            .filter(|&i| self.timelines[i].is_some())
            .map(|i| self.queue_progress(QueueKind::ALL[i], lost))
            .collect();
        let (faults, abort) = if lost {
            self.faults()
        } else {
            (Vec::new(), None)
        };
        DiagnosticsReport {
            lost,
            queues,
            faults,
            abort,
            crash_dump,
        }
    }

    fn queue_progress(&self, queue: QueueKind, lost: bool) -> QueueProgress {
        let timeline = self.timeline(queue.index() as u32);
        let submitted = timeline.submitted.load(Ordering::Acquire);
        // SAFETY: the semaphore is live.
        let completed = unsafe { self.device.get_semaphore_counter_value(timeline.semaphore) }.ok();
        let mut last_completed = None;
        let mut in_flight = Vec::new();
        {
            let trail = self.trail.lock().unwrap();
            for r in trail.records.iter().filter(|r| r.queue == queue) {
                for &(id, name) in &r.passes {
                    let (began, ended) = match self.breadcrumbs.get() {
                        Some(b) => b.state(id),
                        None => match completed {
                            Some(c) if r.value <= c => (true, true),
                            Some(_) => (true, false),
                            None => (false, false),
                        },
                    };
                    if ended {
                        last_completed = Some(name);
                    } else if began {
                        in_flight.push(name);
                    }
                }
            }
        }
        let checkpoints = if lost {
            self.checkpoints(queue)
        } else {
            Vec::new()
        };
        QueueProgress {
            queue,
            submitted,
            completed,
            last_completed,
            in_flight,
            checkpoints,
        }
    }

    fn checkpoints(&self, queue: QueueKind) -> Vec<Checkpoint> {
        let (Some(c), Some(raw)) = (&self.checkpoint_fn, self.queue_handles[queue.index()]) else {
            return Vec::new();
        };
        // SAFETY: the device is lost, which the query requires; the queue is live.
        let n = unsafe { c.get_queue_checkpoint_data2_len(raw) };
        let mut data = vec![vk::CheckpointData2NV::default(); n];
        // SAFETY: as above, with room for `n` entries.
        unsafe { c.get_queue_checkpoint_data2(raw, &mut data) };
        data.iter()
            .map(|d| {
                let (id, end) = unmarker(d.p_checkpoint_marker as usize);
                Checkpoint {
                    stage: format!("{:?}", d.stage),
                    pass: self.pass_name(id),
                    end,
                }
            })
            .collect()
    }

    /// Fault reports and the abort message of a lost device.
    fn faults(&self) -> (Vec<DeviceFault>, Option<String>) {
        let Some(fns) = &self.fault_fn else {
            return (Vec::new(), None);
        };
        let live = self.buffers.read().unwrap();
        let freed = self.freed.lock().unwrap();
        let address = |a: &vk::DeviceFaultAddressInfoKHR| FaultAddress {
            access: a.address_type.into(),
            address: a.reported_address,
            precision: a.address_precision,
            allocation: describe_address(&live, &freed, a.reported_address, a.address_precision),
        };
        let vendor = |v: &vk::DeviceFaultVendorInfoKHR| {
            format!(
                "{} (code {:#x}, data {:#x})",
                c_text(&v.description),
                v.vendor_fault_code,
                v.vendor_fault_data
            )
        };
        match fns {
            FaultFns::Khr(f) => {
                let get = f.fp().get_device_fault_reports_khr;
                let mut count = 0u32;
                // SAFETY: a count query on a lost device.
                let _ = unsafe {
                    get(
                        self.device.handle(),
                        FAULT_TIMEOUT_NS,
                        &mut count,
                        std::ptr::null_mut(),
                    )
                };
                let mut infos = vec![vk::DeviceFaultInfoKHR::default(); count as usize];
                if count > 0 {
                    // SAFETY: room for `count` reports.
                    let _ = unsafe {
                        get(
                            self.device.handle(),
                            FAULT_TIMEOUT_NS,
                            &mut count,
                            infos.as_mut_ptr(),
                        )
                    };
                    infos.truncate(count as usize);
                }
                let faults = infos
                    .iter()
                    .map(|i| {
                        let flags = i.flags;
                        let mut addresses = Vec::new();
                        if flags.contains(vk::DeviceFaultFlagsKHR::FLAG_MEMORY_ADDRESS) {
                            addresses.push(address(&i.fault_address_info));
                        }
                        if flags.contains(vk::DeviceFaultFlagsKHR::FLAG_INSTRUCTION_ADDRESS) {
                            addresses.push(address(&i.instruction_address_info));
                        }
                        DeviceFault {
                            description: c_text(&i.description),
                            watchdog: flags
                                .contains(vk::DeviceFaultFlagsKHR::FLAG_WATCHDOG_TIMEOUT),
                            addresses,
                            vendor: if flags.contains(vk::DeviceFaultFlagsKHR::FLAG_VENDOR) {
                                vec![vendor(&i.vendor_info)]
                            } else {
                                Vec::new()
                            },
                        }
                    })
                    .collect();
                let abort = if self.caps.shader_abort {
                    self.abort_message(f)
                } else {
                    None
                };
                (faults, abort)
            }
            FaultFns::Ext(f) => {
                let get = f.fp().get_device_fault_info_ext;
                let mut counts = vk::DeviceFaultCountsEXT::default();
                // SAFETY: a count query on a lost device.
                let _ = unsafe { get(self.device.handle(), &mut counts, std::ptr::null_mut()) };
                let mut addresses = vec![
                    vk::DeviceFaultAddressInfoKHR::default();
                    counts.address_info_count as usize
                ];
                let mut vendors = vec![
                    vk::DeviceFaultVendorInfoKHR::default();
                    counts.vendor_info_count as usize
                ];
                counts.vendor_binary_size = 0;
                let mut info = vk::DeviceFaultInfoEXT {
                    p_address_infos: addresses.as_mut_ptr(),
                    p_vendor_infos: vendors.as_mut_ptr(),
                    ..Default::default()
                };
                // SAFETY: the arrays hold the counted entries; no vendor binary is asked for.
                let r = unsafe { get(self.device.handle(), &mut counts, &mut info) };
                if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
                    return (Vec::new(), None);
                }
                addresses.truncate(counts.address_info_count as usize);
                vendors.truncate(counts.vendor_info_count as usize);
                let fault = DeviceFault {
                    description: c_text(&info.description),
                    watchdog: false,
                    addresses: addresses.iter().map(address).collect(),
                    vendor: vendors.iter().map(vendor).collect(),
                };
                (vec![fault], None)
            }
        }
    }

    fn abort_message(&self, f: &ash::khr::device_fault::Device) -> Option<String> {
        let get = f.fp().get_device_fault_debug_info_khr;
        let mut message = vk::DeviceFaultShaderAbortMessageInfoKHR::default();
        let mut info = vk::DeviceFaultDebugInfoKHR {
            p_next: std::ptr::from_mut(&mut message).cast(),
            ..Default::default()
        };
        // SAFETY: a size query on a lost device.
        let r = unsafe { get(self.device.handle(), &mut info) };
        let size = message.message_data_size as usize;
        if size == 0 {
            log::debug!("vkGetDeviceFaultDebugInfoKHR: {r:?}, no abort message");
            return None;
        }
        let mut data = vec![0u8; size];
        message.p_message_data = data.as_mut_ptr().cast();
        info.vendor_binary_size = 0;
        info.p_vendor_binary_data = std::ptr::null_mut();
        // SAFETY: the message buffer holds `size` bytes; no vendor binary is asked for.
        let r = unsafe { get(self.device.handle(), &mut info) };
        if r != vk::Result::SUCCESS && r != vk::Result::INCOMPLETE {
            return None;
        }
        data.truncate(message.message_data_size as usize);
        let text = String::from_utf8_lossy(&data);
        Some(text.trim_end_matches('\0').to_owned())
    }
}

impl Device {
    /// What every queue was doing: its recent passes that ended and those still running.
    /// On a lost device, the report made when loss was first seen (it is also logged), with
    /// checkpoints, device faults, the abort message and the Aftermath crash dump.
    pub fn diagnostics_report(&self) -> DiagnosticsReport {
        if let Some(r) = self.shared.lost.get() {
            return r.clone();
        }
        self.shared.diagnostics(false)
    }

    /// Hands Aftermath the debug info of a stripped module, so crash dumps map its faults to
    /// source lines. `stripped` is the SPIR-V pipelines are created from and `sidecar` its
    /// separated debug info (`.dbg.spv`). Does nothing without `Caps::aftermath`.
    pub fn add_shader_debug_info(&self, stripped: &[u32], sidecar: &[u8]) {
        if self.shared.caps.aftermath {
            super::aftermath::add_sidecar(stripped, sidecar);
        }
    }
}

impl fmt::Display for DiagnosticsReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}",
            if self.lost {
                "device lost"
            } else {
                "device progress"
            }
        )?;
        for q in self.queues.iter().filter(|q| q.submitted > 0) {
            write!(f, "  {:?} queue: submitted {}", q.queue, q.submitted)?;
            match q.completed {
                Some(c) => write!(f, ", completed {c}")?,
                None => write!(f, ", completion unknown")?,
            }
            if let Some(p) = q.last_completed {
                write!(f, "; last completed {p:?}")?;
            }
            if !q.in_flight.is_empty() {
                write!(f, "; in flight {:?}", q.in_flight)?;
            }
            writeln!(f)?;
            for c in &q.checkpoints {
                let at = if c.end { "end" } else { "begin" };
                match c.pass {
                    Some(p) => writeln!(f, "    {}: {at} of {p:?}", c.stage)?,
                    None => writeln!(f, "    {}: {at} of an older pass", c.stage)?,
                }
            }
        }
        for fault in &self.faults {
            write!(f, "  fault")?;
            if !fault.description.is_empty() {
                write!(f, ": {}", fault.description)?;
            }
            if fault.watchdog {
                write!(f, " (watchdog timeout)")?;
            }
            writeln!(f)?;
            for a in &fault.addresses {
                write!(f, "    {:?} at {:#x}", a.access, a.address)?;
                if a.precision > 1 {
                    write!(f, " (within {:#x})", a.precision)?;
                }
                if let Some(name) = &a.allocation {
                    write!(f, " in {name}")?;
                }
                writeln!(f)?;
            }
            for v in &fault.vendor {
                writeln!(f, "    vendor: {v}")?;
            }
        }
        if let Some(a) = &self.abort {
            writeln!(f, "  shader abort: {a:?}")?;
        }
        if let Some(d) = &self.crash_dump {
            writeln!(f, "  crash dump: {}", d.display())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(end: u64, name: &str) -> Range {
        Range {
            end,
            buffer: vk::Buffer::null(),
            name: name.into(),
        }
    }

    #[test]
    fn fault_addresses_name_their_allocation() {
        let mut live = BTreeMap::new();
        live.insert(0x1000, range(0x2000, "a"));
        live.insert(0x4000, range(0x4100, "b"));
        let mut freed = VecDeque::new();
        freed.push_back((0x2000, range(0x3000, "old")));
        freed.push_back((0x2000, range(0x2800, "newer")));
        let d = |a, p| describe_address(&live, &freed, a, p);
        assert_eq!(d(0x1040, 0).as_deref(), Some("\"a\" + 0x40"));
        assert_eq!(d(0x2000, 0).as_deref(), Some("freed \"newer\" + 0x0"));
        assert_eq!(d(0x2900, 0).as_deref(), Some("freed \"old\" + 0x900"));
        assert_eq!(d(0x3800, 0), None);
        assert_eq!(d(0x4180, 0x200).as_deref(), Some("near \"b\""));
        assert_eq!(d(0x4100, 0), None);
    }
}
