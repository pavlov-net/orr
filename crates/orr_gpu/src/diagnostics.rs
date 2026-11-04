//! What a device reports about its progress and faults, alike on every backend. The Metal
//! backend fills in less: no checkpoints, fault addresses or crash dumps.

use std::fmt;
use std::path::PathBuf;

use crate::QueueKind;

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
    /// GPU memory statistics as JSON (totals per heap, memory type and pool); lost devices
    /// only.
    pub memory: Option<String>,
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
        if let Some(m) = &self.memory {
            writeln!(f, "  memory: {m}")?;
        }
        Ok(())
    }
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
