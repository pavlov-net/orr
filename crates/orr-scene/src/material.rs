//! The global material table and its writer.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use orr_gpu::{Allocation, Device, MemoryKind};

use crate::memory::UploadWindow;
use crate::upload::Record;
use crate::{Error, MaterialGpu, Result};

/// An index in the global material table.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Pod, Zeroable)]
pub struct MaterialId(pub u32);

/// `MaterialGpu::flags` bits.
pub mod material_flags {
    pub const ALPHA_TESTED: u32 = 1 << 0;
    pub const DOUBLE_SIDED: u32 = 1 << 1;
    pub const THIN_WALLED: u32 = 1 << 2;
    pub const DIELECTRIC: u32 = 1 << 3;
    pub const PRESENCE: u32 = 1 << 4;
    pub const EMISSIVE: u32 = 1 << 5;
    pub const UNLIT: u32 = 1 << 6;
    pub const SUBSURFACE: u32 = 1 << 7;
    pub const DISPLACED: u32 = 1 << 8;
    pub const COAT: u32 = 1 << 9;
    pub const FUZZ: u32 = 1 << 10;
    pub const THIN_FILM: u32 = 1 << 11;
    pub const ANISOTROPY: u32 = 1 << 12;
    /// The cost class, 0-3, in bits 16-17.
    pub const COST_SHIFT: u32 = 16;
    pub const COST_MASK: u32 = 3 << COST_SHIFT;
}

pub(crate) struct MaterialTable {
    pub table: Allocation<MaterialGpu>,
    pub capacity: u32,
    next: Mutex<u32>,
}

impl MaterialTable {
    pub(crate) fn new(device: &Device, capacity: u32) -> Result<Self> {
        Ok(Self {
            table: device.gpu_malloc(MemoryKind::DeviceLocal, u64::from(capacity), "materials")?,
            capacity,
            next: Mutex::new(0),
        })
    }
}

/// Writes material records into the frame's upload window. Send: one per thread.
pub struct MaterialWriter<'f> {
    pub(crate) device: &'f Device,
    pub(crate) table: &'f MaterialTable,
    pub(crate) window: &'f UploadWindow,
    pub(crate) seq: &'f AtomicU64,
    pub(crate) records: &'f Mutex<Vec<Record>>,
}

impl MaterialWriter<'_> {
    /// A new material holding `m`.
    pub fn create(&mut self, m: &MaterialGpu) -> Result<MaterialId> {
        let id = {
            let mut next = self.table.next.lock().unwrap();
            if *next == self.table.capacity {
                return Err(Error::Capacity {
                    what: "materials",
                    capacity: self.table.capacity,
                });
            }
            *next += 1;
            MaterialId(*next - 1)
        };
        self.update(id, m)?;
        Ok(id)
    }

    /// Replaces material `id` from this frame on.
    pub fn update(&mut self, id: MaterialId, m: &MaterialGpu) -> Result<()> {
        assert!(
            id.0 < self.table.capacity,
            "{id:?} is outside the material table"
        );
        let src = self.window.put(self.device, m)?;
        let dst = self.table.table.gpu.offset(u64::from(id.0));
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.records
            .lock()
            .unwrap()
            .push(Record::values(seq, dst, src, 1));
        Ok(())
    }
}
