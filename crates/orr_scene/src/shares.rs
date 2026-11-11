//! What the residency managers on one device hold on its heaps, so each gives back its share
//! of a heap's excess and no more.
//!
//! Every [`Scenes`](crate::Scenes) on a device sees the same heap use. Each posts, per heap,
//! the bytes its classes use and the bytes its pools are giving back, and reads the sums over
//! all of them.

use std::sync::{Arc, Mutex, Weak};

use orr_gpu::Device;

/// The sums for one device.
struct DeviceHeaps {
    device: Device,
    /// Per heap index: bytes held and bytes coming back.
    heaps: Mutex<Vec<[u64; 2]>>,
}

/// The sums of the devices with a live [`HeapShare`].
static DEVICES: Mutex<Vec<Weak<DeviceHeaps>>> = Mutex::new(Vec::new());

/// One residency manager's part of its device's sums, taken out when it drops.
pub(crate) struct HeapShare {
    all: Arc<DeviceHeaps>,
    mine: Vec<[u64; 2]>,
}

impl HeapShare {
    /// A part of the sums of `device`, holding nothing.
    pub(crate) fn new(device: &Device) -> Self {
        let mut devices = DEVICES.lock().unwrap();
        devices.retain(|d| d.strong_count() > 0);
        let found = devices
            .iter()
            .filter_map(Weak::upgrade)
            .find(|d| d.device == *device);
        let all = found.unwrap_or_else(|| {
            let d = Arc::new(DeviceHeaps {
                device: device.clone(),
                heaps: Mutex::default(),
            });
            devices.push(Arc::downgrade(&d));
            d
        });
        Self {
            all,
            mine: Vec::new(),
        }
    }

    /// Sets what this manager holds and is giving back per heap index, `[held, coming]`;
    /// returns the sums over the device's managers as of what each last posted.
    pub(crate) fn post(&mut self, mine: &[[u64; 2]]) -> Vec<[u64; 2]> {
        let mut sums = self.all.heaps.lock().unwrap();
        if sums.len() < mine.len() {
            sums.resize(mine.len(), [0; 2]);
        }
        self.mine.resize(mine.len(), [0; 2]);
        for ((sum, old), new) in sums.iter_mut().zip(&mut self.mine).zip(mine) {
            *sum = [sum[0] - old[0] + new[0], sum[1] - old[1] + new[1]];
            *old = *new;
        }
        sums[..mine.len()].to_vec()
    }
}

impl Drop for HeapShare {
    fn drop(&mut self) {
        let mut sums = self.all.heaps.lock().unwrap();
        for (sum, old) in sums.iter_mut().zip(&self.mine) {
            *sum = [sum[0] - old[0], sum[1] - old[1]];
        }
    }
}
