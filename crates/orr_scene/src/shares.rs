//! What the residency managers on one device hold on its heaps, so each gives back its share
//! of a heap's excess and no more.
//!
//! Every [`Scenes`](crate::Scenes) on a device sees the same heap use. Each posts, per heap,
//! the bytes its classes use and the bytes its pools are giving back, and reads the sums over
//! all of them. What a manager holds stays in the sums until it drops. What it is giving back
//! comes only as it steps, so another manager counts those bytes only while it has posted
//! since that manager's post before.

use std::sync::{Arc, Mutex, Weak};

use orr_gpu::Device;

/// The posts of one device's managers.
struct DeviceHeaps {
    device: Device,
    posts: Mutex<Posts>,
}

#[derive(Default)]
struct Posts {
    /// Posts so far.
    count: u64,
    next_id: u64,
    managers: Vec<Posted>,
}

/// What one manager last posted.
struct Posted {
    id: u64,
    /// `Posts::count` at its last post and at the one before.
    at: u64,
    before: u64,
    /// Per heap index: bytes held and bytes coming back.
    heaps: Vec<[u64; 2]>,
}

/// The posts of the devices with a live [`HeapShare`].
static DEVICES: Mutex<Vec<Weak<DeviceHeaps>>> = Mutex::new(Vec::new());

/// One residency manager's part of its device's posts, taken out when it drops.
pub(crate) struct HeapShare {
    all: Arc<DeviceHeaps>,
    id: u64,
}

impl HeapShare {
    /// A part of the posts of `device`, holding nothing.
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
                posts: Mutex::default(),
            });
            devices.push(Arc::downgrade(&d));
            d
        });
        let id = {
            let mut posts = all.posts.lock().unwrap();
            posts.next_id += 1;
            let id = posts.next_id;
            posts.managers.push(Posted {
                id,
                at: 0,
                before: 0,
                heaps: Vec::new(),
            });
            id
        };
        Self { all, id }
    }

    /// Sets what this manager holds and is giving back per heap index, `[held, coming]`;
    /// returns the sums over the device's managers.
    pub(crate) fn post(&mut self, mine: &[[u64; 2]]) -> Vec<[u64; 2]> {
        self.all.posts.lock().unwrap().post(self.id, mine)
    }
}

impl Posts {
    fn post(&mut self, id: u64, mine: &[[u64; 2]]) -> Vec<[u64; 2]> {
        self.count += 1;
        let count = self.count;
        let me = self
            .managers
            .iter_mut()
            .find(|m| m.id == id)
            .expect("a live share");
        (me.before, me.at) = (me.at, count);
        me.heaps = mine.to_vec();
        let since = me.before;
        let mut sums = vec![[0; 2]; mine.len()];
        for m in &self.managers {
            // A manager that has not stepped since this one's post before gives nothing back
            // meanwhile.
            let stepping = m.id == id || m.at > since;
            for (sum, heap) in sums.iter_mut().zip(&m.heaps) {
                sum[0] += heap[0];
                if stepping {
                    sum[1] += heap[1];
                }
            }
        }
        sums
    }
}

impl Drop for HeapShare {
    fn drop(&mut self) {
        let mut posts = self.all.posts.lock().unwrap();
        posts.managers.retain(|m| m.id != self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two managers on a device read each other's bytes while both post; one that stops keeps
    /// what it holds in the other's sums and loses what it was giving back.
    #[test]
    fn a_manager_that_stops_posting_keeps_its_held_bytes_and_loses_its_coming() {
        let mut posts = Posts::default();
        for id in [1, 2] {
            posts.managers.push(Posted {
                id,
                at: 0,
                before: 0,
                heaps: Vec::new(),
            });
        }
        assert_eq!(posts.post(1, &[[100, 10]]), [[100, 10]]);
        assert_eq!(posts.post(2, &[[300, 30]]), [[400, 40]]);
        assert_eq!(posts.post(1, &[[100, 10]]), [[400, 40]]);
        assert_eq!(posts.post(2, &[[300, 30]]), [[400, 40]]);
        // 2 stops: 1's next post still follows a post of 2, the one after it does not.
        assert_eq!(posts.post(1, &[[100, 10]]), [[400, 40]]);
        assert_eq!(posts.post(1, &[[100, 10]]), [[400, 10]]);
        assert_eq!(posts.post(2, &[[300, 30]]), [[400, 40]]);
        posts.managers.retain(|m| m.id != 1);
        assert_eq!(posts.post(2, &[[300, 30]]), [[300, 30]]);
    }
}
