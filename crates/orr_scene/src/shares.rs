//! What the residency managers on one device hold on its heaps, so each gives back its share
//! of a heap's excess and no more.
//!
//! Every [`Scenes`](crate::Scenes) on a device sees the same heap use. Each posts, per heap,
//! the bytes its classes use and the bytes its pools are giving back, and reads the sums over
//! all of them. What a manager holds stays in the sums until it drops. What it is giving back
//! comes only as it steps, so another manager counts those bytes only while it has posted
//! within the reader's last response period of posts, the horizon over which the reader's
//! budget cuts wait for returning memory.

use std::collections::VecDeque;
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
    /// `Posts::count` at its last post.
    at: u64,
    /// `Posts::count` at its posts before the last, at most its response period of them, the
    /// latest last.
    before: VecDeque<u64>,
    /// Its response period, in its posts.
    period: usize,
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
    /// A part of the posts of `device`, holding nothing, for a manager that posts once a step
    /// and waits `period` steps for what its budget cuts return ([`crate::residency`]).
    pub(crate) fn new(device: &Device, period: u64) -> Self {
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
            posts.managers.push(Posted::new(id, period));
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

impl Posted {
    fn new(id: u64, period: u64) -> Self {
        let period = usize::try_from(period).expect("a period of frames");
        Self {
            id,
            at: 0,
            before: VecDeque::with_capacity(period),
            period,
            heaps: Vec::new(),
        }
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
        if me.before.len() == me.period {
            me.before.pop_front();
        }
        me.before.push_back(me.at);
        me.at = count;
        me.heaps = mine.to_vec();
        let since = me.before[0];
        let mut sums = vec![[0; 2]; mine.len()];
        for m in &self.managers {
            // A manager that has not stepped within this one's period gives nothing back
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

    /// Two managers on a device read each other's bytes while both post, one less often than
    /// the other; one that stops keeps what it holds in the other's sums and loses what it was
    /// giving back once the other has posted a period without it.
    #[test]
    fn a_manager_that_stops_posting_keeps_its_held_bytes_and_loses_its_coming() {
        let mut posts = Posts::default();
        posts
            .managers
            .extend([Posted::new(1, 3), Posted::new(2, 3)]);
        assert_eq!(posts.post(1, &[[100, 10]]), [[100, 10]]);
        // 2 posts once for every two posts of 1.
        for _ in 0..3 {
            assert_eq!(posts.post(2, &[[300, 30]]), [[400, 40]]);
            assert_eq!(posts.post(1, &[[100, 10]]), [[400, 40]]);
            assert_eq!(posts.post(1, &[[100, 10]]), [[400, 40]]);
        }
        // 2 stops: 1 counts what 2 gives back for its period.
        assert_eq!(posts.post(1, &[[100, 10]]), [[400, 40]]);
        assert_eq!(posts.post(1, &[[100, 10]]), [[400, 10]]);
        assert_eq!(posts.post(2, &[[300, 30]]), [[400, 40]]);
        posts.managers.retain(|m| m.id != 1);
        assert_eq!(posts.post(2, &[[300, 30]]), [[300, 30]]);
    }
}
