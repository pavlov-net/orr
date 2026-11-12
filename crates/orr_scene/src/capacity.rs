//! The capacity of a per-frame work list, which follows its floor and what the frames want of
//! it.
//!
//! A list holds its floor at least: what its content can ask of it in a frame, or what its
//! views are estimated to ask where that is less. A scene under the estimate so drops nothing
//! from its first frame on. A floor that rises past what the list holds grows it as [`grown`]
//! says; one that falls before the list has grown asks only what it now is.
//!
//! Past its floor a list grows once a completed frame wanted more than it holds: to twice its
//! size, or half as much again as was wanted when that is more. It shrinks once every frame of
//! a window of [`SHRINK_FRAMES`] completed frames wanted at most a quarter of it, to half as
//! much again as the most any of them wanted, never under its floor. A demand that holds
//! steady moves neither way, so the size cannot swing between two. A list never holds more
//! than its limit, the most entries its readers can name.

use crate::table::grown;

/// Completed frames a list waits before it gives memory back.
const SHRINK_FRAMES: u32 = 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Capacity {
    floor: u32,
    limit: u32,
    /// What the list's memory holds.
    held: u32,
    /// What the frames want the list to hold.
    asked: u32,
    /// The most a frame of the current window wanted, and the frames it has seen.
    peak: u32,
    frames: u32,
}

impl Capacity {
    /// A list that holds nothing yet, of `limit` at most.
    pub(crate) fn new(limit: u32) -> Self {
        Self {
            floor: 0,
            limit,
            held: 0,
            asked: 0,
            peak: 0,
            frames: 0,
        }
    }

    /// The list holds `floor` at least from now on, within its limit, in place of the floor it
    /// had: one that holds less grows as [`grown`] says, and one that is to shrink shrinks no
    /// further.
    pub(crate) fn floor(&mut self, floor: u32) {
        self.floor = floor.min(self.limit);
    }

    /// Takes what a completed frame wanted of the list, which was recorded with this capacity
    /// or an earlier one.
    pub(crate) fn observe(&mut self, wanted: u32) {
        self.peak = self.peak.max(wanted);
        self.frames += 1;
        if wanted > self.held {
            let grown = self.held.saturating_mul(2).max(headroom(wanted));
            self.asked = self.asked.max(grown).min(self.limit);
        } else if self.frames >= SHRINK_FRAMES {
            if self.peak <= self.held / 4 {
                self.asked = headroom(self.peak);
            }
            self.peak = 0;
            self.frames = 0;
        }
    }

    /// The size the list's memory should take: what the frames want, and its floor at least.
    pub(crate) fn wanted(&self) -> u32 {
        let floor = if self.floor > self.held {
            let more = self.floor - self.held;
            grown("work list", self.held, more, self.limit).expect("a floor within the limit")
        } else {
            self.floor
        };
        self.asked.max(floor)
    }

    /// What the list's memory holds.
    pub(crate) fn held(&self) -> u32 {
        self.held
    }

    /// [`Self::wanted`], when it differs from what the memory holds.
    pub(crate) fn resize(&self) -> Option<u32> {
        let wanted = self.wanted();
        (wanted != self.held).then_some(wanted)
    }

    /// The list's memory holds `n` now.
    pub(crate) fn set(&mut self, n: u32) {
        if n == self.held {
            return;
        }
        self.held = n;
        self.asked = n;
        self.peak = 0;
        self.frames = 0;
    }
}

/// `n` and half as much again; the GPU indexes lists with 32 bits.
fn headroom(n: u32) -> u32 {
    n.saturating_add(n / 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Observes `wanted` for `frames` frames, taking every resize.
    fn run(c: &mut Capacity, wanted: u32, frames: u32) {
        for _ in 0..frames {
            c.observe(wanted);
            if let Some(n) = c.resize() {
                c.set(n);
            }
        }
    }

    #[test]
    fn a_list_grows_past_its_demand_at_once_and_shrinks_only_after_a_window_far_under_it() {
        let mut c = Capacity::new(u32::MAX);
        c.floor(100);
        assert_eq!((c.held(), c.resize()), (0, Some(100)), "its floor at first");
        c.set(100);
        run(&mut c, 100, 1);
        assert_eq!(c.wanted(), 100, "what it holds is enough");
        c.observe(101);
        c.observe(1000);
        assert_eq!(c.resize(), Some(1500), "the most a lagging frame wanted");
        c.set(1500);
        run(&mut c, 1000, 1);
        assert_eq!(
            c.wanted(),
            1500,
            "a frame recorded before the growth asks no more"
        );
        run(&mut c, 1600, 1);
        assert_eq!(c.wanted(), 3000, "at least twice");

        // Steady demand over a quarter of it, or a window with one heavy frame, keeps it.
        run(&mut c, 751, 2 * SHRINK_FRAMES);
        assert_eq!(c.wanted(), 3000);
        run(&mut c, 10, SHRINK_FRAMES - 1);
        run(&mut c, 2000, 1);
        assert_eq!(c.wanted(), 3000);
        // A window under a quarter shrinks it, and what it shrinks to holds that demand.
        run(&mut c, 700, SHRINK_FRAMES);
        assert_eq!(c.wanted(), 1050);
        run(&mut c, 700, 4 * SHRINK_FRAMES);
        assert_eq!(c.wanted(), 1050);
        run(&mut c, 0, SHRINK_FRAMES);
        assert_eq!(c.wanted(), 100, "never under its floor");

        let mut c = Capacity::new(250);
        c.floor(100);
        run(&mut c, 400, 2);
        assert_eq!(c.wanted(), 250, "never past its limit");
    }

    #[test]
    fn a_floor_grows_a_list_as_a_table_grows_and_lowers_only_what_it_shrinks_to() {
        let mut c = Capacity::new(1000);
        run(&mut c, 0, 1);
        assert_eq!(c.held(), 0, "a list of nothing holds nothing");
        c.floor(10);
        assert_eq!(c.resize(), Some(10), "what its content asks, from nothing");
        c.set(10);
        c.floor(11);
        assert_eq!(c.resize(), Some(20), "twice what it holds past that");
        c.set(20);
        c.floor(15);
        assert_eq!(c.resize(), None, "a floor it holds moves nothing");
        c.floor(5000);
        assert_eq!(c.resize(), Some(1000), "within its limit");
        c.set(1000);
        c.floor(3);
        assert_eq!(c.resize(), None, "a lower floor gives nothing back at once");
        for _ in 0..SHRINK_FRAMES {
            c.observe(2);
        }
        assert_eq!(
            c.resize(),
            Some(3),
            "a window far under it shrinks it to the floor"
        );
        c.floor(40);
        assert_eq!(
            c.resize(),
            Some(40),
            "or to a floor raised before it has shrunk"
        );

        let mut c = Capacity::new(1000);
        c.set(100);
        c.floor(900);
        c.floor(50);
        assert_eq!(
            c.resize(),
            None,
            "a floor that fell before the list grew to it asks nothing"
        );
    }
}
