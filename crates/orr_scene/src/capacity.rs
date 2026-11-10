//! The capacity of a per-frame work list, which follows what the frames want of it.
//!
//! A list grows once a completed frame wanted more than it holds: to twice its size, or half
//! as much again as was wanted when that is more. It shrinks once every frame of a window of
//! [`SHRINK_FRAMES`] completed frames wanted at most a quarter of it, to half as much again
//! as the most any of them wanted, never under its first size. A demand that holds steady
//! moves neither way, so the size cannot swing between two. A list never holds more than its
//! limit, the most entries its readers can name.

/// Completed frames a list waits before it gives memory back.
const SHRINK_FRAMES: u32 = 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Capacity {
    initial: u32,
    limit: u32,
    /// What the list's memory holds.
    held: u32,
    /// What the list should hold.
    wanted: u32,
    /// The most a frame of the current window wanted, and the frames it has seen.
    peak: u32,
    frames: u32,
}

impl Capacity {
    pub(crate) fn new(initial: u32, limit: u32) -> Self {
        let initial = initial.min(limit);
        Self {
            initial,
            limit,
            held: initial,
            wanted: initial,
            peak: 0,
            frames: 0,
        }
    }

    /// Takes what a completed frame wanted of the list, which was recorded with this capacity
    /// or an earlier one.
    pub(crate) fn observe(&mut self, wanted: u32) {
        self.peak = self.peak.max(wanted);
        self.frames += 1;
        if wanted > self.held {
            let grown = self.held.saturating_mul(2).max(headroom(wanted));
            self.wanted = self.wanted.max(grown).min(self.limit);
        } else if self.frames >= SHRINK_FRAMES {
            if self.peak <= self.held / 4 {
                self.wanted = self.initial.max(headroom(self.peak));
            }
            self.peak = 0;
            self.frames = 0;
        }
    }

    /// The size the list's memory should take.
    pub(crate) fn wanted(&self) -> u32 {
        self.wanted
    }

    /// [`Self::wanted`], when it differs from what the memory holds.
    pub(crate) fn resize(&self) -> Option<u32> {
        (self.wanted != self.held).then_some(self.wanted)
    }

    /// The list's memory holds `n` now.
    pub(crate) fn set(&mut self, n: u32) {
        if n == self.held {
            return;
        }
        self.held = n;
        self.wanted = n;
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
        let mut c = Capacity::new(100, u32::MAX);
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
        assert_eq!(c.wanted(), 100, "never under its first size");

        let mut c = Capacity::new(100, 250);
        run(&mut c, 400, 2);
        assert_eq!(c.wanted(), 250, "never past its limit");
    }
}
