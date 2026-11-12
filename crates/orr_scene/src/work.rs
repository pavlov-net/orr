//! What a scene's content and its views ask of the per-frame work lists: the floors of their
//! capacities ([`crate::capacity::Capacity`]).
//!
//! The content gives each list a bound, the most its copies can ask of it in a frame, summed
//! over the scene's copies from what their models' meshes hold. Where the bound is under what
//! the views are estimated to ask, the list holds the bound and drops nothing. Past it the
//! list starts at the estimate and follows what the frames want.
//!
//! The estimate takes a cluster to cover the area of the error it is drawn at, a square of the
//! detail target's pixels: more clusters than that cannot each show in a visibility buffer. A
//! view's visible clusters take its extent's pixels. The traversal chooses a scene's cuts
//! around its focus whatever the views face, so the scene's lists take the pixels of the whole
//! sphere of directions at the scene's detail, each list the share of its bound that the
//! clusters estimated are of the clusters the content holds.

use std::f64::consts::PI;

/// The most a copy of a model asks of its scene's work lists in a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CopyWork {
    /// Hierarchy nodes one traversal pass visits for it.
    pub(crate) nodes: u32,
    /// Raster candidate items, and the clusters in them.
    pub(crate) items: u32,
    pub(crate) clusters: u32,
    /// The CLAS and the BLAS builds of the ray-tracing cut it gathers.
    pub(crate) rt_clusters: u32,
    pub(crate) rt_blas: u32,
}

impl CopyWork {
    /// What a copy that shows two steps of its chain at once asks of the raster's lists: the
    /// most of each over any two of `steps`, the model's own first. Only its own step gathers
    /// a ray-tracing cut.
    pub(crate) fn fading(steps: &[Self]) -> Self {
        let two = |f: fn(&Self) -> u32| {
            let (mut most, mut next) = (0u32, 0u32);
            for v in steps.iter().map(f) {
                if v > most {
                    (most, next) = (v, most);
                } else {
                    next = next.max(v);
                }
            }
            most.saturating_add(next)
        };
        Self {
            nodes: two(|w| w.nodes),
            items: two(|w| w.items),
            clusters: two(|w| w.clusters),
            ..steps[0]
        }
    }
}

/// The most a scene's copies ask of its work lists in a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Work {
    pub(crate) nodes: u64,
    pub(crate) items: u64,
    pub(crate) clusters: u64,
    pub(crate) rt_clusters: u64,
    pub(crate) rt_blas: u64,
    /// Copies that gather a ray-tracing cut.
    pub(crate) rt_copies: u64,
}

impl Work {
    pub(crate) fn add(&mut self, w: CopyWork, copies: u32) {
        let n = u64::from(copies);
        self.nodes += n * u64::from(w.nodes);
        self.items += n * u64::from(w.items);
        self.clusters += n * u64::from(w.clusters);
        self.rt_clusters += n * u64::from(w.rt_clusters);
        self.rt_blas += n * u64::from(w.rt_blas);
        self.rt_copies += if w.rt_blas > 0 { n } else { 0 };
    }
}

/// What the estimates read of a scene's detail.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DetailTarget {
    /// The scene's detail: render pixels per radian at its focus.
    pub(crate) pixels_per_radian: f32,
    /// The largest errors drawn and traced, in pixels.
    pub(crate) raster_pixels: f32,
    pub(crate) rt_pixels: f32,
    /// Copies of a smaller projected radius gather no cut; 0 for none.
    pub(crate) merge_pixels: f32,
    /// [`crate::Budgets::rt_cut_pixels`].
    pub(crate) rt_cut_pixels: f32,
}

/// The floors of a scene's work lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Floors {
    pub(crate) queue: u32,
    pub(crate) candidates: u32,
    pub(crate) rt_clas: u32,
    pub(crate) rt_blas: u32,
    pub(crate) rt_keys: u32,
}

/// `bound`, or `estimate` where that is less.
fn least(bound: u64, estimate: f64) -> u32 {
    let bound = u32::try_from(bound).unwrap_or(u32::MAX);
    if estimate >= f64::from(bound) {
        bound
    } else {
        estimate.ceil() as u32
    }
}

/// `pixels` in clusters drawn at an error of `error` pixels.
fn clusters_of(pixels: f64, error: f32) -> f64 {
    pixels / (f64::from(error) * f64::from(error))
}

impl DetailTarget {
    /// Pixels of the sphere of directions around the focus.
    fn sphere_pixels(&self) -> f64 {
        let ppr = f64::from(self.pixels_per_radian);
        4.0 * PI * ppr * ppr
    }

    /// The floors of the lists of a scene whose copies ask `work` at most.
    pub(crate) fn floors(&self, work: &Work) -> Floors {
        let sphere = self.sphere_pixels();
        let clusters = clusters_of(sphere, self.raster_pixels);
        // The share of what the content holds that the estimate takes.
        let share = |bound: u64| {
            if clusters >= work.clusters as f64 {
                u32::try_from(bound).unwrap_or(u32::MAX)
            } else {
                least(bound, bound as f64 * clusters / work.clusters as f64)
            }
        };
        // Copies that gather are at least `merge_pixels` in radius, so about this many tile
        // the sphere.
        let gathering = if self.merge_pixels > 0.0 {
            let m = f64::from(self.merge_pixels);
            sphere / (PI * m * m)
        } else {
            f64::INFINITY
        };
        let rt = clusters_of(sphere, self.rt_pixels) / f64::from(self.rt_cut_pixels);
        Floors {
            queue: share(work.nodes),
            candidates: share(work.items),
            rt_clas: least(work.rt_clusters, rt),
            rt_blas: least(work.rt_blas, gathering),
            rt_keys: least(work.rt_copies, gathering),
        }
    }

    /// The floor of the visible clusters of a view of `pixels` render pixels over a scene
    /// whose copies ask `work` at most and whose producers emit `produced` clusters at most.
    pub(crate) fn visible(&self, work: &Work, produced: u64, pixels: u64) -> u32 {
        least(
            work.clusters + produced,
            clusters_of(pixels as f64, self.raster_pixels),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> DetailTarget {
        DetailTarget {
            pixels_per_radian: 100.0,
            raster_pixels: 1.0,
            rt_pixels: 2.0,
            merge_pixels: 50.0,
            rt_cut_pixels: 10.0,
        }
    }

    #[test]
    fn lists_hold_their_bound_under_the_estimate_and_its_share_past_it() {
        let d = target();
        // 4 pi 100^2 pixels of sphere.
        let sphere = 125_663.7f64;
        let mut small = Work::default();
        small.add(
            CopyWork {
                nodes: 7,
                items: 3,
                clusters: 40,
                rt_clusters: 40,
                rt_blas: 1,
            },
            5,
        );
        let f = d.floors(&small);
        assert_eq!(
            f,
            Floors {
                queue: 35,
                candidates: 15,
                rt_clas: 200,
                rt_blas: 5,
                rt_keys: 5
            },
            "a scene under every estimate holds what its content asks"
        );
        assert_eq!(d.visible(&small, 9, 1 << 20), 209);
        assert_eq!(d.visible(&small, 9, 100), 100, "a pixel a cluster");

        let mut large = Work::default();
        large.add(
            CopyWork {
                nodes: 1000,
                items: 500,
                clusters: 4000,
                rt_clusters: 4000,
                rt_blas: 3,
            },
            1000,
        );
        let f = d.floors(&large);
        let share = sphere / 4.0e6;
        assert_eq!(f.queue, (1.0e6 * share).ceil() as u32);
        assert_eq!(f.candidates, (0.5e6 * share).ceil() as u32);
        assert_eq!(
            f.rt_clas,
            (sphere / 4.0 / 10.0).ceil() as u32,
            "the sphere's pixels at the traced error, a CLAS to the setting's pixels"
        );
        assert_eq!((f.rt_blas, f.rt_keys), (16, 16), "4 (100 / 50)^2 copies");
        let whole = DetailTarget {
            merge_pixels: 0.0,
            ..d
        };
        let f = whole.floors(&large);
        assert_eq!((f.rt_blas, f.rt_keys), (3000, 1000), "every copy gathers");
    }

    #[test]
    fn a_fading_copy_asks_its_two_largest_steps_of_the_raster() {
        let step = |n| CopyWork {
            nodes: n,
            items: 2 * n,
            clusters: 3 * n,
            rt_clusters: 3 * n,
            rt_blas: 1,
        };
        let w = CopyWork::fading(&[step(4), step(9), step(9), step(1)]);
        assert_eq!((w.nodes, w.items, w.clusters), (18, 36, 54));
        assert_eq!(
            (w.rt_clusters, w.rt_blas),
            (12, 1),
            "only its own step gathers"
        );
        assert_eq!(CopyWork::fading(&[step(4)]), step(4));
    }
}
