//! Which streamed page a cluster of a streaming mesh stands in for while that page is not
//! resident: the page of the group the cluster was simplified from, whose clusters take its
//! place once they are resident.
//!
//! The format names no such group, but the cook writes a cluster's own bounds and error as
//! exactly that group's `simplified` and `error`, and the group's depth is one less than the
//! depth of the cluster's own group, whose clusters are source clusters at depth 0. The mesh's
//! streamed groups, ordered by depth and by those bits, find it with a binary search; groups in
//! pinned pages are always resident and stay out. Distinct groups can hold the same depth and
//! bits, such as two groups made of clusters simplified from the same groups: a cluster that
//! matches groups in several pages names all of their pages, and stands in while any of them
//! is not resident.

use std::sync::Arc;

use orr_pack::Sphere;
use orr_pack::mesh::{Cluster, Group};

use crate::Result;

/// `tables.slang`'s `FINER_NONE` and `FINER_RUN`.
pub(crate) const FINER_NONE: u32 = !0;
pub(crate) const FINER_RUN: u32 = 1 << 31;

/// A group's depth, then the bits of its `simplified` sphere and its `error`.
type Key = (u8, [u32; 4], u32);

fn key(depth: u8, s: &Sphere, error: f32) -> Key {
    let [x, y, z] = s.center.map(f32::to_bits);
    (depth, [x, y, z, s.radius.to_bits()], error.to_bits())
}

fn group_key(g: &Group) -> Key {
    key(g.depth, &g.simplified, g.error)
}

/// A streaming mesh's streamed groups, ordered for the search: 4 B per streamed group, beside
/// the group table the mount keeps for edits, and the runs of the keys shared across pages.
pub(crate) struct FinerPages {
    groups: Arc<[Group]>,
    /// The streamed groups by key.
    order: Box<[u32]>,
    /// The keys whose groups lie in more than one page: the first of them in `order`, and
    /// the offset of their run in `runs`.
    shared: Box<[(u32, u32)]>,
    /// What `MeshGpu.finer_runs` holds: per shared key, its page count, then its pages
    /// ascending.
    runs: Box<[u32]>,
}

impl FinerPages {
    pub(crate) fn new(groups: Arc<[Group]>, pinned_pages: u32) -> Self {
        let key_of = |g: u32| group_key(&groups[g as usize]);
        let mut order: Vec<u32> = (0..groups.len() as u32)
            .filter(|&g| groups[g as usize].page >= pinned_pages)
            .collect();
        order.sort_unstable_by_key(|&g| key_of(g));
        let (mut shared, mut runs) = (Vec::new(), Vec::new());
        let mut first = 0;
        for run in order.chunk_by(|&a, &b| key_of(a) == key_of(b)) {
            let mut pages: Vec<u32> = run.iter().map(|&g| groups[g as usize].page).collect();
            pages.sort_unstable();
            pages.dedup();
            if pages.len() > 1 {
                shared.push((first, runs.len() as u32));
                runs.push(pages.len() as u32);
                runs.extend(pages);
            }
            first += run.len() as u32;
        }
        Self {
            groups,
            order: order.into(),
            shared: shared.into(),
            runs: runs.into(),
        }
    }

    /// `MeshGpu.finer_runs`; empty when no key is shared across pages.
    pub(crate) fn runs(&self) -> &[u32] {
        &self.runs
    }

    /// `ClusterRef.finer` of each cluster of page `page`, whose records are `clusters`.
    pub(crate) fn of_page(&self, page: u32, clusters: &[Cluster]) -> Result<Vec<u32>> {
        let mut out = vec![FINER_NONE; clusters.len()];
        let start = self.groups.partition_point(|g| g.page < page);
        for g in self.groups[start..].iter().take_while(|g| g.page == page) {
            let first = usize::from(g.first_cluster);
            let at = first..first + usize::from(g.cluster_count);
            let (Some(out), Some(records)) = (out.get_mut(at.clone()), clusters.get(at)) else {
                return Err(orr_pack::Error::Malformed("group outside its page").into());
            };
            // A source cluster was simplified from nothing.
            let Some(depth) = g.depth.checked_sub(1) else {
                continue;
            };
            for (o, c) in out.iter_mut().zip(records) {
                *o = self.finer(key(depth, &c.self_sphere, c.self_error));
            }
        }
        Ok(out)
    }

    fn finer(&self, k: Key) -> u32 {
        let key_of = |g: &u32| group_key(&self.groups[*g as usize]);
        let lo = self.order.partition_point(|g| key_of(g) < k);
        match self.order.get(lo) {
            Some(g) if key_of(g) == k => {}
            _ => return FINER_NONE,
        }
        match self.shared.binary_search_by_key(&(lo as u32), |s| s.0) {
            Ok(s) => FINER_RUN | self.shared[s].1,
            Err(_) => self.groups[self.order[lo] as usize].page,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(page: u32, first: u16, count: u16, depth: u8, center: f32, error: f32) -> Group {
        Group {
            simplified: Sphere {
                center: [center, 0.0, 0.0],
                radius: 1.0,
            },
            error,
            page,
            first_cluster: first,
            cluster_count: count,
            depth,
            _pad: [0; 3],
        }
    }

    fn cluster(center: f32, error: f32) -> Cluster {
        Cluster {
            self_sphere: Sphere {
                center: [center, 0.0, 0.0],
                radius: 1.0,
            },
            self_error: error,
            ..bytemuck::Zeroable::zeroed()
        }
    }

    /// Keys shared across pages name every page in a run, keys shared within one page name
    /// that page, depth tells equal bits apart, and pinned groups and source clusters name
    /// nothing.
    #[test]
    fn clusters_name_the_pages_of_the_groups_their_bounds_match() {
        let groups: Arc<[Group]> = Arc::new([
            // Pinned page 0: a terminal group, and one its third cluster matches.
            group(0, 0, 3, 2, 9.0, f32::INFINITY),
            group(0, 3, 1, 1, 5.0, 0.125),
            // Page 1: a key that group 6 on page 3 shares, and one two of its groups share.
            group(1, 0, 2, 1, 1.0, 0.5),
            group(1, 2, 1, 1, 2.0, 0.5),
            group(1, 3, 1, 1, 2.0, 0.5),
            group(2, 0, 2, 0, 1.0, 0.25),
            group(3, 0, 1, 1, 1.0, 0.5),
            // The bits of groups 3 and 4 one step finer.
            group(3, 1, 1, 0, 2.0, 0.5),
        ]);
        let f = FinerPages::new(groups, 1);
        assert_eq!(f.runs(), &[2, 1, 3]);
        let pinned = [
            cluster(1.0, 0.5),
            cluster(2.0, 0.5),
            cluster(5.0, 0.125),
            cluster(0.0, 0.0),
        ];
        assert_eq!(
            f.of_page(0, &pinned).unwrap(),
            [FINER_RUN, 1, FINER_NONE, FINER_NONE]
        );
        let streamed = [
            cluster(1.0, 0.25),
            cluster(1.0, 0.25),
            cluster(2.0, 0.5),
            cluster(9.0, 0.5),
        ];
        assert_eq!(f.of_page(1, &streamed).unwrap(), [2, 2, 3, FINER_NONE]);
        assert_eq!(f.of_page(2, &streamed[..2]).unwrap(), [FINER_NONE; 2]);
        assert!(
            f.of_page(1, &streamed[..3]).is_err(),
            "a group past the page"
        );
    }
}
