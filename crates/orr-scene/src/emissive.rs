//! The emissive list: one entry per producer entry, then one per (instance slot, EMISSIVE
//! cluster span of its model).

use std::collections::BTreeMap;

use orr_pack::mesh::Cluster;

use crate::material::MaterialId;

/// Consecutive EMISSIVE clusters of one section at a model's finest level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmissiveSpan {
    pub section: u8,
    /// Mesh cluster indices `first_cluster..first_cluster + cluster_count`.
    pub first_cluster: u32,
    pub cluster_count: u32,
    /// Sum of the clusters' `Cluster::area`, model space, m².
    pub area: f32,
}

/// What each copy of a model adds to a scene's emissive list.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelEmission {
    /// The model's section count: the stride of its material sets.
    pub sections: u32,
    /// The material of each section for copies without a material set.
    pub materials: Vec<MaterialId>,
    pub spans: Vec<EmissiveSpan>,
}

/// The EMISSIVE spans of `clusters`, a model's finest level in mesh cluster order from
/// index `first`.
pub fn emissive_spans(
    first: u32,
    clusters: impl IntoIterator<Item = Cluster>,
) -> Vec<EmissiveSpan> {
    let mut spans: Vec<EmissiveSpan> = Vec::new();
    for (i, c) in clusters.into_iter().enumerate() {
        if c.flags & Cluster::EMISSIVE == 0 {
            continue;
        }
        let index = first + i as u32;
        match spans.last_mut() {
            Some(s) if s.section == c.section && s.first_cluster + s.cluster_count == index => {
                s.cluster_count += 1;
                s.area += c.area;
            }
            _ => spans.push(EmissiveSpan {
                section: c.section,
                first_cluster: index,
                cluster_count: 1,
                area: c.area,
            }),
        }
    }
    spans
}

/// The entries of copies `slot..slot + copies` added together: `spans` per copy, copy
/// major, from instance entry `entry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Run {
    pub copies: u32,
    pub entry: u32,
    pub spans: u32,
}

/// Runs by first slot.
#[derive(Debug, Default)]
pub(crate) struct Runs(BTreeMap<u32, Run>);

impl Runs {
    pub(crate) fn insert(&mut self, slot: u32, run: Run) {
        self.0.insert(slot, run);
    }

    /// Takes the entries of slots `first..first + count` out of their runs and returns them
    /// as (first entry, entry count) ranges.
    pub(crate) fn remove(&mut self, first: u32, count: u32) -> Vec<(u32, u32)> {
        let end = first + count;
        let start = self
            .0
            .range(..first)
            .next_back()
            .filter(|(s, r)| *s + r.copies > first)
            .map_or(first, |(s, _)| *s);
        let hit: Vec<(u32, Run)> = self.0.range(start..end).map(|(s, r)| (*s, *r)).collect();
        let mut freed = Vec::new();
        for (slot, run) in hit {
            self.0.remove(&slot);
            let lo = first.max(slot);
            let hi = end.min(slot + run.copies);
            if slot < lo {
                self.0.insert(
                    slot,
                    Run {
                        copies: lo - slot,
                        ..run
                    },
                );
            }
            if hi < slot + run.copies {
                self.0.insert(
                    hi,
                    Run {
                        copies: slot + run.copies - hi,
                        entry: run.entry + (hi - slot) * run.spans,
                        spans: run.spans,
                    },
                );
            }
            freed.push((run.entry + (lo - slot) * run.spans, (hi - lo) * run.spans));
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;

    use super::*;

    #[test]
    fn spans_join_consecutive_emissive_clusters_of_one_section() {
        let cluster = |section: u8, emissive: bool, area: f32| {
            let mut c = Cluster::zeroed();
            c.section = section;
            c.flags = if emissive { Cluster::EMISSIVE } else { 0 };
            c.area = area;
            c
        };
        let clusters = [
            cluster(0, true, 1.0),
            cluster(0, true, 2.0),
            cluster(1, true, 4.0),
            cluster(1, false, 8.0),
            cluster(1, true, 16.0),
        ];
        let spans = emissive_spans(10, clusters);
        let got: Vec<_> = spans
            .iter()
            .map(|s| (s.section, s.first_cluster, s.cluster_count, s.area))
            .collect();
        assert_eq!(got, [(0, 10, 2, 3.0), (1, 12, 1, 4.0), (1, 14, 1, 16.0)]);
    }

    #[test]
    fn removing_slots_splits_runs_and_frees_their_entries() {
        let mut runs = Runs::default();
        runs.insert(
            0,
            Run {
                copies: 10,
                entry: 100,
                spans: 2,
            },
        );
        runs.insert(
            10,
            Run {
                copies: 5,
                entry: 0,
                spans: 1,
            },
        );
        // Slots 8..12 take copies 8, 9 of the first run and 0, 1 of the second.
        assert_eq!(runs.remove(8, 4), [(116, 4), (0, 2)]);
        // Slots 3..5 split the first run in two.
        assert_eq!(runs.remove(3, 2), [(106, 4)]);
        assert_eq!(runs.remove(0, 15), [(100, 6), (110, 6), (2, 3)]);
        assert_eq!(runs.remove(0, 15), []);
    }
}
