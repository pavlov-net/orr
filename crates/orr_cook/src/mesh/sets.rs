//! The cluster sets of a mesh ([`ClusterSets`]): per cut row and ranked section, the
//! section's clusters in the cut, in spatial order, with a fixed-point CDF of their weights
//! and a tree over ranges of that order.
//!
//! Rows cover every cut a mount can trace, so a mesh whose pages stream has its sets from
//! mount on. A discrete mesh has a row per level. A DAG mesh has a row for the finest cut of
//! its pinned pages, then one per threshold of [`cut_thresholds`].

use orr_math::DVec3;
use orr_pack::mesh::{
    Cluster, ClusterSet, ClusterSets, MeshInfo, NO_RANK, SET_LEAF, Section, SetNode,
    cut_thresholds, in_cut, oct_decode, oct_encode,
};

use super::SectionData;
use super::maps::Emission;

/// Where a cluster of a ranked section lies and what it weighs, in mesh space.
#[derive(Copy, Clone, Debug)]
pub(super) struct ClusterBounds {
    center: DVec3,
    /// Reaches every vertex from `center`.
    radius: f64,
    /// The area-weighted mean face normal, unit, and the cosine and sine of the half angle
    /// around it that holds every face normal.
    axis: DVec3,
    cone: (f64, f64),
    /// Square meters.
    area: f64,
    /// `area` times the emission texture's mean over each triangle's footprint, relative to
    /// the material's factor: 1 without a texture.
    emission: f64,
}

impl ClusterBounds {
    /// The bounds of a cluster with `positions` and corners `tris`, its section's emission
    /// texture read at each vertex's uv.
    pub(super) fn new(
        positions: &[DVec3],
        tris: &[[u8; 3]],
        emission: Option<(&Emission, &[[f32; 2]])>,
    ) -> Self {
        let (lo, hi) = positions
            .iter()
            .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), &p| {
                (lo.min(p), hi.max(p))
            });
        let center = 0.5 * (lo + hi);
        let radius = positions
            .iter()
            .map(|&p| p.distance(center))
            .fold(0.0, f64::max);
        let (mut area, mut weighted, mut sum) = (0.0, 0.0, DVec3::ZERO);
        let mut faces = [DVec3::ZERO; 128];
        let mut nf = 0;
        for t in tris {
            let p = t.map(|i| positions[usize::from(i)]);
            let n = (p[1] - p[0]).cross(p[2] - p[0]);
            let a = 0.5 * n.length();
            if a <= 0.0 {
                continue;
            }
            area += a;
            weighted += a * emission.map_or(1.0, |(e, uvs)| e.mean(t.map(|i| uvs[usize::from(i)])));
            sum += n;
            faces[nf] = n / (2.0 * a);
            nf += 1;
        }
        let axis = sum.normalize_or(DVec3::Z);
        let cos = if sum.length() > 0.0 {
            faces[..nf].iter().map(|f| f.dot(axis)).fold(1.0, f64::min)
        } else {
            -1.0
        };
        Self {
            center,
            radius,
            axis,
            cone: (cos, sine(cos)),
            area,
            emission: weighted,
        }
    }
}

/// The sine of an angle in 0..=pi with cosine `c`.
fn sine(c: f64) -> f64 {
    (1.0 - c * c).max(0.0).sqrt()
}

/// What the sets read of a mesh cluster.
pub(super) struct SetCluster {
    pub(super) self_error: f32,
    pub(super) parent_error: f32,
    pub(super) flags: u8,
    pub(super) section: u16,
    /// On a DAG mesh: the group the cluster is in and the group it was simplified from, or -1
    /// for a stage's source cluster.
    pub(super) group: u32,
    pub(super) refined: i32,
    /// `None` on sections without a set.
    pub(super) bounds: Option<ClusterBounds>,
}

/// Sets the mesh's cut rows and sets from `clusters`, in mesh cluster order.
pub(super) fn build(info: &mut MeshInfo, sections: &[SectionData], clusters: &[SetCluster]) {
    let _z = crate::profile::zone!("mesh.sets");
    let merge = info.header.merge_error;
    if info.levels.is_empty() {
        let pinned_cut = info
            .groups
            .iter()
            .filter(|g| g.page >= info.header.pinned_pages && g.error.is_finite())
            .map(|g| g.error)
            .fold(0.0, f32::max);
        info.cut_rows = std::iter::once(pinned_cut)
            .chain(cut_thresholds(clusters.iter().map(|c| c.self_error)))
            .collect();
    }
    let ranked = sections
        .iter()
        .any(|s| s.flags & (Section::EMISSIVE | Section::CASTER) != 0);
    let rows: Vec<Vec<u32>> = if !ranked {
        Vec::new()
    } else if info.levels.is_empty() {
        let selected = |&t: &f32| {
            (0..clusters.len() as u32)
                .filter(|&i| {
                    let c = &clusters[i as usize];
                    in_cut(c.self_error, c.parent_error, c.flags, t, merge)
                })
                .collect()
        };
        info.cut_rows.iter().map(selected).collect()
    } else {
        let level = |l: &orr_pack::mesh::Level| {
            (l.clusters.first..l.clusters.first + l.clusters.count).collect()
        };
        info.levels.iter().map(level).collect()
    };
    let bound = if !ranked || !info.levels.is_empty() {
        Vec::new()
    } else {
        cut_bound(sections.len(), clusters)
    };
    let double_sided = |s: u16| sections[usize::from(s)].flags & Section::DOUBLE_SIDED != 0;
    info.emissive = sets(
        sections,
        Section::EMISSIVE,
        &rows,
        &bound,
        clusters,
        |b| b.emission,
        double_sided,
    );
    info.casters = sets(
        sections,
        Section::CASTER,
        &rows,
        &bound,
        clusters,
        |b| b.area,
        |_| true,
    );
}

/// Per section of a DAG mesh: the most clusters it can have in any cut a traversal takes.
///
/// A cut of one stage kind is its source clusters with each group of a down-closed set
/// replaced by the clusters simplified from it, so a section's count is its source clusters
/// plus each replaced group's outputs less its inputs in the section, and summing only the
/// groups that add clusters bounds every cut. The two stage kinds never share a cut.
fn cut_bound(section_count: usize, clusters: &[SetCluster]) -> Vec<u32> {
    let mut out = vec![0u32; section_count];
    for merged in [false, true] {
        let mut base = vec![0i64; section_count];
        // Per (group, section): +1 for each cluster simplified from the group, -1 for each in
        // it; summed by sorting.
        let key = |group: u32, section: u16| u64::from(group) << 16 | u64::from(section);
        let mut grows: Vec<(u64, i64)> = Vec::with_capacity(2 * clusters.len());
        for c in clusters
            .iter()
            .filter(|c| (c.flags & Cluster::MERGED != 0) == merged)
        {
            match u32::try_from(c.refined) {
                Ok(from) => grows.push((key(from, c.section), 1)),
                Err(_) => base[usize::from(c.section)] += 1,
            }
            grows.push((key(c.group, c.section), -1));
        }
        grows.sort_unstable_by_key(|g| g.0);
        for run in grows.chunk_by(|a, b| a.0 == b.0) {
            let w: i64 = run.iter().map(|g| g.1).sum();
            base[(run[0].0 & 0xFFFF) as usize] += w.max(0);
        }
        for (o, b) in out.iter_mut().zip(base) {
            *o = (*o).max(b as u32);
        }
    }
    out
}

/// A cluster on its way into a set.
#[derive(Copy, Clone)]
struct Item {
    cluster: u32,
    weight: f64,
    bounds: ClusterBounds,
}

/// The sets of the sections with `flag`. A cluster weighs `weight`; sections `two_sided` send
/// light both ways, so their cones hold every direction. `bound` is empty or bounds each
/// section's clusters in any cut, as [`cut_bound`].
fn sets(
    sections: &[SectionData],
    flag: u8,
    rows: &[Vec<u32>],
    bound: &[u32],
    clusters: &[SetCluster],
    weight: impl Fn(&ClusterBounds) -> f64,
    two_sided: impl Fn(u16) -> bool,
) -> ClusterSets {
    let mut out = ClusterSets::default();
    let mut ranked: Vec<u16> = Vec::new();
    out.ranks = sections
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if s.flags & flag == 0 {
                return NO_RANK;
            }
            ranked.push(i as u16);
            ranked.len() as u32 - 1
        })
        .collect();
    if ranked.is_empty() {
        return out;
    }
    if !bound.is_empty() {
        out.widest = ranked
            .iter()
            .map(|&s| bound[usize::from(s)])
            .fold(0, u32::max);
    }
    // The set each rank took at the row before, which an equal row shares.
    let mut last: Vec<(Vec<u32>, ClusterSet)> =
        vec![(Vec::new(), ClusterSet::default()); ranked.len()];
    for row in rows {
        let mut members: Vec<Vec<Item>> = vec![Vec::new(); ranked.len()];
        let mut in_row = vec![0u32; ranked.len()];
        for &i in row {
            let c = &clusters[i as usize];
            let k = out.ranks[usize::from(c.section)];
            if k == NO_RANK {
                continue;
            }
            in_row[k as usize] += 1;
            let bounds = c.bounds.expect("clusters of ranked sections have bounds");
            let w = weight(&bounds);
            if w > 0.0 {
                members[k as usize].push(Item {
                    cluster: i,
                    weight: w,
                    bounds,
                });
            }
        }
        out.widest = in_row.into_iter().fold(out.widest, u32::max);
        for (k, mut items) in members.into_iter().enumerate() {
            let section = ranked[k];
            let ids: Vec<u32> = items.iter().map(|i| i.cluster).collect();
            if ids == last[k].0 {
                out.sets.push(ClusterSet {
                    section: u32::from(section),
                    ..last[k].1
                });
                continue;
            }
            let set = push_set(&mut out, section, &mut items, two_sided(section));
            out.sets.push(set);
            last[k] = (ids, set);
        }
    }
    out
}

/// Appends the lists and tree of a set of `items`, ascending by cluster.
fn push_set(
    out: &mut ClusterSets,
    section: u16,
    items: &mut [Item],
    two_sided: bool,
) -> ClusterSet {
    let n = items.len() as u32;
    let mut set = ClusterSet {
        first: out.clusters.len() as u32,
        count: n,
        section: u32::from(section),
        ..ClusterSet::default()
    };
    if n > SET_LEAF {
        let mut depth = 0;
        while n.div_ceil(1 << depth) > SET_LEAF {
            depth += 1;
        }
        let mut nodes = vec![SetNode::default(); (1 << (depth + 1)) - 1];
        split(items, &mut nodes, 0, 0, depth, two_sided);
        set.first_node = out.nodes.len() as u32;
        set.node_count = nodes.len() as u32;
        out.nodes.extend(nodes);
    }
    let weights: Vec<f64> = items.iter().map(|i| i.weight).collect();
    let total: f64 = weights.iter().sum();
    set.total = total as f32;
    out.cdf.extend(fixed_cdf(&weights));
    out.clusters.extend(items.iter().map(|i| i.cluster));
    let mut index: Vec<[u32; 2]> = items.iter().zip(0..).map(|(i, p)| [i.cluster, p]).collect();
    index.sort_unstable();
    out.index.extend(index);
    set
}

/// The inclusive sums of `weights` in fixed point: every step at least 1, the last value
/// `u32::MAX`. `weights` are positive.
fn fixed_cdf(weights: &[f64]) -> Vec<u32> {
    let n = weights.len() as u64;
    let total: f64 = weights.iter().sum();
    let scale = f64::from(u32::MAX) / total;
    let mut sum = 0.0;
    let mut previous = 0u64;
    let top = u64::from(u32::MAX);
    (0..n)
        .zip(weights)
        .map(|(i, w)| {
            sum += w;
            let room = top - (n - 1 - i);
            let v = if i + 1 == n {
                top
            } else {
                ((sum * scale).round() as u64).clamp(previous + 1, room)
            };
            previous = v;
            v as u32
        })
        .collect()
}

/// Node `node` over `items`, starting at position `first` in the set, and `depth` levels below
/// it: orders `items` by the tree as it goes.
fn split(
    items: &mut [Item],
    nodes: &mut [SetNode],
    node: usize,
    first: u32,
    depth: u32,
    two_sided: bool,
) {
    nodes[node] = bound(items, first, two_sided);
    if depth == 0 {
        return;
    }
    let (lo, hi) = items
        .iter()
        .map(|i| i.bounds.center)
        .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), c| {
            (lo.min(c), hi.max(c))
        });
    let extent = hi - lo;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };
    let mid = items.len() / 2;
    items.select_nth_unstable_by(mid, |a, b| {
        a.bounds.center[axis]
            .total_cmp(&b.bounds.center[axis])
            .then(a.cluster.cmp(&b.cluster))
    });
    let (left, right) = items.split_at_mut(mid);
    split(left, nodes, 2 * node + 1, first, depth - 1, two_sided);
    split(
        right,
        nodes,
        2 * node + 2,
        first + mid as u32,
        depth - 1,
        two_sided,
    );
}

/// The node over `items`.
fn bound(items: &[Item], first: u32, two_sided: bool) -> SetNode {
    let (lo, hi) = items
        .iter()
        .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), i| {
            let (c, r) = (i.bounds.center, DVec3::splat(i.bounds.radius));
            (lo.min(c - r), hi.max(c + r))
        });
    let center = (0.5 * (lo + hi)).as_vec3();
    let stored = center.as_dvec3();
    let radius = items
        .iter()
        .map(|i| i.bounds.center.distance(stored) + i.bounds.radius)
        .fold(0.0, f64::max);
    let (axis, cone_cos) = cone(items, two_sided);
    SetNode {
        center: center.to_array(),
        radius: (radius as f32).next_up(),
        axis,
        cone_cos,
        first,
        count: items.len() as u32,
    }
}

/// The stored axis of a cone holding the face normals of `items`, and its cosine.
fn cone(items: &[Item], two_sided: bool) -> (u32, f32) {
    let sum: DVec3 = items.iter().map(|i| i.bounds.area * i.bounds.axis).sum();
    let unit = sum.normalize_or(DVec3::Z);
    let axis = oct_encode(unit.as_vec3().to_array());
    if two_sided || sum.length() == 0.0 {
        return (axis, -1.0);
    }
    let stored = DVec3::from(oct_decode(axis).map(f64::from));
    let mut cos = 1.0f64;
    for i in items {
        let c = i.bounds.axis.dot(stored).clamp(-1.0, 1.0);
        let s = sine(c);
        let (ch, sh) = i.bounds.cone;
        // The cosine and sine of the angle to the axis plus the cluster's half angle. A half
        // angle of pi can wrap all the way round to a sine of 0.
        let (sum_c, sum_s) = (c * ch - s * sh, s * ch + c * sh);
        cos = cos.min(if sum_s < 0.0 || ch <= -1.0 {
            -1.0
        } else {
            sum_c
        });
    }
    (axis, ((cos as f32).next_down()).max(-1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_cdf_rises_by_at_least_one_and_ends_at_the_top() {
        let check = |w: &[f64]| {
            let c = fixed_cdf(w);
            assert_eq!(c.len(), w.len());
            assert_eq!(*c.last().unwrap(), u32::MAX);
            assert!(c[0] >= 1 && c.windows(2).all(|p| p[1] > p[0]));
            c
        };
        let c = check(&[1.0, 1.0, 2.0]);
        assert!(c[0].abs_diff(u32::MAX / 4) <= 1 && c[1].abs_diff(u32::MAX / 2) <= 1);
        // Weights far below the fixed-point step keep their one step, at either end.
        check(&[1e-30, 1.0, 1e-30, 1e-30]);
        check(&[1.0, 1e-30, 1e-30]);
        check(&[1e-30; 5]);
    }

    #[test]
    fn a_cut_bound_holds_cuts_that_mix_levels_and_counts_one_stage_kind() {
        // (section, group, refined, merged), each a cluster.
        let c = |section, group, refined, merged: bool| SetCluster {
            self_error: 0.0,
            parent_error: 0.0,
            flags: if merged { Cluster::MERGED } else { 0 },
            section,
            group,
            refined,
            bounds: None,
        };
        let mut clusters = Vec::new();
        // Section 0: group 0 simplifies 3 clusters into 1 and group 1 simplifies 1 into 3, both
        // into root group 2. Either level counts 4; group 1 alone simplified counts 6.
        clusters.extend((0..3).map(|_| c(0, 0, -1, false)));
        clusters.push(c(0, 1, -1, false));
        clusters.push(c(0, 2, 0, false));
        clusters.extend((0..3).map(|_| c(0, 2, 1, false)));
        // Section 1: 4 clusters shrink to 2, then to 1.
        clusters.extend((0..4).map(|_| c(1, 3, -1, false)));
        clusters.extend((0..2).map(|_| c(1, 4, 3, false)));
        clusters.push(c(1, 5, 4, false));
        // Its merged stage: 5 clusters into 2, a cut of its own.
        clusters.extend((0..5).map(|_| c(1, 6, -1, true)));
        clusters.extend((0..2).map(|_| c(1, 7, 6, true)));
        assert_eq!(cut_bound(2, &clusters), [6, 5]);
    }

    fn item(cluster: u32, center: [f64; 3], normal: DVec3, cos: f64) -> Item {
        Item {
            cluster,
            weight: 1.0,
            bounds: ClusterBounds {
                center: DVec3::from(center),
                radius: 0.25,
                axis: normal,
                cone: (cos, sine(cos)),
                area: 1.0,
                emission: 1.0,
            },
        }
    }

    #[test]
    fn a_tree_orders_clusters_by_space_and_bounds_every_cluster_below_each_node() {
        // Clusters on a line, shuffled; each faces a little differently.
        let n: u32 = 70;
        let mut items: Vec<Item> = (0..n)
            .map(|i| {
                let k = (i * 37) % n;
                let tilt = f64::from(k) / f64::from(n) - 0.5;
                let normal = DVec3::new(tilt, 0.0, 1.0).normalize();
                item(k, [f64::from(k), 0.0, 0.0], normal, 0.99)
            })
            .collect();
        let mut out = ClusterSets::default();
        let set = push_set(&mut out, 0, &mut items, false);
        assert_eq!((set.count, set.node_count), (n, 15));
        // 70 clusters split four times hold at most 16 a leaf, along x, ascending.
        let x: Vec<f64> = items.iter().map(|i| i.bounds.center.x).collect();
        let first_leaf = out.nodes[7];
        assert!(first_leaf.count <= SET_LEAF);
        for (i, node) in out.nodes.iter().enumerate() {
            let range = node.first as usize..(node.first + node.count) as usize;
            if 2 * i + 2 < out.nodes.len() {
                let (l, r) = (out.nodes[2 * i + 1], out.nodes[2 * i + 2]);
                assert_eq!((l.first, l.first + l.count), (node.first, r.first));
                assert_eq!(r.first + r.count, node.first + node.count);
                let highest_left = x[l.first as usize..r.first as usize]
                    .iter()
                    .fold(f64::MIN, |a, &b| a.max(b));
                assert!(
                    x[r.first as usize..range.end]
                        .iter()
                        .all(|&v| v > highest_left)
                );
            }
            let center = DVec3::from(node.center.map(f64::from));
            let axis = DVec3::from(oct_decode(node.axis).map(f64::from));
            for it in &items[range] {
                let b = &it.bounds;
                assert!(center.distance(b.center) + b.radius <= f64::from(node.radius));
                // The cluster's own cone sits inside the node's.
                let edge = b.axis.dot(axis).min(1.0).acos() + b.cone.0.acos();
                assert!(edge.cos() >= f64::from(node.cone_cos));
            }
        }
    }

    #[test]
    fn a_cone_past_half_a_turn_holds_every_direction() {
        let items = [
            item(0, [0.0; 3], DVec3::Z, 0.0),
            item(1, [0.0; 3], -DVec3::Z, 0.0),
            item(2, [0.0; 3], DVec3::Z, 0.0),
        ];
        assert_eq!(cone(&items, false).1, -1.0);
        assert_eq!(cone(&items[..1], true).1, -1.0);
        assert!(cone(&items[..1], false).1 < 1e-4);
        // A cluster facing every way, exactly opposite the axis the others set.
        let opposite = [
            item(0, [0.0; 3], -DVec3::Z, 1.0),
            item(1, [0.0; 3], -DVec3::Z, 1.0),
            item(2, [0.0; 3], DVec3::Z, -1.0),
        ];
        assert_eq!(cone(&opposite, false).1, -1.0);
    }
}
