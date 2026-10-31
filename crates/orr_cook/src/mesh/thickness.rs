//! Minimum thickness of subsurface clusters.
//!
//! Each triangle of a subsurface cluster casts rays along its reversed face normal, from its
//! centroid and from the points halfway between the centroid and each corner, against the
//! geometry drawn with the cluster: the cluster itself and every cluster of the same stage
//! (part or merged) whose `self_error <= e < parent_error` at the cluster's own
//! `self_error` e. The smallest hit distance, rounded down to f16, is the cluster's
//! `min_thickness`; f16 infinity when no ray hits.

use orr_math::DVec3;
use rayon::prelude::*;

use super::{ClusterData, Grid, SectionData, f16_down};
use orr_pack::mesh::Cluster;

/// Triangles per BVH leaf.
const LEAF: usize = 4;

/// Sets `min_thickness` on the clusters of subsurface sections.
pub(super) fn assign(clusters: Vec<&mut ClusterData>, grid: &Grid, sections: &[SectionData]) {
    let subsurface = |c: &ClusterData| sections[usize::from(c.record.section)].subsurface;
    if !clusters.iter().any(|c| subsurface(c)) {
        return;
    }
    let scene = Scene::new(&clusters, grid);
    let values: Vec<Option<u16>> = clusters
        .par_iter()
        .enumerate()
        .map(|(i, c)| subsurface(c).then(|| scene.cluster_thickness(i)))
        .collect();
    for (c, v) in clusters.into_iter().zip(values) {
        if let Some(v) = v {
            c.record.min_thickness = v;
        }
    }
}

/// Which clusters a query sees.
#[derive(Copy, Clone)]
struct Cut {
    error: f32,
    merged: bool,
}

#[derive(Copy, Clone)]
struct Errors {
    self_error: f32,
    parent_error: f32,
    merged: bool,
}

impl Errors {
    fn of(r: &Cluster) -> Self {
        Self {
            self_error: r.self_error,
            parent_error: r.parent_error,
            merged: r.flags & Cluster::MERGED != 0,
        }
    }

    fn drawn_at(self, cut: Cut) -> bool {
        self.merged == cut.merged && self.self_error <= cut.error && cut.error < self.parent_error
    }
}

struct Node {
    lo: DVec3,
    hi: DVec3,
    /// Smallest `self_error` and largest `parent_error` below the node.
    self_min: f32,
    parent_max: f32,
    /// Bit 0: holds part-stage triangles; bit 1: merged-stage triangles.
    stages: u8,
    /// Leaf: first triangle and count; internal: first child (the second follows) and 0.
    first: u32,
    count: u32,
}

struct Scene {
    /// Corners per triangle, cluster by cluster.
    tris: Vec<[DVec3; 3]>,
    /// Cluster per triangle.
    cluster_of: Vec<u32>,
    /// Each cluster's first triangle, then the triangle count.
    first: Vec<usize>,
    /// Triangles in BVH order.
    order: Vec<u32>,
    errors: Vec<Errors>,
    nodes: Vec<Node>,
    /// Hits closer than this are the surface itself.
    t_min: f64,
}

impl Scene {
    fn new(clusters: &[&mut ClusterData], grid: &Grid) -> Self {
        let step = grid.step();
        let own: Vec<Vec<[DVec3; 3]>> = clusters
            .par_iter()
            .map(|c| {
                let corner = |i: u8| {
                    let l = c.local[usize::from(i)];
                    DVec3::from_array(
                        [0, 1, 2].map(|a| (f64::from(c.record.anchor[a]) + f64::from(l[a])) * step),
                    )
                };
                c.tris
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| t.map(corner))
                    .collect()
            })
            .collect();
        let mut first = vec![0];
        let mut cluster_of = Vec::new();
        for (ci, t) in own.iter().enumerate() {
            cluster_of.extend(std::iter::repeat_n(ci as u32, t.len()));
            first.push(cluster_of.len());
        }
        let mut s = Self {
            tris: own.concat(),
            cluster_of,
            first,
            order: Vec::new(),
            errors: clusters.iter().map(|c| Errors::of(&c.record)).collect(),
            nodes: Vec::new(),
            t_min: step / 8.0,
        };
        let mut order: Vec<u32> = (0..s.tris.len() as u32).collect();
        s.nodes.push(s.node(&order, 0));
        s.split(0, &mut order, 0);
        s.order = order;
        s
    }

    /// A leaf over `order`, whose triangles start at `first`.
    fn node(&self, order: &[u32], first: usize) -> Node {
        let mut n = Node {
            lo: DVec3::INFINITY,
            hi: DVec3::NEG_INFINITY,
            self_min: f32::INFINITY,
            parent_max: f32::NEG_INFINITY,
            stages: 0,
            first: first as u32,
            count: order.len() as u32,
        };
        for &t in order {
            for p in self.tris[t as usize] {
                n.lo = n.lo.min(p);
                n.hi = n.hi.max(p);
            }
            let s = self.errors[self.cluster_of[t as usize] as usize];
            n.self_min = n.self_min.min(s.self_error);
            n.parent_max = n.parent_max.max(s.parent_error);
            n.stages |= 1 << u8::from(s.merged);
        }
        n
    }

    /// Splits leaf `n` over `order` (starting at triangle `first`) at the median centroid
    /// of its longest axis, until leaves hold at most [`LEAF`] triangles.
    fn split(&mut self, n: usize, order: &mut [u32], first: usize) {
        if order.len() <= LEAF {
            return;
        }
        let axis = (self.nodes[n].hi - self.nodes[n].lo).max_position();
        let centroid = |t: u32| self.tris[t as usize].iter().map(|p| p[axis]).sum::<f64>();
        let mid = order.len() / 2;
        order.select_nth_unstable_by(mid, |&a, &b| centroid(a).total_cmp(&centroid(b)));
        let (left, right) = order.split_at_mut(mid);
        let child = self.nodes.len();
        let l = self.node(left, first);
        let r = self.node(right, first + mid);
        self.nodes.extend([l, r]);
        self.nodes[n].first = child as u32;
        self.nodes[n].count = 0;
        self.split(child, left, first);
        self.split(child + 1, right, first + mid);
    }

    fn cluster_thickness(&self, cluster: usize) -> u16 {
        let span = self.errors[cluster];
        let cut = Cut {
            error: span.self_error,
            merged: span.merged,
        };
        // A cluster outside its own cut (equal errors) still sees itself.
        let tris = &self.tris[self.first[cluster]..self.first[cluster + 1]];
        let own = (!span.drawn_at(cut)).then_some(tris);
        let mut best = f64::INFINITY;
        let mut stack = Vec::new();
        for t in tris {
            let n = (t[1] - t[0]).cross(t[2] - t[0]);
            if n.length_squared() == 0.0 {
                continue;
            }
            let dir = -n.normalize();
            let c = (t[0] + t[1] + t[2]) / 3.0;
            for o in [c, (c + t[0]) / 2.0, (c + t[1]) / 2.0, (c + t[2]) / 2.0] {
                best = self.cast(o, dir, cut, best, &mut stack);
                for u in own.into_iter().flatten() {
                    if let Some(d) = hit(o, dir, u)
                        && d > self.t_min
                    {
                        best = best.min(d);
                    }
                }
            }
        }
        if best.is_finite() {
            f16_down(best as f32)
        } else {
            half::f16::INFINITY.to_bits()
        }
    }

    /// The nearest hit closer than `best` among the triangles drawn at `cut`.
    fn cast(&self, o: DVec3, dir: DVec3, cut: Cut, mut best: f64, stack: &mut Vec<usize>) -> f64 {
        let inv = dir.recip();
        let stage = 1 << u8::from(cut.merged);
        stack.push(0);
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n];
            if node.stages & stage == 0
                || node.self_min > cut.error
                || node.parent_max <= cut.error
                || !box_hit(o, inv, node.lo, node.hi, best)
            {
                continue;
            }
            if node.count == 0 {
                stack.extend([node.first as usize, node.first as usize + 1]);
                continue;
            }
            let range = node.first as usize..(node.first + node.count) as usize;
            for &i in &self.order[range] {
                let i = i as usize;
                if !self.errors[self.cluster_of[i] as usize].drawn_at(cut) {
                    continue;
                }
                if let Some(d) = hit(o, dir, &self.tris[i])
                    && d > self.t_min
                {
                    best = best.min(d);
                }
            }
        }
        best
    }
}

/// Whether the ray enters the box before `t_max`.
fn box_hit(o: DVec3, inv: DVec3, lo: DVec3, hi: DVec3, t_max: f64) -> bool {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    // NaN from 0 x infinity (an origin on a slab plane) counts as inside that slab.
    let near = a.min(b).to_array().into_iter().filter(|v| !v.is_nan());
    let far = a.max(b).to_array().into_iter().filter(|v| !v.is_nan());
    let enter = near.fold(0.0, f64::max);
    let exit = far.fold(t_max, f64::min);
    enter <= exit
}

/// Distance along the ray to triangle `t` from either side, edges included.
fn hit(o: DVec3, dir: DVec3, t: &[DVec3; 3]) -> Option<f64> {
    let e1 = t[1] - t[0];
    let e2 = t[2] - t[0];
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 * e1.length() * e2.length() {
        return None;
    }
    let s = o - t[0];
    let u = s.dot(p) / det;
    let q = s.cross(e1);
    let v = dir.dot(q) / det;
    const EDGE: f64 = 1e-9;
    if u < -EDGE || v < -EDGE || u + v > 1.0 + EDGE {
        return None;
    }
    Some(e2.dot(q) / det)
}
