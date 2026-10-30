//! Readers for cooked DAG meshes: clusters with their triangles, cuts by the cluster rule
//! and through the hierarchy, and coverage counts of projected triangles.

use glam::{Vec2, Vec3};
use orr_pack::mesh::{
    self, Cluster, Group, MeshHeader, MeshInfo, NODE_PAGE_MAX, Node, PAGE_MAX, PageView,
};
use orr_pack::{Archive, AssetId, Sphere};

pub(crate) fn read(a: &Archive, b: &orr_pack::Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

/// A cooked cluster: its record, page and triangle positions.
pub(crate) struct C {
    pub(crate) record: Cluster,
    pub(crate) page: usize,
    pub(crate) tris: Vec<[[f32; 3]; 3]>,
}

pub(crate) struct Dag {
    pub(crate) info: MeshInfo,
    pub(crate) clusters: Vec<C>,
    /// Cluster indices of each page, in page order.
    pub(crate) by_page: Vec<Vec<usize>>,
    pub(crate) nodes: Vec<Node>,
}

pub(crate) fn load(a: &Archive, name: &str) -> Dag {
    let e = a.find(AssetId::from_name(name)).unwrap();
    let blobs = a.blobs(e);
    let info = MeshInfo::parse(&read(a, &blobs[0])).unwrap();
    let mut clusters = Vec::new();
    let mut by_page = Vec::new();
    for (pi, p) in info.pages.iter().enumerate() {
        let b = &blobs[p.blob as usize];
        assert_eq!(b.is_mount(), (pi as u32) < info.header.pinned_pages);
        assert!(p.resident_len as usize <= PAGE_MAX);
        let mut page = vec![0; p.resident_len as usize];
        mesh::decode_page(&read(a, b), &mut page).unwrap();
        let view = PageView::new(&page, &info.header).unwrap();
        let mut here = Vec::new();
        for i in 0..view.cluster_count() {
            let c = view.cluster(i);
            let tris = (0..u32::from(c.tri_count))
                .map(|t| {
                    view.triangle(&c, t)
                        .map(|v| view.position(&c, u32::from(v)))
                })
                .collect();
            here.push(clusters.len());
            clusters.push(C {
                record: c,
                page: pi,
                tris,
            });
        }
        by_page.push(here);
    }
    let nodes = if info.node_pages.is_empty() {
        info.nodes.clone()
    } else {
        assert!(info.nodes.is_empty());
        let mut nodes = Vec::new();
        for (i, np) in info.node_pages.iter().enumerate() {
            let b = &blobs[np.blob as usize];
            assert_eq!(b.is_mount(), i == 0, "the top node page is mounted");
            assert_eq!(np.first_node as usize, nodes.len());
            let page = mesh::nodes(&read(a, b)).unwrap();
            assert_eq!(page.len(), np.node_count as usize);
            assert!(page.len() * size_of::<Node>() <= NODE_PAGE_MAX);
            nodes.extend(page);
        }
        nodes
    };
    Dag {
        info,
        clusters,
        by_page,
        nodes,
    }
}

/// The clusterlod projected error of `error` over `sphere` from `eye`.
pub(crate) fn projected(sphere: &Sphere, error: f32, eye: Vec3) -> f32 {
    if error == f32::INFINITY {
        return f32::INFINITY;
    }
    let d = (Vec3::from(sphere.center) - eye).length() - sphere.radius;
    error / d.max(0.01)
}

/// A cut: a uniform threshold, or a projected one from an eye.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Cut {
    Uniform(f32),
    View(Vec3, f32),
}

impl Cut {
    pub(crate) fn over(self, sphere: &Sphere, error: f32) -> bool {
        match self {
            Cut::Uniform(t) => error > t,
            Cut::View(eye, t) => projected(sphere, error, eye) > t,
        }
    }

    pub(crate) fn selects(self, c: &Cluster) -> bool {
        !self.over(&c.self_sphere, c.self_error) && self.over(&c.parent_sphere, c.parent_error)
    }
}

impl Dag {
    pub(crate) fn group_of(&self, g: &Group) -> std::ops::Range<usize> {
        let page = &self.by_page[g.page as usize];
        let f = usize::from(g.first_cluster);
        page[f]..page[f] + usize::from(g.cluster_count)
    }

    /// Clusters a cut selects by the cluster rule.
    pub(crate) fn select(&self, cut: Cut, keep: impl Fn(&Cluster) -> bool) -> Vec<usize> {
        (0..self.clusters.len())
            .filter(|&i| keep(&self.clusters[i].record) && cut.selects(&self.clusters[i].record))
            .collect()
    }

    /// Clusters a cut selects through the hierarchy from `roots`.
    pub(crate) fn traverse(&self, cut: Cut, roots: &[u32]) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack: Vec<u32> = roots
            .iter()
            .copied()
            .filter(|&r| r != MeshHeader::NO_ROOT)
            .collect();
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n as usize];
            if !cut.over(&node.sphere, node.max_error) {
                continue;
            }
            if node.group >= 0 {
                let g = &self.info.groups[node.group as usize];
                out.extend(self.group_of(g).filter(|&c| {
                    !cut.over(
                        &self.clusters[c].record.self_sphere,
                        self.clusters[c].record.self_error,
                    )
                }));
            } else {
                stack.extend(node.child_first..node.child_first + u32::from(node.child_count));
            }
        }
        out.sort_unstable();
        out
    }

    /// Triangles of the part stage per group depth.
    pub(crate) fn part_triangles_per_depth(&self) -> Vec<u64> {
        let mut per_depth = Vec::new();
        for g in &self.info.groups {
            let depth = usize::from(g.depth);
            if per_depth.len() <= depth {
                per_depth.resize(depth + 1, 0);
            }
            per_depth[depth] += self
                .group_of(g)
                .map(|c| &self.clusters[c].record)
                .filter(|r| r.flags & Cluster::MERGED == 0)
                .map(|r| u64::from(r.tri_count))
                .sum::<u64>();
        }
        per_depth
    }

    /// 0 and thresholds just over every distinct group error up to `max`.
    pub(crate) fn thresholds(&self, max: f32) -> Vec<f32> {
        let mut t: Vec<f32> = self
            .info
            .groups
            .iter()
            .map(|g| g.error * 1.0001)
            .filter(|&e| e <= max)
            .collect();
        t.push(0.0);
        t.sort_by(f32::total_cmp);
        t.dedup();
        t
    }
}

/// Samples of a square in xz that triangles projected along y cover, counted per sample;
/// samples on an edge are left out.
pub(crate) struct Coverage {
    pub(crate) lo: Vec2,
    pub(crate) size: f32,
    pub(crate) n: usize,
    pub(crate) count: Vec<u32>,
    pub(crate) edge: Vec<bool>,
}

impl Coverage {
    pub(crate) fn new(lo: Vec2, size: f32, n: usize) -> Self {
        Self {
            lo,
            size,
            n,
            count: vec![0; n * n],
            edge: vec![false; n * n],
        }
    }

    pub(crate) fn sample(&self, i: usize, j: usize) -> Vec2 {
        // Offsets off the dyadic grid, so samples rarely sit on an edge.
        self.lo + Vec2::new(i as f32 + 0.371, j as f32 + 0.613) * (self.size / self.n as f32)
    }

    pub(crate) fn add(&mut self, t: &[[f32; 3]; 3]) {
        let p = t.map(|v| Vec2::new(v[0], v[2]));
        let lo = p[0].min(p[1]).min(p[2]);
        let hi = p[0].max(p[1]).max(p[2]);
        let cell = |v: f32, o: f32| ((v - o) / self.size * self.n as f32).floor() as i64;
        let (i0, i1) = (
            cell(lo.x, self.lo.x).max(0),
            cell(hi.x, self.lo.x).min(self.n as i64 - 1),
        );
        let (j0, j1) = (
            cell(lo.y, self.lo.y).max(0),
            cell(hi.y, self.lo.y).min(self.n as i64 - 1),
        );
        let orient = |a: Vec2, b: Vec2, c: Vec2| {
            f64::from(b.x - a.x) * f64::from(c.y - a.y)
                - f64::from(b.y - a.y) * f64::from(c.x - a.x)
        };
        for j in j0..=j1 {
            for i in i0..=i1 {
                let s = self.sample(i as usize, j as usize);
                let o = [
                    orient(p[0], p[1], s),
                    orient(p[1], p[2], s),
                    orient(p[2], p[0], s),
                ];
                let k = j as usize * self.n + i as usize;
                if o.contains(&0.0) {
                    self.edge[k] = true;
                } else if o.iter().all(|&x| x > 0.0) || o.iter().all(|&x| x < 0.0) {
                    self.count[k] += 1;
                }
            }
        }
    }

    /// Samples inside `keep` covered other than once.
    pub(crate) fn wrong(&self, keep: impl Fn(Vec2) -> bool) -> usize {
        let mut wrong = 0;
        for j in 0..self.n {
            for i in 0..self.n {
                let k = j * self.n + i;
                if !self.edge[k] && keep(self.sample(i, j)) && self.count[k] != 1 {
                    wrong += 1;
                }
            }
        }
        wrong
    }
}

pub(crate) fn covers_once(
    d: &Dag,
    selected: &[usize],
    lo: Vec2,
    size: f32,
    keep: impl Fn(Vec2) -> bool,
) -> usize {
    let mut cov = Coverage::new(lo, size, 192);
    for &c in selected {
        for t in &d.clusters[c].tris {
            cov.add(t);
        }
    }
    cov.wrong(keep)
}

/// Covered pixels of triangles projected along each axis onto a square of side `view.1`
/// centred on `view.0`, `n` x `n` pixels: the six axis views, since opposite orthographic
/// views cover the same pixels mirrored.
pub(crate) fn coverage(tris: &[[[f32; 3]; 3]], view: (Vec3, f32), n: usize) -> [usize; 3] {
    let (centre, size) = view;
    [(0, 1), (1, 2), (0, 2)].map(|(x, y)| {
        let lo = Vec2::new(centre[x], centre[y]) - size / 2.0;
        let mut cov = Coverage::new(lo, size, n);
        for t in tris {
            cov.add(&t.map(|p| [p[x], 0.0, p[y]]));
        }
        cov.count.iter().filter(|&&c| c > 0).count()
    })
}
