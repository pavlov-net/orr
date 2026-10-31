//! Mesh class: blob 0 holds a [`MeshHeader`] and its tables ([`MeshInfo`]); the blobs after
//! it are the pages of whole clusters ([`PageInfo`]), then in DAG mode the hierarchy's node
//! pages when it is paged ([`NodePage`], records read by [`nodes`]).
//!
//! A page is stored in the codec form [`encode_page`] writes, and [`decode_page`] turns it
//! into the resident layout the GPU reads:
//!
//! - [`PageHeader`], then `cluster_count` [`Cluster`] records of 128 B.
//! - Positions at `positions_offset`: u16x3 per vertex, the grid offset from the cluster's
//!   anchor. Mesh-space position is `(anchor + local) x 2^position_exponent`.
//! - Attributes at `attrs_offset`, `attr_stride` bytes per vertex: normal (octahedral, 2x16
//!   unorm), tangent when the mesh has [`MeshHeader::TANGENT`] (octahedral 2x16 unorm,
//!   bitangent sign in the low bit of y: set means -1), uv (2x16 unorm over the cluster's
//!   `uv_offset` and `uv_scale`), and with [`MeshHeader::SKIN`] joints u8x4 and weights
//!   unorm8x4.
//! - Vertex `i` of a cluster sits at index `vertex_first + i` in both the position and the
//!   attribute array. `vertex_first` is even, so every cluster's positions start 4-aligned.
//! - Triangles at `tris_offset + tri_first`: u8x3 cluster-local corners, opaque triangles
//!   first and alpha-tested ones from `alpha_first`, 4-aligned per cluster.
//! - Opacity micromaps from `omm_offset` to the end of the page ([`Micromaps`]).

mod codec;
mod omm;

use bytemuck::{Pod, Zeroable};

pub use codec::{decode_page, encode_page};
pub use omm::*;

use crate::blob::{put_table, read, table};
use crate::{Aabb, Affine3, Error, Result, Span, Sphere};

pub const SCHEMA: u16 = 5;

/// Largest hierarchy node page.
pub const NODE_PAGE_MAX: usize = 64 << 10;

/// Largest resident page.
pub const PAGE_MAX: usize = 128 << 10;

/// Blob 0 header.
///
/// Spans name tables in blob 0: `first` is the table's byte offset in the blob and `count`
/// its record count.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct MeshHeader {
    /// Rest-pose bounds of the positions as stored on the grid.
    pub bounds: Aabb,
    pub sphere: Sphere,
    /// The finest grid step the mesh was cooked with, log2 of metres.
    pub position_step_log2: i8,
    /// The grid step `k`: positions are integers times `2^k` metres.
    pub position_exponent: i8,
    /// Low mantissa bits a CLAS build may truncate without moving any position.
    pub clas_truncate_bits: u8,
    /// [`Self::TANGENT`] and [`Self::SKIN`].
    pub attributes: u8,
    /// Cluster limits the mesh was cooked with.
    pub max_vertices: u16,
    pub max_triangles: u16,
    /// [`Section`] per material slot.
    pub sections: Span,
    /// [`Part`] table; a mesh without parts has one part covering everything.
    pub parts: Span,
    /// [`PartLevel`] per part and discrete level, part-major.
    pub part_levels: Span,
    /// [`Level`] per discrete level, finest first.
    pub levels: Span,
    /// First discrete level whose parts are merged into the body, or [`Self::NO_MERGE_LEVEL`].
    pub merge_level: u8,
    pub _pad: [u8; 3],
    /// Error from which the merged stage serves in DAG mode; infinity for none.
    pub merge_error: f32,
    /// DAG mode: [`Node`] table when the hierarchy is resident; empty when it is paged.
    pub nodes: Span,
    /// DAG mode: [`Group`] per group, in page order.
    pub groups: Span,
    /// DAG mode: [`NodePage`] per node page when the hierarchy is paged; the first is the
    /// top page.
    pub node_pages: Span,
    /// DAG mode: u32 root node per (part, section), part-major, then one per section for
    /// the merged stage; [`Self::NO_ROOT`] where the stage has no triangles.
    pub roots: Span,
    /// [`PageInfo`] per page.
    pub pages: Span,
    /// Pages `0..pinned_pages` are resident from mount.
    pub pinned_pages: u32,
    /// u32 page indices that [`PageInfo::deps`] ranges index.
    pub page_deps: Span,
}

impl MeshHeader {
    pub const TANGENT: u8 = 1;
    pub const SKIN: u8 = 2;
    pub const NO_MERGE_LEVEL: u8 = 0xFF;
    pub const NO_ROOT: u32 = !0;

    /// Bytes per vertex in the attribute array.
    pub fn attr_stride(&self) -> u32 {
        attr_stride(self.attributes)
    }

    /// The grid step in metres.
    pub fn step(&self) -> f32 {
        exp2(self.position_exponent)
    }
}

/// One material slot of the mesh.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Section {
    /// Index into the model's material table.
    pub material_slot: u16,
    pub flags: u8,
    pub _pad: u8,
}

impl Section {
    pub const ALPHA_TESTED: u8 = 1;
    pub const DOUBLE_SIDED: u8 = 2;
    pub const EMISSIVE: u8 = 4;
}

/// A rigid part.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Part {
    /// Rest-pose bounds in model space; [`Aabb::EMPTY`] for a part without triangles.
    pub bounds: Aabb,
    pub pivot: [f32; 3],
    pub joint: u16,
    pub _pad: u16,
}

/// The clusters of one part at one discrete level: a range of mesh cluster indices.
///
/// Mesh cluster indices count clusters page by page in page order.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct PartLevel {
    pub clusters: Span,
}

/// A discrete level: its simplification error and its range of mesh cluster indices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Level {
    pub error: f32,
    pub clusters: Span,
}

/// One page: the blob that holds it and the size it decodes to.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct PageInfo {
    /// Blob index within the mesh entry.
    pub blob: u32,
    pub resident_len: u32,
    pub cluster_count: u16,
    pub _pad: u16,
    /// Range of [`MeshInfo::page_deps`]: the earlier pages holding the coarser clusters this
    /// page's groups simplify into. Empty in discrete mode.
    pub deps: Span,
}

/// A DAG hierarchy node. Traversal enters a node while its projected `max_error` is over
/// the threshold; a leaf names a group.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Node {
    /// Bounds every group below the node.
    pub sphere: Sphere,
    /// The largest group error below the node; infinity when a terminal group is below.
    pub max_error: f32,
    /// Children are nodes `child_first..child_first + child_count`, in one node page.
    pub child_first: u32,
    pub child_count: u16,
    pub _pad: u16,
    /// Group index for a leaf, -1 for an internal node.
    pub group: i32,
}

/// A DAG group: clusters that stream and are replaced together.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Group {
    /// Bounds and error of the coarser geometry the group simplifies into; every cluster
    /// of the group has them as `parent_sphere` and `parent_error`.
    pub simplified: Sphere,
    /// Infinity for a terminal group.
    pub error: f32,
    pub page: u32,
    /// The group's clusters are `first_cluster..first_cluster + cluster_count` of its page.
    pub first_cluster: u16,
    pub cluster_count: u16,
    /// Simplification steps from the stage's source geometry.
    pub depth: u8,
    pub _pad: [u8; 3],
}

/// A page of hierarchy nodes: a contiguous node range whose parents all sit in one page.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct NodePage {
    pub first_node: u32,
    pub node_count: u32,
    /// Blob index within the mesh entry; the blob holds `node_count` [`Node`] records.
    pub blob: u32,
    /// The page holding these nodes' parents; [`Self::TOP`] for the top page.
    pub parent_page: u32,
}

impl NodePage {
    pub const TOP: u32 = !0;
}

const _: () = assert!(size_of::<Node>() == 32);
const _: () = assert!(size_of::<Group>() == 32);

/// Header of a resident page.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct PageHeader {
    pub cluster_count: u32,
    pub attr_stride: u32,
    pub positions_offset: u32,
    pub attrs_offset: u32,
    pub tris_offset: u32,
    pub omm_offset: u32,
    /// Resident length of the page.
    pub len: u32,
    pub _pad: u32,
    /// 4-state micromaps per subdivision level, for the micromap build.
    pub omm_usage: [u32; 13],
    pub _pad2: [u32; 3],
}

/// One cluster record in a resident page.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Cluster {
    pub self_sphere: Sphere,
    /// Error of the cluster's own geometry; 0 for source clusters.
    pub self_error: f32,
    pub parent_sphere: Sphere,
    /// Error of the next coarser geometry; infinity when there is none.
    pub parent_error: f32,
    pub cull_sphere: Sphere,
    /// Backface cone: axis and cutoff, snorm8.
    pub cone: [i8; 4],
    /// Minimum corner in grid steps.
    pub anchor: [i32; 3],
    pub vertex_first: u32,
    /// Byte offset of the cluster's triangles from the page's `tris_offset`.
    pub tri_first: u32,
    pub vertex_count: u16,
    pub tri_count: u8,
    pub alpha_first: u8,
    pub section: u8,
    pub part: u8,
    pub kind: u8,
    pub flags: u8,
    /// Rest AABB is `(anchor, anchor + local_max) x 2^k`.
    pub local_max: [u16; 3],
    /// f16 bits, metres.
    pub deformation_bound: u16,
    pub displacement_min: u16,
    pub displacement_max: u16,
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],
    /// Byte offset of the cluster's micromap block from `omm_offset`, or [`Self::NO_OMM`].
    pub omm_first: u32,
    /// Triangle area at the cluster's own level, model space, square metres.
    pub area: f32,
    /// f16 bits, metres: on clusters of subsurface sections the smallest distance a ray cast
    /// inward from one of its triangles travels to the geometry drawn with it (infinity when
    /// none hits); 0 elsewhere.
    pub min_thickness: u16,
    pub _pad: u16,
}

impl Cluster {
    pub const TRIANGLES: u8 = 0;
    pub const ANIMATED_PART: u8 = 1;
    pub const MERGED: u8 = 2;
    pub const EMISSIVE: u8 = 4;
    pub const NO_OMM: u32 = !0;
}

const _: () = assert!(size_of::<Cluster>() == 128);
const _: () = assert!(size_of::<PageHeader>() == 96);

/// Blob 0 of a mesh: the header and its tables.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshInfo {
    pub header: MeshHeader,
    pub sections: Vec<Section>,
    pub parts: Vec<Part>,
    pub part_levels: Vec<PartLevel>,
    pub levels: Vec<Level>,
    pub nodes: Vec<Node>,
    pub groups: Vec<Group>,
    pub node_pages: Vec<NodePage>,
    pub roots: Vec<u32>,
    pub pages: Vec<PageInfo>,
    pub page_deps: Vec<u32>,
}

impl MeshInfo {
    pub fn parse(blob: &[u8]) -> Result<Self> {
        let header: MeshHeader = read(blob, 0).ok_or(Error::Malformed("mesh header"))?;
        Ok(Self {
            sections: table(blob, header.sections)?,
            parts: table(blob, header.parts)?,
            part_levels: table(blob, header.part_levels)?,
            levels: table(blob, header.levels)?,
            nodes: table(blob, header.nodes)?,
            groups: table(blob, header.groups)?,
            node_pages: table(blob, header.node_pages)?,
            roots: table(blob, header.roots)?,
            pages: table(blob, header.pages)?,
            page_deps: table(blob, header.page_deps)?,
            header,
        })
    }

    /// Blob 0 bytes; the header's table spans are set from the tables.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut h = self.header;
        let mut out = vec![0; size_of::<MeshHeader>()];
        h.sections = put_table(&mut out, &self.sections);
        h.parts = put_table(&mut out, &self.parts);
        h.part_levels = put_table(&mut out, &self.part_levels);
        h.levels = put_table(&mut out, &self.levels);
        h.nodes = put_table(&mut out, &self.nodes);
        h.groups = put_table(&mut out, &self.groups);
        h.node_pages = put_table(&mut out, &self.node_pages);
        h.roots = put_table(&mut out, &self.roots);
        h.pages = put_table(&mut out, &self.pages);
        h.page_deps = put_table(&mut out, &self.page_deps);
        out[..size_of::<MeshHeader>()].copy_from_slice(bytemuck::bytes_of(&h));
        out
    }
}

/// The [`Node`] records of a node page blob.
pub fn nodes(blob: &[u8]) -> Result<Vec<Node>> {
    if !blob.len().is_multiple_of(size_of::<Node>()) {
        return Err(Error::Malformed("node page length"));
    }
    Ok(blob
        .chunks_exact(size_of::<Node>())
        .map(bytemuck::pod_read_unaligned)
        .collect())
}

/// Bytes per vertex in the attribute array of a mesh with these [`MeshHeader::attributes`].
pub fn attr_stride(attributes: u8) -> u32 {
    let has = |a: u8| u32::from(attributes & a != 0);
    8 + 4 * has(MeshHeader::TANGENT) + 8 * has(MeshHeader::SKIN)
}

/// Bytes a cluster of `vertices` and `triangles` takes in a resident page, its record
/// included and its micromaps not.
pub fn cluster_resident_len(vertices: usize, triangles: usize, attr_stride: usize) -> usize {
    size_of::<Cluster>()
        + vertices.next_multiple_of(2) * (6 + attr_stride)
        + (triangles * 3).next_multiple_of(4)
}

fn exp2(k: i8) -> f32 {
    f32::from_bits(((i32::from(k) + 127) as u32) << 23)
}

/// One page for `orr_pack/page_codec.slang`, which decodes a stored page on the GPU into the
/// bytes [`decode_page`] writes. Addresses are device addresses, both 4-aligned;
/// `resident_capacity` is the bytes the page may take at `resident`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct StoredPage {
    pub stored: u64,
    pub resident: u64,
    pub stored_len: u32,
    pub resident_capacity: u32,
}

/// Where the regions of a resident page lie.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PageLayout {
    pub cluster_count: u32,
    pub positions_offset: u32,
    pub attrs_offset: u32,
    pub attr_stride: u32,
    pub tris_offset: u32,
    pub omm_offset: u32,
    pub len: u32,
}

impl PageLayout {
    fn of(h: &PageHeader) -> Self {
        Self {
            cluster_count: h.cluster_count,
            positions_offset: h.positions_offset,
            attrs_offset: h.attrs_offset,
            attr_stride: h.attr_stride,
            tris_offset: h.tris_offset,
            omm_offset: h.omm_offset,
            len: h.len,
        }
    }
}

/// Checks the regions a page header names; returns the vertex slots and the bytes of the
/// triangle region.
fn layout(h: &PageHeader) -> Result<(usize, usize)> {
    let bad = |what| Err(Error::Malformed(what));
    let n = h.cluster_count as usize;
    let regions_ordered = h.positions_offset as usize
        == size_of::<PageHeader>() + n * size_of::<Cluster>()
        && h.attrs_offset >= h.positions_offset
        && h.tris_offset >= h.attrs_offset
        && h.omm_offset >= h.tris_offset
        && h.len >= h.omm_offset
        && h.len as usize <= PAGE_MAX;
    let aligned = h.attr_stride.is_multiple_of(4)
        && h.attrs_offset.is_multiple_of(4)
        && h.omm_offset.is_multiple_of(4);
    if !regions_ordered || h.attr_stride < 8 || !aligned {
        return bad("page regions");
    }
    let slots = (h.attrs_offset - h.positions_offset) as usize / 6;
    if slots * 6 != (h.attrs_offset - h.positions_offset) as usize
        || slots * h.attr_stride as usize != (h.tris_offset - h.attrs_offset) as usize
    {
        return bad("page vertex arrays");
    }
    Ok((slots, (h.omm_offset - h.tris_offset) as usize))
}

/// Checks cluster `c`'s ranges against the page and against `prev`, the cluster before it:
/// clusters take their vertex slots and triangle bytes in order, so no two overlap.
fn check_cluster(c: &Cluster, prev: Option<&Cluster>, slots: usize, tris_len: usize) -> Result<()> {
    let (v0, vn) = (c.vertex_first as usize, usize::from(c.vertex_count));
    let (t0, tn) = (c.tri_first as usize, usize::from(c.tri_count));
    if !v0.is_multiple_of(2) || v0 + vn > slots || !t0.is_multiple_of(4) || t0 + tn * 3 > tris_len {
        return Err(Error::Malformed("cluster outside its page"));
    }
    if let Some(p) = prev {
        let vertices_end =
            p.vertex_first as usize + usize::from(p.vertex_count).next_multiple_of(2);
        let tris_end = p.tri_first as usize + (usize::from(p.tri_count) * 3).next_multiple_of(4);
        if v0 < vertices_end || t0 < tris_end {
            return Err(Error::Malformed("clusters overlap"));
        }
    }
    if c.alpha_first > c.tri_count {
        return Err(Error::Malformed("cluster alpha_first"));
    }
    Ok(())
}

/// Checks a resident page: its regions, every cluster's ranges and triangle corners.
fn check_page(p: &[u8]) -> Result<PageHeader> {
    let Some(h) = read::<PageHeader>(p, 0) else {
        return Err(Error::Malformed("page shorter than its header"));
    };
    let (slots, tris_len) = layout(&h)?;
    if h.len as usize != p.len() {
        return Err(Error::Malformed("page regions"));
    }
    let mut prev = None;
    for i in 0..h.cluster_count as usize {
        let c: Cluster = read(p, size_of::<PageHeader>() + i * size_of::<Cluster>()).unwrap();
        check_cluster(&c, prev.as_ref(), slots, tris_len)?;
        prev = Some(c);
        let vn = usize::from(c.vertex_count);
        let tris = &p[(h.tris_offset + c.tri_first) as usize..][..usize::from(c.tri_count) * 3];
        if tris.iter().any(|&v| usize::from(v) >= vn) {
            return Err(Error::Malformed("triangle corner outside its cluster"));
        }
    }
    Ok(h)
}

/// CPU reader over a resident page, mirroring the GPU loaders.
pub struct PageView<'a> {
    page: &'a [u8],
    header: PageHeader,
    step: f32,
    tangent: bool,
    skin: bool,
}

impl<'a> PageView<'a> {
    /// Checks the page like [`decode_page`].
    pub fn new(page: &'a [u8], mesh: &MeshHeader) -> Result<Self> {
        let header = check_page(page)?;
        if header.attr_stride != mesh.attr_stride() {
            return Err(Error::Malformed("page attribute stride"));
        }
        Ok(Self {
            page,
            header,
            step: mesh.step(),
            tangent: mesh.attributes & MeshHeader::TANGENT != 0,
            skin: mesh.attributes & MeshHeader::SKIN != 0,
        })
    }

    pub fn cluster_count(&self) -> u32 {
        self.header.cluster_count
    }

    pub fn cluster(&self, i: u32) -> Cluster {
        assert!(i < self.header.cluster_count);
        read(
            self.page,
            size_of::<PageHeader>() + i as usize * size_of::<Cluster>(),
        )
        .unwrap()
    }

    /// Grid position of a vertex: `anchor + local`.
    pub fn grid(&self, c: &Cluster, v: u32) -> [i32; 3] {
        let local: [u16; 3] = self.vertex(self.header.positions_offset, 6, c, v);
        [0, 1, 2].map(|a| c.anchor[a] + i32::from(local[a]))
    }

    /// Mesh-space position of a vertex; exact, since grid values fit 24 bits.
    pub fn position(&self, c: &Cluster, v: u32) -> [f32; 3] {
        self.grid(c, v).map(|g| g as f32 * self.step)
    }

    pub fn normal(&self, c: &Cluster, v: u32) -> [f32; 3] {
        oct_decode(self.attr(c, v, 0))
    }

    /// Tangent xyz and bitangent sign, when the mesh has tangents.
    pub fn tangent(&self, c: &Cluster, v: u32) -> Option<[f32; 4]> {
        self.tangent.then(|| {
            let t = self.attr(c, v, 4);
            let [x, y, z] = oct_decode(t);
            [x, y, z, if t & 1 << 16 != 0 { -1.0 } else { 1.0 }]
        })
    }

    pub fn uv(&self, c: &Cluster, v: u32) -> [f32; 2] {
        let q = self.attr(c, v, if self.tangent { 8 } else { 4 });
        let q = [q & 0xFFFF, q >> 16];
        [0, 1].map(|a| c.uv_offset[a] + q[a] as f32 / 65535.0 * c.uv_scale[a])
    }

    /// Skeleton joints of a vertex, when the mesh has [`MeshHeader::SKIN`].
    pub fn joints(&self, c: &Cluster, v: u32) -> Option<[u8; 4]> {
        self.skin
            .then(|| self.attr(c, v, self.skin_at()).to_le_bytes())
    }

    /// Weights of [`Self::joints`], summing to 1.
    pub fn weights(&self, c: &Cluster, v: u32) -> Option<[f32; 4]> {
        self.skin.then(|| {
            let w = self.attr(c, v, self.skin_at() + 4).to_le_bytes();
            w.map(|q| f32::from(q) / 255.0)
        })
    }

    fn skin_at(&self) -> u32 {
        if self.tangent { 12 } else { 8 }
    }

    pub fn triangle(&self, c: &Cluster, t: u32) -> [u8; 3] {
        assert!(t < u32::from(c.tri_count));
        let at = (self.header.tris_offset + c.tri_first + t * 3) as usize;
        self.page[at..at + 3].try_into().unwrap()
    }

    /// Micromap index of triangle `t`: an entry of the page's [`Micromaps`], or negative for
    /// a special index. `None` for opaque triangles (before `alpha_first`) and clusters
    /// without micromaps.
    pub fn omm_index(&self, c: &Cluster, t: u32) -> Option<i16> {
        assert!(t < u32::from(c.tri_count));
        let alpha = u32::from(c.alpha_first);
        if c.omm_first == Cluster::NO_OMM || t < alpha {
            return None;
        }
        let at = [self.header.omm_offset, c.omm_first, 2 * (t - alpha)]
            .map(|o| o as usize)
            .iter()
            .sum();
        Some(read(self.page, at).expect("micromap index outside the page"))
    }

    fn attr(&self, c: &Cluster, v: u32, at: u32) -> u32 {
        let base = self.header.attrs_offset + at;
        self.vertex(base, self.header.attr_stride, c, v)
    }

    fn vertex<T: Pod>(&self, base: u32, stride: u32, c: &Cluster, v: u32) -> T {
        assert!(v < u32::from(c.vertex_count));
        read(self.page, (base + (c.vertex_first + v) * stride) as usize).unwrap()
    }
}

/// Linear blend skinning of a cluster's vertices: vertex `i` lands in `out[i]` at
/// `sum_k weight_k x (palette[joint_k] x position)`, in mesh space. `palette` holds the
/// skinning matrices ([`crate::skeleton::SkeletonInfo::palette`]).
///
/// Panics if the mesh has no [`MeshHeader::SKIN`], a vertex names a joint past `palette`,
/// or `out` is shorter than the cluster's vertices.
pub fn skin(view: &PageView<'_>, c: &Cluster, palette: &[Affine3], out: &mut [[f32; 3]]) {
    assert!(view.skin, "the mesh has no skin");
    let n = usize::from(c.vertex_count);
    for (v, o) in (0..).zip(&mut out[..n]) {
        let p = view.position(c, v);
        let joints = view.joints(c, v).unwrap();
        let weights = view.weights(c, v).unwrap();
        *o = [0.0; 3];
        for (j, w) in joints.into_iter().zip(weights) {
            if w > 0.0 {
                let q = palette[usize::from(j)].transform_point(p);
                (0..3).for_each(|a| o[a] += w * q[a]);
            }
        }
    }
}

/// Octahedral encoding of a unit vector as 2x16 unorm in one u32 (x low).
pub fn oct_encode(n: [f32; 3]) -> u32 {
    let l = n[0].abs() + n[1].abs() + n[2].abs();
    let (mut x, mut y) = (n[0] / l, n[1] / l);
    if n[2] < 0.0 {
        (x, y) = ((1.0 - y.abs()) * sign(x), (1.0 - x.abs()) * sign(y));
    }
    let q = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 65535.0).round() as u32;
    q(x) | q(y) << 16
}

pub fn oct_decode(e: u32) -> [f32; 3] {
    let f = |q: u32| (q & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0;
    let (mut x, mut y) = (f(e), f(e >> 16));
    let z = 1.0 - x.abs() - y.abs();
    if z < 0.0 {
        (x, y) = ((1.0 - y.abs()) * sign(x), (1.0 - x.abs()) * sign(y));
    }
    let l = (x * x + y * y + z * z).sqrt();
    [x / l, y / l, z / l]
}

fn sign(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}
