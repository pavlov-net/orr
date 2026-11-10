// Resident mesh pages (orr_pack mesh pages, the layout `orr_pack::mesh::decode_page` writes).
// The structs match orr_pack::mesh's #[repr(C)] layouts; the loaders follow
// orr_pack::mesh::PageView. Every function takes the page's address; the module holds no
// bindings.
module mesh;

import orr_shaders.packing;

// Largest resident page, bytes.
public static const uint PAGE_MAX = 128 << 10;

// `MeshHeader.attributes`.
public static const uint MESH_TANGENT = 1;
public static const uint MESH_SKIN = 2;
public static const uint MESH_UV1 = 4;
public static const uint MESH_COLOR = 8;

// `Cluster.flags`.
public static const uint CLUSTER_ANIMATED_PART = 1;
public static const uint CLUSTER_MERGED = 2;
public static const uint CLUSTER_TANGENT_SET_SHIFT = 3; // 2 bits: what the mesh's tangents here follow
public static const uint CLUSTER_CLASS_SHIFT = 5; // 2 bits: orr_shaders.trace CLASS_OPAQUE, _DIELECTRIC or _PRESENCE
public static const uint CLUSTER_SOLID = 128;     // a dielectric with an interior: its triangles before `alpha_first` trace opaque

// `Cluster.omm_first` of a cluster without micromaps.
public static const uint CLUSTER_NO_OMM = 0xffffffff;
// `Cluster.from_group` of a cluster simplified from no group.
public static const uint CLUSTER_SOURCE = 0xffffffff;

// Cluster `c`'s class, from orr_shaders.trace.
public uint cluster_class(Cluster c)
{
    return (c.flags >> CLUSTER_CLASS_SHIFT) & 3;
}

// 96 B. Cluster records follow it.
public struct PageHeader
{
    public uint cluster_count;
    public uint attr_stride;
    public uint positions_offset;
    public uint attrs_offset;
    public uint tris_offset;
    public uint omm_offset;
    public uint len;
    public uint _pad;
    public uint omm_usage[13];
    public uint _pad2[3];
}

// A `Cluster.cone` that culls nothing: cutoff 1 in its top byte.
public static const uint CONE_NONE = 0x7f000000u;

// 148 B. Spheres are (center, radius) in mesh space.
public struct Cluster
{
    public float4 self_sphere;
    public float self_error;
    public float4 parent_sphere;
    public float parent_error;
    public float4 cull_sphere;
    public uint cone;              // snorm8 axis xyz, cutoff w; CONE_NONE culls nothing
    public int3 anchor;            // grid steps
    public uint vertex_first;
    public uint tri_first;         // bytes from tris_offset
    public uint16_t vertex_count;
    public uint8_t tri_count;
    public uint8_t alpha_first;
    public uint16_t section;
    public uint8_t flat_triangles; // triangles from 0 whose vertex normals equal the face normal
    public uint8_t flags;
    public uint16_t local_max[3];
    public half deformation_bound; // m
    public half displacement_min;  // m
    public half displacement_max;  // m
    public float2 uv_offset[2];    // per uv set
    public float2 uv_scale[2];
    public uint omm_first;         // bytes from omm_offset
    public float area;             // m^2
    public half min_thickness;     // m
    public uint16_t part;
    public uint from_group;        // DAG mode: the group whose simplification made it, or CLUSTER_SOURCE
}

// A set tree leaf holds at most this many clusters; a set no larger has no tree.
public static const uint SET_LEAF = 16;
// `SetTablesGpu.ranks` entry of a section with no set.
public static const uint NO_RANK = 0xffffffff;

// 24 B. One section's clusters at one cut row (orr_pack::mesh::ClusterSet).
public struct ClusterSet
{
    public uint first;             // in `clusters`, `cdf` and `index`
    public uint count;
    public uint first_node;
    public uint node_count;
    public uint section;
    public float total;            // m^2, weighted
}

// 32 B. A node of a set's heap, children at 2i + 1 and 2i + 2.
public struct SetNode
{
    public float3 center;          // mesh space
    public float radius;
    public uint axis;              // oct 2x16 unorm
    public float cone_cos;         // -1: any direction
    public uint first;             // positions in the set
    public uint count;
}

// 32 B. A DAG hierarchy node: a leaf names a group (`group` >= 0), an internal node has
// children `child_first..child_first + child_count`, all in one node page.
public struct Node
{
    public float4 sphere;          // bounds every group below
    public float max_error;        // the largest group error below; infinity over a terminal group
    public uint child_first;
    public uint16_t child_count;
    public uint16_t _pad;
    public int group;              // -1 for an internal node
}

// 36 B. A DAG group: clusters `first_cluster..first_cluster + cluster_count` of page `page`,
// each with `simplified` and `error` as its parent bounds.
public struct Group
{
    public float4 simplified;
    public float error;            // infinity for a terminal group
    public uint page;
    public uint16_t first_cluster;
    public uint16_t cluster_count;
    public uint8_t depth;          // simplification steps from the stage's source geometry
    public uint8_t _pad[3];
    public uint node_page;         // the node page holding its leaf; 0 in a resident hierarchy
}

// 16 B. Nodes `first_node..first_node + node_count` of a paged hierarchy.
public struct NodePage
{
    public uint first_node;
    public uint node_count;
    public uint blob;
    public uint parent_page;       // NODE_PAGE_TOP for the top page
}

public static const uint NODE_PAGE_TOP = 0xffffffff;
// `MeshHeader.roots` where a stage has no triangles.
public static const uint NO_ROOT = 0xffffffff;

public struct VertexAttributes
{
    public float3 normal;
    public float4 tangent;         // xyz and bitangent sign; (1, 0, 0, 1) without MESH_TANGENT
    public float2 uv[2];           // per uv set; set 1 is zero without MESH_UV1
    public float4 color;           // 1 without MESH_COLOR
    public uint4 joints;           // MESH_SKIN only
    public float4 weights;         // MESH_SKIN only; sum to 1
}

public PageHeader load_page_header(uint8_t* page)
{
    return *(PageHeader*)page;
}

// Where cluster `cluster` of a page is stored.
public Cluster* cluster_at(uint8_t* page, uint cluster)
{
    return (Cluster*)(page + 96) + cluster;
}

public Cluster load_cluster(uint8_t* page, uint cluster)
{
    return *cluster_at(page, cluster);
}

// Grid offsets of a vertex from the cluster's anchor.
public uint3 load_local(uint8_t* page, PageHeader h, Cluster c, uint i)
{
    uint16_t* p = (uint16_t*)(page + h.positions_offset) + (c.vertex_first + i) * 3;
    return uint3(p[0], p[1], p[2]);
}

// Mesh-space position: (anchor + local) x step, step = 2^k (`grid_step`). Bit for bit
// `orr_pack::mesh::PageView::position`.
public float3 load_position(uint8_t* page, PageHeader h, Cluster c, uint i, float step)
{
    return grid_position(c.anchor, load_local(page, h, c, i), step);
}

float2 load_uv(uint q, Cluster c, uint set)
{
    return c.uv_offset[set] + unpack_unorm2x16(q) * c.uv_scale[set];
}

// `attributes` is the mesh's `MeshHeader.attributes`.
public VertexAttributes load_attributes(uint8_t* page, PageHeader h, Cluster c, uint i, uint attributes)
{
    uint* a = (uint*)(page + h.attrs_offset + (c.vertex_first + i) * h.attr_stride);
    VertexAttributes v;
    v.normal = oct_decode_unorm16(a[0]);
    uint at = 1;
    v.tangent = float4(1.0, 0.0, 0.0, 1.0);
    if ((attributes & MESH_TANGENT) != 0)
    {
        uint t = a[at++];
        v.tangent = float4(oct_decode_unorm16(t), (t & (1u << 16)) != 0 ? -1.0 : 1.0);
    }
    v.uv[0] = load_uv(a[at++], c, 0);
    v.uv[1] = float2(0.0);
    if ((attributes & MESH_UV1) != 0)
        v.uv[1] = load_uv(a[at++], c, 1);
    v.color = float4(1.0);
    if ((attributes & MESH_COLOR) != 0)
    {
        v.color = float4(unpack_half2x16(a[at]), unpack_half2x16(a[at + 1]));
        at += 2;
    }
    v.joints = uint4(0);
    v.weights = float4(0.0);
    if ((attributes & MESH_SKIN) != 0)
    {
        uint j = a[at];
        uint w = a[at + 1];
        v.joints = uint4(j, j >> 8, j >> 16, j >> 24) & 0xFF;
        v.weights = unpack_unorm4x8(w);
    }
    return v;
}

// Cluster-local corners of triangle `t`; triangles from `alpha_first` on are alpha-tested.
public uint3 load_triangle(uint8_t* page, PageHeader h, Cluster c, uint t)
{
    uint8_t* p = page + h.tris_offset + c.tri_first + t * 3;
    return uint3(p[0], p[1], p[2]);
}

// Micromap index of alpha-tested triangle `t` (at least `alpha_first`) of a cluster whose
// `omm_first` is not CLUSTER_NO_OMM: an entry of the page's micromap, or negative for a
// special index.
public int load_omm_index(uint8_t* page, PageHeader h, Cluster c, uint t)
{
    int16_t* p = (int16_t*)(page + h.omm_offset + c.omm_first);
    return int(p[t - c.alpha_first]);
}

// Writes the cluster-local corners of triangle `t`. Pointing two corners at one vertex removes
// the triangle from raster and ray tracing.
public void store_triangle(uint8_t* page, PageHeader h, Cluster c, uint t, uint3 corners)
{
    uint8_t* p = page + h.tris_offset + c.tri_first + t * 3;
    p[0] = uint8_t(corners.x);
    p[1] = uint8_t(corners.y);
    p[2] = uint8_t(corners.z);
}

// Writes a vertex's grid offsets from the cluster's anchor.
public void store_local(uint8_t* page, PageHeader h, Cluster c, uint i, uint3 local)
{
    uint16_t* p = (uint16_t*)(page + h.positions_offset) + (c.vertex_first + i) * 3;
    p[0] = uint16_t(local.x);
    p[1] = uint16_t(local.y);
    p[2] = uint16_t(local.z);
}

// Moves cluster `cluster`'s anchor, its vertices staying where they are. A kernel lowers it to
// make room before moving a vertex below it. An edit's rebuild puts it back at the lowest vertex.
public void store_anchor(uint8_t* page, PageHeader h, uint cluster, int3 anchor)
{
    Cluster* at = cluster_at(page, cluster);
    Cluster c = *at;
    int3 shift = c.anchor - anchor;
    for (uint i = 0; i < c.vertex_count; i++)
        store_local(page, h, c, i, uint3(clamp(int3(load_local(page, h, c, i)) + shift, int3(0), int3(65535))));
    at.anchor = anchor;
}

// Moves a vertex to the grid point nearest mesh-space `position`, within the 65,536 grid steps
// above the cluster's anchor (see `store_anchor`). A move farther than that stops at their edge.
public void store_position(uint8_t* page, PageHeader h, Cluster c, uint i, float3 position, float step)
{
    int3 at = int3(round(position / step)) - c.anchor;
    store_local(page, h, c, i, uint3(clamp(at, int3(0), int3(65535))));
}
