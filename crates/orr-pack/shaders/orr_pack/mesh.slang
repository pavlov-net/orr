// Resident mesh pages (orr-pack mesh pages, the layout `orr_pack::mesh::decode_page` writes).
// The structs match orr_pack::mesh's #[repr(C)] layouts; the loaders follow
// orr_pack::mesh::PageView. Every function takes the page's address; the module holds no
// bindings.
module mesh;

import orr.packing;

// `MeshHeader.attributes`.
public static const uint MESH_TANGENT = 1;
public static const uint MESH_SKIN = 2;

// `Cluster.kind`.
public static const uint CLUSTER_TRIANGLES = 0;

// `Cluster.flags`.
public static const uint CLUSTER_ANIMATED_PART = 1;
public static const uint CLUSTER_MERGED = 2;
public static const uint CLUSTER_EMISSIVE = 4;

// `Cluster.omm_first` of a cluster without micromaps.
public static const uint CLUSTER_NO_OMM = 0xffffffff;

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

// 128 B. Spheres are (centre, radius) in mesh space.
public struct Cluster
{
    public float4 self_sphere;
    public float self_error;
    public float4 parent_sphere;
    public float parent_error;
    public float4 cull_sphere;
    public uint cone;              // snorm8 axis xyz, cutoff w
    public int3 anchor;            // grid steps
    public uint vertex_first;
    public uint tri_first;         // bytes from tris_offset
    public uint16_t vertex_count;
    public uint8_t tri_count;
    public uint8_t alpha_first;
    public uint8_t section;
    public uint8_t part;
    public uint8_t kind;
    public uint8_t flags;
    public uint16_t local_max[3];
    public half deformation_bound; // m
    public half displacement_min;  // m
    public half displacement_max;  // m
    public float2 uv_offset;
    public float2 uv_scale;
    public uint omm_first;         // bytes from omm_offset
    public float area;             // m^2
    public half min_thickness;     // m
    public uint16_t _pad;
}

public struct VertexAttributes
{
    public float3 normal;
    public float4 tangent;         // xyz and bitangent sign; (1, 0, 0, 1) without MESH_TANGENT
    public float2 uv;
    public uint4 joints;           // MESH_SKIN only
    public float4 weights;         // MESH_SKIN only; sum to 1
}

public PageHeader load_page_header(uint8_t* page)
{
    return *(PageHeader*)page;
}

public Cluster load_cluster(uint8_t* page, uint cluster)
{
    return ((Cluster*)(page + 96))[cluster];
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
    uint q = a[at++];
    v.uv = c.uv_offset + float2(float(q & 0xFFFF), float(q >> 16)) / 65535.0 * c.uv_scale;
    v.joints = uint4(0);
    v.weights = float4(0.0);
    if ((attributes & MESH_SKIN) != 0)
    {
        uint j = a[at];
        uint w = a[at + 1];
        v.joints = uint4(j, j >> 8, j >> 16, j >> 24) & 0xFF;
        v.weights = float4(uint4(w, w >> 8, w >> 16, w >> 24) & 0xFF) / 255.0;
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
