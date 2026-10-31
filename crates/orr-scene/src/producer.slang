// Producer entries and the batches their kernels emit clusters into, the per-entry state the
// frame keeps of their builds, and the cluster reads the raster and hit resolution share.
module producer;

import orr.abi;
import orr.packing;

// `ProducerClusterGpu.topology` of a directly built cluster.
public static const uint DIRECT_CLUSTER = 0xff;
// Returned when a batch has no room left; the emit is dropped.
public static const uint NO_ROOM = 0xffffffff;

// `ProducerGpu.opacity`.
public static const uint PRODUCER_OPAQUE = 0;
public static const uint PRODUCER_ALPHA_TESTED = 1;
public static const uint PRODUCER_DIELECTRIC = 2;
public static const uint PRODUCER_PRESENCE = 3;
public static const uint PRODUCER_PARTICLE = 4;

// One cluster topology of an entry: triangles as u8x3 cluster-local corners.
public struct TopologyGpu
{
    public uint8_t* triangles;
    public uint tri_count;
    public uint vertex_count;
}

// One emitted cluster. Vertex i is batch vertex `first_vertex + i`.
public struct ProducerClusterGpu
{
    public uint8_t* triangles;   // u8x3 per triangle: the topology's, or the producer's own
    public uint first_vertex;
    public uint user;            // the producer's word for this cluster
    public uint16_t vertex_count;
    public uint8_t tri_count;
    public uint8_t material;     // index in the entry's materials
    public uint8_t topology;     // or DIRECT_CLUSTER
    public uint8_t _pad0;
    public uint16_t _pad1;
}

// A producer entry's batch. Counts reset to 0 before scene compute each frame for
// `EveryFrame` entries, and once at `set_producer` for `Static` ones; they keep counting
// past the maximum, and what lies past it is dropped.
public struct ProducerBatch
{
    public ProducerClusterGpu* clusters;
    public float3* positions;    // relative to the entry's transform
    public float3* previous;     // each vertex's position last frame; null unless ProducerDesc::previous_positions
    public uint2* attrs;         // normal oct 2x16 unorm, uv 2x f16
    public uint* vertex_user;    // null unless ProducerDesc::vertex_user
    public uint* cluster_count;
    public uint* vertex_count;
    public TopologyGpu* topologies;
    public uint max_clusters;
    public uint max_vertices;
    public uint topology_count;
    public uint _pad;
}

// A producer entry. An unset entry is all zero.
public struct ProducerGpu
{
    public uint* materials;      // MaterialId per cluster material index
    public uint material_count;
    public uint group;           // transform group and index
    public uint group_index;
    public uint flags;           // INSTANCE_*
    public uint user_tag;
    public uint opacity;         // PRODUCER_*
    public uint class_id;        // orr.trace CLASS_*
    public uint origin;          // gbuffer ORIGIN_*
    public uint displacement;    // linked IDisplacement id; 0: the material's texture
    public uint tessellated;     // 1: clusters are base patches for the tessellator
}

// `ProducerStateGpu.rt_flags`.
public static const uint PRODUCER_RT_WRITTEN = 1;  // its PTLAS instance holds a BLAS
public static const uint PRODUCER_RT_BUILT = 2;    // built this frame: its instance is rewritten

// What the GPU keeps of a set entry's latest build and of its PTLAS instance.
public struct ProducerStateGpu
{
    public uint3 lo;             // bounds of its built clusters, object space, as `ordered` floats
    public uint clusters;        // CLAS in its BLAS
    public uint3 hi;
    public uint rt_flags;        // PRODUCER_RT_*
    public uint64_t blas;        // 0: none
    public uint2 rt_key;         // hash of what its PTLAS instance was written with
    public uint refused;         // emitted clusters its latest build refused
    public uint _pad;
}

// Floats as uints that order like the floats, for atomic min and max.
public uint ordered(float f)
{
    uint u = asuint(f);
    return (u & 0x80000000u) != 0 ? ~u : u | 0x80000000u;
}

public float unordered(uint u)
{
    return asfloat((u & 0x80000000u) != 0 ? u & 0x7fffffffu : ~u);
}

// The entry's bounds from its state; false while it has built no cluster.
public bool producer_bounds(ProducerStateGpu st, out float3 lo, out float3 hi)
{
    lo = float3(unordered(st.lo.x), unordered(st.lo.y), unordered(st.lo.z));
    hi = float3(unordered(st.hi.x), unordered(st.hi.y), unordered(st.hi.z));
    return st.clusters > 0 && all(lo <= hi);
}

// `count` vertices from the batch: the first one's index, or NO_ROOM.
public uint alloc_vertices(ProducerBatch b, uint count)
{
    uint first;
    InterlockedAdd(*b.vertex_count, count, first);
    return first + count <= b.max_vertices ? first : NO_ROOM;
}

// The attribute word of a vertex with unit `normal` and `uv`, for `ProducerBatch.attrs`.
public uint2 producer_attributes(float3 normal, float2 uv)
{
    return uint2(oct_encode_unorm16(normal), pack_half2x16(uv));
}

public float3 producer_normal(uint2 attrs)
{
    return oct_decode_unorm16(attrs.x);
}

public float2 producer_uv(uint2 attrs)
{
    return unpack_half2x16(attrs.y);
}

uint emit(ProducerBatch b, ProducerClusterGpu c)
{
    uint index;
    InterlockedAdd(*b.cluster_count, 1, index);
    if (index >= b.max_clusters)
        return NO_ROOM;
    b.clusters[index] = c;
    return index;
}

// A cluster of `topology` over vertices from `first_vertex`: its index, or NO_ROOM.
public uint emit_template_cluster(ProducerBatch b, uint topology, uint first_vertex,
                                 uint material, uint user)
{
    TopologyGpu t = (TopologyGpu)0;
    if (topology < b.topology_count)
        t = b.topologies[topology];
    ProducerClusterGpu c;
    c.triangles = t.triangles;
    c.first_vertex = first_vertex;
    c.user = user;
    c.tri_count = uint8_t(t.tri_count);
    c.vertex_count = uint16_t(t.vertex_count);
    c.material = uint8_t(material);
    c.topology = uint8_t(topology);
    c._pad0 = 0;
    c._pad1 = 0;
    return emit(b, c);
}

// A cluster with its own triangles (u8x3 per triangle, at most 128), which stay valid
// until the frame completes: its index, or NO_ROOM.
public uint emit_direct_cluster(ProducerBatch b, uint tri_count, uint vertex_count,
                               uint first_vertex, uint8_t* triangles, uint material, uint user)
{
    ProducerClusterGpu c;
    c.triangles = triangles;
    c.first_vertex = first_vertex;
    c.user = user;
    c.tri_count = uint8_t(tri_count);
    c.vertex_count = uint16_t(vertex_count);
    c.material = uint8_t(material);
    c.topology = uint8_t(DIRECT_CLUSTER);
    c._pad0 = 0;
    c._pad1 = 0;
    return emit(b, c);
}

// The clusters of the batch that count: those below both the count and the maximum.
public uint emitted_clusters(ProducerBatch b)
{
    return min(*b.cluster_count, b.max_clusters);
}

// Whether cluster `k` of `b` can draw and build: 1 to 128 triangles, 1 to 256 vertices, all
// inside the batch, and triangles to read. Clusters that fail are neither drawn nor traced.
public bool producer_cluster_valid(ProducerBatch b, ProducerClusterGpu k)
{
    uint vertices = min(*b.vertex_count, b.max_vertices);
    return k.tri_count > 0 && k.tri_count <= 128 && k.vertex_count > 0 && k.vertex_count <= 256
        && uint(k.first_vertex) + uint(k.vertex_count) <= vertices && k.triangles != nullptr;
}

// The entry-relative box of cluster `k`'s vertices.
public void producer_cluster_box(ProducerBatch b, ProducerClusterGpu k, out float3 lo, out float3 hi)
{
    lo = float3(1e30);
    hi = float3(-1e30);
    for (uint i = 0; i < k.vertex_count; i++)
    {
        float3 x = b.positions[k.first_vertex + i];
        lo = min(lo, x);
        hi = max(hi, x);
    }
}

// The cluster-local corners of triangle `i` of `k`.
public uint3 cluster_corners(ProducerClusterGpu k, uint i)
{
    uint8_t* tri = k.triangles + 3 * i;
    return uint3(tri[0], tri[1], tri[2]);
}

// Corners of triangle `i` of cluster `k` of batch `b`, relative to the entry's transform.
public void cluster_triangle(ProducerBatch b, ProducerClusterGpu k, uint i, out float3 p0,
                             out float3 p1, out float3 p2)
{
    uint3 c = cluster_corners(k, i);
    p0 = b.positions[k.first_vertex + c.x];
    p1 = b.positions[k.first_vertex + c.y];
    p2 = b.positions[k.first_vertex + c.z];
}
