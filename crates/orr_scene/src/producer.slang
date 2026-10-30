// Producer entries and the batches their kernels emit clusters into.
module producer;

import orr.abi;

// `ProducerClusterGpu.topology` of a directly built cluster.
public static const uint DIRECT_CLUSTER = 0xff;
// Returned when a batch has no room left; the emit is dropped.
public static const uint NO_ROOM = 0xffffffff;

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
    public float3* previous;     // null unless ProducerDesc::previous_positions
    public uint2* attrs;         // normal oct 2x16 unorm, uv 2x f16
    public uint* vertex_user;    // null unless ProducerDesc::vertex_user
    public uint* cluster_count;
    public uint* vertex_count;
    public TopologyGpu* topologies;
    public uint max_clusters;
    public uint max_vertices;
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
    public uint _pad;
}

// `count` vertices from the batch: the first one's index, or NO_ROOM.
public uint alloc_vertices(ProducerBatch b, uint count)
{
    uint first;
    InterlockedAdd(*b.vertex_count, count, first);
    return first + count <= b.max_vertices ? first : NO_ROOM;
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
    TopologyGpu t = b.topologies[topology];
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
