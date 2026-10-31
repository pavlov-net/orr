// The acceleration structures of producer entries, built on the frame's list after scene
// compute. `templates` writes an entry's template builds the frame it is set; `begin`
// clears what an entry's build counts; `records` checks each emitted cluster, writes its
// box and the entry's, and writes an instantiation of its topology's template or a direct
// CLAS build; `gather` lists the built CLAS and writes the entry's cluster BLAS record;
// `finish` keeps the BLAS in the entry's state. One root per entry; every kernel runs one
// workgroup per entry, so work follows the emitted counts.
#include "orr/root.h"
import orr.abi;
import producer;
import rtas;
import tables;

// `ProducerBuildGpu.counts`.
public static const uint BUILD_TEMPLATES = 0;
public static const uint BUILD_INSTANTIATED = 1;
public static const uint BUILD_DIRECT = 2;
public static const uint BUILD_BLAS = 3;
public static const uint BUILD_CLAS = 4;
public static const uint BUILD_COUNTS = 8;

// `ProducerBuildGpu.refs`: a cluster's build record, DIRECT_REF for the direct array.
public static const uint REF_NONE = 0xffffffff;
public static const uint DIRECT_REF = 0x80000000;

// An entry's build memory. Results of the implicit builds land in blocks the CPU sized
// for the entry's capacities.
public struct ProducerBuildGpu
{
    public uint8_t* template_records;
    public uint8_t* instantiate_records;
    public uint8_t* direct_records;
    public uint8_t* blas_record;
    public uint64_t* templates;     // per topology: its template
    public uint64_t* instantiated;  // per instantiation record: its CLAS
    public uint64_t* direct;        // per direct record: its CLAS
    public uint64_t* clas;          // the CLAS the entry's BLAS holds
    public uint64_t* blas;          // the BLAS record's result
    public uint* refs;              // per cluster
    public uint* counts;            // BUILD_*
    public float3* curve;           // template positions: vertex i on the moment curve
    public uint topology_count;
    public uint geometry_flags;     // rtas GEOMETRY_* of every triangle
}

public struct BuildRoot
{
    public SceneRoot* scene;
    public ProducerBuildGpu* build;
    public uint entry;
    public uint _pad0;
    public uint64_t _pad1;
}
ORR_ROOT(BuildRoot);

static const uint GROUP = 128;

// One thread per topology.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void templates(uint t: SV_GroupThreadID)
{
    ProducerBuildGpu* b = root.build;
    ProducerBatch batch = root.scene.batches[root.entry];
    if (t == 0)
        b.counts[BUILD_TEMPLATES] = b.topology_count;
    for (uint i = t; i < b.topology_count; i += GROUP)
    {
        TopologyGpu topo = batch.topologies[i];
        TemplateBuild r = (TemplateBuild)0;
        r.cluster.cluster_id = 0;
        r.cluster.tri_count = topo.tri_count;
        r.cluster.vertex_count = topo.vertex_count;
        r.cluster.index_format = 1;
        r.cluster.base_geometry_and_flags = geometry_index_and_flags(0, b.geometry_flags);
        r.cluster.omm_index_format = 4;
        r.cluster.vertices = (uint64_t)b.curve;
        r.cluster.indices = (uint64_t)topo.triangles;
        store((uint64_t)b.template_records, i, r);
    }
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void begin(uint t: SV_GroupThreadID)
{
    ProducerBuildGpu* b = root.build;
    if (t >= BUILD_COUNTS)
        return;
    if (t != BUILD_TEMPLATES)
        b.counts[t] = 0;
    if (t == 0)
    {
        ProducerStateGpu* st = root.scene.producer_state + root.entry;
        st.lo = uint3(0xffffffff);
        st.hi = uint3(0);
        st.clusters = 0;
        st.refused = 0;
    }
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void records(uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ProducerBuildGpu* b = root.build;
    ProducerBatch batch = s.batches[root.entry];
    ProducerStateGpu* st = s.producer_state + root.entry;
    uint n = emitted_clusters(batch);
    for (uint c = t; c < n; c += GROUP)
    {
        ProducerClusterGpu k = batch.clusters[c];
        if (!producer_cluster_valid(batch, k))
        {
            b.refs[c] = REF_NONE;
            InterlockedAdd(st.refused, 1);
            continue;
        }
        float3 lo, hi;
        producer_cluster_box(batch, k, lo, hi);
        InterlockedMin(st.lo.x, ordered(lo.x));
        InterlockedMin(st.lo.y, ordered(lo.y));
        InterlockedMin(st.lo.z, ordered(lo.z));
        InterlockedMax(st.hi.x, ordered(hi.x));
        InterlockedMax(st.hi.y, ordered(hi.y));
        InterlockedMax(st.hi.z, ordered(hi.z));
        uint64_t vertices = (uint64_t)(batch.positions + k.first_vertex);
        uint index;
        if (k.topology == DIRECT_CLUSTER)
        {
            InterlockedAdd(b.counts[BUILD_DIRECT], 1, index);
            ClasBuild r = (ClasBuild)0;
            r.cluster_id = c;
            r.tri_count = k.tri_count;
            r.vertex_count = k.vertex_count;
            r.index_format = 1;
            r.base_geometry_and_flags = geometry_index_and_flags(0, b.geometry_flags);
            r.omm_index_format = 4;
            r.vertices = vertices;
            r.indices = (uint64_t)k.triangles;
            store((uint64_t)b.direct_records, index, r);
            b.refs[c] = index | DIRECT_REF;
        }
        else
        {
            InterlockedAdd(b.counts[BUILD_INSTANTIATED], 1, index);
            Instantiate r;
            r.cluster_id_offset = c;
            r.geometry_index_offset = 0;
            r.template_address = b.templates[k.topology];
            r.vertices = vertices;
            store((uint64_t)b.instantiate_records, index, r);
            b.refs[c] = index;
        }
    }
}

groupshared uint built;

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void gather(uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ProducerBuildGpu* b = root.build;
    ProducerBatch batch = s.batches[root.entry];
    uint n = emitted_clusters(batch);
    for (uint c = t; c < n; c += GROUP)
    {
        uint r = b.refs[c];
        if (r == REF_NONE)
            continue;
        uint64_t clas = (r & DIRECT_REF) != 0 ? b.direct[r & ~DIRECT_REF] : b.instantiated[r];
        if (clas == 0)
            continue;
        uint slot;
        InterlockedAdd(b.counts[BUILD_CLAS], 1, slot);
        b.clas[slot] = clas;
    }
    AllMemoryBarrierWithGroupSync();
    if (t != 0)
        return;
    uint count = b.counts[BUILD_CLAS];
    s.producer_state[root.entry].clusters = count;
    if (count == 0)
        return;
    ClusterBlas r;
    r.clas_addresses = (uint64_t)b.clas;
    r.clas_count = count;
    r.pad = 0;
    store((uint64_t)b.blas_record, 0, r);
    b.counts[BUILD_BLAS] = 1;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void finish(uint t: SV_GroupThreadID)
{
    ProducerBuildGpu* b = root.build;
    ProducerStateGpu* st = root.scene.producer_state + root.entry;
    st.blas = b.counts[BUILD_BLAS] != 0 ? b.blas[0] : 0;
    st.rt_flags |= PRODUCER_RT_BUILT;
}
