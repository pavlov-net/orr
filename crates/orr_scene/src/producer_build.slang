// The acceleration structures of producer entries, built on the frame's list after scene
// compute. `templates` writes an entry's template builds the frame it is set; `begin`
// clears what an entry's build counts; `strands` checks a strand entry's segments, bounds
// them and writes either the index list of its swept-sphere BLAS or its tube batch;
// `records` checks each cluster of the batch the entry builds from, writes its box and the
// entry's, and writes an instantiation of its topology's template or a direct CLAS build;
// `gather` lists the built CLAS and writes the entry's cluster BLAS record; `finish` keeps
// the BLAS in the entry's state. One root per entry; every kernel runs one workgroup per
// entry, so work follows the emitted counts.
#include "orr/root.h"
import orr.abi;
import orr.math;
import producer;
import rtas;
import tables;

// `ProducerBuildGpu.counts`.
public static const uint BUILD_TEMPLATES = 0;
public static const uint BUILD_INSTANTIATED = 1;
public static const uint BUILD_DIRECT = 2;
public static const uint BUILD_BLAS = 3;
public static const uint BUILD_CLAS = 4;
public static const uint BUILD_TUBE_CLUSTERS = 5;  // the tube batch's cluster count
public static const uint BUILD_TUBE_VERTICES = 6;  // the tube batch's vertex count
public static const uint BUILD_COUNTS = 8;

// `ProducerBuildGpu.refs`: a cluster's build record, DIRECT_REF for the direct array.
public static const uint REF_NONE = 0xffffffff;
public static const uint DIRECT_REF = 0x80000000;

// `ProducerBuildGpu.strands`.
public static const uint STRANDS_NONE = 0;
public static const uint STRANDS_SWEPT = 1;   // segments build a swept-sphere BLAS
public static const uint STRANDS_TUBES = 2;   // segments build tube clusters

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
    public ProducerBatch* batch;    // what the cluster builds read: the entry's batch, or a strand entry's tube batch
    public uint* swept_indices;     // a swept-sphere entry's per-segment first vertex, the BLAS's index list
    public uint64_t swept_blas;     // a swept-sphere entry's BLAS
    public uint topology_count;
    public uint geometry_flags;     // rtas GEOMETRY_* of every triangle
    public uint strands;            // STRANDS_*
    public uint _pad;
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
    ProducerBatch batch = *b.batch;
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

void grow_bounds(ProducerStateGpu* st, float3 lo, float3 hi)
{
    InterlockedMin(st.lo.x, ordered(lo.x));
    InterlockedMin(st.lo.y, ordered(lo.y));
    InterlockedMin(st.lo.z, ordered(lo.z));
    InterlockedMax(st.hi.x, ordered(hi.x));
    InterlockedMax(st.hi.y, ordered(hi.y));
    InterlockedMax(st.hi.z, ordered(hi.z));
}

// The ring of tube vertices around `center` at right angles to the unit `tangent`.
void tube_ring(float3* ring, float3 center, float3 tangent, float radius)
{
    float3 u, v;
    orthonormal_basis(tangent, u, v);
    // The polygon's area equals the circle's.
    float sides = float(TUBE_SIDES);
    float r = radius * sqrt(2.0 * PI / (sides * sin(2.0 * PI / sides)));
    for (uint k = 0; k < TUBE_SIDES; k++)
    {
        float a = 2.0 * PI * float(k) / sides;
        ring[k] = center + r * (cos(a) * u + sin(a) * v);
    }
}

// Whether segment `s` of `b` is valid and starts at `vertex`, and its direction.
bool starts_at(ProducerBatch b, uint s, uint vertex, out float3 dir)
{
    dir = float3(0.0);
    Segment g;
    if (!load_segment(b, s, false, g) || g.vertex != vertex)
        return false;
    dir = g.b - g.a;
    return true;
}

// One thread per segment. A swept-sphere entry lists each valid segment's first vertex for
// the BLAS and every other one as the two NaN vertices past the batch's maximum, which the
// build leaves inactive. A tube entry writes a tube per valid segment into the tube batch,
// its rings at joints at right angles to the strand's mean direction so neighboring tubes
// share them, and NaN vertices for the rest, whose triangles are inactive.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void strands(uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ProducerBuildGpu* b = root.build;
    ProducerBatch batch = s.batches[root.entry];
    ProducerStateGpu* st = s.producer_state + root.entry;
    uint n = emitted_segments(batch);
    float nan = asfloat(0x7fc00000u);
    uint tube_count = (n + TUBE_SEGMENTS - 1) / TUBE_SEGMENTS;
    uint listed = b.strands == STRANDS_SWEPT ? batch.max_segments : tube_count * TUBE_SEGMENTS;
    if (b.strands == STRANDS_SWEPT && t < 2)
    {
        batch.positions[batch.max_vertices + t] = float3(nan);
        batch.radii[batch.max_vertices + t] = nan;
    }
    if (b.strands == STRANDS_TUBES && t == 0)
    {
        b.counts[BUILD_TUBE_CLUSTERS] = tube_count;
        b.counts[BUILD_TUBE_VERTICES] = tube_count * TUBE_VERTICES;
    }
    ProducerBatch tubes = *b.batch;
    for (uint k = t; k < listed; k += GROUP)
    {
        Segment g;
        bool valid = load_segment(batch, k, false, g);
        if (!valid && k < n)
            InterlockedAdd(st.refused, 1);
        if (valid)
        {
            float3 lo, hi;
            segment_box(g, lo, hi);
            grow_bounds(st, lo, hi);
            InterlockedAdd(st.clusters, 1);
        }
        if (b.strands == STRANDS_SWEPT)
        {
            b.swept_indices[k] = valid ? g.vertex : batch.max_vertices;
            continue;
        }
        float3* ring = tubes.positions + k * TUBE_SEGMENT_VERTICES;
        if (k % TUBE_SEGMENTS == 0)
        {
            ProducerClusterGpu c = (ProducerClusterGpu)0;
            c.triangles = tubes.topologies[0].triangles;
            c.first_vertex = k * TUBE_SEGMENT_VERTICES;
            c.vertex_count = uint16_t(TUBE_VERTICES);
            c.tri_count = uint8_t(TUBE_SEGMENTS * TUBE_SEGMENT_TRIANGLES);
            tubes.clusters[k / TUBE_SEGMENTS] = c;
        }
        if (!valid)
        {
            for (uint i = 0; i < TUBE_SEGMENT_VERTICES; i++)
                ring[i] = float3(nan);
            continue;
        }
        float3 dir = normalize(g.b - g.a);
        float3 before, after;
        float3 t0 = k > 0 && starts_at(batch, k - 1, g.vertex - 1, before) ? normalize(before) + dir : dir;
        float3 t1 = starts_at(batch, k + 1, g.vertex + 1, after) ? dir + normalize(after) : dir;
        tube_ring(ring, g.a, safe_normalize(t0, dir), g.ra);
        tube_ring(ring + TUBE_SIDES, g.b, safe_normalize(t1, dir), g.rb);
    }
}

[shader("compute")]
[numthreads(GROUP, 1, 1)]
void records(uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ProducerBuildGpu* b = root.build;
    ProducerBatch batch = *b.batch;
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
        // A tube batch takes its bounds from its segments; invalid segments' tubes are NaN.
        if (b.strands == STRANDS_NONE)
        {
            float3 lo, hi;
            producer_cluster_box(batch, k, lo, hi);
            grow_bounds(st, lo, hi);
        }
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
    ProducerBatch batch = *b.batch;
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
    // A tube entry's state counts its segments.
    if (b.strands == STRANDS_NONE)
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
    if (b.strands == STRANDS_SWEPT)
        st.blas = st.clusters > 0 ? b.swept_blas : 0;
    else
        st.blas = b.counts[BUILD_BLAS] != 0 ? b.blas[0] : 0;
    st.rt_flags |= PRODUCER_RT_BUILT;
}
