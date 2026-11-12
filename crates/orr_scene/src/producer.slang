// Producer entries and the batches their kernels emit clusters or strand segments into, the
// per-entry state the frame keeps of their builds, and the cluster and segment reads the
// raster and hit resolution share.
module producer;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
__exported import orr_scene.tessellation;

// `ProducerClusterGpu.topology` of a directly built cluster.
public static const uint DIRECT_CLUSTER = 0xffff;
// Returned when a batch has no room left; the emit is dropped.
public static const uint NO_ROOM = 0xffffffff;

// `ProducerGpu.opacity`.
public static const uint PRODUCER_OPAQUE = 0;
public static const uint PRODUCER_ALPHA_TESTED = 1;
public static const uint PRODUCER_DIELECTRIC = 2;
public static const uint PRODUCER_PRESENCE = 3;
public static const uint PRODUCER_PARTICLE = 4;
public static const uint PRODUCER_SMOKE = 5;
public static const uint PRODUCER_SECTIONS = 6; // each cluster takes the class of its material

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
    public uint8_t _pad;
    public uint16_t material;    // index in the entry's materials
    public uint16_t topology;    // or DIRECT_CLUSTER
}

// A producer entry's batch. Counts reset to 0 before scene compute each frame for
// `EveryFrame` entries, and once at `set_producer` for `Static` ones; they keep counting
// past the maximum, and what lies past it is dropped.
public struct ProducerBatch
{
    public ProducerClusterGpu* clusters;
    public float3* positions;    // relative to the entry's transform
    public float3* previous;     // each vertex's position last frame; null unless ProducerDesc::previous_positions
    public uint3* attrs;         // normal oct 2x16 unorm, then uv 2x f32 (uv set 1 too where `uv1` is null)
    public float2* uv1;          // uv set 1; null but on a subdivision copy that carries it
    public uint2* colors;        // the vertex color stream, 4x f16; null where it reads 1: only a colored subdivision copy's generated vertices hold it
    public uint* vertex_user;    // null unless ProducerDesc::vertex_user
    public uint* cluster_count;
    public uint* vertex_count;
    public TopologyGpu* topologies;
    public GeneratedClusterGpu* generated; // per cluster of a tessellated entry's batch; null unless ProducerDesc::generated_records
    public uint* segments;       // per segment: its first vertex, from which it runs to the next; null outside strand entries
    public uint* segment_user;   // per segment: the producer's word
    public float* radii;         // per vertex of a strand entry: the strand's radius, m, relative to the entry's transform
    public uint* segment_count;
    public uint max_clusters;
    public uint max_vertices;
    public uint topology_count;
    public uint max_segments;
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
    public uint class_id;        // orr_shaders.trace CLASS_*, 0 for PRODUCER_SECTIONS
    public uint origin;          // gbuffer ORIGIN_*
    public uint displacement;    // linked IDisplacement id; 0: the material's texture
    public uint tessellated;     // 1: clusters are base patches for the tessellator
    public uint strands;         // 1: a strand entry, which emits segments instead of clusters
    public uint drawn;           // linked IDrawn id of a drawn entry, whose opaque raster the view's opaque hook draws; 0: its clusters raster
}

// `ProducerStateGpu.rt_flags`.
public static const uint PRODUCER_RT_WRITTEN = 1;  // its PTLAS instance holds a BLAS
public static const uint PRODUCER_RT_BUILT = 2;    // built this frame: its instance is rewritten

// What the GPU keeps of a set entry's latest build and of its PTLAS instance.
public struct ProducerStateGpu
{
    public uint3 lo;             // bounds of its built clusters, object space, as `ordered` floats
    public uint clusters;        // CLAS in its BLAS; segments in a strand entry's swept-sphere BLAS
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

// The attribute words of a vertex with unit `normal` and `uv`, for `ProducerBatch.attrs`. Uvs
// stay f32: vertices come from many threads with no range known to quantize over, and the
// tessellator displaces each vertex at the f32 uv it stores.
public uint3 producer_attributes(float3 normal, float2 uv)
{
    return uint3(oct_encode_unorm16(normal), asuint(uv));
}

public float3 producer_normal(uint3 attrs)
{
    return oct_decode_unorm16(attrs.x);
}

public float2 producer_uv(uint3 attrs)
{
    return asfloat(attrs.yz);
}

// Uv set 1 of vertex `v`, whose uv set 0 is `uv`.
public float2 producer_uv1(ProducerBatch b, uint v, float2 uv)
{
    return b.uv1 != nullptr ? b.uv1[v] : uv;
}

// The word of a vertex with vertex color stream `color`, for `ProducerBatch.colors`.
public uint2 producer_color_word(float4 color)
{
    return uint2(pack_half2x16(color.xy), pack_half2x16(color.zw));
}

public float4 producer_color(uint2 word)
{
    return float4(unpack_half2x16(word.x), unpack_half2x16(word.y));
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
    c.material = uint16_t(material);
    c.topology = uint16_t(topology);
    c._pad = 0;
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
    c.material = uint16_t(material);
    c.topology = uint16_t(DIRECT_CLUSTER);
    c._pad = 0;
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

// Segments per raster cluster of a strand entry: raster cluster g draws segments
// 64 g to 64 g + 63 as two ribbon triangles each.
public static const uint STRAND_GROUP = 64;

// Where strands trace as triangles, each segment is a closed tube: TUBE_SIDES quads between
// two rings and a flat cap on each ring. Caps at joints lie inside the strand. Each tube
// cluster holds TUBE_SEGMENTS consecutive segments.
public static const uint TUBE_SIDES = 6;
public static const uint TUBE_SEGMENTS = 6;
public static const uint TUBE_SEGMENT_VERTICES = 2 * TUBE_SIDES;
public static const uint TUBE_SEGMENT_TRIANGLES = 2 * TUBE_SIDES + 2 * (TUBE_SIDES - 2);
public static const uint TUBE_VERTICES = TUBE_SEGMENTS * TUBE_SEGMENT_VERTICES;

// `count` segments of one strand over vertices `first_vertex` to `first_vertex + count`, each
// from one vertex to the next, all with the word `user`: the first segment's index, or
// NO_ROOM when not all fit. Segments past the batch's maximum are dropped. A strand's
// vertices carry `positions`, `radii` and `attrs` (the uv along the strand; the normal is
// unused).
public uint emit_segments(ProducerBatch b, uint first_vertex, uint count, uint user)
{
    uint first;
    InterlockedAdd(*b.segment_count, count, first);
    for (uint j = 0; j < count && first + j < b.max_segments; j++)
    {
        b.segments[first + j] = first_vertex + j;
        b.segment_user[first + j] = user;
    }
    return first + count <= b.max_segments ? first : NO_ROOM;
}

// The segments of the batch that count: those below both the count and the maximum.
public uint emitted_segments(ProducerBatch b)
{
    return b.segments == nullptr ? 0 : min(*b.segment_count, b.max_segments);
}

// One strand segment, relative to the entry's transform.
public struct Segment
{
    public float3 a;
    public float3 b;
    public float ra;
    public float rb;
    public uint vertex;          // batch vertex of `a`; `b` is the next one
}

// Segment `s` of `b`, at last frame's positions with `previous` where the batch keeps them.
// False when it is neither drawn nor traced: past the emitted count, a vertex outside the
// batch, a position or radius that is not finite, or a negative radius.
public bool load_segment(ProducerBatch b, uint s, bool previous, out Segment g)
{
    g = (Segment)0;
    if (s >= emitted_segments(b))
        return false;
    uint v = b.segments[s];
    uint vertices = min(*b.vertex_count, b.max_vertices);
    if (v >= vertices || v + 1 >= vertices)
        return false;
    float3* positions = previous && b.previous != nullptr ? b.previous : b.positions;
    g.a = positions[v];
    g.b = positions[v + 1];
    g.ra = b.radii[v];
    g.rb = b.radii[v + 1];
    g.vertex = v;
    return all(isfinite(g.a)) && all(isfinite(g.b)) && isfinite(g.ra) && isfinite(g.rb) && g.ra >= 0.0 && g.rb >= 0.0;
}

// The entry-relative box of segment `g` with its radii.
public void segment_box(Segment g, out float3 lo, out float3 hi)
{
    lo = min(g.a - g.ra, g.b - g.rb);
    hi = max(g.a + g.ra, g.b + g.rb);
}

// The entry-relative box of the valid segments of strand raster cluster `c`; false when it
// has none.
public bool strand_cluster_box(ProducerBatch b, uint c, out float3 lo, out float3 hi)
{
    lo = float3(1e30);
    hi = float3(-1e30);
    bool any = false;
    for (uint j = 0; j < STRAND_GROUP; j++)
    {
        Segment g;
        if (!load_segment(b, c * STRAND_GROUP + j, false, g))
            continue;
        float3 l, h;
        segment_box(g, l, h);
        lo = min(lo, l);
        hi = max(hi, h);
        any = true;
    }
    return any;
}

// The unit normal of the sphere swept from `a` with radius `ra` to `b` with radius `rb` at
// its point nearest `p`, and in `u` where along the segment (0 at `a`, 1 at `b`) the sphere
// touching that point is centered.
public float3 swept_normal(float3 a, float3 b, float ra, float rb, float3 p, out float u)
{
    float3 ba = b - a;
    float len = length(ba);
    u = 0.0;
    if (!(len > 0.0))
        return safe_normalize(p - a, float3(0, 0, 1));
    float3 e = ba / len;
    float x = dot(p - a, e);
    float3 across = p - a - x * e;
    float h = length(across);
    // The touching sphere's center lies past the foot of `p` by k h / sqrt(1 - k^2), with k
    // the slope of the radius along the axis.
    float k = clamp((rb - ra) / len, -0.999, 0.999);
    float c = clamp(x + k * h / sqrt(1.0 - k * k), 0.0, len);
    u = c / len;
    float3 side = h > 0.0 ? across / h : perpendicular(e);
    return safe_normalize(p - (a + c * e), side);
}

// The first hit with `t > 0` along the unit ray `o + t d` on the sphere swept from `a` with
// radius `ra` to `b` with radius `rb`, end spheres included: `t` and the unit normal there.
// False on a miss.
public bool swept_intersect(float3 o, float3 d, float3 a, float3 b, float ra, float rb, out float t, out float3 n)
{
    t = 0.0;
    n = float3(0.0);
    float3 ba = b - a;
    float3 oa = o - a;
    float3 ob = o - b;
    float rr = ra - rb;
    float m0 = dot(ba, ba);
    float m1 = dot(ba, oa);
    float m2 = dot(ba, d);
    float m3 = dot(d, oa);
    float m5 = dot(oa, oa);
    float m6 = dot(ob, d);
    float m7 = dot(ob, ob);
    // The cone between the end spheres' circles of contact.
    float d2 = m0 - rr * rr;
    if (d2 > 0.0)
    {
        float k2 = d2 - m2 * m2;
        float k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
        float k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0 - m0 * ra * ra;
        float h = k1 * k1 - k0 * k2;
        if (h >= 0.0 && k2 != 0.0)
        {
            float tc = (-sqrt(h) - k1) / k2;
            float y = m1 - ra * rr + tc * m2;
            if (tc > 0.0 && y > 0.0 && y < d2)
            {
                t = tc;
                n = normalize(d2 * (oa + tc * d) - ba * y);
                return true;
            }
        }
    }
    float best = 1e30;
    float h1 = m3 * m3 - m5 + ra * ra;
    if (h1 >= 0.0 && ra > 0.0)
    {
        float ts = -m3 - sqrt(h1);
        if (ts > 0.0)
        {
            best = ts;
            n = (oa + ts * d) / ra;
        }
    }
    float h2 = m6 * m6 - m7 + rb * rb;
    if (h2 >= 0.0 && rb > 0.0)
    {
        float ts = -m6 - sqrt(h2);
        if (ts > 0.0 && ts < best)
        {
            best = ts;
            n = (ob + ts * d) / rb;
        }
    }
    if (best == 1e30)
        return false;
    t = best;
    n = safe_normalize(n, -d);
    return true;
}

// The distance along the unit ray `o + t d` to its point nearest the segment from `a` to `b`.
public float ray_axis_closest(float3 o, float3 d, float3 a, float3 b)
{
    float3 ba = b - a;
    float3 w = o - a;
    float bb = dot(ba, ba);
    float db = dot(d, ba);
    float denom = bb - db * db;
    float u = denom > 1e-30 ? clamp((dot(w, ba) - db * dot(w, d)) / denom, 0.0, 1.0) : 0.0;
    return max(dot(a + u * ba - o, d), 0.0);
}
