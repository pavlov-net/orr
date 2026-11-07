// Inline ray tracing. This is the only module that names `RayQuery`: kernels trace through
// the functions below and never hold a query themselves.
//
// A hit is any committed status other than none. Linear swept sphere hits commit with a
// status that is neither triangle nor procedural, so `Hit.triangle` is false for them and
// their cluster id is -1. Their primitive index is the segment's.
module trace;

import orr_shaders.abi;

// TLAS instance mask bits.
public static const uint MASK_OPAQUE = 0x01;     // opaque and alpha-tested geometry
public static const uint MASK_PARTICLE = 0x02;   // particle batches
public static const uint MASK_MEDIUM = 0x04;     // medium proxies (AABBs)
public static const uint MASK_LIGHT = 0x08;      // sphere and rect light proxies (AABBs)
public static const uint MASK_DIELECTRIC = 0x80; // dielectric and presence surfaces

// Candidate classes. Cluster geometry holds its class in the geometry index, per cluster, under an
// instance with custom index 0 whose mask covers what it shows; other instances hold it in their
// custom index.
public static const uint CLASS_OPAQUE = 0; // opaque or alpha-tested
public static const uint CLASS_DIELECTRIC = 1;
public static const uint CLASS_PRESENCE = 2;
public static const uint CLASS_PARTICLE = 3;
public static const uint CLASS_MEDIUM = 4;
public static const uint CLASS_LIGHT = 5;

// The class of a candidate or hit from its instance's custom index and its geometry index.
public uint class_of(uint custom_index, uint geometry_index)
{
    return custom_index != 0 ? custom_index : geometry_index;
}

// The mask bit of candidate class `class_id`. Other classes pass every mask.
public uint class_mask(uint class_id)
{
    switch (class_id)
    {
    case CLASS_OPAQUE:
        return MASK_OPAQUE;
    case CLASS_DIELECTRIC:
    case CLASS_PRESENCE:
        return MASK_DIELECTRIC;
    case CLASS_PARTICLE:
        return MASK_PARTICLE;
    case CLASS_MEDIUM:
        return MASK_MEDIUM;
    case CLASS_LIGHT:
        return MASK_LIGHT;
    default:
        return 0xff;
    }
}

// Ray flags beside Slang's `RAY_FLAG_*`: opacity micromaps forced to two states. Builds
// without micromaps drop it.
public static const uint RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE = 0x400;

// A non-opaque candidate as the candidate handlers see it.
public struct Candidate
{
    public uint instance_index; // TLAS or PTLAS instance index
    public uint custom_index;   // the instance's custom index
    public uint geometry_index;
    public uint class_id;       // CLASS_*
    public uint primitive_index;
    public int cluster_id;      // -1 outside cluster BLAS
    public bool triangle;       // false for procedural (AABB) candidates
    public float t;             // triangle candidates only
    public float2 barycentrics; // triangle candidates only
    public bool front_face;     // triangle candidates only
    public float3 object_origin;
    public float3 object_direction;
    // Object-space corner positions where the handler asked for them (zero elsewhere). The
    // BLAS must allow data access.
    public float3 object_positions[3];
}

// The committed hit of a trace. On a miss the index fields are zero and `cluster_id` is -1.
public struct Hit
{
    public bool hit;
    public bool triangle;
    public float t;
    public uint instance_index;
    public uint custom_index;
    public uint geometry_index;
    public uint class_id;  // CLASS_*
    public uint primitive_index;
    public int cluster_id; // -1 outside cluster BLAS and on swept sphere hits
    public float2 barycentrics;
    public bool front_face;
}

// Decides non-opaque triangle candidates for closest and any-hit traces. Implementations
// are pure functions of the candidate that read only GPU data.
public interface IAnyHit
{
    // True commits the candidate as a hit.
    bool accept(Candidate c);
}

// Commits every non-opaque triangle candidate.
public struct AcceptAll : IAnyHit
{
    public bool accept(Candidate c)
    {
        return true;
    }
}

// Gives the transmittance of each non-opaque candidate for `trace_transmittance`. A
// handler may keep state across the candidates of one trace.
public interface ICandidate
{
    // Whether `transmittance` reads the candidate's `object_positions`.
    bool positions(Candidate c);
    // 0 blocks the ray, 1 lets it through unchanged. `counted` is false for candidates
    // that do not count toward the trace's `max_candidates`.
    [mutating]
    float3 transmittance(Candidate c, out bool counted);
}

// Decides candidates for `trace_resolve_translucent`: translucent layers resolve in depth
// order in front of the hit; other candidates commit or not through `accept`.
public interface ITranslucent : IAnyHit
{
    // A translucent layer: premultiplied radiance in rgb, coverage in a; coverage 0 with
    // radiance above 0 is a layer that only emits. A procedural candidate's layer sets its
    // distance in `c.t`. All zero means the candidate is not a layer: a triangle goes to
    // `accept`, and a procedural candidate is never committed.
    float4 layer(inout Candidate c);
}

// The closest hit, with non-opaque triangle candidates committed where `any_hit` accepts
// them. Procedural candidates are skipped.
[ForceInline]
public Hit trace_closest<A : IAnyHit>(Tlas tlas, RayDesc ray, uint mask, uint flags, A any_hit)
{
    float3 positions[3];
    return closest(tlas, ray, mask, flags, any_hit, false, positions);
}

// As above, and the object-space corner positions of a triangle hit, whose BLAS must allow
// data access. On any other hit the first position is the object-space hit point and the
// others are zero.
[ForceInline]
public Hit trace_closest<A : IAnyHit>(Tlas tlas, RayDesc ray, uint mask, uint flags, A any_hit, out float3 positions[3])
{
    return closest(tlas, ray, mask, flags, any_hit, true, positions);
}

// True when anything is hit; the first accepted hit ends the search.
[ForceInline]
public bool trace_any<A : IAnyHit>(Tlas tlas, RayDesc ray, uint mask, uint flags, A any_hit)
{
    RayQuery<RAY_FLAG_NONE> q;
    begin(q, tlas, ray, mask, flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH);
    while (q.Proceed())
    {
        if (opaque_candidate(q, flags))
            q.CommitNonOpaqueTriangleHit();
        else if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            Candidate c = candidate(q);
            if (takes(c, mask) && any_hit.accept(c))
                q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() != COMMITTED_NOTHING;
}

// The product of the transmittance of every non-opaque candidate along the ray, never
// committing one. An opaque hit gives 0. A counted candidate past the first
// `max_candidates` ends the trace with the product so far and sets `overflow`.
[ForceInline]
public float3 trace_transmittance<H : ICandidate>(Tlas tlas, RayDesc ray, uint mask, uint flags, inout H handler, uint max_candidates, out bool overflow)
{
    overflow = false;
    float3 product = 1;
    uint count = 0;
    RayQuery<RAY_FLAG_NONE> q;
    begin(q, tlas, ray, mask, flags);
    while (q.Proceed())
    {
        if (opaque_candidate(q, flags))
        {
            q.CommitNonOpaqueTriangleHit();
            continue;
        }
        Candidate c = candidate(q);
        if (c.triangle && !takes(c, mask))
            continue;
        if (c.triangle && handler.positions(c))
            candidate_positions(q, c.object_positions);
        bool counted;
        float3 t = handler.transmittance(c, counted);
        if (counted)
        {
            if (count == max_candidates)
            {
                overflow = true;
                q.Abort();
                break;
            }
            count++;
        }
        product *= t;
        if (all(product == 0))
        {
            q.Abort();
            break;
        }
    }
    if (q.CommittedStatus() != COMMITTED_NOTHING)
        product = 0;
    return product;
}

// The number of depth-ordered layers `trace_resolve_translucent` keeps; further layers
// composite in arrival order behind them.
public static const uint TRANSLUCENT_BINS = 4;

// The closest hit as `trace_closest`, and the translucent layers in front of it composited
// into `layers` (premultiplied radiance in rgb, total coverage in a), procedural candidates
// among them. The nearest `TRANSLUCENT_BINS` layers composite in depth order, later ones in
// arrival order behind them; the arrival-order tail is dropped only when all of it lies
// behind the hit, so a ray that holds every opaque hit out with `TMax` keeps none behind
// one. At most `max_candidates` candidates are visited.
[ForceInline]
public Hit trace_resolve_translucent<H : ITranslucent>(Tlas tlas, RayDesc ray, uint mask, uint flags, H handler, uint max_candidates, out float4 layers)
{
    float3 positions[3];
    return translucent(tlas, ray, mask, flags, handler, max_candidates, false, positions, layers);
}

// As above, with the hit's corner positions as in `trace_closest`.
[ForceInline]
public Hit trace_resolve_translucent<H : ITranslucent>(Tlas tlas, RayDesc ray, uint mask, uint flags, H handler, uint max_candidates, out float4 layers, out float3 positions[3])
{
    return translucent(tlas, ray, mask, flags, handler, max_candidates, true, positions, layers);
}

[ForceInline]
Hit closest<A : IAnyHit>(Tlas tlas, RayDesc ray, uint mask, uint flags, A any_hit, bool fetch, out float3 positions[3])
{
    RayQuery<RAY_FLAG_NONE> q;
    begin(q, tlas, ray, mask, flags);
    while (q.Proceed())
    {
        if (opaque_candidate(q, flags))
            q.CommitNonOpaqueTriangleHit();
        else if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            Candidate c = candidate(q);
            if (takes(c, mask) && any_hit.accept(c))
                q.CommitNonOpaqueTriangleHit();
        }
    }
    Hit h = committed(q);
    committed_positions(q, fetch, h, positions);
    return h;
}

[ForceInline]
Hit translucent<H : ITranslucent>(Tlas tlas, RayDesc ray, uint mask, uint flags, H handler, uint max_candidates, bool fetch, out float3 positions[3], out float4 layers)
{
    float4 bins[TRANSLUCENT_BINS];
    float depth[TRANSLUCENT_BINS];
    for (uint i = 0; i < TRANSLUCENT_BINS; i++)
    {
        bins[i] = 0;
        depth[i] = ray.TMax;
    }
    float4 tail = 0;
    float tail_near = ray.TMax;
    uint count = 0;
    RayQuery<RAY_FLAG_NONE> q;
    begin(q, tlas, ray, mask, flags);
    while (q.Proceed())
    {
        if (count == max_candidates)
        {
            q.Abort();
            break;
        }
        count++;
        if (opaque_candidate(q, flags))
        {
            q.CommitNonOpaqueTriangleHit();
            continue;
        }
        Candidate c = candidate(q);
        if (c.triangle && !takes(c, mask))
            continue;
        float4 layer = handler.layer(c);
        if (any(layer != 0))
        {
            // Insert in depth order; whatever falls off the last bin joins the tail.
            float t = c.t;
            for (uint i = 0; i < TRANSLUCENT_BINS; i++)
            {
                if (t < depth[i])
                {
                    float4 v = bins[i];
                    float d = depth[i];
                    bins[i] = layer;
                    depth[i] = t;
                    layer = v;
                    t = d;
                }
            }
            if (any(layer != 0))
            {
                tail = over(tail, layer);
                tail_near = min(tail_near, t);
            }
        }
        else if (c.triangle && handler.accept(c))
        {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    Hit h = committed(q);
    committed_positions(q, fetch, h, positions);
    float limit = h.hit ? h.t : ray.TMax;
    layers = 0;
    for (uint i = 0; i < TRANSLUCENT_BINS; i++)
    {
        if (depth[i] < limit)
            layers = over(layers, bins[i]);
    }
    if (tail_near < limit)
        layers = over(layers, tail);
    return h;
}

// `back` composited behind `front`, both premultiplied.
float4 over(float4 front, float4 back)
{
    return front + (1 - front.a) * back;
}

[ForceInline]
void begin(inout RayQuery<RAY_FLAG_NONE> q, Tlas tlas, RayDesc ray, uint mask, uint flags)
{
    __target_switch
    {
    case metal:
        metal_begin(q, tlas.get(), ray.Origin, ray.Direction, ray.TMin, ray.TMax, mask, flags);
    default:
#if defined(ORR_OMM_BUILD)
        spirv_asm
        {
            OpCapability RayTracingOpacityMicromapKHR;
            OpExtension "SPV_KHR_opacity_micromap"
        };
        q.TraceRayInline(tlas.get(), flags, mask, ray);
#else
        q.TraceRayInline(tlas.get(), flags & ~RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, ray);
#endif
    }
}

// Slang's own Metal flag mapping drops RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH.
void metal_begin(inout RayQuery<RAY_FLAG_NONE> q, RaytracingAccelerationStructure tlas, float3 origin, float3 direction, float t_min, float t_max, uint mask, uint flags)
{
    __requirePrelude(R"(
inline metal::raytracing::intersection_params orr_ray_params(uint flags)
{
    metal::raytracing::intersection_params p;
    if (flags & 0x01) p.force_opacity(metal::raytracing::forced_opacity::opaque);
    if (flags & 0x02) p.force_opacity(metal::raytracing::forced_opacity::non_opaque);
    if (flags & 0x04) p.accept_any_intersection(true);
    if (flags & 0x10) p.set_triangle_cull_mode(metal::raytracing::triangle_cull_mode::back);
    if (flags & 0x20) p.set_triangle_cull_mode(metal::raytracing::triangle_cull_mode::front);
    if (flags & 0x40) p.set_opacity_cull_mode(metal::raytracing::opacity_cull_mode::opaque);
    if (flags & 0x80) p.set_opacity_cull_mode(metal::raytracing::opacity_cull_mode::non_opaque);
    if (flags & 0x100) p.set_geometry_cull_mode(metal::raytracing::geometry_cull_mode::triangle);
    if (flags & 0x200) p.set_geometry_cull_mode(metal::raytracing::geometry_cull_mode::bounding_box);
    return p;
}
)");
    __intrinsic_asm "($0)->reset(metal::raytracing::ray($2, $3, $4, $5), $1, $6, orr_ray_params($7))";
}

[ForceInline]
Candidate candidate(inout RayQuery<RAY_FLAG_NONE> q)
{
    Candidate c;
    c.instance_index = q.CandidateInstanceIndex();
    c.custom_index = q.CandidateInstanceID();
    c.geometry_index = q.CandidateGeometryIndex();
    c.primitive_index = q.CandidatePrimitiveIndex();
    c.triangle = q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE;
    c.t = c.triangle ? q.CandidateTriangleRayT() : 0;
    c.barycentrics = c.triangle ? q.CandidateTriangleBarycentrics() : float2(0);
    c.front_face = c.triangle && q.CandidateTriangleFrontFace();
    c.object_origin = q.CandidateObjectRayOrigin();
    c.object_direction = q.CandidateObjectRayDirection();
    c.object_positions[0] = 0;
    c.object_positions[1] = 0;
    c.object_positions[2] = 0;
    __target_switch
    {
    case metal:
        c.cluster_id = -1;
        if (c.triangle && (c.custom_index & METAL_CLUSTER_INSTANCE) != 0)
        {
            uint* d = (uint*)metal_candidate_data(q);
            c.cluster_id = int(d[0]);
            c.geometry_index = d[1] & 0xffffff;
            c.primitive_index = d[2];
        }
        c.custom_index &= 0xffffff;
    default:
        c.cluster_id = q.CandidateClusterID();
    }
    c.class_id = class_of(c.custom_index, c.geometry_index);
    return c;
}

// Metal has no cluster acceleration structures and fetches no corners from hits. The backend
// gives every triangle primitive data: its cluster ID (-1 outside clusters), geometry index
// and flags, triangle in the cluster and its three corners. Its stand-ins for cluster BLAS
// also set bit 31 of their instances' user ID, and instances forced non-opaque bit 30, since
// a query cannot read an instance's options.
static const uint METAL_CLUSTER_INSTANCE = 0x80000000;
static const uint METAL_FORCED_NON_OPAQUE = 0x40000000;
static const uint METAL_GEOMETRY_OPAQUE = 0x80000000;

uint64_t metal_candidate_data(inout RayQuery<RAY_FLAG_NONE> q)
{
    __intrinsic_asm "reinterpret_cast<ulong>(($0)->get_candidate_primitive_data())";
}

// The corners a cluster triangle's data holds.
float3[3] metal_corners(uint64_t data)
{
    float* f = (float*)(data + 12);
    float3 p[3];
    for (uint c = 0; c < 3; c++)
        p[c] = float3(f[c * 3], f[c * 3 + 1], f[c * 3 + 2]);
    return p;
}

uint64_t metal_committed_data(inout RayQuery<RAY_FLAG_NONE> q)
{
    __intrinsic_asm "reinterpret_cast<ulong>(($0)->get_committed_primitive_data())";
}

// Whether a ray with mask `mask` takes triangle candidate `c`. An instance's mask covers every
// class it shows, so a ray whose mask misses a class still meets that class's clusters in a mixed
// BLAS, where the class in the geometry index turns it away. A ray that takes both opaque and
// dielectric geometry has no class to turn away.
[ForceInline]
bool takes(Candidate c, uint mask)
{
    return (mask & (MASK_OPAQUE | MASK_DIELECTRIC)) == (MASK_OPAQUE | MASK_DIELECTRIC) || (class_mask(c.class_id) & mask) != 0;
}

// Whether a non-opaque candidate is an opaque triangle of a cluster: Metal makes a whole
// geometry opaque or not, so a cluster mixing them arrives whole and its triangles' own
// flags decide here, unless its instance is forced non-opaque, as a fading copy is. Always
// false outside Metal.
[ForceInline]
bool opaque_candidate(inout RayQuery<RAY_FLAG_NONE> q, uint flags)
{
    __target_switch
    {
    case metal:
        uint user = q.CandidateInstanceID();
        if ((flags & RAY_FLAG_FORCE_NON_OPAQUE) != 0 || q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE
            || (user & METAL_CLUSTER_INSTANCE) == 0 || (user & METAL_FORCED_NON_OPAQUE) != 0)
            return false;
        return (*(uint*)(metal_candidate_data(q) + 4) & METAL_GEOMETRY_OPAQUE) != 0;
    default:
        return false;
    }
}

[ForceInline]
Hit committed(inout RayQuery<RAY_FLAG_NONE> q)
{
    Hit h;
    uint status = q.CommittedStatus();
    h.hit = status != COMMITTED_NOTHING;
    h.triangle = status == COMMITTED_TRIANGLE_HIT;
    h.t = q.CommittedRayT();
    h.instance_index = h.hit ? q.CommittedInstanceIndex() : 0;
    h.custom_index = h.hit ? q.CommittedInstanceID() : 0;
    h.geometry_index = h.hit ? q.CommittedGeometryIndex() : 0;
    h.primitive_index = h.hit ? q.CommittedPrimitiveIndex() : 0;
    h.barycentrics = h.triangle ? q.CommittedTriangleBarycentrics() : float2(0);
    h.front_face = h.triangle && q.CommittedTriangleFrontFace();
    __target_switch
    {
    case metal:
        h.cluster_id = -1;
        if (h.triangle && (h.custom_index & METAL_CLUSTER_INSTANCE) != 0)
        {
            uint* d = (uint*)metal_committed_data(q);
            h.cluster_id = int(d[0]);
            h.geometry_index = d[1] & 0xffffff;
            h.primitive_index = d[2];
        }
        h.custom_index &= 0xffffff;
    default:
        h.cluster_id = h.hit ? q.CommittedClusterID() : -1;
    }
    h.class_id = class_of(h.custom_index, h.geometry_index);
    return h;
}

[ForceInline]
void candidate_positions(inout RayQuery<RAY_FLAG_NONE> q, inout float3 p[3])
{
    __target_switch
    {
    case metal:
        // Every triangle carries its corners after three words of data.
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            p = metal_corners(metal_candidate_data(q));
    default:
        p = q.CandidateGetIntersectionTriangleVertexPositions();
    }
}

// Triangle hits fetch their corners; other hits have none to fetch, and give their point.
[ForceInline]
void committed_positions(inout RayQuery<RAY_FLAG_NONE> q, bool fetch, Hit h, out float3 p[3])
{
    p[0] = 0;
    p[1] = 0;
    p[2] = 0;
    __target_switch
    {
    case metal:
        if (fetch && h.triangle)
            p = metal_corners(metal_committed_data(q));
        else if (fetch && h.hit)
            p[0] = q.CommittedObjectRayOrigin() + h.t * q.CommittedObjectRayDirection();
    default:
        if (fetch && h.triangle)
            p = q.CommittedGetIntersectionTriangleVertexPositions();
        else if (fetch && h.hit)
            p[0] = q.CommittedObjectRayOrigin() + h.t * q.CommittedObjectRayDirection();
    }
}
