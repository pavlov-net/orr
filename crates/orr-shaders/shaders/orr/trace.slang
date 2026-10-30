// Inline ray tracing. This is the only module that names `RayQuery`: kernels trace through
// the functions below and never hold a query themselves.
//
// A hit is any committed status other than none. Linear swept sphere hits commit with a
// status that is neither triangle nor procedural, so `Hit.triangle` is false for them and
// their cluster id is -1.
module trace;

import orr.abi;

// TLAS instance mask bits.
public static const uint MASK_OPAQUE = 0x01;     // opaque and alpha-tested geometry
public static const uint MASK_PARTICLE = 0x02;   // particle batches
public static const uint MASK_MEDIUM = 0x04;     // medium proxies (AABBs)
public static const uint MASK_DIELECTRIC = 0x80; // dielectric and presence surfaces

// Candidate classes, held in an instance's custom index (`Candidate.custom_index`).
public static const uint CLASS_OPAQUE = 0; // opaque or alpha-tested
public static const uint CLASS_DIELECTRIC = 1;
public static const uint CLASS_PRESENCE = 2;
public static const uint CLASS_PARTICLE = 3;
public static const uint CLASS_MEDIUM = 4;

// Ray flags beside Slang's `RAY_FLAG_*`: opacity micromaps forced to two states. It takes
// effect in the opacity-micromap variant of the module and is dropped from the others.
public static const uint RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE = 0x400;

// The flag is invalid without the micromap capability, which only the variant declares. Its
// value reaches the ray query through this constant: 0 here, set to the flag by the build in
// the variant. The id is reserved and never appears in a module's `Spec`.
[vk::constant_id(2147483648)]
const uint OMM_2_STATE = 0;

// A non-opaque candidate as the candidate handlers see it.
public struct Candidate
{
    public uint instance_index; // TLAS or PTLAS instance index
    public uint custom_index;   // the instance's custom index: its candidate class
    public uint geometry_index;
    public uint primitive_index;
    public int cluster_id;      // -1 outside cluster BLAS
    public bool triangle;       // false for procedural (AABB) candidates
    public float t;             // triangle candidates only
    public float2 barycentrics; // triangle candidates only
    public bool front_face;     // triangle candidates only
    public float3 object_origin;
    public float3 object_direction;
}

// The committed hit of a trace.
public struct Hit
{
    public bool hit;
    public bool triangle;
    public float t;
    public uint instance_index;
    public uint custom_index;
    public uint geometry_index;
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

// Gives the transmittance of each non-opaque candidate for `trace_transmittance`.
public interface ICandidate
{
    // 0 blocks the ray, 1 lets it through unchanged.
    float3 transmittance(Candidate c);
}

// Decides candidates for `trace_resolve_translucent`: translucent layers resolve in depth
// order in front of the hit; other candidates commit or not through `accept`.
public interface ITranslucent : IAnyHit
{
    // A translucent layer: premultiplied radiance in rgb, coverage in a. Coverage 0 means
    // the candidate is not a layer and goes to `accept`.
    float4 layer(Candidate c);
}

// The closest hit, with non-opaque triangle candidates committed where `any_hit` accepts
// them. Procedural candidates are skipped.
[ForceInline]
public Hit trace_closest<A : IAnyHit>(Tlas tlas, RayDesc ray, uint mask, uint flags, A any_hit)
{
    float3 positions[3];
    return closest(tlas, ray, mask, flags, any_hit, false, positions);
}

// As above, and the object-space corner positions of a triangle hit (zero on other hits
// and on Metal). The BLAS must allow data access.
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
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && any_hit.accept(candidate(q)))
            q.CommitNonOpaqueTriangleHit();
    }
    return q.CommittedStatus() != COMMITTED_NOTHING;
}

// The product of the transmittance of every non-opaque candidate along the ray, never
// committing one. An opaque hit gives 0. After `max_candidates` candidates the trace ends
// with the product so far and sets `overflow`.
[ForceInline]
public float3 trace_transmittance<H : ICandidate>(Tlas tlas, RayDesc ray, uint mask, uint flags, H handler, uint max_candidates, out bool overflow)
{
    overflow = false;
    float3 product = 1;
    uint count = 0;
    RayQuery<RAY_FLAG_NONE> q;
    begin(q, tlas, ray, mask, flags);
    while (q.Proceed())
    {
        if (count == max_candidates)
        {
            overflow = true;
            q.Abort();
            break;
        }
        count++;
        product *= handler.transmittance(candidate(q));
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
// into `layers` (premultiplied radiance in rgb, total coverage in a). The nearest
// `TRANSLUCENT_BINS` layers composite in depth order, later ones in arrival order behind
// them; the arrival-order tail is dropped only when all of it lies behind the hit. At most
// `max_candidates` candidates are visited.
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
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE && any_hit.accept(candidate(q)))
            q.CommitNonOpaqueTriangleHit();
    }
    Hit h = committed(q);
    committed_positions(q, fetch && h.triangle, positions);
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
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
            continue;
        Candidate c = candidate(q);
        float4 layer = handler.layer(c);
        if (layer.a > 0)
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
            if (layer.a > 0)
            {
                tail = over(tail, layer);
                tail_near = min(tail_near, t);
            }
        }
        else if (handler.accept(c))
        {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    Hit h = committed(q);
    committed_positions(q, fetch && h.triangle, positions);
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
        uint omm = (flags & RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE) != 0 ? OMM_2_STATE : 0;
        q.TraceRayInline(tlas.get(), (flags & ~RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE) | omm, mask, ray);
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
    __target_switch
    {
    case metal:
        c.cluster_id = -1;
    default:
        c.cluster_id = q.CandidateClusterID();
    }
    return c;
}

[ForceInline]
Hit committed(inout RayQuery<RAY_FLAG_NONE> q)
{
    Hit h;
    uint status = q.CommittedStatus();
    h.hit = status != COMMITTED_NOTHING;
    h.triangle = status == COMMITTED_TRIANGLE_HIT;
    h.t = q.CommittedRayT();
    h.instance_index = q.CommittedInstanceIndex();
    h.custom_index = q.CommittedInstanceID();
    h.geometry_index = q.CommittedGeometryIndex();
    h.primitive_index = q.CommittedPrimitiveIndex();
    h.barycentrics = h.triangle ? q.CommittedTriangleBarycentrics() : float2(0);
    h.front_face = h.triangle && q.CommittedTriangleFrontFace();
    __target_switch
    {
    case metal:
        h.cluster_id = -1;
    default:
        h.cluster_id = h.hit ? q.CommittedClusterID() : -1;
    }
    return h;
}

[ForceInline]
void committed_positions(inout RayQuery<RAY_FLAG_NONE> q, bool fetch, out float3 p[3])
{
    p[0] = 0;
    p[1] = 0;
    p[2] = 0;
    __target_switch
    {
    case metal:
        break;
    default:
        if (fetch)
            p = q.CommittedGetIntersectionTriangleVertexPositions();
    }
}
