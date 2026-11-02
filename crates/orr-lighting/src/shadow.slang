// Shadow rays. One ray query over opaque geometry, particles, medium proxies and dielectric
// and presence layers multiplies what each non-opaque candidate lets through, committing
// none: a dielectric passes its unbent transmission and the Beer transmittance of its
// interior over the distance the ray spends inside, a presence layer or particle 1 minus its
// opacity, a medium proxy a short march of its own medium inside its box. Alpha-tested
// candidates pass or stop the ray by the alpha test, and an opaque hit stops it. The linked
// medium's transmittance covers the scene's medium, which proxies add to.
module shadow;

import orr.abi;
import orr.math;
import orr.random;
import orr.trace;
import orr.openpbr;
import hit;
import bsdf;
import media;

// The mask shadow rays trace.
public static const uint SHADOW_MASK = MASK_OPAQUE | MASK_PARTICLE | MASK_MEDIUM | MASK_DIELECTRIC;

// Non-opaque candidates a shadow ray multiplies before it ends with the product so far.
// Alpha-tested candidates do not count.
public static const uint SHADOW_CANDIDATES = 16;

// Steps a medium proxy's march takes at most over the ray's span inside it; spans longer than
// this many voxels step farther than a voxel.
public static const uint PROXY_STEPS = 64;

// The IOR outside every interior.
public static const float AMBIENT_IOR = 1.0;

// exp(-sigma t) per channel; 1 where sigma is 0, whatever `t`.
public float3 beer(float3 sigma, float t)
{
    return select(sigma > 0.0, exp(-sigma * t), float3(1.0));
}

// The IOR of material `m`'s interior: AMBIENT_IOR for MATERIAL_NONE.
public float interior_ior(SceneRoot* s, uint m)
{
    return m == MATERIAL_NONE ? AMBIENT_IOR : s.materials[m].specular_ior;
}

// The homogeneous medium inside dielectric material `m`, from its `transmission_color` over
// `transmission_depth` and `transmission_scatter` (OpenPBR's mapping): none for
// MATERIAL_NONE, thin-walled materials and a depth of 0.
public struct InteriorMedium
{
    public float3 sigma_t;          // 1/m
    public float3 sigma_s;
    public float anisotropy;
}

public InteriorMedium interior_medium(SceneRoot* s, uint m)
{
    InteriorMedium r = (InteriorMedium)0;
    if (m == MATERIAL_NONE)
        return r;
    MaterialGpu* g = s.materials + m;
    if ((g.flags & MATERIAL_THIN_WALLED) != 0 || !(g.transmission_depth > 0.0))
        return r;
    OpenPBR_HomogeneousVolume v = openpbr_create_volume_from_openpbr_params(
        g.transmission_color, g.transmission_depth, g.transmission_scatter, g.transmission_scatter_anisotropy);
    r.sigma_t = max(v.extinction_coefficient, 0.0);
    r.sigma_s = r.sigma_t * saturate(v.albedo);
    r.anisotropy = v.anisotropy;
    return r;
}

// The candidates of one shadow ray from `origin` along unit `dir` to `t_max`. Dielectric
// interiors add their optical depth to `depth` with the sign of the face crossed: minus the
// candidate's distance times sigma at a front face, plus it at a back face, so the sum is the
// optical depth over the ray's distance inside in any candidate order.
public struct ShadowCandidates<S : ISurface, M : IMedium> : ICandidate
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 origin;
    public float3 dir;
    public float t_max;
    public HitContext context;
    public float3 depth;

    public bool positions(Candidate c)
    {
        return c.custom_index == CLASS_DIELECTRIC;
    }

    [mutating]
    public float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        if (!c.triangle)
        {
            if (c.custom_index != CLASS_MEDIUM)
                return float3(1.0);
            counted = true;
            return proxy(c);
        }
        if (c.cluster_id < 0)
            return float3(1.0);
        uint cluster = uint(c.cluster_id);
        switch (c.custom_index)
        {
        case CLASS_DIELECTRIC:
            counted = true;
            return dielectric(c);
        case CLASS_PRESENCE:
        case CLASS_PARTICLE:
            counted = true;
            return float3(1.0 - saturate(resolve_coverage<S>(scene, view, dir, c.instance_index, cluster, c.primitive_index, c.barycentrics, context)));
        default:
            return resolve_alpha<S>(scene, view, dir, c.instance_index, cluster, c.primitive_index, c.barycentrics, context) ? float3(0.0) : float3(1.0);
        }
    }

    [mutating]
    float3 dielectric(Candidate c)
    {
        SurfaceHit sh = resolve_hit<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, c.object_positions, false, context);
        if (!sh.valid)
            return float3(1.0);
        float3 n = sh.surface.geometric_normal;
        bool front = dot(n, dir) < 0.0;
        depth += interior_medium(scene, sh.surface.material).sigma_t * (front ? -c.t : c.t);
        return bsdf_straight_transmission(sh.surface.bsdf, n, dir, AMBIENT_IOR);
    }

    // A medium proxy is the cube [-1, 1]^3 in its instance's object space. The ray's span
    // inside it takes Beer's law for a proxy without a volume. With one, a march at the
    // volume's voxel spacing (farther past PROXY_STEPS voxels), jittered once per ray, sums
    // the optical depth over the majorant cells that are not empty and skips the rest. The
    // object ray shares the ray's t, and `dir` is unit, so t is in meters.
    float3 proxy(Candidate c)
    {
        float3 o = c.object_origin;
        float3 d = c.object_direction;
        float3 inv = select(d != 0.0, 1.0 / d, float3(1e30));
        float3 a = (-1.0 - o) * inv;
        float3 b = (1.0 - o) * inv;
        float3 lo = min(a, b);
        float3 hi = max(a, b);
        float t0 = max(max(lo.x, max(lo.y, lo.z)), 0.0);
        float t1 = min(min(hi.x, min(hi.y, hi.z)), t_max);
        if (!(t1 > t0))
            return float3(1.0);
        MediumProxyGpu p = scene_proxy(scene, c.instance_index);
        if (p.volume == nullptr)
            return exp(-max(p.extinction, 0.0) * (t1 - t0));
        uint3 key = uint3(asuint(origin.x) ^ (c.instance_index * 0x9E3779B9u), asuint(origin.y) ^ (c.primitive_index * 0x85EBCA6Bu), asuint(origin.z) ^ scene.frame_index);
        float jitter = unit_float(pcg3d(key).x);
        float h = max(proxy_voxel_step(p, d), (t1 - t0) / float(PROXY_STEPS));
        // Steps sit at t0 + (k + jitter) h; `next` is the first k not yet taken.
        uint next = 0;
        float3 tau = float3(0.0);
        float t = t0;
        // The cap on cells only guards against a walk that stops advancing.
        for (uint cell = 0; cell < 4 * PROXY_STEPS && t < t1; cell++)
        {
            float t_exit;
            float m = proxy_majorant_span(p, o, d, t, t_exit);
            float end = min(max(t_exit, t + 1e-5 * (t1 - t0)), t1);
            uint k = max(next, uint(max(ceil((t - t0) / h - jitter), 0.0)));
            for (; m > 0.0 && k < PROXY_STEPS + 1; k++)
            {
                float x = t0 + (float(k) + jitter) * h;
                if (x >= end)
                    break;
                tau += proxy_medium(p, o + d * x).sigma_t * h;
            }
            next = k;
            t = end;
        }
        return exp(-max(tau, 0.0));
    }
}

// What reaches `origin` from `t_max` along unit `dir` (+inf toward directional lights and the
// environment), traced with ray flags `flags`: the product of every candidate along the
// segment, 0 at an opaque hit, times the linked medium's transmittance over it when
// `through_medium`. A ray past SHADOW_CANDIDATES non-opaque candidates keeps the product so
// far and adds 1 to `overflows` (null: not counted).
public float3 visibility<S : ISurface, M : IMedium>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool through_medium, uint* overflows)
{
    ShadowCandidates<S, M> h = { s, v, origin, dir, t_max, context, float3(0.0) };
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    bool overflow;
    float3 product = trace_transmittance(s.tlas, ray, SHADOW_MASK, flags, h, SHADOW_CANDIDATES, overflow);
    if (overflow && overflows != nullptr)
        InterlockedAdd(*overflows, 1u);
    if (all(product == 0.0))
        return product;
    product *= exp(-max(h.depth, 0.0));
    if (through_medium)
        product *= medium_transmittance<M>(s.medium, origin, dir, t_max);
    return product;
}
