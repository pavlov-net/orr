// Shadow rays. One ray query over opaque geometry, particles, medium proxies and dielectric
// and presence layers multiplies what each non-opaque candidate lets through, committing
// none: a presence layer or particle passes 1 minus its opacity, a medium proxy a short march
// of its own medium inside its box. Alpha-tested candidates pass or stop the ray by the alpha
// test, and an opaque hit stops it. What a dielectric does depends on the ray's mode:
// SHADOW_PATH, a path vertex's, passes a thin wall that is an ideal window by its straight
// transmission and ends at any other, and at solid glass, which the AS commits as opaque, since
// light through either bends and BSDF rays and caustic photons carry it; SHADOW_PHOTON, a
// photon's way from its light, ends at every dielectric; SHADOW_UNBENT, what froxels, particle
// lighting and media apply trace, passes every dielectric by its unbent transmission, forcing
// the AS's solid glass non-opaque. That mode keeps its crossings sorted by distance, since
// candidates arrive in no order, and walks them from the interior list at its origin: each
// segment takes the Beer transmittance of the innermost interior around it, each crossing its
// Fresnel against the medium around it, and the walk gives the radiance scale between the two
// ends. The other modes cross no interior surface, so their whole segment lies in the innermost
// interior at the origin. The linked medium's transmittance covers the scene's medium, which
// proxies add to.
module shadow;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_lighting.openpbr;
import orr_scene.hit;
import orr_lighting.bsdf;
import orr_lighting.media;
__exported import orr_lighting.interiors;

// The mask shadow rays trace.
public static const uint SHADOW_MASK = MASK_OPAQUE | MASK_PARTICLE | MASK_MEDIUM | MASK_DIELECTRIC;

// `visibility` modes.
public static const uint SHADOW_PATH = 0;
public static const uint SHADOW_PHOTON = 1;
public static const uint SHADOW_UNBENT = 2;

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
// `transmission_depth` and `transmission_scatter` (OpenPBR's mapping) as `material_medium` reads
// them at `point`: none for MATERIAL_NONE, thin-walled materials and a depth of 0.
public struct InteriorMedium
{
    public float3 sigma_t;          // 1/m
    public float3 sigma_s;
    public float anisotropy;

    public float3 albedo()
    {
        return select(sigma_t > 0.0, sigma_s / sigma_t, float3(0.0));
    }
}

public InteriorMedium interior_medium(SceneRoot* s, uint m, float4 point = NO_MEDIUM_POINT)
{
    InteriorMedium r = (InteriorMedium)0;
    if (m == MATERIAL_NONE)
        return r;
    MaterialGpu* g = s.materials + m;
    MediumParams p = material_medium(g, point);
    if ((g.flags & MATERIAL_THIN_WALLED) != 0 || !(p.depth > 0.0))
        return r;
    OpenPBR_HomogeneousVolume v = openpbr_create_volume_from_openpbr_params(p.color, p.depth, p.scatter, p.anisotropy);
    r.sigma_t = max(v.extinction_coefficient, 0.0);
    r.sigma_s = r.sigma_t * saturate(v.albedo);
    r.anisotropy = v.anisotropy;
    return r;
}

// An unbent shadow ray's crossing of a dielectric: material `material` at `t` m.
struct ShadowCrossing
{
    float t;
    uint material;
    bool front;
    bool thin;      // a thin wall, which bounds no interior
    float4 point;   // what a medium entered here reads: `material_medium`
    StraightSurface surface;
}

// The crossings of the shadow ray in flight. Held outside its handler, which every candidate
// copies in and out.
static ShadowCrossing crossings[SHADOW_CANDIDATES];

// The candidates of one shadow ray from `origin` along unit `dir` to `t_max`.
public struct ShadowCandidates<S : ISurface, M : IMedium, let MODE : uint> : ICandidate
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 origin;
    public float3 dir;
    public float t_max;
    public HitContext context;
    public bool environment;
    // The IOR of the innermost interior at the origin, around a SHADOW_PATH ray's windows.
    public float exterior_ior;
    public uint crossed;

    public bool positions(Candidate c)
    {
        return c.class_id == CLASS_DIELECTRIC;
    }

    [mutating]
    public float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        if (!c.triangle)
        {
            if (c.class_id != CLASS_MEDIUM)
                return float3(1.0);
            counted = true;
            return proxy(c);
        }
        if (c.cluster_id < 0)
            return float3(1.0);
        uint cluster = uint(c.cluster_id);
        switch (c.class_id)
        {
        case CLASS_DIELECTRIC:
            if (!resolve_alpha<S>(scene, view, dir, c.instance_index, cluster, c.primitive_index, c.barycentrics, context))
                return float3(1.0);
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
        if (MODE == SHADOW_PHOTON)
            return float3(0.0);
        SurfaceHit sh = resolve_hit<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, c.object_positions, false, context);
        if (!sh.valid)
            return float3(1.0);
        float3 n = sh.surface.geometric_normal;
        bool thin = !crosses_interior(scene, sh.section);
        if (MODE == SHADOW_PATH)
        {
            if (!thin || !bsdf_thin_mirror(sh.surface.bsdf, exterior_ior))
                return float3(0.0);
            return bsdf_straight_transmission(sh.surface.bsdf, n, dir, exterior_ior, environment);
        }
        if (crossed < SHADOW_CANDIDATES)
        {
            uint at = crossed;
            for (; at > 0 && crossings[at - 1].t > c.t; at--)
                crossings[at] = crossings[at - 1];
            ShadowCrossing crossing = { c.t, sh.section, dot(n, dir) < 0.0, thin, sh.surface.medium_point, straight_surface(sh.surface.bsdf, n, dir) };
            crossings[at] = crossing;
            crossed++;
        }
        return float3(1.0);
    }

    // The optical depth of the interiors along the ray, each segment once, walking its crossings
    // from the list at its origin, and in `straight` what the crossings let through. `start` may
    // lack interiors holding the origin: a back face the walk holds nothing for shows one, inside
    // whatever the walk still holds there, so the first pass completes the list, reading the
    // interior where the ray leaves it, the only point of it the ray knows. An interior a full
    // list then drops takes the segment ending at its back face, read there. Each crossing's
    // Fresnel takes the medium around it: the innermost interior there other than its own.
    // `scale` is the radiance of a light at `t_max` as the walk delivers it, over what an unbent
    // ray gives: radiance grows with the squared IOR, and refraction shrinks the solid angle of a
    // light seen through an interface, which is the paraxial (t_max / (n_light * sum(segment /
    // n)))^2 and 1 for a light at infinity.
    public float3 optical_depth(Interior start, out float scale, out float3 straight)
    {
        scale = 1.0;
        straight = float3(1.0);
        if (MODE != SHADOW_UNBENT)
        {
            uint m = start.innermost();
            float3 sigma = interior_medium(scene, m, start.point(m)).sigma_t;
            return select(sigma > 0.0, sigma * t_max, float3(0.0));
        }
        if (crossed == 0 && start.count == 0)
            return float3(0.0);
        Interior inside = start;
        Interior walked = start;
        for (uint k = 0; k < crossed; k++)
        {
            ShadowCrossing c = crossings[k];
            if (c.thin)
                continue;
            if (c.front || walked.holds(c.material))
                walked.cross(c.material, c.front);
            else
                inside.nest(c.material, walked, c.point);
        }
        float3 depth = float3(0.0);
        float reduced = 0.0;
        float at = 0.0;
        float ior = AMBIENT_IOR;
        for (uint k = 0; k <= crossed; k++)
        {
            bool end = k == crossed;
            ShadowCrossing c = crossings[min(k, SHADOW_CANDIDATES - 1)];
            bool dropped = !end && !c.thin && !c.front && !inside.holds(c.material);
            uint m = dropped ? c.material : inside.innermost();
            float length = (end ? t_max : c.t) - at;
            float3 sigma = interior_medium(scene, m, dropped ? c.point : inside.point(m)).sigma_t;
            ior = interior_ior(scene, m);
            depth += select(sigma > 0.0, sigma * length, float3(0.0));
            reduced += length / ior;
            if (!end)
            {
                straight *= straight_transmission(c.surface, interior_ior(scene, inside.outside(c.material, c.material)), environment);
                if (!c.thin)
                    inside.cross(c.material, c.front, c.point);
                at = c.t;
            }
        }
        if (isfinite(t_max))
        {
            float s = t_max / max(ior * reduced, 1e-30);
            scale = s * s;
        }
        return depth;
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
        float t0, t1;
        if (!proxy_span(o, d, t_max, t0, t1))
            return float3(1.0);
        MediumProxyGpu p = scene_proxy(scene, c.instance_index);
        if (p.volume == nullptr)
            return exp(-max(p.extinction, 0.0) * (t1 - t0));
        uint3 key = uint3(asuint(origin.x) ^ (c.instance_index * 0x9E3779B9u), asuint(origin.y) ^ (c.primitive_index * 0x85EBCA6Bu), asuint(origin.z) ^ scene.frame_index);
        float jitter = unit_float(pcg3d(key).x);
        float h = max(proxy_voxel_step(p, d), (t1 - t0) / float(PROXY_STEPS));
        ProxyDepth depth = { p, o, d, float3(0.0) };
        proxy_march(t0, t1, h, jitter, depth);
        return exp(-max(depth.tau, 0.0));
    }
}

// Sums a proxy's optical depth over the steps of its march.
struct ProxyDepth : IProxyStep
{
    MediumProxyGpu proxy;
    float3 origin;
    float3 direction;
    float3 tau;

    float majorant(float t, out float t_exit)
    {
        return proxy_majorant_span(proxy, origin, direction, t, t_exit);
    }

    [mutating]
    void step(float t, float h)
    {
        tau += proxy_medium(proxy, origin + direction * t).sigma_t * h;
    }
}

// What reaches `origin` from `t_max` along unit `dir` (+inf toward directional lights and the
// environment), traced in mode `MODE` (SHADOW_*) with ray flags `flags` from a point inside the
// interiors of `inside`, which need not name those the ray leaves through: the product of every
// candidate along the segment, 0 at an opaque hit, times the linked medium's transmittance over
// it unless the ray goes to the `environment`, whose map holds that medium. A ray past
// SHADOW_CANDIDATES non-opaque candidates keeps the product so far and adds 1 to `overflows`
// (null: not counted).
public float3 visibility<S : ISurface, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows)
{
    float3 depth;
    float3 product = surface_visibility<S, M, MODE>(s, v, origin, dir, t_max, flags, context, environment, inside, overflows, depth);
    if (all(product == 0.0))
        return product;
    product *= exp(-depth);
    if (!environment)
        product *= medium_transmittance<M>(s.medium, origin, dir, t_max);
    return product;
}

// `visibility` without the linked medium, with the dielectric interiors' optical depth over the
// segment left out of the product and returned in `depth`.
public float3 surface_visibility<S : ISurface, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows, out float3 depth)
{
    ShadowCandidates<S, M, MODE> h;
    h.scene = s;
    h.view = v;
    h.origin = origin;
    h.dir = dir;
    h.t_max = t_max;
    h.context = context;
    h.environment = environment;
    h.exterior_ior = interior_ior(s, inside.innermost());
    h.crossed = 0;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    if (MODE == SHADOW_UNBENT)
        flags |= RAY_FLAG_FORCE_NON_OPAQUE;
    bool overflow;
    // Any opaque hit makes the product 0, so the search ends at the first it finds.
    float3 product = trace_transmittance(s.tlas, ray, SHADOW_MASK, flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, h, SHADOW_CANDIDATES, overflow);
    if (overflow && overflows != nullptr)
        InterlockedAdd(*overflows, 1u);
    float scale;
    float3 straight;
    depth = h.optical_depth(inside, scale, straight);
    return product * straight * scale;
}
