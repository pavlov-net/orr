// Shadow rays. A ray query over opaque geometry, particles, medium proxies and dielectric and
// presence layers multiplies what each non-opaque candidate lets through, committing none: a
// presence layer or particle passes 1 minus its opacity, a medium proxy a short march of its own
// medium inside its box. Alpha-tested candidates pass or stop the ray by the alpha test, and an
// opaque hit stops it. What a dielectric does depends on the ray's mode: SHADOW_PATH, a path
// vertex's, passes a thin wall that is an ideal window by its straight transmission and ends at
// any other, and at solid glass, which the AS commits as opaque, since light through either
// bends and BSDF rays and caustic photons carry it; SHADOW_PHOTON, a photon's way from its
// light, ends at every dielectric; SHADOW_UNBENT, what froxels, particle lighting and media
// apply trace, passes every dielectric by its unbent transmission, tracing dielectrics apart
// from the rest so that it forces the AS's solid glass alone non-opaque. That mode keeps its
// crossings sorted by distance, since candidates arrive in no order, and walks them from the
// interior list at its origin: each segment takes the Beer transmittance of the innermost
// interior around it, each crossing its Fresnel against the medium around it, and the walk
// gives the radiance scale between the two ends. The other modes cross no interior surface, so
// their whole segment lies in the innermost interior at the origin. The linked medium's
// transmittance covers the scene's medium, which proxies add to.
module shadow;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_lighting.openpbr;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.gbuffer;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.specialize;
__exported import orr_lighting.interiors;

// The mask shadow rays trace.
public static const uint SHADOW_MASK = MASK_SHADOW | MASK_PARTICLE | MASK_MEDIUM | MASK_DIELECTRIC;

// Steps a medium proxy's march takes at most over the ray's span inside it; spans longer than
// this many voxels step farther than a voxel.
public static const uint PROXY_STEPS = 64;

// The IOR outside every interior.
public static const float AMBIENT_IOR = 1.0;

// The ray origin floor: `T_MIN_BASE` x (1 + meters from the camera x `T_MIN_SCALE`).
public static const float T_MIN_BASE = 1e-3;
public static const float T_MIN_SCALE = 1.0;

// Rays leaving `p` along `dir`: offset toward `dir`'s side of the geometric normal `ng` by
// `error` (ORIGIN_DEFAULT only) or the distance floor, whichever is larger, plus the f32
// precision at `p`'s magnitude. ORIGIN_TRIANGLE offsets from the visible triangle's plane
// through `plane_point`.
public float3 ray_origin(float3 p, float3 plane_point, float3 ng, float3 dir, uint policy, float error, float camera_distance)
{
    float3 n = dot(ng, dir) < 0.0 ? -ng : ng;
    float floor = T_MIN_BASE * (1.0 + camera_distance * T_MIN_SCALE);
    float d = policy == ORIGIN_DEFAULT ? max(error, floor) : floor;
    float3 base = policy == ORIGIN_TRIANGLE ? p - n * dot(p - plane_point, n) : p;
    float magnitude = max(abs(base.x), max(abs(base.y), abs(base.z)));
    return base + n * (d + magnitude * F32_PRECISION);
}

// The length of the segment from surface point `start` to where the ray from `origin`, offset
// from it, along unit `dir` is `t` m on: the stretch from the surface to the origin is the
// segment's too.
public float segment_length(float3 start, float3 origin, float3 dir, float t)
{
    return isfinite(t) ? length(origin - start + dir * t) : t;
}

// Whether material `m`'s IOR varies with wavelength.
public bool disperses(SceneRoot* s, uint m)
{
    if (m == MATERIAL_NONE)
        return false;
    MaterialGpu* g = s.materials + m;
    return g.transmission_dispersion_scale > 0.0 && g.transmission_dispersion_abbe_number > 0.0;
}

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
    InteriorMedium r = InteriorMedium();
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
public struct ShadowCrossing
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
public struct ShadowCandidates<S : ISurfaces, M : IMedium, let MODE : uint> : ICandidate
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
    // A SHADOW_PATH ray's distance to the solid glass candidate that ended it, +inf for none: where
    // a fading copy keeps solid glass non-opaque, it ends the ray as a committed hit would.
    public float solid_t;

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
        uint flags;
        float3 t = thin_candidate<S>(scene, view, context, MODE, dir, c.class_id, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, may(MAY_LAYERS), may(MAY_PARTICLES), counted, flags);
        // Solid glass ends a path's shadow ray whatever its surface is.
        if ((flags & SHADOW_SOLID) != 0)
            solid_t = c.t;
        if ((flags & SHADOW_CROSSES) == 0)
            return t;
        return dielectric(c);
    }

    // A present dielectric candidate that `thin_candidate` passes with SHADOW_CROSSES.
    [mutating]
    float3 dielectric(Candidate c)
    {
        SurfaceHit sh = resolve_hit<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, c.object_positions, false, context);
        if (!sh.valid)
            return float3(1.0);
        float3 n = sh.surface.geometric_normal;
        if (MODE == SHADOW_PATH)
        {
            if (!bsdf_thin_mirror(sh.surface.bsdf, exterior_ior))
                return float3(0.0);
            return bsdf_straight_transmission(sh.surface.bsdf, n, dir, exterior_ior, environment);
        }
        if (crossed < SHADOW_CANDIDATES)
        {
            bool thin = may2(MAY2_THIN_DIELECTRICS) && !crosses_interior(scene, sh.section);
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
    public float3 optical_depth(Interior start, out float scale, out float3 straight, out float solid)
    {
        straight = float3(1.0);
        solid = positive_infinity();
        if (MODE != SHADOW_UNBENT || crossed == 0)
            return straight_depth(scene, start, t_max, MODE == SHADOW_UNBENT, scale);
        scale = 1.0;
        Interior inside = start;
        Interior walked = start;
        bool first = true;
        for (uint k = 0; k < crossed; k++)
        {
            ShadowCrossing c = crossings[k];
            if (c.thin)
                continue;
            bool own = c.front || walked.holds(c.material);
            // Only a manifold walk asks where the ray met solid glass.
            if (may2(MAY2_MANIFOLDS) && first && own)
                solid = c.t;
            first = false;
            if (own)
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
        float3 beer;
        float t0, t1;
        MediumProxyGpu p;
        if (proxy_unmarched(scene, c, t_max, beer, p, t0, t1))
            return beer;
        return proxy_volume(scene, p, c.instance_index, c.primitive_index, origin, o, d, t0, t1);
    }
}

// What the span from `t0` to `t1` of a ray from render-space `origin` lets through inside proxy
// `p` with a volume, primitive `primitive` of instance `instance_index`, along the object-space
// ray `o + d * t`, by `ShadowCandidates.proxy`'s march.
public float3 proxy_volume(SceneRoot* s, MediumProxyGpu p, uint instance_index, uint primitive, float3 origin, float3 o, float3 d, float t0, float t1)
{
    uint3 key = uint3(asuint(origin.x) ^ (instance_index * 0x9E3779B9u), asuint(origin.y) ^ (primitive * 0x85EBCA6Bu), asuint(origin.z) ^ s.frame_index);
    float jitter = unit_float(pcg3d(key).x);
    float h = max(proxy_voxel_step(p, d), (t1 - t0) / float(PROXY_STEPS));
    ProxyDepth depth = { p, o, d, float3(0.0) };
    proxy_march(t0, t1, h, jitter, depth);
    return exp(-max(depth.tau, 0.0));
}

// As above for the span from `t0` to `t1` of the render-space ray from `origin` along unit `dir`
// through the proxy of instance `instance_index`, whose object space its record places.
public float3 proxy_volume(SceneRoot* s, uint instance_index, uint primitive, float3 origin, float3 dir, float t0, float t1)
{
    MediumProxyGpu p = scene_proxy(s, instance_index);
    // The rows of the inverse of the object-to-world columns, over its determinant.
    float3 x = cross(p.axis_y, p.axis_z);
    float3 y = cross(p.axis_z, p.axis_x);
    float3 z = cross(p.axis_x, p.axis_y);
    float det = dot(p.axis_x, x);
    float3 rel = origin - proxy_render_center(s, p);
    float3 o = float3(dot(x, rel), dot(y, rel), dot(z, rel)) / det;
    float3 d = float3(dot(x, dir), dot(y, dir), dot(z, dir)) / det;
    return proxy_volume(s, p, instance_index, primitive, origin, o, d, t0, t1);
}

// What medium proxy candidate `c` of a ray reaching `t_max` lets through, in `t`, where that
// needs no march: where the ray misses its box, and Beer's law over the ray's span inside a proxy
// without a volume. False for a span that a volume's march takes, with the span in `t0` and `t1`
// and the proxy in `p`.
public bool proxy_unmarched(SceneRoot* s, Candidate c, float t_max, out float3 t, out MediumProxyGpu p, out float t0, out float t1)
{
    t = float3(1.0);
    p = MediumProxyGpu();
    if (!proxy_span(c.object_origin, c.object_direction, t_max, t0, t1))
        return true;
    p = scene_proxy(s, c.instance_index);
    return proxy_beer(p, t0, t1, t);
}

// The interiors' optical depth along a ray from `t_max` that crosses no interior surface, inside
// the list `start` at its origin, as `ShadowCandidates.optical_depth` takes it: in mode
// SHADOW_UNBENT where `unbent`, with in `scale` the radiance scale of a light at `t_max`, else in
// the other modes, whose scale is 1.
public float3 straight_depth(SceneRoot* s, Interior start, float t_max, bool unbent, out float scale)
{
    scale = 1.0;
    uint m = start.innermost();
    if (!unbent)
    {
        float3 sigma = interior_medium(s, m, start.point(m)).sigma_t;
        return select(sigma > 0.0, sigma * t_max, float3(0.0));
    }
    if (start.count == 0)
        return float3(0.0);
    float3 sigma = interior_medium(s, m, start.point(m)).sigma_t;
    float ior = interior_ior(s, m);
    float3 depth = select(sigma > 0.0, sigma * t_max, float3(0.0));
    float reduced = t_max / ior;
    if (isfinite(t_max))
    {
        float k = t_max / max(ior * reduced, 1e-30);
        scale = k * k;
    }
    return depth;
}

// Sums a proxy's optical depth over the steps of its march.
public struct ProxyDepth : IProxyStep
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
// SHADOW_CANDIDATES non-opaque candidates in one of its traces keeps the product so far and adds
// 1 to `overflows` (null: not counted).
public float3 visibility<S : ISurfaces, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows, bool media_only = false)
{
    float3 depth;
    float3 product = surface_visibility<S, M, MODE>(s, v, origin, dir, t_max, flags, context, environment, inside, overflows, depth, media_only);
    if (all(product == 0.0))
        return product;
    product *= exp(-depth);
    if (!environment)
        product *= medium_transmittance<M>(s.medium, origin, dir, t_max);
    return product;
}

// `visibility` without the linked medium, with the dielectric interiors' optical depth over the
// segment left out of the product and returned in `depth`. With `media_only` the ray meets the
// medium proxies alone: a light surfaces cast no shadow from.
public float3 surface_visibility<S : ISurfaces, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows, out float3 depth, bool media_only = false)
{
    bool glass;
    return surface_visibility<S, M, MODE>(s, v, origin, dir, t_max, flags, context, environment, inside, overflows, depth, glass, media_only);
}

// As above, and in `glass` whether a SHADOW_UNBENT ray crosses solid glass and nothing opaque, at
// a crossing that enters the glass or leaves an interior of `inside`: one leaving an interior the
// list at its origin does not hold says the origin's list is not its own.
public float3 surface_visibility<S : ISurfaces, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows, out float3 depth, out bool glass, bool media_only)
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
    h.solid_t = positive_infinity();
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    bool overflow;
    // Any opaque hit makes the product 0, so the search ends at the first it finds.
    flags |= RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    // SHADOW_UNBENT traces everything but dielectrics as the AS has it, then dielectrics alone
    // forced non-opaque, so the AS's solid glass arrives as crossings and opaque geometry takes no
    // candidate handling. Solid glass in an instance that mixes classes commits in the first
    // trace; that ray traces everything forced non-opaque instead. One call site keeps one copy
    // of the candidate handling in the kernel.
    float3 product = float3(1.0);
    overflow = false;
    uint mask = media_only ? MASK_MEDIUM : MODE == SHADOW_UNBENT ? SHADOW_MASK & ~MASK_DIELECTRIC : SHADOW_MASK;
    uint pass_flags = flags;
    for (uint pass = 0; pass < 2; pass++)
    {
        bool more;
        uint c;
        float c_t;
        float3 p = trace_transmittance(scene_traced(s), ray, mask, pass_flags, h, SHADOW_CANDIDATES, more, c, c_t);
        overflow = overflow || more;
        product *= p;
        if (MODE != SHADOW_UNBENT || pass == 1 || media_only)
            break;
        if (c == CLASS_DIELECTRIC)
        {
            mask = SHADOW_MASK;
            product = float3(1.0);
            overflow = false;
        }
        else if (all(product == 0.0))
            break;
        else
            mask = MASK_DIELECTRIC;
        pass_flags = flags | RAY_FLAG_FORCE_NON_OPAQUE;
    }
    if (overflow && overflows != nullptr)
        InterlockedAdd(*overflows, 1u);
    float scale;
    float3 straight;
    float solid;
    depth = h.optical_depth(inside, scale, straight, solid);
    glass = MODE == SHADOW_UNBENT && !all(product == 0.0) && solid < positive_infinity();
    return product * straight * scale;
}

// The direction in a layer of IOR `ior` that light along unit `dir` outside every interior takes
// across planes of unit normal `normal`; false where none refracts into `dir`.
public bool stack_direction(float3 dir, float3 normal, float ior, out float3 bent)
{
    bent = dir;
    if (ior == AMBIENT_IOR)
        return true;
    float c = dot(dir, normal);
    float3 t = (dir - normal * c) * (AMBIENT_IOR / ior);
    float s2 = dot(t, t);
    if (!(s2 < 1.0))
        return false;
    bent = normalize(t + normal * ((c < 0.0 ? -1.0 : 1.0) * sqrt(1.0 - s2)));
    return true;
}

// Commits the opaque and alpha-tested candidates that are present and cooked solid glass that a
// fading copy keeps non-opaque: what ends a stack's leg beside the AS's opaque geometry and solid
// glass.
struct StackAccept<S : ISurfaces> : IAnyHit
{
    SceneRoot* scene;
    ViewConstants* view;
    float3 dir;
    HitContext context;

    bool accept(Candidate c)
    {
        return c.cluster_id >= 0 && cover_decision<S>(scene, view, COVER_STACK, PRESENCE_NONE, 0, context, dir, c.class_id, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics);
    }
}

// The nearest opaque surface or solid glass from `origin` along unit `dir` within `t_max`.
public Hit stack_trace<S : ISurfaces>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context)
{
    StackAccept<S> a = { s, v, dir, context };
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    return trace_closest(scene_traced(s), ray, MASK_SHADOW | MASK_DIELECTRIC, flags, a);
}
