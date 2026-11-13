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

// The ray origin floor: `T_MIN_BASE` x (1 + meters from the camera x `T_MIN_SCALE`).
public static const float T_MIN_BASE = 1e-3;
public static const float T_MIN_SCALE = 1.0;

// The offset per meter of render-space magnitude that clears f32 rounding: 4 to 8 ulps.
public static const float F32_PRECISION = 1.0 / 1048576.0;

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
        bool layer = (may(MAY_LAYERS) && c.class_id == CLASS_PRESENCE) || (may(MAY_PARTICLES) && c.class_id == CLASS_PARTICLE);
        uint section;
        float present = resolve_presence_section<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context, !layer, section);
        if (layer)
        {
            counted = true;
            return float3(1.0 - saturate(present));
        }
        if (!may(MAY_LAYERS) || c.class_id != CLASS_DIELECTRIC)
            return present > 0.0 ? float3(0.0) : float3(1.0);
        if (!(present > 0.0))
            return float3(1.0);
        counted = true;
        return dielectric(c, section);
    }

    // A dielectric candidate of section material `section` that is present.
    [mutating]
    float3 dielectric(Candidate c, uint section)
    {
        if (MODE == SHADOW_PHOTON)
            return float3(0.0);
        bool thin = may2(MAY2_THIN_DIELECTRICS) && !crosses_interior(scene, section);
        // Solid glass ends a path's shadow ray whatever its surface is.
        if (MODE == SHADOW_PATH && !thin)
        {
            solid_t = c.t;
            return float3(0.0);
        }
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
// SHADOW_CANDIDATES non-opaque candidates in one of its traces keeps the product so far and adds
// 1 to `overflows` (null: not counted).
public float3 visibility<S : ISurfaces, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows)
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
public float3 surface_visibility<S : ISurfaces, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows, out float3 depth)
{
    uint committed;
    float committed_t;
    return surface_visibility<S, M, MODE>(s, v, origin, dir, t_max, flags, context, environment, inside, overflows, depth, committed, committed_t);
}

// As above, and in `committed` and `committed_t` the CLASS_* and distance of an opaque hit the AS
// committed, which made the product 0, or ~0 and +inf for none: the first one found, not the
// nearest. A SHADOW_PATH ray ending at solid glass commits CLASS_DIELECTRIC.
public float3 surface_visibility<S : ISurfaces, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, uint* overflows, out float3 depth, out uint committed, out float committed_t)
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
    uint mask = MODE == SHADOW_UNBENT ? SHADOW_MASK & ~MASK_DIELECTRIC : SHADOW_MASK;
    uint pass_flags = flags;
    for (uint pass = 0; pass < 2; pass++)
    {
        bool more;
        uint c;
        float c_t;
        float3 p = trace_transmittance(s.tlas, ray, mask, pass_flags, h, SHADOW_CANDIDATES, more, c, c_t);
        overflow = overflow || more;
        product *= p;
        if (pass == 0)
        {
            committed = c;
            committed_t = c_t;
        }
        if (MODE != SHADOW_UNBENT || pass == 1)
            break;
        if (c == CLASS_DIELECTRIC)
        {
            mask = SHADOW_MASK;
            product = float3(1.0);
            overflow = false;
            committed = ~0u;
            committed_t = positive_infinity();
        }
        else if (all(product == 0.0))
            break;
        else
            mask = MASK_DIELECTRIC;
        pass_flags = flags | RAY_FLAG_FORCE_NON_OPAQUE;
    }
    if (committed == ~0u && h.solid_t < positive_infinity())
    {
        committed = CLASS_DIELECTRIC;
        committed_t = h.solid_t;
    }
    if (overflow && overflows != nullptr)
        InterlockedAdd(*overflows, 1u);
    float scale;
    float3 straight;
    depth = h.optical_depth(inside, scale, straight);
    return product * straight * scale;
}

// Plane stacks: a directional light reaching a point through mirror-smooth, flat solid
// dielectric faces of a constant BSDF (`stack_material`) that are mutually parallel, crossed in
// transmission. Across parallel planes n sin(theta) holds, so the light's direction in each layer
// follows from that layer's IOR alone and needs no solve. A probe from the point along the
// light's own direction finds the first face, whose normal the stack takes; legs then go face to
// face along the bent directions. The bent light keeps its irradiance on the faces' plane, so at
// the point it comes with the ratio of the faces' cosines outside and at the point and no IOR
// term. BSDF rays and photons that cross a stack hold it to the same probe from their own vertex,
// so each light path has one owner.

// Faces a plane stack crosses at most.
public static const uint STACK_CROSSINGS = 4;

// Faces whose normals' cosine falls below this are not parallel.
public static const float STACK_PARALLEL = 0.999999;

// `path_light` outcomes.
public static const uint STACK_NONE = 0;    // the straight ray's light
public static const uint STACK_LIGHT = 1;   // through a stack, 0 where blocked
public static const uint STACK_BROKEN = 2;  // solid glass that no stack takes on to the light

// A light's way to a point: straight, or a directional light's through a plane stack.
public struct StackLight
{
    // What lies along the ray or the legs and Fresnel at the faces, with `path_light`'s `medium`
    // the linked medium too.
    public float3 transmittance;
    // The interiors' optical depth along it.
    public float3 depth;
    // The light's direction at the point, toward the light.
    public float3 bent;
    // |cos| against the faces outside over |cos| at the point.
    public float jacobian;
    // d(bent)/d(dir) in solid angle: `jacobian` over the squared IOR at the point.
    public float density;
    // The faces' unit shading normal; zero without a stack.
    public float3 normal;
}

// The light of a directional light along `dir` that meets no stack.
public StackLight straight_light(float3 dir)
{
    StackLight l = (StackLight)0;
    l.transmittance = float3(1.0);
    l.bent = dir;
    l.jacobian = 1.0;
    l.density = 1.0;
    return l;
}

// The BSDF a plane stack face of material `m` shows, the same at every point: what
// `StandardSurface` gives a material that no texture, vertex color stream, coat, fuzz, thin film,
// anisotropy or decal varies, solid, smooth below MIRROR_ROUGHNESS and transmitting; with
// `dispersion`, not dispersive. False for any other material.
public bool stack_material(SceneRoot* s, uint m, bool dispersion, out BsdfParams p)
{
    p = (BsdfParams)0;
    if (m == MATERIAL_NONE || s.decal_globals != 0 || s.decal_bin_slots != 0)
        return false;
    MaterialGpu* g = s.materials + m;
    uint varies = MATERIAL_THIN_WALLED | MATERIAL_VERTEX | MATERIAL_MORE_TEXTURES | MATERIAL_ANISOTROPY | MATERIAL_HAS_BACK | MATERIAL_DISPLACED | MATERIAL_MEDIUM_TEXTURED;
    if (g.surface != 0 || (g.flags & varies) != 0 || g.coat_weight != 0.0 || g.fuzz_weight != 0.0 || g.thin_film_weight != 0.0)
        return false;
    for (uint k = 0; k < TEXTURE_SLOTS; k++)
    {
        if (material_bound(g, k))
            return false;
    }
    if (!crosses_interior(s, m) || (dispersion && disperses(s, m)))
        return false;
    p.base_color = g.base_weight * g.base_color;
    p.metalness = g.metalness;
    p.specular_ior = g.specular_ior;
    p.specular_color = surface_specular_color(g, g.specular_color, g.specular_ior);
    p.specular_weight = g.specular_weight;
    p.specular_roughness = g.specular_roughness;
    p.transmission_weight = g.transmission_weight;
    p.transmission_color = g.transmission_depth == 0.0 ? g.transmission_color : float3(1.0);
    if ((g.flags & MATERIAL_BASE_TINT) != 0)
        p.transmission_color *= p.base_color;
    if ((g.flags & MATERIAL_FRESNEL_MIX) != 0)
        p.flags |= BSDF_FRESNEL_MIX;
    if ((g.flags & MATERIAL_METAL_FLAT) != 0)
        p.flags |= BSDF_METAL_FLAT;
    return (1.0 - p.metalness) * p.transmission_weight > 0.0 && bsdf_thin_mirror(p, AMBIENT_IOR);
}

// A plane stack face at a hit: its material and BSDF, the unit shading normal its corners share,
// its unit front-face geometric normal, the hit's render-space point and what its medium reads
// there, and the error a ray origin there clears.
public struct StackFace
{
    public uint material;
    public BsdfParams bsdf;
    public float3 normal;
    public float3 geometric_normal;
    public float3 position;
    public float4 medium_point;
    public float error;
}

// Whether hit `h` is on a plane stack face (`stack_material`, a triangle whose corners share one
// normal, at rest), and the face.
public bool stack_face(SceneRoot* s, Hit h, bool dispersion, out StackFace f)
{
    f = (StackFace)0;
    if (!h.hit || !h.triangle || h.class_id != CLASS_DIELECTRIC || h.cluster_id < 0)
        return false;
    TracedTriangle t;
    if (!traced_triangle(s, h.instance_index, uint(h.cluster_id), h.primitive_index, false, t) || t.deformed)
        return false;
    VertexAttributes a0 = t.corners.attributes[0];
    VertexAttributes a1 = t.corners.attributes[1];
    VertexAttributes a2 = t.corners.attributes[2];
    if (any(a0.normal != a1.normal) || any(a0.normal != a2.normal) || !stack_material(s, t.material, dispersion, f.bsdf))
        return false;
    float3 p0 = t.corners.position[0];
    float3 p1 = t.corners.position[1];
    float3 p2 = t.corners.position[2];
    float3x3 unit = unit_linear(t.placement.linear);
    f.material = t.material;
    f.geometric_normal = unit_direction(cross(mul(unit, p1 - p0), mul(unit, p2 - p0))) * linear_mirror(t.placement.linear);
    f.normal = unit_direction(mul(normal_matrix(t.placement.linear), a0.normal), f.geometric_normal);
    float3 w = float3(1.0 - h.barycentrics.x - h.barycentrics.y, h.barycentrics.x, h.barycentrics.y);
    f.position = place(t.placement, w.x * p0 + w.y * p1 + w.z * p2);
    f.medium_point = float4(w.x * a0.uv[0] + w.y * a1.uv[0] + w.z * a2.uv[0], w.x * a0.uv[1] + w.y * a1.uv[1] + w.z * a2.uv[1]);
    f.error = t.error * t.placement.scale;
    return true;
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
        if (c.cluster_id < 0)
            return false;
        uint entry;
        if (c.class_id == CLASS_DIELECTRIC && !hit_producer(scene, c.instance_index, entry))
        {
            ClusterRef r = scene.clusters[uint(c.cluster_id)];
            if ((load_cluster(r.page, r.local).flags & CLUSTER_SOLID) == 0)
                return false;
        }
        else if (c.class_id != CLASS_OPAQUE)
            return false;
        return resolve_alpha<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
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
    return trace_closest(s.tlas, ray, MASK_OPAQUE | MASK_DIELECTRIC, flags, a);
}

// Whether a plane stack's probe from `origin` along unit `dir` toward a directional light meets a
// stack face first, and its unit shading normal.
public bool stack_probe<S : ISurfaces>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, uint flags, HitContext context, bool dispersion, out float3 normal)
{
    StackFace f;
    bool face = stack_face(s, stack_trace<S>(s, v, origin, dir, positive_infinity(), flags, context), dispersion, f);
    normal = f.normal;
    return face;
}

// What reaches `origin`, offset from surface point `start` (`origin` itself in a medium), inside
// the interiors of `inside`, from `t_max` along unit `dir`, by shadow rays in mode SHADOW_PATH
// traced with ray flags `flags`: their product, with the linked medium's transmittance where
// `medium` and the light is not the `environment`, and the interiors' optical depth apart in
// `light.depth`. With `bend`, for a directional light, a ray that ends at an opaque surface or
// solid glass is probed for a plane stack before it, and through one goes on along the bent legs;
// with `dispersion` dispersive faces end a stack. Returns STACK_LIGHT for light through a stack,
// STACK_BROKEN for a ray ending at solid glass that no stack takes on to the light, its light 0,
// and STACK_NONE for the straight ray's light.
public uint path_light<S : ISurfaces, M : IMedium>(SceneRoot* s, ViewConstants* v, float3 start, float3 origin, float3 dir, float t_max, uint flags, HitContext context, bool environment, Interior inside, bool bend, bool dispersion, bool medium, uint* overflows, out StackLight light)
{
    light = straight_light(dir);
    Interior in = inside;
    float3 o = origin;
    float3 from = start;
    float3 d = dir;
    float3 normal = float3(0.0);
    float3 t = float3(1.0);
    float3 tau = float3(0.0);
    Hit h = (Hit)0;
    StackFace face = (StackFace)0;
    bool on_face = false;
    // 0: the straight ray, 1: the probe, 2: the legs.
    uint phase = 0;
    float probe_max = t_max;
    // The probe's face is the first leg's.
    bool probed = false;
    uint crossed = 0;
    for (uint k = 0; k < STACK_CROSSINGS + 3; k++)
    {
        if (phase > 0 && !probed)
        {
            h = stack_trace<S>(s, v, o, d, phase == 1 ? probe_max : positive_infinity(), flags, context);
            on_face = stack_face(s, h, dispersion, face);
        }
        probed = false;
        if (phase != 1)
        {
            float leg = t_max;
            if (phase == 2)
            {
                leg = positive_infinity();
                if (h.hit)
                {
                    float3 at = o + d * h.t;
                    float magnitude = max(abs(at.x), max(abs(at.y), abs(at.z)));
                    leg = max(0.0, h.t - max(1e-4 * h.t, magnitude * F32_PRECISION));
                }
            }
            float3 leg_depth;
            uint committed;
            float committed_t;
            float3 seg = surface_visibility<S, M, SHADOW_PATH>(s, v, o, d, leg, flags, context, phase == 0 && environment, in, overflows, leg_depth, committed, committed_t);
            if (leg > 0.0 && isfinite(leg))
                leg_depth *= segment_length(from, o, d, leg) / leg;
            if (phase == 0 && (!bend || committed == ~0u))
            {
                if (medium && !environment && any(seg > 0.0))
                    seg *= medium_transmittance<M>(s.medium, o, d, leg);
                light.transmittance = seg;
                light.depth = leg_depth;
                return STACK_NONE;
            }
            if (phase == 0)
            {
                // The ray ends at the first opaque surface or solid glass it found, which need not
                // be the nearest; the probe reaches past it by its rounding.
                phase = 1;
                probe_max = committed_t * (1.0 + 1e-4) + T_MIN_BASE;
                continue;
            }
            t *= seg;
            if (all(t == 0.0))
            {
                light.transmittance = t;
                return STACK_LIGHT;
            }
            if (medium)
                t *= medium_transmittance<M>(s.medium, o, d, leg);
            tau += leg_depth;
        }
        if (phase == 1)
        {
            if (!on_face)
            {
                light.transmittance = float3(0.0);
                return h.hit && h.class_id == CLASS_DIELECTRIC ? STACK_BROKEN : STACK_NONE;
            }
            normal = face.normal;
            light.normal = normal;
            if (dot(face.geometric_normal, dir) > 0.0 && !in.holds(face.material))
                in.nest(face.material, in, face.medium_point);
            float ior = interior_ior(s, in.innermost());
            if (!stack_direction(dir, normal, ior, d))
            {
                light.transmittance = float3(0.0);
                return STACK_LIGHT;
            }
            light.bent = d;
            light.jacobian = abs(dot(dir, normal)) / abs(dot(d, normal));
            light.density = light.jacobian / (ior * ior);
            probed = all(d == dir);
            phase = 2;
            continue;
        }
        if (!h.hit)
        {
            if (interior_ior(s, in.innermost()) != AMBIENT_IOR)
                return STACK_BROKEN;
            light.transmittance = t;
            light.depth = tau;
            return STACK_LIGHT;
        }
        if (h.class_id != CLASS_DIELECTRIC)
        {
            light.transmittance = float3(0.0);
            return STACK_LIGHT;
        }
        uint m = face.material;
        uint exterior = in.outside(m, m);
        if (crossed == STACK_CROSSINGS || !on_face || !(abs(dot(face.normal, normal)) >= STACK_PARALLEL))
            return STACK_BROKEN;
        crossed++;
        bool front = dot(face.geometric_normal, d) < 0.0;
        Interior next = in;
        next.cross(m, front, face.medium_point);
        float3 after;
        if (!stack_direction(dir, normal, interior_ior(s, next.innermost()), after))
        {
            light.transmittance = float3(0.0);
            return STACK_LIGHT;
        }
        // Fresnel takes the direction on the exterior side.
        t *= straight_transmission(straight_surface(face.bsdf, face.geometric_normal, front ? d : after), interior_ior(s, exterior));
        in = next;
        d = after;
        from = face.position;
        o = ray_origin(face.position, face.position, face.geometric_normal, d, ORIGIN_DEFAULT, face.error, 0.0);
    }
    return STACK_BROKEN;
}
