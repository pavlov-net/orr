// Dielectric and presence surfaces. After the opaque dispatch, one camera ray per pixel
// against the dielectric mask, up to the opaque depth, finds the pixel's first layer. A
// presence layer passes the ray on with probability 1 - opacity; a hit layer becomes the
// pixel's first vertex and its path replaces the opaque path's radiance. Fresnel picks
// reflection or refraction per path; refracted rays are traced through masks 0 and 7, and
// further layer hits are path vertices up to `dielectric_depth`. Nested dielectrics keep an
// interior list of material ids, innermost last: the IOR outside an interface is the
// innermost other entry's, so glass in water refracts against water. Segments inside an
// interior take its Beer transmittance; a scattering interior takes one distance sample per
// segment, and a scatter there shades by directional next-event estimation through the
// surface without bending, the Henyey-Greenstein phase function and the world cache for the
// rest. A camera inside a medium starts from `camera_medium`, flipped at each surface
// between the eye and the near plane. The dispatch also writes each pixel's layer guide,
// which the guide resolve blends into ray reconstruction's guides.
module dielectric;

import orr.abi;
import orr.math;
import orr.color;
import orr.random;
import orr.sampling;
import orr.trace;
import orr.coherence;
import orr.openpbr;
import hit;
import visibility;
import bsdf;
import media;
import environment;
import lights;
import light_sampling;
import world_cache;
import path;
import guides;

// Interior list entries a path keeps.
public static const uint INTERIOR_ENTRIES = 4;

// The IOR outside every interior.
public static const float AMBIENT_IOR = 1.0;

// Separates the dielectric dispatch's samples from the opaque path's of the same pixel.
public static const uint DIELECTRIC_STREAM = 0x85EBCA6B;

// `PathRoot.dielectric_flags`.
public static const uint DIELECTRIC_REFERENCE = 1 << 0;   // the accumulation reference: scatter on every interior segment, no cache
public static const uint DIELECTRIC_DEBUG_LAYER = 1 << 1; // the radiance shows the first layer's shares


// The dielectric materials whose interiors hold a point, innermost last.
public struct Interior
{
    public uint count;
    public uint entries[INTERIOR_ENTRIES];

    public __init()
    {
        count = 0;
        for (uint i = 0; i < INTERIOR_ENTRIES; i++)
            entries[i] = MATERIAL_NONE;
    }

    public uint innermost()
    {
        return count > 0 ? entries[count - 1] : MATERIAL_NONE;
    }

    // The innermost entry other than `a` and `b`, MATERIAL_NONE for none.
    public uint outside(uint a, uint b)
    {
        for (uint i = count; i > 0; i--)
        {
            uint e = entries[i - 1];
            if (e != a && e != b)
                return e;
        }
        return MATERIAL_NONE;
    }

    // Crosses the surface of material `m`: into it from its front, out of it from its back.
    // A full list drops its outermost entry to take a new one.
    [mutating]
    public void cross(uint m, bool front)
    {
        uint at = INTERIOR_ENTRIES;
        for (uint i = 0; i < count; i++)
        {
            if (entries[i] == m)
                at = i;
        }
        if (front)
        {
            if (at != INTERIOR_ENTRIES)
                return;
            if (count == INTERIOR_ENTRIES)
                at = 0;
            else
            {
                entries[count] = m;
                count++;
                return;
            }
        }
        else if (at == INTERIOR_ENTRIES)
            return;
        for (uint i = at; i + 1 < count; i++)
            entries[i] = entries[i + 1];
        if (front)
            entries[count - 1] = m;
        else
        {
            count--;
            entries[count] = MATERIAL_NONE;
        }
    }
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

// exp(-sigma t) per channel; 1 where sigma is 0, whatever `t`.
public float3 beer(float3 sigma, float t)
{
    return select(sigma > 0.0, exp(-sigma * t), float3(1.0));
}

// Applies an interior segment of length `t` (+inf: unbounded) to `throughput`: its Beer
// transmittance, or with `scatter` one distance sample from the extinction, picking a channel
// uniformly and weighing by the channels' mean density. True with `at` set where the sample
// scatters before `t`. Draws two sampler dimensions.
public bool interior_segment(InteriorMedium m, float t, bool scatter, inout Sampler rng, inout float3 throughput, out float at)
{
    at = 0.0;
    float u_channel = rng.next();
    float u_distance = rng.next();
    if (!scatter || !any(m.sigma_s > 0.0))
    {
        throughput *= beer(m.sigma_t, t);
        return false;
    }
    float sigma = m.sigma_t[min(uint(u_channel * 3.0), 2u)];
    float sampled = sigma > 0.0 ? -log(1.0 - u_distance) / sigma : positive_infinity();
    if (sampled < t)
    {
        float3 tr = beer(m.sigma_t, sampled);
        float pdf = dot(m.sigma_t * tr, float3(1.0 / 3.0));
        throughput *= pdf > 0.0 ? m.sigma_s * tr / pdf : float3(0.0);
        at = sampled;
        return true;
    }
    float3 tr = beer(m.sigma_t, t);
    float p = dot(select(m.sigma_t > 0.0, tr, float3(1.0)), float3(1.0 / 3.0));
    throughput *= p > 0.0 ? tr / p : float3(0.0);
    return false;
}

// Non-opaque candidates of rays that may meet layers: presence layers commit by `presence`,
// the rest by the alpha test.
public static const uint PRESENCE_NONE = 0;      // never: rays that look for dielectrics only
public static const uint PRESENCE_ANY = 1;       // wherever its opacity is above 0
public static const uint PRESENCE_DRAWN = 2;     // where a hashed draw from `seed` falls below its opacity

public struct LayerAccept<S : ISurface> : IAnyHit
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 dir;
    public HitContext context;
    public uint seed;
    public uint presence;           // PRESENCE_*

    public bool accept(Candidate c)
    {
        if (c.cluster_id < 0)
            return false;
        if (c.custom_index == CLASS_PRESENCE)
        {
            if (presence == PRESENCE_NONE)
                return false;
            float coverage = resolve_coverage<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
            if (presence == PRESENCE_ANY)
                return coverage > 0.0;
            uint h = pcg(seed ^ pcg(c.instance_index ^ pcg(uint(c.cluster_id) ^ pcg(c.primitive_index))));
            return unit_float(h) < coverage;
        }
        return resolve_alpha<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
    }
}

public bool is_layer(Hit h)
{
    return h.hit && (h.custom_index == CLASS_DIELECTRIC || h.custom_index == CLASS_PRESENCE);
}

// What leaving `medium` along a shadow ray lets through: its Beer transmittance to the first
// dielectric surface the ray crosses, times 1 - Fresnel there into IOR `exit_ior`, without
// bending. With no surface before `t_max`, the Beer transmittance over it.
public float3 interior_crossing<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, uint medium, float exit_ior, HitContext context)
{
    InteriorMedium m = interior_medium(s, medium);
    LayerAccept<S> a = { s, v, dir, context, 0, PRESENCE_NONE };
    float3 positions[3];
    Hit h = trace_closest(s.tlas, path_ray(origin, dir, t_max), MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    if (!h.hit || h.cluster_id < 0)
        return beer(m.sigma_t, t_max);
    float3 n;
    uint material;
    float c = hit_geometry(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, n, material) ? abs(dot(n, dir)) : 1.0;
    float f = openpbr_fresnel(exit_ior / interior_ior(s, medium), c);
    return beer(m.sigma_t, h.t) * (1.0 - f);
}

// Shadow rays of a vertex with an interior on either side of its outward geometric normal
// (zero inside a medium, where `above` holds the medium): rays leaving through an interior
// cross its surface by `interior_crossing` and take weight 1, since BSDF rays reach the
// light bent.
public struct LayerShadow<S : ISurface, M : IMedium> : IShadow
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 normal;
    public uint below;              // against `normal`, MATERIAL_NONE outside every interior
    public uint above;
    public float below_exit;        // the IOR past `below`'s surface
    public float above_exit;

    public float3 transmittance(float3 origin, float3 dir, float t_max, HitContext context, bool environment, out bool weigh)
    {
        SceneShadow<S, M> plain = { scene, view };
        float3 vis = plain.transmittance(origin, dir, t_max, context, environment, weigh);
        bool down = dot(dir, normal) < 0.0;
        uint medium = down ? below : above;
        if (medium == MATERIAL_NONE)
            return vis;
        weigh = false;
        if (all(vis == 0.0))
            return vis;
        return vis * interior_crossing<S>(scene, view, origin, dir, t_max, medium, down ? below_exit : above_exit, context);
    }
}

// What next-event estimation weighs light by at a dielectric path's vertex: its BSDF, or at
// a scatter in an interior the phase function.
public struct VertexScatter : IScatter
{
    public Bsdf b;
    public bool phase;
    public float3 travel;           // a scatter's: the path's direction into it
    public float anisotropy;

    public float3 eval(float3 wi)
    {
        return phase ? float3(hg_phase(anisotropy, dot(travel, wi))) : b.eval(wi);
    }

    public float pdf(float3 wi)
    {
        return phase ? hg_phase(anisotropy, dot(travel, wi)) : b.pdf(wi);
    }
}

// `LayerVertex.kind`.
public static const uint VERTEX_OPAQUE = 0;
public static const uint VERTEX_LAYER = 1;
public static const uint VERTEX_SCATTER = 2;

// A vertex of a dielectric path.
public struct LayerVertex
{
    public PathVertex x;            // a layer's normals keep the material's orientation
    public uint kind;               // VERTEX_*
    public bool front;              // a layer reached from its front side
    public float anisotropy;        // a scatter's phase function
}

// The vertex of a layer hit by a ray along `dir`: the shading normal leaned toward the ray's
// origin on the side the ray arrived from, both normals kept on the material's front side so
// the BSDF sees which side the path is on.
public LayerVertex layer_vertex(ViewConstants* v, SurfaceHit sh, float3 dir)
{
    LayerVertex l;
    l.kind = VERTEX_LAYER;
    l.front = dot(sh.surface.geometric_normal, dir) < 0.0;
    l.anisotropy = 0.0;
    PathVertex x;
    x.surface = sh.surface;
    x.view = -dir;
    float side = l.front ? 1.0 : -1.0;
    x.surface.normal = side * lean_normal(side * sh.surface.normal, x.view);
    x.origin_error = sh.geometric_error;
    x.plane_point = sh.surface.position - dir * sh.depth_offset;
    x.camera_distance = length(sh.surface.position - v.eye_render.xyz);
    l.x = x;
    return l;
}

public LayerVertex opaque_vertex(PathVertex x)
{
    LayerVertex l;
    l.x = x;
    l.kind = VERTEX_OPAQUE;
    l.front = true;
    l.anisotropy = 0.0;
    return l;
}

// A scatter at `p` in interior `medium`, reached along `dir`.
public LayerVertex scatter_vertex(ViewConstants* v, float3 p, float3 dir, uint medium, float anisotropy)
{
    LayerVertex l;
    l.kind = VERTEX_SCATTER;
    l.front = true;
    l.anisotropy = anisotropy;
    PathVertex x = (PathVertex)0;
    x.surface.position = p;
    x.surface.material = medium;
    x.view = -dir;
    x.plane_point = p;
    x.camera_distance = length(p - v.eye_render.xyz);
    l.x = x;
    return l;
}

public bool thin_walled(GSurface s)
{
    return (s.bsdf.flags & BSDF_THIN_WALLED) != 0;
}

// The BSDF of layer or opaque vertex `l` at path depth `depth`, against the interior list the
// path arrived with: a layer's exterior IOR is the innermost other entry's, an opaque
// vertex's the innermost entry's, so a pool floor meets water.
public Bsdf prepare_vertex(SceneRoot* s, LayerVertex l, Interior interior, uint depth)
{
    BsdfParams p = depth > 1 ? bsdf_collapse(l.x.surface.bsdf) : l.x.surface.bsdf;
    uint m = l.x.surface.material;
    uint outside = l.kind == VERTEX_LAYER ? interior.outside(m, m) : interior.innermost();
    return bsdf_prepare(p, l.x.surface.normal, l.x.view, interior_ior(s, outside));
}

// The shadow of vertex `l` against the interior list it was reached with.
public LayerShadow<S, M> vertex_shadow<S : ISurface, M : IMedium>(SceneRoot* s, ViewConstants* v, LayerVertex l, Interior interior)
{
    LayerShadow<S, M> r;
    r.scene = s;
    r.view = v;
    r.normal = float3(0.0);
    if (l.kind == VERTEX_LAYER)
    {
        uint m = l.x.surface.material;
        uint out = interior.outside(m, m);
        r.normal = l.x.surface.geometric_normal;
        r.above = out;
        r.above_exit = interior_ior(s, interior.outside(out, m));
        r.below = thin_walled(l.x.surface) ? out : m;
        r.below_exit = thin_walled(l.x.surface) ? r.above_exit : interior_ior(s, out);
    }
    else
    {
        uint m = interior.innermost();
        r.above = m;
        r.below = m;
        r.above_exit = interior_ior(s, interior.outside(m, m));
        r.below_exit = r.above_exit;
    }
    return r;
}

// The incident radiance a scatter at `p` takes from the world cache: the mean of its cells
// facing along and against `dir`. Draws eight sampler dimensions.
public float3 interior_ambient(PathRoot r, float3 p, float3 dir, inout Sampler rng)
{
    float3 sum = float3(0.0);
    for (uint k = 0; k < 2; k++)
    {
        CacheCell cell = cache_cell(r.cache, r.scene, p, k == 0 ? dir : -dir, rng.next4().xyz, true);
        float3 cached;
        if (cache_lookup(r.cache, cell.key, cached))
            sum += cached;
    }
    return 0.5 * sum;
}

// What leaves a dielectric path's vertex.
struct LayerRay
{
    float3 origin;
    float3 dir;
    float pdf;                      // 0 for an ideal-specular direction
    float run;                      // the vertex's next-event probability
    float3 normal;                  // the vertex's shading normal, zero in a medium
    bool diffuse;
    bool chain;                     // left a layer or a scatter
    bool scatter;                   // may scatter in an interior
    bool local;                     // left the first vertex, which reads its tile's local light sampler
}

// Iterations of a dielectric path: a vertex and its ray each.
public uint layer_iterations(PathRoot r)
{
    return r.max_bounces + 2 + r.dielectric_depth;
}

// The path from `first`, its camera segment applied in `throughput`, with `interior` the list
// the path reached it with; `layers` counts layer vertices, `first` included. Returns
// unexposed radiance; `rays` counts every ray traced.
public float3 trace_layers<C : ICoherence, S : ISurface, M : IMedium>(PathRoot r, uint2 pixel, LayerVertex first, Interior arrived, float3 throughput, uint seed, inout Sampler rng, inout uint rays, inout uint layers)
{
    Interior interior = arrived;
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    bool count_nee = r.mode != PATH_BSDF_ONLY;
    bool count_bsdf = r.mode != PATH_NEE_ONLY;
    bool one_technique = r.mode == PATH_NEE_ONLY || r.mode == PATH_BSDF_ONLY;
    bool reference = (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
    bool cache = (r.cache_flags & CACHE_TERMINATE) != 0 && !one_technique && r.mode != PATH_DIRECT;

    LayerVertex l = first;
    uint depth = 0;
    Bsdf b = prepare_vertex(s, l, interior, depth);
    RayCone cone = camera_cone(v);
    cone.width = cone_width(cone, l.x.camera_distance);
    float3 radiance = float3(0.0);
    bool alive = l.kind == VERTEX_SCATTER || (l.x.surface.flags & GBUFFER_UNLIT) == 0;
    // Opaque vertices and scatters so far, which the bounce settings count.
    uint opaque = l.kind == VERTEX_LAYER ? 0 : 1;
    // Set once a non-delta ray enters an interior, whose vertex estimated the direct light
    // there through the surface: the path then adds no direct light until its next opaque
    // vertex or scatter, and its layers estimate none.
    bool skip_direct = false;
    // A scatter that took the cache's light: it estimates its direct light and ends.
    bool last = false;
    bool first_opaque = l.kind != VERTEX_OPAQUE;
    LayerRay ray = (LayerRay)0;

    for (uint i = 0; i < layer_iterations(r); i++)
    {
        bool counted = r.mode == PATH_DIRECT ? depth == 0 : r.mode == PATH_INDIRECT ? depth > 0 : true;
        if (alive && l.kind != VERTEX_LAYER && opaque > r.max_bounces + 1)
            alive = false;
        Hit h = (Hit)0;
        float3 positions[3] = { float3(0.0), float3(0.0), float3(0.0) };
        if (alive)
        {
            bool medium_vertex = l.kind == VERTEX_SCATTER;
            LocalLights* local = nullptr;
            if (depth == 0)
                local = pixel_local_lights(r, pixel);
            float run = medium_vertex ? 1.0 : nee_probability(set, l.x.surface);
            float u_run = rng.next();
            uint candidates = depth == 0 ? r.primary_candidates : r.bounce_candidates;
            bool layer_skips = skip_direct && l.kind == VERTEX_LAYER;
            if (count_nee && candidates > 0 && u_run < run && !layer_skips)
            {
                LayerShadow<S, M> shadow = vertex_shadow<S, M>(s, v, l, interior);
                // Layers and scatters light from both sides.
                PathVertex at = l.x;
                if (l.kind != VERTEX_OPAQUE)
                    at.surface.normal = float3(0.0);
                uint light;
                VertexScatter f = { b, medium_vertex, -l.x.view, l.anisotropy };
                float3 e = shade_vertex_nee(r, at, f, shadow, run, pixel, local, depth, candidates, one_technique, rng, rays, light);
                if (counted)
                    radiance += throughput * e;
            }
            if (last)
                alive = false;
        }
        if (alive)
        {
            float3 wi;
            float pdf;
            bool delta = false;
            bool diffuse = false;
            float spread;
            bool crossed = false;
            float4 u = rng.next4();
            float u_roulette = rng.next();
            if (l.kind == VERTEX_SCATTER)
            {
                wi = sample_hg(l.anisotropy, -l.x.view, u.xy);
                pdf = hg_phase(l.anisotropy, dot(-l.x.view, wi));
                spread = 1.0;
            }
            else
            {
                BsdfSample bs = b.sample(u.xyz);
                alive = bs.pdf > 0.0 && (bs.lobe & LOBE_VOLUME) == 0;
                wi = bs.wi;
                pdf = bs.pdf;
                throughput *= bs.weight;
                delta = (bs.lobe & LOBE_SPECULAR) != 0;
                diffuse = (bs.lobe & LOBE_DIFFUSE) != 0;
                float roughness = l.x.surface.bsdf.specular_roughness;
                spread = delta ? 0.0 : diffuse ? 1.0 : 2.0 * roughness * roughness;
                crossed = l.kind == VERTEX_LAYER && (bs.lobe & LOBE_TRANSMISSION) != 0 && !thin_walled(l.x.surface)
                    && (s.materials[l.x.surface.material].flags & MATERIAL_DIELECTRIC) != 0;
            }
            if (alive && l.kind != VERTEX_LAYER && opaque > r.roulette_depth)
            {
                float keep = saturate(max(throughput.x, max(throughput.y, throughput.z)));
                alive = u_roulette < keep;
                throughput /= max(keep, 1e-6);
            }
            alive = alive && any(throughput > 0.0) && all(isfinite(throughput));
            if (alive)
            {
                if (crossed)
                    interior.cross(l.x.surface.material, l.front);
                ray.origin = vertex_origin(l.x, wi);
                ray.dir = wi;
                ray.pdf = delta ? 0.0 : pdf;
                ray.run = count_nee ? (l.kind == VERTEX_SCATTER ? 1.0 : nee_probability(set, l.x.surface)) : 0.0;
                ray.normal = l.kind == VERTEX_OPAQUE ? l.x.surface.normal : float3(0.0);
                ray.diffuse = diffuse;
                ray.chain = l.kind != VERTEX_OPAQUE;
                ray.scatter = reference || depth == 0;
                ray.local = depth == 0;
                if (!delta && count_nee && !one_technique && interior.innermost() != MATERIAL_NONE)
                    skip_direct = true;
                cone.spread += spread;
                uint mask = ray.chain || interior.count > 0 ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
                HitContext context = hit_context(HIT_SECONDARY, depth + 1, cone, 0.0);
                LayerAccept<S> a = { s, v, ray.dir, context, pcg(seed ^ (depth + 1)), PRESENCE_DRAWN };
                rays++;
                h = trace_closest(s.tlas, path_ray(ray.origin, ray.dir, positive_infinity()), mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
            }
        }
        C.point(alive ? path_key(s, h) : coherence_key(false, 0, 0), KEY_BITS);
        if (!alive)
            continue;
        depth++;

        // The segment: an interior's Beer or scatter, else the linked medium.
        float t = h.hit ? h.t : positive_infinity();
        uint medium = interior.innermost();
        if (medium != MATERIAL_NONE)
        {
            InteriorMedium im = interior_medium(s, medium);
            float scattered;
            if (interior_segment(im, t, ray.scatter, rng, throughput, scattered))
            {
                l = scatter_vertex(v, ray.origin + ray.dir * scattered, ray.dir, medium, im.anisotropy);
                cone = cone_after_hit(cone, scattered, 0.0);
                opaque++;
                skip_direct = false;
                if (cache && !reference)
                {
                    if (counted)
                        radiance += throughput * interior_ambient(r, l.x.surface.position, ray.dir, rng);
                    last = true;
                }
                continue;
            }
        }
        else if (h.hit && (ray.chain || t >= INSCATTER_SEGMENT))
        {
            float3 through;
            float3 e = segment_inscatter<M>(set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through);
            if (counted)
                radiance += throughput * e;
            throughput *= through;
        }
        else
        {
            rng.next();
            if (h.hit)
                throughput *= medium_transmittance<M>(s.medium, ray.origin, ray.dir, t);
        }
        if (!any(throughput > 0.0) || !all(isfinite(throughput)))
        {
            alive = false;
            continue;
        }

        bool weigh = !one_technique && ray.pdf > 0.0;
        LocalLights* ray_local = nullptr;
        if (ray.local)
            ray_local = pixel_local_lights(r, pixel);
        if (!h.hit)
        {
            if (count_bsdf && counted && !skip_direct)
            {
                float3 env = environment_map(set.environment, ray.dir);
                float w = weigh ? power_heuristic(ray.pdf, ray.run * nee_environment_pdf(set, ray_local, ray.normal, ray.dir)) : 1.0;
                float3 e = env * w;
                for (uint k = 0; k < set.directional_count; k++)
                {
                    uint id = set.directional_ids[k];
                    float3 d = directional_disc<M>(set, id, ray.origin, ray.dir);
                    if (any(d > 0.0))
                        e += d * (weigh ? power_heuristic(ray.pdf, ray.run * nee_directional_pdf(set, ray.normal, id, ray.dir)) : 1.0);
                }
                radiance += throughput * e;
            }
            alive = false;
            continue;
        }

        HitContext context = hit_context(HIT_SECONDARY, depth, cone, t);
        SurfaceHit sh = resolve_hit<S>(s, v, ray.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
        if (!sh.valid)
        {
            alive = false;
            continue;
        }
        cone = cone_after_hit(cone, t, 0.0);
        bool front = dot(sh.surface.geometric_normal, ray.dir) < 0.0;
        bool layer = is_layer(h);
        if (layer)
        {
            layers++;
            if (layers > r.dielectric_depth)
            {
                alive = false;
                continue;
            }
            l = layer_vertex(v, sh, ray.dir);
        }
        else
        {
            PathVertex x = hit_vertex(v, sh, ray.dir);
            l = opaque_vertex(x);
            opaque++;
        }
        b = prepare_vertex(s, l, interior, depth);
        // Emission seen from the front. Cooked emitters are reached by BSDF rays alone; the
        // light set samples producer entries, so their emission is weighed against it.
        uint entry;
        bool producer = hit_producer(s, h.instance_index, entry);
        if (count_bsdf && counted && front && !(skip_direct && producer))
        {
            float w = 1.0;
            if (weigh && producer)
            {
                float lum = luminance_rec2020(material_emission(s.materials + sh.surface.material));
                w = power_heuristic(ray.pdf, ray.run * nee_emissive_pdf(set, ray_local, ray.normal, entry, lum, t, dot(sh.surface.geometric_normal, ray.dir)));
            }
            radiance += throughput * b.emission() * w;
        }
        if (layer)
            continue;
        skip_direct = false;
        alive = (l.x.surface.flags & GBUFFER_UNLIT) == 0 && r.mode != PATH_DIRECT;
        // Cache termination outside every interior, at the first opaque hit or where the
        // opaque path's rules allow it.
        if (alive && cache && front && interior.count == 0)
        {
            CacheCell cell = cache_cell(r.cache, s, l.x.surface.position, l.x.surface.geometric_normal, rng.next4().xyz, true);
            float3 cached;
            bool may = first_opaque || ray.diffuse || cone.width > cell.size || opaque == r.max_bounces + 1;
            if (may && t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
            {
                if (counted)
                    radiance += throughput * cache_albedo(b.albedo(), l.x.surface.bsdf) * cached;
                alive = false;
            }
        }
        first_opaque = false;
    }
    return radiance;
}

// The surfaces a ray from `eye` along `dir` crosses before `t_near`, flipped into `interior`.
public void eye_crossings<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 eye, float3 dir, float t_near, inout Interior interior)
{
    float3 origin = eye;
    float left = t_near;
    RayCone cone = camera_cone(v);
    for (uint k = 0; k < INTERIOR_ENTRIES && left > 0.0; k++)
    {
        HitContext context = hit_context(HIT_PRIMARY, 0, cone, 0.0);
        LayerAccept<S> a = { s, v, dir, context, 0, PRESENCE_NONE };
        float3 positions[3];
        Hit h = trace_closest(s.tlas, path_ray(origin, dir, left), MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
        if (!h.hit || h.cluster_id < 0)
            return;
        float3 n;
        uint m;
        if (!hit_geometry(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, n, m))
            return;
        if ((s.materials[m].flags & (MATERIAL_DIELECTRIC | MATERIAL_THIN_WALLED)) == MATERIAL_DIELECTRIC)
            interior.cross(m, dot(n, dir) < 0.0);
        float step = h.t + 1e-4 * (1.0 + h.t);
        origin += dir * step;
        left -= step;
    }
}

// Reverse-Z device depth of render-space point `p`, as the jittered projection places it.
public float device_depth(ViewConstants* v, float3 p)
{
    float4 clip = mul(v.clip_from_view_jittered, float4(view_from_relative(v, p - v.eye_render.xyz), 1.0));
    return clip.z / clip.w;
}

// The refracted direction of a ray along `dir` through layer `l`, without roughness: straight
// through thin walls; false on total internal reflection.
public bool layer_refraction(SceneRoot* s, LayerVertex l, Interior interior, float3 dir, out float3 out_dir)
{
    out_dir = dir;
    if (thin_walled(l.x.surface))
        return true;
    uint m = l.x.surface.material;
    float outer = interior_ior(s, interior.outside(m, m));
    float inner = interior_ior(s, m);
    float3 n = l.front ? l.x.surface.normal : -l.x.surface.normal;
    float eta = l.front ? outer / inner : inner / outer;
    out_dir = refract(dir, n, eta);
    return any(out_dir != 0.0);
}

// The guides of a pixel whose first layer `l` (prepared as `b`) was hit by the camera ray
// along `dir` with `interior` the eye's list. Presence layers give their own guides at their
// opacity `coverage`. A dielectric blends its reflection, the surface itself, with what its
// transmission reaches along a roughness-free refraction chain, by throughput; the dominant
// one gives depth and motion: the hit's own where reflection carries more, the virtual point
// of the chain otherwise. Specular motion follows the mirror reflection's virtual point.
public LayerGuide layer_guides<S : ISurface>(PathRoot r, LayerVertex l, Bsdf b, Hit h, float3 dir, Interior interior, float coverage)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LobeAlbedo a = b.albedo();
    LayerGuide g = (LayerGuide)0;
    Guides own = surface_guides(l.x.surface, a);
    float3 lean = l.front ? l.x.surface.normal : -l.x.surface.normal;
    float3 previous;
    if (!hit_previous_position(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, previous))
        previous = l.x.surface.position;
    float own_depth = device_depth(v, l.x.surface.position);
    float2 own_motion = moved_motion(v, l.x.surface.position - v.eye_render.xyz, previous - v.eye_render.xyz, 1.0);
    RayCone cone = camera_cone(v);
    cone.width = cone_width(cone, l.x.camera_distance);

    // The mirror reflection's virtual point.
    float3 reflected = reflect(dir, lean);
    LayerAccept<S> ra = { s, v, reflected, hit_context(HIT_SECONDARY, 1, cone, 0.0), 0, PRESENCE_ANY };
    Hit rh = trace_closest(s.tlas, path_ray(vertex_origin(l.x, reflected), reflected, positive_infinity()), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, ra);
    float3 camera = l.x.surface.position - v.eye_render.xyz;
    g.specular_motion = rh.hit ? reprojected_motion(v, dir * (length(camera) + rh.t), 1.0) : reprojected_motion(v, dir, 0.0);

    if (h.custom_index == CLASS_PRESENCE)
    {
        g.diffuse = own.diffuse;
        g.specular = own.specular;
        g.normal = lean;
        g.roughness = own.roughness;
        g.weight = coverage;
        g.dominant = coverage >= 0.5 ? 1 : 0;
        g.depth = own_depth;
        g.motion = own_motion;
        return g;
    }

    // The transmission chain.
    float3 through = a.transmission;
    Guides behind = sky_guides();
    float travelled = positive_infinity();
    float3 d = dir;
    Interior list = interior;
    LayerVertex x = l;
    float3 bent = d;
    bool open = any(through > 0.0) && layer_refraction(s, x, list, d, bent);
    d = bent;
    if (!open)
        through = float3(0.0);
    else if (!thin_walled(x.x.surface))
        list.cross(x.x.surface.material, x.front);
    float length_so_far = 0.0;
    for (uint k = 0; open && k < r.dielectric_depth; k++)
    {
        LayerAccept<S> ta = { s, v, d, hit_context(HIT_SECONDARY, k + 1, cone, 0.0), 0, PRESENCE_ANY };
        float3 positions[3];
        Hit th = trace_closest(s.tlas, path_ray(vertex_origin(x.x, d), d, positive_infinity()), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, ta, positions);
        through *= beer(interior_medium(s, list.innermost()).sigma_t, th.hit ? th.t : positive_infinity());
        if (!th.hit || th.cluster_id < 0)
        {
            behind = sky_guides();
            break;
        }
        SurfaceHit sh = resolve_hit<S>(s, v, d, th.instance_index, uint(th.cluster_id), th.primitive_index, th.barycentrics, positions, false, hit_context(HIT_SECONDARY, k + 1, cone, th.t));
        if (!sh.valid)
            break;
        length_so_far += th.t;
        if (!is_layer(th))
        {
            LayerVertex o = opaque_vertex(hit_vertex(v, sh, d));
            behind = surface_guides(o.x.surface, prepare_vertex(s, o, list, 2).albedo());
            travelled = length_so_far;
            break;
        }
        x = layer_vertex(v, sh, d);
        Bsdf xb = prepare_vertex(s, x, list, k + 1);
        through *= xb.albedo().transmission;
        bent = d;
        bool bends = any(through > 0.0) && layer_refraction(s, x, list, d, bent);
        d = bent;
        if (!bends)
        {
            through = float3(0.0);
            break;
        }
        if (!thin_walled(x.x.surface))
            list.cross(x.x.surface.material, x.front);
    }
    float wr = luminance_rec2020(a.specular + a.diffuse);
    float wt = luminance_rec2020(through);
    g.diffuse = own.diffuse + through * behind.diffuse;
    g.specular = own.specular + through * behind.specular;
    float3 n = wr * lean + wt * behind.normal;
    g.normal = dot(n, n) > 0.0 ? normalize(n) : lean;
    g.roughness = wr + wt > 0.0 ? (wr * own.roughness + wt * behind.roughness) / (wr + wt) : own.roughness;
    g.weight = 1.0;
    g.dominant = 1;
    if (wr >= wt)
    {
        g.depth = own_depth;
        g.motion = own_motion;
    }
    else if (isfinite(travelled))
    {
        float3 q = dir * (length(camera) + travelled);
        g.depth = device_depth(v, q + v.eye_render.xyz);
        g.motion = reprojected_motion(v, q, 1.0);
    }
    else
    {
        g.depth = 0.0;
        g.motion = reprojected_motion(v, dir, 0.0);
    }
    return g;
}

// The dielectric dispatch's body for `pixel`.
public void dielectric_body<C : ICoherence, S : ISurface, E : IEnvironment, M : IMedium>(PathRoot r, uint2 pixel)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ DIELECTRIC_STREAM);
    uint seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ DIELECTRIC_STREAM)).x;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
    float3 dir = render_direction(v, view_direction(v, uv));
    float3 eye = v.eye_render.xyz;
    float3 near = render_from_view(v, view_position(v, uv, 1.0));
    float t_near = length(near - eye);

    Interior interior = Interior();
    if (r.camera_medium != MATERIAL_NONE)
        interior.cross(r.camera_medium, true);
    eye_crossings<S>(s, v, eye, dir, t_near, interior);

    PathVertex xg;
    bool surface = gbuffer_vertex(s, v, r.gbuffer, pixel, xg);
    float t_opaque = surface ? max(length(xg.surface.position - near) * (1.0 - 1e-5), 0.0) : positive_infinity();
    RayCone cone = camera_cone(v);
    HitContext primary = hit_context(HIT_PRIMARY, 0, cone, 0.0);
    // The first layer `first`, then, past a presence layer the pixel's draw passes, the next;
    // `h` is the layer the pixel shades. One trace call site serves both rays, which keeps the
    // kernel small.
    LayerAccept<S> a = { s, v, dir, primary, seed, PRESENCE_ANY };
    Hit first = (Hit)0;
    Hit h = (Hit)0;
    float3 first_positions[3];
    float3 positions[3];
    float coverage = 1.0;
    float start = 0.0;
    for (uint k = 0; k < 2; k++)
    {
        h = trace_closest(s.tlas, path_ray(near + dir * start, dir, max(t_opaque - start, 0.0)), MASK_DIELECTRIC, RAY_FLAG_NONE, a, positions);
        h.t += start;
        if (k == 1)
            break;
        first = h;
        first_positions = positions;
        float u = rng.next();
        if (!h.hit || h.cluster_id < 0 || h.custom_index != CLASS_PRESENCE)
            break;
        coverage = resolve_coverage<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, primary);
        if (u < coverage)
            break;
        start = h.t + 1e-4 * (1.0 + h.t);
        a.presence = PRESENCE_DRAWN;
    }
    C.point(coherence_key(h.hit, 0, 0), KEY_BITS);
    bool layered = first.hit && first.cluster_id >= 0;
    bool hit_layer = h.hit && h.cluster_id >= 0;
    bool passed = a.presence != PRESENCE_ANY;

    // The first layer's guide, and the first vertex where it is the layer the pixel shades.
    LayerGuide guide = (LayerGuide)0;
    SurfaceHit first_hit = (SurfaceHit)0;
    LayerVertex l = opaque_vertex(xg);
    float3 emission = float3(0.0);
    float wr = 0.0;
    float wt = 0.0;
    if (layered)
    {
        first_hit = resolve_hit<S>(s, v, dir, first.instance_index, uint(first.cluster_id), first.primitive_index, first.barycentrics, first_positions, false, hit_context(HIT_PRIMARY, 0, cone, first.t));
        if (first_hit.valid)
        {
            l = layer_vertex(v, first_hit, dir);
            Bsdf b1 = prepare_vertex(s, l, interior, 0);
            guide = layer_guides<S>(r, l, b1, first, dir, interior, coverage);
            emission = b1.emission();
            wr = luminance_rec2020(b1.albedo().specular + b1.albedo().diffuse);
            wt = luminance_rec2020(b1.albedo().transmission);
        }
    }
    r.layers[index] = guide;
    if (!hit_layer && interior.count == 0)
    {
        for (uint i = 0; i < layer_iterations(r); i++)
            C.point(coherence_key(false, 0, 0), KEY_BITS);
        return;
    }

    // The camera segment, then the first vertex.
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    float3 exposed = float3(0.0);
    uint rays = 0;
    uint layers = 0;
    float t_first = hit_layer ? h.t : t_opaque;
    uint medium = interior.innermost();
    InteriorMedium im = interior_medium(s, medium);
    float scattered;
    bool path = true;
    bool scatters = medium != MATERIAL_NONE && interior_segment(im, t_first, true, rng, throughput, scattered);
    if (scatters)
        l = scatter_vertex(v, near + dir * scattered, dir, medium, im.anisotropy);
    else if (hit_layer)
    {
        SurfaceHit sh = first_hit;
        if (passed)
        {
            sh = resolve_hit<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, hit_context(HIT_PRIMARY, 0, cone, h.t));
            l = layer_vertex(v, sh, dir);
            emission = prepare_vertex(s, l, interior, 0).emission();
        }
        path = sh.valid;
        layers = 1;
        if (path && r.mode != PATH_INDIRECT && l.front)
            radiance += throughput * emission;
    }
    else if (surface)
    {
        l = opaque_vertex(xg);
        // The G-buffer's emission is already pre-exposed.
        if (r.mode != PATH_INDIRECT)
            exposed += throughput * prepare_vertex(s, l, interior, 0).emission();
    }
    else
    {
        path = false;
        if (r.mode != PATH_INDIRECT)
            radiance += throughput * sky_radiance<E, M>(set, eye, dir);
    }
    if (path)
        radiance += trace_layers<C, S, M>(r, pixel, l, interior, throughput, seed, rng, rays, layers);
    else
    {
        for (uint i = 0; i < layer_iterations(r); i++)
            C.point(coherence_key(false, 0, 0), KEY_BITS);
    }

    float3 total = exposed + pre_exposure * radiance;
    float4 out = float4(apply_hdr_ceiling(total), 1.0);
    if (r.mode == PATH_RAYS)
        out = float4(float(rays), all(isfinite(total)) ? 0.0 : 1.0, 0.0, 1.0);
    else if ((r.dielectric_flags & DIELECTRIC_DEBUG_LAYER) != 0)
    {
        float sum = max(wr + wt, 1e-6);
        out = float4(wr / sum, wt / sum, float(layers) / float(max(r.dielectric_depth, 1u)), 1.0);
    }
    if (r.mode == PATH_CACHE || r.mode == PATH_CACHE_LOD || r.mode == PATH_ORIGIN_BIAS)
        return;
    r.radiance.get<RWTexture2D<float4>>()[pixel] = out;
}
