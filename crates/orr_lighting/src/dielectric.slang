// Dielectric and presence surfaces. After the opaque dispatch, one camera ray per pixel of
// the tiles the layers rastered into, and of the tiles beside them, against the dielectric
// mask, up to the opaque depth, finds the pixel's first layer. A presence layer passes the
// ray on with probability 1 - opacity; a hit layer becomes the pixel's first vertex and its
// path replaces the opaque path's radiance. Fresnel picks
// reflection or refraction per path; refracted rays are traced through masks 0 and 7, and
// further layer hits are path vertices up to `dielectric_depth`. Nested dielectrics keep an
// interior list of material ids, innermost last: the IOR outside an interface is the
// innermost other entry's, so glass in water refracts against water. Segments inside an
// interior take its Beer transmittance; a scattering interior takes one distance sample per
// segment. Where the segment scatters less than once on average, a scatter there shades by
// directional next-event estimation through the surface without bending, the
// Henyey-Greenstein phase function and the world cache for the rest. Elsewhere it starts a
// walk, as subsurface materials do, that reflects back or refracts out of the surface by
// Fresnel on the triangle's plane; the way out gathers the light it reaches, and next-event
// estimation there the lights, taking the radiance inside as uniform. A camera inside a
// medium starts from `camera_medium`, flipped at each surface between the eye and the near
// plane; media apply gives its camera segment from the froxels, and only without them, as
// under the reference, a distance sample here does. The dispatch also
// writes each pixel's layer guide, which the guide resolve blends into ray reconstruction's
// guides, and, for the frosted-glass term, what the background behind rough transmission looks
// like and how far its lobes blur it.
module dielectric;

import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.coherence;
import orr_shaders.openpbr;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_scene.lights;
import orr_lighting.light_sampling;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.shadow;
import orr_lighting.interiors;
import orr_lighting.subsurface;
import orr_lighting.guides;
import orr_lighting.froxels;
import orr_lighting.spectrum;

// Separates the dielectric dispatch's samples from the opaque path's of the same pixel.
public static const uint DIELECTRIC_STREAM = 0x85EBCA6B;

// `PathRoot.dielectric_flags`.
public static const uint DIELECTRIC_REFERENCE = 1 << 0;   // the accumulation reference: scatter on every interior segment, no cache
public static const uint DIELECTRIC_DEBUG_LAYER = 1 << 1; // the radiance shows the first layer's shares
public static const uint DIELECTRIC_DISPERSION = 1 << 2;  // dispersive materials split paths by wavelength
public static const uint DIELECTRIC_FROST = 1 << 3;       // layer guides keep `background` and `frost`

// The stream of a dispersive path's hero wavelength.
public static const uint WAVELENGTH_STREAM = 0xD3A2646C;

// `LayerTilesRoot.flags`.
public static const uint LAYER_TILES_EVERYWHERE = 1 << 0; // the eye lies in a medium: trace every tile
public static const uint LAYER_TILES_FROST = 1 << 1;      // keep `background` on tiles left untraced

// What picks the tiles the dielectric dispatch traces.
public struct LayerTilesRoot
{
    public GBufferGpu* gbuffer;
    public LayerGuide* layers;          // the dispatch's layer memory (`traced_tiles`)
    public StorageImageIndex radiance;  // RGBA16F at the G-buffer's extent, read with LAYER_TILES_FROST
    public uint flags;                  // LAYER_TILES_*
    public uint _pad0;
    public uint _pad1;
}

groupshared bool tile_traced;

// One workgroup of LAYER_TILE^2 threads per tile: a tile is traced when it or one beside it
// holds a layer, or everywhere when a layer may lie before the near plane, the eye lies in a
// medium or the G-buffer has no layer tiles. Traced tiles join the list. With
// LAYER_TILES_FROST, untraced tiles keep their pixels' `background`.
public void layer_tiles_body(LayerTilesRoot r, uint2 tile, uint thread)
{
    uint2 extent = r.gbuffer.extent;
    uint columns = layer_tile_columns(extent);
    uint count = layer_tile_count(extent);
    uint rows = count / columns;
    uint index = tile.y * columns + tile.x;
    if (thread == 0)
    {
        uint* marks = r.gbuffer.layer_tiles;
        bool traced = marks == nullptr || (r.flags & LAYER_TILES_EVERYWHERE) != 0 || marks[count] != 0;
        for (int dy = -1; dy <= 1 && !traced; dy++)
        {
            for (int dx = -1; dx <= 1 && !traced; dx++)
            {
                int2 t = int2(tile) + int2(dx, dy);
                if (all(t >= 0) && t.x < int(columns) && t.y < int(rows))
                    traced = marks[uint(t.y) * columns + uint(t.x)] != 0;
            }
        }
        traced_tiles(r.layers, extent)[index] = traced ? 1 : 0;
        if (traced)
        {
            uint at;
            InterlockedAdd(layer_tile_args(r.layers, extent)[0], 1, at);
            layer_tile_list(r.layers, extent)[at] = index;
        }
        tile_traced = traced;
    }
    GroupMemoryBarrierWithGroupSync();
    uint2 pixel = tile * LAYER_TILE + uint2(thread % LAYER_TILE, thread / LAYER_TILE);
    if (tile_traced || (r.flags & LAYER_TILES_FROST) == 0 || any(pixel >= extent))
        return;
    float3 opaque = r.radiance.get<RWTexture2D<float4>>()[pixel].rgb;
    r.layers[pixel.y * extent.x + pixel.x].background = guide_luminance(opaque);
}

// Sizes the dispatch to the listed tiles: two 8 x 4 workgroups per tile for its compute entry,
// LAYER_TILE^2 rays wide and a row per tile for its raygen entry.
public void layer_args_body(LayerTilesRoot r)
{
    uint* args = layer_tile_args(r.layers, r.gbuffer.extent);
    uint n = args[0];
    write_grid(args + 1, 2 * n);
    args[4] = LAYER_TILE * LAYER_TILE;
    args[5] = n;
    args[6] = 1;
}

// The pixel of invocation `lane` (row-major in its tile) of listed tile `entry`.
public uint2 layer_tile_pixel(PathRoot r, uint entry, uint2 lane)
{
    uint2 extent = r.gbuffer.extent;
    uint columns = layer_tile_columns(extent);
    uint index = layer_tile_list(r.layers, extent)[entry];
    return uint2(index % columns, index / columns) * LAYER_TILE + lane;
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

public bool is_layer(Hit h)
{
    return h.hit && (h.custom_index == CLASS_DIELECTRIC || h.custom_index == CLASS_PRESENCE);
}

// Shadow rays of a vertex with an interior on either side of its outward geometric normal
// (zero inside a medium, where `above` holds the medium). The shadow product takes the
// interior's Beer transmittance and its surface's Fresnel transmission, unbent; rays leaving
// through an interior take weight 1, since BSDF rays reach the light bent.
public struct LayerShadow<S : ISurface, M : IMedium> : IShadow
{
    public SceneShadow<S, M> plain;
    public float3 normal;
    public uint below;              // against `normal`, MATERIAL_NONE outside every interior
    public uint above;

    public float3 transmittance(float3 origin, float3 dir, float t_max, HitContext context, bool environment, out bool weigh)
    {
        float3 vis = plain.transmittance(origin, dir, t_max, context, environment, weigh);
        if ((dot(dir, normal) < 0.0 ? below : above) != MATERIAL_NONE)
            weigh = false;
        return vis;
    }
}

// Light from outside as an interior's walk gathers it where it leaves through a surface of
// `eta` times the outside's IOR. The radiance inside is taken as uniform, so what leaves along
// the walk's way out arrives from `wi` by Lambert reflection on outward normal `n` times the
// Fresnel transmission from `wi`, over its mean, which keeps the energy of uniform light.
public struct InteriorExitScatter : IScatter
{
    public float3 n;
    public float eta;
    public float scale;             // 1 / the transmission's cosine-weighted mean

    public float3 eval(float3 wi)
    {
        float c = dot(n, wi);
        if (!(c > 0.0))
            return float3(0.0);
        return float3((1.0 - openpbr_fresnel(eta, c)) * scale * c * INV_PI);
    }

    public float pdf(float3 wi)
    {
        return cosine_hemisphere_pdf(dot(n, wi));
    }
}

public InteriorExitScatter interior_exit(float3 n, float eta)
{
    // The mean, evenly in the cosine squared.
    float sum = 0.0;
    for (uint i = 0; i < 16; i++)
        sum += 1.0 - openpbr_fresnel(eta, sqrt((float(i) + 0.5) / 16.0));
    InteriorExitScatter f = { n, eta, 16.0 / sum };
    return f;
}

// What next-event estimation weighs light by at a dielectric path's vertex: its BSDF, or at
// a scatter in an interior the phase function.
public struct VertexScatter : IScatter
{
    public Bsdf b;
    public bool phase;
    public float3 travel;           // a scatter's: the path's direction into it
    public float anisotropy;
    public bool exit;               // a walk's exit, which scatters as `walk`
    public WalkExitScatter walk;
    public bool leave;              // a walk leaving an interior, which scatters as `way_out`
    public InteriorExitScatter way_out;

    public float3 eval(float3 wi)
    {
        if (phase)
            return float3(hg_phase(anisotropy, dot(travel, wi)));
        if (leave)
            return way_out.eval(wi);
        if (exit)
            return walk.eval(wi);
        return b.eval(wi);
    }

    public float pdf(float3 wi)
    {
        if (phase)
            return hg_phase(anisotropy, dot(travel, wi));
        if (leave)
            return way_out.pdf(wi);
        if (exit)
            return walk.pdf(wi);
        return b.pdf(wi);
    }
}

// `LayerVertex.kind`.
public static const uint VERTEX_OPAQUE = 0;
public static const uint VERTEX_LAYER = 1;
public static const uint VERTEX_SCATTER = 2;
public static const uint VERTEX_LEAVE = 3;      // a walk leaving an interior, along a way it chose

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

// A walk leaving through the surface at `p` with outward normal `n`.
public LayerVertex leave_vertex(ViewConstants* v, float3 p, float3 n)
{
    LayerVertex l;
    l.kind = VERTEX_LEAVE;
    l.front = true;
    l.anisotropy = 0.0;
    PathVertex x = (PathVertex)0;
    x.surface.position = p;
    x.surface.normal = n;
    x.surface.geometric_normal = n;
    x.surface.material = MATERIAL_NONE;
    x.view = n;
    x.plane_point = p;
    x.camera_distance = length(p - v.eye_render.xyz);
    l.x = x;
    return l;
}

public bool thin_walled(GSurface s)
{
    return (s.bsdf.flags & BSDF_THIN_WALLED) != 0;
}

// The parameters of layer or opaque vertex `l` at path depth `depth`.
public BsdfParams vertex_params(LayerVertex l, uint depth)
{
    return depth > 1 ? bsdf_collapse(l.x.surface.bsdf) : l.x.surface.bsdf;
}

// Whether material `m`'s IOR varies with wavelength.
public bool disperses(SceneRoot* s, uint m)
{
    if (m == MATERIAL_NONE)
        return false;
    MaterialGpu* g = s.materials + m;
    return g.transmission_dispersion_scale > 0.0 && g.transmission_dispersion_abbe_number > 0.0;
}

// `ior`, the IOR of material `m` at 587.6 nm, at wavelength `nm`.
public float dispersed_material_ior(SceneRoot* s, uint m, float ior, float nm)
{
    if (m == MATERIAL_NONE)
        return ior;
    MaterialGpu* g = s.materials + m;
    return dispersed_ior(ior, g.transmission_dispersion_scale, g.transmission_dispersion_abbe_number, nm);
}

// Whether layer vertex `l`, reached with `interior`, has a dispersive side.
public bool layer_disperses(SceneRoot* s, LayerVertex l, Interior interior)
{
    uint m = l.x.surface.material;
    return l.kind == VERTEX_LAYER && (disperses(s, m) || disperses(s, interior.outside(m, m)));
}

// The BSDF of layer or opaque vertex `l` at path depth `depth`, against the interior list the
// path arrived with: a layer's exterior IOR is the innermost other entry's, an opaque
// vertex's the innermost entry's, so a pool floor meets water. Unless the vertex `walks`, its
// subsurface share is diffuse reflection. A layer takes both IORs at wavelength `nm` where it
// is above 0, else at 587.6 nm.
public Bsdf prepare_vertex(SceneRoot* s, LayerVertex l, Interior interior, uint depth, bool walks = true, float nm = 0.0)
{
    BsdfParams p = walk_params(vertex_params(l, depth), walks);
    uint m = l.x.surface.material;
    bool layer = l.kind == VERTEX_LAYER;
    uint outside = layer ? interior.outside(m, m) : interior.innermost();
    float exterior = interior_ior(s, outside);
    if (layer && nm > 0.0)
    {
        p.specular_ior = dispersed_material_ior(s, m, p.specular_ior, nm);
        exterior = dispersed_material_ior(s, outside, exterior, nm);
    }
    return bsdf_prepare(p, l.x.surface.normal, l.x.view, exterior);
}

// A path's wavelengths once it meets a dispersive layer; before that its colors are plain
// RGB. From there the path follows the hero's refraction, and every companion's estimate is
// weighed by its own BSDF and density at each direction the hero drew at a dispersive layer,
// over the mean density of all four (the balance heuristic over wavelengths). The path's
// throughput stays the hero's; `transfer` takes its contributions to Rec.2020.
public struct Spectral
{
    public bool on;
    public Wavelengths w;
    public float4 value;            // per wavelength: its BSDF values over the hero's, multiplied along the path
    public float4 density;          // its densities over the hero's
    public float3x3 transfer;

    public __init()
    {
        on = false;
        w = (Wavelengths)0;
        value = float4(1.0);
        density = float4(1.0);
        transfer = float3x3(0.0);
    }

    // Draws the wavelengths from `u`.
    [mutating]
    public void start(float u)
    {
        on = true;
        w = sample_wavelengths(u);
        update();
    }

    // The hero's wavelength, 0 before the path disperses.
    public float hero()
    {
        return on ? w.nm.x : 0.0;
    }

    // Whether a companion still carries an estimate.
    public bool companions()
    {
        return on && any(density.yzw > 0.0);
    }

    // Weighs companion `k` by its BSDF x cos `f` and density `pdf` toward the direction the
    // hero drew with BSDF x cos `hero_f` and density `hero_pdf`.
    [mutating]
    public void weigh(uint k, float3 f, float pdf, float3 hero_f, float hero_pdf)
    {
        float3 basis = spectrum_basis(w.nm[k]);
        float own = dot(f, basis);
        float theirs = dot(hero_f, basis);
        value[k] *= theirs > 0.0 ? own / theirs : 0.0;
        density[k] *= hero_pdf > 0.0 ? pdf / hero_pdf : 0.0;
        if (!(value[k] > 0.0) || !(density[k] > 0.0) || !isfinite(value[k] * density[k]))
        {
            value[k] = 0.0;
            density[k] = 0.0;
        }
    }

    [mutating]
    public void update()
    {
        float mean = dot(density, float4(1.0 / float(WAVELENGTHS)));
        transfer = spectral_transfer(w, mean > 0.0 ? value / mean : float4(0.0));
    }

    // Contribution `c` in Rec.2020.
    public float3 color(float3 c)
    {
        return on ? mul(transfer, c) : c;
    }
}

// The hero wavelength's sample of pixel `pixel`, a dimension of its own.
public float wavelength_sample(PathRoot r, uint2 pixel)
{
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ DIELECTRIC_STREAM ^ WAVELENGTH_STREAM);
    return rng.next();
}

// The shadow of vertex `l` against the interior list it was reached with.
public LayerShadow<S, M> vertex_shadow<S : ISurface, M : IMedium>(PathRoot r, LayerVertex l, Interior interior)
{
    LayerShadow<S, M> shadow;
    shadow.plain = scene_shadow<S, M>(r);
    shadow.normal = float3(0.0);
    if (l.kind == VERTEX_LAYER)
    {
        uint m = l.x.surface.material;
        uint out = interior.outside(m, m);
        shadow.normal = l.x.surface.geometric_normal;
        shadow.above = out;
        shadow.below = thin_walled(l.x.surface) ? out : m;
    }
    else
    {
        shadow.above = interior.innermost();
        shadow.below = shadow.above;
    }
    return shadow;
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

// Steps of an interior's walk through the scene: every face of opaque geometry and layers.
// Faces of its own material seen from behind reflect it back or let it out by Fresnel on their
// plane, the walk's `ior` times the outside's; any other surface ends the walk there.
public struct InteriorWalk<S : ISurface> : IWalkTrace
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public HitContext context;
    public uint seed;
    public uint material;
    public float ior;

    public Hit trace(float3 origin, float3 dir, float t_max, out float3 positions[3])
    {
        LayerAccept<S> a = { scene, view, dir, context, seed, PRESENCE_DRAWN };
        return trace_closest(scene.tlas, path_ray(origin, dir, t_max), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    }

    public WalkMeet meet(Hit h, float3 origin, float3 dir, float3 positions[3], inout Sampler rng)
    {
        float3 normal;
        uint m;
        if (h.cluster_id < 0 || !hit_geometry(scene, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, normal, m))
            return WalkMeet(WALK_BOUNDARY);
        float c = dot(normal, dir);
        if (m != material || !(c > 0.0))
            return WalkMeet(WALK_BOUNDARY);
        float3 p = origin + dir * h.t;
        float3 way_out = refract(dir, -normal, ior);
        if (rng.next() < openpbr_fresnel(1.0 / ior, c) || all(way_out == 0.0))
        {
            WalkMeet back = WalkMeet(WALK_REFLECT);
            back.dir = reflect(dir, normal);
            back.origin = ray_origin(p, p, normal, back.dir, ORIGIN_DEFAULT, 0.0, length(p - view.eye_render.xyz));
            return back;
        }
        WalkMeet leave = WalkMeet(WALK_LEAVE);
        leave.dir = normalize(way_out);
        leave.origin = p;
        leave.normal = normal;
        return leave;
    }
}

// The walk of interior medium `im` from a scatter at `from` reached along `dir`, its surface
// `ior` times the outside's.
public Walk walk_interior(float3 from, float3 dir, InteriorMedium im, float ior, inout Sampler rng)
{
    SubsurfaceVolume volume;
    volume.extinction = im.sigma_t;
    volume.albedo = im.albedo();
    volume.anisotropy = im.anisotropy;
    Sampler walk = walk_sampler(rng);
    float2 u = walk.next2();
    return walk_start(from, sample_hg(im.anisotropy, dir, u), volume, ior, walk);
}

// What traces a walk in interior `medium`, its surface `ior` times the outside's.
public InteriorWalk<S> interior_tracer<S : ISurface>(PathRoot r, uint medium, float ior, uint seed, uint depth)
{
    RayCone cone = { 0.0, 0.0 };
    InteriorWalk<S> tracer = { r.scene, r.view, hit_context(HIT_SECONDARY, depth + 1, cone, 0.0), seed, medium, ior };
    return tracer;
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

// Iterations of a dielectric path: a vertex and its ray each, and one subsurface walk. An
// interior's walk takes none of its own: it runs where a segment scatters, and its way out is
// a vertex that counts as a bounce.
public uint layer_iterations(PathRoot r)
{
    return r.max_bounces + 3 + r.dielectric_depth;
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
    RayCone cone = camera_cone(v);
    cone.width = cone_width(cone, l.x.camera_distance);
    bool walked = false;
    bool disperse = (r.dielectric_flags & DIELECTRIC_DISPERSION) != 0;
    Spectral spectral = Spectral();
    // Whether `l` is a layer with a dispersive side.
    bool dispersive = disperse && layer_disperses(s, l, interior);
    if (dispersive)
        spectral.start(wavelength_sample(r, pixel));
    Bsdf b = prepare_vertex(s, l, interior, depth, l.kind == VERTEX_OPAQUE && may_walk(r, l.x.surface, depth, walked, cone.width), spectral.hero());
    float3 radiance = float3(0.0);
    bool alive = l.kind == VERTEX_SCATTER || (l.x.surface.flags & GBUFFER_UNLIT) == 0;
    // Opaque vertices and scatters so far, which the bounce settings count.
    uint opaque = l.kind == VERTEX_LAYER ? 0 : 1;
    // Set once a non-delta ray enters an interior, whose vertex estimated the direct light
    // there through the surface: the path then adds no direct light until its next opaque
    // vertex or scatter, and its layers estimate none.
    bool skip_direct = false;
    // Set when a walk leaves an interior: while the path skips direct light, what it reaches
    // still adds the environment, which the walk's exit leaves to its way out.
    bool environment_after_walk = false;
    // The interior the path last entered from outside and where, where a walk in that interior
    // leaves at its cap.
    uint entered = MATERIAL_NONE;
    float3 entered_at = float3(0.0);
    float3 entered_normal = float3(0.0);
    float3 leave_dir = float3(0.0);
    InteriorExitScatter leave_scatter = (InteriorExitScatter)0;
    // A scatter that took the cache's light: it estimates its direct light and ends.
    bool last = false;
    bool first_opaque = l.kind != VERTEX_OPAQUE;
    // Set while `l` is a walk's exit, which scatters as `exit_scatter`.
    bool exit = false;
    WalkExitScatter exit_scatter = (WalkExitScatter)0;
    LayerRay ray = (LayerRay)0;
    uint points = walk_points(r);

    // A walk takes an iteration of its own, and its exit vertex the next one at the same
    // depth.
    for (uint i = 0; i < layer_iterations(r); i++)
    {
        bool counted = r.mode == PATH_DIRECT ? depth == 0 : r.mode == PATH_INDIRECT ? depth > 0
            : r.mode == PATH_SUBSURFACE ? walked : true;
        if (alive && l.kind != VERTEX_LAYER && opaque > r.max_bounces + 1)
            alive = false;
        Hit h = (Hit)0;
        float3 positions[3] = { float3(0.0), float3(0.0), float3(0.0) };
        // The particles in front of the ray's hit, premultiplied.
        float4 particles = float4(0.0);
        // A dispersive layer with live companions adds no direct light: its BSDF differs by
        // wavelength, and its rays reach lights at full weight.
        bool split = dispersive && spectral.companions();
        if (alive)
        {
            bool medium_vertex = l.kind == VERTEX_SCATTER;
            LocalLights* local = nullptr;
            if (depth == 0)
                local = pixel_local_lights(r, pixel);
            bool leaving = l.kind == VERTEX_LEAVE;
            float run = medium_vertex || exit || leaving ? 1.0 : nee_probability(set, l.x.surface);
            float u_run = rng.next();
            uint candidates = depth == 0 ? r.primary_candidates : r.bounce_candidates;
            bool layer_skips = (skip_direct && l.kind == VERTEX_LAYER) || split;
            if (count_nee && candidates > 0 && u_run < run && !layer_skips)
            {
                LayerShadow<S, M> shadow = vertex_shadow<S, M>(r, l, interior);
                // Layers and scatters light from both sides.
                PathVertex at = l.x;
                if (l.kind == VERTEX_LAYER || medium_vertex)
                    at.surface.normal = float3(0.0);
                uint light;
                VertexScatter f = { b, medium_vertex, -l.x.view, l.anisotropy, exit, exit_scatter, leaving, leave_scatter };
                // The rays of a walk's way out reach no light this estimate is weighed against.
                float3 e = shade_vertex_nee(r, at, f, shadow, run, pixel, local, depth, candidates, one_technique || leaving, rng, rays, light, leaving && count_bsdf);
                if (counted)
                    radiance += spectral.color(throughput * e);
            }
            if (last)
                alive = false;
        }
        bool walking = false;
        Walk walk = Walk();
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
            uint component = 0;
            float3 hero_f = float3(0.0);
            if (l.kind == VERTEX_SCATTER)
            {
                wi = sample_hg(l.anisotropy, -l.x.view, u.xy);
                pdf = hg_phase(l.anisotropy, dot(-l.x.view, wi));
                spread = 1.0;
            }
            else if (l.kind == VERTEX_LEAVE)
            {
                wi = leave_dir;
                pdf = 1.0;
                delta = true;
                diffuse = true;
                spread = 1.0;
            }
            else
            {
                BsdfSample bs;
                if (exit)
                    bs = walk_exit_sample(exit_scatter, rng);
                else
                    bs = b.sample(u.xyz);
                alive = bs.pdf > 0.0;
                wi = bs.wi;
                pdf = bs.pdf;
                component = bs.component;
                hero_f = bs.weight * bs.pdf;
                throughput *= bs.weight;
                delta = (bs.lobe & LOBE_SPECULAR) != 0;
                diffuse = (bs.lobe & LOBE_DIFFUSE) != 0;
                float roughness = l.x.surface.bsdf.specular_roughness;
                spread = delta ? 0.0 : diffuse ? 1.0 : 2.0 * roughness * roughness;
                // The walk takes the place of an opaque vertex's ray.
                walking = alive && (bs.lobe & LOBE_VOLUME) != 0;
                crossed = l.kind == VERTEX_LAYER && (bs.lobe & LOBE_TRANSMISSION) != 0 && !thin_walled(l.x.surface)
                    && (s.materials[l.x.surface.material].flags & MATERIAL_DIELECTRIC) != 0;
            }
            if (alive && !walking && l.kind != VERTEX_LAYER && opaque > r.roulette_depth)
            {
                float keep = saturate(max(throughput.x, max(throughput.y, throughput.z)));
                alive = u_roulette < keep;
                throughput /= max(keep, 1e-6);
            }
            alive = alive && any(throughput > 0.0) && all(isfinite(throughput));
            if (alive && split)
            {
                for (uint k = 1; k < WAVELENGTHS; k++)
                {
                    if (!(spectral.density[k] > 0.0))
                        continue;
                    Bsdf bk = prepare_vertex(s, l, interior, depth, false, spectral.w.nm[k]);
                    float pk;
                    float3 fk;
                    if (delta)
                        fk = bk.delta_value(component, wi, pk);
                    else
                    {
                        fk = bk.eval(wi);
                        pk = bk.pdf(wi);
                    }
                    spectral.weigh(k, fk, pk, hero_f, pdf);
                }
                spectral.update();
            }
            if (alive && walking)
                walk = walk_from(l.x, vertex_params(l, depth), wi, rng);
            else if (alive)
            {
                if (crossed)
                {
                    interior.cross(l.x.surface.material, l.front);
                    if (l.front)
                    {
                        entered = l.x.surface.material;
                        entered_at = l.x.surface.position;
                        entered_normal = l.x.surface.geometric_normal;
                    }
                }
                ray.origin = vertex_origin(l.x, wi);
                ray.dir = wi;
                ray.pdf = delta ? 0.0 : pdf;
                ray.run = count_nee && !split && !environment_after_walk ? (l.kind == VERTEX_SCATTER || exit ? 1.0 : nee_probability(set, l.x.surface)) : 0.0;
                ray.normal = l.kind == VERTEX_OPAQUE ? l.x.surface.normal : float3(0.0);
                ray.diffuse = diffuse;
                ray.chain = l.kind != VERTEX_OPAQUE;
                ray.scatter = reference || depth == 0;
                ray.local = depth == 0;
                if (!delta && count_nee && !one_technique && !split && interior.innermost() != MATERIAL_NONE)
                    skip_direct = true;
                cone.spread += spread;
                uint mask = ray.chain || interior.count > 0 ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
                // Rays leaving a layer or a scatter, and the first vertex's, see particles.
                if (ray.chain || depth == 0)
                    mask |= MASK_PARTICLE;
                HitContext context = hit_context(HIT_SECONDARY, depth + 1, cone, 0.0);
                LayerAccept<S> a = { s, v, ray.dir, context, pcg(seed ^ (depth + 1)), PRESENCE_DRAWN };
                LocalLights* ray_local = nullptr;
                if (ray.local)
                    ray_local = pixel_local_lights(r, pixel);
                ParticleEmission<LayerAccept<S> > pe = { s, set, ray_local, ray.dir, ray.normal, one_technique ? 0.0 : ray.pdf, ray.run, a };
                rays++;
                h = trace_resolve_translucent(s.tlas, path_ray(ray.origin, ray.dir, positive_infinity()), mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, pe, PARTICLE_CANDIDATES, particles, positions);
            }
        }
        // A segment through an interior scatters or walks before the coherence points, so its
        // walk has them as a vertex's does.
        float t = 0.0;
        uint medium = MATERIAL_NONE;
        bool scatters = false;
        bool interior_walk = false;
        float3 scatter_at = float3(0.0);
        if (alive && !walking)
        {
            depth++;
            // While the path skips direct light, particles add no emission, as producer hits do.
            if (count_bsdf && counted && !skip_direct)
                radiance += spectral.color(throughput * particles.rgb);
            throughput *= 1.0 - particles.a;
            t = h.hit ? h.t : positive_infinity();
            medium = interior.innermost();
            if (medium != MATERIAL_NONE)
            {
                InteriorMedium im = interior_medium(s, medium);
                // A walk takes a bounce; a ray leaving an opaque vertex takes transmittance only.
                bool walks = ray.chain && (reference || !walked) && opaque < r.max_bounces
                    && max(im.sigma_s.x, max(im.sigma_s.y, im.sigma_s.z)) * t >= 1.0;
                float scattered;
                scatters = interior_segment(im, t, ray.scatter || walks, rng, throughput, scattered);
                if (scatters)
                {
                    scatter_at = ray.origin + ray.dir * scattered;
                    cone = cone_after_hit(cone, scattered, 0.0);
                    opaque++;
                    skip_direct = false;
                    environment_after_walk = false;
                    if (walks)
                    {
                        interior_walk = true;
                        walked = true;
                        float ior = interior_ior(s, medium) / interior_ior(s, interior.outside(medium, medium));
                        walk = walk_interior(scatter_at, ray.dir, im, ior, rng);
                    }
                }
            }
        }
        // The iteration's main point follows the walk's last part.
        for (uint p = 0; p <= points; p++)
        {
            if (walk.kind == WALK_RUNNING)
            {
                if (walking)
                    walk_part(r, walk, walk_tracer<S>(r, l.x.surface.instance, depth), rays);
                else
                    walk_part(r, walk, interior_tracer<S>(r, medium, walk.ior, pcg(seed ^ (depth + 1)), depth), rays);
            }
            if (p < points)
                C.point(uint(walk.kind == WALK_RUNNING), 1);
        }
        if (alive && (walking || interior_walk))
        {
            throughput *= walk.throughput;
            alive = walk.kind != WALK_NONE && (walking || walk.kind != WALK_CAP || entered == medium);
            if (walking || walk.kind == WALK_BOUNDARY)
            {
                h = walk.hit;
                positions = walk.positions;
            }
        }
        C.point(alive ? path_key(s, h) : coherence_key(false, 0, 0), KEY_BITS);
        if (!alive)
            continue;
        exit = false;
        if (walking)
        {
            walked = true;
            BsdfParams p = vertex_params(l, depth);
            if (walk.kind == WALK_BOUNDARY)
            {
                HitContext context = hit_context(HIT_SECONDARY, depth + 1, cone, h.t);
                SurfaceHit sh = resolve_hit<S>(s, v, walk.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
                alive = sh.valid;
                if (alive)
                    l = opaque_vertex(exit_vertex(v, sh, walk.dir));
                if (alive && sh.surface.bsdf.subsurface_weight > 0.0)
                    p = sh.surface.bsdf;
            }
            else
                l.x.view = l.x.surface.normal;
            exit = true;
            exit_scatter = walk_exit(l.x, p, rng);
            continue;
        }

        if (medium != MATERIAL_NONE)
        {
            if (scatters && !interior_walk)
            {
                l = scatter_vertex(v, scatter_at, ray.dir, medium, interior_medium(s, medium).anisotropy);
                if (cache && !reference)
                {
                    if (counted)
                        radiance += spectral.color(throughput * interior_ambient(r, l.x.surface.position, ray.dir, rng));
                    last = true;
                }
                continue;
            }
            if (interior_walk)
            {
                if (walk.kind != WALK_BOUNDARY)
                {
                    float3 from = walk.origin;
                    float3 n = walk.normal;
                    leave_dir = walk.dir;
                    if (walk.kind == WALK_CAP)
                    {
                        from = entered_at;
                        n = entered_normal;
                        float3 tangent, bitangent;
                        orthonormal_basis(n, tangent, bitangent);
                        leave_dir = normalize(from_local(sample_cosine_hemisphere(rng.next2()), tangent, bitangent, n));
                    }
                    interior.cross(medium, false);
                    l = leave_vertex(v, from, n);
                    leave_scatter = interior_exit(n, walk.ior);
                    dispersive = false;
                    skip_direct = count_nee && !one_technique;
                    environment_after_walk = skip_direct;
                    continue;
                }
                // The walk met another surface inside: it shades as this ray's hit.
                t = h.t;
                ray.dir = walk.dir;
                ray.pdf = 0.0;
                ray.diffuse = true;
                ray.local = false;
            }
        }
        else if (h.hit && (ray.chain || t >= INSCATTER_SEGMENT))
        {
            float3 through;
            float3 e = froxel_segment_inscatter<M>(r.froxels, v, set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through);
            if (counted)
                radiance += spectral.color(throughput * e);
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
            if (count_bsdf && counted && (!skip_direct || environment_after_walk))
            {
                float3 env = environment_map(set.environment, ray.dir);
                float w = weigh ? power_heuristic(ray.pdf, ray.run * nee_environment_pdf(set, ray_local, ray.normal, ray.dir)) : 1.0;
                float3 e = env * w;
                for (uint k = 0; k < set.directional_count && !skip_direct; k++)
                {
                    uint id = set.directional_ids[k];
                    float3 d = directional_disc<M>(set, id, ray.origin, ray.dir);
                    if (any(d > 0.0))
                        e += d * (weigh ? power_heuristic(ray.pdf, ray.run * nee_directional_pdf(set, ray.normal, id, ray.dir)) : 1.0);
                }
                radiance += spectral.color(throughput * e);
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
        dispersive = disperse && layer_disperses(s, l, interior);
        if (dispersive && !spectral.on)
            spectral.start(wavelength_sample(r, pixel));
        b = prepare_vertex(s, l, interior, depth, !layer && may_walk(r, l.x.surface, depth, walked, cone.width), spectral.hero());
        // Emission seen from the front. Cooked emitters are reached by BSDF rays alone; the
        // light set samples producer entries, so their emission is weighed against it.
        uint entry;
        bool producer = hit_producer(s, h.instance_index, entry);
        if (count_bsdf && counted && front && !(skip_direct && producer))
        {
            float w = 1.0;
            if (weigh && producer)
            {
                float lum = luminance_rec2020(material_light_emission(s.materials + sh.surface.material));
                w = power_heuristic(ray.pdf, ray.run * nee_emissive_pdf(set, ray_local, ray.normal, entry, lum, t, dot(sh.surface.geometric_normal, ray.dir)));
            }
            radiance += spectral.color(throughput * b.emission() * w);
        }
        if (layer)
            continue;
        skip_direct = false;
        environment_after_walk = false;
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
                    radiance += spectral.color(throughput * cache_albedo(b.albedo(), l.x.surface.bsdf) * cached);
                alive = false;
            }
        }
        first_opaque = false;
    }
    return radiance;
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
// Where transmission dominates, `frost` gets the blur the chain's rough interfaces give what
// it reaches: each widens the ray by its roughness squared per m beyond it (half the cone
// spread a rough lobe gives a path), and the widths add in quadrature.
// `front` is the layer's overlay front word where the dominant layer lies behind it, else 0.
public LayerGuide layer_guides<S : ISurface>(PathRoot r, LayerVertex l, Bsdf b, Hit h, float3 dir, Interior interior, float coverage, out uint64_t front)
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
        front = g.dominant == 0 ? overlay_front_word(own_depth, own_motion, coverage) : 0;
        return g;
    }

    // The transmission chain.
    float3 through = a.transmission;
    Guides behind = sky_guides();
    float traveled = positive_infinity();
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
    // What the chain's interiors scatter, which shows in their single-scattering color.
    float3 scattered = float3(0.0);
    // Over the rough interfaces i at chain length s_i with widening w_i: sums of w^2, s w^2
    // and s^2 w^2, which give sum (D - s_i)^2 w_i^2 at chain length D.
    float3 blur = float3(0.0);
    if (open)
        blur += rough_widening(x, 0.0);
    for (uint k = 0; open && k < r.dielectric_depth; k++)
    {
        LayerAccept<S> ta = { s, v, d, hit_context(HIT_SECONDARY, k + 1, cone, 0.0), 0, PRESENCE_ANY };
        float3 positions[3];
        Hit th = trace_closest(s.tlas, path_ray(vertex_origin(x.x, d), d, positive_infinity()), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, ta, positions);
        InteriorMedium im = interior_medium(s, list.innermost());
        float3 tr = beer(im.sigma_t, th.hit ? th.t : positive_infinity());
        scattered += through * (1.0 - tr) * im.albedo();
        through *= tr;
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
            traveled = length_so_far;
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
        blur += rough_widening(x, length_so_far);
        if (!thin_walled(x.x.surface))
            list.cross(x.x.surface.material, x.front);
    }
    float wr = luminance_rec2020(a.specular + a.diffuse + scattered);
    float wt = luminance_rec2020(through);
    g.diffuse = own.diffuse + scattered + through * behind.diffuse;
    g.specular = own.specular + through * behind.specular;
    float3 n = wr * lean + wt * behind.normal;
    g.normal = dot(n, n) > 0.0 ? normalize(n) : lean;
    g.roughness = wr + wt > 0.0 ? (wr * own.roughness + wt * behind.roughness) / (wr + wt) : own.roughness;
    g.weight = 1.0;
    g.dominant = 1;
    front = wr < wt ? overlay_front_word(own_depth, own_motion, wr / (wr + wt)) : 0;
    if (wr >= wt)
    {
        g.depth = own_depth;
        g.motion = own_motion;
    }
    else if (isfinite(traveled))
    {
        float3 q = dir * (length(camera) + traveled);
        g.depth = device_depth(v, q + v.eye_render.xyz);
        g.motion = reprojected_motion(v, q, 1.0);
    }
    else
    {
        g.depth = 0.0;
        g.motion = reprojected_motion(v, dir, 0.0);
    }
    if ((r.dielectric_flags & DIELECTRIC_FROST) != 0 && wt > wr && blur.x > 0.0)
    {
        // The blur's radius over the width of a pixel where the chain ends; the angle alone
        // toward the sky.
        float radius = isfinite(traveled)
            ? sqrt(max(traveled * traveled * blur.x - 2.0 * traveled * blur.y + blur.z, 0.0)) / (cone.spread * (length(camera) + traveled))
            : sqrt(blur.x) / cone.spread;
        g.frost = f32tof16(min(radius, FROST_MAX_RADIUS)) | (f32tof16(guide_luminance(through)) << 16);
    }
    return g;
}

// What rough layer `x`, `s` m along a transmission chain, adds to `layer_guides`'s blur sums.
float3 rough_widening(LayerVertex x, float s)
{
    float roughness = x.x.surface.bsdf.specular_roughness;
    if (roughness < MIRROR_ROUGHNESS)
        return float3(0.0);
    float w2 = roughness * roughness * roughness * roughness;
    return w2 * float3(1.0, s, s * s);
}

// The coherence points of a lane with no layer path: they run in every lane of the dispatch.
void skip_layer_points<C : ICoherence>(PathRoot r)
{
    for (uint i = 0; i < layer_iterations(r) * (walk_points(r) + 1); i++)
        C.point(coherence_key(false, 0, 0), KEY_BITS);
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
    float background = (r.dielectric_flags & DIELECTRIC_FROST) != 0 ? guide_luminance(r.radiance.get<RWTexture2D<float4>>()[pixel].rgb) : 0.0;
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
            uint64_t front;
            guide = layer_guides<S>(r, l, b1, first, dir, interior, coverage, front);
            if (r.overlay_front != nullptr)
                r.overlay_front[index] = front;
            emission = b1.emission();
            wr = luminance_rec2020(b1.albedo().specular + b1.albedo().diffuse);
            wt = luminance_rec2020(b1.albedo().transmission);
        }
    }
    // The froxels hold a camera segment through an interior where this frame has them.
    bool carried = interior.count > 0 && r.froxels == nullptr;
    if (carried)
        guide.medium_end = -1.0;
    else
    {
        guide.eye_medium = interior.innermost() + 1;
        if (hit_layer)
            guide.medium_end = t_near + h.t;
    }
    guide.background = background;
    r.layers[index] = guide;
    if (!hit_layer && interior.count == 0)
    {
        skip_layer_points<C>(r);
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
    bool scatters = carried && interior_segment(im, t_first, true, rng, throughput, scattered);
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
        if (path && r.mode != PATH_INDIRECT && r.mode != PATH_SUBSURFACE && l.front)
            radiance += throughput * emission;
    }
    else if (surface)
    {
        l = opaque_vertex(xg);
        // The G-buffer's emission is already pre-exposed.
        if (r.mode != PATH_INDIRECT && r.mode != PATH_SUBSURFACE)
            exposed += throughput * prepare_vertex(s, l, interior, 0).emission();
    }
    else
    {
        path = false;
        if (r.mode != PATH_INDIRECT && r.mode != PATH_SUBSURFACE)
            radiance += throughput * sky_radiance<E, M>(set, eye, dir);
    }
    if (path)
        radiance += trace_layers<C, S, M>(r, pixel, l, interior, throughput, seed, rng, rays, layers);
    else
        skip_layer_points<C>(r);

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
