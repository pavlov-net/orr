// What the staged loop (`path/stages.slang`) and the pipeline (`path/launch.slang`) share: the
// root's sources, features and predicates, a path's loop, record and context, its starts and a
// view's outputs. The staged loop is the reference the pipeline is tested against; each keeps its
// own shade bodies over these.
module common;

import orr_shaders.packing;
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.hits;
import orr_shaders.coherence;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_scene.lights;
import orr_lighting.light_sampling;
import orr_lighting.world_cache;
import orr_lighting.subsurface;
import orr_lighting.shadow;
import orr_lighting.openpbr;
import orr_lighting.interiors;
import orr_lighting.froxels;
import orr_lighting.spectrum;
import orr_lighting.caustics;
import orr_lighting.guides;
import orr_lighting.path;
import orr_lighting.specialize;

// `PathRoot.source`: which paths a root's stages run.
[Host] public static const uint PATH_SOURCE_VIEW = 0;
[Host] public static const uint PATH_SOURCE_BACKGROUND = 1;
[Host] public static const uint PATH_SOURCE_UPDATE = 2;
[Host] public static const uint PATH_SOURCE_PHOTON = 3;

// Lanes per workgroup of the stages over a queue, and of the pipeline's compute stand-ins.
[Host] public static const uint STAGE_GROUP = 32;

// `PathRoot.features`: a material of the scene may walk, so the walk stage runs; paths may
// split by wavelength; and the shade stages run as ray dispatches, which reorder their lanes by
// coherence key: a workgroup of a shade queue is then a row of STAGE_GROUP rays, and the queue
// counts rows.
[Host] public static const uint FEATURE_WALK = 1 << 0;
[Host] public static const uint FEATURE_DISPERSION = 1 << 2;
[Host] public static const uint FEATURE_RAY_SHADES = 1 << 3;
// And what a path's state holds beside FEATURE_DISPERSION's wavelengths: paths may lie in an
// interior, and may track a plane stack.
[Host] public static const uint FEATURE_INTERIORS = 1 << 4;
[Host] public static const uint FEATURE_STACK = 1 << 5;
// The pipeline path tracer's segments outside interiors march a medium: the build links one, or
// the view has froxels. Without either a segment changes nothing but its sampler dimension.
[Host] public static const uint FEATURE_SEGMENTS = 1 << 6;
// And, read by the pipeline path tracer alone, that its paths may meet dielectric or presence
// layers: their slots hold a layer vertex's words and a shadow ray's crossings; and that their
// next-event estimation may draw an emitter with a coat or fuzz, whose sample their slots keep
// for the shadow ray's end.
[Host] public static const uint FEATURE_LAYERS = 1 << 7;
[Host] public static const uint FEATURE_COATED_EMITTERS = 1 << 8;

// Whether paths of `r` track plane stacks: the light set holds a stack entry and a directional
// light, the one light a stack bends. Without either no stack changes what a path gathers.
public bool stack_tracked(PathRoot r)
{
    return may(MAY_STACK) && (r.features & FEATURE_STACK) != 0 && r.set.stack_entries != 0 && r.set.directional_count != 0;
}

// Whether paths of `r` may split by wavelength.
public bool spectral_tracked(PathRoot r)
{
    return may(MAY_DISPERSION) && (r.features & FEATURE_DISPERSION) != 0;
}

// `r.mode`, which is PATH_FULL in a pipeline that shows no debug view.
public uint path_mode(PathRoot r)
{
    return may(MAY_DEBUG) ? r.mode : PATH_FULL;
}

// Whether the view of `r` has a frosted-glass pass, a thin lens, froxels.
public bool frost_pass(PathRoot r) { return may(MAY_FROST) && (r.dielectric_flags & DIELECTRIC_FROST) != 0; }
public bool thin_lens(PathRoot r) { return may(MAY_LENS) && r.aperture_radius > 0.0; }
public bool has_froxels(PathRoot r) { return may(MAY_FROXELS) && r.froxels != nullptr; }

// Whether a pipeline made for the source of bit `bit` runs no other source's paths, so its root's
// source is known where it is made.
bool sole_source(uint bit)
{
    uint others = (MAY_VIEW | MAY_BACKGROUND | MAY_UPDATE | MAY_PHOTON) & ~bit;
    return PATH_SPECIALIZES && may(bit) && !may_any(others);
}

// Whether the paths of `r` are of `source`, which the pipeline may not run at all, or may run
// alone.
public bool from_view(PathRoot r) { return may(MAY_VIEW) && (sole_source(MAY_VIEW) || r.source == PATH_SOURCE_VIEW); }
public bool from_background(PathRoot r) { return may(MAY_BACKGROUND) && (sole_source(MAY_BACKGROUND) || r.source == PATH_SOURCE_BACKGROUND); }
public bool from_update(PathRoot r) { return may(MAY_UPDATE) && (sole_source(MAY_UPDATE) || r.source == PATH_SOURCE_UPDATE); }
public bool from_photon(PathRoot r) { return may(MAY_PHOTON) && (sole_source(MAY_PHOTON) || r.source == PATH_SOURCE_PHOTON); }

// Whether a pipeline's paths may run next-event estimation: every source's but a photon's.
public bool may_estimate()
{
    return may_any(MAY_VIEW | MAY_BACKGROUND | MAY_UPDATE);
}

// Counts a ray of the path, which only the debug view of rays per pixel reads.
public void count_ray(inout PathLoop p)
{
    if (may(MAY_DEBUG))
        p.rays++;
}

// `PathLoop.flags`. The bits from LOOP_KEY_SHIFT up hold the coherence key of the pending vertex.
public static const uint LOOP_KEY_SHIFT = 16;
public static const uint LOOP_CAMERA = 1 << 0;        // the path has no vertex yet: its ray is the camera ray
public static const uint LOOP_WALKED = 1 << 1;
public static const uint LOOP_DISPERSIVE = 1 << 2;    // the last vertex is a layer with a dispersive side
public static const uint LOOP_THROUGH = 1 << 3;       // the path crossed a chain vertex since it last scattered
public static const uint LOOP_LAST = 1 << 4;          // a scatter took the cache's light
public static const uint LOOP_COUNTED = 1 << 5;       // the iteration's light counts toward the mode
public static const uint LOOP_CARRIED = 1 << 6;       // the froxels hold no camera segment through an interior
public static const uint LOOP_GBUFFER_START = 1 << 7; // the path starts on the G-buffer's surface, its first vertex pending
public static const uint LOOP_MEDIUM = 1 << 8;        // the pending vertex is the trace stage's: a scatter or a way out
public static const uint LOOP_LEAVING = 1 << 9;       // and of those a walk leaving an interior
public static const uint LOOP_GBUFFER_HIT = 1 << 10;  // the camera ray found no layer: the pending vertex is the G-buffer's surface
public static const uint LOOP_SHADOW = 1 << 11;       // the last vertex left a shadow ray for the trace stage
public static const uint LOOP_ENDED = 1 << 12;        // and no ray of its own: the path ends past the shadow ray
static const uint LOOP_SKY = 1 << 13;          // the camera ray met nothing: `PathLoop.sky` is the throughput
public static const uint LOOP_CROSS = 1 << 14;        // the vertex's ray crosses its surface into or out of an interior, which the trace stage does past the shadow ray
public static const uint LOOP_CROSS_FRONT = 1 << 15;  // and enters it

// What a path keeps from its start, which every stage finds again from its slot (`load_fixed`):
// a slot holds the path of the slice's item of its number.
public struct PathFixed
{
    public uint2 pixel;
}

// The loop's state between stages.
public struct PathLoop
{
    public Sampler rng;
    public uint iteration;
    public uint depth;
    public uint opaque;
    public uint layers;
    public uint chains;
    public uint flags;                     // LOOP_*
    public uint rays;
    public uint entered;
    public float3 throughput;
    public float3 radiance;
    public float3 keep;
    public RayCone cone;
    public float3 sky;
    public float3 entered_at;
    public float3 entered_normal;
}

// A deferred shadow ray: what next-event estimation adds where its ray reaches the light. A
// path that tracks no interiors keeps the fields before `start`: no surface it shades bounds
// one, and the vertex's point only scales the optical depth of interiors. The interiors its
// vertex arrived with are the path's until the trace stage has traced it.
public struct PathShadowRay
{
    // The path's throughput at the vertex x the estimate, unshadowed, weighed along the
    // straight way.
    public float3 contribution;
    public uint light;                     // the winner's feedback id, NO_LIGHT for a directional winner
    public float3 origin;
    public float t_max;
    public float3 dir;
    public uint flags;                     // SHADOW_*, and from SHADOW_DEPTH_SHIFT the vertex's depth, which is below the rounds a root may ask for
    public float3 start;                   // the vertex's point
    public uint surface;
    public float4 point;
    public Interior arrived;
}

public static const uint SHADOW_ENVIRONMENT = 1 << 0;
public static const uint SHADOW_DIRECTIONAL = 1 << 1;
public static const uint SHADOW_DISPERSION = 1 << 2;
public static const uint SHADOW_FEEDBACK = 1 << 3;
public static const uint SHADOW_COUNTED = 1 << 4;
public static const uint SHADOW_BEND = 1 << 5;     // a directional light's ray looks for plane stacks
public static const uint SHADOW_INTO = 1 << 6;     // the ray leaves into the surface the vertex lies on
public static const uint SHADOW_VIEW = 1 << 7;     // the light is an emitter whose coat and fuzz its light passes (`emitter_view`)
// Read by the pipeline path tracer alone: the ray is a leg of a plane stack's walk, inside the
// interiors the walk holds.
public static const uint SHADOW_LEG = 1 << 8;
[Host] public static const uint SHADOW_DEPTH_SHIFT = 9;

// A record's words: a view's and a photon's, and an update path's, whose slices hold as many
// fewer paths.
[Host] public static const uint RECORD_WORDS = 2;
[Host] public static const uint UPDATE_RECORD_WORDS = 9;

public bool interiors_tracked(PathRoot r)
{
    return may(MAY_INTERIORS) && (r.features & FEATURE_INTERIORS) != 0;
}

// The parts as their words hold them. A counter is below the rounds a root may ask for, which
// 24 bits hold: the bytes of `chains` lie above `iteration`, `depth` and `opaque`. The sampler
// keeps its dimension alone, beside the ray's direction, as its seed is the pixel's and its
// index the frame's.
public struct StoredLoop
{
    public float3 throughput;
    public uint flags;                     // LOOP_*, without LOOP_SKY's `sky`, which is the throughput
    public float3 radiance;
    public uint iteration;
    public float3 keep;
    public uint depth;
    public RayCone cone;
    public uint opaque;
    public uint layers;
}

static const uint RAY_DIFFUSE = 1 << 0;
static const uint RAY_CHAIN = 1 << 1;
static const uint RAY_SCATTER = 1 << 2;
static const uint RAY_LOCAL = 1 << 3;
static const uint RAY_LIT = 1 << 4;
public static const uint RAY_FLAG_BITS = 5;

// The pixel of the path of item `index` of the root's source: a photon's place in its rows, an
// update block's pixel, or a view's pixel.
public uint2 item_pixel(PathRoot r, uint index)
{
    if (from_photon(r))
        return uint2(index % PHOTON_ROW, index / PHOTON_ROW);
    if (from_update(r))
        return update_pixel(r, index);
    uint2 extent = r.gbuffer.extent;
    return uint2(index % extent.x, index / extent.x);
}

// The pixel of the update path of the view's `index`th block: one of the block's, rotating
// through it over frames.
public uint2 update_pixel(PathRoot r, uint index)
{
    uint n = r.update_block;
    uint2 blocks = (r.gbuffer.extent + n - 1) / n;
    uint2 block = uint2(index % blocks.x, index / blocks.x);
    uint offset = (r.frame + pcg(block.x | (block.y << 16))) % (n * n);
    return block * n + uint2(offset % n, offset / n);
}

// The stream of the sampler a path of the root's source draws from.
public uint path_stream(PathRoot r)
{
    return r.seed ^ (from_background(r) ? DOF_STREAM : from_update(r) ? UPDATE_STREAM : from_photon(r) ? PHOTON_STREAM : 0u);
}

// The sampler a path of `pixel` of the root's source starts with.
public Sampler path_sampler(PathRoot r, uint2 pixel)
{
    return Sampler(pixel, r.frame, path_stream(r));
}

// Whether `pixel` lies on a tile the view's layer guides traced.
public bool pixel_traced(PathRoot r, uint2 pixel)
{
    uint2 extent = r.gbuffer.extent;
    return traced_tiles(r.layers, extent)[layer_tile(extent, pixel)] != 0;
}

// Whether a view's `pixel` also runs a path for the frosted-glass term, in a second pass.
public bool two_passes(PathRoot r, uint2 pixel)
{
    if (!frost_pass(r) || !pixel_traced(r, pixel))
        return false;
    return r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].frost != 0;
}

// The camera ray a path of `pixel` started with, as its source's start stage made it: a view's
// on a traced tile or through the lens, the background layer's, or a photon's, whose hit is
// given.
public PathStart camera_start(PathRoot r, uint2 pixel)
{
    if (from_photon(r))
    {
        PathStart given = PathStart();
        given.mask = MASK_OPAQUE | MASK_DIELECTRIC;
        given.given = true;
        return given;
    }
    if (from_background(r))
    {
        float mask = r.dof_mask.get<Texture2D<float>>().Load(int3(pixel, 0));
        float near = r.view.clip_from_view[2][3];
        Sampler rng = Sampler(pixel, r.frame, r.seed ^ DOF_STREAM);
        return lens_start(r, pixel, r.dof_aperture, *r.dof_focus, near / mask, MASK_CAMERA | MASK_DIELECTRIC, rng);
    }
    if (thin_lens(r))
    {
        Sampler rng = Sampler(pixel, r.frame, r.seed);
        return lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_CAMERA | MASK_DIELECTRIC, rng);
    }
    return pinhole_start(r.view, r.gbuffer, pixel, true);
}

// Whether the root's mode is a debug view of the first vertex alone, which a view's start writes.
public bool first_vertex_mode(PathRoot r)
{
    uint mode = path_mode(r);
    return mode == PATH_CACHE || mode == PATH_CACHE_LOD || mode == PATH_ORIGIN_BIAS || mode == PATH_CAUSTICS;
}

// The start of a view's path of `pixel` with its sampler `rng`: through the thin lens, inside the
// interiors at its point of the lens; on a tile the layer guides traced, a camera ray that looks
// for layers before the G-buffer's surface, inside the interiors at its near-plane point; else
// from the G-buffer's surface, with no camera ray.
public PathStart view_start<S : ISurfaces>(PathRoot r, uint2 pixel, inout Sampler rng)
{
    if (thin_lens(r))
    {
        PathStart start = lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_CAMERA | MASK_DIELECTRIC, rng);
        start.interior = lens_interior<S>(r, start.origin);
        return start;
    }
    bool traced = pixel_traced(r, pixel);
    PathStart start = pinhole_start(r.view, r.gbuffer, pixel, traced);
    if (traced)
        start.interior = Interior(r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior);
    return start;
}

// The start of the background layer's path of `pixel` with its sampler `rng`, on a marked pixel:
// its lens ray past the mask's depth. Elsewhere false, with the pixel's texel (0, 0, 0, -1)
// written.
public bool background_path_start(PathRoot r, uint2 pixel, inout Sampler rng, out PathStart start)
{
    start = PathStart();
    float mask = r.dof_mask.get<Texture2D<float>>().Load(int3(pixel, 0));
    if (!(mask > 0.0))
    {
        r.dof_background.get<RWTexture2D<float4>>()[pixel] = float4(0.0, 0.0, 0.0, -1.0);
        return false;
    }
    float near = r.view.clip_from_view[2][3];
    start = lens_start(r, pixel, r.dof_aperture, *r.dof_focus, near / mask, MASK_CAMERA | MASK_DIELECTRIC, rng);
    return true;
}

// The start of an update path of `pixel`: from its G-buffer surface, inside the interiors around
// it.
public PathStart update_path_start<S : ISurfaces>(PathRoot r, uint2 pixel)
{
    PathStart start = pinhole_start(r.view, r.gbuffer, pixel, false);
    start.interior = gbuffer_interior<S>(r.scene, r.view, r.layers, r.gbuffer.extent, pixel, start);
    return start;
}

// The start of a photon its light emitted as `p`: its first hit given.
public PathStart photon_path_start(PhotonStart p)
{
    PathStart start = PathStart();
    start.origin = p.point;
    start.dir = p.dir;
    start.mask = MASK_OPAQUE | MASK_DIELECTRIC;
    start.interior = p.interior;
    start.throughput = p.flux;
    start.given = true;
    start.hit = p.hit;
    start.positions = p.positions;
    start.directional = p.directional;
    return start;
}

// The record a view's or its background layer's path of `pixel` starts with.
public StoredView view_record_start(PathRoot r, uint2 pixel)
{
    return new_view_record(pcg3d(uint3(pixel, r.frame ^ path_stream(r))).x);
}

// The loop of `v` with sampler `rng` and `rays` rays, the parts but its interiors'.
public PathLoop unpack_loop(StoredLoop v, Sampler rng, uint rays)
{
    PathLoop p;
    p.rng = rng;
    p.iteration = v.iteration & 0xffffff;
    p.depth = v.depth & 0xffffff;
    p.opaque = v.opaque & 0xffffff;
    p.layers = v.layers;
    p.chains = (v.iteration >> 24) | ((v.depth >> 24) << 8) | ((v.opaque >> 24) << 16);
    p.flags = v.flags;
    p.rays = rays;
    p.throughput = v.throughput;
    p.radiance = v.radiance;
    p.keep = v.keep;
    p.cone = v.cone;
    p.sky = has(v.flags, LOOP_SKY) ? v.throughput : float3(0.0);
    // A path enters an interior only where it tracks them.
    p.entered = MATERIAL_NONE;
    p.entered_at = float3(0.0);
    p.entered_normal = float3(0.0);
    return p;
}

// What `StoredLoop` holds of `p`: all but its sampler's dimension, its rays and its interiors.
public StoredLoop pack_loop(PathLoop p)
{
    StoredLoop v;
    v.iteration = p.iteration | ((p.chains & 0xff) << 24);
    v.depth = p.depth | (((p.chains >> 8) & 0xff) << 24);
    v.opaque = p.opaque | ((p.chains >> 16) << 24);
    v.layers = p.layers;
    // The sky a path keeps is its throughput where its camera ray met nothing.
    v.flags = any(p.sky > 0.0) ? p.flags | LOOP_SKY : p.flags & ~LOOP_SKY;
    v.throughput = p.throughput;
    v.radiance = p.radiance;
    v.keep = p.keep;
    v.cone = p.cone;
    return v;
}

// `ray`'s flags as one word holds them: RAY_*, in the bits below RAY_FLAG_BITS.
public uint ray_flags(LayerRay ray)
{
    return (ray.diffuse ? RAY_DIFFUSE : 0) | (ray.chain ? RAY_CHAIN : 0) | (ray.scatter ? RAY_SCATTER : 0) | (ray.local ? RAY_LOCAL : 0) | (ray.lit ? RAY_LIT : 0);
}

// Sets `ray`'s flags from `ray_flags`' word `flags`.
public void set_ray_flags(inout LayerRay ray, uint flags)
{
    ray.diffuse = (flags & RAY_DIFFUSE) != 0;
    ray.chain = (flags & RAY_CHAIN) != 0;
    ray.scatter = (flags & RAY_SCATTER) != 0;
    ray.local = (flags & RAY_LOCAL) != 0;
    ray.lit = (flags & RAY_LIT) != 0;
}

// A view's or the background layer's record (`ViewPath`) as its words hold it. A first vertex
// that is a layer leaves no G-buffer emission, so the two share words.
public struct StoredView
{
    public uint feedback;
    public float total;
    public uint state;
    public float first_distance;
    public float3 own;                     // the G-buffer's emission, pre-exposed; of a first layer its reflected and transmitted luminance
    public float first_t;                  // with its sign bit set on a first layer
}

// The first word of `StoredView`.
public struct StoredViewLights
{
    public uint feedback;
    public float total;
    public uint state;
    public float first_distance;
}

public StoredView new_view_record(uint seed)
{
    StoredView v = { NO_LIGHT, 0.0, seed, 0.0, float3(0.0), 0.0 };
    return v;
}

bool view_first_layer(StoredView v)
{
    return (asuint(v.first_t) >> 31) != 0;
}

public float view_first_t(StoredView v)
{
    return asfloat(asuint(v.first_t) & 0x7fffffff);
}

// The G-buffer's emission the path took, which is already pre-exposed.
public float3 view_exposed(StoredView v)
{
    return view_first_layer(v) ? float3(0.0) : v.own;
}

// A view's record once its path reached its first vertex `x`, prepared as `b`, `t` m from the
// eye: a layer's where `layer` holds.
public void view_arrive(inout StoredView p, PathVertex x, Bsdf b, float t, bool layer)
{
    p.first_distance = x.camera_distance;
    p.own = float3(0.0);
    if (layer)
        p.own = float3(luminance_rec2020(b.albedo().specular + b.albedo().diffuse), luminance_rec2020(b.albedo().transmission), 0.0);
    p.first_t = asfloat((asuint(t) & 0x7fffffff) | (layer ? 0x80000000u : 0u));
}

// Whether light that added `contribution` of luminance takes part in a view's light feedback.
public bool view_light_counts(float contribution)
{
    return contribution > 0.0 && isfinite(contribution);
}

// A view's light feedback once light `id` added `contribution` of luminance, which
// `view_light_counts`: one of the lights, chosen in proportion to what each added.
public void view_light(inout StoredViewLights p, uint id, float contribution)
{
    p.total += contribution;
    p.state = pcg(p.state);
    if (unit_float(p.state) * p.total < contribution)
        p.feedback = id;
}

// What the shade bodies hand a path's events to: its record, and the G-buffer's emission its
// first vertex gives toward the eye, pre-exposed; and what decides whether a path's light
// through a plane stack is next-event estimation's.
public interface IVertexRecord : IPathRecord
{
    [mutating]
    void expose(float3 e);
    // `stack_owns` of a vertex of depth `depth` whose probe leaves `origin` along `dir`.
    bool stack_owned<S : ISurfaces>(PathRoot r, float3 origin, float3 dir, PathStack stack, uint depth, bool dispersion, bool feedback);
}

// ---------------------------------------------------------------------------------------------
// What every stage derives from the root and the path

public struct PathContext
{
    public bool photon;
    public bool count_nee;
    public bool count_bsdf;
    public bool one_technique;
    public bool reference;
    public bool cache;
    public bool disperse;
    public bool caustics;
    public bool feedback;
    public uint seed;
}

public PathContext path_context<R : IPathRecord>(PathRoot r, R record, uint2 pixel)
{
    PathContext c;
    c.photon = record.photon();
    uint mode = path_mode(r);
    c.count_nee = mode != PATH_BSDF_ONLY && !c.photon;
    c.count_bsdf = mode != PATH_NEE_ONLY && !c.photon;
    c.one_technique = mode == PATH_NEE_ONLY || mode == PATH_BSDF_ONLY;
    c.reference = may(MAY_REFERENCE) && (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
    c.cache = may(MAY_CACHE) && (r.cache_flags & CACHE_TERMINATE) != 0 && !c.one_technique && mode != PATH_DIRECT;
    c.disperse = may(MAY_DISPERSION) && record.spectral() && (r.dielectric_flags & DIELECTRIC_DISPERSION) != 0;
    c.caustics = may(MAY_CAUSTICS) && r.caustics != nullptr && !c.one_technique && !c.photon && caustics_hold_light(r.caustics);
    c.seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    c.feedback = !c.photon && texture_feedback_drawn(pixel, r.frame);
    return c;
}

public bool has(uint flags, uint bit)
{
    return (flags & bit) != 0;
}

// What owned light path `p`'s ray `ray` meets keeps: none on a ray a walk's way out lights, the
// chain's share past a chain vertex, else all of it.
public float3 owned_light(PathLoop p, LayerRay ray)
{
    return ray.lit ? float3(0.0) : has(p.flags, LOOP_THROUGH) ? p.keep : float3(1.0);
}

// Whether a path with `throughput` goes on: it carries light, and finite light.
public bool carries(float3 throughput)
{
    return any(throughput > 0.0) && all(isfinite(throughput));
}

// The instance mask of what path `p`'s ray `ray` sees in front of its hit beside what covers it:
// particles for rays leaving a layer or a scatter, and the first vertex's, and for every ray that
// adds emission the proxies of sphere and rect lights, which cover nothing. The camera ray sees
// neither.
public uint front_mask<R : IPathRecord>(PathRoot r, PathLoop p, LayerRay ray, R record, PathContext x)
{
    if (has(p.flags, LOOP_CAMERA))
        return 0;
    bool particles = may(MAY_PARTICLES) && record.particles() && (ray.chain || p.depth == 0) && r.set.particle_entries > 0;
    bool lights = may(MAY_LOCAL_LIGHTS) && x.count_bsdf && has(p.flags, LOOP_COUNTED) && r.set.local_lights > 0;
    return (particles ? MASK_PARTICLE : 0) | (lights ? MASK_LIGHT : 0);
}

// Whether the ray of path `p`, inside `interior`, runs inside an interior, whose medium its segment
// takes: a camera ray only where the froxels hold none of its segment.
public bool in_interior(PathRoot r, PathLoop p, Interior interior)
{
    return interiors_tracked(r) && (!has(p.flags, LOOP_CAMERA) || has(p.flags, LOOP_CARRIED)) && interior.innermost() != MATERIAL_NONE;
}

public void set_flag(inout uint flags, uint bit, bool on)
{
    flags = on ? flags | bit : flags & ~bit;
}

// Whether the iteration's light counts toward the root's mode, from the state at its top.
public bool iteration_counted(PathRoot r, PathLoop p)
{
    return path_mode(r) == PATH_DIRECT ? p.depth == 0 : path_mode(r) == PATH_INDIRECT ? p.depth > 0
        : path_mode(r) == PATH_SUBSURFACE ? has(p.flags, LOOP_WALKED) : true;
}

// The top of an iteration: whether the path goes on into it from vertex kind `kind`.
public bool iteration_runs(PathRoot r, PathLoop p, uint kind, bool alive)
{
    return alive && p.iteration < path_iterations(r) && (kind == VERTEX_LAYER || p.opaque <= r.max_bounces + 1);
}

// What a view writes of a pixel's own path once it ended with `pixel_radiance`: its
// radiance, layer memory and light feedback, and with frost its background's luminance. The
// radiance texel it wrote.
public float4 view_outputs(PathRoot r, PathFixed f, PathLoop p, StoredView pixel_path, float3 pixel_radiance)
{
    uint2 pixel = f.pixel;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    bool arrived = pixel_path.first_distance > 0.0;
    bool first_layer = view_first_layer(pixel_path);
    float4 out = float4(apply_hdr_ceiling(pixel_radiance), arrived ? 1.0 : 0.0);
    if (path_mode(r) == PATH_RAYS)
        out = float4(float(p.rays), all(isfinite(pixel_radiance)) ? 0.0 : 1.0, 0.0, out.a);
    else if (may(MAY_DEBUG) && (r.dielectric_flags & DIELECTRIC_DEBUG_LAYER) != 0 && first_layer)
    {
        float sum = max(pixel_path.own.x + pixel_path.own.y, 1e-6);
        out = float4(pixel_path.own.x / sum, pixel_path.own.y / sum, float(p.layers) / float(max(r.dielectric_depth, 1u)), 1.0);
    }
    r.radiance.get<RWTexture2D<float4>>()[pixel] = out;
    if (pixel_traced(r, pixel))
        r.layers[index].medium_end = has(p.flags, LOOP_CARRIED) ? -1.0 : first_layer ? view_first_t(pixel_path) : 0.0;
    uint feedback = pixel_path.feedback;
    r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(feedback, asuint(pixel_path.first_distance));
    if ((r.light_flags & LIGHT_FEEDBACK_COUNT) != 0 && feedback != NO_LIGHT)
        count_feedback(r.set, feedback);
    if (frost_pass(r))
        r.layers[index].background = guide_luminance(pixel_radiance);
    return out;
}

// The state a path of `pixel` starts from `start` with.
public void start_state(PathRoot r, uint2 pixel, PathStart start, Sampler rng, bool photon, out PathFixed f, out PathLoop p, out ShadeState c)
{
    f.pixel = pixel;

    p.rng = rng;
    p.iteration = 0;
    p.depth = 0;
    p.opaque = 0;
    p.layers = 0;
    p.chains = 0;
    p.flags = 0;
    p.rays = 0;
    p.radiance = float3(0.0);
    p.sky = float3(0.0);
    p.entered_at = float3(0.0);
    p.entered_normal = float3(0.0);
    p.throughput = start.throughput;
    p.keep = float3(1.0);
    p.cone = camera_cone(r.view);
    p.entered = MATERIAL_NONE;
    bool camera = start.mask != 0;
    // The first vertex resolves as the camera's, traced or on the G-buffer's surface.
    set_flag(p.flags, LOOP_CAMERA, true);
    // The froxels hold a camera segment through an interior where this frame has them.
    set_flag(p.flags, LOOP_CARRIED, start.interior.count > 0 && !has_froxels(r));
    // A path without a camera ray starts on the G-buffer's surface as its first vertex.
    set_flag(p.flags, LOOP_GBUFFER_START, !camera);
    set_flag(p.flags, LOOP_GBUFFER_HIT, !camera);
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));

    c.ray = LayerRay();
    c.ray.origin = start.origin;
    c.ray.start = start.origin;
    c.ray.dir = start.dir;
    c.interior = start.interior;
    c.stack = PathStack();
    c.stack.valid = photon && start.directional && r.set.stack_entries != 0;
    // A photon's stack keeps no vertex's ray: its direction is the one the photon started
    // along, which `vertex_nee` probes back along.
    if (photon)
        c.stack.dir = start.dir;
    c.spectral = Spectral();
}

// The hit context of the candidates of a shadow ray of flags `flags` (SHADOW_*, with its vertex's
// depth): a secondary ray's, of no cone.
public HitContext shadow_context(uint flags)
{
    RayCone cone = { 0.0, 0.0 };
    return hit_context(HIT_SECONDARY, (flags >> SHADOW_DEPTH_SHIFT) + 1, cone, 0.0, float2(-1.0), has(flags, SHADOW_FEEDBACK));
}

// The rest of a path's state, as a shade stage holds it.
public struct ShadeState
{
    public LayerRay ray;
    public Interior interior;
    public PathStack stack;
    public Spectral spectral;
}

// The shadow ray a vertex's next-event estimation left, and what weighs its light again where
// the ray meets a plane stack.
public struct ShadeNee
{
    public bool traced;
    // The vertex's stage traces the ray: through a plane stack the vertex's scatter weighs the
    // light again, and an emitter's coat and fuzz pass its light on only where the ray reaches
    // it, neither of which the trace stage keeps a light sample for.
    public bool here;
    public PathShadowRay q;
    public NeeDraw d;
    public float run;                      // the vertex's probability of running next-event estimation
    public bool weight_one;
    public float3 throughput;              // the path's at the vertex
    public float exterior_ior;             // of the medium the vertex lies in
    public float3 keep;                    // what a chain keeps of the vertex's scatter
    public bool chain;
    public Spectral spectral;              // the path's at the vertex
}

// No shadow ray, on a path with `spectral`.
public ShadeNee no_nee(Spectral spectral)
{
    ShadeNee nee;
    nee.traced = false;
    nee.here = false;
    nee.q = PathShadowRay();
    nee.d = NeeDraw();
    nee.run = 0.0;
    nee.weight_one = false;
    nee.throughput = float3(0.0);
    nee.exterior_ior = 0.0;
    nee.keep = float3(0.0);
    nee.chain = false;
    nee.spectral = spectral;
    return nee;
}

// The end of the path of a view's pixel `f.pixel` with record `record`, in its first pass or
// `second`: the pixel's radiance, with the sky its camera ray met where the mode shows it, as the
// pixel's outputs in the first pass, whose radiance texel is in `texel`, and as its background's
// luminance in the second. True where the pixel's second pass follows, from `start` on the
// G-buffer's surface inside what holds it, with `record` readied for it.
public bool view_pass_end<S : ISurfaces, E : IEnvironment, M : IMedium>(PathRoot r, PathFixed f, PathLoop p, inout StoredView record, bool second, out float4 texel, out PathStart start)
{
    ViewConstants* v = r.view;
    uint2 pixel = f.pixel;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float3 pixel_radiance = view_exposed(record) + pre_exposure * p.radiance;
    bool sky = any(p.sky > 0.0) && path_mode(r) != PATH_INDIRECT && path_mode(r) != PATH_SUBSURFACE;
    bool two = two_passes(r, pixel);
    texel = float4(0.0);
    start = PathStart();
    // The camera ray the pixel started with, where what is left to do reads it.
    PathStart camera_ray = PathStart();
    if (sky || (two && !second))
        camera_ray = camera_start(r, pixel);
    if (sky)
        pixel_radiance += pre_exposure * p.sky * sky_radiance<E, M>(r.set, v.eye_render.xyz, camera_ray.dir);
    if (second)
    {
        if (frost_pass(r))
            r.layers[index].background = guide_luminance(pixel_radiance);
        return false;
    }
    texel = view_outputs(r, f, p, record, pixel_radiance);
    if (!two)
        return false;
    if (thin_lens(r))
        start.interior = lens_interior<S>(r, camera_ray.origin);
    else
    {
        start = pinhole_start(v, r.gbuffer, pixel, true);
        start.interior = Interior(r.layers[index].near_interior);
        if (isfinite(start.t_max))
            surface_crossings<S>(r.scene, v, start.origin, start.dir, start.t_max, start.interior);
    }
    start.dir = camera_ray.dir;
    start.throughput = float3(1.0);
    start.mask = 0;
    start.gbuffer = true;
    // The second pass's path takes the G-buffer's emission anew.
    record.own = float3(0.0);
    return true;
}

// What the background layer writes of its path of `pixel` that ended with `radiance` and
// `record`: its texel, the pre-exposed radiance in RGB and the reverse-Z depth of its hit in A (0
// on a miss).
public void background_texel<E : IEnvironment, M : IMedium>(PathRoot r, uint2 pixel, StoredView record, float3 radiance)
{
    ViewConstants* v = r.view;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float near = v.clip_from_view[2][3];
    PathStart camera_ray = camera_start(r, pixel);
    float4 texel;
    if (!(record.first_distance > 0.0))
        texel = float4(apply_hdr_ceiling(pre_exposure * sky_radiance<E, M>(r.set, v.eye_render.xyz, camera_ray.dir)), 0.0);
    else
        texel = float4(apply_hdr_ceiling(pre_exposure * radiance), near / (camera_ray.axis * view_first_t(record)));
    r.dof_background.get<RWTexture2D<float4>>()[pixel] = texel;
}
