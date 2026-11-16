// The lean tracer's common path: the whole vertex of a plain opaque hit, which its hit shader runs
// from the path's hot state, and what a ray that met nothing adds.
//
// A vertex resolves its surface, takes the medium along its ray, prepares its BSDF once, adds what
// its surface emits where its stage is made for emitters, ends in the world cache where the cache
// holds its light, picks one light from a stream of candidates weighed as they are drawn, leaves
// the pick as a pending shadow job, samples its next ray and stores the hot state. What interiors,
// layers and walks add is folded by the mask where a scene holds none of them, and the vertices
// in a medium and where a walk leaves run in callables of their own.
module vertex;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.pipeline;
import orr_color.luminance;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.surface;
import orr_scene.candidates;
import orr_scene.gbuffer;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.lean.state;
import orr_lighting.interiors;
import orr_lighting.subsurface;

// The hot words of the slice's `slot`th path.
public uint4* lean_hot(PathRoot r, uint slot)
{
    return r.launch.hot + slot * LEAN_HOT_WORDS;
}

// The record of the slice's `slot`th path: a view's, or an update path's.
public uint4* lean_record(PathRoot r, uint slot)
{
    return r.launch.records + slot * (from_update(r) ? LEAN_UPDATE_RECORD_WORDS : LEAN_VIEW_RECORD_WORDS);
}

// The cold words of the slice's `slot`th path, where the scene may hold a layer or an interior.
public uint4* lean_cold(PathRoot r, uint slot)
{
    return r.launch.cold + slot * LEAN_COLD_WORDS;
}

// The interior list cold words `at` hold.
public Interior load_interior(uint4* at)
{
    Interior i = Interior();
    i.entries = load_coherent(at);
    i.count = 0;
    for (uint k = 0; k < INTERIOR_ENTRIES; k++)
    {
        i.count += i.entries[k] != MATERIAL_NONE ? 1 : 0;
        i.points[k] = asfloat(load_coherent(at + 1 + k));
    }
    return i;
}

// Stores interior list `i` in cold words `at`, with its innermost medium's extinction, which
// depends on the material and where the path entered it alone: what the stages that do not change
// the list read of it.
public void store_interior(SceneRoot* s, uint4* at, Interior i)
{
    store_coherent(at, i.entries);
    for (uint k = 0; k < INTERIOR_ENTRIES; k++)
        store_coherent(at + 1 + k, asuint(i.points[k]));
    float3 sigma = float3(0.0);
    float4 scatter = float4(0.0);
    if (i.count > 0)
    {
        uint m = i.innermost();
        InteriorMedium im = interior_medium(s, m, i.point(m));
        sigma = im.sigma_t;
        if (may(MAY_SCATTERING))
            scatter = float4(im.sigma_s, im.anisotropy);
    }
    uint4* last = at + 1 + INTERIOR_ENTRIES;
    store_coherent(last, uint4(load_coherent(last).x, asuint(sigma)));
    if (may(MAY_SCATTERING))
        store_coherent(at + LEAN_SCATTER_WORD, asuint(scatter));
}

// The innermost interior of the list whose entries are `entries`, innermost last.
public uint innermost_of(uint4 entries)
{
    uint m = MATERIAL_NONE;
    for (uint k = 0; k < INTERIOR_ENTRIES; k++)
    {
        if (entries[k] != MATERIAL_NONE)
            m = entries[k];
    }
    return m;
}

// The extinction of the innermost interior's medium, which cold words `at` keep.
float3 interior_sigma(uint4* at)
{
    return asfloat(load_coherent(at + 1 + INTERIOR_ENTRIES).yzw);
}

// How far a ray `t` m long whose path's hot words are `at` ran through the medium it left its vertex
// in: from the vertex's surface, which the ray's origin lies off, as the words keep the offset (its
// squared length and its share along the ray). A camera ray has none.
public float inside_length(uint4* at, float t)
{
    float2 o = halves(load_coherent(at + 2).z);
    return sqrt(max(o.x + 2.0 * t * o.y + t * t, 0.0));
}

// The shading normal of the vertex a path's ray left, whose flags are `flags` and second hot word
// `w1`: none (0) where the ray left a point in a medium.
float3 left_normal(uint flags, uint4 w1)
{
    if (may(MAY_SCATTERING) && (flags & HOT_MEDIUM) != 0)
        return float3(0.0);
    return oct_decode_snorm16(w1.y);
}

// The layers a path met, which its cold words `at` count.
public uint met_layers(uint4* at)
{
    return load_coherent(at + 1 + INTERIOR_ENTRIES).x;
}

public void store_met_layers(uint4* at, uint layers)
{
    uint4* last = at + 1 + INTERIOR_ENTRIES;
    store_coherent(last, uint4(layers, load_coherent(last).yzw));
}

// Starts `slot`'s path in the interiors around the eye at its pixel's near-plane point, which the
// layer guides list, its cold words empty but for them; or, `behind`, in those around the G-buffer's
// surface, past the surfaces the camera ray crosses up to it, for a path that starts there. Its hot
// words say whether the path lies in one. The callable a camera inside an interior, and a frosted
// pixel's second path, start with.
public void lean_eye_interior<S : ISurfaces>(PathRoot r, uint slot, bool behind)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    uint4* cold = lean_cold(r, slot);
    store_coherent(cold + 1 + INTERIOR_ENTRIES, uint4(0));
    Interior interior = Interior(r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior);
    if (behind)
    {
        PathStart start = pinhole_start(r.view, r.gbuffer, pixel, true);
        if (isfinite(start.t_max))
            surface_crossings<S>(r.scene, r.view, start.origin, start.dir, start.t_max, interior);
    }
    store_interior(r.scene, cold, interior);
    uint4* at = lean_hot(r, slot);
    uint4 w0 = load_coherent(at);
    store_coherent(at, uint4(w0.xyz, interior.count > 0 ? w0.w | HOT_INTERIOR : w0.w & ~HOT_INTERIOR));
}

// What sets a vertex's BSDF apart by what its surface is, as the stage that runs it knows it: an
// opaque surface, whose path passes no surface past it, or a layer of the dielectric class, whose
// transmission crosses into or out of the interior its section material bounds.
interface IVertexKind
{
    // The vertex `x`'s BSDF, its path inside the interiors its cold words `cold` list where
    // `inside`.
    Bsdf prepare(PathRoot r, PathVertex x, uint4* cold, bool inside);
    // The interiors past a scatter `s` of the vertex, in `cold`: whether the path lies in one.
    bool cross(PathRoot r, BsdfSample s, uint4* cold, bool inside);
}

// An opaque surface: in an interior its BSDF sees the interior's medium outside it.
struct OnOpaque : IVertexKind
{
    Bsdf prepare(PathRoot r, PathVertex x, uint4* cold, bool inside)
    {
        float exterior = 1.0;
        if (inside)
            exterior = interior_ior(r.scene, innermost_of(load_coherent(cold)));
        return bsdf_prepare(x.surface.bsdf, x.surface.normal, x.view, exterior);
    }

    bool cross(PathRoot r, BsdfSample s, uint4* cold, bool inside)
    {
        return inside;
    }
}

// A layer `l`, prepared with the media on its two sides as the path's interiors give them.
struct OnLayer : IVertexKind
{
    LayerVertex l;

    Bsdf prepare(PathRoot r, PathVertex x, uint4* cold, bool inside)
    {
        return prepare_vertex(r.scene, l, inside ? load_interior(cold) : Interior(), false, false, false, 0.0);
    }

    bool cross(PathRoot r, BsdfSample s, uint4* cold, bool inside)
    {
        if ((s.lobe & LOBE_TRANSMISSION) == 0 || !crosses_interior(r.scene, l.section))
            return inside;
        Interior interior = inside ? load_interior(cold) : Interior();
        interior.cross(l.section, l.front, l.x.surface.medium_point);
        store_interior(r.scene, cold, interior);
        // A dense walk stopped at its cap leaves where the path entered the interior it walked.
        if (may(MAY_SCATTERING))
        {
            float3 n = l.x.surface.geometric_normal;
            store_coherent(cold + LEAN_ENTERED_WORD, l.front ? uint4(asuint(l.x.surface.position), oct_encode_snorm16(n)) : uint4(NO_ENTRY));
        }
        return interior.count > 0;
    }
}

// What the root's mode counts of a path's light.
public struct LeanMode
{
    public bool count_nee;
    public bool count_bsdf;
    public bool one_technique;
    public bool cache;
}

public LeanMode lean_mode(PathRoot r)
{
    uint mode = path_mode(r);
    LeanMode m;
    m.count_nee = mode != PATH_BSDF_ONLY;
    m.count_bsdf = mode != PATH_NEE_ONLY;
    m.one_technique = mode == PATH_NEE_ONLY || mode == PATH_BSDF_ONLY;
    m.cache = may(MAY_CACHE) && (r.cache_flags & CACHE_TERMINATE) != 0 && !m.one_technique && mode != PATH_DIRECT;
    return m;
}

// Whether the light a vertex gathers and its ray meets counts toward the root's mode, at a
// vertex the path reached after `scattered` scatters, past a walk where it `walked`.
public bool lean_counted(PathRoot r, uint scattered, bool walked = false)
{
    uint mode = path_mode(r);
    return mode == PATH_DIRECT ? scattered == 0 : mode == PATH_INDIRECT ? scattered > 0 : mode != PATH_SUBSURFACE || walked;
}

// Whether the view's paths trace their camera rays, whose hits are their first vertices and
// write the view's G-buffer.
public bool traces_primary(PathRoot r)
{
    return may(MAY_TRACED) && (r.light_flags & LIGHT_TRACED_PRIMARY) != 0;
}

// Writes the hit `sh` of `slot`'s camera ray along `dir` as `pixel`'s G-buffer surface, as the
// material resolve writes a rastered one: the visibility word, naming the pixel's own visible
// cluster, the surface's words, its motion and its depth, into the HZB's mip 0 too, which is
// the G-buffer's depth until the depth target is drawn from it; and into the path's record the
// surface's distance from the eye. The motion is that of a point that stood still: a hit holds
// no positions of the frame before.
void write_primary(PathRoot r, uint slot, uint2 pixel, SurfaceHit sh)
{
    RasterGpu* raster = (RasterGpu*)r.gbuffer;
    ViewConstants* v = r.view;
    GSurface g = sh.surface;
    float3 relative = g.position - v.eye_render.xyz;
    uint4* record = lean_record(r, slot);
    StoredViewLights lights = reinterpret<StoredViewLights>(load_coherent(record));
    lights.first_distance = length(relative);
    store_coherent(record, reinterpret<uint4>(lights));
    uint index = pixel.y * raster.gbuffer.extent.x + pixel.x;
    float depth = device_depth(v, g.position);
    // The view's list holds an entry per pixel unless memory was short for it, where the pixel
    // reads as one with no geometry.
    if (index < raster.visible_capacity)
    {
        VisibleCluster vc = { nullptr, g.instance, g.cluster, 0, g.material };
        raster.gbuffer.clusters[index] = vc;
        raster.gbuffer.visibility[index] = visibility_word(depth, index, g.primitive);
    }
    g.raster_error = sh.geometric_error;
    g.bsdf.emission = apply_hdr_ceiling(g.bsdf.emission * v.exposure[v.exposure_slot].pre_exposure);
    GBufferWords words = encode_gsurface(g, raster.x.index != IMAGE_NONE);
    uint flags = words.a.w >> 24;
    raster.a.get<RWTexture2D<uint4>>()[pixel] = words.a;
    raster.b.get<RWTexture2D<uint2>>()[pixel] = words.b;
    if ((flags & GBUFFER_EXTENDED) != 0)
        raster.x.get<RWTexture2D<uint4>>()[pixel] = words.x;
    if ((flags & GBUFFER_LAYERS) != 0)
        raster.y.get<RWTexture2D<uint4>>()[pixel] = words.y;
    raster.motion.get<RWTexture2D<float2>>()[pixel] = still_motion(v, relative);
    raster.depth[index] = depth;
    raster.hzb[0].get<RWTexture2D<float>>()[pixel] = depth;
}

// Writes `pixel` as one whose camera ray along `dir` met nothing, as the material resolve writes
// a pixel nothing was rastered on.
void write_primary_sky(PathRoot r, uint2 pixel, float3 dir)
{
    RasterGpu* raster = (RasterGpu*)r.gbuffer;
    uint index = pixel.y * raster.gbuffer.extent.x + pixel.x;
    raster.a.get<RWTexture2D<uint4>>()[pixel] = uint4(0);
    raster.b.get<RWTexture2D<uint2>>()[pixel] = uint2(0);
    raster.motion.get<RWTexture2D<float2>>()[pixel] = sky_motion(r.view, dir);
    raster.depth[index] = 0.0;
    raster.hzb[0].get<RWTexture2D<float>>()[pixel] = 0.0;
}

// The winner of a vertex's light candidates, resampled as they are drawn by the luminance of their
// light times the density the vertex's BSDF draws their direction with: one stays live, and the
// BSDF is evaluated for it alone.
// What weighs a vertex's light candidates and evaluates the direction its scatter drew: a
// surface's BSDF, or the phase function of a point in a medium.
interface ILeanScatter
{
    bool specular_only();
    // Whether the vertex's light pick leaves the environment out.
    bool no_environment();
    float density(float3 wi);
    float3 eval_and_pdf(float3 wi, out float pdf, out float3 diffuse);
}

extension Bsdf : ILeanScatter
{
    bool no_environment()
    {
        return false;
    }
}

extension WalkExitScatter : ILeanScatter
{
    bool no_environment()
    {
        return false;
    }

    float density(float3 wi)
    {
        return pdf(wi);
    }
}

struct LightPick
{
    // What the vertex's shadow job takes of the winner: where it lies, its radiance, its density
    // in the vertex's mixture, its kind and its light.
    float3 dir;
    float distance;
    float3 radiance;
    float pdf;
    uint kind;
    uint light;
    float3 f;                   // the vertex's BSDF x cos toward it
    float scatter_pdf;          // the vertex's own density of its direction
    float target;               // what it was resampled by
    float sum;
    uint drawn;
    // An emissive winner whose radiance is not yet through its side's coat and fuzz
    // (`ResolvedLightSample.view`), and where it lies in its list and cluster.
    bool view;
    uint position;
    uint triangle;
}

// Draws the light candidates of the vertex at `position` with shading normal `normal`, weighed by
// `b`, where the vertex `estimates`: one directional light with the vertex's directional share,
// else `candidates` from its pixel's tile, and at a path's `first` vertex on the G-buffer's
// surface (`FIRST` 1) the points this frame drew of its screen tile's local sampler for that
// sampler's share of them; a hit's stage holds none of that. One sampler dimension and six per
// candidate, as `nee_draw`. The direction `wi` the vertex's scatter drew, where it `scatters` other
// than by a delta lobe, is evaluated beside the winner's, into `f`, `pdf` and `diffuse`, so the
// vertex holds the BSDF's evaluation once. A point in a medium has a zero `normal`.
LightPick pick_light<let FIRST : uint, B : ILeanScatter>(PathRoot r, B b, float3 position, float3 normal, uint2 pixel, uint candidates, bool estimates, bool first, bool scatters, float3 wi, out float3 f, out float pdf, out float3 diffuse, inout Sampler rng)
{
    LightSetGpu* set = r.set;
    LightPick k;
    k.dir = float3(0.0, 0.0, 1.0);
    k.distance = 0.0;
    k.radiance = float3(0.0);
    k.pdf = 0.0;
    k.kind = LIGHT_SAMPLE_ENVIRONMENT;
    k.light = NO_LIGHT;
    k.f = float3(0.0);
    k.scatter_pdf = 0.0;
    k.target = 0.0;
    k.sum = 0.0;
    k.drawn = 0;
    k.view = false;
    k.position = 0;
    k.triangle = 0;
    f = float3(0.0);
    pdf = 0.0;
    diffuse = float3(0.0);
    uint end = rng.dimension + 1 + 6 * candidates;
    NeeVertex v = nee_vertex(set, position, normal, pixel, nullptr);
    bool local = FIRST == 1 && first && may(MAY_LOCAL);
    if (local)
        v.local = pixel_local_lights(r, pixel);
    bool directional = false;
    if (estimates && !b.specular_only())
    {
        directional = rng.next() < v.share;
        k.drawn = directional ? 1 : candidates;
    }
    // A directional light is drawn alone, and a lone candidate takes no weight: its radiance and
    // the BSDF are evaluated once it is picked.
    if (directional)
    {
        ResolvedLightSample c = nee_directional_draw(set, v, rng);
        if (c.pdf > 0.0)
        {
            k.dir = c.dir;
            k.distance = c.distance;
            k.pdf = c.pdf;
            k.kind = c.kind;
            k.light = c.light;
            k.target = 1.0;
            k.sum = 1.0 / c.pdf;
        }
    }
    for (uint i = 0; i < (directional ? 0 : k.drawn); i++)
    {
        ResolvedLightSample c = nee_tile_candidate<FIRST == 1>(set, v, rng);
        float u = rng.next();
        if (!(c.pdf > 0.0) || (b.no_environment() && c.kind == LIGHT_SAMPLE_ENVIRONMENT))
            continue;
        float target = luminance_rec2020(c.radiance) * b.density(c.dir);
        float w = target / c.pdf;
        if (!(w > 0.0) || !isfinite(w))
            continue;
        k.sum += w;
        if (u * k.sum < w)
        {
            k.dir = c.dir;
            k.distance = c.distance;
            k.radiance = c.radiance;
            k.pdf = c.pdf;
            k.kind = c.kind;
            k.light = c.light;
            k.target = target;
            if (may2(MAY2_COATED_EMITTERS))
            {
                k.view = c.view;
                k.position = c.position;
                k.triangle = c.triangle;
            }
        }
    }
    // The winner's value and the scatter's, through one evaluation of the BSDF in the stage.
    for (uint i = 0; i < 2; i++)
    {
        bool winner = i == 0;
        if (winner ? !(k.sum > 0.0) : !scatters)
            continue;
        float density;
        float3 share;
        float3 value = b.eval_and_pdf(winner ? k.dir : wi, density, share);
        if (winner)
        {
            k.f = value;
            k.scatter_pdf = density;
        }
        else
        {
            f = value;
            pdf = density;
            diffuse = share;
        }
    }
    rng.dimension = end;
    return k;
}

// The light a path's vertex takes from its surface, as the stage that runs the vertex knows it:
// a hit's stage made for emitters weighs it against the light pick of the vertex its ray left,
// the first vertex's callable takes the G-buffer's, and every other stage holds none of it.
interface IVertexLight
{
    // What the surface `g` of the vertex, whose BSDF there is `b`, emits toward the path, whose
    // words are `at` and the first of them `w0`: in nits, for the vertex reached after `scattered`
    // scatters, its path's first where `first` holds. With `whole` the path is of one technique.
    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole);
}

// A stage made for surfaces that emit nothing.
struct NoLight : IVertexLight
{
    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole)
    {
        return float3(0.0);
    }
}

// The G-buffer's surface, which holds its emission pre-exposed.
struct GBufferLight : IVertexLight
{
    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole)
    {
        return b.emission() / r.view.exposure[r.view.exposure_slot].pre_exposure;
    }
}

// A ray's hit, with what light sampling draws of its point (`emitter`). A camera ray's vertex
// shows its light whole. A scattered ray's weighs it against the light pick of the vertex the ray
// left, which takes that vertex's density from the path's words: a point the light set never
// draws takes weight 1, as a ray drawn from a delta lobe does and a path of one technique.
struct HitLight : IVertexLight
{
    EmitterHit emitter;

    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole)
    {
        float3 e = b.emission_layer() * g.bsdf.emission;
        float pdf = asfloat(load_coherent(at + 3)).w;
        if (may(MAY_SCATTERING) && pdf == PDF_LIT)
            return float3(0.0);
        if (first || whole || !(pdf > 0.0) || !(emitter.density > 0.0))
            return e;
        uint4 w1 = load_coherent(at + 1);
        float3 toward = emitter.point - asfloat(w0.xyz);
        float distance = length(toward);
        float cos_to = distance > 0.0 ? dot(emitter.normal, toward / distance) : 0.0;
        // The vertex the ray left drew its candidates with its local sampler where it was the
        // path's first.
        LocalLights* local = nullptr;
        if (may(MAY_LOCAL) && scattered == 1)
            local = pixel_local_lights(r, pixel);
        return e * mis_balance(pdf, halves(w1.w).y * nee_emissive_pdf(r.set, local, left_normal(w0.w, w1), emitter, distance, cos_to));
    }
}

// The vertex `x` of `slot`'s path, whose hot words are `at`, reached along `dir` by a ray `t` m
// long whose cone there is `cone`, in a medium `M`: the rest of the vertex past its surface, which
// the shader of a plain opaque hit runs and, for a path's first vertex, the callable that takes it
// from the G-buffer. `FIRST` says where the vertex is its path's first: 1 always, the
// callable's, and 2 where its ray is the camera's, a hit's, which a view that traces its camera
// rays has. The path's state is loaded here, past the surface's evaluation, which holds enough
// of its own. The vertex takes its surface's light through `L` where its stage may emit.
uint lean_vertex<M : IMedium, let FIRST : uint, L : IVertexLight, V : IVertexKind, let LAYER : bool>(PathRoot r, uint slot, uint4* at, PathVertex x, float3 dir, float t, float2 cone, L emits, V on, uint bits)
{
    // Of the path's words the vertex holds its flags and its throughput. The radiance it adds
    // goes into the path's word at once, and the ray and the shadow job it leaves are written
    // whole, so nothing else of the path is live across the surface's light.
    uint4 w0 = load_coherent(at);
    uint flags = w0.w;
    bool first = FIRST == 1 || (FIRST == 2 && may_any(MAY_TRACED | MAY_LAYERS) && (flags & HOT_CAMERA) != 0);
    // Whether light along the ray counts, as the vertex it left had it.
    bool along = (flags & HOT_COUNTED) != 0;
    bool was_diffuse = (flags & HOT_DIFFUSE) != 0;
    flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float3 throughput = hot_throughput(load_coherent(at + 1));
    // A traced camera ray's path runs on the view's share of pixels, as one that starts on the
    // G-buffer's surface does. One that looks for layers in front of the G-buffer's surface
    // runs on the share its raygen gave it.
    if (FIRST == 2 && first && traces_primary(r))
    {
        if (!runs_path(r, pixel))
        {
            store_coherent(at, uint4(w0.xyz, flags));
            return 0;
        }
        throughput /= r.pixel_fraction;
    }
    bool walked = may_any(MAY_WALK | MAY_SCATTERING) && (bits & PAYLOAD_WALKED) != 0;
    // A surface of a stage that may walk shades its subsurface share as diffuse reflection where
    // the vertex walks none of it.
    bool walks = false;
    if (lean_walks(LAYER))
    {
        walks = !from_update(r) && may_walk(r, x.surface, first ? 0 : 1, walked, cone.x);
        x.surface.bsdf = walk_params(x.surface.bsdf, walks);
    }
    GSurface g = x.surface;
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = flags >> HOT_DIMENSION_SHIFT;
    LeanMode mode = lean_mode(r);
    bool update = from_update(r);
    float3 added = float3(0.0);
    // The interiors the path lies in, which its cold words list with the innermost's extinction.
    uint4* cold = lean_cold(r, slot);
    bool inside = may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0;
    float3 sigma = inside ? interior_sigma(cold) : float3(0.0);

    // The medium along the ray: an interior's, or the linked medium's, whose camera segment the
    // view composes from the froxels.
    float u_segment = rng.next();
    // A walk's step weighed the stretch of the ray inside.
    if (inside)
        throughput *= may(MAY_SCATTERING) && (bits & PAYLOAD_WEIGHED) != 0 ? float3(1.0) : beer(sigma, inside_length(at, t));
    else if (!first)
    {
        float3 origin = asfloat(w0.xyz);
        if (t >= INSCATTER_SEGMENT)
        {
            float3 through;
            float3 e = segment_inscatter<M>(r.set, origin, dir, t, r.segment_steps, u_segment, through);
            if (along)
                added += throughput * e;
            throughput *= through;
        }
        else
            throughput *= medium_transmittance<M>(r.scene.medium, origin, dir, t);
    }
    // A layer is no bounce: its path counts the layers it meets apart.
    uint scattered = (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    uint opaque = scattered + (LAYER ? 0 : 1);
    flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(opaque, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
    bool alive = (g.flags & GBUFFER_UNLIT) == 0 && (first || path_mode(r) != PATH_DIRECT) && carries(throughput);
    Bsdf b = on.prepare(r, x, cold, inside);
    if (may(MAY_EMITTER) && along && (first || mode.count_bsdf) && any(g.bsdf.emission > 0.0))
        added += throughput * emits.light(r, at, b, g, pixel, w0, first, scattered, mode.one_technique);

    // The cache holds the light a vertex scatters by its albedo: a path ends in it past its first
    // vertex where its ray was diffuse or its footprint has outgrown the cell.
    float3 u_cell = rng.next4().xyz;
    // Past the root's dielectric depth a path ends at its layer, with the cache's light for the
    // interior there.
    if (may(MAY_LAYERS) && LAYER && alive)
    {
        uint layers = met_layers(cold) + 1;
        store_met_layers(cold, layers);
        if (layers > r.dielectric_depth && !first)
        {
            if (mode.cache && along)
                added += throughput * interior_ambient(r, g.position, dir, rng);
            alive = false;
        }
    }
    if (alive && mode.cache && !first && !LAYER)
    {
        CacheCell cell = cache_cell(r.cache, r.scene, g.position, g.geometric_normal, u_cell, true);
        bool wide = cone.x > cell.size;
        bool ends = update ? scattered >= 2 : was_diffuse || wide || opaque == r.max_bounces + 2;
        float3 cached;
        if (ends && t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
        {
            if (along)
                added += throughput * cache_albedo(b.albedo(), g.bsdf) * cached;
            alive = false;
        }
    }
    if (any(added != 0.0))
    {
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz + added, gathered.w)));
    }
    // The last bounce's hit scatters nothing the path estimates.
    alive = alive && opaque <= r.max_bounces + 1;
    if (!alive)
    {
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
        return 0;
    }
    if (update)
    {
        UpdateVertex v = { g.position, oct_encode_snorm16(g.geometric_normal), cache_albedo(b.albedo(), g.bsdf) * throughput, 0 };
        uint4* record = lean_record(r, slot) + UPDATE_VERTEX_AT;
        store_coherent(record, uint4(asuint(v.position), v.normal));
        store_coherent(record + 1, uint4(asuint(v.albedo), 0));
        flags |= HOT_ARRIVED;
    }
    bool counted = lean_counted(r, scattered, walked);
    flags = counted ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;

    // One light, resampled from the vertex's candidates, as a pending shadow job.
    float run = nee_probability(r.set, g);
    float u_run = rng.next();
    uint candidates = first ? r.primary_candidates : r.bounce_candidates;
    // The scatter's direction is drawn first: the light candidates' loop evaluates it with them.
    float4 u = rng.next4();
    float u_roulette = rng.next();
    BsdfDraw drawn = b.draw(u.xyz);
    float3 scatter_f;
    float scatter_pdf;
    float3 scatter_diffuse;
    bool estimates = mode.count_nee && counted && candidates > 0 && u_run < run;
    LightPick k = pick_light<FIRST>(r, b, g.position, g.normal, pixel, candidates, estimates, first, drawn.drawn && !drawn.delta, drawn.wi, scatter_f, scatter_pdf, scatter_diffuse, rng);
    // The blue of the job's light and its light id share a word with the ray's cone.
    float job_blue = 0.0;
    uint job_light = NO_LIGHT;
    // A surface takes light only from a side its shading and geometric normals agree on, where
    // its BSDF rays leave too.
    if (k.sum > 0.0 && sides_agree(g.normal, g.geometric_normal, k.dir))
    {
        float3 radiance = k.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(r.set, k.light, k.dir) : k.radiance;
        float W = k.sum / (float(k.drawn) * k.target);
        float mis = mode.one_technique ? 1.0 : mis_balance(run * k.pdf, k.scatter_pdf);
        float3 origin = vertex_origin(x, k.dir);
        bool far = !isfinite(k.distance);
        float3 e = throughput * k.f * radiance * (W * mis / run);
        // The medium's transmittance toward the light: the interior's, which a shadow ray leaves
        // through no surface of, or the linked medium's, whose transmittance the environment's
        // map holds of its own.
        if (inside)
            e *= beer(sigma, k.distance);
        else if (k.kind != LIGHT_SAMPLE_ENVIRONMENT)
            e *= medium_transmittance<M>(r.scene.medium, origin, k.dir, k.distance);
        // The point drawn on the light is the candidate's distance along its direction from the
        // vertex.
        float3 target = far ? k.dir : g.position + k.dir * k.distance;
        if (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && k.view && any(e > 0.0))
            e *= emitter_view(at, k, origin, target, inside);
        if (any(e > 0.0))
        {
            store_coherent(at + 4, uint4(asuint(origin), asuint(e.x)));
            store_coherent(at + 5, uint4(asuint(target), asuint(e.y)));
            job_blue = e.z;
            job_light = k.kind != LIGHT_SAMPLE_DIRECTIONAL ? k.light : NO_LIGHT;
            flags |= HOT_SHADOW | (far ? HOT_SHADOW_FAR : 0);
        }
    }

    // The next ray, or the walk that its subsurface lobe starts into the surface, which the path's
    // cold words take for the raygen to run.
    BsdfSample s = b.drawn_sample(drawn, scatter_f, scatter_pdf, scatter_diffuse);
    uint offset = 0;
    uint left = 0;
    bool goes = s.pdf > 0.0 && sides_agree(g.normal, g.geometric_normal, s.wi);
    if (goes)
    {
        throughput *= s.weight;
        if (opaque > r.roulette_depth && !LAYER)
        {
            float survive = saturate(max_component(throughput));
            goes = u_roulette < survive;
            throughput /= max(survive, 1e-6);
        }
        goes = goes && carries(throughput);
    }
    if (goes)
    {
        // The interiors past the vertex.
        if (may(MAY_INTERIORS) && LAYER)
            flags = on.cross(r, s, cold, inside) ? flags | HOT_INTERIOR : flags & ~HOT_INTERIOR;
        if (lean_walks(LAYER) && walks && (s.lobe & LOBE_VOLUME) != 0)
        {
            store_walk(r, cold, x, s.wi, FIRST == 1, rng);
            left = PAYLOAD_WALKS;
        }
        bool delta = (s.lobe & LOBE_SPECULAR) != 0;
        bool diffuse = (s.lobe & LOBE_DIFFUSE) != 0;
        float roughness = g.bsdf.specular_roughness;
        cone.y += delta ? 0.0 : diffuse ? 1.0 : 2.0 * roughness * roughness;
        flags |= HOT_RAY;
        flags = diffuse ? flags | HOT_DIFFUSE : flags & ~HOT_DIFFUSE;
        float3 origin = vertex_origin(x, s.wi);
        // A ray inside an interior keeps the stretch from the surface to its origin, which the
        // medium covers too.
        if (may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0)
        {
            float3 o = origin - g.position;
            offset = as_halves(float2(dot(o, o), dot(o, s.wi)));
        }
        store_coherent(at, uint4(asuint(origin), hot_flags(flags, rng.dimension)));
        store_coherent(at + 1, uint4(oct_encode_snorm16(s.wi), oct_encode_snorm16(g.normal), as_halves(throughput.xy), as_halves(float2(throughput.z, mode.count_nee ? run : 0.0))));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, delta ? 0.0 : s.pdf)));
    }
    else
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
    store_coherent(at + 2, uint4(as_halves(cone), asuint(job_blue), offset, job_light));
    return left;
}

// Whether a stage's opaque vertices may walk: its BSDF holds the subsurface lobe, which its
// surfaces walk where the scene may.
bool lean_walks(bool layer)
{
    return !layer && class_has(CLASS_SUBSURFACE) && may(MAY_WALK);
}

// Stores in a path's cold words `cold` the subsurface walk vertex `x` starts along `wi`, into its
// surface: on the G-buffer's surface where `gbuffer`, else on a hit's. Draws one sampler dimension
// of `rng`, which keys the walk's own sampler.
void store_walk(PathRoot r, uint4* cold, PathVertex x, float3 wi, bool gbuffer, inout Sampler rng)
{
    GSurface g = x.surface;
    // A walk stays on its copy, which a hit names by its instance.
    uint owner = g.instance;
    uint copy;
    if (!gbuffer && hit_copy(r.scene, g.instance, copy))
        owner = copy;
    store_walk_words(cold, walk_from(x, g.bsdf, wi, rng), owner);
}

// Stores walk `e`, which has taken no step, in a path's cold words `cold`, with what it stays in,
// `owner`: a copy for a subsurface walk, an interior's material for one through its medium. Its
// sampler goes on from the dimension it reached: an interior's walk drew its first direction.
public void store_walk_words(uint4* cold, Walk e, uint owner)
{
    store_coherent(cold + LEAN_WALK_WORD, uint4(asuint(e.origin), owner));
    store_coherent(cold + LEAN_WALK_WORD + 1, asuint(float4(e.volume.extinction, e.volume.anisotropy)));
    store_coherent(cold + LEAN_WALK_WORD + 2, uint4(asuint(e.volume.albedo), asuint(e.ior)));
    store_coherent(cold + LEAN_WALK_WORD + 3, uint4(e.rng.seed, oct_encode_snorm16(e.dir), e.rng.dimension, 0));
}

// The walk a path's cold words `cold` hold, as `store_walk_words` left it, with the frame's
// sampler index `frame`, and what it stays in.
public Walk load_walk(uint4* cold, uint frame, out uint owner)
{
    uint4 a = load_coherent(cold + LEAN_WALK_WORD);
    float4 b = asfloat(load_coherent(cold + LEAN_WALK_WORD + 1));
    float4 c = asfloat(load_coherent(cold + LEAN_WALK_WORD + 2));
    uint4 d = load_coherent(cold + LEAN_WALK_WORD + 3);
    owner = a.w;
    SubsurfaceVolume volume;
    volume.extinction = b.xyz;
    volume.anisotropy = b.w;
    volume.albedo = c.xyz;
    Sampler rng;
    rng.index = frame;
    rng.seed = d.x;
    rng.dimension = d.z;
    return walk_start(asfloat(a.xyz), oct_decode_snorm16(d.y), volume, c.w, rng);
}

// What the coat and fuzz of the emitter that light pick `k` drew pass of its light toward a vertex,
// `inside` an interior, along the shadow ray from `origin` to `target`: its callable evaluates the
// emitter's surface there. The ray reaches it in the shadow job's words `at`, which the job then
// takes back.
float3 emitter_view(uint4* at, LightPick k, float3 origin, float3 target, bool inside)
{
    store_coherent(at + 4, uint4(asuint(origin), k.light));
    store_coherent(at + 5, uint4(asuint(target), k.position));
    Payload p;
    p.words = uint4(k.triangle, inside ? 1 : 0, 0, 0);
    // The compute stand-ins (`path.standins`), compiled for their statistics alone, hold no
    // callables.
    __stage_switch
    {
    case raygen:
    case closesthit:
    case miss:
    case callable:
        call(CALL_EMITTER_VIEW, p);
        break;
    default:
        break;
    }
    return asfloat(p.words.xyz);
}

// The emitter view's callable (`emitter_view`) for `slot`'s path, with its payload's `words`: what
// the coat and fuzz of the surface `S` evaluates there pass toward the vertex, in the medium of the
// interior it lies in, into the first three.
public void lean_emitter_view<S : ISurfaces>(PathRoot r, uint slot, inout uint4 words)
{
    uint4* at = lean_hot(r, slot);
    uint4 a = load_coherent(at + 4);
    uint4 b = load_coherent(at + 5);
    float3 to = asfloat(b.xyz) - asfloat(a.xyz);
    ResolvedLightSample c = no_candidate();
    c.distance = length(to);
    c.dir = to / c.distance;
    c.kind = LIGHT_SAMPLE_EMISSIVE;
    c.light = a.w;
    c.position = b.w;
    c.triangle = words.x;
    c.view = true;
    c.point = asfloat(b.xyz);
    uint medium = words.y != 0 ? innermost_of(load_coherent(lean_cold(r, slot))) : MATERIAL_NONE;
    words.xyz = asuint(emitter_view_layer<S>(r.set, c, interior_ior(r.scene, medium)));
}

// What a vertex's scatter leaves of light pick `k` as the path's shadow job: `e` where its ray from
// `origin`, off the vertex at `point`, `inside` an interior or not, reaches the light. The job's
// blue and light id, which share the cone's word, and its flags in `flags`.
void store_job(uint4* at, LightPick k, float3 origin, float3 point, float3 e, bool inside, inout uint flags, out float job_blue, out uint job_light)
{
    job_blue = 0.0;
    job_light = NO_LIGHT;
    bool far = !isfinite(k.distance);
    float3 target = far ? k.dir : point + k.dir * k.distance;
    if (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && k.view && any(e > 0.0))
        e *= emitter_view(at, k, origin, target, inside);
    if (!any(e > 0.0))
        return;
    store_coherent(at + 4, uint4(asuint(origin), asuint(e.x)));
    store_coherent(at + 5, uint4(asuint(target), asuint(e.y)));
    job_blue = e.z;
    job_light = k.kind != LIGHT_SAMPLE_DIRECTIONAL ? k.light : NO_LIGHT;
    flags |= HOT_SHADOW | (far ? HOT_SHADOW_FAR : 0);
}

// The light of pick `k`, which the vertex's scatter weighs by `k.f`, toward a vertex of
// `throughput` that runs next-event estimation with probability `run`, weighed against its scatter
// unless `weight_one`, before the medium toward the light.
float3 picked_light(PathRoot r, LightPick k, float3 throughput, float run, bool weight_one)
{
    float3 radiance = k.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(r.set, k.light, k.dir) : k.radiance;
    float W = k.sum / (float(k.drawn) * k.target);
    float mis = weight_one ? 1.0 : mis_balance(run * k.pdf, k.scatter_pdf);
    return throughput * k.f * radiance * (W * mis / run);
}

// Survival of Russian roulette past the root's roulette depth, at the `opaque`th scatter, by the
// path's `throughput`, which it divides by its chance: false where the path ends.
bool roulette_survives(PathRoot r, uint opaque, float u, inout float3 throughput)
{
    if (opaque <= r.roulette_depth)
        return true;
    float survive = saturate(max_component(throughput));
    throughput /= max(survive, 1e-6);
    return u < survive;
}

// The scatter of a point in an interior's medium: its phase function at a scatter, of anisotropy
// `anisotropy`, reached along `travel`; or, `leave`, where a walk leaves the interior through a
// surface of outward normal `n`, `eta` times the IOR outside, its way out (`InteriorExitScatter`),
// whose rays take the environment's light themselves.
struct LeanMediumScatter : ILeanScatter
{
    bool leave;
    bool skip_environment;
    float anisotropy;
    float3 travel;
    InteriorExitScatter exit;

    bool specular_only()
    {
        return false;
    }

    bool no_environment()
    {
        return skip_environment;
    }

    float density(float3 wi)
    {
        return leave ? exit.pdf(wi) : hg_phase(anisotropy, dot(travel, wi));
    }

    float3 eval_and_pdf(float3 wi, out float pdf, out float3 diffuse)
    {
        if (leave)
            return exit.eval_and_pdf(wi, pdf, diffuse);
        pdf = density(wi);
        diffuse = float3(0.0);
        return float3(pdf);
    }
}

// The vertex of `slot`'s path in an interior's medium, which its raygen made and weighed the path
// by. At a scatter `distance` m along its ray: in real time the world cache's light for the
// interior there ends the path, as past the dielectric depth; otherwise the vertex draws a light
// weighed by the medium's phase function, whose shadow ray the interior's transmittance dims, and
// its next ray from the phase function. A scatter that walks the interior's dense medium leaves
// the walk in the path's cold words (`MEDIUM_WALKS`) for its raygen to run. Where a walk leaves
// the interior (`MEDIUM_LEAVES`), at its hot origin through a surface of the walk's IOR, the hot
// words hold the way out and the surface's outward normal; where one stops at its cap
// (`MEDIUM_CAPPED`) it leaves where the path entered the interior, along a cosine draw, with the
// subsurface table's share. The path then lies past the surface: in real time the vertex takes the
// lights but the environment as `InteriorExitScatter` gathers them, which its ray along the way
// out meets for nothing (`PDF_LIT`); the reference leaves all light to that ray. The walk was the
// bounce of the vertex where it leaves.
public void lean_medium_vertex(PathRoot r, uint slot, float distance, uint kind, bool walked)
{
    uint4* at = lean_hot(r, slot);
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    uint flags = w0.w;
    bool along = (flags & HOT_COUNTED) != 0;
    flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_DIFFUSE);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    bool leaves = kind == MEDIUM_LEAVES || kind == MEDIUM_CAPPED;
    float3 dir = oct_decode_snorm16(w1.x);
    float3 point = asfloat(w0.xyz) + (leaves ? float3(0.0) : dir * distance);
    float3 normal = leaves ? oct_decode_snorm16(w1.y) : float3(0.0);
    float3 throughput = hot_throughput(w1);
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = flags >> HOT_DIMENSION_SHIFT;
    LeanMode mode = lean_mode(r);
    bool reference = may(MAY_REFERENCE) && (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
    uint4* cold = lean_cold(r, slot);
    uint scattered = (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    if (kind == MEDIUM_WALKS)
    {
        Interior interior = load_interior(cold);
        uint medium = interior.innermost();
        float ior = interior_ior(r.scene, medium) / interior_ior(r.scene, interior.outside(medium, medium));
        float4 scatter = asfloat(load_coherent(cold + LEAN_SCATTER_WORD));
        InteriorMedium im = { interior_sigma(cold), scatter.xyz, scatter.w };
        store_walk_words(cold, walk_interior(point, dir, im, ior, rng), medium);
        flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(scattered + 1, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
        flags = lean_counted(r, scattered, true) ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;
        store_coherent(at, uint4(w0.xyz, hot_flags(flags, rng.dimension)));
        return;
    }
    float ior = 1.0;
    if (leaves)
    {
        uint medium;
        Walk e = load_walk(cold, r.frame, medium);
        ior = e.ior;
        if (kind == MEDIUM_CAPPED)
        {
            uint4 entered = load_coherent(cold + LEAN_ENTERED_WORD);
            point = asfloat(entered.xyz);
            normal = oct_decode_snorm16(entered.w);
            float3 tangent, bitangent;
            orthonormal_basis(normal, tangent, bitangent);
            dir = normalize(from_local(sample_cosine_hemisphere(rng.next2()), tangent, bitangent, normal));
            throughput *= capped_walk_share(r.subsurface_table, e.volume.albedo, e.volume.anisotropy, r.sss_max_steps, ior);
        }
        Interior interior = load_interior(cold);
        interior.cross(medium, false);
        store_interior(r.scene, cold, interior);
        flags = interior.count > 0 ? flags | HOT_INTERIOR : flags & ~HOT_INTERIOR;
        // The way out is a refraction: the walk's Fresnel draw chose it, and it scales radiance
        // as a BSDF's does.
        throughput *= ior * ior;
        // The vertex counts as the walk's scatter did.
        scattered--;
    }
    bool inside = (flags & HOT_INTERIOR) != 0;
    float3 sigma = inside ? interior_sigma(cold) : float3(0.0);
    // A scatter is a bounce; a walk's way out is the bounce its walk took.
    uint opaque = scattered + 1;
    flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(opaque, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
    float2 cone = halves(load_coherent(at + 2).x);
    if (!leaves)
        cone.x += cone.y * distance;
    bool alive = carries(throughput) && opaque <= r.max_bounces + 1;
    if (mode.cache && !leaves)
    {
        if (along && alive)
        {
            float4 gathered = asfloat(load_coherent(at + 3));
            store_coherent(at + 3, asuint(float4(gathered.xyz + throughput * interior_ambient(r, point, dir, rng), gathered.w)));
        }
        alive = false;
    }
    if (!alive)
    {
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
        return;
    }
    bool counted = lean_counted(r, scattered, walked);
    flags = counted ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;
    // A way out's next-event estimation leaves the environment to its ray where that ray counts.
    InteriorExitScatter way_out = { normal, ior, 1.0 };
    if (leaves)
        way_out = interior_exit(normal, ior);
    LeanMediumScatter scatter = { leaves, leaves && mode.count_bsdf, asfloat(load_coherent(cold + LEAN_SCATTER_WORD).w), dir, way_out };
    // The lights a way out's ray meets count for nothing where its vertex took them.
    bool lit = leaves && !reference && mode.count_nee && !mode.one_technique;
    float4 u = rng.next4();
    float u_roulette = rng.next();
    float3 wi = leaves ? dir : sample_hg(scatter.anisotropy, dir, u.xy);
    float3 scatter_f;
    float scatter_pdf;
    float3 scatter_diffuse;
    bool estimates = mode.count_nee && counted && r.bounce_candidates > 0 && !(leaves && reference);
    LightPick k = pick_light<0>(r, scatter, point, normal, pixel, r.bounce_candidates, estimates, false, !leaves, wi, scatter_f, scatter_pdf, scatter_diffuse, rng);
    float job_blue = 0.0;
    uint job_light = NO_LIGHT;
    if (k.sum > 0.0)
    {
        float3 origin = leaves ? surface_ray_origin(r, point, normal, k.dir) : point;
        float3 e = picked_light(r, k, throughput, 1.0, mode.one_technique || leaves) * (inside ? beer(sigma, k.distance) : float3(1.0));
        store_job(at, k, origin, point, e, inside, flags, job_blue, job_light);
    }
    // The phase function draws its own density, so the next ray keeps the throughput; a way out is
    // the direction the walk took.
    bool goes = (leaves || scatter_pdf > 0.0) && (leaves || roulette_survives(r, opaque, u_roulette, throughput)) && carries(throughput);
    // A scatter's next ray leaves from the point itself: no offset for the medium to cover. A way
    // out's leaves the interior, and the medium of an outer one covers its offset too.
    float3 origin = leaves ? surface_ray_origin(r, point, normal, wi) : point;
    uint offset = 0;
    if (leaves && inside)
    {
        float3 o = origin - point;
        offset = as_halves(float2(dot(o, o), dot(o, wi)));
    }
    if (goes)
    {
        cone.y += 1.0;
        flags |= HOT_RAY | (leaves ? HOT_DIFFUSE : HOT_MEDIUM);
        store_coherent(at, uint4(asuint(origin), hot_flags(flags, rng.dimension)));
        store_coherent(at + 1, uint4(oct_encode_snorm16(wi), 0, as_halves(throughput.xy), as_halves(float2(throughput.z, mode.count_nee ? 1.0 : 0.0))));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, leaves ? (lit ? PDF_LIT : 0.0) : scatter_pdf)));
    }
    else
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
    store_coherent(at + 2, uint4(as_halves(cone), asuint(job_blue), offset, job_light));
}

// The origin of a ray along `dir` off a surface at `point` with outward normal `n`, past its rounding.
float3 surface_ray_origin(PathRoot r, float3 point, float3 n, float3 dir)
{
    return ray_origin(point, point, n, dir, ORIGIN_DEFAULT, 0.0, length(point - r.view.eye_render.xyz));
}

// The vertex of `slot`'s path where a subsurface walk leaves through surface `x`, in hot words
// `at` whose throughput holds the walk's, its ray's cone `cone` there: it scatters by the walk's
// exit (`WalkExitScatter`) with the surface's parameters, takes light only along a way its normals
// agree on, by next-event estimation every time, and draws its next ray by the exit's cosine lobe.
// The exit is the walk's bounce, which its entry counted. A walk `capped` at its cap takes the
// subsurface table's share of its throughput here.
void lean_exit_vertex<M : IMedium>(PathRoot r, uint slot, uint4* at, PathVertex x, float2 cone, bool capped)
{
    uint4 w0 = load_coherent(at);
    uint flags = w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_DIFFUSE);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float3 throughput = hot_throughput(load_coherent(at + 1));
    if (capped)
    {
        uint owner;
        Walk e = load_walk(lean_cold(r, slot), r.frame, owner);
        throughput *= capped_walk_share(r.subsurface_table, e.volume.albedo, e.volume.anisotropy, r.sss_max_steps, 1.0);
    }
    GSurface g = x.surface;
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = flags >> HOT_DIMENSION_SHIFT;
    LeanMode mode = lean_mode(r);
    uint opaque = (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    bool counted = lean_counted(r, opaque - 1, true);
    flags = counted ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;
    WalkExitScatter f = walk_exit(x, g.bsdf, rng);
    float4 u = rng.next4();
    float u_roulette = rng.next();
    float3 tangent, bitangent;
    orthonormal_basis(f.n, tangent, bitangent);
    float3 wi = normalize(from_local(sample_cosine_hemisphere(u.xy), tangent, bitangent, f.n));
    float3 scatter_f;
    float scatter_pdf;
    float3 scatter_diffuse;
    bool estimates = mode.count_nee && counted && r.bounce_candidates > 0;
    LightPick k = pick_light<0>(r, f, g.position, g.normal, pixel, r.bounce_candidates, estimates, false, true, wi, scatter_f, scatter_pdf, scatter_diffuse, rng);
    float job_blue = 0.0;
    uint job_light = NO_LIGHT;
    if (k.sum > 0.0 && sides_agree(g.normal, g.geometric_normal, k.dir))
    {
        float3 origin = vertex_origin(x, k.dir);
        float3 e = picked_light(r, k, throughput, 1.0, mode.one_technique);
        bool inside = may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0;
        if (inside)
            e *= beer(interior_sigma(lean_cold(r, slot)), k.distance);
        else if (k.kind != LIGHT_SAMPLE_ENVIRONMENT)
            e *= medium_transmittance<M>(r.scene.medium, origin, k.dir, k.distance);
        store_job(at, k, origin, g.position, e, inside, flags, job_blue, job_light);
    }
    bool goes = scatter_pdf > 0.0 && sides_agree(g.normal, g.geometric_normal, wi);
    if (goes)
    {
        throughput *= scatter_f / scatter_pdf;
        goes = roulette_survives(r, opaque, u_roulette, throughput) && carries(throughput);
    }
    if (goes)
    {
        cone.y += 1.0;
        flags |= HOT_RAY | HOT_DIFFUSE;
        store_coherent(at, uint4(asuint(vertex_origin(x, wi)), hot_flags(flags, rng.dimension)));
        store_coherent(at + 1, uint4(oct_encode_snorm16(wi), oct_encode_snorm16(g.normal), as_halves(throughput.xy), as_halves(float2(throughput.z, mode.count_nee ? 1.0 : 0.0))));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, scatter_pdf)));
    }
    else
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
    store_coherent(at + 2, uint4(as_halves(cone), asuint(job_blue), 0, job_light));
}

// The whole vertex of a plain opaque hit `h` of `slot`'s ray, invoked with the ray's cover
// `payload`, through surface `S` with its section material from `material`, in a medium `M`:
// what the hit shader of such a surface runs. A stage made for emitters holds what their light
// is weighed by, and every other none of it.
public void lean_hit<S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial>(PathRoot r, uint slot, inout uint4 payload, T material, H h)
{
    if (may(MAY_EMITTER))
        lean_hit_as<S, M, H, T, true, false>(r, slot, payload, material, h);
    else
        lean_hit_as<S, M, H, T, false, false>(r, slot, payload, material, h);
}

// `lean_hit` of a hit `h` on a layer of the dielectric class: what the hit shader of such a
// surface runs, its vertex crossing into or out of the interior its material bounds.
// A thin wall's hit invoked with a window payload (`window_payload`) runs no vertex: it hands back
// what the wall lets straight through in the payload's first three words.
public void lean_layer_hit<S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial>(PathRoot r, uint slot, inout uint4 payload, T material, H h)
{
    if (may(MAY_EMITTER))
        lean_hit_as<S, M, H, T, true, true>(r, slot, payload, material, h);
    else
        lean_hit_as<S, M, H, T, false, true>(r, slot, payload, material, h);
}

// `lean_hit` of a stage made for emitters, or not, on a layer or an opaque surface.
void lean_hit_as<S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial, let EMITTER : bool, let LAYER : bool>(PathRoot r, uint slot, inout uint4 payload, T material, H h)
{
    uint4* at = lean_hot(r, slot);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // The ray's cone rides in its cover, so the surface resolves before the path's state loads.
    PathCover c = path_cover(payload);
    float t = h.t();
    float3 dir = h.world_direction();
    RayCone cone = { c.context.cone_width, c.context.cone_spread };
    HitContext context = hit_context(c.context.kind, c.context.depth, cone, t, texture_tap(r.view, pixel, r.frame, r.seed, c.context.depth), c.context.feedback);
    float3 positions[3] = { h.corner(0), h.corner(1), h.corner(2) };
    // An emitter's light is weighed by its triangle as light sampling reads it.
    HitTriangle traced;
    SurfaceHit sh = resolve_hit_traced<FeedbackSurface<S>, T, EMITTER>(r.scene, r.view, dir, h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), positions, false, context, material, traced);
    if (!sh.valid)
    {
        uint4 w0 = load_coherent(at);
        store_coherent(at, uint4(w0.xyz, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
        return;
    }
    if (LAYER && may2(MAY2_THIN_DIELECTRICS) && is_window(payload))
    {
        BsdfParams p = sh.surface.bsdf;
        float exterior = asfloat(payload.y);
        float3 through = bsdf_thin_mirror(p, exterior) ? bsdf_straight_transmission(p, sh.surface.geometric_normal, dir, exterior) : float3(0.0);
        payload.xyz = asuint(through);
        return;
    }
    // A view that traces its camera rays marks them by their context.
    if (may(MAY_TRACED) && !LAYER && c.context.kind == HIT_PRIMARY && traces_primary(r))
        write_primary(r, slot, pixel, sh);
    float2 footprint = float2(cone_width(cone, t), cone.spread);
    PathVertex x;
    OnLayer on = { LayerVertex() };
    if (LAYER)
    {
        on.l = layer_vertex(r.view, sh, dir);
        x = on.l.x;
        // A camera ray's layer is what its pixel shows first.
        if (c.context.kind == HIT_PRIMARY)
            first_layer(r, slot, x.camera_distance);
    }
    else
        x = hit_vertex(r.view, sh, dir);
    uint bits = payload.x;
    // Where a subsurface walk leaves, through the surface its last step met or, at its cap, the
    // one it entered by. A surface of a stage without the subsurface lobe has none to let the
    // walk out by: the path ends there.
    if (may(MAY_WALK) && !lean_walks(LAYER) && (bits & PAYLOAD_EXIT) != 0)
    {
        uint4 w0 = load_coherent(at);
        store_coherent(at, uint4(0, 0, 0, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
        payload.y = 0;
        return;
    }
    bool exits = lean_walks(LAYER) && (bits & PAYLOAD_EXIT) != 0;
    uint left = 0;
    if (exits)
        left = PAYLOAD_EXITS;
    else if (EMITTER)
    {
        HitLight emits = { no_emitter() };
        if (any(sh.surface.bsdf.emission > 0.0))
            emits.emitter = emitter_hit_of(r.set, sh, traced, h.instance_index(), uint(h.cluster_id()));
        if (LAYER)
            left = lean_vertex<M, 2, HitLight, OnLayer, true>(r, slot, at, x, dir, t, footprint, emits, on, bits);
        else
            left = lean_vertex<M, 2, HitLight, OnOpaque, false>(r, slot, at, x, dir, t, footprint, emits, OnOpaque(), bits);
    }
    else if (LAYER)
        left = lean_vertex<M, 2, NoLight, OnLayer, true>(r, slot, at, x, dir, t, footprint, NoLight(), on, bits);
    else
        left = lean_vertex<M, 2, NoLight, OnOpaque, false>(r, slot, at, x, dir, t, footprint, NoLight(), OnOpaque(), bits);
    // The surface the vertex where a walk leaves takes: this one, where a walk leaves here, or
    // where one starts here, this one seen from the ray, for where it stops at its cap.
    if (lean_walks(LAYER) && left != 0)
        store_exit(lean_cold(r, slot), exit_vertex_of(x, dir, !exits));
    payload.y = left;
}

// Stores in a path's cold words `cold` vertex `x`, where a subsurface walk leaves, as the G-buffer
// packs a surface.
void store_exit(uint4* cold, PathVertex x)
{
    GBufferWords w = encode_gsurface(x.surface, true);
    uint4* at = cold + LEAN_EXIT_WORD;
    store_coherent(at, asuint(float4(x.surface.position, x.origin_error)));
    store_coherent(at + 1, asuint(float4(x.plane_point, x.camera_distance)));
    store_coherent(at + 2, w.a);
    store_coherent(at + 3, uint4(w.b, 0, 0));
    store_coherent(at + 4, w.x);
    store_coherent(at + 5, w.y);
}

// The vertex where a subsurface walk of `slot`'s path leaves: what the callable of that vertex
// runs, on the surface its path's cold words hold (`store_exit`): the one the walk's last step met,
// or, `capped` where the walk stopped at its cap, the one it entered, which the vertex that started
// it left there.
public void lean_walk_exit<M : IMedium>(PathRoot r, uint slot, bool capped)
{
    uint4* at = lean_hot(r, slot);
    PathVertex x;
    {
        uint4* from = lean_cold(r, slot) + LEAN_EXIT_WORD;
        float4 a = asfloat(load_coherent(from));
        float4 b = asfloat(load_coherent(from + 1));
        GBufferWords w;
        w.a = load_coherent(from + 2);
        w.b = load_coherent(from + 3).xy;
        w.x = load_coherent(from + 4);
        w.y = load_coherent(from + 5);
        x = PathVertex();
        x.surface = decode_gsurface(w);
        x.surface.position = a.xyz;
        x.origin_error = a.w;
        x.plane_point = b.xyz;
        x.camera_distance = b.w;
        x.view = x.surface.normal;
    }
    lean_exit_vertex<M>(r, slot, at, x, halves(load_coherent(at + 2).x), capped);
}

// Vertex `x` where a subsurface walk leaves, reached along `dir`: its normals face out, along the
// walk's last step, or, `capped` where the walk leaves by the surface it entered, along which `dir`
// reached it, against `dir`, the way it was seen from.
PathVertex exit_vertex_of(PathVertex x, float3 dir, bool capped)
{
    float3 out = capped ? -dir : dir;
    float3 ng = dot(x.surface.geometric_normal, out) < 0.0 ? -x.surface.geometric_normal : x.surface.geometric_normal;
    x.surface.geometric_normal = ng;
    x.surface.normal = dot(x.surface.normal, ng) < 0.0 ? -x.surface.normal : x.surface.normal;
    x.view = x.surface.normal;
    return x;
}

// The payload that asks a thin wall's hit for what it lets straight through, inside a medium of
// IOR `exterior`, for a ray whose cone is `cone`: its first word, which a trace never leaves 0, is 0.
public uint4 window_payload(float exterior, float2 cone)
{
    return uint4(0, asuint(exterior), asuint(cone.x), asuint(cone.y));
}

public bool is_window(uint4 payload)
{
    return payload.x == 0;
}

// Writes into the record of `slot`'s view path that its first vertex is a layer `distance` m from
// the eye.
void first_layer(PathRoot r, uint slot, float distance)
{
    if (!from_view(r))
        return;
    uint4* record = lean_record(r, slot);
    StoredViewLights lights = reinterpret<StoredViewLights>(load_coherent(record));
    lights.first_distance = distance;
    store_coherent(record, reinterpret<uint4>(lights));
}

// The first vertex of `slot`'s path, on the G-buffer's surface of its pixel: the raster's
// surface, which the scene's ray tracing geometry need not hold, so no ray finds it. The callable
// a path starts with.
public void lean_first<M : IMedium>(PathRoot r, uint slot, inout uint4 payload)
{
    uint4* at = lean_hot(r, slot);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    PathVertex x;
    if (!gbuffer_vertex(r.scene, r.view, r.gbuffer, pixel, x))
    {
        uint4 w0 = load_coherent(at);
        store_coherent(at, uint4(w0.xyz, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
        return;
    }
    RayCone cone = camera_cone(r.view);
    float2 footprint = float2(cone_width(cone, x.camera_distance), cone.spread);
    payload.y = lean_vertex<M, 1, GBufferLight, OnOpaque, false>(r, slot, at, x, -x.view, x.camera_distance, footprint, GBufferLight(), OnOpaque(), 0);
    // Where the walk the vertex starts stops at its cap it leaves here.
    if (lean_walks(false) && payload.y == PAYLOAD_WALKS)
        store_exit(lean_cold(r, slot), exit_vertex_of(x, -x.view, true));
}

// A debug view of the first vertex alone (`first_vertex_mode`) of `slot`'s pixel, through surface
// `S`: what the callable that every pixel of such a view calls in place of a path writes.
public void lean_first_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    Sampler rng = path_sampler(r, pixel);
    first_vertex_view<S>(r, pixel, rng);
}

// Adds to a view path's record that light `id` gave it `contribution` of luminance.
void lean_feedback(PathRoot r, uint slot, uint id, float contribution)
{
    if (!from_view(r) || !view_light_counts(contribution))
        return;
    uint4* record = lean_record(r, slot);
    StoredViewLights v = reinterpret<StoredViewLights>(load_coherent(record));
    view_light(v, id, contribution);
    store_coherent(record, reinterpret<uint4>(v));
}

// What `slot`'s path ray that met nothing adds: the environment along it and the directional
// lights' discs, each weighed against next-event estimation at the vertex the ray left.
public void lean_miss<M : IMedium, E : IEnvironment>(PathRoot r, uint slot)
{
    uint4* at = lean_hot(r, slot);
    PathHot p = load_hot(at);
    if (may(MAY_SCATTERING) && (p.flags & HOT_MEDIUM) != 0)
        p.normal = float3(0.0);
    LightSetGpu* set = r.set;
    LeanMode mode = lean_mode(r);
    bool camera = (p.flags & HOT_CAMERA) != 0;
    // A ray inside an interior passes its medium to the end.
    if (may(MAY_INTERIORS) && (p.flags & HOT_INTERIOR) != 0)
        p.throughput *= beer(interior_sigma(lean_cold(r, slot)), positive_infinity());
    p.flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA);
    if (camera && traces_primary(r))
        write_primary_sky(r, item_pixel(r, r.slice_base + slot), p.dir);
    if ((p.flags & HOT_COUNTED) != 0 && (mode.count_bsdf || camera))
    {
        bool weigh = !mode.one_technique && p.pdf > 0.0;
        // A ray whose vertex's next-event estimation took the lights meets them for nothing.
        bool lit = may(MAY_SCATTERING) && p.pdf == PDF_LIT;
        // The camera sees the environment itself; every other ray the map its light is drawn from.
        float3 env = camera ? E.radiance(set.scene.environment, p.dir, r.view.eye_render.xyz) : environment_map(set.environment, p.dir);
        // The vertex the ray left drew its candidates with its local sampler where it was the
        // path's first.
        LocalLights* local = nullptr;
        if (may(MAY_LOCAL) && p.depth() == 1)
            local = pixel_local_lights(r, item_pixel(r, r.slice_base + slot));
        float3 e = env * (weigh ? mis_balance(p.pdf, p.run * nee_environment_pdf(set, local, p.normal, p.dir)) : 1.0);
        lean_feedback(r, slot, feedback_environment(set), luminance_rec2020(p.throughput * e));
        if (may(MAY_DIRECTIONAL))
        {
            for (uint k = 0; k < set.directional_count; k++)
            {
                uint id = set.directional_ids[k];
                float3 d = directional_disc<M>(set, id, p.origin, p.dir);
                if (any(d > 0.0) && !lit)
                    e += d * (weigh ? mis_balance(p.pdf, p.run * nee_directional_pdf(set, p.normal, id, p.dir)) : 1.0);
            }
        }
        p.radiance += p.throughput * e;
    }
    store_hot(at, p);
}

// What `slot`'s shadow ray that met nothing adds: `added`, its job's light times what the
// candidates along it let through.
public void lean_shadow_add(PathRoot r, uint slot, float3 added)
{
    uint4* at = lean_hot(r, slot);
    float4 radiance = asfloat(load_coherent(at + 3));
    radiance.xyz += added;
    store_coherent(at + 3, asuint(radiance));
    uint light = load_coherent(at + 2).w;
    if (light != NO_LIGHT)
        lean_feedback(r, slot, light, luminance_rec2020(added));
}
