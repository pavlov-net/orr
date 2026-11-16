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
import orr_shaders.trace;
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

// The plane stack words of the slice's `slot`th path, where the pipeline may meet a stack.
public uint4* lean_stack(PathRoot r, uint slot)
{
    return r.launch.scratch + slot * LEAN_STACK_WORDS;
}

// Whether a directional light may reach `r`'s paths through a plane stack: the light set holds a
// stack entry and a directional light, the one light a stack bends.
public bool lean_stacks(PathRoot r)
{
    return may(MAY_STACK) && r.set.stack_entries != 0 && r.set.directional_count != 0;
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
    store_list(at, i);
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

// The medium of the innermost interior of list `i`, none where it is empty, and `store_interior`
// of the list with it where `stores`: what a vertex that leaves an interior or looks past its
// surface takes of the list outside, its medium drawn once.
InteriorMedium outside_medium(SceneRoot* s, uint4* at, Interior i, bool stores)
{
    InteriorMedium im = { float3(0.0), float3(0.0), 0.0 };
    if (i.count > 0)
    {
        uint m = i.innermost();
        im = interior_medium(s, m, i.point(m));
    }
    if (!stores)
        return im;
    store_list(at, i);
    uint4* last = at + 1 + INTERIOR_ENTRIES;
    store_coherent(last, uint4(load_coherent(last).x, asuint(im.sigma_t)));
    if (may(MAY_SCATTERING))
        store_coherent(at + LEAN_SCATTER_WORD, asuint(float4(im.sigma_s, im.anisotropy)));
    return im;
}

// Stores interior list `i` in words `at`: its entries, then where each was entered.
public void store_list(uint4* at, Interior i)
{
    store_coherent(at, i.entries);
    for (uint k = 0; k < INTERIOR_ENTRIES; k++)
        store_coherent(at + 1 + k, asuint(i.points[k]));
}

// `store_interior` of list `i` whose innermost interior's medium is `im`.
void store_interior_medium(uint4* at, Interior i, InteriorMedium im)
{
    store_list(at, i);
    uint4* last = at + 1 + INTERIOR_ENTRIES;
    store_coherent(last, uint4(load_coherent(last).x, asuint(im.sigma_t)));
    if (may(MAY_SCATTERING))
        store_coherent(at + LEAN_SCATTER_WORD, asuint(float4(im.sigma_s, im.anisotropy)));
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
// words say whether its camera ray takes their medium, which it returns: where it lies in one and
// the frame has no froxels, which otherwise hold the ray's segment as media apply composes it, and
// its first vertex takes the interiors on (HOT_INTERIOR_HELD). A path behind marks its offset word for
// its first vertex, which takes no medium along the camera ray: what lies before the surface, its
// layers and their interiors' media, is the layer chain's to weigh. The callable a camera inside an
// interior, and a frosted pixel's second path, start with.
public bool lean_eye_interior<S : ISurfaces>(PathRoot r, uint slot, bool behind)
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
    if (behind)
    {
        uint4 w2 = load_coherent(at + 2);
        store_coherent(at + 2, uint4(w2.xy, 1, w2.w));
    }
    uint4 w0 = load_coherent(at);
    bool carries = interior.count > 0 && !has_froxels(r);
    uint flags = w0.w & ~(HOT_INTERIOR | HOT_INTERIOR_HELD);
    if (interior.count > 0)
        flags |= carries ? HOT_INTERIOR : HOT_INTERIOR_HELD;
    store_coherent(at, uint4(w0.xyz, flags));
    return carries;
}

// Starts `slot`'s path in the interiors at the point of a thin lens its camera ray leaves, its hot
// origin (`lens_interior`): its cold words empty but for them, and its hot words saying it lies in
// one. The callable a thin lens's path starts with where the pipeline's paths may lie in one.
public void lean_lens_interior<S : ISurfaces>(PathRoot r, uint slot)
{
    uint4* cold = lean_cold(r, slot);
    uint4* at = lean_hot(r, slot);
    store_coherent(cold + 1 + INTERIOR_ENTRIES, uint4(0));
    uint4 w0 = load_coherent(at);
    Interior interior = lens_interior<S>(r, asfloat(w0.xyz));
    store_interior(r.scene, cold, interior);
    if (interior.count > 0)
        store_coherent(at, uint4(w0.xyz, w0.w | (has_froxels(r) ? HOT_INTERIOR_HELD : HOT_INTERIOR)));
}

// A vertex's light lies in the interior the path does, which its cold words list.
static const uint PATH_MEDIUM = MATERIAL_NONE - 1;

// What a layer's vertex meets past its surface, for the light that comes through it and for its
// scatter where that crosses it: the innermost interior of the list past the crossing, MATERIAL_NONE
// where there is none, and its medium, drawn once for both.
struct LayerPast
{
    bool light;
    bool scatter;
    uint innermost;
    InteriorMedium medium;
}

// What sets a vertex's BSDF apart by what its surface is, as the stage that runs it knows it: an
// opaque surface, whose path passes no surface past it, or a layer of the dielectric class, whose
// transmission crosses into or out of the interior its section material bounds.
interface IVertexKind
{
    // The vertex `x`'s BSDF, its path inside the interiors its cold words `cold` list where
    // `inside`.
    Bsdf prepare(PathRoot r, PathVertex x, uint4* cold, bool inside);
    // Whether light toward `dir`, where `light`, and scatter `s` cross the surface into the
    // interiors past it, and those interiors' medium.
    LayerPast meet(PathRoot r, uint4* cold, bool inside, float3 dir, bool light, BsdfSample s);
    // The interiors past a scatter `s` of the vertex, in `cold`, which `past` met: whether the
    // path lies in one.
    bool cross(PathRoot r, BsdfSample s, uint4* cold, bool inside, LayerPast past);
    // Whether light reaching the vertex comes through an interior, whose extinction goes in
    // `sigma`: the path's own, whose extinction `sigma` holds where `inside`, but past a surface
    // the light crosses, as `past` met it. That interior's medium goes in `medium`: MATERIAL_NONE
    // where the light comes through none, PATH_MEDIUM where through the path's own.
    bool toward_light(bool inside, LayerPast past, inout float3 sigma, out uint medium);
    // The interiors past the surface, where light comes through it into the vertex's side.
    Interior past(PathRoot r, uint4* cold, bool inside);
    // Whether scatter `s` crosses a plane stack face ideally into or out of an interior, and the
    // face's unit normal.
    bool crosses_stack(PathRoot r, BsdfSample s, out float3 face);
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

    LayerPast meet(PathRoot r, uint4* cold, bool inside, float3 dir, bool light, BsdfSample s)
    {
        LayerPast past = { false, false, MATERIAL_NONE, InteriorMedium() };
        return past;
    }

    bool cross(PathRoot r, BsdfSample s, uint4* cold, bool inside, LayerPast past)
    {
        return inside;
    }

    bool toward_light(bool inside, LayerPast past, inout float3 sigma, out uint medium)
    {
        medium = inside ? PATH_MEDIUM : MATERIAL_NONE;
        return inside;
    }

    Interior past(PathRoot r, uint4* cold, bool inside)
    {
        return Interior();
    }

    bool crosses_stack(PathRoot r, BsdfSample s, out float3 face)
    {
        face = float3(0.0);
        return false;
    }
}

// A layer `l`, prepared with the media on its two sides as the path's interiors give them, on a
// plane stack face of unit normal `face` where `stack`, which a path's ray that may weigh a
// directional light against next-event estimation through a stack asks.
struct OnLayer : IVertexKind
{
    LayerVertex l;
    bool stack;
    float3 face;

    Bsdf prepare(PathRoot r, PathVertex x, uint4* cold, bool inside)
    {
        return prepare_vertex(r.scene, l, inside ? load_interior(cold) : Interior(), false, false, false, 0.0);
    }

    LayerPast meet(PathRoot r, uint4* cold, bool inside, float3 dir, bool light, BsdfSample s)
    {
        bool crosses = crosses_interior(r.scene, l.section);
        LayerPast past;
        // Normals keep the material's front side, which `front` says the path is on.
        past.light = crosses && light && (dot(dir, l.x.surface.geometric_normal) > 0.0) != l.front;
        past.scatter = crosses && (s.lobe & LOBE_TRANSMISSION) != 0;
        past.innermost = MATERIAL_NONE;
        past.medium = InteriorMedium();
        if (past.light || past.scatter)
        {
            Interior interior = this.past(r, cold, inside);
            past.innermost = interior.innermost();
            past.medium = interior_medium(r.scene, past.innermost, interior.point(past.innermost));
        }
        return past;
    }

    bool cross(PathRoot r, BsdfSample s, uint4* cold, bool inside, LayerPast past)
    {
        if (!past.scatter)
            return inside;
        Interior interior = inside ? load_interior(cold) : Interior();
        interior.cross(l.section, l.front, l.x.surface.medium_point);
        store_interior_medium(cold, interior, past.medium);
        // A dense walk stopped at its cap leaves where the path entered the interior it walked.
        if (may(MAY_SCATTERING))
        {
            float3 n = l.x.surface.geometric_normal;
            store_coherent(cold + LEAN_ENTERED_WORD, l.front ? uint4(asuint(l.x.surface.position), oct_encode_snorm16(n)) : uint4(NO_ENTRY));
        }
        return interior.count > 0;
    }

    bool toward_light(bool inside, LayerPast past, inout float3 sigma, out uint medium)
    {
        medium = inside ? PATH_MEDIUM : MATERIAL_NONE;
        if (!past.light)
            return inside;
        medium = past.innermost;
        sigma = past.medium.sigma_t;
        return medium != MATERIAL_NONE;
    }

    Interior past(PathRoot r, uint4* cold, bool inside)
    {
        Interior interior = inside ? load_interior(cold) : Interior();
        interior.cross(l.section, l.front, l.x.surface.medium_point);
        return interior;
    }

    bool crosses_stack(PathRoot r, BsdfSample s, out float3 face)
    {
        face = this.face;
        uint ideal = LOBE_SPECULAR | LOBE_TRANSMISSION;
        return stack && (s.lobe & ideal) == ideal && crosses_interior(r.scene, l.section);
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
    uint4* cold;

    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole)
    {
        float3 e = b.emission_layer() * g.bsdf.emission;
        float pdf = asfloat(load_coherent(at + 3)).w;
        if (may(MAY_SCATTERING) && pdf == PDF_LIT)
            return float3(0.0);
        if (first || whole || !(pdf > 0.0) || !(emitter.density > 0.0))
            return e;
        uint4 w1 = load_coherent(at + 1);
        // A ray ideal windows passed on measures the light from where its vertex left.
        float3 from = asfloat(w0.xyz);
        if (may2(MAY2_THIN_DIELECTRICS) && (w0.w & HOT_WINDOW) != 0)
            from = asfloat(load_coherent(cold + LEAN_FROM_WORD).xyz);
        float3 toward = emitter.point - from;
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
// of its own. The vertex takes its surface's light through `L` where its stage may emit. A layer's
// vertex says in `exterior` the IOR of the medium its shadow job's ray lies in, which the walls
// it crosses take as their outside: past the layer, or on the side the path's ray left; 0 is the
// path's own.
uint lean_vertex<M : IMedium, let FIRST : uint, L : IVertexLight, V : IVertexKind, let LAYER : bool>(PathRoot r, uint slot, uint4* at, PathVertex x, float3 dir, float t, float2 cone, L emits, V on, uint4 input, out float exterior)
{
    exterior = 0.0;
    // A re-run (`PAYLOAD_BENT`) weighs the light of the job the vertex left where a plane stack
    // bent it, from the words the vertex started with: its scatter alone, toward the light's bent
    // direction.
    bool bent = false;
    if (may(MAY_STACK))
        bent = input.y == PAYLOAD_BENT;
    // Of the path's words the vertex holds its flags and its throughput. The radiance it adds
    // goes into the path's word at once, and the ray and the shadow job it leaves are written
    // whole, so nothing else of the path is live across the surface's light.
    uint4 w0 = load_coherent(at);
    if (may(MAY_STACK))
    {
        if (bent)
            w0 = load_coherent(lean_stack(r, slot) + STACK_BENT_WORD);
    }
    uint flags = w0.w;
    bool first = FIRST == 1 || (FIRST == 2 && may_any(MAY_TRACED | MAY_LAYERS | MAY_LENS | MAY_BACKGROUND) && (flags & HOT_CAMERA) != 0);
    // Whether light along the ray counts, as the vertex it left had it.
    bool along = (flags & HOT_COUNTED) != 0;
    bool was_diffuse = (flags & HOT_DIFFUSE) != 0;
    flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_WINDOW);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float3 throughput = hot_throughput(load_coherent(at + 1));
    if (may(MAY_STACK))
    {
        if (bent)
            throughput = float3(1.0);
    }
    // A traced camera ray's path runs on the view's share of pixels, as one that starts on the
    // G-buffer's surface does, and so does a thin lens's at an opaque surface outside every
    // interior. One that looks for layers in front of the G-buffer's surface runs on the share its
    // raygen gave it.
    bool lens = may_any(MAY_LENS | MAY_BACKGROUND) && !LAYER && (thin_lens(r) || from_background(r)) && (flags & HOT_INTERIOR) == 0;
    if (FIRST == 2 && first && (traces_primary(r) || lens))
    {
        if (!runs_path(r, pixel))
        {
            store_coherent(at, uint4(w0.xyz, flags));
            return 0;
        }
        throughput /= r.pixel_fraction;
    }
    bool walked = may_any(MAY_WALK | MAY_SCATTERING) && (input.x & PAYLOAD_WALKED) != 0;
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
    // A re-run adds no light of the vertex's own, ends in no cache and draws no light.
    bool cache = mode.cache;
    bool count_nee = mode.count_nee;
    if (may(MAY_STACK))
    {
        if (bent)
        {
            along = false;
            cache = false;
            count_nee = false;
        }
    }
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
        throughput *= may(MAY_SCATTERING) && (input.x & PAYLOAD_WEIGHED) != 0 ? float3(1.0) : beer(sigma, inside_length(at, t));
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
    // A camera ray whose segment the froxels hold took none of the medium of the interiors around
    // the eye, but its vertex lies in them.
    if (may(MAY_INTERIORS) && may(MAY_FROXELS) && (flags & HOT_INTERIOR_HELD) != 0)
    {
        inside = true;
        sigma = interior_sigma(cold);
        flags = (flags | HOT_INTERIOR) & ~HOT_INTERIOR_HELD;
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
        if (!bent)
        {
            uint layers = met_layers(cold) + 1;
            store_met_layers(cold, layers);
            if (layers > r.dielectric_depth && !first)
            {
                if (cache && along)
                    added += throughput * interior_ambient(r, g.position, dir, rng);
                alive = false;
            }
        }
    }
    if (alive && cache && !first && !LAYER)
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
    // The cache holds the light of opaque surfaces: a layer's vertex adds to no cell.
    if (update && !LAYER)
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
    // A re-run evaluates its scatter at the bent direction through the same site.
    BsdfDraw drawn = b.draw(u.xyz);
    bool evaluates = drawn.drawn && !drawn.delta;
    float3 wi = drawn.wi;
    if (may(MAY_STACK))
    {
        if (bent)
        {
            evaluates = true;
            wi = bent_direction(lean_stack(r, slot));
        }
    }
    float3 scatter_f;
    float scatter_pdf;
    float3 scatter_diffuse;
    bool estimates = count_nee && counted && candidates > 0 && u_run < run;
    LightPick k = pick_light<FIRST>(r, b, g.position, g.normal, pixel, candidates, estimates, first, evaluates, wi, scatter_f, scatter_pdf, scatter_diffuse, rng);
    if (may(MAY_STACK))
    {
        if (bent)
        {
            bent_add(lean_stack(r, slot), at, scatter_f, scatter_pdf, sides_agree(g.normal, g.geometric_normal, wi));
            return 0;
        }
    }
    // The next ray, or the walk that its subsurface lobe starts into the surface, which the path's
    // cold words take for the raygen to run.
    BsdfSample s = b.drawn_sample(drawn, scatter_f, scatter_pdf, scatter_diffuse);
    LayerPast past = on.meet(r, cold, inside, k.dir, k.sum > 0.0, s);
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
        // The medium's transmittance toward the light: the interior's on the light's side of the
        // vertex, which a shadow ray leaves through no surface of, or the linked medium's, whose
        // transmittance the environment's map holds of its own. A directional light may come
        // through a plane stack, whose walk takes the interior's medium along its legs.
        bool stacked = false;
        if (may(MAY_STACK))
            stacked = k.kind == LIGHT_SAMPLE_DIRECTIONAL && r.set.stack_entries != 0;
        float3 toward = sigma;
        uint medium;
        bool through = on.toward_light(inside, past, toward, medium);
        if (through)
        {
            if (!stacked)
                e *= beer(toward, k.distance);
        }
        else if (k.kind != LIGHT_SAMPLE_ENVIRONMENT)
            e *= medium_transmittance<M>(r.scene.medium, origin, k.dir, k.distance);
        // The point drawn on the light is the candidate's distance along its direction from the
        // vertex.
        float3 target = far ? k.dir : g.position + k.dir * k.distance;
        if (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && k.view && any(e > 0.0))
            e *= emitter_view(at, k, origin, target, medium);
        if (any(e > 0.0))
        {
            store_coherent(at + 4, uint4(asuint(origin), asuint(e.x)));
            store_coherent(at + 5, uint4(asuint(target), asuint(e.y)));
            job_blue = e.z;
            job_light = k.kind != LIGHT_SAMPLE_DIRECTIONAL ? k.light : NO_LIGHT;
            flags |= HOT_SHADOW | (far ? HOT_SHADOW_FAR : 0);
            // A layer's job may lie past the layer, or where the path's ray leaves for.
            if (LAYER && may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
                exterior = interior_ior(r.scene, medium == PATH_MEDIUM ? innermost_of(load_coherent(cold)) : medium);
            if (may(MAY_STACK))
            {
                if (stacked && through)
                {
                    job_light = STACK_JOB_LIGHT;
                    // A layer's scatter may cross into the interiors past it, which its cold
                    // words then list: its job keeps its own list.
                    Interior list = Interior();
                    if (LAYER)
                        list = medium == PATH_MEDIUM ? load_interior(cold) : on.past(r, cold, inside);
                    stack_job(lean_stack(r, slot), LAYER ? 0 : STACK_COLD_LIST, list, g.position, w0, load_coherent(at + 1), input, t, FIRST == 1 ? BENT_FIRST : BENT_HIT, throughput * radiance * (W / run), mode.one_technique ? -1.0 : run * k.pdf);
                }
            }
        }
    }

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
        // A ray that crosses a plane stack face ideally past a vertex that scattered it goes on
        // weighed as that vertex's ray against next-event estimation toward a directional light.
        uint left_normal = oct_encode_snorm16(g.normal);
        bool stacks = false;
        if (LAYER && may(MAY_STACK))
        {
            float3 face;
            if (on.crosses_stack(r, s, face))
                stacks = stack_track(r, slot, at, w0, face, left_normal, inside ? interior_ior(r.scene, innermost_of(load_coherent(cold))) : AMBIENT_IOR);
        }
        // The interiors past the vertex.
        if (may(MAY_INTERIORS) && LAYER)
            flags = on.cross(r, s, cold, inside, past) ? flags | HOT_INTERIOR : flags & ~HOT_INTERIOR;
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
        uint4 w1 = load_coherent(at + 1);
        float ray_run = count_nee ? run : 0.0;
        float ray_pdf = stacks ? PDF_STACK : delta ? 0.0 : s.pdf;
        // An ideal window passes the ray on as it reached it, weighed against next-event
        // estimation as the vertex before it left it, which took the lights past the window
        // through it.
        if (LAYER && may2(MAY2_THIN_DIELECTRICS) && delta && (s.lobe & LOBE_TRANSMISSION) != 0 && thin_walled(g))
        {
            uint kept = HOT_DIFFUSE | HOT_MEDIUM | HOT_COUNTED;
            flags = (flags & ~kept) | (w0.w & kept) | HOT_WINDOW;
            left_normal = w1.y;
            ray_run = halves(w1.w).y;
            ray_pdf = asfloat(load_coherent(at + 3)).w;
            if ((w0.w & HOT_WINDOW) == 0)
                store_coherent(cold + LEAN_FROM_WORD, uint4(w0.xyz, 0));
        }
        float3 origin = vertex_origin(x, s.wi);
        // A ray inside an interior keeps the stretch from the surface to its origin, which the
        // medium covers too.
        if (may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0)
        {
            float3 o = origin - g.position;
            offset = as_halves(float2(dot(o, o), dot(o, s.wi)));
        }
        store_coherent(at, uint4(asuint(origin), hot_flags(flags, rng.dimension)));
        store_coherent(at + 1, uint4(oct_encode_snorm16(s.wi), left_normal, as_halves(throughput.xy), as_halves(float2(throughput.z, ray_run))));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, ray_pdf)));
    }
    else
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
    store_coherent(at + 2, uint4(as_halves(cone), asuint(job_blue), offset, job_light));
    return left;
}

// Leaves in stack words `at` what the walk of a directional job through a plane stack takes of the
// vertex at `point`: the job's ray starts in the interiors of `list`, whose medium its light
// leaves out, or as `cold` says in those of the path's cold list, which the walk copies
// (`STACK_COLD_*`). The light
// may bend at the vertex, and the words keep what weighs it again there: the words the vertex
// started from, `w0` and `w1`, its `input`, the length `t` of the ray that reached it and its kind
// (BENT_*), and of the job its light before the vertex's scatter, `light`, and its density against
// the vertex's own rays, `pdf`, negative where it weighs 1.
void stack_job(uint4* at, uint cold, Interior list, float3 point, uint4 w0, uint4 w1, uint4 input, float t, uint kind, float3 light, float pdf)
{
    if (cold == 0)
        store_list(at + STACK_LIST_WORD, list);
    store_coherent(at + STACK_FROM_WORD, uint4(asuint(point), cold));
    uint4* bent = at + STACK_BENT_WORD;
    store_coherent(bent, w0);
    store_coherent(bent + 1, input);
    store_coherent(bent + 2, uint4(w1.xy, asuint(t), kind));
    store_coherent(bent + 3, uint4(asuint(light), asuint(pdf)));
}

// The direction toward the light at the vertex of the bent job whose stack words are `at`, as its
// walk bent it.
float3 bent_direction(uint4* at)
{
    return asfloat(load_coherent(at + STACK_BEND_WORD).xyz);
}

// Adds to the path in hot words `hot` the light of its bent job, whose stack words are `at`, which
// the vertex's scatter gives `f` toward the bent direction, at density `pdf` there, where the
// light comes from a side it takes light from (`sides`): through the stack's density at the
// vertex, against the job's own.
void bent_add(uint4* at, uint4* hot, float3 f, float pdf, bool sides)
{
    if (!sides)
        return;
    float density = asfloat(load_coherent(at + STACK_BEND_WORD).w);
    float4 light = asfloat(load_coherent(at + STACK_BENT_WORD + 3));
    float4 through = asfloat(load_coherent(at + STACK_THROUGH_WORD));
    float mis = light.w < 0.0 ? 1.0 : mis_balance(light.w, pdf * density);
    float4 radiance = asfloat(load_coherent(hot + 3));
    store_coherent(hot + 3, asuint(float4(radiance.xyz + light.xyz * f * through.xyz * (through.w * mis), radiance.w)));
}

// Whether the ray of `slot`'s path in hot words `at`, whose first word is `w0`, that a layer's
// vertex passes on across a plane stack face of unit normal `face`, which `code` encodes, from a
// medium of IOR `ior`, is
// weighed on as the ray that reached the face toward a directional light: one that a vertex's
// scatter left, whose ray the stack words then keep, or one that crossed parallel faces past one,
// fewer than STACK_CROSSINGS. Its density word says so (`PDF_STACK`).
bool stack_track(PathRoot r, uint slot, uint4* at, uint4 w0, float3 face, uint code, float ior)
{
    if (lean_mode(r).one_technique)
        return false;
    uint4 w1 = load_coherent(at + 1);
    float pdf = asfloat(load_coherent(at + 3)).w;
    uint4* track = lean_stack(r, slot) + STACK_TRACK_WORD;
    if (pdf == PDF_STACK)
    {
        uint4 k = load_coherent(track + 1);
        uint crossed = (k.w >> 16) & 0x7;
        if (crossed == STACK_CROSSINGS || !(abs(dot(face, oct_decode_snorm16(k.z))) >= STACK_PARALLEL))
            return false;
        store_coherent(track + 1, uint4(k.xyz, (k.w & ~(0x7u << 16)) | ((crossed + 1) << 16)));
        return true;
    }
    if (!(pdf > 0.0))
        return false;
    // A ray ideal windows passed on left its vertex from the point the cold words keep.
    float3 origin = asfloat(w0.xyz);
    if (may2(MAY2_THIN_DIELECTRICS) && (w0.w & HOT_WINDOW) != 0)
        origin = asfloat(load_coherent(lean_cold(r, slot) + LEAN_FROM_WORD).xyz);
    bool medium = may(MAY_SCATTERING) && (w0.w & HOT_MEDIUM) != 0;
    uint depth = min((w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK, 0x7fu);
    // The vertex's density of the ray, over the squared IOR it leaves into, which a directional
    // light's density at the vertex through the stack divides by.
    store_coherent(track, uint4(asuint(origin), asuint(pdf / (ior * ior))));
    store_coherent(track + 1, uint4(w1.x, w1.y, code, (w1.w >> 16) | (1u << 16) | (medium ? 1u << 23 : 0u) | (depth << 24)));
    return true;
}

// The density of a directional light's direction `dir` that the ray of `slot`'s path met past
// plane stack faces (`PDF_STACK`) at the vertex that scattered it, which next-event estimation
// there reaches the light through the stack with: the vertex's over the squared IOR it left into
// times the stack's density of `dir` there. That vertex's probability of running next-event
// estimation goes in `run`, its shading normal in `normal`, zero in a medium.
float stack_density(PathRoot r, uint slot, float3 dir, out float run, out float3 normal)
{
    uint4* track = lean_stack(r, slot) + STACK_TRACK_WORD;
    uint4 a = load_coherent(track);
    uint4 b = load_coherent(track + 1);
    float3 face = oct_decode_snorm16(b.z);
    normal = (b.w & (1u << 23)) != 0 ? float3(0.0) : oct_decode_snorm16(b.y);
    run = halves(b.w).x;
    return asfloat(a.w) * abs(dot(dir, face)) / abs(dot(oct_decode_snorm16(b.x), face));
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
// along the shadow ray from `origin` to `target` in the interior of `medium` (MATERIAL_NONE
// outside every interior, PATH_MEDIUM in the path's own): its callable evaluates the emitter's
// surface there. The ray reaches it in the shadow job's words `at`, which the job then takes
// back.
float3 emitter_view(uint4* at, LightPick k, float3 origin, float3 target, uint medium)
{
    store_coherent(at + 4, uint4(asuint(origin), k.light));
    store_coherent(at + 5, uint4(asuint(target), k.position));
    Payload p;
    p.words = uint4(k.triangle, medium, 0, 0);
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
    uint medium = words.y == PATH_MEDIUM ? innermost_of(load_coherent(lean_cold(r, slot))) : words.y;
    words.xyz = asuint(emitter_view_layer<S>(r.set, c, interior_ior(r.scene, medium)));
}

// What a vertex's scatter leaves of light pick `k` as the path's shadow job: `e` where its ray from
// `origin`, off the vertex at `point`, in the interior of `medium` or none, reaches the light. The
// job's blue and light id, which share the cone's word, and its flags in `flags`.
void store_job(uint4* at, LightPick k, float3 origin, float3 point, float3 e, uint medium, inout uint flags, out float job_blue, out uint job_light)
{
    job_blue = 0.0;
    job_light = NO_LIGHT;
    bool far = !isfinite(k.distance);
    float3 target = far ? k.dir : point + k.dir * k.distance;
    if (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && k.view && any(e > 0.0))
        e *= emitter_view(at, k, origin, target, medium);
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
// unless `weight_one`, before the medium toward the light; in `before`, what it is before the
// vertex's scatter and that weight.
float3 picked_light(PathRoot r, LightPick k, float3 throughput, float run, bool weight_one, out float3 before)
{
    float3 radiance = k.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(r.set, k.light, k.dir) : k.radiance;
    float W = k.sum / (float(k.drawn) * k.target);
    float mis = weight_one ? 1.0 : mis_balance(run * k.pdf, k.scatter_pdf);
    before = throughput * radiance * (W / run);
    return before * k.f * mis;
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

// The share of a rough way out's ways in that `rough_way_in` draws evenly over the hemisphere:
// the multiple scattering's transmission reaches past those the visible normals refract.
static const float EVEN_WAYS_IN = 0.1;

// The scatter of a point in an interior's medium: its phase function at a scatter, of anisotropy
// `anisotropy`, reached along `travel`; where a walk leaves the interior through a surface of
// outward normal `n`, `eta` times the IOR outside, its way out (`InteriorExitScatter`), whose rays
// take the environment's light themselves; or, `rough`, where a walk meets or leaves through a
// rough surface, the surface's transmission out of the medium (`InteriorRoughTransmission`).
// Where the walk's last step reached the surface along `travel` from a scatter it reached along
// `into`, `depth` below the surface (`conditioned`), the way out weighs light by what that scatter
// sends through the surface's plane toward it over what it sent along the step, whose throughput
// was divided by the step's chance and multiplied by the least dense channel's transmittance over
// it, exp(-`shift`), and the medium's `extinction` takes the rest. Through a smooth surface the
// light comes along the refraction of its direction; through a rough one of GGX `alpha` below
// `INTERIOR_EXACT_ROUGHNESS` along a way in drawn from it with numbers `u`, and the step's draw is
// the smooth surface's exit lobe from inside times the surface's `draws` where the walk left,
// which a way out no scatter started weighs the surface's transmission by alone.
struct LeanMediumScatter : ILeanScatter
{
    bool leave;
    bool rough;
    bool skip_environment;
    float anisotropy;
    float3 travel;
    InteriorExitScatter exit;
    InteriorRoughTransmission surface;
    bool conditioned;
    float3 into;
    float depth;
    float shift;
    float3 extinction;
    float alpha;
    float draws;
    float3 u;

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
        if (rough && !conditioned)
            return surface.pdf(wi);
        return leave ? exit.pdf(wi) : hg_phase(anisotropy, dot(travel, wi));
    }

    float3 eval_and_pdf(float3 wi, out float pdf, out float3 diffuse)
    {
        diffuse = float3(0.0);
        if (!leave && !rough)
        {
            pdf = density(wi);
            return float3(pdf);
        }
        // Through a conditioned way out, the way inside that leaves toward `wi` and what the light
        // along it takes over the walk's step.
        float own = dot(travel, exit.n);
        float3 toward = float3(0.0);
        float3 weight;
        if (rough)
        {
            InteriorRoughTransmission seen = surface;
            float drawn = 1.0;
            if (conditioned)
            {
                float3 tangent, bitangent;
                orthonormal_basis(exit.n, tangent, bitangent);
                float3 s = float3(dot(wi, tangent), dot(wi, bitangent), dot(wi, exit.n));
                if (!(s.z > 0.0))
                {
                    pdf = 0.0;
                    return float3(0.0);
                }
                float3 r = rough_way_in(alpha, exit.eta, s, u, EVEN_WAYS_IN, drawn);
                toward = -(r.x * tangent + r.y * bitangent + r.z * exit.n);
                seen = InteriorRoughTransmission(alpha, 1.0 / exit.eta, -exit.n, -toward);
            }
            float value = seen.value(wi, pdf) * draws;
            if (!conditioned)
                return float3(value);
            pdf = exit.pdf(wi);
            // The smooth surface's exit lobe from inside draws the step.
            weight = float3(exit.inside(own) * value / drawn);
        }
        else
        {
            float3 lobe = exit.eval_and_pdf(wi, pdf, diffuse);
            if (!conditioned || !any(lobe > 0.0))
                return lobe;
            toward = -refract(-wi, exit.n, 1.0 / exit.eta);
            weight = lobe * own / dot(toward, exit.n);
        }
        float c = dot(toward, exit.n);
        if (!(c > 0.0))
            return float3(0.0);
        return weight * hg_phase(anisotropy, dot(into, toward)) / hg_phase(anisotropy, dot(into, travel)) * exp(shift - extinction * depth / c);
    }
}

// The vertex of `slot`'s path in an interior's medium, which its raygen made and weighed the path
// by, what it runs in its payload's second word (`MEDIUM_*`) and whether the path walked in its
// last. At a scatter the first word's distance along its ray: in real time
// the world cache's light for the interior there ends the path, as past the dielectric depth;
// otherwise the vertex draws a light weighed by the medium's phase function, whose shadow ray the
// interior's transmittance dims, and its next ray from the phase function. A scatter that walks
// the interior's dense medium leaves the walk in the path's cold words (`MEDIUM_WALKS`) for its
// raygen to run. Where a walk leaves the interior (`MEDIUM_LEAVES`), at its hot origin through a
// surface of the walk's IOR, the hot words hold the way out and the surface's outward normal, the
// payload's first word the direction the walk reached the surface along and its third the length
// of the walk's last step where a scatter started it, else -1, and the walk words the direction
// that scatter was reached along and the weight of the surface's draw; where one stops at its cap
// (`MEDIUM_CAPPED`) it leaves where the path entered the interior, along a cosine draw, with the
// subsurface table's share. The path then lies past the surface: in real time the vertex takes
// the lights but the environment as `InteriorExitScatter` gathers them, conditioned on the walk's
// last scatter where one started its last step (`LeanMediumScatter`), which its ray along the way
// out meets for nothing (`PDF_LIT`); a way out no scatter started is a mirror's, whose ray takes
// all of its light, as the reference's take all of theirs. Through a rough surface
// (`MEDIUM_LEAVES_ROUGH`, the payload's first word the direction the walk reached it along, its
// third the weight of the surface's draw of the way out) the vertex weighs every light by the
// surface seen from inside instead, against its ray's own density. Where a walk meets its rough
// surface and goes on inside or ends there (`MEDIUM_MEETS`, the payload's first word the weight
// of the surface's draw, 0 where it was lost), the hot words hold the point, the direction the
// walk reached it along, the outward normal and the throughput there, and the vertex leaves the
// light through the surface there as the shadow job and the throughput the walk goes on with,
// before the raygen goes on with the walk. The walk was the bounce of the vertex where it leaves,
// and of each meeting.
public void lean_medium_vertex<M : IMedium>(PathRoot r, uint slot, inout uint4 words)
{
    uint word = words.x;
    // A re-run (`MEDIUM_BENT`) weighs the light of the job the vertex left where a plane stack
    // bent it, from the words the vertex started with.
    uint kind = words.y;
    uint4* at = lean_hot(r, slot);
    bool bent = false;
    uint4* stack = at;
    uint4* start = at;
    if (may(MAY_STACK))
    {
        bent = (kind & MEDIUM_BENT) != 0;
        kind &= ~MEDIUM_BENT;
        stack = lean_stack(r, slot);
        if (bent)
            start = stack + STACK_BENT_WORD;
    }
    uint4 w0 = load_coherent(start);
    uint4 w1 = load_coherent(at + 1);
    if (may(MAY_STACK))
    {
        if (bent)
            w1 = uint4(load_coherent(start + 2).xy, as_halves(float2(1.0)), as_halves(float2(1.0)));
    }
    uint flags = w0.w;
    bool along = (flags & HOT_COUNTED) != 0;
    flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_DIFFUSE | HOT_WINDOW);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    bool meets = kind == MEDIUM_MEETS;
    bool leaves = kind == MEDIUM_LEAVES || kind == MEDIUM_CAPPED || kind == MEDIUM_LEAVES_ROUGH;
    // The vertex lies on the interior's surface, where a walk took it, and a way out's payload
    // ends with the draws it took there.
    bool surface = leaves || meets;
    bool walked = surface || words.w != 0;
    float3 dir = oct_decode_snorm16(w1.x);
    float3 point = asfloat(w0.xyz) + (surface ? float3(0.0) : dir * asfloat(word));
    float3 normal = surface ? oct_decode_snorm16(w1.y) : float3(0.0);
    float3 travel = kind == MEDIUM_LEAVES || kind == MEDIUM_LEAVES_ROUGH ? oct_decode_snorm16(word) : dir;
    float3 throughput = hot_throughput(w1);
    // A way out of a step that started at a scatter (`conditioned`), its length.
    float length = asfloat(words.z);
    bool conditioned = kind == MEDIUM_LEAVES && length >= 0.0;
    // What the surface's draw weighs the walk by past the vertex, which a way out's walk words
    // hold with the direction its last scatter was reached along.
    uint4* cold = lean_cold(r, slot);
    uint4 last = load_coherent(cold + LEAN_INTO_WORD);
    float weight = asfloat(meets ? word : kind == MEDIUM_LEAVES ? last.w : words.z);
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = flags >> HOT_DIMENSION_SHIFT;
    LeanMode mode = lean_mode(r);
    // A re-run ends in no cache and draws no light.
    if (may(MAY_STACK))
    {
        if (bent)
        {
            mode.cache = false;
            mode.count_nee = false;
        }
    }
    bool reference = may(MAY_REFERENCE) && (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
    uint scattered = (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    if (kind == MEDIUM_WALKS)
    {
        Interior interior = load_interior(cold);
        uint medium = interior.innermost();
        float ior = interior_ior(r.scene, medium) / interior_ior(r.scene, interior.outside(medium, medium));
        float4 scatter = asfloat(load_coherent(cold + LEAN_SCATTER_WORD));
        InteriorMedium im = { interior_sigma(cold), scatter.xyz, scatter.w };
        store_walk_words(cold, walk_interior(point, dir, im, ior, rng), medium);
        store_coherent(cold + LEAN_INTO_WORD, uint4(asuint(dir), 0));
        flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(scattered + 1, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
        flags = lean_counted(r, scattered, true) ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;
        store_coherent(at, uint4(w0.xyz, hot_flags(flags, rng.dimension)));
        return;
    }
    float ior = 1.0;
    uint medium = MATERIAL_NONE;
    bool inside = (flags & HOT_INTERIOR) != 0;
    float3 sigma = float3(0.0);
    // The interior the vertex's light lies in: past a surface, the one outside the walk's.
    uint beyond = MATERIAL_NONE;
    // The phase function of the medium the vertex scatters in, or a walk's way out last
    // scattered in, whose words the way out replaces with the medium outside.
    float anisotropy = asfloat(load_coherent(cold + LEAN_SCATTER_WORD).w);
    float3 extinction = float3(0.0);
    if (surface)
    {
        Walk e = load_walk(cold, r.frame, medium);
        ior = e.ior;
        extinction = e.volume.extinction;
        anisotropy = e.volume.anisotropy;
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
        // The light lies past the surface, outside the interior the walk went through, which a
        // path that leaves it leaves.
        Interior interior = load_interior(cold);
        interior.cross(medium, false);
        inside = interior.count > 0;
        if (inside)
            beyond = interior.innermost();
        sigma = outside_medium(r.scene, cold, interior, leaves).sigma_t;
        if (leaves)
            flags = inside ? flags | HOT_INTERIOR : flags & ~HOT_INTERIOR;
        // The way out is a refraction: the walk's Fresnel draw chose it, and it scales radiance
        // as a BSDF's does.
        throughput *= ior * ior;
        // The vertex counts as the walk's scatter did.
        scattered--;
    }
    else if (inside)
    {
        sigma = interior_sigma(cold);
        beyond = PATH_MEDIUM;
    }
    // A scatter is a bounce; a walk's way out is the bounce its walk took.
    uint opaque = scattered + 1;
    flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(opaque, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
    float2 cone = halves(load_coherent(at + 2).x);
    if (!surface)
        cone.x += cone.y * asfloat(word);
    bool alive = carries(throughput) && opaque <= r.max_bounces + 1;
    // In real time a scatter takes the cache's light for the medium, which ends the path past the
    // vertex's next-event estimation.
    bool cache_ends = false;
    if (mode.cache && !surface)
    {
        if (along && alive)
        {
            float4 gathered = asfloat(load_coherent(at + 3));
            store_coherent(at + 3, asuint(float4(gathered.xyz + throughput * interior_ambient(r, point, dir, rng), gathered.w)));
        }
        cache_ends = true;
    }
    // A meeting's walk goes on whatever its light.
    if (!alive && !meets)
    {
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
        return;
    }
    bool counted = lean_counted(r, scattered, walked);
    flags = counted ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;
    float roughness = surface ? r.scene.materials[medium].specular_roughness : 0.0;
    float alpha = roughness * roughness;
    // A rough way out below the exact roughness (`plane`), which `interior_exact` leaves to the
    // way out alone; and the surface's transmission at a meeting or at a rough way out.
    bool plane = kind == MEDIUM_LEAVES && roughness >= MIRROR_ROUGHNESS;
    bool rough = meets || kind == MEDIUM_LEAVES_ROUGH || plane;
    // A smooth way out no scatter started is a mirror's.
    bool mirror = kind == MEDIUM_LEAVES && !conditioned && !plane;
    // The ways out whose vertex takes the lights for their rays, which meet them for nothing; a
    // way out's next-event estimation leaves the environment to its ray where that ray counts.
    bool lobe = leaves && (!rough || plane) && !mirror;
    InteriorExitScatter way_out = { normal, ior, 1.0 };
    if (leaves && (!rough || plane))
        way_out = interior_exit(normal, ior);
    // Seen from inside, along the way the walk reached the surface.
    InteriorRoughTransmission through = InteriorRoughTransmission(alpha, 1.0 / ior, -normal, -travel);
    // The walk's throughput over its last step's chance holds the least dense channel's
    // transmittance over the step.
    float shift = min(extinction.x, min(extinction.y, extinction.z)) * length;
    float3 into = asfloat(last.xyz);
    LeanMediumScatter scatter = { leaves, rough, lobe && mode.count_bsdf, anisotropy, travel, way_out, through, conditioned, into, length * dot(travel, normal), shift, extinction, alpha, plane ? asfloat(words.w) : 1.0, float3(0.0) };
    // A mirror's way out takes its lights along its ray alone.
    bool lit = lobe && !reference && mode.count_nee && !mode.one_technique;
    float4 u = rng.next4();
    scatter.u = u.xyz;
    float u_roulette = rng.next();
    float3 wi = leaves ? dir : sample_hg(scatter.anisotropy, dir, u.xy);
    if (may(MAY_STACK))
    {
        if (bent)
            wi = bent_direction(stack);
    }
    float3 scatter_f;
    float scatter_pdf;
    float3 scatter_diffuse;
    bool estimates = alive && mode.count_nee && counted && r.bounce_candidates > 0 && !(leaves && reference) && !mirror;
    LightPick k = pick_light<0>(r, scatter, point, normal, pixel, r.bounce_candidates, estimates, false, bent || !surface || kind == MEDIUM_LEAVES_ROUGH, wi, scatter_f, scatter_pdf, scatter_diffuse, rng);
    if (may(MAY_STACK))
    {
        if (bent)
        {
            bent_add(stack, at, scatter_f, scatter_pdf, true);
            return;
        }
    }
    float job_blue = 0.0;
    uint job_light = NO_LIGHT;
    if (k.sum > 0.0)
    {
        float3 origin = surface ? surface_ray_origin(r, point, normal, k.dir) : point;
        bool weight_one = mode.one_technique || lobe;
        float3 before;
        float3 e = picked_light(r, k, throughput, 1.0, weight_one, before);
        // The medium toward the light: the interior's, or past the surface out of every interior
        // the linked medium's, whose transmittance the environment's map holds of its own. A
        // directional light may come through a plane stack, whose walk takes the interior's
        // medium along its legs.
        bool stacked = false;
        if (may(MAY_STACK))
            stacked = k.kind == LIGHT_SAMPLE_DIRECTIONAL && r.set.stack_entries != 0 && inside;
        if (inside && !stacked)
            e *= beer(sigma, k.distance);
        else if (!inside && k.kind != LIGHT_SAMPLE_ENVIRONMENT)
            e *= medium_transmittance<M>(r.scene.medium, origin, k.dir, k.distance);
        store_job(at, k, origin, point, e, beyond, flags, job_blue, job_light);
        if (may(MAY_STACK))
        {
            if (stacked && any(e > 0.0))
            {
                job_light = STACK_JOB_LIGHT;
                stack_job(stack, meets ? STACK_COLD_OUT : STACK_COLD_LIST, Interior(), point, w0, w1, words, 0.0, BENT_MEDIUM, before, weight_one ? -1.0 : k.pdf);
            }
        }
    }
    if (meets)
    {
        // The meeting's shadow job lies outside the walk's interior, which the path stays in:
        // the walls it crosses take that medium's IOR, which the raygen keeps from here.
        if (may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
            words.z = asuint(interior_ior(r.scene, beyond));
        uint4 w2 = load_coherent(at + 2);
        float3 onward = hot_throughput(w1) * weight;
        store_coherent(at, uint4(w0.xyz, hot_flags(flags, rng.dimension)));
        store_coherent(at + 1, uint4(w1.xy, as_halves(onward.xy), as_halves(float2(onward.z, halves(w1.w).y))));
        store_coherent(at + 2, uint4(w2.x, asuint(job_blue), w2.z, job_light));
        return;
    }
    if (rough || kind == MEDIUM_LEAVES)
        throughput *= weight;
    // The way out's ray takes the transmittance over the walk's last step.
    if (conditioned)
        throughput *= exp(shift - extinction * length);
    // The phase function draws its own density, so the next ray keeps the throughput; a way out is
    // the direction the walk took.
    bool goes = !cache_ends && (leaves || scatter_pdf > 0.0) && (leaves || roulette_survives(r, opaque, u_roulette, throughput)) && carries(throughput);
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
        cone.y += rough ? 2.0 * alpha : 1.0;
        flags |= HOT_RAY | (rough ? 0 : leaves ? HOT_DIFFUSE : HOT_MEDIUM);
        store_coherent(at, uint4(asuint(origin), hot_flags(flags, rng.dimension)));
        // The surface's normal, which the walk left in the words, is what the light pick of a way
        // out drew by, and the lights its ray meets are weighed with.
        store_coherent(at + 1, uint4(oct_encode_snorm16(wi), leaves ? w1.y : 0, as_halves(throughput.xy), as_halves(float2(throughput.z, mode.count_nee ? 1.0 : 0.0))));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, lobe || mirror ? (lit ? PDF_LIT : 0.0) : scatter_pdf)));
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
void lean_exit_vertex<M : IMedium>(PathRoot r, uint slot, uint4* at, PathVertex x, float2 cone, uint4 input)
{
    bool capped = input.x != 0;
    // A re-run weighs the light of the job the vertex left where a plane stack bent it, from the
    // words the vertex started with.
    bool bent = false;
    uint4* stack = at;
    uint4* start = at;
    if (may(MAY_STACK))
    {
        bent = input.y == PAYLOAD_BENT;
        stack = lean_stack(r, slot);
        if (bent)
            start = stack + STACK_BENT_WORD;
    }
    uint4 w0 = load_coherent(start);
    uint4 w1 = load_coherent(at + 1);
    uint flags = w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_DIFFUSE | HOT_WINDOW);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float3 throughput = hot_throughput(w1);
    if (may(MAY_STACK))
    {
        if (bent)
            throughput = float3(1.0);
    }
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
    // A re-run draws no light.
    if (may(MAY_STACK))
    {
        if (bent)
            mode.count_nee = false;
    }
    uint opaque = (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    bool counted = lean_counted(r, opaque - 1, true);
    flags = counted ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;
    WalkExitScatter f = walk_exit(x, g.bsdf, rng);
    float4 u = rng.next4();
    float u_roulette = rng.next();
    float3 tangent, bitangent;
    orthonormal_basis(f.n, tangent, bitangent);
    float3 wi = normalize(from_local(sample_cosine_hemisphere(u.xy), tangent, bitangent, f.n));
    if (may(MAY_STACK))
    {
        if (bent)
            wi = bent_direction(stack);
    }
    float3 scatter_f;
    float scatter_pdf;
    float3 scatter_diffuse;
    bool estimates = mode.count_nee && counted && r.bounce_candidates > 0;
    LightPick k = pick_light<0>(r, f, g.position, g.normal, pixel, r.bounce_candidates, estimates, false, true, wi, scatter_f, scatter_pdf, scatter_diffuse, rng);
    if (may(MAY_STACK))
    {
        if (bent)
        {
            bent_add(stack, at, scatter_f, scatter_pdf, sides_agree(g.normal, g.geometric_normal, wi));
            return;
        }
    }
    float job_blue = 0.0;
    uint job_light = NO_LIGHT;
    if (k.sum > 0.0 && sides_agree(g.normal, g.geometric_normal, k.dir))
    {
        float3 origin = vertex_origin(x, k.dir);
        float3 before;
        float3 e = picked_light(r, k, throughput, 1.0, mode.one_technique, before);
        bool inside = may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0;
        // A directional light may come through a plane stack, whose walk takes the interior's
        // medium along its legs.
        bool stacked = false;
        if (may(MAY_STACK))
            stacked = k.kind == LIGHT_SAMPLE_DIRECTIONAL && r.set.stack_entries != 0 && inside;
        if (inside && !stacked)
            e *= beer(interior_sigma(lean_cold(r, slot)), k.distance);
        else if (!inside && k.kind != LIGHT_SAMPLE_ENVIRONMENT)
            e *= medium_transmittance<M>(r.scene.medium, origin, k.dir, k.distance);
        store_job(at, k, origin, g.position, e, inside ? PATH_MEDIUM : MATERIAL_NONE, flags, job_blue, job_light);
        if (may(MAY_STACK))
        {
            if (stacked && any(e > 0.0))
            {
                job_light = STACK_JOB_LIGHT;
                stack_job(stack, STACK_COLD_LIST, Interior(), g.position, w0, w1, input, 0.0, BENT_EXIT, before, mode.one_technique ? -1.0 : k.pdf);
            }
        }
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
// what the wall lets straight through in the payload's first three words; nor does a stack ray's
// hit invoked with the stack question (`is_stack_question`), whose answer goes into the path's
// stack words.
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
    // A stack ray's hit asks whether it is on a plane stack face (the stack question), which only a
    // layer's stage holds and answers from the surface it resolves.
    bool asks = false;
    bool bent = false;
    if (may(MAY_STACK))
    {
        asks = is_stack_question(payload);
        if (!LAYER && asks)
        {
            payload = NO_STACK_FACE;
            return;
        }
        bent = payload.y == PAYLOAD_BENT;
    }
    // The ray's cone rides in its cover, so the surface resolves before the path's state loads.
    PathCover c = path_cover(payload);
    float t = h.t();
    float3 dir = h.world_direction();
    RayCone cone = { c.context.cone_width, c.context.cone_spread };
    HitContext context = hit_context(c.context.kind, c.context.depth, cone, t, texture_tap(r.view, pixel, r.frame, r.seed, c.context.depth), c.context.feedback);
    float3 positions[3] = { h.corner(0), h.corner(1), h.corner(2) };
    // An emitter's light is weighed by its triangle as light sampling reads it, and a plane stack
    // face is flat.
    HitTriangle traced;
    SurfaceHit sh = resolve_hit_traced<FeedbackSurface<S>, T, EMITTER, LAYER>(r.scene, r.view, dir, h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), positions, false, context, material, traced);
    // A layer's vertex asks whether it lies on a plane stack face where its path's ray may go on
    // weighed against next-event estimation through a stack, whose face it may cross.
    // A face's material reads no texture, so its surface is the material's own: the stack's
    // (`stack_material`).
    bool on_stack = false;
    if (LAYER && may(MAY_STACK))
    {
        bool tracks = asks || (!bent && !is_question(payload) && lean_stacks(r) && stack_tracked_ray(at));
        SceneRoot* s = r.scene;
        on_stack = tracks && sh.valid && traced.flat && s.decal_globals == 0 && s.decal_bin_slots == 0 && (s.materials[sh.section].flags & MATERIAL_STACK_FACE) != 0 && bsdf_thin_mirror(sh.surface.bsdf, AMBIENT_IOR);
        if (asks)
        {
            payload = stack_answer(on_stack, sh);
            return;
        }
    }
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
    OnLayer on = { LayerVertex(), on_stack, sh.surface.normal };
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
    float exterior = 0.0;
    if (exits)
        left = PAYLOAD_EXITS;
    else if (EMITTER)
    {
        HitLight emits = { no_emitter(), lean_cold(r, slot) };
        if (any(sh.surface.bsdf.emission > 0.0))
            emits.emitter = emitter_hit_of(r.set, sh, traced, h.instance_index(), uint(h.cluster_id()));
        if (LAYER)
            left = lean_vertex<M, 2, HitLight, OnLayer, true>(r, slot, at, x, dir, t, footprint, emits, on, payload, exterior);
        else
            left = lean_vertex<M, 2, HitLight, OnOpaque, false>(r, slot, at, x, dir, t, footprint, emits, OnOpaque(), payload, exterior);
    }
    else if (LAYER)
        left = lean_vertex<M, 2, NoLight, OnLayer, true>(r, slot, at, x, dir, t, footprint, NoLight(), on, payload, exterior);
    else
        left = lean_vertex<M, 2, NoLight, OnOpaque, false>(r, slot, at, x, dir, t, footprint, NoLight(), OnOpaque(), payload, exterior);
    // The surface the vertex where a walk leaves takes: this one, where a walk leaves here, or
    // where one starts here, this one seen from the ray, for where it stops at its cap.
    if (lean_walks(LAYER) && left != 0)
        store_exit(lean_cold(r, slot), exit_vertex_of(x, dir, !exits));
    payload.y = left;
    // A layer's hit hands its shadow job's exterior to the raygen.
    if (LAYER && may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
        payload.z = asuint(exterior);
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
public void lean_walk_exit<M : IMedium>(PathRoot r, uint slot, uint4 input)
{
    bool capped = input.x != 0;
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
    lean_exit_vertex<M>(r, slot, at, x, halves(load_coherent(at + 2).x), input);
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
    // Where the pipeline may meet a stack, the stack question's second word is 0.
    if (may(MAY_STACK))
        return all(bool2(payload.x == 0, payload.y != 0));
    return payload.x == 0;
}

// Whether a hit is invoked with a question its shader answers in place of running a vertex: a
// window's or the stack question, whose second word is 0.
public bool is_question(uint4 payload)
{
    return payload.x == 0;
}

// The stack question's answer where the hit is on no plane stack's face.
public static const uint4 NO_STACK_FACE = uint4(0, 0, MATERIAL_NONE, 0);

public bool is_stack_question(uint4 payload)
{
    return payload.x == 0 && payload.y == 0;
}

// Whether the ray of the path in hot words `at` may cross a plane stack face weighed as a
// vertex's ray toward a directional light: a vertex's scatter drew it, or it crossed faces since.
bool stack_tracked_ray(uint4* at)
{
    float pdf = asfloat(load_coherent(at + 3)).w;
    return pdf > 0.0 || pdf == PDF_STACK;
}

// The stack question's answer of a layer's hit on surface `sh`, `face` where it is a plane stack's
// face: its shading and geometric normals, its material and its rounding error.
uint4 stack_answer(bool face, SurfaceHit sh)
{
    GSurface g = sh.surface;
    return uint4(oct_encode_snorm16(g.normal), oct_encode_snorm16(g.geometric_normal), face ? sh.section : MATERIAL_NONE, asuint(sh.geometric_error));
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
    // A path behind layers marks its offset word: the camera ray's media are the layer chain's.
    // A re-run of the vertex for its bent job's light reads none of them.
    uint4 w2 = load_coherent(at + 2);
    bool behind = w2.z != 0;
    if (!(may(MAY_STACK) && payload.y == PAYLOAD_BENT))
        store_coherent(at + 2, uint4(w2.xy, 0, w2.w));
    float exterior;
    payload.y = lean_vertex<M, 1, GBufferLight, OnOpaque, false>(r, slot, at, x, -x.view, behind ? 0.0 : x.camera_distance, footprint, GBufferLight(), OnOpaque(), payload, exterior);
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
// lights' discs, each weighed against next-event estimation at the vertex the ray left. A ray past
// plane stack faces weighs a disc against next-event estimation at the vertex before them where
// its probe found the stack there, `stack_owned`, as its raygen says.
public void lean_miss<M : IMedium, E : IEnvironment>(PathRoot r, uint slot, bool stack_owned)
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
                if (!any(d > 0.0) || lit)
                    continue;
                // Past plane stack faces the disc is weighed as the vertex before them drew it.
                float pdf = p.pdf;
                float run = p.run;
                float3 normal = p.normal;
                bool weighs = weigh;
                if (may(MAY_STACK))
                {
                    if (stack_owned && p.pdf == PDF_STACK)
                    {
                        pdf = stack_density(r, slot, p.dir, run, normal);
                        weighs = true;
                    }
                }
                e += d * (weighs ? mis_balance(pdf, run * nee_directional_pdf(set, normal, id, p.dir)) : 1.0);
            }
        }
        p.radiance += p.throughput * e;
    }
    store_hot(at, p);
}

// Where a ray inside an interior's medium of extinction `sigma_t` scatters, by `rng`: a free flight
// by one channel's extinction, the channel picked uniformly, along the ray's stretch in the medium.
// The raygen draws it for the ray's segment, and the callable of what the ray meets in front of its
// hit draws it again, alike, for where the segment ends.
[ForceInline]
public float free_flight(float3 sigma_t, inout Sampler rng)
{
    float u_channel = rng.next();
    float u_distance = rng.next();
    float sigma = sigma_t[min(uint(u_channel * 3.0), 2u)];
    return sigma > 0.0 ? -log(1.0 - u_distance) / sigma : positive_infinity();
}

// What a path's ray sees of a layer in front of its hit (`PathEmission`): the layer's light through
// the medium up to it, as the hit takes its own. Inside an interior's medium (`inside`) that is the
// interior's, and where the ray's draw of where it scatters ends it (`drawn`) its light is over the
// chance the draw reaches the layer, which the channels' mean transmittance gives; the ray's hot
// words `at` keep where its stretch in the medium starts. Elsewhere it is the linked medium `M`'s.
struct FrontLayers<M : IMedium> : ITranslucent
{
    PathEmission emission;
    uint64_t medium;
    bool inside;
    bool drawn;
    float3 sigma_t;
    uint4* at;

    float4 layer(inout Candidate c)
    {
        float4 l = emission.layer(c);
        if (!inside)
            return float4(l.rgb * medium_transmittance<M>(medium, emission.origin, emission.dir, c.t), l.a);
        float3 tr = beer(sigma_t, inside_length(at, c.t));
        return float4(l.rgb * (drawn ? tr / dot(tr, float3(1.0 / 3.0)) : tr), l.a);
    }

    bool accept(Candidate c)
    {
        return false;
    }
}

// The callable that takes what `slot`'s path ray meets in front of its hit `t` m along it, under
// instance mask `mask`: particles, each a layer of its emission at its opacity, and the proxies of
// sphere and rect lights, layers that only emit, in depth order as the staged loop's trace stage
// takes them. What the layers emit toward a ray whose light counts adds to the path's radiance, and
// what they cover leaves its throughput. A view path's ray inside a medium that scatters ends where
// the raygen's draw for its segment (`free_flight`) says it scatters, if that falls before its
// hit. Whether the path still carries light.
public bool lean_front<M : IMedium>(PathRoot r, uint slot, float t, uint mask, bool view)
{
    uint4* at = lean_hot(r, slot);
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    float4 w3 = asfloat(load_coherent(at + 3));
    uint flags = w0.w;
    bool inside = may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0;
    float3 sigma_t = float3(0.0);
    float end = t;
    bool drawn = false;
    if (inside)
    {
        uint4* cold = lean_cold(r, slot);
        sigma_t = interior_sigma(cold);
        if (may(MAY_SCATTERING) && view && any(asfloat(load_coherent(cold + LEAN_SCATTER_WORD)).xyz > 0.0))
        {
            drawn = true;
            float span = inside_length(at, t);
            Sampler rng = path_sampler(r, item_pixel(r, r.slice_base + slot));
            rng.dimension = flags >> HOT_DIMENSION_SHIFT;
            float sampled = free_flight(sigma_t, rng);
            if (sampled < span)
                end = max(sampled - (span - t), 0.0);
        }
    }
    float3 origin = asfloat(w0.xyz);
    float3 dir = oct_decode_snorm16(w1.x);
    LeanMode mode = lean_mode(r);
    // A ray ideal windows passed on measures the light from where its vertex left.
    float3 from = origin;
    if (may2(MAY2_THIN_DIELECTRICS) && (flags & HOT_WINDOW) != 0)
        from = asfloat(load_coherent(lean_cold(r, slot) + LEAN_FROM_WORD).xyz);
    // The vertex the ray left drew its candidates with its local sampler where it was the path's
    // first.
    LocalLights* local = nullptr;
    if (may(MAY_LOCAL) && ((flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK) == 1)
        local = pixel_local_lights(r, item_pixel(r, r.slice_base + slot));
    // A ray whose vertex's next-event estimation took the lights meets what light sampling draws
    // for nothing.
    bool lit = may(MAY_SCATTERING) && w3.w == PDF_LIT;
    bool adds = mode.count_bsdf && (flags & HOT_COUNTED) != 0;
    PathEmission pe = { r.scene, r.set, local, origin, dir, from, left_normal(flags, w1), mode.one_technique ? 0.0 : w3.w, halves(w1.w).y, end, float3(lit ? 0.0 : 1.0), adds };
    FrontLayers<M> f = { pe, r.scene.medium, inside, drawn, sigma_t, at };
    float4 layers;
    trace_resolve_translucent(scene_traced(r.scene), path_ray(origin, dir, end), mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, f, PARTICLE_CANDIDATES, layers);
    float3 throughput = hot_throughput(w1);
    w3.xyz += throughput * layers.rgb;
    throughput *= 1.0 - layers.a;
    store_coherent(at + 1, uint4(w1.xy, as_halves(throughput.xy), as_halves(float2(throughput.z, halves(w1.w).y))));
    store_coherent(at + 3, asuint(w3));
    return carries(throughput);
}

// What the medium proxies with a volume along a shadow ray from `origin` up to `t_max` let through,
// each by its march (`proxy_volume`); a proxy without one took Beer's law in the ray's any-hit.
struct ProxyVolumes : ICandidate
{
    SceneRoot* scene;
    float3 origin;
    float t_max;

    bool positions(Candidate c)
    {
        return false;
    }

    [mutating]
    float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        float3 beer;
        MediumProxyGpu p;
        float t0, t1;
        if (c.triangle || proxy_unmarched(scene, c, t_max, beer, p, t0, t1))
            return float3(1.0);
        return proxy_volume(scene, p, c.instance_index, c.primitive_index, origin, c.object_origin, c.object_direction, t0, t1);
    }
}

// What a shadow ray hands the callable that marches the medium proxies with a volume it passed
// (SHADOW_VOLUME): its origin and reach, and its unit direction. The callable hands back what they
// let through in the first word.
public struct ProxyMarch
{
    public uint4 ray;
    public uint4 dir;
}

// That callable's body: the marches of the proxies along the ray, by a ray query over them alone.
public void lean_proxy_march(PathRoot r, inout ProxyMarch m)
{
    float3 origin = asfloat(m.ray.xyz);
    float t_max = asfloat(m.ray.w);
    ProxyVolumes v = { r.scene, origin, t_max };
    bool overflow;
    float3 through = trace_transmittance(scene_traced(r.scene), path_ray(origin, asfloat(m.dir.xyz), t_max), MASK_MEDIUM, RAY_FLAG_NONE, v, 0, overflow);
    m.ray.xyz = asuint(through);
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
    if (light != NO_LIGHT && light != STACK_JOB_LIGHT)
        lean_feedback(r, slot, light, luminance_rec2020(added));
}
