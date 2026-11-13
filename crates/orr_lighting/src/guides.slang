// Ray reconstruction guides. Per pixel: diffuse and specular albedo from the surface's BSDF,
// with its pre-exposed emission added to the diffuse albedo, world normal and linear roughness,
// depth and motion from the G-buffer, and specular motion. Pixels with a dielectric or presence
// layer blend in its guide by its weight, and take its depth and motion where it dominates.
// A primary surface that is a near-mirror follows its mirror chain and takes the guides of
// the first surface past it, seen through the chain; a dielectric the chain meets gives the
// guides of its reflection and refraction chain, as a pixel's first layer does. Pixels
// smoother than the specular-motion roughness trace the mirror direction and reproject the
// reflection's virtual point.
module guides;

import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.trace;
import orr_lighting.media;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.path;
import orr_lighting.shadow;
import orr_lighting.interiors;

// `GuideRoot.debug`: which guide the radiance image shows.
public static const uint GUIDE_NONE = 0;
public static const uint GUIDE_DIFFUSE_ALBEDO = 1;
public static const uint GUIDE_SPECULAR_ALBEDO = 2;
public static const uint GUIDE_NORMAL = 3;          // 0.5 n + 0.5
public static const uint GUIDE_ROUGHNESS = 4;
public static const uint GUIDE_SPECULAR_MOTION = 5; // uv in rg

// Surfaces smoother than this whose specular albedo exceeds their diffuse albedo are
// near-mirrors: their guides come from what they reflect.
public static const float NEAR_DELTA_ROUGHNESS = 0.05;

// Mirror bounces a guide chain follows.
public static const uint MIRROR_CHAIN = 4;

public static const float SKY_DIFFUSE_ALBEDO = 0.5;

public struct GuideRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public GBufferGpu* gbuffer;
    // This frame's layer memory (`traced_tiles`); null for none.
    public LayerGuide* layers;
    public StorageImageIndex radiance;          // RGBA16F, written only for a GUIDE_* debug view
    public StorageImageIndex diffuse_albedo;    // RGBA8
    public StorageImageIndex specular_albedo;   // RGBA8
    public StorageImageIndex normal_roughness;  // RGBA16F
    public StorageImageIndex specular_motion;   // RG16F
    public StorageImageIndex depth;             // R32F
    public StorageImageIndex motion;            // RG16F
    // R16F depth-of-field guide, cleared on every pixel for the frosted-glass term; NONE
    // writes none.
    public StorageImageIndex dof_guide;
    public float motion_roughness;              // specular motion below this roughness
    public uint debug;                          // GUIDE_*
    public uint dielectric_depth;               // layers a dielectric's refraction chain crosses
    public uint _pad0;
}

// A view's overlay front words, and the targets `guides/overlay_front` resolves them into.
public struct OverlayFrontRoot
{
    public uint64_t* words;             // per pixel, in rows of `width`
    public StorageImageIndex depth;     // R32F
    public StorageImageIndex motion;    // RG16F
    public StorageImageIndex weight;    // R16F
    public TextureIndex overlay;        // the premultiplied overlay; NONE without one
    public uint width;
    public uint height;
}

// The frosted-glass term: on each pixel whose dominant layer is rough transmission, the
// transmission times the background's mean over the disc its lobes blur it by, less the
// background there, added into ray reconstruction's depth-of-field guide, which holds
// `guide_luminance(after - before)` for each blur.
public struct FrostRoot
{
    // The layer memory (`traced_tiles`) of a view dispatch with DIELECTRIC_FROST at
    // `width` x `height`, which keeps every pixel's `background`.
    public LayerGuide* layers;
    public StorageImageIndex guide;     // R16F
    public uint width;
    public uint height;
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// Pixels the frosted-glass term's disc reaches at most.
public static const float FROST_MAX_RADIUS = 64.0;

// Background reads over the disc.
static const uint FROST_TAPS = 24;

public void frost_body(FrostRoot r, uint2 pixel)
{
    LayerGuide own = layer_guide(r.layers, uint2(r.width, r.height), pixel);
    float radius = f16tof32(own.frost & 0xffff);
    float weight = f16tof32(own.frost >> 16);
    if (radius < 0.5 || weight <= 0.0)
        return;
    // A Vogel disc, the same on every pixel and frame, so the term holds still.
    float sum = 0.0;
    for (uint k = 0; k < FROST_TAPS; k++)
    {
        float d = radius * sqrt((float(k) + 0.5) / float(FROST_TAPS));
        float a = float(k) * GOLDEN_ANGLE;
        int2 q = clamp(int2(pixel) + int2(round(d * float2(cos(a), sin(a)))), int2(0), int2(r.width, r.height) - 1);
        sum += r.layers[q.y * r.width + q.x].background;
    }
    RWTexture2D<float> guide = r.guide.get<RWTexture2D<float>>();
    guide[pixel] = guide[pixel] + weight * (sum / float(FROST_TAPS) - own.background);
}

public struct Guides
{
    public float3 diffuse;
    public float3 specular;
    public float3 normal;
    public float roughness;
    // Pre-exposed emission, which the diffuse albedo guide adds.
    public float3 emission;
}

// A surface's guides from its BSDF `b`: subsurface pixels add the subsurface share in the
// subsurface color to the diffuse albedo. `exposure` scales its emission to pre-exposed units.
public Guides surface_guides(GSurface s, Bsdf b, float exposure)
{
    LobeAlbedo a = b.albedo();
    Guides g;
    g.diffuse = a.diffuse + a.subsurface * s.bsdf.subsurface_color;
    g.specular = a.specular;
    g.normal = s.normal;
    g.roughness = s.bsdf.specular_roughness;
    g.emission = exposure * b.emission();
    return g;
}

public Guides sky_guides()
{
    Guides g;
    g.diffuse = float3(SKY_DIFFUSE_ALBEDO);
    g.specular = float3(0.0);
    g.normal = float3(0.0);
    g.roughness = 0.0;
    g.emission = float3(0.0);
    return g;
}

public bool near_delta(Guides g)
{
    return g.roughness < NEAR_DELTA_ROUGHNESS && luminance_rec2020(g.specular) > luminance_rec2020(g.diffuse);
}

// Previous uv minus current uv, unjittered, of a point at eye-relative render-space position
// `q` that was at `previous_q` last frame, relative to this frame's eye (`w` 1), or of the
// direction `q` at infinity (`w` 0, `previous_q` = `q`).
public float2 moved_motion(ViewConstants* v, float3 q, float3 previous_q, float w)
{
    float4 current = mul(v.clip_from_view, float4(view_from_relative(v, q), w));
    float3 p = previous_q + v.eye_motion.xyz * w;
    float4 previous = mul(v.prev_clip_from_view, float4(mul(v.prev_view_from_world_rotation, float4(p, 0.0)), w));
    return motion_between(previous, current);
}

// `moved_motion` of a static point or a direction.
public float2 reprojected_motion(ViewConstants* v, float3 q, float w)
{
    return moved_motion(v, q, q, w);
}

float3x3 mirror_matrix(float3 n)
{
    return float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) - 2.0 * float3x3(n * n.x, n * n.y, n * n.z);
}

// What the mirror direction of a pixel's surface reaches.
public struct Reflection
{
    public Guides guides;
    // The reflection's virtual point relative to the eye (`w` 1), or its direction at
    // infinity (`w` 0).
    public float3 virtual_point;
    public float w;
}

// The refracted direction of a ray along `dir` through layer `l`, without roughness: straight
// through thin walls; false on total internal reflection.
public bool layer_refraction(SceneRoot* s, LayerVertex l, Interior interior, float3 dir, out float3 out_dir)
{
    out_dir = dir;
    if (thin_walled(l.x.surface))
        return true;
    uint m = l.section;
    float outer = interior_ior(s, interior.outside(m, m));
    float inner = interior_ior(s, m);
    float3 n = l.front ? l.x.surface.normal : -l.x.surface.normal;
    float eta = l.front ? outer / inner : inner / outer;
    out_dir = refract(dir, n, eta);
    return any(out_dir != 0.0);
}

// What rough layer `x`, `s` m along a refraction chain, adds to `LayerChain.blur`.
float3 rough_widening(LayerVertex x, float s)
{
    float roughness = x.x.surface.bsdf.specular_roughness;
    if (roughness < MIRROR_ROUGHNESS)
        return float3(0.0);
    float w2 = roughness * roughness * roughness * roughness;
    return w2 * float3(1.0, s, s * s);
}

// What a dielectric layer shows ray reconstruction: its reflection, the surface itself, and
// what its transmission reaches along a roughness-free refraction chain, by throughput.
public struct LayerChain
{
    // The reflection's and the chain's guides added by their throughputs, with normal and
    // roughness weighed by `reflected` and `transmitted`.
    public Guides guides;
    // Luminance of the reflection's albedo with what the chain's interiors scatter, and of
    // the chain's throughput.
    public float reflected;
    public float transmitted;
    public float3 through;
    // m along the chain to the surface it ends on; infinite where it reaches the sky or ends
    // without one.
    public float traveled;
    // Over the chain's rough interfaces i at chain length s_i with widening w_i: sums of w^2,
    // s w^2 and s^2 w^2, which give sum (D - s_i)^2 w_i^2 at chain length D.
    public float3 blur;
}

// `LayerChain` of dielectric layer `l`, prepared as `b`, met along `dir` with `interior` the
// list around the ray before it and `cone` the ray's cone there; the chain crosses at most
// `depth` layers. `exposure` scales emission to pre-exposed units. Each rough interface widens
// the chain by its roughness squared per m beyond it (half the cone spread a rough lobe gives a
// path).
public LayerChain layer_chain<S : ISurfaces>(SceneRoot* s, ViewConstants* v, LayerVertex l, Bsdf b, float3 dir, Interior interior, uint depth, RayCone cone, float exposure)
{
    LobeAlbedo a = b.albedo();
    Guides own = surface_guides(l.x.surface, b, exposure);
    float3 lean = l.front ? l.x.surface.normal : -l.x.surface.normal;
    LayerChain c;
    c.through = a.transmission;
    c.traveled = positive_infinity();
    c.blur = float3(0.0);
    Guides behind = sky_guides();
    float3 d = dir;
    Interior list = interior;
    LayerVertex x = l;
    float3 bent = d;
    bool open = any(c.through > 0.0) && layer_refraction(s, x, list, d, bent);
    d = bent;
    if (!open)
        c.through = float3(0.0);
    else if (crosses_interior(s, x.section))
        list.cross(x.section, x.front, x.x.surface.medium_point);
    float length_so_far = 0.0;
    // What the chain's interiors scatter, which shows in their single-scattering color.
    float3 scattered = float3(0.0);
    if (open)
        c.blur += rough_widening(x, 0.0);
    for (uint k = 0; open && k < depth; k++)
    {
        LayerAccept<S> ta = { s, v, d, hit_context(HIT_SECONDARY, k + 1, cone, 0.0), 0, PRESENCE_ANY };
        float3 positions[3];
        Hit th = trace_closest(s.tlas, path_ray(vertex_origin(x.x, d), d, positive_infinity()), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, ta, positions);
        InteriorMedium im = interior_medium(s, list.innermost(), list.point(list.innermost()));
        float3 tr = beer(im.sigma_t, th.hit ? th.t : positive_infinity());
        scattered += c.through * (1.0 - tr) * im.albedo();
        c.through *= tr;
        if (!th.hit || th.cluster_id < 0)
        {
            behind = sky_guides();
            break;
        }
        SurfaceHit sh = resolve_hit<S>(s, v, d, th.instance_index, uint(th.cluster_id), th.primitive_index, th.barycentrics, positions, false, hit_context(HIT_SECONDARY, k + 1, cone, th.t));
        if (!sh.valid)
            break;
        length_so_far += th.t;
        bool layer = is_layer(th);
        LayerVertex y = layer ? layer_vertex(v, sh, d) : opaque_vertex(hit_vertex(v, sh, d));
        if (!layer)
        {
            behind = surface_guides(y.x.surface, prepare_vertex(s, y, list, true, true, false, 0.0), exposure);
            c.traveled = length_so_far;
            break;
        }
        x = y;
        Bsdf xb = prepare_vertex(s, x, list, k > 0, true, false, 0.0);
        c.through *= xb.albedo().transmission;
        bent = d;
        bool bends = any(c.through > 0.0) && layer_refraction(s, x, list, d, bent);
        d = bent;
        if (!bends)
        {
            c.through = float3(0.0);
            break;
        }
        c.blur += rough_widening(x, length_so_far);
        if (crosses_interior(s, x.section))
            list.cross(x.section, x.front, x.x.surface.medium_point);
    }
    c.reflected = luminance_rec2020(a.specular + a.diffuse + scattered);
    c.transmitted = luminance_rec2020(c.through);
    float wr = c.reflected;
    float wt = c.transmitted;
    c.guides.diffuse = own.diffuse + scattered + c.through * behind.diffuse;
    c.guides.specular = own.specular + c.through * behind.specular;
    c.guides.emission = own.emission + c.through * behind.emission;
    float3 n = wr * lean + wt * behind.normal;
    c.guides.normal = dot(n, n) > 0.0 ? normalize(n) : lean;
    c.guides.roughness = wr + wt > 0.0 ? (wr * own.roughness + wt * behind.roughness) / (wr + wt) : own.roughness;
    return c;
}

// Traces the mirror direction from `pixel`'s G-buffer surface `x1` (guides `g1`), meeting
// dielectrics and passing presence layers. With `chain`, keeps reflecting off near-mirrors for up to `MIRROR_CHAIN` bounces and returns the guides of
// the surface past the last one, or the `LayerChain` guides of the dielectric it meets, scaled
// by the chain's specular albedo, with the normal mirrored through the chain into the view;
// otherwise one bounce and `g1`. The virtual point lies along the camera ray at the camera
// distance plus every bounce's length, and past a dielectric whose refraction chain carries
// more than its reflection, plus that chain's length or at infinity, as a pixel's first layer
// takes it.
public Reflection mirror_chain<S : ISurfaces>(GuideRoot r, uint2 pixel, PathVertex x1, Guides g1, bool chain)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    float exposure = v.exposure[v.exposure_slot].pre_exposure;
    Reflection out;
    out.guides = g1;
    float3 camera_dir = -x1.view;
    float traveled = x1.camera_distance;
    float3 throughput = float3(1.0);
    float3x3 mirrors = float3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    PathVertex x = x1;
    Guides g = g1;
    RayCone cone = camera_cone(v);
    cone.width = cone_width(cone, x1.camera_distance);
    for (uint i = 0; i < MIRROR_CHAIN; i++)
    {
        float3 dir = reflect(-x.view, x.surface.normal);
        HitContext trace_context = hit_context(HIT_SECONDARY, i + 1, cone, 0.0);
        LayerAccept<S> a = { s, v, dir, trace_context, 0, PRESENCE_NONE };
        float3 positions[3];
        Hit h = trace_closest(s.tlas, path_ray(vertex_origin(x, dir), dir, positive_infinity()), MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
        if (chain)
        {
            throughput *= g.specular;
            mirrors = mul(mirrors, mirror_matrix(x.surface.normal));
        }
        if (!h.hit || h.cluster_id < 0)
        {
            if (chain)
            {
                out.guides = sky_guides();
                out.guides.diffuse *= throughput;
            }
            out.virtual_point = camera_dir;
            out.w = 0.0;
            return out;
        }
        traveled += h.t;
        out.virtual_point = camera_dir * traveled;
        out.w = 1.0;
        bool layer = is_layer(h);
        if (!chain && !layer)
            return out;
        cone = cone_after_hit(cone, h.t, 0.0);
        SurfaceHit sh = resolve_hit<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, hit_context(HIT_SECONDARY, i + 1, cone, h.t));
        if (!sh.valid)
            return out;
        if (layer)
        {
            // The chain's surfaces are opaque, so the interiors around the ray are x1's.
            Interior around = gbuffer_interior<S>(s, v, r.layers, r.gbuffer.extent, pixel, pinhole_start(v, r.gbuffer, pixel, true));
            LayerVertex l = layer_vertex(v, sh, dir);
            Bsdf lb = prepare_vertex(s, l, around, true, true, false, 0.0);
            LayerChain c = layer_chain<S>(s, v, l, lb, dir, around, r.dielectric_depth, cone, exposure);
            g = c.guides;
            if (c.transmitted > c.reflected)
            {
                bool ends = isfinite(c.traveled);
                out.virtual_point = ends ? camera_dir * (traveled + c.traveled) : camera_dir;
                out.w = ends ? 1.0 : 0.0;
            }
            if (!chain)
                return out;
        }
        else
        {
            x = hit_vertex(v, sh, dir);
            Bsdf b = bsdf_prepare(bsdf_collapse(x.surface.bsdf), x.surface.normal, x.view);
            g = surface_guides(x.surface, b, exposure);
        }
        out.guides = g;
        out.guides.diffuse *= throughput;
        out.guides.specular *= throughput;
        out.guides.emission *= throughput;
        out.guides.normal = mul(mirrors, g.normal);
        if (layer || !near_delta(g))
            return out;
    }
    return out;
}

// The guide resolve's body for `pixel`, hits shading through linked surface `S`.
public void guide_body<S : ISurfaces>(GuideRoot r, uint2 pixel)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    GBufferGpu* gb = r.gbuffer;
    float depth = gb.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    float2 motion = gb.motion.get<Texture2D<float2>>().Load(int3(pixel, 0));
    Guides g = sky_guides();
    float2 specular_motion = motion;
    PathVertex x1;
    bool surface = gbuffer_vertex(s, v, gb, pixel, x1);
    if (surface)
    {
        Bsdf b = bsdf_prepare(x1.surface.bsdf, x1.surface.normal, x1.view);
        // The G-buffer's emission is already pre-exposed.
        g = surface_guides(x1.surface, b, 1.0);
        bool chain = near_delta(g);
        if (chain || g.roughness < r.motion_roughness)
        {
            Reflection m = mirror_chain<S>(r, pixel, x1, g, chain);
            g = m.guides;
            specular_motion = reprojected_motion(v, m.virtual_point, m.w);
        }
    }
    // Layer guides hold their emission in their diffuse albedo.
    g.diffuse += g.emission;
    if (r.layers != nullptr)
    {
        LayerGuide l = layer_guide(r.layers, gb.extent, pixel);
        if (l.weight > 0.0)
        {
            float w = l.weight;
            float3 n = w * l.normal + (1.0 - w) * g.normal;
            g.diffuse = lerp(g.diffuse, l.diffuse, w);
            g.specular = lerp(g.specular, l.specular, w);
            g.normal = dot(n, n) > 0.0 ? normalize(n) : l.normal;
            g.roughness = lerp(g.roughness, l.roughness, w);
            surface = true;
            if (l.dominant != 0)
            {
                depth = l.depth;
                motion = l.motion;
                specular_motion = l.specular_motion;
            }
        }
    }
    r.diffuse_albedo.get<RWTexture2D<float4>>()[pixel] = float4(saturate(g.diffuse), 1.0);
    r.specular_albedo.get<RWTexture2D<float4>>()[pixel] = float4(saturate(g.specular), 1.0);
    r.normal_roughness.get<RWTexture2D<float4>>()[pixel] = float4(g.normal, g.roughness);
    r.specular_motion.get<RWTexture2D<float2>>()[pixel] = specular_motion;
    r.depth.get<RWTexture2D<float>>()[pixel] = depth;
    r.motion.get<RWTexture2D<float2>>()[pixel] = motion;
    if (r.dof_guide.index != IMAGE_NONE)
        r.dof_guide.get<RWTexture2D<float>>()[pixel] = 0.0;

    float3 shown;
    switch (r.debug)
    {
    case GUIDE_DIFFUSE_ALBEDO: shown = g.diffuse; break;
    case GUIDE_SPECULAR_ALBEDO: shown = g.specular; break;
    case GUIDE_NORMAL: shown = 0.5 * g.normal + 0.5; break;
    case GUIDE_ROUGHNESS: shown = float3(g.roughness); break;
    case GUIDE_SPECULAR_MOTION: shown = float3(specular_motion, 0.0); break;
    default: return;
    }
    r.radiance.get<RWTexture2D<float4>>()[pixel] = float4(shown, surface ? 1.0 : 0.0);
}
