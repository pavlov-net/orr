// Ray reconstruction guides. Per pixel: diffuse and specular albedo from the surface's BSDF,
// world normal and linear roughness, depth and motion from the G-buffer, and specular motion.
// Pixels with a dielectric or presence layer blend in its guide by its weight, and take its
// depth and motion where it dominates.
// A primary surface that is a near-mirror follows its mirror chain and takes the guides of
// the first surface past it, seen through the chain. Pixels smoother than the specular-motion
// roughness trace the mirror direction and reproject the reflection's virtual point.
module guides;

import orr.abi;
import orr.color;
import orr.trace;
import media;
import hit;
import visibility;
import bsdf;
import path;

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
    // One per pixel in rows of the G-buffer's width, from this frame's dielectric dispatch;
    // null for none.
    public LayerGuide* layers;
    public StorageImageIndex radiance;          // RGBA16F, written only for a GUIDE_* debug view
    public StorageImageIndex diffuse_albedo;    // RGBA8
    public StorageImageIndex specular_albedo;   // RGBA8
    public StorageImageIndex normal_roughness;  // RGBA16F
    public StorageImageIndex specular_motion;   // RG16F
    public StorageImageIndex depth;             // R32F
    public StorageImageIndex motion;            // RG16F
    public float motion_roughness;              // specular motion below this roughness
    public uint debug;                          // GUIDE_*
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

public struct Guides
{
    public float3 diffuse;
    public float3 specular;
    public float3 normal;
    public float roughness;
}

// A surface's guides from its BSDF's albedo: subsurface pixels add the subsurface share in
// the subsurface color to the diffuse albedo.
public Guides surface_guides(GSurface s, LobeAlbedo a)
{
    Guides g;
    g.diffuse = a.diffuse + a.subsurface * s.bsdf.subsurface_color;
    g.specular = a.specular;
    g.normal = s.normal;
    g.roughness = s.bsdf.specular_roughness;
    return g;
}

public Guides sky_guides()
{
    Guides g;
    g.diffuse = float3(SKY_DIFFUSE_ALBEDO);
    g.specular = float3(0.0);
    g.normal = float3(0.0);
    g.roughness = 0.0;
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
    return (previous.xy / previous.w - current.xy / current.w) * 0.5;
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

// Traces the mirror direction from `x1` (guides `g1`). With `chain`, keeps reflecting off
// near-mirrors for up to `MIRROR_CHAIN` bounces and returns the guides of the surface past
// the last one, scaled by the chain's specular albedo, with its normal mirrored through the
// chain into the view; otherwise one bounce and `g1`. The virtual point lies along the camera
// ray at the camera distance plus every bounce's length.
public Reflection mirror_chain<S : ISurface>(SceneRoot* s, ViewConstants* v, PathVertex x1, Guides g1, bool chain)
{
    Reflection out;
    out.guides = g1;
    float3 camera_dir = -x1.view;
    float travelled = x1.camera_distance;
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
        Alpha<S> a = { s, v, dir, trace_context };
        float3 positions[3];
        Hit h = trace_closest(s.tlas, path_ray(vertex_origin(x, dir), dir, positive_infinity()), MASK_OPAQUE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
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
        travelled += h.t;
        out.virtual_point = camera_dir * travelled;
        out.w = 1.0;
        if (!chain)
            return out;
        cone = cone_after_hit(cone, h.t, 0.0);
        SurfaceHit sh = resolve_hit<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, hit_context(HIT_SECONDARY, i + 1, cone, h.t));
        if (!sh.valid)
            return out;
        x = hit_vertex(v, sh, dir);
        Bsdf b = bsdf_prepare(bsdf_collapse(x.surface.bsdf), x.surface.normal, x.view);
        g = surface_guides(x.surface, b.albedo());
        out.guides = g;
        out.guides.diffuse *= throughput;
        out.guides.specular *= throughput;
        out.guides.normal = mul(mirrors, g.normal);
        if (!near_delta(g))
            return out;
    }
    return out;
}

// The guide resolve's body for `pixel`, hits shading through linked surface `S`.
public void guide_body<S : ISurface>(GuideRoot r, uint2 pixel)
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
        g = surface_guides(x1.surface, b.albedo());
        bool chain = near_delta(g);
        if (chain || g.roughness < r.motion_roughness)
        {
            Reflection m = mirror_chain<S>(s, v, x1, g, chain);
            g = m.guides;
            specular_motion = reprojected_motion(v, m.virtual_point, m.w);
        }
    }
    if (r.layers != nullptr)
    {
        LayerGuide l = r.layers[pixel.y * gb.extent.x + pixel.x];
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
