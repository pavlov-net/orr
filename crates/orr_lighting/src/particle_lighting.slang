// Light and fog for blended particles drawn at the transparency point. Inside the view's
// froxel volume a particle takes the froxels' isotropic incident light and the visibility of
// their shadowed directional lights; beyond it, the world cache's light on two opposed faces,
// one shadow ray from its point toward each shadowed light and the other directional lights
// unshadowed. Its fog is the in-scatter and transmittance from the eye, the froxels' to the
// scene's `near_range` and the aerial-perspective LUT's past it, so a particle writes
// a x (C x T + S) into the overlay, which composites over radiance that media apply already
// fogged.
module particle_lighting;

import orr_shaders.abi;
import orr_shaders.atomic;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.media;
import orr_lighting.light_sampling;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.shadow;
import orr_lighting.froxels;

// Separates particles' samples from the paths' of the same pixel.
public static const uint PARTICLE_STREAM = 0x68E31DA4;

// What a view's transparency hooks light and fog particles with, for one frame.
public struct ParticleLighting
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public LightSetGpu* set;
    public WorldCacheGpu* cache;
    public FroxelLights* lights;        // the froxels' shadowed directional lights
    public EyeColumn* eye;              // per froxel column; null without froxels
    public LayerGuide* layers;          // the dielectric dispatch's layer memory, null for none
    public uint64_t* overlay_front;     // per pixel of the view, its front layer; null for none
    public TextureIndex incident;       // froxel_incident; NONE without froxels
    public TextureIndex visibility;     // the shadowed lights past the first
    public TextureIndex integrated;     // froxel_integrated
    public TextureIndex integrated_rgb; // rgb transmittance; NONE for its luminance
    public SamplerIndex sampler;        // linear, clamped
    public uint columns;
    public uint rows;
    public uint slices;
    public uint frame;
    public uint seed;
    public TextureIndex aerial;         // the aerial-perspective LUT
    public TextureIndex aerial_rgb;     // its rgb transmittance; NONE for its luminance
    public float ap_range;              // m, the LUT's far end
    public TextureIndex eye_depth;      // per shadowed light, m its ray spends in the column's interior, times its visibility; NONE for none
}

// The light at render-space `x` that does not depend on a direction, as mean radiance over the
// sphere: inside the froxels their incident light, with `vis` the visibility of their shadowed
// lights, and `inside` true; else the world cache's on two opposed faces along unit `to_eye`,
// with `vis` 1. Draws two sampler dimensions.
public float3 particle_ambient(ParticleLighting* l, float3 x, float3 to_eye, inout Sampler rng, out bool inside, out float3 vis[MAX_FROXEL_SHADOWED_LIGHTS])
{
    SceneRoot* s = l.scene;
    float3 coord = float3(0.0);
    inside = l.incident.index != IMAGE_NONE
        && froxel_coord(uint3(l.columns, l.rows, l.slices), l.view, s.near_range, x, coord);
    for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
        vis[j] = float3(1.0);
    if (!inside)
        return particle_irradiance(l.cache, s, x, to_eye, rng) * (0.5 * INV_PI);
    SamplerState ss = l.sampler.get();
    float4 incident = l.incident.get<Texture3D<float4>>().SampleLevel(ss, coord, 0.0);
    float4 more = l.lights.count > 1 ? l.visibility.get<Texture3D<float4>>().SampleLevel(ss, coord, 0.0) : float4(1.0);
    vis[0] = float3(incident.a);
    vis[1] = float3(more.x);
    vis[2] = float3(more.y);
    vis[3] = float3(more.z);
    if (l.eye_depth.index != IMAGE_NONE && l.lights.eye != 0)
    {
        uint2 column = min(uint2(coord.xy * float2(l.columns, l.rows)), uint2(l.columns, l.rows) - 1u);
        EyeColumn eye = l.eye[column.y * l.columns + column.x];
        if (uint(coord.z * float(l.slices)) < eye.slices)
        {
            float3 sigma_t = interior_medium(s, eye.medium).sigma_t;
            float4 depth = l.eye_depth.get<Texture3D<float4>>().SampleLevel(ss, coord, 0.0);
            for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
                vis[j] *= vis[j].x > 0.0 ? beer(sigma_t, depth[j] / vis[j].x) : float3(1.0);
        }
    }
    rng.next2();
    return incident.rgb;
}

// One shadow ray from render-space `x` toward each of the froxels' shadowed lights.
public void particle_shadow_rays<S : ISurface, M : IMedium>(ParticleLighting* l, float3 x, out float3 vis[MAX_FROXEL_SHADOWED_LIGHTS])
{
    SceneRoot* s = l.scene;
    float offset = T_MIN_BASE * (1.0 + length(x - l.view.eye_render.xyz) * T_MIN_SCALE);
    RayCone cone = { 0.0, 0.0 };
    uint count = l.lights.count;
    for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
    {
        vis[j] = float3(1.0);
        if (j >= count)
            continue;
        float3 to_light = s.lights[l.lights.ids[j]].direction;
        float3 depth;
        vis[j] = surface_visibility<S, M>(s, l.view, x + to_light * offset, to_light, positive_infinity(), RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, hit_context(HIT_SECONDARY, 1, cone, 0.0), false, l.set.shadow_overflows, depth);
        vis[j] *= exp(-depth);
    }
}

// The directional light at render-space `x` scattering toward the eye along unit `to_eye` per
// unit albedo, with a Henyey-Greenstein response of `anisotropy`: the shadowed lights by `vis`
// and, outside the froxels, whose incident light holds them inside, the others unshadowed.
public float3 particle_directional<M : IMedium>(ParticleLighting* l, float3 x, float3 to_eye, float anisotropy, bool inside, float3 vis[MAX_FROXEL_SHADOWED_LIGHTS])
{
    SceneRoot* s = l.scene;
    LightSetGpu* set = l.set;
    FroxelLights lights = *l.lights;
    float3 light = float3(0.0);
    for (uint k = 0; k < set.directional_count; k++)
    {
        uint id = set.directional_ids[k];
        float3 v = float3(1.0);
        bool shadowed = false;
        for (uint j = 0; j < lights.count; j++)
        {
            if (lights.ids[j] == id)
            {
                v = vis[j];
                shadowed = true;
            }
        }
        if ((inside && !shadowed) || !any(v > 0.0))
            continue;
        float3 to_light = s.lights[id].direction;
        light += directional_illuminance(set, id) * v * hg_phase(anisotropy, dot(-to_light, to_eye))
            * M.transmittance(s.medium, x, to_light, positive_infinity());
    }
    return light;
}

// Radiance a particle of unit albedo at render-space `x` scatters toward the eye along unit
// `to_eye`, unexposed, with a Henyey-Greenstein response of `anisotropy` to directional
// light; the rest scatters isotropically. Draws two sampler dimensions.
public float3 particle_light<S : ISurface, M : IMedium>(ParticleLighting* l, float3 x, float3 to_eye, float anisotropy, inout Sampler rng)
{
    bool inside;
    float3 vis[MAX_FROXEL_SHADOWED_LIGHTS];
    float3 light = particle_ambient(l, x, to_eye, rng, inside, vis);
    if (!inside)
        particle_shadow_rays<S, M>(l, x, vis);
    return light + particle_directional<M>(l, x, to_eye, anisotropy, inside, vis);
}

// The in-scatter (pre-exposed) and transmittance from the eye to render-space `x` along the
// ray through image point `p`, in render pixels: the froxels' to the scene's `near_range`,
// the aerial-perspective LUT's to `ap_range` and the linked medium `M`'s transmittance past
// it. A camera segment starting in interior `medium` (MATERIAL_NONE for none) goes on through it
// past the froxels.
public float3 particle_fog<M : IMedium>(ParticleLighting* l, float2 p, float3 x, uint medium, out float3 transmittance)
{
    transmittance = float3(1.0);
    if (l.integrated.index == IMAGE_NONE)
        return float3(0.0);
    ViewConstants* v = l.view;
    float near_range = l.scene.near_range;
    float3 eye = v.eye_render.xyz;
    float t = length(x - eye);
    uint3 grid = uint3(l.columns, l.rows, l.slices);
    float3 fog;
    if (l.lights.eye != 0)
    {
        Fog g = eye_fog(l.eye, l.integrated, l.integrated_rgb, grid, v.extent, near_range, p, medium, interior_medium(l.scene, medium).sigma_t, t);
        fog = g.inscatter;
        transmittance = g.transmittance;
        if (medium != MATERIAL_NONE)
            return fog;
    }
    else
        fog = integrated_fog(l.integrated, l.integrated_rgb, l.sampler, grid, v.extent, near_range, p, t, transmittance);
    if (!(t > near_range))
        return fog;
    float ap_range = max(l.ap_range, near_range);
    float3 far = float3(1.0);
    if (ap_range > near_range)
        fog += transmittance * aerial_fog(l.aerial, l.aerial_rgb, l.sampler, v.extent, near_range, ap_range, p, t, far);
    if (t > ap_range)
    {
        float3 dir = (x - eye) / t;
        far *= medium_transmittance<M>(l.scene.medium, eye + dir * ap_range, dir, t - ap_range);
    }
    transmittance *= far;
    return fog;
}

// The overlay value, premultiplied and pre-exposed, of a particle fragment at image point `p`
// (render pixels, `SV_Position.xy`) and render-space `x`, of `albedo`, `emission` (nits,
// unexposed), `opacity` and directional response `anisotropy`: opacity x (C x T + S), with C
// its light and emission and T and S its fog. Nothing behind the dielectric or presence layer
// the pixel shades, which shows what lies behind it through its own rays; no fog where the
// dispatch's path carried the camera segment through an interior.
public float4 particle_shade<S : ISurface, M : IMedium>(ParticleLighting* l, float2 p, float3 x, float3 albedo, float3 emission, float opacity, float anisotropy)
{
    ViewConstants* v = l.view;
    float3 to_eye = v.eye_render.xyz - x;
    float t = length(to_eye);
    to_eye /= max(t, 1e-30);
    float end = 0.0;
    uint medium = MATERIAL_NONE;
    if (l.layers != nullptr)
    {
        LayerGuide guide = layer_guide(l.layers, v.extent, min(uint2(p), v.extent - 1));
        end = guide.medium_end;
        medium = guide.eye_medium - 1;
    }
    if (end > 0.0 && t > end)
        return float4(0.0);
    float3 transmittance = float3(1.0);
    float3 fog = float3(0.0);
    if (!(end < 0.0))
        fog = particle_fog<M>(l, p, x, medium, transmittance);
    Sampler rng = Sampler(uint2(p), l.frame, l.seed ^ PARTICLE_STREAM);
    float3 c = albedo * particle_light<S, M>(l, x, to_eye, anisotropy, rng) + emission;
    float a = saturate(opacity);
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    return float4(apply_hdr_ceiling(a * (c * pre_exposure * transmittance + fog)), a);
}

// Coverage above which an overlay fragment can be the pixel's front layer.
static const float OVERLAY_FRONT_COVERAGE = 0.1;

// Makes a fragment at image point `p` of triangle `primitive` of `hit`, visible cluster `vc`,
// the pixel's front layer if its `coverage` passes OVERLAY_FRONT_COVERAGE and nothing nearer
// holds the pixel. A pixel entry calling it declares [earlydepthstencil], or fragments behind
// the opaque depth would write too.
public void particle_front(ParticleLighting* l, VisibleCluster vc, PrimaryHit hit, uint primitive, float2 p, float coverage)
{
    if (l.overlay_front == nullptr || coverage <= OVERLAY_FRONT_COVERAGE)
        return;
    ViewConstants* v = l.view;
    uint2 pixel = min(uint2(p), v.extent - 1);
    float2 motion = visible_motion(l.scene, v, vc, hit, primitive, hit.relative);
    atomic_max_u64(l.overlay_front + pixel.y * v.extent.x + pixel.x, overlay_front_word(device_depth(v, hit.input.position), motion, 0.0));
}

// The stock particle draw's fragment at image point `p` of triangle `primitive` of the view's
// visible cluster `visible` (`RasterGpu` `raster`'s): `particle_shade` with the surface's base
// color as albedo, its emission, and the material's opacity for presence materials, else the
// surface's coverage. The fragment also enters `particle_front`.
public float4 particle_fragment<S : ISurface, M : IMedium>(ParticleLighting* l, RasterGpu* raster, uint visible, uint primitive, float2 p, float anisotropy)
{
    VisibleCluster vc = raster.gbuffer.clusters[visible];
    PrimaryHit hit = primary_hit(l.scene, l.view, vc, primitive, p);
    SurfaceInput i = hit.input;
    float offset;
    GSurface g = S.surface(i, offset);
    bool presence = (l.scene.materials[i.material].flags & MATERIAL_PRESENCE) != 0;
    float opacity = presence ? material_opacity(i) : S.coverage(i);
    float4 shaded = particle_shade<S, M>(l, p, i.position, g.bsdf.base_color, g.bsdf.emission, opacity, anisotropy);
    particle_front(l, vc, hit, primitive, p, shaded.a);
    return shaded;
}
