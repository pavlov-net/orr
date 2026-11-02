// Media apply: every pixel's camera segment composed with its radiance. The froxels hold the
// linked medium to the scene's `near_range`, the aerial-perspective LUT and a march hold it past
// that. Medium proxies the camera ray crosses are marched per pixel over its spans inside them,
// each step lit by the froxels' isotropic light (past them, the world cache's) and by shadow rays
// from a few points of the span, and seen through the linked medium in front of it.
module camera_media;

import orr.abi;
import orr.math;
import orr.color;
import orr.random;
import orr.trace;
import lights;
import tables;
import view;
import gbuffer;
import hit;
import visibility;
import media;
import light_sampling;
import path;
import shadow;
import froxels;
import particle_lighting;

// Separates media apply's samples from the paths' of the same pixel.
static const uint MEDIA_STREAM = 0xC2B2AE35;

// `MediaApplyRoot.mode`: what media apply writes.
public static const uint MEDIA_COMPOSE = 0;
public static const uint MEDIA_INSCATTER = 1;   // the in-scatter alone
public static const uint MEDIA_FROXELS = 2;     // transmittance in r, slice coordinate / slices in g, a column checker in b
public static const uint MEDIA_OFF = 3;         // radiance left as the paths wrote it

// Medium proxies a camera ray takes, the nearest; farther ones are not drawn on that pixel.
public static const uint MAX_PROXY_SPANS = 4;

// Most shadow rays per light a camera ray traces over a stretch of proxies.
public static const uint MAX_PROXY_RAYS = 8;

public struct MediaApplyRoot
{
    public FroxelRoot* froxels;                 // this frame's
    public ParticleLighting* lighting;          // this frame's: lights the proxies' steps
    public LayerGuide* layers;                  // the dielectric dispatch's layer memory, null for none
    public StorageImageIndex radiance;          // the view's, RGBA16F
    public StorageImageIndex dof_background;    // the opaque dispatch's background layer, NONE for none
    public uint mode;                           // MEDIA_*
    public uint reference;                      // 1: marches the camera segment
    public uint segment_steps;                  // steps of that march and of the march past `ap_range`; most steps per stretch of proxies
    public uint proxy_rays;                     // shadow rays per light over a stretch of proxies, at most MAX_PROXY_RAYS; 0 takes the froxels' visibility
}

// In-scatter (pre-exposed) and transmittance from the eye.
struct Fog
{
    float3 inscatter;
    float3 transmittance;
}

// The linked medium's in-scatter and transmittance from the eye to `t` m along unit `dir`, the
// ray through `pixel`: the froxels to the scene's `near_range`, then the aerial-perspective LUT
// to `ap_range` and `segment_inscatter` past it.
Fog linked_fog<M : IMedium>(MediaApplyRoot r, FroxelRoot f, uint2 pixel, float3 dir, float t)
{
    ViewConstants* v = f.view;
    float near_range = f.scene.near_range;
    Fog g;
    g.inscatter = integrated_fog(f.integrated_sampled, f.integrated_rgb_sampled, f.sampler, uint3(f.columns, f.rows, f.slices), v.extent, near_range, float2(pixel) + 0.5, min(t, near_range), g.transmittance);
    if (!(t > near_range))
        return g;
    float ap_range = max(f.ap_range, near_range);
    float3 through = float3(1.0);
    float3 inscatter = float3(0.0);
    if (ap_range > near_range)
        inscatter = aerial_fog(f.aerial_sampled, f.aerial_rgb_sampled, f.sampler, v.extent, near_range, ap_range, float2(pixel) + 0.5, t, through);
    if (t > ap_range)
    {
        Sampler rng = Sampler(pixel, f.frame, f.seed ^ MEDIA_STREAM);
        float3 more;
        float3 past = segment_inscatter<M>(f.set, v.eye_render.xyz + dir * ap_range, dir, t - ap_range, r.segment_steps, rng.next(), more);
        inscatter += through * past * v.exposure[v.exposure_slot].pre_exposure;
        through *= more;
    }
    g.inscatter += g.transmittance * inscatter;
    g.transmittance *= through;
    return g;
}

// One stretch of a camera ray inside a proxy's cube, with the ray in the cube's object space.
struct ProxySpan
{
    float t0;
    float t1;
    uint instance;
    float3 origin;
    float3 direction;
}

// The nearest MAX_PROXY_SPANS spans of a camera ray up to `t_max`.
struct ProxySpans : ICandidate
{
    float t_max;
    uint count;
    ProxySpan spans[MAX_PROXY_SPANS];

    bool positions(Candidate c)
    {
        return false;
    }

    [mutating]
    float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        ProxySpan s;
        if (c.triangle || c.custom_index != CLASS_MEDIUM || !proxy_span(c.object_origin, c.object_direction, t_max, s.t0, s.t1))
            return float3(1.0);
        s.instance = c.instance_index;
        s.origin = c.object_origin;
        s.direction = c.object_direction;
        uint slot = count;
        if (count == MAX_PROXY_SPANS)
        {
            slot = 0;
            for (uint i = 1; i < count; i++)
            {
                if (spans[i].t0 > spans[slot].t0)
                    slot = i;
            }
            if (spans[slot].t0 <= s.t0)
                return float3(1.0);
        }
        else
            count++;
        spans[slot] = s;
        return float3(1.0);
    }

    // The proxies covering the piece of the ray from `x`, as a bit per span, and that piece's
    // end: the next span boundary, at most `end`.
    uint piece(float x, float end, out float piece_end)
    {
        uint active = 0;
        piece_end = end;
        for (uint i = 0; i < count; i++)
        {
            if (spans[i].t0 <= x && x < spans[i].t1)
            {
                active |= 1u << i;
                piece_end = min(piece_end, spans[i].t1);
            }
            else if (spans[i].t0 > x)
                piece_end = min(piece_end, spans[i].t0);
        }
        return active;
    }

    // Where the last span ends.
    float end()
    {
        float e = 0.0;
        for (uint i = 0; i < count; i++)
            e = max(e, spans[i].t1);
        return e;
    }

    // The step of a march over `len` m of the spans in `active`: the finest volume's voxel
    // spacing, or `len` / `steps` where that is farther.
    float step_size(SceneRoot* scene, uint active, float len, uint steps)
    {
        float voxel = positive_infinity();
        for (uint i = 0; i < count; i++)
        {
            if (((active >> i) & 1u) != 0)
                voxel = min(voxel, proxy_voxel_step(scene_proxy(scene, spans[i].instance), spans[i].direction));
        }
        float h = len / float(max(steps, 1u));
        return isinf(voxel) ? h : max(voxel, h);
    }
}

// Spans of the ray from the eye along `dir` to `t_max` through the proxies on screen.
ProxySpans camera_spans(FroxelRoot f, float3 dir, float t_max)
{
    ProxySpans s = (ProxySpans)0;
    s.t_max = t_max;
    if (f.lights.proxies == 0 || !(t_max > 0.0))
        return s;
    RayDesc ray;
    ray.Origin = f.view.eye_render.xyz;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    bool overflow;
    trace_transmittance(f.scene.tlas, ray, MASK_MEDIUM, RAY_FLAG_NONE, s, 0, overflow);
    return s;
}

// A camera ray's march through the linked medium and its proxies, front to back: between
// steps it adds the linked medium's in-scatter over the distance covered, seen through the
// proxies so far.
struct ProxyMarch<S : ISurface, M : IMedium> : IProxyStep
{
    MediaApplyRoot root;
    FroxelRoot froxels;
    uint2 pixel;
    float3 dir;
    bool sky;
    ProxySpans spans;
    uint active;
    Sampler rng;
    float pre_exposure;
    float3 inscatter;
    float3 through;     // the proxies' transmittance so far
    Fog fog;            // the linked medium's at `at`
    float at;
    // The stretch's shadow rays, evenly spaced along it from `ray_start`, traced as steps first
    // need them: a bit per ray in `traced`.
    float ray_start;
    float ray_spacing;
    uint traced;
    float3 ray_vis[MAX_PROXY_RAYS][MAX_FROXEL_SHADOWED_LIGHTS];
    // The linked medium's in-scatter past `near_range` this march has added: a sky pixel's
    // environment holds it too.
    float3 past;

    [mutating]
    void advance(float x)
    {
        float near_range = froxels.scene.near_range;
        Fog g = linked_fog<M>(root, froxels, pixel, dir, x);
        float3 added = max(g.inscatter - fog.inscatter, 0.0);
        inscatter += through * added;
        if (sky && x > near_range)
            past += at < near_range ? max(g.inscatter - linked_fog<M>(root, froxels, pixel, dir, near_range).inscatter, 0.0) : added;
        fog = g;
        at = x;
    }

    float majorant(float t, out float t_exit)
    {
        float m = 0.0;
        t_exit = positive_infinity();
        for (uint i = 0; i < spans.count; i++)
        {
            if (((active >> i) & 1u) == 0)
                continue;
            ProxySpan span = spans.spans[i];
            float exit;
            m += proxy_majorant_span(scene_proxy(froxels.scene, span.instance), span.origin, span.direction, t, exit);
            t_exit = min(t_exit, exit);
        }
        return m;
    }

    [mutating]
    void begin_stretch(float x0, float x1, uint spans_active, float jitter)
    {
        active = spans_active;
        traced = 0;
        ray_spacing = (x1 - x0) / float(max(root.proxy_rays, 1u));
        ray_start = x0 + jitter * ray_spacing;
    }

    // Light at `p`, `x` m along the ray, scattering toward the eye per unit albedo with a
    // Henyey-Greenstein response of `anisotropy`. The froxels' visibility stands where no rays
    // are asked for; past the froxels there is none, so one ray is traced.
    [mutating]
    float3 light(float3 p, float x, float anisotropy)
    {
        bool inside;
        float3 vis[MAX_FROXEL_SHADOWED_LIGHTS];
        float3 ambient = particle_ambient(root.lighting, p, -dir, rng, inside, vis);
        if (root.proxy_rays > 0 || !inside)
        {
            // Between the two nearest rays, optical depth being what runs linearly along the
            // ray; steps before the first and past the last take theirs.
            uint last = max(root.proxy_rays, 1u) - 1;
            float u = clamp((x - ray_start) / ray_spacing, 0.0, float(last));
            uint a = uint(u);
            uint b = min(a + 1, last);
            for (uint i = a; i <= b; i++)
            {
                if (((traced >> i) & 1u) != 0)
                    continue;
                traced |= 1u << i;
                float3 traced_vis[MAX_FROXEL_SHADOWED_LIGHTS];
                particle_shadow_rays<S, M>(root.lighting, froxels.view.eye_render.xyz + dir * (ray_start + float(i) * ray_spacing), traced_vis);
                ray_vis[i] = traced_vis;
            }
            for (uint j = 0; j < froxels.lights.count; j++)
                vis[j] = exp(lerp(log(max(ray_vis[a][j], 1e-4)), log(max(ray_vis[b][j], 1e-4)), u - float(a)));
        }
        return ambient + particle_directional<M>(root.lighting, p, -dir, anisotropy, inside, vis);
    }

    [mutating]
    void step(float x, float h)
    {
        // Past `ap_range` a lookup marches, so steps there keep the last one.
        if (at == 0.0 || x <= max(froxels.ap_range, froxels.scene.near_range))
            advance(x);
        SceneRoot* s = froxels.scene;
        float3 p = froxels.view.eye_render.xyz + dir * x;
        float3 sigma_t = float3(0.0);
        float3 source = float3(0.0);
        for (uint i = 0; i < spans.count; i++)
        {
            if (((active >> i) & 1u) == 0)
                continue;
            ProxySpan span = spans.spans[i];
            ProxyMedium m = proxy_medium(scene_proxy(s, span.instance), span.origin + span.direction * x);
            sigma_t += m.sigma_t;
            if (any(m.sigma_s > 0.0))
                source += m.sigma_s * (light(p, x, m.anisotropy) + multiple_scattered<M>(froxels.set, p));
        }
        march_step(sigma_t, source * fog.transmittance * pre_exposure, h, inscatter, through);
    }
}

// The camera segment over `t` m (+inf on sky pixels) along unit `dir`, through the proxies of
// `spans`. A sky pixel's environment holds the linked medium past `near_range`, so its
// stretches in front of a proxy past `near_range` come back out of what the proxy lets through.
Fog proxy_fog<S : ISurface, M : IMedium>(MediaApplyRoot r, FroxelRoot f, uint2 pixel, float3 dir, float t, ProxySpans spans)
{
    ViewConstants* v = f.view;
    bool sky = isinf(t);
    Fog clear = { float3(0.0), float3(1.0) };
    ProxyMarch<S, M> march;
    march.root = r;
    march.froxels = f;
    march.pixel = pixel;
    march.dir = dir;
    march.sky = sky;
    march.spans = spans;
    march.rng = Sampler(pixel, f.frame, f.seed ^ MEDIA_STREAM);
    march.pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    march.inscatter = float3(0.0);
    march.through = float3(1.0);
    march.fog = clear;
    march.at = 0.0;
    march.past = float3(0.0);
    float end = sky ? spans.end() : t;
    float x = 0.0;
    for (uint k = 0; k < 2 * MAX_PROXY_SPANS + 1 && x < end; k++)
    {
        float next;
        uint active = spans.piece(x, end, next);
        if (active != 0)
        {
            float jitter = march.rng.next();
            march.begin_stretch(x, next, active, march.rng.next());
            float h = spans.step_size(f.scene, active, next - x, r.segment_steps);
            proxy_march(x, next, h, jitter, march);
        }
        x = next;
    }
    Fog out;
    if (sky)
    {
        float near_range = f.scene.near_range;
        if (march.at < near_range)
            march.advance(near_range);
        out.inscatter = max(march.inscatter - march.through * march.past, 0.0);
        out.transmittance = march.through * linked_fog<M>(r, f, pixel, dir, near_range).transmittance;
    }
    else
    {
        march.advance(t);
        out.inscatter = march.inscatter;
        out.transmittance = march.through * march.fog.transmittance;
    }
    return out;
}

// The reference camera segment: in-scatter (pre-exposed) over `t` m from the eye along `dir`,
// to the scene's `near_range` in the linked medium on sky pixels and to the end of the spans
// in the proxies. Jittered steps at the proxies' voxel spacing, at most `segment_steps` per
// stretch, each lit by one next-event candidate with its shadow ray and by the linked
// medium's multiple scattering; the linked medium's transmittance exactly.
Fog reference_fog<S : ISurface, M : IMedium>(MediaApplyRoot r, FroxelRoot f, uint2 pixel, float3 dir, float t, ProxySpans spans)
{
    SceneRoot* s = f.scene;
    LightSetGpu* set = f.set;
    uint64_t data = s.medium;
    float3 eye = f.view.eye_render.xyz;
    Sampler rng = Sampler(pixel, f.frame, f.seed ^ MEDIA_STREAM);
    float linked_end = isinf(t) ? s.near_range : t;
    float end = isinf(t) ? max(linked_end, spans.end()) : t;
    float3 inscatter = float3(0.0);
    float3 through = float3(1.0);
    float x = 0.0;
    for (uint j = 0; j < 2 * MAX_PROXY_SPANS + 1 && x < end; j++)
    {
        float next;
        uint active = spans.piece(x, end, next);
        float h = spans.step_size(s, active, next - x, r.segment_steps);
        uint steps = uint(ceil((next - x) / h - 1e-3));
        h = (next - x) / float(max(steps, 1u));
        for (uint k = 0; k < steps; k++)
        {
            float d = x + (float(k) + rng.next()) * h;
            float3 p = eye + dir * d;
            MediumPoint m = (MediumPoint)0;
            if (d < linked_end)
                m = M.sample(data, p);
            NeeVertex nv = { p, float3(0.0), pixel, nullptr };
            ResolvedLightSample c = nee_candidate(set, nv, rng);
            float cos_theta = dot(c.dir, dir);
            float3 sigma_t = float3(0.0);
            float3 sigma_s = m.sigma_s;
            float3 scatter = any(m.sigma_s > 0.0) ? m.sigma_s * M.phase(data, m, cos_theta) : float3(0.0);
            for (uint i = 0; i < spans.count; i++)
            {
                if (((active >> i) & 1u) == 0)
                    continue;
                ProxySpan span = spans.spans[i];
                ProxyMedium pm = proxy_medium(scene_proxy(s, span.instance), span.origin + span.direction * d);
                sigma_t += pm.sigma_t;
                sigma_s += pm.sigma_s;
                scatter += pm.sigma_s * hg_phase(pm.anisotropy, cos_theta);
            }
            float3 source = m.emission;
            if (any(sigma_s > 0.0))
            {
                if (c.pdf > 0.0)
                {
                    float3 l = c.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, c.light, c.dir) : c.radiance;
                    float t_max = isfinite(c.distance) ? c.distance * (1.0 - 1e-4) : c.distance;
                    RayCone cone = { 0.0, 0.0 };
                    float3 vis = visibility<S, M>(s, f.view, p, c.dir, t_max, RAY_FLAG_NONE, hit_context(HIT_SECONDARY, 1, cone, 0.0), c.kind != LIGHT_SAMPLE_ENVIRONMENT, set.shadow_overflows);
                    source += scatter * l * vis / c.pdf;
                }
                source += sigma_s * multiple_scattered<M>(set, p);
            }
            march_step(sigma_t, source * medium_transmittance<M>(data, eye, dir, min(d, linked_end)), h, inscatter, through);
        }
        x = next;
    }
    Fog out;
    out.inscatter = inscatter * f.view.exposure[f.view.exposure_slot].pre_exposure;
    out.transmittance = through * medium_transmittance<M>(data, eye, dir, linked_end);
    return out;
}

// The camera segment's in-scatter (pre-exposed) and transmittance over `t` m along unit `dir`,
// the ray through `pixel`: `linked_fog`, or through the proxies of `spans` where it crosses
// any. Sky pixels (`t` infinite) take the froxels alone; the environment carries the rest.
Fog camera_fog<S : ISurface, M : IMedium>(MediaApplyRoot r, FroxelRoot f, uint2 pixel, float3 dir, float t, ProxySpans spans)
{
    if (r.reference != 0)
        return reference_fog<S, M>(r, f, pixel, dir, t, spans);
    if (spans.count != 0 && t > 0.0)
        return proxy_fog<S, M>(r, f, pixel, dir, t, spans);
    return linked_fog<M>(r, f, pixel, dir, isinf(t) ? f.scene.near_range : t);
}

// Media apply for `pixel`: L = S + T x L over the camera segment to the pixel's surface, or
// the dielectric or presence layer it shades; sky pixels take the froxels' whole range, and
// pixels whose eye lies in an interior nothing. Marked pixels of the background layer take
// the same over the segment to their layer's depth. One ray finds the pixel's proxy spans for
// both.
public void media_apply<S : ISurface, M : IMedium>(MediaApplyRoot r, uint2 pixel)
{
    FroxelRoot f = *r.froxels;
    ViewConstants* v = f.view;
    if (r.mode == MEDIA_OFF || any(pixel >= v.extent))
        return;
    RWTexture2D<float4> radiance = r.radiance.get<RWTexture2D<float4>>();
    float4 l = radiance[pixel];
    float depth = f.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
    float3 dir = render_direction(v, view_direction(v, uv));
    float t = depth > 0.0 ? length(view_position(v, uv, depth)) : positive_infinity();
    float end = r.layers != nullptr ? layer_guide(r.layers, f.gbuffer.extent, pixel).medium_end : 0.0;
    if (end != 0.0)
        t = max(end, 0.0);

    bool layered = r.mode == MEDIA_COMPOSE && r.dof_background.index != IMAGE_NONE;
    float4 b = layered ? r.dof_background.get<RWTexture2D<float4>>()[pixel] : float4(-1.0);
    layered = b.a >= 0.0;
    float tb = end < 0.0 ? 0.0 : b.a > 0.0 ? length(view_position(v, uv, b.a)) : positive_infinity();
    ProxySpans spans = camera_spans(f, dir, layered ? max(t, tb) : t);

    // One call site: a linked surface and shadow ray compile once per call.
    for (uint layer = 0; layer < (layered ? 2u : 1u); layer++)
    {
        Fog fog = camera_fog<S, M>(r, f, pixel, dir, layer == 0 ? t : tb, spans);
        if (layer == 1)
        {
            r.dof_background.get<RWTexture2D<float4>>()[pixel] = float4(apply_hdr_ceiling(fog.inscatter + fog.transmittance * b.rgb), b.a);
            continue;
        }
        float4 out = float4(apply_hdr_ceiling(fog.inscatter + fog.transmittance * l.rgb), l.a);
        if (r.mode == MEDIA_INSCATTER)
            out.rgb = apply_hdr_ceiling(fog.inscatter);
        else if (r.mode == MEDIA_FROXELS)
        {
            uint2 column = uint2(round((float2(pixel) + 0.5) / column_spacing(f)));
            out.rgb = float3(luminance_rec2020(fog.transmittance), froxel_slice(t, f.slices, f.scene.near_range) / float(f.slices), float((column.x + column.y) & 1));
        }
        radiance[pixel] = out;
    }
}
