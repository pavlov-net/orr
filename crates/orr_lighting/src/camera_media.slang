// Media apply: every pixel's camera segment composed with its radiance. The froxels hold the
// linked medium to the scene's `near_range`, the aerial-perspective LUT and a march hold it past
// that. Medium proxies the camera ray crosses are marched per pixel over its spans inside them,
// each step lit by the froxels' isotropic light (past them, the world cache's) and by shadow rays
// from a few points of the span, and seen through the linked medium in front of it. Smoke comes
// with the froxels to `near_range`, and past it from one ray per pixel; the reference takes each
// smoke triangle on the ray in turn.
module camera_media;

import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.trace;
import orr_scene.lights;
import orr_scene.tables;
import orr_scene.view;
import orr_scene.gbuffer;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.media;
import orr_lighting.light_sampling;
import orr_lighting.path;
import orr_lighting.shadow;
import orr_lighting.froxels;
import orr_lighting.particle_lighting;

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
    public LayerGuide* layers;                  // the view's layer memory, null for none
    public StorageImageIndex radiance;          // the view's, RGBA16F
    public StorageImageIndex dof_background;    // the background dispatch's layer, NONE for none
    public uint mode;                           // MEDIA_*
    public uint reference;                      // 1: marches the camera segment
    public uint segment_steps;                  // steps of that march and of the march past `ap_range`; most steps per stretch of proxies
    public uint proxy_rays;                     // shadow rays per light over a stretch of proxies, at most MAX_PROXY_RAYS; 0 takes the froxels' visibility
}

// The linked medium's in-scatter and transmittance from the eye to `t` m along unit `dir`, the
// ray through `pixel`: the froxels to the scene's `near_range`, then the aerial-perspective LUT
// to `ap_range` and `segment_inscatter` past it.
Fog linked_fog<M : IMedium>(MediaApplyRoot r, FroxelRoot f, uint2 pixel, float3 dir, float t)
{
    ViewConstants* v = f.view;
    float near_range = f.scene.near_range;
    Fog g;
    if (f.lights.eye != 0)
    {
        uint medium = r.layers != nullptr ? layer_guide(r.layers, f.gbuffer.extent, pixel).near_interior.innermost() : MATERIAL_NONE;
        g = eye_fog(f.eye, f.integrated_sampled, f.integrated_rgb_sampled, uint3(f.columns, f.rows, f.slices), v.extent, near_range, float2(pixel) + 0.5, medium, interior_medium(f.scene, medium).sigma_t, t);
        if (medium != MATERIAL_NONE)
            return g;
    }
    else
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
        if (c.triangle || c.class_id != CLASS_MEDIUM || !proxy_span(c.object_origin, c.object_direction, t_max, s.t0, s.t1))
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

// Gives `handler` every candidate of `mask` on the ray from the eye along `dir` over
// [`t_min`, `t_max`].
[ForceInline]
void trace_camera<H : ICandidate>(FroxelRoot f, float3 dir, uint mask, float t_min, float t_max, inout H handler)
{
    RayDesc ray;
    ray.Origin = f.view.eye_render.xyz;
    ray.Direction = dir;
    ray.TMin = t_min;
    ray.TMax = t_max;
    bool overflow;
    trace_transmittance(f.scene.tlas, ray, mask, RAY_FLAG_NONE, handler, 0, overflow);
}

// Spans of the ray from the eye along `dir` to `t_max` through the proxies on screen.
ProxySpans camera_spans(FroxelRoot f, float3 dir, float t_max)
{
    ProxySpans s = (ProxySpans)0;
    s.t_max = t_max;
    if (f.lights.proxies == 0 || !(t_max > 0.0))
        return s;
    trace_camera(f, dir, MASK_MEDIUM, 0.0, t_max, s);
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

// Smoke triangles the reference's camera ray takes, the nearest; farther ones are not drawn on
// that pixel.
static const uint REFERENCE_SHEETS = 64;

// One smoke triangle on a camera ray.
struct SmokeSheet
{
    float t;
    float opacity;
    float3 albedo;
    float3 emission;
}

// `c` as a smoke triangle; false for any other candidate.
bool smoke_sheet(SceneRoot* scene, Candidate c, out SmokeSheet s)
{
    s = (SmokeSheet)0;
    s.t = c.t;
    return c.triangle && c.class_id == CLASS_PARTICLE && c.cluster_id >= 0
        && resolve_smoke(scene, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, s.albedo, s.emission, s.opacity);
}

// Keeps the nearest smoke triangle of a camera ray.
struct NearestSheet : ICandidate
{
    SceneRoot* scene;
    SmokeSheet sheet = (SmokeSheet)0;

    bool positions(Candidate c)
    {
        return false;
    }

    [mutating]
    float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        SmokeSheet s;
        if (c.t < sheet.t && smoke_sheet(scene, c, s))
            sheet = s;
        return float3(1.0);
    }
}

// The first smoke triangle of the ray from the eye along `dir` between `after` and `t_max`, at
// infinity if none. One ray per triangle, because a list of them in this kernel's locals (3 KB)
// loses the device: the driver then page-faults on stack writes of other compute kernels.
SmokeSheet next_sheet(FroxelRoot f, float3 dir, float after, float t_max)
{
    NearestSheet nearest = { f.scene, (SmokeSheet)0 };
    nearest.sheet.t = positive_infinity();
    if (f.smoke != nullptr && after < t_max)
        trace_camera(f, dir, MASK_PARTICLE, after, t_max, nearest);
    return nearest.sheet;
}

// The reference camera segment: in-scatter (pre-exposed) over `t` m from the eye along `dir`,
// to the scene's `near_range` in the linked medium on sky pixels and to the end of the spans
// in the proxies. Jittered steps at the proxies' voxel spacing, at most `segment_steps` per
// stretch, each lit by one next-event candidate with its shadow ray and by the linked
// medium's multiple scattering, with the smoke triangles between them lit the same way and
// seen through by what their opacity lets through; the linked medium's transmittance exactly.
Fog reference_fog<S : ISurface, M : IMedium>(MediaApplyRoot r, FroxelRoot f, uint2 pixel, float3 dir, float t, ProxySpans spans)
{
    SceneRoot* s = f.scene;
    LightSetGpu* set = f.set;
    uint64_t data = s.medium;
    float3 eye = f.view.eye_render.xyz;
    Sampler rng = Sampler(pixel, f.frame, f.seed ^ MEDIA_STREAM);
    float linked_end = isinf(t) ? s.near_range : t;
    // On a sky pixel smoke reaches as far as the aerial-perspective LUT.
    float end = isinf(t) ? max(max(linked_end, spans.end()), f.smoke != nullptr ? f.ap_range : 0.0) : t;
    SmokeSheet sheet = (SmokeSheet)0;
    float sheet_after = 0.0;
    uint sheets = 0;
    float3 inscatter = float3(0.0);
    float3 through = float3(1.0);
    float x = 0.0;
    for (uint j = 0; j < 2 * MAX_PROXY_SPANS + 2 && x < end; j++)
    {
        float next;
        uint active = spans.piece(x, end, next);
        if (x < linked_end)
            next = min(next, linked_end);
        float h = spans.step_size(s, active, next - x, r.segment_steps);
        // Past the linked medium and the proxies only smoke is left.
        uint steps = active == 0 && x >= linked_end ? 0 : uint(ceil((next - x) / h - 1e-3));
        h = (next - x) / float(max(steps, 1u));
        uint k = 0;
        float step_at = steps > 0 ? x + rng.next() * h : next;
        // One call site each: a linked surface and shadow ray compile once per call.
        while (true)
        {
            if (sheet_after >= 0.0)
            {
                sheet = next_sheet(f, dir, sheets < REFERENCE_SHEETS ? sheet_after : end, end);
                sheet_after = -1.0;
            }
            if (!(k < steps || sheet.t < next))
                break;
            bool at_sheet = sheet.t < step_at;
            float d = at_sheet ? sheet.t : step_at;
            float3 p = eye + dir * d;
            MediumPoint m = (MediumPoint)0;
            if (!at_sheet && d < linked_end)
                m = M.sample(data, p);
            NeeVertex nv = nee_vertex(set, p, float3(0.0), pixel, nullptr);
            ResolvedLightSample c = nee_candidate<S>(set, nv, rng);
            float cos_theta = dot(c.dir, dir);
            float3 sigma_t = float3(0.0);
            float3 sigma_s = m.sigma_s;
            float3 scatter = any(m.sigma_s > 0.0) ? m.sigma_s * M.phase(data, m, cos_theta) : float3(0.0);
            float3 source = m.emission;
            float3 origin = p;
            if (at_sheet)
            {
                sigma_s = sheet.opacity * sheet.albedo;
                scatter = sigma_s * hg_phase(f.smoke_anisotropy, cos_theta);
                source = sheet.opacity * sheet.emission;
                origin += c.dir * T_MIN_BASE * (1.0 + d * T_MIN_SCALE);
            }
            else
            {
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
            }
            if (any(sigma_s > 0.0))
            {
                if (c.pdf > 0.0)
                {
                    float3 l = c.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, c.light, c.dir) : c.radiance;
                    float t_max = isfinite(c.distance) ? c.distance * (1.0 - 1e-4) : c.distance;
                    RayCone cone = { 0.0, 0.0 };
                    float3 vis = visibility<S, M>(s, f.view, origin, c.dir, t_max, RAY_FLAG_NONE, hit_context(HIT_SECONDARY, 1, cone, 0.0), c.kind == LIGHT_SAMPLE_ENVIRONMENT, Interior(), set.shadow_overflows);
                    if (c.view && any(vis > 0.0))
                        vis *= emitter_view_layer<S>(set, c, AMBIENT_IOR);
                    source += scatter * l * vis / c.pdf;
                }
                source += sigma_s * multiple_scattered<M>(set, p);
            }
            source *= medium_transmittance<M>(data, eye, dir, at_sheet ? d : min(d, linked_end));
            if (at_sheet)
            {
                inscatter += through * source;
                through *= 1.0 - sheet.opacity;
                sheet_after = d + T_MIN_BASE * (1.0 + d * T_MIN_SCALE);
                sheets++;
                continue;
            }
            march_step(sigma_t, source, h, inscatter, through);
            k++;
            step_at = k < steps ? x + (float(k) + rng.next()) * h : next;
        }
        x = next;
    }
    Fog out;
    out.inscatter = inscatter * f.view.exposure[f.view.exposure_slot].pre_exposure;
    out.transmittance = through * medium_transmittance<M>(data, eye, dir, linked_end);
    return out;
}

// The smoke triangles of a camera ray past `near_range`, in any order: what they let through,
// and their opacity-weighted distance, albedo and emission.
struct FarSmoke : ICandidate
{
    SceneRoot* scene;
    float through = 1.0;
    float weight = 0.0;
    float distance = 0.0;
    float3 albedo = float3(0.0);
    float3 emission = float3(0.0);

    bool positions(Candidate c)
    {
        return false;
    }

    [mutating]
    float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        SmokeSheet s;
        if (smoke_sheet(scene, c, s))
        {
            through *= 1.0 - s.opacity;
            weight += s.opacity;
            distance += s.opacity * s.t;
            albedo += s.opacity * s.albedo;
            emission += s.opacity * s.emission;
        }
        return float3(1.0);
    }
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

// `camera_fog`, with the smoke past `near_range` on the ray as one lit layer at its mean distance,
// its color the mean weighted by opacity: the fog in front of the layer, the layer, and through
// the layer what lies behind it. The reference takes that smoke in its own march. On sky pixels
// the fog ends at `near_range` and the environment holds the medium past it, so the difference of
// the two fogs puts back what the layer must not take from the stretch in front of it.
Fog smoky_fog<S : ISurface, M : IMedium>(MediaApplyRoot r, FroxelRoot f, uint2 pixel, float3 dir, float t, ProxySpans spans)
{
    float near_range = f.scene.near_range;
    FarSmoke smoke = { f.scene };
    if (r.reference == 0 && f.lights.far_smoke != 0 && t > near_range)
        trace_camera(f, dir, MASK_PARTICLE, near_range, t, smoke);
    bool layer = smoke.weight > 0.0;
    float smoke_t = layer ? smoke.distance / smoke.weight : t;
    Fog front = (Fog)0;
    Fog fog = (Fog)0;
    // One call site: a linked surface and shadow ray compile once per call.
    for (uint k = layer ? 0 : 1; k < 2; k++)
    {
        front = fog;
        fog = camera_fog<S, M>(r, f, pixel, dir, k == 0 ? smoke_t : t, spans);
    }
    if (!layer)
        return fog;
    ViewConstants* v = f.view;
    Sampler rng = Sampler(pixel, f.frame, f.seed ^ PARTICLE_STREAM);
    float3 p = v.eye_render.xyz + dir * smoke_t;
    float3 light = particle_light<S, M>(r.lighting, p, -dir, f.smoke_anisotropy, rng) + multiple_scattered<M>(f.set, p);
    float3 color = (smoke.albedo * light + smoke.emission) / smoke.weight;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    fog.inscatter = front.inscatter + front.transmittance * (1.0 - smoke.through) * color * pre_exposure
        + smoke.through * (fog.inscatter - front.inscatter);
    fog.transmittance *= smoke.through;
    return fog;
}

// Media apply for `pixel`: L = S + T x L over the camera segment to the pixel's surface, or
// the dielectric or presence layer it shades; sky pixels take the froxels' whole range, and
// pixels whose path carried the segment through an interior nothing. Marked pixels of the
// background layer take the same over the segment to their layer's depth. One ray finds the
// pixel's proxy spans for both.
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
        Fog fog = smoky_fog<S, M>(r, f, pixel, dir, layer == 0 ? t : tb, spans);
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
