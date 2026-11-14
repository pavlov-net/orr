// The rays of `hit_rays` traced inline, through orr_lighting's candidate handlers.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.candidates;
import orr_scene.surface;
import orr_lighting.media;
import orr_lighting.path;
import orr_lighting.shadow;
import lighting_suite.hit_rays;

typealias Surface = CoverageFeedback<StandardSurface>;

ORR_ROOT_INLINE(HitRaysRoot);

// Path rays through the inline handlers.
[shader("compute")]
[numthreads(64, 1, 1)]
void inline_paths(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(inline_paths);
    if (id.x >= root.count)
        return;
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RayIn r = root.rays[id.x];
    Hit h;
    float3 positions[3] = { float3(0.0), float3(0.0), float3(0.0) };
    if (r.policy == COVER_LAYERS)
    {
        LayerAccept<Surface> a = { s, v, r.dir, context(root), r.seed, r.presence };
        h = trace_closest(scene_traced(s), ray_of(r), path_mask(r), path_flags(r), a, positions);
    }
    else if (r.policy == COVER_STACK)
        h = stack_trace<Surface>(s, v, r.origin, r.dir, r.t_max, path_flags(r), context(root));
    else
    {
        Alpha<Surface> a = { s, v, r.dir, context(root) };
        h = trace_closest(scene_traced(s), ray_of(r), path_mask(r), path_flags(r), a, positions);
    }
    PathOut o = path_out(h);
    uint material;
    if (h.hit && r.policy != COVER_STACK && h.cluster_id >= 0 && hit_geometry(s, r.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, o.normal, material))
        o.material = material;
    root.paths[id.x] = o;
}

// Shadow rays through `ShadowCandidates` in mode MODE.
ShadowOut inline_shadow<let MODE : uint>(RayIn r)
{
    ShadowCandidates<Surface, NoMedium, MODE> c;
    c.scene = root.scene;
    c.view = root.view;
    c.origin = r.origin;
    c.dir = r.dir;
    c.t_max = r.t_max;
    c.context = shadow_cover_of(root, r).context();
    c.environment = false;
    c.exterior_ior = 1.0;
    c.crossed = 0;
    c.solid_t = positive_infinity();
    bool overflow;
    ShadowOut o = ShadowOut();
    o.product = trace_transmittance(scene_traced(root.scene), ray_of(r), shadow_mask(r), r.flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, c, SHADOW_CANDIDATES, overflow, o.committed_class, o.committed_t);
    o.flags = (overflow ? SHADOW_OVERFLOW : 0) | (c.solid_t < positive_infinity() ? SHADOW_SOLID : 0);
    if (c.solid_t < positive_infinity())
        o.committed_t = c.solid_t;
    o.crossed = c.crossed;
    return o;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void inline_shadows(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(inline_shadows);
    if (id.x >= root.count)
        return;
    RayIn r = root.rays[id.x];
    ShadowOut o;
    if (r.mode == SHADOW_PHOTON)
        o = inline_shadow<SHADOW_PHOTON>(r);
    else if (r.mode == SHADOW_UNBENT)
        o = inline_shadow<SHADOW_UNBENT>(r);
    else
        o = inline_shadow<SHADOW_PATH>(r);
    root.shadows[id.x] = o;
}

