// The pipeline path tracer's shaders (`path.launch`): a raygen per source, the closest-hit shaders
// that resolve a path ray's hit on triangles and on strands, the walk meet's closest-hit and the
// miss shader, the coverage's any-hit shaders and the medium proxies' under shadow rays, what
// LAYERS rays keep of particles, light proxies and medium proxies, and the callables that start,
// shade and finish a path, end its walks and take what lies along its rays.
// `orr_lighting::launch` makes several stages of one entry by its specialization constants
// (`path_may`, `path_may2`, `hit_surface`): a hit shader per kind of geometry and per linked
// surface id, and next-event estimation and the scatter per BSDF class.
#include "orr_root.h"
import orr_shaders.pipeline;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_scene.candidates;
import orr_lighting.path;
import orr_lighting.path.launch;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

[shader("raygeneration")]
void path_view()
{
    ORR_OMM(path_view);
    launch_paths<Medium>(root, CALL_START_VIEW, CALL_FINISH_VIEW, false);
}

[shader("raygeneration")]
void path_update()
{
    ORR_OMM(path_update);
    launch_paths<Medium>(root, CALL_START_UPDATE, CALL_FINISH_RECORD, false);
}

[shader("raygeneration")]
void path_photon()
{
    ORR_OMM(path_photon);
    launch_paths<Medium>(root, CALL_START_PHOTON, CALL_FINISH_RECORD, true);
}

[shader("raygeneration")]
void path_background()
{
    ORR_OMM(path_background);
    launch_paths<Medium>(root, CALL_START_BACKGROUND, CALL_FINISH_BACKGROUND, false);
}

[shader("miss")]
void miss_path(inout Payload p)
{
    ORR_OMM(miss_path);
    launch_miss<Surface, Medium>(root, launch_slot());
    p.words.x = STEP_ENDED;
}

[shader("closesthit")]
void resolve(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(resolve);
    TriangleHit h = { a };
    p.words.x = launch_resolve<Surface, TriangleHit>(root, launch_slot(), h);
}

[shader("closesthit")]
void resolve_strand(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(resolve_strand);
    BuiltInHit h = { a };
    p.words.x = launch_resolve<Surface, BuiltInHit>(root, launch_slot(), h);
}

// The closest-hit of the walk meet's record, which an interior walk's step invokes on the face it
// met in place of the face's own.
[shader("closesthit")]
void walk_meet(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(walk_meet);
    TriangleHit h = { a };
    p.words = launch_walk_meet(root, h);
}

[shader("anyhit")]
void cover_path_hit(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(cover_path_hit);
    TriangleHit h = { a };
    uint d = cover_path<CoverageFeedback<Surface>>(root.scene, root.view, path_cover(p.words), h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
    if (d == ANY_HIT_IGNORE)
        IgnoreHit();
}

// Ends with what the decision says, the payload holding the shadow's state.
void shadow_decided(inout Payload p, ShadowCover c, uint d)
{
    p.words = uint4(asuint(c.product), c.word);
    if (d == ANY_HIT_END)
        AcceptHitAndEndSearch();
    IgnoreHit();
}

[shader("anyhit")]
void cover_shadow_hit(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(cover_shadow_hit);
    TriangleHit h = { a };
    ShadowCover c = { asfloat(p.words.xyz), p.words.w };
    uint d = cover_shadow<CoverageFeedback<Surface>>(root.scene, root.view, c.context(), c, h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
    shadow_decided(p, c, d);
}

struct SpanAttributes
{
    float2 span;
}

[shader("intersection")]
void box_medium_hit()
{
    ORR_OMM(box_medium_hit);
    float2 span = box_medium(ObjectRayOrigin(), ObjectRayDirection(), RayTCurrent());
    SpanAttributes a = { span };
    ReportHit(clamp(span.x, RayTMin(), RayTCurrent()), 0, a);
}

[shader("anyhit")]
void proxy_shadow_hit(inout Payload p, SpanAttributes a)
{
    ORR_OMM(proxy_shadow_hit);
    ShadowCover c = { asfloat(p.words.xyz), p.words.w };
    uint d = proxy_shadow(root.scene, c, InstanceIndex(), a.span);
    shadow_decided(p, c, d);
}

[shader("anyhit")]
void cover_layers_hit(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(cover_layers_hit);
    TriangleHit h = { a };
    layers_triangle(root, p.words, h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.t(), h.barycentrics());
    IgnoreHit();
}

[shader("anyhit")]
void cover_layers_medium_hit(inout Payload p, SpanAttributes a)
{
    ORR_OMM(cover_layers_medium_hit);
    layers_medium(root, p.words, InstanceIndex(), PrimitiveIndex(), a.span);
    IgnoreHit();
}

struct LightAttributes
{
    float t;
}

[shader("intersection")]
void box_light_hit()
{
    ORR_OMM(box_light_hit);
    float t;
    if (box_light(root, InstanceIndex(), WorldRayOrigin(), WorldRayDirection(), RayTCurrent(), t))
    {
        LightAttributes a = { t };
        ReportHit(t, 0, a);
    }
}

[shader("anyhit")]
void cover_layers_light_hit(inout Payload p, LightAttributes a)
{
    ORR_OMM(cover_layers_light_hit);
    layers_light(root, p.words, InstanceIndex(), a.t);
    IgnoreHit();
}

[shader("callable")]
void start_view(inout Payload p)
{
    ORR_OMM(start_view);
    p.words.x = launch_start_view<Surface>(root, launch_slot());
}

[shader("callable")]
void start_update(inout Payload p)
{
    ORR_OMM(start_update);
    p.words.x = launch_start_update(root, launch_slot());
}

[shader("callable")]
void start_photon(inout Payload p)
{
    ORR_OMM(start_photon);
    p.words.x = launch_start_photon<Surface>(root, launch_slot());
}

[shader("callable")]
void start_background(inout Payload p)
{
    ORR_OMM(start_background);
    p.words.x = launch_start_background(root, launch_slot());
}

[shader("callable")]
void resolve_given(inout Payload p)
{
    ORR_OMM(resolve_given);
    p.words.x = launch_resolve_given<Surface>(root, launch_slot());
}

[shader("callable")]
void emission(inout Payload p)
{
    ORR_OMM(emission);
    launch_emission(root, launch_slot());
}

[shader("callable")]
void direct(inout Payload p)
{
    ORR_OMM(direct);
    p.words.x = launch_direct<Surface, Medium>(root, launch_slot());
}

[shader("callable")]
void deposit(inout Payload p)
{
    ORR_OMM(deposit);
    p.words.x = launch_deposit<Surface>(root, launch_slot());
}

[shader("callable")]
void cache_vertex(inout Payload p)
{
    ORR_OMM(cache_vertex);
    launch_cache_vertex(root, launch_slot());
}

[shader("callable")]
void scatter(inout Payload p)
{
    ORR_OMM(scatter);
    p.words.x = launch_scatter<Surface>(root, launch_slot());
}

// Each finish has the path's pass in its data's second word.
[shader("callable")]
void finish_view(inout Payload p)
{
    ORR_OMM(finish_view);
    p.words.x = launch_finish_view<Environment, Medium>(root, launch_slot(), p.words.y);
}

[shader("callable")]
void finish_record(inout Payload p)
{
    ORR_OMM(finish_record);
    p.words.x = launch_finish_record(root, launch_slot());
}

[shader("callable")]
void finish_background(inout Payload p)
{
    ORR_OMM(finish_background);
    p.words.x = launch_finish_background<Environment, Medium>(root, launch_slot());
}

[shader("callable")]
void walk_end(inout Payload p)
{
    ORR_OMM(walk_end);
    p.words.x = launch_walk_end(root, launch_slot());
}

[shader("callable")]
void direct_exit(inout Payload p)
{
    ORR_OMM(direct_exit);
    p.words.x = launch_direct_exit<Surface, Medium>(root, launch_slot());
}

[shader("callable")]
void scatter_exit(inout Payload p)
{
    ORR_OMM(scatter_exit);
    p.words.x = launch_scatter_exit<Surface>(root, launch_slot());
}

[shader("callable")]
void direct_leave(inout Payload p)
{
    ORR_OMM(direct_leave);
    p.words.x = launch_direct_leave<Surface, Medium>(root, launch_slot());
}

[shader("callable")]
void scatter_leave(inout Payload p)
{
    ORR_OMM(scatter_leave);
    p.words.x = launch_scatter_leave<Surface>(root, launch_slot());
}

[shader("callable")]
void segment(inout Payload p)
{
    ORR_OMM(segment);
    p.words.x = launch_segment<Medium>(root, launch_slot(), p.words.x != 0, asfloat(p.words.y));
}

[shader("callable")]
void layers(inout Payload p)
{
    ORR_OMM(layers);
    float bound;
    p.words.x = launch_layers(root, launch_slot(), p.words.x, asfloat(p.words.y), bound);
    p.words.y = asuint(bound);
}

[shader("callable")]
void march_proxy(inout Payload p)
{
    ORR_OMM(march_proxy);
    float bound;
    launch_proxy_march(root, launch_slot(), p.words.x, bound);
    p.words.y = asuint(bound);
}

[shader("callable")]
void direct_medium(inout Payload p)
{
    ORR_OMM(direct_medium);
    p.words.x = launch_direct_medium<Surface, Medium>(root, launch_slot());
}

[shader("callable")]
void scatter_medium(inout Payload p)
{
    ORR_OMM(scatter_medium);
    p.words.x = launch_scatter_medium<Surface>(root, launch_slot());
}
