// The pipeline path tracer's shaders (`path.launch`): a raygen per source, the closest-hit shaders
// that resolve a path ray's opaque, layer or strand hit, the walk meet's and the stack meet's
// closest-hit and the miss shader, the coverage's any-hit shaders and the medium proxies' under
// shadow rays, what LAYERS rays keep of particles, light proxies and medium proxies, and the
// callables that start, shade and finish a path, end its walks, take what lies along its rays,
// pass a shadow ray through a window, walk it through a plane stack and through a coated
// emitter's view. `orr_lighting::launch` makes several stages of one entry by its
// specialization constants (`path_may`, `path_may2`, `hit_surface`): a hit shader per kind of
// geometry and per linked surface id, and next-event estimation and the scatter per BSDF class.
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

// Reads in the payload's first word whether the raygen traced the probe it asked for.
[shader("miss")]
void miss_path(inout Payload p)
{
    ORR_OMM(miss_path);
    p.words.x = launch_miss<Surface, Medium>(root, launch_slot(), p.words.x == STEP_PROBE);
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

// The closest-hit of the stack meet's record, which a plane stack's trace invokes on a dielectric
// face it met in place of the face's own: the face's cluster.
[shader("closesthit")]
void stack_meet(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(stack_meet);
    TriangleHit h = { a };
    p.words.x = asuint(h.cluster_id());
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

[shader("closesthit")]
void layer_resolve(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(layer_resolve);
    TriangleHit h = { a };
    p.words.x = launch_resolve_layer<Surface, TriangleHit>(root, launch_slot(), h);
}

// A shadow ray's crossing step commits nothing: it keeps the nearest crossing past the last.
[shader("anyhit")]
void cover_path_hit(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(cover_path_hit);
    TriangleHit h = { a };
    any_hit_decides(path_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, p.words, h, slot_crossings(root, launch_slot())));
}

[shader("anyhit")]
void cover_shadow_hit(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(cover_shadow_hit);
    TriangleHit h = { a };
    any_hit_decides(shadow_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, p.words, h));
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
    any_hit_decides(proxy_any_hit(root.scene, p.words, InstanceIndex(), a.span));
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
    p.words.x = launch_start_update<Surface>(root, launch_slot());
}

[shader("callable")]
void gbuffer_start(inout Payload p)
{
    ORR_OMM(gbuffer_start);
    p.words.x = launch_gbuffer(root, launch_slot());
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

// The direct callable of vertex class `K` (AT_*), and the deposit, take DIRECT_* in their data's
// second word. A direct callable hands back there the sampler dimension of its record's vertex,
// which `cache_vertex` takes.
void direct_of<let K : uint>(inout Payload p)
{
    uint dimension;
    p.words.x = launch_direct<K, Surface, Medium>(root, launch_slot(), p.words.y, dimension);
    p.words.y = dimension;
}

[shader("callable")]
void direct(inout Payload p)
{
    ORR_OMM(direct);
    direct_of<AT_PLAIN>(p);
}

[shader("callable")]
void deposit(inout Payload p)
{
    ORR_OMM(deposit);
    p.words.x = launch_deposit<Surface>(root, launch_slot(), p.words.y == DIRECT_PROBED);
}

[shader("callable")]
void cache_vertex(inout Payload p)
{
    ORR_OMM(cache_vertex);
    launch_cache_vertex(root, launch_slot(), p.words.y);
}

[shader("callable")]
void scatter(inout Payload p)
{
    ORR_OMM(scatter);
    p.words.x = launch_scatter<AT_PLAIN, Surface>(root, launch_slot());
}

// Each finish has the path's pass in its data's second word.
[shader("callable")]
void finish_view(inout Payload p)
{
    ORR_OMM(finish_view);
    p.words.x = launch_finish_view<Surface, Environment, Medium>(root, launch_slot(), p.words.y);
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
    direct_of<AT_EXIT>(p);
}

[shader("callable")]
void scatter_exit(inout Payload p)
{
    ORR_OMM(scatter_exit);
    p.words.x = launch_scatter<AT_EXIT, Surface>(root, launch_slot());
}

[shader("callable")]
void direct_leave(inout Payload p)
{
    ORR_OMM(direct_leave);
    direct_of<AT_LEAVE>(p);
}

[shader("callable")]
void scatter_leave(inout Payload p)
{
    ORR_OMM(scatter_leave);
    p.words.x = launch_scatter<AT_LEAVE, Surface>(root, launch_slot());
}

[shader("callable")]
void direct_layer(inout Payload p)
{
    ORR_OMM(direct_layer);
    direct_of<AT_LAYER>(p);
}

[shader("callable")]
void scatter_layer(inout Payload p)
{
    ORR_OMM(scatter_layer);
    p.words.x = launch_scatter<AT_LAYER, Surface>(root, launch_slot());
}

// Takes the hit flag and distance in, and hands the step back.
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
    p.words.x = launch_proxy_march(root, launch_slot(), p.words.x, bound);
    p.words.y = asuint(bound);
}

[shader("callable")]
void direct_medium(inout Payload p)
{
    ORR_OMM(direct_medium);
    direct_of<AT_SCATTER>(p);
}

[shader("callable")]
void scatter_medium(inout Payload p)
{
    ORR_OMM(scatter_medium);
    p.words.x = launch_scatter<AT_SCATTER, Surface>(root, launch_slot());
}

[shader("callable")]
void window(inout Payload p)
{
    ORR_OMM(window);
    p.words.x = launch_window<Surface>(root, launch_slot());
}

// Takes the phase in its data's second word, and hands back what follows.
[shader("callable")]
void stack_step(inout Payload p)
{
    ORR_OMM(stack_step);
    p.words.x = launch_stack_step<Medium>(root, launch_slot(), p.words.y);
}

[shader("callable")]
void emitter_view(inout Payload p)
{
    ORR_OMM(emitter_view);
    launch_emitter_view<Surface>(root, launch_slot());
}
