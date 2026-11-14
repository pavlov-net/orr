// The pipeline path tracer's hit shaders (`path.launch`): the closest-hit shaders that resolve a
// path ray's opaque, layer or strand hit, the walk meet's and the stack meet's, the coverage's
// any-hit shaders and the medium proxies' under shadow rays, and what LAYERS rays keep of
// particles, light proxies and medium proxies. `orr_lighting::launch` makes a stage of each
// resolve and coverage entry per kind of geometry and per linked surface id by its specialization
// constants (`path_may2`, `hit_surface`).
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

[shader("closesthit")]
void resolve(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(resolve);
    TriangleHit h = { a };
    float t;
    p.words.x = launch_resolve<Surface, TriangleHit>(root, launch_slot(), h, t);
    p.words.y = asuint(t);
}

[shader("closesthit")]
void resolve_strand(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(resolve_strand);
    BuiltInHit h = { a };
    float t;
    p.words.x = launch_resolve<Surface, BuiltInHit>(root, launch_slot(), h, t);
    p.words.y = asuint(t);
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
    float t;
    p.words.x = launch_resolve_layer<Surface, TriangleHit>(root, launch_slot(), h, t);
    p.words.y = asuint(t);
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
