// The pipeline path tracer's hit shaders (`path.launch`): the closest-hit shaders that resolve a
// path ray's hit on triangles or strands and a shadow ray's crossing in window mode, and the stack
// meet's, the coverage's any-hit shaders and the medium proxies' under shadow rays, and what path
// rays keep of particles and light proxies in front of their hit and LAYERS rays of those and of
// medium proxies. `orr_lighting::launch` makes a
// stage of each resolve and coverage entry per kind of geometry and per linked surface id by its
// specialization constants (`path_may2`, `hit_surface`).
#include "orr_root.h"
import orr_shaders.pipeline;
import orr_shaders.trace;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_scene.candidates;
import orr_scene.runs;
import orr_scene.hit;
import orr_lighting.path;
import orr_lighting.path.launch;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

// The closest-hit shaders of a path ray's hit, chosen by its record: of an opaque surface, of a
// dielectric or a presence layer, and of a strand, which is opaque. Each resolves the hit from
// its payload and its record's section material (`orr_scene.runs`' `HitData`) and runs its whole
// vertex, or only resolves it where the raygen runs the vertex (`launch_hit`); a layer's also
// serves a shadow ray's crossing in window mode. `orr_lighting::launch` makes the opaque one
// twice, with and without the lobes only a rich material has.
[shader("closesthit")]
void vertex_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(vertex_hit);
    TriangleHit h = { a };
    p.words = launch_hit<AT_SURFACE, Surface, Medium, TriangleHit, SectionGiven>(root, launch_slot(), p.words, section_given(data.material), h);
}

[shader("closesthit")]
void layer_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(layer_hit);
    TriangleHit h = { a };
    p.words = launch_hit<AT_LAYER, Surface, Medium, TriangleHit, SectionGiven>(root, launch_slot(), p.words, section_given(data.material), h);
}

[shader("closesthit")]
void strand_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(strand_hit);
    BuiltInHit h = { a };
    p.words = launch_hit<AT_SURFACE, Surface, Medium, BuiltInHit, SectionGiven>(root, launch_slot(), p.words, section_given(data.material), h);
}

// The closest-hit of the stack meet's record, which a plane stack's trace invokes on a dielectric
// face it met in place of the face's own: it keeps the face as the stack ray's hit.
[shader("closesthit")]
void stack_meet(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(stack_meet);
    TriangleHit h = { a };
    keep_stack_hit(root, launch_slot(), true, true, CLASS_DIELECTRIC, h.t(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics(), h.world_origin(), h.world_direction());
}

// The any-hit shaders start from their record's data (`orr_scene.runs`' `HitData`): the class
// their candidate traces as and its section material.
[shader("anyhit")]
void cover_path_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(cover_path_hit);
    TriangleHit h = { a };
    any_hit_decides(path_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, p.words, data, h));
}

[shader("anyhit")]
void cover_shadow_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(cover_shadow_hit);
    TriangleHit h = { a };
    any_hit_decides(shadow_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, p.words, data, h));
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
void front_path_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(front_path_hit);
    TriangleHit h = { a };
    front_triangle(root, p.words, data, h.instance_index(), h.cluster_id(), h.primitive_index(), h.t(), h.barycentrics());
    IgnoreHit();
}

[shader("anyhit")]
void cover_layers_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(cover_layers_hit);
    TriangleHit h = { a };
    layers_triangle(root, p.words, data, h.instance_index(), h.cluster_id(), h.primitive_index(), h.t(), h.barycentrics());
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
void front_path_light_hit(inout Payload p, LightAttributes a)
{
    ORR_OMM(front_path_light_hit);
    front_light(root, p.words, InstanceIndex(), a.t);
    IgnoreHit();
}

[shader("anyhit")]
void cover_layers_light_hit(inout Payload p, LightAttributes a)
{
    ORR_OMM(cover_layers_light_hit);
    layers_light(root, p.words, InstanceIndex(), a.t);
    IgnoreHit();
}
