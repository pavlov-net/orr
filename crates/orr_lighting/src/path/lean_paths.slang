// The lean path tracer's stages (`path.lean`): the raygens of a view's paths, of its background
// layer's and of its update paths; the callables of a path's first vertex and of a debug view of it
// alone, the ones that start a path in the interiors around the eye or at a thin lens's point, the
// one that runs its vertex in an interior's medium, the one of a step of a directional job's walk
// through a plane stack, the one that takes what a path's ray meets in front of its hit and the one
// that marches the medium proxies with a volume a shadow ray passed; the hit shaders of a plain
// opaque surface and of a strand, the any-hit shaders of a path's ray and of its shadow ray at a
// candidate that is not opaque, a medium proxy's intersection and any-hit shaders under a shadow
// ray, and the miss shaders of the two.
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
import orr_lighting.path.lean.vertex;
import orr_lighting.path.lean.raygen;
import orr_lighting.path.lean.stack;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Medium : IMedium = NoMedium;
extern struct Environment : IEnvironment = EnvironmentMap;
export static const bool PATH_SPECIALIZES = true;
export static const uint BSDF_CLASS = 0;

ORR_ROOT_INLINE(PathRoot);

[shader("raygeneration")]
void paths()
{
    ORR_OMM(paths);
    lean_paths(root);
}

[shader("closesthit")]
void surface_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(surface_hit);
    TriangleHit h = { a };
    lean_hit<Surface, Medium, TriangleHit, SectionGiven>(root, DispatchRaysIndex().x, p.words, section_given(data.material), h);
    // Tells the raygen a shader took the hit.
    p.words.x = 0;
}

[shader("closesthit")]
void strand_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(strand_hit);
    BuiltInHit h = { a };
    lean_hit<Surface, Medium, BuiltInHit, SectionGiven>(root, DispatchRaysIndex().x, p.words, section_given(data.material), h);
    // Tells the raygen a shader took the hit.
    p.words.x = 0;
}

[shader("raygeneration")]
void backgrounds()
{
    ORR_OMM(backgrounds);
    lean_backgrounds<Environment, Medium>(root);
}

[shader("raygeneration")]
void updates()
{
    ORR_OMM(updates);
    lean_updates(root);
}

[shader("callable")]
void first_vertex(inout Payload p)
{
    ORR_OMM(first_vertex);
    lean_first<Medium>(root, DispatchRaysIndex().x, p.words);
}

[shader("callable")]
void eye_interior(inout Payload p)
{
    ORR_OMM(eye_interior);
    p.words.x = lean_eye_interior<Surface>(root, DispatchRaysIndex().x, p.words.x != 0) ? 1 : 0;
}

[shader("callable")]
void eye_lens(inout Payload p)
{
    ORR_OMM(eye_lens);
    lean_lens_interior<Surface>(root, DispatchRaysIndex().x);
}

[shader("callable")]
void medium_vertex(inout Payload p)
{
    ORR_OMM(medium_vertex);
    lean_medium_vertex<Medium>(root, DispatchRaysIndex().x, p.words);
}

[shader("callable")]
void stack_step(inout StackStep c)
{
    ORR_OMM(stack_step);
    lean_stack_step<Medium>(root, DispatchRaysIndex().x, c);
}

[shader("callable")]
void front(inout Payload p)
{
    ORR_OMM(front);
    p.words.x = lean_front<Medium>(root, DispatchRaysIndex().x, asfloat(p.words.x), p.words.y, p.words.z != 0) ? 1 : 0;
}

[shader("callable")]
void volume_march(inout ProxyMarch m)
{
    ORR_OMM(volume_march);
    lean_proxy_march(root, m);
}

[shader("callable")]
void first_view(inout Payload p)
{
    ORR_OMM(first_view);
    lean_first_view<Surface>(root, DispatchRaysIndex().x);
}

[shader("anyhit")]
void cover_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(cover_hit);
    TriangleHit h = { a };
    any_hit_decides(path_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, p.words, data, h));
}

[shader("anyhit")]
void shadow_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(shadow_hit);
    TriangleHit h = { a };
    any_hit_decides(shadow_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, p.words, data, h));
}

// A medium proxy under a shadow ray: its box reports the ray's span inside it, at the span's start,
// and the any-hit takes Beer's law over the span, or marks a proxy with a volume for its march.
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

[shader("miss")]
void miss_scatter(inout Payload p)
{
    ORR_OMM(miss_scatter);
    lean_miss<Medium, Environment>(root, DispatchRaysIndex().x, p.words.y != 0);
}

[shader("miss")]
void miss_shadow(inout Payload p)
{
    ORR_OMM(miss_shadow);
    // Whatever traced the ray reads what reached its end from the payload.
    p.words.w |= SHADOW_MISSED;
}
