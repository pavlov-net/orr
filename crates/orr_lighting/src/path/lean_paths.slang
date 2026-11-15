// The lean path tracer's common stages (`path.lean`): the raygens of a view's paths and of its
// update paths, the callable of a path's first vertex, the hit shader of a plain opaque surface, the any-hit shaders of a path's ray and
// of its shadow ray at a candidate that is not opaque, and the miss shaders of the two.
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

extern struct Surface : ISurfaces = StandardSurface;
extern struct Medium : IMedium = NoMedium;
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
    lean_first<Medium>(root, DispatchRaysIndex().x);
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

[shader("miss")]
void miss_scatter(inout Payload p)
{
    ORR_OMM(miss_scatter);
    lean_miss<Medium>(root, DispatchRaysIndex().x);
}

[shader("miss")]
void miss_shadow(inout Payload p)
{
    ORR_OMM(miss_shadow);
    lean_shadow_miss(root, DispatchRaysIndex().x, asfloat(p.words.xyz));
}
