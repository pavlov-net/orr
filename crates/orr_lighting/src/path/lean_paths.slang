// The lean path tracer's common stages (`path.lean`): the hit shader of a plain opaque surface and
// the miss shaders of a path's ray and of its shadow ray.
#include "orr_root.h"
import orr_shaders.pipeline;
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

[shader("miss")]
void miss_scatter(inout Payload p)
{
    ORR_OMM(miss_scatter);
    lean_miss(root, DispatchRaysIndex().x);
}

[shader("miss")]
void miss_shadow(inout Payload p)
{
    ORR_OMM(miss_shadow);
    lean_shadow_miss(root, DispatchRaysIndex().x, asfloat(p.words.xyz));
}
