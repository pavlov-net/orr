// The lean path tracer's stages for surfaces whose BSDF may have the rich lobes of
// `path.lean_rich` and a subsurface lobe, in a scene whose surfaces may walk (`MAY_WALK`): the
// hit shader of such a surface, which starts a vertex's walk and runs the vertex where one leaves,
// and the callables of a path's first vertex and of a debug view of it. In such a scene they take
// the place of `path.lean_rich`'s.
#include "orr_root.h"
import orr_shaders.pipeline;
import orr_shaders.trace;
import orr_lighting.bsdf;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_scene.candidates;
import orr_scene.runs;
import orr_scene.hit;
import orr_lighting.path;
import orr_lighting.path.lean.vertex;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;
export static const uint BSDF_CLASS = CLASS_COAT | CLASS_FUZZ | CLASS_FILM | CLASS_THIN | CLASS_SUBSURFACE;

ORR_ROOT_INLINE(PathRoot);

[shader("closesthit")]
void surface_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(surface_hit);
    TriangleHit h = { a };
    lean_hit<Surface, Medium, TriangleHit, SectionGiven>(root, DispatchRaysIndex().x, p.words, section_given(data.material), h);
    // Tells the raygen a shader took the hit.
    p.words.x = 0;
}

[shader("callable")]
void first_vertex(inout Payload p)
{
    ORR_OMM(first_vertex);
    lean_first<Medium>(root, DispatchRaysIndex().x, p.words);
}

[shader("callable")]
void walk_leaves(inout Payload p)
{
    ORR_OMM(walk_leaves);
    lean_walk_exit<Medium>(root, DispatchRaysIndex().x, p.words);
}

[shader("callable")]
void emitter_view(inout Payload p)
{
    ORR_OMM(emitter_view);
    lean_emitter_view<Surface>(root, DispatchRaysIndex().x, p.words);
}

[shader("callable")]
void first_view(inout Payload p)
{
    ORR_OMM(first_view);
    lean_first_view<Surface>(root, DispatchRaysIndex().x);
}
