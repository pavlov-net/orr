// The lean path tracer's stages for surfaces whose BSDF may have a coat, a fuzz, a thin film or a
// thin wall's transmission (`HIT_LOBED`): the hit shader of such a surface, and the callables of a
// path's first vertex and of a debug view of it, which any surface of the G-buffer may hold.
// `path.lean_paths` holds the other stages; a stage of this kernel folds the lobes a scene's mask
// rules out.
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
export static const uint BSDF_CLASS = CLASS_COAT | CLASS_FUZZ | CLASS_FILM | CLASS_THIN;

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
void first_view(inout Payload p)
{
    ORR_OMM(first_view);
    lean_first_view<Surface>(root, DispatchRaysIndex().x);
}
