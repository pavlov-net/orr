// The lean path tracer's hit shader for layers of the dielectric class (`KIND_LAYER_PATH`): a
// surface whose BSDF transmits, with a coat, a fuzz or a thin film where it has one, and whose
// vertex crosses into or out of the interior its material bounds. `path.lean_paths` holds the
// other stages; a stage of this kernel folds the lobes a scene's mask rules out.
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
import orr_lighting.specialize;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;
export static const uint BSDF_CLASS = CLASS_TRANSMISSION | CLASS_THIN | CLASS_COAT | CLASS_FUZZ | CLASS_FILM;

ORR_ROOT_INLINE(PathRoot);

[shader("closesthit")]
void surface_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(surface_hit);
    TriangleHit h = { a };
    // A window's or the stack question's answer is the payload.
    bool question = is_question(p.words);
    lean_layer_hit<Surface, Medium, TriangleHit, SectionGiven>(root, DispatchRaysIndex().x, p.words, section_given(data.material), h);
    // Tells the raygen a shader took the hit.
    if (!question)
        p.words.x = 0;
}
