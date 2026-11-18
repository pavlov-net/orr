// The path tracer's hit shader for layers of the dielectric class (`KIND_LAYER_PATH`): a
// surface whose BSDF transmits, with a coat, a fuzz or a thin film where it has one, and whose
// vertex crosses into or out of the interior its material bounds; the hit shader of a manifold
// walk's face ray on such a surface (`KIND_LAYER_SHADOW`), which a face ray's hit object runs in
// place of the path record's; the callable that weighs a dispersive layer's companion
// wavelengths; the callable of the vertex where an interior's real-time walk meets the layer's
// surface; and the callable of a manifold walk's step at a surface, which evaluates the surface's
// BSDF toward the light. `path.plain` holds the other
// stages; a stage of this kernel folds the lobes a scene's mask rules out.
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
import orr_lighting.path.vertex;
import orr_lighting.path.manifold;
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
    // A window's answer is the payload.
    bool question = is_question(p.words);
    path_layer_hit<Surface, Medium, TriangleHit, SectionGiven>(root, launch_index().x, p.words, section_given(data.material), h);
    // Tells the raygen a shader took the hit.
    if (!question)
        p.words.x = 0;
}

[shader("closesthit")]
void face_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    ORR_OMM(face_hit);
    TriangleHit h = { a };
    path_face_hit<Surface, TriangleHit, SectionGiven>(root, launch_index().x, p.words, section_given(data.material), h);
}

[shader("callable")]
void disperse(inout DisperseCall c)
{
    ORR_OMM(disperse);
    path_disperse(root, launch_index().x, c);
}

[shader("callable")]
void walk_meets(inout WalkMeetCall c)
{
    ORR_OMM(walk_meets);
    path_walk_meets<Medium>(root, launch_index().x, c);
}

[shader("callable")]
void manifold_face(inout FaceCall c)
{
    ORR_OMM(manifold_face);
    path_manifold_face<Medium>(root, launch_index().x, c);
}
