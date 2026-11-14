// The pipeline path tracer's callables that start and finish a path, resolve a photon's given hit
// and take what lies along a path ray (`path.launch`).
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
