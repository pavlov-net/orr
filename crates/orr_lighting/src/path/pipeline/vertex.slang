// The pipeline path tracer's callables at a path's vertex (`path.launch`): its arrival and its
// shade per vertex class, its emission, and what the raygen runs for the entries a shade leaves:
// the caustic photons' gather, a photon's deposit and an update path's cache vertex; a walk's end;
// and a shadow ray's plane stack step, the weighing of the light a stack bent, and a coated
// emitter's view.
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

// Each takes the vertex's step in its data's first word.
[shader("callable")]
void emission(inout Payload p)
{
    ORR_OMM(emission);
    launch_emission(root, launch_slot(), p.words.x);
}

// The arrival of vertex class `K` (AT_*) takes the length of the ray that reached the vertex in
// its data's second word, and hands back its step, the vertex's probability of running next-event
// estimation and the light candidates it drew.
void arrive_of<let K : uint>(inout Payload p)
{
    p.words = launch_arrive<K, Surface, Medium>(root, launch_slot(), p.words.x, asfloat(p.words.y));
}

// The shade takes that probability and that count in its data's second and third words, and
// hands back its step.
void shade_of<let K : uint>(inout Payload p)
{
    p.words.x = launch_shade<K, Surface, Medium>(root, launch_slot(), p.words.x, asfloat(p.words.y), p.words.z);
}

[shader("callable")]
void arrive(inout Payload p)
{
    ORR_OMM(arrive);
    arrive_of<AT_SURFACE>(p);
}

[shader("callable")]
void shade(inout Payload p)
{
    ORR_OMM(shade);
    shade_of<AT_SURFACE>(p);
}

[shader("callable")]
void arrive_exit(inout Payload p)
{
    ORR_OMM(arrive_exit);
    arrive_of<AT_EXIT>(p);
}

[shader("callable")]
void shade_exit(inout Payload p)
{
    ORR_OMM(shade_exit);
    shade_of<AT_EXIT>(p);
}

[shader("callable")]
void arrive_leave(inout Payload p)
{
    ORR_OMM(arrive_leave);
    arrive_of<AT_LEAVE>(p);
}

[shader("callable")]
void shade_leave(inout Payload p)
{
    ORR_OMM(shade_leave);
    shade_of<AT_LEAVE>(p);
}

[shader("callable")]
void arrive_medium(inout Payload p)
{
    ORR_OMM(arrive_medium);
    arrive_of<AT_SCATTER>(p);
}

[shader("callable")]
void shade_medium(inout Payload p)
{
    ORR_OMM(shade_medium);
    shade_of<AT_SCATTER>(p);
}

[shader("callable")]
void arrive_layer(inout Payload p)
{
    ORR_OMM(arrive_layer);
    arrive_of<AT_LAYER>(p);
}

[shader("callable")]
void shade_layer(inout Payload p)
{
    ORR_OMM(shade_layer);
    shade_of<AT_LAYER>(p);
}

[shader("callable")]
void gather(inout Payload p)
{
    ORR_OMM(gather);
    launch_gather(root, launch_slot());
}

[shader("callable")]
void deposit(inout Payload p)
{
    ORR_OMM(deposit);
    launch_deposit(root, launch_slot());
}

[shader("callable")]
void cache_vertex(inout Payload p)
{
    ORR_OMM(cache_vertex);
    launch_cache_vertex(root, launch_slot());
}

[shader("callable")]
void bent(inout Payload p)
{
    ORR_OMM(bent);
    launch_bent(root, launch_slot(), p.words.x);
}

[shader("callable")]
void walk_end(inout Payload p)
{
    ORR_OMM(walk_end);
    p.words.x = launch_walk_end(root, launch_slot());
}

// Takes the phase in its data's second word, and hands back what follows.
[shader("callable")]
void stack_step(inout Payload p)
{
    ORR_OMM(stack_step);
    p.words.x = launch_stack_step<Medium>(root, launch_slot(), p.words.y);
}

// Hands back whether the raygen traces the stack's probe.
[shader("callable")]
void stack_miss(inout Payload p)
{
    ORR_OMM(stack_miss);
    p.words.x = launch_stack_miss<Medium>(root, launch_slot()) ? 1 : 0;
}

[shader("callable")]
void emitter_view(inout Payload p)
{
    ORR_OMM(emitter_view);
    launch_emitter_view<Surface>(root, launch_slot());
}
