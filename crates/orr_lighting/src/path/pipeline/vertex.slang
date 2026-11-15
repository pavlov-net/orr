// The pipeline path tracer's callables at a path's vertex (`path.launch`): the whole vertex per
// class, for one no hit shader ran; a first vertex's emission where its pixel runs no path; what
// the raygen runs for the entries a vertex leaves: the caustic photons' gather, a photon's
// deposit and an update path's cache vertex; an interior walk's end; and a shadow ray's plane
// stack step, the weighing of the light a stack bent, and a coated emitter's view.
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

// The vertex of class `K` (AT_*) whole, for one no hit shader ran: it takes the length of the ray
// that reached the vertex in its data's second word, and hands back its shade's step.
[shader("callable")]
void vertex(inout Payload p)
{
    ORR_OMM(vertex);
    p.words.x = launch_vertex<AT_SURFACE, Surface, Medium>(root, launch_slot(), p.words.x, asfloat(p.words.y));
}

[shader("callable")]
void vertex_exit(inout Payload p)
{
    ORR_OMM(vertex_exit);
    p.words.x = launch_vertex<AT_EXIT, Surface, Medium>(root, launch_slot(), p.words.x, asfloat(p.words.y));
}

[shader("callable")]
void vertex_leave(inout Payload p)
{
    ORR_OMM(vertex_leave);
    p.words.x = launch_vertex<AT_LEAVE, Surface, Medium>(root, launch_slot(), p.words.x, asfloat(p.words.y));
}

[shader("callable")]
void vertex_medium(inout Payload p)
{
    ORR_OMM(vertex_medium);
    p.words.x = launch_vertex<AT_SCATTER, Surface, Medium>(root, launch_slot(), p.words.x, asfloat(p.words.y));
}

[shader("callable")]
void vertex_layer(inout Payload p)
{
    ORR_OMM(vertex_layer);
    p.words.x = launch_vertex<AT_LAYER, Surface, Medium>(root, launch_slot(), p.words.x, asfloat(p.words.y));
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
