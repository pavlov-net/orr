// The pipeline path tracer's callables at a path's vertex (`path.launch`): its emission, and its
// next-event estimation and scatter per vertex class, which `orr_lighting::launch` makes per BSDF
// class by `path_may`; a photon's deposit, an update path's cache vertex and a walk's end; and a
// shadow ray's plane stack step and coated emitter's view. The emission, direct and deposit
// callables take the length of the ray to the vertex in their data's third word and the step that
// reached it in the fourth.
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
void emission(inout Payload p)
{
    ORR_OMM(emission);
    p.words.x = launch_emission(root, launch_slot(), asfloat(p.words.z), p.words.w);
}

// The direct callable of vertex class `K` (AT_*), and the deposit, take DIRECT_* in their data's
// second word. A direct callable hands back there the sampler dimension of its record's vertex,
// which `cache_vertex` takes.
void direct_of<let K : uint>(inout Payload p)
{
    uint dimension;
    p.words.x = launch_direct<K, Surface, Medium>(root, launch_slot(), p.words.y, asfloat(p.words.z), p.words.w, dimension);
    p.words.y = dimension;
}

[shader("callable")]
void direct(inout Payload p)
{
    ORR_OMM(direct);
    direct_of<AT_PLAIN>(p);
}

[shader("callable")]
void deposit(inout Payload p)
{
    ORR_OMM(deposit);
    p.words.x = launch_deposit<Surface>(root, launch_slot(), p.words.y == DIRECT_PROBED, asfloat(p.words.z), p.words.w);
}

[shader("callable")]
void cache_vertex(inout Payload p)
{
    ORR_OMM(cache_vertex);
    launch_cache_vertex(root, launch_slot(), p.words.y);
}

[shader("callable")]
void scatter(inout Payload p)
{
    ORR_OMM(scatter);
    p.words.x = launch_scatter<AT_PLAIN, Surface>(root, launch_slot());
}

[shader("callable")]
void walk_end(inout Payload p)
{
    ORR_OMM(walk_end);
    p.words.x = launch_walk_end(root, launch_slot());
}

[shader("callable")]
void direct_exit(inout Payload p)
{
    ORR_OMM(direct_exit);
    direct_of<AT_EXIT>(p);
}

[shader("callable")]
void scatter_exit(inout Payload p)
{
    ORR_OMM(scatter_exit);
    p.words.x = launch_scatter<AT_EXIT, Surface>(root, launch_slot());
}

[shader("callable")]
void direct_leave(inout Payload p)
{
    ORR_OMM(direct_leave);
    direct_of<AT_LEAVE>(p);
}

[shader("callable")]
void scatter_leave(inout Payload p)
{
    ORR_OMM(scatter_leave);
    p.words.x = launch_scatter<AT_LEAVE, Surface>(root, launch_slot());
}

[shader("callable")]
void direct_layer(inout Payload p)
{
    ORR_OMM(direct_layer);
    direct_of<AT_LAYER>(p);
}

[shader("callable")]
void scatter_layer(inout Payload p)
{
    ORR_OMM(scatter_layer);
    p.words.x = launch_scatter<AT_LAYER, Surface>(root, launch_slot());
}

[shader("callable")]
void direct_medium(inout Payload p)
{
    ORR_OMM(direct_medium);
    direct_of<AT_SCATTER>(p);
}

[shader("callable")]
void scatter_medium(inout Payload p)
{
    ORR_OMM(scatter_medium);
    p.words.x = launch_scatter<AT_SCATTER, Surface>(root, launch_slot());
}

// Takes the phase in its data's second word, and hands back what follows.
[shader("callable")]
void stack_step(inout Payload p)
{
    ORR_OMM(stack_step);
    p.words.x = launch_stack_step<Medium>(root, launch_slot(), p.words.y);
}

[shader("callable")]
void emitter_view(inout Payload p)
{
    ORR_OMM(emitter_view);
    launch_emitter_view<Surface>(root, launch_slot());
}
