// The pipeline path tracer's callables at a path's vertex (`path.launch`): its emission, its
// next-event estimation and scatter per vertex class, and what those call: light sampling, the
// caustic photons' gather, a photon's deposit and an update path's cache vertex; a walk's end; and
// a shadow ray's plane stack step and coated emitter's view.
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

// The emission and the direct callable of vertex class `K` (AT_*) take DIRECT_* in their data's
// second word, the length of the ray that reached the vertex in its third and the step that
// reached it in its fourth; the direct callable hands back its step, with the vertex's probability
// of running next-event estimation in the last word.
void direct_of<let K : uint>(inout Payload p)
{
    p.words = launch_direct<K, Surface, Medium, CallableCalls>(root, launch_slot(), p.words.y, asfloat(p.words.z), p.words.w, CallableCalls());
}

// The scatter callable takes SCATTER_BENT in its data's second word, or 0, and that probability
// in its third.
void scatter_of<let K : uint>(inout Payload p)
{
    p.words.x = launch_scatter<K, Surface, CallableCalls>(root, launch_slot(), p.words.y == SCATTER_BENT, asfloat(p.words.z), CallableCalls());
}

[shader("callable")]
void direct(inout Payload p)
{
    ORR_OMM(direct);
    direct_of<AT_SURFACE>(p);
}

[shader("callable")]
void scatter(inout Payload p)
{
    ORR_OMM(scatter);
    scatter_of<AT_SURFACE>(p);
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
    scatter_of<AT_EXIT>(p);
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
    scatter_of<AT_LEAVE>(p);
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
    scatter_of<AT_SCATTER>(p);
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
    scatter_of<AT_LAYER>(p);
}

// Takes the vertex's shading normal in its data's first three words and the sampler dimension it
// draws from in its last, and hands back how many candidates it drew.
[shader("callable")]
void lights(inout Payload p)
{
    ORR_OMM(lights);
    p.words.x = launch_lights<Surface>(root, launch_slot(), asfloat(p.words.xyz), p.words.w);
}

// Takes the sampler dimension, the footprint, the vertex's flags and word (`ICalls.gather`), and
// hands back what it gathers and the dimension past its draws.
[shader("callable")]
void gather(inout Payload p)
{
    ORR_OMM(gather);
    uint dimension = p.words.x;
    float3 g = launch_gather<Surface>(root, launch_slot(), dimension, asfloat(p.words.y), p.words.z, p.words.w);
    p.words = uint4(asuint(g), dimension);
}

[shader("callable")]
void deposit(inout Payload p)
{
    ORR_OMM(deposit);
    p.words.x = launch_deposit<Surface>(root, launch_slot(), p.words.x);
}

[shader("callable")]
void cache_vertex(inout Payload p)
{
    ORR_OMM(cache_vertex);
    launch_cache_vertex(root, launch_slot(), p.words.x);
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

[shader("callable")]
void emitter_view(inout Payload p)
{
    ORR_OMM(emitter_view);
    launch_emitter_view<Surface>(root, launch_slot());
}
