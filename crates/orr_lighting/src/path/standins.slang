// The pipeline path tracer's shaders by the bodies they run (`path.launch`,
// `orr_scene.candidates`), as compute stand-ins compiled for the registers and memory the driver
// reports of them (`orr_lighting::standins`). A lane's slot is its index; its hit, its payload or
// its callable's data come from memory, and what its body hands back goes there, so the body is
// the shader's. They are compiled for their statistics and never run.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.hits;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_scene.candidates;
import orr_scene.runs;
import orr_scene.hit;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

// Lane `lane`'s payload or callable data, and what its body hands back.
uint4* lane_words(uint lane)
{
    return (uint4*)root.stages.lists + 2 * lane;
}

// Lane `lane`'s hit.
StoredHit lane_hit(uint lane)
{
    StoredHit h = { (HitInputs*)root.stages.queues + lane };
    return h;
}

// Lane `lane`'s hit record data.
HitData lane_data(uint lane)
{
    uint4 w = lane_words(lane)[1];
    HitData d = { w.x, w.y, w.z, w.w };
    return d;
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void resolve(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(resolve);
    uint4* w = lane_words(id.x);
    HitData data = lane_data(id.x);
    w[0] = launch_resolve<Surface, StoredHit, SectionGiven>(root, id.x, w[0], data.class_id(), section_given(data.material), lane_hit(id.x));
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void emission(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(emission);
    uint4* w = lane_words(id.x);
    w[0].x = launch_emission(root, id.x, asfloat(w[0].z), w[0].w);
}

// What a vertex body's calls hand back, from lane `lane`'s words, which the calls' arguments go
// to: the callables are measured as stand-ins of their own.
struct LaneCalls : ICalls
{
    uint lane;

    uint lights(float3 normal, uint dimension)
    {
        uint4* w = lane_words(lane);
        w[1] = uint4(asuint(normal), dimension);
        return w[0].w;
    }

    float3 gather(inout uint dimension, float width, uint flags, uint word)
    {
        uint4* w = lane_words(lane);
        w[1] = uint4(dimension, asuint(width), flags, word);
        uint4 back = w[0];
        dimension = back.w;
        return asfloat(back.xyz);
    }

    uint deposit(uint dimension)
    {
        uint4* w = lane_words(lane);
        w[1].x = dimension;
        return w[0].w;
    }

    void cache_vertex(uint dimension)
    {
        lane_words(lane)[1].y = dimension;
    }
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void direct(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(direct);
    uint4* w = lane_words(id.x);
    LaneCalls calls = { id.x };
    w[0] = launch_direct<AT_SURFACE, Surface, Medium, LaneCalls>(root, id.x, w[0].y, asfloat(w[0].z), w[0].w, calls);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void scatter(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(scatter);
    uint4* w = lane_words(id.x);
    LaneCalls calls = { id.x };
    w[0].x = launch_scatter<AT_SURFACE, Surface, LaneCalls>(root, id.x, w[0].y != 0, asfloat(w[0].z), calls);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void lights(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(lights);
    uint4* w = lane_words(id.x);
    w[0].x = launch_lights<Surface>(root, id.x, asfloat(w[0].xyz), w[0].w);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void gather(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(gather);
    uint4* w = lane_words(id.x);
    uint4 data = w[0];
    uint dimension = data.x;
    float3 g = launch_gather<Surface>(root, id.x, dimension, asfloat(data.y), data.z, data.w);
    w[0] = uint4(asuint(g), dimension);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void cover_path_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(cover_path_hit);
    uint4* w = lane_words(id.x);
    HitData data = lane_data(id.x);
    uint4 payload = w[0];
    StoredHit h = lane_hit(id.x);
    w[1].x = path_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, payload, data, h, slot_crossings(root, id.x));
    w[0] = payload;
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void cover_shadow_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(cover_shadow_hit);
    uint4* w = lane_words(id.x);
    uint4 payload = w[0];
    HitData data = lane_data(id.x);
    w[1].x = shadow_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, payload, data, lane_hit(id.x));
    w[0] = payload;
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void box_medium_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(box_medium_hit);
    StoredHit h = lane_hit(id.x);
    lane_words(id.x)[0].xy = asuint(box_medium(h.object_origin(), h.object_direction(), h.t()));
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void proxy_shadow_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(proxy_shadow_hit);
    uint4* w = lane_words(id.x);
    uint4 payload = w[0];
    w[1].x = proxy_any_hit(root.scene, payload, lane_hit(id.x).instance_index(), asfloat(w[1].zw));
    w[0] = payload;
}
