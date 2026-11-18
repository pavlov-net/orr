// The path pipeline's shaders by the bodies they run (`path.vertex`,
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
import orr_lighting.path.vertex;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;
export static const uint BSDF_CLASS = 0;

ORR_ROOT_INLINE(PathRoot);

// Lanes per workgroup.
static const uint LANES = 32;

// Lane `lane`'s payload or callable data, and what its body hands back, in the launch's hot words:
// a stand-in never runs, so any memory serves.
uint4* lane_words(uint lane)
{
    return root.launch.hot + 2 * lane;
}

// Lane `lane`'s hit, in the launch's records.
StoredHit lane_hit(uint lane)
{
    StoredHit h = { (HitInputs*)root.launch.records + lane };
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
[numthreads(LANES, 1, 1)]
void vertex_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(vertex_hit);
    uint4* w = lane_words(id.x);
    HitData data = lane_data(id.x);
    uint4 words = w[0];
    path_hit<Surface, Medium, StoredHit, SectionGiven>(root, id.x, words, section_given(data.material), lane_hit(id.x));
}

[shader("compute")]
[numthreads(LANES, 1, 1)]
void first_vertex(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(first_vertex);
    uint4 none = uint4(0);
    path_first<Medium>(root, id.x, none);
}

[shader("compute")]
[numthreads(LANES, 1, 1)]
void cover_path_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(cover_path_hit);
    uint4* w = lane_words(id.x);
    HitData data = lane_data(id.x);
    uint4 payload = w[0];
    StoredHit h = lane_hit(id.x);
    w[1].x = path_any_hit<CoverageFeedback<Surface>>(root.scene, root.view, payload, data, h);
}

[shader("compute")]
[numthreads(LANES, 1, 1)]
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
[numthreads(LANES, 1, 1)]
void box_medium_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(box_medium_hit);
    StoredHit h = lane_hit(id.x);
    lane_words(id.x)[0].xy = asuint(box_medium(h.object_origin(), h.object_direction(), h.t()));
}

[shader("compute")]
[numthreads(LANES, 1, 1)]
void proxy_shadow_hit(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(proxy_shadow_hit);
    uint4* w = lane_words(id.x);
    uint4 payload = w[0];
    w[1].x = proxy_any_hit(root.scene, payload, lane_hit(id.x).instance_index(), asfloat(w[1].zw));
    w[0] = payload;
}
