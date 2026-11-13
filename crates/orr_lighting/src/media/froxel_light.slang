// A view's froxel light pass beside the thin shadow tracer (`rays/shadow.slang`), each kernel a
// lane per froxel the `list` pass found or per flagged ray: `walk_main` finds the interiors around
// each froxel's sample point, `sample_main` draws its light and writes its shadow rays as jobs,
// `retrace_main` retraces the rays the tracer flagged with the full handling, and
// `accumulate_main` lights each froxel from its rays' results.
#include "orr_root.h"
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.froxels;
import orr_lighting.rays;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void walk_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(walk_main);
    froxel_walk_lane<Surface>(root, group, lane);
}

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void sample_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(sample_main);
    froxel_sample_lane<Surface>(root, group, lane);
}

[shader("compute")]
[numthreads(RAY_GROUP, 1, 1)]
void retrace_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(retrace_main);
    froxel_retrace_lane<Surface, Medium>(root, group, lane);
}

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void accumulate_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(accumulate_main);
    froxel_accumulate_lane<Medium>(root, group, lane);
}
