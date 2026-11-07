// The view paths of the listed tiles: one path per pixel from its first layer or the G-buffer's
// surface. The raygen entry reorders lanes at each coherence point; the compute entry runs the same
// body without reordering.
#include "orr_root.h"
import orr_shaders.coherence;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.path;

extern struct Surface : ISurface = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT_INLINE(PathRoot);

// A row of LAYER_TILE^2 rays per listed tile.
[shader("raygeneration")]
void raygen_main()
{
    ORR_OMM(raygen_main);
    uint2 id = DispatchRaysIndex().xy;
    uint2 pixel = layer_tile_pixel(root, id.y, uint2(id.x % LAYER_TILE, id.x / LAYER_TILE));
    if (all(pixel < root.gbuffer.extent))
        view_body<Reorder, Surface, Environment, Medium, true>(root, pixel);
}

// Two workgroups per listed tile, its upper and lower half.
[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 group: SV_GroupID, uint3 id: SV_GroupThreadID)
{
    ORR_OMM(compute_main);
    uint2 pixel;
    if (layer_group_pixel(root, group, id.xy, pixel))
        view_body<NoReorder, Surface, Environment, Medium, true>(root, pixel);
}
