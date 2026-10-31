// A heightfield tile as flat-grid patches: one thread per patch allocates its 9 x 9
// vertices, places them on the height raster with their normals and uv, and emits the
// patch as the grid topology. The patch index is the cluster's user word.
#include "orr/root.h"
import field;
import producer;
import tables;

struct Root
{
    SceneRoot* scene;
    float* heights;      // the height raster: side x side samples, spacing m apart from (origin_x, origin_z)
    uint side;
    float spacing;
    float origin_x;
    float origin_z;
    uint producer;
    uint patches;        // per side
    float patch_size;    // m
    float corner_x;      // the tile's first corner, entry space
    float corner_z;
    uint _pad0;
    uint64_t _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint p = id.x;
    if (p >= root.patches * root.patches)
        return;
    ProducerBatch b = root.scene.batches[root.producer];
    Heights heights = { root.heights, root.side, root.spacing, root.origin_x, root.origin_z };
    uint first = alloc_vertices(b, PATCH_SIDE * PATCH_SIDE);
    if (first == NO_ROOM)
        return;
    float step = root.patch_size / float(PATCH_QUADS);
    float2 corner = float2(root.corner_x, root.corner_z) + float2(p % root.patches, p / root.patches) * root.patch_size;
    for (uint v = 0; v < PATCH_SIDE * PATCH_SIDE; v++)
    {
        float2 xz = corner + float2(v % PATCH_SIDE, v / PATCH_SIDE) * step;
        b.positions[first + v] = float3(xz.x, height_at(heights, xz.x, xz.y), xz.y);
        b.attrs[first + v] = producer_attributes(normal_at(heights, xz.x, xz.y), xz * 0.25);
    }
    emit_template_cluster(b, 0, first, 0, p);
}
