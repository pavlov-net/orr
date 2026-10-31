// A displaced-grid dielectric slab: one thread per grid patch places its 9 x 9 vertices on
// a plane at `level` displaced by travelling waves, with their normals, and emits the patch
// as the grid topology.
#include "orr/root.h"
import field;
import producer;
import tables;

struct Root
{
    SceneRoot* scene;
    uint producer;
    uint patches;        // per side
    float patch_size;    // m
    float corner_x;      // the slab's first corner, entry space
    float corner_z;
    float level;         // m
    float amplitude;     // m
    float time;          // s
    uint64_t _pad0;
}
ORR_ROOT(Root);

float wave(float2 xz, float t)
{
    return root.amplitude * (sin(xz.x * 1.3 + t * 1.1) * 0.6 + sin(xz.y * 1.9 - t * 0.8) * 0.4);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint p = id.x;
    if (p >= root.patches * root.patches)
        return;
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, PATCH_SIDE * PATCH_SIDE);
    if (first == NO_ROOM)
        return;
    float step = root.patch_size / float(PATCH_QUADS);
    float2 corner = float2(root.corner_x, root.corner_z) + float2(p % root.patches, p / root.patches) * root.patch_size;
    for (uint v = 0; v < PATCH_SIDE * PATCH_SIDE; v++)
    {
        float2 xz = corner + float2(v % PATCH_SIDE, v / PATCH_SIDE) * step;
        float e = 0.05;
        float dx = wave(xz + float2(e, 0.0), root.time) - wave(xz - float2(e, 0.0), root.time);
        float dz = wave(xz + float2(0.0, e), root.time) - wave(xz - float2(0.0, e), root.time);
        b.positions[first + v] = float3(xz.x, root.level + wave(xz, root.time), xz.y);
        b.attrs[first + v] = producer_attributes(normalize(float3(-dx, 2.0 * e, -dz)), xz * 0.1);
    }
    emit_template_cluster(b, 0, first, 0, p);
}
