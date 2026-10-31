// Procedural grass on the height raster: one thread per cell of the grass grid within
// `radius` of the camera emits one cluster of eight blades, tall blades (five vertices,
// topology 0) inside half the radius and short ones (three vertices, topology 1) past it,
// so the cluster count follows the camera every frame. Blades sway with the wind; last
// frame's positions go to the batch's previous buffer. Cluster word: the cell's hash, which
// picks a clump's color; vertex word: height along the blade, 0 at the root and 65535 at
// the tip.
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
    uint cells;          // per side
    float cell_size;     // m
    float corner_x;      // the grass grid's first corner, entry space
    float corner_z;
    float camera_x;      // entry space
    float camera_z;
    float radius;        // m
    float time;          // s
    float previous_time;
    uint64_t _pad0;
}
ORR_ROOT(Root);

static const uint BLADES = 8;

// Vertex `k` of a blade of `count` vertices rooted at `base`, `height` tall, turned by
// `yaw`, at wind time `t`; `along` is its height fraction.
float3 blade_vertex(float3 base, float height, float yaw, uint k, uint count, float t, out float along)
{
    float width = 0.025;
    float2 tall[5] = { float2(-1.0, 0.0), float2(1.0, 0.0), float2(-0.6, 0.5), float2(0.6, 0.5), float2(0.0, 1.0) };
    float2 low[3] = { float2(-1.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0) };
    float2 q = count == 5 ? tall[k] : low[k];
    along = q.y;
    float3 side = float3(cos(yaw), 0.0, sin(yaw));
    float gust = sin(t * 1.7 + base.x * 0.35 + base.z * 0.21) * 0.6 + sin(t * 3.1 + base.x * 0.9) * 0.2;
    float bend = 0.35 * height * q.y * q.y;
    return base + side * (q.x * width * (1.0 - 0.7 * q.y)) + float3(bend * gust, height * q.y, bend * 0.4 * gust);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= root.cells * root.cells)
        return;
    float2 cell = float2(root.corner_x, root.corner_z) + (float2(c % root.cells, c / root.cells) + 0.5) * root.cell_size;
    float distance = length(cell - float2(root.camera_x, root.camera_z));
    if (distance > root.radius)
        return;
    uint count = distance < 0.5 * root.radius ? 5 : 3;
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, BLADES * count);
    if (first == NO_ROOM)
        return;
    Heights heights = { root.heights, root.side, root.spacing, root.origin_x, root.origin_z };
    uint h = field_hash(c);
    for (uint blade = 0; blade < BLADES; blade++)
    {
        uint s = field_hash(h + blade * 0x9e3779b9u);
        float2 xz = cell + (float2(field_unit(s), field_unit(s + 1)) - 0.5) * root.cell_size;
        float3 base = float3(xz.x, height_at(heights, xz.x, xz.y), xz.y);
        float height = 0.25 + 0.35 * field_unit(s + 2);
        float yaw = 6.2831853 * field_unit(s + 3);
        float3 n = normalize(float3(-sin(yaw), 0.6, cos(yaw)));
        for (uint k = 0; k < count; k++)
        {
            uint v = first + blade * count + k;
            float along;
            b.positions[v] = blade_vertex(base, height, yaw, k, count, root.time, along);
            if (b.previous != nullptr)
                b.previous[v] = blade_vertex(base, height, yaw, k, count, root.previous_time, along);
            b.attrs[v] = producer_attributes(n, float2(0.5, along));
            if (b.vertex_user != nullptr)
                b.vertex_user[v] = uint(saturate(along) * 65535.0);
        }
    }
    emit_template_cluster(b, count == 5 ? 0 : 1, first, 0, h);
}
