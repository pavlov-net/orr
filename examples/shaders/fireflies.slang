// Emissive particles drifting over the ground: one thread per cluster places 16 particles,
// each a small octahedron of six vertices, on slow loops around fixed homes within `radius`
// of the swarm's center, and emits the cluster as the octahedra topology. Last frame's
// positions go to the batch's previous buffer.
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
    uint clusters;
    float center_x;      // entry space
    float center_z;
    float radius;        // m
    float size;          // m, center to tip
    float time;          // s
    float previous_time;
}
ORR_ROOT(Root);

static const uint PARTICLES = 16;

// Particle `i`'s center at time `t`.
float3 particle(Heights h, uint i, float t)
{
    uint s = field_hash(i * 0x9e3779b9u + 17u);
    float a = 6.2831853 * field_unit(s);
    float r = root.radius * sqrt(field_unit(s + 1));
    float2 home = float2(root.center_x, root.center_z) + r * float2(cos(a), sin(a));
    float rate = 0.3 + 0.4 * field_unit(s + 2);
    float phase = 6.2831853 * field_unit(s + 3);
    float2 xz = home + 1.5 * float2(sin(t * rate + phase), cos(t * rate * 0.8 + phase));
    float lift = 0.8 + 1.6 * field_unit(s + 4) + 0.3 * sin(t * 1.3 + phase);
    return float3(xz.x, height_at(h, xz.x, xz.y) + lift, xz.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.clusters)
        return;
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, PARTICLES * 6);
    if (first == NO_ROOM)
        return;
    Heights h = { root.heights, root.side, root.spacing, root.origin_x, root.origin_z };
    float3 tips[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };
    for (uint k = 0; k < PARTICLES; k++)
    {
        uint i = id.x * PARTICLES + k;
        float3 c = particle(h, i, root.time);
        float3 c0 = particle(h, i, root.previous_time);
        for (uint v = 0; v < 6; v++)
        {
            uint at = first + k * 6 + v;
            b.positions[at] = c + tips[v] * root.size;
            if (b.previous != nullptr)
                b.previous[at] = c0 + tips[v] * root.size;
            b.attrs[at] = producer_attributes(tips[v], float2(0.5));
        }
    }
    emit_template_cluster(b, 0, first, 0, id.x);
}
