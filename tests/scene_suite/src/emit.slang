// A producer: `count` quads, cluster c of edge `size x (1 + c % 3)` at x = 2c, material
// c % 3. Even clusters use topology 0, odd ones their own triangles.
#include "orr/root.h"
import producer;
import tables;

struct Root
{
    SceneRoot* scene;
    uint8_t* triangles;  // the quad's two triangles, for direct clusters
    uint producer;
    uint count;
    float size;
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= root.count)
        return;
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, 4);
    if (first == NO_ROOM)
        return;
    float s = root.size * float(1 + c % 3);
    float3 o = float3(2.0 * float(c), 0.0, 0.0);
    b.positions[first + 0] = o;
    b.positions[first + 1] = o + float3(s, 0.0, 0.0);
    b.positions[first + 2] = o + float3(s, 0.5 * s, s);
    b.positions[first + 3] = o + float3(0.0, 0.0, s);
    if (c % 2 == 0)
        emit_template_cluster(b, 0, first, c % 3, c);
    else
        emit_direct_cluster(b, 2, 4, first, root.triangles, c % 3, c);
}
