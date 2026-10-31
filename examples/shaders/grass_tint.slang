// Colors a shaded view's grass from its producer words: where the visibility buffer shows a
// cluster of entry `entry`, the pixel's hit is found again with `primary_hit`, and its color
// is darkened toward the blade's root by the vertex words (height along the blade, 0 to
// 65535) and shifted per clump by the cluster word.
#include "orr/root.h"
import orr.abi;
import surface;
import visibility;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    StorageImageIndex target;   // RGBA16F display-linear, read and written
    uint entry;                 // the grass producer
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    GBufferGpu* g = root.gbuffer;
    if (any(id.xy >= g.extent))
        return;
    SceneRoot* s = root.scene;
    uint64_t w = g.visibility[id.y * g.extent.x + id.x];
    if (w == 0)
        return;
    VisibleCluster vc = g.clusters[visibility_cluster(w)];
    uint entry;
    if (!visible_producer(s, vc, entry) || entry != root.entry)
        return;
    SurfaceInput i = primary_hit(s, root.view, vc, visibility_triangle(w), float2(id.xy) + 0.5).input;
    float3 words = float3(i.vertex_user) / 65535.0;
    float2 b = i.barycentrics;
    float along = (1.0 - b.x - b.y) * words.x + b.x * words.y + b.y * words.z;
    float clump = float(i.cluster_user & 0xff) / 255.0;
    float3 hue = lerp(float3(0.75, 1.0, 0.55), float3(1.15, 0.95, 0.6), clump);
    RWTexture2D<float4> target = root.target.get<RWTexture2D<float4>>();
    float4 c = target[id.xy];
    target[id.xy] = float4(c.rgb * hue * lerp(0.35, 1.0, along), c.a);
}
