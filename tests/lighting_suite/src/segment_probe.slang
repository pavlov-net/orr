// `segment_inscatter` through the stock height fog on given segments: its march averaged
// over evenly spread jitters, and a fine march of the same segment.
#include "orr_root.h"
import orr_lighting.media;
import lighting_suite.media_types;
import orr_lighting.light_sampling;

struct MarchSegment
{
    float3 origin;
    float t;
    float3 dir;
    float _pad;
}

struct Root
{
    LightSetGpu* set;
    MarchSegment* segments;
    float4* out;            // per segment: the march's mean, then the fine march's
    uint count;
    uint steps;             // the march's
    uint jitters;
    uint fine_steps;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    MarchSegment s = root.segments[id.x];
    float3 through;
    float3 mean = float3(0.0);
    for (uint k = 0; k < root.jitters; k++)
        mean += segment_inscatter<StockFog>(root.set, s.origin, s.dir, s.t, root.steps, (float(k) + 0.5) / float(root.jitters), through);
    float3 fine = segment_inscatter<StockFog>(root.set, s.origin, s.dir, s.t, root.fine_steps, 0.5, through);
    root.out[2 * id.x] = float4(mean / float(root.jitters), 0.0);
    root.out[2 * id.x + 1] = float4(fine, 0.0);
}
