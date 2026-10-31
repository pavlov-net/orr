// The HZB: mip 0 holds the visibility buffer's reverse-Z depth (written by the resolve's
// classify); each further mip holds the farthest depth (the smallest value) of the texels
// it covers, three wide where the finer mip has an odd size.
#include "orr/root.h"
import visibility;

ORR_ROOT(RasterRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void reduce(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    uint mip = root.value;
    uint2 fine = max(r.gbuffer.extent >> (mip - 1), 1);
    uint2 size = max(r.gbuffer.extent >> mip, 1);
    if (any(id.xy >= size))
        return;
    RWTexture2D<float> src = r.hzb[mip - 1].get<RWTexture2D<float>>();
    uint2 base = id.xy * 2;
    uint2 last = min(base + 1 + (fine & 1) * uint2(id.xy == size - 1), fine - 1);
    float d = 1.0;
    for (uint y = base.y; y <= last.y; y++)
        for (uint x = base.x; x <= last.x; x++)
            d = min(d, src[uint2(x, y)]);
    r.hzb[mip].get<RWTexture2D<float>>()[id.xy] = d;
}
