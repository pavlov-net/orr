// A screen warp: rotates the source about the image centre by an angle that falls to zero
// at `radius`, sampled bilinearly with clamped edges.
#include "orr/root.h"
import orr.abi;

struct Root
{
    TextureIndex source;
    StorageImageIndex image;
    uint width;
    uint height;
    float strength;
    float radius;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

float4 texel(int2 p)
{
    int2 q = clamp(p, int2(0), int2(root.width, root.height) - 1);
    return root.source.get<Texture2D<float4>>().Load(int3(q, 0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float2 c = 0.5 * float2(root.width, root.height);
    float2 q = float2(id.xy) + 0.5 - c;
    float t = saturate(1.0 - length(q) / root.radius);
    float a = root.strength * t * t;
    float s = sin(a);
    float co = cos(a);
    float2 at = c + float2(co * q.x - s * q.y, s * q.x + co * q.y) - 0.5;
    float2 i = floor(at);
    float2 f = at - i;
    int2 p = int2(i);
    float4 top = lerp(texel(p), texel(p + int2(1, 0)), f.x);
    float4 bottom = lerp(texel(p + int2(0, 1)), texel(p + int2(1, 1)), f.x);
    root.image.get<RWTexture2D<float4>>()[id.xy] = lerp(top, bottom, f.y);
}
