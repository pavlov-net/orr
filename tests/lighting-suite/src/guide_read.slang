// Copies a view's six guide targets into one buffer, six RGBA values per pixel.
#include "orr/root.h"
import orr.abi;

struct Root
{
    TextureIndex diffuse_albedo;
    TextureIndex specular_albedo;
    TextureIndex normal_roughness;
    TextureIndex specular_motion;
    TextureIndex depth;
    TextureIndex motion;
    float4* out;
    uint width;
    uint count;
    uint64_t _pad0;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    int3 p = int3(id.x % root.width, id.x / root.width, 0);
    float4* o = root.out + id.x * 6;
    o[0] = root.diffuse_albedo.get<Texture2D<float4>>().Load(p);
    o[1] = root.specular_albedo.get<Texture2D<float4>>().Load(p);
    o[2] = root.normal_roughness.get<Texture2D<float4>>().Load(p);
    o[3] = float4(root.specular_motion.get<Texture2D<float2>>().Load(p), 0.0, 0.0);
    o[4] = float4(root.depth.get<Texture2D<float>>().Load(p), 0.0, 0.0, 0.0);
    o[5] = float4(root.motion.get<Texture2D<float2>>().Load(p), 0.0, 0.0);
}
