// A debug overlay: one row and one column at a fixed reverse-Z depth, drawn only where it is
// nearer than the hook's depth.
#include "orr/root.h"
import orr.abi;

struct Root
{
    TextureIndex depth;
    StorageImageIndex image;
    uint width;
    uint height;
    uint row;
    uint column;
    float line_depth;
    uint _pad;
}
ORR_ROOT(Root);

static const float4 LINE_COLOUR = float4(1.0, 0.0, 1.0, 1.0);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)) || (id.y != root.row && id.x != root.column))
        return;
    float depth = root.depth.get<Texture2D<float>>().Load(int3(id.xy, 0));
    if (root.line_depth >= depth)
        root.image.get<RWTexture2D<float4>>()[id.xy] = LINE_COLOUR;
}
