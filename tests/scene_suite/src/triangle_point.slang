// `triangle_point` of one ray and one triangle.
#include "orr_root.h"
import orr_scene.visibility;

struct Root
{
    float4* out;              // the weights of corners 1 and 2 and the distance, the point, then the plane's weights and distance
    uint pad0;
    uint pad1;
    float4 direction;         // the ray's, unit
    float4 corner;            // corner 0, from the eye
    float4 edge1;
    float4 edge2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    float3 at;
    float3 plane;
    float3 b = triangle_point(root.direction.xyz, root.corner.xyz, root.edge1.xyz, root.edge2.xyz, at, plane);
    root.out[0] = float4(b, 0.0);
    root.out[1] = float4(at, 0.0);
    root.out[2] = float4(plane, 0.0);
}
