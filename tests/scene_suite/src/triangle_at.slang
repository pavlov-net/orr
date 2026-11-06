// `triangle_geometry` at one point of a facing triangle with corners at (0, 0, 0), (edge, 0, 0)
// and (0, edge, 0), under `diag(-size, size, size)`.
#include "orr_root.h"
import orr_scene.hit;

struct Root
{
    SceneRoot* scene;
    float4* out;              // the vertex color, the geometric and vertex normals, the side's material and whether it bounds an interior, the tangents of the base and the coat
    float4 view;              // the ray's direction, then the section's material
    float4 color0;
    float4 color1;
    float4 color2;
    float4 point;             // barycentrics of corners 1 and 2, then the size and the edge
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    Corners t = (Corners)0;
    t.position[1] = float3(root.point.w, 0.0, 0.0);
    t.position[2] = float3(0.0, root.point.w, 0.0);
    t.attributes[0].color = root.color0;
    t.attributes[1].color = root.color1;
    t.attributes[2].color = root.color2;
    for (uint c = 0; c < 3; c++)
        t.attributes[c].normal = float3(0.0, 0.0, 1.0);
    // Uv set 1 is set 0 with its axes swapped.
    t.attributes[1].uv[0] = float2(1.0, 0.0);
    t.attributes[2].uv[0] = float2(0.0, 1.0);
    t.attributes[1].uv[1] = float2(0.0, 1.0);
    t.attributes[2].uv[1] = float2(1.0, 0.0);
    Placement pl = (Placement)0;
    float size = root.point.z;
    pl.linear = float3x3(-size, 0.0, 0.0, 0.0, size, 0.0, 0.0, 0.0, size);
    SurfaceInput i = (SurfaceInput)0;
    uint material = uint(root.view.w);
    triangle_geometry(i, t, 0, root.scene, material, pl, root.point.xy, float2(0.0), float2(0.0), root.view.xyz, 1.0);
    root.out[0] = i.vertex_color;
    root.out[1] = float4(i.geometric_normal, 0.0);
    root.out[2] = float4(i.normal, 0.0);
    uint side = side_material(root.scene, material, i);
    root.out[3] = float4(float(side), crosses_interior(root.scene, side) ? 1.0 : 0.0, 0.0, 0.0);
    root.out[4] = i.tangent;
    root.out[5] = i.coat_tangent;
}
