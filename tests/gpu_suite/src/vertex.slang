#include "orr_root.h"

struct Root
{
    float4* rectangles;
    uint first_id;
    uint triangle_list;
}
ORR_ROOT(Root);

struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint id : TEXCOORD0;
}

[shader("vertex")]
Vertex vertex_main(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    const float2 triangles[] = {
        float2(0.0, 0.0), float2(0.25, 0.0), float2(0.0, 1.0),
        float2(0.75, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
    };
    float2 uv = root.triangle_list != 0
        ? triangles[vertex] : float2(vertex & 1, (vertex >> 1) & 1);
    float4 bounds = root.rectangles[instance];
    Vertex result;
    result.position = float4(lerp(bounds.xy, bounds.zw, uv), 0.0, 1.0);
    result.id = instance;
    return result;
}

[shader("pixel")]
uint pixel_main(Vertex vertex) : SV_Target
{
    return root.first_id + vertex.id;
}
