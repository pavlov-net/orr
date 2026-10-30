// A G-buffer pass: the task stage drops hidden triangles and hands the rest to one mesh
// workgroup each; the pixel stage writes three colour targets beside reverse-Z depth.
#include "orr/root.h"

struct Triangle
{
    // Clip positions; z is the triangle's depth.
    float4 a;
    float4 b;
    float4 c;
    float2 motion;
    uint id;
    uint hidden;
}

struct Root
{
    Triangle* triangles;
    uint triangle_count;
    uint pad;
}
ORR_ROOT(Root);

[vk::constant_id(0)] const uint salt = 0;

static const uint GROUP = 32;

struct Payload
{
    uint triangles[GROUP];
}

groupshared Payload payload;
groupshared uint survivors;

[shader("amplification")]
[numthreads(32, 1, 1)]
void task_main(uint group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    if (thread == 0)
        survivors = 0;
    GroupMemoryBarrierWithGroupSync();
    uint i = group * GROUP + thread;
    if (i < root.triangle_count && root.triangles[i].hidden == 0)
    {
        uint slot;
        InterlockedAdd(survivors, 1, slot);
        payload.triangles[slot] = i;
    }
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(survivors, 1, 1, payload);
}

struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint id : ID;
    nointerpolation float2 motion : MOTION;
}

struct Primitive
{
    uint index : SV_PrimitiveID;
}

[shader("mesh")]
[numthreads(3, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    in payload Payload p,
    out indices uint3 triangles[1],
    out vertices Vertex vertices[3],
    out primitives Primitive primitives[1])
{
    uint i = p.triangles[group];
    Triangle t = root.triangles[i];
    SetMeshOutputCounts(3, 1);
    Vertex v;
    v.position = thread == 0 ? t.a : (thread == 1 ? t.b : t.c);
    v.id = t.id;
    v.motion = t.motion;
    vertices[thread] = v;
    if (thread == 0)
    {
        triangles[0] = uint3(0, 1, 2);
        Primitive prim;
        prim.index = i;
        primitives[0] = prim;
    }
}

struct GBuffer
{
    uint4 a : SV_Target0;
    uint2 b : SV_Target1;
    float2 motion : SV_Target2;
}

[shader("pixel")]
GBuffer pixel_main(Vertex v, uint primitive: SV_PrimitiveID)
{
    GBuffer g;
    g.a = uint4(v.id, uint(v.position.x), uint(v.position.y), salt);
    g.b = uint2(primitive, asuint(v.position.z));
    g.motion = v.motion;
    return g;
}
