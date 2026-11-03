// The stock particle draw for transparency hooks: one mesh workgroup per visible particle
// cluster, and per covered pixel its surface's base color as albedo, its emission and its
// opacity, lit and fogged by `particle_shade` and blended premultiplied into the overlay.
#include "orr/root.h"
import media;
import surface;
import particles;
import particle_lighting;

extern struct Surface : ISurface = StandardSurface;
extern struct Medium : IMedium = NoMedium;

public struct ParticleRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public RasterGpu* raster;
    public ParticleLighting* lighting;
    public float anisotropy;
    public uint _pad0;
    public uint64_t _pad1;
}

ORR_ROOT(ParticleRoot);

struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint visible : VISIBLE;
}

struct Primitive
{
    uint triangle : SV_PrimitiveID;
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void particle_mesh(
    uint3 group: SV_GroupID,
    uint3 lane: SV_GroupThreadID,
    out indices uint3 triangles[128],
    out vertices Vertex vertices[256],
    out primitives Primitive primitives[128])
{
    ORR_OMM(particle_mesh);
    uint thread = lane.x;
    ParticleDraw d;
    if (!particle_draw(root.scene, root.view, root.raster, group, d))
    {
        SetMeshOutputCounts(0, 0);
        return;
    }
    SetMeshOutputCounts(uint(d.cluster.vertex_count), uint(d.cluster.tri_count));
    for (uint i = thread; i < uint(d.cluster.vertex_count); i += 128)
    {
        Vertex o;
        o.position = particle_vertex(root.view, d, i);
        o.visible = d.visible;
        vertices[i] = o;
    }
    if (thread < uint(d.cluster.tri_count))
    {
        triangles[thread] = particle_triangle(d, thread);
        Primitive p;
        p.triangle = thread;
        primitives[thread] = p;
    }
}

[shader("pixel")]
[earlydepthstencil]
float4 particle_pixel(Vertex v, int primitive: SV_PrimitiveID) : SV_Target0
{
    ORR_OMM(particle_pixel);
    return particle_fragment<Surface, Medium>(root.lighting, root.raster, v.visible, uint(primitive), v.position.xy, root.anisotropy);
}
