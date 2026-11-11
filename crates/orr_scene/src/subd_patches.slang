// Writes a subdivision copy's base patches into its tessellated entry's patch batch, before
// the tessellator dices them.
//
// `pose` skins a skinned copy's cage points by this frame's and last frame's palettes;
// `stencils` applies the cage's stencils to them for its page points; `fill` evaluates each
// base vertex on the limit surface (its position on the patch every vertex at its cage
// point shares, so they agree bit for bit) and writes the patch clusters and counts.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_scene.producer;
import orr_scene.subd_cage;
import orr_scene.tables;

public struct SubdRoot
{
    public SceneRoot* scene;
    public SubdCopyGpu* copy;
    public uint entry;
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}
ORR_ROOT(SubdRoot);

// Whether every palette entry of the copy is set: an unset one hides the copy.
bool palette_shown(SubdCopyGpu* c)
{
    if (c.joint_count == 0)
        return true;
    GroupGpu* palette = root.scene.groups + c.palette_group;
    if (c.palette_group >= root.scene.group_capacity || palette.current == nullptr
        || c.palette_first + c.joint_count > palette.count)
        return false;
    return palette_set(palette, c.palette_first, c.joint_count);
}

// One thread per cage point.
[shader("compute")]
[numthreads(64, 1, 1)]
void pose(uint3 id: SV_DispatchThreadID)
{
    SubdCopyGpu* c = root.copy;
    SubdCageGpu* cage = c.cage;
    uint i = id.x;
    if (i >= cage.cage_point_count || !palette_shown(c))
        return;
    GroupGpu* palette = root.scene.groups + c.palette_group;
    float3 p = cage.cage_points[i];
    SubdInfluence s = decode_influence(cage.skin[i]);
    for (uint k = 0; k < 2; k++)
    {
        float3x4 m = palette_blend(palette, c.palette_first, c.joint_count, s.joints, s.weights, k == 1);
        c.posed[k * cage.cage_point_count + i] = transform_point(m, p);
    }
}

// One thread per page point.
[shader("compute")]
[numthreads(64, 1, 1)]
void stencils(uint3 id: SV_DispatchThreadID)
{
    SubdCopyGpu* c = root.copy;
    SubdCageGpu* cage = c.cage;
    uint i = id.x;
    if (i >= cage.point_count)
        return;
    float3 now = 0.0;
    float3 before = 0.0;
    uint n = cage.cage_point_count;
    for (uint t = cage.stencils[i]; t < cage.stencils[i + 1]; t++)
    {
        uint p = cage.stencil_points[t];
        float w = cage.stencil_weights[t];
        now += w * c.posed[p];
        before += w * c.posed[n + p];
    }
    c.points[i] = now;
    c.previous[i] = before;
}

// One thread per base vertex; the first threads also write the patch clusters.
[shader("compute")]
[numthreads(64, 1, 1)]
void fill(uint3 id: SV_DispatchThreadID)
{
    SubdCopyGpu* c = root.copy;
    SubdCageGpu* cage = c.cage;
    ProducerBatch b = root.scene.patches[root.entry];
    uint i = id.x;
    bool shown = palette_shown(c);
    if (i == 0)
    {
        *b.cluster_count = shown ? cage.cluster_count : 0;
        *b.vertex_count = shown ? cage.vertex_count : 0;
    }
    if (!shown)
        return;
    if (i < cage.cluster_count)
    {
        uint4 k = cage.clusters[i];
        ProducerClusterGpu out;
        out.triangles = cage.indices + 3 * k.z;
        out.first_vertex = k.x;
        out.user = 0;
        out.vertex_count = uint16_t(k.y);
        out.tri_count = uint8_t(k.w & 0xff);
        out.material = uint16_t(k.w >> 8);
        out.topology = uint16_t(DIRECT_CLUSTER);
        out._pad = 0;
        b.clusters[i] = out;
    }
    if (i >= cage.vertex_count)
        return;
    SubdVertexGpu v = cage.vertices[i];
    float3 du, dv;
    float3 x = subd_limit(cage, c.points, v.patch, v.st, du, dv);
    bool shared = v.source != v.patch || any(v.source_st != v.st);
    if (shared)
    {
        float3 su, sv;
        x = subd_limit(cage, c.points, v.source, v.source_st, su, sv);
    }
    float3 n = subd_normal(du, dv, float3(0.0, 1.0, 0.0));
    b.positions[i] = x;
    b.attrs[i] = producer_attributes(n, subd_limit_uv(cage, v.patch, v.st, 0));
    if (b.uv1 != nullptr)
        b.uv1[i] = subd_limit_uv(cage, v.patch, v.st, 1);
    if (b.previous != nullptr)
        b.previous[i] = subd_limit(cage, c.previous, v.source, v.source_st, du, dv);
}
