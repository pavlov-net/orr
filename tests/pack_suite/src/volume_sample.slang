// Reads a resident volume through orr_pack.volume, one thread per point. `main` gives the
// trilinear sample and majorant at the point and the voxel at the matching index; `spans`
// walks the majorant cells along a ray from the point and gives the majorant that covers
// each of SPAN_SAMPLES evenly spaced points of it.
#include "orr_shaders/root.h"
import orr_pack.volume;

static const uint SPAN_SAMPLES = 64;

struct Root
{
    VolumeGpu* volume;
    float4* points;   // xyz in the volume's space
    int4* voxels;     // xyz
    float4* out;      // sample, majorant, voxel
    float4* dirs;     // xyz: unit direction, w: length of the ray
    float* spans;     // SPAN_SAMPLES per point
    uint count;
    uint grid;
    uint64_t _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    float3 p = root.points[id.x].xyz;
    float s = volume_sample(root.volume, root.grid, p);
    float m = volume_majorant(root.volume, root.grid, p);
    float v = volume_voxel(root.volume, root.grid, root.voxels[id.x].xyz);
    root.out[id.x] = float4(s, m, v, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void spans(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    VolumeGpu* v = root.volume;
    float4 dir = root.dirs[id.x];
    float3 u0 = (root.points[id.x].xyz - v.origin) * v.inv_voxel_size;
    float3 du = dir.xyz * v.inv_voxel_size;
    float t = 0.0;
    float t_exit;
    float m = volume_majorant_span(v, root.grid, u0, du, t, t_exit);
    uint walked = 0;
    for (uint s = 0; s < SPAN_SAMPLES; s++)
    {
        float at = (float(s) + 0.5) / float(SPAN_SAMPLES) * dir.w;
        while (at >= t_exit && walked < 4096)
        {
            t = max(t_exit, t + 1e-6);
            m = volume_majorant_span(v, root.grid, u0, du, t, t_exit);
            walked++;
        }
        root.spans[id.x * SPAN_SAMPLES + s] = m;
    }
}
