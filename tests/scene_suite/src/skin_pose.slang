// A test pose kernel, as a consumer's scene compute writes a skinned palette group: one
// thread per copy samples the clip `time + copy x phase` seconds in, chains the joints'
// local transforms into model space in the group's own entries, then turns each into its
// palette entry with the skeleton's inverse bind.
#include "orr/root.h"
import orr.math;
import orr_pack.skeleton;
import tables;

struct Root
{
    SceneRoot* scene;
    uint8_t* skeleton;
    uint8_t* clip_header;
    uint8_t* clip_samples;
    uint group;
    uint copies;
    float time;
    float phase;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void palette(uint3 id: SV_DispatchThreadID)
{
    uint copy = id.x;
    if (copy >= root.copies)
        return;
    Skeleton k;
    k.blob = root.skeleton;
    Clip c;
    c.header = root.clip_header;
    c.samples = root.clip_samples;
    uint joints = skeleton_joints(k);
    GroupGpu* g = root.scene.groups + root.group;
    float3x4* out = (float3x4*)g.current + copy * joints;
    float t = root.time + float(copy) * root.phase;
    for (uint j = 0; j < joints; j++)
    {
        float3x4 local = trs_affine(clip_sample(c, t, j));
        uint parent = skeleton_parent(k, j);
        out[j] = parent == NO_PARENT ? local : affine_mul(out[parent], local);
    }
    for (uint j = 0; j < joints; j++)
        out[j] = skeleton_palette(k, j, out[j]);
}
