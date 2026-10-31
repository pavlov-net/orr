// Poses a crowd of skinned copies: one thread per copy samples its clip `time + phase` seconds
// in, looping, with the copy's own phase, chains the joints' local transforms into model space
// in the GPU-written palette group's own entries, then turns each into its palette entry
// (skinning matrix) with the skeleton's inverse bind.
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
    float time;          // s
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
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
    float3x4* out = (float3x4*)root.scene.groups[root.group].current + copy * joints;
    float phase = float((copy * 2654435761u) >> 16) / 65536.0;
    float t = fmod(root.time + phase * clip_duration(c), clip_duration(c));
    for (uint j = 0; j < joints; j++)
    {
        float3x4 local = trs_affine(clip_sample(c, t, j));
        uint parent = skeleton_parent(k, j);
        out[j] = parent == NO_PARENT ? local : affine_mul(out[parent], local);
    }
    for (uint j = 0; j < joints; j++)
        out[j] = skeleton_palette(k, j, out[j]);
}
