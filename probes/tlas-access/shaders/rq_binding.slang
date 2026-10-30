// (c) Plain set/binding acceleration structure. The host decides where it comes from: a heap
// mapping (constant offset, push index), an address mapping (push address, indirect address),
// or a classic push descriptor in a non-heap pipeline.
#include "common.slang"

[[vk::binding(1, 0)]] RaytracingAccelerationStructure scene;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    store(id.xy, trace(scene, id.xy));
}
