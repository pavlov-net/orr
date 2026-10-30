// (a) Acceleration structure read from the resource heap through a Slang DescriptorHandle.
#include "common.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    RaytracingAccelerationStructure scene =
        getDescriptorFromHandle(DescriptorHandle<RaytracingAccelerationStructure>(uint2(root.heap_index, 0)));
    store(id.xy, trace(scene, id.xy));
}
