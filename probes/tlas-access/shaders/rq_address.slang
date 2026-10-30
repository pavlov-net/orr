// (b) TLAS device address from the root, converted in-shader (OpConvertUToAccelerationStructureKHR).
#include "common.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    store(id.xy, trace(RaytracingAccelerationStructure(root.tlas), id.xy));
}
