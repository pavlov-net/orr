// (b) TLAS device address loaded through a GPU pointer (root.tlas points at a uint64 holding
// the address), then converted in-shader. This is the scene-table form: the root names a GPU
// struct and the TLAS address lives there.
#include "common.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint64_t* table = (uint64_t*)root.tlas;
    store(id.xy, trace(RaytracingAccelerationStructure(table[0]), id.xy));
}
