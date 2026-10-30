// Decodes instance records through orr_pack.instance, one thread per record.
#include "orr/root.h"
import orr_pack.instance;

struct Root
{
    InstancePageHeader* header;
    InstanceRecord* records;
    float4* out; // three rows per record
    uint count;
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    float3x4 m = decode_instance(*root.header, root.records[id.x]);
    root.out[3 * id.x] = m[0];
    root.out[3 * id.x + 1] = m[1];
    root.out[3 * id.x + 2] = m[2];
}
