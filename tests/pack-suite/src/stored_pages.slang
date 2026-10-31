// Decodes a batch of stored pages through orr_pack.page_codec: `copy` over every page's
// resident dwords, then `decode` with one workgroup per cluster. Group y is the page; each
// page's status collects both passes' bits.
#include "orr/root.h"
import orr_pack.page_codec;

struct Root
{
    StoredPage* pages;
    uint* status;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(256, 1, 1)]
void copy(uint3 id: SV_DispatchThreadID)
{
    uint s = page_copy(root.pages[id.y], id.x);
    if (s != 0)
        InterlockedOr(root.status[id.y], s);
}

[shader("compute")]
[numthreads(PAGE_CODEC_THREADS, 1, 1)]
void decode(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint s = page_decode_cluster(root.pages[group.y], group.x, thread.x);
    if (s != 0 && thread.x == 0)
        InterlockedOr(root.status[group.y], s);
}
