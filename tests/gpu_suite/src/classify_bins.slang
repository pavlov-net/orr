// Writes a device-generated sequence per non-empty bin, picking the bin's pipeline and root,
// and the indirect dispatch arguments of every bin for the CPU-recorded form.
#include "orr/root.h"
import sequence;

struct Root
{
    uint* items;   // items per bin
    uint* count;   // sequences written
    uint* args;    // three words per bin
    uint64_t sequences;
    uint64_t roots; // per-bin roots, ROOT_STRIDE bytes apart
    uint bins;
    uint pad0;
}
ORR_ROOT(Root);

static const uint ROOT_STRIDE = 64;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint b = id.x;
    if (b >= root.bins)
        return;
    uint groups = (root.items[b] + 63) / 64;
    root.args[b * 3] = groups;
    root.args[b * 3 + 1] = 1;
    root.args[b * 3 + 2] = 1;
    if (groups == 0)
        return;
    uint slot;
    InterlockedAdd(*root.count, 1, slot);
    Sequence s;
    s.pipeline = b;
    s.root = root.roots + uint64_t(b) * ROOT_STRIDE;
    s.groups = uint3(groups, 1, 1);
    store(root.sequences, slot, s);
}
