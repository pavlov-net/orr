// Debug check before a PTLAS build: more commands than the build allows, an unknown op or a
// second command of one op type rejects the whole build. The command count becomes 0 and
// `failures` counts the rejection.
#include "orr/root.h"

struct Root
{
    uint* commands;
    uint* count;
    uint* failures;
    uint max_ops;
    uint stride_words;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    uint n = *root.count;
    bool bad = n > root.max_ops;
    uint seen = 0;
    for (uint i = 0; i < min(n, root.max_ops); i++)
    {
        uint op = root.commands[i * root.stride_words];
        uint bit = 1u << min(op, 31u);
        bad = bad || op > 2 || (seen & bit) != 0;
        seen |= bit;
    }
    if (bad)
    {
        *root.count = 0;
        InterlockedAdd(*root.failures, 1u);
    }
}
