// One 16x16 matrix times one FP16 vector per thread through orr.coopvec, plus a bias when
// `bias` is not 0. One entry per weight type.
#include "orr/root.h"
import orr.coopvec;

struct Root
{
    float* input;  // 16 per vector
    float* output; // 16 per vector
    uint64_t weights;
    uint64_t bias; // 0 for none
    uint offset;   // byte offset of the matrix from `weights`
    uint bias_offset;
    uint count;
    uint pad;
}
ORR_ROOT(Root);

[ForceInline]
void body(uint i, constexpr Weights type)
{
    if (i >= root.count)
        return;
    Vector<16> v;
    for (uint k = 0; k < 16; ++k)
        v[k] = half(root.input[i * 16 + k]);
    if (root.bias != 0)
        v = matmul_add<16>(v, root.weights, root.offset, type, root.bias, root.bias_offset);
    else
        v = matmul<16>(v, root.weights, root.offset, type);
    for (uint k = 0; k < 16; ++k)
        root.output[i * 16 + k] = float(v[k]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void f16(uint3 id: SV_DispatchThreadID)
{
    body(id.x, Weights.F16);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void e4m3(uint3 id: SV_DispatchThreadID)
{
    body(id.x, Weights.E4M3);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void e5m2(uint3 id: SV_DispatchThreadID)
{
    body(id.x, Weights.E5M2);
}
