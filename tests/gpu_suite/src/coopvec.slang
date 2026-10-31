// One 16x16 FP16 matrix times one FP16 vector per thread, by OpCooperativeVectorMatrixMulNV
// over a pointer to InferencingOptimal weights.
#include "orr/root.h"

struct Root
{
    float* input;  // 16 per vector
    float* output; // 16 per vector
    uint64_t weights;
    uint offset; // byte offset of the matrix from `weights`
    uint count;
}
ORR_ROOT(Root);

CoopVec<half, 16> matmul(CoopVec<half, 16> v, uint64_t weights, uint offset)
{
    Ptr<uint[]> w = Ptr<uint[]>(weights);
    // Input and matrix interpretation Float16 (0), M = N = 16, InferencingOptimal (2), no
    // transpose, stride 0 (unused by the optimal layouts).
    return spirv_asm {
        OpCapability CooperativeVectorNV;
        OpExtension "SPV_NV_cooperative_vector";
        result: $$CoopVec<half, 16> = OpCooperativeVectorMatrixMulNV $v $(0) $w $offset $(0) $(16) $(16) $(2) $(false) $(0u);
    };
}

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    CoopVec<half, 16> v;
    for (uint k = 0; k < 16; ++k)
        v[k] = half(root.input[i * 16 + k]);
    v = matmul(v, root.weights, root.offset);
    for (uint k = 0; k < 16; ++k)
        root.output[i * 16 + k] = float(v[k]);
}
