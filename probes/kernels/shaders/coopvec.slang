// 16x16 FP16 matrix times FP16 vector with cooperative vectors, weights in InferencingOptimal
// layout. FORM 0: OpCooperativeVectorMatrixMulNV in inline SPIR-V over an address from the
// root; FORM 1: Slang coopVecMatMul over a ByteAddressBuffer mapped by INDIRECT_ADDRESS to the
// root's `weights` field.
#ifndef FORM
#define FORM 0
#endif

struct Root
{
    half* input;   // 16 per vector
    float* output; // 16 per vector
    uint64_t weights;
    uint offset;   // MatrixOffset operand in bytes
    uint count;
    uint iters;
    uint pad[3];
};
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

#if FORM == 0
CoopVec<half, 16> matmul(CoopVec<half, 16> v, uint64_t weights, uint offset)
{
    Ptr<uint[]> w = Ptr<uint[]>(weights);
    // Operands: input interpretation Float16 (0), matrix interpretation Float16 (0), M = N = 16,
    // layout InferencingOptimal (2), no transpose, stride 0 (unused by optimal layouts).
    return spirv_asm {
        OpCapability CooperativeVectorNV;
        OpExtension "SPV_NV_cooperative_vector";
        result: $$CoopVec<half, 16> = OpCooperativeVectorMatrixMulNV $v $(0) $w $offset $(0) $(16) $(16) $(2) $(false) $(0u);
    };
}
#else
[[vk::binding(1, 0)]] ByteAddressBuffer weights;
CoopVec<half, 16> matmul(CoopVec<half, 16> v, uint64_t unused, uint offset)
{
    return coopVecMatMul<half, 16, 16>(v, CoopVecComponentType.Float16, weights, offset, CoopVecComponentType.Float16,
                                       CoopVecMatrixLayout.InferencingOptimal, false, 0);
}
#endif

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    CoopVec<half, 16> v;
    for (uint k = 0; k < 16; ++k)
        v[k] = root.input[i * 16 + k];
    for (uint it = 0; it < root.iters; ++it)
        v = matmul(v, root.weights, root.offset);
    for (uint k = 0; k < 16; ++k)
        root.output[i * 16 + k] = float(v[k]);
}
