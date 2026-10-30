// 16x16 FP16 cooperative-vector matrix multiply with the weights behind an address: inline
// OpCooperativeVectorMatrixMulNV, since coopVecMatMul takes only buffer bindings.
struct Root
{
    half* input;
    float* output;
    uint64_t weights;
    uint offset;
    uint count;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

CoopVec<half, 16> matmul(CoopVec<half, 16> v, uint64_t weights, uint offset)
{
    Ptr<uint[]> w = Ptr<uint[]>(weights);
    // Float16 input and matrix, M = N = 16, InferencingOptimal layout, no transpose.
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
        v[k] = root.input[i * 16 + k];
    v = matmul(v, root.weights, root.offset);
    for (uint k = 0; k < 16; ++k)
        root.output[i * 16 + k] = float(v[k]);
}
