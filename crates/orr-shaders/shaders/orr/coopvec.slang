// Cooperative-vector matrix multiplies with the weights behind a device address. This is the
// only module that names cooperative vectors: kernels hold a `Vector` and multiply through
// the functions below.
//
// Weights are an M x K matrix in the device's inference layout (orr-gpu's
// `MatrixLayout::InferencingOptimal`) at `weights + offset`, where `weights` and `offset` are
// each a multiple of 64 bytes. Biases are M halves at `bias + bias_offset`, each a multiple
// of 16 bytes. Vectors are FP16; with FP8 weights the input is converted to the weights' type
// before the multiply, and the result is FP16. A `Vector` lives in function and private
// storage only: in memory, vectors are plain arrays.
module coopvec;

#include "orr/debug.h"

// An FP16 vector of N elements.
public typealias Vector<let N : int> = CoopVec<half, N>;

// The type weights are stored as. The values are the SPIR-V and Vulkan component types.
public enum Weights
{
    F16 = 0,
    E4M3 = 1000491002,
    E5M2 = 1000491003,
}

// SPIR-V operand values for the matrix layout and the bias component type.
static const int INFERENCING_OPTIMAL = 2;
static const int FLOAT16 = 0;

// `weights * v`.
[ForceInline]
public Vector<M> matmul<let M : int, let K : int>(Vector<K> v, uint64_t weights, uint offset, constexpr Weights type)
{
    orr_assert(((weights | offset) & 63) == 0, "cooperative-vector weights at %llx + %u are not 64-byte aligned", weights, offset);
    Ptr<uint[]> w = Ptr<uint[]>(weights);
    let t = int(type);
    return spirv_asm {
        OpCapability CooperativeVectorNV;
        OpExtension "SPV_NV_cooperative_vector";
        result: $$Vector<M> = OpCooperativeVectorMatrixMulNV $v $(t) $w $offset $(t) $(M) $(K) $(INFERENCING_OPTIMAL) $(false) $(0u);
    };
}

// `weights * v + bias`.
[ForceInline]
public Vector<M> matmul_add<let M : int, let K : int>(Vector<K> v, uint64_t weights, uint offset, constexpr Weights type, uint64_t bias, uint bias_offset)
{
    orr_assert(((weights | offset) & 63) == 0, "cooperative-vector weights at %llx + %u are not 64-byte aligned", weights, offset);
    orr_assert(((bias | bias_offset) & 15) == 0, "cooperative-vector bias at %llx + %u is not 16-byte aligned", bias, bias_offset);
    Ptr<uint[]> w = Ptr<uint[]>(weights);
    Ptr<uint[]> b = Ptr<uint[]>(bias);
    let t = int(type);
    return spirv_asm {
        OpCapability CooperativeVectorNV;
        OpExtension "SPV_NV_cooperative_vector";
        result: $$Vector<M> = OpCooperativeVectorMatrixMulAddNV $v $(t) $w $offset $(t) $b $bias_offset $(FLOAT16) $(M) $(K) $(INFERENCING_OPTIMAL) $(false) $(0u);
    };
}
