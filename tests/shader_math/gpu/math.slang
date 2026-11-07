// Calls every function of orr's math libraries, so the build compiles them to SPIR-V
// (validated) and MSL under the lints.
#include "orr_root.h"
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;

struct Root
{
    float4* out;
    uint64_t _pad0;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    Sampler s = Sampler(id.xy, id.z);
    float2 u = s.next2();
    float4 u4 = s.next4();
    float3 n = sample_uniform_sphere(u);

    float3 t, b;
    orthonormal_basis(n, t, b);
    float3 local = to_local(safe_normalize(t + b), t, b, n);
    float4 q = normalize(u4 - 0.5);
    float3x4 m = affine(quat_to_matrix(q), n);
    float3x4 mm = affine_mul(m, m);
    float3 world = from_local(local, t, b, n) + transform_point(mm, n) + transform_vector(m, t)
        + transform_normal(m, b) + affine_translation(m) + mul(linear_inverse(affine_linear(m)), n)
        + max_component(n) + float(max_component(uint3(id))) + largest_scale(affine_linear(m));

    float3 packed = oct_decode_unorm16(oct_encode_unorm16(n)) + oct_decode_snorm16(oct_encode_snorm16(n))
        + oct_decode_snorm8(oct_encode_snorm8(n)) + unpack_rgb9e5(pack_rgb9e5(n))
        + float3(unpack_unorm2x16(id.x), 0.0) + float3(unpack_snorm2x16(pack_snorm2x16(u)), 0.0)
        + float3(unpack_half2x16(pack_half2x16(u)), 0.0) + unpack_unorm4x8(pack_unorm4x8(u4)).xyz
        + unpack_snorm4x8(id.y).xyz + unpack_unorm8(pack_unorm8(u.x)) + unpack_snorm8(pack_snorm8(u.y)) + grid_position(int3(id), uint3(1, 2, 3), grid_step(-7))
        + mul(instance_rotation(pcg(id.x), (id.x & 1) != 0), n) + decode_rotation(pcg3d(id).x).xyz;

    float3 l = sample_ggx_vndf_bounded(n, float2(0.3, 0.5), u);
    float3 sampled = sample_cosine_hemisphere(u) * cosine_hemisphere_pdf(n.z) + sample_uniform_hemisphere(u)
        + sample_cone(u, 0.1) * cone_pdf(0.1)
        + l * ggx_vndf_bounded_pdf(n, l, float2(0.3, 0.5)) * mis_balance(u.x, u.y);

    float4 hashes = float4(pcg4d(uint4(id, 1))) + float4(r2(id.x), owen_sobol4(id.x, hash_combine(id.y, 3)).xy)
        + float4(pcg_float(id.x), float(sobol(id.x, 2)), float(nested_uniform_scramble(id.x, 9)), s.next());
    root.out[id.x] = float4(world + packed + sampled, PI) + hashes;
}
