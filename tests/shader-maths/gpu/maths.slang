// Calls every function of orr's maths libraries, so the build compiles them to SPIR-V
// (validated) and MSL under the lints.
#include "orr/root.h"
import orr.math;
import orr.packing;
import orr.random;
import orr.sampling;
import orr.color;

struct Root
{
    float4* out;
    AliasEntry* aliases;
    uint alias_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
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
    float3x4 m = affine(quat_to_matrix(quat_mul(q, quat_conjugate(q))), n);
    float3x4 mi = affine_inverse(affine_mul(m, m));
    float3 world = from_local(local, t, b, n) + transform_point(mi, n) + transform_vector(m, t)
        + transform_normal(m, b) + quat_rotate(q, n) + offset_ray_origin(n, b) + affine_translation(m);

    float3 packed = oct_decode_unorm16(oct_encode_unorm16(n)) + oct_decode_snorm16(oct_encode_snorm16(n))
        + oct_decode_snorm8(oct_encode_snorm8(n)) + unpack_r11g11b10f(pack_r11g11b10f(n)) + unpack_rgb9e5(pack_rgb9e5(n))
        + float3(unpack_unorm2x16(pack_unorm2x16(u)), 0.0) + float3(unpack_snorm2x16(pack_snorm2x16(u)), 0.0)
        + float3(unpack_half2x16(pack_half2x16(u)), 0.0) + unpack_unorm4x8(pack_unorm4x8(u4)).xyz
        + unpack_snorm4x8(pack_snorm4x8(u4)).xyz + grid_position(int3(id), uint3(1, 2, 3), grid_step(-7))
        + mul(instance_rotation(pcg(id.x), (id.x & 1) != 0), n) + decode_rotation(pcg3d(id).x).xyz;

    SphericalRectangle rect = SphericalRectangle(float3(0.0), float3(-1.0, -1.0, 2.0), float3(2.0, 0.0, 0.0), float3(0.0, 2.0, 0.0));
    SphericalTriangle tri = SphericalTriangle(float3(0.0), float3(1.0, 0.0, 1.0), float3(0.0, 1.0, 1.0), float3(-1.0, 0.0, 1.0));
    float3 l = sample_ggx_vndf_bounded(n, float2(0.3, 0.5), u);
    float3 sampled = sample_cosine_hemisphere(u) * cosine_hemisphere_pdf(n.z) + sample_uniform_hemisphere(u) * uniform_hemisphere_pdf()
        + sample_cone(u, 0.9) * cone_pdf(0.9) + rect.sample(u) * rect.pdf() + tri.sample(u) * tri.pdf()
        + l * ggx_vndf_bounded_pdf(n, l, float2(0.3, 0.5)) * mis_power(u.x, u.y) * mis_balance(u.x, u.y)
        + float(sample_alias(root.aliases, root.alias_count, u.x)) * uniform_sphere_pdf();

    float3 c = mul(REC2020_TO_REC709, mul(P3_TO_REC2020, mul(XYZ_TO_P3, mul(REC709_TO_XYZ, abs(n)))));
    c = mul(XYZ_TO_REC709, mul(REC2020_TO_XYZ, mul(REC709_TO_REC2020, gamut_compress(c))));
    c = mul(XYZ_TO_REC2020, mul(P3_TO_XYZ, mul(REC2020_TO_P3, c)));
    float3 colour = srgb_decode(srgb_encode(c)) + pq_decode(pq_encode(c)) + rec2020_from_ictcp(ictcp_from_rec2020(c))
        + luminance_rec709(c) + luminance_rec2020(c);

    float4 hashes = float4(pcg4d(uint4(id, 1))) + float4(r2(id.x), owen_sobol4(id.x, hash_combine(id.y, 3)).xy)
        + float4(unit_float(pcg(id.x)), float(sobol(id.x, 2)), float(nested_uniform_scramble(id.x, 9)), s.next());
    root.out[id.x] = float4(world + packed + sampled + colour, PI) + hashes;
}
