// The OpenPBR 1.1.1 reference BSDF (adobe/openpbr-bsdf, unedited): `openpbr_prepare`,
// `openpbr_eval`, `openpbr_sample`, `openpbr_pdf` and their types. `Build` and `CpuLibrary`
// put the fetched openpbr-bsdf directory on the include path.
//
// Feature toggles are specialisation constants 100-103 (default true). Kernels read the eight
// lookup tables from consecutive heap slots starting at specialisation constant 104, in
// `OpenPBR_LutId` order (`orr_shaders::openpbr::LUTS`), and filter them in the shader the way
// the array mode does, so both modes give the same values. With `ORR_OPENPBR_ARRAY_LUTS`
// defined the tables are compiled in instead (CPU tests).
module openpbr;

__exported import orr.openpbr_bsdf;

#ifndef ORR_OPENPBR_ARRAY_LUTS

import orr.abi;

[vk::constant_id(104)] const uint openpbr_lut_slot = 0;

// The table sizes of openpbr_data_constants.h.
static const int SIZE = 32;
static const int METAL_AVERAGE_ID = 6;

float4 texel_2d(int id, int2 p)
{
    TextureIndex t;
    t.index = openpbr_lut_slot + uint(id);
    return t.get<Texture2D<float4>>().Load(int3(p, 0));
}

float4 texel_3d(int id, int3 p)
{
    TextureIndex t;
    t.index = openpbr_lut_slot + uint(id);
    return t.get<Texture3D<float4>>().Load(int4(p, 0));
}

public export float4 orr_openpbr_sample_2d(int id, float2 uv)
{
    int2 size = int2(SIZE, id == METAL_AVERAGE_ID ? 1 : SIZE);
    float2 x = clamp(uv * float2(size) - 0.5, float2(0.0), float2(size - 1));
    int2 i0 = int2(floor(x));
    int2 i1 = min(i0 + 1, size - 1);
    float2 f = x - float2(i0);
    float4 a = lerp(texel_2d(id, i0), texel_2d(id, int2(i1.x, i0.y)), f.x);
    float4 b = lerp(texel_2d(id, int2(i0.x, i1.y)), texel_2d(id, i1), f.x);
    return lerp(a, b, f.y);
}

public export float4 orr_openpbr_sample_3d(int id, float3 uvw)
{
    float3 x = clamp(uvw * float(SIZE) - 0.5, float3(0.0), float3(SIZE - 1));
    int3 i0 = int3(floor(x));
    int3 i1 = min(i0 + 1, SIZE - 1);
    float3 f = x - float3(i0);
    float4 c00 = lerp(texel_3d(id, i0), texel_3d(id, int3(i1.x, i0.y, i0.z)), f.x);
    float4 c10 = lerp(texel_3d(id, int3(i0.x, i1.y, i0.z)), texel_3d(id, int3(i1.x, i1.y, i0.z)), f.x);
    float4 c01 = lerp(texel_3d(id, int3(i0.x, i0.y, i1.z)), texel_3d(id, int3(i1.x, i0.y, i1.z)), f.x);
    float4 c11 = lerp(texel_3d(id, int3(i0.x, i1.y, i1.z)), texel_3d(id, i1), f.x);
    return lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z);
}

#endif
