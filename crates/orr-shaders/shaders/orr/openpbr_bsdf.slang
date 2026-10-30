// adobe/openpbr-bsdf compiled as one module; kernels import `orr.openpbr`, which re-exports
// it. The module imports nothing: in a module with an import, Slang 2026.20 ignores
// `#pragma warning(disable)` for flow-analysis diagnostics.

#define OPENPBR_LANGUAGE_TARGET_SLANG 1
// Slang does not parse the interop default, `unsigned short`.
#define OPENPBR_UINT16 uint16_t

#ifdef ORR_OPENPBR_ARRAY_LUTS

// Every feature on. Specialisation constants would become kernel parameters on the C++
// target, which `export __extern_cpp` functions cannot take.
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) true

#else

#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) ORR_OPENPBR_##name
#define ORR_OPENPBR_EnableSheenAndCoat openpbr_enable_sheen_and_coat
#define ORR_OPENPBR_EnableDispersion openpbr_enable_dispersion
#define ORR_OPENPBR_EnableTranslucency openpbr_enable_translucency
#define ORR_OPENPBR_EnableMetallic openpbr_enable_metallic

[vk::constant_id(100)] const bool openpbr_enable_sheen_and_coat = true;
[vk::constant_id(101)] const bool openpbr_enable_dispersion = true;
[vk::constant_id(102)] const bool openpbr_enable_translucency = true;
[vk::constant_id(103)] const bool openpbr_enable_metallic = true;

// Defined in orr/openpbr.slang.
extern float4 orr_openpbr_sample_2d(int id, float2 uv);
extern float4 orr_openpbr_sample_3d(int id, float3 uvw);
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_SAMPLE_2D_TEXTURE(id, uv) orr_openpbr_sample_2d(id, uv)
#define OPENPBR_SAMPLE_3D_TEXTURE(id, uvw) orr_openpbr_sample_3d(id, uvw)

#endif

// openpbr-bsdf fills per-channel arrays under a dispersion test that Slang's flow analysis
// cannot follow (E41035, possibly uninitialised), which `-warnings-as-errors all` rejects.
#pragma warning(disable : 41035)
#include "openpbr.h"
