// OpenPBR reference BSDF through Slang, exported for the CPU.
#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_UINT16 uint16_t
#include "openpbr.h"

export __extern_cpp void orr_test_openpbr_eval(float* out3, float r, float g, float b, float rough,
    float ox, float oy, float oz, float ix, float iy, float iz)
{
    OpenPBR_ResolvedInputs inputs = openpbr_make_default_resolved_inputs();
    inputs.base_color = float3(r, g, b);
    inputs.specular_roughness = rough;
    OpenPBR_PreparedBsdf p = openpbr_prepare(inputs, float3(1.0), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor,
                                             normalize(float3(ox, oy, oz)));
    OpenPBR_DiffuseSpecular ds = openpbr_eval(p, normalize(float3(ix, iy, iz)));
    float3 v = ds.diffuse + ds.specular;
    out3[0] = v.x;
    out3[1] = v.y;
    out3[2] = v.z;
}
