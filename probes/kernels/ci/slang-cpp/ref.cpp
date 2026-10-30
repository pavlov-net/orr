// OpenPBR C++ reference (adobe/openpbr-bsdf with GLM), same entry shape as the Slang wrapper.
#include <glm/glm.hpp>
#include "openpbr.h"

extern "C" void ref_openpbr_eval(float* out3, float r, float g, float b, float rough,
    float ox, float oy, float oz, float ix, float iy, float iz)
{
    OpenPBR_ResolvedInputs inputs = openpbr_make_default_resolved_inputs();
    inputs.base_color = vec3(r, g, b);
    inputs.specular_roughness = rough;
    const OpenPBR_PreparedBsdf p = openpbr_prepare(inputs, vec3(1.0f), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor,
                                                   normalize(vec3(ox, oy, oz)));
    const OpenPBR_DiffuseSpecular ds = openpbr_eval(p, normalize(vec3(ix, iy, iz)));
    const vec3 v = ds.diffuse + ds.specular;
    out3[0] = v.x;
    out3[1] = v.y;
    out3[2] = v.z;
}
