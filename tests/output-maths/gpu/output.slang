// Calls orr-output's maths, so the build compiles them to SPIR-V (validated) and MSL under
// the lints and generates the Rust layout of Gt7Curve.
#include "orr/root.h"
import gt7;
import exposure;

struct Root
{
    float4* out;
    Gt7Curve* curve;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    float x = float(id.x) * 0.25;
    float3 c = gt7_tonemap(float3(x, 0.5 * x, 0.25 * x), *root.curve);
    float s = exposure_from_ev100(ev100_from_luminance(x)) * luminance_from_ev100(ev100_from_exposure(MID_GREY));
    root.out[id.x] = float4(c, s);
}
