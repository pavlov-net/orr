// Reads the linked environment and medium through their data addresses.
#include "orr/root.h"
import environment;
import media;

extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;

struct SkyRoot
{
    float4* out;
    uint64_t environment;
    uint64_t medium;
    uint64_t pad;
}

ORR_ROOT(SkyRoot);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    root.out[0] = float4(Environment.radiance(root.environment, float3(0.0, 1.0, 0.0), float3(0.0)), 0.0);
    MediumPoint m = Medium.sample(root.medium, float3(0.0));
    root.out[1] = float4(m.sigma_s, m.phase_parameter);
    root.out[2] = float4(Medium.transmittance(root.medium, float3(0.0), float3(0.0, 1.0, 0.0), 2.0), 0.0);
}
