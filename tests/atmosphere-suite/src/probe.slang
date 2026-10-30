// Probes of the atmosphere module: LUT parameterisation round trips, altitudes, LUT texels
// and the sky-view LUT integrated over the upper hemisphere.
#include "orr/root.h"
import orr.math;
import atmosphere;

struct Root
{
    AtmosphereGpu* atmosphere;
    float4* input;
    float4* output;
    uint count;
    // Transmittance or multiple-scattering layer; sky-view LUT.
    uint layer;
    // Light direction and the LUT eye's altitude in w.
    float4 light;
    // The light's Rec.2020 illuminance.
    float4 colour;
}
ORR_ROOT(Root);

// Per transmittance texel: altitude, zenith cosine, and the round trip's error in texels.
[shader("compute")]
[numthreads(64, 1, 1)]
void transmittance_texels(uint3 id: SV_DispatchThreadID)
{
    uint2 size = TRANSMITTANCE_SIZE;
    if (id.x >= size.x * size.y)
        return;
    AtmosphereGpu* a = root.atmosphere;
    float2 unit = float2(id.x % size.x, id.x / size.x) / float2(size - 1);
    float h, mu;
    atmosphere_transmittance_point(a, unit, h, mu);
    float2 back = atmosphere_transmittance_unit(a, h, mu);
    root.output[id.x] = float4(h, mu, abs(back - unit) * float2(size - 1));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void multiple_scattering_texels(uint3 id: SV_DispatchThreadID)
{
    uint2 size = MULTIPLE_SCATTERING_SIZE;
    if (id.x >= size.x * size.y)
        return;
    AtmosphereGpu* a = root.atmosphere;
    float2 unit = float2(id.x % size.x, id.x / size.x) / float2(size - 1);
    float h, mu;
    atmosphere_multiple_scattering_point(a, unit, h, mu);
    float2 back = atmosphere_multiple_scattering_unit(a, h, mu);
    root.output[id.x] = float4(h, mu, abs(back - unit) * float2(size - 1));
}

// Per sky-view texel from a LUT eye at altitude `light.w`: the direction and the round
// trip's error in texels, the longitude's taken across the seam.
[shader("compute")]
[numthreads(64, 1, 1)]
void sky_view_texels(uint3 id: SV_DispatchThreadID)
{
    uint2 size = SKY_VIEW_SIZE;
    if (id.x >= size.x * size.y)
        return;
    AtmosphereGpu* a = root.atmosphere;
    uint row = id.x / size.x;
    float2 uv = (float2(id.x % size.x, row) + 0.5) / float2(size);
    float3 dir = atmosphere_sky_view_direction(a, root.light.w, uv);
    float2 d = abs(atmosphere_sky_view_uv(a, root.light.w, dir) - uv);
    d.x = min(d.x, 1.0 - d.x);
    // The zenith and nadir rows have no longitude.
    if (row == 0 || row == size.y - 1)
        d.x = 0.0;
    root.output[id.x] = float4(dir, max(d.x * float(size.x), d.y * float(size.y)));
}

// Plane and spherical altitude of each render-space point.
[shader("compute")]
[numthreads(64, 1, 1)]
void altitude(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    AtmosphereGpu* a = root.atmosphere;
    float3 p = root.input[id.x].xyz;
    root.output[id.x] =
        float4(atmosphere_altitude(a, p), atmosphere_spherical_altitude(a, p), 0.0, 0.0);
}

// Every texel of transmittance layer `layer`, row by row.
[shader("compute")]
[numthreads(64, 1, 1)]
void transmittance_lut(uint3 id: SV_DispatchThreadID)
{
    uint2 size = TRANSMITTANCE_SIZE;
    if (id.x < size.x * size.y)
        root.output[id.x] = root.atmosphere.transmittance.get<Texture2DArray<float4>>()
            .Load(int4(id.x % size.x, id.x / size.x, root.layer, 0));
}

// Every texel of sky-view LUT `layer`, row by row.
[shader("compute")]
[numthreads(64, 1, 1)]
void sky_view_lut(uint3 id: SV_DispatchThreadID)
{
    uint2 size = SKY_VIEW_SIZE;
    if (id.x < size.x * size.y)
        root.output[id.x] = root.atmosphere.sky_view[root.layer].get<Texture2D<float4>>()
            .Load(int3(id.x % size.x, id.x / size.x, 0));
}

static groupshared float3 sums[64];

// Irradiance on a horizontal surface at the eye `input[0]` (render space): output 0 is the
// sky-view LUT over the upper hemisphere (`count` directions), output 1 the light `colour`
// through the atmosphere; both Rec.2020 lux.
[shader("compute")]
[numthreads(64, 1, 1)]
void irradiance(uint lane: SV_GroupIndex)
{
    AtmosphereGpu* a = root.atmosphere;
    float3 up = a.up.xyz;
    float3 x, y;
    orthonormal_basis(up, x, y);
    float3 eye = root.input[0].xyz;
    float3 e = float3(0.0);
    for (uint k = lane; k < root.count; k += 64)
    {
        float c = 1.0 - (float(k) + 0.5) / float(root.count);
        float s = sqrt(max(1.0 - c * c, 0.0));
        float phi = 2.39996322972865332 * float(k);
        float3 dir = s * cos(phi) * x + s * sin(phi) * y + c * up;
        e += atmosphere_sky_view(a, dir, eye) * c;
    }
    sums[lane] = e;
    GroupMemoryBarrierWithGroupSync();
    for (uint w = 32; w > 0; w >>= 1)
    {
        if (lane < w)
            sums[lane] += sums[lane + w];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane != 0)
        return;
    float mu = dot(root.light.xyz, up);
    float h = atmosphere_spherical_altitude(a, eye);
    root.output[0] = float4(sums[0] * (TWO_PI / float(root.count)), 0.0);
    root.output[1] =
        float4(root.colour.rgb * atmosphere_transmittance_to_space(a, h, mu) * max(mu, 0.0), 0.0);
}
