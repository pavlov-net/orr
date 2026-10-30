// Probes of the atmosphere module: LUT parameterisation round trips, altitudes, LUT texels,
// the sky-view LUT integrated over the upper hemisphere, `AtmosphereSky` directly and as
// lighting bakes it, the night background, and a light's disc seen through the atmosphere.
#include "orr/root.h"
import orr.math;
import orr.color;
import lights;
import media;
import environment;
import light_sampling;
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
    // Lighting's `LightSetGpu`, for `disc`.
    uint64_t set;
    uint64_t _pad0;
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

// The sum of `e` over the 64 lanes of the group.
float3 group_sum(uint lane, float3 e)
{
    sums[lane] = e;
    GroupMemoryBarrierWithGroupSync();
    for (uint w = 32; w > 0; w >>= 1)
    {
        if (lane < w)
            sums[lane] += sums[lane + w];
        GroupMemoryBarrierWithGroupSync();
    }
    return sums[0];
}

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
    e = group_sum(lane, e);
    if (lane != 0)
        return;
    float mu = dot(root.light.xyz, up);
    float h = atmosphere_spherical_altitude(a, eye);
    root.output[0] = float4(e * (TWO_PI / float(root.count)), 0.0);
    root.output[1] =
        float4(root.colour.rgb * atmosphere_transmittance_to_space(a, h, mu) * max(mu, 0.0), 0.0);
}

// `AtmosphereSky.radiance` along each direction `input[i]` from the eye `light.xyz` (render
// space).
[shader("compute")]
[numthreads(64, 1, 1)]
void sky_radiance(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.output[id.x] = float4(AtmosphereSky.radiance((uint64_t)root.atmosphere,
                                                          root.input[id.x].xyz, root.light.xyz), 0.0);
}

// The night background before extinction from altitude `light.w` at zenith cosine
// `input[i].x`.
[shader("compute")]
[numthreads(64, 1, 1)]
void night_background(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.output[id.x] =
            float4(atmosphere_night_background(root.atmosphere, root.light.w, root.input[id.x].x));
}

// Output 0: the mean over the texels of a `count` x `count` equal-area map of what lighting's
// bake puts in each for `AtmosphereSky` seen from the eye `light.xyz`, luminance in w.
[shader("compute")]
[numthreads(64, 1, 1)]
void bake_mean(uint lane: SV_GroupIndex)
{
    uint n = root.count;
    float3 e = float3(0.0);
    for (uint k = lane; k < n * n; k += 64)
        e += bake_texel<AtmosphereSky>((uint64_t)root.atmosphere, root.light.xyz, uint2(k % n, k / n), n);
    e = group_sum(lane, e);
    if (lane == 0)
    {
        float3 m = e / float(n * n);
        root.output[0] = float4(m, luminance_rec2020(m));
    }
}

// The atmosphere's transmittance to space as a medium with no scattering; `data` is the block.
struct SkyTransmittance : IMedium
{
    static MediumPoint sample(uint64_t data, float3 p_render)
    {
        return NoMedium.sample(data, p_render);
    }

    static float3 phase(uint64_t data, MediumPoint m, float cos_theta)
    {
        return float3(0.0);
    }

    static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light)
    {
        return float3(0.0);
    }

    static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        return atmosphere_transmittance_to_space(a, atmosphere_altitude(a, origin), dot(dir, a.up.xyz));
    }
}

// Light `layer`'s disc seen from the eye `light.xyz` through the atmosphere: output 0 its
// radiance integrated over the cone, weighted by the cosine to the axis (`count` rings of 64
// directions, equal in solid angle); output 1 its illuminance times the transmittance along
// the axis.
[shader("compute")]
[numthreads(64, 1, 1)]
void disc(uint lane: SV_GroupIndex)
{
    LightSetGpu* set = (LightSetGpu*)root.set;
    LightGpu l = set.scene.lights[root.layer];
    float3 eye = root.light.xyz;
    float3 t, b;
    orthonormal_basis(l.direction, t, b);
    float sin_max = sin(l.radius);
    float phi = TWO_PI * (float(lane) + 0.5) / 64.0;
    float3 e = float3(0.0);
    for (uint k = 0; k < root.count; k++)
    {
        float s = sin_max * sqrt((float(k) + 0.5) / float(root.count));
        float3 dir = sqrt(1.0 - s * s) * l.direction + s * (cos(phi) * t + sin(phi) * b);
        e += directional_disc<SkyTransmittance>(set, root.layer, eye, dir);
    }
    e = group_sum(lane, e);
    if (lane != 0)
        return;
    root.output[0] = float4(e * (PI * sin_max * sin_max / (64.0 * float(root.count))), 0.0);
    root.output[1] = float4(l.color * SkyTransmittance.transmittance(set.scene.medium, eye,
                                                                     l.direction, positive_infinity()), 0.0);
}
