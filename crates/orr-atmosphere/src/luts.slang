// The atmosphere's LUT passes: transmittance, multiple scattering and ground irradiance after
// each `set`, then the fold and the sky-view LUTs every frame. Each pass runs the wavelength
// groups three at a time.
#include "orr/root.h"
import orr.abi;
import orr.math;
import orr.color;
import tables;
import lights;
import media;
import light_sampling;
import atmosphere;

struct Root
{
    AtmosphereGpu* atmosphere;
    // The LUT the pass writes.
    StorageImageIndex output;
    // The sky-view LUT, for `sky_view`.
    uint view;
}
ORR_ROOT(Root);

static const uint COLUMN_STEPS = 64;
static const uint MULTIPLE_SCATTERING_STEPS = 32;
static const uint GROUND_STEPS = 32;
static const uint SKY_VIEW_STEPS = 40;
// Directions per multiple-scattering and ground irradiance texel: 64 lanes, 4 each.
static const uint DIRECTIONS = 256;
static const float GOLDEN_ANGLE = 2.39996322972865332;
// Candidates more than 18 deg below the focus horizon add less than the night background.
static const float FOLD_HORIZON = -0.309016994;

float3 to_rec2020(AtmosphereGroups g, float3 v)
{
    return float3(dot(g.rec2020[0].xyz, v), dot(g.rec2020[1].xyz, v), dot(g.rec2020[2].xyz, v));
}

float3 light_groups(AtmosphereGroups g, float3 colour)
{
    return float3(dot(g.light_weight[0].xyz, colour), dot(g.light_weight[1].xyz, colour),
                  dot(g.light_weight[2].xyz, colour));
}

// The sample point and length of step i of n over [t0, t1], denser towards t0.
float2 step_at(float t0, float t1, uint i, uint n)
{
    float u0 = float(i) / float(n);
    float u1 = float(i + 1) / float(n);
    float um = (float(i) + 0.5) / float(n);
    return float2(t0 + (t1 - t0) * um * um, (t1 - t0) * (u1 * u1 - u0 * u0));
}

// Integral of exp(-sigma s) over a step of length dt.
float3 step_integral(float3 sigma, float dt)
{
    float3 x = sigma * dt;
    float3 series = dt * (1.0 - x * (0.5 - x / 6.0));
    float3 exact = (1.0 - exp(-x)) / max(sigma, 1e-30);
    return float3(x.x < 1e-3 ? series.x : exact.x, x.y < 1e-3 ? series.y : exact.y,
                  x.z < 1e-3 ? series.z : exact.z);
}

// Unit vector k of DIRECTIONS spread evenly over the sphere (`hemisphere`: the half
// around +z), in a frame with +z as `z`.
float3 fibonacci(uint k, bool hemisphere, float3 x, float3 y, float3 z)
{
    float f = (float(k) + 0.5) / float(DIRECTIONS);
    float c = hemisphere ? 1.0 - f : 1.0 - 2.0 * f;
    float s = sqrt(max(1.0 - c * c, 0.0));
    float phi = GOLDEN_ANGLE * float(k);
    return s * cos(phi) * x + s * sin(phi) * y + c * z;
}

// Rayleigh, aerosol and ozone densities integrated over [0, s] from altitude h along zenith
// cosine mu >= 0, with samples dense near the start.
float3 column_up(AtmosphereGpu* a, float h, float mu, float s)
{
    float3 c = float3(0.0);
    for (uint i = 0; i < COLUMN_STEPS; ++i)
    {
        float u = (float(i) + 0.5) / float(COLUMN_STEPS);
        float t = s * u * u;
        c += atmosphere_densities(a, atmosphere_altitude_along(a, h, mu, t)) * (2.0 * s * u);
    }
    return c / float(COLUMN_STEPS);
}

// The column to the top of the atmosphere along a ray that does not hit the ground. A
// downward ray is split at its lowest point: from there back to the eye, twice, plus the
// eye's mirrored upward ray.
float3 column_to_top(AtmosphereGpu* a, float h, float mu)
{
    if (mu >= 0.0)
        return column_up(a, h, mu, atmosphere_distance_to_top(a, h, mu));
    float back = -(a.planet_radius + h) * mu;
    return 2.0 * column_up(a, atmosphere_tangent_altitude(a, h, mu), 0.0, back)
        + column_up(a, h, -mu, atmosphere_distance_to_top(a, h, -mu));
}

// Scattering and extinction of three groups at altitude h.
struct Medium
{
    float3 rayleigh;
    float3 aerosol;
    float3 fog;
    float3 extinction;
}

Medium medium(AtmosphereGpu* a, AtmosphereGroups g, float h, bool fog)
{
    float3 d = atmosphere_densities(a, h);
    float f = fog ? atmosphere_fog_extinction(a, h) : 0.0;
    Medium m;
    m.rayleigh = g.rayleigh.xyz * d.x;
    m.aerosol = g.aerosol_scattering.xyz * d.y;
    m.fog = g.fog_albedo.xyz * f;
    m.extinction = m.rayleigh + g.aerosol_extinction.xyz * d.y + g.ozone.xyz * d.z + f;
    return m;
}

// Sky irradiance on the ground per unit spectral irradiance of a light at zenith cosine mu.
float3 ground_sky_irradiance(AtmosphereGpu* a, uint pass, float mu)
{
    float x = saturate(mu * 0.5 + 0.5) * float(GROUND_SIZE - 1);
    uint i = min(uint(x), GROUND_SIZE - 2);
    float f = x - float(i);
    float3 e0 = a.ground_irradiance[i * a.group_passes + pass].xyz;
    float3 e1 = a.ground_irradiance[(i + 1) * a.group_passes + pass].xyz;
    return lerp(e0, e1, f);
}

struct Lights
{
    float3 direction[MAX_FOLDED_LIGHTS];
    // Spectral irradiance of the pass's three groups.
    float3 groups[MAX_FOLDED_LIGHTS];
    uint count;
}

// Radiance of one pass's three groups arriving at distance t0 along unit `dir` from a LUT
// eye at altitude h0, from the medium over [t0, t1] and, with `ground`, the ground at t1.
// Multiple scattering, and the sky's light on the ground, are scaled by `multiple`.
float3 march(AtmosphereGpu* a, uint pass, float h0, float3 dir, float t0, float t1, bool ground,
             uint steps, Lights lights, float multiple)
{
    AtmosphereGroups g = a.groups[pass];
    float3 up = a.up.xyz;
    float mu = dot(dir, up);
    float r0 = a.planet_radius + h0;
    bool fog = a.fog_density > 0.0;
    float3 phase[MAX_FOLDED_LIGHTS];
    for (uint l = 0; l < lights.count; ++l)
    {
        float c = dot(dir, lights.direction[l]);
        phase[l] = float3(rayleigh_phase(c), cornette_shanks_phase(a.aerosol_asymmetry, c),
                          fog ? fog_phase(a.fog_diameter, c) : 0.0);
    }
    float3 radiance = float3(0.0);
    float3 transmittance = float3(1.0);
    for (uint i = 0; i < steps; ++i)
    {
        float2 s = step_at(t0, t1, i, steps);
        float h = atmosphere_altitude_along(a, h0, mu, s.x);
        float r = a.planet_radius + h;
        Medium m = medium(a, g, h, fog);
        float3 scattering = m.rayleigh + m.aerosol + m.fog;
        float3 source = float3(0.0);
        for (uint l = 0; l < lights.count; ++l)
        {
            float3 ld = lights.direction[l];
            float mu_l = (r0 * dot(up, ld) + s.x * dot(dir, ld)) / r;
            float3 single = m.rayleigh * phase[l].x + m.aerosol * phase[l].y + m.fog * phase[l].z;
            source += lights.groups[l] * (atmosphere_light_transmittance(a, h, mu_l, pass) * single
                + multiple * atmosphere_lut_multiple_scattering(a, h, mu_l, pass) * scattering);
        }
        radiance += transmittance * source * step_integral(m.extinction, s.y);
        transmittance *= exp(-m.extinction * s.y);
    }
    if (ground)
    {
        float3 n = (r0 * up + t1 * dir) / a.planet_radius;
        float3 lit = float3(0.0);
        for (uint l = 0; l < lights.count; ++l)
        {
            float mu_l = dot(n, lights.direction[l]);
            lit += lights.groups[l] * (atmosphere_light_transmittance(a, 0.0, mu_l, pass) * max(mu_l, 0.0)
                + multiple * ground_sky_irradiance(a, pass, mu_l));
        }
        radiance += transmittance * g.ground_albedo.xyz * INV_PI * lit;
    }
    return radiance;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void transmittance(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= TRANSMITTANCE_SIZE))
        return;
    AtmosphereGpu* a = root.atmosphere;
    float h, mu;
    atmosphere_transmittance_point(a, float2(id.xy) / float2(TRANSMITTANCE_SIZE - 1), h, mu);
    float3 column = column_to_top(a, h, mu);
    RWTexture2DArray<float4> lut = root.output.get<RWTexture2DArray<float4>>();
    float3 rgb = float3(0.0);
    for (uint p = 0; p < a.group_passes; ++p)
    {
        AtmosphereGroups g = a.groups[p];
        float3 t = exp(-(g.rayleigh.xyz * column.x + g.aerosol_extinction.xyz * column.y
                         + g.ozone.xyz * column.z));
        lut[uint3(id.xy, p)] = float4(t, 1.0);
        rgb += to_rec2020(g, t);
    }
    lut[uint3(id.xy, a.group_passes)] = float4(rgb / a.solar_colour.xyz, 1.0);
}

static groupshared float3 second_order[64];
static groupshared float3 transfer[64];

// Hillaire's (2020) multiple scattering: the second-order radiance arriving at the point
// from every direction under an isotropic phase, times 1 / (1 - f), where f is the share an
// isotropic point scatters back to itself. Clear air only.
[shader("compute")]
[numthreads(64, 1, 1)]
void multiple_scattering(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    AtmosphereGpu* a = root.atmosphere;
    float h, mu_light;
    atmosphere_multiple_scattering_point(a, float2(group.xy) / float2(MULTIPLE_SCATTERING_SIZE - 1),
                                         h, mu_light);
    float3 up = a.up.xyz;
    float3 x, y;
    orthonormal_basis(up, x, y);
    float3 light = sqrt(max(1.0 - mu_light * mu_light, 0.0)) * x + mu_light * up;
    float r0 = a.planet_radius + h;
    RWTexture2DArray<float4> lut = root.output.get<RWTexture2DArray<float4>>();
    float3 rgb = float3(0.0);
    for (uint p = 0; p < a.group_passes; ++p)
    {
        AtmosphereGroups g = a.groups[p];
        float3 l2 = float3(0.0);
        float3 f = float3(0.0);
        for (uint k = lane; k < DIRECTIONS; k += 64)
        {
            float3 dir = fibonacci(k, false, x, y, up);
            float mu = dot(dir, up);
            bool ground = atmosphere_ray_hits_ground(a, h, mu);
            float end = ground ? atmosphere_distance_to_ground(a, h, mu)
                               : atmosphere_distance_to_top(a, h, mu);
            float3 t = float3(1.0);
            for (uint i = 0; i < MULTIPLE_SCATTERING_STEPS; ++i)
            {
                float2 s = step_at(0.0, end, i, MULTIPLE_SCATTERING_STEPS);
                float hs = atmosphere_altitude_along(a, h, mu, s.x);
                float mu_l = (r0 * mu_light + s.x * dot(dir, light)) / (a.planet_radius + hs);
                Medium m = medium(a, g, hs, false);
                float3 scattering = m.rayleigh + m.aerosol;
                float3 w = t * scattering * step_integral(m.extinction, s.y);
                l2 += w * atmosphere_lut_transmittance(a, hs, mu_l, p) * INV_FOUR_PI;
                f += w;
                t *= exp(-m.extinction * s.y);
            }
            if (ground)
            {
                float3 n = (r0 * up + end * dir) / a.planet_radius;
                float mu_g = dot(n, light);
                l2 += t * atmosphere_lut_transmittance(a, 0.0, mu_g, p) * max(mu_g, 0.0)
                    * g.ground_albedo.xyz * INV_PI;
            }
        }
        second_order[lane] = l2;
        transfer[lane] = f;
        GroupMemoryBarrierWithGroupSync();
        for (uint w = 32; w > 0; w >>= 1)
        {
            if (lane < w)
            {
                second_order[lane] += second_order[lane + w];
                transfer[lane] += transfer[lane + w];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0)
        {
            float n = float(DIRECTIONS);
            float3 psi = (second_order[0] / n) / (1.0 - transfer[0] / n);
            lut[uint3(group.xy, p)] = float4(psi, 1.0);
            rgb += to_rec2020(g, psi);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        lut[uint3(group.xy, a.group_passes)] = float4(rgb / a.solar_colour.xyz, 1.0);
}

static groupshared float3 irradiance[64];

// Sky irradiance on the ground per unit spectral irradiance of a light, over the light's
// zenith cosine: the sky radiance (single scattering with every phase function, multiple
// scattering from the LUT, fog included) integrated over the upper hemisphere.
[shader("compute")]
[numthreads(64, 1, 1)]
void ground_irradiance(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    AtmosphereGpu* a = root.atmosphere;
    float mu_light = float(group.x) / float(GROUND_SIZE - 1) * 2.0 - 1.0;
    float3 up = a.up.xyz;
    float3 x, y;
    orthonormal_basis(up, x, y);
    Lights lights;
    lights.count = 1;
    lights.direction[0] = sqrt(max(1.0 - mu_light * mu_light, 0.0)) * x + mu_light * up;
    lights.groups[0] = float3(1.0);
    for (uint p = 0; p < a.group_passes; ++p)
    {
        float3 e = float3(0.0);
        for (uint k = lane; k < DIRECTIONS; k += 64)
        {
            float3 dir = fibonacci(k, true, x, y, up);
            float3 l = march(a, p, 0.0, dir, 0.0, atmosphere_distance_to_top(a, 0.0, dot(dir, up)),
                             false, GROUND_STEPS, lights, 1.0);
            e += l * dot(dir, up);
        }
        irradiance[lane] = e;
        GroupMemoryBarrierWithGroupSync();
        for (uint w = 32; w > 0; w >>= 1)
        {
            if (lane < w)
                irradiance[lane] += irradiance[lane + w];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0)
            a.ground_irradiance[group.x * a.group_passes + p] =
                float4(irradiance[0] * (TWO_PI / float(DIRECTIONS)), 0.0);
        GroupMemoryBarrierWithGroupSync();
    }
}

// Fold membership: the `sky_lights` brightest eligible candidates, ties to the earlier one.
// A candidate is eligible when it names a directional light with some illuminance no more
// than 18 deg below the focus horizon. Writes each candidate's illuminance.
[shader("compute")]
[numthreads(1, 1, 1)]
void fold(uint3 id: SV_DispatchThreadID)
{
    AtmosphereGpu* a = root.atmosphere;
    SceneRoot* scene = atmosphere_scene(a);
    FoldGpu* f = a.fold;
    float luminance[MAX_SKY_LIGHTS];
    for (uint i = 0; i < MAX_SKY_LIGHTS; ++i)
    {
        luminance[i] = 0.0;
        f.illuminance[i] = float4(0.0);
        uint index = a.candidates[i];
        if (i >= a.candidate_count || index >= scene.light_capacity)
            continue;
        LightGpu l = scene.lights[index];
        if (l.kind != LIGHT_DIRECTIONAL || dot(normalize(l.direction), a.up.xyz) < FOLD_HORIZON)
            continue;
        float3 e = l.disc == DISC_TEXTURED && all(l.color == 0.0)
            ? textured_disc_integral(scene, index) : l.color;
        f.illuminance[i] = float4(e, 0.0);
        luminance[i] = max(luminance_rec2020(e), 0.0);
    }
    for (uint i = 0; i < MAX_SKY_LIGHTS; ++i)
    {
        uint brighter = 0;
        for (uint j = 0; j < MAX_SKY_LIGHTS; ++j)
            if (luminance[j] > luminance[i] || (luminance[j] == luminance[i] && j < i))
                ++brighter;
        f.weight[i] = luminance[i] > 0.0 && brighter < a.sky_lights ? 1.0 : 0.0;
    }
    uint taken = 0;
    uint count = 0;
    for (; count < MAX_FOLDED_LIGHTS; ++count)
    {
        int best = -1;
        for (uint i = 0; i < MAX_SKY_LIGHTS; ++i)
            if ((taken & (1u << i)) == 0 && f.weight[i] > 0.0
                && (best < 0 || f.weight[i] > f.weight[best]))
                best = int(i);
        if (best < 0)
            break;
        taken |= 1u << uint(best);
        f.folded[count] = uint(best);
    }
    f.folded_count = count;
}

// One sky-view LUT: from `near_range` along each texel's direction, the folded lights'
// in-scatter, and the virtual ground below the horizon.
[shader("compute")]
[numthreads(8, 8, 1)]
void sky_view(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= SKY_VIEW_SIZE))
        return;
    AtmosphereGpu* a = root.atmosphere;
    float h0 = a.sky_view_altitude[root.view];
    float3 dir = atmosphere_sky_view_direction(a, h0, (float2(id.xy) + 0.5) / float2(SKY_VIEW_SIZE));
    float mu = dot(dir, a.up.xyz);
    bool ground = atmosphere_ray_hits_ground(a, h0, mu);
    float end = ground ? atmosphere_distance_to_ground(a, h0, mu)
                       : atmosphere_distance_to_top(a, h0, mu);
    SceneRoot* scene = atmosphere_scene(a);
    float start = min(scene.near_range, end);

    FoldGpu* f = a.fold;
    float3 colour[MAX_FOLDED_LIGHTS];
    Lights lights;
    lights.count = f.folded_count;
    for (uint l = 0; l < lights.count; ++l)
    {
        uint slot = f.folded[l];
        lights.direction[l] = normalize(scene.lights[a.candidates[slot]].direction);
        colour[l] = f.illuminance[slot].rgb * f.weight[slot];
    }
    float3 rgb = float3(0.0);
    for (uint p = 0; p < a.group_passes; ++p)
    {
        AtmosphereGroups g = a.groups[p];
        for (uint l = 0; l < lights.count; ++l)
            lights.groups[l] = light_groups(g, colour[l]);
        rgb += to_rec2020(g, march(a, p, h0, dir, start, end, ground, SKY_VIEW_STEPS, lights,
                                   a.multiple_scattering_scale));
    }
    root.output.get<RWTexture2D<float4>>()[id.xy] = float4(rgb, 1.0);
}
