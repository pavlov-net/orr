// The atmosphere's LUT passes: transmittance, multiple scattering and ground irradiance after
// each `set`, then the fold and the sky-view LUTs every frame. Each pass runs the wavelength
// groups three at a time.
//
// Multiple scattering is Bruneton and Neyret's (2008) iteration over scattering orders on the
// spherical planet: the radiance field of one order, a table over altitude, the light's and the
// ray's zenith cosines and the azimuth between them, gives the next order's in-scatter by the
// medium's phase functions, and that in-scatter, marched along each ray, the next radiance field.
// The phase functions act on the radiance's azimuthal cosine modes through an operator built once
// for the medium's phase parameters and radii (`ms_kernel`). The in-scatter of the orders from
// the second sums into the table the medium and the sky-view march read.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_scene.tables;
import orr_scene.lights;
import orr_lighting.media;
import orr_lighting.light_sampling;
import orr_atmosphere.atmosphere;

struct Root
{
    AtmosphereGpu* atmosphere;
    // Per light zenith texel and group pass, the sky irradiance on the ground of the order
    // `ms_scatter` reads, which `ms_gather` reflects.
    float4* ground;
    // The multiple-scattering build's radiance table, phase operator and the rest it keeps
    // (`ms_scatter`).
    float4* radiance;
    float4* kernel;
    float4* aux;
    // The running sum of the orders' in-scatter, linear, by texel (`ms_sum_index`).
    float4* sum;
    // The LUT the pass writes; the march table for `ms_scatter`.
    StorageImageIndex output;
    // The sky-view LUT, for `sky_view`.
    uint view;
    // The march table, for `ms_gather` and `ms_fog`.
    TextureIndex input;
    // The multiple-scattering table, for `ms_scatter` and `ms_fog`.
    StorageImageIndex table;
    // The scattering order `ms_scatter` makes, the azimuth texels of the radiance it reads, and
    // the last order the table sums, past which a tail stands for the rest.
    uint order;
    uint azimuths;
    uint last;
    uint _pad;
}
ORR_ROOT(Root);

static const uint COLUMN_STEPS = 64;
static const uint GROUND_STEPS = 32;
static const uint SKY_VIEW_STEPS = 40;
// Steps of the sky-view march from the eye to `near_range`.
static const uint SKY_VIEW_NEAR_STEPS = 4;
// Lanes of the workgroups that sum over themselves (`workgroup_sum`).
static const uint WORKGROUP_LANES = 256;
// Directions per ground irradiance texel, a lane each.
static const uint DIRECTIONS = WORKGROUP_LANES;
// Candidates more than 18 deg below the focus horizon add less than the night background.
static const float FOLD_HORIZON = -0.309016994;
// Seconds a fold weight takes to go between 0 and 1.
static const float FOLD_FADE = 0.5;
// A folded light ranks as if this much brighter, times its weight, so near-equal candidates
// do not trade places every frame.
static const float FOLD_HOLD = 0.02;

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
    return medium_of(g, atmosphere_densities(a, h), fog ? atmosphere_fog_extinction(a, h) : 0.0);
}

// The same from the densities `d` (`atmosphere_densities`) and the fog's extinction `f`.
Medium medium_of(AtmosphereGroups g, float3 d, float f)
{
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
    float x = atmosphere_ms_light_unit(mu) * float(GROUND_SIZE - 1);
    uint i = min(uint(x), GROUND_SIZE - 2);
    float f = x - float(i);
    float3 e0 = a.ground_irradiance[i * a.group_passes + pass].xyz;
    float3 e1 = a.ground_irradiance[(i + 1) * a.group_passes + pass].xyz;
    return exp(lerp(e0, e1, f));
}

struct Lights
{
    float3 direction[MAX_FOLDED_LIGHTS];
    // Rec.2020 illuminance.
    float3 color[MAX_FOLDED_LIGHTS];
    uint count;
    // Each light is a unit spectral irradiance in every group in place of its color.
    bool unit;
}

// A light's spectral irradiance in group pass `g`'s three groups.
float3 light_groups(Lights lights, uint l, AtmosphereGroups g)
{
    return lights.unit ? float3(1.0) : atmosphere_light_groups(g, lights.color[l]);
}

// Radiance of the three groups of each of `passes` group passes from `first` arriving at
// distance t0 along unit `dir` from a LUT eye at altitude h0, from the medium over [t0, t1] and,
// with `ground`, the ground at t1. Multiple scattering, and the sky's light on the ground, are
// scaled by `multiple`. The fog scatters the lights once by its Draine part, and with `lobe` by
// its forward lobe too; with multiple scattering the light that lobe keeps on its way on both
// legs is included (`atmosphere_fog_lobe`), and reaches t0 through the extinction the lobe's
// scattering leaves out. What the passes share at a step is found once there. `lobe_depth` is the
// lobe's scattering depth over [t0, t1]. With `before`, the view started before t0, over a lobe
// depth of `lobe_before` up to it: `lobe_more` is what that adds at t0 to the lobe's light, which
// keeps on its way along the view from where it started, seen through the full extinction before
// t0.
void march(AtmosphereGpu* a, float h0, float3 dir, float t0, float t1, bool ground, uint steps,
           Lights lights, float multiple, bool lobe, uint first, uint passes, bool before,
           float3 lobe_before[MAX_GROUP_PASSES], out float3 radiance[MAX_GROUP_PASSES],
           out float3 transmittance[MAX_GROUP_PASSES], out float3 lobe_depth[MAX_GROUP_PASSES],
           out float3 lobe_more[MAX_GROUP_PASSES])
{
    float3 up = a.up.xyz;
    float mu = dot(dir, up);
    float r0 = a.planet_radius + h0;
    bool fog = a.fog_density > 0.0;
    float4 droplets = fog_phase_parameters(a.fog_diameter);
    // The share of the fog's scattering in its lobe, where the march sums the light the lobe
    // keeps on its way; single scattering takes the lobe as it is.
    float lobe_weight = lobe && fog && multiple > 0.0 ? 1.0 - droplets.w : 0.0;
    float3 phase[MAX_FOLDED_LIGHTS];
    for (uint l = 0; l < lights.count; ++l)
    {
        float c = dot(dir, lights.direction[l]);
        float f = droplets.w * draine_phase(droplets.y, droplets.z, c);
        if (lobe && !(multiple > 0.0))
            f += (1.0 - droplets.w) * hg_phase(droplets.x, c);
        phase[l] = float3(rayleigh_phase(c), cornette_shanks_phase(a.aerosol_asymmetry, c), fog ? f : 0.0);
    }
    // The lobe's largest share of the fog's extinction over the groups, and its largest depth
    // before t0.
    float lobe_share = 0.0;
    float lobe_ahead = 0.0;
    for (uint p = 0; p < passes; ++p)
    {
        radiance[p] = float3(0.0);
        transmittance[p] = float3(1.0);
        lobe_depth[p] = float3(0.0);
        lobe_more[p] = float3(0.0);
        lobe_share = max(lobe_share, max_component(a.groups[first + p].fog_albedo.xyz) * lobe_weight);
        if (before)
            lobe_ahead = max(lobe_ahead, max_component(lobe_before[p]));
    }
    // Where the lobe's in-scatter can't reach f32's rounding of the air's own, the march leaves
    // its sum out. Per unit fog extinction its in-scatter is at most the lobe's share times the
    // lobe's peak, times e^depth for the light the lobe keeps on its way over its depth `depth`
    // on both legs, which the march sees through the transmittance without the lobe; the air's
    // is at least its Rayleigh scattering times that phase function's least.
    float lobe_peak = lobe_share * hg_phase(droplets.x, 1.0);
    float rayleigh_floor = 0x1p-24 * min(min(a.rayleigh.x, a.rayleigh.y), a.rayleigh.z) * rayleigh_phase(0.0);
    // The fog's optical depth from t0.
    float fog_along = 0.0;
    for (uint i = 0; i < steps; ++i)
    {
        float2 s = step_at(t0, t1, i, steps);
        float h = atmosphere_altitude_along(a, h0, mu, s.x);
        float r = a.planet_radius + h;
        float3 densities = atmosphere_densities(a, h);
        float f = fog ? atmosphere_fog_extinction(a, h) : 0.0;
        float3 source[MAX_GROUP_PASSES];
        float3 lobe_source[MAX_GROUP_PASSES];
        float3 lobe_ahead_source[MAX_GROUP_PASSES];
        for (uint p = 0; p < passes; ++p)
        {
            source[p] = float3(0.0);
            lobe_source[p] = float3(0.0);
            lobe_ahead_source[p] = float3(0.0);
        }
        for (uint l = 0; l < lights.count; ++l)
        {
            float3 ld = lights.direction[l];
            float c = dot(dir, ld);
            float mu_l = (r0 * dot(up, ld) + s.x * c) / r;
            bool shadowed = atmosphere_ray_hits_ground(a, h, mu_l);
            float2 uv = atmosphere_transmittance_uv(a, h, mu_l);
            float fog_depth = fog && !shadowed ? atmosphere_fog_optical_depth_to_space(a, h, mu_l) : 0.0;
            float4 unit = multiple > 0.0 ? atmosphere_ms_unit(a, h, mu_l, (r0 * mu + s.x) / r, c) : float4(0.0);
            float depth = (fog_depth + fog_along + 0.5 * f * s.y) * lobe_share + lobe_ahead;
            bool lobe_here = lobe_share > 0.0 && log(f * lobe_peak) + depth > log(rayleigh_floor * densities.x);
            FogLobe terms = {};
            if (lobe_here)
                terms = atmosphere_fog_lobe_terms(droplets.x, c);
            for (uint p = 0; p < passes; ++p)
            {
                AtmosphereGroups g = a.groups[first + p];
                Medium m = medium_of(g, densities, f);
                float3 e = light_groups(lights, l, g);
                float3 clear = shadowed ? float3(0.0) : atmosphere_lut_transmittance_at(a, uv, first + p);
                float3 lit = clear * exp(-fog_depth) * (m.rayleigh * phase[l].x + m.aerosol * phase[l].y + m.fog * phase[l].z);
                if (multiple > 0.0)
                    lit += multiple * (m.rayleigh + m.aerosol + m.fog) * atmosphere_lut_multiple_scattering_at(a, unit, first + p);
                source[p] += e * lit;
                if (lobe_here)
                {
                    float3 share = g.fog_albedo.xyz * lobe_weight;
                    float3 sigma = m.fog * lobe_weight;
                    float3 kept = e * clear * exp(-fog_depth * (1.0 - share)) * sigma;
                    float3 depth_here = fog_depth * share + lobe_depth[p] + 0.5 * sigma * s.y;
                    lobe_source[p] += kept * atmosphere_fog_lobe_sum(terms, depth_here);
                    if (before)
                        lobe_ahead_source[p] += kept * atmosphere_fog_lobe_sum(terms, depth_here + lobe_before[p]);
                }
            }
        }
        for (uint p = 0; p < passes; ++p)
        {
            Medium m = medium_of(a.groups[first + p], densities, f);
            radiance[p] += transmittance[p] * source[p] * step_integral(m.extinction, s.y);
            // The lobe's light reaches t0 through the transmittance without the lobe's scattering,
            // the transmittance times e^lobe_depth, up to LOBE_VIEW_DEPTH.
            if (lobe_share > 0.0)
            {
                float3 sigma = m.fog * lobe_weight;
                float3 kept = exp(min(lobe_depth[p], LOBE_VIEW_DEPTH)) * lobe_source[p];
                float3 lobe_step = transmittance[p] * step_integral(m.extinction - sigma, s.y);
                radiance[p] += kept * lobe_step;
                if (before)
                    lobe_more[p] += lobe_step * (exp(min(lobe_depth[p] + lobe_before[p], LOBE_VIEW_DEPTH)) * lobe_ahead_source[p] - kept);
                lobe_depth[p] += sigma * s.y;
            }
            transmittance[p] *= exp(-m.extinction * s.y);
        }
        fog_along += f * s.y;
    }
    if (ground)
    {
        float3 n = (r0 * up + t1 * dir) / a.planet_radius;
        for (uint l = 0; l < lights.count; ++l)
        {
            float mu_l = dot(n, lights.direction[l]);
            for (uint p = 0; p < passes; ++p)
            {
                AtmosphereGroups g = a.groups[first + p];
                float3 e = atmosphere_light_transmittance(a, 0.0, mu_l, first + p) * max(mu_l, 0.0);
                if (multiple > 0.0)
                    e += multiple * ground_sky_irradiance(a, first + p, mu_l);
                radiance[p] += transmittance[p] * g.ground_albedo.xyz * INV_PI * light_groups(lights, l, g) * e;
            }
        }
    }
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
        rgb += atmosphere_rec2020(g, t);
    }
    lut[uint3(id.xy, a.group_passes)] = float4(rgb / a.solar_color.xyz, 1.0);
}

// The multiple-scattering build, every group pass at once. The radiance table (`root.radiance`)
// holds the radiance of one order arriving at a point along a ray, per unit spectral irradiance
// of the light, linear; the march table (`root.output`) the in-scatter of an order toward a view
// per unit scattering coefficient, encoded as `atmosphere_ms_table` reads it. The running sum of
// the orders' in-scatter is kept linear (`root.sum`) until the last order writes the table.
//
// The fog's forward Henyey-Greenstein lobe is all but a delta, which no table resolves: the build
// takes the light it scatters as light that passes on unscattered, in a medium whose fog
// extinguishes and scatters by its Draine part alone (`ms_medium`), and `ms_fog` puts the lobe
// back for the table's reader.

// Azimuthal cosine modes of the radiance the phase operator acts on: as many as the first
// order's azimuth texels.
[Host] static const uint MS_MODES = 16;
// Azimuth texels of the first order's radiance, and the steps of the marches of the first order
// and the later ones.
[Host] static const uint MS_FIRST_AZIMUTHS = 16;
static const uint MS_FIRST_STEPS = 40;
static const uint MS_LATER_STEPS = 24;
// The grid that builds the phase operator about each view texel: cells of the ray zenith
// coordinate, and of the azimuth about the zenith from the view over half a turn.
static const uint MS_KERNEL_ZENITHS = 512;
static const uint MS_KERNEL_AZIMUTHS = 128;
static const uint MS_KERNEL_LANES = 256;
static const uint MS_KERNEL_COMPONENTS = 3;
// The phase operator sums its cells in fixed point, so the sum is the same every build.
static const float MS_KERNEL_SCALE = 536870912.0;
// Steps of the zenith cosine over which the ground's weights of the ray texels are summed.
static const uint MS_GROUND_STEPS = 20480;
// `root.aux`: the ground's weights of the radiance table's ray texels, the per-slice sums of
// the next to last order's in-scatter, and the sum of the orders' sky irradiance on the ground.
static const uint MS_AUX_GROUND_WEIGHTS = 0;
static const uint MS_AUX_SLICES = MS_RAY_ZENITHS;
static const uint MS_AUX_GROUND_SUM = MS_AUX_SLICES + MS_ALTITUDES * MS_LIGHT_ZENITHS * MAX_GROUP_PASSES;

// The light and a direction at zenith cosine mu and azimuth phi from it, in a frame whose zenith
// is +z and whose light lies in the x-z plane.
float3 ms_light(float mu_light)
{
    return float3(sqrt(max(1.0 - mu_light * mu_light, 0.0)), 0.0, mu_light);
}

float3 ms_direction(float mu, float phi)
{
    float s = sqrt(max(1.0 - mu * mu, 0.0));
    return float3(s * cos(phi), s * sin(phi), mu);
}

// The share of the fog's extinction the build keeps: all but the light its forward lobe
// scatters.
float3 ms_fog_keep(AtmosphereGroups g, float4 droplets)
{
    return 1.0 - g.fog_albedo.xyz * (1.0 - droplets.w);
}

// The densities at altitude h the build's medium reads: Rayleigh, aerosol and ozone in xyz, the
// fog's extinction in w.
float4 ms_densities(AtmosphereGpu* a, float h)
{
    return float4(atmosphere_densities(a, h), a.fog_density > 0.0 ? atmosphere_fog_extinction(a, h) : 0.0);
}

// The build's medium of densities `d` (`ms_densities`): the fog scatters by its Draine part and
// extinguishes by what that part and absorption take.
Medium ms_medium(AtmosphereGroups g, float4 d, float4 droplets)
{
    Medium m;
    m.rayleigh = g.rayleigh.xyz * d.x;
    m.aerosol = g.aerosol_scattering.xyz * d.y;
    m.fog = g.fog_albedo.xyz * d.w * droplets.w;
    m.extinction = m.rayleigh + g.aerosol_extinction.xyz * d.y + g.ozone.xyz * d.z + d.w * ms_fog_keep(g, droplets);
    return m;
}

// The fog's optical depth to space from altitude h along zenith cosine mu, 0 where the ray meets
// the ground and its light is gone.
float ms_fog_depth(AtmosphereGpu* a, float h, float mu)
{
    return atmosphere_ray_hits_ground(a, h, mu) ? 0.0 : atmosphere_fog_optical_depth_to_space(a, h, mu);
}

// The build medium's light transmittance to altitude h along zenith cosine mu, where the fog's
// optical depth that way is `fog_depth` (`ms_fog_depth`).
float3 ms_light_transmittance(AtmosphereGpu* a, AtmosphereGroups g, float h, float mu, uint pass, float fog_depth, float4 droplets)
{
    return atmosphere_lut_transmittance(a, h, mu, pass) * exp(-fog_depth * ms_fog_keep(g, droplets));
}

// The ray texel pair and weight of the radiance table that hold zenith cosine mu at altitude h:
// a ray reads only texels on its own side of the horizon.
void ms_ray_pair(AtmosphereGpu* a, float h, float mu, out uint x0, out float tx)
{
    uint half = MS_RAY_ZENITHS / 2;
    float v = atmosphere_ms_ray_unit(a, h, mu);
    bool below = v >= 0.5;
    float x = v * float(MS_RAY_ZENITHS - 1) - (below ? float(half) : 0.0);
    x = clamp(x, 0.0, float(half - 1));
    uint i = min(uint(x), half - 2);
    x0 = (below ? half : 0) + i;
    tx = x - float(i);
}

// The running sum's entry of the table's texel `index` (altitude, light, view, azimuth) for group
// pass `pass`.
uint ms_sum_index(uint4 index, uint pass)
{
    return (((index.x * MS_LIGHT_ZENITHS + index.y) * MAX_GROUP_PASSES + pass) * MS_AZIMUTHS + index.w) * MS_VIEW_ZENITHS + index.z;
}

uint ms_radiance_index(uint altitude, uint light, uint pass, uint ray, uint azimuth)
{
    return (((altitude * MS_LIGHT_ZENITHS + light) * MAX_GROUP_PASSES + pass) * MS_RAY_ZENITHS + ray) * MS_AZIMUTHS + azimuth;
}

// The altitude, light and ray zenith cosines and azimuth of radiance texel `id` among
// `azimuths`: x is azimuth * MS_RAY_ZENITHS + ray zenith, y the light, z the altitude.
void ms_ray_texel(AtmosphereGpu* a, uint3 id, uint azimuths, out float h, out float mu_light, out float mu, out float phi)
{
    h = atmosphere_ms_altitude(a, float(id.z) / float(MS_ALTITUDES - 1));
    mu_light = atmosphere_ms_light(float(id.y) / float(MS_LIGHT_ZENITHS - 1));
    mu = atmosphere_ms_ray(a, h, float(id.x % MS_RAY_ZENITHS) / float(MS_RAY_ZENITHS - 1));
    phi = PI * float(id.x / MS_RAY_ZENITHS) / float(azimuths - 1);
}

// Writes `j` per group pass to the table's texel `index`, encoded, and the Rec.2020 layer.
void ms_write_table(AtmosphereGpu* a, uint4 index, float3 j[MAX_GROUP_PASSES])
{
    RWTexture3D<float4> table = root.table.get<RWTexture3D<float4>>();
    float3 rgb = float3(0.0);
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        table[atmosphere_ms_texel(index, MS_VIEW_ZENITHS, p)] = float4(atmosphere_ms_encode(j[p]), 0.0);
        rgb += atmosphere_rec2020(a.groups[p], j[p]);
    }
    table[atmosphere_ms_texel(index, MS_VIEW_ZENITHS, MAX_GROUP_PASSES)] = float4(atmosphere_ms_encode(rgb / a.solar_color.xyz), 0.0);
}

static groupshared int ms_kernel_sum[MS_KERNEL_COMPONENTS * MS_MODES * MS_RAY_ZENITHS];
static groupshared int ms_ground_sum[MS_RAY_ZENITHS];

// The phase operator of each scattering component at one view texel (group x) and altitude
// (group y): for azimuthal cosine mode m of the radiance arriving along the ray texels, what it
// scatters toward the view, `root.kernel[((altitude * MS_VIEW_ZENITHS + view) * MS_MODES + m) *
// MS_RAY_ZENITHS + ray]` with Rayleigh, aerosol and the fog's Draine part in xyz. A lane takes
// cells of the ray zenith coordinate, each a band of directions whose solid angle follows from
// the coordinate's slope, and sums each phase function over the band's azimuths from the view in
// every mode; the band's sums go to the ray texels that hold its zenith. The first group also
// writes the ground's weights of the ray texels for the sky irradiance there.
[shader("compute")]
[numthreads(MS_KERNEL_LANES, 1, 1)]
void ms_kernel(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    AtmosphereGpu* a = root.atmosphere;
    for (uint i = lane; i < MS_KERNEL_COMPONENTS * MS_MODES * MS_RAY_ZENITHS; i += MS_KERNEL_LANES)
        ms_kernel_sum[i] = 0;
    if (lane < MS_RAY_ZENITHS)
        ms_ground_sum[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    float h = atmosphere_ms_altitude(a, float(group.y) / float(MS_ALTITUDES - 1));
    float mu_view = atmosphere_ms_view(a, h, float(group.x) / float(MS_VIEW_ZENITHS - 1));
    float sin_view = sqrt(max(1.0 - mu_view * mu_view, 0.0));
    float g = a.aerosol_asymmetry;
    float4 droplets = fog_phase_parameters(a.fog_diameter);
    for (uint k = lane; k < MS_KERNEL_ZENITHS; k += MS_KERNEL_LANES)
    {
        float2 theta = atmosphere_ms_ray_angle(a, h, (float(k) + 0.5) / float(MS_KERNEL_ZENITHS));
        float mu = cos(theta.x);
        float s = sin(theta.x);
        // The band's solid angle per azimuth cell, both sides of the view's plane.
        float band = s * theta.y / float(MS_KERNEL_ZENITHS) * (2.0 * PI / float(MS_KERNEL_AZIMUTHS));
        float sums[MS_KERNEL_COMPONENTS][MS_MODES];
        for (uint c = 0; c < MS_KERNEL_COMPONENTS; ++c)
            for (uint m = 0; m < MS_MODES; ++m)
                sums[c][m] = 0.0;
        for (uint j = 0; j < MS_KERNEL_AZIMUTHS; ++j)
        {
            float phi = (float(j) + 0.5) * PI / float(MS_KERNEL_AZIMUTHS);
            float c1 = cos(phi);
            float cosine = mu_view * mu + sin_view * s * c1;
            float3 phase = float3(rayleigh_phase(cosine), cornette_shanks_phase(g, cosine),
                                  draine_phase(droplets.y, droplets.z, cosine));
            // cos(m phi) by its recurrence.
            float previous = c1;
            float current = 1.0;
            for (uint m = 0; m < MS_MODES; ++m)
            {
                sums[0][m] += phase.x * current;
                sums[1][m] += phase.y * current;
                sums[2][m] += phase.z * current;
                float next = 2.0 * c1 * current - previous;
                previous = current;
                current = next;
            }
        }
        uint x0;
        float tx;
        ms_ray_pair(a, h, mu, x0, tx);
        for (uint c = 0; c < MS_KERNEL_COMPONENTS; ++c)
        {
            for (uint m = 0; m < MS_MODES; ++m)
            {
                float w = sums[c][m] * band * MS_KERNEL_SCALE;
                uint i = (c * MS_MODES + m) * MS_RAY_ZENITHS + x0;
                InterlockedAdd(ms_kernel_sum[i], int(round(w * (1.0 - tx))));
                InterlockedAdd(ms_kernel_sum[i + 1], int(round(w * tx)));
            }
        }
    }
    // The ground's sky irradiance is the m = 0 mode times 2 pi mu over the upper hemisphere.
    if (all(group.xy == 0))
    {
        for (uint s = lane; s < MS_GROUND_STEPS; s += MS_KERNEL_LANES)
        {
            float mu = (float(s) + 0.5) / float(MS_GROUND_STEPS);
            uint x0;
            float tx;
            ms_ray_pair(a, 0.0, mu, x0, tx);
            float w = TWO_PI * mu / float(MS_GROUND_STEPS);
            InterlockedAdd(ms_ground_sum[x0], int(round(w * (1.0 - tx) * MS_KERNEL_SCALE)));
            InterlockedAdd(ms_ground_sum[x0 + 1], int(round(w * tx * MS_KERNEL_SCALE)));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = lane; i < MS_MODES * MS_RAY_ZENITHS; i += MS_KERNEL_LANES)
    {
        uint stride = MS_MODES * MS_RAY_ZENITHS;
        float3 k = float3(float(ms_kernel_sum[i]), float(ms_kernel_sum[stride + i]), float(ms_kernel_sum[2 * stride + i]));
        root.kernel[(group.y * MS_VIEW_ZENITHS + group.x) * stride + i] = float4(k / MS_KERNEL_SCALE, 0.0);
    }
    if (all(group.xy == 0) && lane < MS_RAY_ZENITHS)
        root.aux[MS_AUX_GROUND_WEIGHTS + lane] = float4(float(ms_ground_sum[lane]) / MS_KERNEL_SCALE, 0.0, 0.0, 0.0);
}

bool ms_ray_outside(uint3 id, uint azimuths)
{
    return any(id >= uint3(MS_RAY_ZENITHS * azimuths, MS_LIGHT_ZENITHS, MS_ALTITUDES));
}

// The first order: single scattering along each ray and the ground's reflection of the direct
// light where the ray meets the ground, at MS_FIRST_AZIMUTHS.
[shader("compute")]
[numthreads(8, 8, 1)]
void ms_single(uint3 id: SV_DispatchThreadID)
{
    if (ms_ray_outside(id, MS_FIRST_AZIMUTHS))
        return;
    AtmosphereGpu* a = root.atmosphere;
    float h, mu_light, mu, phi;
    ms_ray_texel(a, id, MS_FIRST_AZIMUTHS, h, mu_light, mu, phi);
    float nu = dot(ms_direction(mu, phi), ms_light(mu_light));
    float4 droplets = fog_phase_parameters(a.fog_diameter);
    float3 phase = float3(rayleigh_phase(nu), cornette_shanks_phase(a.aerosol_asymmetry, nu),
                          draine_phase(droplets.y, droplets.z, nu));
    bool ground = atmosphere_ray_hits_ground(a, h, mu);
    float end = ground ? atmosphere_distance_to_ground(a, h, mu) : atmosphere_distance_to_top(a, h, mu);
    float r0 = a.planet_radius + h;
    float3 radiance[MAX_GROUP_PASSES];
    float3 transmittance[MAX_GROUP_PASSES];
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        radiance[p] = float3(0.0);
        transmittance[p] = float3(1.0);
    }
    for (uint i = 0; i < MS_FIRST_STEPS; ++i)
    {
        float2 s = step_at(0.0, end, i, MS_FIRST_STEPS);
        float hs = atmosphere_altitude_along(a, h, mu, s.x);
        float mu_ls = (r0 * mu_light + s.x * nu) / (a.planet_radius + hs);
        float4 d = ms_densities(a, hs);
        float depth = ms_fog_depth(a, hs, mu_ls);
        for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
        {
            AtmosphereGroups g = a.groups[p];
            Medium m = ms_medium(g, d, droplets);
            float3 lit = ms_light_transmittance(a, g, hs, mu_ls, p, depth, droplets)
                * (m.rayleigh * phase.x + m.aerosol * phase.y + m.fog * phase.z);
            radiance[p] += transmittance[p] * lit * step_integral(m.extinction, s.y);
            transmittance[p] *= exp(-m.extinction * s.y);
        }
    }
    float mu_ground = (r0 * mu_light + end * nu) / a.planet_radius;
    float ground_depth = ms_fog_depth(a, 0.0, mu_ground);
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        AtmosphereGroups g = a.groups[p];
        if (ground)
            radiance[p] += transmittance[p] * g.ground_albedo.xyz * INV_PI
                * ms_light_transmittance(a, g, 0.0, mu_ground, p, ground_depth, droplets) * max(mu_ground, 0.0);
        root.radiance[ms_radiance_index(id.z, id.y, p, id.x % MS_RAY_ZENITHS, id.x / MS_RAY_ZENITHS)] = float4(radiance[p], 0.0);
    }
}

// The (view, mode) items and the view and azimuth texels each lane of `ms_scatter` takes; the
// in-scatter's modes reuse `ms_hat`, which holds the radiance's for at least as many ray texels.
static const uint MS_MODE_SPAN = (MS_VIEW_ZENITHS * MS_MODES + WORKGROUP_LANES - 1) / WORKGROUP_LANES;
static const uint MS_TEXEL_SPAN = (MS_VIEW_ZENITHS * MS_AZIMUTHS + WORKGROUP_LANES - 1) / WORKGROUP_LANES;

// The radiance's cosine modes, then the in-scatter's, three floats each: Metal gives a float3
// 16 bytes, which would put the kernel past its 32 KB of threadgroup memory.
static groupshared float ms_hat[3 * MS_RAY_ZENITHS * MS_MODES * MAX_GROUP_PASSES];

float3 ms_hat_at(uint i)
{
    return float3(ms_hat[3 * i], ms_hat[3 * i + 1], ms_hat[3 * i + 2]);
}

void ms_hat_set(uint i, float3 v)
{
    ms_hat[3 * i] = v.x;
    ms_hat[3 * i + 1] = v.y;
    ms_hat[3 * i + 2] = v.z;
}
static groupshared float ms_cos_in[MS_MODES * MS_AZIMUTHS];
static groupshared float ms_cos_out[MS_MODES * MS_AZIMUTHS];
static groupshared float3 ms_waves[WORKGROUP_LANES / 8 * MAX_GROUP_PASSES];

// The sum over a workgroup of WORKGROUP_LANES of each lane's `v`, in every lane; `slot` keeps
// sums of up to MAX_GROUP_PASSES values apart.
float3 workgroup_sum(float3 v, uint lane, uint slot)
{
    float3 wave = WaveActiveSum(v);
    uint waves = WORKGROUP_LANES / WaveGetLaneCount();
    if (WaveIsFirstLane())
        ms_waves[lane / WaveGetLaneCount() * MAX_GROUP_PASSES + slot] = wave;
    GroupMemoryBarrierWithGroupSync();
    float3 sum = float3(0.0);
    for (uint w = 0; w < waves; ++w)
        sum += ms_waves[w * MAX_GROUP_PASSES + slot];
    GroupMemoryBarrierWithGroupSync();
    return sum;
}

// The in-scatter of order `root.order` from the last order's radiance at `root.azimuths`: per
// group pass, the radiance's azimuthal cosine modes along each ray texel, each mode's phase
// operator (`ms_kernel`) by the components' shares of the build medium's scattering, and the
// modes summed back at the table's azimuths. A workgroup holds one light (group x) and altitude
// (group y). Before the last order it writes the order's logs for the next march and adds it to
// the sum; the last adds itself and the tail of the orders past it, a geometric series in the
// ratio of the last two orders' sums over the slice where that is below 1, and writes the table
// (or, with fog, the logs `ms_fog` finishes the table from). At the ground it also writes the
// last order's sky irradiance there, and its sum over the orders.
[shader("compute")]
[numthreads(WORKGROUP_LANES, 1, 1)]
void ms_scatter(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    AtmosphereGpu* a = root.atmosphere;
    uint light = group.x;
    uint altitude = group.y;
    uint azimuths = root.azimuths;
    {
        uint m = lane / MS_AZIMUTHS;
        uint k = lane % MS_AZIMUTHS;
        ms_cos_in[lane] = cos(PI * float(m * k) / float(azimuths - 1));
        ms_cos_out[lane] = cos(PI * float(m * k) / float(MS_AZIMUTHS - 1));
    }
    GroupMemoryBarrierWithGroupSync();

    // Cosine modes of the radiance over its azimuths, which run from 0 to pi.
    for (uint item = lane; item < MS_RAY_ZENITHS * MS_MODES; item += WORKGROUP_LANES)
    {
        uint ray = item / MS_MODES;
        uint m = item % MS_MODES;
        float3 c[MAX_GROUP_PASSES];
        for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
            c[p] = float3(0.0);
        if (m < azimuths)
        {
            for (uint k = 0; k < azimuths; ++k)
            {
                float w = (k == 0 || k == azimuths - 1 ? 0.5 : 1.0) * ms_cos_in[m * MS_AZIMUTHS + k];
                for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
                    c[p] += w * root.radiance[ms_radiance_index(altitude, light, p, ray, k)].rgb;
            }
        }
        float scale = (m == 0 || m == azimuths - 1 ? 1.0 : 2.0) / float(azimuths - 1);
        for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
            ms_hat_set(item * MAX_GROUP_PASSES + p, c[p] * scale);
    }
    GroupMemoryBarrierWithGroupSync();

    float h = atmosphere_ms_altitude(a, float(altitude) / float(MS_ALTITUDES - 1));
    float4 droplets = fog_phase_parameters(a.fog_diameter);
    float3 jm[MS_MODE_SPAN][MAX_GROUP_PASSES];
    for (uint i = 0; i < MS_MODE_SPAN; ++i)
    {
        for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
            jm[i][p] = float3(0.0);
        uint item = lane + i * WORKGROUP_LANES;
        uint view = item / MS_MODES;
        uint m = item % MS_MODES;
        if (view < MS_VIEW_ZENITHS && m < azimuths)
        {
            float3 rayleigh[MAX_GROUP_PASSES];
            float3 aerosol[MAX_GROUP_PASSES];
            float3 fog[MAX_GROUP_PASSES];
            for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
            {
                rayleigh[p] = float3(0.0);
                aerosol[p] = float3(0.0);
                fog[p] = float3(0.0);
            }
            uint row = ((altitude * MS_VIEW_ZENITHS + view) * MS_MODES + m) * MS_RAY_ZENITHS;
            for (uint ray = 0; ray < MS_RAY_ZENITHS; ++ray)
            {
                float4 k = root.kernel[row + ray];
                for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
                {
                    float3 l = ms_hat_at((ray * MS_MODES + m) * MAX_GROUP_PASSES + p);
                    rayleigh[p] += k.x * l;
                    aerosol[p] += k.y * l;
                    fog[p] += k.z * l;
                }
            }
            float4 d = ms_densities(a, h);
            for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
            {
                Medium md = ms_medium(a.groups[p], d, droplets);
                float3 scattering = md.rayleigh + md.aerosol + md.fog;
                float3 j = (md.rayleigh * rayleigh[p] + md.aerosol * aerosol[p] + md.fog * fog[p]) / scattering;
                jm[i][p] = select(scattering > 0.0, j, float3(0.0));
            }
        }
    }

    // The ground's sky irradiance by the last order.
    if (altitude == 0 && lane < MAX_GROUP_PASSES)
    {
        float3 e = float3(0.0);
        for (uint ray = 0; ray < MS_RAY_ZENITHS; ++ray)
            e += root.aux[MS_AUX_GROUND_WEIGHTS + ray].x * ms_hat_at((ray * MS_MODES) * MAX_GROUP_PASSES + lane);
        uint at = light * MAX_GROUP_PASSES + lane;
        root.ground[at] = float4(e, 0.0);
        float4* sum = root.aux + MS_AUX_GROUND_SUM;
        sum[at] = float4(root.order == 2 ? e : sum[at].xyz + e, 0.0);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0; i < MS_MODE_SPAN; ++i)
        if (lane + i * WORKGROUP_LANES < MS_VIEW_ZENITHS * MS_MODES)
            for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
                ms_hat_set((lane + i * WORKGROUP_LANES) * MAX_GROUP_PASSES + p, jm[i][p]);
    GroupMemoryBarrierWithGroupSync();

    // The in-scatter at the table's texels, a few per lane.
    uint4 index[MS_TEXEL_SPAN];
    bool held[MS_TEXEL_SPAN];
    float3 jt[MS_TEXEL_SPAN][MAX_GROUP_PASSES];
    for (uint i = 0; i < MS_TEXEL_SPAN; ++i)
    {
        uint texel = lane + i * WORKGROUP_LANES;
        held[i] = texel < MS_VIEW_ZENITHS * MS_AZIMUTHS;
        uint view = texel % MS_VIEW_ZENITHS;
        uint azimuth = min(texel / MS_VIEW_ZENITHS, MS_AZIMUTHS - 1);
        index[i] = uint4(altitude, light, view, azimuth);
        for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
        {
            float3 j = float3(0.0);
            for (uint m = 0; m < MS_MODES; ++m)
                j += ms_hat_at((view * MS_MODES + m) * MAX_GROUP_PASSES + p) * ms_cos_out[m * MS_AZIMUTHS + azimuth];
            // A cut series rings below 0 where the radiance falls fast.
            jt[i][p] = held[i] ? max(j, 0.0) : float3(0.0);
        }
    }
    RWTexture3D<float4> march = root.output.get<RWTexture3D<float4>>();
    float3 total[MS_TEXEL_SPAN][MAX_GROUP_PASSES];
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        float3 lane_sum = float3(0.0);
        for (uint i = 0; i < MS_TEXEL_SPAN; ++i)
            lane_sum += jt[i][p];
        uint slice = (altitude * MS_LIGHT_ZENITHS + light) * MAX_GROUP_PASSES + p;
        if (root.order == root.last - 1)
        {
            float3 s = workgroup_sum(lane_sum, lane, p);
            if (lane == 0)
                root.aux[MS_AUX_SLICES + slice] = float4(s, 0.0);
        }
        float3 ratio = float3(0.0);
        if (root.order == root.last)
        {
            float3 last = workgroup_sum(lane_sum, lane, p);
            float3 before = root.aux[MS_AUX_SLICES + slice].xyz;
            ratio = select(before > 0.0, last / before, float3(0.0));
        }
        for (uint i = 0; i < MS_TEXEL_SPAN; ++i)
        {
            if (!held[i])
                continue;
            uint3 texel = atmosphere_ms_texel(index[i], MS_VIEW_ZENITHS, p);
            uint at = ms_sum_index(index[i], p);
            float3 sum = root.order == 2 ? float3(0.0) : root.sum[at].xyz;
            if (root.order < root.last)
            {
                march[texel] = float4(atmosphere_ms_encode(jt[i][p]), 0.0);
                root.sum[at] = float4(sum + jt[i][p], 0.0);
                continue;
            }
            total[i][p] = sum + jt[i][p] * (1.0 + select(ratio > 0.0 && ratio < 1.0, ratio / (1.0 - ratio), float3(0.0)));
            if (a.fog_density > 0.0)
                march[texel] = float4(atmosphere_ms_encode(total[i][p]), 0.0);
        }
    }
    if (root.order == root.last && a.fog_density <= 0.0)
        for (uint i = 0; i < MS_TEXEL_SPAN; ++i)
            if (held[i])
                ms_write_table(a, index[i], total[i]);
}

// The sky irradiance on the ground under a light at zenith cosine mu_light from `sky`, per
// light texel and group pass.
float3 ms_ground_at(float4* sky, float mu_light, uint pass)
{
    float x = atmosphere_ms_light_unit(mu_light) * float(MS_LIGHT_ZENITHS - 1);
    uint i = min(uint(x), MS_LIGHT_ZENITHS - 2);
    return lerp(sky[i * MAX_GROUP_PASSES + pass].xyz, sky[(i + 1) * MAX_GROUP_PASSES + pass].xyz, x - float(i));
}

// The order's radiance at `root.azimuths` from its in-scatter (`root.input`): the in-scatter
// marched along each ray, and where the ray meets the ground, the ground's reflection of the last
// order's sky irradiance.
[shader("compute")]
[numthreads(8, 8, 1)]
void ms_gather(uint3 id: SV_DispatchThreadID)
{
    if (ms_ray_outside(id, root.azimuths))
        return;
    AtmosphereGpu* a = root.atmosphere;
    Texture3D<float4> input = root.input.get<Texture3D<float4>>();
    float h, mu_light, mu, phi;
    ms_ray_texel(a, id, root.azimuths, h, mu_light, mu, phi);
    float nu = dot(ms_direction(mu, phi), ms_light(mu_light));
    float4 droplets = fog_phase_parameters(a.fog_diameter);
    bool ground = atmosphere_ray_hits_ground(a, h, mu);
    float end = ground ? atmosphere_distance_to_ground(a, h, mu) : atmosphere_distance_to_top(a, h, mu);
    float r0 = a.planet_radius + h;
    float3 radiance[MAX_GROUP_PASSES];
    float3 transmittance[MAX_GROUP_PASSES];
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        radiance[p] = float3(0.0);
        transmittance[p] = float3(1.0);
    }
    for (uint i = 0; i < MS_LATER_STEPS; ++i)
    {
        float2 s = step_at(0.0, end, i, MS_LATER_STEPS);
        float hs = atmosphere_altitude_along(a, h, mu, s.x);
        float r = a.planet_radius + hs;
        float4 unit = atmosphere_ms_unit(a, hs, (r0 * mu_light + s.x * nu) / r, (r0 * mu + s.x) / r, nu);
        float4 d = ms_densities(a, hs);
        for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
        {
            Medium m = ms_medium(a.groups[p], d, droplets);
            float3 j = atmosphere_ms_table(input, a.sampler.get(), unit, MS_VIEW_ZENITHS, p, MAX_GROUP_PASSES);
            radiance[p] += transmittance[p] * (m.rayleigh + m.aerosol + m.fog) * j * step_integral(m.extinction, s.y);
            transmittance[p] *= exp(-m.extinction * s.y);
        }
    }
    float mu_ground = (r0 * mu_light + end * nu) / a.planet_radius;
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        if (ground)
            radiance[p] += transmittance[p] * a.groups[p].ground_albedo.xyz * INV_PI * ms_ground_at(root.ground, mu_ground, p);
        root.radiance[ms_radiance_index(id.z, id.y, p, id.x % MS_RAY_ZENITHS, id.x / MS_RAY_ZENITHS)] = float4(radiance[p], 0.0);
    }
}

// With fog, the table from the build medium's sum of orders (`root.input`): what the build left
// out, put back per unit of the medium's own scattering. The fog's forward lobe scatters the
// build medium's radiance of every order arriving along the view (its single scattering, the
// march of the sum and the ground's light) on toward it, and the other components the light the
// build took as unscattered, its transmittance less the medium's.
[shader("compute")]
[numthreads(8, 8, 1)]
void ms_fog(uint3 id: SV_DispatchThreadID)
{
    if (any(id >= uint3(MS_VIEW_ZENITHS * MS_AZIMUTHS, MS_LIGHT_ZENITHS, MS_ALTITUDES)))
        return;
    AtmosphereGpu* a = root.atmosphere;
    Texture3D<float4> input = root.input.get<Texture3D<float4>>();
    uint4 index = uint4(id.z, id.y, id.x % MS_VIEW_ZENITHS, id.x / MS_VIEW_ZENITHS);
    float h = atmosphere_ms_altitude(a, float(index.x) / float(MS_ALTITUDES - 1));
    float mu_light = atmosphere_ms_light(float(index.y) / float(MS_LIGHT_ZENITHS - 1));
    float mu = atmosphere_ms_view(a, h, float(index.z) / float(MS_VIEW_ZENITHS - 1));
    float nu = dot(ms_direction(mu, PI * float(index.w) / float(MS_AZIMUTHS - 1)), ms_light(mu_light));
    float4 droplets = fog_phase_parameters(a.fog_diameter);
    float3 phase = float3(rayleigh_phase(nu), cornette_shanks_phase(a.aerosol_asymmetry, nu),
                          draine_phase(droplets.y, droplets.z, nu));
    bool ground = atmosphere_ray_hits_ground(a, h, mu);
    float end = ground ? atmosphere_distance_to_ground(a, h, mu) : atmosphere_distance_to_top(a, h, mu);
    float r0 = a.planet_radius + h;
    float3 along[MAX_GROUP_PASSES];
    float3 transmittance[MAX_GROUP_PASSES];
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        along[p] = float3(0.0);
        transmittance[p] = float3(1.0);
    }
    for (uint i = 0; i < MS_FIRST_STEPS; ++i)
    {
        float2 s = step_at(0.0, end, i, MS_FIRST_STEPS);
        float hs = atmosphere_altitude_along(a, h, mu, s.x);
        float r = a.planet_radius + hs;
        float mu_ls = (r0 * mu_light + s.x * nu) / r;
        float4 unit = atmosphere_ms_unit(a, hs, mu_ls, (r0 * mu + s.x) / r, nu);
        float4 d = ms_densities(a, hs);
        float depth = ms_fog_depth(a, hs, mu_ls);
        for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
        {
            AtmosphereGroups g = a.groups[p];
            Medium m = ms_medium(g, d, droplets);
            float3 j = atmosphere_ms_table(input, a.sampler.get(), unit, MS_VIEW_ZENITHS, p, MAX_GROUP_PASSES);
            float3 lit = ms_light_transmittance(a, g, hs, mu_ls, p, depth, droplets)
                    * (m.rayleigh * phase.x + m.aerosol * phase.y + m.fog * phase.z)
                + (m.rayleigh + m.aerosol + m.fog) * j;
            along[p] += transmittance[p] * lit * step_integral(m.extinction, s.y);
            transmittance[p] *= exp(-m.extinction * s.y);
        }
    }
    float mu_ground = (r0 * mu_light + end * nu) / a.planet_radius;
    float ground_depth = ms_fog_depth(a, 0.0, mu_ground);
    float light_depth = ms_fog_depth(a, h, mu_light);
    float3 total[MAX_GROUP_PASSES];
    for (uint p = 0; p < MAX_GROUP_PASSES; ++p)
    {
        AtmosphereGroups g = a.groups[p];
        if (ground)
            along[p] += transmittance[p] * g.ground_albedo.xyz * INV_PI
                * (ms_light_transmittance(a, g, 0.0, mu_ground, p, ground_depth, droplets) * max(mu_ground, 0.0)
                   + ms_ground_at(root.aux + MS_AUX_GROUND_SUM, mu_ground, p));
        Medium m = medium(a, g, h, true);
        float3 scattering = m.rayleigh + m.aerosol + m.fog;
        float3 lobe = m.fog * (1.0 - droplets.w);
        float3 broad = m.rayleigh * phase.x + m.aerosol * phase.y + m.fog * droplets.w * phase.z;
        float3 beam = atmosphere_lut_transmittance(a, h, mu_light, p)
            * (exp(-light_depth * ms_fog_keep(g, droplets)) - exp(-light_depth));
        float3 built = atmosphere_ms_decode(input.Load(int4(atmosphere_ms_texel(index, MS_VIEW_ZENITHS, p), 0)).rgb);
        float3 j = ((scattering - lobe) * built + lobe * along[p] + broad * beam) / scattering;
        total[p] = select(scattering > 0.0, j, float3(0.0));
    }
    ms_write_table(a, index, total);
}

// Sky irradiance on the ground per unit spectral irradiance of a light, over the light's
// zenith cosine (group x) for group pass group y: the sky radiance (single scattering with every
// phase function, multiple scattering from the LUT, fog included) integrated over the upper
// hemisphere, a direction per lane. The fog's forward lobe keeps the light it scatters within a
// degree or so of the light, which no lane's direction resolves: that light reaches the ground
// as the light's own beam does, the light the fog's Draine part and absorption leave less the
// light all of the fog leaves.
[shader("compute")]
[numthreads(DIRECTIONS, 1, 1)]
void ground_irradiance(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    AtmosphereGpu* a = root.atmosphere;
    float mu_light = atmosphere_ms_light(float(group.x) / float(GROUND_SIZE - 1));
    float3 up = a.up.xyz;
    float3 x, y;
    orthonormal_basis(up, x, y);
    Lights lights;
    lights.count = 1;
    lights.unit = true;
    lights.direction[0] = sqrt(max(1.0 - mu_light * mu_light, 0.0)) * x + mu_light * up;
    float3 dir = fibonacci(lane, true, x, y, up);
    float3 none[MAX_GROUP_PASSES] = {};
    float3 l[MAX_GROUP_PASSES];
    float3 passed[MAX_GROUP_PASSES];
    float3 lobe_depth[MAX_GROUP_PASSES];
    float3 lobe_more[MAX_GROUP_PASSES];
    march(a, 0.0, dir, 0.0, atmosphere_distance_to_top(a, 0.0, dot(dir, up)), false, GROUND_STEPS, lights,
          1.0, false, group.y, 1, false, none, l, passed, lobe_depth, lobe_more);
    float3 e = workgroup_sum(l[0] * dot(dir, up), lane, 0) * (TWO_PI / float(DIRECTIONS));
    float depth = ms_fog_depth(a, 0.0, mu_light);
    e += atmosphere_lut_transmittance(a, 0.0, mu_light, group.y) * max(mu_light, 0.0)
        * (exp(-depth * ms_fog_keep(a.groups[group.y], fog_phase_parameters(a.fog_diameter))) - exp(-depth));
    if (lane == 0)
        a.ground_irradiance[group.x * a.group_passes + group.y] = float4(max(log(e), MS_LOG_LOW), 0.0);
}

// Fold membership and weights. Members are the `sky_lights` brightest eligible candidates,
// ranked by luminance times 1 + FOLD_HOLD x the candidate's weight, ties to the earlier one;
// a candidate is eligible when it names a directional light with some illuminance no more
// than 18 deg below the focus horizon. Each candidate's weight, kept per light across
// frames, moves toward 1 for a member and 0 otherwise by `delta_time / FOLD_FADE`, or jumps
// there when `fold_reset` is set. Members come first in `folded`, then the rest by weight; a
// light that does not fit in `MAX_FOLDED_LIGHTS` drops to 0. Writes each candidate's
// illuminance.
[shader("compute")]
[numthreads(1, 1, 1)]
void fold(uint3 id: SV_DispatchThreadID)
{
    AtmosphereGpu* a = root.atmosphere;
    SceneRoot* scene = atmosphere_scene(a);
    FoldGpu* f = a.fold;
    bool reset = a.fold_reset != 0;
    float luminance[MAX_SKY_LIGHTS];
    float held[MAX_SKY_LIGHTS];
    float rank[MAX_SKY_LIGHTS];
    bool lit[MAX_SKY_LIGHTS];
    for (uint i = 0; i < MAX_SKY_LIGHTS; ++i)
    {
        luminance[i] = 0.0;
        held[i] = 0.0;
        rank[i] = 0.0;
        lit[i] = false;
        f.illuminance[i] = float4(0.0);
        uint index = a.candidates[i];
        if (i >= a.candidate_count || index >= scene.light_capacity)
            continue;
        LightGpu l = scene.lights[index];
        if (l.kind != LIGHT_DIRECTIONAL)
            continue;
        float3 e = l.disc == DISC_TEXTURED && all(l.color == 0.0)
            ? textured_disc_integral(scene, index) : l.color;
        f.illuminance[i] = float4(e, 0.0);
        float y = luminance_rec2020(e);
        lit[i] = y > 0.0;
        if (lit[i] && dot(normalize(l.direction), a.up.xyz) >= FOLD_HORIZON)
            luminance[i] = y;
        for (uint j = 0; j < MAX_SKY_LIGHTS && !reset; ++j)
            if (f.light[j] == index)
                held[i] = max(held[i], f.weight[j]);
        rank[i] = luminance[i] * (1.0 + FOLD_HOLD * held[i]);
    }
    float step = max(scene.delta_time, 0.0) / FOLD_FADE;
    float weight[MAX_SKY_LIGHTS];
    bool member[MAX_SKY_LIGHTS];
    for (uint i = 0; i < MAX_SKY_LIGHTS; ++i)
    {
        uint brighter = 0;
        for (uint j = 0; j < MAX_SKY_LIGHTS; ++j)
            if (rank[j] > rank[i] || (rank[j] == rank[i] && j < i))
                ++brighter;
        member[i] = luminance[i] > 0.0 && brighter < a.sky_lights;
        float target = member[i] ? 1.0 : 0.0;
        float w = reset ? target : clamp(held[i] + (member[i] ? step : -step), 0.0, 1.0);
        weight[i] = lit[i] ? w : 0.0;
    }
    uint taken = 0;
    uint count = 0;
    for (; count < MAX_FOLDED_LIGHTS; ++count)
    {
        int best = -1;
        for (uint i = 0; i < MAX_SKY_LIGHTS; ++i)
            if ((taken & (1u << i)) == 0 && weight[i] > 0.0
                && (best < 0 || (member[i] && !member[best])
                    || (member[i] == member[best] && weight[i] > weight[best])))
                best = int(i);
        if (best < 0)
            break;
        taken |= 1u << uint(best);
        f.folded[count] = uint(best);
    }
    f.folded_count = count;
    for (uint i = 0; i < MAX_SKY_LIGHTS; ++i)
    {
        f.weight[i] = (taken & (1u << i)) != 0 ? weight[i] : 0.0;
        f.light[i] = i < a.candidate_count ? a.candidates[i] : ~0u;
    }
}

// One sky-view LUT: from `near_range` along each texel's direction, the folded lights'
// in-scatter, and the virtual ground below the horizon (SKY_VIEW_FAR); the same seen through the
// medium from the eye to `near_range` (SKY_VIEW_EYE); and what of it the fog's lobe within
// `near_range` keeps on its way (SKY_VIEW_KEPT).
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
    Lights lights;
    lights.count = f.folded_count;
    lights.unit = false;
    for (uint l = 0; l < lights.count; ++l)
    {
        uint slot = f.folded[l];
        lights.direction[l] = normalize(scene.lights[a.candidates[slot]].direction);
        lights.color[l] = f.illuminance[slot].rgb * f.weight[slot];
    }
    // From the eye to `near_range` first, whose lobe depth the light the lobe keeps on its way
    // from past it carries on to the eye.
    float3 none[MAX_GROUP_PASSES] = {};
    float3 near[MAX_GROUP_PASSES];
    float3 near_passed[MAX_GROUP_PASSES];
    float3 near_lobe[MAX_GROUP_PASSES];
    float3 unused[MAX_GROUP_PASSES];
    march(a, h0, dir, 0.0, start, false, SKY_VIEW_NEAR_STEPS, lights, a.multiple_scattering_scale, true, 0,
          a.group_passes, false, none, near, near_passed, near_lobe, unused);
    float3 far[MAX_GROUP_PASSES];
    float3 passed[MAX_GROUP_PASSES];
    float3 far_lobe[MAX_GROUP_PASSES];
    float3 more[MAX_GROUP_PASSES];
    march(a, h0, dir, start, end, ground, SKY_VIEW_STEPS, lights, a.multiple_scattering_scale, true, 0,
          a.group_passes, true, near_lobe, far, passed, far_lobe, more);
    float3 rgb = float3(0.0);
    float3 eye = float3(0.0);
    float3 kept = float3(0.0);
    for (uint p = 0; p < a.group_passes; ++p)
    {
        rgb += atmosphere_rec2020(a.groups[p], far[p]);
        eye += atmosphere_rec2020(a.groups[p], near[p] + near_passed[p] * (far[p] + more[p]));
        kept += atmosphere_rec2020(a.groups[p], more[p]);
    }
    RWTexture2DArray<float4> lut = root.output.get<RWTexture2DArray<float4>>();
    lut[uint3(id.xy, SKY_VIEW_FAR)] = float4(rgb, 1.0);
    lut[uint3(id.xy, SKY_VIEW_EYE)] = float4(eye, 1.0);
    lut[uint3(id.xy, SKY_VIEW_KEPT)] = float4(kept, 1.0);
}
