// Planetary sky and atmosphere: the data block every atmosphere pass reads, its geometry,
// the LUT parameterisations and lookups.
//
// Two altitudes exist. The medium's is height above the flat ground plane through the
// planet top point. The LUTs are spherical: each sits at one altitude above the sphere of
// `planet_radius`, with `up` as its zenith, and a point along a LUT ray has its altitude
// above that sphere.
module atmosphere;

import orr.abi;
import orr.math;
import tables;
import media;
import environment;

public static const uint MAX_SKY_LIGHTS = 16;
public static const uint MAX_FOLDED_LIGHTS = 8;
// The focus view's sky-view LUT and the extra views'.
public static const uint MAX_SKY_VIEWS = 5;

public static const uint2 TRANSMITTANCE_SIZE = uint2(256, 128);
public static const uint2 MULTIPLE_SCATTERING_SIZE = uint2(64, 32);
public static const uint2 SKY_VIEW_SIZE = uint2(400, 200);
// Entries of the ground irradiance table, over the light's zenith cosine.
public static const uint GROUND_SIZE = 64;

// The LUT passes' coefficients for three adjacent wavelength groups. Each float4 holds the
// three groups in xyz.
public struct AtmosphereGroups
{
    // Rayleigh scattering at the ground, 1/m.
    public float4 rayleigh;
    // Aerosol per unit vertical column: times exp(-h / H) / H gives 1/m.
    public float4 aerosol_extinction;
    public float4 aerosol_scattering;
    // Ozone absorption at the ozone peak, 1/m.
    public float4 ozone;
    public float4 fog_albedo;
    public float4 ground_albedo;
    // Row c: each group's share of Rec.2020 channel c, nits per unit of the group's value.
    public float4 rec2020[3];
    // Row j: group j's spectral irradiance from a light's Rec.2020 illuminance, as a dot
    // product with it.
    public float4 light_weight[3];
}

// Which candidates the sky-view passes fold in.
public struct FoldGpu
{
    // Per candidate slot: the light's Rec.2020 illuminance above the atmosphere (a `Textured`
    // disc with zero illuminance gets its disc integral).
    public float4 illuminance[MAX_SKY_LIGHTS];
    // Per candidate slot.
    public float weight[MAX_SKY_LIGHTS];
    // Candidate slots with a non-zero weight, highest weight first.
    public uint folded[MAX_FOLDED_LIGHTS];
    public uint folded_count;
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// One atmosphere: what `AtmosphereState` writes and the linked types read.
public struct AtmosphereGpu
{
    // The scene's `SceneRoot`; `atmosphere_scene` reads it.
    public uint64_t scene;
    // `group_passes` entries.
    public AtmosphereGroups* groups;
    public FoldGpu* fold;
    // Sky irradiance on the ground per unit spectral irradiance of a light: entry
    // `i * group_passes + pass` for the light's zenith cosine -1 + 2 i / (GROUND_SIZE - 1).
    public float4* ground_irradiance;
    // The planet top point: whole metres and the rest.
    public int4 top_metres;
    public float4 top_offset;
    public float4 up;
    // Linear Rec.2020 of sunlight above the atmosphere, luminance 1.
    public float4 solar_colour;
    // The medium's three solar-weighted Rec.2020 bands, in the units of `AtmosphereGroups`.
    public float4 rayleigh;
    public float4 aerosol_extinction;
    public float4 aerosol_scattering;
    public float4 ozone;
    public float4 fog_albedo;
    public float planet_radius;
    public float top_radius;
    public float rayleigh_scale;
    public float aerosol_scale;
    public float aerosol_asymmetry;
    public float ozone_peak;
    public float ozone_half_width;
    // Zenith radiance of the night background before extinction, nits; 0 without it.
    public float night_background;
    // Fog extinction at `fog_base`, 1/m; 0 without fog.
    public float fog_density;
    public float fog_base;
    public float fog_falloff;
    public float fog_diameter;
    public uint group_passes;
    public uint sky_lights;
    public uint candidate_count;
    public uint sky_view_count;
    // Light table indices.
    public uint candidates[MAX_SKY_LIGHTS];
    // Altitude above the ground plane of each sky-view LUT's eye, m.
    public float sky_view_altitude[MAX_SKY_VIEWS];
    // Groups in layers of three, then the Rec.2020 layer.
    public TextureIndex transmittance;
    public TextureIndex multiple_scattering;
    // Linear filtering, repeating.
    public SamplerIndex sampler;
    public TextureIndex sky_view[MAX_SKY_VIEWS];
    // Scales the multiple scattering the sky-view LUTs add: 1, or 0 for single scattering.
    public float multiple_scattering_scale;
    public uint _pad0;
    public uint _pad1;
}

public SceneRoot* atmosphere_scene(AtmosphereGpu* a)
{
    return (SceneRoot*)a.scene;
}

// Geometry.

// The planet top point in render space.
public float3 atmosphere_top_render(AtmosphereGpu* a)
{
    int4 origin = atmosphere_scene(a).origin_cell;
    return float3(a.top_metres.xyz - (origin.xyz << origin.w)) + a.top_offset.xyz;
}

// Height above the ground plane.
public float atmosphere_altitude(AtmosphereGpu* a, float3 p_render)
{
    return dot(p_render - atmosphere_top_render(a), a.up.xyz);
}

// Height above the sphere, in a form that keeps render-space precision far from the centre.
public float atmosphere_spherical_altitude(AtmosphereGpu* a, float3 p_render)
{
    float3 q = p_render - atmosphere_top_render(a);
    float r = a.planet_radius;
    float3 up = a.up.xyz;
    return (dot(q, q) + 2.0 * r * dot(q, up)) / (length(q + r * up) + r);
}

// rho^2 = r^2 - R^2 at altitude h.
float rho_squared(AtmosphereGpu* a, float h)
{
    return h * (2.0 * a.planet_radius + h);
}

public bool atmosphere_ray_hits_ground(AtmosphereGpu* a, float h, float mu)
{
    float r = a.planet_radius + h;
    return mu < 0.0 && r * r * mu * mu >= rho_squared(a, h);
}

// Distance from altitude h along zenith cosine mu to the top of the atmosphere.
public float atmosphere_distance_to_top(AtmosphereGpu* a, float h, float mu)
{
    float r = a.planet_radius + h;
    float c = (a.top_radius - r) * (a.top_radius + r);
    float s = sqrt(max(r * r * mu * mu + c, 0.0));
    return max(mu > 0.0 ? c / (r * mu + s) : s - r * mu, 0.0);
}

// Distance to the ground along a ray that hits it.
public float atmosphere_distance_to_ground(AtmosphereGpu* a, float h, float mu)
{
    float r = a.planet_radius + h;
    float rr = rho_squared(a, h);
    return rr / (-r * mu + sqrt(max(r * r * mu * mu - rr, 0.0)));
}

// Altitude at distance t from altitude h along zenith cosine mu.
public float atmosphere_altitude_along(AtmosphereGpu* a, float h, float mu, float t)
{
    float r = a.planet_radius + h;
    float x = t * t + 2.0 * r * mu * t + rho_squared(a, h);
    return x / (sqrt(x + a.planet_radius * a.planet_radius) + a.planet_radius);
}

// The medium.

// Rayleigh, aerosol (per unit column) and ozone densities at altitude h.
public float3 atmosphere_densities(AtmosphereGpu* a, float h)
{
    return float3(exp(-h / a.rayleigh_scale),
                  exp(-h / a.aerosol_scale) / a.aerosol_scale,
                  max(0.0, 1.0 - abs(h - a.ozone_peak) / a.ozone_half_width));
}

public float atmosphere_fog_extinction(AtmosphereGpu* a, float h)
{
    if (a.fog_density <= 0.0)
        return 0.0;
    return height_fog_extinction(a.fog_density, a.fog_base, a.fog_falloff, h);
}

public float rayleigh_phase(float cos_theta)
{
    return 3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta);
}

public float cornette_shanks_phase(float g, float cos_theta)
{
    float d = 1.0 + g * g - 2.0 * g * cos_theta;
    return 3.0 / (8.0 * PI) * (1.0 - g * g) * (1.0 + cos_theta * cos_theta)
        / ((2.0 + g * g) * d * sqrt(d));
}

// exp(z^2) erfc(z) for z >= 0, within 4e-4 relative.
float erfcx(float z)
{
    if (z < 2.5)
    {
        // Abramowitz and Stegun 7.1.26 without its exp(-z^2).
        float t = 1.0 / (1.0 + 0.3275911 * z);
        return t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
            + t * (-1.453152027 + t * 1.061405429))));
    }
    // Laplace's continued fraction.
    float f = z;
    for (int k = 6; k >= 1; --k)
        f = z + 0.5 * float(k) / f;
    return 1.0 / (sqrt(PI) * f);
}

// Optical depth of an exponential layer to space along a spherical ray, from radius r with
// zenith cosine mu, in units of the layer's density at r times its scale height `scale`,
// for r much larger than `scale`. The ray must not hit the ground.
float chapman_up(float r, float mu, float scale)
{
    float x = r / scale;
    float k = sqrt(0.5 * x / max(1.0 - mu * mu, 1e-12));
    return sqrt(PI) * k * erfcx(mu * k);
}

// Altitude of the lowest point of the ray from altitude h along zenith cosine mu < 0, where
// it is tangent to a sphere about the planet centre.
public float atmosphere_tangent_altitude(AtmosphereGpu* a, float h, float mu)
{
    float r = a.planet_radius + h;
    float rt = r * sqrt(1.0 - mu * mu);
    return max((rho_squared(a, h) - r * r * mu * mu) / (rt + a.planet_radius), 0.0);
}

// Fog optical depth from altitude h to space along zenith cosine mu on the sphere. The ray
// must not hit the ground.
public float atmosphere_fog_optical_depth_to_space(AtmosphereGpu* a, float h, float mu)
{
    if (a.fog_density <= 0.0)
        return 0.0;
    float scale = a.fog_falloff;
    float r = a.planet_radius + h;
    float here = scale * atmosphere_fog_extinction(a, h);
    if (mu >= 0.0)
        return here * chapman_up(r, mu, scale);
    // Down to the tangent point and up again: twice the tangent point's half-path, less the
    // part behind the eye.
    float ht = atmosphere_tangent_altitude(a, h, mu);
    float tangent = scale * atmosphere_fog_extinction(a, ht);
    return 2.0 * tangent * chapman_up(a.planet_radius + ht, 0.0, scale)
        - here * chapman_up(r, -mu, scale);
}

// LUT parameterisations. Each maps a point to unit coordinates in [0, 1]^2 whose texel
// centres sit at 0 and 1; `lut_coord` turns them into texture coordinates.

public float2 atmosphere_lut_coord(float2 unit, uint2 size)
{
    return (0.5 + unit * float2(size - 1)) / float2(size);
}

// Bruneton's (r, mu) mapping with the r axis as sqrt(rho / H), which puts more rows near
// the ground where the aerosol is.
public float2 atmosphere_transmittance_unit(AtmosphereGpu* a, float h, float mu)
{
    float big_r = a.planet_radius;
    float top = a.top_radius - big_r;
    float horizon = sqrt(top * (a.top_radius + big_r));
    float rho = sqrt(rho_squared(a, h));
    float d = atmosphere_distance_to_top(a, h, mu);
    float d_min = top - h;
    float d_max = rho + horizon;
    return float2(saturate((d - d_min) / (d_max - d_min)), sqrt(rho / horizon));
}

public void atmosphere_transmittance_point(AtmosphereGpu* a, float2 unit, out float h,
                                           out float mu)
{
    float big_r = a.planet_radius;
    float top = a.top_radius - big_r;
    float horizon = sqrt(top * (a.top_radius + big_r));
    float rho = horizon * unit.y * unit.y;
    h = rho * rho / (sqrt(rho * rho + big_r * big_r) + big_r);
    float r = big_r + h;
    float d_min = top - h;
    float d_max = rho + horizon;
    float d = d_min + unit.x * (d_max - d_min);
    mu = d <= 0.0 ? 1.0 : clamp((horizon * horizon - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
}

// Light's zenith cosine on x, altitude on y, both linear.
public float2 atmosphere_multiple_scattering_unit(AtmosphereGpu* a, float h, float mu_light)
{
    return saturate(float2(mu_light * 0.5 + 0.5, h / (a.top_radius - a.planet_radius)));
}

public void atmosphere_multiple_scattering_point(AtmosphereGpu* a, float2 unit, out float h,
                                                 out float mu_light)
{
    mu_light = unit.x * 2.0 - 1.0;
    h = unit.y * (a.top_radius - a.planet_radius);
}

// The sky-view LUT's frame: zenith `up`, longitude measured from `t` towards `b`.
void sky_view_frame(AtmosphereGpu* a, out float3 t, out float3 b)
{
    orthonormal_basis(a.up.xyz, t, b);
}

// Zenith angle of the horizon seen from altitude h.
float horizon_zenith_angle(AtmosphereGpu* a, float h)
{
    return PI - acos(clamp(sqrt(rho_squared(a, h)) / (a.planet_radius + h), 0.0, 1.0));
}

// Sky-view texture coordinates of `dir` from a LUT eye at altitude h. Latitude is warped
// quadratically towards the horizon, which sits at v = 0.5 between two rows; longitude is
// absolute and wraps.
public float2 atmosphere_sky_view_uv(AtmosphereGpu* a, float h, float3 dir)
{
    float3 t, b;
    sky_view_frame(a, t, b);
    float horizon = horizon_zenith_angle(a, h);
    float theta = acos(clamp(dot(dir, a.up.xyz), -1.0, 1.0));
    float v = theta < horizon ? 0.5 - 0.5 * sqrt(max(1.0 - theta / horizon, 0.0))
                              : 0.5 + 0.5 * sqrt(max((theta - horizon) / (PI - horizon), 0.0));
    float u = atan2(dot(dir, b), dot(dir, t)) * INV_TWO_PI;
    u = u - floor(u);
    return float2(u, (0.5 + v * float(SKY_VIEW_SIZE.y - 1)) / float(SKY_VIEW_SIZE.y));
}

// The direction of sky-view texture coordinates `uv` from a LUT eye at altitude h.
public float3 atmosphere_sky_view_direction(AtmosphereGpu* a, float h, float2 uv)
{
    float3 t, b;
    sky_view_frame(a, t, b);
    float horizon = horizon_zenith_angle(a, h);
    float v = saturate((uv.y * float(SKY_VIEW_SIZE.y) - 0.5) / float(SKY_VIEW_SIZE.y - 1));
    float theta;
    if (v < 0.5)
    {
        float c = 1.0 - 2.0 * v;
        theta = horizon * (1.0 - c * c);
    }
    else
    {
        float c = 2.0 * v - 1.0;
        theta = horizon + (PI - horizon) * c * c;
    }
    float phi = uv.x * TWO_PI;
    float s = sin(theta);
    return s * cos(phi) * t + s * sin(phi) * b + cos(theta) * a.up.xyz;
}

// Lookups.

float3 transmittance_texel(AtmosphereGpu* a, float h, float mu, uint layer)
{
    float2 uv = atmosphere_lut_coord(atmosphere_transmittance_unit(a, h, mu), TRANSMITTANCE_SIZE);
    return a.transmittance.get<Texture2DArray<float4>>()
        .SampleLevel(a.sampler.get(), float3(uv, float(layer)), 0.0).rgb;
}

// Transmittance to space of the clear air (no fog) from altitude h along zenith cosine mu:
// three groups from `layer`, or the Rec.2020 layer at `group_passes`. 0 when the ray hits
// the ground.
public float3 atmosphere_lut_transmittance(AtmosphereGpu* a, float h, float mu, uint layer)
{
    if (atmosphere_ray_hits_ground(a, h, mu))
        return float3(0.0);
    return transmittance_texel(a, h, mu, layer);
}

// As `atmosphere_lut_transmittance`, fog included.
public float3 atmosphere_light_transmittance(AtmosphereGpu* a, float h, float mu, uint layer)
{
    if (atmosphere_ray_hits_ground(a, h, mu))
        return float3(0.0);
    return transmittance_texel(a, h, mu, layer)
        * exp(-atmosphere_fog_optical_depth_to_space(a, h, mu));
}

// Multiply-scattered radiance per unit scattering coefficient and per unit spectral
// irradiance of a light at zenith cosine mu_light, at altitude h: groups from `layer`, or
// the Rec.2020 layer at `group_passes`.
public float3 atmosphere_lut_multiple_scattering(AtmosphereGpu* a, float h, float mu_light,
                                                 uint layer)
{
    float2 uv = atmosphere_lut_coord(atmosphere_multiple_scattering_unit(a, h, mu_light),
                                     MULTIPLE_SCATTERING_SIZE);
    return a.multiple_scattering.get<Texture2DArray<float4>>()
        .SampleLevel(a.sampler.get(), float3(uv, float(layer)), 0.0).rgb;
}

// Rec.2020 transmittance to space of sunlight, fog included, from altitude h above the
// sphere along zenith cosine mu: 0 when the ray hits the ground.
public float3 atmosphere_transmittance_to_space(AtmosphereGpu* a, float h, float mu)
{
    return atmosphere_light_transmittance(a, h, mu, a.group_passes);
}

// The sky-view LUT whose eye altitude is nearest the ground-plane altitude h.
public uint atmosphere_sky_view_index(AtmosphereGpu* a, float h)
{
    uint best = 0;
    for (uint i = 1; i < a.sky_view_count; ++i)
        if (abs(a.sky_view_altitude[i] - h) < abs(a.sky_view_altitude[best] - h))
            best = i;
    return best;
}

// Sky radiance along `dir` from the nearest sky-view LUT: nits, linear Rec.2020, the medium
// beyond `near_range` and the virtual ground below the horizon, no light discs.
public float3 atmosphere_sky_view(AtmosphereGpu* a, float3 dir, float3 eye_render)
{
    uint i = atmosphere_sky_view_index(a, atmosphere_altitude(a, eye_render));
    return sky_view_lookup(a, i, dir);
}

float3 sky_view_lookup(AtmosphereGpu* a, uint i, float3 dir)
{
    float h = max(a.sky_view_altitude[i], 0.0);
    return a.sky_view[i].get<Texture2D<float4>>()
        .SampleLevel(a.sampler.get(), atmosphere_sky_view_uv(a, h, dir), 0.0).rgb;
}

// Altitude of the airglow layer, m.
static const float AIRGLOW_ALTITUDE = 90000.0;

// Night background radiance before extinction seen from altitude h above the sphere along
// zenith cosine mu, nits of a neutral colour: two thirds airglow from a thin layer with van
// Rhijn brightening toward the horizon, one third zodiacal light and starlight, uniform.
public float atmosphere_night_background(AtmosphereGpu* a, float h, float mu)
{
    float s = (a.planet_radius + h) / (a.planet_radius + AIRGLOW_ALTITUDE);
    float airglow = s < 1.0 ? rsqrt(1.0 - s * s * (1.0 - mu * mu)) : 0.0;
    return a.night_background * (2.0 / 3.0 * airglow + 1.0 / 3.0);
}

// The sky as orr's environment link point reads it: the nearest sky-view LUT (the medium
// beyond `near_range`, the folded lights' in-scatter and the virtual ground below the
// horizon) plus the night background, extinguished from `near_range` to space. No light
// discs. Link it with `Build::link(LinkPoint::Environment, ..)`.
public struct AtmosphereSky : IEnvironment
{
    public static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        uint i = atmosphere_sky_view_index(a, atmosphere_altitude(a, eye_render));
        float3 l = sky_view_lookup(a, i, dir);
        float h = max(a.sky_view_altitude[i], 0.0);
        float mu = dot(dir, a.up.xyz);
        if (a.night_background <= 0.0 || atmosphere_ray_hits_ground(a, h, mu))
            return l;
        float t = atmosphere_scene(a).near_range;
        float hs = atmosphere_altitude_along(a, h, mu, t);
        float mu_s = ((a.planet_radius + h) * mu + t) / (a.planet_radius + hs);
        return l + atmosphere_night_background(a, h, mu)
            * atmosphere_transmittance_to_space(a, hs, mu_s);
    }
}
