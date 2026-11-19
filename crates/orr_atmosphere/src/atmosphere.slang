// Planetary sky and atmosphere: the data block every atmosphere pass reads, its geometry,
// the LUT parameterizations and lookups.
//
// Two altitudes exist. The medium's is height above the flat ground plane through the
// planet top point. The LUTs are spherical: each sits at one altitude above the sphere of
// `planet_radius`, with `up` as its zenith, and a point along a LUT ray has its altitude
// above that sphere.
module atmosphere;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_scene.tables;
import orr_lighting.media;
import orr_lighting.environment;

public static const uint MAX_SKY_LIGHTS = 16;
public static const uint MAX_FOLDED_LIGHTS = 8;
// The focus view's sky-view LUT and the extra views'.
public static const uint MAX_SKY_VIEWS = 5;

public static const uint2 TRANSMITTANCE_SIZE = uint2(256, 128);
// The multiple-scattering tables' texels per axis: altitude, the light's zenith cosine, the
// view's zenith cosine (the radiance table's count is even, which puts the horizon between two
// texels) and the azimuth between the view and the light.
[Host] public static const uint MS_ALTITUDES = 16;
[Host] public static const uint MS_LIGHT_ZENITHS = 32;
[Host] public static const uint MS_VIEW_ZENITHS = 16;
[Host] public static const uint MS_RAY_ZENITHS = 32;
[Host] public static const uint MS_AZIMUTHS = 16;
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

// Which candidates the sky-view passes fold in, and how much of each. The fold keeps it
// across frames.
public struct FoldGpu
{
    // Per candidate slot: the light's Rec.2020 illuminance above the atmosphere (a `Textured`
    // disc with zero illuminance gets its disc integral).
    public float4 illuminance[MAX_SKY_LIGHTS];
    // Per candidate slot: the share of its in-scatter the sky-view LUTs hold, in [0, 1].
    public float weight[MAX_SKY_LIGHTS];
    // Per candidate slot: the light table index the weight belongs to.
    public uint light[MAX_SKY_LIGHTS];
    // Candidate slots with a non-zero weight: members, then the lights fading out.
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
    // Sky irradiance on the ground per unit spectral irradiance of a light, as natural logs no
    // lower than MS_LOG_LOW: entry `i * group_passes + pass` for the light's zenith cosine
    // `atmosphere_ms_light(i / (GROUND_SIZE - 1))`, which crowds the entries about the horizon
    // where the irradiance falls fastest.
    public float4* ground_irradiance;
    // The planet top point: whole meters and the rest.
    public int4 top_meters;
    public float4 top_offset;
    public float4 up;
    // Linear Rec.2020 of sunlight above the atmosphere, luminance 1.
    public float4 solar_color;
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
    // The multiple-scattering table (`atmosphere_lut_multiple_scattering`): a 3D texture with
    // the same layers.
    public TextureIndex multiple_scattering;
    // Linear filtering, repeating.
    public SamplerIndex sampler;
    public TextureIndex sky_view[MAX_SKY_VIEWS];
    // How much of each sky-view LUT a view reads, the rest from the focus LUT (entry 0).
    public float sky_view_weight[MAX_SKY_VIEWS];
    // Scales the multiple scattering the sky-view LUTs and the medium add: 1, or 0 for single
    // scattering.
    public float multiple_scattering_scale;
    // Non-zero: the fold sets every weight to its target instead of fading.
    public uint fold_reset;
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
    return float3(a.top_meters.xyz - (origin.xyz << origin.w)) + a.top_offset.xyz;
}

// Height above the ground plane.
public float atmosphere_altitude(AtmosphereGpu* a, float3 p_render)
{
    return dot(p_render - atmosphere_top_render(a), a.up.xyz);
}

// Height above the sphere, in a form that keeps render-space precision far from the center.
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

// Extinction at altitude h above the ground plane, 1/m, in the medium's Rec.2020 bands.
public float3 atmosphere_extinction(AtmosphereGpu* a, float h)
{
    float3 d = atmosphere_densities(a, h);
    return a.rayleigh.rgb * d.x + a.aerosol_extinction.rgb * d.y + a.ozone.rgb * d.z
        + atmosphere_fog_extinction(a, h);
}

// Larger columns than this stop all light at any wavelength; holding them finite keeps a zero
// coefficient's product with them zero.
static const float LARGEST_COLUMN = 1e30;

// Integral of exp(-h / scale) over t finite meters from altitude h0, where the altitude
// changes by d per meter.
float exponential_column(float scale, float h0, float d, float t)
{
    return min(height_fog_optical_depth(1.0, 0.0, scale, h0, d, t), LARGEST_COLUMN);
}

// Integral of the ozone tent over t meters from altitude h0, where the altitude changes by d
// per meter: each side of the tent is linear, so its mean over a piece is its middle value.
float ozone_column(AtmosphereGpu* a, float h0, float d, float t)
{
    float h1 = h0 + d * t;
    float lo = min(h0, h1);
    float hi = max(h0, h1);
    float peak = a.ozone_peak;
    float width = a.ozone_half_width;
    if (!(hi > lo))
        return t * max(0.0, 1.0 - abs(h0 - peak) / width);
    float sum = 0.0;
    float knots[3] = { peak - width, peak, peak + width };
    for (uint i = 0; i < 2; ++i)
    {
        float from = max(lo, knots[i]);
        float to = min(hi, knots[i + 1]);
        if (to > from)
            sum += (to - from) / (hi - lo) * (1.0 - abs(0.5 * (from + to) - peak) / width);
    }
    return t * sum;
}

// Optical depth over t finite meters from `p_render` along unit `dir`, in the medium's Rec.2020
// bands: altitude above the ground plane is linear in distance, so every layer has a closed
// form.
public float3 atmosphere_optical_depth(AtmosphereGpu* a, float3 p_render, float3 dir, float t)
{
    float h = atmosphere_altitude(a, p_render);
    float d = dot(dir, a.up.xyz);
    float fog = 0.0;
    if (a.fog_density > 0.0)
        fog = height_fog_optical_depth(a.fog_density, a.fog_base, a.fog_falloff, h, d, t);
    return a.rayleigh.rgb * exponential_column(a.rayleigh_scale, h, d, t)
        + a.aerosol_extinction.rgb * (exponential_column(a.aerosol_scale, h, d, t) / a.aerosol_scale)
        + a.ozone.rgb * ozone_column(a, h, d, t)
        + fog;
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
// it is tangent to a sphere about the planet center.
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

// LUT parameterizations. Each maps a point to unit coordinates in [0, 1]^2 whose texel
// centers sit at 0 and 1; `lut_coord` turns them into texture coordinates.

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

// The multiple-scattering tables' axes. Altitude is quadratic in its coordinate, so rows crowd
// near the ground where the aerosol and the fog are; the light's zenith cosine is
// quadratic either side of the horizon, where sunlight changes fastest; the view's zenith
// angle is warped toward the horizon as the sky-view LUT's latitude is; the azimuth between the
// view and the light is linear in angle.
public float atmosphere_ms_altitude_unit(AtmosphereGpu* a, float h)
{
    return sqrt(saturate(h / (a.top_radius - a.planet_radius)));
}

public float atmosphere_ms_altitude(AtmosphereGpu* a, float unit)
{
    return (a.top_radius - a.planet_radius) * unit * unit;
}

public float atmosphere_ms_light_unit(float mu_light)
{
    return 0.5 + 0.5 * sign(mu_light) * sqrt(min(abs(mu_light), 1.0));
}

public float atmosphere_ms_light(float unit)
{
    float t = 2.0 * unit - 1.0;
    return sign(t) * t * t;
}

public float atmosphere_ms_view_unit(AtmosphereGpu* a, float h, float mu_view)
{
    return horizon_unit(horizon_zenith_angle(a, h), acos(clamp(mu_view, -1.0, 1.0)));
}

public float atmosphere_ms_view(AtmosphereGpu* a, float h, float unit)
{
    return cos(horizon_angle(horizon_zenith_angle(a, h), unit));
}

// The zenith angle at view coordinate `unit` from altitude h, and in y its derivative by `unit`.
public float2 atmosphere_ms_view_angle(AtmosphereGpu* a, float h, float unit)
{
    float horizon = horizon_zenith_angle(a, h);
    float c = abs(2.0 * unit - 1.0);
    float side = unit < 0.5 ? horizon : PI - horizon;
    return float2(horizon_angle(horizon, unit), 4.0 * side * c);
}

// The azimuth coordinate of a view at zenith cosine mu_view and a light at mu_light whose
// directions are at cosine nu: 0 toward the light's azimuth, 1 away from it. Straight up or
// down every azimuth is the same direction.
public float atmosphere_ms_azimuth_unit(float mu_view, float mu_light, float nu)
{
    float s = sqrt(max((1.0 - mu_view * mu_view) * (1.0 - mu_light * mu_light), 0.0));
    float c = s > 0.0 ? clamp((nu - mu_view * mu_light) / s, -1.0, 1.0) : 1.0;
    return acos(c) * INV_PI;
}

// Coordinates (altitude, light, view, azimuth) of a point at altitude h lit from zenith cosine
// mu_light and seen along zenith cosine mu_view, at cosine nu to the light.
public float4 atmosphere_ms_unit(AtmosphereGpu* a, float h, float mu_light, float mu_view, float nu)
{
    return float4(atmosphere_ms_altitude_unit(a, h), atmosphere_ms_light_unit(mu_light),
                  atmosphere_ms_view_unit(a, h, mu_view),
                  atmosphere_ms_azimuth_unit(mu_view, mu_light, nu));
}

// The natural logs a table's 16-bit texels span (`atmosphere_ms_encode`). The planet's shadow
// holds zeros, which store as the lowest; the largest in-scatter per unit scattering is a phase
// function's peak, a few per steradian.
public static const float MS_LOG_LOW = -44.0;
public static const float MS_LOG_HIGH = 4.0;

// A value as a table's texel stores it: its natural log, which interpolates the exponential change
// toward the horizon and the light, over [MS_LOG_LOW, MS_LOG_HIGH] in [0, 1].
public float3 atmosphere_ms_encode(float3 v)
{
    return saturate((log(v) - MS_LOG_LOW) / (MS_LOG_HIGH - MS_LOG_LOW));
}

public float3 atmosphere_ms_decode(float3 texel)
{
    return exp(MS_LOG_LOW + texel * (MS_LOG_HIGH - MS_LOG_LOW));
}

// A 4D table in a 3D texture of `layers` layers: along x a block of `views` view texels for each
// azimuth, along y the light, along z `MS_ALTITUDES` altitude texels per layer. Filtering stays
// inside a block and a layer, and the azimuth is interpolated here. Texels are encoded logs
// (`atmosphere_ms_encode`).
public float3 atmosphere_ms_table(Texture3D<float4> t, SamplerState s, float4 unit, uint views, uint layer, uint layers)
{
    float x = unit.z * float(views - 1) + 0.5;
    float p = unit.w * float(MS_AZIMUTHS - 1);
    float p0 = min(floor(p), float(MS_AZIMUTHS - 2));
    float width = float(views * MS_AZIMUTHS);
    float y = (unit.y * float(MS_LIGHT_ZENITHS - 1) + 0.5) / float(MS_LIGHT_ZENITHS);
    float z = (float(layer * MS_ALTITUDES) + unit.x * float(MS_ALTITUDES - 1) + 0.5) / float(MS_ALTITUDES * layers);
    float3 l0 = t.SampleLevel(s, float3((p0 * float(views) + x) / width, y, z), 0.0).rgb;
    float3 l1 = t.SampleLevel(s, float3(((p0 + 1.0) * float(views) + x) / width, y, z), 0.0).rgb;
    return atmosphere_ms_decode(lerp(l0, l1, p - p0));
}

// The texel of a 4D table that holds the value at coordinates index / (count - 1) on each axis
// (altitude, light, view, azimuth).
public uint3 atmosphere_ms_texel(uint4 index, uint views, uint layer)
{
    return uint3(index.w * views + index.z, index.y, layer * MS_ALTITUDES + index.x);
}

// The sky-view LUT's frame: zenith `up`, longitude measured from `t` toward `b`.
void sky_view_frame(AtmosphereGpu* a, out float3 t, out float3 b)
{
    orthonormal_basis(a.up.xyz, t, b);
}

// Zenith angle of the horizon seen from altitude h.
float horizon_zenith_angle(AtmosphereGpu* a, float h)
{
    return PI - acos(clamp(sqrt(rho_squared(a, h)) / (a.planet_radius + h), 0.0, 1.0));
}

// Zenith angle theta as a coordinate in [0, 1] warped quadratically toward the horizon at zenith
// angle `horizon`, which sits at 0.5.
float horizon_unit(float horizon, float theta)
{
    return theta < horizon ? 0.5 - 0.5 * sqrt(max(1.0 - theta / horizon, 0.0))
                           : 0.5 + 0.5 * sqrt(max((theta - horizon) / (PI - horizon), 0.0));
}

float horizon_angle(float horizon, float v)
{
    if (v < 0.5)
    {
        float c = 1.0 - 2.0 * v;
        return horizon * (1.0 - c * c);
    }
    float c = 2.0 * v - 1.0;
    return horizon + (PI - horizon) * c * c;
}

// Sky-view texture coordinates of `dir` from a LUT eye at altitude h. Latitude is warped
// quadratically toward the horizon, which sits at v = 0.5 between two rows; longitude is
// absolute and wraps.
public float2 atmosphere_sky_view_uv(AtmosphereGpu* a, float h, float3 dir)
{
    float3 t, b;
    sky_view_frame(a, t, b);
    float v = horizon_unit(horizon_zenith_angle(a, h), acos(clamp(dot(dir, a.up.xyz), -1.0, 1.0)));
    float u = atan2(dot(dir, b), dot(dir, t)) * INV_TWO_PI;
    u = u - floor(u);
    return float2(u, (0.5 + v * float(SKY_VIEW_SIZE.y - 1)) / float(SKY_VIEW_SIZE.y));
}

// The direction of sky-view texture coordinates `uv` from a LUT eye at altitude h.
public float3 atmosphere_sky_view_direction(AtmosphereGpu* a, float h, float2 uv)
{
    float3 t, b;
    sky_view_frame(a, t, b);
    float v = saturate((uv.y * float(SKY_VIEW_SIZE.y) - 0.5) / float(SKY_VIEW_SIZE.y - 1));
    float theta = horizon_angle(horizon_zenith_angle(a, h), v);
    float phi = uv.x * TWO_PI;
    float s = sin(theta);
    return s * cos(phi) * t + s * sin(phi) * b + cos(theta) * a.up.xyz;
}

// Quadrature.

// The sample point and length of step i of n over [t0, t1], denser toward t0.
public float2 step_at(float t0, float t1, uint i, uint n)
{
    float u0 = float(i) / float(n);
    float u1 = float(i + 1) / float(n);
    float um = (float(i) + 0.5) / float(n);
    return float2(t0 + (t1 - t0) * um * um, (t1 - t0) * (u1 * u1 - u0 * u0));
}

// Integral of exp(-sigma s) over a step of length dt.
public float3 step_integral(float3 sigma, float dt)
{
    float3 x = sigma * dt;
    float3 series = dt * (1.0 - x * (0.5 - x / 6.0));
    float3 exact = (1.0 - exp(-x)) / max(sigma, 1e-30);
    return float3(x.x < 1e-3 ? series.x : exact.x, x.y < 1e-3 ? series.y : exact.y,
                  x.z < 1e-3 ? series.z : exact.z);
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

// What a point at altitude h scatters toward a view along zenith cosine mu_view by every order
// from the second, per unit scattering coefficient, of a light at zenith cosine mu_light at
// cosine nu to the view: three groups from `layer`, per unit spectral irradiance, or the
// Rec.2020 layer at `group_passes`, per unit of the light's Rec.2020 illuminance. Every order
// scatters the one before with the medium's phase functions, so the aerosol and the fog send
// more of it on toward the light's side.
public float3 atmosphere_lut_multiple_scattering(AtmosphereGpu* a, float h, float mu_light,
                                                 float mu_view, float nu, uint layer)
{
    return atmosphere_ms_table(a.multiple_scattering.get<Texture3D<float4>>(), a.sampler.get(),
                               atmosphere_ms_unit(a, h, mu_light, mu_view, nu), MS_VIEW_ZENITHS,
                               layer, a.group_passes + 1);
}

// Rec.2020 transmittance to space of sunlight, fog included, from altitude h above the
// sphere along zenith cosine mu: 0 when the ray hits the ground. An altitude outside the
// atmosphere reads as its nearer end. Light reddened past Rec.2020 clamps to it.
public float3 atmosphere_transmittance_to_space(AtmosphereGpu* a, float h, float mu)
{
    float inside = clamp(h, 0.0, a.top_radius - a.planet_radius);
    return max(atmosphere_light_transmittance(a, inside, mu, a.group_passes), 0.0);
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

// Sky radiance along `dir` from the nearest sky-view LUT, blended with the focus LUT by its
// weight: nits, linear Rec.2020, the medium beyond `near_range` and the virtual ground below
// the horizon, no light discs. `eye_render` is the same for every invocation that calls it,
// since it chooses the LUT the heap read takes as uniform.
public float3 atmosphere_sky_view(AtmosphereGpu* a, float3 dir, float3 eye_render)
{
    uint i = atmosphere_sky_view_index(a, atmosphere_altitude(a, eye_render));
    float3 l = sky_view_lookup(a, i, dir);
    float w = a.sky_view_weight[i];
    return w < 1.0 ? lerp(sky_view_lookup(a, 0, dir), l, w) : l;
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
// zenith cosine mu, nits of a neutral color: two thirds airglow from a thin layer with van
// Rhijn brightening toward the horizon, one third zodiacal light and starlight, uniform.
public float atmosphere_night_background(AtmosphereGpu* a, float h, float mu)
{
    float s = (a.planet_radius + h) / (a.planet_radius + AIRGLOW_ALTITUDE);
    float airglow = s < 1.0 ? rsqrt(1.0 - s * s * (1.0 - mu * mu)) : 0.0;
    return a.night_background * (2.0 / 3.0 * airglow + 1.0 / 3.0);
}

// Sky-view LUT i along `dir` plus the night background, extinguished from `near_range` to
// space.
float3 sky_view_radiance(AtmosphereGpu* a, uint i, float3 dir)
{
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

// The wavelength group passes an atmosphere has (`group_passes`, `PASSES` in `block.rs`).
[Host] public static const uint MAX_GROUP_PASSES = 4;

float sum3(float3 v)
{
    return v.x + v.y + v.z;
}

// One wavelength group pass's three groups, `v`, in Rec.2020.
public float3 atmosphere_rec2020(AtmosphereGroups g, float3 v)
{
    return float3(dot(g.rec2020[0].xyz, v), dot(g.rec2020[1].xyz, v), dot(g.rec2020[2].xyz, v));
}

// A light of Rec.2020 color `color` in one wavelength group pass's three groups.
public float3 atmosphere_light_groups(AtmosphereGroups g, float3 color)
{
    return float3(dot(g.light_weight[0].xyz, color), dot(g.light_weight[1].xyz, color),
                  dot(g.light_weight[2].xyz, color));
}

// Legs a reference sky path takes at most, the leg from which Russian roulette may end it, and
// the steps of each leg's quadrature: at least REFERENCE_SKY_STEPS, and more where the leg rises
// through thin layers or starts.
static const uint REFERENCE_SKY_LEGS = 256;
static const uint REFERENCE_SKY_ROULETTE = 1;
static const uint REFERENCE_SKY_STEPS = 24;
static const float REFERENCE_SKY_RISE = 0.25;
static const float REFERENCE_SKY_GROWTH = 0.25;
static const float REFERENCE_SKY_FIRST = 10.0;

// One wavelength group pass's three groups at altitude h: Rayleigh, aerosol and fog scattering,
// and the extinction.
struct ReferenceMedium
{
    float3 rayleigh;
    float3 aerosol;
    float3 fog;
    float3 extinction;
}

ReferenceMedium reference_medium(AtmosphereGpu* a, AtmosphereGroups g, float3 d, float fog)
{
    ReferenceMedium m;
    m.rayleigh = g.rayleigh.xyz * d.x;
    m.aerosol = g.aerosol_scattering.xyz * d.y;
    m.fog = g.fog_albedo.xyz * fog;
    m.extinction = m.rayleigh + g.aerosol_extinction.xyz * d.y + g.ozone.xyz * d.z + fog;
    return m;
}

// Where a reference sky path's leg met the medium: the point, and per group its scattering
// components and its throughput there over the leg's density of the point.
struct ReferenceEvent
{
    bool met;
    float3 point;
    float3 rayleigh[MAX_GROUP_PASSES];
    float3 aerosol[MAX_GROUP_PASSES];
    float3 fog[MAX_GROUP_PASSES];
    float3 weight[MAX_GROUP_PASSES];
}

// One leg of a reference sky path, from `q` (relative to the planet's center) along unit `d`, over
// steps of constant medium. With `quadrature` it adds to `radiance` the light of every group that
// reaches `q` along the leg scattered once from every folded light, and the ground's reflection of
// their direct light where the leg ends on the ground, and to `background` the night background
// where it leaves to space, each times `throughput`. Its next event lies where the groups' mean
// transmittance falls to `target`: in the medium there, or past the leg, where `ground` says
// whether it reaches the ground, at `end` with each group's transmittance over the mean in
// `passed`.
ReferenceEvent reference_leg(AtmosphereGpu* a, float3 q, float3 d, float target, bool quadrature, float3 throughput[MAX_GROUP_PASSES],
                             inout float3 radiance[MAX_GROUP_PASSES], inout float3 background[MAX_GROUP_PASSES],
                             out bool ground, out float end, out float3 passed[MAX_GROUP_PASSES])
{
    SceneRoot* scene = atmosphere_scene(a);
    FoldGpu* fold = a.fold;
    float big_r = a.planet_radius;
    uint passes = a.group_passes;
    float r0 = length(q);
    float h0 = r0 - big_r;
    float mu = dot(q, d) / r0;
    ground = atmosphere_ray_hits_ground(a, h0, mu);
    end = ground ? atmosphere_distance_to_ground(a, h0, mu) : atmosphere_distance_to_top(a, h0, mu);
    float3 transmittance[MAX_GROUP_PASSES];
    for (uint p = 0; p < passes; ++p)
        transmittance[p] = float3(1.0);
    float mean = 1.0;
    float channels = float(3 * passes);
    ReferenceEvent e;
    e.met = false;
    e.point = q;
    // The folded lights, whose phase functions are the leg's throughout.
    float3 to_lights[MAX_FOLDED_LIGHTS];
    float3 phases[MAX_FOLDED_LIGHTS];
    float3 colors[MAX_FOLDED_LIGHTS];
    for (uint l = 0; l < fold.folded_count; ++l)
    {
        uint slot = fold.folded[l];
        to_lights[l] = normalize(scene.lights[a.candidates[slot]].direction);
        float c = dot(d, to_lights[l]);
        phases[l] = float3(rayleigh_phase(c), cornette_shanks_phase(a.aerosol_asymmetry, c),
                           a.fog_density > 0.0 ? fog_phase(a.fog_diameter, c) : 0.0);
        colors[l] = fold.illuminance[slot].rgb * fold.weight[slot];
    }
    float start = 0.0;
    while (start < end)
    {
        // A step rises at most REFERENCE_SKY_RISE of the thinnest layer's scale height there,
        // spans at most 1 / REFERENCE_SKY_STEPS of the leg, and near the leg's start grows from
        // REFERENCE_SKY_FIRST m by REFERENCE_SKY_GROWTH of the distance covered.
        float3 x0 = q + d * start;
        float r = length(x0);
        float h_start = r - big_r;
        float scale = a.rayleigh_scale;
        if (h_start < 40.0 * a.aerosol_scale)
            scale = min(scale, a.aerosol_scale);
        if (a.fog_density > 0.0 && h_start - a.fog_base < 40.0 * a.fog_falloff)
            scale = min(scale, a.fog_falloff);
        float rise = abs(dot(x0, d)) / r;
        float step = min(end / float(REFERENCE_SKY_STEPS), max(REFERENCE_SKY_GROWTH * start, REFERENCE_SKY_FIRST));
        if (rise > 0.0)
            step = min(step, REFERENCE_SKY_RISE * scale / rise);
        float2 s = float2(0.0, min(step, end - start));
        s.x = start + 0.5 * s.y;
        float3 x = q + d * s.x;
        float h = length(x) - big_r;
        float3 up = x / (big_r + h);
        float3 densities = atmosphere_densities(a, h);
        float fog = atmosphere_fog_extinction(a, h);
        // Each light's zenith cosine and its fog transmittance there, which every group shares.
        float mus[MAX_FOLDED_LIGHTS];
        float fogged[MAX_FOLDED_LIGHTS];
        for (uint l = 0; l < (quadrature ? fold.folded_count : 0); ++l)
        {
            mus[l] = dot(up, to_lights[l]);
            fogged[l] = 1.0;
            if (a.fog_density > 0.0 && !atmosphere_ray_hits_ground(a, h, mus[l]))
                fogged[l] = exp(-atmosphere_fog_optical_depth_to_space(a, h, mus[l]));
        }
        float next = 0.0;
        for (uint p = 0; p < passes; ++p)
        {
            AtmosphereGroups g = a.groups[p];
            ReferenceMedium m = reference_medium(a, g, densities, fog);
            float3 source = float3(0.0);
            for (uint l = 0; l < (quadrature ? fold.folded_count : 0); ++l)
            {
                float3 e_light = atmosphere_light_groups(g, colors[l]);
                source += e_light * atmosphere_lut_transmittance(a, h, mus[l], p) * fogged[l]
                    * (m.rayleigh * phases[l].x + m.aerosol * phases[l].y + m.fog * phases[l].z);
            }
            if (quadrature)
                radiance[p] += throughput[p] * transmittance[p] * source * step_integral(m.extinction, s.y);
            // The event's step: solve for where the mean transmittance meets the target.
            if (!e.met)
            {
                e.rayleigh[p] = m.rayleigh;
                e.aerosol[p] = m.aerosol;
                e.fog[p] = m.fog;
                e.weight[p] = m.extinction;
            }
            transmittance[p] *= exp(-m.extinction * s.y);
            next += transmittance[p].x + transmittance[p].y + transmittance[p].z;
        }
        next /= channels;
        if (!e.met && next <= target)
        {
            // Newton's method on the mean of the groups' exponentials over the step, which
            // falls monotonically from `mean` to `next`.
            float u = s.y * (mean - target) / (mean - next);
            for (uint k = 0; k < 8; ++k)
            {
                float f = -target;
                float df = 0.0;
                for (uint p = 0; p < passes; ++p)
                {
                    float3 before = transmittance[p] * exp(e.weight[p] * s.y);
                    float3 at = before * exp(-e.weight[p] * u);
                    f += (at.x + at.y + at.z) / channels;
                    df -= dot(e.weight[p], at) / channels;
                }
                u = clamp(u - f / df, 0.0, s.y);
            }
            float density = 0.0;
            for (uint p = 0; p < passes; ++p)
            {
                float3 at = transmittance[p] * exp(e.weight[p] * (s.y - u));
                density += dot(e.weight[p], at) / channels;
                e.weight[p] = at;
            }
            for (uint p = 0; p < passes; ++p)
                e.weight[p] *= (e.rayleigh[p] + e.aerosol[p] + e.fog[p]) / density;
            e.met = true;
            e.point = q + d * (start + u);
        }
        mean = next;
        start += s.y;
        // Past its event a leg without quadrature has nothing left to add.
        if (e.met && !quadrature)
            break;
    }
    for (uint p = 0; p < passes; ++p)
        passed[p] = transmittance[p] / mean;
    if (!quadrature)
        return e;
    if (ground)
    {
        float3 n = normalize(q + d * end);
        for (uint p = 0; p < passes; ++p)
            radiance[p] += throughput[p] * transmittance[p] * a.groups[p].ground_albedo.xyz * INV_PI
                * reference_ground_lit(a, p, n);
    }
    else
    {
        float night = atmosphere_night_background(a, h0, mu);
        for (uint p = 0; p < passes; ++p)
            background[p] += throughput[p] * transmittance[p] * night;
    }
    return e;
}

// Adds to `radiance` the folded lights' light that event `e` scatters back along unit `d`, times
// `throughput`.
void reference_event_light(AtmosphereGpu* a, ReferenceEvent e, float3 d, float3 throughput[MAX_GROUP_PASSES],
                           inout float3 radiance[MAX_GROUP_PASSES])
{
    SceneRoot* scene = atmosphere_scene(a);
    FoldGpu* fold = a.fold;
    uint passes = a.group_passes;
    float h = length(e.point) - a.planet_radius;
    float3 up = normalize(e.point);
    // Each group's weight per unit scattering.
    float3 weight[MAX_GROUP_PASSES];
    for (uint p = 0; p < passes; ++p)
    {
        float3 scattering = e.rayleigh[p] + e.aerosol[p] + e.fog[p];
        weight[p] = throughput[p] * select(scattering > 0.0, e.weight[p] / scattering, float3(0.0));
    }
    for (uint l = 0; l < fold.folded_count; ++l)
    {
        uint slot = fold.folded[l];
        float3 to_light = normalize(scene.lights[a.candidates[slot]].direction);
        float c = dot(d, to_light);
        float3 phase = float3(rayleigh_phase(c), cornette_shanks_phase(a.aerosol_asymmetry, c),
                              a.fog_density > 0.0 ? fog_phase(a.fog_diameter, c) : 0.0);
        float mu = dot(up, to_light);
        if (atmosphere_ray_hits_ground(a, h, mu))
            continue;
        float fogged = a.fog_density > 0.0 ? exp(-atmosphere_fog_optical_depth_to_space(a, h, mu)) : 1.0;
        float3 color = fold.illuminance[slot].rgb * fold.weight[slot] * fogged;
        for (uint p = 0; p < passes; ++p)
        {
            AtmosphereGroups g = a.groups[p];
            float3 e_light = atmosphere_light_groups(g, color);
            radiance[p] += weight[p] * e_light * atmosphere_lut_transmittance(a, h, mu, p)
                * (e.rayleigh[p] * phase.x + e.aerosol[p] * phase.y + e.fog[p] * phase.z);
        }
    }
}

// The folded lights' direct light on the ground at unit normal `n`, of group pass p.
float3 reference_ground_lit(AtmosphereGpu* a, uint p, float3 n)
{
    SceneRoot* scene = atmosphere_scene(a);
    FoldGpu* fold = a.fold;
    AtmosphereGroups g = a.groups[p];
    float3 lit = float3(0.0);
    for (uint l = 0; l < fold.folded_count; ++l)
    {
        uint slot = fold.folded[l];
        float c = dot(n, normalize(scene.lights[a.candidates[slot]].direction));
        float3 color = fold.illuminance[slot].rgb * fold.weight[slot];
        float3 e_light = atmosphere_light_groups(g, color);
        lit += e_light * max(c, 0.0) * atmosphere_light_transmittance(a, 0.0, c, p);
    }
    return lit;
}

// What `AtmosphereSky` approximates, by path tracing the medium on the spherical planet, as the
// f64 reference does: the radiance arriving at an eye at altitude h0 along unit `dir` from
// `near_range` on, the folded lights' light scattered any number of times, the virtual ground's
// reflection of it and the night background, from random numbers seeded by `seed`. With
// `quadrature` the view ray adds the light scattered once along it by quadrature, which lowers the
// draw's variance for what it costs. Each leg's next event is drawn from the groups' mean density
// along it, each group weighed by its own over that mean, and scatters by a phase component drawn
// by the groups' mean share of it, or reflects off the ground by a cosine draw, and every event
// past the integrated view ray takes the lights' light there. All wavelength groups ride every
// path. Rec.2020, nits.
public float3 atmosphere_reference_sky(AtmosphereGpu* a, float h0, float3 dir, uint seed, bool quadrature)
{
    uint state = pcg(seed);
    uint passes = a.group_passes;
    float big_r = a.planet_radius;
    float3 up = a.up.xyz;
    float mu = dot(dir, up);
    bool hits = atmosphere_ray_hits_ground(a, h0, mu);
    float reach = hits ? atmosphere_distance_to_ground(a, h0, mu) : atmosphere_distance_to_top(a, h0, mu);
    float3 q = (big_r + h0) * up + dir * min(atmosphere_scene(a).near_range, reach);
    float3 d = dir;
    float3 throughput[MAX_GROUP_PASSES];
    float3 radiance[MAX_GROUP_PASSES];
    float3 background[MAX_GROUP_PASSES];
    for (uint p = 0; p < passes; ++p)
    {
        throughput[p] = float3(1.0);
        radiance[p] = float3(0.0);
        background[p] = float3(0.0);
    }
    for (uint leg = 0; leg < REFERENCE_SKY_LEGS; ++leg)
    {
        state = pcg(state);
        bool ground;
        float end;
        float3 passed[MAX_GROUP_PASSES];
        bool integrated = quadrature && leg == 0;
        ReferenceEvent e = reference_leg(a, q, d, 1.0 - unit_float(state), integrated, throughput, radiance, background, ground, end, passed);
        float largest = 0.0;
        if (e.met)
        {
            if (!integrated)
                reference_event_light(a, e, d, throughput, radiance);
            // The phase component by the groups' mean share of the scattering there.
            float3 shares = float3(0.0);
            for (uint p = 0; p < passes; ++p)
            {
                float3 scattering = e.rayleigh[p] + e.aerosol[p] + e.fog[p];
                shares += float3(sum3(select(scattering > 0.0, e.rayleigh[p] / scattering, float3(0.0))), sum3(select(scattering > 0.0, e.aerosol[p] / scattering, float3(0.0))),
                                 sum3(select(scattering > 0.0, e.fog[p] / scattering, float3(0.0))));
            }
            shares /= float(3 * passes);
            state = pcg(state);
            float pick = unit_float(state);
            state = pcg(state);
            float u0 = unit_float(state);
            state = pcg(state);
            float u1 = unit_float(state);
            state = pcg(state);
            float u2 = unit_float(state);
            float3 wi;
            if (pick < shares.x)
                wi = direction_about(d, sample_rayleigh_cosine(u0), TWO_PI * u1);
            else if (pick < shares.x + shares.y)
                wi = direction_about(d, sample_hg_cosine(a.aerosol_asymmetry, u0), TWO_PI * u1);
            else
            {
                float unused;
                wi = sample_fog_phase(a.fog_diameter, d, float3(u0, u1, u2), unused);
            }
            float c = dot(wi, d);
            float3 phase = float3(rayleigh_phase(c), cornette_shanks_phase(a.aerosol_asymmetry, c),
                                  a.fog_density > 0.0 ? fog_phase(a.fog_diameter, c) : 0.0);
            float pdf = shares.x * rayleigh_phase(c) + shares.y * hg_phase(a.aerosol_asymmetry, c);
            if (shares.z > 0.0)
                pdf += shares.z * fog_phase_pdf(a.fog_diameter, c);
            for (uint p = 0; p < passes; ++p)
            {
                float3 scattering = e.rayleigh[p] + e.aerosol[p] + e.fog[p];
                float3 value = select(scattering > 0.0, (e.rayleigh[p] * phase.x + e.aerosol[p] * phase.y + e.fog[p] * phase.z) / scattering, float3(0.0));
                throughput[p] *= e.weight[p] * value / pdf;
                largest = max(largest, max_component(throughput[p]));
            }
            q = e.point;
            d = wi;
        }
        else if (ground)
        {
            q += d * end;
            float3 n = normalize(q);
            q = n * big_r;
            for (uint p = 0; p < passes; ++p)
            {
                throughput[p] *= passed[p] * a.groups[p].ground_albedo.xyz;
                if (!integrated)
                    radiance[p] += throughput[p] * INV_PI * reference_ground_lit(a, p, n);
                largest = max(largest, max_component(throughput[p]));
            }
            state = pcg(state);
            float u0 = unit_float(state);
            state = pcg(state);
            d = direction_about(n, sqrt(u0), TWO_PI * unit_float(state));
        }
        else
        {
            if (!integrated)
            {
                float r = length(q);
                float night = atmosphere_night_background(a, r - big_r, dot(q, d) / r);
                for (uint p = 0; p < passes; ++p)
                    background[p] += throughput[p] * passed[p] * night;
            }
            break;
        }
        if (leg >= REFERENCE_SKY_ROULETTE)
        {
            float survive = min(largest, 1.0);
            state = pcg(state);
            if (!(unit_float(state) < survive))
                break;
            for (uint p = 0; p < passes; ++p)
                throughput[p] /= survive;
        }
    }
    float3 rgb = float3(0.0);
    float3 night = float3(0.0);
    for (uint p = 0; p < passes; ++p)
    {
        AtmosphereGroups g = a.groups[p];
        rgb += atmosphere_rec2020(g, radiance[p]);
        night += atmosphere_rec2020(g, background[p]);
    }
    return rgb + night / a.solar_color.xyz;
}

// The sky as orr's environment link point reads it: the nearest sky-view LUT (the medium
// beyond `near_range`, the folded lights' in-scatter and the virtual ground below the
// horizon) plus the night background, extinguished from `near_range` to space, blended with
// the focus LUT's by the nearest LUT's weight. No light discs. Link it with
// `Build::link(LinkPoint::Environment, ..)`.
public struct AtmosphereSky : IEnvironment
{
    public static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        uint i = atmosphere_sky_view_index(a, atmosphere_altitude(a, eye_render));
        float3 l = sky_view_radiance(a, i, dir);
        float w = a.sky_view_weight[i];
        return w < 1.0 ? lerp(sky_view_radiance(a, 0, dir), l, w) : l;
    }

    // `atmosphere_reference_sky` from the eye's own altitude; where the camera sees along `dir` it
    // integrates the view ray's single scattering.
    public static float3 reference_radiance(uint64_t data, float3 dir, float3 eye_render, uint seed, bool camera)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        return atmosphere_reference_sky(a, max(atmosphere_altitude(a, eye_render), 0.0), dir, seed, camera);
    }
}

// Newton steps `AtmosphereMedium.distance` takes at most, and the error in optical depth, relative
// to the depth sought, at which it stops.
static const uint DISTANCE_ITERATIONS = 32;
static const float DISTANCE_TOLERANCE = 1e-5;

// The planet's medium as orr's medium link point reads it. Altitude is height above the flat
// ground plane and coefficients are the three solar-weighted Rec.2020 bands. The phase has three
// components: fog in `phase_share`, aerosol in `phase_share_second` and Rayleigh, the flattest,
// as the rest, which takes the rounding of the froxels' shares. Toward lights, `transmittance`
// reads the spherical transmittance LUT at the point's altitude. Link it with
// `Build::link(LinkPoint::Medium, ..)`.
public struct AtmosphereMedium : IMedium
{
    public static MediumPoint sample(uint64_t data, float3 p_render)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        float h = atmosphere_altitude(a, p_render);
        float3 d = atmosphere_densities(a, h);
        float3 rayleigh = a.rayleigh.rgb * d.x;
        float3 aerosol = a.aerosol_scattering.rgb * d.y;
        float3 fog = atmosphere_fog_extinction(a, h) * a.fog_albedo.rgb;
        MediumPoint m = {};
        m.sigma_s = rayleigh + aerosol + fog;
        m.sigma_t = atmosphere_extinction(a, h);
        bool3 scatters = m.sigma_s > 0.0;
        m.phase_share = select(scatters, fog / m.sigma_s, float3(0.0));
        m.phase_share_second = select(scatters, aerosol / m.sigma_s, float3(0.0));
        return m;
    }

    public static float3 phase(uint64_t data, MediumPoint m, float cos_theta)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        float3 rest = max(1.0 - m.phase_share - m.phase_share_second, 0.0);
        float3 p = m.phase_share_second * cornette_shanks_phase(a.aerosol_asymmetry, cos_theta)
            + rest * rayleigh_phase(cos_theta);
        if (a.fog_density > 0.0)
            p += m.phase_share * fog_phase(a.fog_diameter, cos_theta);
        return p;
    }

    public static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light, float3 dir)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        // Below the ground plane the horizon is undefined: the table's lowest row stands.
        float h = max(atmosphere_altitude(a, p_render), 0.0);
        float3 up = a.up.xyz;
        return a.multiple_scattering_scale
            * atmosphere_lut_multiple_scattering(a, h, dot(dir_to_light, up), dot(dir, up),
                                                 dot(dir, dir_to_light), a.group_passes);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        if (isinf(t))
            return atmosphere_transmittance_to_space(a, atmosphere_altitude(a, origin), dot(dir, a.up.xyz));
        return exp(-atmosphere_optical_depth(a, origin, dir, t));
    }

    // Newton's method on the closed-form optical depth, kept inside the bracket that bisection
    // narrows: the depth grows monotonically with distance.
    public static float distance(uint64_t data, float3 origin, float3 dir, float t_max, uint channel, float depth)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        float total = atmosphere_optical_depth(a, origin, dir, t_max)[channel];
        if (!(total > depth))
            return positive_infinity();
        float lo = 0.0;
        float hi = t_max;
        float t = t_max * depth / total;
        for (uint i = 0; i < DISTANCE_ITERATIONS; ++i)
        {
            float f = atmosphere_optical_depth(a, origin, dir, t)[channel] - depth;
            if (abs(f) <= DISTANCE_TOLERANCE * depth)
                break;
            if (f > 0.0)
                hi = t;
            else
                lo = t;
            float next = t - f / atmosphere_extinction(a, atmosphere_altitude(a, origin + dir * t))[channel];
            t = next > lo && next < hi ? next : 0.5 * (lo + hi);
        }
        return t;
    }

    // A component by its share of the scattering, averaged over the channels: Rayleigh's own
    // distribution, a Henyey-Greenstein lobe of the aerosol's asymmetry for its Cornette-Shanks
    // function, and the fog's lobes (`sample_fog_phase`).
    public static float3 sample_phase(uint64_t data, MediumPoint m, float3 dir, float3 u, out float pdf)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        float fog = a.fog_density > 0.0 ? (m.phase_share.x + m.phase_share.y + m.phase_share.z) / 3.0 : 0.0;
        float aerosol = (m.phase_share_second.x + m.phase_share_second.y + m.phase_share_second.z) / 3.0;
        float3 wi;
        if (u.z < fog)
        {
            float unused;
            wi = sample_fog_phase(a.fog_diameter, dir, float3(u.xy, u.z / fog), unused);
        }
        else if (u.z < fog + aerosol)
            wi = direction_about(dir, sample_hg_cosine(a.aerosol_asymmetry, u.x), TWO_PI * u.y);
        else
            wi = direction_about(dir, sample_rayleigh_cosine(u.x), TWO_PI * u.y);
        pdf = AtmosphereMedium.phase_pdf(data, m, dot(wi, dir));
        return wi;
    }

    public static float phase_pdf(uint64_t data, MediumPoint m, float cos_theta)
    {
        AtmosphereGpu* a = (AtmosphereGpu*)data;
        float fog = a.fog_density > 0.0 ? (m.phase_share.x + m.phase_share.y + m.phase_share.z) / 3.0 : 0.0;
        float aerosol = (m.phase_share_second.x + m.phase_share_second.y + m.phase_share_second.z) / 3.0;
        float pdf = aerosol * hg_phase(a.aerosol_asymmetry, cos_theta)
            + (1.0 - fog - aerosol) * rayleigh_phase(cos_theta);
        if (fog > 0.0)
            pdf += fog * fog_phase_pdf(a.fog_diameter, cos_theta);
        return pdf;
    }
}
