// Participating media: the medium link point, the stock height-fog functions implementations
// call, the transmittance every lighting segment uses, the froxel volume's layout and
// material encoding, and the aerial-perspective LUT's layout.
module media;

import orr.abi;
import orr.math;
import orr.color;
import orr.packing;

// One point of a medium. Coefficients are per meter in linear Rec.2020.
public struct MediumPoint
{
    public float3 sigma_s;
    public float3 sigma_t;
    // Emitted radiance per meter.
    public float3 emission;
    // The point's phase parameter; fixed parameters live in the medium's data.
    public float phase_parameter;
    // Per channel, the shares of `sigma_s` the first and second phase components hold; a
    // third holds the rest.
    public float3 phase_share;
    public float3 phase_share_second;
}

// A medium linked into orr's kernels. `data` is the address the scene was given for it.
public interface IMedium
{
    static MediumPoint sample(uint64_t data, float3 p_render);
    // Phase function per channel at the cosine between the incoming light's travel direction
    // and the scattered direction.
    static float3 phase(uint64_t data, MediumPoint m, float cos_theta);
    // Multiply-scattered radiance per unit `sigma_s` and per unit illuminance of the light
    // before the medium; 0 when the medium has none.
    static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light);
    // Transmittance over `t` meters from `origin` along unit `dir`; `t` is +inf toward
    // directional lights.
    static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t);
}

// No medium: transmittance 1, nothing scatters or emits.
public struct NoMedium : IMedium
{
    public static MediumPoint sample(uint64_t data, float3 p_render)
    {
        return {};
    }

    public static float3 phase(uint64_t data, MediumPoint m, float cos_theta)
    {
        return float3(0.0);
    }

    public static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light)
    {
        return float3(0.0);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        return float3(1.0);
    }
}

// Transmittance of the linked medium `M` over a segment; `t` is +inf toward directional
// lights.
public float3 medium_transmittance<M : IMedium>(uint64_t data, float3 origin, float3 dir, float t)
{
    if (!(t > 0.0))
        return float3(1.0);
    return M.transmittance(data, origin, dir, t);
}

public float positive_infinity()
{
    return asfloat(0x7f800000u);
}

// Stock height fog. Extinction is `density * exp(-(h - base_height) / falloff_height)` at
// altitude `h`, equal at every wavelength and unclamped below `base_height`; scattering is
// extinction times an rgb albedo.

public float height_fog_extinction(float density, float base_height, float falloff_height, float h)
{
    return density * exp(-(h - base_height) / falloff_height);
}

// Optical depth of height fog over `t` meters from altitude `h0`, where `dir_y` is the
// altitude change per meter. `t` may be +inf.
public float height_fog_optical_depth(float density, float base_height, float falloff_height,
                                      float h0, float dir_y, float t)
{
    float start = height_fog_extinction(density, base_height, falloff_height, h0);
    float k = dir_y / falloff_height;
    if (isinf(t))
        return k > 0.0 ? start / k : (start > 0.0 ? positive_infinity() : 0.0);
    float x = k * t;
    if (abs(x) < 1e-3)
        return start * t * (1.0 - x * (0.5 - x / 6.0));
    // The difference of the two ends: far above the layer the start is 0 and exp(-x) infinite,
    // a NaN product. Far below it both ends overflow, and the depth with them.
    if (isinf(start))
        return start;
    float end = height_fog_extinction(density, base_height, falloff_height, h0 + dir_y * t);
    return (start - end) / k;
}

public float height_fog_transmittance(float density, float base_height, float falloff_height,
                                      float h0, float dir_y, float t)
{
    return exp(-height_fog_optical_depth(density, base_height, falloff_height, h0, dir_y, t));
}

// Henyey-Greenstein phase function, per steradian.
public float hg_phase(float g, float cos_theta)
{
    float d = 1.0 + g * g - 2.0 * g * cos_theta;
    return INV_FOUR_PI * (1.0 - g * g) / (d * sqrt(d));
}

// A direction scattered from travel direction `d` (unit) by the Henyey-Greenstein phase
// function of anisotropy `g`, from two uniform numbers; its density is `hg_phase(g, cos)`.
public float3 sample_hg(float g, float3 d, float2 u)
{
    float c;
    if (abs(g) < 1e-3)
        c = 1.0 - 2.0 * u.x;
    else
    {
        float s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u.x);
        c = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
    }
    float sine = sqrt(max(0.0, 1.0 - c * c));
    float phi = TWO_PI * u.y;
    float3 t, b;
    orthonormal_basis(d, t, b);
    return normalize(t * (sine * cos(phi)) + b * (sine * sin(phi)) + d * c);
}

// Draine's phase function, per steradian.
public float draine_phase(float g, float alpha, float cos_theta)
{
    return hg_phase(g, cos_theta) * (1.0 + alpha * cos_theta * cos_theta)
        / (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0);
}

// Cloud and fog droplet phase function for droplet diameter `diameter_um` (clamped to
// 5-50 um): `(1 - w_D) HG(g_HG) + w_D Draine(g_D, alpha)`, with Jendersie and d'Eon's (2023)
// fit of Mie scattering.
public float fog_phase(float diameter_um, float cos_theta)
{
    float d = clamp(diameter_um, 5.0, 50.0);
    float g_hg = exp(-0.0990567 / (d - 1.67154));
    float g_d = exp(-2.20679 / (d + 3.91029) - 0.428934);
    float alpha = exp(3.62489 - 8.29288 / (d + 5.52825));
    float w_d = exp(-0.599085 / (d - 0.641583) - 0.665888);
    return (1.0 - w_d) * hg_phase(g_hg, cos_theta) + w_d * draine_phase(g_d, alpha, cos_theta);
}

// One `dt` m step of a medium march with extinction `sigma_t` and `source` radiance per m:
// adds its in-scatter, seen through `transmittance`, to `inscatter`, then its transmittance
// to `transmittance`.
public void march_step(float3 sigma_t, float3 source, float dt, inout float3 inscatter, inout float3 transmittance)
{
    float3 step = exp(-sigma_t * dt);
    inscatter += transmittance * source * select(sigma_t > 0.0, (1.0 - step) / max(sigma_t, 1e-30), float3(dt));
    transmittance *= step;
}

// Froxels: a view's volume of ceil(width / FROXEL_TILE) x ceil(height / FROXEL_TILE) columns
// along unjittered rays spread evenly from one edge of the image to the other, each cut
// into `slices` shells of distance from the eye to the scene's `near_range`, squared toward
// the far end: slice coordinate c in [0, slices] lies at near_range (c / slices)^2.
public static const uint FROXEL_TILE = 8;

// Most directional lights the froxels trace shadow rays toward.
public static const uint MAX_FROXEL_SHADOWED_LIGHTS = 4;

// `froxel_material` holds coefficients times this, so clear air keeps f16 precision; dense
// media up to 16 /m fit.
public static const float FROXEL_SIGMA_SCALE = 4096.0;

// The directional lights a view's froxels trace shadow rays toward, as LightIds, in the
// order their visibility is stored.
public struct FroxelLights
{
    public uint ids[MAX_FROXEL_SHADOWED_LIGHTS];
    public uint count;
    // 1 when the lights differ from the last frame's: their visibility history restarts.
    public uint changed;
    // Medium proxies in the view's frustum this frame; 0 spares media apply its ray per pixel.
    public uint proxies;
    // 1 when a visible smoke triangle reaches past `near_range` this frame; 0 spares media apply
    // its ray per pixel for smoke there.
    public uint far_smoke;
}

// One frame's froxel volume of a view, for passes that read it after its integration.
public struct FroxelVolume
{
    public FroxelLights* lights;
    public StorageImageIndex sigma;             // froxel_material, RGBA16F
    public StorageImageIndex words;             // froxel_material, RGBA32_UINT
    public StorageImageIndex incident;          // RGBA16F
    public StorageImageIndex visibility;        // shadowed lights past the first, one channel each
    public uint columns;
    public uint rows;
    public uint slices;
    public uint chromatic;
}

// Render pixels between neighboring column rays of a `grid` of columns over an `extent`
// image. The first and last rays of each axis lie on the image's edges.
public float2 froxel_spacing(uint2 extent, uint2 grid)
{
    return float2(extent) / max(float2(grid) - 1.0, 1.0);
}

// Distance from the eye of slice coordinate `c`.
public float froxel_distance(float c, uint slices, float near_range)
{
    float u = c / float(slices);
    return near_range * u * u;
}

// Slice coordinate of distance `t`, clamped to the volume.
public float froxel_slice(float t, uint slices, float near_range)
{
    return float(slices) * sqrt(saturate(t / near_range));
}

// The aerial-perspective LUT: AERIAL_SIZE x AERIAL_SIZE columns along unjittered rays spread
// evenly from one image edge to the other, as the froxels' are, each cut into AERIAL_SIZE
// shells of distance from the eye between the scene's `near_range` and the view's
// `ap_range`, squared toward the far end: slice coordinate c in [0, AERIAL_SIZE] lies at
// near_range + (ap_range - near_range) (c / AERIAL_SIZE)^2.
public static const uint AERIAL_SIZE = 32;

// Distance from the eye of aerial slice coordinate `c`.
public float aerial_distance(float c, float near_range, float ap_range)
{
    float u = c / float(AERIAL_SIZE);
    return near_range + (ap_range - near_range) * u * u;
}

// Aerial slice coordinate of distance `t`, clamped to the LUT.
public float aerial_slice(float t, float near_range, float ap_range)
{
    return float(AERIAL_SIZE) * sqrt(saturate((t - near_range) / (ap_range - near_range)));
}

// A froxel's medium: `sigma` (RGBA16F) holds sigma_s rgb and the luminance of sigma_t, both
// times FROXEL_SIGMA_SCALE; `words` (RGBA32_UINT) the emission as rgb9e5, the phase
// parameter as f16, and the two share triples at 10 bits per channel.
public struct FroxelMaterial
{
    public float4 sigma;
    public uint4 words;
}

uint pack_shares(float3 s)
{
    uint3 q = uint3(round(saturate(s) * 1023.0));
    return q.x | (q.y << 10) | (q.z << 20);
}

float3 unpack_shares(uint p)
{
    return float3(uint3(p, p >> 10, p >> 20) & 0x3FF) / 1023.0;
}

public FroxelMaterial froxel_material(MediumPoint m)
{
    FroxelMaterial f;
    f.sigma = float4(m.sigma_s, luminance_rec2020(m.sigma_t)) * FROXEL_SIGMA_SCALE;
    f.words = uint4(pack_rgb9e5(m.emission), f32tof16(m.phase_parameter),
                    pack_shares(m.phase_share), pack_shares(m.phase_share_second));
    return f;
}

// The point a froxel's material describes. Its extinction is gray, the stored luminance;
// with `chromatic`, sigma_s per channel plus gray absorption, the luminance's excess over
// sigma_s's.
public MediumPoint froxel_point(FroxelMaterial f, bool chromatic)
{
    MediumPoint m;
    m.sigma_s = f.sigma.rgb / FROXEL_SIGMA_SCALE;
    float sigma_t = f.sigma.a / FROXEL_SIGMA_SCALE;
    m.sigma_t = chromatic ? m.sigma_s + max(0.0, sigma_t - luminance_rec2020(m.sigma_s)) : float3(sigma_t);
    m.emission = unpack_rgb9e5(f.words.x);
    m.phase_parameter = f16tof32(f.words.y);
    m.phase_share = unpack_shares(f.words.z);
    m.phase_share_second = unpack_shares(f.words.w);
    return m;
}
