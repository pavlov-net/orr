// Light selection and sampling. Directional lights are a reserved share of every vertex's
// candidates, weighted by n.l; the scene's other lights (sphere and rect lights, emissive
// entries and the baked environment) are drawn from a selection CDF into presampled light
// tiles each frame, and vertices take candidates from the tiles. The selection blends
// emitted power with the light feedback of earlier frames, and a view's first vertices may
// also draw from the lights that fed back in their screen tile.
module light_sampling;

import orr.abi;
import orr.math;
import orr.color;
import orr.packing;
import orr.random;
import lights;
import material;
import producer;
import tables;
import view;
import media;
import environment;

// Presampled light tiles: `TILE_COUNT` tiles of `TILE_SIZE` samples, redrawn every frame.
public static const uint TILE_COUNT = 128;
public static const uint TILE_SIZE = 1024;
// Pixels per side of the screen blocks that share one tile in a frame.
public static const uint TILE_BLOCK = 8;
// A block's tile moves by this many tiles each frame; odd, so a block visits every tile.
public static const uint TILE_FRAME_STEP = 37;

// Share of a vertex's candidates taken from the directional lights, when at least one of
// them lights the vertex and the tiles hold other lights.
public static const float DIRECTIONAL_SHARE = 0.5;
// Share of tile samples drawn from the environment when other lights are present.
public static const float ENVIRONMENT_SHARE = 0.25;

// Each frame the selection's feedback weights keep this share of their value and add the
// pixel counts of the frame before.
public static const float FEEDBACK_HISTORY = 0.5;
// Pixels per side of the screen tiles that share one local sampler.
public static const uint LOCAL_TILE = 16;
// Lights one local sampler holds.
public static const uint LOCAL_LIGHTS = 8;

// `LightSample` kinds (bits 0-1), and the kind of a directional candidate.
public static const uint LIGHT_SAMPLE_ENVIRONMENT = 0;
public static const uint LIGHT_SAMPLE_SPHERE = 1;
public static const uint LIGHT_SAMPLE_RECT = 2;
public static const uint LIGHT_SAMPLE_EMISSIVE = 3;
public static const uint LIGHT_SAMPLE_DIRECTIONAL = 4;
// `LightSample.bits`: the light emits from both faces.
public static const uint LIGHT_SAMPLE_TWO_SIDED = 4;

// What lighting's scene pass derives per directional light each frame.
public struct DirectionalState
{
    // Illuminance at the scene focus after the medium.
    public float3 focus_illuminance;
    // Luminance of `focus_illuminance`: the light's selection weight before n.l.
    public float weight;
    // Multiplies the disc profile's shape to give its radiance.
    public float3 radiance_scale;
    public float _pad0;
    // `DISC_TEXTURED`: `textured_disc_integral`; 0 otherwise.
    public float3 integral;
    public float _pad1;
    // The medium's transmittance from the focus toward the light.
    public float3 transmittance;
    public float _pad2;
}

// Selection state of one emissive list entry.
public struct EmissiveSelection
{
    // Probability that a tile sample comes from this entry.
    public float probability;
    // The entry's cluster CDF: `count` values from `LightSetGpu.cluster_cdf[first]`,
    // inclusive sums of luminance x area per cluster, ending at `total`.
    public uint first;
    public uint count;
    public float total;
}

// One presampled light sample, 32 B, in render space. Its density includes the selection
// probability: per m² of the light for rect and emissive samples, per sr (stored negated)
// for the environment; a sphere sample holds the sphere, whose point is drawn at resolve.
public struct LightSample
{
    // Environment: unit direction. Sphere: centre. Rect, emissive: a point on the light.
    public float3 point;
    // Octahedral 2x16 snorm: the surface normal (rect, emissive) or spot axis (sphere).
    public uint normal;
    // Emitted radiance, nits: shared-exponent rgb (`pack_rgb9e5`), times 2^(bits 8-15 - 128).
    public uint radiance;
    // Environment: -(probability per sr). Rect, emissive: probability per m². Sphere: the
    // light's selection probability. 0: no sample.
    public float pdf;
    // The light's feedback id (`feedback_environment`).
    public uint light;
    // Bits 0-1: LIGHT_SAMPLE_*. Bit 2: LIGHT_SAMPLE_TWO_SIDED. Bits 8-15: the radiance
    // exponent + 128. Bits 16-31: the sphere radius, half float.
    public uint bits;
}

// The scene's light set as lighting's scene pass leaves it each frame.
public struct LightSetGpu
{
    public SceneRoot* scene;
    public EnvironmentGpu* environment;
    // By LightId; valid for the lights in `directional_ids`.
    public DirectionalState* directional;
    // The scene's directional lights in LightId order.
    public uint* directional_ids;
    // Selection weights over the candidates (sphere and rect lights by LightId, then the
    // emissive list), as inclusive sums.
    public float* cdf;
    // By LightId: probability that a tile sample comes from that sphere or rect light.
    public float* light_probability;
    // By emissive list entry.
    public EmissiveSelection* emissive;
    public float* cluster_cdf;
    public LightSample* tiles;
    // Per feedback id, the pixels whose feedback named it this frame; the next scene pass
    // reads and clears them.
    public uint* feedback_counts;
    public uint directional_count;
    public uint candidate_count;
    // The last value of `cdf`.
    public float candidate_total;
    // Probability that a tile sample comes from the environment.
    public float environment_probability;
    // rgb: the summed `focus_illuminance` of every directional light.
    public float4 directional_illuminance;
    public uint frame;
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// Counts and statistics of one frame's scene pass, read back by the CPU.
public struct LightSetStats
{
    public SkyStats sky;
    public uint directional_lights;
    // Sphere and rect lights with non-zero power.
    public uint local_lights;
    // Emissive entries with a non-zero selection probability.
    public uint emissive_entries;
    // Clusters in their CDFs.
    public uint emissive_clusters;
    // Emitting entries left out of the selection because the cluster pool was full.
    public uint dropped_entries;
    public float environment_probability;
    // Pixels whose feedback named a light in the frame before.
    public uint feedback_pixels;
    public uint _pad1;
}

// The lights that fed back in one screen tile of a view in its last frame, as a selection.
public struct LocalLights
{
    // Feedback ids.
    public uint ids[LOCAL_LIGHTS];
    // Inclusive sums of the pixels that named each.
    public float cdf[LOCAL_LIGHTS];
    public uint count;
    // Probability that a tile-light candidate of the tile's first vertices comes from here.
    public float share;
    public uint _pad0;
    public uint _pad1;
}

// A receiver of next-event estimation.
public struct NeeVertex
{
    // Render space.
    public float3 position;
    // The shading normal; zero for a point in a medium, which every direction lights.
    public float3 normal;
    // The pixel the path belongs to; it picks the tile.
    public uint2 pixel;
    // The local sampler of the pixel's screen tile, or null.
    public LocalLights* local;
}

// One light candidate at a vertex, before visibility.
public struct ResolvedLightSample
{
    // Unit direction from the vertex toward the light.
    public float3 dir;
    // Distance to the light's surface; +inf for the environment and directional lights.
    public float distance;
    // Incident radiance along `dir`: 0 when the light faces away; a directional light's
    // carries the medium's transmittance at the scene focus.
    public float3 radiance;
    // Density per sr of this candidate in the vertex's mixture; 0 when there is none.
    public float pdf;
    // LIGHT_SAMPLE_*.
    public uint kind;
    // LIGHT_SAMPLE_DIRECTIONAL: the light's LightId; otherwise its feedback id, or NO_LIGHT
    // without a candidate.
    public uint light;
}

// Root of the scene pass over directional lights.
public struct DirectionalRoot
{
    public SceneRoot* scene;
    public DirectionalState* states;
    public uint* ids;
    // The pass writes `directional_count` and `directional_illuminance`.
    public LightSetGpu* set;
    // The scene focus in render space.
    public float4 focus;
}

// 1 - cos and sin of a cone's half angle, accurate for small cones.
struct Cone
{
    float one_minus_cos;
    float sin_max;

    __init(float angular_radius)
    {
        float h = sin(0.5 * angular_radius);
        one_minus_cos = 2.0 * h * h;
        sin_max = sin(angular_radius);
    }
}

// A direction drawn uniformly from the cone around unit `axis` with 1 - cos of its half
// angle `one_minus_cos`.
float3 cone_direction(float3 axis, float one_minus_cos, float2 u)
{
    float one_minus_z = u.x * one_minus_cos;
    float r = sqrt(max(0.0, one_minus_z * (2.0 - one_minus_z)));
    float phi = TWO_PI * u.y;
    float3 t, b;
    orthonormal_basis(axis, t, b);
    return normalize(from_local(float3(r * cos(phi), r * sin(phi), 1.0 - one_minus_z), t, b, axis));
}

// Squared distance from the disc centre of direction `dir`, on the unit disc, or a value above
// 1 outside the disc.
float disc_r2(LightGpu l, float3 dir)
{
    float3 s = cross(l.direction, dir);
    float sin_max = Cone(l.radius).sin_max;
    return dot(l.direction, dir) > 0.0 ? dot(s, s) / (sin_max * sin_max) : 2.0;
}

// Neckel and Labs (1994) fifth-order limb darkening at 680, 550 and 440 nm for Rec.2020's
// red, green and blue, centre intensity 1, at `mu` = cos of the angle from the disc normal.
float3 limb_darkening(float mu)
{
    float3 a0 = float3(0.34685, 0.26073, 0.15248);
    float3 a1 = float3(1.37539, 1.27428, 1.38517);
    float3 a2 = float3(-2.04425, -1.30352, -1.49615);
    float3 a3 = float3(2.70493, 1.47085, 1.99886);
    float3 a4 = float3(-1.94290, -0.96618, -1.48155);
    float3 a5 = float3(0.55999, 0.26384, 0.44119);
    return a0 + mu * (a1 + mu * (a2 + mu * (a3 + mu * (a4 + mu * a5))));
}

// The disc mean of `limb_darkening`, 2 x integral of I(mu) mu dmu.
static const float3 LIMB_DARKENING_MEAN = float3(0.835987476, 0.800152857, 0.759599952);

// Right and up axes of a disc as seen from a receiver looking along its direction.
void disc_frame(LightGpu l, out float3 right, out float3 up)
{
    float3 axis = l.direction;
    up = l.up - dot(l.up, axis) * axis;
    if (dot(up, up) > 1e-12)
    {
        up = normalize(up);
        right = cross(axis, up);
    }
    else
    {
        orthonormal_basis(axis, right, up);
    }
}

// Unscaled radiance of a `Textured` disc at point `p` of the unit disc (+x right, +y up):
// albedo x the sum over its `lit_by` lights of E_i max(0, n.l_i) / pi, with n the sphere
// normal there.
float3 textured_shape(SceneRoot* scene, LightGpu l, float3 right, float3 up, float2 p)
{
    float mu = sqrt(max(0.0, 1.0 - dot(p, p)));
    float3 n = p.x * right + p.y * up - mu * l.direction;
    float3 lit = float3(0.0);
    for (uint i = 0; i < 4; i++)
    {
        uint j = l.lit_by[i];
        if (j >= scene.light_capacity)
            continue;
        LightGpu s = scene.lights[j];
        if (s.kind == LIGHT_DIRECTIONAL && s.disc != DISC_TEXTURED)
            lit += s.color * max(0.0, dot(n, s.direction));
    }
    if (l.albedo.index == 0xffffffffu)
        return lit * INV_PI;
    Texture2D<float4> albedo = l.albedo.get<Texture2D<float4>>();
    uint2 size;
    albedo.GetDimensions(size.x, size.y);
    float2 q = float2(0.5 + 0.5 * p.x, 0.5 - 0.5 * p.y) * float2(size) - 0.5;
    int2 i0 = int2(floor(q));
    float2 f = q - float2(i0);
    int2 last = int2(size) - 1;
    float3 a00 = albedo.Load(int3(clamp(i0, 0, last), 0)).rgb;
    float3 a10 = albedo.Load(int3(clamp(i0 + int2(1, 0), 0, last), 0)).rgb;
    float3 a01 = albedo.Load(int3(clamp(i0 + int2(0, 1), 0, last), 0)).rgb;
    float3 a11 = albedo.Load(int3(clamp(i0 + int2(1, 1), 0, last), 0)).rgb;
    float3 a = lerp(lerp(a00, a10, f.x), lerp(a01, a11, f.x), f.y);
    return a * lit * INV_PI;
}

// The fixed 64-point disc pattern: 4 rings at the Gauss-Legendre nodes in r^2 (equal area),
// 16 angles each, every other ring turned by half a step. Returns the point and its weight;
// the weights sum to 1.
float2 disc_point(uint k, out float weight)
{
    static const float RING_R2[4] = { 0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558 };
    static const float RING_WEIGHT[4] = { 0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226 };
    uint ring = k >> 4;
    float phi = TWO_PI * (float(k & 15) + 0.5 * float(ring & 1)) / 16.0;
    weight = RING_WEIGHT[ring] / 16.0;
    return sqrt(RING_R2[ring]) * float2(cos(phi), sin(phi));
}

// Illuminance the `Textured` directional light `index` of the scene emits before the medium
// when its illuminance is zero: the integral of its unscaled radiance over the disc's cone,
// weighted by the cosine to its axis, over the 64-point pattern. Lighting's scene pass calls
// this; scene compute kernels call it for the same value.
public float3 textured_disc_integral(SceneRoot* scene, uint index)
{
    LightGpu l = scene.lights[index];
    float3 right, up;
    disc_frame(l, right, up);
    float3 sum = float3(0.0);
    for (uint k = 0; k < 64; k++)
    {
        float weight;
        float2 p = disc_point(k, weight);
        sum += weight * textured_shape(scene, l, right, up, p);
    }
    float sin_max = Cone(l.radius).sin_max;
    return PI * sin_max * sin_max * sum;
}

// Radiance scale per channel of a light's disc, so that the cosine-weighted integral of its
// radiance over the cone is its illuminance; for a `Textured` disc with zero illuminance the
// scale is 1, so the disc emits `integral`.
float3 radiance_scale(LightGpu l, float3 integral)
{
    float sin_max = Cone(l.radius).sin_max;
    float projected = PI * sin_max * sin_max;
    switch (l.disc)
    {
    case DISC_LIMB_DARKENED:
        return l.color / (projected * LIMB_DARKENING_MEAN);
    case DISC_TEXTURED:
        if (all(l.color == 0.0))
            return float3(1.0);
        return select(integral > 0.0, l.color / integral, float3(0.0));
    default:
        return l.color / projected;
    }
}

// Illuminance a directional light emits before the medium, with `integral` its disc's
// `textured_disc_integral`: the integral for a `Textured` light of zero illuminance, else its
// illuminance.
float3 emitted_illuminance(LightGpu l, float3 integral)
{
    return l.disc == DISC_TEXTURED && all(l.color == 0.0) ? integral : l.color;
}

// The state of directional light `index` at the focus, with `transmittance` the medium's
// toward it.
public DirectionalState directional_state(SceneRoot* scene, uint index, float3 transmittance)
{
    LightGpu l = scene.lights[index];
    DirectionalState s = {};
    s.integral = l.disc == DISC_TEXTURED ? textured_disc_integral(scene, index) : float3(0.0);
    s.radiance_scale = radiance_scale(l, s.integral);
    s.focus_illuminance = emitted_illuminance(l, s.integral) * transmittance;
    s.weight = luminance_rec2020(s.focus_illuminance);
    s.transmittance = transmittance;
    return s;
}

static groupshared float3 directional_sums[64];
static groupshared uint directional_flags[64];

// The scene pass over directional lights for one 64-lane workgroup: every directional
// light's state, their LightIds in order, and their summed focus illuminance.
public void directional_pass<M : IMedium>(DirectionalRoot r, uint lane)
{
    SceneRoot* s = r.scene;
    float3 sum = float3(0.0);
    uint count = 0;
    for (uint base = 0; base < s.light_capacity; base += 64)
    {
        uint i = base + lane;
        bool directional = i < s.light_capacity && s.lights[i].kind == LIGHT_DIRECTIONAL;
        if (directional)
        {
            float3 dir = s.lights[i].direction;
            float3 t = medium_transmittance<M>(s.medium, r.focus.xyz, dir, positive_infinity());
            DirectionalState st = directional_state(s, i, t);
            r.states[i] = st;
            sum += st.focus_illuminance;
        }
        directional_flags[lane] = directional ? 1 : 0;
        GroupMemoryBarrierWithGroupSync();
        uint before = 0;
        uint chunk = 0;
        for (uint k = 0; k < 64; k++)
        {
            before += k < lane ? directional_flags[k] : 0;
            chunk += directional_flags[k];
        }
        if (directional)
            r.ids[count + before] = i;
        count += chunk;
        GroupMemoryBarrierWithGroupSync();
    }
    directional_sums[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 32; k > 0; k >>= 1)
    {
        if (lane < k)
            directional_sums[lane] += directional_sums[lane + k];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
    {
        r.set.directional_count = count;
        r.set.directional_illuminance = float4(directional_sums[0], 0.0);
    }
}

// Radiance of directional light `id`'s disc in direction `dir`, before the medium; 0 outside
// it.
public float3 disc_radiance(LightSetGpu* set, uint id, float3 dir)
{
    LightGpu l = set.scene.lights[id];
    float r2 = disc_r2(l, dir);
    if (r2 > 1.0)
        return float3(0.0);
    float3 shape;
    switch (l.disc)
    {
    case DISC_LIMB_DARKENED:
        shape = limb_darkening(sqrt(max(0.0, 1.0 - r2)));
        break;
    case DISC_TEXTURED:
    {
        float3 right, up;
        disc_frame(l, right, up);
        float2 p = float2(dot(dir, right), dot(dir, up)) / Cone(l.radius).sin_max;
        shape = textured_shape(set.scene, l, right, up, p);
        break;
    }
    default:
        shape = float3(1.0);
        break;
    }
    return shape * set.directional[id].radiance_scale;
}

// Radiance of directional light `id`'s disc in direction `dir` reaching `position` through the
// linked medium `M`. Next-event estimation and BSDF hits both call this.
public float3 directional_disc<M : IMedium>(LightSetGpu* set, uint id, float3 position, float3 dir)
{
    float3 l = disc_radiance(set, id, dir);
    if (all(l == 0.0))
        return l;
    return l * medium_transmittance<M>(set.scene.medium, position, dir, positive_infinity());
}

// A direction drawn uniformly from directional light `l`'s cone, with its density per sr.
public float3 directional_sample(LightGpu l, float2 u, out float pdf)
{
    Cone c = Cone(l.radius);
    pdf = 1.0 / (TWO_PI * c.one_minus_cos);
    return cone_direction(l.direction, c.one_minus_cos, u);
}

// Density of `directional_sample` for light `l` in direction `dir`.
public float directional_pdf(LightGpu l, float3 dir)
{
    return disc_r2(l, dir) <= 1.0 ? 1.0 / (TWO_PI * Cone(l.radius).one_minus_cos) : 0.0;
}

// Radiance of a primary sky pixel looking along `dir` from `eye`: the linked environment
// `E` plus every directional disc through the linked medium `M`.
public float3 sky_radiance<E : IEnvironment, M : IMedium>(LightSetGpu* set, float3 eye, float3 dir)
{
    float3 l = E.radiance(set.scene.environment, dir, eye);
    for (uint k = 0; k < set.directional_count; k++)
        l += directional_disc<M>(set, set.directional_ids[k], eye, dir);
    return l;
}

// Directional light `id`'s weight at a vertex with shading normal `normal`.
public float directional_weight(LightSetGpu* set, uint id, float3 normal)
{
    float w = set.directional[id].weight;
    if (any(normal != 0.0))
        w *= max(0.0, dot(normal, set.scene.lights[id].direction));
    return w;
}

// Share of a vertex's candidates drawn from directional lights, and the sum of their
// weights there.
public float directional_share(LightSetGpu* set, float3 normal, out float total)
{
    total = 0.0;
    for (uint k = 0; k < set.directional_count; k++)
        total += directional_weight(set, set.directional_ids[k], normal);
    if (!(total > 0.0))
        return 0.0;
    bool tiles = set.environment_probability > 0.0 || set.candidate_total > 0.0;
    return tiles ? DIRECTIONAL_SHARE : 1.0;
}

// Probability that a vertex with `normal` draws directional light `id`, among all its
// candidates.
public float directional_probability(LightSetGpu* set, float3 normal, uint id)
{
    float total;
    float share = directional_share(set, normal, total);
    return share > 0.0 ? share * directional_weight(set, id, normal) / total : 0.0;
}

// Shared-exponent radiance and its exponent byte (`LightSample.bits` 8-15).
uint2 encode_radiance(float3 c)
{
    float m = max(c.r, max(c.g, c.b));
    if (!(m > 0.0) || !isfinite(m))
        return uint2(0, 128);
    int e = int((asuint(m) >> 23) & 0xff) - 127;
    int k = clamp(e - 14, -128, 127);
    return uint2(pack_rgb9e5(c * exp2(float(-k))), uint(k + 128));
}

public float3 decode_radiance(LightSample s)
{
    int k = int((s.bits >> 8) & 0xff) - 128;
    return unpack_rgb9e5(s.radiance) * exp2(float(k));
}

LightSample light_sample(uint kind, uint light, float3 point, float3 normal, float3 radiance, float pdf)
{
    uint2 r = encode_radiance(radiance);
    LightSample s;
    s.point = point;
    s.normal = oct_encode_snorm16(normal);
    s.radiance = r.x;
    s.pdf = pdf;
    s.light = light;
    s.bits = kind | (r.y << 8);
    return s;
}

// Spot falloff at `c`, the cosine between the spot axis and the direction from the light;
// full within the inner angle, none beyond the outer.
float spot_falloff(float c, float cos_inner, float cos_outer)
{
    if (cos_outer <= -1.0)
        return 1.0;
    if (cos_inner <= cos_outer)
        return c >= cos_outer ? 1.0 : 0.0;
    return smoothstep(cos_outer, cos_inner, c);
}

// Producer entry `entry`'s transform: the anchor of its group in render space and its
// [R | t] relative to that anchor. False when its group hides it.
public bool producer_transform(SceneRoot* s, uint entry, out float3 anchor, out float3x4 m)
{
    ProducerGpu p = s.producers[entry];
    GroupGpu* g = s.groups + p.group;
    anchor = float3(0.0);
    m = float3x4(0.0);
    if (g.current == nullptr || p.group_index >= g.count)
        return false;
    m = group_transform(g, p.group_index);
    anchor = view_relative(g.anchor_cell, g.anchor_offset, s.origin_cell.xyz, float3(0.0),
                           uint(s.origin_cell.w));
    return true;
}

// The material of producer cluster `k`, or null when its index is out of range.
public MaterialGpu* cluster_material(SceneRoot* s, ProducerGpu p, ProducerClusterGpu k)
{
    if (k.material >= p.material_count)
        return nullptr;
    return s.materials + p.materials[k.material];
}

// The front-face normal of triangle p0 p1 p2 under `m`, at twice the triangle's area. Taken
// from transformed edges, so it keeps its precision far from the render origin.
public float3 transformed_normal(float3x4 m, float3 p0, float3 p1, float3 p2)
{
    return cross(mul(m, float4(p1 - p0, 0.0)), mul(m, float4(p2 - p0, 0.0)));
}

// Render-space position of `p` under producer transform `anchor`, `m`.
public float3 render_point(float3 anchor, float3x4 m, float3 p)
{
    return anchor + mul(m, float4(p, 1.0));
}

// Render-space area of producer cluster `k` under `m`.
float cluster_area(ProducerBatch b, ProducerClusterGpu k, float3x4 m)
{
    float area = 0.0;
    for (uint i = 0; i < k.tri_count; i++)
    {
        float3 p0, p1, p2;
        cluster_triangle(b, k, i, p0, p1, p2);
        area += 0.5 * length(transformed_normal(m, p0, p1, p2));
    }
    return area;
}

// Luminance x render-space area of cluster `c` of producer entry `entry`'s batch.
public float producer_cluster_weight(SceneRoot* s, uint entry, uint c)
{
    float3 anchor;
    float3x4 m;
    if (!producer_transform(s, entry, anchor, m))
        return 0.0;
    ProducerBatch b = s.batches[entry];
    ProducerClusterGpu k = b.clusters[c];
    MaterialGpu* mat = cluster_material(s, s.producers[entry], k);
    if (mat == nullptr)
        return 0.0;
    float y = luminance_rec2020(material_emission(mat));
    return y > 0.0 ? y * cluster_area(b, k, m) : 0.0;
}

// The first of `count` inclusive sums from `cdf` above `target`, which lies below the last.
uint upper_bound(float* cdf, uint count, float target)
{
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        if (cdf[mid] > target)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

// `x` x `total`, kept below `total`.
float below(float x, float total)
{
    return min(x * total, asfloat(asuint(total) - 1));
}

// Density per m² of tile samples on emissive entry `entry` at a surface emitting
// `luminance` nits.
public float emissive_area_pdf(LightSetGpu* set, uint entry, float luminance)
{
    EmissiveSelection e = set.emissive[entry];
    return e.total > 0.0 ? e.probability * luminance / e.total : 0.0;
}

// Sample of emissive entry `entry` from `u`, the entry chosen with `probability`: a cluster
// by its CDF, a triangle by area and a point uniform in the triangle. A triangle's front
// face winds counter-clockwise.
LightSample emissive_sample(LightSetGpu* set, uint entry, float4 u, float probability)
{
    SceneRoot* s = set.scene;
    EmissiveSelection e = set.emissive[entry];
    LightSample none = {};
    float3 anchor;
    float3x4 m;
    if (!(e.total > 0.0) || e.count == 0 || !producer_transform(s, entry, anchor, m))
        return none;
    uint c = upper_bound(set.cluster_cdf + e.first, e.count, below(u.x, e.total));
    ProducerBatch b = s.batches[entry];
    ProducerClusterGpu k = b.clusters[c];
    MaterialGpu* mat = cluster_material(s, s.producers[entry], k);
    float area = cluster_area(b, k, m);
    if (mat == nullptr || !(area > 0.0))
        return none;
    float target = below(u.y, area);
    float3 p0, p1, p2, n;
    float sum = 0.0;
    for (uint i = 0; i < k.tri_count; i++)
    {
        float3 q0, q1, q2;
        cluster_triangle(b, k, i, q0, q1, q2);
        float3 ni = transformed_normal(m, q0, q1, q2);
        float a = 0.5 * length(ni);
        if (!(a > 0.0))
            continue;
        p0 = q0;
        p1 = q1;
        p2 = q2;
        n = ni;
        if (target < sum + a)
            break;
        sum += a;
    }
    float su = sqrt(u.z);
    float b0 = 1.0 - su;
    float b1 = u.w * su;
    float3 p = b0 * p0 + b1 * p1 + (1.0 - b0 - b1) * p2;
    float3 l = material_emission(mat);
    LightSample r = light_sample(LIGHT_SAMPLE_EMISSIVE, s.light_capacity + entry, render_point(anchor, m, p),
                                 normalize(n), l, probability * luminance_rec2020(l) / e.total);
    if ((mat.flags & MATERIAL_DOUBLE_SIDED) != 0)
        r.bits |= LIGHT_SAMPLE_TWO_SIDED;
    return r;
}

// A sphere light's radius as tile samples store it.
float sample_radius(float radius)
{
    return unpack_half2x16(pack_half2x16(float2(radius, 0.0))).x;
}

// Sample of sphere or rect light `id` from `u`, the light chosen with `probability`.
LightSample local_sample(LightSetGpu* set, uint id, float2 u, float probability)
{
    LightGpu l = set.scene.lights[id];
    if (l.kind == LIGHT_RECT)
    {
        float3 n = cross(l.edge_x, l.edge_y);
        float area = length(n);
        float3 x = l.position + (u.x - 0.5) * l.edge_x + (u.y - 0.5) * l.edge_y;
        return light_sample(LIGHT_SAMPLE_RECT, id, x, n / area, l.color, probability / area);
    }
    float r = sample_radius(l.radius);
    float3 axis = l.cone_cos_outer > -1.0 ? l.direction : float3(0.0, 0.0, 1.0);
    LightSample s = light_sample(LIGHT_SAMPLE_SPHERE, id, l.position, axis, l.color / (PI * r * r), probability);
    s.bits |= (pack_half2x16(float2(r, 0.0)) & 0xffff) << 16;
    return s;
}

// The environment's feedback id. Sphere and rect lights take their LightId, and emissive
// entries the light capacity plus their entry.
public uint feedback_environment(LightSetGpu* set)
{
    return set.candidate_count;
}

// Probability that a tile sample comes from the light with feedback id `id`.
public float global_selection(LightSetGpu* set, uint id)
{
    if (id == feedback_environment(set))
        return set.environment_probability;
    uint lights = set.scene.light_capacity;
    return id < lights ? set.light_probability[id] : set.emissive[id - lights].probability;
}

// Probability that local sampler `local` draws the light with feedback id `id`.
public float local_selection(LocalLights* local, uint id)
{
    uint n = min(local.count, LOCAL_LIGHTS);
    for (uint k = 0; k < n; k++)
    {
        if (local.ids[k] == id)
            return (local.cdf[k] - (k > 0 ? local.cdf[k - 1] : 0.0)) / local.cdf[n - 1];
    }
    return 0.0;
}

// Probability that a tile-light candidate of a vertex reading `local` (null for none) is the
// light with feedback id `id`.
public float light_selection(LightSetGpu* set, LocalLights* local, uint id)
{
    float g = global_selection(set, id);
    if (local == nullptr)
        return g;
    return local.share * local_selection(local, id) + (1.0 - local.share) * g;
}

// Sample of the light with feedback id `id` from `u`, the light chosen with `probability`.
LightSample tile_light_sample(LightSetGpu* set, uint id, float4 u, float probability)
{
    if (id == feedback_environment(set))
    {
        float pdf;
        float3 d = environment_sample(set.environment, float3(u.z, u.xy), pdf);
        return light_sample(LIGHT_SAMPLE_ENVIRONMENT, id, d, float3(0.0, 0.0, 1.0),
                            environment_map(set.environment, d), -probability * pdf);
    }
    uint lights = set.scene.light_capacity;
    if (id < lights)
        return local_sample(set, id, u.xy, probability);
    return emissive_sample(set, id - lights, u, probability);
}

// Tile sample `index` of `frame`, drawn from the light set's selection.
public LightSample presample_light(LightSetGpu* set, uint index, uint frame)
{
    uint4 h0 = pcg4d(uint4(index, frame, 0x6c69u, 1u));
    uint4 h1 = pcg4d(uint4(index, frame, 0x6c69u, 2u));
    float x = unit_float(h0.x);
    float4 u = float4(unit_float(h0.y), unit_float(h0.z), unit_float(h0.w), unit_float(h1.x));
    float env = set.environment_probability;
    uint id = feedback_environment(set);
    if (!(x < env))
    {
        float total = set.candidate_total;
        if (!(total > 0.0))
        {
            LightSample none = {};
            return none;
        }
        id = upper_bound(set.cdf, set.candidate_count, below((x - env) / (1.0 - env), total));
    }
    return tile_light_sample(set, id, u, global_selection(set, id));
}

// The local sampler candidate of a vertex at `x` from `pick` and `u`, with the density of
// its point alone (the light chosen with probability 1).
ResolvedLightSample local_candidate(LightSetGpu* set, LocalLights* local, float3 x, float pick, float4 u)
{
    uint n = min(local.count, LOCAL_LIGHTS);
    float target = below(pick, local.cdf[n - 1]);
    uint k = 0;
    while (k + 1 < n && !(local.cdf[k] > target))
        k++;
    return resolve_light_sample(set, tile_light_sample(set, local.ids[k], u, 1.0), x, u.xy);
}

// Adds one pixel whose feedback names `id` to the scene's feedback counts.
public void count_feedback(LightSetGpu* set, uint id)
{
    InterlockedAdd(set.feedback_counts[id], 1u);
}

public ResolvedLightSample no_candidate()
{
    ResolvedLightSample r;
    r.dir = float3(0.0, 0.0, 1.0);
    r.distance = 0.0;
    r.radiance = float3(0.0);
    r.pdf = 0.0;
    r.kind = LIGHT_SAMPLE_ENVIRONMENT;
    r.light = NO_LIGHT;
    return r;
}

// Light sample `s` of `set` seen from `x`: a sphere's point is drawn here with `u`, uniform
// over the cone the sphere subtends. The density is the sample's, without the vertex's
// directional share.
public ResolvedLightSample resolve_light_sample(LightSetGpu* set, LightSample s, float3 x, float2 u)
{
    if (s.pdf == 0.0)
        return no_candidate();
    ResolvedLightSample r = no_candidate();
    uint kind = s.bits & 3;
    r.kind = kind;
    r.light = s.light;
    float3 l = decode_radiance(s);
    if (kind == LIGHT_SAMPLE_ENVIRONMENT)
    {
        r.dir = s.point;
        r.distance = positive_infinity();
        r.radiance = l;
        r.pdf = -s.pdf;
        return r;
    }
    if (kind == LIGHT_SAMPLE_SPHERE)
    {
        float radius = unpack_half2x16(s.bits >> 16).x;
        float3 to_centre = s.point - x;
        float d2 = dot(to_centre, to_centre);
        float r2 = radius * radius;
        if (!(d2 > r2))
            return no_candidate();
        float sin2 = r2 / d2;
        float one_minus_cos = sin2 / (1.0 + sqrt(1.0 - sin2));
        float3 axis = to_centre * rsqrt(d2);
        r.dir = cone_direction(axis, one_minus_cos, u);
        float b = dot(to_centre, r.dir);
        float3 h = to_centre - b * r.dir;
        r.distance = b - sqrt(max(0.0, r2 - dot(h, h)));
        LightGpu* light = set.scene.lights + s.light;
        float c = dot(oct_decode_snorm16(s.normal), -axis);
        r.radiance = l * spot_falloff(c, light.cone_cos_inner, light.cone_cos_outer);
        r.pdf = s.pdf / (TWO_PI * one_minus_cos);
        return r;
    }
    float3 v = s.point - x;
    float d2 = dot(v, v);
    if (!(d2 > 0.0))
        return no_candidate();
    r.distance = sqrt(d2);
    r.dir = v / r.distance;
    float cos_light = -dot(oct_decode_snorm16(s.normal), r.dir);
    if (cos_light == 0.0)
        return no_candidate();
    bool lit = cos_light > 0.0 || (s.bits & LIGHT_SAMPLE_TWO_SIDED) != 0;
    r.radiance = lit ? l : float3(0.0);
    r.pdf = s.pdf * d2 / abs(cos_light);
    return r;
}

// The tile that pixel `pixel`'s block reads in `frame`.
public uint light_tile(uint2 pixel, uint frame)
{
    uint2 b = pixel / TILE_BLOCK;
    return (pcg(b.x | (b.y << 16)) + frame * TILE_FRAME_STEP) % TILE_COUNT;
}

// One next-event candidate for vertex `v`: with the vertex's directional share, a directional
// light chosen by weight x n.l and a direction in its cone; otherwise, with the local
// sampler's share, a light it holds and a point on it, or else a sample at a random offset
// in the vertex's tile. The density is the candidate's in the vertex's whole mixture. Draws
// three sampler dimensions.
public ResolvedLightSample nee_candidate(LightSetGpu* set, NeeVertex v, inout Sampler rng)
{
    float total;
    float share = directional_share(set, v.normal, total);
    float choice = rng.next();
    float pick = rng.next();
    float4 u4 = rng.next4();
    float2 u = u4.xy;
    if (choice < share)
    {
        float target = pick * total;
        uint id = NO_LIGHT;
        float w = 0.0;
        float sum = 0.0;
        for (uint k = 0; k < set.directional_count; k++)
        {
            uint j = set.directional_ids[k];
            float wj = directional_weight(set, j, v.normal);
            if (wj > 0.0)
            {
                id = j;
                w = wj;
                if (target < sum + wj)
                    break;
            }
            sum += wj;
        }
        ResolvedLightSample r = no_candidate();
        float cone_pdf;
        r.dir = directional_sample(set.scene.lights[id], u, cone_pdf);
        r.distance = positive_infinity();
        r.radiance = disc_radiance(set, id, r.dir) * set.directional[id].transmittance;
        r.pdf = share * w / total * cone_pdf;
        r.kind = LIGHT_SAMPLE_DIRECTIONAL;
        r.light = id;
        return r;
    }
    LocalLights* local = v.local;
    ResolvedLightSample r;
    if (local != nullptr && (choice - share) < local.share * (1.0 - share))
    {
        r = local_candidate(set, local, v.position, pick, u4);
        if (r.pdf > 0.0)
            r.pdf *= light_selection(set, local, r.light);
    }
    else
    {
        uint tile = light_tile(v.pixel, set.frame);
        uint index = min(uint(pick * float(TILE_SIZE)), TILE_SIZE - 1);
        r = resolve_light_sample(set, set.tiles[tile * TILE_SIZE + index], v.position, u);
        if (local != nullptr && r.pdf > 0.0)
            r.pdf *= light_selection(set, local, r.light) / global_selection(set, r.light);
    }
    r.pdf *= 1.0 - share;
    return r;
}

// Densities per sr of the candidates of a vertex with `normal` and local sampler `local`
// (null for none) in direction `dir`, for weighting BSDF samples against next-event
// estimation: from the environment, from directional light `id`, and from emissive entry
// `entry` hit at `distance` on a surface emitting `luminance` nits whose normal makes
// `cos_light` with the ray.
public float nee_environment_pdf(LightSetGpu* set, LocalLights* local, float3 normal, float3 dir)
{
    float total;
    float share = directional_share(set, normal, total);
    float p = light_selection(set, local, feedback_environment(set));
    return (1.0 - share) * p * environment_pdf(set.environment, dir);
}

public float nee_directional_pdf(LightSetGpu* set, float3 normal, uint id, float3 dir)
{
    return directional_probability(set, normal, id) * directional_pdf(set.scene.lights[id], dir);
}

public float nee_emissive_pdf(LightSetGpu* set, LocalLights* local, float3 normal, uint entry,
                              float luminance, float distance, float cos_light)
{
    float total;
    float share = directional_share(set, normal, total);
    EmissiveSelection e = set.emissive[entry];
    if (!(e.total > 0.0))
        return 0.0;
    float p = light_selection(set, local, set.scene.light_capacity + entry);
    return (1.0 - share) * p * luminance / e.total * distance * distance / abs(cos_light);
}

// Density per sr of tile samples from sphere or rect light `id` seen from `x` in direction
// `dir`, without the vertex's directional share; 0 when `dir` misses the light.
public float local_light_pdf(LightSetGpu* set, uint id, float3 x, float3 dir)
{
    LightGpu l = set.scene.lights[id];
    float p = set.light_probability[id];
    if (l.kind == LIGHT_SPHERE)
    {
        float r = sample_radius(l.radius);
        float3 to_centre = l.position - x;
        float d2 = dot(to_centre, to_centre);
        float r2 = r * r;
        float b = dot(to_centre, dir);
        float3 h = to_centre - b * dir;
        if (!(d2 > r2) || b <= 0.0 || dot(h, h) > r2)
            return 0.0;
        float sin2 = r2 / d2;
        return p / (TWO_PI * sin2 / (1.0 + sqrt(1.0 - sin2)));
    }
    if (l.kind != LIGHT_RECT)
        return 0.0;
    float3 n = cross(l.edge_x, l.edge_y);
    float area = length(n);
    n /= area;
    float dn = dot(dir, n);
    if (dn == 0.0)
        return 0.0;
    float t = dot(l.position - x, n) / dn;
    float3 q = x + t * dir - l.position;
    float ex = dot(q, l.edge_x) / dot(l.edge_x, l.edge_x);
    float ey = dot(q, l.edge_y) / dot(l.edge_y, l.edge_y);
    if (t <= 0.0 || abs(ex) > 0.5 || abs(ey) > 0.5)
        return 0.0;
    return p / area * t * t / abs(dn);
}

// Illuminance of directional light `id` before the medium.
public float3 directional_illuminance(LightSetGpu* set, uint id)
{
    return emitted_illuminance(set.scene.lights[id], set.directional[id].integral);
}

// Light the linked medium `M` scatters toward `origin` along the `t` m segment from `origin`
// along unit `dir`, from every directional light and the medium's own emission, marched in
// `steps` steps offset by `jitter` in [0, 1); `transmittance` is the segment's. Traces no
// rays.
public float3 segment_inscatter<M : IMedium>(LightSetGpu* set, float3 origin, float3 dir, float t, uint steps, float jitter, out float3 transmittance)
{
    uint64_t data = set.scene.medium;
    float dt = t / float(max(steps, 1u));
    float3 inscatter = float3(0.0);
    transmittance = float3(1.0);
    for (uint k = 0; k < max(steps, 1u); k++)
    {
        float3 x = origin + dir * ((float(k) + jitter) * dt);
        MediumPoint m = M.sample(data, x);
        float3 source = m.emission;
        if (any(m.sigma_s > 0.0))
        {
            for (uint j = 0; j < set.directional_count; j++)
            {
                uint id = set.directional_ids[j];
                float3 to_light = set.scene.lights[id].direction;
                float3 e = directional_illuminance(set, id);
                float3 phase = M.phase(data, m, dot(to_light, dir));
                float3 lit = M.transmittance(data, x, to_light, positive_infinity());
                source += m.sigma_s * e * (phase * lit + M.multiple_scattering(data, x, to_light));
            }
        }
        float3 step = exp(-m.sigma_t * dt);
        float3 through = select(m.sigma_t > 0.0, (1.0 - step) / max(m.sigma_t, 1e-30), float3(dt));
        inscatter += transmittance * source * through;
        transmittance *= step;
    }
    return inscatter;
}
