// Synthetic RR and FG inputs from an analytic screen-space scene: a grid of coloured tiles with
// grout lines, shaded bumps and small emitters, panning horizontally. Every kernel reads the
// root through a pushed address and reaches images through resource-heap storage descriptors.

// C layout; mirrors `Root` in src/main.rs.
struct Root
{
    float4* out;      // reference_main, copy_main: one float4 per output pixel
    float* rows;      // rows_main: luminance sum per output row
    uint img[16];     // resource-heap slots, see `Slot` in src/main.rs
    uint2 render;
    uint2 output;
    float2 pan;       // scene offset in uv units this frame
    float2 dpan;      // pan(frame) - pan(frame - 1)
    float2 jitter;    // NGX jitter, render pixels: sample position = pixel centre - jitter
    float exposure;   // pre-exposure applied to the colour input
    uint frame;
    uint seed;
    uint flags;
    float nits;       // backbuffer_main: nits at value 1.0
    uint pad;
};

[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

static const uint REC2020 = 1;   // RR inputs (colour, albedo) in linear Rec.2020, else Rec.709
static const uint WIDE = 2;      // backbuffer: three tile colours are Rec.2020 primaries
static const uint CLIP = 4;      // backbuffer: content clipped to Rec.709 before encoding
static const uint DOF = 8;       // blurred region in the colour input plus the DoF guide
static const uint JITTER_FLIP_X = 16; // sample x = pixel centre + jitter x
static const uint JITTER_FLIP_Y = 32; // sample y = pixel centre + jitter y

static const uint COLOR = 0, DIFFUSE = 1, SPECULAR = 2, NORMAL = 3, DEPTH = 4, MOTION = 5,
                  SPEC_MOTION = 6, DOF_GUIDE = 7, RR_OUT = 8, BACKBUFFER = 9;

RWTexture2D<float4> image(uint slot)
{
    return getDescriptorFromHandle(DescriptorHandle<RWTexture2D<float4>>(uint2(root.img[slot], 0)));
}

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

float rand01(inout uint state)
{
    state = hash(state);
    return float(state >> 8) * (1.0 / 16777216.0);
}

float3 rec709_to_rec2020(float3 c)
{
    return float3(dot(float3(0.6274039, 0.3292830, 0.0433131), c),
                  dot(float3(0.0690973, 0.9195404, 0.0113623), c),
                  dot(float3(0.0163914, 0.0880133, 0.8955953), c));
}

float3 rec2020_to_rec709(float3 c)
{
    return float3(dot(float3(1.6604910, -0.5876411, -0.0728499), c),
                  dot(float3(-0.1245505, 1.1328999, -0.0083494), c),
                  dot(float3(-0.0181508, -0.1005789, 1.1187297), c));
}

// Tile colours, Rec.709. A switch, not an array: no function-local arrays (output.md 2.3).
float3 palette(uint i)
{
    switch (i & 7)
    {
    case 0: return float3(0.80, 0.10, 0.08);
    case 1: return float3(0.10, 0.65, 0.12);
    case 2: return float3(0.08, 0.15, 0.80);
    case 3: return float3(0.85, 0.75, 0.10);
    case 4: return float3(0.10, 0.70, 0.75);
    case 5: return float3(0.75, 0.12, 0.70);
    case 6: return float3(0.70, 0.70, 0.70);
    default: return float3(0.95, 0.45, 0.10);
    }
}

static const float3 PALETTE_MEAN = float3(0.53125, 0.45250, 0.42125);
static const float3 LIGHT = float3(0.3713907, -0.5570860, 0.7427814);
static const float SUN = 2.5;
static const float AMBIENT = 0.8;

struct Surface
{
    float3 albedo;
    float spec;
    float3 n;
    float rough;
    float depth;   // reverse-Z device depth
    float3 emit;
    uint palette_id; // tile colour index, for the wide-gamut backbuffer
};

Surface surface(float2 uv)
{
    float aspect = float(root.output.x) / float(root.output.y);
    float2 p = (uv + root.pan) * float2(aspect, 1.0) * 9.0;
    float2 cell = floor(p);
    float2 f = p - cell;
    uint h = hash(uint(int(cell.x) + 4096) * 7919u + uint(int(cell.y) + 4096));
    Surface s;
    s.palette_id = h & 7;
    s.albedo = palette(s.palette_id);
    s.spec = 0.04;
    s.n = float3(0, 0, 1);
    s.rough = 0.6;
    s.depth = 0.1 / 5.0;
    s.emit = float3(0, 0, 0);
    float2 d = f - 0.5;
    float r = 0.32;
    float edge = min(min(f.x, f.y), min(1.0 - f.x, 1.0 - f.y));
    if (edge < 0.03)
    {
        s.albedo = float3(0.05, 0.05, 0.05);
        s.palette_id = 6;
        s.rough = 0.9;
    }
    else if (((h >> 3) & 1) != 0 && dot(d, d) < r * r)
    {
        float nz = sqrt(max(0.0, 1.0 - dot(d, d) / (r * r)));
        s.n = normalize(float3(d.x / r, -d.y / r, nz));
        s.palette_id = (h >> 4) & 7;
        s.albedo = palette(s.palette_id);
        s.spec = 0.3;
        s.rough = 0.25;
        s.depth = 0.1 / (5.0 - 0.2 * nz);
    }
    else if ((h >> 8) % 7 == 0 && dot(d, d) < 0.18 * 0.18)
    {
        s.palette_id = (h >> 12) & 7;
        s.emit = palette(s.palette_id) * 6.0;
        s.albedo = float3(0.02, 0.02, 0.02);
    }
    return s;
}

float3 specular(Surface s)
{
    float3 hv = normalize(LIGHT + float3(0, 0, 1));
    float e = clamp(2.0 / (s.rough * s.rough * s.rough * s.rough) - 2.0, 1.0, 4096.0);
    float ndl = max(0.0, dot(s.n, LIGHT));
    return s.spec * SUN * (e + 2.0) / 8.0 * pow(max(0.0, dot(s.n, hv)), e) * ndl;
}

// Expected radiance, Rec.709.
float3 truth(Surface s)
{
    float ndl = max(0.0, dot(s.n, LIGHT));
    return s.albedo * (SUN * ndl + AMBIENT * PALETTE_MEAN) + specular(s) + s.emit;
}

// One noisy sample with expectation `truth`: Bernoulli light visibility and an indirect bounce
// off a random tile colour.
float3 noisy(Surface s, inout uint state)
{
    float ndl = max(0.0, dot(s.n, LIGHT));
    float vis = rand01(state) < 0.5 ? 2.0 : 0.0;
    float3 bounce = palette(uint(rand01(state) * 8.0));
    return s.albedo * (SUN * ndl * vis + AMBIENT * bounce) + specular(s) + s.emit;
}

bool in_dof(float2 uv)
{
    return (root.flags & DOF) != 0 && uv.x > 0.55 && uv.x < 0.95 && uv.y > 0.1 && uv.y < 0.6;
}

// Disc blur of the expected radiance, 16 Vogel taps of radius 0.015 uv.
float3 blurred(float2 uv)
{
    float3 sum = 0;
    for (uint k = 0; k < 16; k++)
    {
        float a = float(k) * 2.3999632;
        float rr = 0.015 * sqrt((float(k) + 0.5) / 16.0);
        sum += truth(surface(uv + rr * float2(cos(a), sin(a) * float(root.output.x) / float(root.output.y))));
    }
    return sum / 16.0;
}

float guide_luminance(float3 c)
{
    return (c.r + 2.0 * c.g + c.b) / 4.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void inputs_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= root.render))
        return;
    float2 j = -root.jitter * float2((root.flags & JITTER_FLIP_X) != 0 ? -1.0 : 1.0, (root.flags & JITTER_FLIP_Y) != 0 ? -1.0 : 1.0);
    float2 uv = (float2(id.xy) + 0.5 + j) / float2(root.render);
    Surface s = surface(uv);
    uint state = hash(id.x * 73856093u ^ id.y * 19349663u ^ root.frame * 83492791u ^ root.seed * 2654435761u);
    float3 radiance = noisy(s, state);
    float dof = 0.0;
    if (in_dof(uv))
    {
        float3 after = blurred(uv);
        dof = guide_luminance(after - radiance);
        radiance = after;
    }
    float3 albedo = s.albedo;
    if ((root.flags & REC2020) != 0)
    {
        radiance = rec709_to_rec2020(radiance);
        albedo = rec709_to_rec2020(albedo);
    }
    image(COLOR)[id.xy] = float4(radiance * root.exposure, 1.0);
    image(DIFFUSE)[id.xy] = float4(albedo, 1.0);
    image(SPECULAR)[id.xy] = float4(s.spec, s.spec, s.spec, 1.0);
    image(NORMAL)[id.xy] = float4(s.n, s.rough);
    image(DEPTH)[id.xy] = float4(s.depth, 0, 0, 0);
    image(MOTION)[id.xy] = float4(root.dpan, 0, 0);
    image(SPEC_MOTION)[id.xy] = float4(root.dpan, 0, 0);
    image(DOF_GUIDE)[id.xy] = float4(dof, 0, 0, 0);
}

// Expected radiance at output resolution, Rec.709, box-filtered over 8x8 sub-samples.
[shader("compute")]
[numthreads(8, 8, 1)]
void reference_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= root.output))
        return;
    float3 sum = 0;
    for (uint k = 0; k < 64; k++)
    {
        float2 uv = (float2(id.xy) + (float2(k % 8, k / 8) + 0.5) / 8.0) / float2(root.output);
        sum += in_dof(uv) ? blurred(uv) : truth(surface(uv));
    }
    root.out[id.y * root.output.x + id.x] = float4(sum / 64.0, 1.0);
}

float3 pq_encode(float3 nits)
{
    float3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), 78.84375);
}

// Display-encoded PQ backbuffer, Rec.2020 primaries, 4x4 sub-samples. With WIDE, tiles of
// colour 0-2 are the Rec.2020 primaries at 0.8; with CLIP the content is clipped to Rec.709.
[shader("compute")]
[numthreads(8, 8, 1)]
void backbuffer_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= root.output))
        return;
    float3 sum = 0;
    for (uint k = 0; k < 16; k++)
    {
        float2 uv = (float2(id.xy) + (float2(k % 4, k / 4) + 0.5) / 4.0) / float2(root.output);
        Surface s = surface(uv);
        float3 c = rec709_to_rec2020(truth(s));
        if ((root.flags & WIDE) != 0 && s.palette_id < 3 && all(s.emit == 0))
        {
            float3 primary = float3(s.palette_id == 0 ? 0.8 : 0.0, s.palette_id == 1 ? 0.8 : 0.0, s.palette_id == 2 ? 0.8 : 0.0);
            float ndl = max(0.0, dot(s.n, LIGHT));
            c = primary * (SUN * ndl + AMBIENT * 0.47) + rec709_to_rec2020(specular(s));
        }
        if ((root.flags & CLIP) != 0)
            c = rec709_to_rec2020(max(rec2020_to_rec709(c), 0.0));
        sum += c;
    }
    image(BACKBUFFER)[id.xy] = float4(pq_encode(sum / 16.0 * root.nits), 1.0);
}

// Heap read of the RR output after NGX: copies it to `out`.
[shader("compute")]
[numthreads(8, 8, 1)]
void copy_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= root.output))
        return;
    root.out[id.y * root.output.x + id.x] = image(RR_OUT)[id.xy];
}

// Rec.709 luminance sum of each RR output row (output in Rec.709).
[shader("compute")]
[numthreads(64, 1, 1)]
void rows_main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.output.y)
        return;
    RWTexture2D<float4> rr = image(RR_OUT);
    float sum = 0;
    for (uint x = 0; x < root.output.x; x++)
        sum += dot(rr[uint2(x, id.x)].rgb, float3(0.2126, 0.7152, 0.0722));
    root.rows[id.x] = sum;
}
