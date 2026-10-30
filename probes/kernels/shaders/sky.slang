// Atmosphere LUT passes in the shape the atmosphere plan describes: transmittance 256x128,
// multiple scattering 64x32 (64 directions per texel), sky-view lat-long with a quadratic
// latitude warp, folding `light_count` directional lights. Earth constants, three channels.
struct Light
{
    float4 dir_lux; // direction to the light, top-of-atmosphere illuminance
    float4 color;
};

struct Root
{
    Light lights[8];
    uint light_count;
    uint trans_img; // storage views
    uint ms_img;
    uint sky_img;
    uint trans_tex; // sampled views
    uint ms_tex;
    uint sampler;
    uint sky_w;
    uint sky_h;
    uint steps;
    float altitude; // m above the planet surface
    uint pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

static const float R_GROUND = 6360e3;
static const float R_TOP = 6460e3;
static const float3 RAYLEIGH_S = float3(5.802e-6, 13.558e-6, 33.1e-6);
static const float MIE_S = 3.996e-6;
static const float MIE_E = 4.40e-6;
static const float3 OZONE_A = float3(0.650e-6, 1.881e-6, 0.085e-6);
static const float PI = 3.14159265;

RWTexture2D<float4> storage(uint i) { return getDescriptorFromHandle(DescriptorHandle<RWTexture2D<float4>>(uint2(i, 0))); }
Texture2D<float4> tex(uint i) { return getDescriptorFromHandle(DescriptorHandle<Texture2D<float4>>(uint2(i, 0))); }
SamplerState linear_clamp() { return getDescriptorFromHandle(DescriptorHandle<SamplerState>(uint2(root.sampler, 0))); }

struct Medium
{
    float3 scatter_r;
    float scatter_m;
    float3 extinction;
};

Medium medium(float r)
{
    float h = max(r - R_GROUND, 0.0);
    float dr = exp(-h / 8000.0);
    float dm = exp(-h / 1200.0);
    float dozone = max(0.0, 1.0 - abs(h - 25000.0) / 15000.0);
    Medium m;
    m.scatter_r = RAYLEIGH_S * dr;
    m.scatter_m = MIE_S * dm;
    m.extinction = m.scatter_r + MIE_E * dm + OZONE_A * dozone;
    return m;
}

// Distance to the sphere of radius `rad` along mu from radius r (far root), or -1.
float ray_sphere(float r, float mu, float rad)
{
    float disc = r * r * (mu * mu - 1.0) + rad * rad;
    return disc < 0.0 ? -1.0 : -r * mu + sqrt(disc);
}

bool hits_ground(float r, float mu)
{
    return mu < 0.0 && r * r * (mu * mu - 1.0) + R_GROUND * R_GROUND >= 0.0;
}

float dist_to_boundary(float r, float mu)
{
    if (hits_ground(r, mu))
        return -r * mu - sqrt(max(r * r * (mu * mu - 1.0) + R_GROUND * R_GROUND, 0.0));
    return max(ray_sphere(r, mu, R_TOP), 0.0);
}

// Bruneton (r, mu) <-> uv for the transmittance LUT.
float2 trans_uv(float r, float mu)
{
    float H = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
    float rho = sqrt(max(r * r - R_GROUND * R_GROUND, 0.0));
    float d = max(ray_sphere(r, mu, R_TOP), 0.0);
    float d_min = R_TOP - r;
    float d_max = rho + H;
    return float2((d - d_min) / (d_max - d_min), rho / H);
}

void trans_rmu(float2 uv, out float r, out float mu)
{
    float H = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
    float rho = H * uv.y;
    r = sqrt(rho * rho + R_GROUND * R_GROUND);
    float d_min = R_TOP - r;
    float d_max = rho + H;
    float d = d_min + uv.x * (d_max - d_min);
    mu = d == 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
}

float3 transmittance_to_top(float r, float mu)
{
    if (hits_ground(r, mu))
        return float3(0.0);
    return tex(root.trans_tex).SampleLevel(linear_clamp(), trans_uv(r, mu), 0.0).rgb;
}

float3 multiple_scattering(float r, float mu_light)
{
    float2 uv = float2(mu_light * 0.5 + 0.5, (r - R_GROUND) / (R_TOP - R_GROUND));
    return tex(root.ms_tex).SampleLevel(linear_clamp(), uv, 0.0).rgb;
}

float rayleigh_phase(float c) { return 3.0 / (16.0 * PI) * (1.0 + c * c); }

float cs_phase(float c)
{
    const float g = 0.7;
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + c * c) / pow(1.0 + g * g - 2.0 * g * c, 1.5);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void transmittance(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= 256 || id.y >= 128)
        return;
    float r, mu;
    trans_rmu((float2(id.xy) + 0.5) / float2(256.0, 128.0), r, mu);
    float d = max(ray_sphere(r, mu, R_TOP), 0.0);
    float3 tau = float3(0.0);
    const uint n = 40;
    float dt = d / float(n);
    for (uint i = 0; i < n; ++i)
    {
        float t = (float(i) + 0.5) * dt;
        float ri = sqrt(r * r + t * t + 2.0 * r * mu * t);
        tau += medium(ri).extinction * dt;
    }
    storage(root.trans_img)[id.xy] = float4(exp(-tau), 1.0);
}

groupshared float3 g_l2[64];
groupshared float3 g_fms[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void multiscatter(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID)
{
    float2 uv = (float2(gid.xy) + 0.5) / float2(64.0, 32.0);
    float mu_s = uv.x * 2.0 - 1.0;
    float r = R_GROUND + uv.y * (R_TOP - R_GROUND);
    // Fibonacci direction for this thread.
    float k = float(tid.x) + 0.5;
    float z = 1.0 - 2.0 * k / 64.0;
    float phi = 2.39996323 * float(tid.x);
    float s = sqrt(max(1.0 - z * z, 0.0));
    float3 dir = float3(s * cos(phi), z, s * sin(phi));
    float3 sun = float3(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0);
    float d = dist_to_boundary(r, dir.y);
    const uint n = 20;
    float dt = d / float(n);
    float3 thr = float3(1.0);
    float3 l2 = float3(0.0);
    float3 fms = float3(0.0);
    for (uint i = 0; i < n; ++i)
    {
        float t = (float(i) + 0.5) * dt;
        float3 p = float3(0.0, r, 0.0) + dir * t;
        float ri = length(p);
        float3 up = p / ri;
        Medium m = medium(ri);
        float3 scatter = m.scatter_r + m.scatter_m;
        float3 step_t = exp(-m.extinction * dt);
        float3 integ = (1.0 - step_t) / max(m.extinction, 1e-12);
        float3 ts = transmittance_to_top(ri, dot(up, sun));
        l2 += thr * integ * scatter * ts * (1.0 / (4.0 * PI));
        fms += thr * integ * scatter;
        thr *= step_t;
    }
    g_l2[tid.x] = l2 / 64.0;
    g_fms[tid.x] = fms / 64.0;
    GroupMemoryBarrierWithGroupSync();
    for (uint w = 32; w > 0; w >>= 1)
    {
        if (tid.x < w)
        {
            g_l2[tid.x] += g_l2[tid.x + w];
            g_fms[tid.x] += g_fms[tid.x + w];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0)
        storage(root.ms_img)[gid.xy] = float4(g_l2[0] / max(1.0 - g_fms[0], 1e-3), 1.0);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void skyview(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.sky_w || id.y >= root.sky_h)
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(root.sky_w, root.sky_h);
    // Latitude warped toward the horizon, absolute longitude.
    float v = uv.y * 2.0 - 1.0;
    float lat = sign(v) * v * v * (PI * 0.5);
    float lon = uv.x * 2.0 * PI;
    float3 dir = float3(cos(lat) * cos(lon), sin(lat), cos(lat) * sin(lon));
    float r = R_GROUND + root.altitude;
    float d = dist_to_boundary(r, dir.y);
    uint n = root.steps;
    float dt = d / float(n);
    float3 thr = float3(1.0);
    float3 radiance = float3(0.0);
    for (uint i = 0; i < n; ++i)
    {
        float t = (float(i) + 0.5) * dt;
        float3 p = float3(0.0, r, 0.0) + dir * t;
        float ri = length(p);
        float3 up = p / ri;
        Medium m = medium(ri);
        float3 inscatter = float3(0.0);
        for (uint l = 0; l < root.light_count; ++l)
        {
            float3 ld = root.lights[l].dir_lux.xyz;
            float c = dot(dir, ld);
            float mu_l = dot(up, ld);
            float3 tl = transmittance_to_top(ri, mu_l);
            float3 ms = multiple_scattering(ri, mu_l);
            float3 single = tl * (m.scatter_r * rayleigh_phase(c) + m.scatter_m * cs_phase(c));
            inscatter += root.lights[l].dir_lux.w * root.lights[l].color.rgb * (single + ms * (m.scatter_r + m.scatter_m));
        }
        float3 step_t = exp(-m.extinction * dt);
        radiance += thr * inscatter * (1.0 - step_t) / max(m.extinction, 1e-12);
        thr *= step_t;
    }
    storage(root.sky_img)[id.xy] = float4(radiance, 1.0);
}
