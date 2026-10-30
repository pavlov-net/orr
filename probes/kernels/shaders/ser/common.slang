// Scene access, alpha test, materials and sampling shared by every entry point. Root and the
// scene records mirror `Root`, `Inst` and `ClusterInfo` in main.rs (C layout).

struct Inst
{
    float m[12]; // object -> world, row-major 3x4, uniform scale
    uint proto;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct ClusterInfo
{
    uint vbase; // first vertex in pos/uv
    uint ibase; // first index in idx (indices are cluster-local)
    uint material;
    uint flags;
};

struct Root
{
    uint64_t tlas;
    Inst* insts;
    ClusterInfo* clusters;
    float* pos;
    float* uv;
    uint* idx;
    float* out_rad; // 4 floats per pixel
    uint* out_hash;
    uint* out_stats; // 2 words per pixel (STATS builds)
    uint width;
    uint height;
    uint frame;
    uint bounces;
    uint tex;
    uint flags;
    uint salt;
    uint shade_scale;
    float cam[16]; // pos.xyz, tan_half_y; fwd.xyz, aspect; right.xyz, 0; up.xyz, 0
    float sun[4];
};

#if ROOT_PUSH
// Classic pipeline layout (no descriptor heap): root in push constants.
[[vk::push_constant]] ConstantBuffer<Root> root;
#else
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;
#endif

// Never equal to a pixel x; varied per run so the driver cannot reuse a cached pipeline.
[[vk::constant_id(0)]] const uint SALT = 0;

static const uint CF_ALPHA = 1;

// Pipeline traversal (any-hit GeometryIndex(), traced hit objects) on cluster geometry returns
// the geometry index with unrelated high bits set on 621.01; ray query and hit objects recorded
// from a ray query return it intact. The probe scene has fewer than 4096 clusters.
uint pipe_geom(uint g)
{
    return g & 0xfffu;
}
static const uint MAT_GROUND = 0;
static const uint MAT_BARK = 1;
static const uint MAT_LEAF = 2;
static const uint MAT_HEAVY = 3;

struct Payload
{
    uint unused;
};

RaytracingAccelerationStructure scene()
{
    return RaytracingAccelerationStructure(root.tlas);
}

Texture2D<float4> atlas()
{
    return getDescriptorFromHandle(DescriptorHandle<Texture2D<float4>>(uint2(root.tex, 0)));
}

SamplerState nearest()
{
    return getDescriptorFromHandle(DescriptorHandle<SamplerState>(uint2(0, 0)));
}

uint vertex(ClusterInfo c, uint prim, uint k)
{
    return root.idx[c.ibase + prim * 3 + k] + c.vbase;
}

float2 load_uv(uint v)
{
    return float2(root.uv[v * 2], root.uv[v * 2 + 1]);
}

float3 load_pos(uint v)
{
    return float3(root.pos[v * 3], root.pos[v * 3 + 1], root.pos[v * 3 + 2]);
}

float2 interp_uv(ClusterInfo c, uint prim, float2 bary)
{
    float2 t0 = load_uv(vertex(c, prim, 0));
    float2 t1 = load_uv(vertex(c, prim, 1));
    float2 t2 = load_uv(vertex(c, prim, 2));
    return t0 * (1.0 - bary.x - bary.y) + t1 * bary.x + t2 * bary.y;
}

// Alpha test for one candidate; mirrors the CPU micromap classification in main.rs.
bool alpha_pass(uint geom, uint prim, float2 bary)
{
    ClusterInfo c = root.clusters[geom];
    if ((c.flags & CF_ALPHA) == 0)
        return true;
    return atlas().SampleLevel(nearest(), interp_uv(c, prim, bary), 0.0).a >= 0.5;
}

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(inout uint s)
{
    s = pcg(s);
    return float(s >> 8) * (1.0 / 16777216.0);
}

uint hash_mix(uint h, uint v)
{
    return (h ^ v) * 16777619u;
}

float vnoise(float3 p)
{
    float3 fl = floor(p);
    float3 f = p - fl;
    int3 i = int3(fl);
    f = f * f * (3.0 - 2.0 * f);
    float c[8];
    for (uint k = 0; k < 8; ++k)
    {
        int3 o = i + int3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        c[k] = float(pcg(uint(o.x) * 73856093u ^ uint(o.y) * 19349663u ^ uint(o.z) * 83492791u) & 0xffffu) / 65535.0;
    }
    float x0 = lerp(lerp(c[0], c[1], f.x), lerp(c[2], c[3], f.x), f.y);
    float x1 = lerp(lerp(c[4], c[5], f.x), lerp(c[6], c[7], f.x), f.y);
    return lerp(x0, x1, f.z);
}

float fbm(float3 p, uint octaves)
{
    float sum = 0.0;
    float amp = 0.5;
    float norm = 0.0;
    for (uint o = 0; o < octaves; ++o)
    {
        sum += amp * vnoise(p);
        norm += amp;
        p = p * 2.03 + float3(1.7, 9.2, 4.1);
        p -= 4096.0 * floor(p * (1.0 / 4096.0)); // keep lattice coordinates finite and in int range
        amp *= 0.55;
    }
    return sum / max(norm, 1e-6);
}

struct Hit
{
    bool valid;
    uint inst;
    uint geom;
    uint prim;
    float2 bary;
};

Hit miss_hit()
{
    Hit h;
    h.valid = false;
    h.inst = 0;
    h.geom = 0;
    h.prim = 0;
    h.bary = float2(0.0);
    return h;
}

struct Surface
{
    float3 p;
    float3 n; // faces the incoming ray
    float2 uv;
    uint material;
};

float3 xform_point(Inst t, float3 p)
{
    return float3(t.m[0] * p.x + t.m[1] * p.y + t.m[2] * p.z + t.m[3],
                  t.m[4] * p.x + t.m[5] * p.y + t.m[6] * p.z + t.m[7],
                  t.m[8] * p.x + t.m[9] * p.y + t.m[10] * p.z + t.m[11]);
}

float3 xform_dir(Inst t, float3 d)
{
    return float3(t.m[0] * d.x + t.m[1] * d.y + t.m[2] * d.z,
                  t.m[4] * d.x + t.m[5] * d.y + t.m[6] * d.z,
                  t.m[8] * d.x + t.m[9] * d.y + t.m[10] * d.z);
}

// Material of a hit. With root.flags bit 0, the leaves of one instance in four take the
// expensive material, so neighbouring rays diverge in shading cost.
uint material_of(uint inst, uint geom)
{
    uint m = root.clusters[geom].material;
    if ((root.flags & 1u) != 0 && m == MAT_LEAF && (pcg(inst * 2654435761u) & 3u) == 0)
        m = MAT_HEAVY;
    return m;
}

Surface surface(Hit h, float3 dir)
{
    ClusterInfo c = root.clusters[h.geom];
    uint v0 = vertex(c, h.prim, 0);
    uint v1 = vertex(c, h.prim, 1);
    uint v2 = vertex(c, h.prim, 2);
    float3 p0 = load_pos(v0);
    float3 p1 = load_pos(v1);
    float3 p2 = load_pos(v2);
    float w = 1.0 - h.bary.x - h.bary.y;
    Inst t = root.insts[h.inst];
    Surface s;
    s.p = xform_point(t, p0 * w + p1 * h.bary.x + p2 * h.bary.y);
    s.n = normalize(xform_dir(t, cross(p1 - p0, p2 - p0)));
    if (dot(s.n, dir) > 0.0)
        s.n = -s.n;
    s.uv = load_uv(v0) * w + load_uv(v1) * h.bary.x + load_uv(v2) * h.bary.y;
    s.material = material_of(h.inst, h.geom);
    return s;
}

// Material-dependent shading with deliberately different cost per material.
float3 albedo(Surface s)
{
    uint k = max(root.shade_scale, 1u);
    switch (s.material)
    {
    case MAT_GROUND:
    {
        float f = fbm(s.p * 0.35, 4 * k);
        return lerp(float3(0.22, 0.16, 0.09), float3(0.16, 0.28, 0.07), f);
    }
    case MAT_BARK:
    {
        float f = fbm(s.p * float3(6.0, 0.8, 6.0), 6 * k);
        return float3(0.30, 0.22, 0.15) * (0.45 + 0.55 * f);
    }
    case MAT_LEAF:
    {
        float3 t = atlas().SampleLevel(nearest(), s.uv, 0.0).rgb;
        return t * (0.8 + 0.4 * fbm(s.p * 3.0, 2 * k));
    }
    default:
    {
        // Layered bark: many octaves standing in for an expensive layered material.
        float f = fbm(s.p * float3(5.0, 1.3, 5.0), 40 * k);
        float g = fbm(s.p.zxy * 2.0, 8 * k);
        return lerp(float3(0.35, 0.30, 0.26), float3(0.12, 0.20, 0.10), f * g * 1.6);
    }
    }
}

float3 sky(float3 d)
{
    float t = saturate(d.y * 0.5 + 0.5);
    return lerp(float3(0.9, 0.95, 1.0), float3(0.35, 0.55, 1.0), t) * 1.2;
}

float3 cosine_dir(float3 n, inout uint rng)
{
    float u1 = rnd(rng);
    float u2 = rnd(rng);
    float r = sqrt(u1);
    float phi = 6.2831853 * u2;
    float3 t = abs(n.x) > 0.5 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 b1 = normalize(cross(t, n));
    float3 b2 = cross(n, b1);
    return normalize(b1 * (r * cos(phi)) + b2 * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

RayDesc make_ray(float3 o, float3 d, float tmax)
{
    RayDesc r;
    r.Origin = o;
    r.Direction = d;
    r.TMin = 0.0;
    r.TMax = tmax;
    return r;
}

RayDesc camera_ray(uint2 px, inout uint rng)
{
    float jx = rnd(rng);
    float jy = rnd(rng);
    float2 ndc = (float2(px) + float2(jx, jy)) / float2(root.width, root.height) * 2.0 - 1.0;
    float3 o = float3(root.cam[0], root.cam[1], root.cam[2]);
    float th = root.cam[3];
    float3 fwd = float3(root.cam[4], root.cam[5], root.cam[6]);
    float aspect = root.cam[7];
    float3 right = float3(root.cam[8], root.cam[9], root.cam[10]);
    float3 up = float3(root.cam[12], root.cam[13], root.cam[14]);
    return make_ray(o, normalize(fwd + right * (ndc.x * th * aspect) - up * (ndc.y * th)), 1e4);
}
