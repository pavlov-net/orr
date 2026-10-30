// Root-heavy compute and mesh kernels. ROOT_PTR 0: root is a uniform buffer mapped to the
// address in push data (PUSH_ADDRESS); 1: push data holds a pointer and every root read goes
// through it. Everything else is identical.
#ifndef ROOT_PTR
#define ROOT_PTR 0
#endif

struct Root
{
    float4x4 m;
    float4 p[16];
    float* src[8];
    uint* out;
    uint n;
    uint iters;
    uint patches_x;
    uint pad[3];
};

#if ROOT_PTR
struct Push
{
    Root* root;
};
[[vk::push_constant]] ConstantBuffer<Push> push;
#define R (*push.root)
#else
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;
#define R root
#endif

// Uniform-heavy: a matrix and indexed parameters read every iteration.
[shader("compute")]
[numthreads(32, 1, 1)]
void params(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= R.n)
        return;
    float4 x = float4(float(i & 1023u), float(i >> 10), 1.0, 0.5) * 1e-3;
    for (uint k = 0; k < R.iters; ++k)
        x = frac(mul(R.m, x) + R.p[k & 15u]);
    R.out[i] = asuint(x.x + x.y + x.z + x.w);
}

// Pointer-heavy: every iteration loads through one of the root's pointers.
[shader("compute")]
[numthreads(32, 1, 1)]
void pointers(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= R.n)
        return;
    float acc = 0.0;
    uint j = i;
    for (uint k = 0; k < R.iters; ++k)
    {
        acc += R.src[k & 7u][j & 65535u] * R.p[k & 15u].x;
        j = j * 1103515245u + 12345u;
    }
    R.out[i] = asuint(acc);
}

// Mesh: one 8x8-vertex grid patch per group (98 triangles); every vertex reads the matrix,
// four parameters and a height through a root pointer.
struct V
{
    float4 pos : SV_Position;
    nointerpolation uint id : ID;
};

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void mesh_main(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID, out vertices V verts[64], out indices uint3 tris[98])
{
    SetMeshOutputCounts(64, 98);
    uint v = tid.x;
    uint px = gid.x % R.patches_x;
    uint py = gid.x / R.patches_x;
    float2 uv = float2(float(px * 7u + (v & 7u)), float(py * 7u + (v >> 3)));
    float h = R.src[0][(gid.x * 64u + v) & 65535u];
    float4 p = float4(uv * R.p[0].xy + R.p[1].xy, h * R.p[2].x, 1.0);
    p += R.p[3 + (v & 7u)] * 1e-6;
    verts[v].pos = mul(R.m, p);
    verts[v].id = gid.x;
    if (v < 49)
    {
        uint x = v % 7u;
        uint y = v / 7u;
        uint a = y * 8u + x;
        tris[v * 2] = uint3(a, a + 1, a + 8);
        tris[v * 2 + 1] = uint3(a + 1, a + 9, a + 8);
    }
}

[shader("pixel")]
uint pixel_main(V input) : SV_Target0
{
    return input.id ^ asuint(R.p[4].x);
}
