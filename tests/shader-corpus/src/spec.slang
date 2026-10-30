// Specialisation constants with explicit ids; a literal workgroup size.
[vk::constant_id(0)] const uint GROUP_LIMIT = 64;
[vk::constant_id(1)] const float SCALE = 1.5;
[vk::constant_id(2)] const bool ALPHA_TEST = true;
[vk::constant_id(5)] const int BIAS = -3;

struct Root
{
    float* data;
    uint count;
    uint pad;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

groupshared float s[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uint gi: SV_GroupIndex)
{
    s[gi] = root.data[id.x] * SCALE + float(BIAS);
    GroupMemoryBarrierWithGroupSync();
    if (ALPHA_TEST && gi < GROUP_LIMIT)
        root.data[id.x] = s[(gi + 1) % 64];
}
