// A root with every member kind: pointers, heap slots, TLAS, matrix, 64-bit scalar.
import handles;
import records;

struct View
{
    float4x4 clip_from_world;
    float3 camera_pos;
    uint frame;
}

struct Root
{
    View* view;
    Instance* instances;
    Material* materials;
    float4* out;
    float4x4 world;
    uint64_t raw_address;
    TextureIndex albedo;
    StorageImageIndex target;
    SamplerIndex linear;
    uint instance_count;
    Tlas scene;
    Pair<float>* pairs;
    Batch* batches;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.instance_count)
        return;
    Instance inst = root.instances[id.x];
    Material m = root.materials[inst.material];
    float4 c = root.albedo.get<Texture2D<float4>>().SampleLevel(root.linear.get(), float2(m.uv_scale), 0.0);
    float3 p = mul(inst.world_from_local, float4(root.view->camera_pos, 1.0));
    float4 w = mul(root.world, float4(p, 1.0));
    root.target.get<RWTexture2D<float4>>()[id.xy] = c;
    root.out[id.x] = float4(m.base_color * c.rgb * w.xyz, m.roughness + root.pairs[id.x].second + float(root.raw_address + root.scene.address) + root.batches[0].material->roughness);
}
