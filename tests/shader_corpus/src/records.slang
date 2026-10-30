// Records shared by several kernels; generated once into `types`.
// Mapped to a hand-written Rust type by `extern_type`.
public struct Material
{
    public float3 base_color;
    public float roughness;
    public uint base_color_tex;
    public uint sampler_index;
    public half2 uv_scale;
}

public struct Instance
{
    public float3x4 world_from_local;
    public uint material;
    public uint flags;
}

public struct Batch
{
    public Material* material;
    public uint first;
    public uint count;
}

public struct Pair<T>
{
    public T first;
    public T second;
}
