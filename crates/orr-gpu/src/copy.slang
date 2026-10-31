// Region records of indirect copies. Kernels fill the structs below and write them with
// `store` into the arrays `CommandList::copy_indirect` and `copy_to_texture_indirect` read.
// Addresses are device addresses.
module copy;

public static const uint MEMORY_COPY_STRIDE = 24;
public static const uint TEXTURE_COPY_STRIDE = 56;

// Bytes from `src` to `dst`. Both addresses and `bytes` are multiples of 4; 0 bytes copies
// nothing.
public struct MemoryCopy
{
    public uint64_t src;
    public uint64_t dst;
    public uint64_t bytes;
}

// Texels from `src` into a box of one mip of a color texture. `mip`, `base_layer` and
// `layers` must match the entry the CPU passes for this index. Rows start `row_texels`
// texels apart (0 packs them tightly). The extent is not empty.
public struct TextureCopy
{
    public uint64_t src;
    public uint row_texels;
    public uint mip;
    public uint base_layer;
    public uint layers;
    public uint3 offset;
    public uint3 extent;
}

uint* record(uint64_t list, uint index, uint stride)
{
    return (uint*)(list + uint64_t(index) * stride);
}

void put64(uint* p, uint word, uint64_t v)
{
    p[word] = uint(v);
    p[word + 1] = uint(v >> 32);
}

public void store(uint64_t list, uint index, MemoryCopy v)
{
    uint* p = record(list, index, MEMORY_COPY_STRIDE);
    put64(p, 0, v.src);
    put64(p, 2, v.dst);
    put64(p, 4, v.bytes);
}

public void store(uint64_t list, uint index, TextureCopy v)
{
    uint* p = record(list, index, TEXTURE_COPY_STRIDE);
    put64(p, 0, v.src);
    p[2] = v.row_texels;
    p[3] = 0; // image rows: tight
    p[4] = 1; // color aspect
    p[5] = v.mip;
    p[6] = v.base_layer;
    p[7] = v.layers;
    for (uint i = 0; i < 3; i++)
    {
        p[8 + i] = v.offset[i];
        p[11 + i] = v.extent[i];
    }
}
