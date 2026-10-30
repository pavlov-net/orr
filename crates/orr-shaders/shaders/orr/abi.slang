// The types kernels share with Rust: heap slots and the TLAS address. Every heap and
// acceleration-structure access goes through this module.
//
// Mesh-shader kernels: task (`amplification`), mesh and pixel entries live in one file and
// pass a groupshared payload to `DispatchMesh`. Mesh outputs are written as whole structs,
// never field by field. Pixel entries read per-primitive data through `SV_PrimitiveID`
// lookups or flat per-vertex attributes, not per-primitive inputs. Workgroup sizes are
// literals.
module abi;

import orr.debug;

// A sampled image view in the resource heap. Sampled and storage views share one slot
// space.
public struct TextureIndex
{
    public uint index;

    // The view as `T`: `Texture2D<float4>`, `Texture2DArray<float>`, `TextureCube<float4>`, ...
    [ForceInline]
    public T get<T : IOpaqueDescriptor>()
    {
        __target_switch
        {
        case metal:
            return metal_resource<T>(metal_texture_pool + uint64_t(index));
        default:
            return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
        }
    }
}

// A storage image view in the resource heap, in the slot space of `TextureIndex`.
public struct StorageImageIndex
{
    public uint index;

    // The view as `T`: `RWTexture2D<float4>`, ... A 64-bit image names its format:
    // `RWTexture2D<uint64_t, 0, 40>`.
    [ForceInline]
    public T get<T : IOpaqueDescriptor>()
    {
        __target_switch
        {
        case metal:
            return metal_resource<T>(metal_texture_pool + uint64_t(index));
        default:
            return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
        }
    }
}

// A sampler in the sampler heap.
public struct SamplerIndex
{
    public uint index;

    [ForceInline]
    public SamplerState get()
    {
        __target_switch
        {
        case metal:
            return metal_resource<SamplerState>(metal_samplers[index].id);
        default:
            return getDescriptorFromHandle(DescriptorHandle<SamplerState>(uint2(index, 0)));
        }
    }
}

// A TLAS or PTLAS device address. Traced only through `orr.trace`.
public struct Tlas
{
    public uint64_t address;

    // Address 0 is not a valid acceleration structure; with `ASSERTS` it aborts.
    [ForceInline]
    public RaytracingAccelerationStructure get()
    {
        __target_switch
        {
        case metal:
            return metal_resource<RaytracingAccelerationStructure>(address);
        default:
            if (ASSERTS && address == 0)
                abort("Tlas.get() on address 0");
            return RaytracingAccelerationStructure(address);
        }
    }
}

// Marks a heap index that differs between lanes.
[ForceInline]
public uint nonuniform(uint index)
{
    __target_switch
    {
    case metal:
        return index;
    default:
        return NonUniformResourceIndex(index);
    }
}

// Metal: the root is at argument-table buffer 0, the texture view pool's base resource ID
// at 1 and the sampler resource IDs at 2. SPIR-V never uses these.
[[vk::binding(1, 0)]] ConstantBuffer<uint64_t> metal_texture_pool;
struct MetalSampler
{
    uint64_t id;
}
[[vk::binding(2, 0)]] StructuredBuffer<MetalSampler> metal_samplers;

// A resource of type `T` from its Metal resource ID.
T metal_resource<T>(uint64_t id)
{
    __requirePrelude(R"(template<typename T> T orr_resource_from_id(ulong id) { return *reinterpret_cast<thread T*>(&id); })");
    __intrinsic_asm "orr_resource_from_id<$TR>($0)";
}
