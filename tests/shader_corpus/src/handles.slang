// Heap slots and the TLAS address, as named structs so reflection keeps their kind.
public struct TextureIndex
{
    public uint index;
    [ForceInline] public T get<T : IOpaqueDescriptor>() { return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0))); }
}

public struct StorageImageIndex
{
    public uint index;
    [ForceInline] public T get<T : IOpaqueDescriptor>() { return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0))); }
}

public struct SamplerIndex
{
    public uint index;
    [ForceInline] public SamplerState get() { return getDescriptorFromHandle(DescriptorHandle<SamplerState>(uint2(index, 0))); }
}

public struct Tlas
{
    public uint64_t address;
    [ForceInline] public RaytracingAccelerationStructure get() { return RaytracingAccelerationStructure(address); }
}
