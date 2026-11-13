// Ray tracing pipeline shaders. This is the only module that names `TraceRay`, `HitObject`,
// `CallShader` and `ReorderThread`, as `trace` is for `RayQuery`; Metal compiles none of it.
//
// Every stage of a pipeline declares its root with `ORR_ROOT_INLINE`, over one root type.
// A shader's record data is its entry's `uniform` parameters: Slang places them in the shader
// record after the group handle, and the build reports their extent as the entry's
// `record_bytes`. A raygen reads a traced hit's record data before invoking it with
// `Traced.record`.
//
// The trace's `ray_type` and `ray_types` are the hit record offset and stride a hit's record is
// chosen by, `instance contribution + geometry index * ray_types + ray_type`; only their low
// four bits count.
//
// Memory one shader writes and another reads after a trace, invoke, call or reorder between
// them goes through `load_coherent` and `store_coherent`, at device scope: on NVIDIA's driver
// a plain access, and one coherent at shader call scope (GLSL's `shadercallcoherent`), can
// read what the memory held before a store a callable made ahead of a reorder.
module pipeline;

import orr_shaders.abi;

// What crosses a trace, invoke or call in registers. A launch pays registers in every thread
// for its widest payload past 16 bytes, so pipelines pass this and keep the rest in memory.
public struct Payload
{
    public uint4 words;
}

// The attributes of a triangle hit; an intersection shader reports its own type.
public typealias TriangleAttributes = BuiltInTriangleIntersectionAttributes;

// Ray flags beside Slang's `RAY_FLAG_*`: opacity micromaps forced to two states. Pipelines
// see micromaps only when created for them; builds without micromaps drop the flag.
public static const uint RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE = 0x400;

// Loads `*p` as another shader of the path stored it, made visible at device scope.
[ForceInline]
public T load_coherent<T>(T* p)
{
    return spirv_asm {
        OpCapability VulkanMemoryModelDeviceScope;
        result : $$T = OpLoad $p Aligned|MakePointerVisible|NonPrivatePointer !4 $(1u);
    };
}

// Stores `v` to `*p` for the other shaders of the path, made available at device scope.
[ForceInline]
public void store_coherent<T>(T* p, T v)
{
    spirv_asm {
        OpCapability VulkanMemoryModelDeviceScope;
        OpStore $p $v Aligned|MakePointerAvailable|NonPrivatePointer !4 $(1u);
    };
}

[ForceInline]
uint ray_flags(uint flags)
{
#if defined(ORR_OMM_BUILD)
    if ((flags & RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE) != 0)
    {
        spirv_asm
        {
            OpCapability RayTracingOpacityMicromapKHR;
            OpExtension "SPV_KHR_opacity_micromap"
        };
    }
    return flags;
#else
    return flags & ~RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
#endif
}

// Traces `ray` and runs the closest-hit or miss shader it finds, from a raygen, closest-hit or
// miss shader.
[ForceInline]
public void trace<P>(Tlas tlas, uint flags, uint mask, uint ray_type, uint ray_types, uint miss, RayDesc ray, inout P payload)
{
    TraceRay(tlas.get(), ray_flags(flags), mask, ray_type, ray_types, miss, ray, payload);
}

// Runs callable shader `index` of the binding table's callable region.
[ForceInline]
public void call<D>(uint index, inout D data)
{
    CallShader(index, data);
}

// Lets the system regroup this thread with others of equal `hint` (its low `bits` bits)
// before it goes on. Raygen only.
[ForceInline]
public void reorder(uint hint, uint bits)
{
    ReorderThread(hint, bits);
}

// The record data a hit object names. The build decorates this type's instances as glslang
// decorates a `hitobjectshaderrecordext` buffer reference, which Slang has no syntax for: the
// driver reads the record only through a pointer to such a type.
struct HitObjectShaderRecordData<T>
{
    T value;
}

// A traced hit or miss whose shader has not run, from `trace_hit`. Raygen only, where its
// value lives across reorders and the invoke. It is the hit object itself: Slang drops the
// copy of a struct that holds one, leaving the copy's hit object undefined.
public typealias Traced = HitObject;

public extension HitObject
{
    public bool hit()
    {
        return this.IsHit();
    }

    public bool miss()
    {
        return this.IsMiss();
    }

    // Neither hit nor miss: invoking it runs nothing.
    public bool empty()
    {
        return this.IsNop();
    }

    // The hit's distance; on a miss, the ray's TMax.
    public float t()
    {
        return this.GetRayTCurrent();
    }

    public uint instance_index()
    {
        return this.GetInstanceIndex();
    }

    public uint custom_index()
    {
        return this.GetInstanceID();
    }

    public uint geometry_index()
    {
        return this.GetGeometryIndex();
    }

    public uint primitive_index()
    {
        return this.GetPrimitiveIndex();
    }

    public uint hit_kind()
    {
        return this.GetHitKind();
    }

    // A triangle hit's barycentrics.
    public float2 barycentrics()
    {
        return this.GetAttributes<TriangleAttributes>().barycentrics;
    }

    // The index of the hit or miss record `invoke` runs.
    public uint record_index()
    {
        return this.GetShaderTableIndex();
    }

    // Makes `invoke` run hit record `index`, counted from the hit region's start, in place of
    // the one traversal chose. A hit only.
    [mutating]
    public void set_record_index(uint index)
    {
        this.SetShaderTableIndex(index);
    }

    // The record data of the record `invoke` runs, as type `T`. A hit or miss only.
    public T record<T>()
    {
        uint2 a = this.GetShaderRecordBufferHandle();
        let p = (HitObjectShaderRecordData<T>*)(uint64_t(a.x) | (uint64_t(a.y) << 32));
        return p.value;
    }

    // Lets the system regroup this thread by the shader `invoke` runs, then where it hit.
    public void reorder()
    {
        ReorderThread(this);
    }

    // As above, with `hint`'s low `bits` bits ranked between the shader and the place.
    public void reorder(uint hint, uint bits)
    {
        ReorderThread(this, hint, bits);
    }

    // Runs the closest-hit or miss shader of the record, with `payload`.
    public void invoke<P>(inout P payload)
    {
        HitObject::Invoke(this, payload);
    }
}

// Traces `ray` running only intersection and any-hit shaders: what it found, to read, reorder
// and invoke. Raygen only.
[ForceInline]
public Traced trace_hit<P>(Tlas tlas, uint flags, uint mask, uint ray_type, uint ray_types, uint miss, RayDesc ray, inout P payload)
{
    return HitObject::TraceRay(tlas.get(), ray_flags(flags), mask, ray_type, ray_types, miss, ray, payload);
}

// A miss of `ray` that runs miss record `miss` when invoked. Raygen only.
[ForceInline]
public Traced traced_miss(uint miss, RayDesc ray)
{
    return HitObject::MakeMiss(miss, ray);
}

// Neither hit nor miss: reorders by hint alone and invokes nothing. Raygen only.
[ForceInline]
public Traced traced_nothing()
{
    return HitObject::MakeNop();
}

// What a hit shader's body reads of the hit, so the same body compiles as the shader and as
// a compute stand-in that reads it from memory.
public interface IHit
{
    // The index of the thread in the launch.
    uint3 launch();
    uint instance_index();
    uint custom_index();
    uint geometry_index();
    uint primitive_index();
    // -1 outside cluster geometry.
    int cluster_id();
    float t();
    float2 barycentrics();
    bool front_face();
    float3 world_origin();
    float3 world_direction();
    float3 object_origin();
    float3 object_direction();
    // Corner `i` of a triangle hit in object space; the geometry allows data access.
    float3 corner(uint i);
}

// The hit as built-ins give it, in closest-hit and any-hit shaders of triangle geometry.
public struct TriangleHit : IHit
{
    public TriangleAttributes attributes;

    public uint3 launch()
    {
        return DispatchRaysIndex();
    }

    public uint instance_index()
    {
        return InstanceIndex();
    }

    public uint custom_index()
    {
        return InstanceID();
    }

    public uint geometry_index()
    {
        return GeometryIndex();
    }

    public uint primitive_index()
    {
        return PrimitiveIndex();
    }

    public int cluster_id()
    {
        return GetClusterID();
    }

    public float t()
    {
        return RayTCurrent();
    }

    public float2 barycentrics()
    {
        return attributes.barycentrics;
    }

    public bool front_face()
    {
        return HitKind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    }

    public float3 world_origin()
    {
        return WorldRayOrigin();
    }

    public float3 world_direction()
    {
        return WorldRayDirection();
    }

    public float3 object_origin()
    {
        return ObjectRayOrigin();
    }

    public float3 object_direction()
    {
        return ObjectRayDirection();
    }

    public float3 corner(uint i)
    {
        return HitTriangleVertexPosition(i);
    }
}

// A hit as a compute stand-in reads it: the values `TriangleHit` gives, from memory.
public struct HitInputs
{
    public uint3 launch;
    public uint instance_index;
    public uint custom_index;
    public uint geometry_index;
    public uint primitive_index;
    public int cluster_id;
    public float2 barycentrics;
    public float t;
    public uint front_face;
    public float3 world_origin;
    public uint pad0;
    public float3 world_direction;
    public uint pad1;
    public float3 object_origin;
    public uint pad2;
    public float3 object_direction;
    public uint pad3;
    public float3 corners[3];
    public uint pad4;
}

// The hit a compute stand-in reads from `HitInputs` in memory.
public struct StoredHit : IHit
{
    public HitInputs* at;

    public uint3 launch()
    {
        return at.launch;
    }

    public uint instance_index()
    {
        return at.instance_index;
    }

    public uint custom_index()
    {
        return at.custom_index;
    }

    public uint geometry_index()
    {
        return at.geometry_index;
    }

    public uint primitive_index()
    {
        return at.primitive_index;
    }

    public int cluster_id()
    {
        return at.cluster_id;
    }

    public float t()
    {
        return at.t;
    }

    public float2 barycentrics()
    {
        return at.barycentrics;
    }

    public bool front_face()
    {
        return at.front_face != 0;
    }

    public float3 world_origin()
    {
        return at.world_origin;
    }

    public float3 world_direction()
    {
        return at.world_direction;
    }

    public float3 object_origin()
    {
        return at.object_origin;
    }

    public float3 object_direction()
    {
        return at.object_direction;
    }

    public float3 corner(uint i)
    {
        return at.corners[i];
    }
}
