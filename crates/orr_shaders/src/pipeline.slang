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
import orr_shaders.trace;
__exported import orr_shaders.hits;

// What crosses a trace, invoke or call in registers. A launch pays registers in every thread
// for its widest payload past 16 bytes, so pipelines pass this and keep the rest in memory.
public struct Payload
{
    public uint4 words;
}

// The attributes of a triangle hit; an intersection shader reports its own type.
public typealias TriangleAttributes = BuiltInTriangleIntersectionAttributes;

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

// `load_coherent` of texel `at` of `image`.
[ForceInline]
public float4 load_texel_coherent(RWTexture2D<float4> image, uint2 at)
{
    return spirv_asm {
        OpCapability VulkanMemoryModelDeviceScope;
        result : $$float4 = OpImageRead $image $at MakeTexelVisible|NonPrivateTexel $(1u);
    };
}

// `store_coherent` of texel `at` of `image`.
[ForceInline]
public void store_texel_coherent(RWTexture2D<float4> image, uint2 at, float4 v)
{
    spirv_asm {
        OpCapability VulkanMemoryModelDeviceScope;
        OpImageWrite $image $at $v MakeTexelAvailable|NonPrivateTexel $(1u);
    };
}

// `flags` with `orr_shaders.trace`'s RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, which pipelines see
// only when created for micromaps, and builds without micromaps drop.
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

// Ends an any-hit shader by its decision `d` (ANY_HIT_*): ignores the candidate, accepts it, or
// accepts it and ends the search.
[ForceInline]
public void any_hit_decides(uint d)
{
    if (d == ANY_HIT_END)
        AcceptHitAndEndSearch();
    if (d == ANY_HIT_IGNORE)
        IgnoreHit();
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
        return this.GetRayDesc().TMax;
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

    // A triangle hit's corners in object space, as its acceleration structure holds them.
    public float3[3] corners()
    {
        return this.GetTriangleVertexPositions();
    }

    // The linear part of a hit's object-to-world transform, which takes an object-space direction
    // `v` to `mul(linear, v)`.
    public float3x3 object_to_world()
    {
        float4x3 m = this.GetObjectToWorld();
        return transpose(float3x3(m[0], m[1], m[2]));
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

// `h.reorder()` then `h.invoke(payload)` as the extension's one instruction, which Slang has no
// name for: the driver keeps no hit object of the caller's across the regrouping.
[ForceInline]
public void reorder_invoke<P>(Traced h, inout P payload)
{
    [__vulkanRayPayload]
    static P p;
    p = payload;
    spirv_asm
    {
        OpExtension "SPV_EXT_shader_invocation_reorder";
        OpCapability ShaderInvocationReorderEXT;
        OpHitObjectReorderExecuteShaderEXT &h &p
    };
    payload = p;
}

// As above, after `h.reorder(hint, bits)`.
[ForceInline]
public void reorder_invoke<P>(Traced h, uint hint, uint bits, inout P payload)
{
    [__vulkanRayPayload]
    static P p;
    p = payload;
    spirv_asm
    {
        OpExtension "SPV_EXT_shader_invocation_reorder";
        OpCapability ShaderInvocationReorderEXT;
        OpHitObjectReorderExecuteShaderEXT &h &p $hint $bits
    };
    payload = p;
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

// The hit as built-ins give it in closest-hit and any-hit shaders: `TriangleHit` for geometry of
// triangles only, `BuiltInHit` for geometry that is triangles or linear swept spheres. A
// swept-sphere hit has no corners, cluster or barycentrics: its first corner is the hit point in
// object space, the others and its barycentrics 0, and its cluster id -1, as `orr_shaders.trace`
// gives an inline hit on one.
public struct BuiltIns<let SWEPT : bool> : IHit
{
    public TriangleAttributes attributes;

    // Whether the hit is on a triangle.
    public bool triangle()
    {
        if (!SWEPT)
            return true;
        uint kind = HitKind();
        return kind == HIT_KIND_TRIANGLE_FRONT_FACE || kind == HIT_KIND_TRIANGLE_BACK_FACE;
    }

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
        return triangle() ? GetClusterID() : -1;
    }

    public float t()
    {
        return RayTCurrent();
    }

    public float2 barycentrics()
    {
        return triangle() ? attributes.barycentrics : float2(0.0);
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
        if (triangle())
            return HitTriangleVertexPosition(i);
        return i == 0 ? ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection() : float3(0.0);
    }
}

public typealias TriangleHit = BuiltIns<false>;
public typealias BuiltInHit = BuiltIns<true>;
