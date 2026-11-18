// Ray tracing pipeline shaders. This is the only module that names `TraceRay`, `HitObject`,
// `CallShader`, `ReorderThread` and the other intrinsics of the pipeline's stages, as `trace` is
// for `RayQuery`. Targets without such a pipeline compile its compute form instead
// (ORR_PIPELINE_COMPUTE, below).
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
import orr_shaders.debug;
import orr_shaders.trace;
__exported import orr_shaders.hits;

// What crosses a trace, invoke or call in registers. A launch pays registers in every thread
// for its widest payload past 16 bytes, so pipelines pass this and keep the rest in memory.
public struct Payload
{
    public uint4 words;
}

#if !defined(ORR_PIPELINE_COMPUTE)

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

// The thread's index in the launch, in any shader of the pipeline.
[ForceInline]
public uint3 launch_index()
{
    return DispatchRaysIndex();
}

// The launch's extent.
[ForceInline]
public uint3 launch_size()
{
    return DispatchRaysDimensions();
}

// Whether the thread's launch index is inside the launch's extent: always, here.
[ForceInline]
public bool launch_inside()
{
    return true;
}

// Ends an any-hit shader by its decision `d` (ANY_HIT_*): ignores the candidate, accepts it, or
// accepts it and ends the search. The shader's last act; its payload writes stay whatever it
// decides.
[ForceInline]
public void any_hit_decides(uint d)
{
    if (d == ANY_HIT_END)
        AcceptHitAndEndSearch();
    if (d == ANY_HIT_IGNORE)
        IgnoreHit();
}

// Ends an any-hit shader ignoring its candidate, as `any_hit_decides(ANY_HIT_IGNORE)`.
[ForceInline]
public void ignore_hit()
{
    IgnoreHit();
}

// Reports a hit of an intersection shader at `t` with hit kind `kind` (below 128) and attributes
// `a`, of at most 8 bytes, which its group's any-hit and closest-hit shaders take. The shader's
// last act: a hit outside the ray's span is not reported, and a shader reports one hit at most.
[ForceInline]
public void report_hit<A>(float t, uint kind, A a)
{
    ReportHit(t, kind, a);
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

// `call` from a stage of a pipeline; nothing from a compute kernel, as a compute stand-in of a
// stage's body is (`orr_lighting.path.standins`), which holds no callables.
[ForceInline]
public void call_from_stage<D>(uint index, inout D data)
{
    __stage_switch
    {
    case raygen:
    case closesthit:
    case miss:
    case callable:
        CallShader(index, data);
        break;
    default:
        break;
    }
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

// Whether every thread of a group reaches `group_any` and `reorder_invoke` in the same sequence,
// as a form that reorders invokes across its group needs: a raygen loop that reorders runs while
// `group_any` says any thread's work goes on and reaches one `reorder_invoke` an iteration, where
// a thread with nothing to invoke, or whose work ended, passes `traced_nothing()`. Here the
// system reorders, and each thread's loop ends where its own work does.
public static const bool UNIFORM_REORDER = false;

// Whether `live` holds on any thread of the group, under UNIFORM_REORDER, where every thread calls
// it together; here, `live`.
[ForceInline]
public bool group_any(bool live)
{
    return live;
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

// What an intersection shader, and the any-hit and closest-hit shaders of a group with one, read of
// the ray and of the candidate or hit, as built-ins give it. `t` is the candidate's distance in an
// any-hit shader, the hit's in a closest-hit shader and the nearest hit's so far in an
// intersection shader.
public struct RayBuiltIns
{
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

    public float t_min()
    {
        return RayTMin();
    }

    public float t()
    {
        return RayTCurrent();
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
}

// The built-ins of the shader that calls it.
[ForceInline]
public RayBuiltIns ray_builtins()
{
    RayBuiltIns r = {};
    return r;
}

#else

// The compute form, under ORR_PIPELINE_COMPUTE: a target with no ray tracing pipeline (Metal) runs
// each raygen as a launch kernel of 64-thread groups, and its other stages as functions the build
// wraps: closest-hit, miss and callable stages as functions of a table the launch block names,
// indexed by their place in the pipeline's stages; any-hit and intersection stages as intersection
// functions the hit table names, which the traversal calls for each candidate. The build generates
// a body per stage that calls its entry with what it takes, and wraps each body in the target's
// signature (`orr_shaders::build`'s Metal templates); what the bodies and wrappers share with this
// module is laid out below:
//
// - A stage gets its payload or callable data by reference to the caller's own, and its hit by
//   value as a `Traced`: a triangle group's entries take it as their `TriangleAttributes`, and the
//   built-ins of `BuiltIns` and `RayBuiltIns` read it. Ambient, per wrapper call, are the launch
//   index the stage runs for, the hit `RayBuiltIns` reads, an any-hit's decision and an
//   intersection's report.
// - `report_hit`, `ignore_hit` and `any_hit_decides` set what their shader returns; they end
//   nothing, so each is its shader's last act. A procedural group's any-hit runs on the report
//   once its intersection returns.
// - A hit record starts with its 32-byte handle: the intersection function's resource ID (8
//   bytes), then the closest-hit, any-hit and intersection stage indices, each plus 1, 0 for none.
//   A miss or callable record's handle is its stage index plus 1. Record data follows the handle.

// Bytes of a record's handle, which its data follows.
static const uint HANDLE_BYTES = 32;

// The kinds a `Traced` holds in the top byte of its custom index word, past the hit kinds a hit
// has: 0 to 127 a procedural hit's, HIT_KIND_TRIANGLE_FRONT_FACE and HIT_KIND_TRIANGLE_BACK_FACE a
// triangle's.
static const uint TRACED_MISS = 0x80;
static const uint TRACED_NOTHING = 0x81;

// The attributes of a triangle hit: its barycentrics, and the hit, which `BuiltIns` reads.
public struct TriangleAttributes
{
    public Traced hit;

    public property float2 barycentrics
    {
        get
        {
            return hit.barycentrics();
        }
    }
}

// The attributes a triangle group's entries take for `hit`.
public TriangleAttributes triangle_attributes(Traced hit)
{
    TriangleAttributes a = { hit };
    return a;
}

// A traced hit or miss, or nothing, in 64 bytes, as raygens keep it and stages are handed it. The
// Metal templates write these words, so their layout is fixed:
// - w0: record index (the hit or miss record), t (asuint; a miss's is the ray's TMax), instance
//   index, custom index (24 bits) with the kind (TRACED_* or a hit kind) in the top byte.
// - w1: a triangle hit's primitive data address (xy) and barycentrics (zw); a procedural hit's
//   geometry and primitive index (xy) and attributes (zw, up to 8 bytes).
// - w2, w3: the ray's world origin and direction, and the address of the `Tlas` it traced.
// A triangle's primitive data is what the backend gives every triangle (`orr_shaders.trace`): its
// cluster ID, geometry index and flags, triangle in the cluster and corners.
public struct Traced
{
    public uint4 w0;
    public uint4 w1;
    public uint4 w2;
    public uint4 w3;

    uint kind()
    {
        return w0.w >> 24;
    }

    public bool hit()
    {
        return kind() < TRACED_MISS || kind() > TRACED_NOTHING;
    }

    public bool miss()
    {
        return kind() == TRACED_MISS;
    }

    // Neither hit nor miss: invoking it runs nothing.
    public bool empty()
    {
        return kind() == TRACED_NOTHING;
    }

    // A hit on a triangle rather than a procedural primitive.
    public bool triangle()
    {
        return kind() == HIT_KIND_TRIANGLE_FRONT_FACE || kind() == HIT_KIND_TRIANGLE_BACK_FACE;
    }

    // The hit's distance; on a miss, the ray's TMax.
    public float t()
    {
        return asfloat(w0.y);
    }

    public uint instance_index()
    {
        return w0.z;
    }

    public uint custom_index()
    {
        return w0.w & 0xffffff;
    }

    public uint hit_kind()
    {
        return kind();
    }

    // A triangle's primitive data, as 32-bit words.
    uint* data()
    {
        return (uint*)(uint64_t(w1.x) | (uint64_t(w1.y) << 32));
    }

    public uint geometry_index()
    {
        return triangle() ? data()[1] & 0xffffff : w1.x;
    }

    public uint primitive_index()
    {
        return triangle() ? data()[2] : w1.y;
    }

    // -1 outside cluster geometry.
    public int cluster_id()
    {
        return triangle() ? int(data()[0]) : -1;
    }

    // A triangle hit's barycentrics.
    public float2 barycentrics()
    {
        return asfloat(w1.zw);
    }

    // A procedural hit's attributes, as its intersection reported them.
    public A attributes<A>()
    {
        return reinterpret<A>(w1.zw);
    }

    // A triangle hit's corners in object space, from its primitive data.
    public float3[3] corners()
    {
        float* f = (float*)(data() + 3);
        float3 p[3];
        for (uint c = 0; c < 3; c++)
            p[c] = float3(f[c * 3], f[c * 3 + 1], f[c * 3 + 2]);
        return p;
    }

    public float3 world_origin()
    {
        return asfloat(w2.xyz);
    }

    public float3 world_direction()
    {
        return asfloat(uint3(w2.w, w3.xy));
    }

    uint64_t tlas()
    {
        return uint64_t(w3.z) | (uint64_t(w3.w) << 32);
    }

    // The linear part of a hit's object-to-world transform, which takes an object-space direction
    // `v` to `mul(linear, v)`.
    public float3x3 object_to_world()
    {
        return instance_linear(instance_descriptor(tlas(), instance_index()));
    }

    // The ray in the hit instance's object space.
    public float3 object_origin()
    {
        uint64_t d = instance_descriptor(tlas(), instance_index());
        return mul(inverse_linear(instance_linear(d)), world_origin() - instance_column(d, 3));
    }

    public float3 object_direction()
    {
        uint64_t d = instance_descriptor(tlas(), instance_index());
        return mul(inverse_linear(instance_linear(d)), world_direction());
    }

    // The index of the hit or miss record `invoke` runs.
    public uint record_index()
    {
        return w0.x;
    }

    // Makes `invoke` run hit record `index`, counted from the hit region's start, in place of
    // the one traversal chose. A hit only.
    [mutating]
    public void set_record_index(uint index)
    {
        w0.x = index;
    }

    // The address of the record `invoke` runs.
    uint64_t record_address()
    {
        PipelineLaunch l = metal_launch;
        if (miss())
            return l.misses + uint64_t(record_index()) * l.miss_stride;
        return l.hits + uint64_t(record_index()) * l.hit_stride;
    }

    // The record data of the record `invoke` runs, as type `T`. A hit or miss only.
    public T record<T>()
    {
        return *(T*)(record_address() + HANDLE_BYTES);
    }

    // The stage `invoke` runs, plus 1; 0 for none.
    uint stage()
    {
        if (empty())
            return 0;
        uint* handle = (uint*)record_address();
        return miss() ? handle[0] : handle[2];
    }

    // Nothing here: a raygen regroups its threads with `reorder_invoke`.
    public void reorder()
    {
    }

    public void reorder(uint hint, uint bits)
    {
    }

    // Runs the closest-hit or miss shader of the record, with `payload`.
    public void invoke<P>(inout P payload)
    {
        invoke_as(this, orr_index, payload);
    }
}

// The Metal instance descriptor of instance `instance` of the TLAS at `tlas`, from the address
// and stride of its descriptors that every TLAS's header holds at bytes 48 and 56.
uint64_t instance_descriptor(uint64_t tlas, uint instance)
{
    uint64_t base = *(uint64_t*)(tlas + 48);
    uint stride = *(uint*)(tlas + 56);
    return base + uint64_t(instance) * stride;
}

// Column `c` of an instance descriptor's object-to-world transform: 3 the translation.
float3 instance_column(uint64_t descriptor, uint c)
{
    float* f = (float*)(descriptor + 12 * c);
    return float3(f[0], f[1], f[2]);
}

float3x3 instance_linear(uint64_t descriptor)
{
    return transpose(float3x3(instance_column(descriptor, 0), instance_column(descriptor, 1), instance_column(descriptor, 2)));
}

float3x3 inverse_linear(float3x3 m)
{
    float3 c0 = cross(m[1], m[2]);
    float3 c1 = cross(m[2], m[0]);
    float3 c2 = cross(m[0], m[1]);
    return transpose(float3x3(c0, c1, c2)) / dot(m[0], c0);
}

// The hit group contribution of an instance descriptor, its intersection function table offset.
uint instance_contribution(uint64_t descriptor)
{
    return *(uint*)(descriptor + 56);
}

// The launch index the running stage works for; a launch kernel's thread's own, or the owner's of
// what it runs while a reorder has it run another thread's work.
static uint3 orr_index = 0;
// A launch kernel thread's index in its group.
static uint orr_lane = 0;
// The hit `RayBuiltIns` reads: the candidate of an any-hit or intersection, the hit of a
// closest-hit.
static Traced orr_hit = { 0, 0, 0, 0 };
// The ray's TMin, in an any-hit or intersection.
static float orr_t_min = 0;
// An any-hit's decision (ANY_HIT_*).
static uint orr_decision = 0;
// An intersection's report: x 1 plus its hit kind times 2, 0 for none; y its t; zw its attributes.
static uint4 orr_report = 0;

// What a launch kernel sets before its raygen runs: the thread's launch index and its index in its
// group.
public void pipeline_launch(uint3 index, uint lane)
{
    orr_index = index;
    orr_lane = lane;
    if (lane == 0)
        orr_any[0] = 0;
    GroupMemoryBarrierWithGroupSync();
}

// The thread's index in the launch, in any shader of the pipeline. A launch runs whole groups of 64
// threads: a raygen thread whose index is past the extent runs with nothing to do, and reaches its
// group's `group_any` and `reorder_invoke` with the others.
public uint3 launch_index()
{
    return orr_index;
}

// The launch's extent.
public uint3 launch_size()
{
    PipelineLaunch l = metal_launch;
    if (l.extent_address != 0)
        return *(uint3*)l.extent_address;
    return l.extent.xyz;
}

// Whether the thread's launch index is inside the launch's extent. A raygen that reaches no
// `group_any` returns at once where it is not.
public bool launch_inside()
{
    return all(orr_index < launch_size());
}

// Loads `*p` as another shader of the path stored it: a reorder or invoke between them orders the
// memory, which the stage's thread reaches through its group.
[ForceInline]
public T load_coherent<T>(T* p)
{
    return *p;
}

[ForceInline]
public void store_coherent<T>(T* p, T v)
{
    *p = v;
}

// Ends an any-hit shader by its decision `d` (ANY_HIT_*): ignores the candidate, accepts it, or
// accepts it and ends the search. The shader's last act; its payload writes stay whatever it
// decides.
public void any_hit_decides(uint d)
{
    orr_decision = d;
}

// Ends an any-hit shader ignoring its candidate, as `any_hit_decides(ANY_HIT_IGNORE)`.
public void ignore_hit()
{
    orr_decision = ANY_HIT_IGNORE;
}

// Reports a hit of an intersection shader at `t` with hit kind `kind` (below 128) and attributes
// `a`, of at most 8 bytes, which its group's any-hit and closest-hit shaders take. The shader's
// last act: a hit outside the ray's span is not reported, and a shader reports one hit at most.
public void report_hit<A>(float t, uint kind, A a)
{
    orr_report = uint4(1 + (kind << 1), asuint(t), reinterpret<uint2>(a));
}

// `flags` without RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE: the target has no micromaps.
uint ray_flags(uint flags)
{
    return flags & ~RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
}

// Traces `ray` and runs the closest-hit or miss shader it finds, from a raygen, closest-hit or
// miss shader.
public void trace<P>(Tlas tlas, uint flags, uint mask, uint ray_type, uint ray_types, uint miss, RayDesc ray, inout P payload)
{
    Traced h = trace_hit(tlas, flags, mask, ray_type, ray_types, miss, ray, payload);
    if ((flags & RAY_FLAG_SKIP_CLOSEST_HIT_SHADER) == 0 || h.miss())
        invoke_as(h, orr_index, payload);
}

// Traces `ray` running only intersection and any-hit shaders: what it found, to read, reorder
// and invoke. Raygen only.
public Traced trace_hit<P>(Tlas tlas, uint flags, uint mask, uint ray_type, uint ray_types, uint miss, RayDesc ray, inout P payload)
{
    PipelineLaunch l = metal_launch;
    Traced h;
    h.w2 = uint4(asuint(ray.Origin), asuint(ray.Direction.x));
    h.w3 = uint4(asuint(ray.Direction.yz), uint(tlas.address), uint(tlas.address >> 32));
    uint function_id = 0;
    bool found = metal_trace(tlas.metal_id(), tlas.address, ray_flags(flags), mask, ray_type, ray_types, ray.Origin, ray.Direction, ray.TMin, ray.TMax, payload, l.hits, l.hit_stride, l.hit_count, orr_index, metal_root, metal_texture_pool, metal_samplers, metal_launch, h.w0, h.w1, function_id);
    if (!found)
    {
        h.w0 = uint4(miss, asuint(ray.TMax), 0, TRACED_MISS << 24);
        h.w1 = 0;
        return h;
    }
    uint64_t d = instance_descriptor(tlas.address, h.instance_index());
    h.w0.x = instance_contribution(d) + h.geometry_index() * (ray_types & 15) + (ray_type & 15);
    // Metal adds the instance offsets of every level it passes through, or of the last one; the
    // backend's middle levels add 0, so either gives the index here.
    if (ASSERTS && h.w0.x != function_id)
        pipeline_check_failed(PIPELINE_CHECK_FUNCTION_ID);
    return h;
}

// A miss of `ray` that runs miss record `miss` when invoked. Raygen only.
public Traced traced_miss(uint miss, RayDesc ray)
{
    Traced h;
    h.w0 = uint4(miss, asuint(ray.TMax), 0, TRACED_MISS << 24);
    h.w1 = 0;
    h.w2 = uint4(asuint(ray.Origin), asuint(ray.Direction.x));
    h.w3 = uint4(asuint(ray.Direction.yz), 0, 0);
    return h;
}

// Neither hit nor miss: reorders by hint alone and invokes nothing. Raygen only.
public Traced traced_nothing()
{
    Traced h;
    h.w0 = uint4(0, 0, 0, TRACED_NOTHING << 24);
    h.w1 = 0;
    h.w2 = 0;
    h.w3 = 0;
    return h;
}

// Runs the closest-hit or miss stage of `h`'s record with `payload`, for launch index `index`.
void invoke_as<P>(Traced h, uint3 index, inout P payload)
{
    uint stage = h.stage();
    if (stage != 0)
        table_call(stage - 1, payload, h, h.record_address(), index);
}

// Runs callable shader `index` of the binding table's callable region.
public void call<D>(uint index, inout D data)
{
    PipelineLaunch l = metal_launch;
    uint64_t record = l.callables + uint64_t(index) * l.callable_stride;
    uint stage = *(uint*)record;
    if (stage != 0)
        table_call(stage - 1, data, traced_nothing(), record, orr_index);
}

// `call`: every stage here is a function the launch kernel calls.
public void call_from_stage<D>(uint index, inout D data)
{
    call(index, data);
}

// Nothing here: a raygen regroups its threads with `reorder_invoke`.
public void reorder(uint hint, uint bits)
{
}

// Whether every thread of a group reaches `group_any` and `reorder_invoke` in the same sequence,
// as a form that reorders invokes across its group needs: a raygen loop that reorders runs while
// `group_any` says any thread's work goes on and reaches one `reorder_invoke` an iteration, where
// a thread with nothing to invoke, or whose work ended, passes `traced_nothing()`. Here the
// group's threads sort their invokes by the stage each runs, then by hint.
public static const bool UNIFORM_REORDER = true;

static const uint GROUP = 64;

// `group_any`'s words, a round's in turn; a round clears the next one's.
groupshared uint orr_any[3];
static uint orr_any_round = 0;

// Whether `live` holds on any thread of the group, under UNIFORM_REORDER, where every thread calls
// it together.
public bool group_any(bool live)
{
    uint round = orr_any_round;
    uint next = round == 2 ? 0 : round + 1;
    orr_any_round = next;
    if (orr_lane == 0)
        orr_any[next] = 0;
    if (live)
        orr_any[round] = 1;
    GroupMemoryBarrierWithGroupSync();
    return orr_any[round] != 0;
}

// `reorder_invoke`'s exchange: each thread's sort key and work by its lane, and the lane whose
// work each sorted place runs.
groupshared uint orr_keys[GROUP];
groupshared uint orr_order[GROUP];
groupshared Traced orr_work[GROUP];
groupshared uint4 orr_payloads[GROUP];
groupshared uint3 orr_indices[GROUP];

// `h.reorder()` then `h.invoke(payload)`, as one step every thread of the group takes together
// (UNIFORM_REORDER): the group's work is sorted by the stage it runs, and each thread runs the
// work at its place in that order for the thread whose work it is, which gets its payload back.
// The payload is at most 16 bytes, as `Payload`.
public void reorder_invoke<P>(Traced h, inout P payload)
{
    reorder_invoke(h, 0, 0, payload);
}

// As above, sorted by the stage, then by `hint`'s low `bits` bits.
public void reorder_invoke<P>(Traced h, uint hint, uint bits, inout P payload)
{
    uint lane = orr_lane;
    uint key = (h.stage() << bits) | (hint & ((1u << bits) - 1));
    orr_keys[lane] = key;
    orr_work[lane] = h;
    orr_payloads[lane] = reinterpret<uint4>(payload);
    orr_indices[lane] = orr_index;
    GroupMemoryBarrierWithGroupSync();
    uint place = 0;
    for (uint i = 0; i < GROUP; i++)
    {
        uint k = orr_keys[i];
        place += k < key || (k == key && i < lane) ? 1 : 0;
    }
    orr_order[place] = lane;
    GroupMemoryBarrierWithGroupSync();
    uint owner = orr_order[lane];
    P p = reinterpret<P>(orr_payloads[owner]);
    invoke_as(orr_work[owner], orr_indices[owner], p);
    orr_payloads[owner] = reinterpret<uint4>(p);
    // The stage's stores, made on the owner's behalf, reach the owner past this.
    AllMemoryBarrierWithGroupSync();
    payload = reinterpret<P>(orr_payloads[lane]);
}

// The hit as a closest-hit or any-hit shader of a triangle group reads it: `TriangleHit` for
// geometry of triangles only, `BuiltInHit` for geometry that is triangles or linear swept
// spheres, which this target has none of.
public struct BuiltIns<let SWEPT : bool> : IHit
{
    public TriangleAttributes attributes;

    public bool triangle()
    {
        return true;
    }

    public uint3 launch()
    {
        return orr_index;
    }

    public uint instance_index()
    {
        return attributes.hit.instance_index();
    }

    public uint custom_index()
    {
        return attributes.hit.custom_index();
    }

    public uint geometry_index()
    {
        return attributes.hit.geometry_index();
    }

    public uint primitive_index()
    {
        return attributes.hit.primitive_index();
    }

    public int cluster_id()
    {
        return attributes.hit.cluster_id();
    }

    public float t()
    {
        return attributes.hit.t();
    }

    public float2 barycentrics()
    {
        return attributes.hit.barycentrics();
    }

    public bool front_face()
    {
        return attributes.hit.hit_kind() == HIT_KIND_TRIANGLE_FRONT_FACE;
    }

    public float3 world_origin()
    {
        return attributes.hit.world_origin();
    }

    public float3 world_direction()
    {
        return attributes.hit.world_direction();
    }

    public float3 object_origin()
    {
        return attributes.hit.object_origin();
    }

    public float3 object_direction()
    {
        return attributes.hit.object_direction();
    }

    public float3 corner(uint i)
    {
        return attributes.hit.corners()[i];
    }
}

public typealias TriangleHit = BuiltIns<false>;
public typealias BuiltInHit = BuiltIns<true>;

// What an intersection shader, and the any-hit and closest-hit shaders of a group with one, read of
// the ray and of the candidate or hit, as built-ins give it. `t` is the candidate's distance in an
// any-hit shader, the hit's in a closest-hit shader and the nearest hit's so far in an
// intersection shader.
public struct RayBuiltIns
{
    public uint3 launch()
    {
        return orr_index;
    }

    public uint instance_index()
    {
        return orr_hit.instance_index();
    }

    public uint custom_index()
    {
        return orr_hit.custom_index();
    }

    public uint geometry_index()
    {
        return orr_hit.geometry_index();
    }

    public uint primitive_index()
    {
        return orr_hit.primitive_index();
    }

    public float t_min()
    {
        return orr_t_min;
    }

    public float t()
    {
        return orr_hit.t();
    }

    public float3 world_origin()
    {
        return orr_hit.world_origin();
    }

    public float3 world_direction()
    {
        return orr_hit.world_direction();
    }

    public float3 object_origin()
    {
        return orr_hit.object_origin();
    }

    public float3 object_direction()
    {
        return orr_hit.object_direction();
    }
}

// The built-ins of the shader that calls it.
public RayBuiltIns ray_builtins()
{
    RayBuiltIns r = {};
    return r;
}

// Checks of `ASSERTS` builds that the device counts (`PipelineLaunch.checks`), where the target
// cannot abort.
static const uint PIPELINE_CHECK_FUNCTION_ID = 1;

void pipeline_check_failed(uint check)
{
    uint64_t at = metal_launch.checks;
    if (at != 0)
        InterlockedOr(*(uint*)at, check);
}

// The bodies and wrappers the build generates call these.

// The address of the launch's raygen record, whose data follows the handle.
public uint64_t pipeline_raygen_record()
{
    return metal_launch.raygen;
}

// What an any-hit body returns: whether its candidate is accepted and whether the search goes on.
public uint2 any_hit_result()
{
    return uint2(orr_decision != ANY_HIT_IGNORE ? 1 : 0, orr_decision != ANY_HIT_END ? 1 : 0);
}

// Clears an any-hit's decision to accept, before its entry runs.
public void any_hit_begin()
{
    orr_decision = ANY_HIT_ACCEPT;
}

// Clears an intersection's report, before its entry runs.
public void intersection_begin()
{
    orr_report = 0;
}

// The intersection's report: x 1 plus its hit kind times 2, 0 for none; y its t; zw its
// attributes.
public uint4 intersection_report()
{
    return orr_report;
}

// Calls stage `stage` of the launch's table with `data`, `hit` and the address of its record, for
// launch index `index`.
void table_call<D>(uint stage, inout D data, Traced hit, uint64_t record, uint3 index)
{
    metal_table_call(metal_launch.table, stage, data, hit, record, index, metal_root, metal_texture_pool, metal_samplers, metal_launch);
}

// The definitions the Metal intrinsics below and the build's wrappers share, once a program: the
// build's stage program has it through `pipeline_prelude`. Every intersector and intersection
// function of a pipeline has the tags `triangle_data, instancing, world_space_data, max_levels<3>,
// intersection_function_buffer`, which the build's wrappers repeat.
static const String METAL_PRELUDE = R"(
// A stage function: the caller's data, the hit, the record's address, the launch index, the root,
// the texture pool base, the samplers and the launch block.
typedef void orr_stage_fn(thread void*, thread const void*, ulong, uint3, constant void*, constant ulong*, device void*, constant void*);
// What a trace hands its intersection functions with its payload: what a stage's wrapper needs to
// run its body, the trace's hit table and the report of a procedural hit it accepted.
template <typename P>
struct orr_ray_data
{
    constant void* root;
    constant ulong* pool;
    device void* samplers;
    constant void* launch;
    ulong tlas;
    ulong hits;
    uint hit_stride;
    uint index_x;
    uint index_y;
    uint index_z;
    uint kind;
    uint attribute_x;
    uint attribute_y;
    uint pad;
    // Last, so a function that reads none of it finds the rest where any payload's trace put it.
    P payload;
};
// What an any-hit's and an intersection's functions return.
struct orr_any_result
{
    bool accept [[accept_intersection]];
    bool more [[continue_search]];
};
struct orr_box_result
{
    bool accept [[accept_intersection]];
    bool more [[continue_search]];
    float distance [[distance]];
};
template <typename T>
T orr_from_id(ulong id)
{
    return *reinterpret_cast<thread T*>(&id);
}
)";

void metal_table_call<D>(uint64_t table, uint stage, inout D data, inout Traced hit, uint64_t record, uint3 index, ConstantBuffer<MetalRoot> root, ConstantBuffer<uint64_t> pool, StructuredBuffer<MetalSampler> samplers, ConstantBuffer<PipelineLaunch> launch)
{
    __requirePrelude(METAL_PRELUDE);
    __requirePrelude(R"(
template <typename D>
inline void orr_table_call(ulong table, uint stage, thread D* data, thread const void* hit, ulong record, uint3 index, constant void* root, constant ulong* pool, device void* samplers, constant void* launch)
{
    metal::visible_function_table<orr_stage_fn> t = orr_from_id<metal::visible_function_table<orr_stage_fn>>(table);
    t[stage]((thread void*)data, hit, record, index, root, pool, (device void*)samplers, launch);
}
)");
    __intrinsic_asm "orr_table_call($0, $1, $2, $3, $4, $5, $6, $7, $8, $9)";
}

// Traces with an intersector over the hit table as intersection function buffer: false for a
// miss; for a hit, `w0` (but its record index) and `w1` of its `Traced`, and the index of the
// function buffer entry Metal chose for it.
bool metal_trace<P>(uint64_t id, uint64_t tlas, uint flags, uint mask, uint ray_type, uint ray_types, float3 origin, float3 direction, float t_min, float t_max, inout P payload, uint64_t hits, uint hit_stride, uint hit_count, uint3 index, ConstantBuffer<MetalRoot> root, ConstantBuffer<uint64_t> pool, StructuredBuffer<MetalSampler> samplers, ConstantBuffer<PipelineLaunch> launch, inout uint4 w0, inout uint4 w1, inout uint function_id)
{
    __requirePrelude(METAL_PRELUDE);
    __requirePrelude(R"(
template <typename P>
inline bool orr_trace(ulong id, ulong tlas, uint flags, uint mask, uint ray_type, uint ray_types, float3 origin, float3 direction, float t_min, float t_max, thread P* payload, ulong hits, uint hit_stride, uint hit_count, uint3 index, constant void* root, constant ulong* pool, device void* samplers, constant void* launch, thread uint4* w0, thread uint4* w1, thread uint* function_id)
{
    using namespace metal::raytracing;
    intersector<triangle_data, instancing, world_space_data, max_levels<3>, intersection_function_buffer> i;
    i.set_base_id(ray_type);
    i.set_geometry_multiplier(ray_types);
    if (flags & 0x01) i.force_opacity(forced_opacity::opaque);
    if (flags & 0x02) i.force_opacity(forced_opacity::non_opaque);
    if (flags & 0x04) i.accept_any_intersection(true);
    if (flags & 0x10) i.set_triangle_cull_mode(triangle_cull_mode::back);
    if (flags & 0x20) i.set_triangle_cull_mode(triangle_cull_mode::front);
    if (flags & 0x40) i.set_opacity_cull_mode(opacity_cull_mode::opaque);
    if (flags & 0x80) i.set_opacity_cull_mode(opacity_cull_mode::non_opaque);
    if (flags & 0x100) i.set_geometry_cull_mode(geometry_cull_mode::triangle);
    if (flags & 0x200) i.set_geometry_cull_mode(geometry_cull_mode::bounding_box);
    intersection_function_buffer_arguments ifb;
    ifb.intersection_function_buffer = reinterpret_cast<const device void*>(hits);
    ifb.intersection_function_buffer_size = size_t(hit_count) * hit_stride;
    ifb.intersection_function_stride = hit_stride;
    orr_ray_data<P> rd;
    rd.payload = *payload;
    rd.root = root;
    rd.pool = pool;
    rd.samplers = samplers;
    rd.launch = launch;
    rd.tlas = tlas;
    rd.hits = hits;
    rd.hit_stride = hit_stride;
    rd.index_x = index.x;
    rd.index_y = index.y;
    rd.index_z = index.z;
    rd.kind = 0;
    rd.attribute_x = 0;
    rd.attribute_y = 0;
    rd.pad = 0;
    acceleration_structure<instancing> as = orr_from_id<acceleration_structure<instancing>>(id);
    auto r = i.intersect(ray(origin, direction, t_min, t_max), mask, as, ifb, rd);
    *payload = rd.payload;
    if (r.type == intersection_type::none)
        return false;
    bool tri = r.type == intersection_type::triangle;
    uint kind = tri ? (r.triangle_front_facing ? 0xfeu : 0xffu) : rd.kind;
    *w0 = uint4(0, as_type<uint>(r.distance), r.instance_id[0], (r.user_instance_id[0] & 0xffffffu) | (kind << 24));
    ulong data = reinterpret_cast<ulong>(r.primitive_data);
    *w1 = tri ? uint4(uint(data), uint(data >> 32), as_type<uint2>(r.triangle_barycentric_coord))
              : uint4(r.geometry_id, r.primitive_id, rd.attribute_x, rd.attribute_y);
    *function_id = r.function_id;
    return true;
}
)");
    __intrinsic_asm "orr_trace($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)";
}

// Brings the prelude into a program that has none of the intrinsics above, as the build's stage
// program, whose wrappers use it.
public void pipeline_prelude()
{
    __requirePrelude(METAL_PRELUDE);
    __intrinsic_asm "";
}

#endif
