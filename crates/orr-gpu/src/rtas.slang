// Argument records of cluster and partitioned acceleration-structure operations. Kernels fill
// the logical structs below and write them with `store`, which packs each record in the
// layout the backend reads; `CommandList::rtas_execute` consumes the arrays. Addresses are
// device addresses; vertex positions are f32x3 with a 12-byte stride.
module rtas;

import orr.debug;

// `ClasBuild.flags`: the micromap index may name `OMM_INDEX_DISABLE` per triangle.
public static const uint CLUSTER_ALLOW_DISABLE_OMM = 1;

// Geometry flags in `geometry_index_and_flags`.
public static const uint GEOMETRY_CULL_DISABLE = 1;
public static const uint GEOMETRY_NO_DUPLICATE_ANY_HIT = 2;
public static const uint GEOMETRY_OPAQUE = 4;

// Special micromap indices of a triangle in a cluster.
public static const int OMM_INDEX_TRANSPARENT = -1;
public static const int OMM_INDEX_OPAQUE = -2;
public static const int OMM_INDEX_UNKNOWN_TRANSPARENT = -3;
public static const int OMM_INDEX_UNKNOWN_OPAQUE = -4;
// The triangle ignores the micromap and follows its geometry flags.
public static const int OMM_INDEX_DISABLE = -5;

// Instance flags of `PtlasWrite` and `ClassicInstance`.
public static const uint INSTANCE_CULL_DISABLE = 1;
public static const uint INSTANCE_FLIP_FACING = 2;
public static const uint INSTANCE_FORCE_OPAQUE = 4;
public static const uint INSTANCE_FORCE_NO_OPAQUE = 8;
// `PtlasWrite.aabb_min` and `aabb_max` bound the instance in world space (the transform
// applied, the partition translation not).
public static const uint INSTANCE_EXPLICIT_AABB = 16;

// `PtlasWrite.partition` of an instance in the global partition.
public static const uint PTLAS_GLOBAL_PARTITION = 0xffffffff;

// `PtlasCommand.op`. One build takes at most one command of each.
public static const uint PTLAS_WRITE = 0;
public static const uint PTLAS_UPDATE = 1;
public static const uint PTLAS_TRANSLATE = 2;

// A geometry index (24 bits) with `GEOMETRY_*` flags, for `ClasBuild.base_geometry_and_flags`
// and per-triangle flag words. A triangle's word adds its index to the base and ORs its
// flags with the base flags.
public uint geometry_index_and_flags(uint index, uint flags)
{
    return (index & 0xffffff) | (flags << 29);
}

// One cluster (CLAS) build.
public struct ClasBuild
{
    public uint cluster_id;
    public uint flags;         // CLUSTER_* bits
    public uint tri_count;
    public uint vertex_count;
    public uint truncate_bits; // low mantissa bits dropped from every position
    public uint index_format;  // bytes per index: 1, 2 or 4
    public uint base_geometry_and_flags;
    public uint omm_index_format; // bytes per micromap index: 1, 2 or 4
    public uint64_t vertices;
    public uint64_t indices;   // three per triangle
    public uint64_t triangle_flags; // a geometry_index_and_flags word per triangle; 0: none
    public uint64_t omm_array; // the micromap's address; 0: none
    public uint64_t omm_indices; // a micromap index per triangle
}

// One cluster template build: a cluster whose positions come at instantiation.
public struct TemplateBuild
{
    public ClasBuild cluster;
    // Two float3 corners (min, max) bounding every instantiation, 32-byte aligned; 0: none.
    public uint64_t bounds;
}

// One template instantiation into a CLAS.
public struct Instantiate
{
    public uint cluster_id_offset;
    public uint geometry_index_offset;
    public uint64_t template_address;
    // The positions, one per template vertex. Required: 0 loses the device.
    public uint64_t vertices;
}

// One cluster BLAS over `clas_count` CLAS addresses (u64 each) at `clas_addresses`.
public struct ClusterBlas
{
    public uint64_t clas_addresses;
    public uint clas_count;
    public uint pad;
}

// One object move.
public struct Move
{
    public uint64_t src;
}

// Writes an instance into a PTLAS partition, replacing whatever the index held. A BLAS
// address of 0 disables the instance; it comes back only by another write.
public struct PtlasWrite
{
    public float3x4 transform; // object to partition space, row-major
    public float3 aabb_min;    // with INSTANCE_EXPLICIT_AABB
    public float3 aabb_max;
    public uint64_t blas;
    public uint custom_index;  // 24 bits
    public uint mask;
    public uint flags;         // INSTANCE_* bits
    public uint instance_index;
    public uint partition;     // or PTLAS_GLOBAL_PARTITION
    public uint pad;
}

// Points an instance at another BLAS without rebuilding its partition.
public struct PtlasUpdate
{
    public uint64_t blas;
    public uint instance_index;
    public uint pad;
}

// Sets a partition's translation, added to the transform of every instance in it. Replaces
// the previous translation.
public struct PtlasTranslate
{
    public float3 translation;
    public uint partition;
}

// One argument array of a PTLAS build: `count` records of the `op`'s type at `args`.
public struct PtlasCommand
{
    public uint64_t args;
    public uint op;
    public uint count;
}

// One classic TLAS instance in the Vulkan layout (`VkAccelerationStructureInstanceKHR`),
// as `classic_instance` packs it. A BLAS address of 0 makes it inactive.
public struct ClassicInstance
{
    public float3x4 transform; // object to world, row-major
    public uint custom_index_and_mask;
    public uint sbt_offset_and_flags;
    public uint64_t blas;
}

// A classic instance: a 24-bit custom index, a visibility mask and INSTANCE_* flags except
// INSTANCE_EXPLICIT_AABB.
public ClassicInstance classic_instance(float3x4 transform, uint custom_index, uint mask, uint flags, uint64_t blas)
{
    if (ASSERTS && custom_index >= (1u << 24))
        abort("rtas: an instance custom index above 24 bits");
    ClassicInstance v;
    v.transform = transform;
    v.custom_index_and_mask = (custom_index & 0xffffff) | (mask << 24);
    v.sbt_offset_and_flags = flags << 24;
    v.blas = blas;
    return v;
}

// The translation of a partition whose cell is `cell` while the origin sits in
// `origin_cell`, for cells of 2^cell_log2 metres. Exact in f32.
public float3 partition_translation(int3 cell, int3 origin_cell, uint cell_log2)
{
    return float3((cell - origin_cell) << cell_log2);
}

// Record sizes in the backend's layout.
public static const uint CLAS_BUILD_STRIDE = 64;
public static const uint TEMPLATE_BUILD_STRIDE = 72;
public static const uint INSTANTIATE_STRIDE = 32;
public static const uint CLUSTER_BLAS_STRIDE = 16;
public static const uint MOVE_STRIDE = 8;
public static const uint PTLAS_WRITE_STRIDE = 104;
public static const uint PTLAS_UPDATE_STRIDE = 16;
public static const uint PTLAS_TRANSLATE_STRIDE = 16;
public static const uint PTLAS_COMMAND_STRIDE = 24;
public static const uint CLASSIC_INSTANCE_STRIDE = 64;

uint* record(uint64_t args, uint index, uint stride)
{
    return (uint*)(args + uint64_t(index) * stride);
}

void put64(uint* p, uint word, uint64_t v)
{
    p[word] = uint(v);
    p[word + 1] = uint(v >> 32);
}

void put_transform(uint* p, float3x4 m)
{
    for (uint r = 0; r < 3; r++)
        for (uint c = 0; c < 4; c++)
            p[r * 4 + c] = asuint(m[r][c]);
}

void put_cluster(uint* p, ClasBuild v)
{
    bool omm = v.omm_indices != 0;
    p[0] = v.cluster_id;
    p[1] = v.flags;
    p[2] = v.tri_count | (v.vertex_count << 9) | (v.truncate_bits << 18) | (v.index_format << 24)
        | ((omm ? v.omm_index_format : 1) << 28);
    p[3] = v.base_geometry_and_flags;
    p[4] = v.index_format | (12 << 16);
    p[5] = (v.triangle_flags != 0 ? 4 : 0) | ((omm ? v.omm_index_format : 0) << 16);
    put64(p, 6, v.indices);
    put64(p, 8, v.vertices);
    put64(p, 10, v.triangle_flags);
    put64(p, 12, v.omm_array);
    put64(p, 14, v.omm_indices);
}

public void store(uint64_t args, uint index, ClasBuild v)
{
    put_cluster(record(args, index, CLAS_BUILD_STRIDE), v);
}

public void store(uint64_t args, uint index, TemplateBuild v)
{
    uint* p = record(args, index, TEMPLATE_BUILD_STRIDE);
    put_cluster(p, v.cluster);
    put64(p, 16, v.bounds);
}

public void store(uint64_t args, uint index, Instantiate v)
{
    if (ASSERTS && v.vertices == 0)
        abort("rtas: a template instantiation without vertices");
    uint* p = record(args, index, INSTANTIATE_STRIDE);
    p[0] = v.cluster_id_offset;
    p[1] = v.geometry_index_offset & 0xffffff;
    put64(p, 2, v.template_address);
    put64(p, 4, v.vertices);
    put64(p, 6, 12);
}

public void store(uint64_t args, uint index, ClusterBlas v)
{
    uint* p = record(args, index, CLUSTER_BLAS_STRIDE);
    p[0] = v.clas_count;
    p[1] = 8;
    put64(p, 2, v.clas_addresses);
}

public void store(uint64_t args, uint index, Move v)
{
    put64(record(args, index, MOVE_STRIDE), 0, v.src);
}

public void store(uint64_t args, uint index, PtlasWrite v)
{
    if (ASSERTS && v.custom_index >= (1u << 24))
        abort("rtas: an instance custom index above 24 bits");
    uint* p = record(args, index, PTLAS_WRITE_STRIDE);
    put_transform(p, v.transform);
    p[12] = asuint(v.aabb_min.x);
    p[13] = asuint(v.aabb_min.y);
    p[14] = asuint(v.aabb_min.z);
    p[15] = asuint(v.aabb_max.x);
    p[16] = asuint(v.aabb_max.y);
    p[17] = asuint(v.aabb_max.z);
    p[18] = v.custom_index;
    p[19] = v.mask;
    p[20] = 0;
    p[21] = v.flags;
    p[22] = v.instance_index;
    p[23] = v.partition;
    put64(p, 24, v.blas);
}

public void store(uint64_t args, uint index, PtlasUpdate v)
{
    uint* p = record(args, index, PTLAS_UPDATE_STRIDE);
    p[0] = v.instance_index;
    p[1] = 0;
    put64(p, 2, v.blas);
}

public void store(uint64_t args, uint index, PtlasTranslate v)
{
    uint* p = record(args, index, PTLAS_TRANSLATE_STRIDE);
    p[0] = v.partition;
    p[1] = asuint(v.translation.x);
    p[2] = asuint(v.translation.y);
    p[3] = asuint(v.translation.z);
}

public void store(uint64_t args, uint index, PtlasCommand v)
{
    uint stride = v.op == PTLAS_WRITE ? PTLAS_WRITE_STRIDE
        : v.op == PTLAS_UPDATE ? PTLAS_UPDATE_STRIDE : PTLAS_TRANSLATE_STRIDE;
    uint* p = record(args, index, PTLAS_COMMAND_STRIDE);
    p[0] = v.op;
    p[1] = v.count;
    put64(p, 2, v.args);
    put64(p, 4, stride);
}

public void store(uint64_t args, uint index, ClassicInstance v)
{
    uint* p = record(args, index, CLASSIC_INSTANCE_STRIDE);
    put_transform(p, v.transform);
    p[12] = v.custom_index_and_mask;
    p[13] = v.sbt_offset_and_flags;
    put64(p, 14, v.blas);
}
