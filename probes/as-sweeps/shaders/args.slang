// GPU-side argument writer: packs cluster build, template, instantiate, cluster-BLAS and
// TLAS instance records from CPU-side descriptors, and writes the op count. All layouts mirror
// main.rs; packed words follow VK_NV_cluster_acceleration_structure bit patterns.
struct ArgsRoot
{
    uint64_t src;
    uint64_t dst;
    uint64_t count_dst;
    uint64_t aux;
    uint n;
    uint op;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] ConstantBuffer<ArgsRoot> root;

// Mirrors ClusterDesc in main.rs (112 bytes).
struct ClusterDesc
{
    uint64_t index_addr;
    uint64_t vertex_addr;
    uint64_t gflags_addr;
    uint64_t omm_array;
    uint64_t omm_index;
    uint64_t bbox_addr;
    uint id;
    uint tris;
    uint verts;
    uint index_type;
    uint truncate;
    uint base_geom;
    uint geom_flags;
    uint cluster_flags;
    uint index_stride;
    uint vertex_stride;
    uint gflags_stride;
    uint omm_index_type;
    uint omm_index_stride;
    uint pad0;
    uint pad1;
    uint pad2;
};

// Mirrors InstantiateDesc in main.rs (32 bytes).
struct InstantiateDesc
{
    uint64_t template_addr_ptr;
    uint64_t vertex_addr;
    uint64_t vertex_stride;
    uint cluster_id_offset;
    uint geometry_index_offset;
};

// Mirrors BlasDesc in main.rs (16 bytes).
struct BlasDesc
{
    uint count;
    uint first;
    uint64_t pad;
};

// Mirrors InstanceDesc in main.rs (64 bytes).
struct InstanceDesc
{
    float m[12];
    uint custom;
    uint mask;
    uint flags;
    uint blas_index;
};

static const uint OP_CLUSTER = 0;
static const uint OP_TEMPLATE = 1;
static const uint OP_INSTANTIATE = 2;
static const uint OP_GATHER = 3;
static const uint OP_BLAS = 4;
static const uint OP_INSTANCE = 5;

void put(uint* p, uint w, uint v)
{
    p[w] = v;
}

void put64(uint* p, uint w, uint64_t v)
{
    p[w] = uint(v);
    p[w + 1] = uint(v >> 32);
}

uint64_t get64(uint64_t addr)
{
    uint* p = (uint*)addr;
    return uint64_t(p[0]) | (uint64_t(p[1]) << 32);
}

// VkClusterAccelerationStructureBuildTriangleClusterInfoNV (64 bytes); the template info adds
// instantiationBoundingBoxLimit (72 bytes).
void write_cluster(uint* o, ClusterDesc d, bool template_info)
{
    put(o, 0, d.id);
    put(o, 1, d.cluster_flags);
    put(o, 2, d.tris | (d.verts << 9) | (d.truncate << 18) | (d.index_type << 24) | (d.omm_index_type << 28));
    put(o, 3, (d.base_geom & 0xffffff) | (d.geom_flags << 29));
    put(o, 4, d.index_stride | (d.vertex_stride << 16));
    put(o, 5, d.gflags_stride | (d.omm_index_stride << 16));
    put64(o, 6, d.index_addr);
    put64(o, 8, d.vertex_addr);
    put64(o, 10, d.gflags_addr);
    put64(o, 12, d.omm_array);
    put64(o, 14, d.omm_index);
    if (template_info)
        put64(o, 16, d.bbox_addr);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i == 0 && root.count_dst != 0)
        *(uint*)root.count_dst = root.n;
    if (i >= root.n)
        return;
    switch (root.op)
    {
    case OP_CLUSTER:
        write_cluster((uint*)(root.dst + i * 64), ((ClusterDesc*)root.src)[i], false);
        break;
    case OP_TEMPLATE:
        write_cluster((uint*)(root.dst + i * 72), ((ClusterDesc*)root.src)[i], true);
        break;
    case OP_INSTANTIATE:
    {
        // VkClusterAccelerationStructureInstantiateClusterInfoNV (32 bytes).
        InstantiateDesc d = ((InstantiateDesc*)root.src)[i];
        uint* o = (uint*)(root.dst + i * 32);
        put(o, 0, d.cluster_id_offset);
        put(o, 1, d.geometry_index_offset & 0xffffff);
        put64(o, 2, get64(d.template_addr_ptr));
        put64(o, 4, d.vertex_addr);
        put64(o, 6, d.vertex_stride);
        break;
    }
    case OP_GATHER:
    {
        // dst[i] = aux[src[i]]: CLAS addresses into per-BLAS reference lists.
        uint index = ((uint*)root.src)[i];
        put64((uint*)(root.dst + i * 8), 0, get64(root.aux + index * 8));
        break;
    }
    case OP_BLAS:
    {
        // VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV (16 bytes); references
        // live in the gathered list at aux.
        BlasDesc d = ((BlasDesc*)root.src)[i];
        uint* o = (uint*)(root.dst + i * 16);
        put(o, 0, d.count);
        put(o, 1, 8);
        put64(o, 2, root.aux + uint64_t(d.first) * 8);
        break;
    }
    case OP_INSTANCE:
    {
        // VkAccelerationStructureInstanceKHR (64 bytes); BLAS address from the build output.
        InstanceDesc d = ((InstanceDesc*)root.src)[i];
        uint* o = (uint*)(root.dst + i * 64);
        for (uint k = 0; k < 12; ++k)
            put(o, k, asuint(d.m[k]));
        put(o, 12, (d.custom & 0xffffff) | (d.mask << 24));
        put(o, 13, d.flags << 24);
        put64(o, 14, get64(root.aux + uint64_t(d.blas_index) * 8));
        break;
    }
    default:
        break;
    }
}
