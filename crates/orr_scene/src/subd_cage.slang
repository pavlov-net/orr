// Subdivision cages on the GPU: a mounted cage's patch tables and base patches, a copy's
// evaluated points, and the limit surface the tessellator dices. Patch coordinates (s, t)
// are in [0, 1] over one patch of the cage's patch table; points are page points, each a
// stencil of cage points.
module subd_cage;

__exported import orr_pack.subd;

// `SubdPartGpu.bits >> 16` of uv set 0; set 1's is the next bit. Those below are color channels.
public static const uint SUBD_PART_UV = 16;

// One patch: its kind and boundary edges for positions, and its parts.
public struct SubdPatchGpu
{
    public uint bits;            // kind | boundary << 8
    public uint cvs;             // its first point in the cage's `cvs`
    public uint part_first;      // its first part in the cage's `parts`
    public uint part_count;
}

// A uv set or some channels of the vertex color stream over a patch: a patch of its own, which
// lies in the patch's coordinates.
public struct SubdPartGpu
{
    public uint bits;            // kind | boundary << 8 | channels << 16
    public uint cvs;             // its first point in the cage's `part_cvs`
    public float2 offset;        // its coordinates are st * scale - offset
    public float scale;
    public uint _pad;
}

// A vertex of a cage's base patches: the patch and coordinates its uvs and normal come from,
// and those its position comes from, which every vertex at the same cage point shares.
public struct SubdVertexGpu
{
    public uint patch;
    public uint source;
    public float2 st;
    public float2 source_st;
}

// A base triangle: its patch and its corners' patch coordinates in halves, s | t << 2 per
// corner from bit 4 x corner.
public struct SubdTriangleGpu
{
    public uint patch;
    public uint corners;
}

// A mounted cage: its patch table over all pages, its base patches as patch clusters, and
// what skinning a copy reads.
public struct SubdCageGpu
{
    public SubdPatchGpu* patches;
    public uint* cvs;               // page point per patch point
    public float3* points;          // rest page points
    public SubdVertexGpu* vertices; // per base vertex
    public SubdTriangleGpu* triangles;
    public uint4* clusters;         // first vertex, vertex count, first triangle, triangle count | material << 8
    public uint8_t* indices;        // u8x3 per base triangle, cluster-local
    public uint* stencils;          // per page point plus one: its terms
    public uint* stencil_points;    // cage point per term
    public float* stencil_weights;
    public float3* cage_points;     // rest
    public uint2* skin;             // per cage point: joints u8x4, weights unorm8x4
    public SubdPartGpu* parts;
    public uint* part_cvs;          // value point per part point
    public float4* value_points;    // a color, or a uv in its first two
    public uint point_count;        // page points
    public uint cage_point_count;
    public uint vertex_count;
    public uint cluster_count;
}

// One copy of a cage: the page points it evaluates this frame and last frame, and for a
// skinned copy its posed cage points and palette.
public struct SubdCopyGpu
{
    public SubdCageGpu* cage;
    public float3* points;          // the cage's rest points for an unskinned copy
    public float3* previous;
    public float3* posed;           // skinned: this frame's then last frame's cage points
    public uint palette_group;
    public uint palette_first;
    public uint joint_count;        // 0: unskinned
    public uint _pad;
}

public float2 triangle_corner(SubdTriangleGpu t, uint corner)
{
    uint c = t.corners >> (4 * corner);
    return float2(float(c & 3), float((c >> 2) & 3)) * 0.5;
}

// The limit surface of patch `patch` at `st` from page points `points`, and its derivatives
// in patch coordinates.
public float3 subd_limit(SubdCageGpu* c, float3* points, uint patch, float2 st, out float3 du, out float3 dv)
{
    SubdPatchGpu p = c.patches[patch];
    float w[20], wu[20], wv[20];
    uint n = subd_basis(p.bits & 0xff, (p.bits >> 8) & 0xff, st, w, wu, wv);
    float3 x = 0.0;
    du = 0.0;
    dv = 0.0;
    for (uint i = 0; i < n; i++)
    {
        float3 q = points[c.cvs[p.cvs + i]];
        x += w[i] * q;
        du += wu[i] * q;
        dv += wv[i] * q;
    }
    return x;
}

// Part `part` of a patch at the patch's `st`.
float4 subd_part_value(SubdCageGpu* c, SubdPartGpu part, float2 st)
{
    float w[20], wu[20], wv[20];
    uint n = subd_basis(part.bits & 0xff, (part.bits >> 8) & 0xff, st * part.scale - part.offset, w, wu, wv);
    float4 value = 0.0;
    for (uint i = 0; i < n; i++)
        value += w[i] * c.value_points[c.part_cvs[part.cvs + i]];
    return value;
}

// Uv set `set` of patch `patch` at `st`; 0 where none of its parts gives the set.
public float2 subd_limit_uv(SubdCageGpu* c, uint patch, float2 st, uint set)
{
    SubdPatchGpu p = c.patches[patch];
    for (uint k = p.part_count; k > 0; k--)
    {
        SubdPartGpu part = c.parts[p.part_first + k - 1];
        if (((part.bits >> 16) & (SUBD_PART_UV << set)) != 0)
            return subd_part_value(c, part, st).xy;
    }
    return 0.0;
}

// The vertex color stream of patch `patch` at `st`; 1 in the channels none of its parts gives.
public float4 subd_limit_color(SubdCageGpu* c, uint patch, float2 st)
{
    SubdPatchGpu p = c.patches[patch];
    float4 color = 1.0;
    for (uint k = 0; k < p.part_count; k++)
    {
        SubdPartGpu part = c.parts[p.part_first + k];
        uint channels = (part.bits >> 16) & (SUBD_PART_UV - 1);
        if (channels != 0)
            color = select((uint4(channels) & uint4(1, 2, 4, 8)) != 0, subd_part_value(c, part, st), color);
    }
    return color;
}

// The unit normal of the limit surface from its derivatives, oriented by the face winding;
// `fallback` where they degenerate.
public float3 subd_normal(float3 du, float3 dv, float3 fallback)
{
    float3 n = cross(du, dv);
    float l = length(n);
    return l > 0.0 ? n / l : fallback;
}
