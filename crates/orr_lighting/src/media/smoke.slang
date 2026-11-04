// Adds the view's visible smoke clusters to the froxels, and notes whether any reaches past them
// for media apply: one workgroup per cluster. Each triangle adds to the froxel of each column cell
// it covers, at the distance of the part it covers, an optical depth that keeps the mean
// transmittance of the cell: -ln(1 - coverage x opacity), with opacity, base color and emission
// read at the triangle's uv centroid. A cluster of two triangles on a shared edge counts as one
// quad, since two disjoint parts of a cell let through 1 - (c1 + c2) x opacity, not
// (1 - c1 x opacity) x (1 - c2 x opacity).
#include "orr_root.h"
import orr_lighting.media;
import orr_lighting.froxels;
import orr_scene.particles;
import orr_scene.producer;

ORR_ROOT(FroxelRoot);

// In front of this distance along the view's axis, m, a triangle is clipped away.
static const float SMOKE_NEAR = 0.01;

// A quad takes up to one corner from the near plane and one from each side of a cell.
static const uint POLYGON_CORNERS = 9;

// The most opacity a cell takes from one triangle, so its optical depth stays finite.
static const float SMOKE_MAX_OPACITY = 0.999;

void add_smoke(FroxelRoot r, uint3 f, float3 tint, float3 emission, float depth)
{
    uint first = smoke_index(r, f);
    float3 scatter = tint * depth * SMOKE_DEPTH_SCALE;
    float3 light = emission * depth * SMOKE_EMISSION_SCALE;
    InterlockedAdd(r.smoke[first], uint(depth * SMOKE_DEPTH_SCALE + 0.5));
    for (uint c = 0; c < 3; c++)
    {
        InterlockedAdd(r.smoke[first + 1 + c], uint(scatter[c] + 0.5));
        InterlockedAdd(r.smoke[first + 4 + c], uint(light[c] + 0.5));
    }
}

// Adds to the froxels the planar polygon with view-space corners `q`, `count` of them in order
// around it. Its corners and each cell's part of it are arrays local to this function: written
// as a polygon struct with clip, area and center methods, this kernel came out of Metal's
// compiler with every corner past the first wrong, so a polygon covered at most one cell. The
// same struct in a small kernel compiles right.
void splat_polygon(FroxelRoot r, float3 q[4], uint count, float opacity, float3 albedo, float3 emission)
{
    // The polygon in column coordinates, clipped to the near plane; `lo` bounds its distances
    // from the eye from below, `hi` from above.
    float2 corners[POLYGON_CORNERS];
    uint n = 0;
    float lo = positive_infinity();
    float hi = 0.0;
    for (uint i = 0; i < count; i++)
    {
        float3 a = q[i];
        float3 b = q[i + 1 == count ? 0 : i + 1];
        bool a_in = a.z < -SMOKE_NEAR;
        bool b_in = b.z < -SMOKE_NEAR;
        for (uint k = 0; k < 2; k++)
        {
            if (k == 0 ? !a_in : a_in == b_in)
                continue;
            float3 x = k == 0 ? a : lerp(a, b, (a.z + SMOKE_NEAR) / (a.z - b.z));
            corners[n++] = view_column(uint2(r.columns, r.rows), r.view, x);
            lo = min(lo, -x.z);
            hi = max(hi, length(x));
        }
    }
    float near_range = r.scene.near_range;
    if (n < 3)
        return;
    if (hi >= near_range)
        InterlockedOr(r.lights.far_smoke, 1u);
    if (!(lo < near_range))
        return;
    float2 low = corners[0];
    float2 high = low;
    for (uint i = 1; i < n; i++)
    {
        low = min(low, corners[i]);
        high = max(high, corners[i]);
    }
    int2 first = max(int2(floor(low + 0.5)), int2(0));
    int2 last = min(int2(floor(high + 0.5)), int2(r.columns, r.rows) - 1);
    float3 normal = cross(q[1] - q[0], q[2] - q[0]);
    for (int y = first.y; y <= last.y; y++)
    {
        for (int x = first.x; x <= last.x; x++)
        {
            // The cell's part: the polygon clipped to each side of the cell in turn, keeping
            // the part with side x (corner[axis] - limit) >= 0.
            float2 cell[POLYGON_CORNERS] = corners;
            uint m = n;
            for (uint plane = 0; plane < 4; plane++)
            {
                uint axis = plane / 2;
                float side = plane % 2 == 0 ? 1.0 : -1.0;
                float limit = float(axis == 0 ? x : y) - 0.5 * side;
                float2 out[POLYGON_CORNERS];
                uint kept = 0;
                for (uint i = 0; i < m; i++)
                {
                    float2 a = cell[i];
                    float2 b = cell[i + 1 == m ? 0 : i + 1];
                    float da = side * ((axis == 0 ? a.x : a.y) - limit);
                    float db = side * ((axis == 0 ? b.x : b.y) - limit);
                    if (da >= 0.0 && kept < POLYGON_CORNERS)
                        out[kept++] = a;
                    if ((da < 0.0) != (db < 0.0) && kept < POLYGON_CORNERS)
                        out[kept++] = lerp(a, b, da / (da - db));
                }
                cell = out;
                m = kept;
            }
            float twice_area = 0.0;
            float2 sum = float2(0.0);
            for (uint i = 0; i < m; i++)
            {
                float2 a = cell[i];
                float2 b = cell[i + 1 == m ? 0 : i + 1];
                twice_area += a.x * b.y - a.y * b.x;
                sum += a;
            }
            float coverage = 0.5 * abs(twice_area);
            if (!(coverage > 0.0))
                continue;
            float2 center = sum / float(max(m, 1u));
            float3 ray = normalize(unjittered_ray(r.view, column_pixel(r, center)));
            float along = dot(normal, ray);
            float t = abs(along) > 1e-6 * length(normal) ? dot(normal, q[0]) / along : 0.5 * (lo + hi);
            t = clamp(t, lo, hi);
            if (!(t < near_range))
                continue;
            uint slice = min(uint(froxel_slice(t, r.slices, near_range)), r.slices - 1);
            float depth = -log(1.0 - min(saturate(coverage) * opacity, SMOKE_MAX_OPACITY));
            add_smoke(r, uint3(x, y, slice), albedo, emission, depth);
        }
    }
}

// The corners of the cluster's two triangles in order around their outline, when they share an
// edge.
bool quad_outline(ProducerClusterGpu k, out uint4 outline)
{
    outline = uint4(0);
    uint3 a = cluster_corners(k, 0);
    uint3 b = cluster_corners(k, 1);
    for (uint i = 0; i < 3; i++)
    {
        uint u = a[i];
        uint v = a[(i + 1) % 3];
        if (any(b == u) && any(b == v))
        {
            outline = uint4(u, b.x + b.y + b.z - u - v, v, a[(i + 2) % 3]);
            return true;
        }
    }
    return false;
}

// Splats the polygon over the cluster-local corners `corners`, `count` of them, with the look at
// their uv centroid.
void splat_corners(ParticleDraw d, uint4 corners, uint count)
{
    float3 q[4];
    float2 uv = float2(0.0);
    for (uint i = 0; i < count; i++)
    {
        uint vertex = d.cluster.first_vertex + corners[i];
        q[i] = view_from_relative(root.view, place(d.placement, d.batch.positions[vertex]));
        uv += producer_uv(d.batch.attrs[vertex]);
    }
    float3 albedo, emission;
    float opacity;
    smoke_look(particle_input(root.scene, d.vc.material, uv / float(count)), albedo, emission, opacity);
    if (opacity > 0.0)
        splat_polygon(root, q, count, opacity, albedo, emission);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 group: SV_GroupID)
{
    ParticleDraw d;
    if (!smoke_draw(root.scene, root.view, root.raster, group, d))
        return;
    uint4 outline;
    if (uint(d.cluster.tri_count) == 2 && quad_outline(d.cluster, outline))
    {
        splat_corners(d, outline, 4);
        return;
    }
    for (uint t = 0; t < uint(d.cluster.tri_count); t++)
        splat_corners(d, uint4(cluster_corners(d.cluster, t), 0), 3);
}
