// Subdivision cage pages (orr_pack::subd): patch kinds, the patch basis `PatchParam::basis`
// is the reference for, and per-point skin influences. Patch coordinates (s, t) are in
// [0, 1] over one patch; the basis is bicubic B-spline (missing points beyond boundary edges
// extrapolated), Gregory basis or bilinear.
module subd;

import orr_shaders.packing;

// `PatchParam.kind`.
public static const uint SUBD_REGULAR = 0;
public static const uint SUBD_GREGORY = 1;
public static const uint SUBD_LINEAR = 2;
public static const uint SUBD_NONE = 0xff;

// `Influence`: four joints (u8) and their weights (unorm8).
public struct SubdInfluence
{
    public uint4 joints;
    public float4 weights;
}

public SubdInfluence decode_influence(uint2 packed)
{
    SubdInfluence i;
    i.joints = uint4(packed.x, packed.x >> 8, packed.x >> 16, packed.x >> 24) & 0xff;
    i.weights = unpack_unorm4x8(packed.y);
    return i;
}

void bspline_curve(float t, out float4 b, out float4 d)
{
    float t2 = t * t;
    float t3 = t2 * t;
    b = float4(1.0 - 3.0 * (t - t2) - t3, 4.0 - 6.0 * t2 + 3.0 * t3, 1.0 + 3.0 * (t + t2 - t3), t3) / 6.0;
    d = float4(-0.5 * t2 + t - 0.5, 1.5 * t2 - 2.0 * t, -1.5 * t2 + t + 0.5, 0.5 * t2);
}

void bezier_curve(float t, out float4 b, out float4 d)
{
    float c = 1.0 - t;
    b = float4(c * c * c, 3.0 * c * c * t, 3.0 * t * t * c, t * t * t);
    d = float4(-3.0 * c * c, 9.0 * t * t - 12.0 * t + 3.0, -9.0 * t * t + 6.0 * t, 3.0 * t * t);
}

// Moves the weights of points beyond boundary edges onto the points that extrapolate them:
// a missing point is 2 x next - next but one.
void fold_boundary(uint boundary, inout float w[20])
{
    if ((boundary & 1) != 0)
    {
        for (uint i = 0; i < 4; i++)
        {
            w[i + 8] -= w[i];
            w[i + 4] += 2.0 * w[i];
            w[i] = 0.0;
        }
    }
    if ((boundary & 2) != 0)
    {
        for (uint i = 0; i < 16; i += 4)
        {
            w[i + 1] -= w[i + 3];
            w[i + 2] += 2.0 * w[i + 3];
            w[i + 3] = 0.0;
        }
    }
    if ((boundary & 4) != 0)
    {
        for (uint i = 0; i < 4; i++)
        {
            w[i + 4] -= w[i + 12];
            w[i + 8] += 2.0 * w[i + 12];
            w[i + 12] = 0.0;
        }
    }
    if ((boundary & 8) != 0)
    {
        for (uint i = 0; i < 16; i += 4)
        {
            w[i + 2] -= w[i];
            w[i + 1] += 2.0 * w[i];
            w[i] = 0.0;
        }
    }
}

static const uint GREGORY_BOUNDARY[12] = { 0, 1, 7, 5, 2, 6, 16, 12, 15, 17, 11, 10 };
static const uint GREGORY_BOUNDARY_COL[12] = { 0, 1, 2, 3, 0, 3, 0, 3, 0, 1, 2, 3 };
static const uint GREGORY_BOUNDARY_ROW[12] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3 };
static const uint GREGORY_INTERIOR[8] = { 3, 4, 8, 9, 13, 14, 18, 19 };
static const uint GREGORY_INTERIOR_COL[8] = { 1, 1, 2, 2, 2, 2, 1, 1 };
static const uint GREGORY_INTERIOR_ROW[8] = { 1, 1, 1, 1, 2, 2, 2, 2 };

float inverse_or_one(float d)
{
    return d <= 0.0 ? 1.0 : 1.0 / d;
}

// Weights of a patch's points and their derivatives at patch coordinates `st`; returns the
// point count.
public uint subd_basis(uint kind, uint boundary, float2 st, out float w[20], out float du[20], out float dv[20])
{
    for (uint i = 0; i < 20; i++)
    {
        w[i] = 0.0;
        du[i] = 0.0;
        dv[i] = 0.0;
    }
    float s = st.x;
    float t = st.y;
    if (kind == SUBD_REGULAR)
    {
        float4 bs, ds, bt, dt;
        bspline_curve(s, bs, ds);
        bspline_curve(t, bt, dt);
        for (uint i = 0; i < 4; i++)
        {
            for (uint j = 0; j < 4; j++)
            {
                w[4 * i + j] = bs[j] * bt[i];
                du[4 * i + j] = ds[j] * bt[i];
                dv[4 * i + j] = bs[j] * dt[i];
            }
        }
        if (boundary != 0)
        {
            fold_boundary(boundary, w);
            fold_boundary(boundary, du);
            fold_boundary(boundary, dv);
        }
        return 16;
    }
    if (kind == SUBD_GREGORY)
    {
        float4 bs, ds, bt, dt;
        bezier_curve(s, bs, ds);
        bezier_curve(t, bt, dt);
        float sc = 1.0 - s;
        float tc = 1.0 - t;
        float d0 = inverse_or_one(s + t);
        float d1 = inverse_or_one(sc + t);
        float d2 = inverse_or_one(sc + tc);
        float d3 = inverse_or_one(s + tc);
        float g[8] = { s * d0, 1.0 - s * d0, t * d1, 1.0 - t * d1, sc * d2, 1.0 - sc * d2, tc * d3, 1.0 - tc * d3 };
        for (uint i = 0; i < 12; i++)
        {
            uint c = GREGORY_BOUNDARY_COL[i];
            uint r = GREGORY_BOUNDARY_ROW[i];
            uint k = GREGORY_BOUNDARY[i];
            w[k] = bs[c] * bt[r];
            du[k] = ds[c] * bt[r];
            dv[k] = bs[c] * dt[r];
        }
        for (uint i = 0; i < 8; i++)
        {
            uint c = GREGORY_INTERIOR_COL[i];
            uint r = GREGORY_INTERIOR_ROW[i];
            uint k = GREGORY_INTERIOR[i];
            w[k] = bs[c] * bt[r] * g[i];
            du[k] = ds[c] * bt[r] * g[i];
            dv[k] = bs[c] * dt[r] * g[i];
        }
        return 20;
    }
    if (kind == SUBD_LINEAR)
    {
        float sc = 1.0 - s;
        float tc = 1.0 - t;
        w[0] = sc * tc;
        w[1] = s * tc;
        w[2] = s * t;
        w[3] = sc * t;
        du[0] = -tc;
        du[1] = tc;
        du[2] = t;
        du[3] = -t;
        dv[0] = -sc;
        dv[1] = -s;
        dv[2] = s;
        dv[3] = sc;
        return 4;
    }
    return 0;
}
