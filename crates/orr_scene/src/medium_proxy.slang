// Medium proxies: boxes of participating medium that shadow and camera rays march through
// (TLAS mask MASK_MEDIUM, candidate class CLASS_MEDIUM). A proxy's PTLAS instance places the
// cube [-1, 1]^3 of its object space; the cube maps onto a box of the proxy's volume space,
// where its density is a grid of a resident volume, or 1 without one.
module medium_proxy;

import orr_shaders.math;
import orr_pack.volume;

// `MediumProxyGpu.flags`.
public static const uint PROXY_LIVE = 1;

public struct MediumProxyGpu
{
    public int3 cell;              // world position of the box center
    public uint flags;             // PROXY_*
    public float3 offset;
    public uint grid;              // in `volume`
    public float3 axis_x;          // object-to-world columns, m: the box's half extents turned and scaled
    public float anisotropy;       // Henyey-Greenstein g
    public float3 axis_y;
    public uint _pad0;
    public float3 axis_z;
    public uint _pad1;
    public float3 box_center;      // the box in the volume's space, m
    public uint _pad2;
    public float3 box_half;
    public uint _pad3;
    public float3 extinction;      // per unit density, 1/m, linear Rec.2020
    public uint _pad4;
    public float3 albedo;          // scattering over extinction
    public uint _pad5;
    public VolumeGpu* volume;      // null: density 1 throughout the box
}

// The medium at a point of a proxy. Coefficients are per meter in linear Rec.2020.
public struct ProxyMedium
{
    public float3 sigma_t;
    public float3 sigma_s;
    public float anisotropy;
}

// The volume-space point of object-space point `x`.
public float3 proxy_volume_point(MediumProxyGpu p, float3 x)
{
    return p.box_center + p.box_half * x;
}

// Density at object-space point `x` of the cube.
public float proxy_density(MediumProxyGpu p, float3 x)
{
    if (p.volume == nullptr)
        return 1.0;
    return max(volume_sample(p.volume, p.grid, proxy_volume_point(p, x)), 0.0);
}

public ProxyMedium proxy_medium(MediumProxyGpu p, float3 x)
{
    ProxyMedium m;
    m.sigma_t = p.extinction * proxy_density(p, x);
    m.sigma_s = m.sigma_t * p.albedo;
    m.anisotropy = p.anisotropy;
    return m;
}

// The t along the object-space ray direction `d` that moves at most one voxel on every
// axis of the proxy's volume; +inf without a volume.
public float proxy_voxel_step(MediumProxyGpu p, float3 d)
{
    if (p.volume == nullptr)
        return asfloat(0x7f800000u);
    float3 du = abs(p.box_half * d * p.volume.inv_voxel_size);
    return 1.0 / max_component(du);
}

// Walks the majorant cells along the object-space ray `o + d * t`: at least the largest
// channel of `sigma_t` along the stretch from `t` that one cell covers, and the t at which
// it ends (+inf where it does not, as for a proxy without a volume).
public float proxy_majorant_span(MediumProxyGpu p, float3 o, float3 d, float t, out float t_exit)
{
    float e = max_component(p.extinction);
    if (p.volume == nullptr)
    {
        t_exit = asfloat(0x7f800000u);
        return e;
    }
    VolumeGpu* v = p.volume;
    float3 u0 = (proxy_volume_point(p, o) - v.origin) * v.inv_voxel_size;
    float3 du = p.box_half * d * v.inv_voxel_size;
    return e * max(volume_majorant_span(v, p.grid, u0, du, t, t_exit), 0.0);
}

// The span [t0, t1] of the object-space ray `o + d * t` inside the cube [-1, 1]^3, from 0 to
// at most `t_max`; false where the ray misses it.
public bool proxy_span(float3 o, float3 d, float t_max, out float t0, out float t1)
{
    float3 inv = select(d != 0.0, 1.0 / d, float3(1e30));
    float3 a = (-1.0 - o) * inv;
    float3 b = (1.0 - o) * inv;
    float3 lo = min(a, b);
    float3 hi = max(a, b);
    t0 = max(max_component(lo), 0.0);
    t1 = min(min(hi.x, min(hi.y, hi.z)), t_max);
    return t1 > t0;
}

// What a `proxy_march` walks and steps through.
public interface IProxyStep
{
    // At least the largest channel of `sigma_t` along the stretch from `t` that one majorant cell
    // covers, and the t at which it ends.
    float majorant(float t, out float t_exit);
    // A step of `h` at `t` along the ray.
    [mutating]
    void step(float t, float h);
}

// Majorant cells a march walks at most; the cap only guards against a walk that stops advancing.
static const uint PROXY_MARCH_CELLS = 256;

// Marches over [t0, t1] in steps of `h` at `t0 + (k + jitter) * h`, skipping the steps in empty
// majorant cells.
public void proxy_march<V : IProxyStep>(float t0, float t1, float h, float jitter, inout V visit)
{
    // `next` is the first k not yet taken.
    uint next = 0;
    uint steps = uint(ceil((t1 - t0) / h)) + 1;
    float t = t0;
    for (uint cell = 0; cell < PROXY_MARCH_CELLS && t < t1; cell++)
    {
        float t_exit;
        float m = visit.majorant(t, t_exit);
        float end = min(max(t_exit, t + 1e-5 * (t1 - t0)), t1);
        uint k = max(next, uint(max(ceil((t - t0) / h - jitter), 0.0)));
        for (; m > 0.0 && k < steps; k++)
        {
            float x = t0 + (float(k) + jitter) * h;
            if (x >= end)
                break;
            visit.step(x, h);
        }
        next = k;
        t = end;
    }
}
