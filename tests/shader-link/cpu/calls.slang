// C entry points that call the linked types through the same declarations the renderer's
// kernels use.
import surface;
import displacement;
import environment;
import media;
import orr.coherence;

extern struct Surface : ISurface = StandardSurface;
extern struct Displace : IDisplacement = TextureDisplacement;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;

// out: albedo, depth offset, coverage.
export __extern_cpp void link_surface(uint id, uint kind, uint depth, float cone_width, float cone_spread, float* out)
{
    SurfaceInput i = {};
    i.context.kind = kind;
    i.context.depth = depth;
    i.context.cone_width = cone_width;
    i.context.cone_spread = cone_spread;
    i.surface = id;
    float depth_offset;
    float4 a = Surface.surface(i, depth_offset).albedo;
    out[0] = a.x;
    out[1] = a.y;
    out[2] = a.z;
    out[3] = a.w;
    out[4] = depth_offset;
    out[5] = Surface.coverage(i);
}

export __extern_cpp void link_displace(uint id, float* position, float* normal, float* out)
{
    DisplacementInput i = {};
    i.position = float3(position[0], position[1], position[2]);
    i.normal = float3(normal[0], normal[1], normal[2]);
    i.displacement = id;
    float3 d = Displace.displace(i);
    out[0] = d.x;
    out[1] = d.y;
    out[2] = d.z;
}

export __extern_cpp void link_environment(float* dir, float* out)
{
    float3 r = Environment.radiance(0, float3(dir[0], dir[1], dir[2]), float3(0.0));
    out[0] = r.x;
    out[1] = r.y;
    out[2] = r.z;
}

// out: sigma_s, sigma_t, phase parameter, transmittance over `t`.
export __extern_cpp void link_medium(uint64_t data, float t, float* out)
{
    MediumPoint m = Medium.sample(data, float3(0.0));
    float3 tr = Medium.transmittance(data, float3(0.0), float3(0.0, 1.0, 0.0), t);
    out[0] = m.sigma_s.x;
    out[1] = m.sigma_s.y;
    out[2] = m.sigma_s.z;
    out[3] = m.sigma_t.x;
    out[4] = m.sigma_t.y;
    out[5] = m.sigma_t.z;
    out[6] = m.phase_parameter;
    out[7] = tr.x;
    out[8] = tr.y;
    out[9] = tr.z;
}

export __extern_cpp uint link_key_bits()
{
    return KEY_BITS;
}
