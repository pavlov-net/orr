// An unlit surface that shows its hit's path depth: red at depths 0 and 1, green from 2 on.
module context;

import surface;

public static const float3 NEAR = float3(40.0, 4.0, 4.0);
public static const float3 FAR = float3(4.0, 40.0, 4.0);

public struct Context : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        GSurface s = StandardSurface.surface(i, depth_offset);
        s.bsdf.emission = i.context.depth < 2 ? NEAR : FAR;
        s.flags |= GBUFFER_UNLIT;
        return s;
    }
}
