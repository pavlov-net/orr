// The surface link point as contracts.md 6 shapes it, cut down to what the tests read.
module surface;

public static const uint HIT_PRIMARY = 0;
public static const uint HIT_SECONDARY = 1;

public struct HitContext
{
    public uint kind;
    public uint depth;
    public float cone_width;
    public float cone_spread;
}

public struct SurfaceInput
{
    public HitContext context;
    public uint surface;
    public uint material;
    public uint64_t data;
}

public struct GSurface
{
    public float4 albedo;
}

public interface ISurface
{
    static GSurface surface(SurfaceInput i, out float depth_offset);
    static float coverage(SurfaceInput i)
    {
        return 1.0;
    }
}

public struct StandardSurface : ISurface
{
    public static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        depth_offset = 0.0;
        return { float4(0.5, 0.5, 0.5, 1.0) };
    }

    public override static float coverage(SurfaceInput i)
    {
        return 0.25;
    }
}
