// The surface link point as orr_scene shapes it, cut down to what the tests read.
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
    static bool surface(SurfaceInput i, out GSurface g, out float depth_offset)
    {
        g = GSurface();
        depth_offset = 0.0;
        return false;
    }
    static void adjust(SurfaceInput i, inout GSurface g, inout float depth_offset)
    {
    }
    static bool coverage(SurfaceInput i, out float coverage)
    {
        coverage = 0.0;
        return false;
    }
    static void feedback(SurfaceInput i)
    {
    }
    static void coverage_feedback(SurfaceInput i)
    {
    }
}

public struct NoSurface : ISurface
{
}

public interface ISurfaces
{
    static GSurface surface(SurfaceInput i, out float depth_offset, bool report);
    static float coverage(SurfaceInput i, bool presence, bool report);
}

public struct Surfaces<L : ISurface> : ISurfaces
{
    public static GSurface surface(SurfaceInput i, out float depth_offset, bool report)
    {
        GSurface g;
        if (!L.surface(i, g, depth_offset))
        {
            depth_offset = 0.0;
            g = { float4(0.5, 0.5, 0.5, 1.0) };
            L.adjust(i, g, depth_offset);
        }
        return g;
    }

    public static float coverage(SurfaceInput i, bool presence, bool report)
    {
        float coverage;
        if (presence || !L.coverage(i, coverage))
            coverage = 0.25;
        return coverage;
    }
}

public typealias StandardSurface = Surfaces<NoSurface>;
