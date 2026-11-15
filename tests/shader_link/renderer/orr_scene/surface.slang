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

// The id the linked types' switch reads.
public uint surface_id(SurfaceInput i)
{
    return i.surface;
}

public struct GSurface
{
    public float4 albedo;
}

public struct PlainSurface
{
    public float4 albedo;
}

public struct TransmissionLobe { public float weight; }
public struct SubsurfaceLobe { public float weight; }
public struct CoatLobe { public float weight; }
public struct FuzzLobe { public float weight; }
public struct ThinFilmLobe { public float weight; }
public struct EmissionLobe { public float weight; }
public struct AnisotropyLobe { public float weight; }

public interface ISurface
{
    static bool surface(SurfaceInput i, out PlainSurface s, out float depth_offset)
    {
        s = PlainSurface();
        depth_offset = 0.0;
        return false;
    }
    static void adjust(SurfaceInput i, inout PlainSurface s, inout float depth_offset)
    {
    }
    static void transmission(SurfaceInput i, PlainSurface s, inout TransmissionLobe lobe)
    {
    }
    static void subsurface(SurfaceInput i, PlainSurface s, inout SubsurfaceLobe lobe)
    {
    }
    static void coat(SurfaceInput i, PlainSurface s, inout CoatLobe lobe)
    {
    }
    static void fuzz(SurfaceInput i, PlainSurface s, inout FuzzLobe lobe)
    {
    }
    static void thin_film(SurfaceInput i, PlainSurface s, inout ThinFilmLobe lobe)
    {
    }
    static void emission(SurfaceInput i, PlainSurface s, inout EmissionLobe lobe)
    {
    }
    static void anisotropy(SurfaceInput i, PlainSurface s, inout AnisotropyLobe lobe)
    {
    }
    static void other_side()
    {
    }
    static void always_shades()
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
    static uint coverage_reads(SurfaceInput i)
    {
        return 0xffffffffu;
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
        PlainSurface s;
        if (!L.surface(i, s, depth_offset))
        {
            depth_offset = 0.0;
            s = { float4(0.5, 0.5, 0.5, 1.0) };
            L.adjust(i, s, depth_offset);
        }
        FuzzLobe fuzz = { 0.0 };
        L.fuzz(i, s, fuzz);
        GSurface g = { s.albedo + float4(fuzz.weight, 0.0, 0.0, 0.0) };
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
