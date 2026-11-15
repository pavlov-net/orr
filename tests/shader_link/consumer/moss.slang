// A surface that adjusts the standard one where its hit is past the first bounce, declares a
// fuzz, and leaves coverage to the standard one.
import orr_scene.surface;

public struct Moss : ISurface
{
    override static void adjust(SurfaceInput i, inout PlainSurface s, inout float depth_offset)
    {
        if (i.context.depth == 0)
            return;
        s.albedo.w = i.context.cone_width;
        depth_offset += 0.25;
    }

    override static void fuzz(SurfaceInput i, PlainSurface s, inout FuzzLobe lobe)
    {
        lobe.weight = 4.0;
    }
}
