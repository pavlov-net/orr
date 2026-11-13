// A surface that adjusts the standard one where its hit is past the first bounce, and leaves
// coverage to it.
import orr_scene.surface;

struct Moss : ISurface
{
    override static void adjust(SurfaceInput i, inout GSurface g, inout float depth_offset)
    {
        if (i.context.depth == 0)
            return;
        g.albedo.w = i.context.cone_width;
        depth_offset += 0.25;
    }
}
