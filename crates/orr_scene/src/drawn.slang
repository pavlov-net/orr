// The drawn link point. A drawn producer entry's opaque raster is not its batch: the consumer's
// task and mesh stages derive triangles and draw them into a view's visibility buffer from the
// view's opaque hook (`orr_scene.drawn_raster`), with nothing stored per triangle. A pixel
// names its triangle by the word the stages gave its visible record and the triangle's index
// in the record. The material resolve hands both to the entry's linked `IDrawn`, which derives
// the triangle again, and shades it as it shades a producer cluster's triangle: through the
// material's linked surface, with motion from where last frame drew the triangle. Rays never
// meet drawn triangles; they meet the clusters the entry's kernel emits into its batch.
module drawn;

import orr_scene.tables;
import orr_scene.view;

// The triangle a linked `IDrawn` derives.
public struct DrawnInput
{
    public uint drawn;          // the entry's linked id
    public uint entry;          // the producer entry
    public uint user_tag;       // the entry's
    public uint word;           // the visible record's, from `drawn_visible`
    public uint primitive;      // the triangle in the record, below 128
    public SceneRoot* scene;
    public ViewConstants* view;
    public uint64_t data;       // the record's material's consumer data, else the scene's
}

public struct DrawnTriangle
{
    public float3 position[3];  // relative to the entry's transform
    public float3 previous[3];  // where last frame's draw placed the corners
    public float3 normal[3];    // unit
    public float2 uv[3];
    public uint cluster_user;   // what a producer cluster's word is to the surface
    public uint3 vertex_user;   // per corner
    // m, entry space: how far the triangle lies from what rays trace of the entry at most; 0
    // where the entry's clusters hold the same triangle. Rays leaving it start that far off
    // it under `OriginPolicy::Default`.
    public float error;
}

public interface IDrawn
{
    // Triangle `i.primitive` of the visible record with word `i.word`: the corners the mesh
    // stage gave that primitive, in its order, and where it gave them last frame.
    static DrawnTriangle triangle(DrawnInput i);
}

// Serves id 0 and every id nothing links.
public struct NoDrawn : IDrawn
{
    public static DrawnTriangle triangle(DrawnInput i)
    {
        return (DrawnTriangle)0;
    }
}
