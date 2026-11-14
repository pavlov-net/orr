// The path loop as one ray tracing pipeline launch per source (`path/pipeline/*.slang`). A raygen
// owns each path from its start to its end and holds nothing across a call but the step a shader
// returned: it traces the path's ray, composites the particles and light proxies in front of its
// hit by LAYERS traces, calls the segment that runs along it, inside an interior or outside every
// one, reorders on the hit and invokes the hit's resolve or the miss shader, then calls the
// vertex's emission, next-event estimation and scatter, tracing the shadow ray between the last
// two, walking the thin walls it passed, marching the medium proxies with a volume it passed and
// walking a directional light's ray through the plane stack it comes through.
// Every wider value lives in the path's slot (`launch/slot.slang`), found from the launch index,
// which each shader loads and stores in the parts it reads and writes; across a trace, reorder,
// invoke or call that memory is coherent at device scope. What crosses in registers beside the
// step is the length of the ray to the vertex a resolve reached, which the raygen hands the
// vertex's next-event estimation. A vertex is shaded by the pipeline's own
// copies of the staged loop's bodies (`launch/shade_surface.slang`, `shade_other.slang`,
// `nee.slang`, `scatter.slang`) over its packed surface: an opaque one's, a layer's or a scatter's
// in an interior. The modules of `launch/` hold the loop by family; this one exports them to the
// kernel files.
module launch;

__exported import orr_lighting.path.launch.steps;
__exported import orr_lighting.path.launch.slot;
__exported import orr_lighting.path.launch.record;
__exported import orr_lighting.path.launch.load;
__exported import orr_lighting.path.launch.walk;
__exported import orr_lighting.path.launch.media;
__exported import orr_lighting.path.launch.start;
__exported import orr_lighting.path.launch.hit;
__exported import orr_lighting.path.launch.vertex;
__exported import orr_lighting.path.launch.shadow;
__exported import orr_lighting.path.launch.stack;
__exported import orr_lighting.path.launch.front;
__exported import orr_lighting.path.launch.finish;
__exported import orr_lighting.path.launch.raygen_shadow;
__exported import orr_lighting.path.launch.raygen_walk;
__exported import orr_lighting.path.launch.raygen;
