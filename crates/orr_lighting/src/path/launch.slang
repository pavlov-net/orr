// The path loop as one ray tracing pipeline launch per source (`path/pipeline/*.slang`). A raygen
// owns each path from its start to its end and holds nothing across a call but the step a shader
// returned: it traces the path's ray, composites the particles and light proxies in front of its
// hit by LAYERS traces, calls the segment that runs along it inside an interior, reorders on the
// hit and invokes the hit's resolve or the miss shader, then runs the vertex: its arrival, which
// draws its light candidates, its shade, which leaves its shadow job and its next ray, and what
// the shade left entries for. It traces the shadow ray, walking the thin walls it passed,
// marching the medium proxies with a volume it passed and walking a directional light's ray
// through the plane stack it comes through.
// A path's state is its slot (`launch/slot.slang`), found from the launch index: each shader
// loads the words it reads and stores the ones it changed, calls nothing with them live, and
// across a trace, reorder, invoke or call that memory is coherent at device scope. What crosses
// in registers is a step and at most three words beside it. The modules of `launch/` hold the
// loop by family; this one exports them to the kernel files.
module launch;

__exported import orr_lighting.path.launch.steps;
__exported import orr_lighting.path.launch.slot;
__exported import orr_lighting.path.launch.record;
__exported import orr_lighting.path.launch.boundary;
__exported import orr_lighting.path.launch.walk;
__exported import orr_lighting.path.launch.media;
__exported import orr_lighting.path.launch.start;
__exported import orr_lighting.path.launch.hit;
__exported import orr_lighting.path.launch.nee;
__exported import orr_lighting.path.launch.vertex;
__exported import orr_lighting.path.launch.shadow;
__exported import orr_lighting.path.launch.stack;
__exported import orr_lighting.path.launch.front;
__exported import orr_lighting.path.launch.finish;
__exported import orr_lighting.path.launch.raygen_shadow;
__exported import orr_lighting.path.launch.raygen_walk;
__exported import orr_lighting.path.launch.raygen_vertex;
__exported import orr_lighting.path.launch.raygen;
