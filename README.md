# orr

orr (OpenPBR Raytracing Renderer) is an experimental renderer for Rust and Vulkan, built around
path tracing and OpenPBR materials, and aimed at today's top end GPUs and the ones coming next.

It starts from one question: what would you build if you were starting now and looking ahead?
So all lighting is raytraced, geometry is meshlets from the start, the GPU decides what to draw,
and assets are built ahead of time into the forms the GPU reads directly. There's no shadow map
fallback and no older path kept alongside.

Full documentation is coming as a separate site.

## Get the data onto the GPU and let it work

orr follows the thinking in Sebastian Aaltonen's
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api), across the whole
renderer and not only at the GPU layer:

- **Data, not objects.** The scene is plain structs in GPU memory, reached through 64-bit pointers
  and 32-bit indices. There are no descriptor sets, vertex input or resource states to manage,
  and no CPU side copy of the scene to keep in sync.
- **You write data.** Updating the scene means writing to mapped memory in a known layout, then
  making a few calls in a fixed frame order. One render thread runs those steps, with no render
  graph or task scheduler. orr starts only a loading thread and a present thread, and other CPU
  work runs on your threads.
- **The GPU does the work.** Level of detail selection, culling, draw generation and streaming
  requests all happen on the GPU, and textures and meshes decompress there too.
- **Your code is linked in.** Surfaces, displacement, skies and media are Slang functions compiled
  into orr's shaders. There are no material graphs or plugin registries.

orr is a renderer and stays low level. Your application owns the window, the game logic and the
world.

## What's in it

**Geometry**

- Meshlets in a cluster LOD DAG, with two pass HZB occlusion culling
- Mesh shaders for large triangles and compute rasterization for small ones, writing a visibility
  buffer
- Cluster acceleration structures and a partitioned TLAS, so raytracing sees the same detail as
  raster
- Opacity micromaps for alpha tested geometry
- GPU skinning through cluster templates, Catmull-Clark subdivision and displacement as tessellated
  clusters, and strands as linear swept spheres
- Device generated commands, and geometry from your own GPU code

**Streaming**

- Meshlet streaming: mesh pages and their cluster acceleration structures stream in as the view
  needs them
- Texture streaming through sparse images, driven by what shaders actually sample
- Neural texture compression (NTC), decoded on the GPU with cooperative vectors, with its latents
  streamed as tiles
- Instance pages that stream and evict, and your own data streamed into your buffers and images
- Fast file IO through IoRing or io_uring, GPU decompression with GDeflate, and budgets that shrink
  when VRAM runs short

**Lighting and materials**

- Path tracing with shader execution reordering
- A real time OpenPBR BSDF, checked against the reference implementation
- Light selection across many lights, including area lights and emissive meshes, guided by per
  pixel feedback, with light and BSDF samples weighed against each other, plus a world radiance
  cache
- Glass with nested interiors, absorption and hero wavelength dispersion, which every path ray
  bends at
- Caustics through glass and mirrors from photons traced each frame into a world space grid, and
  a progressive photon map in the accumulation reference
- Random walk subsurface scattering
- Froxel fog, smoke, and volumes from cooked or your own data
- A physically based, spectral sky, sun and moon in `orr_atmosphere`

**Output**

- HDR first: rendering happens in linear Rec.2020, with HDR10 (PQ), scRGB and SDR output that
  follows the display's settings
- Gran Turismo 7's tone mapping, gamut compression for SDR, and lights in physical units
- DLSS ray reconstruction, super resolution, DLAA and frame generation, with Reflex for low latency,
  behind the default `dlss` feature, which does nothing on Apple targets
- Auto exposure, white balance, local exposure, depth of field, motion blur and glare
- A UI layer you can draw into, with `orr_egui` as a simple way to get debug UI on screen for now

**Assets**

- `orr_cook` turns glTF, USD, MaterialX and OpenVDB sources into archives of GPU friendly data: mesh
  pages, LOD DAGs, opacity micromaps, BCn and NTC textures, and sparse volume bricks
- Use it as an offline build step, or as a library to cook in process at run time
- Cooks are incremental, and results are cached by content

## Getting started

You'll need:

- an NVIDIA RTX card with a current driver
- Rust, with the version picked by `rust-toolchain.toml`
- a C++17 compiler
- optionally, the NVIDIA DLSS SDK, with `DLSS_SDK` set, and the Vulkan SDK for its headers, with
  `VULKAN_SDK` set (orr builds without them)

Then:

```sh
bash tools/fetch-slang.sh
cargo run -p orr_examples --example viewer
```

The other programs in `examples/` show one feature each: try `sky`, `dielectrics` for glass,
`skinned` for animated characters, and `dense-vegetation` for a large scene. When you're ready to
read code, start with those examples and the `orr` crate.

### Profiling

Each kernel's entry point is named after its module, such as `orr_lighting.path.staged.trace`, so
a GPU profiler tells kernels apart, and passes carry debug labels. For hotspots at Slang source
lines, build with line tables and the source text in the SPIR-V:

```sh
ORR_SHADER_DEBUG_INFO=lines cargo build --release -p orr_examples --example viewer
```

The variable reaches every crate that builds kernels with `orr_shaders::build`, a program's own
too. Release builds also leave out shader asserts. Slang optimizes less around line tables, so a
`lines` build's kernels keep some locals in memory and are about a percent larger than a normal
build's: its numbers are close to a normal build's but not identical.

## Contributing

Install the pre-commit hook, which runs fmt, clippy and typos:

```sh
cargo install --locked prek typos-cli
prek install
```
