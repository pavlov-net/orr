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
a GPU profiler tells kernels apart, and passes carry debug labels. Kernels follow the Rust debug
info of the build: where it has some, they carry line tables and their Slang source, so a
profiler maps hotspots to source lines; release builds have none. The `profiling` profile is
release with line tables:

```sh
cargo build --profile profiling -p orr_examples --example viewer
```

A program built on orr gets the same from its own profiles, which govern orr's build scripts.
`ORR_SHADER_DEBUG_INFO` (`off`, `lines`, `embedded`, `separate`) overrides the profile for every
crate that builds kernels with `orr_shaders::build`. Slang optimizes less around line tables, so
kernels with them keep some locals in memory and are about a percent larger: a profiling
build's numbers are close to a release build's but not identical. Release and profiling builds
also leave out shader asserts.

Per-pass hardware counters come from NVIDIA's Nsight Perf SDK, with orr_gpu's `nsight-perf`
feature, off by default and Vulkan only. The build reads the SDK's headers from
`NVPERF_SDK_PATH`. At run time the host library loads from the SDK a program names in
`CounterDesc::sdk`, else from beside the program or the system's library path; the examples
name `NVPERF_SDK_PATH`'s. `bench` then adds one frame's counters to its table and report:

```sh
cargo run --release -p orr_examples --features nsight-perf --example bench -- --headless
```

A program loads the SDK with `DeviceDesc::perf_counters`, opens a `CounterSession` for the
metrics it wants with `Queue::start_counters`, and puts each frame it counts between the
session's `begin_frame` and `end_frame`. Each pass's `PassResult::counters` then sits beside its
time in `Device::pass_results` and `FrameStats::gpu`, one value per metric under the SDK's name
and unit, `None` for a metric the GPU doesn't expose. `DEFAULT_COUNTER_METRICS` (pass time, SM
activity and instruction issue, occupancy, L2 hit rate, DRAM and RT core throughput) fit in one
frame on Blackwell GPUs, so every counted frame has its own counters. Metrics that don't fit
together, such as the SDK's `sm__throughput`, take several frames (`CounterSession::replays`),
each counting a part: the program renders the same frame again, the same passes from the same
scene, view and settings, and holds still whatever state it carries from frame to frame
(accumulation, history, animation, random sequences); otherwise each metric describes a
slightly different frame.

While it counts, the SDK drains the GPU before and after every pass. A counted frame's passes
don't overlap, its pass times include the SDK's few microseconds of range work, and collecting
costs the CPU a few milliseconds per frame; frames outside `begin_frame` and `end_frame` cost
nothing more. Counters are the whole GPU's, other queues' and programs' work included. Pass
timestamps can end before a pass's last workgroup finishes, so for a pass with a long tail the
counted `gpu__time_duration` is the truer time. A pass of a few microseconds is mostly the
SDK's range work, so its rates and percentages say little. For numbers that compare from run
to run, keep the scene, camera, resolution, settings and frame the same, and lock the GPU's
clocks as the SDK's guide describes under "Ensuring Stable Measurements".

## Contributing

Install the pre-commit hook, which runs fmt, clippy and typos:

```sh
cargo install --locked prek typos-cli
prek install
```
