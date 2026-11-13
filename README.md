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

Each pass's GPU time comes from timestamps around it (`PassResult`, in `Device::pass_results` and
`FrameStats::gpu`). Passes on a queue overlap, and the driver can write a pass's end timestamp
before its last workgroups finish and its begin timestamp before the work ahead of it does, so a
pass's time can lose its tail to the passes after it while the frame's total stays right. For
exact times a program asks for the frames to measure: `Renderer::set_exact_timing(true)` before
their `begin_frame` and `false` after, or `CommandList::set_exact_timing` on lists of its own.
Each of their passes then waits for everything before it on its queue to finish before each of
its timestamps, so its time is its own work's and its `PassResult::exact` is set. Their passes
stop overlapping, so those frames take longer than the frames around them: time a few and leave
the rest alone. `bench` times three frames of its first script exactly and prints their passes
beside the others; `--exact-frame N` and `--exact-frames K` choose them. On Metal every pass's
timestamps wait so already.

Per-pass hardware counters come from NVIDIA's Nsight Perf SDK, with orr_gpu's `nsight-perf`
feature, off by default and Vulkan only. The build reads the SDK's headers from
`NVPERF_SDK_PATH`. At run time the host library loads from the SDK a program names in
`CounterDesc::sdk`, else from beside the program or the system's library path; the examples
name `NVPERF_SDK_PATH`'s. `bench` then adds one frame's counters to its table and report:

```sh
cargo run --release -p orr_examples --features nsight-perf --example bench -- --headless
```

Its `--counter-metrics` takes other metrics, comma-separated, by the names the SDK lists under
`NvPerf/doc/metrics`, and `--stage-passes` times each stage of the path loop as a pass of its own.

A program loads the SDK with `DeviceDesc::perf_counters`, opens a `CounterSession` for the
metrics it wants with `Queue::start_counters`, and puts each frame it counts between the
session's `begin_frame` and `end_frame`. Each pass's `PassResult::counters` then sits beside its
time in `Device::pass_results` and `FrameStats::gpu`, one value per metric under the SDK's name
and unit, `None` for a metric the GPU doesn't expose. Metrics that don't fit together take
several frames (`CounterSession::replays`), each counting a part: the program renders the same
frame again, the same passes from the same scene, view and settings, and holds still whatever
state it carries from frame to frame (accumulation, history, animation, random sequences);
otherwise each metric describes a slightly different frame. Only the last frame of a collection
carries counters. `DEFAULT_COUNTER_METRICS` (pass time, SM activity and instruction issue,
occupancy, L1 and L2 hit rates, DRAM and RT core throughput) take two frames on Blackwell GPUs,
so these rules hold for the default; the SDK's `sm__throughput` takes dozens.

While it counts, the SDK drains the GPU before and after every pass. A counted frame's passes
don't overlap and are timed exactly, its pass times include the SDK's few microseconds of range
work, and collecting costs the CPU a few milliseconds per frame; frames outside `begin_frame`
and `end_frame` cost nothing more. Counters are the whole GPU's, other queues' and programs'
work included. A pass of a few microseconds is mostly the SDK's range work, so its rates and
percentages say little. For numbers that compare from run to run, keep the scene, camera,
resolution, settings and frame the same, and lock the GPU's clocks as the SDK's guide describes
under "Ensuring Stable Measurements".

CPU and GPU timelines come from the [Tracy](https://github.com/wolfpld/tracy) profiler, with
orr_gpu's `tracy` feature, which `orr` forwards; off by default, and without it nothing is
compiled in. orr's steps get zones (`begin_frame`, `update_scenes`, `render_view`, `end` and the
work inside each, such as the wait for an earlier frame, streaming, uploads and the recording of
each part of a view), as do submits, presents, the loading thread's work and loading jobs, and
pipeline creation, whose zone shows the pipeline's name (a path stage's with its mask). orr names
the threads it starts, marks a frame at `Frame::end` and a `presented image` frame at each
present, and plots the path pipelines compiling, the loading jobs queued, the streams and mount
reads in flight and each frame's upload bytes. Each pass is a zone on its queue's GPU context,
placed on the CPU timeline by its `PassResult::cpu` times, so GPU zones need
`Caps::calibrated_timestamps`, arrive a few frames late, and are exact only in frames timed
exactly. In Tracy a GPU zone's CPU-side times are when orr read its result, not
when it was recorded. Metal has no GPU zones. `orr_cook` has a `tracy` feature of its own.

The examples' `tracy` feature turns on orr's and the cook's, with the Tracy client's default
features. tracy-client's `only-localhost` keeps the client off the network:

```sh
cargo run --profile profiling -p orr_examples --features tracy,tracy-client/only-localhost --example bench -- --headless
tracy-capture -a 127.0.0.1 -o bench.tracy
```

A program built on orr turns on `orr`'s `tracy` feature and links the same Tracy client: it
depends on `tracy-client` `=0.19.0` and `tracy-client-sys` `=0.30.1`, the versions orr pins,
which speak the protocol of Tracy 0.14.1, so capture with that release's tools. orr asks only for
the client's `enable` feature; the program chooses the rest (`only-localhost`, `ondemand`,
sampling and so on) on its own dependency. orr never starts the client: it runs from process start
by default, and under `manual-lifetime` orr records nothing until the program starts it. The
program's own zones around orr's calls hold orr's zones in the same capture. orr marks the frames,
so the program marks none of its own, or marks secondary frames under names of its own. With the
feature, a zone costs a timestamp and a queue write whether or not a profiler is connected, and
without `ondemand` the client keeps every event in memory until one connects.

## Contributing

Install the pre-commit hook, which runs fmt, clippy and typos:

```sh
cargo install --locked prek typos-cli
prek install
```
