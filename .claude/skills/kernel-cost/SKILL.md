---
name: kernel-cost
description: Explains what orr's Slang GPU kernels cost in the driver and on the GPU (registers and spill, driver compile time, per-thread state, code a scene does not run) and how to measure each. Use when editing or adding .slang code that a path stage, a support kernel, a surface evaluation or a hit resolve can reach; adding a material, light, medium or geometry feature; adding or changing a path_may or path_may2 bit, per-path state or a ray payload; restructuring, splitting or merging kernels, or moving work into a ray tracing pipeline; or looking into a slow pass, slow pipeline compiles, cold start, spill or low occupancy. Use it for small changes too, because one more call site of a heavy body is a full copy of it in every kernel that reaches the site.
---

# Kernel cost

A kernel's cost comes from how the driver compiles it and how the GPU runs it, not from its
size or structure. This skill holds no measurements: read the current numbers from the driver
and the counters, for the GPU, driver, scene and mask in front of you.

## Where the cost comes from

- **Registers.** A GPU core keeps many warps resident and switches to another when one waits
  on memory or a ray. Its register file is fixed, so the registers each thread gets decide how
  many warps fit. At the register ceiling a core has few to switch between and idles.
- **Spill.** The driver picks a kernel's register count when it compiles the pipeline, from
  the point where the most values are live (values kept across a ray query or a loop, a large
  struct kept for a later use, a candidate handler inside a query's loop), whether or not a
  scene runs that code. What the registers cannot hold spills to memory, which each thread
  crossing that point loads and stores.
- **Per-thread state.** Every stage loads and stores per-path state every round. An array
  indexed at run time lives in each thread's local memory, and every invocation writes a
  `static` array's initializer whether or not anything reads the array.
- **Call sites.** Slang inlines every call. That is what makes the code fast, and it is not
  the thing to fight. What costs is a heavy body reached from many sites: each is another full
  copy for the driver to compile and for threads to hold registers for, often where only part
  of it is needed. The driver's compile time grows faster than the code, and a scene's first
  lit frame waits for its slowest pipeline.
- **Ray tracing pipelines.** A launch has one register count for all its threads, set by the
  largest shader linked into it whether or not that shader runs, under a ceiling lower than a
  compute kernel's. Payload bytes cost registers in every thread.

## What the work aims for

- **No spill:** each kernel under its launch's register ceiling.
- **Driver compile time a person can live with:** about ten to fifteen seconds cold for a
  large kernel on a scene's mask.
- **Little state that every thread pays for:** per-path state, per-thread arrays, payload.
- **Code a scene does not use costs that scene nothing:** gated by the mask.

A kernel of any size or structure that meets these is fine, and none of them picks a design.
For a heavy body, look at how many sites reach it and whether each needs it: one site where
possible, a narrower helper where a site needs only part, the mask where a scene needs none.

## Gating with the mask

`path_may` and `path_may2` say what a scene's paths may meet. The fold
(`crates/orr_gpu/src/vulkan/specialize.rs`) deletes the blocks a scene's mask rules out before
the driver sees the module; specialization constants alone leave the driver to read all of it.

- Gate large code that not every scene uses with a direct test where the feature is used:
  `if (may2(MAY2_DECALS) && has_decal) { ... }`. The fold follows a bit through integer and
  boolean tests and scalar locals; a gate that reads it from a root or record field removes
  nothing.
- Set the bit from what the scene holds, as narrowly as the scene allows (`SceneFeatures`,
  `scene_may`, `features_may`). A bit set without need keeps code that holds registers and
  compile time; a bit clear where the feature is used gives a wrong image with no error.
- Add a rendered test that fails with the bit forced clear, like the `specialized_pipelines_*`
  tests in `tests/lighting_suite`. The driver may round two specializations differently, so
  last-bit differences alone are not a wrong bit.

## Measuring

Measure what ships (release, or a dev build with `ORR_SHADER_DEBUG_INFO=off` and
`ORR_SHADER_ASSERTS=0`), before and after, on the same machine, scene, mask and build, and
report those with the GPU and driver beside every number.

- **Registers, spill, code size, compile time.** With `RUST_LOG=orr_gpu=debug`, orr_gpu logs
  each pipeline it creates: its name (a path stage's carries its mask), the creation time and
  every statistic the driver reports. On NVIDIA those include the register count, the machine
  code size, and local and shared memory, which hold spill beside the kernel's own arrays and
  `groupshared` data. The local memory value's low 32 bits are bytes per thread. Kernels that
  overflow all report the same register count: the ceiling. A creation time is a compile only
  with an empty pipeline store (`ORR_TEST_PIPELINE_STORE`) and driver disk cache (NVIDIA:
  `__GL_SHADER_DISK_CACHE_PATH`), and it varies between runs, so compare within one run or
  over several. The examples read `RUST_LOG`, as do the tests on `gpu_suite`'s device (the
  lighting suite's among them), whose output the test runner writes to `target/test-logs`, a
  passing test's only with `-- --nocapture`.
- **Call sites.** `python -I ${CLAUDE_SKILL_DIR}/scripts/spv_size.py --lines --stage trace`,
  on a build with line tables (dev, or the `profiling` profile), lists a path stage's
  instructions by source file and the lines with the most copies: a heavy body inlined at many
  sites.
- **What holds registers or spill.** The driver reports no liveness per line, so find the
  holder by removal: clear one mask bit, or replace one part with a load from memory keyed on
  that part's inputs, then recompile and compare. At the ceiling, spill bytes are the measure
  of progress; under it the register count can rise as code is removed, because the driver
  picks it. A stand-in that loads a whole struct makes every field live and can raise spill.
- **Why a pass is slow.** The README's Profiling section covers exact pass times, per-pass
  hardware counters and `bench`, which reports both. The counters tell the causes apart (the
  words in parentheses find them in the SDK's metric list):
  - too few threads: few launched for the items handled (`threads_launched`) and cores mostly
    idle, where a dispatch sized from the data gives each item a thread;
  - register-limited: a full register file (`registers_allocated`) with few warps resident
    (`warps_active`); the two also give a ray tracing launch's registers per thread where the
    driver reports no statistics for its pipeline;
  - waiting: warps stall on the long scoreboard (`warp_issue_stalled`), with cache hit rates
    and DRAM throughput pointing at memory, RT core throughput at traversal, `sysmem` and
    `pcie` traffic at a buffer in host memory, and `mem_local` bytes at spill or arrays;
  - instruction fetch: warps stall with no instruction, which points at code size.

Metal has neither the statistics nor the counters.

## Numbers from a half-built design

Most structural changes here pay off only once several pieces are in: the mask folding, heavy
bodies at one site, state slimmed, kernels under the register ceiling. A number taken before
then measures the stand-ins as much as the idea.

- With an intermediate or probe result, say what was unbuilt or still fat on each side. A
  technique measured on a kernel at the register ceiling says little about it on a thin one,
  and a restructured kernel can measure slower for something else that changed with it, such
  as where its buffers live.
- Build the correct, image-identical steps of an agreed design through, and judge it on whole
  frames of real scenes once the pieces are in. Building it all is cheap, and a design that
  then fails can be rolled back.
- A failure seen once with an unknown cause says nothing yet about the technique.

## Facts about the API

- The host records the barriers between a pass's rounds, and nothing in Vulkan lets the GPU
  skip them, so an empty round still costs its barriers.
- Lanes that split at a branch are guaranteed back together for subgroup (wave) operations
  only where the whole workgroup entered the branch together, unless the kernel declares
  maximal reconvergence (`[MaximallyReconverges]`, as the software raster does).
