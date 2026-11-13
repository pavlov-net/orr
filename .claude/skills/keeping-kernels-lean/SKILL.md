---
name: keeping-kernels-lean
description: Keeps orr's Slang GPU kernels quick for the driver to compile and small in per-path state. Use it whenever you edit or add .slang code that a path stage, a support kernel, a surface evaluation or a hit resolve can reach; add a material, light, medium or geometry feature; add or change a path_may or path_may2 bit or per-path state; or look into slow pipeline compiles, cold start or shader size. Use it for small changes too, because one ungated call can add seconds to every scene's first lit frame.
---

# Keeping kernels lean

A scene's first lit frame waits for the driver to compile that scene's path stages and the
support kernels it reads, so kernel compile time is something a user sees.

## How the cost works

- Slang inlines every call, so each call site is a full copy of the callee.
- The driver reads every instruction it is given. Its back end grows faster than linearly with
  what survives its own cleanup, so each extra copy costs more than the last. Ray queries with
  candidate handling inside a loop cost the most.
- Pipelines compile in parallel, so cold start is set by the slowest pipeline the frame waits
  for, whether a path stage or a support kernel.
- Each path stage is compiled per scene. `path_may` (`crates/orr_lighting/src/specialize.slang`)
  and `path_may2` (`crates/orr_scene/src/scene_may.slang`) say what a scene's paths may meet; a
  scene's values for the two are its mask. Before the driver sees a module, the fold
  (`crates/orr_gpu/src/vulkan/specialize.rs`) applies the mask and deletes the blocks it rules
  out. Passing the mask as specialization constants is not enough: a stage the driver folds
  itself compiles several times slower than one with the code removed.
- A plain scene's stage should be much smaller than the stage that may meet everything. If it
  isn't, a gate isn't folding.
- Every round of a pass runs as stages, however few paths reach it: a late round pays its
  dispatches and barriers even when its queues are empty. Keep that fixed cost small rather
  than folding rounds into a looping kernel, which holds every stage it loops over.
- The pipeline store fills with the number of distinct pipelines more than with their code size,
  so each new axis of specialization multiplies it.

## When you add or change code a kernel reaches

1. **Put it in the stage that owns it.** The opaque shade stage stays plain: lobes a plain surface
   lacks go to the rich stage, layers and interiors to the other shade stage, ray queries to trace
   and walk. The stages and the path's specification are in
   `crates/orr_lighting/src/path/stages.slang`. Probe and diagnostic kernels get their own
   pipelines and call functions the stages already export, rather than adding to a stage.
2. **Gate large code that not every scene uses.** Code that is small, or that every scene needs,
   stays ungated. Use the bit whose comment covers your case, or add one and say in its comment
   what it covers. Test it directly where the feature is used:
   `if (may2(MAY2_DECALS) && has_decal) { ... }`. The fold follows a bit through integer and
   boolean tests, phis and scalar locals. It does not follow a flag stored in a root or record
   field, so a gate written that way removes nothing.
3. **Set the bit from the scene.** Detection goes in `crates/orr_scene/src/features.rs`
   (`SceneFeatures`, `surface_may`) or in `scene_may` and `features_may` in
   `crates/orr_lighting/src/view.rs`. Derive it from what the scene holds, not from a view
   setting. A bit set when it need not be only costs time; a bit clear when the scene uses the
   feature gives a wrong image with no error.
4. **Add a rendered test that fails with the bit forced clear.** Follow the
   `specialized_pipelines_*` tests in `tests/lighting_suite/tests/path.rs`: render through the
   pipelines that may meet everything and through the scene's own, and require the same image.
   Force the bit clear by hand once to see the test fail, then restore it. If it still passes,
   the test doesn't reach the feature. When one of these tests fails, look at a single path
   first: the same decisions with last-bit differences mean the driver rounded the two pipelines
   differently, not that a bit is wrong.
5. **Count the call sites of large bodies.** A surface evaluation, `bsdf_prepare`, a hit resolve
   or a shadow trace with candidates is thousands of instructions per site, and a site brings
   everything it can reach: a shadow candidate that can be a dielectric pulls a hit resolve and a
   surface evaluation into every shadow-ray site. Code repeated per case, such as per material
   slot, multiplies the same way. Feed an existing site instead of adding one, with a loop over
   cases or in the stage that already has it. Small bodies copied a few times are rarely worth
   cutting.
6. **Keep per-path state small.** Every stage reads and writes it every round, so its size shows
   directly in frame time. What costs is the words a stage loads and keeps live, not the
   allocation. Before adding a word, check whether the value can be recomputed after the stage
   boundary from what is already stored, and whether it belongs in a part that only scenes with
   the feature allocate (`StateParts` in `crates/orr_lighting/src/staged.rs`).
7. **Compare sizes before and after** for the stages your change reaches, as in Measuring. An
   unexpected jump is usually a new call site.

## Settled by measurement

Don't retry these without new evidence. The ratios come from one NVIDIA GPU and driver.

These change the SPIR-V, so they should hold on any driver:

- **Optimizing the SPIR-V at build time** (`-O2`, SPIRV-Tools performance passes): compiles get
  faster, frames get about twice as slow, and the shader build takes over ten times longer. An
  SSA-only pass list keeps frame time but stops helping once several stages compile at once, and
  grows the modules. `--if-conversion` slows frames.
- **`[noinline]` on large bodies:** frames 1.6-1.9 times slower, and the driver can fail to
  compile the result or lose the device on it.

These depend on the driver, so measure again on other hardware:

- **One looping kernel for the whole path, or for a pass's late rounds:** faster per frame, but
  it takes minutes to compile where a stage takes seconds. A rest stage that ran the rounds
  from the fifth on in one lane saved about a tenth of a small bed's frame and compiled
  seven to twelve times as long as the slowest stage.
- **One entry point per module:** no gain in the driver, and longer builds.
- **Shade stages as ray-generation entries** (for reordering): about twice as slow as compute
  with no reordering, cause unknown.
- **A fold that removes more than the mask makes unreachable:** lost the device, cause unknown.
  The fold removes only what the mask makes unreachable.

## Measuring

Compare with the same stage before your change, on the same machine; numbers from another machine
or an older build don't transfer. Compile times of the same module vary a lot between runs and
processes, so trust only differences measured in the same run, or of several times.

- **Size:** `python -I ${CLAUDE_SKILL_DIR}/scripts/spv_size.py --stage trace` reports the newest
  build of a stage module (`primary`, `shade_opaque`, `shade_other`, `trace`, `walk`,
  `sources`, `reset`) and prints its path; it also accepts module paths. It reads the module as
  emitted, before the fold, so the size is the same for every mask and every GPU. With `--lines`
  on a build made with `ORR_SHADER_DEBUG_INFO=lines`, it lists instructions by source file and
  the lines with the most copies, which finds a body inlined at many sites.
- **Compile time:** only a cold compile counts, because the pipeline store and the driver's cache
  serve later ones. Time with an empty store (`ORR_TEST_PIPELINE_STORE` set to a new directory)
  and an empty driver cache directory, and record the mask, GPU and driver.
- **Frame time:** a smaller kernel can be slower, and code that never runs can still cost frame
  time. Compare pass times on the same scene before calling a size reduction a win.
