# orr-dlss

DLSS ray reconstruction, super resolution, DLAA and frame generation through NVIDIA NGX on
Vulkan, without Streamline.

## Building

The DLSS SDK is not part of this crate. Download it from NVIDIA under its licence and set
`DLSS_SDK` to the absolute path of the copy. The build generates bindings from its headers
with bindgen, so it needs clang and the Vulkan SDK headers (`VULKAN_SDK`).

| orr-dlss | DLSS SDK |
| --- | --- |
| 0.0.1 | v310.9.16 |

Without `DLSS_SDK` the crate builds without NGX and reports DLSS unavailable at run time.
Features:

- `mock`: no SDK needed. Every feature reports unsupported, and ray reconstruction and super
  resolution copy their color input to the output with a bilinear upscale.
- `debug_overlay`: loads the SDK's `dev` DLLs, which draw the DLSS debug overlay. Never ship
  these.

## Use

Create the orr-gpu device with `DeviceDesc::hooks` set to `DlssSdk::device_hooks(project_id)`,
then `DlssSdk::new` on it. `RayReconstruction::record`, `SuperResolution::record` and
`FrameGeneration::record` record into an orr command list. Ray reconstruction upscales path
traced views; super resolution upscales views that are not path traced, such as raster debug
views. Frame generation is usually driven by orr-output's `Display`, which presents the
generated images. The `raw` module offers the same features on ash handles for programs
without orr-gpu.

## Runtime DLLs

NGX looks for its feature DLLs beside the executable first. Debug builds also look in the
SDK the crate was built against, and every build looks in `$DLSS_SDK` at run time.

## Shipping

- Copy `lib/Windows_x86_64/rel/nvngx_dlss.dll` (super resolution and DLAA),
  `nvngx_dlssd.dll` (ray reconstruction) and `nvngx_dlssg.dll` (frame generation) from the SDK
  beside the executable. Never ship the `dev` DLLs.
- When code-signing, append your signature; never strip NVIDIA's.
- Include the copyright and licence notices listed in section 9.5 of the DLSS programming
  guide that comes with the SDK.
- The SDK's `LICENSE.txt` (section 1c) allows distributing the SDK's files in object code
  inside an application, except developer tools meant for internal use.
- The licence requires notice to NVIDIA before a commercial release (section 4), and NVIDIA
  marks on the splash screen, the about box and the credits (section 7.1b).
- NGX is initialised with your project ID (a UUID you choose) and the custom engine type.
