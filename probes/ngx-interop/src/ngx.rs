//! NGX direct: init with logging, RR (DLSSD) and FG (DLSSG) create and evaluate by parameter
//! keys, recorded into the caller's command buffer.

use crate::gpu::{Gpu, Img};
use crate::sys::*;
use ash::vk::{self, Handle};
use std::ffi::{CStr, CString, c_void};
use std::sync::Mutex;

pub static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "C" fn log_callback(message: *const std::os::raw::c_char, _level: NVSDK_NGX_Logging_Level, _source: NVSDK_NGX_Feature) {
    unsafe {
        let s = CStr::from_ptr(message).to_string_lossy().trim_end().to_owned();
        LOG.lock().unwrap().push(s);
    }
}

pub fn ok(r: NVSDK_NGX_Result) -> bool {
    r == NVSDK_NGX_Result_NVSDK_NGX_Result_Success
}

fn name(b: &[u8]) -> *const std::os::raw::c_char {
    b.as_ptr().cast()
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// NGX's instance and device extensions.
pub unsafe fn required_extensions() -> (Vec<CString>, Vec<CString>) {
    unsafe {
        let (mut ic, mut dc) = (0u32, 0u32);
        let (mut ip, mut dp) = (std::ptr::null_mut(), std::ptr::null_mut());
        let r = NVSDK_NGX_VULKAN_RequiredExtensions(&mut ic, &mut ip, &mut dc, &mut dp);
        assert!(ok(r), "NVSDK_NGX_VULKAN_RequiredExtensions {r:#x}");
        let read = |p: *mut *const std::os::raw::c_char, n: u32| (0..n as usize).map(|i| CStr::from_ptr(*p.add(i)).to_owned()).collect::<Vec<_>>();
        (read(ip, ic), read(dp, dc))
    }
}

fn resource(img: &Img, rw: bool) -> NVSDK_NGX_Resource_VK {
    unsafe {
        let mut r: NVSDK_NGX_Resource_VK = std::mem::zeroed();
        r.Resource.ImageViewInfo = NVSDK_NGX_ImageViewInfo_VK {
            ImageView: img.view.as_raw() as usize as VkImageView,
            Image: img.image.as_raw() as usize as VkImage,
            SubresourceRange: VkImageSubresourceRange { aspectMask: 1, baseMipLevel: 0, levelCount: 1, baseArrayLayer: 0, layerCount: 1 },
            Format: img.format.as_raw(),
            Width: img.extent[0],
            Height: img.extent[1],
        };
        r.Type = NVSDK_NGX_Resource_VK_Type_NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
        r.ReadWrite = rw;
        r
    }
}

const PROJECT_ID: &CStr = c"6a3c1e0d-9b2f-4c7a-8e15-3d0f7b9a2c41";

pub struct Ngx {
    pub caps: *mut NVSDK_NGX_Parameter,
    device: vk::Device,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Quality {
    Performance,
    Balanced,
    Quality,
    Dlaa,
}

impl Quality {
    fn value(self) -> NVSDK_NGX_PerfQuality_Value {
        match self {
            Quality::Performance => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf,
            Quality::Balanced => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced,
            Quality::Quality => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality,
            Quality::Dlaa => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_DLAA,
        }
    }
    fn preset_key(self) -> &'static [u8] {
        match self {
            Quality::Performance => NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance,
            Quality::Balanced => NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
            Quality::Quality => NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
            Quality::Dlaa => NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
        }
    }
}

impl Ngx {
    pub unsafe fn new(gpu: &Gpu, verbose: bool) -> Result<Ngx, String> {
        unsafe {
            let dir = std::env::temp_dir().join("ngx-interop-probe");
            let _ = std::fs::create_dir_all(&dir);
            let data = wide(&dir.to_string_lossy());
            let dll_dir = wide(env!("PROBE_NGX_DLLS"));
            let paths = [dll_dir.as_ptr()];
            let mut info: NVSDK_NGX_FeatureCommonInfo = std::mem::zeroed();
            info.PathListInfo.Path = paths.as_ptr();
            info.PathListInfo.Length = 1;
            info.LoggingInfo.LoggingCallback = Some(log_callback);
            info.LoggingInfo.MinimumLoggingLevel =
                if verbose { NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_VERBOSE } else { NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_ON };
            info.LoggingInfo.DisableOtherLoggingSinks = true;
            let gipa: PFN_vkGetInstanceProcAddr = std::mem::transmute(gpu.entry.static_fn().get_instance_proc_addr);
            let gdpa: PFN_vkGetDeviceProcAddr = std::mem::transmute(gpu.instance.fp_v1_0().get_device_proc_addr);
            let r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                c"0.0.0".as_ptr(),
                data.as_ptr(),
                gpu.instance.handle().as_raw() as usize as VkInstance,
                gpu.pdev.as_raw() as usize as VkPhysicalDevice,
                gpu.device.handle().as_raw() as usize as VkDevice,
                gipa,
                gdpa,
                &info,
                NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            );
            if !ok(r) {
                return Err(format!("init {r:#x}"));
            }
            let mut caps = std::ptr::null_mut();
            NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut caps);
            Ok(Ngx { caps, device: gpu.device.handle() })
        }
    }

    pub unsafe fn caps_i(&self, key: &[u8]) -> i32 {
        unsafe {
            let mut v = -1i32;
            NVSDK_NGX_Parameter_GetI(self.caps, name(key), &mut v);
            v
        }
    }

    pub unsafe fn caps_ui(&self, key: &[u8]) -> u32 {
        unsafe {
            let mut v = 0u32;
            NVSDK_NGX_Parameter_GetUI(self.caps, name(key), &mut v);
            v
        }
    }

    /// RR render size for `quality` at `output` (DLSSD optimal settings).
    pub unsafe fn rr_render_size(&self, output: [u32; 2], quality: Quality) -> Result<[u32; 2], String> {
        unsafe {
            if quality == Quality::Dlaa {
                return Ok(output);
            }
            let mut cb: *mut c_void = std::ptr::null_mut();
            NVSDK_NGX_Parameter_GetVoidPointer(self.caps, name(NVSDK_NGX_Parameter_DLSSDOptimalSettingsCallback), &mut cb);
            if cb.is_null() {
                return Err("no DLSSDOptimalSettingsCallback".into());
            }
            let f: unsafe extern "C" fn(*mut NVSDK_NGX_Parameter) -> NVSDK_NGX_Result = std::mem::transmute(cb);
            NVSDK_NGX_Parameter_SetUI(self.caps, name(NVSDK_NGX_Parameter_Width), output[0]);
            NVSDK_NGX_Parameter_SetUI(self.caps, name(NVSDK_NGX_Parameter_Height), output[1]);
            NVSDK_NGX_Parameter_SetI(self.caps, name(NVSDK_NGX_Parameter_PerfQualityValue), quality.value());
            NVSDK_NGX_Parameter_SetI(self.caps, name(NVSDK_NGX_Parameter_RTXValue), 0);
            let r = f(self.caps);
            if !ok(r) {
                return Err(format!("optimal settings {r:#x}"));
            }
            let (mut w, mut h) = (0u32, 0u32);
            NVSDK_NGX_Parameter_GetUI(self.caps, name(NVSDK_NGX_Parameter_OutWidth), &mut w);
            NVSDK_NGX_Parameter_GetUI(self.caps, name(NVSDK_NGX_Parameter_OutHeight), &mut h);
            Ok([w, h])
        }
    }

    /// VRAM NGX reports for its RR features (DLSSD stats callback), bytes.
    pub unsafe fn rr_vram(&self) -> Option<u64> {
        unsafe {
            let mut cb: *mut c_void = std::ptr::null_mut();
            NVSDK_NGX_Parameter_GetVoidPointer(self.caps, name(NVSDK_NGX_Parameter_DLSSDGetStatsCallback), &mut cb);
            if cb.is_null() {
                return None;
            }
            let f: unsafe extern "C" fn(*mut NVSDK_NGX_Parameter) -> NVSDK_NGX_Result = std::mem::transmute(cb);
            if !ok(f(self.caps)) {
                return None;
            }
            let mut v = 0u64;
            NVSDK_NGX_Parameter_GetULL(self.caps, name(NVSDK_NGX_Parameter_SizeInBytes), &mut v);
            Some(v)
        }
    }

    pub unsafe fn shutdown(self) {
        unsafe {
            NVSDK_NGX_VULKAN_Shutdown1(self.device.as_raw() as usize as VkDevice);
        }
    }
}

/// Row-major, post-multiplied (v * M) camera matrices.
type Mat = [f32; 16];

fn identity() -> Mat {
    let mut m = [0.0; 16];
    for i in 0..4 {
        m[i * 5] = 1.0;
    }
    m
}

/// Reverse-Z infinite perspective, 60 degree vertical field of view, near 0.1.
fn projection(aspect: f32) -> Mat {
    let f = 1.0 / (30f32.to_radians()).tan();
    let mut m = [0.0; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[11] = -1.0;
    m[14] = 0.1;
    m
}

fn clip_translate(dx: f32) -> Mat {
    let mut m = identity();
    m[12] = dx;
    m
}

pub struct RrInputs<'a> {
    pub color: &'a Img,
    pub diffuse: &'a Img,
    pub specular: &'a Img,
    pub normals: &'a Img,
    pub depth: &'a Img,
    pub motion: &'a Img,
    pub spec_motion: &'a Img,
    pub output: &'a Img,
    pub dof_guide: Option<&'a Img>,
}

pub struct Rr {
    handle: *mut NVSDK_NGX_Handle,
    params: *mut NVSDK_NGX_Parameter,
    pub render: [u32; 2],
    pub output: [u32; 2],
    res: Vec<NVSDK_NGX_Resource_VK>,
    mats: Box<[Mat; 2]>,
}

impl Rr {
    /// `preset` 0 = NGX default, else a `NVSDK_NGX_RayReconstruction_Hint_Render_Preset` value.
    pub unsafe fn new(cmd: vk::CommandBuffer, render: [u32; 2], output: [u32; 2], quality: Quality, preset: u32) -> Result<Rr, NVSDK_NGX_Result> {
        unsafe {
            let mut p = std::ptr::null_mut();
            NVSDK_NGX_VULKAN_AllocateParameters(&mut p);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_CreationNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_VisibilityNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Width), render[0]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Height), render[1]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_OutWidth), output[0]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_OutHeight), output[1]);
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_PerfQualityValue), quality.value());
            let flags = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR
                | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
                | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags), flags);
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects), 0);
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_DLSS_Denoise_Mode), NVSDK_NGX_DLSS_Denoise_Mode_NVSDK_NGX_DLSS_Denoise_Mode_DLUnified);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_DLSS_Roughness_Mode), NVSDK_NGX_DLSS_Roughness_Mode_NVSDK_NGX_DLSS_Roughness_Mode_Packed as u32);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Use_HW_Depth), NVSDK_NGX_DLSS_Depth_Type_NVSDK_NGX_DLSS_Depth_Type_HW as u32);
            if preset != 0 {
                NVSDK_NGX_Parameter_SetUI(p, name(quality.preset_key()), preset);
            }
            let mut handle = std::ptr::null_mut();
            let r = NVSDK_NGX_VULKAN_CreateFeature(cmd.as_raw() as usize as VkCommandBuffer, NVSDK_NGX_Feature_NVSDK_NGX_Feature_RayReconstruction, p, &mut handle);
            if !ok(r) {
                NVSDK_NGX_VULKAN_DestroyParameters(p);
                return Err(r);
            }
            let aspect = output[0] as f32 / output[1] as f32;
            Ok(Rr { handle, params: p, render, output, res: Vec::new(), mats: Box::new([identity(), projection(aspect)]) })
        }
    }

    /// Records RR. `jitter` in render pixels.
    pub unsafe fn eval(&mut self, cmd: vk::CommandBuffer, i: &RrInputs, jitter: [f32; 2], reset: bool) -> NVSDK_NGX_Result {
        unsafe {
            self.res.clear();
            let mut list = vec![
                (i.color, false),
                (i.output, true),
                (i.depth, false),
                (i.motion, false),
                (i.diffuse, false),
                (i.specular, false),
                (i.normals, false),
                (i.spec_motion, false),
            ];
            if let Some(g) = i.dof_guide {
                list.push((g, false));
            }
            for (img, rw) in list {
                self.res.push(resource(img, rw));
            }
            let r = self.res.as_mut_ptr();
            let p = self.params;
            let set = |n: &[u8], k: usize| NVSDK_NGX_Parameter_SetVoidPointer(p, name(n), r.add(k).cast::<c_void>());
            set(NVSDK_NGX_Parameter_Color, 0);
            set(NVSDK_NGX_Parameter_Output, 1);
            set(NVSDK_NGX_Parameter_Depth, 2);
            set(NVSDK_NGX_Parameter_MotionVectors, 3);
            set(NVSDK_NGX_Parameter_DiffuseAlbedo, 4);
            set(NVSDK_NGX_Parameter_SpecularAlbedo, 5);
            set(NVSDK_NGX_Parameter_GBuffer_Normals, 6);
            set(NVSDK_NGX_Parameter_MotionVectorsReflection, 7);
            if i.dof_guide.is_some() {
                set(NVSDK_NGX_Parameter_DLSSD_DepthOfFieldGuide, 8);
            } else {
                NVSDK_NGX_Parameter_SetVoidPointer(p, name(NVSDK_NGX_Parameter_DLSSD_DepthOfFieldGuide), std::ptr::null_mut());
            }
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_Jitter_Offset_X), jitter[0]);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_Jitter_Offset_Y), jitter[1]);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_MV_Scale_X), self.render[0] as f32);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_MV_Scale_Y), self.render[1] as f32);
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_Reset), reset as i32);
            // The value NGX's own eval helper writes when no exposure is given (RR guide 3.7).
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_DLSS_Pre_Exposure), 1.0);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_DLSS_Exposure_Scale), 1.0);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width), self.render[0]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height), self.render[1]);
            let m = self.mats.as_mut_ptr();
            NVSDK_NGX_Parameter_SetVoidPointer(p, name(NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX), m.cast());
            NVSDK_NGX_Parameter_SetVoidPointer(p, name(NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX), m.add(1).cast());
            NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd.as_raw() as usize as VkCommandBuffer, self.handle, p, None)
        }
    }

    pub unsafe fn release(self) {
        unsafe {
            NVSDK_NGX_VULKAN_ReleaseFeature(self.handle);
            NVSDK_NGX_VULKAN_DestroyParameters(self.params);
        }
    }
}

pub struct Fg {
    handle: *mut NVSDK_NGX_Handle,
    params: *mut NVSDK_NGX_Parameter,
    output: [u32; 2],
    res: Vec<NVSDK_NGX_Resource_VK>,
    mats: Box<[Mat; 5]>,
}

pub struct FgInputs<'a> {
    pub backbuffer: &'a Img,
    pub depth: &'a Img,
    pub motion: &'a Img,
    pub interpolated: &'a Img,
}

impl Fg {
    pub unsafe fn new(cmd: vk::CommandBuffer, output: [u32; 2], format: vk::Format) -> Result<Fg, NVSDK_NGX_Result> {
        unsafe {
            let mut q = std::ptr::null_mut();
            NVSDK_NGX_VULKAN_AllocateParameters(&mut q);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_CreationNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_VisibilityNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_Width), output[0]);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_Height), output[1]);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_DLSSG_Parameter_BackbufferFormat), format.as_raw() as u32);
            let mut handle = std::ptr::null_mut();
            let r = NVSDK_NGX_VULKAN_CreateFeature(cmd.as_raw() as usize as VkCommandBuffer, NVSDK_NGX_Feature_NVSDK_NGX_Feature_FrameGeneration, q, &mut handle);
            if !ok(r) {
                NVSDK_NGX_VULKAN_DestroyParameters(q);
                return Err(r);
            }
            Ok(Fg { handle, params: q, output, res: Vec::new(), mats: Box::new([identity(); 5]) })
        }
    }

    /// Records interpolation `index` of `count` between the previous and this backbuffer.
    /// `dpan_uv` is the frame's horizontal pan in uv units (clip-space shift 2 * dpan).
    pub unsafe fn eval(&mut self, cmd: vk::CommandBuffer, i: &FgInputs, render: [u32; 2], index: u32, count: u32, frame: u64, reset: bool, dpan_uv: f32, hdr: bool) -> NVSDK_NGX_Result {
        unsafe {
            self.res.clear();
            for (img, rw) in [(i.backbuffer, false), (i.depth, false), (i.motion, false), (i.interpolated, true)] {
                self.res.push(resource(img, rw));
            }
            let aspect = self.output[0] as f32 / self.output[1] as f32;
            let proj = projection(aspect);
            *self.mats = [proj, invert_projection(&proj), clip_translate(2.0 * dpan_uv), clip_translate(-2.0 * dpan_uv), identity()];
            let r = self.res.as_mut_ptr();
            let q = self.params;
            let set = |n: &[u8], k: usize| NVSDK_NGX_Parameter_SetVoidPointer(q, name(n), r.add(k).cast::<c_void>());
            set(NVSDK_NGX_DLSSG_Parameter_Backbuffer, 0);
            set(NVSDK_NGX_DLSSG_Parameter_Depth, 1);
            set(NVSDK_NGX_DLSSG_Parameter_MVecs, 2);
            set(NVSDK_NGX_DLSSG_Parameter_OutputInterpolated, 3);
            let ui = |n: &[u8], v: u32| NVSDK_NGX_Parameter_SetUI(q, name(n), v);
            let f = |n: &[u8], v: f32| NVSDK_NGX_Parameter_SetF(q, name(n), v);
            ui(NVSDK_NGX_DLSSG_Parameter_MultiFrameCount, count);
            ui(NVSDK_NGX_DLSSG_Parameter_MultiFrameIndex, index);
            ui(NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, frame as u32);
            let m = self.mats.as_mut_ptr();
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_CameraViewToClip), m.cast());
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_ClipToCameraView), m.add(1).cast());
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_ClipToLensClip), m.add(4).cast());
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip), m.add(2).cast());
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_PrevClipToClip), m.add(3).cast());
            f(NVSDK_NGX_DLSSG_Parameter_MvecScaleX, render[0] as f32);
            f(NVSDK_NGX_DLSSG_Parameter_MvecScaleY, render[1] as f32);
            f(NVSDK_NGX_DLSSG_Parameter_JitterOffsetX, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_JitterOffsetY, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraPosX, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraPosY, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraPosZ, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraUpX, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraUpY, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraUpZ, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraRightX, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraRightY, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraRightZ, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFwdX, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFwdY, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFwdZ, -1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraNear, 0.1);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFar, 10000.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFOV, 60f32.to_radians());
            f(NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio, aspect);
            ui(NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR, hdr as u32);
            ui(NVSDK_NGX_DLSSG_Parameter_DepthInverted, 1);
            ui(NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded, 1);
            ui(NVSDK_NGX_DLSSG_Parameter_Reset, reset as u32);
            ui(NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames, 0);
            ui(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectWidth, render[0]);
            ui(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectHeight, render[1]);
            ui(NVSDK_NGX_DLSSG_Parameter_DepthSubrectWidth, render[0]);
            ui(NVSDK_NGX_DLSSG_Parameter_DepthSubrectHeight, render[1]);
            NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd.as_raw() as usize as VkCommandBuffer, self.handle, q, None)
        }
    }

    pub unsafe fn interpolation_disabled(&self) -> u32 {
        unsafe {
            let mut v = 0u32;
            NVSDK_NGX_Parameter_GetUI(self.params, name(NVSDK_NGX_DLSSG_Parameter_OutputDisableInterpolation), &mut v);
            v
        }
    }

    pub unsafe fn release(self) {
        unsafe {
            NVSDK_NGX_VULKAN_ReleaseFeature(self.handle);
            NVSDK_NGX_VULKAN_DestroyParameters(self.params);
        }
    }
}

fn invert_projection(p: &Mat) -> Mat {
    // p = [a 0 0 0; 0 b 0 0; 0 0 0 -1; 0 0 n 0] (row-major, v * M).
    let mut m = [0.0; 16];
    m[0] = 1.0 / p[0];
    m[5] = 1.0 / p[5];
    m[11] = 1.0 / p[14];
    m[14] = -1.0;
    m
}
