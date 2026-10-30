//! NGX ray reconstruction and frame generation on synthetic inputs, recorded into the frame's
//! command buffer. Only the CPU cost of the evaluate calls matters here.

use crate::gpu::{Gpu, Img};
use crate::sys::*;
use ash::vk::{self, Handle};
use std::ffi::{CStr, CString, c_void};

fn ok(r: NVSDK_NGX_Result) -> bool {
    r == NVSDK_NGX_Result_NVSDK_NGX_Result_Success
}

fn name(b: &[u8]) -> *const std::os::raw::c_char {
    b.as_ptr().cast()
}

pub fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// NGX's instance and device extensions.
pub unsafe fn required_extensions() -> (Vec<CString>, Vec<CString>) {
    unsafe {
        let (mut ic, mut dc) = (0u32, 0u32);
        let (mut ip, mut dp) = (std::ptr::null_mut(), std::ptr::null_mut());
        let r = NVSDK_NGX_VULKAN_RequiredExtensions(&mut ic, &mut ip, &mut dc, &mut dp);
        if !ok(r) {
            return (vec![], vec![]);
        }
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

pub struct Inputs {
    pub color: Img,
    pub depth: Img,
    pub motion: Img,
    pub diffuse: Img,
    pub specular: Img,
    pub normals: Img,
    pub rr_out: Img,
}

pub struct Ngx {
    rr_params: *mut NVSDK_NGX_Parameter,
    fg_params: *mut NVSDK_NGX_Parameter,
    rr: *mut NVSDK_NGX_Handle,
    fg: *mut NVSDK_NGX_Handle,
    pub fg_ok: bool,
    res: Vec<NVSDK_NGX_Resource_VK>,
    mats: Box<[[f32; 16]; 4]>,
    render: [u32; 2],
    output: [u32; 2],
}

const PROJECT_ID: &CStr = c"6a3c1e0d-9b2f-4c7a-8e15-3d0f7b9a2c41";

impl Ngx {
    /// Initialises NGX and creates RR (render -> output) and FG (output, `fg_format`). `cmd` is
    /// recording; the caller submits and waits it.
    pub unsafe fn new(gpu: &Gpu, cmd: vk::CommandBuffer, render: [u32; 2], output: [u32; 2], fg_format: vk::Format) -> Result<Ngx, String> {
        unsafe {
            let data = wide(&std::env::temp_dir().join("present-probe-ngx").to_string_lossy());
            let _ = std::fs::create_dir_all(std::env::temp_dir().join("present-probe-ngx"));
            let dll_dir = wide(env!("PROBE_NGX_DLLS"));
            let paths = [dll_dir.as_ptr()];
            let mut info: NVSDK_NGX_FeatureCommonInfo = std::mem::zeroed();
            info.PathListInfo.Path = paths.as_ptr();
            info.PathListInfo.Length = 1;
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
            let (mut rr_av, mut fg_av, mut max) = (0i32, 0i32, 0u32);
            NVSDK_NGX_Parameter_GetI(caps, name(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available), &mut rr_av);
            NVSDK_NGX_Parameter_GetI(caps, name(NVSDK_NGX_Parameter_FrameGeneration_Available), &mut fg_av);
            NVSDK_NGX_Parameter_GetUI(caps, name(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax), &mut max);
            crate::row("ngx", format!("RR available {rr_av}, FG available {fg_av}, MultiFrameCountMax {max}"));

            let mut rr_params = std::ptr::null_mut();
            NVSDK_NGX_VULKAN_AllocateParameters(&mut rr_params);
            let p = rr_params;
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_CreationNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_VisibilityNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Width), render[0]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Height), render[1]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_OutWidth), output[0]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_OutHeight), output[1]);
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_PerfQualityValue), NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf);
            let flags = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR
                | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
                | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags), flags);
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_DLSS_Denoise_Mode), NVSDK_NGX_DLSS_Denoise_Mode_NVSDK_NGX_DLSS_Denoise_Mode_DLUnified);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_DLSS_Roughness_Mode), NVSDK_NGX_DLSS_Roughness_Mode_NVSDK_NGX_DLSS_Roughness_Mode_Packed as u32);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_Use_HW_Depth), 0);
            let vkcmd = cmd.as_raw() as usize as VkCommandBuffer;
            let mut rr = std::ptr::null_mut();
            let r = NVSDK_NGX_VULKAN_CreateFeature(vkcmd, NVSDK_NGX_Feature_NVSDK_NGX_Feature_RayReconstruction, p, &mut rr);
            if !ok(r) {
                return Err(format!("RR create {r:#x}"));
            }

            let mut fg_params = std::ptr::null_mut();
            NVSDK_NGX_VULKAN_AllocateParameters(&mut fg_params);
            let q = fg_params;
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_CreationNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_VisibilityNodeMask), 1);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_Width), output[0]);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_Parameter_Height), output[1]);
            NVSDK_NGX_Parameter_SetUI(q, name(NVSDK_NGX_DLSSG_Parameter_BackbufferFormat), fg_format.as_raw() as u32);
            let mut fg = std::ptr::null_mut();
            let r = NVSDK_NGX_VULKAN_CreateFeature(vkcmd, NVSDK_NGX_Feature_NVSDK_NGX_Feature_FrameGeneration, q, &mut fg);
            let fg_ok = ok(r);
            if !fg_ok {
                crate::row("ngx", format!("FG create failed {r:#x}"));
            }
            let mut mats = Box::new([[0f32; 16]; 4]);
            for m in mats.iter_mut() {
                for i in 0..4 {
                    m[i * 5] = 1.0;
                }
            }
            Ok(Ngx { rr_params, fg_params, rr, fg, fg_ok, res: Vec::new(), mats, render, output })
        }
    }

    /// Records RR. Returns the NGX result.
    pub unsafe fn eval_rr(&mut self, cmd: vk::CommandBuffer, i: &Inputs, frame: u64) -> NVSDK_NGX_Result {
        unsafe {
            self.res.clear();
            for (img, rw) in [(&i.color, false), (&i.rr_out, true), (&i.depth, false), (&i.motion, false), (&i.diffuse, false), (&i.specular, false), (&i.normals, false)] {
                self.res.push(resource(img, rw));
            }
            let r = self.res.as_mut_ptr();
            let p = self.rr_params;
            let set = |n: &[u8], k: usize| NVSDK_NGX_Parameter_SetVoidPointer(p, name(n), r.add(k).cast::<c_void>());
            set(NVSDK_NGX_Parameter_Color, 0);
            set(NVSDK_NGX_Parameter_Output, 1);
            set(NVSDK_NGX_Parameter_Depth, 2);
            set(NVSDK_NGX_Parameter_MotionVectors, 3);
            set(NVSDK_NGX_Parameter_DiffuseAlbedo, 4);
            set(NVSDK_NGX_Parameter_SpecularAlbedo, 5);
            set(NVSDK_NGX_Parameter_GBuffer_Normals, 6);
            let jitter = ((frame % 8) as f32 / 8.0) - 0.5;
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_Jitter_Offset_X), jitter);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_Jitter_Offset_Y), -jitter);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_MV_Scale_X), self.render[0] as f32);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_MV_Scale_Y), self.render[1] as f32);
            NVSDK_NGX_Parameter_SetI(p, name(NVSDK_NGX_Parameter_Reset), 0);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width), self.render[0]);
            NVSDK_NGX_Parameter_SetUI(p, name(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height), self.render[1]);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_DLSS_Pre_Exposure), 1.0);
            NVSDK_NGX_Parameter_SetF(p, name(NVSDK_NGX_Parameter_DLSS_Exposure_Scale), 1.0);
            let m = self.mats.as_mut_ptr();
            NVSDK_NGX_Parameter_SetVoidPointer(p, name(NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX), m.cast());
            NVSDK_NGX_Parameter_SetVoidPointer(p, name(NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX), m.add(1).cast());
            NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd.as_raw() as usize as VkCommandBuffer, self.rr, p, None)
        }
    }

    /// Records FG for `index` of `count` into `out`.
    pub unsafe fn eval_fg(&mut self, cmd: vk::CommandBuffer, backbuffer: &Img, depth: &Img, motion: &Img, out: &Img, index: u32, count: u32, frame: u64) -> NVSDK_NGX_Result {
        unsafe {
            if !self.fg_ok {
                return 0;
            }
            self.res.clear();
            for (img, rw) in [(backbuffer, false), (depth, false), (motion, false), (out, true)] {
                self.res.push(resource(img, rw));
            }
            let r = self.res.as_mut_ptr();
            let q = self.fg_params;
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
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_ClipToLensClip), m.add(2).cast());
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip), m.add(2).cast());
            NVSDK_NGX_Parameter_SetVoidPointer(q, name(NVSDK_NGX_DLSSG_Parameter_PrevClipToClip), m.add(3).cast());
            f(NVSDK_NGX_DLSSG_Parameter_MvecScaleX, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_MvecScaleY, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraPosX, 0.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraUpY, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraRightX, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFwdZ, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraNear, 0.1);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFar, 10000.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraFOV, 1.0);
            f(NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio, self.output[0] as f32 / self.output[1] as f32);
            ui(NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR, 0);
            ui(NVSDK_NGX_DLSSG_Parameter_DepthInverted, 1);
            ui(NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded, 1);
            ui(NVSDK_NGX_DLSSG_Parameter_Reset, 0);
            ui(NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames, 0);
            ui(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectWidth, self.render[0]);
            ui(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectHeight, self.render[1]);
            ui(NVSDK_NGX_DLSSG_Parameter_DepthSubrectWidth, self.render[0]);
            ui(NVSDK_NGX_DLSSG_Parameter_DepthSubrectHeight, self.render[1]);
            NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd.as_raw() as usize as VkCommandBuffer, self.fg, q, None)
        }
    }

    pub unsafe fn shutdown(self, gpu: &Gpu) {
        unsafe {
            NVSDK_NGX_VULKAN_ReleaseFeature(self.rr);
            if self.fg_ok {
                NVSDK_NGX_VULKAN_ReleaseFeature(self.fg);
            }
            NVSDK_NGX_VULKAN_DestroyParameters(self.rr_params);
            NVSDK_NGX_VULKAN_DestroyParameters(self.fg_params);
            NVSDK_NGX_VULKAN_Shutdown1(gpu.device.handle().as_raw() as usize as VkDevice);
        }
    }
}
