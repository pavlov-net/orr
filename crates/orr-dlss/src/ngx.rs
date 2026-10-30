use crate::raw::{Image, RrInputs, Vulkan};
use crate::sys::*;
use crate::{DlssError, Quality, Support, Uuid};
use ash::vk::{self, Handle, TaggedStructure};
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::Mutex;

static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "C" fn log_callback(
    message: *const c_char,
    _level: NVSDK_NGX_Logging_Level,
    _source: NVSDK_NGX_Feature,
) {
    if message.is_null() {
        return;
    }
    // SAFETY: NGX passes a NUL-terminated string valid for the call.
    let line = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .trim_end()
        .to_owned();
    log::debug!(target: "ngx", "{line}");
    LOG.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(line);
}

pub(crate) fn take_log() -> Vec<String> {
    std::mem::take(
        &mut *LOG
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    )
}

fn check(call: &'static str, result: NVSDK_NGX_Result) -> Result<(), DlssError> {
    if result == NVSDK_NGX_Result_NVSDK_NGX_Result_Success {
        Ok(())
    } else {
        Err(DlssError::Ngx {
            call,
            result: result as u32,
        })
    }
}

fn key(name: &[u8]) -> *const c_char {
    name.as_ptr().cast()
}

fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    s.encode_wide().chain(std::iter::once(0)).collect()
}

/// The strings NGX's init and discovery calls point into.
struct Setup {
    project_id: CString,
    engine_version: CString,
    data_path: Vec<u16>,
    dll_paths: Vec<Vec<u16>>,
    dll_path_ptrs: Vec<*const u16>,
}

impl Setup {
    fn new(project_id: Uuid) -> Self {
        let data = std::env::temp_dir().join("orr-ngx");
        let _ = std::fs::create_dir_all(&data);
        let dll_paths: Vec<Vec<u16>> = dll_dirs().iter().map(|p| wide(p.as_os_str())).collect();
        let dll_path_ptrs = dll_paths.iter().map(Vec::as_ptr).collect();
        Self {
            project_id: CString::new(project_id.hyphenated().to_string()).unwrap(),
            engine_version: CString::new(env!("CARGO_PKG_VERSION")).unwrap(),
            data_path: wide(data.as_os_str()),
            dll_paths,
            dll_path_ptrs,
        }
    }

    fn common_info(&self) -> NVSDK_NGX_FeatureCommonInfo {
        // SAFETY: a plain C struct; zero is its documented default.
        let mut info: NVSDK_NGX_FeatureCommonInfo = unsafe { std::mem::zeroed() };
        info.PathListInfo.Path = self.dll_path_ptrs.as_ptr();
        info.PathListInfo.Length = self.dll_paths.len() as u32;
        info.LoggingInfo.LoggingCallback = Some(log_callback);
        info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_Logging_Level_NVSDK_NGX_LOGGING_LEVEL_ON;
        info.LoggingInfo.DisableOtherLoggingSinks = true;
        info
    }

    fn discovery(
        &self,
        feature: NVSDK_NGX_Feature,
        common: &NVSDK_NGX_FeatureCommonInfo,
    ) -> NVSDK_NGX_FeatureDiscoveryInfo {
        // SAFETY: a plain C struct; every field the call reads is set below.
        let mut info: NVSDK_NGX_FeatureDiscoveryInfo = unsafe { std::mem::zeroed() };
        info.SDKVersion = NVSDK_NGX_Version_NVSDK_NGX_Version_API;
        info.FeatureID = feature;
        info.Identifier.IdentifierType =
            NVSDK_NGX_Application_Identifier_Type_NVSDK_NGX_Application_Identifier_Type_Project_Id;
        info.Identifier.v.ProjectDesc = NVSDK_NGX_ProjectIdDescription {
            ProjectId: self.project_id.as_ptr(),
            EngineType: NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
            EngineVersion: self.engine_version.as_ptr(),
        };
        info.ApplicationDataPath = self.data_path.as_ptr();
        info.FeatureInfo = common;
        info
    }
}

/// Directories NGX searches for its feature DLLs after the executable's own.
fn dll_dirs() -> Vec<PathBuf> {
    let sub = if cfg!(feature = "debug_overlay") {
        "dev"
    } else {
        "rel"
    };
    let mut dirs = Vec::new();
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(PathBuf::from))
    {
        dirs.push(dir);
    }
    #[cfg(debug_assertions)]
    dirs.push(PathBuf::from(env!("ORR_DLSS_BUILD_DLLS")));
    if let Some(sdk) = std::env::var_os("DLSS_SDK").filter(|s| !s.is_empty()) {
        dirs.push(PathBuf::from(sdk).join("lib/Windows_x86_64").join(sub));
    }
    dirs
}

const FEATURES: [(NVSDK_NGX_Feature, &str); 3] = [
    (
        NVSDK_NGX_Feature_NVSDK_NGX_Feature_RayReconstruction,
        "ray reconstruction",
    ),
    (
        NVSDK_NGX_Feature_NVSDK_NGX_Feature_SuperSampling,
        "super resolution",
    ),
    (
        NVSDK_NGX_Feature_NVSDK_NGX_Feature_FrameGeneration,
        "frame generation",
    ),
];

/// Reads `count` extension names NGX returned, adding new ones to `out`.
///
/// # Safety
///
/// `props` points at `count` extension properties.
unsafe fn collect(out: &mut Vec<CString>, props: *const vk::ExtensionProperties, count: u32) {
    if props.is_null() {
        return;
    }
    // SAFETY: forwarded from the caller.
    let props = unsafe { std::slice::from_raw_parts(props, count as usize) };
    for p in props {
        if let Ok(name) = p.extension_name_as_c_str()
            && !out.iter().any(|e| e.as_c_str() == name)
        {
            out.push(name.to_owned());
        }
    }
}

pub(crate) fn instance_extensions(project_id: Uuid) -> Result<Vec<CString>, DlssError> {
    let setup = Setup::new(project_id);
    let common = setup.common_info();
    let mut out = Vec::new();
    for (feature, name) in FEATURES {
        let info = setup.discovery(feature, &common);
        let (mut count, mut props) = (0u32, ptr::null_mut());
        // SAFETY: `info` and the strings it points at outlive the call.
        let r = unsafe {
            NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &mut count, &mut props)
        };
        if check(
            "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements",
            r,
        )
        .is_err()
        {
            log::info!("NGX lists no instance extensions for {name}: result {r:#x}");
            continue;
        }
        // SAFETY: NGX returned `count` properties at `props`.
        unsafe { collect(&mut out, props.cast(), count) };
    }
    Ok(out)
}

/// Device extension lists by device UUID. NGX's discovery calls return
/// `FAIL_NotImplemented` once NGX has been shut down in the process, so a device recreated
/// later reuses the first answer.
static DEVICE_EXTENSIONS: Mutex<Vec<([u8; 16], Vec<CString>)>> = Mutex::new(Vec::new());

pub(crate) unsafe fn device_extensions(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    project_id: Uuid,
) -> Result<Vec<CString>, DlssError> {
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push(&mut id);
    // SAFETY: a query on a live physical device (caller).
    unsafe { instance.get_physical_device_properties2(physical_device, &mut props) };
    let uuid = id.device_uuid;
    let mut cache = DEVICE_EXTENSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, list)) = cache.iter().find(|(u, _)| *u == uuid) {
        return Ok(list.clone());
    }

    let setup = Setup::new(project_id);
    let common = setup.common_info();
    let raw_instance = instance.handle().as_raw() as usize as VkInstance;
    let raw_physical = physical_device.as_raw() as usize as VkPhysicalDevice;
    let mut out = Vec::new();
    for (feature, name) in FEATURES {
        let info = setup.discovery(feature, &common);
        // SAFETY: a plain C struct NGX fills.
        let mut req: NVSDK_NGX_FeatureRequirement = unsafe { std::mem::zeroed() };
        // SAFETY: the handles are live (caller), `info` outlives the call.
        let r = unsafe {
            NVSDK_NGX_VULKAN_GetFeatureRequirements(raw_instance, raw_physical, &info, &mut req)
        };
        check("NVSDK_NGX_VULKAN_GetFeatureRequirements", r)?;
        if req.FeatureSupported != 0 {
            log::info!(
                "NGX {name} unavailable: reasons {:#x}",
                req.FeatureSupported
            );
            continue;
        }
        let (mut count, mut props) = (0u32, ptr::null_mut());
        // SAFETY: as above.
        let r = unsafe {
            NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                raw_instance,
                raw_physical,
                &info,
                &mut count,
                &mut props,
            )
        };
        check("NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements", r)?;
        // SAFETY: NGX returned `count` properties at `props`.
        unsafe { collect(&mut out, props.cast(), count) };
    }
    cache.push((uuid, out.clone()));
    Ok(out)
}

pub(crate) struct Ngx {
    device: vk::Device,
    caps: *mut NVSDK_NGX_Parameter,
    _setup: Setup,
}

impl Ngx {
    pub(crate) unsafe fn new(vulkan: Vulkan<'_>, project_id: Uuid) -> Result<Self, DlssError> {
        let setup = Setup::new(project_id);
        let common = setup.common_info();
        // SAFETY: ash's function pointers have the C signatures NGX's typedefs name.
        let (gipa, gdpa): (PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr) = unsafe {
            (
                std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, PFN_vkGetInstanceProcAddr>(
                    vulkan.entry.static_fn().get_instance_proc_addr,
                ),
                std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, PFN_vkGetDeviceProcAddr>(
                    vulkan.instance.fp_v1_0().get_device_proc_addr,
                ),
            )
        };
        let device = vulkan.device.handle();
        // SAFETY: live handles (caller); the strings outlive the call and `setup` is kept.
        let r = unsafe {
            NVSDK_NGX_VULKAN_Init_with_ProjectID(
                setup.project_id.as_ptr(),
                NVSDK_NGX_EngineType_NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                setup.engine_version.as_ptr(),
                setup.data_path.as_ptr(),
                vulkan.instance.handle().as_raw() as usize as VkInstance,
                vulkan.physical_device.as_raw() as usize as VkPhysicalDevice,
                device.as_raw() as usize as VkDevice,
                gipa,
                gdpa,
                &common,
                NVSDK_NGX_Version_NVSDK_NGX_Version_API,
            )
        };
        check("NVSDK_NGX_VULKAN_Init_with_ProjectID", r)?;
        let mut caps = ptr::null_mut();
        // SAFETY: NGX is initialised.
        let r = unsafe { NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut caps) };
        let this = Self {
            device,
            caps,
            _setup: setup,
        };
        check("NVSDK_NGX_VULKAN_GetCapabilityParameters", r)?;
        Ok(this)
    }

    fn get_i(&self, name: &[u8]) -> i32 {
        let mut v = 0;
        // SAFETY: `caps` is NGX's capability map; a missing key leaves `v` unchanged.
        unsafe { NVSDK_NGX_Parameter_GetI(self.caps, key(name), &mut v) };
        v
    }

    fn get_ui(&self, name: &[u8]) -> u32 {
        let mut v = 0;
        // SAFETY: as in `get_i`.
        unsafe { NVSDK_NGX_Parameter_GetUI(self.caps, key(name), &mut v) };
        v
    }

    /// Calls a callback NGX stores in its capability map, with the map as its parameters.
    fn callback(&self, name: &[u8]) -> Option<NVSDK_NGX_Result> {
        let mut f: *mut c_void = ptr::null_mut();
        // SAFETY: as in `get_i`.
        unsafe { NVSDK_NGX_Parameter_GetVoidPointer(self.caps, key(name), &mut f) };
        if f.is_null() {
            return None;
        }
        // SAFETY: NGX stores these callbacks as `PFN_NVSDK_NGX_DLSS_GetStatsCallback` /
        // `GetOptimalSettingsCallback`, which take the parameter map.
        let f: unsafe extern "C" fn(*mut NVSDK_NGX_Parameter) -> NVSDK_NGX_Result =
            unsafe { std::mem::transmute(f) };
        // SAFETY: see above.
        Some(unsafe { f(self.caps) })
    }

    pub(crate) fn support(&self) -> Support {
        let fg = self.get_i(NVSDK_NGX_Parameter_FrameGeneration_Available) != 0;
        Support {
            rr: self.get_i(NVSDK_NGX_Parameter_SuperSamplingDenoising_Available) != 0,
            sr: self.get_i(NVSDK_NGX_Parameter_SuperSampling_Available) != 0,
            fg,
            max_generated: if fg {
                self.get_ui(NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax)
                    .max(1)
            } else {
                0
            },
        }
    }

    pub(crate) fn vram_bytes(&self) -> u64 {
        let mut total = 0;
        for name in [
            &NVSDK_NGX_Parameter_DLSSDGetStatsCallback[..],
            NVSDK_NGX_Parameter_DLSSGetStatsCallback,
        ] {
            if self.callback(name) == Some(NVSDK_NGX_Result_NVSDK_NGX_Result_Success) {
                let mut v = 0u64;
                // SAFETY: as in `get_i`.
                unsafe {
                    NVSDK_NGX_Parameter_GetULL(
                        self.caps,
                        key(NVSDK_NGX_Parameter_SizeInBytes),
                        &mut v,
                    )
                };
                total += v;
            }
        }
        total
    }

    pub(crate) fn render_size(
        &self,
        output: [u32; 2],
        quality: Quality,
    ) -> Result<[u32; 2], DlssError> {
        if quality == Quality::Dlaa {
            return Ok(output);
        }
        if !self.support().rr {
            return Err(DlssError::Unsupported("ray reconstruction"));
        }
        // SAFETY: as in `get_i`; the optimal-settings callback reads these keys.
        unsafe {
            NVSDK_NGX_Parameter_SetUI(self.caps, key(NVSDK_NGX_Parameter_Width), output[0]);
            NVSDK_NGX_Parameter_SetUI(self.caps, key(NVSDK_NGX_Parameter_Height), output[1]);
            NVSDK_NGX_Parameter_SetI(
                self.caps,
                key(NVSDK_NGX_Parameter_PerfQualityValue),
                perf_quality(quality),
            );
            NVSDK_NGX_Parameter_SetI(self.caps, key(NVSDK_NGX_Parameter_RTXValue), 0);
        }
        let r = self
            .callback(NVSDK_NGX_Parameter_DLSSDOptimalSettingsCallback)
            .ok_or(DlssError::Unsupported("ray reconstruction"))?;
        check("DLSSDOptimalSettingsCallback", r)?;
        Ok([
            self.get_ui(NVSDK_NGX_Parameter_OutWidth),
            self.get_ui(NVSDK_NGX_Parameter_OutHeight),
        ])
    }
}

impl Drop for Ngx {
    fn drop(&mut self) {
        // SAFETY: the caller of `new` keeps the device alive past this and drops every
        // feature first.
        unsafe {
            NVSDK_NGX_VULKAN_DestroyParameters(self.caps);
            NVSDK_NGX_VULKAN_Shutdown1(self.device.as_raw() as usize as VkDevice);
        }
    }
}

fn perf_quality(quality: Quality) -> NVSDK_NGX_PerfQuality_Value {
    match quality {
        Quality::Dlaa => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_DLAA,
        Quality::Quality => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxQuality,
        Quality::Balanced => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_Balanced,
        Quality::Performance => NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_MaxPerf,
        Quality::UltraPerformance => {
            NVSDK_NGX_PerfQuality_Value_NVSDK_NGX_PerfQuality_Value_UltraPerformance
        }
    }
}

fn preset_key(quality: Quality) -> &'static [u8] {
    match quality {
        Quality::Dlaa => NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA,
        Quality::Quality => NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality,
        Quality::Balanced => NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
        Quality::Performance => {
            NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance
        }
        Quality::UltraPerformance => {
            NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance
        }
    }
}

fn resource(image: &Image, read_write: bool) -> NVSDK_NGX_Resource_VK {
    // SAFETY: a plain C struct; the fields NGX reads are set below.
    let mut r: NVSDK_NGX_Resource_VK = unsafe { std::mem::zeroed() };
    r.Resource.ImageViewInfo = NVSDK_NGX_ImageViewInfo_VK {
        ImageView: image.view.as_raw() as usize as VkImageView,
        Image: image.image.as_raw() as usize as VkImage,
        SubresourceRange: VkImageSubresourceRange {
            aspectMask: vk::ImageAspectFlags::COLOR.as_raw(),
            baseMipLevel: 0,
            levelCount: 1,
            baseArrayLayer: 0,
            layerCount: 1,
        },
        Format: image.format.as_raw(),
        Width: image.extent[0],
        Height: image.extent[1],
    };
    r.Type = NVSDK_NGX_Resource_VK_Type_NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    r.ReadWrite = read_write;
    r
}

pub(crate) struct Rr {
    handle: *mut NVSDK_NGX_Handle,
    params: *mut NVSDK_NGX_Parameter,
    resources: [NVSDK_NGX_Resource_VK; SLOTS],
}

const SLOTS: usize = 11;

impl Rr {
    pub(crate) unsafe fn new(
        ngx: &Ngx,
        cmd: vk::CommandBuffer,
        render: [u32; 2],
        output: [u32; 2],
        quality: Quality,
        preset: Option<u32>,
    ) -> Result<Self, DlssError> {
        if !ngx.support().rr {
            return Err(DlssError::Unsupported("ray reconstruction"));
        }
        let mut params = ptr::null_mut();
        // SAFETY: NGX is initialised.
        check("NVSDK_NGX_VULKAN_AllocateParameters", unsafe {
            NVSDK_NGX_VULKAN_AllocateParameters(&mut params)
        })?;
        let flags = NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_IsHDR
            | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_MVLowRes
            | NVSDK_NGX_DLSS_Feature_Flags_NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
        let mut handle = ptr::null_mut();
        // SAFETY: `params` is a live map; `cmd` is recording on NGX's device (caller).
        let r = unsafe {
            let ui = |name: &[u8], v: u32| NVSDK_NGX_Parameter_SetUI(params, key(name), v);
            let i = |name: &[u8], v: i32| NVSDK_NGX_Parameter_SetI(params, key(name), v);
            ui(NVSDK_NGX_Parameter_CreationNodeMask, 1);
            ui(NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
            ui(NVSDK_NGX_Parameter_Width, render[0]);
            ui(NVSDK_NGX_Parameter_Height, render[1]);
            ui(NVSDK_NGX_Parameter_OutWidth, output[0]);
            ui(NVSDK_NGX_Parameter_OutHeight, output[1]);
            i(NVSDK_NGX_Parameter_PerfQualityValue, perf_quality(quality));
            i(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);
            i(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
            i(
                NVSDK_NGX_Parameter_DLSS_Denoise_Mode,
                NVSDK_NGX_DLSS_Denoise_Mode_NVSDK_NGX_DLSS_Denoise_Mode_DLUnified,
            );
            ui(
                NVSDK_NGX_Parameter_DLSS_Roughness_Mode,
                NVSDK_NGX_DLSS_Roughness_Mode_NVSDK_NGX_DLSS_Roughness_Mode_Packed as u32,
            );
            ui(
                NVSDK_NGX_Parameter_Use_HW_Depth,
                NVSDK_NGX_DLSS_Depth_Type_NVSDK_NGX_DLSS_Depth_Type_HW as u32,
            );
            if let Some(preset) = preset {
                ui(preset_key(quality), preset);
            }
            NVSDK_NGX_VULKAN_CreateFeature(
                cmd.as_raw() as usize as VkCommandBuffer,
                NVSDK_NGX_Feature_NVSDK_NGX_Feature_RayReconstruction,
                params,
                &mut handle,
            )
        };
        if let Err(e) = check("NVSDK_NGX_VULKAN_CreateFeature", r) {
            // SAFETY: allocated above and not shared.
            unsafe { NVSDK_NGX_VULKAN_DestroyParameters(params) };
            return Err(e);
        }
        Ok(Self {
            handle,
            params,
            // SAFETY: plain C structs; each is written before NGX reads it.
            resources: unsafe { std::mem::zeroed() },
        })
    }

    pub(crate) unsafe fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        inputs: &RrInputs<'_>,
        render: [u32; 2],
    ) -> Result<(), DlssError> {
        let slots: [(&[u8], Option<&Image>, bool); SLOTS] = [
            (NVSDK_NGX_Parameter_Color, Some(inputs.color), false),
            (NVSDK_NGX_Parameter_Output, Some(inputs.output), true),
            (NVSDK_NGX_Parameter_Depth, Some(inputs.depth), false),
            (
                NVSDK_NGX_Parameter_MotionVectors,
                Some(inputs.motion),
                false,
            ),
            (
                NVSDK_NGX_Parameter_DiffuseAlbedo,
                Some(inputs.diffuse_albedo),
                false,
            ),
            (
                NVSDK_NGX_Parameter_SpecularAlbedo,
                Some(inputs.specular_albedo),
                false,
            ),
            (
                NVSDK_NGX_Parameter_GBuffer_Normals,
                Some(inputs.normal_roughness),
                false,
            ),
            (
                NVSDK_NGX_Parameter_GBuffer_SpecularMvec,
                Some(inputs.specular_motion),
                false,
            ),
            (
                NVSDK_NGX_Parameter_DLSS_TransparencyLayer,
                inputs.transparency,
                false,
            ),
            (
                NVSDK_NGX_Parameter_DLSSD_ColorBeforeTransparency,
                inputs.color_before_transparency,
                false,
            ),
            (
                NVSDK_NGX_Parameter_DLSSD_DepthOfFieldGuide,
                inputs.depth_of_field_guide,
                false,
            ),
        ];
        let p = self.params;
        // SAFETY: `params` is this feature's map; the resource array lives in `self` and is not
        // touched until the next call; `cmd` is recording on NGX's device (caller).
        let r = unsafe {
            for (slot, &(name, image, rw)) in self.resources.iter_mut().zip(&slots) {
                let value = match image {
                    Some(image) => {
                        *slot = resource(image, rw);
                        ptr::from_mut(slot).cast::<c_void>()
                    }
                    None => ptr::null_mut(),
                };
                NVSDK_NGX_Parameter_SetVoidPointer(p, key(name), value);
            }
            NVSDK_NGX_Parameter_SetF(
                p,
                key(NVSDK_NGX_Parameter_Jitter_Offset_X),
                inputs.jitter[0],
            );
            NVSDK_NGX_Parameter_SetF(
                p,
                key(NVSDK_NGX_Parameter_Jitter_Offset_Y),
                inputs.jitter[1],
            );
            NVSDK_NGX_Parameter_SetF(p, key(NVSDK_NGX_Parameter_MV_Scale_X), render[0] as f32);
            NVSDK_NGX_Parameter_SetF(p, key(NVSDK_NGX_Parameter_MV_Scale_Y), render[1] as f32);
            NVSDK_NGX_Parameter_SetI(p, key(NVSDK_NGX_Parameter_Reset), i32::from(inputs.reset));
            NVSDK_NGX_Parameter_SetUI(
                p,
                key(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width),
                render[0],
            );
            NVSDK_NGX_Parameter_SetUI(
                p,
                key(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height),
                render[1],
            );
            NVSDK_NGX_VULKAN_EvaluateFeature_C(
                cmd.as_raw() as usize as VkCommandBuffer,
                self.handle,
                p,
                None,
            )
        };
        check("NVSDK_NGX_VULKAN_EvaluateFeature_C", r)
    }
}

impl Drop for Rr {
    fn drop(&mut self) {
        // SAFETY: the caller of `new` drops the feature after the GPU is done with it.
        unsafe {
            NVSDK_NGX_VULKAN_ReleaseFeature(self.handle);
            NVSDK_NGX_VULKAN_DestroyParameters(self.params);
        }
    }
}
