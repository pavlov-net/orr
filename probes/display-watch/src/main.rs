#![allow(non_snake_case)]
use std::ffi::c_void;
use windows::core::*;
use windows::Graphics::Display::*;
use windows::Foundation::TypedEventHandler;
use windows::Win32::Devices::Display::*;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::WinRT::*;
use windows::Win32::UI::ColorSystem::*;
use windows::Win32::UI::WindowsAndMessaging::*;

#[windows_core::interface("7449121C-382B-4705-8DA7-A795BA482013")]
unsafe trait IDisplayInformationStaticsInterop: windows_core::IUnknown {
    fn GetIids(&self, n: *mut u32, iids: *mut *mut GUID) -> HRESULT;
    fn GetRuntimeClassName(&self, name: *mut *mut c_void) -> HRESULT;
    fn GetTrustLevel(&self, level: *mut i32) -> HRESULT;
    fn GetForWindow(&self, window: HWND, riid: *const GUID, info: *mut *mut c_void) -> HRESULT;
    fn GetForMonitor(&self, monitor: HMONITOR, riid: *const GUID, info: *mut *mut c_void) -> HRESULT;
}

fn wstr(a: &[u16]) -> String { String::from_utf16_lossy(&a[..a.iter().position(|&c| c == 0).unwrap_or(a.len())]) }
fn t() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64() % 1000.0 }

static mut RECTS: Vec<RECT> = Vec::new();
fn dxgi() {
    unsafe {
        let f: IDXGIFactory1 = CreateDXGIFactory1().unwrap();
        let mut i = 0;
        while let Ok(a) = f.EnumAdapters1(i) {
            let ad = a.GetDesc1().unwrap();
            let mut j = 0;
            while let Ok(o) = a.EnumOutputs(j) {
                if let Ok(o6) = o.cast::<IDXGIOutput6>() {
                    let d = o6.GetDesc1().unwrap();
                    #[allow(static_mut_refs)] if RECTS.len() < 2 { RECTS.push(d.DesktopCoordinates); }
                    println!("dxgi adapter={} output={} cs={} min={} max={} maxff={} r={:?} g={:?} b={:?} w={:?}",
                        wstr(&ad.Description), wstr(&d.DeviceName), d.ColorSpace.0, d.MinLuminance, d.MaxLuminance,
                        d.MaxFullFrameLuminance, d.RedPrimary, d.GreenPrimary, d.BluePrimary, d.WhitePoint);
                }
                j += 1;
            }
            i += 1;
        }
    }
}

fn displayconfig() {
    unsafe {
        let (mut np, mut nm) = (0u32, 0u32);
        GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut np, &mut nm).ok().unwrap();
        let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); np as usize];
        let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); nm as usize];
        QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &mut np, paths.as_mut_ptr(), &mut nm, modes.as_mut_ptr(), None).ok().unwrap();
        for p in &paths[..np as usize] {
            let mut src = DISPLAYCONFIG_SOURCE_DEVICE_NAME::default();
            src.header = DISPLAYCONFIG_DEVICE_INFO_HEADER { r#type: DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, size: size_of_val(&src) as u32, adapterId: p.sourceInfo.adapterId, id: p.sourceInfo.id };
            DisplayConfigGetDeviceInfo(&mut src.header);
            let mut tgt = DISPLAYCONFIG_TARGET_DEVICE_NAME::default();
            tgt.header = DISPLAYCONFIG_DEVICE_INFO_HEADER { r#type: DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME, size: size_of_val(&tgt) as u32, adapterId: p.targetInfo.adapterId, id: p.targetInfo.id };
            DisplayConfigGetDeviceInfo(&mut tgt.header);
            let mut wl = DISPLAYCONFIG_SDR_WHITE_LEVEL::default();
            wl.header = DISPLAYCONFIG_DEVICE_INFO_HEADER { r#type: DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL, size: size_of_val(&wl) as u32, adapterId: p.targetInfo.adapterId, id: p.targetInfo.id };
            DisplayConfigGetDeviceInfo(&mut wl.header);
            let scope = ColorProfileGetDisplayUserScope(p.sourceInfo.adapterId, p.sourceInfo.id).unwrap_or(WCS_PROFILE_MANAGEMENT_SCOPE_SYSTEM_WIDE);
            let prof = |sub| match ColorProfileGetDisplayDefault(scope, p.sourceInfo.adapterId, p.sourceInfo.id, CPT_ICC, sub) {
                Ok(s) => s.to_string().unwrap_or_default(), Err(e) => format!("err {e}") };
            println!("dcfg gdi={} monitor={} sdr_white_nits={} scope={} extended_profile={:?} standard_profile={:?}",
                wstr(&src.viewGdiDeviceName), wstr(&tgt.monitorFriendlyDeviceName), wl.SDRWhiteLevel as f32 * 80.0 / 1000.0,
                scope.0, prof(CPST_EXTENDED_DISPLAY_COLOR_MODE), prof(CPST_STANDARD_DISPLAY_COLOR_MODE));
        }
    }
}

fn aci(tag: &str, d: &DisplayInformation) {
    let a = d.GetAdvancedColorInfo().unwrap();
    println!("{tag} t={:.2} kind={:?} min={} max={} maxff={} sdr_white={} r={:?} w={:?}", t(), a.CurrentAdvancedColorKind().unwrap(),
        a.MinLuminanceInNits().unwrap(), a.MaxLuminanceInNits().unwrap(), a.MaxAverageFullFrameLuminanceInNits().unwrap(),
        a.SdrWhiteLevelInNits().unwrap(), a.RedPrimary().unwrap(), a.WhitePoint().unwrap());
}

extern "system" fn wndproc(h: HWND, m: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    unsafe {
        match m {
            WM_TIMER if w.0 == 2 || w.0 == 3 => {
                #[allow(static_mut_refs)] let r = RECTS[if w.0 == 2 { 1 } else { 0 }];
                println!("move t={:.2} to {:?}", t(), (r.left, r.top));
                let _ = SetWindowPos(h, None, r.left + 8, r.top + 8, 32, 32, SWP_NOACTIVATE | SWP_NOZORDER);
                if w.0 == 2 { SetTimer(Some(h), 3, 1500, None); } let _ = KillTimer(Some(h), w.0); LRESULT(0) }
            WM_TIMER => { let _ = PostMessageW(Some(h), WM_CLOSE, WPARAM(0), LPARAM(0)); LRESULT(0) }
            WM_DESTROY => { PostQuitMessage(0); LRESULT(0) }
            _ => DefWindowProcW(h, m, w, l),
        }
    }
}

fn main() -> Result<()> {
    let secs: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    dxgi();
    displayconfig();
    unsafe {
        RoInitialize(RO_INIT_SINGLETHREADED)?;
        let _dq = CreateDispatcherQueueController(DispatcherQueueOptions { dwSize: size_of::<DispatcherQueueOptions>() as u32, threadType: DQTYPE_THREAD_CURRENT, apartmentType: DQTAT_COM_NONE })?;
        let inst = GetModuleHandleW(None)?;
        let cls = w!("orr_display_probe");
        RegisterClassW(&WNDCLASSW { lpfnWndProc: Some(wndproc), hInstance: inst.into(), lpszClassName: cls, ..Default::default() });
        let h = CreateWindowExW(WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, cls, w!("probe"), WS_POPUP | WS_VISIBLE, 0, 0, 32, 32, None, None, Some(inst.into()), None)?;
        let f = windows_core::factory::<DisplayInformation, IDisplayInformationStaticsInterop>()?;
        let mut raw = std::ptr::null_mut();
        f.GetForWindow(h, &<DisplayInformation as Interface>::IID, &mut raw).ok()?;
        let di = DisplayInformation::from_raw(raw);
        aci("aci initial", &di);
        di.AdvancedColorInfoChanged(&TypedEventHandler::new(|s: Ref<DisplayInformation>, _| { aci("aci EVENT", s.as_ref().unwrap()); Ok(()) }))?;
        di.ColorProfileChanged(&TypedEventHandler::new(|_: Ref<DisplayInformation>, _| { println!("colorprofile EVENT t={:.2}", t()); Ok(()) }))?;
        SetTimer(Some(h), 1, secs * 1000, None);
        SetTimer(Some(h), 2, 1000, None);
        println!("pumping {secs}s");
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() { let _ = TranslateMessage(&msg); DispatchMessageW(&msg); }
        aci("aci final", &di);
    }
    dxgi();
    Ok(())
}
