use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;

/// Counts validation warnings and errors and logs each one.
#[derive(Default)]
pub(crate) struct MessageCount(AtomicU32);

impl MessageCount {
    pub(crate) fn get(&self) -> u32 {
        self.0.load(Ordering::Relaxed)
    }
}

/// The messenger's create info; `count` must outlive the messenger.
pub(crate) fn messenger_info(count: &MessageCount) -> vk::DebugUtilsMessengerCreateInfoEXT<'_> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(callback))
        .user_data(std::ptr::from_ref(count).cast_mut().cast())
}

unsafe extern "system" fn callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes valid callback data for the duration of the call.
    let data = unsafe { &*data };
    let text = |p: *const std::ffi::c_char| {
        if p.is_null() {
            String::new()
        } else {
            // SAFETY: non-null strings in the callback data are NUL-terminated.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    };
    let id = text(data.p_message_id_name);
    let message = text(data.p_message);
    // General messages come from the loader (layer versions, environment overrides); the
    // settings ones report how the layer itself is configured (GPU-AV runs).
    if types == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        || id == "VALIDATION-SETTINGS"
        || id.starts_with("WARNING-Setting-")
    {
        log::debug!("{id}: {message}");
        return vk::FALSE;
    }
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("{id}: {message}");
    } else {
        log::warn!("{id}: {message}");
    }
    // SAFETY: `user` is the `MessageCount` the messenger was created with, which outlives it.
    unsafe { &*user.cast::<MessageCount>() }
        .0
        .fetch_add(1, Ordering::Relaxed);
    vk::FALSE
}

pub(crate) fn name_object<H: vk::Handle>(
    debug: &ash::ext::debug_utils::Device,
    handle: H,
    name: &str,
) {
    let Ok(name) = CString::new(name) else { return };
    let info = vk::DebugUtilsObjectNameInfoEXT::default()
        .object_handle(handle)
        .object_name(&name);
    // SAFETY: the handle is a live object of the device the loader table belongs to.
    let _ = unsafe { debug.set_debug_utils_object_name(&info) };
}
