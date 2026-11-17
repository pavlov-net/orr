/// Why a MetalFX effect could not be made or recorded.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The device runs no Metal 4 variant of the effect.
    #[error("this device runs no Metal 4 MetalFX {0}")]
    Unsupported(&'static str),
    /// A render size outside the effect's input scale range on this device.
    #[error(
        "render {render:?} for output {output:?} is outside MetalFX's scale range {min}..={max}"
    )]
    Scale {
        render: [u32; 2],
        output: [u32; 2],
        min: f32,
        max: f32,
    },
    /// MetalFX returned no effect for the descriptor, as for formats it does not take.
    #[error("MetalFX made no {effect} for formats {formats}")]
    Create {
        effect: &'static str,
        formats: String,
    },
    /// An input texture lacks usage the effect needs.
    #[error("the {input} texture has usage {have:?}, and MetalFX needs {need:?}")]
    Usage {
        input: &'static str,
        have: objc2_metal::MTLTextureUsage,
        need: objc2_metal::MTLTextureUsage,
    },
}
