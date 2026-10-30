//! The Slang probe corpus, compiled by `orr_shaders::build`. The generated Rust compiling
//! at all is most of the test; the tests below pin the facts the corpus exists for.

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub base_color_tex: u32,
    pub sampler_index: u32,
    pub uv_scale: [u16; 2],
}

#[cfg(test)]
mod tests {
    use core::mem::{offset_of, size_of};

    use orr_shaders::Stage;
    use orr_shaders::abi::SpecEntry;

    use super::shaders::*;

    #[test]
    fn array_member_offsets_come_from_spirv() {
        // Reflection reports 28 here.
        assert_eq!(offset_of!(layout::array::C, after), 32);
        assert_eq!(size_of::<layout::array::A>(), 16);
    }

    #[test]
    fn spec_constants_keep_ids_types_and_defaults() {
        let spec = spec::Spec::default();
        assert_eq!(
            (spec.group_limit, spec.scale, spec.alpha_test, spec.bias),
            (64, 1.5, true, -3)
        );
        assert_eq!(
            spec::Spec::ENTRIES,
            &[
                SpecEntry {
                    constant_id: 0,
                    offset: 0,
                    size: 4
                },
                SpecEntry {
                    constant_id: 1,
                    offset: 4,
                    size: 4
                },
                SpecEntry {
                    constant_id: 2,
                    offset: 8,
                    size: 4
                },
                SpecEntry {
                    constant_id: 5,
                    offset: 12,
                    size: 4
                },
            ]
        );
        let bytes = spec::Spec {
            alpha_test: false,
            ..spec
        }
        .bytes();
        assert_eq!(bytes[8..12], [0; 4]);
        assert_eq!(bytes[12..16], (-3i32).to_le_bytes());
    }

    #[test]
    fn entry_points_and_variants() {
        let stages = |m: &orr_shaders::ShaderModule| -> Vec<(Stage, [u32; 3])> {
            m.entry_points
                .iter()
                .map(|e| (e.stage, e.workgroup))
                .collect()
        };
        assert_eq!(
            stages(&mesh::MODULE),
            [
                (Stage::Task, [1, 1, 1]),
                (Stage::Mesh, [32, 1, 1]),
                (Stage::Pixel, [1, 1, 1])
            ]
        );
        assert_eq!(
            stages(&ser::MODULE),
            [(Stage::RayGen, [1, 1, 1]), (Stage::Compute, [8, 4, 1])]
        );
        // Only modules with a ray query get the opacity-micromap variant.
        assert!(!rq::MODULE.spirv_omm.is_empty() && !ser::MODULE.spirv_omm.is_empty());
        assert!(spec::MODULE.spirv_omm.is_empty() && mesh::MODULE.spirv_omm.is_empty());
        assert_eq!(root::MODULE.spirv[0], 0x0723_0203);
    }
}
