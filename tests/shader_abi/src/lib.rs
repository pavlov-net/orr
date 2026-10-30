//! Kernels over `orr.abi`, `orr.trace`, `orr.coherence` and `orr.atomic`, built with the MSL
//! check: the build passing is the check. The kernels under `asserts` link `ASSERTS = true`.

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

pub mod assert_shaders {
    include!(concat!(env!("OUT_DIR"), "/asserts/orr_shaders.rs"));
}

#[cfg(test)]
mod tests {
    use core::mem::{offset_of, size_of};

    use super::{assert_shaders, shaders};

    const OP_ABORT: u32 = 5121;

    fn aborts(spirv: &[u32]) -> bool {
        let mut at = 5;
        while let Some(&first) = spirv.get(at) {
            if first & 0xffff == OP_ABORT {
                return true;
            }
            at += (first >> 16).max(1) as usize;
        }
        false
    }

    #[test]
    fn asserts_abort_only_when_linked() {
        assert!(aborts(assert_shaders::null_tlas::MODULE.spirv));
        assert!(!aborts(shaders::alpha::MODULE.spirv));
        assert!(!aborts(shaders::clusters::MODULE.spirv));
    }

    #[test]
    fn records_keep_c_layout() {
        use shaders::clusters::ClusterHit;
        assert_eq!(offset_of!(ClusterHit, positions), 16);
        assert_eq!(size_of::<ClusterHit>(), 52);
    }
}
