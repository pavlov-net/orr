//! The debug check rejects a PTLAS build with two commands of one op type, which the driver
//! would apply only in part. In its own process: the rejection counts as a validation
//! message on the shared device.

use gpu_suite::clusters::*;
use gpu_suite::device;

#[test]
#[cfg(debug_assertions)]
fn ptlas_build_with_a_repeated_op_is_rejected() {
    let clusters = base_clusters();
    let mut b = Builder::new();
    let clas = b.build_clas(&clusters, data_flags(), 0, None, None);
    let blas = b.build_blas(clas.addrs.gpu, &[vec![0, 1, 2, 3], vec![4]], None);
    let limits = ptlas_limits(2);
    let (dst, scratch) = b.ptlas_blocks(limits);
    let writes = |front_x: f32| PtlasOps {
        writes: vec![
            ptlas_write(0, 0, FRONT_CUSTOM, [front_x, 0.0, FRONT_Z], 0, false),
            ptlas_write(1, 1, BACK_CUSTOM, [0.0; 3], 1, false),
        ],
        ..Default::default()
    };
    b.build_ptlas(limits, blas.addrs.gpu, &writes(0.0), dst, scratch, false);
    b.flush();
    assert_eq!(device().validation_messages(), 0, "a valid build passes");

    let mut repeated = writes(0.25);
    repeated.repeat_first = true;
    b.build_ptlas(limits, blas.addrs.gpu, &repeated, dst, scratch, true);
    let out = b.trace(orr_gpu::Tlas(dst.0), 0);
    b.flush();
    assert_eq!(
        device().validation_messages(),
        1,
        "the repeated write is rejected"
    );
    let unchanged = [
        ref_inst(
            FRONT_CUSTOM,
            0,
            [0.0, 0.0, FRONT_Z],
            &clusters[..4].iter().collect::<Vec<_>>(),
        ),
        ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[&clusters[4]]),
    ];
    check(out.as_slice().unwrap(), &unchanged, 0).assert_ok("the rejected build changed nothing");
}
