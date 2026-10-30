use std::path::Path;

use orr_shaders::build::Toolchain;

#[test]
fn compiles_a_trivial_kernel() {
    let toolchain = Toolchain::locate().unwrap_or_else(|e| panic!("{e}"));
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/kernels/fill.slang");
    let spv = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fill.spv");
    let output = toolchain
        .command()
        .arg(&src)
        .arg("-o")
        .arg(&spv)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "slangc failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = std::fs::read(&spv).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    assert_eq!(bytes[..4], 0x0723_0203u32.to_le_bytes());
}
