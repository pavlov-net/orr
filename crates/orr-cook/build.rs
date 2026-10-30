//! Puts the locked versions of ctt and its encoder crates into `ORR_COOK_ENCODERS`, which
//! enters texture cook keys.

use std::path::Path;

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let lock = Path::new(&dir)
        .ancestors()
        .map(|d| d.join("Cargo.lock"))
        .find(|p| p.exists());
    let mut versions = Vec::new();
    if let Some(lock) = lock {
        println!("cargo:rerun-if-changed={}", lock.display());
        let text = std::fs::read_to_string(&lock).expect("Cargo.lock");
        let mut name = None;
        for line in text.lines() {
            if let Some(n) = line.strip_prefix("name = ") {
                name = Some(n.trim_matches('"').to_string());
            } else if let (Some(v), Some(n)) = (line.strip_prefix("version = "), name.take())
                && (n == "ctt" || n.starts_with("ctt-"))
            {
                versions.push(format!("{n} {}", v.trim_matches('"')));
            }
        }
    }
    versions.sort();
    println!("cargo:rustc-env=ORR_COOK_ENCODERS={}", versions.join(", "));
}
