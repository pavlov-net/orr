//! Downloading a scene published online into a directory of the run's own.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Downloads that run at once.
const CONNECTIONS: usize = 8;

/// A directory of this run's own, removed when it drops, however the run ends.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            log::warn!("could not remove {}: {e}", self.0.display());
        }
    }
}

/// Downloads `{base}{name}` to `dir/name` for each of `names`, a few at a time, and returns the
/// files' paths in the order of `names`.
pub fn download_all<S: AsRef<str> + Sync>(base: &str, names: &[S], dir: &Path) -> Vec<PathBuf> {
    let started = Instant::now();
    let agent = ureq::Agent::new_with_defaults();
    let paths: Vec<PathBuf> = names.iter().map(|n| dir.join(n.as_ref())).collect();
    let next = AtomicUsize::new(0);
    let bytes: u64 = std::thread::scope(|s| {
        let workers: Vec<_> = (0..CONNECTIONS.min(names.len()))
            .map(|_| {
                s.spawn(|| {
                    let mut bytes = 0;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(name) = names.get(i) else {
                            return bytes;
                        };
                        bytes += download(&agent, &format!("{base}{}", name.as_ref()), &paths[i]);
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("download"))
            .sum()
    });
    log::info!(
        "downloaded {} files from {base}, {:.1} MB in {:.1} s",
        names.len(),
        bytes as f64 / 1e6,
        started.elapsed().as_secs_f64()
    );
    paths
}

/// Downloads `url` to `path` and returns its size in bytes.
fn download(agent: &ureq::Agent, url: &str, path: &Path) -> u64 {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }
    let mut response = agent
        .get(url)
        .call()
        .unwrap_or_else(|e| panic!("{url}: {e}"));
    let mut file =
        std::fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    std::io::copy(&mut response.body_mut().as_reader(), &mut file)
        .unwrap_or_else(|e| panic!("{url}: {e}"))
}
