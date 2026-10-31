#![allow(clippy::print_stdout)] // the bench reports measured figures

//! GPU page decode rate on the loading queue: every mesh page of the `.orr` archives under
//! `ORR_COOKED_SET`, decoded through orr_pack.page_codec in batches of 64 pages. Prints
//! resident GB/s over the copy and decode passes' GPU time (median of 5 runs per batch, into
//! host-visible memory).

use orr_pack::mesh::PageHeader;
use pack_suite::{cooked_set_pages, decode_stored};

const BATCH: usize = 64;
const RUNS: usize = 5;

fn main() {
    let Ok(dir) = std::env::var("ORR_COOKED_SET") else {
        println!("set ORR_COOKED_SET to a directory of cooked archives");
        return;
    };
    let set = cooked_set_pages(dir.as_ref());
    let pages: Vec<Vec<u8>> = set.into_iter().flat_map(|(_, p)| p).collect();
    let resident: u64 = pages
        .iter()
        .map(|p| {
            u64::from(bytemuck::pod_read_unaligned::<PageHeader>(&p[..size_of::<PageHeader>()]).len)
        })
        .sum();
    let mut ms = 0.0;
    for batch in pages.chunks(BATCH) {
        let mut runs: Vec<f64> = (0..RUNS).map(|_| decode_stored(batch).gpu_ms).collect();
        runs.sort_by(f64::total_cmp);
        ms += runs[RUNS / 2];
    }
    println!(
        "{} pages, {:.1} MB resident: {ms:.2} ms, {:.2} GB/s",
        pages.len(),
        resident as f64 / 1e6,
        resident as f64 / (ms * 1e6)
    );
}
