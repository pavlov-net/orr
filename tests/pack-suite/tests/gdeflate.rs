//! GDeflate blobs written by the cook's encoder decompress bit-exact on the loading queue's
//! indirect-count path, one region per chunk as the archive stores them.

use std::sync::Arc;
use std::time::Duration;

use gpu_suite::{assert_clean, device, readback, render, require, upload};
use orr_cook::gdeflate::Libdeflate;
use orr_gpu::{DecompressRegion, Hazards, MemoryKind, QueueKind, Stages};
use orr_pack::{Archive, ArchiveWriter, BlobIn, Class, Codec, GDEFLATE_PAGE, Hash32, Target};

const WAIT: Duration = Duration::from_secs(60);

/// Text, all-zero and incompressible pages in turn, then a short last page.
fn source() -> Vec<u8> {
    let page = GDEFLATE_PAGE as usize;
    let mut x = 11u64;
    let mut random = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    };
    let mut bytes = Vec::new();
    for i in 0..30 {
        match i % 3 {
            0 => bytes.extend((0..page).map(|j| b"orr loads pages "[(j + i) % 16])),
            1 => bytes.resize(bytes.len() + page, 0),
            _ => bytes.extend(random(page)),
        }
    }
    bytes.extend(random(1001));
    bytes
}

/// A memory archive holding `bytes` as one GDeflate blob at `level`.
fn cooked(bytes: &[u8], level: i32) -> Archive {
    let mut w = ArchiveWriter::create(Target::Memory).unwrap();
    w.set_gdeflate(Some(Arc::new(Libdeflate::new(level).unwrap())));
    w.add(
        "pages",
        Class::USER_FIRST,
        1,
        Hash32([0; 32]),
        &[BlobIn::new(bytes)],
    )
    .unwrap();
    Archive::from_memory(w.finish().unwrap().unwrap()).unwrap()
}

#[test]
fn cooked_pages_decompress_on_the_loading_queue() {
    let d = device();
    if !require(
        d.caps().memory_decompression,
        "GDeflate needs memory decompression",
    ) {
        return;
    }
    let source = source();
    let loading = d.queue(QueueKind::Loading).unwrap();
    for level in [1, 12] {
        let a = cooked(&source, level);
        let b = a.blobs(&a.entries()[0])[0];
        assert_eq!(b.codec(), Codec::GDeflate);
        let chunks = a.chunks(&b);
        let n = chunks.len();
        let stored = upload(a.stored(&b, None).unwrap(), "gdeflate input");
        let out = d
            .gpu_malloc::<u8>(MemoryKind::DeviceLocal, source.len() as u64, "gdeflate out")
            .unwrap();
        let mut src = stored.gpu.address().0;
        let regions: Vec<DecompressRegion> = (0..n)
            .map(|i| {
                let region = DecompressRegion {
                    src,
                    dst: out.gpu.address().0 + u64::from(b.chunk_raw_len()) * i as u64,
                    compressed_bytes: u64::from(chunks[i].stored_len),
                    decompressed_bytes: u64::from(b.chunk_len(i as u32)),
                };
                src += region.compressed_bytes;
                region
            })
            .collect();
        let regions = upload(&regions, "gdeflate regions");
        let cleared = upload(&vec![0xcdu8; source.len()], "gdeflate clear");
        let host = readback::<u8>(source.len() as u64, "gdeflate readback");
        // A count below the maximum leaves the regions after it untouched.
        for count in [n as u32, n as u32 - 5] {
            let count_at = upload(&[count], "gdeflate count");
            let mut list = loading.command_list().unwrap();
            list.copy(out.gpu, cleared.gpu, out.size);
            list.barrier(Stages::TRANSFER, Stages::DECOMPRESS, Hazards::empty());
            list.decompress(regions.gpu, count_at.gpu, n as u32);
            list.barrier(Stages::DECOMPRESS, Stages::TRANSFER, Hazards::empty());
            let decompressed = loading.submit(vec![list], &[]).unwrap();
            let mut list = render().command_list().unwrap();
            list.copy(host.gpu, out.gpu, out.size);
            list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
            let read = render().submit(vec![list], &[decompressed]).unwrap();
            render().wait(read, WAIT).unwrap();
            let got = host.as_slice().unwrap();
            let page = b.chunk_raw_len() as usize;
            for i in 0..n {
                let span = i * page..(i * page + page).min(source.len());
                if i < count as usize {
                    assert!(
                        got[span.clone()] == source[span],
                        "level {level}: page {i} of {count} differs"
                    );
                } else {
                    assert!(
                        got[span].iter().all(|&v| v == 0xcd),
                        "level {level}: page {i} past the count of {count} was written"
                    );
                }
            }
        }
    }
    assert_clean();
}
