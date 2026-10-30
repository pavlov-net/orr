//! The loading side on orr-gpu: sparse texture tiles, CLAS-pool page rebinds, GDeflate
//! decompression, host image copy and the staging path, and the queues they run on. The
//! loading and sparse probes' correctness cases are regression tests here.

use std::time::Duration;

use gpu_suite::clusters::*;
use gpu_suite::shaders::{step, texel_read};
use gpu_suite::{assert_clean, device, pipeline, readback, render, upload};
use orr_gpu::{
    Allocation, BlasLimits, ClusterVertexFormat, CommandList, DecompressRegion, Device, DeviceDesc,
    Format, GpuAddress, Hazards, MemoryKind, PageBind, Queue, QueueKind, QueueRequest, RtasFlags,
    RtasLimits, RtasMode, RtasOp, RtasOpDesc, SparseBinds, Stages, Texture, TextureDesc,
    TextureKind, TextureMemory, TextureRegion, TextureUsage, TileBind, TimelinePoint, ViewDesc,
};

const WAIT: Duration = Duration::from_secs(60);

fn queue(kind: QueueKind) -> &'static Queue {
    device().queue(kind).unwrap()
}

/// Submits `list` on its queue after `wait` and blocks until it has run.
fn run_on(queue: &Queue, list: CommandList<'_>, wait: &[TimelinePoint]) -> TimelinePoint {
    let point = queue.submit(vec![list], wait).unwrap();
    queue.wait(point, WAIT).unwrap();
    point
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut v: Vec<u8> = (0..n.div_ceil(8))
            .flat_map(|_| self.next().to_le_bytes())
            .collect();
        v.truncate(n);
        v
    }
}

fn mip_extent(desc: &TextureDesc, mip: u32) -> [u32; 2] {
    [
        (desc.extent[0] >> mip).max(1),
        (desc.extent[1] >> mip).max(1),
    ]
}

/// Tight random texels per layer and mip.
fn texels(desc: &TextureDesc, rng: &mut Rng) -> Vec<Vec<Vec<u8>>> {
    (0..desc.layers)
        .map(|_| {
            (0..desc.mips)
                .map(|m| {
                    let [w, h] = mip_extent(desc, m);
                    rng.bytes(desc.format.region_bytes([w, h, 1]) as usize)
                })
                .collect()
        })
        .collect()
}

/// Every tile above the mip tail as (layer, mip, tile).
fn tiles_of(t: &Texture) -> Vec<(u32, u32, [u32; 2])> {
    let desc = t.desc();
    let layout = t.sparse_layout().unwrap();
    let mut v = Vec::new();
    for layer in 0..desc.layers {
        for mip in 0..layout.tail_first_mip {
            let [nx, ny] = layout.tiles(desc, mip);
            for y in 0..ny {
                for x in 0..nx {
                    v.push((layer, mip, [x, y]));
                }
            }
        }
    }
    v
}

/// Bytes of a block holding every tile and mip tail of `t`.
fn sparse_bytes(t: &Texture) -> u64 {
    let layout = t.sparse_layout().unwrap();
    tiles_of(t).len() as u64 * layout.tile_bytes
        + u64::from(layout.tails(t.desc())) * layout.tail_bytes
}

/// Binds every tile of `t` to slot `slots[i]` of `block`, then its tails after the tiles.
fn bind_all(t: &Texture, block: &Allocation<u8>, slots: &[u64]) -> TimelinePoint {
    let layout = *t.sparse_layout().unwrap();
    let tiles = tiles_of(t);
    let binds: Vec<TileBind<'_>> = tiles
        .iter()
        .zip(slots)
        .map(|(&(layer, mip, tile), &slot)| TileBind {
            mip,
            layer,
            tile,
            memory: Some((block, slot * layout.tile_bytes)),
        })
        .collect();
    let mut batch = SparseBinds::new();
    batch.tiles(t, &binds).unwrap();
    let tails_at = tiles.len() as u64 * layout.tile_bytes;
    for layer in 0..layout.tails(t.desc()) {
        let offset = tails_at + u64::from(layer) * layout.tail_bytes;
        batch.tail(t, layer, Some((block, offset))).unwrap();
    }
    device().bind_sparse(&batch).unwrap()
}

/// Copies the cooked layout of `data` into `t` on the loading queue: one 64 KiB staging
/// tile per bound tile, rows at the full tile width and padded edge tiles, then tight tail
/// mips.
fn staged_tile_upload(t: &Texture, data: &[Vec<Vec<u8>>]) -> TimelinePoint {
    let desc = *t.desc();
    let layout = *t.sparse_layout().unwrap();
    let format = desc.format;
    let block = format.block_extent();
    let block_bytes = format.block_bytes() as usize;
    let tiles = tiles_of(t);
    let tail_bytes: usize = (0..desc.layers)
        .flat_map(|l| (layout.tail_first_mip..desc.mips).map(move |m| (l, m)))
        .map(|(l, m)| data[l as usize][m as usize].len())
        .sum();
    let staging_bytes = tiles.len() * layout.tile_bytes as usize + tail_bytes;
    let mut staging = device()
        .gpu_malloc::<u8>(MemoryKind::Upload, staging_bytes as u64, "tile staging")
        .unwrap();
    let mut list = queue(QueueKind::Loading).command_list().unwrap();
    list.discard(t);
    let tile_row_bytes = (layout.tile_extent[0] / block) as usize * block_bytes;
    let gpu = staging.gpu;
    let mapped = staging.as_mut_slice().unwrap();
    for (i, &(layer, mip, tile)) in tiles.iter().enumerate() {
        let region = layout.tile_region(&desc, mip, layer, tile);
        let [w, _] = mip_extent(&desc, mip);
        let mip_row_bytes = w.div_ceil(block) as usize * block_bytes;
        let rows = region.extent[1].div_ceil(block) as usize;
        let row_bytes = region.extent[0].div_ceil(block) as usize * block_bytes;
        let x0 = (region.offset[0] / block) as usize * block_bytes;
        let y0 = (region.offset[1] / block) as usize;
        let at = i * layout.tile_bytes as usize;
        let src = &data[layer as usize][mip as usize];
        for r in 0..rows {
            let from = (y0 + r) * mip_row_bytes + x0;
            let to = at + r * tile_row_bytes;
            mapped[to..to + row_bytes].copy_from_slice(&src[from..from + row_bytes]);
        }
        list.copy_rows_to_texture(t, &region, gpu.offset(at as u64), layout.tile_extent[0]);
    }
    let mut at = tiles.len() * layout.tile_bytes as usize;
    for layer in 0..desc.layers {
        for mip in layout.tail_first_mip..desc.mips {
            let src = &data[layer as usize][mip as usize];
            mapped[at..at + src.len()].copy_from_slice(src);
            let region = TextureRegion {
                base_layer: layer,
                layers: 1,
                ..TextureRegion::whole_mip(&desc, mip)
            };
            list.copy_to_texture(t, &region, gpu.offset(at as u64));
            at += src.len();
        }
    }
    let loading = queue(QueueKind::Loading);
    let point = loading.submit(vec![list], &[]).unwrap();
    device().gpu_free(staging, point);
    point
}

/// Reads every mip of `t` back on the render queue after `wait`, per layer and mip.
fn read_mips(t: &Texture, wait: &[TimelinePoint]) -> Vec<Vec<Vec<u8>>> {
    let desc = *t.desc();
    let sizes: Vec<u64> = (0..desc.mips)
        .map(|m| TextureRegion::whole_mip(&desc, m).bytes(desc.format))
        .collect();
    let out = readback::<u8>(sizes.iter().sum(), "mips readback");
    let mut list = render().command_list().unwrap();
    let mut at = 0;
    for (m, &bytes) in sizes.iter().enumerate() {
        let region = TextureRegion::whole_mip(&desc, m as u32);
        list.copy_from_texture(out.gpu.offset(at), t, &region);
        at += bytes;
    }
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run_on(render(), list, wait);
    let bytes = out.as_slice().unwrap();
    let mut per_layer: Vec<Vec<Vec<u8>>> = vec![Vec::new(); desc.layers as usize];
    let mut at = 0;
    for &size in &sizes {
        let layer_bytes = (size / u64::from(desc.layers)) as usize;
        for layer in per_layer.iter_mut() {
            layer.push(bytes[at..at + layer_bytes].to_vec());
            at += layer_bytes;
        }
    }
    per_layer
}

/// Uploads cooked tiles into a fully bound sparse texture, reads every mip back, then
/// rebinds every tile to shuffled slots of another block and does it again.
fn tiles_round_trip(desc: &TextureDesc, seed: u64) {
    let d = device();
    let t = d
        .create_texture(desc, TextureMemory::Sparse, "sparse tiles")
        .unwrap();
    let layout = *t.sparse_layout().unwrap();
    assert_eq!(layout.tile_bytes, 65536, "{desc:?}");
    let mut rng = Rng(seed);
    let data = texels(desc, &mut rng);
    let n = tiles_of(&t).len();
    let blocks: Vec<Allocation<u8>> = (0..2)
        .map(|_| {
            d.gpu_malloc::<u8>(
                MemoryKind::DeviceLocal,
                sparse_bytes(&t).max(1),
                "tile block",
            )
            .unwrap()
        })
        .collect();
    let in_order: Vec<u64> = (0..n as u64).collect();
    let mut shuffled = in_order.clone();
    for i in (1..n).rev() {
        shuffled.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    for (pass, (block, slots)) in [(&blocks[0], &in_order), (&blocks[1], &shuffled)]
        .into_iter()
        .enumerate()
    {
        bind_all(&t, block, slots);
        let copied = staged_tile_upload(&t, &data);
        let got = read_mips(&t, &[copied]);
        for layer in 0..desc.layers as usize {
            for mip in 0..desc.mips as usize {
                assert!(
                    got[layer][mip] == data[layer][mip],
                    "{desc:?} pass {pass}: layer {layer} mip {mip} differs (tail from mip {})",
                    layout.tail_first_mip
                );
            }
        }
    }
    drop(t);
    drop(blocks);
    d.collect_garbage();
}

#[test]
fn tiles_exact_by_staged_copy() {
    let formats = [
        Format::Bc7Unorm,
        Format::Bc7Srgb,
        Format::Bc4Unorm,
        Format::Bc5Unorm,
        Format::Bc6hUfloat,
        Format::R16Unorm,
        Format::R8Unorm,
        Format::Rgba8Unorm,
    ];
    let sizes = [(4096, 4096), (1300, 700), (700, 1300), (2600, 300)];
    for (i, format) in formats.into_iter().enumerate() {
        for (j, (w, h)) in sizes.into_iter().enumerate() {
            let mut desc = TextureDesc::d2(format, w, h, TextureUsage::SAMPLED);
            desc.mips = 32 - w.max(h).leading_zeros();
            tiles_round_trip(&desc, 1 + (i * 4 + j) as u64);
        }
    }
    assert_clean();
}

#[test]
fn cube_tiles_and_tails_exact() {
    let desc = TextureDesc {
        kind: TextureKind::Cube,
        format: Format::Bc7Unorm,
        extent: [1024, 1024, 1],
        mips: 11,
        layers: 6,
        usage: TextureUsage::SAMPLED,
    };
    let t = device()
        .create_texture(&desc, TextureMemory::Sparse, "cube layout")
        .unwrap();
    assert!(
        !t.sparse_layout().unwrap().single_tail,
        "cube faces bind their own tails"
    );
    drop(t);
    tiles_round_trip(&desc, 77);
    assert_clean();
}

/// A tile bound and filled on the loading queue reads back through a sampled view on the
/// render queue once the render submit waits the loading point.
#[test]
fn tile_sampled_after_loading_wait() {
    let d = device();
    let desc = TextureDesc::d2(Format::Rgba8Unorm, 512, 512, TextureUsage::SAMPLED);
    let t = d
        .create_texture(&desc, TextureMemory::Sparse, "sampled tile")
        .unwrap();
    let layout = *t.sparse_layout().unwrap();
    assert_eq!(layout.tail_first_mip, 1, "512 x 512 RGBA8 has no tail");
    let tile = [1, 2];
    let block = d
        .gpu_malloc::<u8>(MemoryKind::DeviceLocal, layout.tile_bytes, "one tile")
        .unwrap();
    let mut batch = SparseBinds::new();
    batch
        .tiles(
            &t,
            &[TileBind {
                mip: 0,
                layer: 0,
                tile,
                memory: Some((&block, 0)),
            }],
        )
        .unwrap();
    d.bind_sparse(&batch).unwrap();
    let region = layout.tile_region(&desc, 0, 0, tile);
    let texels = Rng(5).bytes(region.bytes(desc.format) as usize);
    let staging = upload(&texels, "tile texels");
    let mut list = queue(QueueKind::Loading).command_list().unwrap();
    list.discard(&t);
    list.copy_to_texture(&t, &region, staging.gpu);
    let loaded = queue(QueueKind::Loading).submit(vec![list], &[]).unwrap();

    let slot = d.alloc_image_slot().unwrap();
    let source = d.write_sampled(slot, &t, &ViewDesc::default());
    let [w, h, _] = region.extent;
    let out = readback::<[f32; 4]>(u64::from(w * h), "tile texels out");
    let root = upload(
        &[texel_read::Root {
            out: out.gpu,
            source,
            x: region.offset[0],
            y: region.offset[1],
            width: w,
            height: h,
            mip: 0,
        }],
        "texel read root",
    );
    let read = pipeline(&texel_read::MODULE, "main");
    let mut list = render().command_list().unwrap();
    list.dispatch(&read, root.gpu, [w.div_ceil(8), h.div_ceil(8), 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = run_on(render(), list, &[loaded]);
    let wrong = out
        .as_slice()
        .unwrap()
        .iter()
        .zip(texels.chunks(4))
        .filter(|(v, b)| (0..4).any(|c| (v[c] * 255.0).round() as u8 != b[c]))
        .count();
    assert_eq!(wrong, 0, "texels differing from the uploaded tile");
    d.free_image_slot(slot, point);
    drop((staging, root));
    let unbind = [TileBind {
        mip: 0,
        layer: 0,
        tile,
        memory: None,
    }];
    let mut batch = SparseBinds::new();
    batch.tiles(&t, &unbind).unwrap();
    let unbound = d.bind_sparse(&batch).unwrap();
    queue(QueueKind::Loading).wait(unbound, WAIT).unwrap();
    assert_clean();
}

fn front_back(clusters: &[Cluster], front_z: f32) -> [RefInst; 2] {
    let (back, front) = clusters.split_last().unwrap();
    let front: Vec<&Cluster> = front.iter().collect();
    [
        ref_inst(FRONT_CUSTOM, 0, [0.0, 0.0, front_z], &front),
        ref_inst(BACK_CUSTOM, 1, [0.0; 3], &[back]),
    ]
}

/// CLAS and cluster BLAS of the base clusters at `at` in a sparse range, under a TLAS with
/// the front instance at `front_z`.
fn build_at(b: &mut Builder, at: u64, blas_off: u64, front_z: f32) -> orr_gpu::Tlas {
    let clusters = base_clusters();
    let clas = b.build_clas(&clusters, data_flags(), 0, Some(GpuAddress(at)), None);
    let blas = b.build_blas(
        clas.addrs.gpu,
        &[vec![0, 1, 2, 3], vec![4]],
        Some(GpuAddress(at + blas_off)),
    );
    b.keep_any(clas);
    let tlas = b.build_tlas(
        blas.addrs.gpu,
        &[
            instance([0.0, 0.0, front_z], FRONT_CUSTOM, 0),
            instance([0.0; 3], BACK_CUSTOM, 1),
        ],
    );
    b.keep_any(blas);
    tlas
}

/// The CLAS pool's page rebind: region A holds CLAS under a BLAS that stays live; each
/// cycle rebinds region B to the other block, builds and traces there, then unbinds it on
/// the loading queue while the render queue traces A. Every ray matches the reference.
#[test]
fn clas_pages_rebind_beside_live_blas() {
    const CYCLES: u32 = 24;
    let d = device();
    let clusters = base_clusters();
    let clas_bytes = d
        .rtas_sizes(&clas_desc(
            RtasOp::BuildClas,
            RtasMode::Implicit,
            data_flags(),
            &clusters,
        ))
        .result_max;
    let blas_bytes = d
        .rtas_sizes(&RtasOpDesc {
            op: RtasOp::BuildClusterBlas,
            mode: RtasMode::Implicit,
            flags: RtasFlags::FAST_TRACE,
            vertex_format: ClusterVertexFormat::F32x3,
            limits: RtasLimits::Blas(BlasLimits {
                max_args: 2,
                max_clas_per_blas: 4,
                max_total_clas: 5,
            }),
        })
        .result_max;
    let page = d.gpu_malloc_sparse::<u8>(1, "page size").unwrap().page_size;
    let blas_off = clas_bytes.next_multiple_of(256);
    let region = (blas_off + blas_bytes).div_ceil(page);
    let range = d
        .gpu_malloc_sparse::<u8>(2 * region * page, "clas pool")
        .unwrap();
    let block = |name| {
        d.gpu_malloc::<u8>(MemoryKind::DeviceLocal, region * page, name)
            .unwrap()
    };
    let (a, b_blocks) = (
        block("region a"),
        [block("region b 0"), block("region b 1")],
    );
    let base = range.gpu.address().0;
    let mut b = Builder::new();
    let bound_a = d
        .bind_sparse_pages(
            &range,
            &[PageBind {
                first_page: 0,
                count: region,
                memory: Some((&a, 0)),
            }],
        )
        .unwrap();
    b.waits.push(bound_a);
    let tlas_a = build_at(&mut b, base, blas_off, FRONT_Z);
    let ref_a = front_back(&clusters, FRONT_Z);
    let first = b.trace(tlas_a, 0);
    b.flush();
    check(first.as_slice().unwrap(), &ref_a, 0).assert_ok("region a");
    let want_a = first.as_slice().unwrap().to_vec();
    let mut want_b: [Option<Vec<u32>>; 3] = Default::default();
    // A trace equal to one already checked passes; any other goes through the reference.
    let verify = |got: &[u32], want: &[u32], insts: &[RefInst], what: String| {
        if got != want {
            check(got, insts, 0).assert_ok(&what);
        }
    };
    let mut unbound = None;
    for cycle in 0..CYCLES {
        let variant = (cycle % 3) as usize;
        let front_z = FRONT_Z * (1 + variant) as f32;
        let rebound = d
            .bind_sparse_pages(
                &range,
                &[PageBind {
                    first_page: region,
                    count: region,
                    memory: Some((&b_blocks[cycle as usize % 2], 0)),
                }],
            )
            .unwrap();
        b.waits.push(rebound);
        let tlas_b = build_at(&mut b, base + region * page, blas_off, front_z);
        let out_b = b.trace(tlas_b, 0);
        let out_a = b.trace(tlas_a, 0);
        b.flush();
        let got_b = out_b.as_slice().unwrap();
        let ref_b = front_back(&clusters, front_z);
        match &want_b[variant] {
            Some(want) => verify(got_b, want, &ref_b, format!("region b, cycle {cycle}")),
            None => {
                check(got_b, &ref_b, 0).assert_ok(&format!("region b, cycle {cycle}"));
                want_b[variant] = Some(got_b.to_vec());
            }
        }
        verify(
            out_a.as_slice().unwrap(),
            &want_a,
            &ref_a,
            format!("region a beside the rebind, cycle {cycle}"),
        );
        unbound = Some(
            d.bind_sparse_pages(
                &range,
                &[PageBind {
                    first_page: region,
                    count: region,
                    memory: None,
                }],
            )
            .unwrap(),
        );
        let during = b.trace(tlas_a, 0);
        b.flush();
        verify(
            during.as_slice().unwrap(),
            &want_a,
            &ref_a,
            format!("region a during the unbind, cycle {cycle}"),
        );
    }
    queue(QueueKind::Loading)
        .wait(unbound.unwrap(), WAIT)
        .unwrap();
    drop(b);
    drop((range, a, b_blocks));
    assert_clean();
}

/// GDeflate 64 KiB regions compressed on the CPU decompress bit-exact on the loading
/// queue; a GPU count below the maximum leaves the regions after it untouched.
#[test]
fn gdeflate_bit_exact() {
    const REGION: usize = 64 << 10;
    const REGIONS: usize = 64;
    let d = device();
    if !d.caps().memory_decompression {
        log::warn!("no memory decompression; skipped");
        return;
    }
    let mut rng = Rng(11);
    // Halves of compressible text-like bytes and incompressible random bytes.
    let source: Vec<u8> = (0..REGIONS)
        .flat_map(|i| {
            if i % 2 == 0 {
                (0..REGION)
                    .map(|j| b"orr loads pages "[(j + i) % 16] ^ (rng.next() % 3) as u8)
                    .collect::<Vec<u8>>()
            } else {
                rng.bytes(REGION)
            }
        })
        .collect();
    let mut compressor = gdeflate::Compressor::new(gdeflate::CompressionLevel::Level12).unwrap();
    let chunks: Vec<Vec<u8>> = source
        .chunks(REGION)
        .map(|c| compressor.compress(c, REGION).unwrap().bytes)
        .collect();
    let mut packed = Vec::new();
    let mut offsets = Vec::new();
    for c in &chunks {
        packed.resize(packed.len().next_multiple_of(4), 0);
        offsets.push(packed.len() as u64);
        packed.extend_from_slice(c);
    }
    packed.resize(packed.len().next_multiple_of(4), 0);
    let compressed = upload(&packed, "gdeflate input");
    let out = d
        .gpu_malloc::<u8>(
            MemoryKind::DeviceLocal,
            (REGIONS * REGION) as u64,
            "gdeflate out",
        )
        .unwrap();
    let regions: Vec<DecompressRegion> = (0..REGIONS)
        .map(|i| DecompressRegion {
            src: compressed.gpu.address().0 + offsets[i],
            dst: out.gpu.address().0 + (i * REGION) as u64,
            compressed_bytes: chunks[i].len() as u64,
            decompressed_bytes: REGION as u64,
        })
        .collect();
    let regions = upload(&regions, "gdeflate regions");
    let fill = vec![0xcdu8; REGIONS * REGION];
    let cleared = upload(&fill, "gdeflate clear");
    let host = readback::<u8>((REGIONS * REGION) as u64, "gdeflate readback");
    let loading = queue(QueueKind::Loading);
    for count in [REGIONS as u32, REGIONS as u32 - 5] {
        let count_at = upload(&[count], "gdeflate count");
        let mut list = loading.command_list().unwrap();
        list.copy(out.gpu, cleared.gpu, out.size);
        list.barrier(Stages::TRANSFER, Stages::DECOMPRESS, Hazards::empty());
        list.decompress(regions.gpu, count_at.gpu, REGIONS as u32);
        list.barrier(Stages::DECOMPRESS, Stages::TRANSFER, Hazards::empty());
        let decompressed = loading.submit(vec![list], &[]).unwrap();
        let mut list = render().command_list().unwrap();
        list.copy(host.gpu, out.gpu, out.size);
        list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
        run_on(render(), list, &[decompressed]);
        let got = host.as_slice().unwrap();
        for i in 0..REGIONS {
            let span = i * REGION..(i + 1) * REGION;
            if i < count as usize {
                assert!(
                    got[span.clone()] == source[span],
                    "region {i} of {count} differs"
                );
            } else {
                assert!(
                    got[span].iter().all(|&b| b == 0xcd),
                    "region {i} past the count of {count} was written"
                );
            }
        }
    }
    assert_clean();
}

/// Address-based upload and readback of a texture in one submission, ordered by a transfer
/// barrier alone, read back exact every run (the 617.14 stale-copy repro).
#[test]
fn texture_round_trip_in_one_submit() {
    const RUNS: u32 = 1000;
    let d = device();
    let cases = [
        TextureDesc::d2(Format::Rgba8Unorm, 256, 256, TextureUsage::SAMPLED),
        TextureDesc::d2(Format::Bc7Unorm, 512, 512, TextureUsage::SAMPLED),
        TextureDesc {
            kind: TextureKind::D3,
            extent: [64, 64, 16],
            ..TextureDesc::d2(Format::Rgba8Unorm, 64, 64, TextureUsage::SAMPLED)
        },
        TextureDesc {
            layers: 4,
            ..TextureDesc::d2(Format::Rgba8Unorm, 128, 128, TextureUsage::SAMPLED)
        },
        TextureDesc {
            kind: TextureKind::Cube,
            layers: 6,
            ..TextureDesc::d2(Format::Rgba8Unorm, 128, 128, TextureUsage::SAMPLED)
        },
    ];
    for (k, desc) in cases.iter().enumerate() {
        let t = d
            .create_texture(desc, TextureMemory::Dedicated, "round trip")
            .unwrap();
        let region = TextureRegion::whole_mip(desc, 0);
        let words = (region.bytes(desc.format) / 4) as usize;
        let mut up = d
            .gpu_malloc::<u32>(MemoryKind::Upload, words as u64, "round trip up")
            .unwrap();
        let mut down = readback::<u32>(words as u64, "round trip down");
        let mut wrong = 0;
        for run in 0..RUNS {
            let tag = run.wrapping_mul(0x9e37_79b9) ^ (k as u32) << 28;
            for (w, v) in up.as_mut_slice().unwrap().iter_mut().enumerate() {
                *v = tag ^ (w as u32).wrapping_mul(0x85eb_ca6b);
            }
            down.as_mut_slice().unwrap().fill(0xcdcd_cdcd);
            let mut list = render().command_list().unwrap();
            list.discard(&t);
            list.copy_to_texture(&t, &region, up.gpu.cast());
            list.barrier(Stages::TRANSFER, Stages::TRANSFER, Hazards::empty());
            list.copy_from_texture(down.gpu.cast(), &t, &region);
            list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
            run_on(render(), list, &[]);
            wrong += u32::from(down.as_slice().unwrap() != up.as_slice().unwrap());
        }
        assert_eq!(wrong, 0, "{desc:?}: runs read back wrong");
    }
    assert_clean();
}

/// Render, loading and copy queues copy upload -> scratch -> readback from three threads
/// at once, each checking every run (the 617.14 three-queue repro).
#[test]
fn three_queues_copy_concurrently() {
    const RUNS: u32 = 1000;
    const WORDS: u64 = 64;
    std::thread::scope(|s| {
        for (q, kind) in [QueueKind::Render, QueueKind::Loading, QueueKind::Copy]
            .into_iter()
            .enumerate()
        {
            s.spawn(move || {
                let d = device();
                let queue = queue(kind);
                let mut up = d
                    .gpu_malloc::<u32>(MemoryKind::Upload, WORDS, "copy up")
                    .unwrap();
                let scratch = d
                    .gpu_malloc::<u32>(MemoryKind::DeviceLocal, WORDS, "copy scratch")
                    .unwrap();
                let mut down = readback::<u32>(WORDS, "copy down");
                for run in 1..=RUNS {
                    for (w, v) in up.as_mut_slice().unwrap().iter_mut().enumerate() {
                        *v = (q as u32) << 24 | run << 8 | w as u32;
                    }
                    down.as_mut_slice().unwrap().fill(0xcdcd_cdcd);
                    let mut list = queue.command_list().unwrap();
                    list.copy(scratch.gpu.cast(), up.gpu.cast(), WORDS * 4);
                    list.barrier(Stages::TRANSFER, Stages::TRANSFER, Hazards::empty());
                    list.copy(down.gpu.cast(), scratch.gpu.cast(), WORDS * 4);
                    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
                    run_on(queue, list, &[]);
                    assert_eq!(
                        down.as_slice().unwrap(),
                        up.as_slice().unwrap(),
                        "{kind:?} run {run}"
                    );
                }
            });
        }
    });
    assert_clean();
}

/// Mips written by host image copy from worker threads, one thread per mip, match the
/// source both by host readback and by a GPU copy, as do the same texels staged through
/// the copy queue into a DeviceLocal texture.
#[test]
fn host_copied_and_staged_texels_match() {
    let d = device();
    if !d.caps().host_image_copy {
        log::warn!("no host image copy; skipped");
        return;
    }
    for (seed, format) in [Format::Bc7Unorm, Format::Rgba8Unorm, Format::R16Unorm]
        .into_iter()
        .enumerate()
    {
        let mut desc = TextureDesc::d2(format, 1024, 512, TextureUsage::SAMPLED);
        desc.mips = 11;
        let data = texels(&desc, &mut Rng(seed as u64 + 3));
        let mips = &data[0];
        let host_desc = TextureDesc {
            usage: TextureUsage::SAMPLED | TextureUsage::HOST_COPY,
            ..desc
        };
        let host = d
            .create_texture(&host_desc, TextureMemory::Dedicated, "host copied")
            .unwrap();
        std::thread::scope(|s| {
            for (m, texels) in mips.iter().enumerate() {
                let host = &host;
                s.spawn(move || {
                    let region = TextureRegion::whole_mip(&host_desc, m as u32);
                    d.host_copy_to_texture(host, &region, texels).unwrap();
                });
            }
        });
        for (m, texels) in mips.iter().enumerate() {
            let region = TextureRegion::whole_mip(&host_desc, m as u32);
            let mut back = vec![0u8; texels.len()];
            d.host_copy_from_texture(&mut back, &host, &region).unwrap();
            assert!(&back == texels, "{format:?} mip {m}: host readback differs");
        }
        let gpu_read = read_mips(&host, &[]);
        assert!(
            gpu_read[0] == *mips,
            "{format:?}: GPU copy of host-copied texels differs"
        );

        let staged = d
            .create_texture(&desc, TextureMemory::Dedicated, "staged")
            .unwrap();
        let copied = staged_upload(d, d.queue(QueueKind::Copy).unwrap(), &staged, mips);
        let staged_read = read_mips(&staged, &[copied]);
        assert!(staged_read[0] == *mips, "{format:?}: staged texels differ");
    }
    assert_clean();
}

/// Uploads tight `mips` into `t` through Upload staging and a copy on `queue`.
fn staged_upload(d: &Device, queue: &Queue, t: &Texture, mips: &[Vec<u8>]) -> TimelinePoint {
    let bytes: usize = mips.iter().map(Vec::len).sum();
    let mut staging = d
        .gpu_malloc::<u8>(MemoryKind::Upload, bytes as u64, "staging")
        .unwrap();
    let gpu = staging.gpu;
    let mapped = staging.as_mut_slice().unwrap();
    let mut list = queue.command_list().unwrap();
    list.discard(t);
    let mut at = 0;
    for (m, texels) in mips.iter().enumerate() {
        mapped[at..at + texels.len()].copy_from_slice(texels);
        let region = TextureRegion::whole_mip(t.desc(), m as u32);
        list.copy_to_texture(t, &region, gpu.offset(at as u64));
        at += texels.len();
    }
    let point = queue.submit(vec![list], &[]).unwrap();
    d.gpu_free(staging, point);
    point
}

/// A device forced onto the staging path reports no resizable BAR or host image copy,
/// refuses host-copy textures, and moves texels through its copy queue to the render queue.
#[test]
fn forced_staging_path() {
    gpu_suite::log_to_stderr();
    let d = Device::new(&DeviceDesc {
        name: "gpu-suite staging",
        validation: true,
        force_staging: true,
        queues: QueueRequest {
            async_compute: true,
            copy: true,
            ..QueueRequest::default()
        },
        ..DeviceDesc::default()
    })
    .unwrap();
    assert!(!d.caps().rebar && !d.caps().host_image_copy);
    let host_desc = TextureDesc::d2(
        Format::Rgba8Unorm,
        64,
        64,
        TextureUsage::SAMPLED | TextureUsage::HOST_COPY,
    );
    assert!(
        d.create_texture(&host_desc, TextureMemory::Dedicated, "refused")
            .is_err()
    );
    let mut desc = TextureDesc::d2(Format::Bc7Unorm, 512, 256, TextureUsage::SAMPLED);
    desc.mips = 10;
    let mips = texels(&desc, &mut Rng(21)).remove(0);
    let t = d
        .create_texture(&desc, TextureMemory::Dedicated, "staged")
        .unwrap();
    let copied = staged_upload(&d, d.queue(QueueKind::Copy).unwrap(), &t, &mips);
    let render = d.queue(QueueKind::Render).unwrap();
    for (m, texels) in mips.iter().enumerate() {
        let region = TextureRegion::whole_mip(&desc, m as u32);
        let out = d
            .gpu_malloc::<u8>(MemoryKind::Readback, texels.len() as u64, "staged out")
            .unwrap();
        let mut list = render.command_list().unwrap();
        list.copy_from_texture(out.gpu, &t, &region);
        list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
        run_on(render, list, &[copied]);
        assert!(out.as_slice().unwrap() == &texels[..], "mip {m} differs");
    }
    assert_eq!(
        d.validation_messages(),
        0,
        "validation messages (see the log)"
    );
}

/// A chain of dispatches alternating between the render-side compute queue and the render
/// queue, each waiting the last one's timeline point, gives the exact composed result.
#[test]
fn compute_queue_orders_by_timeline() {
    const N: u32 = 1 << 16;
    const LINKS: usize = 6;
    let d = device();
    let start: Vec<u32> = (0..N).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    let buffers: Vec<Allocation<u32>> = (0..=LINKS)
        .map(|i| {
            if i == 0 {
                upload(&start, "chain start")
            } else {
                d.gpu_malloc::<u32>(MemoryKind::DeviceLocal, u64::from(N), "chain link")
                    .unwrap()
            }
        })
        .collect();
    let roots: Vec<_> = (0..LINKS)
        .map(|i| {
            upload(
                &[step::Root {
                    src: buffers[i].gpu,
                    dst: buffers[i + 1].gpu,
                    n: N,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                }],
                "chain root",
            )
        })
        .collect();
    let p = pipeline(&step::MODULE, "main");
    let compute = queue(QueueKind::Compute);
    let mut last = None;
    for (i, root) in roots.iter().enumerate() {
        let q = if i % 2 == 0 { compute } else { render() };
        let mut list = q.command_list().unwrap();
        if i % 2 == 0 {
            list.set_occupancy(0.25, 0.0);
        }
        list.dispatch(&p, root.gpu, [N.div_ceil(64), 1, 1]);
        let waits: Vec<TimelinePoint> = last.into_iter().collect();
        last = Some(q.submit(vec![list], &waits).unwrap());
    }
    let out = readback::<u32>(u64::from(N), "chain out");
    let mut list = render().command_list().unwrap();
    list.copy(out.gpu.cast(), buffers[LINKS].gpu.cast(), u64::from(N) * 4);
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    run_on(render(), list, &[last.unwrap()]);
    let expected: Vec<u32> = start
        .iter()
        .map(|&v| (0..LINKS).fold(v, |v, _| v.wrapping_mul(3).wrapping_add(1)))
        .collect();
    assert!(
        out.as_slice().unwrap() == &expected[..],
        "chain result differs"
    );
    assert_clean();
}
