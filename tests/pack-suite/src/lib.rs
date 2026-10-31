//! Headless GPU tests of orr-pack's Slang modules. Every test runs on
//! `orr_scene::testing`'s device with validation on and fails if the layer reports anything.

use std::path::{Path, PathBuf};
use std::time::Duration;

use orr_gpu::{
    GpuPtr, Hazards, MemoryKind, QueueKind, ShaderModule, ShaderStage, SpecData, Stages,
};
use orr_pack::mesh::{self, Cluster, MeshInfo, PageHeader, StoredPage};
use orr_pack::{Archive, Blob, Class, Entry};
use orr_scene::testing::{device, wait};

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// Runs `module`'s `main` once over `groups` workgroups with `root` and waits for it.
pub fn dispatch<R>(module: &'static ShaderModule, root: GpuPtr<R>, groups: u32) {
    let stage = ShaderStage::new(module, "main", SpecData::default()).unwrap();
    let p = device()
        .create_compute_pipeline(&stage, module.name)
        .unwrap();
    let render = device().queue(QueueKind::Render).unwrap();
    let mut cmd = render.command_list().unwrap();
    cmd.dispatch(&p, root, [groups, 1, 1]);
    cmd.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    wait(render.submit(vec![cmd], &[]).unwrap());
}

/// The decoded bytes of blob `b`.
pub fn read(a: &Archive, b: &Blob) -> Vec<u8> {
    let mut v = vec![0; b.raw_len as usize];
    a.read(b, &mut v).unwrap();
    v
}

/// What [`decode_stored`] leaves in bytes it does not write.
pub const UNWRITTEN: u8 = 0xCD;

/// Results of [`decode_stored`].
pub struct StoredDecode {
    /// Each page's status bits from both passes; 0 for a page that decoded.
    pub status: Vec<u32>,
    /// Each page's [`mesh::PAGE_MAX`]-byte slot.
    pub resident: Vec<Vec<u8>>,
    /// GPU time of the two passes.
    pub gpu_ms: f64,
}

/// Decodes `stored` pages as one batch on the loading queue through orr_pack.page_codec,
/// into slots of [`mesh::PAGE_MAX`] bytes filled with [`UNWRITTEN`] beforehand.
pub fn decode_stored(stored: &[Vec<u8>]) -> StoredDecode {
    use shaders::stored_pages::{MODULE, Root};
    let d = device();
    let slot = mesh::PAGE_MAX;
    let n = stored.len();
    let mut offsets = Vec::with_capacity(n);
    let mut total = 0;
    for p in stored {
        offsets.push(total);
        total += p.len().next_multiple_of(4);
    }
    let mut src = d
        .gpu_malloc::<u8>(MemoryKind::Upload, total.max(4) as u64, "stored pages")
        .unwrap();
    let mut dst = d
        .gpu_malloc::<u8>(MemoryKind::Readback, (n * slot) as u64, "resident pages")
        .unwrap();
    dst.as_mut_slice().unwrap().fill(UNWRITTEN);
    let mut status = d
        .gpu_malloc::<u32>(MemoryKind::Readback, n as u64, "status")
        .unwrap();
    status.as_mut_slice().unwrap().fill(0);
    let mut pages = d
        .gpu_malloc::<StoredPage>(MemoryKind::Upload, n as u64, "pages")
        .unwrap();
    let mut clusters = 1;
    let (src_at, dst_at) = (src.gpu.address().0, dst.gpu.address().0);
    let (bytes, records) = (src.as_mut_slice().unwrap(), pages.as_mut_slice().unwrap());
    for (i, p) in stored.iter().enumerate() {
        bytes[offsets[i]..][..p.len()].copy_from_slice(p);
        records[i] = StoredPage {
            stored: src_at + offsets[i] as u64,
            resident: dst_at + (i * slot) as u64,
            stored_len: p.len() as u32,
            resident_capacity: slot as u32,
        };
        if let Some(h) = p.get(..size_of::<PageHeader>()) {
            let h: PageHeader = bytemuck::pod_read_unaligned(h);
            let most = (mesh::PAGE_MAX - size_of::<PageHeader>()) / size_of::<Cluster>();
            clusters = clusters.max(h.cluster_count.min(most as u32));
        }
    }
    let mut root = d.gpu_malloc::<Root>(MemoryKind::Upload, 1, "root").unwrap();
    root.as_mut_slice().unwrap()[0] = Root {
        pages: pages.gpu,
        status: status.gpu,
    };
    let pipeline = |entry| {
        let stage = ShaderStage::new(&MODULE, entry, SpecData::default()).unwrap();
        d.create_compute_pipeline(&stage, entry).unwrap()
    };
    let (copy, decode) = (pipeline("copy"), pipeline("decode"));
    let loading = d.queue(QueueKind::Loading).unwrap();
    let mut cmd = loading.command_list().unwrap();
    cmd.begin_pass("page codec", false);
    cmd.dispatch(&copy, root.gpu, [(slot / 4 / 256) as u32, n as u32, 1]);
    cmd.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    cmd.dispatch(&decode, root.gpu, [clusters, n as u32, 1]);
    cmd.end_pass();
    cmd.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let done = loading.submit(vec![cmd], &[]).unwrap();
    loading.wait(done, Duration::from_secs(10)).unwrap();
    let gpu_ms = d
        .pass_results(done)
        .unwrap_or_default()
        .iter()
        .map(|r| r.gpu_ms)
        .sum();
    let dst = dst.as_slice().unwrap();
    StoredDecode {
        status: status.as_slice().unwrap().to_vec(),
        resident: dst.chunks_exact(slot).map(<[u8]>::to_vec).collect(),
        gpu_ms,
    }
}

/// Every `.orr` archive under `dir`: each archive's path and the stored pages of all its
/// meshes.
pub fn cooked_set_pages(dir: &Path) -> Vec<(PathBuf, Vec<Vec<u8>>)> {
    fn files(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                files(&p, out);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("orr")) {
                out.push(p);
            }
        }
    }
    let mut paths = Vec::new();
    files(dir, &mut paths);
    paths
        .into_iter()
        .map(|path| {
            let a = Archive::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let pages = a
                .entries()
                .iter()
                .filter(|e| e.class == Class::Mesh as u16)
                .flat_map(|e| stored_pages(&a, e))
                .collect();
            (path, pages)
        })
        .collect()
}

/// Every stored page of mesh entry `e`.
pub fn stored_pages(a: &Archive, e: &Entry) -> Vec<Vec<u8>> {
    let blobs = a.blobs(e);
    let info = MeshInfo::parse(&read(a, &blobs[0])).unwrap();
    info.pages
        .iter()
        .map(|p| read(a, &blobs[p.blob as usize]))
        .collect()
}
