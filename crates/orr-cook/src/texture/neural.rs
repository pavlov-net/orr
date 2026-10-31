//! Neural texture sets: a material's textures, except opacity, compressed together by
//! RTXNTC's LibNTC into latent images and a decoder network ([`orr_pack::neural`]). The BC
//! textures always cook and ship beside the set.
//!
//! - Members: the material's bound textures in slot order whose cooked mip 0 is the largest
//!   among them, while their channels fit LibNTC's 16. BC6H textures and uncompressed ones
//!   other than displacement stay BC only.
//! - Input: the same mip chains the BC encoders get (alpha coverage and Toksvig roughness
//!   included). Color textures are stored as sRGB, normal maps as x and y at `0.5 v + 0.5`
//!   (as BC5 holds them), the rest linear.
//! - LibNTC's adaptive search looks for [`NeuralSettings::target_psnr`] over all channels at
//!   no more bits per texel than the members' BC textures. A member enters the set's texture
//!   table only when every one of its channels reaches [`NeuralSettings::min_psnr`] at mip 0,
//!   and none does when the set holds more bytes than the members' BC textures.
//! - Compression runs on the GPU through CUDA, one set at a time.
//! - The LibNTC library and file schema versions enter the cook key: a set is readable only
//!   by the inference shaders of the version that wrote it.

use std::sync::Mutex;

use orr_pack::neural::{NeuralHeader, NeuralInfo, NeuralTexture, NeuralWeights};
use orr_pack::texture::{FormatInfo, mip_size, vk};
use orr_pack::{ArchiveWriter, AssetId, BlobIn, Class, Hash32, Span};
use serde::Deserialize;

use super::material::{ImageFile, TextureJob};
use super::{Built, Encoding, Primaries, linear_to_srgb};
use crate::key::Key;
use crate::report::{NeuralReport, NeuralTextureReport};
use crate::{Error, Result};

/// How neural texture sets cook.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeuralSettings {
    /// PSNR over all channels that LibNTC's adaptive search aims for, dB.
    pub target_psnr: f32,
    /// Lowest PSNR of every channel of a texture at mip 0 for the texture to enter the set,
    /// dB.
    pub min_psnr: f32,
    /// Training steps per compression run.
    pub training_steps: u32,
    /// Training seed; LibNTC's stable training mode is on.
    pub seed: u32,
    /// CUDA device that trains.
    pub cuda_device: i32,
}

impl Default for NeuralSettings {
    fn default() -> Self {
        Self {
            target_psnr: 40.0,
            min_psnr: 40.0,
            training_steps: 100_000,
            seed: 1,
            cuda_device: 0,
        }
    }
}

/// LibNTC's channel limit.
const MAX_CHANNELS: usize = 16;

/// LibNTC's `ChannelFormat` values.
const UNORM8: i32 = 1;
const UNORM16: i32 = 2;

/// One texture of a set.
#[derive(Clone)]
struct Member {
    job: TextureJob,
    channels: u8,
    /// A [`NeuralTexture`] storage value.
    stored: u8,
}

/// A neural set to cook.
pub(crate) struct NeuralJob {
    pub name: String,
    pub key: Hash32,
    asset: String,
    members: Vec<Member>,
    size: [u32; 2],
    settings: NeuralSettings,
}

/// Channels and storage of a texture in a set, or `None` when it stays BC only.
fn channels(job: &TextureJob) -> Option<(u8, u8)> {
    match job.encoding() {
        Encoding::Bc7Srgb => Some((3, NeuralTexture::SRGB)),
        Encoding::Bc7 => Some((4, NeuralTexture::LINEAR)),
        Encoding::Bc5 if job.is_normal() => Some((2, NeuralTexture::NORMAL)),
        Encoding::Bc5 => Some((2, NeuralTexture::LINEAR)),
        Encoding::Bc4 | Encoding::Raw(super::DataFormat::R16) => Some((1, NeuralTexture::LINEAR)),
        Encoding::Bc6h | Encoding::Raw(_) => None,
    }
}

/// The set of a material's textures (`jobs` in slot order, opacity left out); `None` when
/// none can join.
pub(crate) fn plan(
    asset: &str,
    suffix: &str,
    jobs: &[&TextureJob],
    settings: &NeuralSettings,
) -> Result<Option<NeuralJob>> {
    let candidates: Vec<(&TextureJob, (u8, u8))> = jobs
        .iter()
        .filter_map(|j| channels(j).map(|c| (*j, c)))
        .collect();
    let Some(size) = candidates
        .iter()
        .map(|(j, _)| j.cooked_size())
        .max_by_key(|[w, h]| u64::from(*w) * u64::from(*h))
    else {
        return Ok(None);
    };
    let mut members = Vec::new();
    let mut used = 0;
    for (job, (n, stored)) in candidates {
        if job.cooked_size() != size || used + usize::from(n) > MAX_CHANNELS {
            continue;
        }
        used += usize::from(n);
        members.push(Member {
            job: job.clone(),
            channels: n,
            stored,
        });
    }
    if members.is_empty() {
        return Ok(None);
    }
    let name = format!("{asset}/neural/{suffix}");
    let fail = |message: String| Error::Texture {
        asset: asset.into(),
        message,
    };
    let mut key = Key::new(Class::NeuralTextureSet, &name)
        .str(&libntc::library().map_err(fail)?.key)
        .debug(settings);
    for m in &members {
        key = key.hash(m.job.key).bytes(&[m.channels, m.stored]);
    }
    Ok(Some(NeuralJob {
        name,
        key: key.finish(),
        asset: asset.into(),
        members,
        size,
        settings: settings.clone(),
    }))
}

/// Bytes of a BC texture's mips from `size` down to 1x1.
fn bc_bytes(e: Encoding, [w, h]: [u32; 2]) -> u64 {
    let f = FormatInfo::of(e.vk_format()).expect("cooked formats are known");
    (0..orr_pack::texture::full_mip_count(w, h))
        .map(|m| {
            let [mw, mh] = mip_size(w, h, m);
            f.mip_bytes(mw, mh) as u64
        })
        .sum()
}

/// PSNR of values in 0..1, dB.
fn psnr(reference: impl Iterator<Item = f32>, decoded: impl Iterator<Item = f32>) -> f32 {
    let (mut sum, mut n) = (0.0f64, 0u64);
    for (a, b) in reference.zip(decoded) {
        sum += f64::from(a - b).powi(2);
        n += 1;
    }
    let mse = sum / n.max(1) as f64;
    if mse == 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (1.0 / mse).log10()) as f32
}

/// A cooked set: blob 0 and the weight blobs, its latent texture and its report.
pub(crate) struct NeuralBuilt {
    info: NeuralInfo,
    weights: Vec<Vec<u8>>,
    latents: Built,
    pub report: NeuralReport,
}

/// LibNTC compresses one set at a time on the GPU.
static GPU: Mutex<()> = Mutex::new(());

impl NeuralJob {
    /// Ids of the member textures, which identify the set.
    pub(crate) fn members(&self) -> Vec<AssetId> {
        self.members
            .iter()
            .map(|m| AssetId::from_name(&m.job.name))
            .collect()
    }

    /// Estimated peak bytes: the members' texture jobs, then 4 bytes per channel texel of
    /// the chains and the decoded mip 0.
    pub(crate) fn working_set(&self, images: &[ImageFile]) -> u64 {
        let jobs: u64 = self.members.iter().map(|m| m.job.working_set(images)).sum();
        let channels: u64 = self.members.iter().map(|m| u64::from(m.channels)).sum();
        jobs + u64::from(self.size[0]) * u64::from(self.size[1]) * channels * 4 * 3
    }

    fn fail(&self, message: String) -> Error {
        Error::Texture {
            asset: self.asset.clone(),
            message,
        }
    }

    pub(crate) fn run(&self, images: &[ImageFile]) -> Result<NeuralBuilt> {
        let [w, h] = self.size;
        let mut layout = Vec::new();
        let mut first = 0u8;
        for m in &self.members {
            layout.push((first, m.channels, m.stored));
            first += m.channels;
        }
        let channels = usize::from(first);
        let chains: Vec<_> = self
            .members
            .iter()
            .map(|m| m.job.chain(images))
            .collect::<Result<_>>()?;
        let mips = chains[0].len();
        if chains.iter().any(|c| c.len() != mips) {
            return Err(self.fail("neural set members differ in mip count".into()));
        }
        // Per mip, the members' channels interleaved, linear; normals at 0.5 v + 0.5.
        let data: Vec<Vec<f32>> = (0..mips)
            .map(|mip| {
                let texels = chains[0][mip].px.len();
                let mut out = Vec::with_capacity(texels * channels);
                for t in 0..texels {
                    for (c, &(_, n, stored)) in chains.iter().zip(&layout) {
                        let px = c[mip].px[t];
                        for v in &px[..usize::from(n)] {
                            out.push(if stored == NeuralTexture::NORMAL {
                                0.5 * v + 0.5
                            } else {
                                v.clamp(0.0, 1.0)
                            });
                        }
                    }
                }
                out
            })
            .collect();
        let srgb: Vec<bool> = layout
            .iter()
            .flat_map(|&(_, n, stored)| (0..n).map(move |_| stored == NeuralTexture::SRGB))
            .collect();
        let bc: u64 = self
            .members
            .iter()
            .map(|m| bc_bytes(m.job.encoding(), self.size))
            .sum();
        let max_bpp = (bc * 8) as f32 / (u64::from(w) * u64::from(h)) as f32;
        let textures: Vec<libntc::Texture> = layout
            .iter()
            .zip(&self.members)
            .map(|(&(first, n, stored), m)| libntc::Texture {
                first_channel: i32::from(first),
                channel_count: i32::from(n),
                srgb: i32::from(stored == NeuralTexture::SRGB),
                format: match m.job.encoding() {
                    Encoding::Raw(_) => UNORM16,
                    _ => UNORM8,
                },
            })
            .collect();
        let out = {
            let _gpu = GPU
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            libntc::compress(&libntc::Request {
                size: self.size,
                channels,
                mips: &data,
                srgb: &srgb,
                textures: &textures,
                target_psnr: self.settings.target_psnr,
                max_bits_per_pixel: max_bpp,
                training_steps: self.settings.training_steps,
                seed: self.settings.seed,
                cuda_device: self.settings.cuda_device,
            })
            .map_err(|e| self.fail(format!("neural set: {e}")))?
        };

        // Per channel PSNR at mip 0 in the stored encoding.
        let reference = |c: usize| {
            let srgb = srgb[c];
            data[0]
                .iter()
                .skip(c)
                .step_by(channels)
                .map(move |&v| if srgb { linear_to_srgb(v) } else { v })
        };
        let decoded = |c: usize| out.decoded.iter().skip(c).step_by(channels).copied();
        let channel_psnr: Vec<f32> = (0..channels)
            .map(|c| psnr(reference(c), decoded(c)))
            .collect();

        let weight_bytes: u64 = out.weights.iter().map(|w| w.data.len() as u64).sum();
        let latent_bytes: u64 = out.latents.iter().flatten().map(|l| l.len() as u64).sum();
        let bytes = latent_bytes + weight_bytes;
        let mut report = NeuralReport {
            bits_per_pixel: out.bits_per_pixel,
            psnr: out.psnr,
            runs: out.runs,
            bytes,
            bc_bytes: bc,
            textures: Vec::new(),
            warnings: Vec::new(),
        };
        if bytes > bc {
            report.warnings.push(format!(
                "{bytes} bytes against {bc} for the BC textures; the set stands in for none"
            ));
        }
        let mut table = Vec::new();
        for (&(first, n, stored), m) in layout.iter().zip(&self.members) {
            let range = usize::from(first)..usize::from(first + n);
            let worst = channel_psnr[range.clone()]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let kept = bytes <= bc && worst >= self.settings.min_psnr;
            report.textures.push(NeuralTextureReport {
                name: m.job.name.clone(),
                psnr: channel_psnr[range].to_vec(),
                kept,
            });
            if kept {
                table.push(NeuralTexture {
                    texture: AssetId::from_name(&m.job.name),
                    first_channel: first,
                    channel_count: n,
                    stored,
                    _pad: 0,
                    psnr: worst,
                });
            }
        }

        let latents_name = format!("{}/latents", self.name);
        let latents = super::layout(
            vk::A4R4G4B4_UNORM_PACK16,
            [out.latent_size[0], out.latent_size[1]],
            false,
            Primaries::Rec709,
            f32::NAN,
            Vec::new(),
            &out.latents,
        );
        let lib = libntc::library().map_err(|e| self.fail(e))?;
        let info = NeuralInfo {
            header: NeuralHeader {
                library: lib.version,
                _pad: 0,
                interface: lib.interface,
                file_schema: lib.schema,
                width: w,
                height: h,
                mip_count: mips as u8,
                channel_count: first,
                latent_features: out.latent_features,
                _pad1: 0,
                bits_per_pixel: out.bits_per_pixel,
                latents: AssetId::from_name(&latents_name),
                textures: Span::default(),
                weights: Span::default(),
            },
            textures: table,
            weights: out
                .weights
                .iter()
                .enumerate()
                .map(|(i, wt)| {
                    (
                        NeuralWeights {
                            kind: wt.kind,
                            blob: 1 + i as u32,
                            converted_bytes: wt.converted_bytes,
                            constants: Span::default(),
                        },
                        wt.constants.clone(),
                    )
                })
                .collect(),
        };
        Ok(NeuralBuilt {
            info,
            weights: out.weights.into_iter().map(|w| w.data).collect(),
            latents,
            report,
        })
    }
}

/// Adds a cooked set and its latent texture.
pub(crate) fn add(w: &mut ArchiveWriter, name: &str, key: Hash32, set: &NeuralBuilt) -> Result<()> {
    let head = set.info.to_bytes();
    let mut blobs = vec![BlobIn {
        mount: true,
        ..BlobIn::new(&head)
    }];
    blobs.extend(set.weights.iter().map(|b| BlobIn {
        mount: true,
        ..BlobIn::new(b)
    }));
    w.add(
        name,
        Class::NeuralTextureSet as u16,
        orr_pack::neural::SCHEMA,
        key,
        &blobs,
    )?;
    super::add(w, &format!("{name}/latents"), key, &set.latents)?;
    Ok(())
}

/// LibNTC, loaded when the first set is planned.
mod libntc {
    /// A texture's channels in the set.
    #[repr(C)]
    pub(super) struct Texture {
        pub first_channel: i32,
        pub channel_count: i32,
        pub srgb: i32,
        pub format: i32,
    }

    #[cfg_attr(
        not(feature = "ntc"),
        expect(dead_code, reason = "without LibNTC no set is planned")
    )]
    pub(super) struct Request<'a> {
        pub size: [u32; 2],
        pub channels: usize,
        /// Per mip, texels of `channels` values, linear.
        pub mips: &'a [Vec<f32>],
        /// Per channel, whether the network stores it as sRGB.
        pub srgb: &'a [bool],
        pub textures: &'a [Texture],
        pub target_psnr: f32,
        pub max_bits_per_pixel: f32,
        pub training_steps: u32,
        pub seed: u32,
        pub cuda_device: i32,
    }

    pub(super) struct Weights {
        pub kind: u32,
        pub data: Vec<u8>,
        pub converted_bytes: u64,
        pub constants: Vec<u8>,
    }

    pub(super) struct Compressed {
        pub latent_size: [u32; 2],
        pub latent_features: u8,
        /// `latents[layer][mip]`, rows tight.
        pub latents: Vec<Vec<Vec<u8>>>,
        pub weights: Vec<Weights>,
        /// Mip 0 output, `channels` values per texel, sRGB channels encoded.
        pub decoded: Vec<f32>,
        pub bits_per_pixel: f32,
        pub psnr: f32,
        pub runs: u32,
    }

    /// The loaded library's versions.
    pub(super) struct Library {
        /// Major, minor, point.
        pub version: [u16; 3],
        pub interface: u32,
        /// NTC file schema version of the build's checkout.
        pub schema: u32,
        /// Library, interface and schema versions, for cook keys.
        pub key: String,
    }

    #[cfg(feature = "ntc")]
    pub(super) use loaded::{compress, library};

    #[cfg(not(feature = "ntc"))]
    pub(super) fn library() -> Result<&'static Library, String> {
        Err("neural texture sets need orr-cook's ntc feature".into())
    }

    #[cfg(not(feature = "ntc"))]
    pub(super) fn compress(_: &Request<'_>) -> Result<Compressed, String> {
        unreachable!("no set is planned without a library")
    }

    #[cfg(feature = "ntc")]
    mod loaded {
        use std::ffi::{CStr, c_char, c_void};
        use std::path::PathBuf;
        use std::sync::OnceLock;

        use super::{Compressed, Library, Request, Texture, Weights};

        #[repr(C)]
        struct Api {
            create_context: *const c_void,
            destroy_context: *const c_void,
            last_error: *const c_void,
            loss_to_psnr: *const c_void,
        }

        #[repr(C)]
        struct Input {
            width: i32,
            height: i32,
            mips: i32,
            channels: i32,
            mip_data: *const *const f32,
            channel_srgb: *const u8,
            texture_count: i32,
            textures: *const Texture,
            target_psnr: f32,
            max_bits_per_pixel: f32,
            training_steps: i32,
            seed: u32,
            cuda_device: i32,
        }

        #[repr(C)]
        #[derive(Default)]
        struct Output {
            latent_width: i32,
            latent_height: i32,
            latent_layers: i32,
            latent_mips: i32,
            latent_features: i32,
            bits_per_pixel: f32,
            psnr: f32,
            runs: i32,
        }

        #[repr(C)]
        struct VersionInfo {
            major: i32,
            minor: i32,
            point: i32,
            branch: *const c_char,
            commit: *const c_char,
        }

        const LATENT: i32 = 0;
        const WEIGHTS: i32 = 1;
        const CONSTANTS: i32 = 2;
        const DECODED: i32 = 3;

        type Sink = unsafe extern "C" fn(*mut c_void, i32, i32, *const c_void, usize, u64);

        unsafe extern "C" {
            fn orr_ntc_compress(
                api: *const Api,
                input: *const Input,
                out: *mut Output,
                sink: Sink,
                user: *mut c_void,
                error: *mut c_char,
                error_len: usize,
            ) -> i32;
        }

        struct Lib {
            api: Api,
            library: Library,
            _lib: libloading::Library,
        }

        // SAFETY: the pointers are plain functions of a library that stays loaded, callable
        // from any thread.
        unsafe impl Send for Lib {}
        // SAFETY: as for `Send`; nothing in `Lib` is mutated after loading.
        unsafe impl Sync for Lib {}

        /// `libntc.dll` (`libntc.so`) from `ORR_NTC_LIB`, else from the build's checkout.
        fn path() -> PathBuf {
            if let Some(p) = std::env::var_os("ORR_NTC_LIB").filter(|p| !p.is_empty()) {
                return p.into();
            }
            let sdk = PathBuf::from(env!("ORR_COOK_NTC_SDK"));
            if cfg!(windows) {
                sdk.join("bin/windows-x64/libntc.dll")
            } else {
                sdk.join("bin/linux-x64/libntc.so")
            }
        }

        fn load() -> Result<Lib, String> {
            let path = path();
            let fail = |e: libloading::Error| format!("{}: {e} (set ORR_NTC_LIB)", path.display());
            // SAFETY: LibNTC's initializers have no preconditions.
            let lib = unsafe { libloading::Library::new(&path) }.map_err(fail)?;
            let get = |name: &str| -> Result<*const c_void, String> {
                // SAFETY: the symbol is only called through its declared signature.
                unsafe { lib.get::<*const c_void>(name) }
                    .map(|s| *s)
                    .map_err(fail)
            };
            // SAFETY: both are `extern "C"` functions of these signatures in ntc.h.
            let (iface, version) = unsafe {
                let iface: libloading::Symbol<unsafe extern "C" fn() -> u32> =
                    lib.get("GetInterfaceVersion").map_err(fail)?;
                let version: libloading::Symbol<unsafe extern "C" fn() -> VersionInfo> =
                    lib.get("GetLibraryVersion").map_err(fail)?;
                (iface(), version())
            };
            let built = env!("ORR_COOK_NTC_INTERFACE");
            let interface = u32::from_str_radix(built.trim_start_matches("0x"), 16)
                .expect("hex interface version");
            if iface != interface {
                return Err(format!(
                    "{} has interface {iface:#x}; orr-cook was built against {interface:#x} from {}",
                    path.display(),
                    env!("ORR_COOK_NTC_SDK")
                ));
            }
            let schema = env!("ORR_COOK_NTC_SCHEMA").parse().expect("schema version");
            let text = |p: *const c_char| {
                if p.is_null() {
                    String::new()
                } else {
                    // SAFETY: LibNTC returns static NUL-terminated strings.
                    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
                }
            };
            let key = format!(
                "libntc {}.{}.{} {} {}, interface {iface:#x}, schema {schema}",
                version.major,
                version.minor,
                version.point,
                text(version.branch),
                text(version.commit),
            );
            let api = Api {
                create_context: get("CreateContext")?,
                destroy_context: get("DestroyContext")?,
                last_error: get("GetLastErrorMessage")?,
                loss_to_psnr: get("LossToPSNR")?,
            };
            let num = |v: i32| u16::try_from(v).unwrap_or(u16::MAX);
            Ok(Lib {
                api,
                library: Library {
                    version: [num(version.major), num(version.minor), num(version.point)],
                    interface,
                    schema,
                    key,
                },
                _lib: lib,
            })
        }

        fn lib() -> Result<&'static Lib, String> {
            static LIB: OnceLock<Result<Lib, String>> = OnceLock::new();
            LIB.get_or_init(load).as_ref().map_err(Clone::clone)
        }

        pub(in super::super) fn library() -> Result<&'static Library, String> {
            Ok(&lib()?.library)
        }

        unsafe extern "C" fn sink(
            user: *mut c_void,
            kind: i32,
            index: i32,
            data: *const c_void,
            len: usize,
            extra: u64,
        ) {
            // SAFETY: `user` is the `Compressed` passed to `orr_ntc_compress`, and `data`
            // holds `len` bytes for the duration of the call.
            let (c, bytes) = unsafe {
                (
                    &mut *user.cast::<Compressed>(),
                    std::slice::from_raw_parts(data.cast::<u8>(), len),
                )
            };
            let index = index as usize;
            match kind {
                LATENT => {
                    let layer = extra as usize;
                    if c.latents.len() <= layer {
                        c.latents.resize(layer + 1, Vec::new());
                    }
                    let level = &mut c.latents[layer];
                    if level.len() <= index {
                        level.resize(index + 1, Vec::new());
                    }
                    level[index] = bytes.to_vec();
                }
                WEIGHTS => c.weights.push(Weights {
                    kind: index as u32,
                    data: bytes.to_vec(),
                    converted_bytes: extra,
                    constants: Vec::new(),
                }),
                CONSTANTS => {
                    if let Some(w) = c.weights.iter_mut().find(|w| w.kind == index as u32) {
                        w.constants = bytes.to_vec();
                    }
                }
                DECODED => {
                    c.decoded = bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                }
                _ => {}
            }
        }

        pub(in super::super) fn compress(r: &Request<'_>) -> Result<Compressed, String> {
            let lib = lib()?;
            let mips: Vec<*const f32> = r.mips.iter().map(Vec::as_ptr).collect();
            let srgb: Vec<u8> = r.srgb.iter().map(|&s| u8::from(s)).collect();
            let input = Input {
                width: r.size[0] as i32,
                height: r.size[1] as i32,
                mips: mips.len() as i32,
                channels: r.channels as i32,
                mip_data: mips.as_ptr(),
                channel_srgb: srgb.as_ptr(),
                texture_count: r.textures.len() as i32,
                textures: r.textures.as_ptr(),
                target_psnr: r.target_psnr,
                max_bits_per_pixel: r.max_bits_per_pixel,
                training_steps: r.training_steps as i32,
                seed: r.seed,
                cuda_device: r.cuda_device,
            };
            let mut c = Compressed {
                latent_size: [0; 2],
                latent_features: 0,
                latents: Vec::new(),
                weights: Vec::new(),
                decoded: Vec::new(),
                bits_per_pixel: 0.0,
                psnr: 0.0,
                runs: 0,
            };
            let mut out = Output::default();
            let mut error = vec![0 as c_char; 1024];
            // SAFETY: every pointer in `input` outlives the call; the shim writes at most
            // `error.len()` bytes and calls `sink` with `c` only during the call.
            let status = unsafe {
                orr_ntc_compress(
                    &lib.api,
                    &input,
                    &mut out,
                    sink,
                    std::ptr::from_mut(&mut c).cast(),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if status != 0 {
                // SAFETY: the shim NUL-terminates its message.
                return Err(unsafe { CStr::from_ptr(error.as_ptr()) }
                    .to_string_lossy()
                    .into_owned());
            }
            let (layers, mips) = (out.latent_layers as usize, out.latent_mips as usize);
            if c.latents.len() != layers || c.latents.iter().any(|l| l.len() != mips) {
                return Err("LibNTC returned an incomplete latent texture".into());
            }
            if c.weights.is_empty() || c.weights.iter().any(|w| w.constants.is_empty()) {
                return Err("LibNTC returned no inference weights".into());
            }
            c.latent_size = [out.latent_width as u32, out.latent_height as u32];
            c.latent_features = out.latent_features as u8;
            c.bits_per_pixel = out.bits_per_pixel;
            c.psnr = out.psnr;
            c.runs = out.runs as u32;
            Ok(c)
        }
    }
}
