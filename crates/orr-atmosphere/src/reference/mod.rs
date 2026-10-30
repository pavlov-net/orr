//! f64 spectral reference for the atmosphere's GPU passes.
//!
//! [`Reference`] holds one [`Atmosphere`]'s medium; [`Reference::run`] evaluates it at a
//! spectral resolution ([`Channels`]): the 48 bins, the LUT passes' wavelength groups, or
//! the three Rec.2020 bands of the medium's `sample`. Single scattering is integrated by
//! Gauss-Legendre quadrature along the view ray. Multiple scattering (every path with two
//! or more scattering or ground events) is a Monte Carlo estimate: each path is shared by
//! all channels, which carry their own throughputs; scattering points are drawn from the
//! channels' mean free-flight distribution mixed with a uniform one, directions from the
//! phase function mixed with a lobe toward the lit sky, and every leg after the first adds
//! the single scattering along it by quadrature. Paths are traced until [`Precision`] is
//! met, spread over the machine's threads; the result does not depend on the thread count.
//! Light transmittance at scattering points comes from a column table on the spherical
//! planet.
//!
//! Positions are in a local frame: the ground plane is y = 0 with +y up and the planet's
//! centre is at (0, -planet_radius, 0). `planet_top` and `up` of the [`Atmosphere`] are
//! not used. Lights are directional and have no disc.

mod medium;
mod phase;

use glam::DVec3;

use self::medium::{COMPONENTS, Columns, Densities, Leg, Profile, Ray};
use self::phase::{FogPhase, Rng};
use crate::colour::{REC709_TO_REC2020, REC2020_TO_XYZ, mul};
use crate::spectrum::{self, Bands, rgb_at};
use crate::{Atmosphere, DOBSON_UNIT};

/// Interactions after which a path ends.
const MAX_INTERACTIONS: u32 = 256;
/// Paths between precision checks.
const PATHS_PER_ROUND: u32 = 2048;
/// Length over which scattering points are also drawn uniformly, m.
const UNIFORM_SPAN: f64 = 300_000.0;
/// Lowest elevation of the light lobe's axis, radians.
const LOBE_FLOOR: f64 = 0.035;

/// How altitude follows from position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Geometry {
    /// Altitude above the spherical planet; the zenith points away from its centre.
    Sphere,
    /// Altitude above the flat ground plane, zenith +y everywhere: the medium's model.
    /// Light transmittance still comes from the spherical planet at that altitude.
    Plane,
}

/// Spectral resolution of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channels {
    /// The 48 bins of 10 nm.
    Spectral,
    /// The LUT passes' wavelength groups.
    Groups,
    /// The three solar-weighted Rec.2020 bands, with light transmittance from the groups
    /// summed to Rec.2020, as the medium's `sample`, finite `t` and transmittance LUT give.
    Rgb,
}

/// A directional light.
#[derive(Clone, Copy, Debug)]
pub struct Light {
    /// Unit direction toward the light.
    pub direction: DVec3,
    /// Linear Rec.2020 colour relative to the solar colour; (1, 1, 1) is sunlight.
    pub colour: [f64; 3],
}

impl Light {
    /// A light with the solar spectrum.
    pub fn solar(direction: DVec3) -> Self {
        Light {
            direction: direction.normalize(),
            colour: [1.0; 3],
        }
    }
}

/// When the multiple-scattering estimate stops.
#[derive(Clone, Copy, Debug)]
pub struct Precision {
    /// Stop once the standard error of each of X, Y and Z is below this fraction of the
    /// value.
    pub relative_error: f64,
    /// Stop after this many paths regardless.
    pub max_samples: u32,
}

impl Default for Precision {
    /// 0.5%, at most 2^24 paths.
    fn default() -> Self {
        Precision {
            relative_error: 0.005,
            max_samples: 1 << 24,
        }
    }
}

/// In-scattered radiance per lux of top-of-atmosphere illuminance.
///
/// Per-channel values are radiance over the light's spectral irradiance (1/sr): times the
/// channel's `rec2020` row they give nits per lux.
#[derive(Clone, Debug)]
pub struct Radiance {
    /// Single plus multiple scattering, per channel.
    pub values: Vec<f64>,
    /// CIE XYZ of `values`, cd/m^2 per lux.
    pub xyz: [f64; 3],
    /// Standard error of `xyz`.
    pub std_error: [f64; 3],
    /// Paths traced.
    pub samples: u32,
}

/// In-scatter and transmittance along a finite segment.
#[derive(Clone, Debug)]
pub struct Segment {
    pub inscatter: Radiance,
    /// Per channel.
    pub transmittance: Vec<f64>,
}

/// One atmosphere's medium, at any spectral resolution.
pub struct Reference {
    atmosphere: Atmosphere,
    profile: Profile,
    columns: Columns,
    fog_phase: FogPhase,
}

impl Reference {
    /// Builds the light-transmittance column table (a few hundred thousand optical-depth
    /// integrals, spread over the machine's threads).
    pub fn new(atmosphere: &Atmosphere) -> Self {
        let profile = Profile::new(atmosphere);
        let columns = Columns::new(&profile);
        let fog_phase = FogPhase::new(
            atmosphere
                .height_fog
                .map_or(10.0, |f| f64::from(f.droplet_diameter)),
        );
        Reference {
            atmosphere: *atmosphere,
            profile,
            columns,
            fog_phase,
        }
    }

    pub fn run(&self, channels: Channels) -> Run<'_> {
        let a = &self.atmosphere;
        let alpha = f64::from(a.aerosol.angstrom_exponent);
        let chans = match channels {
            Channels::Spectral => self.chans(&spectrum::SPECTRAL, |k| {
                (spectrum::SPECTRAL.wavelength[k] / 550.0).powf(-alpha)
            }),
            Channels::Groups => self.chans(&spectrum::GROUPS, |k| {
                spectrum::GROUPS.aerosol_factor(k, alpha)
            }),
            Channels::Rgb => self.chans(&spectrum::RGB, |k| spectrum::RGB.aerosol_factor(k, alpha)),
        };
        let light_groups = (channels == Channels::Rgb).then(|| {
            self.chans(&spectrum::GROUPS, |k| {
                spectrum::GROUPS.aerosol_factor(k, alpha)
            })
        });
        Run {
            reference: self,
            channels,
            chans,
            light_groups,
        }
    }

    fn chans<const N: usize>(
        &self,
        bands: &Bands<N>,
        aerosol_factor: impl Fn(usize) -> f64,
    ) -> Vec<Chan> {
        let a = &self.atmosphere;
        let rec2020 = |c: glam::Vec3| mul(&REC709_TO_REC2020, c.as_dvec3().to_array());
        let ground = rec2020(a.ground_albedo);
        let (fog_ext, fog_albedo) = a.height_fog.map_or((0.0, [0.0; 3]), |f| {
            (f64::from(f.density), rec2020(f.albedo))
        });
        let ozone_peak = f64::from(a.ozone.column) * DOBSON_UNIT / f64::from(a.ozone.half_width);
        (0..N)
            .map(|k| {
                let per_channel = |rgb: [f64; 3]| {
                    if N == 3 {
                        rgb[k]
                    } else {
                        rgb_at(bands.wavelength[k], rgb)
                    }
                };
                let rayleigh = bands.rayleigh[k] * f64::from(a.rayleigh.density);
                let aerosol = f64::from(a.aerosol.optical_depth_550) * aerosol_factor(k);
                Chan {
                    ext: [
                        rayleigh,
                        aerosol,
                        bands.ozone_cross_section[k] * ozone_peak,
                        fog_ext,
                    ],
                    sca: [
                        rayleigh,
                        aerosol * f64::from(a.aerosol.single_scattering_albedo),
                        0.0,
                        fog_ext * per_channel(fog_albedo),
                    ],
                    rec2020: bands.rec2020[k],
                    wavelength: bands.wavelength[k],
                    ground_albedo: per_channel(ground),
                }
            })
            .collect()
    }
}

/// `f` over `0..n` on the machine's threads, in index order.
fn parallel<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk = n.div_ceil(threads).max(1);
    let f = &f;
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|first| {
                scope.spawn(move || (first..n.min(first + chunk)).map(f).collect::<Vec<_>>())
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap())
            .collect()
    })
}

struct Chan {
    ext: Densities,
    sca: Densities,
    rec2020: [f64; 3],
    wavelength: f64,
    ground_albedo: f64,
}

fn dot(a: &Densities, b: &Densities) -> f64 {
    (0..COMPONENTS).map(|c| a[c] * b[c]).sum()
}

/// A [`Reference`] at one spectral resolution.
pub struct Run<'a> {
    reference: &'a Reference,
    channels: Channels,
    chans: Vec<Chan>,
    /// The group channels that give the Rec.2020 bands their light transmittance.
    light_groups: Option<Vec<Chan>>,
}

impl Run<'_> {
    /// Linear Rec.2020 of per-channel values, nits per lux for radiance.
    pub fn rec2020(&self, values: &[f64]) -> [f64; 3] {
        let mut out = [0.0; 3];
        for (v, ch) in values.iter().zip(&self.chans) {
            for (o, w) in out.iter_mut().zip(ch.rec2020) {
                *o += v * w;
            }
        }
        out
    }

    /// CIE XYZ of per-channel values.
    pub fn xyz(&self, values: &[f64]) -> [f64; 3] {
        mul(&REC2020_TO_XYZ, self.rec2020(values))
    }

    /// Transmittance from `origin` along `dir` over `t` metres (infinite: to space, zero
    /// when the ray reaches the ground), by numerical integration of the optical depth.
    pub fn transmittance(&self, origin: DVec3, dir: DVec3, t: f64, geometry: Geometry) -> Vec<f64> {
        let p = &self.reference.profile;
        let ray = Ray::new(origin, dir, geometry, p.radius);
        let (exit, ground) = ray.exit(p.thickness());
        if t > exit && ground {
            return vec![0.0; self.chans.len()];
        }
        let column = p.column(&ray, 0.0, t.min(exit));
        self.chans
            .iter()
            .map(|ch| (-dot(&ch.ext, &column)).exp())
            .collect()
    }

    /// Sky radiance seen from `eye` along `dir` on the spherical planet, arriving at the
    /// point `start` metres along the ray, from the medium beyond it. Below the horizon it
    /// includes the virtual ground and the medium in front of it.
    pub fn sky(
        &self,
        eye: DVec3,
        dir: DVec3,
        light: &Light,
        start: f64,
        precision: &Precision,
    ) -> Radiance {
        let p = &self.reference.profile;
        let ray = Ray::new(eye, dir, Geometry::Sphere, p.radius);
        let (exit, ground) = ray.exit(p.thickness());
        self.inscatter(&ray, start.min(exit), exit, ground, light, precision)
            .0
    }

    /// In-scatter toward `origin` and transmittance along `length` metres of `dir`. A
    /// segment that reaches the ground stops there and gets no light from it.
    pub fn segment(
        &self,
        origin: DVec3,
        dir: DVec3,
        length: f64,
        light: &Light,
        geometry: Geometry,
        precision: &Precision,
    ) -> Segment {
        let p = &self.reference.profile;
        let ray = Ray::new(origin, dir, geometry, p.radius);
        let end = length.min(ray.exit(p.thickness()).0);
        let (inscatter, transmittance) = self.inscatter(&ray, 0.0, end, false, light, precision);
        Segment {
            inscatter,
            transmittance,
        }
    }

    fn tint(&self, light: &Light) -> Vec<f64> {
        self.chans
            .iter()
            .enumerate()
            .map(|(k, ch)| {
                if self.channels == Channels::Rgb {
                    light.colour[k]
                } else {
                    rgb_at(ch.wavelength, light.colour)
                }
            })
            .collect()
    }

    /// Light transmittance per channel at altitude `h` toward a light at cosine `mu` to the
    /// zenith, into `out`; false when the planet hides the light. The Rec.2020 bands take it
    /// from the groups, as the medium reads it from the group-built transmittance LUT.
    fn light_transmittance(&self, h: f64, mu: f64, out: &mut [f64]) -> bool {
        let Some(column) = self.reference.columns.get(h, mu) else {
            return false;
        };
        match &self.light_groups {
            None => {
                for (o, ch) in out.iter_mut().zip(&self.chans) {
                    *o = (-dot(&ch.ext, &column)).exp();
                }
            }
            Some(groups) => {
                out.fill(0.0);
                for g in groups {
                    let t = (-dot(&g.ext, &column)).exp();
                    for (c, o) in out.iter_mut().enumerate() {
                        *o += t * g.rec2020[c] / spectrum::SOLAR_COLOUR[c];
                    }
                }
            }
        }
        true
    }

    /// Per-component phase values for the cosine between the travel and light directions.
    fn phases(&self, cos: f64) -> Densities {
        let g = f64::from(self.reference.atmosphere.aerosol.asymmetry);
        [
            phase::rayleigh(cos),
            phase::cornette_shanks(g, cos),
            0.0,
            self.reference.fog_phase.eval(cos),
        ]
    }

    /// Radiance over `[ta, tb]` of `ray` (plus the ground at `tb` when `ground`), and the
    /// transmittance over the same range.
    fn inscatter(
        &self,
        ray: &Ray,
        ta: f64,
        tb: f64,
        ground: bool,
        light: &Light,
        precision: &Precision,
    ) -> (Radiance, Vec<f64>) {
        let tint = self.tint(light);
        let view = &Leg::new(&self.reference.profile, ray, ta, tb);
        let transmittance = self
            .chans
            .iter()
            .map(|ch| (-dot(&ch.ext, &view.total)).exp())
            .collect();
        let single = self.single(ray, view, ground, light);
        let single: Vec<f64> = single.iter().zip(&tint).map(|(s, t)| s * t).collect();
        let single_xyz = self.xyz(&single);

        let n = self.chans.len();
        let xyz_w: Vec<[f64; 3]> = self
            .chans
            .iter()
            .zip(&tint)
            .map(|(ch, t)| mul(&REC2020_TO_XYZ, ch.rec2020).map(|w| w * t))
            .collect();
        let (mut sum, mut sum_xyz, mut sum_xyz2) = (vec![0.0; n], [0.0; 3], [0.0; 3]);
        let mut samples = 0u32;
        let mut std_error = [0.0; 3];
        while samples < precision.max_samples {
            // Rounds have a fixed length and are summed in sample order, so the estimate does
            // not depend on the thread count or scheduling.
            let round = PATHS_PER_ROUND.min(precision.max_samples - samples);
            let paths = parallel(round as usize, |i| {
                self.path(ray, view, ground, light, samples + i as u32)
            });
            for v in &paths {
                let mut x = [0.0; 3];
                for k in 0..n {
                    sum[k] += v[k];
                    for j in 0..3 {
                        x[j] += v[k] * xyz_w[k][j];
                    }
                }
                for j in 0..3 {
                    sum_xyz[j] += x[j];
                    sum_xyz2[j] += x[j] * x[j];
                }
            }
            samples += round;
            let m = f64::from(samples);
            let mut done = true;
            for j in 0..3 {
                let mean = sum_xyz[j] / m;
                let var = (sum_xyz2[j] / m - mean * mean).max(0.0) / (m - 1.0);
                std_error[j] = var.sqrt();
                if std_error[j] > precision.relative_error * (single_xyz[j] + mean).abs() {
                    done = false;
                }
            }
            if done {
                break;
            }
        }
        // With no paths, single scattering alone.
        let m = f64::from(samples.max(1));
        let values: Vec<f64> = single
            .iter()
            .zip(&sum)
            .zip(&tint)
            .map(|((s, v), t)| s + v / m * t)
            .collect();
        let xyz = self.xyz(&values);
        (
            Radiance {
                values,
                xyz,
                std_error,
                samples,
            },
            transmittance,
        )
    }

    /// Single scattering over `leg` of `ray` (plus the ground at its end when `ground`),
    /// before the light's tint.
    fn single(&self, ray: &Ray, leg: &Leg, ground: bool, light: &Light) -> Vec<f64> {
        let l = light.direction;
        let phases = self.phases(ray.dir.dot(l));
        let n = self.chans.len();
        let (mut out, mut lit) = (vec![0.0; n], vec![0.0; n]);
        for seg in &leg.segments {
            for node in &seg.nodes {
                if !self.light_transmittance(node.altitude, ray.up(node.t).dot(l), &mut lit) {
                    continue;
                }
                for (k, ch) in self.chans.iter().enumerate() {
                    let source: f64 = (0..COMPONENTS)
                        .map(|c| ch.sca[c] * node.densities[c] * phases[c])
                        .sum();
                    out[k] += node.weight * source * (-dot(&ch.ext, &node.column)).exp() * lit[k];
                }
            }
        }
        let mu = ray.up(leg.end).dot(l);
        if ground && self.light_transmittance(0.0, mu, &mut lit) {
            for (k, ch) in self.chans.iter().enumerate() {
                out[k] += (-dot(&ch.ext, &leg.total)).exp() * ch.ground_albedo
                    / std::f64::consts::PI
                    * mu.max(0.0)
                    * lit[k];
            }
        }
        out
    }

    /// One path's multiple scattering per channel, before the light's tint. All channels
    /// share the path and each carries its own throughput. Every leg after the first adds
    /// the single scattering arriving along it, so the path counts light from its second
    /// interaction on.
    fn path(&self, first: &Ray, view: &Leg, ground: bool, light: &Light, sample: u32) -> Vec<f64> {
        let p = &self.reference.profile;
        let n = self.chans.len();
        let (mut throughput, mut sum) = (vec![1.0; n], vec![0.0; n]);
        let (mut ray, mut ends_on_ground) = (*first, ground);
        let mut later: Option<Leg> = None;
        for i in 0..MAX_INTERACTIONS {
            let leg = later.as_ref().unwrap_or(view);
            if i > 0 {
                let single = self.single(&ray, leg, ends_on_ground, light);
                for k in 0..n {
                    sum[k] += throughput[k] * single[k];
                }
            }
            let mut rng = Rng::new(sample, i);
            let Some((pos, dir)) = self.interact(
                &ray,
                leg,
                ends_on_ground,
                light.direction,
                &mut throughput,
                &mut rng,
            ) else {
                break;
            };
            let largest = throughput.iter().copied().fold(0.0, f64::max);
            if largest <= 0.0 {
                break;
            }
            if i >= 2 && largest < 0.25 {
                let survive = largest / 0.25;
                if rng.next() >= survive {
                    break;
                }
                throughput.iter_mut().for_each(|b| *b /= survive);
            }
            ray = Ray::new(pos, dir, first.geometry(), p.radius);
            let (exit, g) = ray.exit(p.thickness());
            ends_on_ground = g;
            later = Some(Leg::new(p, &ray, 0.0, exit));
        }
        sum
    }

    /// The next interaction along `leg` of `ray`: a scattering point, or the ground at its
    /// end when `ground`. A leg into space always scatters; the throughput carries the
    /// probability. The event is drawn from the average of the channels' distributions and
    /// each channel's throughput is weighted by its own over that average. Returns the point
    /// and the new direction, or `None` when no channel can interact.
    fn interact(
        &self,
        ray: &Ray,
        leg: &Leg,
        ground: bool,
        light: DVec3,
        throughput: &mut [f64],
        rng: &mut Rng,
    ) -> Option<(DVec3, DVec3)> {
        let p = &self.reference.profile;
        let range = (leg.start, leg.end);
        // An empty leg that ends on the ground can only reflect.
        if leg.segments.is_empty() {
            if !ground {
                return None;
            }
            for (b, ch) in throughput.iter_mut().zip(&self.chans) {
                *b *= ch.ground_albedo;
            }
            return Some((
                ray.at(range.1),
                phase::cosine_direction(ray.up(range.1), rng),
            ));
        }
        let tau: Vec<f64> = self
            .chans
            .iter()
            .map(|ch| dot(&ch.ext, &leg.total))
            .collect();
        let usable: Vec<usize> = (0..self.chans.len())
            .filter(|&k| ground || tau[k] > 0.0)
            .collect();
        if usable.is_empty() {
            return None;
        }
        let share = 1.0 / usable.len() as f64;
        // Half the events follow the channels' own distributions, half are a distance drawn
        // uniformly over the first UNIFORM_SPAN metres of the leg.
        let span = (range.1 - range.0).min(UNIFORM_SPAN);
        let t = if rng.next() < 0.5 {
            let j = usable[((rng.next() * usable.len() as f64) as usize).min(usable.len() - 1)];
            if ground && rng.next() < (-tau[j]).exp() {
                let mass = 0.5 * share * usable.iter().map(|&k| (-tau[k]).exp()).sum::<f64>();
                for (k, ch) in self.chans.iter().enumerate() {
                    throughput[k] *= (-tau[k]).exp() / mass * ch.ground_albedo;
                }
                let n = ray.up(range.1);
                return Some((ray.at(range.1), phase::cosine_direction(n, rng)));
            }
            let target = -(-rng.next() * -(-tau[j]).exp_m1()).ln_1p();
            leg.invert(p, ray, &self.chans[j].ext, target)
        } else {
            range.0 + rng.next() * span
        };
        // Scattering on a ground leg has density sigma_t T; on a leg into space it is
        // conditioned on happening, so it is divided by 1 - T over the leg.
        let scatter = |k: usize| if ground { 1.0 } else { -(-tau[k]).exp_m1() };
        let column = leg.column_to(p, ray, t);
        let d = p.densities(ray.altitude(t));
        let pdf = 0.5
            * share
            * usable
                .iter()
                .map(|&k| {
                    let ext = &self.chans[k].ext;
                    dot(ext, &d) * (-dot(ext, &column)).exp() / scatter(k)
                })
                .sum::<f64>()
            + if t <= range.0 + span { 0.5 / span } else { 0.0 };

        // The new direction comes from the phase function (a component by its mean share of
        // sigma_s over the channels) or from a lobe around the light, half the time each;
        // paths into the lit sky carry the twilight and the aureole.
        let mut shares = [0.0; COMPONENTS];
        let mut counted = 0.0;
        for ch in &self.chans {
            let sigma_s = dot(&ch.sca, &d);
            if sigma_s > 0.0 {
                for (s, c) in shares.iter_mut().zip(0..COMPONENTS) {
                    *s += ch.sca[c] * d[c] / sigma_s;
                }
                counted += 1.0;
            }
        }
        if counted == 0.0 || pdf <= 0.0 {
            return None;
        }
        shares.iter_mut().for_each(|s| *s /= counted);
        // A light below the local horizon lights the sky above the horizon in its azimuth.
        let up = ray.up(t);
        let below = (LOBE_FLOOR.sin() - light.dot(up)).max(0.0);
        let lobe = if below > 0.0 {
            let flat = (light - up * light.dot(up)).normalize_or(up);
            (flat * LOBE_FLOOR.cos() + up * LOBE_FLOOR.sin()).normalize()
        } else {
            light
        };
        let g = f64::from(self.reference.atmosphere.aerosol.asymmetry);
        let dir = if rng.next() < 0.5 {
            let mut pick = rng.next();
            let component = (0..COMPONENTS)
                .find(|&c| {
                    pick -= shares[c];
                    pick < 0.0
                })
                .unwrap_or(0);
            let cos = match component {
                0 => phase::sample_cornette_shanks(0.0, rng),
                1 => phase::sample_cornette_shanks(g, rng),
                _ => self.reference.fog_phase.sample(rng),
            };
            phase::around(ray.dir, cos, rng)
        } else {
            phase::around(lobe, phase::sample_toward_light(rng), rng)
        };
        let phases = self.phases(ray.dir.dot(dir));
        let pdf_phase: f64 = (0..COMPONENTS).map(|c| shares[c] * phases[c]).sum();
        let pdf_dir = 0.5 * (pdf_phase + phase::toward_light(dir.dot(lobe)));
        for (k, ch) in self.chans.iter().enumerate() {
            let scattered: f64 = (0..COMPONENTS).map(|c| ch.sca[c] * d[c] * phases[c]).sum();
            throughput[k] *= scattered * (-dot(&ch.ext, &column)).exp() / (pdf * pdf_dir);
        }
        Some((ray.at(t), dir))
    }
}
