//! Density profiles, rays through them, and optical-depth integration.

use glam::DVec3;

use super::Geometry;
use crate::Atmosphere;

/// Rayleigh, aerosol, ozone and fog.
pub(super) const COMPONENTS: usize = 4;
pub(super) type Densities = [f64; COMPONENTS];

/// Integrals from 0 to each Gauss-Legendre node of the cubic through the four nodes: row j
/// holds the weights of the node values.
const GAUSS_PARTIAL: [[f64; 4]; 4] = [
    [
        0.086_963_711_284_363_48,
        -0.026_604_180_084_998_8,
        0.012_627_462_689_404_722,
        -0.003_555_149_685_795_681,
    ],
    [
        0.188_118_117_499_868_06,
        0.163_036_288_715_636_44,
        -0.027_880_428_602_470_87,
        0.006_735_500_594_538_164,
    ],
    [
        0.167_191_921_974_188_64,
        0.353_953_006_033_743_7,
        0.163_036_288_715_636_47,
        -0.014_190_694_931_141_001,
    ],
    [
        0.177_482_572_254_523_02,
        0.313_445_114_741_867_54,
        0.352_676_757_516_272_7,
        0.086_963_711_284_363_6,
    ],
];
/// Gauss-Legendre nodes on [0, 1] and their weights.
const GAUSS: [(f64, f64); 4] = [
    (0.069_431_844_202_973_71, 0.173_927_422_568_726_93),
    (0.330_009_478_207_571_87, 0.326_072_577_431_273_07),
    (0.669_990_521_792_428_1, 0.326_072_577_431_273_07),
    (0.930_568_155_797_026_3, 0.173_927_422_568_726_93),
];
/// Farthest a ray is followed in the plane geometry, m.
const PLANE_REACH: f64 = 1e7;
/// Longest segment, m.
const MAX_STEP: f64 = 50_000.0;
/// An exponential layer this many scale heights above its reference is treated as gone
/// when choosing segment lengths.
const NEGLIGIBLE_SCALES: f64 = 40.0;
/// Altitude change per segment, in scale heights.
const SEGMENT_RISE: f64 = 1.0;

/// Density shapes in altitude, each relative to the component's own unit.
pub(super) struct Profile {
    pub(super) radius: f64,
    pub(super) top: f64,
    rayleigh_scale: f64,
    aerosol_scale: f64,
    ozone_peak: f64,
    ozone_half_width: f64,
    fog_base: f64,
    fog_falloff: f64,
    active: [bool; COMPONENTS],
}

impl Profile {
    pub(super) fn new(a: &Atmosphere) -> Self {
        let fog = a.height_fog;
        Profile {
            radius: f64::from(a.planet_radius),
            top: f64::from(a.top_radius),
            rayleigh_scale: f64::from(a.rayleigh.scale_height),
            aerosol_scale: f64::from(a.aerosol.scale_height),
            ozone_peak: f64::from(a.ozone.peak_altitude),
            ozone_half_width: f64::from(a.ozone.half_width),
            fog_base: fog.map_or(0.0, |f| f64::from(f.base_height)),
            fog_falloff: fog.map_or(1.0, |f| f64::from(f.falloff_height)),
            active: [
                a.rayleigh.density > 0.0,
                a.aerosol.optical_depth_550 > 0.0,
                a.ozone.column > 0.0,
                fog.is_some_and(|f| f.density > 0.0),
            ],
        }
    }

    /// Height of the top of the atmosphere above the ground.
    pub(super) fn thickness(&self) -> f64 {
        self.top - self.radius
    }

    /// Rayleigh relative to the ground, aerosol per unit vertical column (1/m), ozone
    /// relative to its peak, fog relative to its base.
    pub(super) fn densities(&self, h: f64) -> Densities {
        [
            (-h / self.rayleigh_scale).exp(),
            (-h / self.aerosol_scale).exp() / self.aerosol_scale,
            self.tent(h),
            (-(h - self.fog_base) / self.fog_falloff).exp(),
        ]
    }

    fn tent(&self, h: f64) -> f64 {
        (1.0 - (h - self.ozone_peak).abs() / self.ozone_half_width).max(0.0)
    }

    /// The shortest scale height of the exponential layers present at altitude `h`.
    fn scale(&self, h: f64) -> f64 {
        let mut s = f64::INFINITY;
        if self.active[0] {
            s = s.min(self.rayleigh_scale);
        }
        if self.active[1] && h < NEGLIGIBLE_SCALES * self.aerosol_scale {
            s = s.min(self.aerosol_scale);
        }
        if self.active[3] && h - self.fog_base < NEGLIGIBLE_SCALES * self.fog_falloff {
            s = s.min(self.fog_falloff);
        }
        s
    }

    /// Splits `[t0, t1]` along `ray` into pieces over which the altitude changes by at most
    /// `SEGMENT_RISE` scale heights and which do not cross an ozone knot.
    pub(super) fn segments<'a>(&'a self, ray: &'a Ray, t0: f64, t1: f64) -> Segments<'a> {
        Segments {
            profile: self,
            ray,
            t: t0,
            end: t1,
        }
    }

    /// Integral of the densities along `ray` over `[t0, t1]`.
    pub(super) fn column(&self, ray: &Ray, t0: f64, t1: f64) -> Densities {
        let mut acc = [0.0; COMPONENTS];
        for seg in self.segments(ray, t0, t1) {
            let len = seg.tb - seg.ta;
            for (x, w) in GAUSS {
                let d = self.densities(ray.altitude(seg.ta + len * x));
                for c in 0..COMPONENTS {
                    acc[c] += w * len * d[c];
                }
            }
        }
        acc
    }
}

pub(super) struct Segment {
    pub(super) ta: f64,
    pub(super) tb: f64,
}

pub(super) struct Segments<'a> {
    profile: &'a Profile,
    ray: &'a Ray,
    t: f64,
    end: f64,
}

impl Iterator for Segments<'_> {
    type Item = Segment;

    fn next(&mut self) -> Option<Segment> {
        let ta = self.t;
        if ta >= self.end {
            return None;
        }
        let (p, ray) = (self.profile, self.ray);
        let a = ray.altitude(ta);
        let step = SEGMENT_RISE * p.scale(a);
        let floor = ta + 1e-6;
        let mut tb = self.end.min(ta + MAX_STEP);
        let knots = [
            p.ozone_peak - p.ozone_half_width,
            p.ozone_peak,
            p.ozone_peak + p.ozone_half_width,
        ];
        for h in [a + step, a - step].into_iter().chain(knots) {
            for t in ray.crossings(h) {
                if t > floor && t < tb {
                    tb = t;
                }
            }
        }
        self.t = tb;
        Some(Segment { ta, tb })
    }
}

/// A leg of a ray: its segments with each Gauss-Legendre node's densities and the column
/// from the leg's start to the node.
pub(super) struct Leg {
    pub(super) start: f64,
    pub(super) end: f64,
    pub(super) segments: Vec<LegSegment>,
    pub(super) total: Densities,
}

pub(super) struct LegSegment {
    ta: f64,
    /// Column from the leg's start to `ta`.
    before: Densities,
    pub(super) nodes: [Node; 4],
}

#[derive(Clone, Copy, Default)]
pub(super) struct Node {
    pub(super) t: f64,
    /// Quadrature weight times the segment length.
    pub(super) weight: f64,
    pub(super) altitude: f64,
    pub(super) densities: Densities,
    /// Column from the leg's start.
    pub(super) column: Densities,
}

impl Leg {
    pub(super) fn new(p: &Profile, ray: &Ray, start: f64, end: f64) -> Self {
        let mut total = [0.0; COMPONENTS];
        let segments = p
            .segments(ray, start, end)
            .map(|seg| {
                let len = seg.tb - seg.ta;
                let mut nodes = [Node::default(); 4];
                for (node, (x, w)) in nodes.iter_mut().zip(GAUSS) {
                    let t = seg.ta + len * x;
                    let altitude = ray.altitude(t);
                    *node = Node {
                        t,
                        weight: w * len,
                        altitude,
                        densities: p.densities(altitude),
                        column: [0.0; 4],
                    };
                }
                for j in 0..4 {
                    nodes[j].column = std::array::from_fn(|c| {
                        total[c]
                            + len
                                * (0..4)
                                    .map(|i| GAUSS_PARTIAL[j][i] * nodes[i].densities[c])
                                    .sum::<f64>()
                    });
                }
                let before = total;
                for node in &nodes {
                    for (t, d) in total.iter_mut().zip(node.densities) {
                        *t += node.weight * d;
                    }
                }
                LegSegment {
                    ta: seg.ta,
                    before,
                    nodes,
                }
            })
            .collect();
        Leg {
            start,
            end,
            segments,
            total,
        }
    }

    pub(super) fn column_to(&self, p: &Profile, ray: &Ray, t: f64) -> Densities {
        let i = self.segments.partition_point(|s| s.ta <= t).max(1) - 1;
        let seg = &self.segments[i];
        let part = p.column(ray, seg.ta, t);
        std::array::from_fn(|c| seg.before[c] + part[c])
    }

    /// The distance at which the optical depth for coefficients `ext` reaches `target`.
    pub(super) fn invert(&self, p: &Profile, ray: &Ray, ext: &Densities, target: f64) -> f64 {
        let depth = |c: &Densities| (0..COMPONENTS).map(|k| ext[k] * c[k]).sum::<f64>();
        let i = self
            .segments
            .partition_point(|s| depth(&s.before) <= target)
            .max(1)
            - 1;
        let seg = &self.segments[i];
        let ta = seg.ta;
        let tb = self.segments.get(i + 1).map_or(self.end, |s| s.ta);
        let rest = target - depth(&seg.before);
        let (mut lo, mut hi) = (ta, tb);
        let mut t = 0.5 * (ta + tb);
        for _ in 0..64 {
            let f = depth(&p.column(ray, ta, t)) - rest;
            if f > 0.0 {
                hi = t;
            } else {
                lo = t;
            }
            if f.abs() <= 1e-14 * target || hi - lo <= 1e-9 * hi.max(1.0) {
                break;
            }
            let next = t - f / depth(&p.densities(ray.altitude(t)));
            t = if next > lo && next < hi {
                next
            } else {
                0.5 * (lo + hi)
            };
        }
        t
    }
}

/// A ray in the local frame: the ground plane is y = 0 with +y up, and the planet's
/// centre is at (0, -radius, 0).
#[derive(Clone, Copy)]
pub(super) struct Ray {
    pub(super) origin: DVec3,
    pub(super) dir: DVec3,
    geometry: Geometry,
    radius: f64,
    /// Origin relative to the planet centre.
    q: DVec3,
}

impl Ray {
    pub(super) fn new(origin: DVec3, dir: DVec3, geometry: Geometry, radius: f64) -> Self {
        let dir = dir.normalize();
        Ray {
            origin,
            dir,
            geometry,
            radius,
            q: origin + DVec3::Y * radius,
        }
    }

    pub(super) fn geometry(&self) -> Geometry {
        self.geometry
    }

    pub(super) fn at(&self, t: f64) -> DVec3 {
        self.origin + self.dir * t
    }

    pub(super) fn altitude(&self, t: f64) -> f64 {
        match self.geometry {
            Geometry::Sphere => (self.q + self.dir * t).length() - self.radius,
            Geometry::Plane => self.origin.y + self.dir.y * t,
        }
    }

    pub(super) fn up(&self, t: f64) -> DVec3 {
        match self.geometry {
            Geometry::Sphere => (self.q + self.dir * t).normalize(),
            Geometry::Plane => DVec3::Y,
        }
    }

    /// Distances at which the ray passes altitude `h`; NaN where it does not.
    fn crossings(&self, h: f64) -> [f64; 2] {
        match self.geometry {
            Geometry::Sphere => {
                let b = self.q.dot(self.dir);
                let r = self.radius + h;
                let disc = b * b - (self.q.length_squared() - r * r);
                if disc < 0.0 {
                    [f64::NAN; 2]
                } else {
                    let s = disc.sqrt();
                    [-b - s, -b + s]
                }
            }
            Geometry::Plane => [(h - self.origin.y) / self.dir.y, f64::NAN],
        }
    }

    /// Where the ray leaves the atmosphere from t = 0, and whether it leaves through the
    /// ground. In the plane geometry a ray is followed for at most `PLANE_REACH`.
    pub(super) fn exit(&self, thickness: f64) -> (f64, bool) {
        match self.geometry {
            Geometry::Sphere => {
                let b = self.q.dot(self.dir);
                let qq = self.q.length_squared();
                let ground = b * b - (qq - self.radius * self.radius);
                if b < 0.0 && ground >= 0.0 {
                    return ((-b - ground.sqrt()).max(0.0), true);
                }
                let top = self.radius + thickness;
                (
                    (-b + (b * b - (qq - top * top)).max(0.0).sqrt()).max(0.0),
                    false,
                )
            }
            Geometry::Plane => {
                let (t, ground) = if self.dir.y < 0.0 {
                    (-self.origin.y / self.dir.y, true)
                } else {
                    ((thickness - self.origin.y) / self.dir.y, false)
                };
                if t > PLANE_REACH {
                    (PLANE_REACH, false)
                } else {
                    (t.max(0.0), ground)
                }
            }
        }
    }
}

/// Columns of the four components from a point to the top of the atmosphere, on
/// Bruneton's (r, mu) mapping of the spherical planet; rays that hit the ground have none.
pub(super) struct Columns {
    data: Vec<Densities>,
    radius: f64,
    top: f64,
}

const COLUMNS_R: usize = 256;
const COLUMNS_MU: usize = 1024;

impl Columns {
    pub(super) fn new(profile: &Profile) -> Self {
        let (radius, top) = (profile.radius, profile.top);
        let horizon = (top * top - radius * radius).sqrt();
        let data = super::parallel(COLUMNS_R * COLUMNS_MU, |i| {
            let (ir, im) = (i / COLUMNS_MU, i % COLUMNS_MU);
            let rho = horizon * ir as f64 / (COLUMNS_R - 1) as f64;
            let r = (rho * rho + radius * radius).sqrt();
            let (dmin, dmax) = (top - r, rho + horizon);
            let d = dmin + (dmax - dmin) * im as f64 / (COLUMNS_MU - 1) as f64;
            let mu = if d == 0.0 {
                1.0
            } else {
                ((horizon * horizon - rho * rho - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0)
            };
            let dir = DVec3::new((1.0 - mu * mu).max(0.0).sqrt(), mu, 0.0);
            let ray = Ray::new(DVec3::Y * (r - radius), dir, Geometry::Sphere, radius);
            profile.column(&ray, 0.0, d)
        });
        Columns { data, radius, top }
    }

    /// The column from altitude `h` along a direction whose cosine to the zenith is `mu`.
    pub(super) fn get(&self, h: f64, mu: f64) -> Option<Densities> {
        let (radius, top) = (self.radius, self.top);
        let r = (radius + h).clamp(radius, top);
        if mu < 0.0 && r * r * (mu * mu - 1.0) + radius * radius >= 0.0 {
            return None;
        }
        let horizon = (top * top - radius * radius).sqrt();
        let rho = (r * r - radius * radius).max(0.0).sqrt();
        let d = -r * mu + (r * r * (mu * mu - 1.0) + top * top).max(0.0).sqrt();
        let (dmin, dmax) = (top - r, rho + horizon);
        let xm = if dmax > dmin {
            ((d - dmin) / (dmax - dmin)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let fr = rho / horizon * (COLUMNS_R - 1) as f64;
        let fm = xm * (COLUMNS_MU - 1) as f64;
        let (ir, im) = (
            (fr as usize).min(COLUMNS_R - 2),
            (fm as usize).min(COLUMNS_MU - 2),
        );
        let (ar, am) = (fr - ir as f64, fm - im as f64);
        let at = |a: usize, b: usize| self.data[a * COLUMNS_MU + b];
        let (c00, c01, c10, c11) = (
            at(ir, im),
            at(ir, im + 1),
            at(ir + 1, im),
            at(ir + 1, im + 1),
        );
        Some(std::array::from_fn(|c| {
            (1.0 - ar) * ((1.0 - am) * c00[c] + am * c01[c])
                + ar * ((1.0 - am) * c10[c] + am * c11[c])
        }))
    }
}
