// The glare band pyramid's filters on the CPU, one channel in f64: the same taps, sample
// positions and level sizes as `post/glare.slang`, with exact bilinear sampling and no
// storage rounding. Shared by the build script, which fits band weights to its kernels, and
// the output suite (through `include!`), so it has no inner attributes or doc comments.

// Halvings to the chain's last level: 1/256 of the extent.
pub const LAST_LEVEL: u32 = 8;
// The largest value the chain stores (RGBA16F).
pub const STORAGE_MAX: f64 = 65504.0;

// One level of the chain: `size` texels, of which the window at `origin` is held; texels
// outside the window are zero, texel indices outside the level clamp to its edge.
#[derive(Clone, Debug)]
pub struct Plane {
    pub size: [u32; 2],
    pub origin: [i64; 2],
    pub window: [usize; 2],
    pub data: Vec<f64>,
}

impl Plane {
    // A whole level, rows top to bottom.
    pub fn dense(size: [u32; 2], data: Vec<f64>) -> Self {
        assert_eq!(data.len(), (size[0] * size[1]) as usize);
        Self {
            size,
            origin: [0, 0],
            window: [size[0] as usize, size[1] as usize],
            data,
        }
    }

    // A level of `size` that is zero except `value` at texel `at`.
    pub fn impulse(size: [u32; 2], at: [i64; 2], value: f64) -> Self {
        Self {
            size,
            origin: at,
            window: [1, 1],
            data: vec![value],
        }
    }

    pub fn get(&self, x: i64, y: i64) -> f64 {
        let x = x.clamp(0, i64::from(self.size[0]) - 1) - self.origin[0];
        let y = y.clamp(0, i64::from(self.size[1]) - 1) - self.origin[1];
        if x < 0 || y < 0 || x >= self.window[0] as i64 || y >= self.window[1] as i64 {
            return 0.0;
        }
        self.data[y as usize * self.window[0] + x as usize]
    }

    // Bilinear sample at texel coordinates `p` (texel centers at i + 0.5).
    fn sample(&self, p: [f64; 2]) -> f64 {
        let (tx, ty) = (p[0] - 0.5, p[1] - 0.5);
        let (x, y) = (tx.floor(), ty.floor());
        let (fx, fy) = (tx - x, ty - y);
        let (x, y) = (x as i64, y as i64);
        let top = self.get(x, y) * (1.0 - fx) + self.get(x + 1, y) * fx;
        let bottom = self.get(x, y + 1) * (1.0 - fx) + self.get(x + 1, y + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    // Texel indices `[x0, x1)` and `[y0, y1)` of the window.
    fn bounds(&self) -> [[i64; 2]; 2] {
        [
            [self.origin[0], self.origin[0] + self.window[0] as i64],
            [self.origin[1], self.origin[1] + self.window[1] as i64],
        ]
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }
}

// Texel indices `[first, last)` of a level of `size` whose taps, `reach` of its own texels
// around each center, can touch the window `[lo, hi)` of a level `ratio` times coarser per
// texel.
fn reached(lo: i64, hi: i64, ratio: f64, reach: f64, size: u32) -> [i64; 2] {
    let first = ((lo as f64 - 1.0) * ratio - reach).floor() as i64 - 1;
    let last = ((hi as f64 + 1.0) * ratio + reach).ceil() as i64 + 1;
    [first.max(0), last.min(i64::from(size))]
}

fn fill(size: [u32; 2], xs: [i64; 2], ys: [i64; 2], f: impl Fn(i64, i64) -> f64) -> Plane {
    let window = [
        (xs[1] - xs[0]).max(0) as usize,
        (ys[1] - ys[0]).max(0) as usize,
    ];
    let mut data = Vec::with_capacity(window[0] * window[1]);
    for y in ys[0]..ys[1] {
        for x in xs[0]..xs[1] {
            data.push(f(x, y));
        }
    }
    Plane {
        size,
        origin: [xs[0], ys[0]],
        window,
        data,
    }
}

// The 13-tap downsample of the Jimenez pyramid to a level of `size`: bilinear taps at 0,
// +-1 and +-2 times `stride` source texels around each texel's center, weighted as five 2x2
// boxes (center 1/2, corners 1/8 each). On the first step (`first` = the exposure) each tap
// is scaled and clamped to [0, STORAGE_MAX].
pub fn downsample(src: &Plane, size: [u32; 2], stride: f64, first: Option<f64>) -> Plane {
    let ratio = [
        f64::from(src.size[0]) / f64::from(size[0]),
        f64::from(src.size[1]) / f64::from(size[1]),
    ];
    let [bx, by] = src.bounds();
    let reach = 3.0 * stride;
    let xs = reached(bx[0], bx[1], 1.0 / ratio[0], reach / ratio[0], size[0]);
    let ys = reached(by[0], by[1], 1.0 / ratio[1], reach / ratio[1], size[1]);
    let tap = |c: [f64; 2], o: [f64; 2]| {
        let v = src.sample([c[0] + o[0] * stride, c[1] + o[1] * stride]);
        match first {
            Some(scale) => {
                let v = v * scale;
                if v > 0.0 { v.min(STORAGE_MAX) } else { 0.0 }
            }
            None => v,
        }
    };
    fill(size, xs, ys, |x, y| {
        let c = [(x as f64 + 0.5) * ratio[0], (y as f64 + 0.5) * ratio[1]];
        let t = |ox: f64, oy: f64| tap(c, [ox, oy]);
        let center = t(0.0, 0.0);
        let inner = t(-1.0, -1.0) + t(1.0, -1.0) + t(-1.0, 1.0) + t(1.0, 1.0);
        let edges = t(0.0, -2.0) + t(-2.0, 0.0) + t(2.0, 0.0) + t(0.0, 2.0);
        let corners = t(-2.0, -2.0) + t(2.0, -2.0) + t(-2.0, 2.0) + t(2.0, 2.0);
        0.125 * center + 0.125 * inner + 0.0625 * edges + 0.03125 * corners
    })
}

// `w * own + coarse_weight * tent(coarse)` at a level of `size`: the 3x3 tent (1 2 1) of
// bilinear samples of `coarse` one texel of this level apart.
pub fn upsample(
    own: Option<(&Plane, f64)>,
    coarse: &Plane,
    coarse_weight: f64,
    size: [u32; 2],
) -> Plane {
    let ratio = [
        f64::from(size[0]) / f64::from(coarse.size[0]),
        f64::from(size[1]) / f64::from(coarse.size[1]),
    ];
    let [cx, cy] = coarse.bounds();
    let mut xs = reached(cx[0], cx[1], ratio[0], 2.0, size[0]);
    let mut ys = reached(cy[0], cy[1], ratio[1], 2.0, size[1]);
    if let Some((p, _)) = own {
        let [ox, oy] = p.bounds();
        xs = [
            xs[0].min(ox[0].max(0)),
            xs[1].max(ox[1].min(i64::from(size[0]))),
        ];
        ys = [
            ys[0].min(oy[0].max(0)),
            ys[1].max(oy[1].min(i64::from(size[1]))),
        ];
    }
    fill(size, xs, ys, |x, y| {
        let mut tent = 0.0;
        for dy in -1..=1_i64 {
            for dx in -1..=1_i64 {
                let w = ((2 - dx.abs()) * (2 - dy.abs())) as f64 / 16.0;
                let p = [
                    (x as f64 + 0.5 + dx as f64) / ratio[0],
                    (y as f64 + 0.5 + dy as f64) / ratio[1],
                ];
                tent += w * coarse.sample(p);
            }
        }
        own.map_or(0.0, |(p, w)| w * p.get(x, y)) + coarse_weight * tent
    })
}

// Level sizes of the chain for `extent` starting `first_level` halvings down: mip m is
// mip 0 halved m times, rounding down as texture mips do, and the chain ends at LAST_LEVEL
// or at a level one texel across.
pub fn levels(extent: [u32; 2], first_level: u32) -> Vec<[u32; 2]> {
    let first = extent.map(|n| (n >> first_level).max(1));
    let max = (LAST_LEVEL + 1 - first_level) as usize;
    let fit = (32 - first[0].min(first[1]).leading_zeros()) as usize;
    (0..max.min(fit))
        .map(|m| first.map(|n| (n >> m).max(1)))
        .collect()
}

// The chain's downsampled levels of `input`, the first step's taps scaled by `scale`.
pub fn downsamples(input: &Plane, first_level: u32, scale: f64) -> Vec<Plane> {
    let sizes = levels(input.size, first_level);
    let stride = f64::from(1u32 << (first_level - 1));
    let mut downs = vec![downsample(input, sizes[0], stride, Some(scale))];
    for &size in &sizes[1..] {
        let next = downsample(downs.last().unwrap(), size, 1.0, None);
        downs.push(next);
    }
    downs
}

// The glare on one channel at the input's resolution: `w0 * scale * c + sum_m w_m * band_m`
// with `weights[0]` = w0 and `weights[1 + m]` the weight of mip m of `downs`. The tent
// upsample sums the chain from its last level up, each level adding its weighted band.
pub fn compose(input: &Plane, downs: &[Plane], weights: &[f64], scale: f64) -> Plane {
    let last = downs.len() - 1;
    let mut up: Option<Plane> = None;
    for m in (0..last).rev() {
        let (coarse, coarse_weight) = match &up {
            Some(p) => (p, 1.0),
            None => (&downs[last], weights[1 + last]),
        };
        let size = downs[m].size;
        up = Some(upsample(
            Some((&downs[m], weights[1 + m])),
            coarse,
            coarse_weight,
            size,
        ));
    }
    let (chain, chain_weight) = match &up {
        Some(p) => (p, 1.0),
        None => (&downs[0], weights[1]),
    };
    upsample(
        Some((input, weights[0] * scale)),
        chain,
        chain_weight,
        input.size,
    )
}

// `weights` for a chain of `mips` levels: the bands past its last mip fold into it.
pub fn fold(weights: &[f64], mips: usize) -> Vec<f64> {
    let mut w = weights[..=mips].to_vec();
    w[mips] += weights[mips + 1..].iter().sum::<f64>();
    w
}
