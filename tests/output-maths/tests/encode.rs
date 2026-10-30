use output_maths as slang;

#[test]
fn pq_round_trips_in_f64() {
    let mut worst = 0.0f64;
    for i in 0..=100_000 {
        let nits = 10f64.powf(-3.0 + 7.0 * f64::from(i) / 100_000.0);
        let back = orr_output::pq_decode(orr_output::pq_encode(nits));
        worst = worst.max((back - nits).abs() / nits);
    }
    assert!(worst < 1e-6, "PQ round trip {worst:e}");
    let mut worst = 0.0f64;
    for i in 0..=100_000 {
        let x = f64::from(i) / 100_000.0;
        worst = worst.max((orr_output::srgb_decode(orr_output::srgb_encode(x)) - x).abs());
    }
    assert!(worst < 1e-12, "sRGB round trip {worst:e}");
}

#[test]
fn encoders_match_orr_color() {
    // orr.color is f32: its PQ signal is within 1.3e-5 of ST 2084 and its decode within
    // 3.6e-5 relative of the exact decode.
    for i in 0..=4096 {
        let nits = 10f64.powf(-3.0 + 7.0 * f64::from(i) / 4096.0);
        let s = slang::pq_encode(nits as f32);
        assert!(
            (f64::from(s) - orr_output::pq_encode(nits)).abs() < 1.5e-5,
            "pq_encode {nits}"
        );
        let d = f64::from(slang::pq_decode(s));
        let exact = orr_output::pq_decode(f64::from(s));
        assert!((d - exact).abs() <= 4e-5 * exact, "pq_decode {s}");
    }
    for i in -64..=4096 {
        let x = f64::from(i) / 4096.0;
        let (e, d) = (slang::srgb_encode(x as f32), slang::srgb_decode(x as f32));
        assert!(
            (f64::from(e) - orr_output::srgb_encode(x)).abs() < 1e-6,
            "srgb_encode {x}"
        );
        assert!(
            (f64::from(d) - orr_output::srgb_decode(x)).abs() < 1e-6,
            "srgb_decode {x}"
        );
    }
}

#[test]
fn exposure_units_match_the_shader() {
    for i in -400..=400 {
        let ev = i as f32 * 0.05;
        let l = orr_output::luminance_from_ev100(ev);
        let s = orr_output::exposure_from_ev100(ev);
        assert!((slang::luminance_from_ev100(ev) - l).abs() <= 1e-6 * l);
        assert!((slang::exposure_from_ev100(ev) - s).abs() <= 1e-6 * s);
        assert!((slang::ev100_from_luminance(l) - ev).abs() < 1e-5);
        assert!((slang::ev100_from_exposure(s) - ev).abs() < 1e-5);
    }
}
