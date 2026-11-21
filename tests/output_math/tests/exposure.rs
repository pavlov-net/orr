use output_math as slang;

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
