//! The lighting test bed: every analytic scene builds and renders, and accumulated images
//! depend on the seed and nothing else.

use lighting_suite::assert_clean;
use lighting_suite::bed::{Analytic, Bed, FlatColor};
use orr_scene::testing::max_difference;

const SKY: [f32; 3] = [0.2, 0.3, 0.5];

#[test]
fn flat_color_accumulation_repeats_per_seed_and_differs_across_seeds() {
    let mut bed = Bed::new(Analytic::CornellBox);
    let flat = FlatColor::new(SKY);
    let mut run = |seed| bed.accumulate(8, seed, &mut |list, f| flat.record(list, f));
    let a = run(1);
    let again = run(1);
    let other = run(2);
    assert_eq!(max_difference(&a.image(), &again.image()), 0.0);
    let e = other.flip(&a, 1.0);
    assert!(
        e.mean > 1e-3,
        "seeds 1 and 2 gave nearly the same image: {e:?}"
    );
    assert_clean();
}

#[test]
fn every_analytic_scene_draws_surfaces_over_a_quarter_of_the_view() {
    let flat = FlatColor::new(SKY);
    for analytic in Analytic::ALL {
        let image = Bed::new(analytic).accumulate(1, 0, &mut |list, f| flat.record(list, f));
        let drawn = image.mean.iter().filter(|p| p[3] > 0.0).count();
        assert!(
            drawn > image.mean.len() / 4,
            "{analytic:?}: only {drawn} of {} pixels drew a surface",
            image.mean.len()
        );
    }
    assert_clean();
}
