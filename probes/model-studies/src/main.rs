mod atmo;
mod atmo_run;
mod exposure;
mod gamut;
mod gbuf;
mod openpbr;
mod sss;
mod util;

fn main() {
    let cmd = std::env::args().nth(1).unwrap_or_default();
    match cmd.as_str() {
        "gbuf" => gbuf::run(),
        "gamut" => gamut::run(),
        "gamut-sweep" => gamut::sweep(),
        "exposure" => exposure::run(),
        "atmo-bands" => atmo_run::bands(),
        "atmo-curvature" => atmo_run::curvature(),
        "atmo-phase" => atmo_run::phase(),
        "atmo-eye" => atmo_run::eye(),
        "atmo-aerosol" => atmo_run::aerosol(),
        "atmo-f32" => atmo_run::altitude_f32(),
        "sss-entry" => sss::run_entry(),
        "sss-dwivedi" => sss::run_dwivedi(),
        _ => eprintln!("usage: model-studies <gbuf|sss-entry|sss-dwivedi|gamut|gamut-sweep|exposure|atmo-bands|atmo-curvature|atmo-phase|atmo-eye|atmo-aerosol|atmo-f32>"),
    }
}
