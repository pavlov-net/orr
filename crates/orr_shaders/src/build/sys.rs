//! Bindings to the pinned release's `slang.h`.
#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_imports,
    unreachable_pub,
    unused_qualifications,
    clippy::all,
    clippy::pedantic,
    clippy::restriction
)]

include!(concat!(env!("OUT_DIR"), "/slang_sys.rs"));
