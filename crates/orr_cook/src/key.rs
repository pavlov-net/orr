//! Cook keys: a hash of everything that shapes an entry's bytes.
//!
//! A key covers the class's cook version, the zstd library and level, the libraries of the
//! class (locked versions from `Cargo.lock`, build script), the entry name and the parts the
//! caller adds: settings, input content hashes and the keys of the entries it depends on.

use orr_pack::{Class, Hash32};

/// zstd level of every cooked blob.
pub(crate) const ZSTD_LEVEL: i32 = 19;

/// Cook version per class. Bump a class's version whenever the cook writes different bytes
/// for the same inputs; the golden test fails until it is bumped.
pub(crate) const VERSIONS: &[(Class, u32)] = &[
    (Class::Mesh, 5),
    (Class::Texture, 1),
    (Class::Material, 1),
    (Class::Model, 2),
    (Class::InstancePage, 2),
    (Class::RasterSet, 1),
    (Class::Skeleton, 1),
    (Class::Clip, 1),
];

/// Cook version of user classes, which the cook stores as given.
pub(crate) const USER_VERSION: u32 = 1;

pub(crate) fn version(class: u16) -> u32 {
    VERSIONS
        .iter()
        .find(|(c, _)| *c as u16 == class)
        .map_or(USER_VERSION, |(_, v)| *v)
}

fn libraries(class: u16) -> &'static str {
    match Class::from_u16(class) {
        Some(Class::Mesh | Class::Skeleton | Class::Clip) => env!("ORR_COOK_MESH_LIBS"),
        Some(Class::Texture) => env!("ORR_COOK_TEXTURE_LIBS"),
        Some(Class::RasterSet) => env!("ORR_COOK_RASTER_LIBS"),
        _ => "",
    }
}

pub(crate) struct Key(blake3::Hasher);

impl Key {
    pub(crate) fn new(class: impl Into<u16>, name: &str) -> Self {
        let class = class.into();
        Self::with_libraries(class, libraries(class), name)
    }

    fn with_libraries(class: u16, libraries: &str, name: &str) -> Self {
        let mut k = Self(blake3::Hasher::new());
        k.0.update(&class.to_le_bytes());
        k.0.update(&version(class).to_le_bytes());
        k.0.update(&ZSTD_LEVEL.to_le_bytes());
        k.0.update(&zstd::zstd_safe::version_number().to_le_bytes());
        k.str(env!("ORR_COOK_ZSTD_LIBS")).str(libraries).str(name)
    }

    /// Adds one part; parts are length-prefixed, so their boundaries are part of the key.
    pub(crate) fn bytes(mut self, b: &[u8]) -> Self {
        self.0.update(&(b.len() as u64).to_le_bytes());
        self.0.update(b);
        self
    }

    pub(crate) fn str(self, s: &str) -> Self {
        self.bytes(s.as_bytes())
    }

    /// Adds a value through its `Debug` form, which names every field.
    pub(crate) fn debug(self, v: &impl std::fmt::Debug) -> Self {
        self.str(&format!("{v:?}"))
    }

    pub(crate) fn hash(self, h: Hash32) -> Self {
        self.bytes(&h.0)
    }

    pub(crate) fn finish(self) -> Hash32 {
        Hash32(*self.0.finalize().as_bytes())
    }
}

/// Every version and library that enters any key; source records carry it, so a change
/// to any of them drops every record.
pub(crate) fn salt() -> Hash32 {
    let mut k = Key::new(0u16, "salt").str(env!("ORR_COOK_MESH_LIBS"));
    k = k.str(env!("ORR_COOK_TEXTURE_LIBS"));
    for (c, v) in VERSIONS {
        k = k.bytes(&(*c as u16).to_le_bytes()).bytes(&v.to_le_bytes());
    }
    k.bytes(&USER_VERSION.to_le_bytes()).finish()
}

pub(crate) fn hex(h: Hash32) -> String {
    format!("{h:?}")
}

pub(crate) fn from_hex(s: &str) -> Option<Hash32> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(Hash32(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_versions_enter_keys() {
        assert!(libraries(Class::Mesh as u16).contains("meshopt 0.7.0"));
        assert!(libraries(Class::Texture as u16).contains("ctt 0.6.1"));
        assert!(env!("ORR_COOK_ZSTD_LIBS").contains("zstd-sys "));
        let key = |libs| {
            Key::with_libraries(Class::Mesh as u16, libs, "m")
                .str("settings")
                .finish()
        };
        assert_ne!(key("meshopt 0.7.0"), key("meshopt 0.7.1"));
    }
}
