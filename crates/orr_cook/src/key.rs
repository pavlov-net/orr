//! Cook keys: a hash of everything that shapes an entry's bytes.
//!
//! A key covers the class's cook version, the pack schema `orr_pack` writes the class's entries
//! in, the zstd library, the libraries of the class (locked versions from `Cargo.lock`, build
//! script), the entry name and the parts the caller adds: settings, input content hashes and the
//! keys of the entries it depends on.
//! The key an archive stores also covers the zstd level, and for GDeflate archives the
//! encoder's commit and level.

use orr_pack::{Class, Hash32};

/// GDeflate level of the payload blobs of GDeflate archives.
pub(crate) const GDEFLATE_LEVEL: i32 = 12;

/// Cook version per class, part of every key: raising one recooks that class. A schema change
/// recooks the class by itself (`schema`); the version is for changes in what the cook writes.
const VERSIONS: &[(Class, u32)] = &[
    (Class::Mesh, 40),
    (Class::Texture, 15),
    (Class::Material, 17),
    (Class::Model, 7),
    (Class::InstancePage, 13),
    (Class::RasterSet, 4),
    (Class::Skeleton, 5),
    (Class::Clip, 6),
    (Class::NeuralTextureSet, 6),
    (Class::Subd, 5),
    (Class::Volume, 2),
    (Class::TextureTiles, 1),
];

/// Cook version of user classes, which the cook stores as given.
const USER_VERSION: u32 = 1;

pub(crate) fn version(class: u16) -> u32 {
    VERSIONS
        .iter()
        .find(|(c, _)| *c as u16 == class)
        .map_or(USER_VERSION, |(_, v)| *v)
}

/// The schema `orr_pack` writes `class`'s entries in, part of every key, so an entry cooked under
/// another schema is never reused; 0 for a user class, which the cook stores as given.
fn schema(class: u16) -> u16 {
    Class::from_u16(class).and_then(Class::schema).unwrap_or(0)
}

fn libraries(class: u16) -> &'static str {
    match Class::from_u16(class) {
        Some(Class::Mesh | Class::Skeleton | Class::Clip) => env!("ORR_COOK_MESH_LIBS"),
        Some(Class::Texture | Class::NeuralTextureSet) => env!("ORR_COOK_TEXTURE_LIBS"),
        Some(Class::RasterSet) => env!("ORR_COOK_RASTER_LIBS"),
        Some(Class::Subd) => env!("ORR_COOK_SUBD_LIBS"),
        Some(Class::Volume) => env!("ORR_COOK_VOLUME_LIBS"),
        _ => "",
    }
}

pub(crate) struct Key(blake3::Hasher);

impl Key {
    pub(crate) fn new(class: impl Into<u16>, name: &str) -> Self {
        let class = class.into();
        Self::with_parts(class, schema(class), libraries(class), name)
    }

    fn with_parts(class: u16, schema: u16, libraries: &str, name: &str) -> Self {
        let mut k = Self(blake3::Hasher::new());
        k.0.update(&class.to_le_bytes());
        k.0.update(&version(class).to_le_bytes());
        k.0.update(&schema.to_le_bytes());
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

/// The key of an entry in an archive of blobs at zstd level `zstd_level`, with or without
/// GDeflate payload blobs.
pub(crate) fn stored_as(key: Hash32, zstd_level: i32, gdeflate: bool) -> Hash32 {
    let k = Key(blake3::Hasher::new()).bytes(&zstd_level.to_le_bytes());
    if !gdeflate {
        return k.hash(key).finish();
    }
    #[cfg(feature = "gdeflate")]
    let commit = crate::gdeflate::COMMIT;
    #[cfg(not(feature = "gdeflate"))]
    let commit = "";
    k.str("gdeflate")
        .str(commit)
        .bytes(&GDEFLATE_LEVEL.to_le_bytes())
        .hash(key)
        .finish()
}

/// Every version and library that enters any key; source records carry it, so a change
/// to any of them drops every record.
pub(crate) fn salt() -> Hash32 {
    let mut k = Key::new(0u16, "salt").str(env!("ORR_COOK_MESH_LIBS"));
    k = k.str(env!("ORR_COOK_TEXTURE_LIBS"));
    k = k.str(env!("ORR_COOK_SUBD_LIBS"));
    k = k.str(env!("ORR_COOK_RASTER_LIBS"));
    k = k.str(env!("ORR_COOK_VOLUME_LIBS"));
    for (c, v) in VERSIONS {
        let class = *c as u16;
        k = k
            .bytes(&class.to_le_bytes())
            .bytes(&v.to_le_bytes())
            .bytes(&schema(class).to_le_bytes());
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
            let mesh = Class::Mesh as u16;
            Key::with_parts(mesh, schema(mesh), libs, "m")
                .str("settings")
                .finish()
        };
        assert_ne!(key("meshopt 0.7.0"), key("meshopt 0.7.1"));
    }

    /// An entry's key follows its class's schema: one cooked before a schema change is another
    /// entry, whatever its bytes.
    #[test]
    fn schemas_enter_keys() {
        let material = Class::Material as u16;
        let key = |schema| Key::with_parts(material, schema, "", "m").finish();
        assert_eq!(
            Key::new(material, "m").finish(),
            key(orr_pack::material::SCHEMA)
        );
        assert_ne!(key(3), key(4));
    }
}
