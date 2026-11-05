//! Tile tokens in file names: `<UDIM>` of USD and MaterialX, `<UVTILE>` of MaterialX.
//!
//! The specifications replace a token per point of the surface, from the integer parts U and V of
//! its uv: Mari's four digits `1001 + U + V * 10`, or Mudbox's `u{U + 1}_v{V + 1}`. So a name with
//! a token is a set of images, one to each unit square that has a file. A point past the files
//! names a file that isn't there, and reads as a file that can't be resolved does.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Mari's token.
pub(crate) const UDIM: &str = "<UDIM>";
/// Mudbox's token.
const UVTILE: &str = "<UVTILE>";

/// A tile's column U and row V.
pub(crate) type Cell = [u16; 2];

/// The first and last Mari tiles USD reads, ten to a row. MaterialX states no last, and its four
/// digits end at 9999.
pub(crate) const UDIM_TILES: std::ops::RangeInclusive<u16> = 1001..=1100;
const UDIM_LAST: u16 = 9999;

/// Mari tile `n` as a cell.
pub(crate) fn udim_cell(n: u16) -> Cell {
    [(n - 1001) % 10, (n - 1001) / 10]
}

/// Whether a file name holds a tile token.
pub(crate) fn tiled(file: &str) -> bool {
    file.contains(UDIM) || file.contains(UVTILE)
}

/// What the text at the start of a file name says under a token.
#[derive(Debug, PartialEq)]
enum Named {
    Cell(Cell),
    /// A tile past the last column or row a tile set holds.
    Past,
    /// No tile.
    Other,
}

/// What the text at the start of `s` names under token `token`.
fn parse(token: &str, s: &str) -> Named {
    if token == UDIM {
        let n = s.get(..4).and_then(|n| n.parse().ok());
        return match n.filter(|n| (1001..=UDIM_LAST).contains(n)) {
            Some(n) => Named::Cell(udim_cell(n)),
            None => Named::Other,
        };
    }
    let digits = |s: &str| s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let Some(s) = s.strip_prefix('u') else {
        return Named::Other;
    };
    let (u, rest) = s.split_at(digits(s));
    let Some(v) = rest.strip_prefix("_v") else {
        return Named::Other;
    };
    let v = &v[..digits(v)];
    // A tile is 1 + its cell, so one past the last u16 is still a cell.
    let cell = |n: &str| match n.trim_start_matches('0') {
        "" => None,
        n => Some(
            n.parse::<u32>()
                .ok()
                .and_then(|n| u16::try_from(n - 1).ok()),
        ),
    };
    match (cell(u), cell(v)) {
        (Some(Some(u)), Some(Some(v))) => Named::Cell([u, v]),
        (Some(_), Some(_)) => Named::Past,
        _ => Named::Other,
    }
}

/// `path` with its tokens naming `cell`.
pub(crate) fn named(path: &str, [u, v]: Cell) -> PathBuf {
    let udim = 1001 + u32::from(u) + 10 * u32::from(v);
    let uvtile = format!("u{}_v{}", u32::from(u) + 1, u32::from(v) + 1);
    PathBuf::from(
        path.replace(UDIM, &udim.to_string())
            .replace(UVTILE, &uvtile),
    )
}

/// What a name with a token holds.
#[derive(Default)]
pub(crate) struct Scan {
    /// Its files, in cell order.
    pub(crate) files: Vec<(Cell, PathBuf)>,
    /// The directories whose names say what tiles there are: a file appearing in one adds a tile.
    /// A token in a directory's name gives each tile a directory of its own, and those count too.
    pub(crate) holders: Vec<PathBuf>,
    /// Names of files past the last column or row a tile set holds.
    pub(crate) past: Vec<String>,
}

/// The files of `path`, a name with a token.
pub(crate) fn scan(path: &Path) -> Scan {
    let text = path.to_string_lossy();
    let token = [UDIM, UVTILE]
        .into_iter()
        .filter_map(|t| text.find(t).map(|at| (at, t)))
        .min();
    let Some((at, token)) = token else {
        return Scan::default();
    };
    let head = &text[..at];
    let cut = head.rfind(['/', '\\']).map_or(0, |i| i + 1);
    let (dir, stem) = head.split_at(cut);
    let dir = PathBuf::from(if dir.is_empty() { "." } else { dir });
    // What follows the token in the same file name, up to any other token.
    let tail = text[at + token.len()..].split(['/', '\\', '<']).next();
    let mut cells: Vec<Cell> = Vec::new();
    let mut past = Vec::new();
    for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix(stem) else {
            continue;
        };
        match parse(token, rest) {
            Named::Cell(cell) => cells.push(cell),
            Named::Past if rest.ends_with(tail.unwrap_or_default()) => past.push(name.to_string()),
            Named::Past | Named::Other => {}
        }
    }
    past.sort_unstable();
    cells.sort_unstable_by_key(|[u, v]| (*v, *u));
    cells.dedup();
    let candidates: Vec<_> = cells
        .into_iter()
        .map(|cell| (cell, named(&text, cell)))
        .collect();
    let parents = candidates.iter().filter_map(|(_, file)| file.parent());
    let mut holders = BTreeSet::from([dir]);
    holders.extend(
        parents
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf),
    );
    let files = candidates
        .into_iter()
        .filter(|(_, file)| file.is_file())
        .collect();
    Scan {
        files,
        holders: holders.into_iter().collect(),
        past,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_name_cells_by_the_clauses_formulas() {
        assert_eq!(parse(UDIM, "1001.png"), Named::Cell([0, 0]));
        assert_eq!(parse(UDIM, "1012.png"), Named::Cell([1, 1]));
        assert_eq!(parse(UDIM, "1010.png"), Named::Cell([9, 0]));
        assert_eq!(parse(UDIM, "1000.png"), Named::Other);
        assert_eq!(parse(UVTILE, "u1_v1.png"), Named::Cell([0, 0]));
        assert_eq!(parse(UVTILE, "u12_v3.png"), Named::Cell([11, 2]));
        assert_eq!(parse(UVTILE, "u0_v1.png"), Named::Other);
        assert_eq!(parse(UVTILE, "u65536_v1.png"), Named::Cell([65535, 0]));
        assert_eq!(parse(UVTILE, "u1_v65537.png"), Named::Past);
        assert_eq!(parse(UVTILE, "u99999999999_v1.png"), Named::Past);
        let path = "a/<UDIM>/b.<UVTILE>.png";
        assert_eq!(named(path, [1, 1]), Path::new("a/1012/b.u2_v2.png"));
    }
}
