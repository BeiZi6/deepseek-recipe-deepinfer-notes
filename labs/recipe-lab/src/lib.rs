//! Shared helpers for the learning binaries.
//!
//! The tokenizer is not vendored (6.6 MB JSON). Point `DEEPSEEK_TOKENIZER` at
//! any DeepSeek V4.1 `tokenizer.json`, or clone deepseek-recipe into
//! `.research/` as the README describes.

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;

/// The DSML marker is `<｜DSML｜...>`; written with escapes so the source stays ASCII.
pub const DSML: &str = "\u{FF5C}DSML\u{FF5C}";

pub fn tokenizer_path() -> PathBuf {
    if let Ok(path) = std::env::var("DEEPSEEK_TOKENIZER") {
        return PathBuf::from(path);
    }
    // Resolve against the crate, not the cwd, so `cargo run --manifest-path`
    // from the repo root works too.
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        crate_dir.join("tokenizer.json"),
        crate_dir.join("../../.research/deepseek-recipe/static/tokenizers/v41/tokenizer.json"),
    ];
    candidates
        .iter()
        .find(|path| path.is_file())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "no tokenizer found; set DEEPSEEK_TOKENIZER or clone deepseek-recipe into .research/"
            )
        })
}

pub fn load_tokenizer() -> Tokenizer {
    Tokenizer::from_file(tokenizer_path()).expect("tokenizer.json loads")
}
