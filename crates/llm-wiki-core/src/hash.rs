//! Content hashing (PRD §8.3): SHA-256 everywhere, hex-encoded.
//!
//! Content hashes are the change-detection basis; mtimes are only a
//! pre-filter and never enter an identity.

use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    to_hex(&hasher.finalize())
}

pub fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Normalizes text for content fingerprints: NFC + line-ending and trailing
/// whitespace collapse, so that equivalent section bodies share a fingerprint
/// (PRD §45 Section Matcher input).
pub fn fingerprint_text(text: &str) -> String {
    let nfc: String = text.nfc().collect();
    let mut out = String::with_capacity(nfc.len());
    for line in nfc.lines() {
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out.trim().to_owned()
}

pub fn content_fingerprint(text: &str) -> String {
    sha256_hex(fingerprint_text(text).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_stable_hex() {
        let h = sha256_hex(b"abc");
        assert_eq!(
            h,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn fingerprint_ignores_line_endings_and_trailing_ws() {
        let a = content_fingerprint("line one\r\nline two\n");
        let b = content_fingerprint("line one\nline two   \n");
        assert_eq!(a, b);
    }

    #[test]
    fn fingerprint_is_nfc_insensitive() {
        let decomposed = "caf\u{e9}".to_owned();
        let composed = "cafe\u{301}".to_owned();
        assert_eq!(
            content_fingerprint(&decomposed),
            content_fingerprint(&composed)
        );
    }
}
