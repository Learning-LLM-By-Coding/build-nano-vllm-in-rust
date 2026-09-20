/// Analogy: the pantry — 256 numbered jars, one per possible byte.
///
/// Text is UTF-8 bytes, so the vocabulary is fixed at 256 and no input can
/// ever be out-of-vocabulary.
pub const VOCAB_SIZE: usize = 256;

/// Text -> token ids, one id per UTF-8 byte.
pub fn encode(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// Token ids -> text. Returns None if an id is not a byte (> 255) or the
/// bytes are not valid UTF-8 (e.g. a lone continuation byte).
pub fn decode(ids: &[u32]) -> Option<String> {
    let bytes = ids
        .iter()
        .map(|&id| u8::try_from(id).ok())
        .collect::<Option<Vec<u8>>>()?;
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii() {
        let ids = encode("hello");
        assert_eq!(ids, vec![104, 101, 108, 108, 111]);
        assert_eq!(decode(&ids), Some("hello".to_string()));
    }

    #[test]
    fn multibyte_text_uses_more_tokens_than_chars() {
        // é is two bytes in UTF-8, so 5 characters become 6 tokens.
        let ids = encode("héllo");
        assert_eq!(ids.len(), 6);
        assert_eq!(decode(&ids), Some("héllo".to_string()));
    }

    #[test]
    fn decode_rejects_ids_larger_than_a_byte() {
        assert_eq!(decode(&[104, 999]), None);
    }

    #[test]
    fn decode_rejects_invalid_utf8() {
        // 0xF0 opens a 4-byte sequence; 0x28 is not a valid continuation.
        assert_eq!(decode(&[0xF0, 0x28]), None);
    }
}
