use candle_core::{Device, Result, Tensor};

use crate::tokenizer::{VOCAB_SIZE, encode};

/// The tiny corpus the bigram model "trains" on. Counting its byte pairs is
/// the only training in this course until real checkpoints arrive on Day 6.
pub const CORPUS: &str = "rust. rust. rust. rush.";

/// Analogy: the cook's tally sheet — which byte followed which.
///
/// A bigram model: row b of `weights` counts how often each byte followed
/// byte b in the corpus. Higher count = higher logit.
pub struct Bigram {
    weights: Tensor, // [VOCAB_SIZE, VOCAB_SIZE] f32 counts
}

impl Bigram {
    /// Analogy: fill the tally sheet.
    ///
    /// "Training", in miniature: count the data.
    pub fn from_text(text: &str, device: &Device) -> Result<Self> {
        let ids = encode(text);
        let mut counts = vec![0_f32; VOCAB_SIZE * VOCAB_SIZE];
        for pair in ids.windows(2) {
            counts[pair[0] as usize * VOCAB_SIZE + pair[1] as usize] += 1.0;
        }
        let weights = Tensor::from_vec(counts, (VOCAB_SIZE, VOCAB_SIZE), device)?;
        Ok(Self { weights })
    }

    /// Analogy: the cook's suggestion.
    ///
    /// Logits for the token after `token`, shape [1, VOCAB_SIZE].
    pub fn forward(&self, token: u32) -> Result<Tensor> {
        one_hot(token, self.weights.device())?.matmul(&self.weights)
    }
}

/// One-hot row for a token id: shape [1, VOCAB_SIZE], all zeros except id.
fn one_hot(id: u32, device: &Device) -> Result<Tensor> {
    let mut row = vec![0_f32; VOCAB_SIZE];
    row[id as usize] = 1.0;
    Tensor::from_vec(row, (1, VOCAB_SIZE), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_the_corpus_matches_hand_tallies() -> Result<()> {
        let model = Bigram::from_text(CORPUS, &Device::Cpu)?;
        let after_s = model.forward(115)?.to_vec2::<f32>()?; // after 's'
        assert_eq!(after_s[0][116], 3.0); // "st" appears 3 times
        assert_eq!(after_s[0][104], 1.0); // "sh" appears once
        Ok(())
    }

    #[test]
    fn forward_scores_every_token() -> Result<()> {
        let model = Bigram::from_text(CORPUS, &Device::Cpu)?;
        assert_eq!(model.forward(114)?.dims(), &[1, VOCAB_SIZE]); // after 'r'
        Ok(())
    }

    #[test]
    fn unseen_bytes_have_all_zero_logits() -> Result<()> {
        let model = Bigram::from_text(CORPUS, &Device::Cpu)?;
        let row = model.forward(122)?.to_vec2::<f32>()?; // 'z' is not in the corpus
        assert!(row[0].iter().all(|&score| score == 0.0));
        Ok(())
    }
}
