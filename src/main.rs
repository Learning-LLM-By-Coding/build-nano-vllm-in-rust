use candle_core::{Device, Result, Tensor};

mod tokenizer;

/// A deterministic toy "model": after byte b it scores byte b+1 highest.
/// Real, learned weights arrive on Day 2 — today the point is the pipeline.
fn next_byte_weights(device: &Device) -> Result<Tensor> {
    let mut w = vec![0_f32; tokenizer::VOCAB_SIZE * tokenizer::VOCAB_SIZE];
    for b in 0..tokenizer::VOCAB_SIZE {
        w[b * tokenizer::VOCAB_SIZE + (b + 1) % tokenizer::VOCAB_SIZE] = 1.0;
    }
    Tensor::from_vec(w, (tokenizer::VOCAB_SIZE, tokenizer::VOCAB_SIZE), device)
}

/// One-hot row for a token id: shape [1, VOCAB_SIZE], all zeros except id.
fn one_hot(id: u32, device: &Device) -> Result<Tensor> {
    let mut row = vec![0_f32; tokenizer::VOCAB_SIZE];
    row[id as usize] = 1.0;
    Tensor::from_vec(row, (1, tokenizer::VOCAB_SIZE), device)
}

/// Analogy: the judge.
///
/// Greedy sampling: pick the highest-scoring token id (ties -> lowest id).
fn greedy_pick(logits: &Tensor) -> Result<u32> {
    let rows = logits.to_vec2::<f32>()?;
    let row = &rows[0];
    let mut best = 0;
    for (id, score) in row.iter().enumerate() {
        if *score > row[best] {
            best = id;
        }
    }
    Ok(best as u32)
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let text = "hi";
    let ids = tokenizer::encode(text);
    println!("text {text:?} -> token ids {ids:?}");

    let last = *ids.last().expect("prompt must not be empty");
    let input = one_hot(last, &device)?; // [1, 256]
    let weights = next_byte_weights(&device)?; // [256, 256]
    let logits = input.matmul(&weights)?; // [1, 256]: the inner 256s cancel
    println!("logits shape: {:?}", logits.dims());

    let next = greedy_pick(&logits)?;
    let piece = tokenizer::decode(&[next]).expect("next byte is not decodable on its own");
    println!("greedy next token: {next} -> {piece:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_matmul_matches_hand_arithmetic() -> Result<()> {
        let device = Device::Cpu;
        let input = Tensor::new(&[[1_f32, 2.0, 3.0]], &device)?;
        let weights = Tensor::new(&[[1_f32, 0.0], [0_f32, 1.0], [1_f32, 1.0]], &device)?;
        let output = input.matmul(&weights)?;
        assert_eq!(output.dims(), &[1, 2]);
        // Exact float comparison is safe here: every value is a small
        // integer, which f32 represents exactly.
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![4.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn matmul_rejects_mismatched_shapes() {
        let device = Device::Cpu;
        let input = Tensor::new(&[[1_f32, 2.0, 3.0]], &device).unwrap();
        // [1, 3] x [2, 2]: inner dimensions 3 and 2 do not match.
        let bad_weights = Tensor::new(&[[1_f32, 0.0], [0_f32, 1.0]], &device).unwrap();
        assert!(input.matmul(&bad_weights).is_err());
    }

    #[test]
    fn toy_model_predicts_the_next_ascii_byte() -> Result<()> {
        let device = Device::Cpu;
        let logits = one_hot(104, &device)?.matmul(&next_byte_weights(&device)?)?;
        assert_eq!(greedy_pick(&logits)?, 105); // 'h' -> 'i'
        Ok(())
    }

    #[test]
    fn greedy_pick_breaks_ties_toward_the_lowest_id() -> Result<()> {
        let device = Device::Cpu;
        let flat = Tensor::zeros((1, tokenizer::VOCAB_SIZE), candle_core::DType::F32, &device)?;
        assert_eq!(greedy_pick(&flat)?, 0);
        Ok(())
    }
}
