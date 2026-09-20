use candle_core::{Device, Result, Tensor};

mod tokenizer;

fn main() -> Result<()> {
    let device = Device::Cpu;

    // Shape [1, 3]: one row of three features.
    let input = Tensor::new(&[[1_f32, 2.0, 3.0]], &device)?;

    // Shape [3, 2]: turns three input features into two output scores.
    let weights = Tensor::new(&[[1_f32, 0.0], [0_f32, 1.0], [1_f32, 1.0]], &device)?;

    // [1, 3] x [3, 2] -> [1, 2]. The inner dimensions (3) must match.
    let output = input.matmul(&weights)?;

    println!("output shape: {:?}", output.dims());
    println!("{:?}", output.to_vec2::<f32>()?);

    let text = "hi";
    let ids = tokenizer::encode(text);
    println!(
        "vocab size {} -> text {text:?} -> token ids {ids:?}",
        tokenizer::VOCAB_SIZE
    );
    let back = tokenizer::decode(&ids).expect("ids from valid text always decode");
    println!("decoded back: {back:?}");
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
}
