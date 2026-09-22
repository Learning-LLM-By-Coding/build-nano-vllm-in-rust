use candle_core::{Device, Result, Tensor};

mod layers;
mod model;
mod tokenizer;

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

/// Analogy: cook the next course from the last, over and over.
///
/// Autoregressive generation: predict one token, append it, feed it back
/// in, `steps` times. Decode once at the end so multi-byte characters stay
/// intact.
fn generate(model: &model::Bigram, prompt: &str, steps: usize) -> Result<String> {
    let mut ids = tokenizer::encode(prompt);
    for _ in 0..steps {
        let last = *ids.last().expect("prompt must not be empty");
        let logits = model.forward(last)?;
        ids.push(greedy_pick(&logits)?);
    }
    Ok(tokenizer::decode(&ids).expect("an ascii corpus only generates ascii bytes"))
}

/// Deterministic made-up weights: value = index * scale. Real, trained
/// weights arrive on Day 6; until then the shapes are the lesson.
fn ramp(rows: usize, cols: usize, scale: f32, device: &Device) -> Result<Tensor> {
    let data: Vec<f32> = (0..rows * cols).map(|v| v as f32 * scale).collect();
    Tensor::from_vec(data, (rows, cols), device)
}

/// Walk a tiny prompt through the pre-attention layers, printing shapes.
fn run_layers_demo() -> Result<()> {
    let device = Device::Cpu;
    let embed = layers::Embedding::new(ramp(tokenizer::VOCAB_SIZE, 8, 0.01, &device)?);

    let text = "aa";
    let ids = tokenizer::encode(text);
    println!("text {text:?} -> ids {ids:?}");

    let x = embed.forward(&ids)?; // [seq, dim] = [2, 8]
    println!("embed:  {:?}", x.dims());
    let rows = x.to_vec2::<f32>()?;
    println!(
        "row 0 starts [{:.2}, {:.2}, {:.2}, ...]",
        rows[0][0], rows[0][1], rows[0][2]
    );
    println!("rows equal: {}", rows[0] == rows[1]);
    Ok(())
}

fn run_generate(prompt: &str, steps: usize) -> Result<()> {
    let device = Device::Cpu;
    let model = model::Bigram::from_text(model::CORPUS, &device)?;
    // Debug-print so invisible bytes (like the \0 an unseen prompt causes)
    // show up on screen instead of silently vanishing.
    println!("{:?}", generate(&model, prompt, steps)?);
    Ok(())
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = raw.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["generate", prompt] => run_generate(prompt, 40),
        ["generate", prompt, steps] => {
            run_generate(prompt, steps.parse().expect("steps must be a number"))
        }
        ["layers"] => run_layers_demo(),
        _ => {
            eprintln!("usage: cargo run -- generate <prompt> [steps]");
            eprintln!("       cargo run -- layers");
            std::process::exit(2);
        }
    }
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
    fn greedy_pick_breaks_ties_toward_the_lowest_id() -> Result<()> {
        let device = Device::Cpu;
        let flat = Tensor::zeros((1, tokenizer::VOCAB_SIZE), candle_core::DType::F32, &device)?;
        assert_eq!(greedy_pick(&flat)?, 0);
        Ok(())
    }

    #[test]
    fn generation_is_deterministic_and_cyclic() -> Result<()> {
        let model = model::Bigram::from_text(model::CORPUS, &Device::Cpu)?;
        assert_eq!(generate(&model, "ru", 12)?, "rust. rust. ru");
        Ok(())
    }

    #[test]
    fn unseen_bytes_generate_token_zero_garbage() -> Result<()> {
        let model = model::Bigram::from_text(model::CORPUS, &Device::Cpu)?;
        // 'z' never appears in the corpus: all-zero logits, so greedy picks
        // token 0 forever. The model knows nothing outside its data.
        assert_eq!(generate(&model, "z", 3)?, "z\0\0\0");
        Ok(())
    }
}
