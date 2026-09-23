use candle_core::{DType, Device, Result, Tensor};

mod attention;
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
    let norm = layers::RmsNorm::new(Tensor::ones(8, DType::F32, &device)?, 1e-5);
    let proj = layers::Linear::new(ramp(8, 4, 0.1, &device)?);

    let text = "aa";
    let ids = tokenizer::encode(text);
    println!("text {text:?} -> ids {ids:?}");

    let x = embed.forward(&ids)?; // [seq, dim] = [2, 8]
    println!("embed:   {:?}", x.dims());
    let rows = x.to_vec2::<f32>()?;
    println!(
        "row 0 starts [{:.2}, {:.2}, {:.2}, ...]",
        rows[0][0], rows[0][1], rows[0][2]
    );

    let x = norm.forward(&x)?; // [2, 8], every row now at unit magnitude
    let rows = x.to_vec2::<f32>()?;
    let rms: f32 = (rows[0].iter().map(|v| v * v).sum::<f32>() / 8.0).sqrt();
    println!("rmsnorm: {:?}  rms of row 0 = {rms:.3}", x.dims());

    let x = proj.forward(&x)?; // [2, 8] x [8, 4] -> [2, 4]
    println!("linear:  {:?}", x.dims());

    let rows = x.to_vec2::<f32>()?;
    let equal_before = rows[0] == rows[1];

    let x = layers::rope(&x)?; // [2, 4], each row rotated by its position
    println!("rope:    {:?}", x.dims());
    let rows = x.to_vec2::<f32>()?;
    println!(
        "same token, same vector?  before rope: {equal_before}, after rope: {}",
        rows[0] == rows[1]
    );
    Ok(())
}

/// Print a [seq, seq] table of the conversation: one row per listening
/// guest, one column per guest being heard, labels kept clear of values.
fn print_table(table: &Tensor) -> Result<()> {
    let rows = table.to_vec2::<f32>()?;
    let header: Vec<String> = (0..rows.len()).map(|s| format!("seat {s}")).collect();
    println!("           {}", header.join("  "));
    for (seat, row) in rows.iter().enumerate() {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:>6.3}")).collect();
        println!("  seat {seat} | {}", cells.join("  "));
    }
    Ok(())
}

/// Analogy: seat tonight's guests and hold one round of the table
/// conversation, printing the result of every step.
fn run_attention_demo() -> Result<()> {
    let device = Device::Cpu;
    // Day 3's front door and volume knob, unchanged. No demo translator
    // this time: the conversation carries its own re-mixing inside.
    let embed = layers::Embedding::new(ramp(tokenizer::VOCAB_SIZE, 8, 0.01, &device)?);
    let norm = layers::RmsNorm::new(Tensor::ones(8, DType::F32, &device)?, 1e-5);

    let text = "abc";
    let ids = tokenizer::encode(text);
    println!("text {text:?} -> ids {ids:?}");
    let cards = embed.forward(&ids)?; // [3, 8]: three guests' profile cards
    println!("cards:  {:?}", cards.dims());

    // One head over whole cards for now: level the volume, stamp the seats,
    // then hold the listening round by hand — printing after every step.
    let leveled = norm.forward(&cards)?;
    let q = layers::rope(&leveled)?;
    let k = layers::rope(&leveled)?;
    let scores = attention::scaled_scores(&q, &k)?;
    println!("scores (how well each question matches each topic card):");
    print_table(&scores)?;
    let masked = attention::mask_future(&scores)?;
    println!("masked (etiquette: every score in the future becomes -inf):");
    print_table(&masked)?;
    let shares = attention::softmax_rows(&masked)?;
    println!("who listens to whom (each row sums to 1; 0.000 = the future):");
    print_table(&shares)?;
    // attend() runs the same three steps in one call, then blends the
    // statements (v) by those shares — here the statements are the cards.
    let heard = attention::attend(&q, &k, &leveled)?;
    println!("heard:  {:?}", heard.dims());

    // The packaged layer: the same round through real projections.
    println!("q proj [8, 8]: 4 heads;  k,v proj [8, 4]: 2 SHARED heads (half the table space)");
    let attn = attention::Attention::new(
        layers::Linear::new(ramp(8, 8, 0.02, &device)?),
        layers::Linear::new(ramp(8, 4, 0.03, &device)?),
        layers::Linear::new(ramp(8, 4, 0.05, &device)?),
        layers::Linear::new(ramp(8, 8, 0.02, &device)?),
        4,
        2,
    );
    let out = attn.forward(&leveled)?;
    println!("attn:   {:?}", out.dims());
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
        ["attention"] => run_attention_demo(),
        _ => {
            eprintln!("usage: cargo run -- generate <prompt> [steps]");
            eprintln!("       cargo run -- layers");
            eprintln!("       cargo run -- attention");
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
