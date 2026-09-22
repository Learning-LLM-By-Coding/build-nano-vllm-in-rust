use candle_core::{D, Result, Tensor};

/// Analogy: the dining room's front door — swap a guest's badge number
/// for their profile card.
///
/// A learned lookup table: row t is the dense vector for token id t;
/// mathematically one_hot(t) x weight, fetched directly.
pub struct Embedding {
    weight: Tensor, // [vocab_size, dim]
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    /// Token ids -> their vectors: [seq] ids become a [seq, dim] tensor.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(ids, self.weight.device())?;
        self.weight.index_select(&ids, 0)
    }
}

/// Analogy: the translator — re-mix a card's numbers for the next
/// activity.
///
/// out = x @ weight, [seq, in] x [in, out] -> [seq, out]; no bias, like the
/// Llama family. (Checkpoints store the transpose — Day 6 handles it.)
pub struct Linear {
    weight: Tensor, // [in, out]
}

impl Linear {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight)
    }
}

/// Analogy: the volume knob — every card leaves at loudness 1.0, whatever
/// it arrived at.
///
/// RMS normalization: x / sqrt(mean(x^2) + eps) * scale, row by row.
pub struct RmsNorm {
    scale: Tensor, // [dim]
    eps: f64,
}

impl RmsNorm {
    pub fn new(scale: Tensor, eps: f64) -> Self {
        Self { scale, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Each row's loudness: the square root of its mean squared entry
        // (eps keeps a silent row from dividing by zero). Shape [seq, 1].
        let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        // Divide each row by its own loudness, then apply the learned knobs.
        x.broadcast_div(&rms)?.broadcast_mul(&self.scale)
    }
}

/// Analogy: the seat stamp — wind each guest's card-dials by where they
/// sit.
///
/// Rotary position embeddings: rotate each (2i, 2i+1) pair of features by
/// angle pos * theta_i, where theta_i = 10000^(-2i/dim). Early pairs spin
/// fast, later pairs slow — a multi-speed clock, one unique hand
/// configuration per position. Rotation preserves lengths, and dot products
/// between rotated vectors depend only on the POSITION GAP — exactly the
/// property attention scores will consume on Day 4.
/// Deliberately a plain scalar loop: obvious first, fast later.
pub fn rope(x: &Tensor) -> Result<Tensor> {
    let (seq, dim) = x.dims2()?;
    assert!(dim % 2 == 0, "rope needs an even feature dimension");
    let mut rows = x.to_vec2::<f32>()?;
    for (pos, row) in rows.iter_mut().enumerate() {
        // Each pair of features (2i, 2i+1) is one dial on this guest's card.
        for i in 0..dim / 2 {
            // How fast this dial spins — dial 0 fastest, later ones slower.
            let theta = 10_000_f64.powf(-2.0 * i as f64 / dim as f64);
            // Wind it: sitting in seat `pos` turns the dial by pos * theta.
            let (sin, cos) = (pos as f64 * theta).sin_cos();
            // A standard 2D rotation of the pair by that angle.
            let (a, b) = (row[2 * i] as f64, row[2 * i + 1] as f64);
            row[2 * i] = (a * cos - b * sin) as f32;
            row[2 * i + 1] = (a * sin + b * cos) as f32;
        }
    }
    Tensor::from_vec(rows.concat(), (seq, dim), x.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn tiny_table(device: &Device) -> Result<Tensor> {
        // vocab 3, dim 2: row t is recognizable at a glance.
        Tensor::new(&[[1_f32, 2.0], [3_f32, 4.0], [5_f32, 6.0]], device)
    }

    #[test]
    fn lookup_returns_the_selected_rows() -> Result<()> {
        let device = Device::Cpu;
        let embed = Embedding::new(tiny_table(&device)?);
        let out = embed.forward(&[2, 0])?;
        assert_eq!(out.dims(), &[2, 2]);
        assert_eq!(out.to_vec2::<f32>()?, vec![vec![5.0, 6.0], vec![1.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn lookup_equals_one_hot_matmul() -> Result<()> {
        // The trick from Days 1-2 and the real thing are the same operation.
        let device = Device::Cpu;
        let table = tiny_table(&device)?;
        let embed = Embedding::new(table.clone());
        let one_hot = Tensor::new(&[[0_f32, 1.0, 0.0]], &device)?; // id 1
        assert_eq!(
            one_hot.matmul(&table)?.to_vec2::<f32>()?,
            embed.forward(&[1])?.to_vec2::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn linear_reuses_day_1_arithmetic() -> Result<()> {
        // Same numbers as Day 1's warm-up matmul: [1,2,3] through the
        // 3->2 recipe matrix gives [4, 5], exactly.
        let device = Device::Cpu;
        let proj = Linear::new(Tensor::new(
            &[[1_f32, 0.0], [0_f32, 1.0], [1_f32, 1.0]],
            &device,
        )?);
        let out = proj.forward(&Tensor::new(&[[1_f32, 2.0, 3.0]], &device)?)?;
        assert_eq!(out.to_vec2::<f32>()?, vec![vec![4.0, 5.0]]);
        Ok(())
    }

    #[test]
    fn rmsnorm_matches_hand_math() -> Result<()> {
        // rms([3, 4]) = sqrt((9 + 16) / 2) = sqrt(12.5); dividing gives
        // [0.84852814, 1.1313708]. sqrt makes exact equality impossible, so
        // this is the course's first APPROXIMATE assertion: close within a
        // tolerance, not bit-identical.
        let device = Device::Cpu;
        let norm = RmsNorm::new(Tensor::ones(2, candle_core::DType::F32, &device)?, 0.0);
        let out = norm.forward(&Tensor::new(&[[3_f32, 4.0]], &device)?)?;
        let row = &out.to_vec2::<f32>()?[0];
        assert!((row[0] - 0.848_528_1).abs() < 1e-5);
        assert!((row[1] - 1.131_370_8).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn rmsnorm_output_has_unit_rms() -> Result<()> {
        let device = Device::Cpu;
        let norm = RmsNorm::new(Tensor::ones(4, candle_core::DType::F32, &device)?, 0.0);
        let out = norm.forward(&Tensor::new(&[[1_f32, -2.0, 3.0, -4.0]], &device)?)?;
        let row = &out.to_vec2::<f32>()?[0];
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn rope_leaves_position_zero_untouched() -> Result<()> {
        // Position 0 means angle 0 everywhere: rotating by nothing.
        let device = Device::Cpu;
        let x = Tensor::new(&[[1_f32, 2.0, 3.0, 4.0]], &device)?;
        assert_eq!(rope(&x)?.to_vec2::<f32>()?, x.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn identical_tokens_differ_by_position() -> Result<()> {
        let device = Device::Cpu;
        let same = [1_f32, 2.0, 3.0, 4.0];
        let x = Tensor::new(&[same, same], &device)?;
        let out = rope(&x)?.to_vec2::<f32>()?;
        assert_ne!(out[0], out[1]);
        Ok(())
    }

    #[test]
    fn rotation_preserves_vector_length() -> Result<()> {
        let device = Device::Cpu;
        let v = [0.5_f32, -1.0, 2.0, 0.25];
        let out = rope(&Tensor::new(&[v, v], &device)?)?.to_vec2::<f32>()?;
        assert!((length(&out[1]) - length(&v)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn dot_products_depend_only_on_position_gap() -> Result<()> {
        // q at position 1 against k at position 4 (gap 3), then both
        // shifted by +3 (positions 4 and 7): attention's dot product must
        // not change — RoPE encodes RELATIVE position.
        let device = Device::Cpu;
        let q = [1_f32, 2.0, 3.0, 4.0];
        let k = [0.5_f32, -1.0, 2.0, 0.25];
        let near = dot(&rotated(q, 1, &device)?, &rotated(k, 4, &device)?);
        let far = dot(&rotated(q, 4, &device)?, &rotated(k, 7, &device)?);
        assert!((near - far).abs() < 1e-4);
        Ok(())
    }

    /// The vector `v`, rope-rotated as if it sat at position `pos`.
    fn rotated(v: [f32; 4], pos: usize, device: &Device) -> Result<Vec<f32>> {
        let rows: Vec<f32> = std::iter::repeat_n(v, pos + 1).flatten().collect();
        let x = Tensor::from_vec(rows, (pos + 1, 4), device)?;
        Ok(rope(&x)?.to_vec2::<f32>()?.swap_remove(pos))
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn length(v: &[f32]) -> f32 {
        v.iter().map(|a| a * a).sum::<f32>().sqrt()
    }
}
