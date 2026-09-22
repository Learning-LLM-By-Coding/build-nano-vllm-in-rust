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
}
