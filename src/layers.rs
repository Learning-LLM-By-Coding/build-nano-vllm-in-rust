use candle_core::{Result, Tensor};

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
}
