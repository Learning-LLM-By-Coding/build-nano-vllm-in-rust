use candle_core::{Result, Tensor};

use crate::attention::Attention;
use crate::layers::{Linear, RmsNorm};

/// Analogy: after the conversation, each guest thinks it over alone — two
/// independent readings of their own card, one deciding how much of the
/// other to keep.
///
/// SwiGLU: down(silu(gate(x)) * up(x)). up widens the card into scratch
/// space; gate decides, slot by slot, how much survives (silu sends
/// negative readings toward zero and lets confident ones pass); down
/// shrinks the result back to card size. The block's only nonlinearity.
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(gate_proj: Linear, up_proj: Linear, down_proj: Linear) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // silu(g) = g * sigmoid(g): strongly negative readings shut their
        // gate toward zero, confident ones pass through almost unchanged.
        let gate = self.gate_proj.forward(x)?.silu()?;
        self.down_proj.forward(&(gate * self.up_proj.forward(x)?)?)
    }
}

/// Analogy: one full round of dinner. Level the volume, hold the
/// conversation, and pencil what you heard ONTO your card — never rewrite
/// it; level again, think it over, pencil in the conclusions too.
///
/// Pre-norm residual block: x + attn(norm(x)), then x + mlp(norm(x)). The
/// additions are residual connections — the original card always survives,
/// each sublayer only annotates it.
pub struct Block {
    attn_norm: RmsNorm,
    attn: Attention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    pub fn new(attn_norm: RmsNorm, attn: Attention, mlp_norm: RmsNorm, mlp: Mlp) -> Self {
        Self {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Volume knob, conversation, pencil the notes onto the card...
        let x = (x + self.attn.forward(&self.attn_norm.forward(x)?)?)?;
        // ...volume knob, reflection, pencil in the conclusions.
        &x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Deterministic test weights: value = index * scale.
    fn counting(rows: usize, cols: usize, scale: f32, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..rows * cols).map(|v| v as f32 * scale).collect();
        Tensor::from_vec(data, (rows, cols), device)
    }

    /// A full block over 8-number cards: 4 heads (2 shared kv), hidden 16.
    fn tiny_block(scale: f32, device: &Device) -> Result<Block> {
        let attn = Attention::new(
            Linear::new(counting(8, 8, scale, device)?),
            Linear::new(counting(8, 4, scale, device)?),
            Linear::new(counting(8, 4, scale, device)?),
            Linear::new(counting(8, 8, scale, device)?),
            4,
            2,
        );
        let mlp = Mlp::new(
            Linear::new(counting(8, 16, scale, device)?),
            Linear::new(counting(8, 16, scale, device)?),
            Linear::new(counting(16, 8, scale, device)?),
        );
        let ones = Tensor::ones(8, DType::F32, device)?;
        Ok(Block::new(
            RmsNorm::new(ones.clone(), 1e-5),
            attn,
            RmsNorm::new(ones, 1e-5),
            mlp,
        ))
    }

    #[test]
    fn silu_gate_matches_hand_math() -> Result<()> {
        // A 1-wide Mlp with all weights 1 computes silu(x) * x:
        // silu(1) = 1 * sigmoid(1) = 0.731059, so the output is 0.731059.
        let device = Device::Cpu;
        let one = Tensor::new(&[[1_f32]], &device)?;
        let mlp = Mlp::new(
            Linear::new(one.clone()),
            Linear::new(one.clone()),
            Linear::new(one.clone()),
        );
        let out = mlp.forward(&one)?.to_vec2::<f32>()?;
        assert!((out[0][0] - 0.731_059).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn a_silent_conversation_changes_nobody() -> Result<()> {
        // All-zero weights: attention says nothing, the reflection thinks
        // nothing — and thanks to the residuals the cards leave the block
        // bit-identical. The card always survives.
        let device = Device::Cpu;
        let block = tiny_block(0.0, &device)?;
        let x = counting(3, 8, 0.1, &device)?;
        assert_eq!(block.forward(&x)?.to_vec2::<f32>()?, x.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn the_block_keeps_the_card_size() -> Result<()> {
        // [3, 8] in, [3, 8] out: blocks annotate cards, never resize them —
        // which is exactly what lets Day 6 stack them.
        let device = Device::Cpu;
        let block = tiny_block(0.01, &device)?;
        let x = counting(3, 8, 0.1, &device)?;
        assert_eq!(block.forward(&x)?.dims(), &[3, 8]);
        Ok(())
    }

    #[test]
    fn the_whole_block_is_still_causal() -> Result<()> {
        // Same property as Day 4's attention test, proven through the full
        // block: norms, residuals, and the reflection are all per-guest,
        // so causality survives end to end.
        let device = Device::Cpu;
        let block = tiny_block(0.01, &device)?;
        let mut rows = counting(3, 8, 0.1, &device)?.to_vec2::<f32>()?;
        let a = block.forward(&Tensor::new(rows.clone(), &device)?)?;
        rows[2] = vec![9.0; 8]; // a different third guest
        let b = block.forward(&Tensor::new(rows, &device)?)?;
        let (a, b) = (a.to_vec2::<f32>()?, b.to_vec2::<f32>()?);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
        Ok(())
    }
}
