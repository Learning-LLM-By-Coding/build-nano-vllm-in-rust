use candle_core::{Result, Tensor};

use crate::layers::Linear;

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

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
}
