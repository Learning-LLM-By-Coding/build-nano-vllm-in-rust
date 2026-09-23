use candle_core::{D, Result, Tensor};

use crate::layers::{Linear, rope};

/// Analogy: how well one guest's question matches another's topic card —
/// every question dotted against every topic card, in one matmul.
///
/// scores[i][j] = q_i . k_j / sqrt(head_dim). Longer cards make raw dot
/// products bigger by pure arithmetic (more terms in the sum), so dividing
/// by sqrt(len) keeps scores comparable whatever the card size.
pub fn scaled_scores(q: &Tensor, k: &Tensor) -> Result<Tensor> {
    let (_seq, head_dim) = q.dims2()?;
    q.matmul(&k.t()?)? * (1.0 / (head_dim as f64).sqrt())
}

/// Analogy: table etiquette — a guest may listen only to guests seated
/// before them, who have already spoken; later seats have not.
///
/// Every score in a seat's future becomes -inf, which softmax will turn
/// into a share of exactly zero.
pub fn mask_future(scores: &Tensor) -> Result<Tensor> {
    let (rows, cols) = scores.dims2()?;
    let mask: Vec<f32> = (0..rows)
        .flat_map(|i| (0..cols).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    scores + Tensor::from_vec(mask, (rows, cols), scores.device())?
}

/// Analogy: split one guest's attention into shares that sum to 1 — a
/// high score earns a big share, -inf gets exactly none.
///
/// Softmax: exp each score, divide by the row's total. The row max is
/// subtracted first so exp never overflows; the shares are unchanged,
/// because the top and the bottom of the division both shrink by the same
/// e^max.
pub fn softmax_rows(scores: &Tensor) -> Result<Tensor> {
    let max = scores.max_keepdim(D::Minus1)?;
    let exp = scores.broadcast_sub(&max)?.exp()?;
    let total = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&total)
}

/// Analogy: one full listening round at the table. Each guest asks their
/// question (q) of every earlier guest's topic card (k), splits their
/// attention into shares, and walks away with the share-weighted blend of
/// what those guests actually said (v).
///
/// Scaled dot-product attention for a single head.
pub fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    softmax_rows(&mask_future(&scaled_scores(q, k)?)?)?.matmul(v)
}

/// Analogy: the table conversation, packaged. Several conversations run
/// at once — each head is one ear, listening for its own kind of
/// connection — and to save table space, heads share topic/statement cards
/// in groups.
///
/// Grouped-query attention: n_heads askers over n_kv_heads shared kv heads
/// — the choice that will shrink Day 8's mise-en-place shelf. RoPE is
/// applied to q and k only: position should shape whom you listen to,
/// never what you say.
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
}

impl Attention {
    pub fn new(
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Self {
        assert!(
            n_heads.is_multiple_of(n_kv_heads),
            "every shared kv head serves an equal group of q heads"
        );
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads,
            n_kv_heads,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // From every guest's card: questions, topic cards, statements.
        let q = self.q_proj.forward(x)?; // [seq, n_heads * head_dim]
        let k = self.k_proj.forward(x)?; // [seq, n_kv_heads * head_dim]
        let v = self.v_proj.forward(x)?; // [seq, n_kv_heads * head_dim]
        // The projected widths carry the geometry — k is n_kv_heads cards wide.
        let head_dim = k.dims2()?.1 / self.n_kv_heads;
        let group = self.n_heads / self.n_kv_heads;
        let mut heard = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            // This head's slice of every question — and the SHARED kv slice
            // its group listens through (h / group picks the group's card).
            let q_h = rope(&q.narrow(1, h * head_dim, head_dim)?)?;
            let kv = h / group;
            let k_h = rope(&k.narrow(1, kv * head_dim, head_dim)?)?;
            let v_h = v.narrow(1, kv * head_dim, head_dim)?;
            heard.push(attend(&q_h, &k_h, &v_h)?);
        }
        // Stitch what the ears heard back side by side, then re-mix once.
        self.o_proj.forward(&Tensor::cat(&heard, 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn scores_match_hand_arithmetic() -> Result<()> {
        // q = [1, 2] against k = [3, 4]: dot product 11, head_dim 2,
        // 11 / sqrt(2) = 7.778175.
        let device = Device::Cpu;
        let q = Tensor::new(&[[1_f32, 2.0]], &device)?;
        let k = Tensor::new(&[[3_f32, 4.0]], &device)?;
        let s = scaled_scores(&q, &k)?.to_vec2::<f32>()?;
        assert!((s[0][0] - 7.778_175).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn softmax_rows_are_shares_that_sum_to_one() -> Result<()> {
        // exp(0) = 1 and exp(ln 4) = 4, so the shares must be 1/5 and 4/5.
        let device = Device::Cpu;
        let x = Tensor::new(&[[0_f32, 4_f32.ln()]], &device)?;
        let w = softmax_rows(&x)?.to_vec2::<f32>()?;
        assert!((w[0][0] - 0.2).abs() < 1e-5);
        assert!((w[0][1] - 0.8).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn future_seats_get_a_zero_share() -> Result<()> {
        // Identical scores everywhere — yet after the mask, everything
        // above the diagonal must come out as exactly 0.0, not merely small.
        let device = Device::Cpu;
        let scores = Tensor::ones((3, 3), candle_core::DType::F32, &device)?;
        let w = softmax_rows(&mask_future(&scores)?)?.to_vec2::<f32>()?;
        for (i, row) in w.iter().enumerate() {
            for (j, share) in row.iter().enumerate() {
                if j > i {
                    assert_eq!(*share, 0.0, "seat {i} heard future seat {j}");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn seat_zero_hears_only_its_own_statement() -> Result<()> {
        // The first guest has nobody before them: their listening round
        // returns exactly their own statement, bit for bit.
        let device = Device::Cpu;
        let q = Tensor::new(&[[1_f32, 0.0], [0_f32, 1.0]], &device)?;
        let k = Tensor::new(&[[2_f32, 1.0], [1_f32, 3.0]], &device)?;
        let v = Tensor::new(&[[5_f32, 6.0], [7_f32, 8.0]], &device)?;
        let out = attend(&q, &k, &v)?.to_vec2::<f32>()?;
        assert_eq!(out[0], vec![5.0, 6.0]);
        Ok(())
    }

    /// Deterministic test weights: value = index * scale.
    fn counting(rows: usize, cols: usize, scale: f32, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = (0..rows * cols).map(|v| v as f32 * scale).collect();
        Tensor::from_vec(data, (rows, cols), device)
    }

    /// A 4-head attention over 8-number cards, with 2 shared kv heads.
    fn tiny_attention(device: &Device) -> Result<Attention> {
        Ok(Attention::new(
            Linear::new(counting(8, 8, 0.02, device)?),
            Linear::new(counting(8, 4, 0.03, device)?),
            Linear::new(counting(8, 4, 0.05, device)?),
            Linear::new(counting(8, 8, 0.02, device)?),
            4,
            2,
        ))
    }

    #[test]
    fn every_guest_leaves_with_a_full_card() -> Result<()> {
        // 4 heads of width 2 read 8-number cards and write 8-number cards.
        let device = Device::Cpu;
        let x = counting(3, 8, 0.1, &device)?;
        assert_eq!(tiny_attention(&device)?.forward(&x)?.dims(), &[3, 8]);
        Ok(())
    }

    #[test]
    fn the_future_cannot_reach_back() -> Result<()> {
        // Change the LAST guest only: every earlier guest's output must be
        // bit-identical. Day 8's KV cache is built entirely on this fact.
        let device = Device::Cpu;
        let attn = tiny_attention(&device)?;
        let mut rows = counting(3, 8, 0.1, &device)?.to_vec2::<f32>()?;
        let a = attn.forward(&Tensor::new(rows.clone(), &device)?)?;
        rows[2] = vec![9.0; 8]; // a different third guest
        let b = attn.forward(&Tensor::new(rows, &device)?)?;
        let (a, b) = (a.to_vec2::<f32>()?, b.to_vec2::<f32>()?);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], b[1]);
        assert_ne!(a[2], b[2]);
        Ok(())
    }

    #[test]
    fn a_shared_topic_card_equals_its_own_copies() -> Result<()> {
        // Sharing is not an approximation: one kv head serving 2 q heads
        // computes exactly what 2 duplicated kv heads would.
        let device = Device::Cpu;
        let (wq, wo) = (
            counting(8, 4, 0.02, &device)?,
            counting(4, 8, 0.02, &device)?,
        );
        let (wk, wv) = (
            counting(8, 2, 0.03, &device)?,
            counting(8, 2, 0.05, &device)?,
        );
        let shared = Attention::new(
            Linear::new(wq.clone()),
            Linear::new(wk.clone()),
            Linear::new(wv.clone()),
            Linear::new(wo.clone()),
            2,
            1,
        );
        let copied = Attention::new(
            Linear::new(wq),
            Linear::new(Tensor::cat(&[&wk, &wk], 1)?),
            Linear::new(Tensor::cat(&[&wv, &wv], 1)?),
            Linear::new(wo),
            2,
            2,
        );
        let x = counting(3, 8, 0.1, &device)?;
        assert_eq!(
            shared.forward(&x)?.to_vec2::<f32>()?,
            copied.forward(&x)?.to_vec2::<f32>()?
        );
        Ok(())
    }
}
