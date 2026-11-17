# Day 5: The decoder block

> **Day 5 of 15** · builds on `v1-day-04` · ≈ 1 h

| Section | Builds | Checkpoint |
|---------|--------|------------|
| A | The SwiGLU feed-forward | `v1-day-05a` |
| B | The decoder block | `v1-day-05b` = `v1-day-05` |

**Pick your mode before starting** (explained fully in
[Setup](setup.md#4-choose-how-you-follow-along)):

```bash
# Build mode — start (or restart) your branch from yesterday's checkpoint.
# Already on my-course from Day 4? Just keep going — no command needed.
git switch -c my-course v1-day-04

# Inspect mode — read along with the finished code instead:
git switch --detach v1-day-05a         # then 05b as you pass the section
```

<div class="callout">

**Today's typing budget** — so nothing sneaks up on you:

| Section | The layer (the learning) | Wiring & tests (copy-friendly) |
|---------|--------------------------|--------------------------------|
| A | `Mlp` · ~20 lines | `mod` line, 1 demo addition, 1 test |
| B | `Block` · ~25 lines | a 3-name demo rename, demo finish, 3 tests, 1 CLI test |

Two sections are two natural sittings, and every checkpoint is a save
point — nothing says a "day" must fit in one. After yesterday's climb,
today is a short walk downhill. You write two small structs that snap
everything you already built into one unit.

</div>

## 1. Why we need it

**The story so far** — remember [the restaurant](story.md): the kitchen (Day 1), the
cook who learns from old tickets (Day 2), the dining room where [guests](story.md#guest) (one per token of your
prompt) got their [profile cards](story.md#embedding) (Day 3). And last
night, at last, came the [table conversation](story.md#attention): every guest walked away with a card full of borrowed
information (Day 4). Today the evening becomes something the restaurant
can *repeat*, night after night, forty rounds deep.

Why repeat it? One round of conversation is not a model. A real LLM runs
dozens of rounds in a row, each with its own learned weights. Here is the
whole machine, with the day each part arrives:

```text
token ids           the prompt as numbers (Day 1's tokenizer)
    │ embedding — each id becomes a vector of numbers (Day 3)
    ▼
▶ block ◀           attention (Day 4) + the digest, packaged — TODAY'S BUILD
    │ …then the same block again, dozens of copies with their own
    │ learned weights (Day 6 stacks them)
    ▼
scoring layer       the final vectors become one score per possible
    │               next token (Day 6)
    ▼
next token          highest score wins (Day 1's greedy pick) — and
                    Day 2's loop feeds it back in for the token after
```

The repeating unit in the middle is the **transformer decoder block**, or
**block** for short. Your first instinct might be to make Day 4's
`Attention` the block and stack it forty times. Three things go wrong:

- **Nobody digests what they heard.** Attention moves information
  *between* guests, but each guest only walks away with a share-weighted
  average of other guests' statements. Embedding and `Linear` are weighted
  sums too, and stacked weighted sums collapse: algebra can fold any
  number of purely linear steps into one equivalent step. Softmax and
  RMSNorm are not linear, but neither bends the card's numbers one by one.
  Softmax only decides *how much* of each guest to blend, and RMSNorm
  scales the whole card by one number. So forty rounds could only keep
  re-mixing what is already on the cards.

  The fix is a **digest**: a small
  network each guest runs alone on its own card, with a curve in the
  middle. That curve, a **nonlinearity**, is the block's only *per-feature*
  nonlinearity: the one step that sends each of the card's numbers through
  a bend of its own. It is what lets round two learn something round one
  could not.

- **Rounds erase the cards.** Suppose every round *replaced* each guest's
  card with that round's output. The original guest would be gone within
  a few rounds, and forty rounds deep, the model would long ago have
  erased its own input. The fix is to *add* each round's output onto the
  card instead, like a note penciled onto it. That addition is a
  **residual connection**.

- **The notes snowball.** Each round now adds more onto the card, so the
  card's numbers can grow round after round. A step that reads bigger
  numbers writes bigger notes, and the growth feeds itself. The fix is
  Day 3's [volume knob](story.md#rmsnorm) (`RmsNorm`), turned before every
  step on the copy that step reads. Leveling first is called
  **pre-norm**.

You build these fixes in two sections. Section A builds the digest (the
SwiGLU **feed-forward**). Section B packages the conversation, the digest,
two volume knobs, and two residual additions into one `Block`. Along the
way the demo grows into `cargo run -- block`, which ends with a live
experiment: it changes the last token and checks that the earlier tokens'
results do not move.

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=aircAruvnKk">▶ what a per-token network is (3Blue1Brown)</a>
<a href="https://nn.labml.ai/transformers/feed_forward.html">gated feed-forward networks, formally (labml)</a>
</div>

## 2. Mental model

Tonight the evening earns its rhythm. Walk its last two rituals, then
watch the whole thing get packaged:

- **Think it over, alone.** After the conversation, each guest digests.
  They take two fresh readings of their card. One reading goes through a
  soft switch (`silu`, defined in section A) and becomes the **gate**.
  Each gate number decides how much of the matching number in the other
  reading gets through. The result is mixed back down to card size. That
  is the SwiGLU feed-forward, all per-guest, with no cross-talk. (The
  conversation decides *where* information flows. The digest is where
  each guest *makes something of it*.)

- **Pencil, never rewrite.** Neither the conversation nor the digestion
  *replaces* a guest's card. Their output is *added on*, like notes
  penciled onto the original. These **residual connections** mean the
  card always survives. After forty rounds, each guest is still
  recognizably themselves, only heavily annotated.

- **The knob turns before every activity.** Before the conversation, and
  again before the digestion, the familiar volume knob levels the copy of
  the card that the activity reads. Knob-first is **pre-norm**, and it is
  what the Llama/Qwen family actually runs.

- **One evening, packaged.** A **block** is that whole sequence (knob,
  conversation, pencil; knob, digestion, pencil) as a single struct. A
  real model runs this same evening dozens of times in a row. The
  choreography never changes, but every round carries its own learned
  weights, so every round looks for something new. If you could watch one
  card travel down the stack, you would see it grow richer round by round.
  The early rounds pick up surface things, like spellings and word
  endings, and the later rounds begin to catch meaning. Today you build
  one round, and Day 6 stacks them.

```text
x [3, 8]            3 guests × 8 numbers on each card
    │ volume knob (RmsNorm)
    │ the conversation (Day 4's Attention)
    ▼
x + notes [3, 8]    what each guest heard, penciled onto their own card
    │ volume knob again (a second RmsNorm)
    │ think it over (the SwiGLU Mlp — per guest, no cross-talk)
    ▼
x + more [3, 8]     conclusions penciled in too — same card size as it arrived
```

The reading rule is the same as always: every shape is `[how many
guests, how many numbers each]`. A block never changes the shape, and
that is exactly what makes it stackable.

## 3. Inference concepts introduced

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://nn.labml.ai/transformers/feed_forward.html">gated feed-forward networks (labml)</a>
<a href="https://www.youtube.com/watch?v=aircAruvnKk">▶ neural nets &amp; nonlinearity, visually (3Blue1Brown)</a>
</div>

- **Nonlinearity is why depth exists.** Without it, stacking layers is
  cosmetic, because any pile of linear steps collapses into one. The
  `silu` gate is the block's only per-feature nonlinearity. It is what
  lets each stacked block capture patterns the previous one couldn't.

- **Residuals + pre-norm are why depth survives.** Each sublayer only
  *adds* its contribution, and it computes that contribution from a
  leveled copy. So signals can't explode or vanish across dozens of
  blocks. That is the same reason RMSNorm existed at all, now applied at
  the block scale.

- **The feed-forward is where most of the model lives.** It expands each
  card to a wider hidden size and back (8 → 16 → 8 here). Those two fat
  matrices dominate the parameter count. In today's toy, the digest holds
  384 weights (three matrices of 8 × 16 = 128) against attention's 192
  (8 × 8 + 8 × 4 + 8 × 4 + 8 × 8). Real models keep the digest the biggest
  part too. In Qwen3-0.6B it holds about 1.5 weights for every attention
  weight: 3 × 1024 × 3072 ≈ 9.4 million against about 6.3 million (Day 6
  prints these shapes). When Day 9 measures where the time goes, remember
  where the weights are.

- **"A 28-layer model" means 28 of these blocks.** The whole engine will
  be organized around the block. Day 6 stacks them, and Day 8's KV cache
  will be bookkept *per block*.

## 4. Build it, section by section

Today you add one new file, `src/block.rs`, which holds the digestion and
the packaging. You compose it entirely from pieces you already built.

### Section A — the SwiGLU feed-forward

<div class="rust-concepts"><span>Rust used here:</span> nothing new — one
struct, three fields, straight-line tensor calls.</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://nn.labml.ai/transformers/feed_forward.html">gated feed-forward networks, formally (labml)</a>
</div>

After all that listening, every guest needs a quiet moment to think.
Create `src/block.rs` and give them one, the private thinking step:

*(Typing it out builds finger-memory, but copying it straight from here is
just as legitimate.)*

```rust,ignore
use candle_core::{Result, Tensor};

use crate::layers::Linear;

/// Analogy: after the conversation, each guest thinks it over alone — two
/// independent readings of their own card, one deciding how much of the
/// other to keep.
///
/// SwiGLU: down(silu(gate(x)) * up(x)). up widens the card into scratch
/// space; gate decides, slot by slot, how much survives (silu sends
/// negative readings toward zero and lets confident ones pass); down
/// shrinks the result back to card size. The block's only per-feature nonlinearity.
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
```

You typed three translators there, so meet them properly. `up_proj`
widens each card into a bigger scratch space, room to think. In today's
demo, 8 numbers become 16. `gate_proj` reads the *same* card and produces
one control value per scratch slot. And `down_proj` shrinks the finished
thought back to card size. The flow in `forward` is exactly that: widen,
gate, shrink.

The `*` in `gate * self.up_proj.forward(x)?` is not a matmul. Both sides
have the same shape, and `*` multiplies them slot by slot: gate number 0
times scratch number 0, gate number 1 times scratch number 1, and so on.

Now the soft switch itself. It is built from **sigmoid**,
`sigmoid(g) = 1 / (1 + e^(-g))`, which squashes any number into the range
0 to 1. `silu(g) = g × sigmoid(g)` is a **soft switch**, not a 0-to-1
value. A large positive `g` passes through almost unchanged, a large
negative `g` comes out near 0, and in between the curve bends smoothly
(its lowest point is about −0.28).

So why a *gate*, instead of squashing the widened card with some fixed
function? A fixed squash treats every slot the same way, every time. The
gate is computed from its own reading of the card, so it decides *per
slot, per card*. `silu` sends a strongly negative gate reading toward
zero, and multiplying by nearly-zero erases that slot's thought. A
confident reading lets its thought through almost unchanged. So the
network learns not just *what to compute* but *what to keep*.
That gate-times-value pattern is the "GLU" family. The `silu`-gated
variant is the one the Llama/Qwen models actually ship, and Candle
provides `silu()` ready-made.

The file is new, so declare it. In `src/main.rs`, add this line between
`mod attention;` and `mod layers;`:

```rust,ignore
mod block;
```

Pin it with one hand-checked test at the bottom of `src/block.rs`. The
numbers are small enough to verify on a napkin. With every weight 1, all
three translators copy the input, so the output is `silu(1) × 1`. And
`sigmoid(1) = 1 / (1 + 1/e) = 1 / 1.368 = 0.731`:

```rust,ignore
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
```

Now you are back to **wiring**, so copy freely. Watch the digest run. In
`src/main.rs`, at the end
of `run_attention_demo` (after the `attn:` print, before `Ok(())`), build
one and push the conversation's output through it. The `⋮` and the framing
lines are already in your file. Type only what sits between the two
markers. If you type the marker comments too, no harm done:

```rust,ignore
fn run_attention_demo() -> Result<()> {
    ⋮
    ⋮
    let out = attn.forward(&leveled)?;
    println!("attn:   {:?}", out.dims());

    // ──────── new code starts here (around line 145) ────────
    // The digest, alone for now: run what the guests carried away through
    // one think-it-over step. Day 5 section B packages both into the block.
    let mlp = block::Mlp::new(
        layers::Linear::new(ramp(8, 16, 0.01, &device)?),
        layers::Linear::new(ramp(8, 16, 0.02, &device)?),
        layers::Linear::new(ramp(16, 8, 0.01, &device)?),
    );
    println!("mlp:    {:?}", mlp.forward(&out)?.dims());
    // ─────────────────── new code ends here ─────────────────────
    Ok(())
}
```

Look at the widths: `8 → 16` up, then `16 → 8` back down. The expansion
gives the digest room to work, and it is where most of a real model's
weights live. `cargo run -- attention` now ends with one more line:

```text
text "abc" -> ids [97, 98, 99]
cards:  [3, 8]
scores (how well each question matches each topic card):
           seat 0  seat 1  seat 2
  seat 0 |  2.828   2.502   1.820
  seat 1 |  2.502   2.828   2.502
  seat 2 |  1.820   2.502   2.828
masked (etiquette: every score in the future becomes -inf):
           seat 0  seat 1  seat 2
  seat 0 |  2.828    -inf    -inf
  seat 1 |  2.502   2.828    -inf
  seat 2 |  1.820   2.502   2.828
who listens to whom (each row sums to 1; 0.000 = the future):
           seat 0  seat 1  seat 2
  seat 0 |  1.000   0.000   0.000
  seat 1 |  0.419   0.581   0.000
  seat 2 |  0.175   0.346   0.479
heard:  [3, 8]
q proj: 8 in -> 8 out (4 heads x 2);  k,v proj: 8 in -> 4 out (2 SHARED heads x 2)
attn:   [3, 8]
───────────── everything above: Day 4's output, unchanged ─────────────
mlp:    [3, 8]
```

`mlp:    [3, 8]` is the digest's result: 3 guests × 8 numbers, the same
shape attention handed it. Inside, each card was widened to 16 numbers,
gated, and shrunk back to 8.

**Section checkpoint:** `./scripts/compare.sh v1-day-05a`

### Section B — the decoder block

<div class="rust-concepts"><span>Rust used here:</span> nothing new —
one struct composed of everything already built.</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=aircAruvnKk">▶ what the per-token network is (3Blue1Brown)</a>
</div>

Now comes the payoff of five days of parts: boxing up the evening. The
`Block` owns Day 4's `Attention`, today's `Mlp`, and two volume knobs of
its own. So first widen the imports. At the top of `src/block.rs`,
replace the `use crate::layers::Linear;` line with these two, so the code
below can name all three:

```rust,ignore
use crate::attention::Attention;
use crate::layers::{Linear, RmsNorm};
```

Next comes the block itself. It goes between `Mlp`'s closing brace and
the `#[cfg(test)]` tests module. The frame shows both neighbors, and the
markers bracket what you type:

*(It is short and load-bearing: two lines of `forward` carry the residual
idea.)*

```rust,ignore
⋮
        let gate = self.gate_proj.forward(x)?.silu()?;
        self.down_proj.forward(&(gate * self.up_proj.forward(x)?)?)
    }
}

// ──────── new code starts here (around line 35) ────────
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
// ──────── new code ends here ────────

#[cfg(test)]
mod tests {
    ⋮
```

Look at the shape of both lines in `forward`: `x + something`. That little
`+` is the residual, the pencil. Whatever the conversation or the digest
produces gets *added onto* the card, and the card itself is never
replaced.

The two lines are written slightly differently, for Rust reasons. Every
Candle addition returns a `Result`. In the first line, `x` is the
function's argument, already a borrowed `&Tensor`, so `x + …` only reads
it. The `( … )?` unwraps the `Result`, and the new `x` is a plain `Tensor`
that replaces the old name. In the second line, `x` is that owned
`Tensor`. Writing `x + …` would *move* it into the addition while the
right-hand side still needs to read it, so the line *borrows* it with
`&x`. That line needs no `?`, because its `Result` is exactly what
`forward` returns.

One subtlety is worth slowing down for. Each sublayer *reads* a leveled
copy of the card. That is the `self.attn_norm.forward(x)?` tucked inside
the parentheses. But the sublayer's result is added onto the *original*,
un-leveled `x`. The volume knob only ever changes what a sublayer *sees*,
never what the card *is*. This arrangement, leveling the copy you read
and adding onto the original, is called **pre-norm**. It is exactly what
the Llama/Qwen family runs.

Three tests pin the block: the residual guarantee, the shape guarantee,
and causality end to end. They live in the existing `tests` module, below
`silu_gate_matches_hand_math`, with two small helpers they share. Make one
tiny edit first: the module's `use` line gains `DType`, so make it
`use candle_core::{DType, Device};`. Then add:

```rust,ignore
#[cfg(test)]
mod tests {
    ⋮
    ⋮
    // ──────── new code starts here (around line 56) ────────
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
    // ──────── new code ends here ────────
}
```

`a_silent_conversation_changes_nobody` makes the residual idea testable.
Zero out every sublayer, and the block is a perfect no-op, because the
card's own path through `forward` is a bare `+`. Depth can only ever *add*.

Here comes one last stretch of **wiring**, so copy freely. Your demo shows
a full block now and has outgrown its name. Make five small edits, in
order:

1. In `src/main.rs`, rename `run_attention_demo` to `run_block_demo`, in
   both its definition and the call in the `match` arm.

2. In the same `match`, change the arm's pattern `["attention"]` to
   `["block"]`.

3. In the usage message just below, change the line to
   `cargo run -- block`.

4. In `scripts/check.sh`, make the same swap in the smoke command:
   `cargo run --quiet -- block`.

5. Back in `run_block_demo`, replace the `mlp:` print line with the block
   assembly and the day's headline, shown next. Keep the `let mlp = …`
   construction above it.

The `⋮`, the framing lines, and the `let mlp` construction are already in
your file. The markers bracket what's new:

```rust,ignore
fn run_block_demo() -> Result<()> {
    ⋮
    ⋮
    // The digest — built here, run inside the block below.
    let mlp = block::Mlp::new(
        layers::Linear::new(ramp(8, 16, 0.01, &device)?),
        layers::Linear::new(ramp(8, 16, 0.02, &device)?),
        layers::Linear::new(ramp(16, 8, 0.01, &device)?),
    );

    // ──────── new code starts here (around line 154) ────────
    // The full round of dinner: both knobs, the conversation, reflection.
    let block = block::Block::new(
        layers::RmsNorm::new(Tensor::ones(8, DType::F32, &device)?, 1e-5),
        attn,
        layers::RmsNorm::new(Tensor::ones(8, DType::F32, &device)?, 1e-5),
        mlp,
    );
    let out = block.forward(&cards)?;
    println!("block:  {:?}", out.dims());

    // The day's headline, live: swap only the LAST guest — the earlier
    // rows must not move, because the future cannot reach back.
    let swapped = block.forward(&embed.forward(&tokenizer::encode("abz"))?)?;
    let (a, b) = (out.to_vec2::<f32>()?, swapped.to_vec2::<f32>()?);
    println!(
        "swap guest 3 \"c\"->\"z\": row 0 moved: {}, row 1 moved: {}, row 2 moved: {}",
        a[0] != b[0],
        a[1] != b[1],
        a[2] != b[2]
    );
    // ─────────────────── new code ends here ─────────────────────
    Ok(())
}
```

Two small things in that code are worth noticing. First, the demo feeds
the block the raw `cards`, not `leveled`. Turning the volume knobs is the
block's own job now, so the demo hands it untouched cards and lets it
level them inside. Second, there is an ownership detail. `Block::new(…)`
*moves* `attn` and `mlp` into the block. After that line they belong to
the block, and the old variables can't be used again. That is why the
demo builds the block *after* the `attn:` print: one last use, then the
demo hands them over.

The headline is now user-facing behavior, so pin it from
the outside in `tests/cli.rs`, above
`missing_subcommand_exits_nonzero_with_usage`:

```rust,ignore
#[test]
fn block_demo_shows_causality() {
    let first = run_cli(&["block"]);
    let second = run_cli(&["block"]);
    assert!(first.status.success());
    assert_eq!(first.stdout, second.stdout);
    let text = String::from_utf8(first.stdout).expect("stdout is text");
    assert!(text.contains("row 0 moved: false"));
    assert!(text.contains("row 2 moved: true"));
}
```

**Section checkpoint:** `./scripts/compare.sh v1-day-05b` (identical to
`v1-day-05`, because the last section of a day *is* the day).

## 5. Run it

```bash
cargo run -- block
```

## 6. Expected result

```text
text "abc" -> ids [97, 98, 99]
cards:  [3, 8]
scores (how well each question matches each topic card):
           seat 0  seat 1  seat 2
  seat 0 |  2.828   2.502   1.820
  seat 1 |  2.502   2.828   2.502
  seat 2 |  1.820   2.502   2.828
masked (etiquette: every score in the future becomes -inf):
           seat 0  seat 1  seat 2
  seat 0 |  2.828    -inf    -inf
  seat 1 |  2.502   2.828    -inf
  seat 2 |  1.820   2.502   2.828
who listens to whom (each row sums to 1; 0.000 = the future):
           seat 0  seat 1  seat 2
  seat 0 |  1.000   0.000   0.000
  seat 1 |  0.419   0.581   0.000
  seat 2 |  0.175   0.346   0.479
heard:  [3, 8]
q proj: 8 in -> 8 out (4 heads x 2);  k,v proj: 8 in -> 4 out (2 SHARED heads x 2)
attn:   [3, 8]
block:  [3, 8]
swap guest 3 "c"->"z": row 0 moved: false, row 1 moved: false, row 2 moved: true
```

Everything from Day 4 still holds: scores, mask, shares, and the GQA
shapes. Read the two new lines against what you built:

- `block:  [3, 8]` — `Block::forward` ran the whole evening (knob,
  conversation, pencil; knob, digest, pencil). It handed back cards the
  same size they arrived, which is the stackability guarantee, live.

- `swap guest 3 "c"->"z": row 0 moved: false, row 1 moved: false, row 2
  moved: true` — the demo ran the block on `"abc"`, ran it again on
  `"abz"`, and compared the outputs row by row. The label counts guests
  from 1, the way you'd say it aloud, so guest 3 is the third token, `c`.
  The rows count from 0, like Day 4's seats, so guest 3's output is row 2.
  Rows 0 and 1 are bit-identical across the two runs, because
  `mask_future` gives the changed guest exactly zero share in them, and
  every other step in the block works on each guest alone. Only row 2
  moved.

This one-way-street rule is **causality**: a token can never affect the
tokens before it, the way changing a novel's last page cannot rewrite its
first chapter. You observed it through the complete block instead of
taking it on faith. The engine cares because generation adds one token at
a time. If earlier tokens' work stays valid whatever comes next, the
engine can compute that work once and *keep* it. Day 8 turns exactly that
into the engine's biggest speed trick.

## 7. Test it

Your five new tests (one for the digest, three for the block, one CLI
pin) bring the suite to 36 (32 unit, 4 integration):

```bash
cargo test block
```

```text
running 4 tests
test block::tests::silu_gate_matches_hand_math ... ok
test block::tests::the_block_keeps_the_card_size ... ok
test block::tests::a_silent_conversation_changes_nobody ... ok
test block::tests::the_whole_block_is_still_causal ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 28 filtered out; finished in 0.00s

running 1 test
test block_demo_shows_causality ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.23s
```

## 8. Break it deliberately

Now remove one pencil. In `Block::forward`, drop the second residual, so
the block returns the digest's output *instead of* adding it on:

```diff
-        &x + self.mlp.forward(&self.mlp_norm.forward(&x)?)?
+        self.mlp.forward(&self.mlp_norm.forward(&x)?)
```

`cargo test`:

```text
test block::tests::a_silent_conversation_changes_nobody ... FAILED

  left: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...
 right: [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], ...
```

The guests were erased. With zero weights, a digest that *replaces* the
card hands back all zeros, and the original cards (`right`) are gone
entirely.
That is precisely what "rewrite instead of annotate" does. A 28-block
model built this way would overwrite its own input 28 times before
predicting anything. Notice what stayed green, too. The causality test
still passes. The digest works on each guest alone and never mixes
guests, so even a broken digest can't carry the future backward.
One property can be perfectly healthy while another is catastrophically
broken. That is why the block carries three independent guarantees
instead of one. Restore the `&x +` and confirm `cargo test` is green.

## 9. Checkpoint

Every checkpoint, each section and the day alike, is done only when the
full green bar passes:

```bash
./scripts/check.sh
```

Then close out the day in whichever mode you are following:

**Built along on your own branch?** Confirm your code matches the finished
day. The final ✓ line means it does:

```bash
./scripts/compare.sh v1-day-05
```

**Inspecting instead of typing?** Jump to today's exact finished state,
poke at it, and come back:

```bash
git switch --detach v1-day-05
cargo test
git switch -          # back to where you were
```

**Either way**, today's complete change set, docs included, reads as one
diff:

```bash
git diff v1-day-04 v1-day-05
```

## 10. Recap

**What changed today:** you wrote `src/block.rs`. It holds the SwiGLU
`Mlp` (`down(silu(gate(x)) * up(x))`, the block's only per-feature nonlinearity) and
the pre-norm decoder `Block` with residual connections. The demo grew
into `cargo run -- block`, and it now demonstrates causality through the
complete block, pinned by a CLI test. You also wrote five tests, including
the residual no-op guarantee (zero weights leave every card
bit-identical).

**Deliberately naive:** one block stands alone, while real models stack
dozens (Day 6). The weights are still ramp-filled fakes (Day 6). Qwen's
extra per-head q/k normalization also waits for the real checkpoint
(Day 6).

**Takeaways:**

- A block is attention plus a digest, wrapped in two volume knobs and two
  residual additions. It keeps the card size, so a model can stack it
  dozens of times.

- Without a per-feature nonlinearity, stacked rounds collapse into one.
  The `silu` gate in the SwiGLU feed-forward is that nonlinearity.

- Add each sublayer's result onto the card instead of replacing it. The
  residual is why the original token survives forty rounds.

- Level the copy a sublayer reads, never the card itself (pre-norm). That
  keeps the numbers from snowballing as notes pile up.

- Only attention mixes guests, and its mask points backward. So the whole
  block is causal, and Day 8 can keep earlier tokens' work.

**Tomorrow's limitation, resolved on Day 6:** the machine is complete,
but the mind is fake. Ramp weights make grammatical-looking shapes and
meaningless content. Day 6 hires the trained chef. You stack the blocks,
download a real checkpoint, and load its actual weights into these exact
structs, putting a real mind in the machine you built by hand. (One
piece stays borrowed a day longer: real *text* needs the real tokenizer,
and that is Day 7's opening move.)
