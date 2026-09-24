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
today is a short walk downhill: two small structs that snap everything
you already built into one unit.

</div>

## 1. What you will have running

Yesterday ended with tokens that can finally listen to each other. Today
you finish the machine around that: `cargo run -- block` sends the
three-token prompt through one complete **transformer decoder block** —
the unit real LLMs are made of.

Where does that unit sit in a finished model? Follow a prompt through the
whole machine: tokens walk in through Day 3's embedding, pass through
dozens of identical blocks — copies of the one you build today — and leave
through a single scoring layer that names the next token. Day 6 builds
that entrance and that exit; today is all about the block itself. Here is
the whole machine, with the day each part arrives:

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

And the block is closer than you might think: yesterday already built its
biggest part, the conversation. Only two pieces of the block are missing,
and you build one per section:

- **The digest** — a small per-token network (the SwiGLU **feed-forward**)
  that lets each token quietly make something of what attention just
  handed it. No token sees any other token during this step.

- **The packaging** — the `Block` that runs the evening's whole ritual in
  order: level the volume, hold the conversation, add what was heard;
  level again, digest, add the conclusions. Each step's result is *added
  onto* the token instead of replacing it — a **residual connection**.

And the demo saves the best for last — a tiny experiment. It runs the
block twice: once on `"abc"`, once on `"abz"` — the same prompt except for
the last token — and compares the two outputs, token by token. The first
two tokens come out identical, down to the last bit: changing a token can
never affect the tokens *before* it. That one-way-street rule is called
**causality**, and today you watch the whole block prove it on your
screen.

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=aircAruvnKk">▶ what a per-token network is (3Blue1Brown)</a>
<a href="https://nn.labml.ai/transformers/feed_forward.html">gated feed-forward networks, formally (labml)</a>
</div>

## 2. What you will be seeing in the output

A sneak peek before anything else: today's run keeps everything from
Day 4 and adds exactly two new lines at the very end. Here is the full
run, with a plain-English note under each new line — if they don't fully
click yet, that's exactly what the day is for:

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
q proj [8, 8]: 4 heads;  k,v proj [8, 4]: 2 SHARED heads (half the table space)
attn:   [3, 8]
───────────── everything above: Day 4's output, unchanged ─────────────
block:  [3, 8]
swap guest 3 "c"->"z": row 0 moved: false, row 1 moved: false, row 2 moved: true
```

- `block:  [3, 8]` — the complete unit runs end to end: 3 tokens in, 3
  tokens out, each still carrying 8 numbers. Same size in, same size out —
  which is exactly what lets a real model stack this unit dozens of times.

- the swap line — the day's proof, and worth a moment. The model reads
  strictly left to right, the way you read a sentence one word at a time:
  everything it makes of `a` is finished before it has even seen `c`. If
  that is *really* true, then editing the ending must leave the beginning
  untouched — the way changing a novel's last page cannot rewrite its
  first chapter. So the demo tests exactly that: it runs the block on
  `"abc"`, runs it again on `"abz"` — same prompt, different last token —
  and compares the outputs token by token. The first two tokens come out
  identical to the bit; only the swapped token's own output moved. And
  here is why the engine cares: generation adds one token at a time, so if
  earlier tokens' work stays valid no matter what comes next, that work
  can be computed once and *kept* instead of redone — Day 8 turns exactly
  that into the engine's biggest speed trick.

## 3. Why we need it

**The story so far** — remember [the restaurant](story.md): the kitchen (Day 1), the
cook who learns from old tickets (Day 2), the dining room where [guests](story.md#guest) — one per token of your
prompt — got their [profile cards](story.md#embedding) (Day 3) — and last
night, at last, the [table conversation](story.md#attention): every guest walked away with a card full of borrowed
information (Day 4). Today the evening becomes something the restaurant
can *repeat*, night after night, forty rounds deep.

Two things stand between yesterday's conversation and a repeatable
evening:

- **Everything so far is re-mixing.** Embedding, `Linear`, attention's
  blend — each is a weighted sum. And stacked weighted sums collapse:
  algebra can fold any number of purely linear steps into a single
  equivalent one, so forty rounds of them would teach the model nothing a
  single round couldn't. The digest brings the one ingredient no matmul
  can imitate — a **nonlinearity** (the gate) — and that is what makes
  round two able to learn something round one could not.

- **Nothing protects the cards.** If every round *replaced* each guest's
  card with that round's output, the original guest would be gone within a
  few rounds — forty rounds deep, the model would long ago have erased its
  own input. Residual connections make every round an *annotation* on the
  card instead of a rewrite, and the [volume knob](story.md#rmsnorm) (Day 3's `RmsNorm`)
  re-levels before every step so the annotations can't snowball.

## 4. Mental model

Tonight the evening earns its rhythm. Walk its last two rituals, then
watch the whole thing get packaged:

- **Think it over, alone.** After the conversation, each guest digests:
  two fresh readings of their card, one squashed into a 0-to-1 **gate**
  that decides how much of the other gets through, mixed back down to
  card size. That is the SwiGLU feed-forward — all per-guest, no
  cross-talk. (The conversation decides *where* information flows; the
  digest is where each guest *makes something of it*.)

- **Pencil, never rewrite.** Neither the conversation nor the digestion
  *replaces* a guest's card. Their output is *added on*: notes penciled
  onto the original. These **residual connections** mean the card always
  survives — after forty rounds each guest is still recognizably
  themselves, just heavily annotated.

- **The knob turns before every activity.** Before the conversation and
  again before the digestion, the familiar volume knob levels the card.
  Knob-first is called **pre-norm**, and it is what the Llama/Qwen family
  actually runs.

- **One evening, packaged.** A **block** is that whole sequence — knob,
  conversation, pencil; knob, digestion, pencil — as a single struct. And
  a real model? It runs this same evening dozens of times in a row — that
  is the whole trick. The choreography never changes, but every round carries its own
  learned weights, so every round looks for something new. If you could
  watch one card travel down the stack, you would see it grow richer
  round by round: the early rounds pick up surface things — spellings,
  word endings — and the later rounds begin to catch meaning. Today you
  build one round; Day 6 stacks them.

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

Same reading rule as always: every shape is `[how many guests, how many
numbers each]`. A block never changes the shape — that is exactly what
makes it stackable.

## 5. Inference concepts introduced

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://nn.labml.ai/transformers/feed_forward.html">gated feed-forward networks (labml)</a>
<a href="https://www.youtube.com/watch?v=aircAruvnKk">▶ neural nets &amp; nonlinearity, visually (3Blue1Brown)</a>
</div>

- **Nonlinearity is why depth exists.** Without it, stacking layers is
  cosmetic — any pile of linear steps collapses into one. The `silu` gate
  is the block's only nonlinear moment, and it is what lets each stacked
  block capture patterns the previous one couldn't.

- **Residuals + pre-norm are why depth survives.** Each sublayer only
  *adds* its contribution to an already-leveled input, so signals can't
  explode or vanish across dozens of blocks — the same reason RMSNorm
  existed at all, now applied at the block scale.

- **The feed-forward is where most of the model lives.** It expands each
  card to a wider hidden size and back (8 → 16 → 8 here), and those two
  fat matrices dominate the parameter count — in today's toy the digest
  holds 384 weights against attention's 192, and real models keep
  roughly that ratio. When Day 9 measures where the time goes, remember
  where the weights are.

- **"A 28-layer model" means 28 of these blocks.** The block is the unit
  the whole engine will be organized around — Day 6 stacks them, and Day
  8's KV cache will be bookkept *per block*.

## 6. Build it, section by section

One new file today — `src/block.rs`, the digestion and the packaging —
composed entirely of pieces you already built.

### Section A — the SwiGLU feed-forward

<div class="rust-concepts"><span>Rust used here:</span> nothing new — one
struct, three fields, straight-line tensor calls.</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://nn.labml.ai/transformers/feed_forward.html">gated feed-forward networks, formally (labml)</a>
</div>

After all that listening, every guest needs a quiet moment to think.
Create `src/block.rs` and give them one — the private thinking step:

*(Typing it out builds finger-memory — but copying it straight from here is
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
```

You just typed three translators — let's meet them properly. `up_proj`
widens each card into a bigger scratch space — in today's demo, 8 numbers
become 16 — room to think. `gate_proj` reads the *same* card and produces
one control value per scratch slot. And `down_proj` shrinks the finished
thought back to card size. The flow in `forward` is exactly that: widen,
gate, shrink.

So why a *gate*, instead of just squashing the widened card with some
fixed function? Because a fixed squash treats every slot the same way,
every time. The gate decides *per slot, per card*: `silu` sends a strongly
negative gate reading toward zero — and multiplying by nearly-zero erases
that slot's thought — while a confident reading lets its thought through
almost unchanged. So the network learns not just *what to compute* but
*what to keep*. That gate-times-value pattern is the "GLU" family; the
`silu`-gated variant is the one the Llama/Qwen models actually ship, and
Candle provides `silu()` ready-made.

The file is new, so declare it — in `src/main.rs`, between
`mod attention;` and `mod layers;`:

```rust,ignore
mod block;
```

Pin it with one hand-checked test at the bottom of `src/block.rs` — the
numbers small enough to verify on a napkin:

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

Back to **wiring** — copy freely. Now watch the digest run: in
`src/main.rs`, at the end
of `run_attention_demo` — after the `attn:` print, before `Ok(())` — build
one and push the conversation's output through it. The `⋮` and the framing
lines are already in your file — type only what sits between the two
markers (and if you type the marker comments too, no harm done):

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

Note the widths: `8 → 16` up, `16 → 8` back down — the expansion is where
the digest gets room to work, and where most of a real model's weights
live. `cargo run -- attention` now ends with one more line:

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
q proj [8, 8]: 4 heads;  k,v proj [8, 4]: 2 SHARED heads (half the table space)
attn:   [3, 8]
───────────── everything above: Day 4's output, unchanged ─────────────
mlp:    [3, 8]
```

**Section checkpoint:** `./scripts/compare.sh v1-day-05a`

### Section B — the decoder block

<div class="rust-concepts"><span>Rust used here:</span> nothing new —
one struct composed of everything already built.</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=aircAruvnKk">▶ what the per-token network is (3Blue1Brown)</a>
</div>

Now for the payoff of five days of parts: boxing up the evening. The
`Block` owns Day 4's `Attention`, today's `Mlp`, and two volume knobs of
its own — so first widen the imports: at the top of `src/block.rs`,
replace the `use crate::layers::Linear;` line with these two, so the code
below can name all three:

```rust,ignore
use crate::attention::Attention;
use crate::layers::{Linear, RmsNorm};
```

Then the block itself. It goes between `Mlp`'s closing brace and the
`#[cfg(test)]` tests module — the frame shows both neighbors, and the
markers bracket what you type:

*(Short and load-bearing: two lines of `forward` carry the residual idea.)*

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
`+` is the residual — the pencil. Whatever the conversation or the digest
produces gets *added onto* the card; the card itself is never replaced.

And one subtlety is worth slowing down for. Each sublayer *reads* a
leveled copy of the card — that is the `self.attn_norm.forward(x)?` tucked
inside the parentheses — but its result is added onto the *original*,
un-leveled `x`. The volume knob only ever changes what a sublayer *sees*,
never what the card *is*. That arrangement — level the copy you read, add
onto the original — is called **pre-norm**, and it is exactly what the
Llama/Qwen family runs.

Three tests pin the block — the residual guarantee, the shape guarantee,
and causality end to end. They live in the existing `tests` module, below
`silu_gate_matches_hand_math`, with two small helpers they share. One tiny
edit first: the module's `use` line gains `DType` — make it
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

`a_silent_conversation_changes_nobody` is the residual idea made testable:
zero out every sublayer and the block is a perfect no-op, because the
card's own path through `forward` is a bare `+`. Depth can only ever *add*.

One last stretch of **wiring** — copy freely. Your demo has outgrown its
name — it shows a full block now — so five small edits, in order:

1. In `src/main.rs`, rename `run_attention_demo` to `run_block_demo` —
   both its definition and the call in the `match` arm.

2. In the same `match`, change the arm's pattern `["attention"]` to
   `["block"]`.

3. In the usage message just below, change the line to
   `cargo run -- block`.

4. In `scripts/check.sh`, make the same swap in the smoke command:
   `cargo run --quiet -- block`.

5. Back in `run_block_demo`: replace the `mlp:` print line — keeping the
   `let mlp = …` construction above it — with the block assembly and the
   day's headline, shown next.

The `⋮`, the framing lines, and the `let mlp` construction are already in
your file — the markers bracket what's new:

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

Two small things worth noticing in that code. First, the block is fed the
raw `cards`, not `leveled`: turning the volume knobs is the block's own
job now, so the demo hands it untouched cards and lets it do its own
leveling inside. Second, an ownership detail: `Block::new(…)` *moves*
`attn` and `mlp` into the block — after that line they belong to the
block, and the old variables can't be used again. That is why the demo
builds the block *after* the `attn:` print: one last use, then hand them
over.

And since the headline is now user-facing behavior, pin it from
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
`v1-day-05` — the last section of a day *is* the day).

## 7. Run it

```bash
cargo run -- block
```

## 8. Expected result

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
q proj [8, 8]: 4 heads;  k,v proj [8, 4]: 2 SHARED heads (half the table space)
attn:   [3, 8]
block:  [3, 8]
swap guest 3 "c"->"z": row 0 moved: false, row 1 moved: false, row 2 moved: true
```

Everything from Day 4 still holds — scores, mask, shares, the GQA shapes.
Read the two new lines against what you just built:

- `block:  [3, 8]` — `Block::forward` ran the whole evening — knob,
  conversation, pencil; knob, digest, pencil — and handed back cards the
  same size they arrived: the stackability guarantee, live.

- `swap guest 3 "c"->"z": row 0 moved: false, row 1 moved: false, row 2
  moved: true` — the demo ran the block twice on prompts differing only in
  the last guest, then compared rows. Rows 0 and 1 are bit-identical
  across the two runs, because `mask_future` gives the changed guest
  exactly zero share in them; only the swapped guest's own row moved.
  Causality observed through the complete block, not just asserted.

## 9. Test it

Five new tests — one for the digest, three for the block, one CLI pin —
bring the suite to 36 (32 unit, 4 integration):

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

## 10. Break it deliberately

Remove one pencil. In `Block::forward`, drop the second residual — return
the digest's output *instead of* adding it on:

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
card hands back all zeros — the original cards (`right`) are gone
entirely.
That is precisely what "rewrite instead of annotate" does, and a 28-block
model built this way would overwrite its own input 28 times before
predicting anything. Notice what stayed green, too: the causality test
still passes, because erasing every guest uniformly is still per-guest —
one property can be perfectly healthy while another is catastrophically
broken, which is why the block carries three independent guarantees
instead of one. Restore the `&x +` and confirm `cargo test` is green.

## 11. Checkpoint

Every checkpoint — each section and the day — is done only when the full
green bar passes:

```bash
./scripts/check.sh
```

Then close out the day in whichever mode you are following:

**Built along on your own branch?** Confirm your code matches the finished
day — the final ✓ line means it does:

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

**Either way**, today's complete change set — docs included — reads as one
diff:

```bash
git diff v1-day-04 v1-day-05
```

## 12. Recap

**What changed today:** `src/block.rs` — the SwiGLU `Mlp`
(`down(silu(gate(x)) * up(x))`, the model's only nonlinearity) and the
pre-norm decoder `Block` with residual connections; the demo grew into
`cargo run -- block` and now demonstrates causality through the complete
block, pinned by a CLI test; five tests, including the residual no-op
guarantee (zero weights leave every card bit-identical).

**Deliberately naive:** one block stands alone (real models stack dozens —
Day 6); the weights are still ramp-filled fakes (Day 6); and Qwen's extra
per-head q/k normalization waits for the real checkpoint (Day 6).

**Tomorrow's limitation, resolved on Day 6:** the machine is complete but
the mind is fake — ramp weights make grammatical-looking shapes and
meaningless content. Day 6 hires the trained chef: stack the blocks,
download a real checkpoint, and load its actual weights into these exact
structs — a real mind in the machine you built by hand. (One piece stays
borrowed a day longer: real *text* needs the real tokenizer, and that is
Day 7's opening move.)
