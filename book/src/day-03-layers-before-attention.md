# Day 3: The layers before attention

> **Day 3 of 15** · builds on `v1-day-02` · ≈ 1.5–2 h

| Section | Builds | Checkpoint |
|---------|--------|------------|
| A | Token embeddings | `v1-day-03a` |
| B | Linear layers & RMSNorm | `v1-day-03b` |
| C | Rotary position embeddings | `v1-day-03c` = `v1-day-03` |

**Pick your mode before starting** (explained fully in
[Setup](setup.md#4-choose-how-you-follow-along)):

```bash
# Build mode — start (or restart) your branch from yesterday's checkpoint.
# Already on my-course from Day 2? Just keep going — no command needed.
git switch -c my-course v1-day-02

# Inspect mode — read along with the finished code instead:
git switch --detach v1-day-03a         # then 03b, 03c as you pass each section
```

<div class="callout">

**Theory-density warning — and a promise.** Today is the most
concept-heavy day so far: embeddings, normalization, rotary positions.
Remember what this course is about: *inference engineering*, not deep
learning math — today's formulas are cargo the engine needs, not material
you will be tested on. You will always be told *what* each block is and
*why* the engine needs it — what's optional is the math of *how* it works
inside. Type the blocks out anyway (fingers build a familiarity that
reading skips) — or copy them straight from this page; both are legitimate,
the code just has to compile. Every concept comes with an everyday
analogy in the mental model below, and the teal **LLM concepts used
here** chips open a short video (▶) or the formal write-up in a side
panel whenever you want more. Understanding compounds: Day 4 is where
today's pieces click together.

</div>

<div class="callout">

**Today's typing budget** — so nothing sneaks up on you:

| Section | The layer (the learning) | Wiring & tests (copy-friendly) |
|---------|--------------------------|--------------------------------|
| A | `Embedding` · ~15 lines | demo subcommand, 2 tests |
| B | `Linear` + `RmsNorm` · ~30 lines | 3 demo edits, 3 tests |
| C | `rope` · ~20 lines | demo finish, 4 tests + helpers, 1 CLI test |

Three sections are three natural sittings, and every checkpoint is a save
point — nothing says a "day" must fit in one.

</div>

## 1. What you will have running

A new `cargo run -- layers` subcommand that walks a tiny prompt through
every *layer* — every processing station the data flows through on its way
to a prediction — that a **transformer** applies *before* **attention**.
Two more words worth defining on first contact: a *transformer* is the neural-network
design behind every modern LLM — a stack of identical blocks in which
tokens first exchange information, then each token thinks over what it
collected. *Attention* is the exchanging step: every token looks at the
tokens before it and decides which ones matter to it right now. Attention
is Day 4's whole subject — today builds what it consumes.

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=wjZofJX0v4M">▶ Transformers, visually (3Blue1Brown)</a>
<a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">▶ Attention, step by step (3Blue1Brown)</a>
<a href="https://www.deeplearning.ai/short-courses/how-transformer-llms-work/">full course: How Transformer LLMs Work (DeepLearning.AI) ↗</a>
</div>

## 2. What you will be seeing in the output

A sneak peek before anything else: here is the exact output you will
produce today, with a plain-English note under each line. Don't worry if a
note doesn't fully click yet — the rest of the day builds each piece, and
when you run this yourself at the end, every line will read like an old
friend:

```text
text "aa" -> ids [97, 97]
embed:   [2, 8]
row 0 starts [7.76, 7.77, 7.78, ...]
rmsnorm: [2, 8]  rms of row 0 = 1.000
linear:  [2, 4]
rope:    [2, 4]
same token, same vector?  before rope: true, after rope: false
```

- `text "aa" -> ids [97, 97]` — the same letter twice, on purpose: two
  identical tokens make today's problem visible.

- `embed:   [2, 8]` — each token's single number is traded for a *list of
  8 numbers* — room to carry meaning instead of just identity. Read
  `[2, 8]` as 2 tokens × 8 numbers each.

- `row 0 starts [7.76, ...]` — a peek at the first token's numbers.
  (Made-up today; real learned numbers arrive on Day 6.)

- `rmsnorm: [2, 8]  rms of row 0 = 1.000` — a volume control: every
  token's numbers are rescaled to the same overall loudness, so stacking
  many steps can't blow the numbers up or fade them out.

- `linear:  [2, 4]` — a re-mixer: each token's 8 numbers are blended down
  into 4 new ones. The shape change is the visible part; the *why* comes
  in the chapter.

- `rope:    [2, 4]` — same shape, new information: each token's numbers
  are twisted by *where that token sits* in the sentence.

- `before rope: true, after rope: false` — the punch line: until that last
  step, the two identical letters carried *identical* numbers — the model
  literally could not tell first from second. Position fixed it.

## 3. Why we need it

Day 2's [bigram](story.md#bigram) sees exactly **one token of context**, and the root cause is
its representation. A one-hot row is an ID badge: it says *which* token this
is, and nothing else.

**The story so far** — remember [the restaurant](story.md): Day 1 built the kitchen,
Day 2 taught the cook to choose each course from the last. Today the
**dining room** opens. Your prompt is tonight's table of guests — **one
guest per token**, so a prompt of five tokens seats five guests. (With our
byte tokenizer, one token is one byte, which for plain English text is one
letter: the prompt `"hi"` seats two guests, `h` and `i`. On Day 7 the real
tokenizer arrives, and a guest becomes a word piece instead.) The engine
is the host seating them, and the conversation the guests will eventually
have at the table is *attention* — that's Day 4. Now picture the badge
problem in that dining room: a table where every guest wears only a number.
Nobody can hold a conversation with that. Today is everything the host must
do *before* anyone can talk, and each of today's layers is one of the
host's moves.

At tomorrow's conversation, every guest gets to
look around at the guests seated before them and borrow information from
the ones that matter; that is how a model connects an "it" back to "the
cat" from twenty words ago. But organized looking-around only helps if
each guest carries something worth looking *at*. Today builds that carrier,
one section per ingredient:

- **A. Embeddings** — swap the ID badge for a *profile card*: a short list
  of numbers with room for meaning, where similar tokens can carry similar
  numbers.

- **B. Linear + RMSNorm** — a learnable *translator* that re-mixes a card's
  numbers into more useful ones, plus a *volume knob* that resets every
  card to the same loudness — so that dozens of re-mixing steps stacked on
  top of each other can't blow the numbers up to infinity or shrink them
  to nothing.

- **C. RoPE** — stamp each guest's *position* onto their card, so "dog bites
  man" finally stops being the same input as "man bites dog".

## 4. Mental model

<div class="callout">

**Reading tip:** this walk and the "Inference concepts" section after it
are the day's densest reading — all ideas, no code. If they start to
blur, that is normal and nothing is lost: jump ahead to section 6 and
*build* the thing, then come back — the ideas are far easier to hold
once you have watched the demo print them.

</div>

Now walk the dining room in order — one host's move per layer:

- **Arrival — the badge problem.** Each guest walks in wearing only a
  numbered badge (`a` is badge 97). Two guests with the same badge are
  indistinguishable, and a number by itself tells you nothing about anyone.
  That badge is Day 2's one-hot representation, and it is why the bigram
  was so limited.

- **Embedding — swap the badge for a profile card.** Right now each guest
  wears only a numbered badge: the letter `a` is badge 97, and that number
  says *which* letter it is, nothing more. At the door, the host takes the
  badge and hands back a profile card from a filing cabinet — card 97 for
  badge 97. The card holds eight numbers that describe the guest, and the
  same badge always gets the same card. Today our cards hold made-up
  numbers. In a trained model (you will load one on Day 6), the numbers
  were learned, so that guests used in similar ways carry similar numbers.
  That is all an embedding is: a lookup table that turns a token's id into
  a short list of numbers describing it.

- **RMSNorm — set everyone to the same volume.** Think of a card's numbers
  as how loudly its guest speaks. Some cards arrive loud (big numbers) and
  some faint (tiny numbers). Before each activity, the host turns a volume
  knob on every card, so that every guest speaks at the same loudness, 1.0.
  The knob changes how loud a card is, never what it says: all its numbers
  shrink or grow together, keeping their proportions. Why bother? A real
  model repeats the evening dozens of times, and every round multiplies the
  numbers again. Without the knob, a card that is a little too loud gets
  louder round after round until its numbers explode toward infinity, and a
  card that is a little too faint fades until its numbers vanish to zero.
  Leveling everyone before every round keeps both from happening.

- **Linear — the translator.** Each activity later in the evening needs a
  card's information arranged differently — the way the same dish is
  described one way on the menu and another way in the recipe. So before
  each activity, a translator rewrites every card. Each number on the new
  card is a blend of all the numbers on the old one: a bit of the first, a
  lot of the third, none of the fifth, all added up. How much of each to
  take — the *blend weights* — is the recipe the translator follows. Today
  those weights are made up, so only the *shapes* of the cards are real; on
  Day 6, a trained chef brings the real weights.

- **RoPE — stamp the seat number, as clock dials.** Identical twins still
  carry identical cards, so the host records *where each guest sits*: every
  pair of numbers on the card is a little dial, and sitting in seat N winds
  each dial forward N clicks — the first dial spins fast, later dials
  slower, like second, minute, and hour hands. The clever part: comparing
  two guests' dials reveals only *how many seats apart* they are, never the
  absolute row number — and "how far apart" is precisely what a
  conversation cares about.

Tomorrow, **attention** is the conversation itself: every guest compares
cards with everyone seated before them and decides whom to listen to. Today
ends with the cards ready:

```text
ids [97, 97]     two guests — so far just badge numbers
    │ embed
    ▼
[2, 8]           2 guests × 8 numbers on each profile card
    │ rmsnorm
    ▼
[2, 8]           same shape — only the loudness changed (every card at 1.0)
    │ linear
    ▼
[2, 4]           2 guests × 4 re-mixed numbers per card
    │ rope
    ▼
[2, 4]           same shape — the seat number is now wound into the dials
```

Read every shape the same way all course long: `[how many guests, how many
numbers each]` — or in engine terms, `[sequence length, features]`.

## 5. Inference concepts introduced

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=viZrOnJclY0">▶ word embeddings (StatQuest)</a>
<a href="https://nn.labml.ai/normalization/layer_norm/index.html">LayerNorm (labml)</a>
<a href="https://www.youtube.com/watch?v=o29P0Kpobz0">▶ RoPE (Efficient NLP)</a>
</div>

- **Dense representations.** A one-hot badge needs 256 slots just to say
  "I am token 97", and says nothing else. A profile card says more with
  less: eight numbers here, thousands in real models. Those numbers are
  *learnable*, and that is the whole point — everything a model "knows"
  about a token is literally stored in that short list of numbers.

- **Normalization.** A deep model is the same mixing step repeated dozens
  of times, and repeated multiplication snowballs: numbers either grow
  without bound or fade toward zero. Re-leveling every token to volume 1.0
  between steps is what makes depth survivable. RMSNorm is the lighter
  cousin of LayerNorm, and it is what the Llama/Qwen family actually ships.

- **Positional information.** Attention on its own treats a sentence as a
  *bag* of words — shuffle them and nothing would change. RoPE adds order
  by winding each card's dials, engineered so that when attention compares
  two tokens, only the *distance between them* shows through — a property
  today's tests prove rather than assume.

## 6. Build it, section by section

Almost no new Rust today — Day 3's difficulty is math, not language (only
section C's strip has entries). All three layers accumulate in one new
file, `src/layers.rs`, built up section by section.

### Section A — token embeddings

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=viZrOnJclY0">▶ word embeddings, clearly explained (StatQuest)</a>
</div>

Section A is the dining room's front door — the filing cabinet where badge
numbers are swapped for profile cards. Start `src/layers.rs` with that
cabinet, the lookup table:

*(Typing it out builds finger-memory — but copying it straight from here is
just as legitimate.)*

```rust
# extern crate candle_core;
# fn main() {}
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
```

The file is brand new, so wire it into the crate —
in `src/main.rs`, add the declaration next to the existing modules; without
it, nothing in `layers.rs` even compiles:

```rust,ignore
mod layers;
```

You have *already used* an embedding twice without
the name: Day 1's `one_hot(id) × weights` and Day 2's bigram forward are row
lookups performed by multiplication. `index_select(ids, 0)` — "pick these
rows along dimension 0" — is the honest, fast version. That equivalence is
too good a claim to leave untested, so the section's tests pin it — add
this test module at the bottom of `src/layers.rs`:

```rust,ignore
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
```

And that's the heart of the section — everything below is **wiring**,
honest plumbing you're welcome to copy. Now let's actually *see* some
vectors: give `src/main.rs` a demo subcommand. Two helpers above `main` (weights are a deterministic ramp —
fake on purpose; real trained weights arrive on Day 6):

```rust,ignore
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
```

Wire it into the CLI `match` (and add the line to the usage message):

```rust,ignore
        ["layers"] => run_layers_demo(),
```

Add `cargo run --quiet -- layers` as a second smoke command in
`scripts/check.sh` — from today on, the green bar exercises both
subcommands. Then run it:

```bash
cargo run -- layers
```

```text
text "aa" -> ids [97, 97]
embed:  [2, 8]
row 0 starts [7.76, 7.77, 7.78, ...]
rows equal: true
```

Read it line by line, against the code you just typed:

- `text "aa" -> ids [97, 97]` — Day 1's `encode`: the same guest (`a`,
  badge 97) arrives twice, on purpose.

- `embed:  [2, 8]` — `Embedding::forward` swapped each badge for a profile
  card: 2 guests × 8 numbers each.

- `row 0 starts [7.76, 7.77, 7.78, ...]` — the ramp table is fake but
  checkable by hand: row 97 starts at `97 × 8 × 0.01 = 7.76`.

- `rows equal: true` — the section's real finding: same badge, same card.
  The embedding has **no idea where a token sits**. Hold that thought
  until section C.

**Section checkpoint:** `./scripts/compare.sh v1-day-03a`

### Section B — linear layers and RMSNorm

<div class="rust-concepts"><span>Rust used here:</span> nothing new — this
section is pure tensor math (and the course's first tolerance-based test).</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=aircAruvnKk">▶ neural nets &amp; weights, visually (3Blue1Brown)</a>
<a href="https://nn.labml.ai/normalization/layer_norm/index.html">LayerNorm — RMSNorm's parent (labml)</a>
</div>

Next you build the middle two stations of the day's pipeline — and both
are old friends from the dining room walk: `Linear`, the translator that
re-mixes a card's numbers, and `RmsNorm`, the volume knob that resets every
card to the same loudness. Each is only a few lines of Rust.

One new name appears in that code: `D`, candle's *dimension picker*.
`RmsNorm` must answer the question *"average over which direction?"* — a
`[2, 8]` tensor could be averaged down its 2 rows or across its 8
features — and it answers with `D::Minus1`, "the last dimension" (the
`-1` works like Python's negative indexing), which is where a token's
features live. Add `D` to the import at the top of `src/layers.rs` so the
code below compiles:

```rust,ignore
use candle_core::{D, Result, Tensor};
```

Then the two layers. They go in `src/layers.rs`, between the `Embedding`
impl's closing `}` and the `#[cfg(test)]` tests module — around line 23:

*(Typing them out builds finger-memory — but copying them straight from here
is just as legitimate.)*

```rust
# extern crate candle_core;
# fn main() {}
# use candle_core::{D, Result, Tensor};
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
```

`Linear` is literally Day 1's [matmul](story.md#matmul) wearing a struct. `RmsNorm` reads off
its own formula: square, mean along the last dimension (`D::Minus1`,
keeping the dimension so the shape is `[seq, 1]`), add `eps`, square root —
then divide each row by its own rms (`broadcast_div` stretches the `[seq,
1]` divisor across the row) and multiply by the learned per-feature `scale`.
`eps` is a guard rail: with it, a row of zeros divides by `sqrt(eps)`
instead of by zero.

The tests reuse Day 1's arithmetic for `Linear` — and introduce something
new for `RmsNorm`: a **tolerance**. The moment `sqrt` enters, exact float
equality is over; from here to the Day 7 oracle, math is verified as "close
within epsilon". Add all three inside the `tests` module at the bottom of
`src/layers.rs`, below section A's tests:

```rust,ignore
#[cfg(test)]
mod tests {
    ⋮
    ⋮
    // ──────── new code starts here (around line 57) ────────
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
    // ──────── new code ends here ────────
}
```

That's the learning done — the rest of the section is **wiring**, yours
to copy freely. The demo in `src/main.rs` grows by three small edits.

**First**, `Tensor::ones` (below) needs candle's `DType`, so add it to the
import at the top of the file:

```rust,ignore
use candle_core::{DType, Device, Result, Tensor};
```

**Second**, in `run_layers_demo`, directly under the `let embed = …` line,
build the two new stations:

```rust,ignore
    let norm = layers::RmsNorm::new(Tensor::ones(8, DType::F32, &device)?, 1e-5);
    let proj = layers::Linear::new(ramp(8, 4, 0.1, &device)?);
```

**Third**, lower in the same function — between the `row 0 starts …` print
and the final `rows equal` print — thread `x` through both stations:

```rust,ignore
    let x = norm.forward(&x)?; // [2, 8], every row now at unit magnitude
    let rows = x.to_vec2::<f32>()?;
    let rms: f32 = (rows[0].iter().map(|v| v * v).sum::<f32>() / 8.0).sqrt();
    println!("rmsnorm: {:?}  rms of row 0 = {rms:.3}", x.dims());

    let x = proj.forward(&x)?; // [2, 8] x [8, 4] -> [2, 4]
    println!("linear:  {:?}", x.dims());
```

(One cosmetic touch: give the `embed:` label one extra space —
`"embed:   {:?}"` — so its column lines up with `rmsnorm:`.) Note the
`let x =` shadowing — reusing the name for each stage is idiomatic Rust
for pipelines. Run it:

```text
text "aa" -> ids [97, 97]
embed:   [2, 8]
row 0 starts [7.76, 7.77, 7.78, ...]
rmsnorm: [2, 8]  rms of row 0 = 1.000
linear:  [2, 4]
rows equal: true
```

Read the two new lines against the two layers you just built:

- `rmsnorm: [2, 8]  rms of row 0 = 1.000` — `RmsNorm::forward` leveled
  every card to loudness 1.000: the volume knob doing its one job. Same
  shape as before — normalizing changes values, never sizes.

- `linear:  [2, 4]` — `Linear::forward` re-mixed each 8-number card into
  4 numbers: `[2, 8] × [8, 4] → [2, 4]`, Day 1's shape rule at work.

And `rows equal: true` still closes the transcript: leveled and translated,
the twins carry identical cards *yet again*. Two more layers, and position
is still nowhere in sight.

**Section checkpoint:** `./scripts/compare.sh v1-day-03b`

### Section C — rotary position embeddings

<div class="rust-concepts"><span>Rust used here:</span>
<a href="https://google.github.io/comprehensive-rust/references/exclusive.html">exclusive references (<code>&amp;mut</code>, <code>iter_mut</code>)</a>
<a href="https://google.github.io/comprehensive-rust/tuples-and-arrays/tuples.html">tuples</a>
</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=o29P0Kpobz0">▶ RoPE, explained (Efficient NLP)</a>
<a href="https://nn.labml.ai/transformers/rope/index.html">RoPE, formally (labml)</a>
</div>

This is the host's final move before the conversation: recording where
each guest sits. Without it, a sentence is just a *bag* of tokens — feed
"dog bites man" and "man bites dog" through everything built so far and
the model sees the same evening twice.

The obvious fix first. Write the seat number straight onto the card: keep
a second filing cabinet of *position* cards — one learned vector for seat
0, one for seat 1, … — and add the right one to each guest's card. Early
transformers (GPT-2 included) did exactly that, and it has two real
problems. The cabinet is printed in advance, so the restaurant can never
seat a guest past its last position card — a hard maximum prompt length.
And it stamps *absolute* seats, when a conversation mostly cares about
*distance*: "the cat … sat" should connect the same way whether that pair
sits at seats 2 and 5 or seats 42 and 45.

**RoPE** — rotary position embeddings, what the Llama/Qwen family actually
ships — fixes both by *winding the card's dials* instead of adding a seat
card: rotate each pair of features `(2i, 2i+1)` by an angle proportional
to the seat number, the multi-speed clock from the dining room. Rotation
has exactly the property the naive stamp lacks: compare two wound cards (a
dot product — tomorrow's attention score) and the result depends only on
how far apart the seats are, never on the raw seat numbers. No cabinet to
run out of, and relative distance falls out of the math for free — the
section's last test proves that claim. Append to `src/layers.rs`, between
the `RmsNorm` impl's closing `}` and the `#[cfg(test)]` tests module —
around line 65:

*(Today's densest block. Typing it out builds finger-memory — but copying it
straight from here is just as legitimate; the line comments walk you
through either way.)*

```rust
# extern crate candle_core;
# fn main() {}
# use candle_core::{Result, Tensor};
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
```

Read it line by line, against what each piece does:

- `dims2()` unpacks the shape into a `(seq, dim)` tuple.

- `iter_mut()` hands out *exclusive* (`&mut`) references, so each row can
  be edited in place.

- The two assignment lines are the 2×2 rotation matrix, written out.

- The math runs in `f64` so repeated trig doesn't erode `f32` precision.

It is a slow scalar loop **on purpose** — you can read every operation. It
gets vectorized only when a benchmark says it hurts.

The four tests are the section's real content — each pins one property you
would otherwise take on faith. They go inside the same `tests` module at
the bottom of `src/layers.rs`, below section B's tests, along with the two
small helpers after them:

```rust,ignore
#[cfg(test)]
mod tests {
    ⋮
    ⋮
    // ──────── new code starts here (around line 139) ────────
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
    // ──────── new code ends here ────────
}
```

The last one is *the* reason RoPE exists: shift both tokens three positions
to the right, and the score attention would compute between them stays the
same. Position is encoded, but only *relative* position leaks into dot
products.

One last stretch of **wiring** — copy freely. Time to finish the demo:
in `src/main.rs`, at the end of `run_layers_demo`, replace
the final `rows equal` print — its last two lines before `Ok(())` — with a
remember-then-compare ending: capture whether the rows match *before* the
rotation, then print one verdict line after it:

```rust,ignore
    let rows = x.to_vec2::<f32>()?;
    let equal_before = rows[0] == rows[1];

    let x = layers::rope(&x)?; // [2, 4], each row rotated by its position
    println!("rope:    {:?}", x.dims());
    let rows = x.to_vec2::<f32>()?;
    println!(
        "same token, same vector?  before rope: {equal_before}, after rope: {}",
        rows[0] == rows[1]
    );
```

And since the demo is now a user-facing behavior, pin it from the outside, at the
bottom of `tests/cli.rs`:

```rust,ignore
#[test]
fn layers_demo_is_deterministic() {
    let first = run_cli(&["layers"]);
    let second = run_cli(&["layers"]);
    assert!(first.status.success());
    assert_eq!(first.stdout, second.stdout);
    let text = String::from_utf8(first.stdout).expect("stdout is text");
    assert!(text.contains("before rope: true"));
    assert!(text.contains("after rope: false"));
}
```

**Section checkpoint:** `./scripts/compare.sh v1-day-03c` (identical to
`v1-day-03` — the last section of a day *is* the day).

## 7. Run it

```bash
cargo run -- layers
```

## 8. Expected result

```text
text "aa" -> ids [97, 97]
embed:   [2, 8]
row 0 starts [7.76, 7.77, 7.78, ...]
rmsnorm: [2, 8]  rms of row 0 = 1.000
linear:  [2, 4]
rope:    [2, 4]
same token, same vector?  before rope: true, after rope: false
```

Read it line by line — every number is checkable:

- `ids [97, 97]` — `a` is byte 97; the twin-guest prompt, on purpose.

- `embed:   [2, 8]` — badge → profile card (`Embedding`): 2 guests × 8
  numbers each.

- `row 0 starts [7.76, ...]` — the ramp table's row 97: `97 × 8 × 0.01`.

- `rmsnorm: [2, 8]  rms of row 0 = 1.000` — the volume knob's entire job,
  done.

- `linear:  [2, 4]` — the translator: `[2, 8] × [8, 4] → [2, 4]`.

- `rope:    [2, 4]` — same shape: `rope` only *rotates* the dials, never
  resizes the card.

- `before rope: true, after rope: false` — the day's headline: the seat
  stamp finally told the twins apart. Position has entered the
  representation.

## 9. Test it

Nine new tests — two per layer plus four for RoPE's properties — bring the
suite to 24 (21 unit, 3 integration):

```bash
cargo test layers
```

```text
running 9 tests
test layers::tests::rotation_preserves_vector_length ... ok
test layers::tests::rope_leaves_position_zero_untouched ... ok
test layers::tests::identical_tokens_differ_by_position ... ok
test layers::tests::dot_products_depend_only_on_position_gap ... ok
test layers::tests::lookup_returns_the_selected_rows ... ok
test layers::tests::rmsnorm_output_has_unit_rms ... ok
test layers::tests::rmsnorm_matches_hand_math ... ok
test layers::tests::linear_reuses_day_1_arithmetic ... ok
test layers::tests::lookup_equals_one_hot_matmul ... ok

test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out; finished in 0.00s
```

## 10. Break it deliberately

Flip one sign in `rope` — the classic transcription slip when copying a
rotation matrix from paper:

```diff
-            row[2 * i + 1] = (a * sin + b * cos) as f32;
+            row[2 * i + 1] = (a * sin - b * cos) as f32;
```

`cargo test rope`:

```text
test layers::tests::rope_leaves_position_zero_untouched ... FAILED

assertion `left == right` failed
  left: [[1.0, -2.0, 3.0, -4.0]]
 right: [[1.0, 2.0, 3.0, 4.0]]
```

Here is what makes math bugs dangerous: the broken version still *runs*,
still produces plausible-looking numbers, and even still preserves vector
lengths (a sign flip turns the rotation into a reflection, which is also
length-preserving — so `rotation_preserves_vector_length` alone would have
stayed green!). Only the *identity-at-position-zero* property catches it
instantly. That is why section C pinned four independent properties instead
of one: math code does not crash when it is wrong, it just quietly computes
the wrong thing, and a set of property tests is the only tripwire. Flip the
sign back by hand and confirm `cargo test` is green again.

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
./scripts/compare.sh v1-day-03
```

**Inspecting instead of typing?** Jump to today's exact finished state,
poke at it, and come back:

```bash
git switch --detach v1-day-03
cargo test
git switch -          # back to where you were
```

**Either way**, today's complete change set — docs included — reads as one
diff:

```bash
git diff v1-day-02 v1-day-03
```

## 12. Recap

**What changed today:** a new `src/layers.rs` with the four pre-attention
building blocks — `Embedding` (provably the one-hot trick, done honestly),
`Linear` (Day 1's matmul in a struct), `RmsNorm` (unit-magnitude rows, with
the course's first tolerance-based tests), and `rope` (position as
multi-speed rotation, with four property tests) — plus a `layers` CLI demo,
a third integration test, and a second smoke command in the green bar.

**Deliberately naive:** the weights are ramp-filled fakes (real checkpoints
load on Day 6); `rope` is a scalar loop that revisits every element (it
gets vectorized when a benchmark demands it); the demo runs layers one at a
time on a two-token prompt; and nothing yet *looks at* any other token.

**Tomorrow's limitation, resolved on Day 4:** the cards are written,
leveled, re-mixed, and seat-stamped — but each guest still stands alone;
nobody has spoken. Day 4 is the conversation itself — causal self-attention
with grouped KV heads (GQA) — and Day 5 packages it, with the SwiGLU
feed-forward, into the complete decoder block: the machine that finally
lets "man bites dog" mean something different from "dog bites man".
