# Day 4: Causal attention

> **Day 4 of 15** · builds on `v1-day-03` · ≈ 1.5 h

| Section | Builds | Checkpoint |
|---------|--------|------------|
| A | Attention scores, causal mask, softmax | `v1-day-04a` |
| B | Grouped-query attention | `v1-day-04b` = `v1-day-04` |

**Pick your mode before starting** (explained fully in
[Setup](setup.md#4-choose-how-you-follow-along)):

```bash
# Build mode — start (or restart) your branch from yesterday's checkpoint.
# Already on my-course from Day 3? Just keep going — no command needed.
git switch -c my-course v1-day-03

# Inspect mode — read along with the finished code instead:
git switch --detach v1-day-04a         # then 04b as you pass the section
```

<div class="callout">

**Theory-density warning — and a promise.** Today is the summit:
attention itself, the one mechanism everything else in this course
serves. It is also the steepest math of the whole course — and the climb
ends today: Day 5 assembles the finished transformer block with barely
any new math, and from Day 6 on the work shifts back to this course's
real subject, engineering the machine you now hold. The same promise as
yesterday applies: every block tells you *what* it is and *why* the
engine needs it; the math of *how* it works inside is optional, typing
can always become copying, and the teal chips open a video or the formal
write-up whenever you want more. One extra comfort: today's ideas are
*checked, not believed* — each property you might take on faith gets its
own test, and you will watch the machine prove itself.

</div>

<div class="callout">

**Today's typing budget** — so nothing sneaks up on you:

| Section | The layer (the learning) | Wiring & tests (copy-friendly) |
|---------|--------------------------|--------------------------------|
| A | 4 small functions · ~45 lines | mod line, demo subcommand, 4 tests |
| B | `Attention` · ~55 lines | 1 import edit, 2 demo additions, 3 tests |

Two sections are two natural sittings, and every checkpoint is a save
point — nothing says a "day" must fit in one.

</div>

## 1. What you will have running

A new `cargo run -- attention` subcommand: a three-token prompt goes
through **causal self-attention** — the step where tokens finally
interact. Yesterday defined it in one line: every token looks at the
tokens before it and decides which ones matter to it right now. Today that
line becomes running code, in two moves:

- **The rules** — how a pair of tokens is scored for relevance, why a
  token may never use tokens *after* its own position (the **causal**
  rule), and how raw scores become shares of attention that sum to 1
  (softmax).

- **The full layer** — the same rules, run several times side by side.
  One pass of the rules can only look for one kind of connection, so the
  real layer runs several small passes at once — the **heads** — each with
  its own learned taste: one might track grammar, another might track
  which earlier word an "it" points back to. And because part of what each
  head computes must be kept around for every later token, heads are
  grouped to *share* that kept part — a memory saver called
  **grouped-query attention (GQA)**, the exact layout Day 8's cache will
  be built on.

The demo prints the machinery mid-flight, one table per step: the raw
match scores, the same table with the future silenced, and the final
who-listens-to-whom shares.

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">▶ Attention, step by step (3Blue1Brown)</a>
<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">▶ Let's build GPT — attention from scratch (Karpathy)</a>
<a href="https://www.deeplearning.ai/short-courses/how-transformer-llms-work/">full course: How Transformer LLMs Work (DeepLearning.AI) ↗</a>
</div>

## 2. What you will be seeing in the output

A sneak peek before anything else: here is the output you will produce
today, with a plain-English note under each line. Don't worry if a note
doesn't fully click yet — the whole chapter exists to build each piece,
and when you run this yourself, every line will read like an old friend.
(Section B appends two more lines; the full transcript is in section 8.)

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
```

- `ids [97, 98, 99]` — three characters become three numbered tokens:
  `a` is 97, `b` is 98, `c` is 99. A "seat" in the tables below is a
  token's position: seat 0 is the first token.

- `cards:  [3, 8]` — Day 3's work, reused: each token now carries 8
  numbers instead of one. Read it as 3 tokens × 8 numbers each.

- `scores` — the new idea, step one: every token is compared with every
  token, one number per pair — *how relevant are you to me right now?*
  Bigger means stronger. Each row is one token doing the asking; each
  column, a token being considered.

- `masked` — step two, a hard rule: text is generated left to right, so a
  token may not use tokens that come *after* it. Every score that peeks
  into the future is replaced by `-inf` — the upper-right triangle goes
  silent.

- `who listens to whom` — step three: each row of surviving scores becomes
  *shares of attention* that add up to exactly 1, like splitting 100% of
  your focus. `-inf` becomes exactly `0.000`: the future gets no share at
  all.

- `heard:  [3, 8]` — the payoff: every token walks away with a fresh set
  of 8 numbers, blended from the tokens it paid attention to. For the
  first time in this course, tokens *inform each other*.

The three tables are the *same table transformed twice* — that is the
entire mechanism, and today you build each transformation as a small
function.

## 3. Why we need it

**The story so far** — remember [the restaurant](story.md): Day 1 built the kitchen,
Day 2 taught the cook to serve each course from the last one, and yesterday
the dining room opened — every [guest](story.md#guest) (one per token of
your prompt) traded their numbered badge for a
[profile card](story.md#embedding), the host leveled the
[volumes](story.md#rmsnorm), [re-mixed the cards](story.md#linear), and
stamped on [seat numbers](story.md#rope). Then the chapter ended, and the room went quiet. Tonight
comes the scene the whole restaurant has been building toward: **the table
conversation** — the guests finally talk, each one reading the cards of
the guests seated before them and borrowing what they need. That
conversation is *attention*, and building it is today's whole job.

Why does the restaurant need a conversation at all? Look at what the
engine can actually do so far. The `generate` loop still chooses every
course from the *previous byte alone* — ask it to continue
`"the cat sat on the"` and everything it knows is the final letter, that
`"e"`; the cat, the sitting, the whole sentence of setup are invisible to
it. And Day 3, for all its work, did not change that: every profile card
was written, leveled, and stamped *in isolation* — a guest's card still
says nothing about any other guest at the table.

The conversation is the missing move: the moment one guest's card finally
reaches another's — how the model will connect an `"it"` back to
`"the cat"` from twenty words earlier. Today builds the conversation in
two sections:

- **A. The rules of the conversation** — three small functions, each one
  rule. Scoring: how relevant is one guest to another right now — one
  number per pair of guests. Etiquette: nobody may listen to a guest who
  hasn't spoken yet, so every score pointing at the future is silenced
  (the **causal mask**). Sharing out: each guest has exactly 100% of
  attention to spend, and **softmax** turns the surviving scores into
  that split.

- **B. The conversation at full scale** — the same rules, upgraded twice.
  Every guest listens through several narrow ears at once, each tuned to
  its own kind of connection (**heads**). And because the table is small,
  groups of ears share their reading material (**grouped-query
  attention**, GQA).

## 4. Mental model

<div class="callout">

**Reading tip:** this walk and the "Inference concepts" section after it
are the day's densest reading — all ideas, no code. If they start to
blur, that is normal and nothing is lost: jump ahead to section 6 and
*build* the thing, then come back — the ideas are far easier to hold
once you have watched the demo print them.

</div>

Walk one round of the conversation, move by move:

- **Three things from one card.** Before talking, each guest prepares
  three slips from their profile card: a **question** — what am I looking
  for right now? (`q`); a **topic card** — what can I be asked about?
  (`k`); and a **statement** — what will I actually contribute if someone
  listens to me? (`v`). All three are made the Day 3 way: a translator
  (`Linear`) re-mixes the card's numbers. Why three separate slips? What
  you *seek*, what you can be *found by*, and what you *say* are different
  roles — one card could never play all three at once.

- **Match questions to topic cards.** Guest *i* holds their question up
  against guest *j*'s topic card; the better the match (a dot product),
  the higher the score. Long cards inflate scores by sheer arithmetic —
  more numbers in the sum — so every score is divided by the square root
  of the card length to keep things fair.

- **Etiquette: only listen to guests who have spoken.** During generation
  the engine produces one token at a time, so at the moment guest *i*
  speaks, guests *i+1, i+2, …* haven't arrived yet. The mask enforces at
  training-shape what is physically true at generation-time: every score
  pointing at a later seat is set to negative infinity.

- **Split your attention into shares.** Every guest has exactly 100% of
  attention to distribute, so raw scores must become shares that sum to 1.
  The obvious way — divide each score by the row's total — breaks
  immediately: scores can be negative (a negative share of attention means
  nothing), and the masked `-inf` must come out as *exactly* no attention.
  So **softmax** exponentiates first: `exp` makes every score positive,
  sends `-inf` to exactly 0, and stretches the gaps so a clearly better
  match earns a clearly bigger share — *then* divide by the row's total. A
  huge score takes most of the attention, and the guest walks away with
  the share-weighted blend of the statements they listened to.

- **Several conversations at once.** One blend per guest is crude — "the
  cat sat on *it*" needs one ear on grammar and another on which thing
  "it" was. So the card is split among several **heads**, each a narrow
  ear running the identical routine on its own slice, listening for its
  own kind of connection.

- **Share the topic cards — the table is small.** Questions are cheap to
  hold (asked once, discarded), but topic cards and statements must stay
  on the table for every *future* guest to consult. So groups of ears
  share one topic-card/statement pair: 4 askers, 2 shared pairs here.
  That is **grouped-query attention (GQA)** — and the shared pairs are
  exactly what Day 8 will cache, so halving them now halves that shelf
  forever. A rare treat: an architecture choice made *for the inference
  engineer*.

- **The seat stamp moves inside.** Yesterday's RoPE stamps the *question*
  and the *topic card* — not the statement, and never the master card.
  Where you sit should shape *whom you listen to* (the dials make nearby
  guests match louder), not *what you say* once heard.

One round, end to end:

```text
x [3, 8]           3 guests × 8 numbers on each prepped card
    │ three translators: q (questions), k (topic cards), v (statements)
    ▼
q [3, 8]           4 ears × 2 numbers of question per guest
k, v [3, 4]        only 2 SHARED topic/statement pairs — half the table space
    │ per ear: stamp seats onto q and k, match, mask the future, split shares
    ▼
shares [3, 3]      who listens to whom — every row sums to 1, zeros above the diagonal
    │ blend each ear's statements by its shares, stitch the ears, re-mix (o)
    ▼
[3, 8]             what each guest carries away from the conversation
```

Same reading rule as yesterday: every shape is `[how many guests, how many
numbers each]`. What each guest *does* with what they carried away — and
how the whole evening gets packaged into one repeatable unit — is
tomorrow: Day 5 builds the digestion step and the decoder block.

## 5. Inference concepts introduced

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">▶ attention, visually (3Blue1Brown)</a>
<a href="https://nn.labml.ai/transformers/mha.html">multi-head attention, formally (labml)</a>
<a href="https://arxiv.org/abs/2305.13245">the GQA paper ↗</a>
</div>

- **Causality is an inference contract, not a math nicety.** The engine
  generates left to right, appending one token at a time. Because the
  future provably cannot reach back, everything computed for old tokens
  stays valid as new ones arrive — which is why it can be computed *once*
  and cached. Break the mask and no cache is ever safe; Day 8 stands
  entirely on today's `the_future_cannot_reach_back` test.

- **GQA is why real models are servable.** At serving time the k and v
  slips of every past token must be kept around (the KV cache — Day 8).
  Cutting kv heads from 16 to 8 halves that memory *for the life of every
  request*. Qwen, Llama, and friends all ship GQA for exactly this
  reason.

- **Softmax turns scores into a budget.** Shares that always sum to 1 make
  attention a weighted *average* — outputs stay in a sane range no matter
  how wild the scores get, and `-inf` from the mask becomes exactly 0
  share, not merely a small one.

## 6. Build it, section by section

One new file today — `src/attention.rs`, the conversation — leaning on
yesterday's `layers.rs` for `Linear` and `rope`.

### Section A — attention scores, the causal mask, softmax

<div class="rust-concepts"><span>Rust used here:</span>
<a href="https://google.github.io/comprehensive-rust/closures.html">closures (and <code>move</code>)</a>
<a href="https://google.github.io/comprehensive-rust/iterators/iterator.html">iterator chains (<code>flat_map</code>)</a>
</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=eMlx5fFNoYc">▶ attention, step by step (3Blue1Brown)</a>
<a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">▶ the same code, in Python (Karpathy)</a>
</div>

Section A writes the rules of the conversation as four small functions —
match questions to topic cards, silence the future, split attention into
shares, and one function composing them into a full listening round. Start
a new file, `src/attention.rs`, with all four:

*(Today's core ideas, ~45 lines. Typing them out builds finger-memory — but
copying them straight from here is just as legitimate; every function is
four lines of body or less.)*

```rust
# extern crate candle_core;
# fn main() {}
use candle_core::{D, Result, Tensor};

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
```

One name in that code arrives a little early: the card width is called
`head_dim`, and `attend`'s doc says "for a single head", because section B
will run this exact routine several times in parallel, each run on its own
narrow slice of the cards — one such run is called a **head**. In the
story: a guest doesn't listen with one giant ear; they have several small
ones, each tuned to its own kind of connection — and each ear is one head.
For now the guest listens with a single ear the size of the whole card.

Two Rust newcomers in `mask_future`: `flat_map` runs the inner closure for
every `i` and flattens the resulting rows into one flat `Vec` (a tensor is
built from flat data plus a shape, as `from_vec` shows); and that inner
closure needs `move` because it captures `i` — without it, the closure
would try to *borrow* `i` and outlive it. Everything else is Day 3
machinery: `matmul`, `broadcast_*`, `keepdim`, and `D::Minus1` picking the
last dimension.

The file is brand new, so wire it into the crate — in `src/main.rs`, add
the declaration above the existing `mod layers;`; without it, nothing in
`attention.rs` even compiles:

```rust,ignore
mod attention;
```

Now pin the rules with four tests — every one checkable by hand — at the
bottom of `src/attention.rs`:

```rust,ignore
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
}
```

And that's the heart of the section — the rest is **wiring**, yours to
copy freely. Now comes the fun part: *watching* your rules at work. Give
`src/main.rs` a third demo subcommand. It preps cards exactly as Day 3 did, then holds one listening
round *by hand* — using the whole card as question, topic card, and
statement alike (the per-role slips arrive in section B) — and prints the
table **after every step**, so you can watch each function you just wrote
change it. Add both the small table printer and the demo above
`run_generate`:

```rust,ignore
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
    Ok(())
}
```

Wire it into the CLI `match` (and add
`eprintln!("       cargo run -- attention");` to the usage message below
it):

```rust,ignore
        ["attention"] => run_attention_demo(),
```

Add `cargo run --quiet -- attention` as a third smoke command in
`scripts/check.sh`, then run it:

```bash
cargo run -- attention
```

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
```

The same table, three times — once after each function you just wrote.
Read it table by table:

- `text "abc" -> ids [97, 98, 99]` — three *different* guests this time:
  `a` (badge 97), `b` (98), and `c` (99). A conversation needs more than
  twins.

- `cards:  [3, 8]` — Day 3's front door, unchanged: three badges become
  three profile cards.

- the table layout, same for all three tables — each printed row is one
  guest *listening*; each column (named in the header) is a guest being
  *heard*. The cell at row `seat 1`, column `seat 0` reads "how strongly
  guest `b` attends to guest `a`."

- **`scores`** — `scaled_scores`' raw matches, before any rule applies.
  Two things are visible in the numbers. The table is *symmetric* (the
  same `2.502` above and below the diagonal) because the demo used the
  same cards for questions and topic cards. And every guest matches
  *itself* loudest (`2.828` down the diagonal) while the match *fades with
  distance* (`2.502` one seat apart, `1.820` two apart) — that fade is the
  seat stamps at work: `rope` on q and k makes nearby guests match louder.

- **`masked`** — the same table after `mask_future`: every score above the
  diagonal — every look into the future — replaced by `-inf`. Nothing
  below the diagonal changed.

- **`who listens to whom`** — the same table after `softmax_rows`: each
  row of scores became shares that sum to 1.000, and `-inf` became
  *exactly* `0.000`, not merely small. Seat 0, with nobody before it,
  keeps all 100%; seat 2, the only guest allowed to hear everyone, splits
  its attention three ways.

- `heard:  [3, 8]` — `attend` ran those same three steps in one call and
  blended each guest's statements by the shares: everyone walks away with
  a full card of what they gathered.

**Section checkpoint:** `./scripts/compare.sh v1-day-04a`

### Section B — grouped-query attention

<div class="rust-concepts"><span>Rust used here:</span> nothing new — the
grouping is one integer division, and the head loop is a plain `for`.</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://nn.labml.ai/transformers/mha.html">multi-head attention, formally (labml)</a>
<a href="https://arxiv.org/abs/2305.13245">GQA: the paper ↗</a>
</div>

Your section A conversation already works — one big ear, whole cards for
every role. So why touch it? Put yourself in a card's position for a
moment: you are expected to say what your guest is *looking for*, to
advertise what they can be *asked about*, and to carry what they would
actually *say* — all with the same eight numbers. Three jobs, one card —
it can't be great at any of them.

The fix is what you'd do in real life: stop making one card work three
jobs and write three slips instead, each translated off the card — a
question slip (the `q` projection), a topic card (`k`), and a statement
(`v`). Add the two upgrades you met in the mental model — listening
through four narrow ears, with pairs of ears sharing their topic cards to
save table space — and you have grouped-query attention, exactly as
promised.

Everything here is built from pieces you already own: Day 3's `Linear`
does the translating, and `rope` stamps the seats. Invite them in first —
in `src/attention.rs`, directly under the `use candle_core…` line, add the
crate import so the code below can name them:

```rust,ignore
use crate::layers::{Linear, rope};
```

Then the layer itself. It goes in `src/attention.rs`, between `attend`
and the tests module — the frame shows both neighbors, and the markers
bracket what you type:

*(The day's biggest single block, ~55 lines — half of it is the
constructor's plumbing. Typing it out builds finger-memory — but copying it
straight from here is just as legitimate; the comments narrate every move.)*

```rust,ignore
⋮
    softmax_rows(&mask_future(&scaled_scores(q, k)?)?)?.matmul(v)
}

// ──────── new code starts here (around line 50) ────────
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
// ──────── new code ends here ────────

#[cfg(test)]
mod tests {
    ⋮
```

Read `forward` as the evening's choreography. Three translators make the
three slips — note `k` and `v` come out *narrower* than `q` when heads are
shared. `narrow(1, start, len)` slices columns `start..start+len`: each
ear `h` takes its own slice of the questions, but `h / group` sends whole
*groups* of ears to the same shared kv slice — with 4 ears and 2 shared
cards, ears 0–1 get card 0, ears 2–3 get card 1. Sharing is nothing
fancier than *reading the same columns*. Each ear runs section A's `attend` unchanged;
`Tensor::cat` stitches the narrow answers back into full cards; one last
translator (`o_proj`) re-mixes them.

Three tests — the second is the day's most important, and the third
proves GQA does exactly what "sharing" claims. They go inside the existing
`tests` module, below `seat_zero_hears_only_its_own_statement`, along with
the two helpers they build on:

```rust,ignore
#[cfg(test)]
mod tests {
    ⋮
    ⋮
    // ──────── new code starts here (around line 108) ────────
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
    // ──────── new code ends here ────────
}
```

`the_future_cannot_reach_back` asserts **bit-identical** equality, not
"close within tolerance" — masked-out shares are exactly zero, so a
changed future token contributes exactly nothing. Hold on to that: it is
the strongest guarantee in the whole engine, and Day 8 spends it.

Back to **wiring** — copy freely. Let the demo show off the packaged
layer: in `src/main.rs`, at the end of `run_attention_demo` — after the
`heard:` print, before `Ok(())` — build one from ramp weights and run the
same cards through it. The `⋮` and the framing lines are already in your
file — type only what sits between the two markers (and if you type the
marker comments too, no harm done):

```rust,ignore
fn run_attention_demo() -> Result<()> {
    ⋮
    ⋮
    let heard = attention::attend(&q, &k, &leveled)?;
    println!("heard:  {:?}", heard.dims());

    // ──────── new code starts here (around line 132) ────────
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
    // ─────────────────── new code ends here ─────────────────────
    Ok(())
}
```

Run `cargo run -- attention` again — two new lines at the bottom:

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
──────────── everything above: section A's output, unchanged ────────────
q proj [8, 8]: 4 heads;  k,v proj [8, 4]: 2 SHARED heads (half the table space)
attn:   [3, 8]
```

- The `q proj` line is GQA visible in the *weights*, before a single number
  is computed: questions get a full-width `[8, 8]` translator, while topic
  cards and statements share a half-width `[8, 4]` one — four ears, two
  shared card sets.

- `attn:   [3, 8]` — `Attention::forward` ran section A's `attend` once
  per ear on its own slice, stitched the four answers with `Tensor::cat`,
  and re-mixed them through `o_proj`: every guest leaves the packaged
  layer with a full card, exactly like the hand-held round above.

**Section checkpoint:** `./scripts/compare.sh v1-day-04b` (identical to
`v1-day-04` — the last section of a day *is* the day).

## 7. Run it

```bash
cargo run -- attention
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
```

Read it line by line — every number is checkable:

- `ids [97, 98, 99]` — `a`, `b`, `c`: three different guests.

- `cards:  [3, 8]` — Day 3's front door, unchanged.

- `scores` — raw matches (`scaled_scores`): loudest on the diagonal,
  fading with seat distance — the seat stamps at work.

- `masked` — the future replaced by `-inf` (`mask_future`); the past
  untouched.

- `who listens to whom` — every row now sums to 1.000 (`softmax_rows`),
  every `-inf` became exactly zero, and seat 0 keeps everything: nobody
  spoke before them.

- `q proj [8, 8] … k,v proj [8, 4]` — GQA in the weights: four ears'
  questions, two shared topic/statement sets.

- `heard:` and `attn:` — both `[3, 8]`: the hand-held round and the
  packaged `Attention` layer each hand every guest back a full card.

## 9. Test it

Seven new tests bring the suite to 31 (28 unit, 3 integration):

```bash
cargo test attention
```

```text
running 7 tests
test attention::tests::scores_match_hand_arithmetic ... ok
test attention::tests::future_seats_get_a_zero_share ... ok
test attention::tests::seat_zero_hears_only_its_own_statement ... ok
test attention::tests::softmax_rows_are_shares_that_sum_to_one ... ok
test attention::tests::every_guest_leaves_with_a_full_card ... ok
test attention::tests::a_shared_topic_card_equals_its_own_copies ... ok
test attention::tests::the_future_cannot_reach_back ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 21 filtered out; finished in 0.00s
```

## 10. Break it deliberately

Delete the etiquette. In `attend`, drop the mask:

```diff
-    softmax_rows(&mask_future(&scaled_scores(q, k)?)?)?.matmul(v)
+    softmax_rows(&scaled_scores(q, k)?)?.matmul(v)
```

`cargo test`:

```text
test attention::tests::seat_zero_hears_only_its_own_statement ... FAILED
test attention::tests::the_future_cannot_reach_back ... FAILED

  left: [5.660477, 6.6604767]
 right: [5.0, 6.0]
```

Two tripwires fire. The first guest's answer is now `[5.66, 6.66]` instead
of their own `[5.0, 6.0]` statement — a slice of the *second* guest's
words leaked backward in time. Note what did **not** happen: no crash, no
error, and `future_seats_get_a_zero_share` stays green because it tests
the mask directly — only the properties that watch `attend` end to end
caught the leak. An unmasked model still runs and still prints
sane-looking numbers while silently reading the future; when Day 8 starts
*reusing* old computations, that leak would poison every cached token.
That is why causality gets independent property tests instead of a comment
saying "trust me". Restore the mask and confirm `cargo test` is green.

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
./scripts/compare.sh v1-day-04
```

**Inspecting instead of typing?** Jump to today's exact finished state,
poke at it, and come back:

```bash
git switch --detach v1-day-04
cargo test
git switch -          # back to where you were
```

**Either way**, today's complete change set — docs included — reads as one
diff:

```bash
git diff v1-day-03 v1-day-04
```

## 12. Recap

**What changed today:** `src/attention.rs` — the conversation rules
(`scaled_scores`, `mask_future`, a hand-rolled `softmax_rows`, `attend`)
and the packaged `Attention` layer with grouped kv heads; an `attention`
CLI demo that prints the who-listens-to-whom triangle live; seven tests,
including bit-exact proofs that the future cannot reach back and that
GQA's sharing is exact.

**Deliberately naive:** the head loop runs one narrow `attend` at a time
(real engines batch heads into one big matmul — Day 9's benchmark decides
when we care); every step recomputes attention over the *whole* prompt
from scratch (living with that pain until Day 8 makes it obvious); the
weights are still ramp-filled fakes (Day 6); and Qwen's extra per-head
q/k normalization waits for the real checkpoint (Day 6).

**Tomorrow's limitation, resolved on Day 5:** the guests can finally
listen — but nobody digests what they hear, nothing protects the cards
through dozens of rounds, and the evening is still loose parts. Day 5
adds the think-it-over step (the SwiGLU feed-forward) and packages the
whole evening into the transformer decoder block — the repeating unit
real models are a stack of.
