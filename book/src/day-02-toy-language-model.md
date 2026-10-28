# Day 2: A complete toy language model

> **Day 2 of 15** · builds on `v1-day-01` · ≈ 1–1.5 h

| Section | Builds | Checkpoint |
|---------|--------|------------|
| A | A bigram model learned by counting | `v1-day-02a` |
| B | Autoregressive generation & the `generate` CLI | `v1-day-02b` = `v1-day-02` |

**Pick your mode before starting** (explained fully in
[Setup](setup.md#4-choose-how-you-follow-along)):

```bash
# Build mode — start (or restart) your branch from yesterday's checkpoint.
# Already on my-course from Day 1? Just keep going — no command needed.
git switch -c my-course v1-day-01

# Inspect mode — read along with the finished code instead:
git switch --detach v1-day-02a         # then 02b as you pass each section
```

<div class="callout">

**Today's typing budget** — so nothing sneaks up on you:

| Section | The learning | Wiring & tests (copy-friendly) |
|---------|--------------|--------------------------------|
| A | `CORPUS`, `Bigram`, `from_text`, `forward` · ~28 lines | import edit, move `one_hot`, a new `main`, 3 tests |
| B | `generate` · ~12 lines | the CLI `main`, 2 tests, 2 CLI tests |

Two sections are two natural sittings, and every checkpoint is a save
point — nothing says a "day" must fit in one.

</div>

## 1. What you will have running

By the end of today, `cargo run -- generate "ru" 40` prints the engine's
first self-generated text. It is a complete language model, end to end. Its
weights are **learned from data** (by counting). An **autoregressive loop**
feeds each predicted token back in as input. And a **CLI** wraps it, so you
can hand it to someone else. Run it a hundred times, and you get the same
output a hundred times. It is also hilariously dumb. This chapter measures
those dumbnesses precisely, because they are exactly what the next thirteen
days repair.

## 2. What you will be seeing in the output

Here is a sneak peek: the exact output you will produce today, with a
plain-English note under each line. Don't worry if a note doesn't fully
click yet. The rest of the day builds each piece, and when you run this
yourself at the end, every line will read like an old friend:

```text
"rust. rust. rust. rust. rust. rust. rust. "
```

- It is one line, but the model built it one character at a time. It
  predicts a single next character and glues it onto the end of the text.
  Then it asks *what comes next?* again, 40 times here. That feedback loop
  is the day's real product.

- The looping is honest, not a bug. This model remembers only *one*
  character back. So after `.` it always picks space, after space it always
  picks `r`, and the same six-character groove repeats forever. Today you
  build exactly that loop, and you measure exactly that dumbness.

## 3. Why we need it

Your Day 1 pipeline works, but it predicts **one** token with **fake**
weights: a hard-coded "next byte in ASCII order" matrix. Two things
separate that from a language model, and today's two sections build one
each:

- **Section A** gives you weights that come from *data*. Over a training
  text, you count how often each byte follows each byte. The counts *are*
  the model.

- **Section B** handles the fact that language is many tokens. You loop the
  pipeline and feed each chosen token back in as the new input. That loop
  is autoregression, and every LLM you have ever used runs it.

**The story so far** — remember [the restaurant](story.md): Day 1 built the kitchen,
with its pantry of numbered jars, its recipe board, and its judge. Today the
cook stops obeying a hard-coded rulebook and *learns*. Section A builds the
cook's memory: a tally sheet of what followed what in the old order
tickets. Section B has the cook serve a whole tasting menu, choosing every
next course by looking at the last. So A builds the model, and B wraps the
loop around it.

## 4. Mental model

Walk the story in its two halves. First see how the cook learns, then what
happens when the cook starts serving:

**How the cook learns (section A).** The cook keeps a 256×256 tally sheet.
It has one row per "jar I just used" (the byte just seen) and one column
per "jar that came next". The cook reads the old order tickets, the
training text, once. For every adjacent pair, the cook puts a tally mark in
cell (current, next). To suggest the next jar, the cook looks up the
current row and takes the column with the most marks. Counting is
**training, in miniature**. Real training also adjusts numbers in matrices
based on data. Gradient descent is a smarter version of "+1 per sighting",
not a different kind of thing.

**Serving course after course (section B).** The cook looks at the last jar
used, consults the tally sheet, cooks the winner, and repeats. Every course
is chosen only from the one before. That loop is **autoregression**, and
you have already seen its famous failure. Keep tapping the middle
suggestion on your phone keyboard, and the sentences go in circles ("…I am
a beautiful day and I am a beautiful day and…"), because the keyboard only
remembers a step back. Our cook remembers exactly one *byte*. That is why
`"ru"` will become `"rust. rust. rust. "` forever:

```text
ids [.., last]      every course served so far — the cook reads only the last
    │ forward
    ▼
logits [1, 256]     1 row of 256 scores — one for every byte that could come next
    │ argmax
    ▼
next token          the judge's pick — appended to the end of `ids`
    │
    └──► back to the top; repeat `steps` times, then decode `ids` once
```

The model still answers only one question, "what comes after this byte?",
asked over and over.

## 5. Inference concepts introduced

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=PaCmpygFfXo">▶ build a bigram language model (Karpathy)</a>
<a href="https://nn.labml.ai/sampling/index.html">sampling &amp; decoding strategies (labml)</a>
</div>

- **Weights learned from data.** The bigram counts are the simplest possible
  learned parameters. Until real trained checkpoints arrive on Day 6, this is
  the only "training" in the course. It is enough to expose every part of
  the serving problem.

- **The autoregressive loop.** Generation is not one model call. It is a
  loop of calls, and each call depends on the previous choice. This loop
  shape is why serving LLMs is hard and interesting: Days 8–14 schedule,
  cache, and batch exactly this loop.

- **Context length.** A bigram conditions on exactly **one** token of
  history. That is why its output locks into a cycle. Everything in Days 3–5
  (embeddings, attention) exists to widen this to thousands of tokens.

## 6. Build it, section by section

### Section A — a bigram model learned by counting

<div class="rust-concepts"><span>Rust used here:</span>
<a href="https://google.github.io/comprehensive-rust/user-defined-types/named-structs.html">structs</a>
<a href="https://google.github.io/comprehensive-rust/methods-and-traits/methods.html">methods &amp; <code>impl</code></a>
<a href="https://google.github.io/comprehensive-rust/modules/paths.html"><code>use</code> paths</a>
<a href="https://google.github.io/comprehensive-rust/user-defined-types/const.html">constants</a>
</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://www.youtube.com/watch?v=PaCmpygFfXo">▶ build a bigram language model (Karpathy)</a>
</div>

Section A builds the cook's memory, the tally sheet. The model deserves its
own module, so create `src/model.rs`. Start with the training data and the
model type:

```rust
# extern crate candle_core;
# use candle_core::Tensor;
/// The tiny corpus the bigram model "trains" on. Counting its byte pairs is
/// the only training in this course until real checkpoints arrive on Day 6.
pub const CORPUS: &str = "rust. rust. rust. rush.";

/// Analogy: the cook's tally sheet — which byte followed which.
///
/// A bigram model: row b of `weights` counts how often each byte followed
/// byte b in the corpus. Higher count = higher logit.
pub struct Bigram {
    weights: Tensor, // [VOCAB_SIZE, VOCAB_SIZE] f32 counts
}
```

The corpus is engineered, and honestly so. `rust` appears three times and
`rush` once, so after `s` the tally reads **t: 3, h: 1**. That is a real
argmax over real counts, decided by data instead of by a hard-coded rule.
`weights` has no `pub`, so outside code can hold a `Bigram` but cannot
reach into it. The only ways in are the methods we choose to expose.

Now add the two methods, training and forward:

```rust
# extern crate candle_core;
# use candle_core::{Device, Result, Tensor};
# mod tokenizer {
#     pub const VOCAB_SIZE: usize = 256;
#     pub fn encode(text: &str) -> Vec<u32> {
#         text.bytes().map(u32::from).collect()
#     }
# }
# use tokenizer::{VOCAB_SIZE, encode};
# pub struct Bigram {
#     weights: Tensor,
# }
impl Bigram {
    /// Analogy: fill the tally sheet.
    ///
    /// "Training", in miniature: count the data.
    pub fn from_text(text: &str, device: &Device) -> Result<Self> {
        let ids = encode(text);
        let mut counts = vec![0_f32; VOCAB_SIZE * VOCAB_SIZE];
        for pair in ids.windows(2) {
            counts[pair[0] as usize * VOCAB_SIZE + pair[1] as usize] += 1.0;
        }
        let weights = Tensor::from_vec(counts, (VOCAB_SIZE, VOCAB_SIZE), device)?;
        Ok(Self { weights })
    }

    /// Analogy: the cook's suggestion.
    ///
    /// Logits for the token after `token`, shape [1, VOCAB_SIZE].
    pub fn forward(&self, token: u32) -> Result<Tensor> {
        one_hot(token, self.weights.device())?.matmul(&self.weights)
    }
}
# fn one_hot(id: u32, device: &Device) -> Result<Tensor> {
#     let mut row = vec![0_f32; VOCAB_SIZE];
#     row[id as usize] = 1.0;
#     Tensor::from_vec(row, (1, VOCAB_SIZE), device)
# }
```

- `ids.windows(2)` slides a length-2 window over the token ids. It visits
  every adjacent pair, with no index arithmetic to get wrong.

- `counts[pair[0] * VOCAB + pair[1]] += 1.0` is the tally sheet flattened
  into one `Vec`. It is row-major, the same layout `Tensor::from_vec`
  expects.

- `from_text` returns `Self`. It is an associated function used as a
  constructor, called as `Bigram::from_text(...)`.

- `forward` is your Day 1 pipeline fragment turned into a method:
  `[1, 256] × [256, 256] → [1, 256]`, one-hot row lookup and all.

`forward` leans on two things the new file still needs. First, add imports
at the top of `src/model.rs`:

```rust,ignore
use candle_core::{Device, Result, Tensor};

use crate::tokenizer::{VOCAB_SIZE, encode};
```

The second line is new Rust. `crate::` means "start from my own crate's
root". Here, `model.rs` reaches over to its sibling `tokenizer` module for
the vocabulary size and the encoder.

Second, move `one_hot` over, unchanged from Day 1. Cut it from
`src/main.rs` and paste it at the bottom of `src/model.rs`. It stays private
(no `pub`), because only `forward` calls it now.

```rust
# extern crate candle_core;
# use candle_core::{Device, Result, Tensor};
# const VOCAB_SIZE: usize = 256;
/// One-hot row for a token id: shape [1, VOCAB_SIZE], all zeros except id.
fn one_hot(id: u32, device: &Device) -> Result<Tensor> {
    let mut row = vec![0_f32; VOCAB_SIZE];
    row[id as usize] = 1.0;
    Tensor::from_vec(row, (1, VOCAB_SIZE), device)
}
```

That completes `src/model.rs`. Now make three edits in `src/main.rs`:

1. Register the new module: add `mod model;` next to `mod tokenizer;`.

2. Delete `next_byte_weights`. The fake model is gone.

3. Replace `main` so it asks the real model instead:

```rust
# extern crate candle_core;
# use candle_core::{Device, Result, Tensor};
# mod tokenizer {
#     pub const VOCAB_SIZE: usize = 256;
#     pub fn encode(text: &str) -> Vec<u32> {
#         text.bytes().map(u32::from).collect()
#     }
#     pub fn decode(ids: &[u32]) -> Option<String> {
#         let bytes = ids
#             .iter()
#             .map(|&id| u8::try_from(id).ok())
#             .collect::<Option<Vec<u8>>>()?;
#         String::from_utf8(bytes).ok()
#     }
# }
# mod model {
#     use super::tokenizer::{VOCAB_SIZE, encode};
#     use candle_core::{Device, Result, Tensor};
#     pub const CORPUS: &str = "rust. rust. rust. rush.";
#     pub struct Bigram {
#         weights: Tensor,
#     }
#     impl Bigram {
#         pub fn from_text(text: &str, device: &Device) -> Result<Self> {
#             let ids = encode(text);
#             let mut counts = vec![0_f32; VOCAB_SIZE * VOCAB_SIZE];
#             for pair in ids.windows(2) {
#                 counts[pair[0] as usize * VOCAB_SIZE + pair[1] as usize] += 1.0;
#             }
#             let weights = Tensor::from_vec(counts, (VOCAB_SIZE, VOCAB_SIZE), device)?;
#             Ok(Self { weights })
#         }
#         pub fn forward(&self, token: u32) -> Result<Tensor> {
#             one_hot(token, self.weights.device())?.matmul(&self.weights)
#         }
#     }
#     fn one_hot(id: u32, device: &Device) -> Result<Tensor> {
#         let mut row = vec![0_f32; VOCAB_SIZE];
#         row[id as usize] = 1.0;
#         Tensor::from_vec(row, (1, VOCAB_SIZE), device)
#     }
# }
# fn greedy_pick(logits: &Tensor) -> Result<u32> {
#     let rows = logits.to_vec2::<f32>()?;
#     let row = &rows[0];
#     let mut best = 0;
#     for (id, score) in row.iter().enumerate() {
#         if *score > row[best] {
#             best = id;
#         }
#     }
#     Ok(best as u32)
# }
fn main() -> Result<()> {
    let device = Device::Cpu;
    let model = model::Bigram::from_text(model::CORPUS, &device)?;

    let ids = tokenizer::encode("ru");
    let last = *ids.last().expect("prompt must not be empty");
    let logits = model.forward(last)?;
    println!("logits shape: {:?}", logits.dims());

    let next = greedy_pick(&logits)?;
    let piece = tokenizer::decode(&[next]).expect("an ascii corpus predicts ascii bytes");
    println!("after \"ru\" the bigram predicts token {next} -> {piece:?}");
    Ok(())
}
```

```bash
cargo run
```

```text
logits shape: [1, 256]
after "ru" the bigram predicts token 115 -> "s"
```

Read it against what you just built. `logits shape: [1, 256]` is unchanged
from Day 1, one score per [pantry jar](story.md#token). But the scores behind it now come
from `from_text`'s tally sheet, not a formula. The new second line shows
`forward` consulting the sheet. In the corpus, the most-marked column in
row `u` is `s` (byte 115), so `forward` scores it highest. One Day 1 test
goes away with the fake model: delete
`toy_model_predicts_the_next_ascii_byte`. Its job passes to three tests at
the bottom of `src/model.rs` that pin the counting itself:

```rust,ignore
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_the_corpus_matches_hand_tallies() -> Result<()> {
        let model = Bigram::from_text(CORPUS, &Device::Cpu)?;
        let after_s = model.forward(115)?.to_vec2::<f32>()?; // after 's'
        assert_eq!(after_s[0][116], 3.0); // "st" appears 3 times
        assert_eq!(after_s[0][104], 1.0); // "sh" appears once
        Ok(())
    }

    #[test]
    fn forward_scores_every_token() -> Result<()> {
        let model = Bigram::from_text(CORPUS, &Device::Cpu)?;
        assert_eq!(model.forward(114)?.dims(), &[1, VOCAB_SIZE]); // after 'r'
        Ok(())
    }

    #[test]
    fn unseen_bytes_have_all_zero_logits() -> Result<()> {
        let model = Bigram::from_text(CORPUS, &Device::Cpu)?;
        let row = model.forward(122)?.to_vec2::<f32>()?; // 'z' is not in the corpus
        assert!(row[0].iter().all(|&score| score == 0.0));
        Ok(())
    }
}
```

```bash
cargo test model
```

```text
running 3 tests
test model::tests::unseen_bytes_have_all_zero_logits ... ok
test model::tests::forward_scores_every_token ... ok
test model::tests::counting_the_corpus_matches_hand_tallies ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.00s
```

That third test foreshadows the rest of today. The model has **nothing to
say** about a byte it never saw, and section B turns that silence into
visible garbage, on purpose.

**Section checkpoint:** `./scripts/compare.sh v1-day-02a`

### Section B — autoregressive generation and the `generate` CLI

<div class="rust-concepts"><span>Rust used here:</span>
<a href="https://google.github.io/comprehensive-rust/pattern-matching/match.html"><code>match</code> &amp; patterns</a>
<a href="https://google.github.io/comprehensive-rust/testing/other.html">integration tests</a>
</div>

<div class="dl-concepts"><span>LLM concepts used here:</span>
<a href="https://nn.labml.ai/sampling/index.html">sampling &amp; decoding strategies (labml)</a>
</div>

Your model now makes one good prediction. But one token is a prediction,
and a sequence is generation. From here on, the cook chooses every next
course from the last one, over and over. The loop goes in `src/main.rs`,
above `main`:

```rust
# extern crate candle_core;
# use candle_core::{Result, Tensor};
# mod tokenizer {
#     pub fn encode(text: &str) -> Vec<u32> {
#         text.bytes().map(u32::from).collect()
#     }
#     pub fn decode(ids: &[u32]) -> Option<String> {
#         let bytes = ids
#             .iter()
#             .map(|&id| u8::try_from(id).ok())
#             .collect::<Option<Vec<u8>>>()?;
#         String::from_utf8(bytes).ok()
#     }
# }
# mod model {
#     use candle_core::{Result, Tensor};
#     pub struct Bigram;
#     impl Bigram {
#         pub fn forward(&self, _token: u32) -> Result<Tensor> {
#             unimplemented!()
#         }
#     }
# }
# fn greedy_pick(_logits: &Tensor) -> Result<u32> {
#     unimplemented!()
# }
/// Analogy: cook the next course from the last, over and over.
///
/// Autoregressive generation: predict one token, append it, feed it back
/// in, `steps` times. Decode once at the end so multi-byte characters stay
/// intact.
fn generate(model: &model::Bigram, prompt: &str, steps: usize) -> Result<String> {
    let mut ids = tokenizer::encode(prompt);
    for _ in 0..steps {
        let last = *ids.last().expect("prompt must not be empty");
        let logits = model.forward(last)?;
        ids.push(greedy_pick(&logits)?);
    }
    Ok(tokenizer::decode(&ids).expect("an ascii corpus only generates ascii bytes"))
}
```

That is nine lines, and it is *the* loop, the same one every production
engine runs. Notice two details. First, `generate` decodes the whole
sequence **once at the end**. That quietly fixes Day 1's "a single byte of
`é` doesn't decode" caveat. Second, each iteration's input is the
*previous iteration's output*. That data dependency is why generation
cannot be parallelized within one sequence. It is also why Days 8–14 fight
so hard to batch *across* sequences instead.

Now replace `main` with a real command-line interface:

```rust,no_run
# extern crate candle_core;
# use candle_core::{Device, Result};
# mod model {
#     use candle_core::{Device, Result};
#     pub const CORPUS: &str = "rust. rust. rust. rush.";
#     pub struct Bigram;
#     impl Bigram {
#         pub fn from_text(_text: &str, _device: &Device) -> Result<Self> {
#             Ok(Bigram)
#         }
#     }
# }
# fn generate(_m: &model::Bigram, _p: &str, _s: usize) -> Result<String> {
#     Ok(String::new())
# }
fn run_generate(prompt: &str, steps: usize) -> Result<()> {
    let device = Device::Cpu;
    let model = model::Bigram::from_text(model::CORPUS, &device)?;
    // Debug-print so invisible bytes (like the \0 an unseen prompt causes)
    // show up on screen instead of silently vanishing.
    println!("{:?}", generate(&model, prompt, steps)?);
    Ok(())
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = raw.iter().map(String::as_str).collect();
    match args.as_slice() {
        ["generate", prompt] => run_generate(prompt, 40),
        ["generate", prompt, steps] => {
            run_generate(prompt, steps.parse().expect("steps must be a number"))
        }
        _ => {
            eprintln!("usage: cargo run -- generate <prompt> [steps]");
            std::process::exit(2);
        }
    }
}
```

- `std::env::args()` yields the command-line arguments, and `.skip(1)` drops
  the binary's own name.

- `match args.as_slice()` with `["generate", prompt]` is a **slice
  pattern**. It matches a two-element slice whose first element is exactly
  `"generate"`, and it binds the second to `prompt`. The `_` arm catches
  everything else, because `match` refuses to compile unless every case is
  covered.

- A bad invocation prints usage to **stderr** and exits with code 2, the
  Unix convention for misused arguments. (We add no dependencies yet.
  `clap` can wait until the engine has real configuration to parse.)

Pin the loop's behavior with two tests at the bottom of the `tests` module
in `src/main.rs`. One pins the exact happy cycle, and the other pins the
garbage case that section A promised:

```rust,ignore
#[cfg(test)]
mod tests {
    ⋮
    ⋮
    // ──────── new code starts here (around line 69) ────────
    #[test]
    fn generation_is_deterministic_and_cyclic() -> Result<()> {
        let model = model::Bigram::from_text(model::CORPUS, &Device::Cpu)?;
        assert_eq!(generate(&model, "ru", 12)?, "rust. rust. ru");
        Ok(())
    }

    #[test]
    fn unseen_bytes_generate_token_zero_garbage() -> Result<()> {
        let model = model::Bigram::from_text(model::CORPUS, &Device::Cpu)?;
        // 'z' never appears in the corpus: all-zero logits, so greedy picks
        // token 0 forever. The model knows nothing outside its data.
        assert_eq!(generate(&model, "z", 3)?, "z\0\0\0");
        Ok(())
    }
    // ──────── new code ends here ────────
}
```

The CLI is now a user-facing contract, so give it the course's first
**integration tests**. These tests run the compiled binary from the
outside, exactly like a user would. They live in `tests/cli.rs`, inside a
new top-level `tests/` directory. (Cargo compiles each file there as its
own separate test program.)

```rust,ignore
//! Integration tests: drive the compiled binary exactly like a user would.
//! Cargo exposes the binary's path as CARGO_BIN_EXE_<name> to these tests.

use std::process::{Command, Output};

fn run_cli(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_nano-vllm-rs"))
        .args(args)
        .output()
        .expect("the course binary should run")
}

#[test]
fn generate_cli_is_deterministic() {
    let first = run_cli(&["generate", "ru", "12"]);
    let second = run_cli(&["generate", "ru", "12"]);
    assert!(first.status.success());
    assert_eq!(first.stdout, second.stdout);
    let text = String::from_utf8(first.stdout).expect("stdout is text");
    assert_eq!(text.trim(), "\"rust. rust. ru\"");
}

#[test]
fn missing_subcommand_exits_nonzero_with_usage() {
    let out = run_cli(&[]);
    assert!(!out.status.success());
    let usage = String::from_utf8(out.stderr).expect("stderr is text");
    assert!(usage.contains("usage:"));
}
```

Unit tests check functions from the inside. These tests check the
*product* from the outside: arguments, stdout, stderr, and exit codes. Both
determinism tests matter. In Day 15's closing benchmark, and in production
debugging, "same input, same output" is the property you will miss most
when it is gone.

**Section checkpoint:** `./scripts/compare.sh v1-day-02b`. It is identical
to `v1-day-02`, because the last section of a day *is* the day.

## 7. Run it

```bash
cargo run -- generate "ru" 40
```

(`generate <prompt>` alone defaults to 40 steps.)

## 8. Expected result

```text
"rust. rust. rust. rust. rust. rust. rust. "
```

Trace the first steps by hand with the tally sheet. After `u`, the only
mark is `s`. After `s`, it is **t: 3 vs h: 1**, so [greedy](story.md#greedy) takes `t`. After
`t` comes `.`, then space, then `r`, and the cycle closes. Notice what
greedy costs you. The corpus contains `rush`, but greedy *never* takes a
minority branch, so this model will say `rust.` forever and `rush.` never.
(Temperature sampling, on Day 10, is exactly the fix for "never".)

## 9. Test it

Fourteen tests now pin the engine: Day 1's seven survivors, section A's three
counting tests, section B's two loop tests, and two integration tests.

```bash
cargo test
```

```text
running 12 tests
test tokenizer::tests::decode_rejects_ids_larger_than_a_byte ... ok
test tokenizer::tests::roundtrip_ascii ... ok
test tokenizer::tests::multibyte_text_uses_more_tokens_than_chars ... ok
test tokenizer::tests::decode_rejects_invalid_utf8 ... ok
test tests::greedy_pick_breaks_ties_toward_the_lowest_id ... ok
test tests::matmul_rejects_mismatched_shapes ... ok
test tests::tiny_matmul_matches_hand_arithmetic ... ok
test model::tests::forward_scores_every_token ... ok
test model::tests::unseen_bytes_have_all_zero_logits ... ok
test model::tests::counting_the_corpus_matches_hand_tallies ... ok
test tests::unseen_bytes_generate_token_zero_garbage ... ok
test tests::generation_is_deterministic_and_cyclic ... ok

test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

running 2 tests
test missing_subcommand_exits_nonzero_with_usage ... ok
test generate_cli_is_deterministic ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.18s
```

The second `running 2 tests` block is `tests/cli.rs`. Cargo builds and runs
it as its own binary, after the unit tests.

## 10. Break it deliberately

You need no code change today. The engine breaks from the *outside*:

```bash
cargo run -- generate z 5
```

```text
"z\0\0\0\0\0"
```

The corpus never contains `z`, so its logits row is all zeros and every
token ties. Greedy's tie-break then picks token 0, the NUL byte, forever.
The debug-print makes the garbage visible. A raw print would show an
innocent blank line. You get three lessons in one:

- a model **knows nothing outside its data**;

- a deterministic tie-break turns "no information" into *confidently
  repeated* garbage, not noise;

- and thanks to your byte-level tokenizer, the engine still never crashed.
  Unknown input degrades; it does not panic.

Real models blunt this failure with smoothing, which gives every count a
tiny head start so nothing is ever *impossible*. They also use more data
and subword vocabularies. You will meet all three in time.

Also try `cargo run -- oops`. The usage message comes with exit code 2,
which the integration test pins.

## 11. Checkpoint

Every checkpoint — each section and the day — is done only when the full
green bar passes:

```bash
./scripts/check.sh
```

Then close out the day in whichever mode you are following:

**Built along on your own branch?** Confirm that your code matches the
finished day. The final ✓ line means it does:

```bash
./scripts/compare.sh v1-day-02
```

**Inspecting instead of typing?** Jump to today's exact finished state,
explore it, and come back:

```bash
git switch --detach v1-day-02
cargo test
git switch -          # back to where you were
```

**Either way**, you can read today's complete change set, docs included,
as one diff:

```bash
git diff v1-day-01 v1-day-02
```

## 12. Recap

**What changed today:** your engine gained `src/model.rs`, with its first
weights learned from data (a counted bigram). It gained an autoregressive
`generate` loop that decodes once at the end, and a `generate` CLI with
Unix-honest exit codes. It also gained its first integration tests, which
drive the compiled binary from the outside. That makes fourteen tests
across three modules and two test layers.

**Deliberately naive:** the context is one byte, so output cycles forever.
Unseen bytes produce confident NUL garbage, because there is no smoothing.
Generation runs a fixed step count, since there is no stop token. Greedy
never explores minority branches, so `rush.` is unreachable. And the corpus
is 23 bytes long.

**Tomorrow's limitation, resolved on Day 3:** one byte of context is the
wall. A cook who remembers a single jar can only cook in circles. To look
farther back, tokens must first become *vectors* that the model can mix
and transform. Day 3 builds token embeddings, linear layers with RMSNorm,
and rotary position embeddings. Those are the layers a transformer runs
before attention ever happens.
