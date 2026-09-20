# Build nano-vLLM in Rust

An executable book. Over 15 days you build a single-GPU, text-only LLM
inference engine in Rust — from your first tensor to a streaming HTTP server
with real continuous batching, a paged KV cache, and honest benchmarks.

Each day is one theme built as two to four related sections, and the
repository history **is** the course: every day ends green and receives an
immutable checkpoint tag. Days 0–5 live in this repository, free; Days
6–15 continue in the Pro edition (see
[Get the full course](#get-the-full-course)). The full day-by-day syllabus
is in the book's [introduction](book/src/introduction.md).

## Prerequisites

Some programming experience, in any language — that is the whole list.
${\color{red}\textbf{Rust is not a prerequisite}}$: the language is taught as you go, one
concept chip at a time (each opens Comprehensive Rust in a side panel).
${\color{red}\textbf{Deep learning is not a prerequisite}}$: every concept arrives with a
plain-English analogy and a ▶ video, and the math is never required to
progress (see "What this course is — and isn't — about" below). No GPU is
needed until Day 9, and never for the tests.

## Why vLLM?

[vLLM](https://github.com/vllm-project/vllm) is the engine that serves
the AI boom: the open-source system most inference providers run or
imitate, and the project whose signature ideas — the paged KV cache,
continuous batching — every modern serving stack now lives on. When a
hosted model answers you, machinery shaped like vLLM produced the answer.

- **Inference is where LLMs meet production — and it is its own
  discipline.** Training gets the papers; serving gets the pager.
  Latency, throughput, memory ceilings, overload — the day-by-day
  syllabus below is, one for one, the skill set that operates every
  deployed model.

- **Using an engine teaches its flags; building one teaches its
  physics.** After you have built the KV cache, paged its memory, and
  batched requests through it with your own hands, no serving dashboard
  is mysterious again: you know why throughput falls off a cliff, what
  the memory ceiling is made of, and which knob actually moves.

By Day 15 the name on the cover is earned: you will have typed the
load-bearing parts of a vLLM-shaped engine yourself — and every serving
system you meet afterwards will read as familiar.

## Why Rust?

Fair question — the ML world speaks Python. Three honest reasons:

- **The second half of this course is systems programming.** The KV
  cache, paged memory, continuous batching, surviving load — those days
  are all one question in different clothes: *who owns which block of
  memory, and when is it safe to reuse?* Rust's ownership model makes you
  answer that out loud, so the compiler becomes a second teacher. In
  Python, those chapters would be hand-waving.

- **Every checkpoint has to work forever.** The whole course rests on
  `git checkout <any tag>` → green bar. One toolchain, no virtualenv
  drift, no "works on my machine": a Rust checkpoint that compiled on
  tag day still compiles years later. A Python version of that promise
  rots quietly.

- **Inference is a server you harden, not a notebook you poke.** Python
  is the right tool for *training* — quick experiments, instant
  feedback. But an inference engine is long-running, concurrent,
  memory-hungry production software, and that is Rust's home turf (it is
  why so much serious inference tooling is C++ or Rust under the hood).

And since the language is taught as you go (see Prerequisites), you get
a working knowledge of Rust as a side effect — arguably a fourth reason.

## What you build (Volume 1)

One engine — single GPU, text only — built in four arcs:

- **A working model (Days 1–7):** tensors and a byte tokenizer, a real
  transformer built from visible pieces, and a pinned small checkpoint whose
  outputs are verified token-for-token against a Python oracle.

- **Fast single-user inference (Days 8–9):** prefill/decode separation, a KV
  cache with correctness tests, honest TTFT/TPOT benchmarks, and a CUDA
  backend that always keeps a CPU fallback.

- **Many users at once (Days 10–14):** continuous batching, paged KV-cache
  management, and survival under pressure — admission control, preemption,
  chunked prefill, prefix caching, zero-leak cancellation.

- **A real service (Day 15):** an Axum server with SSE token streaming,
  backpressure, metrics, and a reproducible benchmark against Transformers,
  nano-vllm, and vLLM.

**Volume 2** (planned): tensor parallelism, custom CUDA kernels,
quantization, speculative decoding, multimodal inference.

## What this course is — and isn't — about

This is a course about **inference engineering**: batching, caching,
scheduling, memory management, serving. It is *not* a course about deep
learning or its math. Some days (3–6 especially) necessarily contain
math-flavored code — normalization formulas, rotations, attention — because
a real engine has to run a real model, and the code has to compile.

The contract for those parts: every concept gets a plain-English analogy, a
▶ video chip for intuition, and a link to a formal treatment — but
**understanding the math is never required to progress**. What you will
never be asked to do is type *blindly*: every block comes with a plain
statement of **what it is and why the engine needs it** — only the math of
*how it works inside* is optional. Typing the math-heavy blocks out is
still recommended (fingers build a familiarity that reading skips), and
copying them straight from the chapter is a completely legitimate choice:
that code exists so the engine works, not to test you. The lessons this
course actually cares about — the engine-side ideas — never hide inside
the math.

## The 15 days

Roughly 25–30 hours end to end. Each day is one theme built as two to four
related sections, and every section ends at a runnable, tagged checkpoint —
the per-section breakdown lives in the book's
[introduction](book/src/introduction.md).

| Day | Theme | In plain words | ≈&nbsp;Time |
|-----|-------|----------------|--------|
| 0&nbsp;✅ | Setup: toolchain and the empty tested scaffold | Install the tools and end with a tested, running (if empty) program. | 0.5&nbsp;h |
| 1 | From text to the next token | Text becomes numbers, numbers become scores, the top score becomes the next character. | 1.5–&#8288;2&nbsp;h |
| 2 | A complete toy language model | The model learns from real text by counting, then writes one predicted character at a time. | 1–&#8288;1.5&nbsp;h |
| 3 | The layers before attention | Each token trades its ID for a list of numbers with room for meaning — leveled, re-mixed, position-stamped. | 1.5–&#8288;2&nbsp;h |
| 4 | Causal attention | Tokens finally look at each other — earlier ones only — and decide whom to listen to. | 1.5&nbsp;h |
| 5 | The decoder block | Attention plus a think-it-over step, packaged into the unit real models stack dozens of times. | 1&nbsp;h |
| 6&nbsp;🔒 | A real model skeleton | Stack the blocks and load real trained weights — the machine you built by hand speaks. | 2.5&nbsp;h |
| 7&nbsp;🔒 | Real text, proven correct | A real tokenizer, and a Python referee confirming your output token for token. | 2&nbsp;h |
| 8&nbsp;🔒 | The KV cache | Stop redoing old work: keep what earlier tokens computed, and generation gets fast. | 2&nbsp;h |
| 9&nbsp;🔒 | Measure it, then GPU it | Time everything honestly first, then move the math to a GPU and measure again. | 1.5&nbsp;h |
| 10&nbsp;🔒 | Many requests, one engine | Several prompts share one engine, each with its own settings and place in line. | 2&nbsp;h |
| 11&nbsp;🔒 | Continuous batching | New requests hop into the running batch as old ones finish — no waiting for a full bus. | 2&nbsp;h |
| 12&nbsp;🔒 | Paged KV memory | Memory managed like a library: fixed-size pages, checked out, shared, returned. | 2&nbsp;h |
| 13&nbsp;🔒 | Attention over pages | Attention learns to read from those scattered pages, with the pool sized by real measurement. | 2&nbsp;h |
| 14&nbsp;🔒 | Survive pressure | Too many requests? Bump, pause, resume, split, cancel — without leaking a byte. | 2.5&nbsp;h |
| 15&nbsp;🔒 | Turn it into a service | An HTTP server streams tokens as they are born, refuses overload politely, and proves itself in a final benchmark. | 3&nbsp;h |

✅ = finished — the day's chapter and checkpoint tags are in this repository.
🔒 = Pro edition — see [Get the full course](#get-the-full-course).
Days 0–5 are free in this repository.

## Get the full course

**Days 0–5 are free, right here** — the complete arc from your first
tensor to a full transformer decoder block, with every checkpoint tag and
the whole book.

**Days 6–15 are the Pro edition**: loading real Qwen3 weights, a
token-for-token correctness oracle, the KV cache, GPU execution,
continuous batching, paged KV memory, survival under pressure, and the
final HTTP service — the same executable-book format, in a private
repository with lifetime access to Volume 1 updates.

And this book is one of a pair being written side by side: its sibling is
**[Build nano-Ray in Rust](https://github.com/Learning-LLM-By-Coding/build-nano-ray-in-rust)**
— a distributed compute engine, same format. Each book stands alone, and
owning one will earn a discount on the next; they meet only if you want
them to, on nano-Ray's final day, which can run this book's engine on
that book's cluster. Its planned syllabus is already up — star it there
to catch its launch too.

**[Get the Pro edition — $39, one-time →](https://buy.polar.sh/polar_cl_9N9RQ0HXzENjtCxbCucJuAn8E20HV6knlXOKF2VPZTl)**

One purchase, one seat: the invite to the private repository arrives
automatically, with days 6–7 ready today, days 8–15 as they ship, and
lifetime updates to Volume 1.

## How to follow the course

Read the book chapters in `book/` (rendered with [mdBook](https://rust-lang.github.io/mdBook/)):

```bash
mdbook serve book --open
```

`main` always holds the most recently finished day; every day's exact state
lives forever under its tag. Then pick one of two modes:

```bash
# Mode 1 — inspect the exact completed state of any day
git switch --detach v1-day-08
cargo test

# Mode 2 — build it yourself: start from yesterday, follow today's chapter
git switch -c my-course v1-day-07
# ...write the code from the Day 8 chapter, section by section...
./scripts/compare.sh v1-day-08a   # code-only diff after section A
./scripts/compare.sh v1-day-08    # the finished day
```

Every *section* of a day is one commit with its own checkpoint tag
(`v1-day-08a`, `v1-day-08b`, …), and the day's last section doubles as the
day tag. The compare script checks only the code you write (`src/`,
`tests/`, `Cargo.toml`) — never chapters, README, or course infrastructure —
and ignores comments, blank lines, and top-level item order, so its ✓ means
the section is done.

## The green bar

A checkpoint is not done until this passes at its tag:

```bash
./scripts/check.sh
```

The script runs, in order: `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test`, the day's smoke
run, and `mdbook test` — so the book's own code blocks compile and run
against the same pinned dependencies as the engine.

## Credits & further learning

This course stands on excellent public work. The reference chips throughout
the book link straight into these — follow them whenever you want more than
the course itself provides:

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — the ~1,200-line
  Python inference engine whose spirit (and name) this course borrows. Read
  it after Day 15; it will feel like meeting a pen pal.

- [Candle](https://github.com/huggingface/candle) — the Rust tensor library
  this engine builds on.

- [Comprehensive Rust](https://github.com/google/comprehensive-rust)
  (Google, Apache-2.0) — every Rust concept chip opens it in the side panel;
  the exact revision is pinned as the `third_party/comprehensive-rust`
  submodule.

- The ▶ video chips: [3Blue1Brown](https://www.youtube.com/@3blue1brown)'s
  deep-learning series, [StatQuest](https://www.youtube.com/@statquest),
  [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)'s *Neural
  Networks: Zero to Hero*, and
  [Efficient NLP](https://www.youtube.com/@EfficientNLP)'s RoPE explainer.

- [labml.ai annotated implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
  (MIT) — the "formally" chips: research papers implemented and annotated
  line by line.

- [How Transformer LLMs Work](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/)
  (DeepLearning.AI) — a free short course, a good guided tour of the
  architecture before our Day 4.

- [mdBook](https://github.com/rust-lang/mdBook) — renders this book.

None of these projects are affiliated with this course; all of them deserve
your time.

## License

MIT — see [LICENSE](LICENSE).
