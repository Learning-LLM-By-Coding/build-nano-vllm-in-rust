# Introduction

You are going to build an LLM inference engine in Rust, from scratch, in 15
days. It is the kind of program that sits behind an API endpoint. It accepts
many prompts at once and streams the generated tokens back, while it carefully
juggles GPU memory.

By Day 15 your engine will have:

- a real transformer that loads a small pinned checkpoint and matches a
  Python reference implementation token-for-token,

- prefill/decode separation with a KV cache,

- **continuous batching** — new requests join the running batch as old ones
  finish, in one model call per step, not one call per request,

- **paged KV-cache management** — fixed-size memory blocks with allocation,
  reference counting, and zero-leak cancellation,

- admission control, preemption, chunked prefill, and prefix caching,

- an HTTP server with token streaming, overload handling, and metrics,

- a reproducible benchmark comparing it against Transformers, nano-vllm, and
  vLLM.

## Who this is for

You need some programming experience in any language. You do **not** need to
know Rust, transformer math, or GPU programming. You meet each one when the
engine first needs it. Everything through Day 8 runs on a plain CPU. A GPU
first becomes useful on Day 9, and the tests never require one.

## One story holds it all together

Every day adds a scene to one running story: a little restaurant. Each
piece of the engine is one thing in it — a jar in the pantry, a guest at a
table, a chef with a recipe book.
[The restaurant: an analogy to help you build the right mental model](story.md) keeps the whole story on one page.
Read it once before you start, as a map of the course. Come back to it
whenever an earlier idea gets fuzzy. In the chapters, a concept name with a
dotted underline links to that page. Click it, and a short reminder opens
right where you are reading. It tells you what the concept is, what it is in
the story, and which day introduced it. The story rides along in the code,
too. Every `Analogy:` comment is highlighted, so you can spot where a scene
from the story meets the code that builds it.

## How the course works

Each day is one chapter and one theme, built as **two to four related
sections**. Each section adds one focused capability that you can run when
you finish it, and a day's sections always tell a single story. Every
section is one git commit with its own checkpoint tag (`v1-day-03a`,
`v1-day-03b`, …). The day's last section doubles as the day tag
(`v1-day-03`). When you finish a section, `./scripts/compare.sh <tag>`
checks your code against that checkpoint. It looks only at your code
(`src/`, `tests/`, `Cargo.toml`), never at chapters or the README. It
ignores comments, blank lines, and the order of top-level items, so its ✓
means the section is done.

Each section introduces at most one new *systems* concept. You learn the
Rust language by reference instead of by detour. Every code section opens
with a small **Rust used here** strip. It links exactly the concepts its
code leans on (borrowing, `Option`, iterators, …) to the matching pages of
Google's [Comprehensive Rust](https://google.github.io/comprehensive-rust/)
course. Those links open in a side panel *next to* the code, so you can fill
a gap without leaving the page (Esc closes the panel). A git submodule
(`third_party/comprehensive-rust`) pins the exact upstream revision the
course references. One optional script stages the linked pages locally for
fully offline reading — see [Setup](setup.md). The course assumes no prior
Rust, so read only the strips you need.

Deep-learning theory gets the same deal. Sections that introduce ML
machinery (embeddings, normalization, attention, …) carry a teal **LLM
concepts used here** strip. For intuition, it offers short, excellent
teaching videos (3Blue1Brown, StatQuest, Andrej Karpathy). They are marked
▶ and play right inside the panel. For formal depth, it links [labml.ai's
annotated implementations](https://nn.labml.ai/). **It is fine not to fully
understand a concept the day you meet it.** Every one arrives with an
everyday analogy, the code is small and tested, and a good explanation is
one click away whenever you are ready.

Every chapter follows the same shape, so you always know where you are.
It shows what you will have running and what its output will show. It
explains why yesterday's engine is not enough and gives you a mental model.
Then come the code section by section, the exact commands to run, the
output you should see, tests, one deliberate way to break it, and a
checkpoint.

## The 15 days

| Day | Theme | Sections | ≈ Time |
|-----|-------|----------|--------|
| 1 | From text to the next token | Tensors & matmul · a byte tokenizer · logits & the greedy choice | 1.5–2 h |
| 2 | A complete toy language model | A bigram model · autoregressive generation & the `generate` CLI | 1–1.5 h |
| 3 | The layers before attention | Token embeddings · linear layers & RMSNorm · rotary positions (RoPE) | 1.5–2 h |
| 4 | Causal attention | Attention scores, the causal mask & softmax · grouped-query attention (GQA) | 1.5 h |
| 5 | The decoder block | The SwiGLU feed-forward · the pre-norm decoder block with residuals | 1 h |
| 6 | A real model skeleton | Stacked blocks & LM head · config & safetensors inspection · loading real weights | 2.5 h |
| 7 | Real text, proven correct | Real tokenizer & pinned checkpoint · Python correctness oracle | 2 h |
| 8 | The KV cache | Prefill/decode split · contiguous KV cache · cache correctness tests | 2 h |
| 9 | Measure it, then GPU it | TTFT/TPOT/memory/throughput benchmarks · CUDA backend with CPU fallback | 1.5 h |
| 10 | Many requests, one engine | Sequence state machine & per-request SamplingParams · temperature sampling (greedy = temperature 0, seeded) · FIFO scheduler · round-robin interleaving | 2 h |
| 11 | Continuous batching | True batched decode · continuous admission & a load generator | 2 h |
| 12 | Paged KV memory | Fixed-size BlockPool · per-sequence block tables · CPU reference paged-KV store | 2 h |
| 13 | Attention over pages | Block-table attention (reference) · GPU paged attention · VRAM-measured pool sizing & memory-aware admission | 2 h |
| 14 | Survive pressure | Preemption with recomputation · chunked prefill · prefix caching · zero-leak cancellation | 2.5 h |
| 15 | Turn it into a service | HTTP endpoint · SSE streaming · backpressure & 429s · observability · final benchmark & v0.1 | 3 h |

Plan on roughly 25–30 hours end to end.

Two rules keep the course honest:

1. **Every day ends green.** At every tag, formatting, lints, tests, the
   day's smoke command, and the book's own code examples all pass.

2. **Naive before optimized.** Every optimization arrives after a slow
   version you already understand and can see is correct. It also keeps an
   off-switch, so you can measure the difference yourself.

Head to [Setup](setup.md) to get your toolchain ready.
