# Setup

This is Day 0. At the end of this page you have a cloned repository where
`cargo test` passes. That empty-but-green scaffold is checkpoint `v1-day-00`.

## 1. Install Rust

Install [rustup](https://rustup.rs/), the Rust toolchain manager:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

You do not need to pick a version. The repository contains a
`rust-toolchain.toml` that pins the exact compiler this course was built and
tested with; rustup reads it and installs that version automatically the
first time you run `cargo` inside the repo.

## 2. Clone and verify

One thing to know before your first command: **`main` always holds the most
recently finished day** of the course, while every checkpoint's exact state
lives forever under its tag. This page is about Day 0, so jump to its
checkpoint first:

```bash
git clone https://github.com/Learning-LLM-By-Coding/build-nano-vllm-in-rust.git
cd build-nano-vllm-in-rust
git switch --detach v1-day-00
cargo test
cargo run
```

Expected output from `cargo run`:

```text
nano-vllm-rs scaffold is alive
```

If you see that line, your toolchain works and every later day will build.
(If you skip the `git switch` and run straight from `main`, you will see the
*latest* day's output instead — e.g. Day 1 prints
`text "hi" -> token ids [104, 105]` and friends. Nothing is wrong; you are
just ahead of this page.) Return to the newest state anytime with
`git switch main`.

Two cloning rules, because the checkpoints live in *tags* and tags are easy
to lose. The symptom is the same for both:
`fatal: invalid reference: v1-day-…` on a tag that plainly exists upstream.

1. **Clone the original, not a fork.** GitHub forks copy only the default
   branch — *none of the tags*. You never push to this repository, so you
   don't need a fork to follow the course. If you do want your build-along
   branch on your own GitHub, fork for pushing but fetch the checkpoints
   from the original:

   ```bash
   git remote add upstream https://github.com/Learning-LLM-By-Coding/build-nano-vllm-in-rust.git
   git fetch upstream --tags
   ```

2. **Clone plainly, as shown above.** A *shallow* clone (`--depth 1`, or a
   tool that shallow-clones for speed) skips every checkpoint tag that
   points into the unfetched history. The fix:

   ```bash
   git fetch --unshallow --tags
   ```

## 3. Install mdBook (optional but recommended)

The book you are reading is built with [mdBook](https://rust-lang.github.io/mdBook/).
To read it locally with live reload:

```bash
cargo install mdbook --locked
mdbook serve book --open
```

### Offline Rust reference (optional)

Each chapter's Rust concept links open Google's
[Comprehensive Rust](https://google.github.io/comprehensive-rust/) in a side
panel, straight from the live site. Two optional commands make that fully
self-contained:

```bash
# Stage the linked pages locally (internet needed once; gitignored).
./scripts/vendor-rust-ref.sh

# Check out the pinned upstream revision this course was written against.
git submodule update --init --depth 1 third_party/comprehensive-rust
```

The first fills a local folder the panel automatically prefers over the live
site. It stages the pages linked *at the time you run it*, so re-run it after
pulling new days if you want those pages offline too — until you do, the
panel quietly loads any missing page from the live site instead. The second
gives you the upstream course's own source — browsable offline, license and
all; nothing from it is copied into this repository's history. (The
deep-learning references — the ▶ videos and nn.labml.ai — stay online-only.)

## 4. Choose how you follow along

Every day is built as a few sections, and **every section is one commit with
its own checkpoint tag**: Day 5's sections are `v1-day-05a`, `v1-day-05b`, …
and the day's last section doubles as the day tag `v1-day-05`. Two workflows:

**Inspect mode** — jump to any finished checkpoint and poke at it:

```bash
git switch --detach v1-day-05
cargo test
```

**Build mode** — start from yesterday's checkpoint, write today's code
yourself by following the chapter, and compare after each section:

```bash
git switch -c my-course v1-day-04
# ...follow the Day 5 chapter, section A...
./scripts/compare.sh v1-day-05a    # code-only diff against the checkpoint
# ...section B...
./scripts/compare.sh v1-day-05b
# ...and after the last section:
./scripts/compare.sh v1-day-05     # the finished day
```

The compare script checks only the code you are expected to write (`src/`,
`tests/`, `Cargo.toml`); chapters, the README, `Cargo.lock`, and course
infrastructure are excluded on purpose. It is also forgiving about style:
comments, blank lines, extra whitespace, and the order of top-level items
(your `fn` above or below the checkpoint's — both fine) are all ignored.
It reads your files exactly as they sit on disk — no need to `git add` or
commit before comparing, and comparing never touches your git state.
**Its final ✓ line means the section is done**, even though your copy of the
book still shows yesterday's prose.

<div class="callout">

**While the course is under construction**, checkpoint tags are
occasionally re-pointed when earlier days receive fixes — and plain
`git fetch` / `git pull` deliberately does *not* move a tag you already
have. If a checkpoint ever looks stale (a missing file, output that does
not match its chapter), refresh your tags first:

```bash
git fetch --tags --force
```

</div>

## 5. The green bar

Every checkpoint of this course — including this one — must end green. Get
in the habit now:

```bash
./scripts/check.sh
```

The script (a few lines of bash — read it) runs five gates in order:

```text
cargo fmt --check                          # formatting
cargo clippy --all-targets -- -D warnings  # lints, warnings are errors
cargo test                                 # all tests
cargo run                                  # the day's smoke command
mdbook test book -L target/doctest-libs    # the book's own code blocks
```

That last gate compiles and runs the Rust code blocks inside the book
chapters themselves, so the prose can never silently rot away from the code.

**Checkpoint reached: `v1-day-00`.** On Day 1 you build the whole skeleton
of an LLM: text in, one greedily-chosen token out.
