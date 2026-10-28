# Setup

This is Day 0. By the end of this page you will have a cloned repository
where `cargo test` passes. That scaffold is empty but green, and it is
checkpoint `v1-day-00`.

## 1. Install Rust

Install [rustup](https://rustup.rs/), the Rust toolchain manager:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

You do not need to pick a version. The repository contains a
`rust-toolchain.toml` that pins the exact compiler this course was built and
tested with. The first time you run `cargo` inside the repo, rustup reads
that file and installs the pinned version for you.

## 2. Clone and verify

Before your first command, know this: **`main` always holds the most
recently finished day** of the course. Every checkpoint's exact state lives
forever under its tag. This page is about Day 0, so jump to its checkpoint
first:

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
*latest* day's output instead. Day 1, for example, prints
`text "hi" -> token ids [104, 105]` and friends. Nothing is wrong. You are
ahead of this page.) You can return to the newest state anytime with
`git switch main`.

The checkpoints live in *tags*, and tags are easy to lose. Two cloning
rules keep yours safe. Breaking either rule shows the same symptom:
`fatal: invalid reference: v1-day-…` on a tag that plainly exists upstream.

1. **Clone the original, not a fork.** GitHub forks copy only the default
   branch, with *none of the tags*. You never push to this repository, so
   you don't need a fork to follow the course. If you want your build-along
   branch on your own GitHub, push to a fork but fetch the checkpoints from
   the original:

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
panel, straight from the live site. If you want to read offline, two
optional commands make the panel fully self-contained:

```bash
# Stage the linked pages locally (internet needed once; gitignored).
./scripts/vendor-rust-ref.sh

# Check out the pinned upstream revision this course was written against.
git submodule update --init --depth 1 third_party/comprehensive-rust
```

The first command fills a local folder, and the panel prefers that folder
over the live site. It stages the pages linked *at the time you run it*.
After you pull new days, re-run it if you want those pages offline too.
Until you do, the panel quietly loads any missing page from the live site.
The second command gives you the upstream course's own source, browsable
offline, license and all. Nothing from it is copied into this repository's
history. (The deep-learning references, the ▶ videos and nn.labml.ai, stay
online-only.)

## 4. Choose how you follow along

Every day is built as a few sections, and **every section is one commit with
its own checkpoint tag**. Day 5's sections are `v1-day-05a`, `v1-day-05b`, …,
and the day's last section doubles as the day tag `v1-day-05`. You can
follow along in one of two ways:

**Inspect mode** — jump to any finished checkpoint and explore it:

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
`tests/`, `Cargo.toml`). It leaves out chapters, the README, `Cargo.lock`,
and course infrastructure on purpose. It is also forgiving about style. It
ignores comments, blank lines, extra whitespace, and the order of top-level
items, so your `fn` may sit above or below the checkpoint's. It reads your
files exactly as they sit on disk. You don't need to `git add` or commit
before comparing, and comparing never touches your git state.
**Its final ✓ line means the section is done**, even though your copy of the
book still shows yesterday's prose.

<div class="callout">

**While the course is under construction**, a checkpoint tag occasionally
moves when an earlier day receives a fix. A plain `git fetch` or `git pull`
deliberately does *not* move a tag you already have. If a checkpoint ever
looks stale (a missing file, or output that does not match its chapter),
refresh your tags first:

```bash
git fetch --tags --force
```

</div>

## 5. The green bar

Every checkpoint of this course must end green, including this one. It
helps to get in the habit now:

```bash
./scripts/check.sh
```

The script is a few lines of bash, and it is worth reading. It runs five
gates in order:

```text
cargo fmt --check                          # formatting
cargo clippy --all-targets -- -D warnings  # lints, warnings are errors
cargo test                                 # all tests
cargo run                                  # the day's smoke command
mdbook test book -L target/doctest-libs    # the book's own code blocks
```

That last gate compiles and runs the Rust code blocks inside the book
chapters themselves. It keeps the prose from silently drifting away from the
code.

**Checkpoint reached: `v1-day-00`.** On [Day 1](day-01-next-token.md) you
will build the whole skeleton of an LLM. Text goes in, and one
greedily-chosen token comes out.
