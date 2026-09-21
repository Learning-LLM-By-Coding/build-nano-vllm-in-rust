#!/usr/bin/env bash
# The course green bar: a checkpoint is done when this script exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."

cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build
# The day's smoke command — update it whenever the CLI's shape changes.
cargo run --quiet -- generate "ru" 40

# `mdbook test` compiles and runs the book's Rust code blocks against our
# compiled dependencies. We stage only the actual libraries into a clean
# directory: clippy/`cargo check` also leave .rmeta files in deps/, and
# rustdoc refuses ambiguous duplicate candidates (E0464). If E0464 ever
# appears anyway, run `cargo clean` once — it means stale artifacts from an
# older dependency version are still lying around.
STAGE=target/doctest-libs
rm -rf "$STAGE" && mkdir -p "$STAGE"
# No rlibs exist until the project's first dependency arrives (Day 1).
cp target/debug/deps/*.rlib "$STAGE" 2>/dev/null || true
# Proc-macro crates compile to dynamic libraries, not rlibs.
cp target/debug/deps/*.dylib "$STAGE" 2>/dev/null || true
cp target/debug/deps/*.so "$STAGE" 2>/dev/null || true
mdbook test book -L "$STAGE"

echo "GREEN: all gates passed"
