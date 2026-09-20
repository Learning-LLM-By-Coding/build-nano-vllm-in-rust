#!/usr/bin/env bash
# Stage the Comprehensive Rust pages this book links into
# book/src/comprehensive-rust/ so the Rust reference side panel works
# offline. The staging directory is GITIGNORED — nothing from the upstream
# course is committed to this repository. The upstream revision the course
# was written against is pinned as the third_party/comprehensive-rust
# submodule (source markdown; the panel needs the *built* pages, which
# upstream publishes only on its website, so this fetches those).
# Requires wget (brew install wget / apt install wget) and internet, once.
set -euo pipefail
cd "$(dirname "$0")/.."

SITE="https://google.github.io/comprehensive-rust/"
DEST="book/src/comprehensive-rust"

command -v wget >/dev/null || {
    echo "wget is required (brew install wget / apt install wget)" >&2
    exit 1
}

# Every comprehensive-rust URL the book links, plus the course root.
urls="$(grep -rhoE 'https://google\.github\.io/comprehensive-rust/[A-Za-z0-9/_.-]*' \
    book/src --include='*.md' | sort -u)"

rm -rf "$DEST"
mkdir -p "$DEST"
for url in $SITE $urls; do
    # Fetch the page plus the css/js/fonts it references, laid out with the
    # same relative paths the live site uses.
    wget --quiet --page-requisites --no-host-directories --cut-dirs=1 \
        --domains=google.github.io --directory-prefix="$DEST" "$url" ||
        echo "warn: could not stage $url" >&2
done

cat >"$DEST/README.md" <<'EOF'
Local, gitignored staging of the Comprehensive Rust pages this book links,
so the reference side panel works offline. Apache-2.0, (c) the Comprehensive
Rust authors. The pinned upstream source lives in the
third_party/comprehensive-rust submodule. Regenerate this directory any time
with scripts/vendor-rust-ref.sh.
EOF
date -u +"vendored %Y-%m-%dT%H:%M:%SZ" >"$DEST/vendored.ok"

echo "staged $(find "$DEST" -name '*.html' | wc -l | tr -d ' ') html files into $DEST (gitignored)"
