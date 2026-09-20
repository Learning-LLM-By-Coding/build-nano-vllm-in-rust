#!/usr/bin/env bash
# Compare YOUR code against a checkpoint tag, ignoring docs, infra, and
# style: comments, blank lines, extra whitespace, and the order of top-level
# items (fn A above fn B or below it — both fine) are all forgiven. Only the
# substance of the code has to match.
#
#   ./scripts/compare.sh v1-day-01a    # after finishing a section
#   ./scripts/compare.sh v1-day-01     # after finishing a day
#
# A final ✓ line means your code matches the checkpoint. Differences are
# shown in normalized form; for a raw line-by-line diff use
#   git diff <tag> -- src tests Cargo.toml
#
# The script reads your files exactly as they are on disk — new untracked
# files and unstaged edits included. You never need to `git add` or commit
# before comparing, and comparing never touches your git state.
#
# The normalizer is a heuristic: it strips //-comments textually and counts
# braces to find top-level items, so an unbalanced brace inside a string
# literal could confuse it, and items nested inside a block (e.g. tests
# inside `mod tests`) still compare in order. Good enough for course code.
set -euo pipefail
cd "$(dirname "$0")/.."
export LC_ALL=C

if [ $# -ne 1 ]; then
    echo "usage: $0 <checkpoint-tag>" >&2
    echo "available checkpoints:" >&2
    git tag --list 'v1-day-*' >&2
    exit 2
fi
TAG=$1
git rev-parse -q --verify "$TAG^{commit}" >/dev/null || {
    echo "unknown checkpoint '$TAG' — available:" >&2
    git tag --list 'v1-day-*' >&2
    if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
        echo "this is a SHALLOW clone — older checkpoint tags were never fetched." >&2
        echo "fix: git fetch --unshallow --tags" >&2
    elif [ -z "$(git tag --list 'v1-day-*')" ]; then
        echo "no checkpoint tags at all — this is usually a FORK clone (GitHub" >&2
        echo "forks copy only the default branch, never tags). Fetch them from" >&2
        echo "the original repository:" >&2
        echo "  git remote add upstream https://github.com/Learning-LLM-By-Coding/build-nano-vllm-in-rust.git" >&2
        echo "  git fetch upstream --tags" >&2
    fi
    exit 2
}

# Only the paths learners write. Chapters (book/), README, the generated
# Cargo.lock, and course infrastructure (scripts/) are deliberately excluded.
CODE_PATHS=(src tests Cargo.toml)

# Rust: strip comments, blanks, and extra whitespace, then sort the
# top-level brace-delimited items so declaration order doesn't matter.
normalize_rust() {
    sed -E -e 's|^[[:space:]]*//.*$||' -e 's|[[:space:]]+//.*$||' -e 's|/\*[^*]*\*/||g' |
        awk '
            {
                gsub(/[ \t]+/, " ")
                gsub(/^ +| +$/, "")
                if ($0 == "") next
                block = block $0 "\001"
                depth += gsub(/\{/, "{") - gsub(/\}/, "}")
                if (depth <= 0) {
                    print block
                    block = ""
                    depth = 0
                }
            }
            END { if (block != "") print block }
        ' | sort | awk '{ gsub(/\001/, "\n"); print }'
}

# Everything else (Cargo.toml): strip #-comments and blanks; line order in
# TOML tables is not significant, so sort.
normalize_generic() {
    sed -E -e 's|^[[:space:]]*#.*$||' -e 's|[[:space:]]+#.*$||' |
        awk '{ gsub(/[ \t]+/, " "); gsub(/^ +| +$/, ""); if ($0 != "") print }' |
        sort
}

normalize_file() { # $1 = filename; content on stdin
    case "$1" in
    *.rs) normalize_rust ;;
    *) normalize_generic ;;
    esac
}

# Union of checkpoint files and your files (tracked or not) under CODE_PATHS.
files=$(
    {
        git ls-tree -r --name-only "$TAG" -- "${CODE_PATHS[@]}"
        git ls-files -co --exclude-standard -- "${CODE_PATHS[@]}"
    } | sort -u
)

status=0
for f in $files; do
    in_tag=1
    tag_raw=$(git show "$TAG:$f" 2>/dev/null) || in_tag=0
    if [ "$in_tag" = 0 ]; then
        echo "✗ $f: not in $TAG (extra file)"
        status=1
        continue
    fi
    if [ ! -f "$f" ]; then
        echo "✗ $f: missing from your tree"
        status=1
        continue
    fi
    if [ "$tag_raw" = "$(cat "$f")" ]; then
        continue # byte-identical
    fi
    tag_norm=$(printf '%s\n' "$tag_raw" | normalize_file "$f")
    work_norm=$(normalize_file "$f" <"$f")
    if [ "$tag_norm" = "$work_norm" ]; then
        echo "~ $f: matches (only comments, blank lines, or item order differ)"
    else
        echo "✗ $f:"
        diff -u --label "checkpoint/$f" --label "yours/$f" \
            <(printf '%s\n' "$tag_norm") <(printf '%s\n' "$work_norm") || true
        status=1
    fi
done

if [ "$status" = 0 ]; then
    echo "✓ your code matches $TAG (comments, blank lines, and top-level item order ignored)"
else
    echo "✗ differences from $TAG shown above (raw diff: git diff $TAG -- ${CODE_PATHS[*]})" >&2
fi
exit "$status"
