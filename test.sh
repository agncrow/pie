#!/bin/bash
# Pie's whole suite against a crow binary, from this repository's root:
# each package's test blocks and its example in both modes, the parity
# programs (interpreted against compiled), and the RAG example's tests from
# its recorded fixtures in both modes.
# Usage: ./test.sh [path to crow]   (the crow on the PATH by default)
set -u
CROW=${1:-crow}
CROW=$(command -v "$CROW" || echo "$CROW")
cd "$(dirname "$0")" || exit 1
failed=()
step() {
    local name=$1
    shift
    echo "== $name"
    "$@" || failed+=("$name")
    echo
}
step packages "$CROW" test --packages
step parity "$CROW" test --parity
rag() {
    (cd examples/rag && "$CROW" test chat/corpus.co && "$CROW" test chat/retrieve.co && "$CROW" test --fixtures chat/page.co && "$CROW" test --fixtures --compiled chat/page.co)
}
step rag rag
if [ ${#failed[@]} -eq 0 ]; then
    echo "✓ pie: every step passes ($CROW)"
else
    echo "✗ pie: ${failed[*]} failed ($CROW)"
    exit 1
fi
