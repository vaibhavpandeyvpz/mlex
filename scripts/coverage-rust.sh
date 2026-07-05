#!/usr/bin/env bash
# Runs `cargo llvm-cov` for mlex (lib + integration tests) and writes an
# lcov report to target/llvm-cov/lcov.info plus a human-readable summary.
#
# On macOS with a Homebrew-installed rustc, the LLVM version bundled with
# Xcode's command line tools rarely matches the one rustc was built against,
# so `cargo llvm-cov` fails to find compatible `llvm-cov`/`llvm-profdata`
# binaries. We point it at Homebrew's `llvm` keg instead, which tracks the
# same LLVM release rustc links against.
#
# Usage:
#   scripts/coverage-rust.sh              # unit tests only (fast, no models)
#   scripts/coverage-rust.sh --all        # unit + integration tests (loads models)
#
# Env vars:
#   MLEX_MODELS_DIR   restrict which local model snapshots are discovered by
#                      the test registry (see crates/mlex/tests/common).
#                      Defaults to unset (full registry) unless already set
#                      by the caller.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v rustup >/dev/null 2>&1; then
  # CI (and any rustup-managed toolchain) - the standard, version-matched way.
  rustup component add llvm-tools-preview >/dev/null 2>&1 || true
elif command -v brew >/dev/null 2>&1 && brew --prefix llvm >/dev/null 2>&1; then
  # Homebrew rustc has no rustup component story; Homebrew's `llvm` keg
  # tracks the same LLVM release rustc links against.
  LLVM_PREFIX="$(brew --prefix llvm)"
  export LLVM_COV="${LLVM_COV:-$LLVM_PREFIX/bin/llvm-cov}"
  export LLVM_PROFDATA="${LLVM_PROFDATA:-$LLVM_PREFIX/bin/llvm-profdata}"
fi

mkdir -p target/llvm-cov

TARGETS=(--lib)
if [[ "${1:-}" == "--all" ]]; then
  TARGETS=(--lib --tests)
fi

cargo llvm-cov "${TARGETS[@]}" -p mlex \
  --lcov --output-path target/llvm-cov/lcov.info

cargo llvm-cov report --summary-only
