#!/usr/bin/env bash
# Downloads the model set used by CI into an HF-hub-cache-layout directory
# (models--<org>--<name>/snapshots/<rev>/...) that
# crates/mlex/tests/common's `registry()` (and its TS mirror) expect.
#
# Usage:
#   scripts/ci-download-models.sh <target-dir> [hf-repo...]
#
# If one or more HF repos are passed explicitly, only those are downloaded.
# Otherwise this falls back to the default minimal CI set used by the
# single-job coverage path.
#
# Env:   HF_TOKEN (optional but recommended - avoids anonymous rate limits)
set -euo pipefail

TARGET_DIR="${1:?usage: ci-download-models.sh <target-dir>}"
mkdir -p "$TARGET_DIR"
shift || true

if (($# > 0)); then
  MODELS=("$@")
else
  MODELS=(
    "mlx-community/gemma-4-E2B-it-qat-4bit"
  )
fi

VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlex-hf-cli.XXXXXX")"
trap 'rm -rf "$VENV_DIR"' EXIT
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade "huggingface_hub[cli]"
HF_CLI="$VENV_DIR/bin/huggingface-cli"

for repo in "${MODELS[@]}"; do
  echo "Downloading $repo -> $TARGET_DIR"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    "$HF_CLI" download "$repo" --cache-dir "$TARGET_DIR" --token "$HF_TOKEN"
  else
    "$HF_CLI" download "$repo" --cache-dir "$TARGET_DIR"
  fi
done
