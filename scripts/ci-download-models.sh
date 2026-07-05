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

VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlex-hf.XXXXXX")"
trap 'rm -rf "$VENV_DIR"' EXIT
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade "huggingface_hub"
HF_PYTHON="$VENV_DIR/bin/python"

for repo in "${MODELS[@]}"; do
  echo "Downloading $repo -> $TARGET_DIR"
  REPO_ID="$repo" CACHE_DIR="$TARGET_DIR" HF_TOKEN="${HF_TOKEN:-}" "$HF_PYTHON" - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["REPO_ID"],
    cache_dir=os.environ["CACHE_DIR"],
    token=os.environ["HF_TOKEN"] or None,
)
PY
done
