#!/usr/bin/env bash
# Temporary helper: copy the currently-writing checkpoint files into
# checkpoints/<run_name>/ so each run's artifacts are preserved.
# Delete once train.py's --run_name flag is used for all launches.
set -e

RUN_NAME="${1:?usage: $0 <run_name>}"
SRC="checkpoints"
DEST="checkpoints/${RUN_NAME}"

mkdir -p "$DEST"
# Only copy when the destination is older than the source — runs launched
# with train.py --run_name write directly into checkpoints/<run_name>/, so
# copying from root would overwrite live state with stale root files.
for f in checkpoint_best.pt checkpoint_latest.pt training_history.json labels.txt; do
  if [[ -f "$SRC/$f" ]]; then
    if [[ ! -f "$DEST/$f" || "$SRC/$f" -nt "$DEST/$f" ]]; then
      cp -p "$SRC/$f" "$DEST/"
    fi
  fi
done
echo "Synced checkpoints/* → $DEST (newer-only)"
