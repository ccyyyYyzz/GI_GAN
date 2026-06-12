#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-mnist5}"

case "$TASK" in
  hadamard5_medium)
    RUN_NAME="hadamard5_medium_noise001"
    ;;
  rademacher10)
    RUN_NAME="rademacher10_full_noise001"
    ;;
  scrambled10)
    RUN_NAME="scrambled_hadamard10_full_noise001"
    ;;
  mnist5)
    RUN_NAME="mnist_hadamard5_full"
    ;;
  fashion5)
    RUN_NAME="fashion_hadamard5_full"
    ;;
  hadamard5_push)
    RUN_NAME="hadamard5_push_hq"
    ;;
  *)
    RUN_NAME="$TASK"
    ;;
esac

OUTPUT_ROOT="/content/ns_mc_gan_gi_outputs"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
ARCHIVE_DIR="/content/ns_mc_gan_gi_archives"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$ARCHIVE_DIR/${RUN_NAME}_${STAMP}.tar.gz"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Missing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$ARCHIVE_DIR"
tar -czf "$ARCHIVE" -C "$OUTPUT_ROOT" "$RUN_NAME"

echo "$ARCHIVE"
