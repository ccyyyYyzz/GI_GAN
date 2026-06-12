#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-hadamard5_medium}"

case "$TASK" in
  hadamard5_medium)
    CONFIG="configs/colab/hadamard5_medium_noise001_colab.yaml"
    ;;
  rademacher10)
    CONFIG="configs/colab/rademacher10_full_noise001_colab.yaml"
    ;;
  scrambled10)
    CONFIG="configs/colab/scrambled_hadamard10_full_noise001_colab.yaml"
    ;;
  mnist5)
    CONFIG="configs/colab/mnist_hadamard5_full_colab.yaml"
    ;;
  fashion5)
    CONFIG="configs/colab/fashion_hadamard5_full_colab.yaml"
    ;;
  hadamard5_push)
    CONFIG="configs/colab/hadamard5_push_hq_colab.yaml"
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    echo "Valid: hadamard5_medium, rademacher10, scrambled10, mnist5, fashion5, hadamard5_push" >&2
    exit 2
    ;;
esac

bash scripts/colab_run_task.sh "$CONFIG"
