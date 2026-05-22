#!/usr/bin/env bash
set -euo pipefail

: "${CODET5_SH_DIR:?Set CODET5_SH_DIR to the upstream CodeT5/sh directory}"

cd "$CODET5_SH_DIR"

# The initial continual-learning checkpoint is the fully fine-tuned CodeT5-C model.
model="${MODEL_TAG:-codet5_base}"
lang="${LANG_TAG:-decomC}"
log_file="${LOG_FILE:-log.txt}"

python3 run_exp.py --model_tag "$model" --task summarize --sub_task "$lang" "$@" |& tee "$log_file"
