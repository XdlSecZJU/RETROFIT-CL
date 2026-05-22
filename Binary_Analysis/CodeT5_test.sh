#!/usr/bin/env bash
set -euo pipefail

: "${CODET5_SH_DIR:?Set CODET5_SH_DIR to the upstream CodeT5/sh directory}"

cd "$CODET5_SH_DIR"

model="${MODEL_TAG:-codet5_base}"
log_file="${LOG_FILE:-log.txt}"

if [ "$#" -gt 0 ]; then
  langs=("$@")
else
  langs=(decomC demiStripped strippedDecomC)
fi

: > "$log_file"
for lang in "${langs[@]}"; do
  python3 run_exp.py --model_tag "$model" --task summarize --sub_task "$lang" |& tee -a "$log_file"
done
