#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Example:
# python Retrofit-Merge.py --base_model_path ... --adapter_paths ... --val_files ... --out_adapter_dir ... --out_full_pt_path ...
python "$SCRIPT_DIR/Retrofit-Merge.py" "$@"

# Optional second step when you want a standard merged HuggingFace checkpoint:
# python "$SCRIPT_DIR/Util/merge_and_unload_adapter.py" \
#   --base_model_path /path/to/base_model \
#   --adapter_path /path/to/adapter \
#   --save_dir /path/to/save_dir
