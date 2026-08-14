#!/bin/bash
set -euo pipefail

# Run with:
#   bash submit_qwen_noranker.sh --model-path /path/to/model.gguf --model-alias NAME

WRAPPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$WRAPPER_DIR/run_scripts_qwen"
source "$WRAPPER_DIR/submit_model_utils.sh"
configure_model_submission "$(basename "$0")" "$@"

[[ -d "$SCRIPT_DIR" ]] || {
    echo "Directory not found: $SCRIPT_DIR" >&2
    exit 1
}

cd "$SCRIPT_DIR"
mkdir -p sbatch_output sbatch_error

j5d=$(submit_model_job step_5d_norerank.sh)

j5e=$(submit_model_job --dependency=afterok:"$j5d" step_5e_norerank.sh)

j6=$(submit_model_job --dependency=afterok:"$j5e" step_6_norerank.sh)

jpost=$(submit_model_job --dependency=afterok:"$j6" post_processing_norerank.sh)

printf 'Submitted dependency chain:\n'
printf '5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j5d" "$j5e" "$j6" "$jpost"
