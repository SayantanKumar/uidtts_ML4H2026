#!/bin/bash
set -euo pipefail

# Run with:
#   bash submit_qwen_e2rr.sh --model-path /path/to/model.gguf --model-alias NAME

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

j5d=$(submit_model_job step_5d_e2rr.sh)

j5e=$(submit_model_job --dependency=afterok:"$j5d" step_5e_e2rr.sh)

j6=$(submit_model_job --dependency=afterok:"$j5e" step_6_e2rr.sh)

jpost=$(submit_model_job --dependency=afterok:"$j6" post_processing_e2rr.sh)

printf 'Submitted dependency chain:\n'
printf '5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j5d" "$j5e" "$j6" "$jpost"
