#!/bin/bash
set -euo pipefail

# Run with:
#   bash submit_qwen.sh --model-path /path/to/model.gguf --model-alias NAME

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

j12=$(submit_model_job step_12_uid_tagging.sh)

j34=$(submit_model_job --dependency=afterok:"$j12" step_34_temporal.sh)

j5a=$(submit_model_job --dependency=afterok:"$j34" step_5a.sh)

j5b=$(submit_model_job --dependency=afterok:"$j5a" step_5b.sh)

j5c=$(submit_model_job --dependency=afterok:"$j5b" step_5c.sh)

j5d=$(submit_model_job --dependency=afterok:"$j5c" step_5d.sh)

j5e=$(submit_model_job --dependency=afterok:"$j5d" step_5e.sh)

j6=$(submit_model_job --dependency=afterok:"$j5e" step6_final_timeline.sh)

jpost=$(submit_model_job --dependency=afterok:"$j6" post_processing.sh)

printf 'Submitted dependency chain:\n'
printf '12=%s → 34=%s → 5a=%s → 5b=%s → 5c=%s → 5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j12" "$j34" "$j5a" "$j5b" "$j5c" "$j5d" "$j5e" "$j6" "$jpost"
