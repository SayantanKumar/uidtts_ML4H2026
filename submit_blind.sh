#!/bin/bash
set -euo pipefail

# Run with:
#   bash submit_blind.sh --model-path /path/to/model.gguf --model-alias NAME

WRAPPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$WRAPPER_DIR/run_scripts_slurm"
source "$WRAPPER_DIR/submit_model_utils.sh"
configure_model_submission "$(basename "$0")" "$@"

[[ -d "$SCRIPT_DIR" ]] || {
    echo "Directory not found: $SCRIPT_DIR" >&2
    exit 1
}

cd "$SCRIPT_DIR"
mkdir -p sbatch_output sbatch_error

j6=$(submit_model_job step_6_blind.sh)

jpost=$(submit_model_job --dependency=afterok:"$j6" post_processing_blind.sh)

printf 'Submitted dependency chain:\n'
printf '6=%s → post=%s\n' \
    "$j6" "$jpost"
