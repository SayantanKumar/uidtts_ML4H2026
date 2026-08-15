#!/bin/bash
set -euo pipefail

# Run with the hard-coded Qwen defaults:
#   bash submit_nouid.sh

WRAPPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$WRAPPER_DIR/run_scripts_slurm"
source "$WRAPPER_DIR/submit_model_utils.sh"

WRAPPER_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
WRAPPER_MODEL_ALIAS="qwen35"
WRAPPER_NOTES_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n40_notes/"
WRAPPER_STRUCTURED_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n40_filtered_time/"
WRAPPER_NOTE_LINK="/data/grimalditersans/mimic_uid_data/note_link_i2m4.csv"
configure_model_submission "$(basename "$0")" \
    --model-path "$WRAPPER_MODEL_PATH" \
    --model-alias "$WRAPPER_MODEL_ALIAS" \
    --notes-dir "$WRAPPER_NOTES_DIR" \
    --structured-dir "$WRAPPER_STRUCTURED_DIR" \
    --note-link "$WRAPPER_NOTE_LINK" \
    "$@"

[[ -d "$SCRIPT_DIR" ]] || {
    echo "Directory not found: $SCRIPT_DIR" >&2
    exit 1
}

cd "$SCRIPT_DIR"
mkdir -p sbatch_output sbatch_error

j6=$(submit_model_job step_6_nouid.sh)

jpost=$(submit_model_job --dependency=afterok:"$j6" post_processing_nouid.sh)

printf 'Submitted dependency chain:\n'
printf '6=%s → post=%s\n' \
    "$j6" "$jpost"
