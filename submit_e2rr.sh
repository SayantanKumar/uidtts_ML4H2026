#!/bin/bash
set -euo pipefail

# Run with the hard-coded Qwen defaults:
#   bash submit_e2rr.sh

WRAPPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$WRAPPER_DIR/run_scripts_slurm"
source "$WRAPPER_DIR/submit_model_utils.sh"

WRAPPER_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/DeepSeek-V3.2-UD-IQ2_M-00001-of-00005.gguf"
WRAPPER_MODEL_ALIAS="dsk_32"
WRAPPER_NOTES_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n40_notes/"
WRAPPER_STRUCTURED_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n40_filtered_time/"
WRAPPER_NOTE_LINK="/data/grimalditersans/mimic_uid_data/note_link_i2m4.csv"
WRAPPER_GPU_TYPE="a100"
WRAPPER_GPU_COUNT=4
configure_model_submission "$(basename "$0")" \
    --model-path "$WRAPPER_MODEL_PATH" \
    --model-alias "$WRAPPER_MODEL_ALIAS" \
    --notes-dir "$WRAPPER_NOTES_DIR" \
    --structured-dir "$WRAPPER_STRUCTURED_DIR" \
    --note-link "$WRAPPER_NOTE_LINK" \
    --gpu-type "$WRAPPER_GPU_TYPE" \
    --gpu-count "$WRAPPER_GPU_COUNT" \
    "$@"

[[ -d "$SCRIPT_DIR" ]] || {
    echo "Directory not found: $SCRIPT_DIR" >&2
    exit 1
}

cd "$SCRIPT_DIR"
mkdir -p sbatch_output sbatch_error

j5d=$(submit_model_job \
    --job-name="${WRAPPER_MODEL_ALIAS}_5d_e2rr" \
    step_5d_e2rr.sh)

j5e=$(submit_model_job \
    --dependency=afterok:"$j5d" \
    --job-name="${WRAPPER_MODEL_ALIAS}_5e_e2rr" \
    step_5e_e2rr.sh)

j6=$(submit_llm_job \
    --dependency=afterok:"$j5e" \
    --job-name="${WRAPPER_MODEL_ALIAS}_6_e2rr" \
    step_6_e2rr.sh)

jpost=$(submit_model_job \
    --dependency=afterok:"$j6" \
    --job-name="${WRAPPER_MODEL_ALIAS}_post_e2rr" \
    post_processing_e2rr.sh)

printf 'Submitted dependency chain for %s:\n' "$WRAPPER_MODEL_ALIAS"
printf '5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j5d" "$j5e" "$j6" "$jpost"
