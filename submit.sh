#!/bin/bash
set -euo pipefail

# Run with the hard-coded Qwen defaults:
#   bash submit.sh

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

j12=$(submit_llm_job step_12_uid_tagging.sh)

j34=$(submit_llm_job --dependency=afterok:"$j12" step_34_temporal.sh)

j5a=$(submit_model_job --dependency=afterok:"$j34" step_5a.sh)

j5b=$(submit_llm_job --dependency=afterok:"$j5a" step_5b.sh)

j5c=$(submit_model_job --dependency=afterok:"$j5b" step_5c.sh)

j5d=$(submit_model_job --dependency=afterok:"$j5c" step_5d.sh)

j5e=$(submit_model_job --dependency=afterok:"$j5d" step_5e.sh)

j6=$(submit_llm_job --dependency=afterok:"$j5e" step6_final_timeline.sh)

jpost=$(submit_model_job --dependency=afterok:"$j6" post_processing.sh)

printf 'Submitted dependency chain:\n'
printf '12=%s → 34=%s → 5a=%s → 5b=%s → 5c=%s → 5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j12" "$j34" "$j5a" "$j5b" "$j5c" "$j5d" "$j5e" "$j6" "$jpost"
