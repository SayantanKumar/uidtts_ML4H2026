#!/bin/bash
set -euo pipefail

# Submit the dependency chain with the hard-coded defaults:
#   bash submit.sh
#
# Or, from a shell on an already allocated interactive node, run every step
# sequentially without submitting additional Slurm jobs:
#   bash submit.sh --interactive

RUN_INTERACTIVE=false
MODEL_OPTIONS=()
for option in "$@"; do
    case "$option" in
        --interactive)
            RUN_INTERACTIVE=true
            ;;
        *)
            MODEL_OPTIONS+=("$option")
            ;;
    esac
done

WRAPPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$WRAPPER_DIR/run_scripts_slurm"
source "$WRAPPER_DIR/submit_model_utils.sh"

WRAPPER_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf"
WRAPPER_MODEL_ALIAS="test"
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
    "${MODEL_OPTIONS[@]}"

[[ -d "$SCRIPT_DIR" ]] || {
    echo "Directory not found: $SCRIPT_DIR" >&2
    exit 1
}

cd "$SCRIPT_DIR"
mkdir -p sbatch_output sbatch_error

if [[ "$RUN_INTERACTIVE" == true ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "Interactive mode must be run inside a Slurm allocation." >&2
        echo "Start one with sinteractive, then rerun: bash submit.sh --interactive" >&2
        exit 1
    fi

    for model_path in "$INSTRUCT_MODEL_PATH" "$REASONING_MODEL_PATH"; do
        if [[ "$model_path" != *.gguf ]]; then
            echo "Interactive llama-server run requires a GGUF model: $model_path" >&2
            exit 1
        fi
    done

    interactive_steps=(
        step_12_uid_tagging.sh
        step_34_temporal.sh
        step_5a.sh
        step_5b.sh
        step_5c.sh
        step_5d.sh
        step_5e.sh
        step6_final_timeline.sh
        post_processing.sh
    )

    printf 'Running pipeline sequentially in interactive Slurm job %s.\n' "$SLURM_JOB_ID"
    for step_script in "${interactive_steps[@]}"; do
        printf '\n==> Running %s\n' "$step_script"
        bash "$SCRIPT_DIR/$step_script"
    done

    printf '\nInteractive pipeline completed for %s.\n' "$WRAPPER_MODEL_ALIAS"
    exit 0
fi

j12=$(submit_llm_job \
    --job-name="${WRAPPER_MODEL_ALIAS}_12" \
    step_12_uid_tagging.sh)

j34=$(submit_llm_job \
    --dependency=afterok:"$j12" \
    --job-name="${WRAPPER_MODEL_ALIAS}_34" \
    step_34_temporal.sh)

j5a=$(submit_model_job \
    --dependency=afterok:"$j34" \
    --job-name="${WRAPPER_MODEL_ALIAS}_5a" \
    step_5a.sh)

j5b=$(submit_llm_job \
    --dependency=afterok:"$j5a" \
    --job-name="${WRAPPER_MODEL_ALIAS}_5b" \
    step_5b.sh)

j5c=$(submit_model_job \
    --dependency=afterok:"$j5b" \
    --job-name="${WRAPPER_MODEL_ALIAS}_5c" \
    step_5c.sh)

j5d=$(submit_model_job \
    --dependency=afterok:"$j5c" \
    --job-name="${WRAPPER_MODEL_ALIAS}_5d" \
    step_5d.sh)

j5e=$(submit_model_job \
    --dependency=afterok:"$j5d" \
    --job-name="${WRAPPER_MODEL_ALIAS}_5e" \
    step_5e.sh)

j6=$(submit_llm_job \
    --dependency=afterok:"$j5e" \
    --job-name="${WRAPPER_MODEL_ALIAS}_6" \
    step6_final_timeline.sh)

jpost=$(submit_model_job \
    --dependency=afterok:"$j6" \
    --job-name="${WRAPPER_MODEL_ALIAS}_post" \
    post_processing.sh)

printf 'Submitted dependency chain for %s:\n' "$WRAPPER_MODEL_ALIAS"
printf '12=%s → 34=%s → 5a=%s → 5b=%s → 5c=%s → 5d=%s → 5e=%s → 6=%s → post=%s\n' \
    "$j12" "$j34" "$j5a" "$j5b" "$j5c" "$j5d" "$j5e" "$j6" "$jpost"
