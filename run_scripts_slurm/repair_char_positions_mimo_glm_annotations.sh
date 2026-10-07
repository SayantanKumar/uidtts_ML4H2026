#!/bin/bash
#SBATCH --job-name=mimo_glm_charpos
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1200g
#SBATCH --partition=gpuh200
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:h200:8
#SBATCH --output=sbatch_output/mimo_glm_charpos_%j.out
#SBATCH --error=sbatch_error/mimo_glm_charpos_%j.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/grimalditersans/uid_anchor_detection}"
MODEL_DIR="${MODEL_DIR:-/data/grimalditersans/models/MiMo-V2.6-Pro-RL}"
INPUT_DIR="${INPUT_DIR:-/data/grimalditersans/MLHC_TTA/compare_tts/glm_annotations}"
NOTE_DIR="${NOTE_DIR:-/data/grimalditersans/mimic_uid_data/i2m4_n40_notes}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/grimalditersans/MLHC_TTA/compare_tts/glm_annotations_charpos_fixed}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---dtype auto}"

case_ids=(13129657 11649157)

source /data/grimalditersans/myconda
conda activate /data/grimalditersans/mymamba/envs/gen-vllm

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR"

for case_id in "${case_ids[@]}"; do
    timeline="$INPUT_DIR/$case_id.bsv.gz"
    note="$NOTE_DIR/$case_id.txt"
    output="$OUTPUT_DIR/$case_id.bsv.gz"

    for required_path in "$REPO_DIR/repair_char_positions_llm.py" "$MODEL_DIR" "$timeline" "$note"; do
        if [[ ! -e "$required_path" ]]; then
            echo "ERROR: required path does not exist: $required_path" >&2
            exit 1
        fi
    done

    echo "Repairing case $case_id"
    command=(
        python3 "$REPO_DIR/repair_char_positions_llm.py"
        --timeline "$timeline"
        --note "$note"
        --output "$output"
        --model-path "$MODEL_DIR"
        --tensor-parallel-size 8
        --startup-timeout 3600
        --repair-all
        --overwrite
    )
    if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
        command+=(--vllm-extra-args "$VLLM_EXTRA_ARGS")
    fi

    set +e
    "${command[@]}"
    status=$?
    set -e

    if [[ "$status" -eq 1 && -s "$output" ]]; then
        echo "Case $case_id completed with unresolved rows; inspect ${output}.audit.jsonl" >&2
    elif [[ "$status" -ne 0 ]]; then
        exit "$status"
    fi
done
