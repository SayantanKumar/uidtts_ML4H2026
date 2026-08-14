#!/bin/bash
#SBATCH --job-name=uid_step5d
#SBATCH --time=2-00:00:00
#SBATCH --mem=128g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=sbatch_output/slurm_%j.out
#SBATCH --error=sbatch_error/slurm_%j.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step5d_e2rr.sh

set -eo pipefail

###############################################################################
# Environment
###############################################################################

module load CUDA/12.8.1 gcc cmake

module purge
module load CUDA/12.8.1 gcc/11.3.0 cmake/3.30.2

source /data/grimalditersans/myconda
conda activate /data/grimalditersans/mymamba/envs/TTA

if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -n "${CUDA_PATH:-}" && -d "$CUDA_PATH/lib64" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

###############################################################################
# Paths and settings
###############################################################################

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/outputs/${MODEL_OUTPUT_NAME:-qwen35}}"

export BUNDLE_DIR="$OUTPUT_DIR/bundle"

# Step 4 input.
export TML_DIR="$OUTPUT_DIR/temporal/answers"

# Step 5B input.
export AQG_INFIX="uidtts"
export AQG_DIR="$BUNDLE_DIR/$AQG_INFIX"

# Step 5C input.
export RAG_INFIX="uid_infix"
export RAG_OUTPUT_ROOT="$BUNDLE_DIR/tabular"
export RAG_DIR="$RAG_OUTPUT_ROOT/$RAG_INFIX"

# Step 5D output.
export COMPARISON_DIR="$AQG_DIR/e2rr_comparison_results"

export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"
export STEP5_RANK_LOG="$LLAMA_LOG_DIR/step5d_e2rr.log"

export ROW_RANKER_SCRIPT="$PROJECT_DIR/row_match_ranker_hf.py"

###############################################################################
# Hugging Face models
###############################################################################

export HF_HOME="/data/grimalditersans/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"

export HF_DEVICE="auto"
export EMBEDDING_BATCH_SIZE=8
export RERANKER_BATCH_SIZE=4
export HF_MAX_LENGTH=8192

export TOP_K=10

###############################################################################
# File selection
###############################################################################

export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

###############################################################################
# Helpers
###############################################################################

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $*"
}

die() {
    echo "[$(timestamp)] ERROR: $*" >&2
    exit 1
}

check_path() {
    local path="$1"
    local label="$2"
    [[ -e "$path" ]] || die "Missing $label: $path"
}

has_nonblank_content() {
    local file="$1"
    [[ -s "$file" ]] &&
        awk 'NF { found=1 } END { exit found ? 0 : 1 }' "$file"
}

###############################################################################
# Load file-selection helpers
###############################################################################

SELECTION_UTILS="$PROJECT_DIR/run_scripts/file_selection_utils.sh"
if [[ ! -f "$SELECTION_UTILS" ]]; then
    RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELECTION_UTILS="$RUN_SCRIPT_DIR/file_selection_utils.sh"
fi

check_path "$SELECTION_UTILS" "file_selection_utils.sh"
source "$SELECTION_UTILS"
init_file_selection

###############################################################################
# Preflight
###############################################################################

mkdir -p \
    "$HF_HOME" \
    "$TRANSFORMERS_CACHE" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$COMPARISON_DIR" \
    "$LLAMA_LOG_DIR" \
    "$SELECTION_WORK_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$BUNDLE_DIR" "bundle directory"
check_path "$AQG_DIR" "Step 5B anchor-query directory"
check_path "$TML_DIR" "Step 4 timeline directory"

check_path "$RAG_DIR" "Step 5C RAG directory"
check_path "$ROW_RANKER_SCRIPT" "row ranker script"

shopt -s nullglob

# anchor_jsons=("$AQG_DIR"/*.json)
# filter_selected_files_in_place anchor_jsons

# [[ "${#anchor_jsons[@]}" -gt 0 ]] ||
#     die "No selected Step 5B anchor-query files found in $AQG_DIR"

# for anchor_json in "${anchor_jsons[@]}"; do
#     has_nonblank_content "$anchor_json" ||
#         die "Anchor-query file is empty: $anchor_json"

#     python -m json.tool "$anchor_json" >/dev/null ||
#         die "Anchor-query file is invalid JSON: $anchor_json"
# done

timeline_files=("$TML_DIR"/*.bsv.gz)
filter_selected_files_in_place timeline_files

[[ "${#timeline_files[@]}" -gt 0 ]] ||
    die "No selected Step 4 timeline files found in $TML_DIR"

for timeline_file in "${timeline_files[@]}"; do
    has_nonblank_content "$timeline_file" ||
        die "Timeline file is empty: $timeline_file"
done

rag_dirs=("$RAG_DIR"/*)

[[ "${#rag_dirs[@]}" -gt 0 ]] ||
    die "No Step 5C RAG indexes found in $RAG_DIR"

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Step 4 input directory: $TML_DIR"
log "Step 5B directory: $AQG_DIR"
log "Step 5C input directory: $RAG_DIR"
log "Step 5D output directory: $COMPARISON_DIR"
log_file_selection
log "Timeline files: ${#timeline_files[@]}"
log "Embedding model: $EMBEDDING_MODEL"
log "Reranker model: $RERANKER_MODEL"
log "Device: $HF_DEVICE"
log "Top K: $TOP_K"
log "Using Python: $(which python)"
log "Python version: $(python --version)"

###############################################################################
# Prepare effective folder layout
###############################################################################

EFFECTIVE_BUNDLE_DIR="$BUNDLE_DIR"
TML_INPUT_DIR="$TML_DIR"

if selection_enabled; then
    EFFECTIVE_BUNDLE_DIR="$SELECTION_WORK_DIR/step5d_e2rr_bundle_${SLURM_JOB_ID:-manual}_$$"
    EFFECTIVE_TML_DIR="$EFFECTIVE_BUNDLE_DIR/temporal_answers"
    EFFECTIVE_RAG_DIR="$EFFECTIVE_BUNDLE_DIR/tabular/$RAG_INFIX"

    mkdir -p "$EFFECTIVE_TML_DIR" "$EFFECTIVE_RAG_DIR"

    stage_selected_files_for_bases \
        "$EFFECTIVE_TML_DIR" \
        "$TML_DIR/{}.bsv.gz"

    stage_selected_files_for_bases \
        "$EFFECTIVE_RAG_DIR" \
        "$RAG_DIR/${RAG_INFIX}_{}.csv"

    TML_INPUT_DIR="$EFFECTIVE_TML_DIR"
fi

###############################################################################
# Step 5D: compare events against RAG summaries
###############################################################################

log "Starting Step 5D event matching and reranking"

if ! python "$ROW_RANKER_SCRIPT" compare \
    --folder "$EFFECTIVE_BUNDLE_DIR" \
    --tml-dir "$TML_INPUT_DIR" \
    --rag-subdir "tabular/$RAG_INFIX" \
    --output-dir "$COMPARISON_DIR" \
    --k "$TOP_K" \
    --embedding-model "$EMBEDDING_MODEL" \
    --reranker-model "$RERANKER_MODEL" \
    --device "$HF_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --reranker-batch-size "$RERANKER_BATCH_SIZE" \
    --max-length "$HF_MAX_LENGTH" \
    > "$STEP5_RANK_LOG" 2>&1; then
    log "WARNING: row_match_ranker_hf.py returned a nonzero status"
fi

###############################################################################
# Validate Step 5D outputs
###############################################################################

validated_count=0

# for anchor_json in "${anchor_jsons[@]}"; do
#     case_name="$(basename "$anchor_json" .json)"
#     results_file="$COMPARISON_DIR/${case_name}_results.json"

#     has_nonblank_content "$results_file" ||
#         die "Missing or empty Step 5D result: $results_file"

#     python -m json.tool "$results_file" >/dev/null ||
#         die "Invalid Step 5D result JSON: $results_file"

#     validated_count=$((validated_count + 1))
# done

validated_count=0

for timeline_file in "${timeline_files[@]}"; do
    case_name="$(basename "$timeline_file" .bsv.gz)"
    results_file="$COMPARISON_DIR/${case_name}_results.json"

    has_nonblank_content "$results_file" ||
        die "Missing or empty Step 5D result: $results_file"

    python -m json.tool "$results_file" >/dev/null ||
        die "Invalid Step 5D result JSON: $results_file"

    validated_count=$((validated_count + 1))
done

log "Step 5D complete"
log "Validated comparison results: $validated_count"
log "Output for Step 5E: $COMPARISON_DIR"
log "Ranker log: $STEP5_RANK_LOG"
