#!/bin/bash
#SBATCH --job-name=uid_step5c
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
#   sbatch step5c_rag_ingest.sh

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

# Step 5C input directory.
export TABULAR_DIR="${TABULAR_DIR:-/data/grimalditersans/mimic_uid_data/i2m4_n30_tabular/}"

# Step 5C output:
#   $RAG_DIR/uid_infix_<patient_id>.csv/
export BUNDLE_DIR="$OUTPUT_DIR/bundle"
export RAG_OUTPUT_ROOT="$BUNDLE_DIR/tabular"
export RAG_INFIX="uid_infix"
export RAG_DIR="$RAG_OUTPUT_ROOT/$RAG_INFIX"

export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"
export STEP5_RAG_LOG="$LLAMA_LOG_DIR/step5c_rag_ingest.log"

export RAG_PIPELINE_SCRIPT="$PROJECT_DIR/rag_pipeline_llama_embed_folder_summarized_hf.py"

###############################################################################
# Hugging Face embedding model
###############################################################################

export HF_HOME="/data/grimalditersans/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export HF_DEVICE="auto"
export EMBEDDING_BATCH_SIZE=8
export HF_MAX_LENGTH=8192

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

    [[ -s "$file" ]] || return 1

    if [[ "$file" == *.gz ]]; then
        gzip -t "$file" >/dev/null 2>&1 || return 1
        gzip -cd "$file" |
            awk 'NF { found=1 } END { exit found ? 0 : 1 }'
    else
        awk 'NF { found=1 } END { exit found ? 0 : 1 }' "$file"
    fi
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
    "$RAG_OUTPUT_ROOT" \
    "$RAG_DIR" \
    "$LLAMA_LOG_DIR" \
    "$SELECTION_WORK_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$TABULAR_DIR" "tabular input directory"
check_path "$RAG_PIPELINE_SCRIPT" "RAG ingestion script"

shopt -s nullglob

tabular_files=("$TABULAR_DIR"/*.csv.gz)
filter_selected_files_in_place tabular_files

[[ "${#tabular_files[@]}" -gt 0 ]] ||
    die "No selected .csv.gz files found in $TABULAR_DIR"

for tabular_file in "${tabular_files[@]}"; do
    has_nonblank_content "$tabular_file" ||
        die "Tabular input is empty or invalid: $tabular_file"
done

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Tabular input directory: $TABULAR_DIR"
log "Step 5C output directory: $RAG_DIR"
log_file_selection
log "Tabular files: ${#tabular_files[@]}"
log "Embedding model: $EMBEDDING_MODEL"
log "Embedding device: $HF_DEVICE"
log "Embedding batch size: $EMBEDDING_BATCH_SIZE"
log "Maximum token length: $HF_MAX_LENGTH"
log "Using Python: $(which python)"
log "Python version: $(python --version)"

###############################################################################
# Prepare selected tabular inputs
###############################################################################

EFFECTIVE_TABULAR_DIR="$TABULAR_DIR"

if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/step5c_tabular_${SLURM_JOB_ID:-manual}_$$" \
        "$TABULAR_DIR/{}.csv.gz"

    EFFECTIVE_TABULAR_DIR="$STAGED_SELECTED_DIR"
fi

###############################################################################
# Step 5C: ingest tabular data into RAG indexes
###############################################################################

log "Starting Step 5C RAG ingestion"

if ! python "$RAG_PIPELINE_SCRIPT" \
    --folder "$EFFECTIVE_TABULAR_DIR/" \
    --output-folder "$RAG_OUTPUT_ROOT/" \
    --infix "$RAG_INFIX" \
    --embedding-model "$EMBEDDING_MODEL" \
    --device "$HF_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --max-length "$HF_MAX_LENGTH" \
    ingest \
    > "$STEP5_RAG_LOG" 2>&1; then
    log "WARNING: RAG ingestion returned a nonzero status"
fi

###############################################################################
# Validate Step 5C outputs
###############################################################################

validated_count=0

for tabular_file in "${tabular_files[@]}"; do
    # Example: 11121324.csv.gz becomes 11121324.csv.
    input_stem="$(basename "$tabular_file" .gz)"
    patient_rag_dir="$RAG_DIR/${RAG_INFIX}_${input_stem}"
    mapping_file="$patient_rag_dir/summary_mapping.json"
    chroma_database="$patient_rag_dir/chroma.sqlite3"

    [[ -d "$patient_rag_dir" ]] ||
        die "Missing RAG directory for $tabular_file: $patient_rag_dir"

    [[ -s "$mapping_file" ]] ||
        die "Missing or empty summary mapping: $mapping_file"

    python -m json.tool "$mapping_file" >/dev/null ||
        die "Invalid summary mapping JSON: $mapping_file"

    [[ -s "$chroma_database" ]] ||
        die "Missing or empty Chroma database: $chroma_database"

    validated_count=$((validated_count + 1))
done

log "Step 5C complete"
log "Validated RAG indexes: $validated_count"
log "Output for Step 5D: $RAG_DIR"
log "Ingestion log: $STEP5_RAG_LOG"
