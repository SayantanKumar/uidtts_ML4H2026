#!/bin/bash
#SBATCH --job-name=uid_step12
#SBATCH --time=2-00:00:00
#SBATCH --mem=32g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step_12_uid_tagging.sh

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

# Avoid Biowulf/Squid proxy for local llama-server calls.
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

###############################################################################
# Edit these paths/settings
###############################################################################

export REPO_ROOT="/data/CHARM-MIMIC/grimalditersans/LLM_pratice/Textual_tabular_alignment-main/uid"

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export SERVER_UTILS="$PROJECT_DIR/llm_server_utils.sh"

export INPUT_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n30_notes/"
#export INPUT_DIR="/data/grimalditersans/LLM_pratice/sample_case_reports/"  # what is this?

# Run either every .txt file in INPUT_DIR or only the hard-coded subset below.
# Use FILE_SELECTION_MODE="subset" and list filenames exactly as they appear in
# INPUT_DIR. You may also list bare case ids, e.g. "11121324".
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

export OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/outputs/${MODEL_OUTPUT_NAME:-qwen35}}"
# export OUTPUT_DIR="/data/CHARM-MIMIC/grimalditersans/LLM_pratice/sample_annotations/"  # what is this?

# Set to "txt.gz" if INPUT_DIR contains compressed reports.
export INPUT_TYPE="txt"
export OUT_EXTENSION=".txt"

export TAG_PROMPT_FILE="$PROJECT_DIR/templates/tts_tag.txt"

# Step 1 long-note handling. Files above STEP1_MAX_INPUT_LINES are split into
# chunks instead of skipped, then tag ids are renumbered globally in the final
# tagged report.
export STEP1_MAX_INPUT_LINES=500
export STEP1_CHUNK_LINES=450

export LLAMA_SERVER="/data/grimalditersans/llama.cpp/build-cuda-gcc11/bin/llama-server"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"

export CUDA_DEVICES="0,1"
export LLAMA_TIMEOUT=1200

# Model roles. Wrapper-provided values take precedence over these Qwen defaults.
export INSTRUCT_MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export INSTRUCT_MODEL_ALIAS="${INSTRUCT_MODEL_ALIAS:-qwen35}"

export REASONING_MODEL_PATH="${REASONING_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export REASONING_MODEL_ALIAS="${REASONING_MODEL_ALIAS:-qwen35}"
export PORT=8080

# Leave empty unless the llama-server model needs extra OpenAI body options.
export EXTRA_BODY=""

###############################################################################
# Derived paths
###############################################################################

export TAGGED_DIR="$OUTPUT_DIR/tagged"
export TAGS_DIR="$OUTPUT_DIR/tags"

PROMPT_IO_SCRIPT="$PROJECT_DIR/prompt_io.py"
EXTRACT_TAGS_SCRIPT="$PROJECT_DIR/extract_tagged_table.py"

STEP1_LOG="$LLAMA_LOG_DIR/step1_prompt_io.log"
STEP2_LOG="$LLAMA_LOG_DIR/step2_extract_tags.log"

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

cleanup_server() {
    if [[ -n "${LLAMA_SERVER_PID:-}" ]]; then
        kill_server "$LLAMA_SERVER_PID" "$PORT" || true
        unset LLAMA_SERVER_PID
    fi
}

SELECTION_UTILS="$PROJECT_DIR/run_scripts/file_selection_utils.sh"
if [[ ! -f "$SELECTION_UTILS" ]]; then
    RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELECTION_UTILS="$RUN_SCRIPT_DIR/file_selection_utils.sh"
fi
source "$SELECTION_UTILS"

###############################################################################
# Preflight
###############################################################################

mkdir -p "$TAGGED_DIR" "$TAGS_DIR" "$LLAMA_LOG_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$INPUT_DIR" "input directory"
check_path "$TAG_PROMPT_FILE" "tag prompt template"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$INSTRUCT_MODEL_PATH" "instruct model"
check_path "$PROMPT_IO_SCRIPT" "prompt_io.py"
check_path "$EXTRACT_TAGS_SCRIPT" "extract_tagged_table.py"
check_path "$PROJECT_DIR/call_lcpp.py" "call_lcpp.py"

init_file_selection
export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"
prepare_effective_input_dir

source "$SERVER_UTILS"

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Input directory: $INPUT_DIR"
log_file_selection
log "Output directory: $OUTPUT_DIR"
log "Input type: $INPUT_TYPE"
log "Using python: $(which python)"
log "Python version: $(python --version)"
log "Instruct model path: $INSTRUCT_MODEL_PATH"
log "Instruct model alias: $INSTRUCT_MODEL_ALIAS"
log "Reasoning model path: $REASONING_MODEL_PATH"
log "Reasoning model alias: $REASONING_MODEL_ALIAS"
log "Port: $PORT"

###############################################################################
# Step 1: tag clinical event mentions
###############################################################################

log "Starting llama-server for Step 1"
start_server "$INSTRUCT_MODEL_PATH" "$INSTRUCT_MODEL_ALIAS" "$PORT"

set -e

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib":$LD_LIBRARY_PATH

trap cleanup_server EXIT

log "Step 1: tagging clinical mentions with prompt_io.py"

if ! python "$PROMPT_IO_SCRIPT" \
    --in_dir "$EFFECTIVE_INPUT_DIR" \
    --input_type "$INPUT_TYPE" \
    --out_dir "$TAGGED_DIR" \
    --prompt_file "$TAG_PROMPT_FILE" \
    --llm_mechanism llama.cpp \
    --lcpp_url "http://127.0.0.1:${PORT}" \
    --model_name "$INSTRUCT_MODEL_ALIAS" \
    --endpoint_type openai \
    --extra_body "$EXTRA_BODY" \
    --max_input_lines "$STEP1_MAX_INPUT_LINES" \
    --chunk_overlength \
    --chunk_line_count "$STEP1_CHUNK_LINES" \
    --out_extension "$OUT_EXTENSION" \
    > "$STEP1_LOG" 2>&1; then
    log "WARNING: Step 1 prompt_io.py returned a nonzero status; continuing with any tagged reports that were produced"
fi

log "Step 1 complete. Tagged reports: $TAGGED_DIR"

###############################################################################
# Stop server before Step 2
###############################################################################

cleanup_server
trap - EXIT

###############################################################################
# Step 2: extract tag table and assign uid4 IDs
###############################################################################

log "Step 2: extracting tag tables with extract_tagged_table.py"

EFFECTIVE_TAGGED_DIR="$TAGGED_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/tagged_${SLURM_JOB_ID:-manual}_$$" \
        "$TAGGED_DIR/{}.txt" \
        "$TAGGED_DIR/{}.txt.gz"
    EFFECTIVE_TAGGED_DIR="$STAGED_SELECTED_DIR"
fi

if ! python "$EXTRACT_TAGS_SCRIPT" \
    --input-dir "$EFFECTIVE_TAGGED_DIR" \
    --output-dir "$TAGS_DIR" \
    > "$STEP2_LOG" 2>&1; then
    log "WARNING: Step 2 extract_tagged_table.py returned a nonzero status; continuing with any tag tables that were produced"
fi

log "Step 2 complete. UID tag tables: $TAGS_DIR"

###############################################################################
# Summary
###############################################################################

log "Done."
log "Step 1 outputs: $TAGGED_DIR"
log "Step 2 outputs: $TAGS_DIR"
log "Logs:"
log "  llama-server: $LLAMA_LOG_DIR/server_${PORT}.log"
log "  step 1: $STEP1_LOG"
log "  step 2: $STEP2_LOG"
