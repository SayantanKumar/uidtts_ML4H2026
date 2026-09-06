#!/bin/bash
#SBATCH --job-name=uid_step5b
#SBATCH --time=1-00:00:00
#SBATCH --mem=128g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --qos=gpunlm2025.2
#SBATCH --output=sbatch_output/slurm_%j.out
#SBATCH --error=sbatch_error/slurm_%j.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step5b_generate_anchor_queries.sh

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

export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

###############################################################################
# Paths and settings
###############################################################################

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/outputs/${MODEL_OUTPUT_NAME:-single_case_qwen}}"

# Step 5A input directory.
export BUNDLE_DIR="$OUTPUT_DIR/bundle"

# Step 5B output directory:
#   /data/grimalditersans/uid_anchor_detection/outputs/gpt-oss-120b/bundle/uidtts
export AQG_INFIX="uidtts"
export AQG_DIR="$BUNDLE_DIR/$AQG_INFIX"

export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"
export STEP5_AQG_LOG="$LLAMA_LOG_DIR/step5b_anchor_query_generator.log"

export SERVER_UTILS="$PROJECT_DIR/llm_server_utils.sh"
export ANCHOR_QUERY_SCRIPT="$PROJECT_DIR/anchor_query_generator.py"

export LLAMA_SERVER="/data/grimalditersans/llama.cpp/build-cuda-gcc11/bin/llama-server"
export CUDA_DEVICES="${LLM_CUDA_DEVICES:-0,1}"
export LLAMA_TIMEOUT=1200
export PORT=8081

export REASONING_MODEL_PATH="${REASONING_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export REASONING_MODEL_ALIAS="${REASONING_MODEL_ALIAS:-qwen35}"

export AQG_SYSTEM_PROMPT="$PROJECT_DIR/templates/aqg_system.txt"
export AQG_USER_PROMPT="$PROJECT_DIR/templates/aqg_uid.txt"

export AQG_CHUNK_SIZE=25
export AQG_CONTEXT_WINDOW_CHARS=700
export AQG_MAX_REPORT_CHARS=25000
export AQG_MAX_QUERIES_PER_UID=3

export FILE_SELECTION_MODE="subset"  # all|subset
SELECTED_INPUT_FILES=(
    "11121324"
    "11649157"
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
    [[ -s "$file" ]] && awk 'NF { found=1 } END { exit found ? 0 : 1 }' "$file"
}

cleanup_llm_server() {
    if [[ -n "${LLAMA_SERVER_PID:-}" ]]; then
        kill_server "$LLAMA_SERVER_PID" "$PORT" || true
        unset LLAMA_SERVER_PID
    fi
}

###############################################################################
# File selection
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

mkdir -p "$AQG_DIR" "$LLAMA_LOG_DIR" "$SELECTION_WORK_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$BUNDLE_DIR" "Step 5A bundle directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$REASONING_MODEL_PATH" "reasoning model"
check_path "$AQG_SYSTEM_PROMPT" "anchor-query system prompt"
check_path "$AQG_USER_PROMPT" "anchor-query user prompt"
check_path "$ANCHOR_QUERY_SCRIPT" "anchor_query_generator.py"

source "$SERVER_UTILS"

shopt -s nullglob

bundle_reports=("$BUNDLE_DIR"/*.txt)
filter_selected_files_in_place bundle_reports

[[ "${#bundle_reports[@]}" -gt 0 ]] ||
    die "No selected Step 5A bundle reports found in $BUNDLE_DIR"

for bundle_report in "${bundle_reports[@]}"; do
    has_nonblank_content "$bundle_report" ||
        die "Step 5A bundle report is empty: $bundle_report"
done

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Step 5A input directory: $BUNDLE_DIR"
log "Step 5B output directory: $AQG_DIR"
log_file_selection
log "Bundle reports: ${#bundle_reports[@]}"
log "Using Python: $(which python)"
log "Python version: $(python --version)"
log "Model: $REASONING_MODEL_ALIAS"
log "Chunk size: $AQG_CHUNK_SIZE"
log "Context window characters: $AQG_CONTEXT_WINDOW_CHARS"
log "Maximum report characters: $AQG_MAX_REPORT_CHARS"
log "Maximum queries per UID: $AQG_MAX_QUERIES_PER_UID"

###############################################################################
# Prepare the effective input directory
###############################################################################

EFFECTIVE_BUNDLE_DIR="$BUNDLE_DIR"

if selection_enabled; then
    EFFECTIVE_BUNDLE_DIR="$SELECTION_WORK_DIR/step5b_bundle_${SLURM_JOB_ID:-manual}_$$"

    stage_selected_files_for_bases \
        "$EFFECTIVE_BUNDLE_DIR" \
        "$BUNDLE_DIR/{}.txt" \
        "$BUNDLE_DIR/{}.txt.txt"

    # anchor_query_generator.py writes into <folder>/<AQG_INFIX>.
    # This link keeps subset outputs in the permanent Step 5B directory.
    ln -sfn "$AQG_DIR" "$EFFECTIVE_BUNDLE_DIR/$AQG_INFIX"
fi

###############################################################################
# Step 5B: generate anchor queries
###############################################################################

# DeepSeek V3.2 workaround for llama.cpp PEG-native parsing failure.
case "$(basename "$INSTRUCT_MODEL_PATH")" in
    DeepSeek-V3.2-*)
        export LLAMA_ARG_SKIP_CHAT_PARSING=true
        ;;
    *)
        unset LLAMA_ARG_SKIP_CHAT_PARSING
        ;;
esac

log "Skip chat parsing: ${LLAMA_ARG_SKIP_CHAT_PARSING:-false}"
log "Starting llama-server"
start_server \
    "$REASONING_MODEL_PATH" \
    "$REASONING_MODEL_ALIAS" \
    "$PORT"

trap cleanup_llm_server EXIT

log "Generating anchor queries"

if ! python "$ANCHOR_QUERY_SCRIPT" generate \
    --folder "$EFFECTIVE_BUNDLE_DIR" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --infix "$AQG_INFIX" \
    --system-prompt "$AQG_SYSTEM_PROMPT" \
    --user-prompt "$AQG_USER_PROMPT" \
    --model-name "$REASONING_MODEL_ALIAS" \
    --chunk-size "$AQG_CHUNK_SIZE" \
    --context-window-chars "$AQG_CONTEXT_WINDOW_CHARS" \
    --max-report-chars "$AQG_MAX_REPORT_CHARS" \
    --max-queries-per-uid "$AQG_MAX_QUERIES_PER_UID" \
    > "$STEP5_AQG_LOG" 2>&1; then
    log "WARNING: anchor_query_generator.py returned a nonzero status"
fi

cleanup_llm_server
trap - EXIT

###############################################################################
# Validate Step 5B outputs
###############################################################################

anchor_jsons=("$AQG_DIR"/*.json)
filter_selected_files_in_place anchor_jsons

[[ "${#anchor_jsons[@]}" -gt 0 ]] ||
    die "No anchor-query JSON files were produced in $AQG_DIR"

for anchor_json in "${anchor_jsons[@]}"; do
    has_nonblank_content "$anchor_json" ||
        die "Anchor-query output is empty: $anchor_json"

    python -m json.tool "$anchor_json" >/dev/null ||
        die "Anchor-query output is invalid JSON: $anchor_json"
done

log "Step 5B complete"
log "Anchor-query files: ${#anchor_jsons[@]}"
log "Output for later steps: $AQG_DIR"
log "Generator log: $STEP5_AQG_LOG"
log "Server log: $LLAMA_LOG_DIR/server_${PORT}.log"
