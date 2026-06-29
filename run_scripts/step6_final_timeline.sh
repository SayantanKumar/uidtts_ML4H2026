#!/bin/bash
#SBATCH --job-name=uid_step6
#SBATCH --time=2-00:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:4
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=kumarsayantan94@gmail.com

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step6_final_timeline.sh

set -eo pipefail

###############################################################################
# Environment
###############################################################################

module load CUDA/12.8.1 gcc cmake

set +u
eval "$(conda shell.bash hook)"
conda activate /data/kumars33/conda/envs/TTA
set -u

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

export PROJECT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection"
export SERVER_UTILS="$PROJECT_DIR/llm_server_utils.sh"

export OUTPUT_DIR="$PROJECT_DIR/outputs/qwen35/"

# Run either every completed Step 5 case or only the hard-coded subset below.
# Use the original input filenames or bare case ids.
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

# File with note_id/hadm_id/admittime/dischtime mappings.
# Must match the file used in Step 5.
export NOTE_LINK_PATH="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/note_link_i2m4.csv"

export LLAMA_SERVER="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/llama.cpp/build2603/bin/llama-server"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"

export CUDA_DEVICES="0,1"
export LLAMA_TIMEOUT=1200

# Model roles. Both point to DeepSeek V3.2 for now; change either later if you
# want separate instruct/reasoning models. Step 6 uses the reasoning model.
export INSTRUCT_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
export INSTRUCT_MODEL_ALIAS="qwen35"

export REASONING_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
export REASONING_MODEL_ALIAS="qwen35"
export PORT=8082

export AQG_INFIX="uidtts"
export TAU=0.05
export AJT_PROMPT_FILE="$PROJECT_DIR/templates/ajt.txt"

###############################################################################
# Derived paths
###############################################################################

export BUNDLE_DIR="$OUTPUT_DIR/bundle"
export TEMPORAL_ANSWERS_DIR="$OUTPUT_DIR/temporal/answers"
export COMPARISON_DIR="$BUNDLE_DIR/$AQG_INFIX/comparison_results"

ASSIGN_TIMESTAMPS_SCRIPT="$PROJECT_DIR/assign_joint_timestamps.py"

STEP6_LOG="$LLAMA_LOG_DIR/step6_assign_joint_timestamps.log"
STEP6_RETRY_LOG_DIR="$LLAMA_LOG_DIR/step6_retries"

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
        gzip -cd "$file" | awk 'NF { found=1 } END { exit found ? 0 : 1 }'
    else
        awk 'NF { found=1 } END { exit found ? 0 : 1 }' "$file"
    fi
}

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_SCRIPT_DIR/file_selection_utils.sh"

strip_known_suffixes() {
    local name="$1"
    name="${name%.txt.gz}"
    name="${name%.txt}"
    name="${name%.txt}"
    echo "$name"
}

find_case_report() {
    local base="$1"
    local cleaned
    cleaned="$(strip_known_suffixes "$base")"

    local candidates=(
        "$BUNDLE_DIR/${base}.txt"
        "$BUNDLE_DIR/${base}.txt.txt"
        "$BUNDLE_DIR/${base}.txt.gz"
        "$BUNDLE_DIR/${cleaned}.txt"
        "$BUNDLE_DIR/${cleaned}.txt.txt"
        "$BUNDLE_DIR/${cleaned}.txt.gz"
    )

    local file
    for file in "${candidates[@]}"; do
        if [[ -e "$file" ]]; then
            echo "$file"
            return 0
        fi
    done

    return 1
}

find_mention_table() {
    local base="$1"
    local cleaned
    cleaned="$(strip_known_suffixes "$base")"

    local candidates=(
        "$TEMPORAL_ANSWERS_DIR/${base}_extracted.bsv.gz"
        "$TEMPORAL_ANSWERS_DIR/${cleaned}_extracted.bsv.gz"
    )

    local file
    for file in "${candidates[@]}"; do
        if [[ -e "$file" ]]; then
            echo "$file"
            return 0
        fi
    done

    return 1
}

validate_step5_and_step4_inputs() {
    shopt -s nullglob

    local joined_files=("$COMPARISON_DIR"/*_joined.csv)
    filter_selected_files_in_place joined_files
    local joined_file
    local base
    local case_report
    local mention_table
    local count=0

    [[ "${#joined_files[@]}" -gt 0 ]] || die "No Step 5 joined CSV files found in $COMPARISON_DIR"

    for joined_file in "${joined_files[@]}"; do
        has_nonblank_content "$joined_file" || die "Joined CSV is empty: $joined_file"

        base="$(basename "$joined_file" _joined.csv)"
        case_report="$(find_case_report "$base")" || die "Missing bundle case report for $base"
        mention_table="$(find_mention_table "$base")" || die "Missing Step 4 mention table for $base"

        has_nonblank_content "$case_report" || die "Bundle case report is empty: $case_report"
        has_nonblank_content "$mention_table" || die "Mention table is empty: $mention_table"

        count=$((count + 1))
    done

    log "Validated Step 6 inputs for $count cases"
}

cleanup_server() {
    if [[ -n "${LLAMA_SERVER_PID:-}" ]]; then
        kill_server "$LLAMA_SERVER_PID" "$PORT" || true
        unset LLAMA_SERVER_PID
    fi
}

###############################################################################
# Preflight
###############################################################################

mkdir -p "$LLAMA_LOG_DIR" "$STEP6_RETRY_LOG_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$BUNDLE_DIR" "Step 5 bundle directory"
check_path "$TEMPORAL_ANSWERS_DIR" "Step 4 temporal answers directory"
check_path "$COMPARISON_DIR" "Step 5 comparison directory"
check_path "$NOTE_LINK_PATH" "note link/admission file"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$REASONING_MODEL_PATH" "reasoning model"
check_path "$AJT_PROMPT_FILE" "final timeline prompt template"
check_path "$ASSIGN_TIMESTAMPS_SCRIPT" "assign_joint_timestamps.py"
check_path "$PROJECT_DIR/call_lcpp.py" "call_lcpp.py"

init_file_selection
export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"

source "$SERVER_UTILS"

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Output directory: $OUTPUT_DIR"
log_file_selection
log "Comparison directory: $COMPARISON_DIR"
log "Using python: $(which python)"
log "Python version: $(python --version)"
log "Instruct model path: $INSTRUCT_MODEL_PATH"
log "Instruct model alias: $INSTRUCT_MODEL_ALIAS"
log "Reasoning model path: $REASONING_MODEL_PATH"
log "Reasoning model alias: $REASONING_MODEL_ALIAS"
log "Tau: $TAU"

validate_step5_and_step4_inputs

###############################################################################
# Step 6: final joint timestamp revision
###############################################################################

log "Starting llama-server for Step 6"
start_server "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS" "$PORT"
trap cleanup_server EXIT

log "Step 6: generating final timeline JSON files"
: > "$STEP6_LOG"

shopt -s nullglob
joined_files=("$COMPARISON_DIR"/*_joined.csv)
filter_selected_files_in_place joined_files
timeline_count=0

for joined_file in "${joined_files[@]}"; do
    base="$(basename "$joined_file" _joined.csv)"
    case_report="$(find_case_report "$base")"
    mention_table="$(find_mention_table "$base")"
    timeline_json="$COMPARISON_DIR/${base}_timelines.json"

    log "Assigning final timestamps for $base" | tee -a "$STEP6_LOG"

    python "$ASSIGN_TIMESTAMPS_SCRIPT" \
        --anchor-results "$joined_file" \
        --case-report "$case_report" \
        --mention-table "$mention_table" \
        --output "$timeline_json" \
        --log-dir "$STEP6_RETRY_LOG_DIR" \
        --tau "$TAU" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --template "$AJT_PROMPT_FILE" \
        --note-link "$NOTE_LINK_PATH" \
        --endpoint-type openai \
        --overwrite \
        >> "$STEP6_LOG" 2>&1

    has_nonblank_content "$timeline_json" || die "Timeline JSON is empty: $timeline_json"
    timeline_count=$((timeline_count + 1))
done

cleanup_server
trap - EXIT

log "Step 6 complete. Created $timeline_count timeline JSON files"

###############################################################################
# Summary
###############################################################################

log "Done."
log "Timeline JSON outputs: $COMPARISON_DIR/*_timelines.json"
log "Logs:"
log "  llama-server: $LLAMA_LOG_DIR/server_${PORT}.log"
log "  step 6: $STEP6_LOG"
log "  retries: $STEP6_RETRY_LOG_DIR"
