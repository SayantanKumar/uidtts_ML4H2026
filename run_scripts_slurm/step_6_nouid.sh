#!/bin/bash
#SBATCH --job-name=uid_step6
#SBATCH --time=2-00:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step6_final_timeline.sh

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

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export SERVER_UTILS="$PROJECT_DIR/llm_server_utils.sh"

export OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/outputs/${MODEL_OUTPUT_NAME:-qwen35}}"

# Run either every completed Step 5 case or only the hard-coded subset below.
# Use the original input filenames or bare case ids.
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

# File with note_id/hadm_id/admittime/dischtime mappings.
# Must match the file used in Step 5.
export NOTE_LINK_PATH="${NOTE_LINK_PATH:-/data/grimalditersans/mimic_uid_data/note_link_i2m4.csv}"

export LLAMA_SERVER="/data/grimalditersans/llama.cpp/build-cuda-gcc11/bin/llama-server"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"

export CUDA_DEVICES="${LLM_CUDA_DEVICES:-0,1}"
export LLAMA_TIMEOUT=1200

# Model roles. Wrapper-provided values take precedence over these Qwen defaults.
# Step 6 uses the reasoning model.
export INSTRUCT_MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export INSTRUCT_MODEL_ALIAS="${INSTRUCT_MODEL_ALIAS:-qwen35}"

export REASONING_MODEL_PATH="${REASONING_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export REASONING_MODEL_ALIAS="${REASONING_MODEL_ALIAS:-qwen35}"
export PORT=8082

export AQG_INFIX="uidtts"
export TAU=0.05
export AJT_PROMPT_FILE="$PROJECT_DIR/templates/ajt_nouid.txt"

# Step 6 prompt-size controls. The failed cases were hitting the llama-server
# context limit, so keep the final-timeline prompt compact while still asking
# for three alternate timelines.
export STEP6_MAX_TOTAL_CANDIDATES=100
export STEP6_TOP_K=3
export STEP6_NUM_TIMELINES=3
export STEP6_MAX_CANDIDATE_FIELD_CHARS=160
export STEP6_MAX_CASE_REPORT_CHARS=30000
export STEP6_CASE_CONTEXT_CHARS=220

###############################################################################
# Derived paths
###############################################################################

export BUNDLE_DIR="$OUTPUT_DIR/bundle"
export TEMPORAL_ANSWERS_DIR="$OUTPUT_DIR/temporal/answers"
export COMPARISON_DIR="$BUNDLE_DIR/$AQG_INFIX/comparison_results"
export TIMELINE_OUTPUT_DIR="$BUNDLE_DIR/$AQG_INFIX/nouid_comparison_results"

ASSIGN_TIMESTAMPS_SCRIPT="$PROJECT_DIR/assign_joint_timestamp_nouid.py"

STEP6_LOG="$LLAMA_LOG_DIR/assign_joint_timestamp_nouid.log"
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

SELECTION_UTILS="$PROJECT_DIR/run_scripts/file_selection_utils.sh"
if [[ ! -f "$SELECTION_UTILS" ]]; then
    RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELECTION_UTILS="$RUN_SCRIPT_DIR/file_selection_utils.sh"
fi
source "$SELECTION_UTILS"

strip_known_suffixes() {
    local name="$1"
    name="${name%.txt.gz}"
    name="${name%.txt}"
    name="${name%.txt}"
    echo "$name"
}

numeric_tokens_from_name() {
    printf "%s\n" "$1" | grep -Eo '[0-9]{5,}' || true
}

path_matches_subject_id() {
    local file="$1"
    local subject_id="$2"
    local name
    name="$(basename "$file")"
    [[ "$name" =~ (^|[^0-9])${subject_id}([^0-9]|$) ]]
}

normalized_case_base() {
    local base="$1"
    if declare -F case_base >/dev/null 2>&1; then
        case_base "$base"
    elif declare -F strip_known_suffixes >/dev/null 2>&1; then
        strip_known_suffixes "$base"
    else
        base="$(basename "$base")"
        base="${base%.gz}"
        base="${base%.txt}"
        base="${base%.txt}"
        base="${base%_modified}"
        echo "$base"
    fi
}

find_case_report() {
    local base="$1"
    local cleaned
    cleaned="$(normalized_case_base "$base")"

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
        [[ -e "$file" ]] && echo "$file" && return 0
    done

    local token
    while IFS= read -r token; do
        [[ -n "$token" ]] || continue
        for file in "$BUNDLE_DIR"/*"$token"*.txt "$BUNDLE_DIR"/*"$token"*.txt.txt "$BUNDLE_DIR"/*"$token"*.txt.gz; do
            [[ -e "$file" ]] || continue
            path_matches_subject_id "$file" "$token" || continue
            echo "$file"
            return 0
        done
    done < <(numeric_tokens_from_name "$cleaned")

    return 1
}

find_mention_table() {
    local base="$1"
    local cleaned
    cleaned="$(normalized_case_base "$base")"

    local candidates=(
        "$TEMPORAL_ANSWERS_DIR/${base}_extracted.bsv.gz"
        "$TEMPORAL_ANSWERS_DIR/${cleaned}_extracted.bsv.gz"
    )

    local file
    for file in "${candidates[@]}"; do
        [[ -e "$file" ]] && echo "$file" && return 0
    done

    local token
    while IFS= read -r token; do
        [[ -n "$token" ]] || continue
        for file in "$TEMPORAL_ANSWERS_DIR"/*"$token"*"_extracted.bsv.gz"; do
            [[ -e "$file" ]] || continue
            path_matches_subject_id "$file" "$token" || continue
            echo "$file"
            return 0
        done
    done < <(numeric_tokens_from_name "$cleaned")

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
    local skipped_count=0

    [[ "${#joined_files[@]}" -gt 0 ]] || die "No Step 5 joined CSV files found in $COMPARISON_DIR"

    for joined_file in "${joined_files[@]}"; do
        base="$(basename "$joined_file" _joined.csv)"

        if ! has_nonblank_content "$joined_file"; then
            log "WARNING: Step 6 skipping $base; joined CSV is empty: $joined_file"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! case_report="$(find_case_report "$base")"; then
            log "WARNING: Step 6 skipping $base; missing bundle case report"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! mention_table="$(find_mention_table "$base")"; then
            log "WARNING: Step 6 skipping $base; missing Step 4 mention table"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! has_nonblank_content "$case_report"; then
            log "WARNING: Step 6 skipping $base; bundle case report is empty: $case_report"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! has_nonblank_content "$mention_table"; then
            log "WARNING: Step 6 skipping $base; mention table is empty: $mention_table"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        count=$((count + 1))
    done

    [[ "$count" -gt 0 ]] || die "No valid Step 6 inputs after checking Step 5 joined CSVs, bundle reports, and Step 4 mention tables"

    log "Validated Step 6 inputs for $count cases; skipped $skipped_count invalid/missing cases"
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

mkdir -p "$LLAMA_LOG_DIR" "$STEP6_RETRY_LOG_DIR" "$TIMELINE_OUTPUT_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$BUNDLE_DIR" "Step 5 bundle directory"
check_path "$TEMPORAL_ANSWERS_DIR" "Step 4 temporal answers directory"
check_path "$COMPARISON_DIR" "Step 5 comparison directory"
check_path "$NOTE_LINK_PATH" "note link/admission file"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$REASONING_MODEL_PATH" "reasoning model"
check_path "$AJT_PROMPT_FILE" "final timeline prompt template"
check_path "$ASSIGN_TIMESTAMPS_SCRIPT" "assign_joint_timestamp_nouid.py"
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
failed_timeline_count=0
STEP6_FAILED_CASES="$LLAMA_LOG_DIR/step6_failed_cases.txt"
: > "$STEP6_FAILED_CASES"

for joined_file in "${joined_files[@]}"; do
    base="$(basename "$joined_file" _joined.csv)"
    timeline_json="$TIMELINE_OUTPUT_DIR/${base}_timelines.json"

    if ! has_nonblank_content "$joined_file"; then
        log "WARNING: joined CSV is empty for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|empty_joined_csv|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi
    if ! case_report="$(find_case_report "$base")"; then
        log "WARNING: missing bundle case report for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|missing_case_report|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi
    if ! mention_table="$(find_mention_table "$base")"; then
        log "WARNING: missing Step 4 mention table for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|missing_mention_table|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi
    if ! has_nonblank_content "$case_report"; then
        log "WARNING: bundle case report is empty for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|empty_case_report|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi
    if ! has_nonblank_content "$mention_table"; then
        log "WARNING: mention table is empty for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|empty_mention_table|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi

    log "Assigning final timestamps for $base" | tee -a "$STEP6_LOG"

    if ! python "$ASSIGN_TIMESTAMPS_SCRIPT" \
        --anchor-results "$joined_file" \
        --case-report "$case_report" \
        --mention-table "$mention_table" \
        --output "$timeline_json" \
        --log-dir "$STEP6_RETRY_LOG_DIR" \
        --model-input-log-dir "$LLAMA_LOG_DIR/model_inputs_nouid" \
        --tau "$TAU" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --template "$AJT_PROMPT_FILE" \
        --note-link "$NOTE_LINK_PATH" \
        --endpoint-type openai \
        --max-total-candidates "$STEP6_MAX_TOTAL_CANDIDATES" \
        --top-k "$STEP6_TOP_K" \
        --num-timelines "$STEP6_NUM_TIMELINES" \
        --max-candidate-field-chars "$STEP6_MAX_CANDIDATE_FIELD_CHARS" \
        --max-case-report-chars "$STEP6_MAX_CASE_REPORT_CHARS" \
        --case-context-chars "$STEP6_CASE_CONTEXT_CHARS" \
        --overwrite \
        >> "$STEP6_LOG" 2>&1; then
        log "WARNING: assign_joint_timestamp_nouid.py failed for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|python_failed|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi

    if ! has_nonblank_content "$timeline_json"; then
        log "WARNING: timeline JSON is empty for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|empty_timeline_json|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi
    timeline_count=$((timeline_count + 1))
done

cleanup_server
trap - EXIT

[[ "$timeline_count" -gt 0 ]] || die "No timeline JSON files were created successfully"

log "Step 6 complete. Created $timeline_count timeline JSON files; failed $failed_timeline_count cases"
if [[ "$failed_timeline_count" -gt 0 ]]; then
    log "Step 6 failed-case list: $STEP6_FAILED_CASES"
fi

###############################################################################
# Summary
###############################################################################

log "Done."
log "Timeline JSON outputs: $TIMELINE_OUTPUT_DIR/*_timelines.json"
log "Logs:"
log "  llama-server: $LLAMA_LOG_DIR/server_${PORT}.log"
log "  step 6: $STEP6_LOG"
log "  retries: $STEP6_RETRY_LOG_DIR"
