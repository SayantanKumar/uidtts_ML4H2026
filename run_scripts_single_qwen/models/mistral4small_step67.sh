#!/bin/bash
#SBATCH --job-name=mistral4small_step67
#SBATCH --time=2-00:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=sayantan.kumar@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step67.sh

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
export OUTPUT_DIR="$PROJECT_DIR/outputs/mistral4small/"

# Run either every completed upstream case or only the hard-coded subset below.
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

# Model roles. Both point to Qwen3.5 for now; change either later if you want
# separate instruct/reasoning models. Step 6 uses the reasoning model.
export INSTRUCT_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Mistral-Small-4-119B-2603-UD-Q6_K-00001-of-00003.gguf"
export INSTRUCT_MODEL_ALIAS="mistral4small"

export REASONING_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Mistral-Small-4-119B-2603-UD-Q6_K-00001-of-00003.gguf"
export REASONING_MODEL_ALIAS="mistral4small"
export PORT=8082

export AQG_INFIX="uidtts"
export TAU=0.05
export AJT_PROMPT_FILE="$PROJECT_DIR/templates/ajt.txt"

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
export TIMELINES_DIR="$OUTPUT_DIR/timelines"
export CHARPOS_DIR="$BUNDLE_DIR/charpos"
export CHARPOS_UNREVISED_DIR="$BUNDLE_DIR/charpos_unrevised"
export LOG_DIR="$OUTPUT_DIR/logs"

ASSIGN_TIMESTAMPS_SCRIPT="$PROJECT_DIR/assign_joint_timestamps.py"
EXTRACT_TIMELINES_SCRIPT="$PROJECT_DIR/extract_timelines.py"
GET_CHARPOS_SCRIPT="$PROJECT_DIR/get_tag_charpos_table.py"

STEP6_LOG="$LLAMA_LOG_DIR/step6_assign_joint_timestamps.log"
STEP6_RETRY_LOG_DIR="$LLAMA_LOG_DIR/step6_retries"
STEP6_FAILED_CASES="$LLAMA_LOG_DIR/step6_failed_cases.txt"
POST_LOG="$LOG_DIR/post_processing.log"

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

has_valid_timeline_json() {
    local file="$1"

    [[ -s "$file" ]] || return 1

    python - "$file" <<'PY' >/dev/null 2>&1
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

if not isinstance(data, dict):
    sys.exit(1)

timelines = data.get("timelines")
if not isinstance(timelines, list) or not timelines:
    sys.exit(1)

for item in timelines:
    if not isinstance(item, dict):
        sys.exit(1)
    table = item.get("data")
    if not isinstance(table, str) or len(table.strip()) < 20:
        sys.exit(1)
    lines = [line for line in table.splitlines() if line.strip()]
    if len(lines) < 2:
        sys.exit(1)
    if "|" not in lines[0]:
        sys.exit(1)

sys.exit(0)
PY
}

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

validate_step6_outputs() {
    shopt -s nullglob

    local timeline_jsons=("$COMPARISON_DIR"/*_timelines.json)
    filter_selected_files_in_place timeline_jsons
    local file
    local valid_count=0
    local invalid_count=0

    [[ "${#timeline_jsons[@]}" -gt 0 ]] || die "No Step 6 timeline JSON files found in $COMPARISON_DIR"

    for file in "${timeline_jsons[@]}"; do
        if has_valid_timeline_json "$file"; then
            valid_count=$((valid_count + 1))
        else
            log "WARNING: invalid or incomplete timeline JSON will be skipped by post-processing: $file"
            invalid_count=$((invalid_count + 1))
        fi
    done

    [[ "$valid_count" -gt 0 ]] || die "No valid Step 6 timeline JSON files found in $COMPARISON_DIR"

    log "Validated $valid_count Step 6 timeline JSON files; $invalid_count invalid/incomplete"
}

validate_bundle_reports() {
    shopt -s nullglob

    local bundle_reports=("$BUNDLE_DIR"/*.txt "$BUNDLE_DIR"/*.txt.txt)
    filter_selected_files_in_place bundle_reports
    local file
    local valid_count=0
    local invalid_count=0

    [[ "${#bundle_reports[@]}" -gt 0 ]] || die "No bundle reports found in $BUNDLE_DIR"

    for file in "${bundle_reports[@]}"; do
        if has_nonblank_content "$file"; then
            valid_count=$((valid_count + 1))
        else
            log "WARNING: empty bundle report will be ignored by post-processing: $file"
            invalid_count=$((invalid_count + 1))
        fi
    done

    [[ "$valid_count" -gt 0 ]] || die "No valid bundle report files found in $BUNDLE_DIR"

    log "Validated $valid_count bundle report files; $invalid_count empty/invalid"
}

validate_temporal_answers() {
    shopt -s nullglob

    local temporal_tables=("$TEMPORAL_ANSWERS_DIR"/*_extracted.bsv.gz)
    filter_selected_files_in_place temporal_tables
    local file
    local valid_count=0
    local invalid_count=0

    [[ "${#temporal_tables[@]}" -gt 0 ]] || die "No unrevised temporal tables found in $TEMPORAL_ANSWERS_DIR"

    for file in "${temporal_tables[@]}"; do
        if has_nonblank_content "$file"; then
            valid_count=$((valid_count + 1))
        else
            log "WARNING: empty unrevised temporal table will be ignored by post-processing: $file"
            invalid_count=$((invalid_count + 1))
        fi
    done

    [[ "$valid_count" -gt 0 ]] || die "No valid unrevised temporal tables found in $TEMPORAL_ANSWERS_DIR"

    log "Validated $valid_count unrevised temporal tables; $invalid_count empty/invalid"
}

validate_generated_timelines() {
    shopt -s nullglob

    local timeline_bsvs=("$TIMELINES_DIR"/*.bsv.gz)
    filter_selected_files_in_place timeline_bsvs
    local file
    local valid_count=0
    local invalid_count=0

    [[ "${#timeline_bsvs[@]}" -gt 0 ]] || die "No final timeline BSV files found in $TIMELINES_DIR"

    for file in "${timeline_bsvs[@]}"; do
        if has_nonblank_content "$file"; then
            valid_count=$((valid_count + 1))
        else
            log "WARNING: empty final timeline BSV will be ignored: $file"
            invalid_count=$((invalid_count + 1))
        fi
    done

    [[ "$valid_count" -gt 0 ]] || die "No valid final timeline BSV files found in $TIMELINES_DIR"

    log "Validated $valid_count final timeline BSV files; $invalid_count empty/invalid"
}

validate_charpos_outputs() {
    shopt -s nullglob

    local final_files=("$CHARPOS_DIR"/*.bsv)
    local unrevised_files=("$CHARPOS_UNREVISED_DIR"/*.bsv)
    filter_selected_files_in_place final_files
    filter_selected_files_in_place unrevised_files
    local file
    local final_valid_count=0
    local unrevised_valid_count=0
    local invalid_count=0

    [[ "${#final_files[@]}" -gt 0 ]] || die "No final charpos BSV files found in $CHARPOS_DIR"
    [[ "${#unrevised_files[@]}" -gt 0 ]] || die "No unrevised charpos BSV files found in $CHARPOS_UNREVISED_DIR"

    for file in "${final_files[@]}"; do
        if has_nonblank_content "$file"; then
            final_valid_count=$((final_valid_count + 1))
        else
            log "WARNING: empty final charpos output will be ignored: $file"
            invalid_count=$((invalid_count + 1))
        fi
    done

    for file in "${unrevised_files[@]}"; do
        if has_nonblank_content "$file"; then
            unrevised_valid_count=$((unrevised_valid_count + 1))
        else
            log "WARNING: empty unrevised charpos output will be ignored: $file"
            invalid_count=$((invalid_count + 1))
        fi
    done

    [[ "$final_valid_count" -gt 0 ]] || die "No valid final charpos BSV files found in $CHARPOS_DIR"
    [[ "$unrevised_valid_count" -gt 0 ]] || die "No valid unrevised charpos BSV files found in $CHARPOS_UNREVISED_DIR"

    log "Validated $final_valid_count final charpos files and $unrevised_valid_count unrevised charpos files; $invalid_count empty/invalid"
}

select_bundle_pattern() {
    shopt -s nullglob

    local bundle_input_dir="${EFFECTIVE_BUNDLE_DIR:-$BUNDLE_DIR}"
    local txttxt_files=("$bundle_input_dir"/*.txt.txt)
    local txt_files=("$bundle_input_dir"/*.txt)

    if [[ "${#txttxt_files[@]}" -gt 0 ]]; then
        echo "*.txt.txt"
    elif [[ "${#txt_files[@]}" -gt 0 ]]; then
        echo "*.txt"
    else
        return 1
    fi
}

rename_unrevised_charpos_outputs() {
    shopt -s nullglob

    local file
    for file in "$CHARPOS_UNREVISED_DIR"/*_extracted.bsv; do
        local target="${file%_extracted.bsv}_1.bsv"
        if [[ -e "$target" ]]; then
            log "Skipping rename for $file; target already exists"
            continue
        fi
        mv "$file" "$target"
    done

    for file in "$CHARPOS_UNREVISED_DIR"/*_timeline_extracted.bsv; do
        local target="${file%_timeline_extracted.bsv}_1.bsv"
        if [[ -e "$target" ]]; then
            log "Skipping rename for $file; target already exists"
            continue
        fi
        mv "$file" "$target"
    done
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

mkdir -p "$LLAMA_LOG_DIR" "$STEP6_RETRY_LOG_DIR" "$TIMELINES_DIR" "$CHARPOS_DIR" "$CHARPOS_UNREVISED_DIR" "$LOG_DIR"

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
check_path "$EXTRACT_TIMELINES_SCRIPT" "extract_timelines.py"
check_path "$GET_CHARPOS_SCRIPT" "get_tag_charpos_table.py"
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
log "Step 6 max total candidates: $STEP6_MAX_TOTAL_CANDIDATES"
log "Step 6 top-k per anchor: $STEP6_TOP_K"
log "Step 6 requested timelines: $STEP6_NUM_TIMELINES"
log "Step 6 max candidate field chars: $STEP6_MAX_CANDIDATE_FIELD_CHARS"
log "Step 6 max case report chars: $STEP6_MAX_CASE_REPORT_CHARS"
log "Step 6 case context chars: $STEP6_CASE_CONTEXT_CHARS"

validate_step5_and_step4_inputs

###############################################################################
# Step 6: final joint timestamp revision
###############################################################################

: > "$STEP6_LOG"
: > "$STEP6_FAILED_CASES"

shopt -s nullglob
joined_files=("$COMPARISON_DIR"/*_joined.csv)
filter_selected_files_in_place joined_files
timeline_count=0
skipped_timeline_count=0
pending_timeline_count=0
failed_timeline_count=0

for joined_file in "${joined_files[@]}"; do
    base="$(basename "$joined_file" _joined.csv)"
    timeline_json="$COMPARISON_DIR/${base}_timelines.json"

    if has_valid_timeline_json "$timeline_json"; then
        skipped_timeline_count=$((skipped_timeline_count + 1))
    else
        pending_timeline_count=$((pending_timeline_count + 1))
    fi
done

if [[ "$pending_timeline_count" -gt 0 ]]; then
    log "Starting llama-server for Step 6"
    start_server "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS" "$PORT"
    trap cleanup_server EXIT

    log "Step 6: generating $pending_timeline_count missing/incomplete timeline JSON files"
else
    log "Step 6: all $skipped_timeline_count timeline JSON files already exist; skipping LLM generation"
fi

for joined_file in "${joined_files[@]}"; do
    base="$(basename "$joined_file" _joined.csv)"
    timeline_json="$COMPARISON_DIR/${base}_timelines.json"

    if has_valid_timeline_json "$timeline_json"; then
        log "Skipping final timestamp assignment for $base; timeline JSON already exists" | tee -a "$STEP6_LOG"
        continue
    fi

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
        log "WARNING: assign_joint_timestamps.py failed for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|python_failed|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi

    if has_valid_timeline_json "$timeline_json"; then
        timeline_count=$((timeline_count + 1))
    else
        log "WARNING: timeline JSON is missing, empty, or invalid for $base; continuing" | tee -a "$STEP6_LOG"
        echo "$base|invalid_or_empty_timeline_json|$timeline_json" >> "$STEP6_FAILED_CASES"
        failed_timeline_count=$((failed_timeline_count + 1))
        continue
    fi
done

if [[ "$pending_timeline_count" -gt 0 ]]; then
    cleanup_server
    trap - EXIT
fi

log "Step 6 complete. Created $timeline_count timeline JSON files; skipped $skipped_timeline_count existing timeline JSON files; failed $failed_timeline_count cases"
if [[ "$failed_timeline_count" -gt 0 ]]; then
    log "Step 6 failed-case list: $STEP6_FAILED_CASES"
fi

###############################################################################
# Post-processing
###############################################################################

validate_step6_outputs
validate_bundle_reports
validate_temporal_answers

BUNDLE_PATTERN="$(select_bundle_pattern)" || die "No bundle report pattern found in $BUNDLE_DIR"
log "Using bundle report pattern for charpos extraction: $BUNDLE_PATTERN"

log "Post-processing: extracting final timeline BSV files"
: > "$POST_LOG"

EFFECTIVE_COMPARISON_DIR="$COMPARISON_DIR"
EFFECTIVE_BUNDLE_DIR="$BUNDLE_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/step67_comparison_${SLURM_JOB_ID:-manual}_$$" \
        "$COMPARISON_DIR/{}_timelines.json"
    EFFECTIVE_COMPARISON_DIR="$STAGED_SELECTED_DIR"

    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/step67_bundle_${SLURM_JOB_ID:-manual}_$$" \
        "$BUNDLE_DIR/{}.txt" \
        "$BUNDLE_DIR/{}.txt.txt"
    EFFECTIVE_BUNDLE_DIR="$STAGED_SELECTED_DIR"
fi

if ! python "$EXTRACT_TIMELINES_SCRIPT" \
    --directory "$EFFECTIVE_COMPARISON_DIR" \
    --pattern "*_timelines.json" \
    --output-dir "$TIMELINES_DIR/" \
    --verbose \
    >> "$POST_LOG" 2>&1; then
    log "WARNING: extract_timelines.py returned a nonzero status; continuing with any final timeline files that were produced"
fi

validate_generated_timelines

log "Post-processing: extracting final character-position tables"

if ! python "$GET_CHARPOS_SCRIPT" \
    --input-dir "$EFFECTIVE_BUNDLE_DIR/" \
    --pattern "$BUNDLE_PATTERN" \
    --output-dir "$CHARPOS_DIR" \
    --join-mention \
    --timeline-dir "$TIMELINES_DIR" \
    >> "$POST_LOG" 2>&1; then
    log "WARNING: get_tag_charpos_table.py returned a nonzero status for final timelines; continuing with any charpos files that were produced"
fi

log "Post-processing: extracting unrevised character-position tables"

if ! python "$GET_CHARPOS_SCRIPT" \
    --input-dir "$EFFECTIVE_BUNDLE_DIR/" \
    --pattern "$BUNDLE_PATTERN" \
    --output-dir "$CHARPOS_UNREVISED_DIR" \
    --join-mention \
    --timeline-dir "$TEMPORAL_ANSWERS_DIR" \
    >> "$POST_LOG" 2>&1; then
    log "WARNING: get_tag_charpos_table.py returned a nonzero status for unrevised timelines; continuing with any charpos files that were produced"
fi

rename_unrevised_charpos_outputs
validate_charpos_outputs

###############################################################################
# Summary
###############################################################################

log "Done."
log "Timeline JSON outputs: $COMPARISON_DIR/*_timelines.json"
log "Final timelines: $TIMELINES_DIR"
log "Final charpos tables: $CHARPOS_DIR"
log "Unrevised charpos tables: $CHARPOS_UNREVISED_DIR"
log "Logs:"
log "  llama-server: $LLAMA_LOG_DIR/server_${PORT}.log"
log "  step 6: $STEP6_LOG"
log "  retries: $STEP6_RETRY_LOG_DIR"
log "  post-processing: $POST_LOG"



# OUT=/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/outputs/mistral4small

# watch -n 30 "
# echo 'Step 6 timeline JSONs:'; find $OUT/bundle/uidtts/comparison_results -name '*_timelines.json' 2>/dev/null | wc -l
# echo 'Final timeline BSVs:'; find $OUT/timelines -name '*.bsv.gz' 2>/dev/null | wc -l
# echo 'Final charpos:'; find $OUT/bundle/charpos -name '*.bsv' 2>/dev/null | wc -l
# echo 'Unrevised charpos:'; find $OUT/bundle/charpos_unrevised -name '*.bsv' 2>/dev/null | wc -l
# "
