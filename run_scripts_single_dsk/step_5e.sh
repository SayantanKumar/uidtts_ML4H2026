#!/bin/bash
#SBATCH --job-name=uid_step5e
#SBATCH --time=1-00:00:00
#SBATCH --mem=128g
#SBATCH --output=sbatch_output/slurm_%j.out
#SBATCH --error=sbatch_error/slurm_%j.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step5e_flatten_join.sh

set -eo pipefail

###############################################################################
# Environment
###############################################################################

module load CUDA/12.8.1 gcc cmake

module purge
module load CUDA/12.8.1 gcc/11.3.0 cmake/3.30.2

source /data/grimalditersans/myconda
conda activate /data/grimalditersans/mymamba/envs/TTA

###############################################################################
# Paths and settings
###############################################################################

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/outputs/${MODEL_OUTPUT_NAME:-qwen35}}"

# Original per-patient tabular data.
export TABULAR_DIR="${TABULAR_DIR:-/data/grimalditersans/mimic_uid_data/i2m4_n30_tabular/}"

# note_id/hadm_id/admittime/dischtime mappings.
export NOTE_LINK_PATH="${NOTE_LINK_PATH:-/data/grimalditersans/mimic_uid_data/note_link_i2m4.csv}"

export BUNDLE_DIR="$OUTPUT_DIR/bundle"

# Step 5C input.
export RAG_INFIX="uid_infix"
export RAG_OUTPUT_ROOT="$BUNDLE_DIR/tabular"
export RAG_DIR="$RAG_OUTPUT_ROOT/$RAG_INFIX"

# Step 5D input and Step 5E output.
export AQG_INFIX="uidtts"
export AQG_DIR="$BUNDLE_DIR/$AQG_INFIX"
export COMPARISON_DIR="$AQG_DIR/comparison_results"

export VIEW_SCORES_SCRIPT="$PROJECT_DIR/view_match_scores.py"
export JOIN_ANCHOR_SCRIPT="$PROJECT_DIR/join_anchor_with_source_timed.py"

export LOG_DIR="$OUTPUT_DIR/logs"
export STEP5_JOIN_LOG="$LOG_DIR/step5e_flatten_join.log"

###############################################################################
# File selection
###############################################################################

export FILE_SELECTION_MODE="subset"  # all|subset
SELECTED_INPUT_FILES=(
    "18533644"
    "11649157"
    "19471635"
)

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

case_base() {
    local name
    name="$(basename "$1")"
    name="${name%.gz}"
    name="${name%.json}"
    name="${name%.csv}"
    name="${name%.txt}"
    name="${name%_results}"
    name="${name%_modified}"
    echo "$name"
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

find_mapping_file() {
    shopt -s nullglob

    local patient_id="$1"
    local exact="$RAG_DIR/${RAG_INFIX}_${patient_id}.csv/summary_mapping.json"
    local matches

    if [[ -e "$exact" ]]; then
        echo "$exact"
        return 0
    fi

    matches=(
        "$RAG_DIR"/"${RAG_INFIX}_${patient_id}"*.csv/summary_mapping.json
    )

    if [[ "${#matches[@]}" -gt 0 ]]; then
        echo "${matches[0]}"
        return 0
    fi

    return 1
}

find_line_level_file() {
    shopt -s nullglob

    local patient_id="$1"
    local exact="$TABULAR_DIR/${patient_id}.csv.gz"
    local matches

    if [[ -e "$exact" ]]; then
        echo "$exact"
        return 0
    fi

    matches=("$TABULAR_DIR"/"${patient_id}"*.csv.gz)

    if [[ "${#matches[@]}" -gt 0 ]]; then
        echo "${matches[0]}"
        return 0
    fi

    return 1
}

subject_id_from_name() {
    local raw_name="$1"
    local base
    local token

    base="$(case_base "$raw_name")"

    if [[ -e "$TABULAR_DIR/${base}.csv.gz" ]]; then
        echo "$base"
        return 0
    fi

    while IFS= read -r token; do
        [[ -n "$token" ]] || continue

        if [[ -e "$TABULAR_DIR/${token}.csv.gz" ]]; then
            echo "$token"
            return 0
        fi
    done < <(printf "%s\n" "$base" | grep -Eo '[0-9]{5,}' || true)

    return 1
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

mkdir -p "$COMPARISON_DIR" "$LOG_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$COMPARISON_DIR" "Step 5D comparison directory"
check_path "$RAG_DIR" "Step 5C RAG directory"
check_path "$TABULAR_DIR" "tabular input directory"
check_path "$NOTE_LINK_PATH" "note/admission mapping file"
check_path "$VIEW_SCORES_SCRIPT" "view_match_scores.py"
check_path "$JOIN_ANCHOR_SCRIPT" "join_anchor_with_source_timed.py"

has_nonblank_content "$NOTE_LINK_PATH" ||
    die "Note/admission mapping file is empty: $NOTE_LINK_PATH"

shopt -s nullglob

result_files=("$COMPARISON_DIR"/*_results.json)
filter_selected_files_in_place result_files

[[ "${#result_files[@]}" -gt 0 ]] ||
    die "No selected Step 5D result files found in $COMPARISON_DIR"

for results_file in "${result_files[@]}"; do
    has_nonblank_content "$results_file" ||
        die "Step 5D result is empty: $results_file"

    python -m json.tool "$results_file" >/dev/null ||
        die "Step 5D result is invalid JSON: $results_file"
done

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Step 5D input directory: $COMPARISON_DIR"
log "Step 5C RAG directory: $RAG_DIR"
log "Tabular source directory: $TABULAR_DIR"
log "Step 5E output directory: $COMPARISON_DIR"
log_file_selection
log "Step 5D result files: ${#result_files[@]}"
log "Using Python: $(which python)"
log "Python version: $(python --version)"

###############################################################################
# Step 5E: flatten scores and join to source rows
###############################################################################

: > "$STEP5_JOIN_LOG"

created_count=0
skipped_count=0
failed_count=0

for results_file in "${result_files[@]}"; do
    base_name="$(basename "$results_file" _results.json)"

    if ! patient_id="$(subject_id_from_name "$base_name")"; then
        log "WARNING: could not resolve subject ID for $base_name" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    summary_csv="$COMPARISON_DIR/${patient_id}_summary.csv"
    joined_csv="$COMPARISON_DIR/${patient_id}_joined.csv"

    if has_nonblank_content "$joined_csv"; then
        log "Skipping $base_name; joined CSV already exists: $joined_csv" |
            tee -a "$STEP5_JOIN_LOG"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    log "Flattening Step 5D scores for $base_name" |
        tee -a "$STEP5_JOIN_LOG"

    if ! python "$VIEW_SCORES_SCRIPT" \
        --results-file "$results_file" \
        --output "$summary_csv" \
        --sort-by score \
        --overwrite \
        >> "$STEP5_JOIN_LOG" 2>&1; then
        log "WARNING: score flattening failed for $base_name" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! has_nonblank_content "$summary_csv"; then
        log "WARNING: summary CSV is empty: $summary_csv" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! mapping_file="$(find_mapping_file "$patient_id")"; then
        log "WARNING: missing summary_mapping.json for patient $patient_id" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! has_nonblank_content "$mapping_file"; then
        log "WARNING: summary mapping is empty: $mapping_file" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! python -m json.tool "$mapping_file" >/dev/null; then
        log "WARNING: summary mapping is invalid JSON: $mapping_file" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! line_level_file="$(find_line_level_file "$patient_id")"; then
        log "WARNING: missing source tabular CSV for patient $patient_id" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! has_nonblank_content "$line_level_file"; then
        log "WARNING: source tabular CSV is empty or invalid: $line_level_file" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    log "Joining $base_name back to source tabular rows" |
        tee -a "$STEP5_JOIN_LOG"

    if ! python "$JOIN_ANCHOR_SCRIPT" \
        --anchor-results "$summary_csv" \
        --summary-mapping "$mapping_file" \
        --line-level-data "$line_level_file" \
        --note-adms "$NOTE_LINK_PATH" \
        --output "$joined_csv" \
        --overwrite \
        >> "$STEP5_JOIN_LOG" 2>&1; then
        log "WARNING: source-row join failed for $base_name" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    if ! has_nonblank_content "$joined_csv"; then
        log "WARNING: joined CSV is empty: $joined_csv" |
            tee -a "$STEP5_JOIN_LOG"
        failed_count=$((failed_count + 1))
        continue
    fi

    created_count=$((created_count + 1))
done

###############################################################################
# Final validation
###############################################################################

[[ $((created_count + skipped_count)) -gt 0 ]] ||
    die "Step 5E did not create or find any valid joined CSV files"

if [[ "$failed_count" -gt 0 ]]; then
    die "Step 5E failed for $failed_count case(s); see $STEP5_JOIN_LOG"
fi

log "Step 5E complete"
log "Created joined CSVs: $created_count"
log "Previously completed CSVs: $skipped_count"
log "Final outputs: $COMPARISON_DIR"
log "Log: $STEP5_JOIN_LOG"
