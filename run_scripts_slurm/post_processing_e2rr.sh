#!/bin/bash
#SBATCH --job-name=uid_post
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch post_processing.sh

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
# Edit these paths/settings
###############################################################################

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/outputs/${MODEL_OUTPUT_NAME:-qwen35}}"

# Run either every completed Step 6 case or only the hard-coded subset below.
# Use the original input filenames or bare case ids.
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

export AQG_INFIX="uidtts"

###############################################################################
# Derived paths
###############################################################################

export BUNDLE_DIR="$OUTPUT_DIR/bundle"
export TEMPORAL_ANSWERS_DIR="$OUTPUT_DIR/temporal/answers"
export TIMELINES_DIR="$OUTPUT_DIR/timelines_e2rr"
export CHARPOS_DIR="$BUNDLE_DIR/charpos"
export CHARPOS_UNREVISED_DIR="$BUNDLE_DIR/charpos_unrevised"
export COMPARISON_DIR="$BUNDLE_DIR/$AQG_INFIX/e2rr_comparison_results"
export LOG_DIR="$OUTPUT_DIR/logs"

EXTRACT_TIMELINES_SCRIPT="$PROJECT_DIR/extract_timelines.py"
GET_CHARPOS_SCRIPT="$PROJECT_DIR/get_tag_charpos_table.py"

POST_LOG="$LOG_DIR/post_processing_e2rr.log"

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

validate_step6_outputs() {
    shopt -s nullglob

    local timeline_jsons=("$COMPARISON_DIR"/*_timelines.json)
    filter_selected_files_in_place timeline_jsons
    local file
    local valid_count=0
    local invalid_count=0

    [[ "${#timeline_jsons[@]}" -gt 0 ]] || die "No Step 6 timeline JSON files found in $COMPARISON_DIR"

    for file in "${timeline_jsons[@]}"; do
        if has_nonblank_content "$file"; then
            valid_count=$((valid_count + 1))
        else
            log "WARNING: empty timeline JSON will be ignored by post-processing: $file"
            invalid_count=$((invalid_count + 1))
        fi
    done

    [[ "$valid_count" -gt 0 ]] || die "No valid Step 6 timeline JSON files found in $COMPARISON_DIR"

    log "Validated $valid_count Step 6 timeline JSON files; $invalid_count empty/invalid"
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

###############################################################################
# Preflight
###############################################################################

mkdir -p "$TIMELINES_DIR" "$CHARPOS_DIR" "$CHARPOS_UNREVISED_DIR" "$LOG_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$BUNDLE_DIR" "bundle directory"
check_path "$TEMPORAL_ANSWERS_DIR" "Step 4 temporal answers directory"
check_path "$COMPARISON_DIR" "Step 6 comparison directory"
check_path "$EXTRACT_TIMELINES_SCRIPT" "extract_timelines.py"
check_path "$GET_CHARPOS_SCRIPT" "get_tag_charpos_table.py"

init_file_selection
export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Output directory: $OUTPUT_DIR"
log_file_selection
log "Using python: $(which python)"
log "Python version: $(python --version)"

validate_step6_outputs
validate_bundle_reports
validate_temporal_answers

EFFECTIVE_COMPARISON_DIR="$COMPARISON_DIR"
EFFECTIVE_BUNDLE_DIR="$BUNDLE_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/post_comparison_${SLURM_JOB_ID:-manual}_$$" \
        "$COMPARISON_DIR/{}_timelines.json"
    EFFECTIVE_COMPARISON_DIR="$STAGED_SELECTED_DIR"

    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/post_bundle_${SLURM_JOB_ID:-manual}_$$" \
        "$BUNDLE_DIR/{}.txt" \
        "$BUNDLE_DIR/{}.txt.txt"
    EFFECTIVE_BUNDLE_DIR="$STAGED_SELECTED_DIR"
fi

BUNDLE_PATTERN="$(select_bundle_pattern)" || die "No bundle report pattern found in $BUNDLE_DIR"
log "Using bundle report pattern for charpos extraction: $BUNDLE_PATTERN"

###############################################################################
# Post-processing 1: extract final timeline BSV files
###############################################################################

log "Extracting final timeline BSV files"
: > "$POST_LOG"

if ! python "$EXTRACT_TIMELINES_SCRIPT" \
    --directory "$EFFECTIVE_COMPARISON_DIR" \
    --pattern "*_timelines.json" \
    --output-dir "$TIMELINES_DIR/" \
    --verbose \
    >> "$POST_LOG" 2>&1; then
    log "WARNING: extract_timelines.py returned a nonzero status; continuing with any final timeline files that were produced"
fi

validate_generated_timelines

###############################################################################
# Post-processing 2: character positions joined to final timelines
###############################################################################

log "Extracting final character-position tables"

if ! python "$GET_CHARPOS_SCRIPT" \
    --input-dir "$EFFECTIVE_BUNDLE_DIR/" \
    --pattern "$BUNDLE_PATTERN" \
    --output-dir "$CHARPOS_DIR" \
    --join-mention \
    --timeline-dir "$TIMELINES_DIR" \
    >> "$POST_LOG" 2>&1; then
    log "WARNING: get_tag_charpos_table.py returned a nonzero status for final timelines; continuing with any charpos files that were produced"
fi

###############################################################################
# Post-processing 3: character positions joined to unrevised Step 4 timelines
###############################################################################

log "Extracting unrevised character-position tables"

if ! python "$GET_CHARPOS_SCRIPT" \
    --input-dir "$EFFECTIVE_BUNDLE_DIR/" \
    --pattern "$BUNDLE_PATTERN" \
    --output-dir "$CHARPOS_UNREVISED_DIR" \
    --join-mention \
    --timeline-dir "$TEMPORAL_ANSWERS_DIR" \
    >> "$POST_LOG" 2>&1; then
    log "WARNING: get_tag_charpos_table.py returned a nonzero status for unrevised timelines; continuing with any charpos files that were produced"
fi

# Match the original pipeline's naming convention for unrevised charpos files.
shopt -s nullglob
for file in "$CHARPOS_UNREVISED_DIR"/*_extracted.bsv; do
    mv "$file" "${file%_extracted.bsv}_1.bsv"
done

validate_charpos_outputs

###############################################################################
# Summary
###############################################################################

log "Done."
log "Final timelines: $TIMELINES_DIR"
log "Final charpos tables: $CHARPOS_DIR"
log "Unrevised charpos tables: $CHARPOS_UNREVISED_DIR"
log "Log: $POST_LOG"
