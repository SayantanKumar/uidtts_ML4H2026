#!/bin/bash
#SBATCH --job-name=uid_post
#SBATCH --time=12:00:00
#SBATCH --mem=32g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:4
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=kumarsayantan94@gmail.com

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch post_processing.sh

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

###############################################################################
# Edit these paths/settings
###############################################################################

export PROJECT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection"
export OUTPUT_DIR="$PROJECT_DIR/outputs/qwen35/"

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
export TIMELINES_DIR="$OUTPUT_DIR/timelines"
export CHARPOS_DIR="$BUNDLE_DIR/charpos"
export CHARPOS_UNREVISED_DIR="$BUNDLE_DIR/charpos_unrevised"
export COMPARISON_DIR="$BUNDLE_DIR/$AQG_INFIX/comparison_results"
export LOG_DIR="$OUTPUT_DIR/logs"

EXTRACT_TIMELINES_SCRIPT="$PROJECT_DIR/extract_timelines.py"
GET_CHARPOS_SCRIPT="$PROJECT_DIR/get_tag_charpos_table.py"

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

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_SCRIPT_DIR/file_selection_utils.sh"

validate_step6_outputs() {
    shopt -s nullglob

    local timeline_jsons=("$COMPARISON_DIR"/*_timelines.json)
    filter_selected_files_in_place timeline_jsons
    local file

    [[ "${#timeline_jsons[@]}" -gt 0 ]] || die "No Step 6 timeline JSON files found in $COMPARISON_DIR"

    for file in "${timeline_jsons[@]}"; do
        has_nonblank_content "$file" || die "Timeline JSON is empty: $file"
    done

    log "Validated ${#timeline_jsons[@]} Step 6 timeline JSON files"
}

validate_bundle_reports() {
    shopt -s nullglob

    local bundle_reports=("$BUNDLE_DIR"/*.txt "$BUNDLE_DIR"/*.txt.txt)
    filter_selected_files_in_place bundle_reports
    local file

    [[ "${#bundle_reports[@]}" -gt 0 ]] || die "No bundle reports found in $BUNDLE_DIR"

    for file in "${bundle_reports[@]}"; do
        has_nonblank_content "$file" || die "Bundle report is empty: $file"
    done

    log "Validated ${#bundle_reports[@]} bundle report files"
}

validate_temporal_answers() {
    shopt -s nullglob

    local temporal_tables=("$TEMPORAL_ANSWERS_DIR"/*_extracted.bsv.gz)
    filter_selected_files_in_place temporal_tables
    local file

    [[ "${#temporal_tables[@]}" -gt 0 ]] || die "No unrevised temporal tables found in $TEMPORAL_ANSWERS_DIR"

    for file in "${temporal_tables[@]}"; do
        has_nonblank_content "$file" || die "Temporal answer table is empty: $file"
    done

    log "Validated ${#temporal_tables[@]} unrevised temporal tables"
}

validate_generated_timelines() {
    shopt -s nullglob

    local timeline_bsvs=("$TIMELINES_DIR"/*.bsv.gz)
    filter_selected_files_in_place timeline_bsvs
    local file

    [[ "${#timeline_bsvs[@]}" -gt 0 ]] || die "No final timeline BSV files found in $TIMELINES_DIR"

    for file in "${timeline_bsvs[@]}"; do
        has_nonblank_content "$file" || die "Final timeline BSV is empty: $file"
    done

    log "Validated ${#timeline_bsvs[@]} final timeline BSV files"
}

validate_charpos_outputs() {
    shopt -s nullglob

    local final_files=("$CHARPOS_DIR"/*.bsv)
    local unrevised_files=("$CHARPOS_UNREVISED_DIR"/*.bsv)
    filter_selected_files_in_place final_files
    filter_selected_files_in_place unrevised_files
    local file

    [[ "${#final_files[@]}" -gt 0 ]] || die "No final charpos BSV files found in $CHARPOS_DIR"
    [[ "${#unrevised_files[@]}" -gt 0 ]] || die "No unrevised charpos BSV files found in $CHARPOS_UNREVISED_DIR"

    for file in "${final_files[@]}" "${unrevised_files[@]}"; do
        has_nonblank_content "$file" || die "Charpos output is empty: $file"
    done

    log "Validated ${#final_files[@]} final charpos files and ${#unrevised_files[@]} unrevised charpos files"
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

python "$EXTRACT_TIMELINES_SCRIPT" \
    --directory "$EFFECTIVE_COMPARISON_DIR" \
    --pattern "*_timelines.json" \
    --output-dir "$TIMELINES_DIR/" \
    --verbose \
    >> "$POST_LOG" 2>&1

validate_generated_timelines

###############################################################################
# Post-processing 2: character positions joined to final timelines
###############################################################################

log "Extracting final character-position tables"

python "$GET_CHARPOS_SCRIPT" \
    --input-dir "$EFFECTIVE_BUNDLE_DIR/" \
    --pattern "$BUNDLE_PATTERN" \
    --output-dir "$CHARPOS_DIR" \
    --join-mention \
    --timeline-dir "$TIMELINES_DIR" \
    >> "$POST_LOG" 2>&1

###############################################################################
# Post-processing 3: character positions joined to unrevised Step 4 timelines
###############################################################################

log "Extracting unrevised character-position tables"

python "$GET_CHARPOS_SCRIPT" \
    --input-dir "$EFFECTIVE_BUNDLE_DIR/" \
    --pattern "$BUNDLE_PATTERN" \
    --output-dir "$CHARPOS_UNREVISED_DIR" \
    --join-mention \
    --timeline-dir "$TEMPORAL_ANSWERS_DIR" \
    >> "$POST_LOG" 2>&1

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
