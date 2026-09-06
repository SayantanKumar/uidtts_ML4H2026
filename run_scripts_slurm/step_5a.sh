#!/bin/bash
#SBATCH --job-name=uid_step5a
#SBATCH --time=1-00:00:00
#SBATCH --mem=128g
#SBATCH --output=sbatch_output/slurm_%j.out
#SBATCH --error=sbatch_error/slurm_%j.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step5a_prepare_uid_reports.sh

set -euo pipefail

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
export OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/outputs/${MODEL_OUTPUT_NAME:-single_case_qwen}}"

export TAGGED_DIR="$OUTPUT_DIR/tagged"
export TAGS_DIR="$OUTPUT_DIR/tags"

# Permanent Step 5A outputs used by Step 5B.
export BUNDLE_DIR="$OUTPUT_DIR/bundle"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"
export STEP5_MASK_LOG="$LLAMA_LOG_DIR/step5a_mask_cr_append_mt.log"

export MASK_SCRIPT="$PROJECT_DIR/mask_cr_append_mt.py"

# Run all completed cases or only the listed subset.
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324"
    # "11649157"
)

export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"
export STEP5_TAGGED_GZ_DIR="$SELECTION_WORK_DIR/step5a_tagged_${SLURM_JOB_ID:-manual}_$$"
export STEP5_EFFECTIVE_TAGS_DIR="$SELECTION_WORK_DIR/step5a_tags_${SLURM_JOB_ID:-manual}_$$"

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
    name="${name%.txt}"
    name="${name%_modified}"
    echo "$name"
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

find_tag_file_for_base() {
    local base="$1"

    if [[ -e "$TAGS_DIR/${base}_tags.bsv.gz" ]]; then
        echo "$TAGS_DIR/${base}_tags.bsv.gz"
    elif [[ -e "$TAGS_DIR/${base}_tags.bsv" ]]; then
        echo "$TAGS_DIR/${base}_tags.bsv"
    else
        return 1
    fi
}

cleanup() {
    rm -rf "$STEP5_TAGGED_GZ_DIR" "$STEP5_EFFECTIVE_TAGS_DIR"
}
trap cleanup EXIT

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

mkdir -p \
    "$BUNDLE_DIR" \
    "$LLAMA_LOG_DIR" \
    "$SELECTION_WORK_DIR" \
    "$STEP5_TAGGED_GZ_DIR" \
    "$STEP5_EFFECTIVE_TAGS_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$TAGGED_DIR" "Step 1 tagged output directory"
check_path "$TAGS_DIR" "Step 2 tag table directory"
check_path "$MASK_SCRIPT" "mask_cr_append_mt.py"

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Output directory: $OUTPUT_DIR"
log_file_selection
log "Using Python: $(which python)"
log "Python version: $(python --version)"

###############################################################################
# Step 5A: stage valid Step 1/2 pairs
###############################################################################

shopt -s nullglob

tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
filter_selected_files_in_place tagged_files

[[ "${#tagged_files[@]}" -gt 0 ]] ||
    die "No selected tagged reports found in $TAGGED_DIR"

staged_count=0
skipped_count=0

for tagged_file in "${tagged_files[@]}"; do
    base="$(case_base "$tagged_file")"

    if ! has_nonblank_content "$tagged_file"; then
        log "WARNING: skipping $base; tagged report is empty or invalid"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    if ! tag_file="$(find_tag_file_for_base "$base")"; then
        log "WARNING: skipping $base; corresponding tag table is missing"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    if ! has_nonblank_content "$tag_file"; then
        log "WARNING: skipping $base; tag table is empty or invalid: $tag_file"
        skipped_count=$((skipped_count + 1))
        continue
    fi

    staged_report="$STEP5_TAGGED_GZ_DIR/${base}.txt.gz"
    staged_tags="$STEP5_EFFECTIVE_TAGS_DIR/${base}_tags.bsv.gz"

    if [[ "$tagged_file" == *.gz ]]; then
        cp "$tagged_file" "$staged_report"
    else
        gzip -c "$tagged_file" > "$staged_report"
    fi

    if [[ "$tag_file" == *.gz ]]; then
        ln -s "$tag_file" "$staged_tags"
    else
        gzip -c "$tag_file" > "$staged_tags"
    fi

    staged_count=$((staged_count + 1))
done

[[ "$staged_count" -gt 0 ]] ||
    die "No valid Step 1/2 input pairs were available for Step 5A"

log "Staged $staged_count valid input pairs; skipped $skipped_count"

###############################################################################
# Step 5A: prepare UID-tagged bundle reports
###############################################################################

log "Running Step 5A with mask_cr_append_mt.py"

if ! python "$MASK_SCRIPT" \
    --mention_table_dir "$STEP5_EFFECTIVE_TAGS_DIR/" \
    --tagged_case_report_dir "$STEP5_TAGGED_GZ_DIR/" \
    --output_dir "$BUNDLE_DIR/" \
    > "$STEP5_MASK_LOG" 2>&1; then
    log "WARNING: mask_cr_append_mt.py returned a nonzero status"
fi

bundle_reports=("$BUNDLE_DIR"/*.txt)
filter_selected_files_in_place bundle_reports

[[ "${#bundle_reports[@]}" -gt 0 ]] ||
    die "Step 5A produced no bundle reports in $BUNDLE_DIR"

for bundle_report in "${bundle_reports[@]}"; do
    has_nonblank_content "$bundle_report" ||
        die "Step 5A output is empty: $bundle_report"
done

log "Step 5A complete"
log "Prepared bundle reports: ${#bundle_reports[@]}"
log "Output for later steps: $BUNDLE_DIR"
log "Log: $STEP5_MASK_LOG"
