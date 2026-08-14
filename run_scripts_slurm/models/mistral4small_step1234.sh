#!/bin/bash
#SBATCH --job-name=mistral_medium35_step1234
#SBATCH --time=2-00:00:00
#SBATCH --mem=32g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=sayantan.kumar@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step_1234_tag_time.sh

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

#export INPUT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/i2m4/notes"
export INPUT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/i2m4_n30_notes/"

# Run either every .txt file in INPUT_DIR or only the hard-coded subset below.
# Use FILE_SELECTION_MODE="subset" and list filenames exactly as they appear in
# INPUT_DIR. You may also list bare case ids, e.g. "11121324".
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

# Step 1/2 write here; Step 3/4 read the Step 1/2 outputs from the same root.
export OUTPUT_DIR="$PROJECT_DIR/outputs/mistral_medium35/"

mkdir -p "$OUTPUT_DIR"

# Set to "txt.gz" if INPUT_DIR contains compressed reports.
export INPUT_TYPE="txt"
export OUT_EXTENSION=".txt"

export TAG_PROMPT_FILE="$PROJECT_DIR/templates/tts_tag.txt"
export TEMPORAL_PROMPT_FILE="$PROJECT_DIR/templates/temporal_prompt.txt"

# Step 1 long-note handling. Files above STEP1_MAX_INPUT_LINES are split into
# chunks instead of skipped, then tag ids are renumbered globally in the final
# tagged report.
export STEP1_MAX_INPUT_LINES=500
export STEP1_CHUNK_LINES=450

# Step 3 long-case handling. If a tagged report has many extracted mentions,
# process the temporal table in mention batches and merge the final output.
export STEP3_TAG_CHUNK_SIZE=60

export LLAMA_SERVER="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/llama.cpp/build2603/bin/llama-server"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"

export CUDA_DEVICES="0,1"
export LLAMA_TIMEOUT=1200

# Model roles. Both point to the same model for now; change either later if you
# want separate instruct/reasoning models.
export INSTRUCT_MODEL_PATH="/data/kumars33/.cache/huggingface/gguf/Mistral-Medium-3.5-128B-UD-Q6_K_XL-00001-of-00004.gguf"
export INSTRUCT_MODEL_ALIAS="mistral_medium35"

export REASONING_MODEL_PATH="/data/kumars33/.cache/huggingface/gguf/Mistral-Medium-3.5-128B-UD-Q6_K_XL-00001-of-00004.gguf"
export REASONING_MODEL_ALIAS="mistral_medium35"
export PORT=8083

# Leave empty unless the llama-server model needs extra OpenAI body options.
export EXTRA_BODY=""

###############################################################################
# Derived paths
###############################################################################

export TAGGED_DIR="$OUTPUT_DIR/tagged"
export TAGS_DIR="$OUTPUT_DIR/tags"
export TEMPORAL_DIR="$OUTPUT_DIR/temporal"
export TEMPORAL_ANSWERS_DIR="$TEMPORAL_DIR/answers"

PROMPT_IO_SCRIPT="$PROJECT_DIR/prompt_io.py"
EXTRACT_TAGS_SCRIPT="$PROJECT_DIR/extract_tagged_table.py"
TEMPORAL_IO_SCRIPT="$PROJECT_DIR/temporal_prompt_io.py"
EXTRACT_TEMPORAL_SCRIPT="$PROJECT_DIR/extract_temporal_table.py"

STEP1_LOG="$LLAMA_LOG_DIR/step1_prompt_io.log"
STEP2_LOG="$LLAMA_LOG_DIR/step2_extract_tags.log"
STEP3_LOG="$LLAMA_LOG_DIR/step3_temporal_prompt_io.log"
STEP4_LOG="$LLAMA_LOG_DIR/step4_extract_temporal_table.log"

SERVER_LOG="$LLAMA_LOG_DIR/server_${PORT}.log"
STEP1_SERVER_LOG="$LLAMA_LOG_DIR/server_${PORT}_step1.log"
STEP3_SERVER_LOG="$LLAMA_LOG_DIR/server_${PORT}_step3.log"

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
    name="${name%.txt}"
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
    local gz_file="$TAGS_DIR/${base}_tags.bsv.gz"
    local plain_file="$TAGS_DIR/${base}_tags.bsv"

    if [[ -e "$gz_file" ]]; then
        echo "$gz_file"
    elif [[ -e "$plain_file" ]]; then
        echo "$plain_file"
    else
        return 1
    fi
}

cleanup_server() {
    if [[ -n "${LLAMA_SERVER_PID:-}" ]]; then
        kill_server "$LLAMA_SERVER_PID" "$PORT" || true
        unset LLAMA_SERVER_PID
    fi
}

archive_server_log() {
    local dest="$1"
    if [[ -s "$SERVER_LOG" ]]; then
        cp -f "$SERVER_LOG" "$dest"
    fi
}

SELECTION_UTILS="$PROJECT_DIR/run_scripts/file_selection_utils.sh"
if [[ ! -f "$SELECTION_UTILS" ]]; then
    RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELECTION_UTILS="$RUN_SCRIPT_DIR/file_selection_utils.sh"
fi
source "$SELECTION_UTILS"

validate_step12_outputs() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    local tag_files=("$TAGS_DIR"/*_tags.bsv "$TAGS_DIR"/*_tags.bsv.gz)
    filter_selected_files_in_place tagged_files
    filter_selected_files_in_place tag_files

    [[ "${#tagged_files[@]}" -gt 0 ]] || die "No tagged reports found in $TAGGED_DIR"
    [[ "${#tag_files[@]}" -gt 0 ]] || die "No tag tables found in $TAGS_DIR"

    local valid_pair_count=0
    local skipped_pair_count=0
    local tag_count=0
    local invalid_tag_count=0
    local tagged_file
    local tag_file
    local base

    for tagged_file in "${tagged_files[@]}"; do
        if ! has_nonblank_content "$tagged_file"; then
            log "WARNING: skipping invalid Step 1 tagged report: $tagged_file"
            skipped_pair_count=$((skipped_pair_count + 1))
            continue
        fi

        base="$(case_base "$tagged_file")"
        if ! tag_file="$(find_tag_file_for_base "$base")"; then
            log "WARNING: skipping $base for downstream steps; missing Step 2 tag table"
            skipped_pair_count=$((skipped_pair_count + 1))
            continue
        fi
        if ! has_nonblank_content "$tag_file"; then
            log "WARNING: skipping $base for downstream steps; Step 2 tag table is empty or invalid: $tag_file"
            skipped_pair_count=$((skipped_pair_count + 1))
            continue
        fi

        valid_pair_count=$((valid_pair_count + 1))
    done

    for tag_file in "${tag_files[@]}"; do
        if has_nonblank_content "$tag_file"; then
            tag_count=$((tag_count + 1))
        else
            log "WARNING: invalid tag table will be ignored: $tag_file"
            invalid_tag_count=$((invalid_tag_count + 1))
        fi
    done

    [[ "$valid_pair_count" -gt 0 ]] || die "No valid Step 1/2 tagged-report + tag-table pairs found"

    log "Validated Step 1/2 outputs: $valid_pair_count valid pairs; skipped $skipped_pair_count invalid/missing pairs; $tag_count non-empty tag tables; $invalid_tag_count invalid tag tables"
}

validate_step3_outputs() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    filter_selected_files_in_place tagged_files
    local temporal_file
    local tagged_file
    local base
    local tag_file
    local expected_count=0
    local count=0
    local skipped_prereq_count=0
    local missing_or_empty_count=0

    for tagged_file in "${tagged_files[@]}"; do
        if ! has_nonblank_content "$tagged_file"; then
            skipped_prereq_count=$((skipped_prereq_count + 1))
            continue
        fi

        base="$(case_base "$tagged_file")"
        if ! tag_file="$(find_tag_file_for_base "$base")" || ! has_nonblank_content "$tag_file"; then
            skipped_prereq_count=$((skipped_prereq_count + 1))
            continue
        fi

        expected_count=$((expected_count + 1))
        temporal_file="$TEMPORAL_DIR/${base}.txt.gz"

        if [[ ! -e "$temporal_file" ]]; then
            log "WARNING: missing Step 3 temporal output for $base: $temporal_file"
            missing_or_empty_count=$((missing_or_empty_count + 1))
            continue
        fi
        if ! has_nonblank_content "$temporal_file"; then
            log "WARNING: Step 3 temporal output is empty or invalid for $base: $temporal_file"
            missing_or_empty_count=$((missing_or_empty_count + 1))
            continue
        fi

        count=$((count + 1))
    done

    [[ "$expected_count" -gt 0 ]] || die "No valid Step 1/2 pairs available to validate Step 3"
    [[ "$count" -gt 0 ]] || die "No valid Step 3 temporal outputs found"

    log "Validated Step 3 outputs: $count valid temporal outputs; $missing_or_empty_count missing/empty outputs; $skipped_prereq_count cases skipped because Step 1/2 prerequisites were invalid"
}

validate_step4_outputs() {
    shopt -s nullglob

    local temporal_files=("$TEMPORAL_DIR"/*.txt "$TEMPORAL_DIR"/*.txt.gz)
    filter_selected_files_in_place temporal_files
    local temporal_file
    local extracted_file
    local base
    local count=0
    local missing_or_empty_count=0

    [[ "${#temporal_files[@]}" -gt 0 ]] || die "No temporal LLM outputs found in $TEMPORAL_DIR"

    for temporal_file in "${temporal_files[@]}"; do
        if ! has_nonblank_content "$temporal_file"; then
            log "WARNING: skipping invalid Step 3 temporal output during Step 4 validation: $temporal_file"
            missing_or_empty_count=$((missing_or_empty_count + 1))
            continue
        fi

        base="$(case_base "$temporal_file")"
        extracted_file="$TEMPORAL_ANSWERS_DIR/${base}_extracted.bsv.gz"

        if [[ ! -e "$extracted_file" ]]; then
            log "WARNING: missing Step 4 extracted table for $base: $extracted_file"
            missing_or_empty_count=$((missing_or_empty_count + 1))
            continue
        fi
        if ! has_nonblank_content "$extracted_file"; then
            log "WARNING: Step 4 extracted table is empty or invalid for $base: $extracted_file"
            missing_or_empty_count=$((missing_or_empty_count + 1))
            continue
        fi

        count=$((count + 1))
    done

    [[ "$count" -gt 0 ]] || die "No valid Step 4 extracted temporal tables found"

    log "Validated Step 4 outputs: $count valid extracted temporal tables; $missing_or_empty_count missing/empty outputs"
}

prepare_step3_inputs() {
    shopt -s nullglob

    local stage_root="$1"
    local case_stage="$stage_root/cases"
    local tag_stage="$stage_root/tags"
    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    filter_selected_files_in_place tagged_files

    local tagged_file
    local tag_file
    local base
    local staged_count=0
    local skipped_count=0

    rm -rf "$stage_root"
    mkdir -p "$case_stage" "$tag_stage"

    for tagged_file in "${tagged_files[@]}"; do
        base="$(case_base "$tagged_file")"

        if ! has_nonblank_content "$tagged_file"; then
            log "WARNING: Step 3 skipping $base; tagged report is empty or invalid"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! tag_file="$(find_tag_file_for_base "$base")"; then
            log "WARNING: Step 3 skipping $base; missing tag table"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! has_nonblank_content "$tag_file"; then
            log "WARNING: Step 3 skipping $base; tag table is empty or invalid: $tag_file"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        ln -sfn "$tagged_file" "$case_stage/$(basename "$tagged_file")"
        ln -sfn "$tag_file" "$tag_stage/$(basename "$tag_file")"
        staged_count=$((staged_count + 1))
    done

    [[ "$staged_count" -gt 0 ]] || die "No valid Step 3 inputs after filtering Step 1/2 outputs"

    EFFECTIVE_STEP3_TAGGED_DIR="$case_stage"
    EFFECTIVE_STEP3_TAGS_DIR="$tag_stage"
    log "Prepared Step 3 inputs: $staged_count valid cases; skipped $skipped_count cases with missing/empty prerequisites"
}

prepare_step4_inputs() {
    shopt -s nullglob

    local stage_dir="$1"
    local temporal_files=("$TEMPORAL_DIR"/*.txt "$TEMPORAL_DIR"/*.txt.gz)
    filter_selected_files_in_place temporal_files

    local temporal_file
    local staged_count=0
    local skipped_count=0

    rm -rf "$stage_dir"
    mkdir -p "$stage_dir"

    for temporal_file in "${temporal_files[@]}"; do
        if ! has_nonblank_content "$temporal_file"; then
            log "WARNING: Step 4 skipping invalid temporal output: $temporal_file"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        ln -sfn "$temporal_file" "$stage_dir/$(basename "$temporal_file")"
        staged_count=$((staged_count + 1))
    done

    [[ "$staged_count" -gt 0 ]] || die "No valid Step 4 inputs after filtering Step 3 outputs"

    EFFECTIVE_TEMPORAL_INPUT_DIR="$stage_dir"
    log "Prepared Step 4 inputs: $staged_count valid temporal outputs; skipped $skipped_count invalid outputs"
}

###############################################################################
# Preflight
###############################################################################

mkdir -p "$TAGGED_DIR" "$TAGS_DIR" "$TEMPORAL_DIR" "$TEMPORAL_ANSWERS_DIR" "$LLAMA_LOG_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$INPUT_DIR" "input directory"
check_path "$TAG_PROMPT_FILE" "tag prompt template"
check_path "$TEMPORAL_PROMPT_FILE" "temporal prompt template"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$INSTRUCT_MODEL_PATH" "instruct model"
check_path "$REASONING_MODEL_PATH" "reasoning model"
check_path "$PROMPT_IO_SCRIPT" "prompt_io.py"
check_path "$EXTRACT_TAGS_SCRIPT" "extract_tagged_table.py"
check_path "$TEMPORAL_IO_SCRIPT" "temporal_prompt_io.py"
check_path "$EXTRACT_TEMPORAL_SCRIPT" "extract_temporal_table.py"
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
log "Tagged reports: $TAGGED_DIR"
log "Tag tables: $TAGS_DIR"
log "Temporal outputs: $TEMPORAL_DIR"
log "Temporal extracted tables: $TEMPORAL_ANSWERS_DIR"
log "Using python: $(which python)"
log "Python version: $(python --version)"
log "Instruct model path: $INSTRUCT_MODEL_PATH"
log "Instruct model alias: $INSTRUCT_MODEL_ALIAS"
log "Reasoning model path: $REASONING_MODEL_PATH"
log "Reasoning model alias: $REASONING_MODEL_ALIAS"
log "Port: $PORT"

###############################################################################
# NOTE: Step 1/2 below are intentionally commented out; existing Step 1/2 outputs are used and execution resumes at Step 3.
# # Step 1: tag clinical event mentions
# ###############################################################################
#
# log "Starting llama-server for Step 1"
# start_server "$INSTRUCT_MODEL_PATH" "$INSTRUCT_MODEL_ALIAS" "$PORT"
# trap cleanup_server EXIT
#
# log "Step 1: tagging clinical mentions with prompt_io.py"
#
# if ! python "$PROMPT_IO_SCRIPT" \
#     --in_dir "$EFFECTIVE_INPUT_DIR" \
#     --input_type "$INPUT_TYPE" \
#     --out_dir "$TAGGED_DIR" \
#     --prompt_file "$TAG_PROMPT_FILE" \
#     --llm_mechanism llama.cpp \
#     --lcpp_url "http://127.0.0.1:${PORT}" \
#     --model_name "$INSTRUCT_MODEL_ALIAS" \
#     --endpoint_type openai \
#     --extra_body "$EXTRA_BODY" \
#     --max_input_lines "$STEP1_MAX_INPUT_LINES" \
#     --chunk_overlength \
#     --chunk_line_count "$STEP1_CHUNK_LINES" \
#     --out_extension "$OUT_EXTENSION" \
#     > "$STEP1_LOG" 2>&1; then
#     log "WARNING: Step 1 prompt_io.py returned a nonzero status; continuing with any tagged reports that were produced"
# fi
#
# log "Step 1 complete. Tagged reports: $TAGGED_DIR"
#
# cleanup_server
# archive_server_log "$STEP1_SERVER_LOG"
# trap - EXIT
#
# ###############################################################################
# # Step 2: extract tag table and assign uid4 IDs
# ###############################################################################
#
# log "Step 2: extracting tag tables with extract_tagged_table.py"
#
# EFFECTIVE_TAGGED_DIR="$TAGGED_DIR"
# if selection_enabled; then
#     stage_selected_files_for_bases \
#         "$SELECTION_WORK_DIR/tagged_${SLURM_JOB_ID:-manual}_$$" \
#         "$TAGGED_DIR/{}.txt" \
#         "$TAGGED_DIR/{}.txt.gz"
#     EFFECTIVE_TAGGED_DIR="$STAGED_SELECTED_DIR"
# fi
#
# if ! python "$EXTRACT_TAGS_SCRIPT" \
#     --input-dir "$EFFECTIVE_TAGGED_DIR" \
#     --output-dir "$TAGS_DIR" \
#     > "$STEP2_LOG" 2>&1; then
#     log "WARNING: Step 2 extract_tagged_table.py returned a nonzero status; continuing with any tag tables that were produced"
# fi
#
# log "Step 2 complete. UID tag tables: $TAGS_DIR"
#
validate_step12_outputs

###############################################################################
# Step 3: generate temporal tables
###############################################################################

prepare_step3_inputs "$SELECTION_WORK_DIR/step3_valid_${SLURM_JOB_ID:-manual}_$$"

log "Starting llama-server for Step 3"
start_server "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS" "$PORT"
trap cleanup_server EXIT

log "Step 3: generating temporal tables with temporal_prompt_io.py"

if ! python "$TEMPORAL_IO_SCRIPT" \
    --case-dir "$EFFECTIVE_STEP3_TAGGED_DIR" \
    --tags-dir "$EFFECTIVE_STEP3_TAGS_DIR" \
    --output-dir "$TEMPORAL_DIR" \
    --prompt-file "$TEMPORAL_PROMPT_FILE" \
    --llm_mechanism llama.cpp \
    --lcpp_url "http://127.0.0.1:${PORT}" \
    --keep_columns 1 2 \
    --tag_chunk_size "$STEP3_TAG_CHUNK_SIZE" \
    --endpoint_type openai \
    --model_name "$REASONING_MODEL_ALIAS" \
    --extra_body "$EXTRA_BODY" \
    > "$STEP3_LOG" 2>&1; then
    log "WARNING: Step 3 temporal_prompt_io.py returned a nonzero status; continuing with any temporal outputs that were produced"
fi

log "Step 3 complete. Temporal LLM outputs: $TEMPORAL_DIR"

validate_step3_outputs

cleanup_server
archive_server_log "$STEP3_SERVER_LOG"
trap - EXIT

###############################################################################
# Step 4: extract temporal tables
###############################################################################

log "Step 4: extracting temporal tables with extract_temporal_table.py"

prepare_step4_inputs "$SELECTION_WORK_DIR/step4_valid_${SLURM_JOB_ID:-manual}_$$"

if ! python "$EXTRACT_TEMPORAL_SCRIPT" \
    --input-dir "$EFFECTIVE_TEMPORAL_INPUT_DIR" \
    --output-dir "$TEMPORAL_ANSWERS_DIR" \
    --keep-columns uid4 mention time bounds known "context uid4s" \
    --include-header \
    --no-header \
    > "$STEP4_LOG" 2>&1; then
    log "WARNING: Step 4 extract_temporal_table.py returned a nonzero status; continuing with any extracted temporal tables that were produced"
fi

log "Step 4 complete. Extracted temporal tables: $TEMPORAL_ANSWERS_DIR"

validate_step4_outputs

###############################################################################
# Summary
###############################################################################

log "Done."
log "Step 1 outputs: $TAGGED_DIR"
log "Step 2 outputs: $TAGS_DIR"
log "Step 3 outputs: $TEMPORAL_DIR"
log "Step 4 outputs: $TEMPORAL_ANSWERS_DIR"
log "Logs:"
log "  step 1 server: $STEP1_SERVER_LOG"
log "  step 3 server: $STEP3_SERVER_LOG"
log "  step 1: $STEP1_LOG"
log "  step 2: $STEP2_LOG"
log "  step 3: $STEP3_LOG"
log "  step 4: $STEP4_LOG"
