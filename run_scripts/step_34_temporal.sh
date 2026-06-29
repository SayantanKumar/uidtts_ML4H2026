#!/bin/bash
#SBATCH --job-name=uid_step34
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
#   sbatch step_34_temporal.sh

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
export INPUT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/multimodal_simple/new_annotations/files"

# Run either every completed Step 1/2 case or only the hard-coded subset below.
# Use the original input filenames or bare case ids.
export FILE_SELECTION_MODE="subset"  # all|subset
SELECTED_INPUT_FILES=("11121324.txt","11649157.txt", "103859.txt", "11083378.txt")

# Must match the Step 12 output directory because Step 34 reads
# "$OUTPUT_DIR/tagged" and "$OUTPUT_DIR/tags".
#export OUTPUT_DIR="$PROJECT_DIR/outputs"
export OUTPUT_DIR="$PROJECT_DIR/outputs/qwen35/"

# Set to "txt.gz" if INPUT_DIR contains compressed reports.
export INPUT_TYPE="txt"
export OUT_EXTENSION=".txt"

export TEMPORAL_PROMPT_FILE="$PROJECT_DIR/templates/temporal_prompt.txt"

# Step 3 long-case handling. If a tagged report has many extracted mentions,
# process the temporal table in mention batches and merge the final output.
export STEP3_TAG_CHUNK_SIZE=60

export LLAMA_SERVER="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/llama.cpp/build2603/bin/llama-server"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"

export CUDA_DEVICES="0,1"
export LLAMA_TIMEOUT=1200

# Model roles. Both point to DeepSeek V3.2 for now; change either later if you
# want separate instruct/reasoning models.
export INSTRUCT_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
export INSTRUCT_MODEL_ALIAS="baal"

export REASONING_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
export REASONING_MODEL_ALIAS="baal"
export PORT=8080

# Leave empty unless the llama-server model needs extra OpenAI body options.
export EXTRA_BODY=""

###############################################################################
# Derived paths
###############################################################################

export TAGGED_DIR="$OUTPUT_DIR/tagged"
export TAGS_DIR="$OUTPUT_DIR/tags"
export TEMPORAL_DIR="$OUTPUT_DIR/temporal"
export TEMPORAL_ANSWERS_DIR="$TEMPORAL_DIR/answers"

TEMPORAL_IO_SCRIPT="$PROJECT_DIR/temporal_prompt_io.py"
EXTRACT_TEMPORAL_SCRIPT="$PROJECT_DIR/extract_temporal_table.py"

STEP3_LOG="$LLAMA_LOG_DIR/step3_temporal_prompt_io.log"
STEP4_LOG="$LLAMA_LOG_DIR/step4_extract_temporal_table.log"

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

RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RUN_SCRIPT_DIR/file_selection_utils.sh"

validate_step12_outputs() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    local tag_files=("$TAGS_DIR"/*_tags.bsv "$TAGS_DIR"/*_tags.bsv.gz)
    filter_selected_files_in_place tagged_files
    filter_selected_files_in_place tag_files

    [[ "${#tagged_files[@]}" -gt 0 ]] || die "No tagged reports found in $TAGGED_DIR"
    [[ "${#tag_files[@]}" -gt 0 ]] || die "No tag tables found in $TAGS_DIR"

    local tagged_count=0
    local tag_count=0
    local tagged_file
    local tag_file
    local base

    for tagged_file in "${tagged_files[@]}"; do
        has_nonblank_content "$tagged_file" || die "Tagged report is empty or invalid: $tagged_file"

        base="$(case_base "$tagged_file")"
        tag_file="$(find_tag_file_for_base "$base")" || die "Missing tag table for tagged report: $tagged_file"
        has_nonblank_content "$tag_file" || die "Tag table is empty or invalid: $tag_file"

        tagged_count=$((tagged_count + 1))
    done

    for tag_file in "${tag_files[@]}"; do
        has_nonblank_content "$tag_file" || die "Tag table is empty or invalid: $tag_file"
        tag_count=$((tag_count + 1))
    done

    log "Validated Step 1/2 outputs: $tagged_count tagged reports, $tag_count tag tables"
}

validate_step3_outputs() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    filter_selected_files_in_place tagged_files
    local temporal_file
    local tagged_file
    local base
    local count=0

    for tagged_file in "${tagged_files[@]}"; do
        base="$(case_base "$tagged_file")"
        temporal_file="$TEMPORAL_DIR/${base}.txt.gz"

        [[ -e "$temporal_file" ]] || die "Missing Step 3 temporal output for $base: $temporal_file"
        has_nonblank_content "$temporal_file" || die "Step 3 temporal output is empty or invalid: $temporal_file"

        count=$((count + 1))
    done

    log "Validated Step 3 outputs: $count temporal LLM outputs"
}

validate_step4_outputs() {
    shopt -s nullglob

    local temporal_files=("$TEMPORAL_DIR"/*.txt "$TEMPORAL_DIR"/*.txt.gz)
    filter_selected_files_in_place temporal_files
    local temporal_file
    local extracted_file
    local base
    local count=0

    [[ "${#temporal_files[@]}" -gt 0 ]] || die "No temporal LLM outputs found in $TEMPORAL_DIR"

    for temporal_file in "${temporal_files[@]}"; do
        base="$(case_base "$temporal_file")"
        extracted_file="$TEMPORAL_ANSWERS_DIR/${base}_extracted.bsv.gz"

        [[ -e "$extracted_file" ]] || die "Missing Step 4 extracted table for $base: $extracted_file"
        has_nonblank_content "$extracted_file" || die "Step 4 extracted table is empty or invalid: $extracted_file"

        count=$((count + 1))
    done

    log "Validated Step 4 outputs: $count extracted temporal tables"
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

mkdir -p "$TEMPORAL_DIR" "$TEMPORAL_ANSWERS_DIR" "$LLAMA_LOG_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$TAGGED_DIR" "Step 1 tagged output directory"
check_path "$TAGS_DIR" "Step 2 tag table directory"
check_path "$TEMPORAL_PROMPT_FILE" "temporal prompt template"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$REASONING_MODEL_PATH" "reasoning model"
check_path "$TEMPORAL_IO_SCRIPT" "temporal_prompt_io.py"
check_path "$EXTRACT_TEMPORAL_SCRIPT" "extract_temporal_table.py"
check_path "$PROJECT_DIR/call_lcpp.py" "call_lcpp.py"

init_file_selection
export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"

source "$SERVER_UTILS"

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Output directory: $OUTPUT_DIR"
log_file_selection
log "Tagged reports: $TAGGED_DIR"
log "Tag tables: $TAGS_DIR"
log "Using python: $(which python)"
log "Python version: $(python --version)"
log "Instruct model path: $INSTRUCT_MODEL_PATH"
log "Instruct model alias: $INSTRUCT_MODEL_ALIAS"
log "Reasoning model path: $REASONING_MODEL_PATH"
log "Reasoning model alias: $REASONING_MODEL_ALIAS"
log "Port: $PORT"

validate_step12_outputs

###############################################################################
# Step 3: generate temporal tables
###############################################################################

EFFECTIVE_STEP3_TAGGED_DIR="$TAGGED_DIR"
EFFECTIVE_STEP3_TAGS_DIR="$TAGS_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/step3_tagged_${SLURM_JOB_ID:-manual}_$$" \
        "$TAGGED_DIR/{}.txt" \
        "$TAGGED_DIR/{}.txt.gz"
    EFFECTIVE_STEP3_TAGGED_DIR="$STAGED_SELECTED_DIR"

    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/step3_tags_${SLURM_JOB_ID:-manual}_$$" \
        "$TAGS_DIR/{}_tags.bsv.gz" \
        "$TAGS_DIR/{}_tags.bsv"
    EFFECTIVE_STEP3_TAGS_DIR="$STAGED_SELECTED_DIR"
fi

log "Starting llama-server for Step 3"
start_server "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS" "$PORT"

trap cleanup_server EXIT

log "Step 3: generating temporal tables with temporal_prompt_io.py"

python "$TEMPORAL_IO_SCRIPT" \
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
    > "$STEP3_LOG" 2>&1

log "Step 3 complete. Temporal LLM outputs: $TEMPORAL_DIR"

validate_step3_outputs

###############################################################################
# Stop server before Step 4
###############################################################################

cleanup_server
trap - EXIT

###############################################################################
# Step 4: extract temporal tables
###############################################################################

log "Step 4: extracting temporal tables with extract_temporal_table.py"

EFFECTIVE_TEMPORAL_INPUT_DIR="$TEMPORAL_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/step4_temporal_${SLURM_JOB_ID:-manual}_$$" \
        "$TEMPORAL_DIR/{}.txt.gz" \
        "$TEMPORAL_DIR/{}.txt"
    EFFECTIVE_TEMPORAL_INPUT_DIR="$STAGED_SELECTED_DIR"
fi

python "$EXTRACT_TEMPORAL_SCRIPT" \
    --input-dir "$EFFECTIVE_TEMPORAL_INPUT_DIR" \
    --output-dir "$TEMPORAL_ANSWERS_DIR" \
    --keep-columns uid4 mention time bounds known "context uid4s" \
    --include-header \
    --no-header \
    > "$STEP4_LOG" 2>&1

log "Step 4 complete. Extracted temporal tables: $TEMPORAL_ANSWERS_DIR"

validate_step4_outputs

###############################################################################
# Summary
###############################################################################

log "Done."
log "Step 3 outputs: $TEMPORAL_DIR"
log "Step 4 outputs: $TEMPORAL_ANSWERS_DIR"
log "Logs:"
log "  llama-server: $LLAMA_LOG_DIR/server_${PORT}.log"
log "  step 3: $STEP3_LOG"
log "  step 4: $STEP4_LOG"
