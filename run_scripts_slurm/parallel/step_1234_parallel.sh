#!/bin/bash
#SBATCH --job-name=qwen35_step1234_pp
#SBATCH --time=2-00:00:00
#SBATCH --mem=48g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=sayantan.kumar@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step_1234_parallel.sh

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

export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

###############################################################################
# Edit these paths/settings
###############################################################################

export PROJECT_DIR="${PROJECT_DIR:-/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection}"

# EDIT THIS for your notes folder.
# export INPUT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/multimodal_simple/new_annotations/files"
export INPUT_DIR="${INPUT_DIR:-/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/i2m4_n30_notes/}"

# Run every .txt file in INPUT_DIR, or only the hard-coded subset below.
export FILE_SELECTION_MODE="${FILE_SELECTION_MODE:-all}"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

# Kept separate from outputs/qwen35 so parallel experiments do not overwrite
# conventional runs. Change to outputs/qwen35/ if you want to reuse that folder.
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/qwen35_parallel/}"
mkdir -p "$OUTPUT_DIR"

export INPUT_TYPE="${INPUT_TYPE:-txt}"
export OUT_EXTENSION="${OUT_EXTENSION:-.txt}"

export TAG_PROMPT_FILE="${TAG_PROMPT_FILE:-$PROJECT_DIR/templates/tts_tag.txt}"
export TEMPORAL_PROMPT_FILE="${TEMPORAL_PROMPT_FILE:-$PROJECT_DIR/templates/temporal_prompt.txt}"

export STEP1_MAX_INPUT_LINES="${STEP1_MAX_INPUT_LINES:-500}"
export STEP1_CHUNK_LINES="${STEP1_CHUNK_LINES:-450}"
export STEP3_TAG_CHUNK_SIZE="${STEP3_TAG_CHUNK_SIZE:-60}"

export LLAMA_SERVER="${LLAMA_SERVER:-/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/llama.cpp/build2603/bin/llama-server}"
export LLAMA_LOG_DIR="${LLAMA_LOG_DIR:-$OUTPUT_DIR/logs}"

export CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
export LLAMA_TIMEOUT="${LLAMA_TIMEOUT:-1200}"

export INSTRUCT_MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export INSTRUCT_MODEL_ALIAS="${INSTRUCT_MODEL_ALIAS:-qwen35}"
export REASONING_MODEL_PATH="${REASONING_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export REASONING_MODEL_ALIAS="${REASONING_MODEL_ALIAS:-qwen35}"

# One server, multiple request slots. With --ctx-size 72000 and -np 2,
# llama.cpp effectively gives each slot roughly half the total context.
export PORT="${PORT:-8085}"
export LLAMA_PARALLEL_SLOTS="${LLAMA_PARALLEL_SLOTS:-2}"
export LLAMA_CONTEXT_SIZE="${LLAMA_CONTEXT_SIZE:-72000}"
export LLAMA_N_PREDICT="${LLAMA_N_PREDICT:-72000}"
export LLAMA_THREADS="${LLAMA_THREADS:-16}"
export LLAMA_TEMPERATURE="${LLAMA_TEMPERATURE:-0.6}"
export LLAMA_EXTRA_SERVER_ARGS="${LLAMA_EXTRA_SERVER_ARGS:-}"

# Number of concurrent Python client processes. Keep this <= LLAMA_PARALLEL_SLOTS.
export STEP1_WORKERS="${STEP1_WORKERS:-2}"
export STEP3_WORKERS="${STEP3_WORKERS:-2}"

export EXTRA_BODY="${EXTRA_BODY:-}"

###############################################################################
# Derived paths
###############################################################################

export TAGGED_DIR="$OUTPUT_DIR/tagged"
export TAGS_DIR="$OUTPUT_DIR/tags"
export TEMPORAL_DIR="$OUTPUT_DIR/temporal"
export TEMPORAL_ANSWERS_DIR="$TEMPORAL_DIR/answers"
export PARALLEL_WORK_DIR="$OUTPUT_DIR/pipeline_work/parallel_${SLURM_JOB_ID:-manual}_$$"

PROMPT_IO_SCRIPT="$PROJECT_DIR/prompt_io.py"
EXTRACT_TAGS_SCRIPT="$PROJECT_DIR/extract_tagged_table.py"
TEMPORAL_IO_SCRIPT="$PROJECT_DIR/temporal_prompt_io.py"
EXTRACT_TEMPORAL_SCRIPT="$PROJECT_DIR/extract_temporal_table.py"

STEP1_LOG_DIR="$LLAMA_LOG_DIR/step1_parallel"
STEP3_LOG_DIR="$LLAMA_LOG_DIR/step3_parallel"
STEP2_LOG="$LLAMA_LOG_DIR/step2_extract_tags.log"
STEP4_LOG="$LLAMA_LOG_DIR/step4_extract_temporal_table.log"

SERVER_LOG="$LLAMA_LOG_DIR/server_${PORT}.log"
STEP1_SERVER_LOG="$LLAMA_LOG_DIR/server_${PORT}_step1_parallel.log"
STEP3_SERVER_LOG="$LLAMA_LOG_DIR/server_${PORT}_step3_parallel.log"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

###############################################################################
# Helpers
###############################################################################

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

epoch_now() {
    date +%s
}

format_duration() {
    local seconds="$1"
    printf "%02d:%02d:%02d" $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
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

input_extension() {
    if [[ "$INPUT_TYPE" == "txt.gz" ]]; then
        echo ".txt.gz"
    else
        echo ".txt"
    fi
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

SELECTION_UTILS="$PROJECT_DIR/run_scripts/file_selection_utils.sh"
if [[ ! -f "$SELECTION_UTILS" ]]; then
    RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELECTION_UTILS="$RUN_SCRIPT_DIR/../file_selection_utils.sh"
fi
source "$SELECTION_UTILS"

start_parallel_server() {
    local model="$1"
    local alias="$2"
    local port="$3"
    local log_file="$LLAMA_LOG_DIR/server_${port}.log"
    local server_cmd
    local extra_args=()

    mkdir -p "$LLAMA_LOG_DIR"

    log "Starting llama-server with model: $model"
    log "Alias: $alias"
    log "CUDA devices: $CUDA_DEVICES"
    log "Parallel slots: $LLAMA_PARALLEL_SLOTS"
    log "Total ctx-size: $LLAMA_CONTEXT_SIZE"
    log "Server log: $log_file"

    server_cmd=(
        "$LLAMA_SERVER"
        -m "$model"
        -a "$alias"
        -t "$LLAMA_THREADS"
        -ngl 999
        -n "$LLAMA_N_PREDICT"
        --ctx-size "$LLAMA_CONTEXT_SIZE"
        -np "$LLAMA_PARALLEL_SLOTS"
        --temp "$LLAMA_TEMPERATURE"
        --port "$port"
        --jinja
    )

    if [[ -n "${LLAMA_EXTRA_SERVER_ARGS:-}" ]]; then
        read -r -a extra_args <<< "$LLAMA_EXTRA_SERVER_ARGS"
        server_cmd+=("${extra_args[@]}")
    fi

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${server_cmd[@]}" > "$log_file" 2>&1 &
    local server_pid=$!
    log "Server PID: $server_pid"

    sleep 5

    local url="http://127.0.0.1:${port}/v1/chat/completions"
    local start_time
    start_time="$(epoch_now)"
    local timeout="${LLAMA_TIMEOUT:-300}"
    local attempts=0
    local health_prompt="ping"

    log "Waiting for llama-server to be ready..."
    while true; do
        attempts=$((attempts + 1))

        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server process died unexpectedly"
            tail -n 40 "$log_file" 2>/dev/null || true
            return 1
        fi

        local raw
        raw=$(timeout 10 curl --noproxy "localhost,127.0.0.1,::1" -s -w '\n%{http_code}' -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"default\",
                \"messages\":[{\"role\":\"user\",\"content\":\"$health_prompt\"}],
                \"max_tokens\": 16
            }" 2>/dev/null || echo -e "\n000")

        local http_code body content trimmed
        http_code=${raw##*$'\n'}
        body=${raw%$'\n'$http_code}

        if [[ "$http_code" == "200" && -n "$body" ]]; then
            content=$(echo "$body" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")
            trimmed=$(echo "$content" | xargs)
            if [[ -n "$trimmed" && "$trimmed" != "null" && "$trimmed" != "$health_prompt" ]]; then
                log "llama-server ready with content: '$trimmed'"
                break
            fi
            if [[ -z "$content" ]]; then
                log "llama-server ready with non-empty body"
                break
            fi
        fi

        if (( attempts % 10 == 0 )); then
            log "Still waiting... attempt=$attempts HTTP=$http_code"
            echo "Last body (truncated): '$(echo "$body" | head -c 200)'"
        fi

        if (( $(epoch_now) - start_time > timeout )); then
            echo "Timed out waiting for llama-server after ${timeout} seconds"
            tail -n 80 "$log_file" 2>/dev/null || true
            kill "$server_pid" 2>/dev/null || true
            return 1
        fi

        sleep 3
    done

    export LLAMA_SERVER_PID="$server_pid"
    return 0
}

kill_server_pid() {
    local pid="${1:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        log "Stopping llama-server PID $pid"
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

cleanup_server() {
    if [[ -n "${LLAMA_SERVER_PID:-}" ]]; then
        kill_server_pid "$LLAMA_SERVER_PID"
        unset LLAMA_SERVER_PID
    fi
}

archive_server_log() {
    local dest="$1"
    if [[ -s "$SERVER_LOG" ]]; then
        cp -f "$SERVER_LOG" "$dest"
    fi
}

validate_worker_count() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be an integer; got '$value'"
    (( value >= 1 )) || die "$name must be >= 1"
    (( value <= LLAMA_PARALLEL_SLOTS )) || die "$name=$value must be <= LLAMA_PARALLEL_SLOTS=$LLAMA_PARALLEL_SLOTS"
}

collect_input_filenames() {
    local ext
    ext="$(input_extension)"
    mapfile -t INPUT_CASE_FILES < <(
        find -L "$EFFECTIVE_INPUT_DIR" -maxdepth 1 -type f -name "*${ext}" -printf "%f\n" | sort
    )
    [[ "${#INPUT_CASE_FILES[@]}" -gt 0 ]] || die "No input files matching *${ext} found in $EFFECTIVE_INPUT_DIR"
}

make_case_list_shards() {
    local out_dir="$1"
    local worker_count="$2"
    shift 2
    local files=("$@")
    local i shard

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    SHARD_FILES=()
    for ((i = 0; i < worker_count; i++)); do
        SHARD_FILES+=("$out_dir/shard_${i}.txt")
        : > "${SHARD_FILES[$i]}"
    done

    for ((i = 0; i < ${#files[@]}; i++)); do
        shard=$((i % worker_count))
        printf "%s\n" "${files[$i]}" >> "${SHARD_FILES[$shard]}"
    done
}

wait_for_worker_group() {
    local label="$1"
    local failed=0
    local i pid worker_label

    for ((i = 0; i < ${#WORKER_PIDS[@]}; i++)); do
        pid="${WORKER_PIDS[$i]}"
        worker_label="${WORKER_LABELS[$i]}"
        if wait "$pid"; then
            log "$label worker success: $worker_label"
        else
            log "WARNING: $label worker failed: $worker_label"
            failed=1
        fi
    done

    return "$failed"
}

validate_step12_outputs() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    local tag_files=("$TAGS_DIR"/*_tags.bsv "$TAGS_DIR"/*_tags.bsv.gz)
    filter_selected_files_in_place tagged_files
    filter_selected_files_in_place tag_files

    [[ "${#tagged_files[@]}" -gt 0 ]] || die "No tagged reports found in $TAGGED_DIR"
    [[ "${#tag_files[@]}" -gt 0 ]] || die "No tag tables found in $TAGS_DIR"

    local tagged_file tag_file base count=0 skipped_count=0 invalid_tag_count=0 tag_count=0
    for tagged_file in "${tagged_files[@]}"; do
        if ! has_nonblank_content "$tagged_file"; then
            log "WARNING: skipping invalid Step 1 tagged report: $tagged_file"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        base="$(case_base "$tagged_file")"
        if ! tag_file="$(find_tag_file_for_base "$base")"; then
            log "WARNING: skipping $base for downstream steps; missing Step 2 tag table"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! has_nonblank_content "$tag_file"; then
            log "WARNING: skipping $base for downstream steps; Step 2 tag table is empty or invalid: $tag_file"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        count=$((count + 1))
    done

    for tag_file in "${tag_files[@]}"; do
        if has_nonblank_content "$tag_file"; then
            tag_count=$((tag_count + 1))
        else
            log "WARNING: invalid tag table will be ignored: $tag_file"
            invalid_tag_count=$((invalid_tag_count + 1))
        fi
    done

    [[ "$count" -gt 0 ]] || die "No valid Step 1/2 tagged-report + tag-table pairs found"

    log "Validated Step 1/2 outputs: $count valid pairs; skipped $skipped_count invalid/missing pairs; $tag_count non-empty tag tables; $invalid_tag_count invalid tag tables"
}

validate_step3_outputs() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    filter_selected_files_in_place tagged_files
    local tagged_file base temporal_file tag_file count=0 expected_count=0 skipped_prereq_count=0 missing_or_empty_count=0

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
    local temporal_file base extracted_file count=0 missing_or_empty_count=0

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

make_step3_shards() {
    shopt -s nullglob

    local worker_count="$1"
    local shard_root="$2"
    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    filter_selected_files_in_place tagged_files

    [[ "${#tagged_files[@]}" -gt 0 ]] || die "No tagged reports found for Step 3 sharding"

    rm -rf "$shard_root"
    mkdir -p "$shard_root"

    STEP3_CASE_DIRS=()
    STEP3_TAG_DIRS=()
    local i shard tagged_file tag_file base added_count=0 skipped_count=0
    for ((i = 0; i < worker_count; i++)); do
        STEP3_CASE_DIRS+=("$shard_root/shard_${i}/cases")
        STEP3_TAG_DIRS+=("$shard_root/shard_${i}/tags")
        mkdir -p "${STEP3_CASE_DIRS[$i]}" "${STEP3_TAG_DIRS[$i]}"
    done

    for ((i = 0; i < ${#tagged_files[@]}; i++)); do
        tagged_file="${tagged_files[$i]}"
        base="$(case_base "$tagged_file")"
        if ! has_nonblank_content "$tagged_file"; then
            log "WARNING: Step 3 skipping $base; tagged report is empty or invalid"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! tag_file="$(find_tag_file_for_base "$base")"; then
            log "WARNING: Step 3 skipping $base; missing tag file"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! has_nonblank_content "$tag_file"; then
            log "WARNING: Step 3 skipping $base; tag table is empty or invalid: $tag_file"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        shard=$((added_count % worker_count))
        ln -sfn "$tagged_file" "${STEP3_CASE_DIRS[$shard]}/$(basename "$tagged_file")"
        ln -sfn "$tag_file" "${STEP3_TAG_DIRS[$shard]}/$(basename "$tag_file")"
        added_count=$((added_count + 1))
    done

    [[ "$added_count" -gt 0 ]] || die "No valid Step 3 inputs after filtering Step 1/2 outputs"
    log "Prepared Step 3 shards: $added_count valid cases; skipped $skipped_count cases with missing/empty prerequisites"
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

mkdir -p "$TAGGED_DIR" "$TAGS_DIR" "$TEMPORAL_DIR" "$TEMPORAL_ANSWERS_DIR" \
    "$LLAMA_LOG_DIR" "$STEP1_LOG_DIR" "$STEP3_LOG_DIR" "$PARALLEL_WORK_DIR"

check_path "$PROJECT_DIR" "project directory"
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

validate_worker_count "STEP1_WORKERS" "$STEP1_WORKERS"
validate_worker_count "STEP3_WORKERS" "$STEP3_WORKERS"

init_file_selection
export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"
prepare_effective_input_dir
collect_input_filenames

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Input directory: $INPUT_DIR"
log_file_selection
log "Effective input directory: $EFFECTIVE_INPUT_DIR"
log "Output directory: $OUTPUT_DIR"
log "Input type: $INPUT_TYPE"
log "Model path: $INSTRUCT_MODEL_PATH"
log "Parallel server slots: $LLAMA_PARALLEL_SLOTS"
log "Step 1 workers: $STEP1_WORKERS"
log "Step 3 workers: $STEP3_WORKERS"
log "Total input files: ${#INPUT_CASE_FILES[@]}"

###############################################################################
# Step 1: tag clinical event mentions in parallel
###############################################################################

section_start="$(epoch_now)"
log "Starting llama-server for Step 1 with $LLAMA_PARALLEL_SLOTS slots"
start_parallel_server "$INSTRUCT_MODEL_PATH" "$INSTRUCT_MODEL_ALIAS" "$PORT"
trap cleanup_server EXIT

log "Step 1: sharding ${#INPUT_CASE_FILES[@]} input files across $STEP1_WORKERS workers"
make_case_list_shards "$PARALLEL_WORK_DIR/step1_case_lists" "$STEP1_WORKERS" "${INPUT_CASE_FILES[@]}"

WORKER_PIDS=()
WORKER_LABELS=()
for ((worker = 0; worker < STEP1_WORKERS; worker++)); do
    shard_file="${SHARD_FILES[$worker]}"
    worker_log="$STEP1_LOG_DIR/worker_${worker}.log"
    if [[ ! -s "$shard_file" ]]; then
        log "Step 1 worker $worker has no files; skipping"
        continue
    fi
    log "Step 1 worker $worker starting with $(wc -l < "$shard_file") files; log=$worker_log"
    python "$PROMPT_IO_SCRIPT" \
        --in_dir "$EFFECTIVE_INPUT_DIR" \
        --input_type "$INPUT_TYPE" \
        --case_list "$shard_file" \
        --out_dir "$TAGGED_DIR" \
        --prompt_file "$TAG_PROMPT_FILE" \
        --llm_mechanism llama.cpp \
        --lcpp_url "http://127.0.0.1:${PORT}" \
        --model_name "$INSTRUCT_MODEL_ALIAS" \
        --endpoint_type openai \
        --extra_body "$EXTRA_BODY" \
        --max_input_lines "$STEP1_MAX_INPUT_LINES" \
        --chunk_overlength \
        --chunk_line_count "$STEP1_CHUNK_LINES" \
        --out_extension "$OUT_EXTENSION" \
        > "$worker_log" 2>&1 &
    WORKER_PIDS+=("$!")
    WORKER_LABELS+=("step1_worker_${worker}")
done

if ! wait_for_worker_group "Step 1"; then
    log "WARNING: at least one Step 1 worker failed; continuing with any tagged reports that were produced. See $STEP1_LOG_DIR"
fi
log "Step 1 complete. Duration=$(format_duration "$(( $(epoch_now) - section_start ))")"

cleanup_server
archive_server_log "$STEP1_SERVER_LOG"
trap - EXIT

###############################################################################
# Step 2: extract tag table and assign uid4 IDs
###############################################################################

section_start="$(epoch_now)"
log "Step 2: extracting tag tables with extract_tagged_table.py"

EFFECTIVE_TAGGED_DIR="$TAGGED_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/tagged_${SLURM_JOB_ID:-manual}_$$" \
        "$TAGGED_DIR/{}.txt" \
        "$TAGGED_DIR/{}.txt.gz"
    EFFECTIVE_TAGGED_DIR="$STAGED_SELECTED_DIR"
fi

if ! python "$EXTRACT_TAGS_SCRIPT" \
    --input-dir "$EFFECTIVE_TAGGED_DIR" \
    --output-dir "$TAGS_DIR" \
    > "$STEP2_LOG" 2>&1; then
    log "WARNING: Step 2 extract_tagged_table.py returned a nonzero status; continuing with any tag tables that were produced"
fi

log "Step 2 complete. Duration=$(format_duration "$(( $(epoch_now) - section_start ))")"
validate_step12_outputs

###############################################################################
# Step 3: generate temporal tables in parallel
###############################################################################

section_start="$(epoch_now)"
log "Starting llama-server for Step 3 with $LLAMA_PARALLEL_SLOTS slots"
start_parallel_server "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS" "$PORT"
trap cleanup_server EXIT

log "Step 3: sharding tagged reports across $STEP3_WORKERS workers"
make_step3_shards "$STEP3_WORKERS" "$PARALLEL_WORK_DIR/step3_shards"

WORKER_PIDS=()
WORKER_LABELS=()
for ((worker = 0; worker < STEP3_WORKERS; worker++)); do
    case_dir="${STEP3_CASE_DIRS[$worker]}"
    tag_dir="${STEP3_TAG_DIRS[$worker]}"
    worker_log="$STEP3_LOG_DIR/worker_${worker}.log"
    file_count="$(find -L "$case_dir" -maxdepth 1 -type f \( -name "*.txt" -o -name "*.txt.gz" \) | wc -l)"
    if [[ "$file_count" -eq 0 ]]; then
        log "Step 3 worker $worker has no files; skipping"
        continue
    fi
    log "Step 3 worker $worker starting with $file_count files; log=$worker_log"
    python "$TEMPORAL_IO_SCRIPT" \
        --case-dir "$case_dir" \
        --tags-dir "$tag_dir" \
        --output-dir "$TEMPORAL_DIR" \
        --prompt-file "$TEMPORAL_PROMPT_FILE" \
        --llm_mechanism llama.cpp \
        --lcpp_url "http://127.0.0.1:${PORT}" \
        --keep_columns 1 2 \
        --tag_chunk_size "$STEP3_TAG_CHUNK_SIZE" \
        --endpoint_type openai \
        --model_name "$REASONING_MODEL_ALIAS" \
        --extra_body "$EXTRA_BODY" \
        > "$worker_log" 2>&1 &
    WORKER_PIDS+=("$!")
    WORKER_LABELS+=("step3_worker_${worker}")
done

if ! wait_for_worker_group "Step 3"; then
    log "WARNING: at least one Step 3 worker failed; continuing with any temporal outputs that were produced. See $STEP3_LOG_DIR"
fi
log "Step 3 complete. Duration=$(format_duration "$(( $(epoch_now) - section_start ))")"

validate_step3_outputs

cleanup_server
archive_server_log "$STEP3_SERVER_LOG"
trap - EXIT

###############################################################################
# Step 4: extract temporal tables
###############################################################################

section_start="$(epoch_now)"
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

log "Step 4 complete. Duration=$(format_duration "$(( $(epoch_now) - section_start ))")"
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
log "  step 1 workers: $STEP1_LOG_DIR"
log "  step 2: $STEP2_LOG"
log "  step 3 workers: $STEP3_LOG_DIR"
log "  step 4: $STEP4_LOG"
