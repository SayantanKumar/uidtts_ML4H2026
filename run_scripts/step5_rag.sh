#!/bin/bash
#SBATCH --job-name=uid_step5
#SBATCH --time=2-00:00:00
#SBATCH --mem=128g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch step5_rag.sh

set -eo pipefail

###############################################################################
# Environment
###############################################################################

module load CUDA/12.8.1 gcc cmake

set +u
eval "$(conda shell.bash hook)"
conda activate /data/grimalditersans/mymamba/envs/TTA
set -u

if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -n "${CUDA_PATH:-}" && -d "$CUDA_PATH/lib64" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# Avoid Biowulf/Squid proxy for local server calls.
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

###############################################################################
# Edit these paths/settings
###############################################################################

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export SERVER_UTILS="$PROJECT_DIR/llm_server_utils.sh"

export OUTPUT_DIR="$PROJECT_DIR/outputs/gpt-oss-120b/"

# Run either every completed upstream case or only the hard-coded subset below.
# Use the original input filenames or bare case ids.
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

# Folder containing per-patient structured EHR CSVs, e.g. <patient_id>.csv.gz.
# EDIT THIS if your tabular files are elsewhere.
export TABULAR_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n30_tabular/"

# File with note_id/hadm_id/admittime/dischtime mappings.
# EDIT THIS before running Step 5 source joining.
export NOTE_LINK_PATH="/data/grimalditersans/mimic_uid_data/note_link_i2m4.csv"

export LLAMA_SERVER="/data/grimalditersans/llama.cpp/build-cuda-gcc11/bin/llama-server"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"

export CUDA_DEVICES="0,1"
export LLAMA_TIMEOUT=1200

# Model roles. Both point to DeepSeek V3.2 for now; change either later if you
# want separate instruct/reasoning models. Step 5B uses the reasoning model for
# anchor-query generation.
export INSTRUCT_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/gpt-oss-120b-F16.gguf"
export INSTRUCT_MODEL_ALIAS="gpt-oss-120b"

export REASONING_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/gpt-oss-120b-F16.gguf"
export REASONING_MODEL_ALIAS="gpt-oss-120b"
export PORT=8081

# Local Hugging Face embedding/reranker setup. These are loaded directly with
# transformers; no sglang, vLLM, or embedding/reranker HTTP server is used.
#
# HF_HOME controls where Hugging Face downloads/caches models. Use a user-owned
# cache so transformers can create lock files and resume downloads safely.
# You can also replace EMBEDDING_MODEL/RERANKER_MODEL with full local snapshot
# paths if the models are already downloaded elsewhere.
export HF_HOME="/data/grimalditersans/.cache/huggingface/"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"

export HF_DEVICE="auto"
export EMBEDDING_BATCH_SIZE=8
export RERANKER_BATCH_SIZE=4
export HF_MAX_LENGTH=8192

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

export AQG_INFIX="uidtts"
export RAG_INFIX="uid_infix"
export TOP_K=10

export AQG_SYSTEM_PROMPT="$PROJECT_DIR/templates/aqg_system.txt"
export AQG_USER_PROMPT="$PROJECT_DIR/templates/aqg_uid.txt"

# Step 5B anchor-query generation. Large bundle reports are processed in UID
# batches to avoid LLM context/completion truncation.
export AQG_CHUNK_SIZE=25
export AQG_CONTEXT_WINDOW_CHARS=700
export AQG_MAX_REPORT_CHARS=25000
export AQG_MAX_QUERIES_PER_UID=3

###############################################################################
# Derived paths
###############################################################################

export TAGGED_DIR="$OUTPUT_DIR/tagged"
export TAGS_DIR="$OUTPUT_DIR/tags"
export STEP5_TAGGED_GZ_DIR="$OUTPUT_DIR/tagged_step5_gz"
export BUNDLE_DIR="$OUTPUT_DIR/bundle"
export AQG_DIR="$BUNDLE_DIR/$AQG_INFIX"
export RAG_OUTPUT_ROOT="$BUNDLE_DIR/tabular"
export RAG_DIR="$RAG_OUTPUT_ROOT/$RAG_INFIX"
export COMPARISON_DIR="$AQG_DIR/comparison_results"

MASK_SCRIPT="$PROJECT_DIR/mask_cr_append_mt.py"
ANCHOR_QUERY_SCRIPT="$PROJECT_DIR/anchor_query_generator.py"
RAG_PIPELINE_SCRIPT="$PROJECT_DIR/rag_pipeline_llama_embed_folder_summarized_hf.py"
ANCHOR_RANKER_SCRIPT="$PROJECT_DIR/anchor_match_ranker_hf.py"
VIEW_SCORES_SCRIPT="$PROJECT_DIR/view_match_scores.py"
JOIN_ANCHOR_SCRIPT="$PROJECT_DIR/join_anchor_with_source_timed.py"

STEP5_MASK_LOG="$LLAMA_LOG_DIR/step5a_mask_cr_append_mt.log"
STEP5_AQG_LOG="$LLAMA_LOG_DIR/step5b_anchor_query_generator.log"
STEP5_RAG_LOG="$LLAMA_LOG_DIR/step5c_rag_ingest.log"
STEP5_RANK_LOG="$LLAMA_LOG_DIR/step5d_anchor_ranker.log"
STEP5_JOIN_LOG="$LLAMA_LOG_DIR/step5e_flatten_join.log"

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

validate_step12_outputs() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    local tag_files=("$TAGS_DIR"/*_tags.bsv "$TAGS_DIR"/*_tags.bsv.gz)
    filter_selected_files_in_place tagged_files
    filter_selected_files_in_place tag_files
    local tagged_file
    local tag_file
    local base
    local valid_pair_count=0
    local skipped_pair_count=0
    local valid_tag_count=0
    local invalid_tag_count=0

    [[ "${#tagged_files[@]}" -gt 0 ]] || die "No tagged reports found in $TAGGED_DIR"
    [[ "${#tag_files[@]}" -gt 0 ]] || die "No tag tables found in $TAGS_DIR"

    for tagged_file in "${tagged_files[@]}"; do
        if ! has_nonblank_content "$tagged_file"; then
            log "WARNING: skipping invalid Step 1 tagged report for Step 5: $tagged_file"
            skipped_pair_count=$((skipped_pair_count + 1))
            continue
        fi
        base="$(case_base "$tagged_file")"
        if ! tag_file="$(find_tag_file_for_base "$base")"; then
            log "WARNING: skipping $base for Step 5; missing Step 2 tag table"
            skipped_pair_count=$((skipped_pair_count + 1))
            continue
        fi
        if ! has_nonblank_content "$tag_file"; then
            log "WARNING: skipping $base for Step 5; Step 2 tag table is empty or invalid: $tag_file"
            skipped_pair_count=$((skipped_pair_count + 1))
            continue
        fi
        valid_pair_count=$((valid_pair_count + 1))
    done

    for tag_file in "${tag_files[@]}"; do
        if has_nonblank_content "$tag_file"; then
            valid_tag_count=$((valid_tag_count + 1))
        else
            log "WARNING: invalid tag table will be ignored by Step 5: $tag_file"
            invalid_tag_count=$((invalid_tag_count + 1))
        fi
    done

    [[ "$valid_pair_count" -gt 0 ]] || die "No valid Step 1/2 tagged-report + tag-table pairs found for Step 5"

    log "Validated Step 1/2 outputs for Step 5: $valid_pair_count valid pairs; skipped $skipped_pair_count invalid/missing pairs; $valid_tag_count non-empty tag tables; $invalid_tag_count invalid tag tables"
}

validate_tabular_inputs() {
    shopt -s nullglob

    local tabular_files=("$TABULAR_DIR"/*.csv.gz)
    filter_selected_files_in_place tabular_files
    local file

    [[ "${#tabular_files[@]}" -gt 0 ]] || die "No .csv.gz tabular files found in $TABULAR_DIR"

    for file in "${tabular_files[@]}"; do
        has_nonblank_content "$file" || die "Tabular file is empty or invalid: $file"
    done

    log "Validated ${#tabular_files[@]} tabular input files"
}

stage_tagged_reports_as_gz() {
    shopt -s nullglob

    local tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
    filter_selected_files_in_place tagged_files
    local tagged_file
    local tag_file
    local base
    local dest
    local valid_tags_dir="$SELECTION_WORK_DIR/step5_valid_tags_${SLURM_JOB_ID:-manual}_$$"
    local staged_count=0
    local skipped_count=0

    rm -rf "$STEP5_TAGGED_GZ_DIR" "$valid_tags_dir"
    mkdir -p "$STEP5_TAGGED_GZ_DIR" "$valid_tags_dir"

    for tagged_file in "${tagged_files[@]}"; do
        base="$(case_base "$tagged_file")"
        dest="$STEP5_TAGGED_GZ_DIR/${base}.txt.gz"

        if ! has_nonblank_content "$tagged_file"; then
            log "WARNING: Step 5A skipping $base; tagged report is empty or invalid"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! tag_file="$(find_tag_file_for_base "$base")"; then
            log "WARNING: Step 5A skipping $base; missing tag table"
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if ! has_nonblank_content "$tag_file"; then
            log "WARNING: Step 5A skipping $base; tag table is empty or invalid: $tag_file"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        if has_nonblank_content "$dest"; then
            log "Skipping staged tagged report for $base; already exists"
        elif [[ "$tagged_file" == *.gz ]]; then
            cp -f "$tagged_file" "$dest"
        else
            gzip -c "$tagged_file" > "$dest"
        fi

        ln -sfn "$tag_file" "$valid_tags_dir/$(basename "$tag_file")"
        staged_count=$((staged_count + 1))
    done

    [[ "$staged_count" -gt 0 ]] || die "No valid Step 5A inputs after filtering Step 1/2 outputs"

    STEP5_EFFECTIVE_TAGS_DIR="$valid_tags_dir"
    log "Staged tagged reports for mask_cr_append_mt.py: $STEP5_TAGGED_GZ_DIR ($staged_count valid; $skipped_count skipped)"
}

cleanup_llm_server() {
    if [[ -n "${LLAMA_SERVER_PID:-}" ]]; then
        kill_server "$LLAMA_SERVER_PID" "$PORT" || true
        unset LLAMA_SERVER_PID
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

    matches=("$RAG_DIR/${RAG_INFIX}_${patient_id}"*.csv/summary_mapping.json)
    if [[ "${#matches[@]}" -gt 0 ]]; then
        echo "${matches[0]}"
        return 0
    fi

    return 1
}

SELECTION_UTILS="$PROJECT_DIR/run_scripts/file_selection_utils.sh"
if [[ ! -f "$SELECTION_UTILS" ]]; then
    RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELECTION_UTILS="$RUN_SCRIPT_DIR/file_selection_utils.sh"
fi
source "$SELECTION_UTILS"

find_line_level_file() {
    shopt -s nullglob

    local patient_id="$1"
    local exact="$TABULAR_DIR/${patient_id}.csv.gz"
    local matches

    if [[ -e "$exact" ]]; then
        echo "$exact"
        return 0
    fi

    matches=("$TABULAR_DIR/${patient_id}"*.csv.gz)
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

flatten_and_join_results() {
    shopt -s nullglob

    local result_files=("$COMPARISON_DIR"/*_results.json)
    filter_selected_files_in_place result_files
    local results_file
    local base_name
    local patient_id
    local summary_csv
    local joined_csv
    local mapping_file
    local line_level_file
    local joined_count=0
    local skipped_count=0
    local failed_count=0

    [[ "${#result_files[@]}" -gt 0 ]] || die "No ranker result JSON files found in $COMPARISON_DIR"

    : > "$STEP5_JOIN_LOG"

    for results_file in "${result_files[@]}"; do
        base_name="$(basename "$results_file" _results.json)"

        if ! has_nonblank_content "$results_file"; then
            log "WARNING: skipping $base_name; ranker result is empty: $results_file" | tee -a "$STEP5_JOIN_LOG"
            failed_count=$((failed_count + 1))
            continue
        fi

        if ! patient_id="$(subject_id_from_name "$base_name")"; then
            log "WARNING: skipping $base_name; could not resolve subject ID from filename" | tee -a "$STEP5_JOIN_LOG"
            failed_count=$((failed_count + 1))
            continue
        fi
        summary_csv="$COMPARISON_DIR/${patient_id}_summary.csv"
        joined_csv="$COMPARISON_DIR/${patient_id}_joined.csv"

        if has_nonblank_content "$joined_csv"; then
            log "Skipping $base_name; joined CSV already exists" | tee -a "$STEP5_JOIN_LOG"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        if has_nonblank_content "$summary_csv"; then
            log "Using existing score summary for $base_name" | tee -a "$STEP5_JOIN_LOG"
        else
            log "Flattening scores for $base_name" | tee -a "$STEP5_JOIN_LOG"
            if ! python "$VIEW_SCORES_SCRIPT" \
                --results-file "$results_file" \
                --output "$summary_csv" \
                --sort-by score \
                --overwrite \
                >> "$STEP5_JOIN_LOG" 2>&1; then
                log "WARNING: skipping $base_name; score flattening failed" | tee -a "$STEP5_JOIN_LOG"
                failed_count=$((failed_count + 1))
                continue
            fi
        fi

        if ! has_nonblank_content "$summary_csv"; then
            log "WARNING: skipping $base_name; summary CSV is empty: $summary_csv" | tee -a "$STEP5_JOIN_LOG"
            failed_count=$((failed_count + 1))
            continue
        fi

        if ! mapping_file="$(find_mapping_file "$patient_id")"; then
            log "WARNING: skipping $base_name; missing summary_mapping.json for patient_id=$patient_id" | tee -a "$STEP5_JOIN_LOG"
            failed_count=$((failed_count + 1))
            continue
        fi
        if ! line_level_file="$(find_line_level_file "$patient_id")"; then
            log "WARNING: skipping $base_name; missing tabular CSV for patient_id=$patient_id" | tee -a "$STEP5_JOIN_LOG"
            failed_count=$((failed_count + 1))
            continue
        fi

        log "Joining anchor matches back to source rows for $base_name" | tee -a "$STEP5_JOIN_LOG"
        if ! python "$JOIN_ANCHOR_SCRIPT" \
            --anchor-results "$summary_csv" \
            --summary-mapping "$mapping_file" \
            --line-level-data "$line_level_file" \
            --output "$joined_csv" \
            --note-adms "$NOTE_LINK_PATH" \
            --overwrite \
            >> "$STEP5_JOIN_LOG" 2>&1; then
            log "WARNING: skipping $base_name; join_anchor_with_source_timed.py failed" | tee -a "$STEP5_JOIN_LOG"
            failed_count=$((failed_count + 1))
            continue
        fi

        if ! has_nonblank_content "$joined_csv"; then
            log "WARNING: skipping $base_name; joined CSV is empty: $joined_csv" | tee -a "$STEP5_JOIN_LOG"
            failed_count=$((failed_count + 1))
            continue
        fi
        joined_count=$((joined_count + 1))
    done

    [[ $((joined_count + skipped_count)) -gt 0 ]] || die "No valid joined anchor-source CSV files were created or found"

    log "Created $joined_count joined anchor-source CSV files; skipped $skipped_count existing joined CSV files; failed $failed_count cases"
}

###############################################################################
# Preflight
###############################################################################

mkdir -p "$BUNDLE_DIR" "$AQG_DIR" "$RAG_OUTPUT_ROOT" "$COMPARISON_DIR" "$LLAMA_LOG_DIR"

check_path "$PROJECT_DIR" "project directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$TAGGED_DIR" "Step 1 tagged output directory"
check_path "$TAGS_DIR" "Step 2 tag table directory"
check_path "$TABULAR_DIR" "tabular input directory"
check_path "$NOTE_LINK_PATH" "note link/admission file"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$REASONING_MODEL_PATH" "reasoning model"
check_path "$AQG_SYSTEM_PROMPT" "anchor-query system prompt"
check_path "$AQG_USER_PROMPT" "anchor-query user prompt"
check_path "$MASK_SCRIPT" "mask_cr_append_mt.py"
check_path "$ANCHOR_QUERY_SCRIPT" "anchor_query_generator.py"
check_path "$RAG_PIPELINE_SCRIPT" "rag_pipeline_llama_embed_folder_summarized.py"
check_path "$ANCHOR_RANKER_SCRIPT" "anchor_match_ranker.py"
check_path "$VIEW_SCORES_SCRIPT" "view_match_scores.py"
check_path "$JOIN_ANCHOR_SCRIPT" "join_anchor_with_source_timed.py"

init_file_selection
export SELECTION_WORK_DIR="$OUTPUT_DIR/selection_work"
if selection_enabled; then
    export STEP5_TAGGED_GZ_DIR="$SELECTION_WORK_DIR/tagged_step5_gz_${SLURM_JOB_ID:-manual}_$$"
fi

source "$SERVER_UTILS"

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Output directory: $OUTPUT_DIR"
log_file_selection
log "Tabular directory: $TABULAR_DIR"
log "Using python: $(which python)"
log "Python version: $(python --version)"
log "Instruct model path: $INSTRUCT_MODEL_PATH"
log "Instruct model alias: $INSTRUCT_MODEL_ALIAS"
log "Reasoning model path: $REASONING_MODEL_PATH"
log "Reasoning model alias: $REASONING_MODEL_ALIAS"
log "Anchor-query model: $REASONING_MODEL_ALIAS"
log "Anchor-query chunk size: $AQG_CHUNK_SIZE"
log "Anchor-query context window chars: $AQG_CONTEXT_WINDOW_CHARS"
log "Anchor-query max report chars: $AQG_MAX_REPORT_CHARS"
log "Anchor-query max queries per UID: $AQG_MAX_QUERIES_PER_UID"
log "Embedding model: $EMBEDDING_MODEL"
log "Reranker model: $RERANKER_MODEL"
log "HF_HOME: $HF_HOME"
log "HF device: $HF_DEVICE"

validate_step12_outputs
validate_tabular_inputs

###############################################################################
# Step 5A: prepare UID-tagged case reports
###############################################################################

stage_tagged_reports_as_gz

log "Step 5A: preparing UID-tagged reports with mask_cr_append_mt.py"

EFFECTIVE_TAGS_DIR="${STEP5_EFFECTIVE_TAGS_DIR:-$TAGS_DIR}"

if ! python "$MASK_SCRIPT" \
    --mention_table_dir "$EFFECTIVE_TAGS_DIR/" \
    --tagged_case_report_dir "$STEP5_TAGGED_GZ_DIR/" \
    --output_dir "$BUNDLE_DIR/" \
    > "$STEP5_MASK_LOG" 2>&1; then
    log "WARNING: Step 5A mask_cr_append_mt.py returned a nonzero status; continuing with any bundle reports that were produced"
fi

shopt -s nullglob
bundle_reports=("$BUNDLE_DIR"/*.txt)
filter_selected_files_in_place bundle_reports
[[ "${#bundle_reports[@]}" -gt 0 ]] || die "No prepared bundle reports found in $BUNDLE_DIR"
log "Step 5A complete. Prepared ${#bundle_reports[@]} bundle reports"

###############################################################################
# Step 5B: generate anchor queries
###############################################################################

log "Starting llama-server for Step 5B anchor-query generation"
start_server "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS" "$PORT"
trap cleanup_llm_server EXIT

log "Step 5B: generating anchor queries"

EFFECTIVE_BUNDLE_DIR="$BUNDLE_DIR"
if selection_enabled; then
    EFFECTIVE_BUNDLE_DIR="$SELECTION_WORK_DIR/bundle_${SLURM_JOB_ID:-manual}_$$"
    mkdir -p "$EFFECTIVE_BUNDLE_DIR"
    stage_selected_files_for_bases \
        "$EFFECTIVE_BUNDLE_DIR" \
        "$BUNDLE_DIR/{}.txt" \
        "$BUNDLE_DIR/{}.txt.txt"
    ln -sfn "$AQG_DIR" "$EFFECTIVE_BUNDLE_DIR/$AQG_INFIX"
    ln -sfn "$RAG_OUTPUT_ROOT" "$EFFECTIVE_BUNDLE_DIR/tabular"
fi

if ! python "$ANCHOR_QUERY_SCRIPT" generate \
    --folder "$EFFECTIVE_BUNDLE_DIR" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --infix "$AQG_INFIX" \
    --system-prompt "$AQG_SYSTEM_PROMPT" \
    --user-prompt "$AQG_USER_PROMPT" \
    --model-name "$REASONING_MODEL_ALIAS" \
    --chunk-size "$AQG_CHUNK_SIZE" \
    --context-window-chars "$AQG_CONTEXT_WINDOW_CHARS" \
    --max-report-chars "$AQG_MAX_REPORT_CHARS" \
    --max-queries-per-uid "$AQG_MAX_QUERIES_PER_UID" \
    > "$STEP5_AQG_LOG" 2>&1; then
    log "WARNING: Step 5B anchor_query_generator.py returned a nonzero status; continuing with any anchor-query JSON files that were produced"
fi

cleanup_llm_server
trap - EXIT

anchor_jsons=("$AQG_DIR"/*.json)
filter_selected_files_in_place anchor_jsons
[[ "${#anchor_jsons[@]}" -gt 0 ]] || die "No anchor-query JSON files found in $AQG_DIR"
log "Step 5B complete. Anchor query files: ${#anchor_jsons[@]}"

###############################################################################
# Step 5C/D: RAG ingestion and anchor matching
###############################################################################

log "Step 5C: ingesting tabular data into RAG index with local HF embeddings"

EFFECTIVE_TABULAR_DIR="$TABULAR_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/tabular_${SLURM_JOB_ID:-manual}_$$" \
        "$TABULAR_DIR/{}.csv.gz"
    EFFECTIVE_TABULAR_DIR="$STAGED_SELECTED_DIR"
fi

if ! python "$RAG_PIPELINE_SCRIPT" \
    --folder "$EFFECTIVE_TABULAR_DIR/" \
    --output-folder "$RAG_OUTPUT_ROOT/" \
    --infix "$RAG_INFIX" \
    --embedding-model "$EMBEDDING_MODEL" \
    --device "$HF_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --max-length "$HF_MAX_LENGTH" \
    ingest \
    > "$STEP5_RAG_LOG" 2>&1; then
    log "WARNING: Step 5C RAG ingestion returned a nonzero status; continuing with any RAG outputs that were produced"
fi

rag_dirs=("$RAG_DIR"/*)
[[ "${#rag_dirs[@]}" -gt 0 ]] || die "No RAG outputs found in $RAG_DIR"
log "Step 5C complete. RAG outputs found in $RAG_DIR"

log "Step 5D: comparing anchor queries against RAG summaries with local HF models"

if ! python "$ANCHOR_RANKER_SCRIPT" compare \
    --folder "$EFFECTIVE_BUNDLE_DIR" \
    --anchor-subdir "$AQG_INFIX" \
    --rag-subdir "tabular/$RAG_INFIX/" \
    --k "$TOP_K" \
    --embedding-model "$EMBEDDING_MODEL" \
    --reranker-model "$RERANKER_MODEL" \
    --device "$HF_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --reranker-batch-size "$RERANKER_BATCH_SIZE" \
    --max-length "$HF_MAX_LENGTH" \
    > "$STEP5_RANK_LOG" 2>&1; then
    log "WARNING: Step 5D anchor_match_ranker.py returned a nonzero status; continuing with any comparison results that were produced"
fi

result_jsons=("$COMPARISON_DIR"/*_results.json)
filter_selected_files_in_place result_jsons
[[ "${#result_jsons[@]}" -gt 0 ]] || die "No comparison result JSON files found in $COMPARISON_DIR"
log "Step 5D complete. Comparison result files: ${#result_jsons[@]}"

###############################################################################
# Step 5E: flatten scores and join back to source rows
###############################################################################

log "Step 5E: flattening scores and joining to source tabular rows"
flatten_and_join_results

joined_files=("$COMPARISON_DIR"/*_joined.csv)
filter_selected_files_in_place joined_files
[[ "${#joined_files[@]}" -gt 0 ]] || die "No joined anchor-source CSV files found in $COMPARISON_DIR"

###############################################################################
# Summary
###############################################################################

log "Done."
log "Step 5 outputs:"
log "  Bundle reports: $BUNDLE_DIR"
log "  Anchor queries: $AQG_DIR"
log "  RAG indexes: $RAG_DIR"
log "  Comparison/joined results: $COMPARISON_DIR"
log "Logs:"
log "  mask: $STEP5_MASK_LOG"
log "  anchor query: $STEP5_AQG_LOG"
log "  RAG ingest: $STEP5_RAG_LOG"
log "  ranker: $STEP5_RANK_LOG"
log "  flatten/join: $STEP5_JOIN_LOG"





# X="/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/atp_tts/data/processed250729__AllData_by_subjectid/"
# Y="/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/LLM_TTS/new_annotations/files/"
# OUT="/Users/kumars33/Desktop/TTA/Textual_tabular_alignment-main/atp_tts/data/m4_new10_tabular/"

# mkdir -p "$OUT"

# find "$Y" -maxdepth 1 -type f -name "*.txt" -print0 |
# while IFS= read -r -d '' txt; do
#     base="$(basename "$txt")"

#     found=0

#     # Extract every numeric token from the txt filename, then keep the one
#     # that has a matching <subject_id>.csv.gz file in folder X.
#     while IFS= read -r subject_id; do
#         csv="$X/${subject_id}.csv.gz"

#         if [[ -f "$csv" ]]; then
#             cp -n "$csv" "$OUT/"
#             echo "Copied: $csv"
#             found=1
#             break
#         fi
#     done < <(printf '%s\n' "$base" | grep -oE '[0-9]+')

#     if [[ "$found" -eq 0 ]]; then
#         echo "Missing CSV for txt file: $base" >&2
#     fi
# done



# OUTPUT_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/outputs/qwen35"
# TABULAR_DIR="/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/i2m4_n30_tabular"

# DRY_RUN=0   # change to 0 after checking printed renames

# rename_one_dir() {
#     local dir="$1"
#     local suffix="$2"

#     echo "Checking: $dir"

#     find "$dir" -maxdepth 1 -type f -print0 |
#     while IFS= read -r -d '' file; do
#         fname="$(basename "$file")"

#         subject_id=""
#         while IFS= read -r token; do
#             if [[ -f "$TABULAR_DIR/${token}.csv.gz" ]]; then
#                 subject_id="$token"
#                 break
#             fi
#         done < <(printf '%s\n' "$fname" | grep -oE '[0-9]+')

#         if [[ -z "$subject_id" ]]; then
#             echo "No matching subject_id found: $file" >&2
#             continue
#         fi

#         new_file="$dir/${subject_id}${suffix}"

#         if [[ "$file" == "$new_file" ]]; then
#             continue
#         fi

#         if [[ -e "$new_file" ]]; then
#             echo "SKIP, target exists: $new_file" >&2
#             continue
#         fi

#         echo "mv '$file' '$new_file'"

#         if [[ "$DRY_RUN" -eq 0 ]]; then
#             mv "$file" "$new_file"
#         fi
#     done
# }

# rename_one_dir "$OUTPUT_DIR/tagged" ".txt"
# rename_one_dir "$OUTPUT_DIR/tags" "_tags.bsv.gz"
# rename_one_dir "$OUTPUT_DIR/temporal" ".txt.gz"
# rename_one_dir "$OUTPUT_DIR/temporal/answers" "_extracted.bsv.gz"


# OUT=/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection/outputs/qwen35

# watch -n 30 "
# echo '5A bundle reports:'; find $OUT/bundle -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l
# echo '5B anchor jsons:'; find $OUT/bundle/uidtts -maxdepth 1 -name '*.json' 2>/dev/null | wc -l
# echo '5C RAG mappings:'; find $OUT/bundle/tabular/uid_infix -name 'summary_mapping.json' 2>/dev/null | wc -l
# echo '5D ranker results:'; find $OUT/bundle/uidtts/comparison_results -name '*_results.json' 2>/dev/null | wc -l
# echo '5E joined csvs:'; find $OUT/bundle/uidtts/comparison_results -name '*_joined.csv' 2>/dev/null | wc -l
# "
