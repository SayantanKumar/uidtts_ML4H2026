#!/bin/bash
#SBATCH --job-name=uid_all_steps
#SBATCH --time=4-00:00:00
#SBATCH --mem=64g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:4
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=grimalditersans@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch run_all_uid_pipeline.sh

set -uo pipefail
shopt -s nullglob

###############################################################################
# Environment
###############################################################################

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

export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"

###############################################################################
# Edit these paths/settings
###############################################################################

export PROJECT_DIR="/data/grimalditersans/uid_anchor_detection"
export SERVER_UTILS="$PROJECT_DIR/llm_server_utils.sh"

#export INPUT_DIR="/data/CHARM-MIMIC/grimalditersans/LLM_pratice/Textual_tabular_alignment-main/multimodal_simple/new_annotations/files"
export INPUT_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n30_notes/"
export INPUT_TYPE="txt"
export OUT_EXTENSION=".txt"

# Run either every .txt file in INPUT_DIR or only the hard-coded subset below.
# Use FILE_SELECTION_MODE="subset" and list filenames exactly as they appear in
# INPUT_DIR. You may also list bare case ids, e.g. "11121324".
export FILE_SELECTION_MODE="all"  # all|subset
SELECTED_INPUT_FILES=(
    # "11121324.txt"
    # "11649157.txt"
)

export OUTPUT_DIR="$PROJECT_DIR/outputs/qwen35/"

export TABULAR_DIR="/data/grimalditersans/mimic_uid_data/i2m4_n30_tabular/"
export NOTE_LINK_PATH="/data/grimalditersans/mimic_uid_data/note_link_i2m4.csv"

export TAG_PROMPT_FILE="$PROJECT_DIR/templates/tts_tag.txt"
export TEMPORAL_PROMPT_FILE="$PROJECT_DIR/templates/temporal_prompt.txt"
export AQG_SYSTEM_PROMPT="$PROJECT_DIR/templates/aqg_system.txt"
export AQG_USER_PROMPT="$PROJECT_DIR/templates/aqg_uid.txt"
export AJT_PROMPT_FILE="$PROJECT_DIR/templates/ajt.txt"

export LLAMA_SERVER="/data/grimalditersans/llama.cpp/build-cuda-gcc11/bin/llama-server"
export LLAMA_LOG_DIR="$OUTPUT_DIR/logs"
export CUDA_DEVICES="0,1,2,3"
export LLAMA_TIMEOUT=1200
export PORT=8080

export INSTRUCT_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
export INSTRUCT_MODEL_ALIAS="qwen35"

export REASONING_MODEL_PATH="/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf"
export REASONING_MODEL_ALIAS="qwen35"

export EXTRA_BODY=""

# Step 1 long-note controls.
export STEP1_MAX_INPUT_LINES=500
export STEP1_CHUNK_LINES=450

# Step 3 long-case handling. If a tagged report has many extracted mentions,
# process the temporal table in mention batches and merge the final output.
export STEP3_TAG_CHUNK_SIZE=60

# Local Hugging Face embedding/reranker setup for Step 5C/D.
export HF_HOME="/data/grimalditersans/.cache/huggingface/"  # what is this?
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
export RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"
export HF_DEVICE="auto"
export EMBEDDING_BATCH_SIZE=8
export RERANKER_BATCH_SIZE=4
export HF_MAX_LENGTH=8192

export AQG_INFIX="uidtts"
export RAG_INFIX="uid_infix"
export TOP_K=10

# Step 5B anchor-query controls.
export AQG_CHUNK_SIZE=25
export AQG_CONTEXT_WINDOW_CHARS=700
export AQG_MAX_REPORT_CHARS=25000
export AQG_MAX_QUERIES_PER_UID=3

# Step 6 final timeline controls.
export TAU=0.05
export STEP6_MAX_TOTAL_CANDIDATES=100
export STEP6_TOP_K=3
export STEP6_NUM_TIMELINES=3
export STEP6_MAX_CANDIDATE_FIELD_CHARS=160
export STEP6_MAX_CASE_REPORT_CHARS=30000
export STEP6_CASE_CONTEXT_CHARS=220

###############################################################################
# Derived paths
###############################################################################

export TAGGED_DIR="$OUTPUT_DIR/tagged"
export TAGS_DIR="$OUTPUT_DIR/tags"
export TEMPORAL_DIR="$OUTPUT_DIR/temporal"
export TEMPORAL_ANSWERS_DIR="$TEMPORAL_DIR/answers"
export STEP5_TAGGED_GZ_DIR="$OUTPUT_DIR/tagged_step5_gz"
export BUNDLE_DIR="$OUTPUT_DIR/bundle"
export AQG_DIR="$BUNDLE_DIR/$AQG_INFIX"
export RAG_OUTPUT_ROOT="$BUNDLE_DIR/tabular"
export RAG_DIR="$RAG_OUTPUT_ROOT/$RAG_INFIX"
export COMPARISON_DIR="$AQG_DIR/comparison_results"
export TIMELINES_DIR="$OUTPUT_DIR/timelines"
export CHARPOS_DIR="$BUNDLE_DIR/charpos"
export CHARPOS_UNREVISED_DIR="$BUNDLE_DIR/charpos_unrevised"
export WORK_DIR="$OUTPUT_DIR/pipeline_work"

PROMPT_IO_SCRIPT="$PROJECT_DIR/prompt_io.py"
EXTRACT_TAGS_SCRIPT="$PROJECT_DIR/extract_tagged_table.py"
TEMPORAL_IO_SCRIPT="$PROJECT_DIR/temporal_prompt_io.py"
EXTRACT_TEMPORAL_SCRIPT="$PROJECT_DIR/extract_temporal_table.py"
MASK_SCRIPT="$PROJECT_DIR/mask_cr_append_mt.py"
ANCHOR_QUERY_SCRIPT="$PROJECT_DIR/anchor_query_generator.py"
RAG_PIPELINE_SCRIPT="$PROJECT_DIR/rag_pipeline_llama_embed_folder_summarized_hf.py"
ANCHOR_RANKER_SCRIPT="$PROJECT_DIR/anchor_match_ranker_hf.py"
VIEW_SCORES_SCRIPT="$PROJECT_DIR/view_match_scores.py"
JOIN_ANCHOR_SCRIPT="$PROJECT_DIR/join_anchor_with_source_timed.py"
ASSIGN_TIMESTAMPS_SCRIPT="$PROJECT_DIR/assign_joint_timestamps.py"
EXTRACT_TIMELINES_SCRIPT="$PROJECT_DIR/extract_timelines.py"
GET_CHARPOS_SCRIPT="$PROJECT_DIR/get_tag_charpos_table.py"

PIPELINE_LOG="$LLAMA_LOG_DIR/all_pipeline.log"
SUMMARY_TSV="$LLAMA_LOG_DIR/all_pipeline_summary.tsv"
FAILED_CASES_TSV="$LLAMA_LOG_DIR/all_pipeline_failed_cases.tsv"

STEP1_LOG="$LLAMA_LOG_DIR/step1_prompt_io.log"
STEP2_LOG="$LLAMA_LOG_DIR/step2_extract_tags.log"
STEP3_LOG="$LLAMA_LOG_DIR/step3_temporal_prompt_io.log"
STEP4_LOG="$LLAMA_LOG_DIR/step4_extract_temporal_table.log"
STEP5A_LOG="$LLAMA_LOG_DIR/step5a_mask_cr_append_mt.log"
STEP5B_LOG="$LLAMA_LOG_DIR/step5b_anchor_query_generator.log"
STEP5C_LOG="$LLAMA_LOG_DIR/step5c_rag_ingest.log"
STEP5D_LOG="$LLAMA_LOG_DIR/step5d_anchor_ranker.log"
STEP5E_LOG="$LLAMA_LOG_DIR/step5e_flatten_join.log"
STEP6_LOG="$LLAMA_LOG_DIR/step6_assign_joint_timestamps.log"
STEP6_RETRY_LOG_DIR="$LLAMA_LOG_DIR/step6_retries"
POST_LOG="$LLAMA_LOG_DIR/post_processing.log"

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
    echo "[$(timestamp)] $*" | tee -a "$PIPELINE_LOG"
}

die() {
    log "ERROR: $*"
    exit 1
}

check_path() {
    local path="$1"
    local label="$2"
    [[ -e "$path" ]] || die "Missing $label: $path"
}

start_section() {
    SECTION_NAME="$1"
    SECTION_START_EPOCH="$(epoch_now)"
    SECTION_SUCCESS=0
    SECTION_FAILED=0
    SECTION_SKIPPED=0
    log "===== START $SECTION_NAME ====="
}

end_section() {
    local end_epoch
    local duration
    end_epoch="$(epoch_now)"
    duration=$((end_epoch - SECTION_START_EPOCH))
    log "===== END $SECTION_NAME | success=$SECTION_SUCCESS failed=$SECTION_FAILED skipped=$SECTION_SKIPPED duration=$(format_duration "$duration") ====="
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$SECTION_NAME" "$SECTION_START_EPOCH" "$end_epoch" "$(format_duration "$duration")" \
        "$SECTION_SUCCESS" "$SECTION_FAILED" "$SECTION_SKIPPED" >> "$SUMMARY_TSV"
}

record_failure() {
    local step="$1"
    local case_id="$2"
    local reason="$3"
    local path="${4:-}"
    printf "%s\t%s\t%s\t%s\t%s\n" "$(timestamp)" "$step" "$case_id" "$reason" "$path" >> "$FAILED_CASES_TSV"
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

output_suffix() {
    local base_ext
    base_ext="${OUT_EXTENSION%.gz}"
    if [[ "$OUT_EXTENSION" == *.gz ]]; then
        echo "${base_ext}.gz"
    else
        echo "$base_ext"
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


find_mapping_file() {
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

find_line_level_file() {
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

has_valid_anchor_json() {
    local file="$1"
    [[ -s "$file" ]] || return 1
    python - "$file" <<'PY' >/dev/null 2>&1
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
if not isinstance(data, dict) or not data:
    sys.exit(1)
if len(data) == 1 and data.get(" ") is None:
    sys.exit(1)
for key, values in data.items():
    if key and isinstance(values, list) and any(str(v).strip() for v in values):
        sys.exit(0)
sys.exit(1)
PY
}

has_valid_timeline_json() {
    local file="$1"
    [[ -s "$file" ]] || return 1
    python - "$file" <<'PY' >/dev/null 2>&1
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
timelines = data.get("timelines") if isinstance(data, dict) else None
if not isinstance(timelines, list) or not timelines:
    sys.exit(1)
for item in timelines:
    if not isinstance(item, dict):
        sys.exit(1)
    table = item.get("data")
    if not isinstance(table, str) or len(table.strip()) < 20:
        sys.exit(1)
    lines = [line for line in table.splitlines() if line.strip()]
    if len(lines) < 2 or "|" not in lines[0]:
        sys.exit(1)
sys.exit(0)
PY
}

cleanup_server() {
    if [[ -n "${LLAMA_SERVER_PID:-}" ]]; then
        kill_server "$LLAMA_SERVER_PID" "$PORT" || true
        unset LLAMA_SERVER_PID
    fi
}

run_logged() {
    local log_file="$1"
    shift
    "$@" >> "$log_file" 2>&1
}

make_single_input_dirs() {
    local base="$1"
    local source_file="$2"
    local tag_file="${3:-}"
    SINGLE_CASE_DIR="$WORK_DIR/single_case/$base"
    SINGLE_TAG_DIR="$WORK_DIR/single_tags/$base"
    rm -rf "$SINGLE_CASE_DIR" "$SINGLE_TAG_DIR"
    mkdir -p "$SINGLE_CASE_DIR" "$SINGLE_TAG_DIR"
    ln -sf "$source_file" "$SINGLE_CASE_DIR/$(basename "$source_file")"
    if [[ -n "$tag_file" ]]; then
        ln -sf "$tag_file" "$SINGLE_TAG_DIR/$(basename "$tag_file")"
    fi
}

select_bundle_pattern() {
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

SELECTION_UTILS="$PROJECT_DIR/run_scripts/file_selection_utils.sh"
if [[ ! -f "$SELECTION_UTILS" ]]; then
    RUN_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SELECTION_UTILS="$RUN_SCRIPT_DIR/file_selection_utils.sh"
fi
source "$SELECTION_UTILS"

rename_unrevised_charpos_outputs() {
    local file
    for file in "$CHARPOS_UNREVISED_DIR"/*_extracted.bsv; do
        local target="${file%_extracted.bsv}_1.bsv"
        [[ -e "$target" ]] || mv "$file" "$target"
    done
    for file in "$CHARPOS_UNREVISED_DIR"/*_timeline_extracted.bsv; do
        local target="${file%_timeline_extracted.bsv}_1.bsv"
        [[ -e "$target" ]] || mv "$file" "$target"
    done
}

###############################################################################
# Preflight
###############################################################################

mkdir -p "$LLAMA_LOG_DIR" "$TAGGED_DIR" "$TAGS_DIR" "$TEMPORAL_DIR" "$TEMPORAL_ANSWERS_DIR" \
    "$STEP5_TAGGED_GZ_DIR" "$BUNDLE_DIR" "$AQG_DIR" "$RAG_OUTPUT_ROOT" "$COMPARISON_DIR" \
    "$TIMELINES_DIR" "$CHARPOS_DIR" "$CHARPOS_UNREVISED_DIR" "$WORK_DIR" \
    "$STEP6_RETRY_LOG_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE"

: > "$PIPELINE_LOG"
printf "step\tstart_epoch\tend_epoch\tduration\tsuccess\tfailed\tskipped\n" > "$SUMMARY_TSV"
printf "timestamp\tstep\tcase_id\treason\tpath\n" > "$FAILED_CASES_TSV"

check_path "$PROJECT_DIR" "project directory"
check_path "$SERVER_UTILS" "llm_server_utils.sh"
check_path "$INPUT_DIR" "input directory"
check_path "$TABULAR_DIR" "tabular directory"
check_path "$NOTE_LINK_PATH" "note link file"
check_path "$LLAMA_SERVER" "llama-server binary"
check_path "$INSTRUCT_MODEL_PATH" "instruct model"
check_path "$REASONING_MODEL_PATH" "reasoning model"
check_path "$TAG_PROMPT_FILE" "tag prompt"
check_path "$TEMPORAL_PROMPT_FILE" "temporal prompt"
check_path "$AQG_SYSTEM_PROMPT" "AQG system prompt"
check_path "$AQG_USER_PROMPT" "AQG user prompt"
check_path "$AJT_PROMPT_FILE" "AJT prompt"

for script in "$PROMPT_IO_SCRIPT" "$EXTRACT_TAGS_SCRIPT" "$TEMPORAL_IO_SCRIPT" "$EXTRACT_TEMPORAL_SCRIPT" \
    "$MASK_SCRIPT" "$ANCHOR_QUERY_SCRIPT" "$RAG_PIPELINE_SCRIPT" "$ANCHOR_RANKER_SCRIPT" "$VIEW_SCORES_SCRIPT" \
    "$JOIN_ANCHOR_SCRIPT" "$ASSIGN_TIMESTAMPS_SCRIPT" "$EXTRACT_TIMELINES_SCRIPT" "$GET_CHARPOS_SCRIPT"; do
    check_path "$script" "script"
done

init_file_selection
export SELECTION_WORK_DIR="$WORK_DIR/selection"
prepare_effective_input_dir

source "$SERVER_UTILS"
trap cleanup_server EXIT

log "Job ID: ${SLURM_JOB_ID:-manual}"
log "Project directory: $PROJECT_DIR"
log "Input directory: $INPUT_DIR"
log_file_selection
log "Output directory: $OUTPUT_DIR"
log "Tabular directory: $TABULAR_DIR"
log "Using python: $(which python)"
log "Python version: $(python --version)"
log "Instruct model: $INSTRUCT_MODEL_PATH ($INSTRUCT_MODEL_ALIAS)"
log "Reasoning model: $REASONING_MODEL_PATH ($REASONING_MODEL_ALIAS)"
log "HF embedding model: $EMBEDDING_MODEL"
log "HF reranker model: $RERANKER_MODEL"

###############################################################################
# LLM phase 1: Steps 1, 3, and 5B use one long-lived llama-server.
###############################################################################

start_section "llm_server_phase_1_start"
if start_server "$INSTRUCT_MODEL_PATH" "$INSTRUCT_MODEL_ALIAS" "$PORT" >> "$PIPELINE_LOG" 2>&1; then
    SECTION_SUCCESS=1
else
    SECTION_FAILED=1
    end_section
    die "Could not start llama-server for Steps 1/3/5B"
fi
end_section

###############################################################################
# NOTE: Step 1/2 sections below are intentionally commented out; existing Step 1/2 outputs are used and execution resumes at Step 3.
# # Step 1
# ###############################################################################
#
# start_section "step1_tag_clinical_mentions"
# input_ext="$(input_extension)"
# out_suffix="$(output_suffix)"
# input_files=("$EFFECTIVE_INPUT_DIR"/*"$input_ext")
# filter_selected_files_in_place input_files
# [[ "${#input_files[@]}" -gt 0 ]] || die "Step 1 prerequisite failed: no input files matching *$input_ext in $EFFECTIVE_INPUT_DIR"
#
# for input_file in "${input_files[@]}"; do
#     base="$(case_base "$input_file")"
#     tagged_out="$TAGGED_DIR/${base}${out_suffix}"
#     log "Step 1 running: $base"
#     case_start="$(epoch_now)"
#
#     if has_nonblank_content "$tagged_out"; then
#         log "Step 1 skip existing: $base"
#         SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
#         continue
#     fi
#
#     case_list="$WORK_DIR/step1_${base}.case_list"
#     printf "%s\n" "$(basename "$input_file")" > "$case_list"
#
#     if run_logged "$STEP1_LOG" python "$PROMPT_IO_SCRIPT" \
#         --in_dir "$EFFECTIVE_INPUT_DIR" \
#         --input_type "$INPUT_TYPE" \
#         --case_list "$case_list" \
#         --out_dir "$TAGGED_DIR" \
#         --prompt_file "$TAG_PROMPT_FILE" \
#         --llm_mechanism llama.cpp \
#         --lcpp_url "http://127.0.0.1:${PORT}" \
#         --model_name "$INSTRUCT_MODEL_ALIAS" \
#         --endpoint_type openai \
#         --extra_body "$EXTRA_BODY" \
#         --max_input_lines "$STEP1_MAX_INPUT_LINES" \
#         --chunk_overlength \
#         --chunk_line_count "$STEP1_CHUNK_LINES" \
#         --out_extension "$OUT_EXTENSION" \
#         && has_nonblank_content "$tagged_out"; then
#         duration=$(( $(epoch_now) - case_start ))
#         log "Step 1 success: $base duration=$(format_duration "$duration")"
#         SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
#     else
#         duration=$(( $(epoch_now) - case_start ))
#         log "Step 1 failed: $base duration=$(format_duration "$duration")"
#         record_failure "step1" "$base" "tagging_failed_or_empty_output" "$tagged_out"
#         SECTION_FAILED=$((SECTION_FAILED + 1))
#         [[ -e "$tagged_out" ]] && [[ ! -s "$tagged_out" ]] && rm -f "$tagged_out"
#     fi
# done
# end_section
#
# ###############################################################################
# # Step 2
# ###############################################################################
#
# start_section "step2_extract_uid_tags"
# tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
# filter_selected_files_in_place tagged_files
# [[ "${#tagged_files[@]}" -gt 0 ]] || die "Step 2 prerequisite failed: no tagged reports in $TAGGED_DIR"
#
# for tagged_file in "${tagged_files[@]}"; do
#     base="$(case_base "$tagged_file")"
#     tag_out="$TAGS_DIR/${base}_tags.bsv.gz"
#     log "Step 2 running: $base"
#     case_start="$(epoch_now)"
#
#     if has_nonblank_content "$tag_out"; then
#         log "Step 2 skip existing: $base"
#         SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
#         continue
#     fi
#
#     if run_logged "$STEP2_LOG" python "$EXTRACT_TAGS_SCRIPT" \
#         --input "$tagged_file" \
#         --output-dir "$TAGS_DIR" \
#         && has_nonblank_content "$tag_out"; then
#         duration=$(( $(epoch_now) - case_start ))
#         log "Step 2 success: $base duration=$(format_duration "$duration")"
#         SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
#     else
#         duration=$(( $(epoch_now) - case_start ))
#         log "Step 2 failed: $base duration=$(format_duration "$duration")"
#         record_failure "step2" "$base" "tag_extraction_failed_or_empty_output" "$tag_out"
#         SECTION_FAILED=$((SECTION_FAILED + 1))
#     fi
# done
# end_section
#
# ###############################################################################
# Step 3
###############################################################################

start_section "step3_temporal_llm"
tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
filter_selected_files_in_place tagged_files
[[ "${#tagged_files[@]}" -gt 0 ]] || die "Step 3 prerequisite failed: no tagged reports in $TAGGED_DIR"

for tagged_file in "${tagged_files[@]}"; do
    base="$(case_base "$tagged_file")"
    tag_file="$(find_tag_file_for_base "$base" || true)"
    temporal_out="$TEMPORAL_DIR/${base}.txt.gz"
    log "Step 3 running: $base"
    case_start="$(epoch_now)"

    if [[ -z "$tag_file" ]] || ! has_nonblank_content "$tag_file"; then
        log "Step 3 failed prerequisite for $base: missing/nonblank tag table"
        record_failure "step3" "$base" "missing_or_empty_tag_table" "${tag_file:-}"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        continue
    fi

    if has_nonblank_content "$temporal_out"; then
        log "Step 3 skip existing: $base"
        SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
        continue
    fi

    make_single_input_dirs "$base" "$tagged_file" "$tag_file"
    if run_logged "$STEP3_LOG" python "$TEMPORAL_IO_SCRIPT" \
        --case-dir "$SINGLE_CASE_DIR" \
        --tags-dir "$SINGLE_TAG_DIR" \
        --output-dir "$TEMPORAL_DIR" \
        --prompt-file "$TEMPORAL_PROMPT_FILE" \
        --llm_mechanism llama.cpp \
        --lcpp_url "http://127.0.0.1:${PORT}" \
        --keep_columns 1 2 \
        --tag_chunk_size "$STEP3_TAG_CHUNK_SIZE" \
        --endpoint_type openai \
        --model_name "$REASONING_MODEL_ALIAS" \
        --extra_body "$EXTRA_BODY" \
        && has_nonblank_content "$temporal_out"; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 3 success: $base duration=$(format_duration "$duration")"
        SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
    else
        duration=$(( $(epoch_now) - case_start ))
        log "Step 3 failed: $base duration=$(format_duration "$duration")"
        record_failure "step3" "$base" "temporal_llm_failed_or_empty_output" "$temporal_out"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        [[ -e "$temporal_out" ]] && [[ ! -s "$temporal_out" ]] && rm -f "$temporal_out"
    fi
done
end_section

###############################################################################
# Step 4
###############################################################################

start_section "step4_extract_temporal_tables"
temporal_files=("$TEMPORAL_DIR"/*.txt "$TEMPORAL_DIR"/*.txt.gz)
filter_selected_files_in_place temporal_files
[[ "${#temporal_files[@]}" -gt 0 ]] || die "Step 4 prerequisite failed: no temporal LLM outputs in $TEMPORAL_DIR"

for temporal_file in "${temporal_files[@]}"; do
    base="$(case_base "$temporal_file")"
    extracted_out="$TEMPORAL_ANSWERS_DIR/${base}_extracted.bsv.gz"
    log "Step 4 running: $base"
    case_start="$(epoch_now)"

    if has_nonblank_content "$extracted_out"; then
        log "Step 4 skip existing: $base"
        SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
        continue
    fi

    tmp_temporal_dir="$WORK_DIR/step4_temporal/$base"
    rm -rf "$tmp_temporal_dir"
    mkdir -p "$tmp_temporal_dir"
    ln -sf "$temporal_file" "$tmp_temporal_dir/$(basename "$temporal_file")"

    if run_logged "$STEP4_LOG" python "$EXTRACT_TEMPORAL_SCRIPT" \
        --input-dir "$tmp_temporal_dir" \
        --output-dir "$TEMPORAL_ANSWERS_DIR" \
        --keep-columns uid4 mention time bounds known "context uid4s" \
        --include-header \
        --no-header \
        && has_nonblank_content "$extracted_out"; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 4 success: $base duration=$(format_duration "$duration")"
        SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
    else
        duration=$(( $(epoch_now) - case_start ))
        log "Step 4 failed: $base duration=$(format_duration "$duration")"
        record_failure "step4" "$base" "temporal_extraction_failed_or_empty_output" "$extracted_out"
        SECTION_FAILED=$((SECTION_FAILED + 1))
    fi
done
end_section

###############################################################################
# Step 5A
###############################################################################

start_section "step5a_prepare_uid_bundles"
tagged_files=("$TAGGED_DIR"/*.txt "$TAGGED_DIR"/*.txt.gz)
filter_selected_files_in_place tagged_files
[[ "${#tagged_files[@]}" -gt 0 ]] || die "Step 5A prerequisite failed: no tagged reports in $TAGGED_DIR"

for tagged_file in "${tagged_files[@]}"; do
    base="$(case_base "$tagged_file")"
    tag_file="$(find_tag_file_for_base "$base" || true)"
    staged_gz="$STEP5_TAGGED_GZ_DIR/${base}.txt.gz"
    bundle_out="$BUNDLE_DIR/${base}.txt"
    uid4_out="$BUNDLE_DIR/${base}_uid4.bsv"
    log "Step 5A running: $base"
    case_start="$(epoch_now)"

    if [[ -z "$tag_file" ]] || ! has_nonblank_content "$tag_file"; then
        log "Step 5A failed prerequisite for $base: missing/nonblank tag table"
        record_failure "step5a" "$base" "missing_or_empty_tag_table" "${tag_file:-}"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        continue
    fi

    if has_nonblank_content "$bundle_out" && has_nonblank_content "$uid4_out"; then
        log "Step 5A skip existing: $base"
        SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
        continue
    fi

    if [[ "$tagged_file" == *.gz ]]; then
        cp -f "$tagged_file" "$staged_gz"
    else
        gzip -c "$tagged_file" > "$staged_gz"
    fi

    tmp_case_dir="$WORK_DIR/step5a_case/$base"
    tmp_tag_dir="$WORK_DIR/step5a_tags/$base"
    rm -rf "$tmp_case_dir" "$tmp_tag_dir"
    mkdir -p "$tmp_case_dir" "$tmp_tag_dir"
    ln -sf "$staged_gz" "$tmp_case_dir/${base}.txt.gz"
    if [[ "$tag_file" == *.gz ]]; then
        ln -sf "$tag_file" "$tmp_tag_dir/${base}_tags.bsv.gz"
    else
        gzip -c "$tag_file" > "$tmp_tag_dir/${base}_tags.bsv.gz"
    fi

    if run_logged "$STEP5A_LOG" python "$MASK_SCRIPT" \
        --mention_table_dir "$tmp_tag_dir/" \
        --tagged_case_report_dir "$tmp_case_dir/" \
        --output_dir "$BUNDLE_DIR/" \
        && has_nonblank_content "$bundle_out" && has_nonblank_content "$uid4_out"; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5A success: $base duration=$(format_duration "$duration")"
        SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
    else
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5A failed: $base duration=$(format_duration "$duration")"
        record_failure "step5a" "$base" "bundle_preparation_failed_or_empty_output" "$bundle_out"
        SECTION_FAILED=$((SECTION_FAILED + 1))
    fi
done
end_section

###############################################################################
# Step 5B
###############################################################################

start_section "step5b_generate_anchor_queries"
bundle_reports=("$BUNDLE_DIR"/*.txt)
filter_selected_files_in_place bundle_reports
[[ "${#bundle_reports[@]}" -gt 0 ]] || die "Step 5B prerequisite failed: no bundle reports in $BUNDLE_DIR"

for bundle_file in "${bundle_reports[@]}"; do
    base="$(case_base "$bundle_file")"
    anchor_out="$AQG_DIR/${base}.json"
    log "Step 5B running: $base"
    case_start="$(epoch_now)"

    if has_valid_anchor_json "$anchor_out"; then
        log "Step 5B skip existing: $base"
        SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
        continue
    fi

    tmp_bundle_dir="$WORK_DIR/step5b_bundle/$base"
    rm -rf "$tmp_bundle_dir"
    mkdir -p "$tmp_bundle_dir"
    ln -sf "$bundle_file" "$tmp_bundle_dir/$(basename "$bundle_file")"

    if run_logged "$STEP5B_LOG" python "$ANCHOR_QUERY_SCRIPT" generate \
        --folder "$tmp_bundle_dir" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --infix "$AQG_INFIX" \
        --system-prompt "$AQG_SYSTEM_PROMPT" \
        --user-prompt "$AQG_USER_PROMPT" \
        --model-name "$REASONING_MODEL_ALIAS" \
        --chunk-size "$AQG_CHUNK_SIZE" \
        --context-window-chars "$AQG_CONTEXT_WINDOW_CHARS" \
        --max-report-chars "$AQG_MAX_REPORT_CHARS" \
        --max-queries-per-uid "$AQG_MAX_QUERIES_PER_UID"; then
        generated_anchor="$tmp_bundle_dir/$AQG_INFIX/${base}.json"
        if has_valid_anchor_json "$generated_anchor"; then
            cp -f "$generated_anchor" "$anchor_out"
            duration=$(( $(epoch_now) - case_start ))
            log "Step 5B success: $base duration=$(format_duration "$duration")"
            SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
        else
            duration=$(( $(epoch_now) - case_start ))
            log "Step 5B failed validation: $base duration=$(format_duration "$duration")"
            record_failure "step5b" "$base" "anchor_json_missing_or_invalid" "$generated_anchor"
            SECTION_FAILED=$((SECTION_FAILED + 1))
        fi
    else
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5B failed: $base duration=$(format_duration "$duration")"
        record_failure "step5b" "$base" "anchor_query_generation_failed" "$anchor_out"
        SECTION_FAILED=$((SECTION_FAILED + 1))
    fi
done
end_section

###############################################################################
# Stop llama-server before HF local model steps.
###############################################################################

start_section "llm_server_phase_1_stop"
cleanup_server
server_log="$LLAMA_LOG_DIR/server_${PORT}.log"
[[ -s "$server_log" ]] && cp -f "$server_log" "$LLAMA_LOG_DIR/server_${PORT}_steps1_3_5b.log"
SECTION_SUCCESS=1
end_section

###############################################################################
# Step 5C
###############################################################################

start_section "step5c_rag_ingest"
tabular_files=("$TABULAR_DIR"/*.csv.gz)
filter_selected_files_in_place tabular_files
[[ "${#tabular_files[@]}" -gt 0 ]] || die "Step 5C prerequisite failed: no .csv.gz files in $TABULAR_DIR"
log "Step 5C running folder ingest for ${#tabular_files[@]} tabular files. Per-file progress is in $STEP5C_LOG"

EFFECTIVE_TABULAR_DIR="$TABULAR_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/tabular_${SLURM_JOB_ID:-manual}_$$" \
        "$TABULAR_DIR/{}.csv.gz"
    EFFECTIVE_TABULAR_DIR="$STAGED_SELECTED_DIR"
fi

if run_logged "$STEP5C_LOG" python "$RAG_PIPELINE_SCRIPT" \
    --folder "$EFFECTIVE_TABULAR_DIR/" \
    --output-folder "$RAG_OUTPUT_ROOT/" \
    --infix "$RAG_INFIX" \
    --embedding-model "$EMBEDDING_MODEL" \
    --device "$HF_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --max-length "$HF_MAX_LENGTH" \
    ingest; then
    rag_mappings=("$RAG_DIR"/*/summary_mapping.json)
    SECTION_SUCCESS="${#rag_mappings[@]}"
    log "Step 5C complete: RAG mapping files found=${#rag_mappings[@]}"
else
    SECTION_FAILED=1
    record_failure "step5c" "folder" "rag_ingest_command_failed" "$STEP5C_LOG"
    end_section
    die "Step 5C failed globally; see $STEP5C_LOG"
fi
end_section

###############################################################################
# Step 5D
###############################################################################

start_section "step5d_anchor_ranking"
anchor_jsons=("$AQG_DIR"/*.json)
filter_selected_files_in_place anchor_jsons
rag_dirs=("$RAG_DIR"/*)
[[ "${#anchor_jsons[@]}" -gt 0 ]] || die "Step 5D prerequisite failed: no anchor JSON files in $AQG_DIR"
[[ "${#rag_dirs[@]}" -gt 0 ]] || die "Step 5D prerequisite failed: no RAG directories in $RAG_DIR"
log "Step 5D running folder comparison for ${#anchor_jsons[@]} anchor files. Per-file progress is in $STEP5D_LOG"

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

if run_logged "$STEP5D_LOG" python "$ANCHOR_RANKER_SCRIPT" compare \
    --folder "$EFFECTIVE_BUNDLE_DIR" \
    --anchor-subdir "$AQG_INFIX" \
    --rag-subdir "tabular/$RAG_INFIX/" \
    --k "$TOP_K" \
    --embedding-model "$EMBEDDING_MODEL" \
    --reranker-model "$RERANKER_MODEL" \
    --device "$HF_DEVICE" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --reranker-batch-size "$RERANKER_BATCH_SIZE" \
    --max-length "$HF_MAX_LENGTH"; then
    result_jsons=("$COMPARISON_DIR"/*_results.json)
    filter_selected_files_in_place result_jsons
    SECTION_SUCCESS="${#result_jsons[@]}"
    log "Step 5D complete: result JSON files found=${#result_jsons[@]}"
else
    SECTION_FAILED=1
    record_failure "step5d" "folder" "anchor_ranking_command_failed" "$STEP5D_LOG"
    end_section
    die "Step 5D failed globally; see $STEP5D_LOG"
fi
end_section

###############################################################################
# Step 5E
###############################################################################

start_section "step5e_flatten_and_join"
result_jsons=("$COMPARISON_DIR"/*_results.json)
filter_selected_files_in_place result_jsons
[[ "${#result_jsons[@]}" -gt 0 ]] || die "Step 5E prerequisite failed: no result JSON files in $COMPARISON_DIR"

for results_file in "${result_jsons[@]}"; do
    has_nonblank_content "$results_file" || {
        base="$(basename "$results_file" _results.json)"
        record_failure "step5e" "$base" "empty_result_json" "$results_file"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        continue
    }

    base="$(basename "$results_file" _results.json)"
    patient_id="$(subject_id_from_name "$base" || true)"
    log "Step 5E running: $base"
    case_start="$(epoch_now)"

    if [[ -z "$patient_id" ]]; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5E failed: could not resolve subject ID for $base duration=$(format_duration "$duration")"
        record_failure "step5e" "$base" "could_not_resolve_subject_id_from_filename" "$results_file"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        continue
    fi

    summary_csv="$COMPARISON_DIR/${patient_id}_summary.csv"
    joined_csv="$COMPARISON_DIR/${patient_id}_joined.csv"

    if has_nonblank_content "$joined_csv"; then
        log "Step 5E skip existing: $base"
        SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
        continue
    fi

    if ! has_nonblank_content "$summary_csv"; then
        if ! run_logged "$STEP5E_LOG" python "$VIEW_SCORES_SCRIPT" \
            --results-file "$results_file" \
            --output "$summary_csv" \
            --sort-by score \
            --overwrite; then
            duration=$(( $(epoch_now) - case_start ))
            log "Step 5E failed flattening: $base duration=$(format_duration "$duration")"
            record_failure "step5e" "$base" "score_flattening_failed" "$summary_csv"
            SECTION_FAILED=$((SECTION_FAILED + 1))
            continue
        fi
    fi

    if ! has_nonblank_content "$summary_csv"; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5E failed: summary CSV empty for $base duration=$(format_duration "$duration")"
        record_failure "step5e" "$base" "empty_summary_csv" "$summary_csv"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        continue
    fi

    mapping_file="$(find_mapping_file "$patient_id" || true)"
    line_level_file="$(find_line_level_file "$patient_id" || true)"
    if [[ -z "$mapping_file" || -z "$line_level_file" ]]; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5E failed prerequisite for $base duration=$(format_duration "$duration")"
        record_failure "step5e" "$base" "missing_mapping_or_tabular_file" "$mapping_file $line_level_file"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        continue
    fi

    if run_logged "$STEP5E_LOG" python "$JOIN_ANCHOR_SCRIPT" \
        --anchor-results "$summary_csv" \
        --summary-mapping "$mapping_file" \
        --line-level-data "$line_level_file" \
        --output "$joined_csv" \
        --note-adms "$NOTE_LINK_PATH" \
        --overwrite \
        && has_nonblank_content "$joined_csv"; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5E success: $base duration=$(format_duration "$duration")"
        SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
    else
        duration=$(( $(epoch_now) - case_start ))
        log "Step 5E failed joining: $base duration=$(format_duration "$duration")"
        record_failure "step5e" "$base" "join_failed_or_empty_output" "$joined_csv"
        SECTION_FAILED=$((SECTION_FAILED + 1))
    fi
done
end_section

###############################################################################
# LLM phase 2: Step 6.
###############################################################################

start_section "llm_server_phase_2_start"
if start_server "$REASONING_MODEL_PATH" "$REASONING_MODEL_ALIAS" "$PORT" >> "$PIPELINE_LOG" 2>&1; then
    SECTION_SUCCESS=1
else
    SECTION_FAILED=1
    end_section
    die "Could not start llama-server for Step 6"
fi
end_section

###############################################################################
# Step 6
###############################################################################

start_section "step6_final_timelines"
joined_files=("$COMPARISON_DIR"/*_joined.csv)
filter_selected_files_in_place joined_files
[[ "${#joined_files[@]}" -gt 0 ]] || die "Step 6 prerequisite failed: no joined CSV files in $COMPARISON_DIR"

for joined_file in "${joined_files[@]}"; do
    base="$(basename "$joined_file" _joined.csv)"
    case_report="$(find_case_report "$base" || true)"
    mention_table="$(find_mention_table "$base" || true)"
    timeline_json="$COMPARISON_DIR/${base}_timelines.json"
    log "Step 6 running: $base"
    case_start="$(epoch_now)"

    if has_valid_timeline_json "$timeline_json"; then
        log "Step 6 skip existing valid timeline JSON: $base"
        SECTION_SKIPPED=$((SECTION_SKIPPED + 1))
        continue
    fi

    if [[ -z "$case_report" || -z "$mention_table" ]] || ! has_nonblank_content "$joined_file"; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 6 failed prerequisite for $base duration=$(format_duration "$duration")"
        record_failure "step6" "$base" "missing_case_report_mention_table_or_joined_csv" "$case_report $mention_table $joined_file"
        SECTION_FAILED=$((SECTION_FAILED + 1))
        continue
    fi

    if run_logged "$STEP6_LOG" python "$ASSIGN_TIMESTAMPS_SCRIPT" \
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
        && has_valid_timeline_json "$timeline_json"; then
        duration=$(( $(epoch_now) - case_start ))
        log "Step 6 success: $base duration=$(format_duration "$duration")"
        SECTION_SUCCESS=$((SECTION_SUCCESS + 1))
    else
        duration=$(( $(epoch_now) - case_start ))
        log "Step 6 failed: $base duration=$(format_duration "$duration")"
        record_failure "step6" "$base" "timeline_generation_failed_or_invalid_output" "$timeline_json"
        SECTION_FAILED=$((SECTION_FAILED + 1))
    fi
done
end_section

start_section "llm_server_phase_2_stop"
cleanup_server
server_log="$LLAMA_LOG_DIR/server_${PORT}.log"
[[ -s "$server_log" ]] && cp -f "$server_log" "$LLAMA_LOG_DIR/server_${PORT}_step6.log"
SECTION_SUCCESS=1
end_section

###############################################################################
# Post-processing
###############################################################################

start_section "postprocess_extract_timeline_bsv"
timeline_jsons=("$COMPARISON_DIR"/*_timelines.json)
filter_selected_files_in_place timeline_jsons
[[ "${#timeline_jsons[@]}" -gt 0 ]] || die "Post-processing prerequisite failed: no timeline JSON files in $COMPARISON_DIR"
log "Post-processing running extract_timelines.py. Per-file progress is in $POST_LOG"

EFFECTIVE_COMPARISON_DIR="$COMPARISON_DIR"
if selection_enabled; then
    stage_selected_files_for_bases \
        "$SELECTION_WORK_DIR/post_comparison_${SLURM_JOB_ID:-manual}_$$" \
        "$COMPARISON_DIR/{}_timelines.json"
    EFFECTIVE_COMPARISON_DIR="$STAGED_SELECTED_DIR"
fi

if run_logged "$POST_LOG" python "$EXTRACT_TIMELINES_SCRIPT" \
    --directory "$EFFECTIVE_COMPARISON_DIR" \
    --pattern "*_timelines.json" \
    --output-dir "$TIMELINES_DIR/" \
    --verbose; then
    timeline_bsvs=("$TIMELINES_DIR"/*.bsv.gz)
    SECTION_SUCCESS="${#timeline_bsvs[@]}"
else
    SECTION_FAILED=1
    record_failure "postprocess_timelines" "folder" "extract_timelines_command_failed" "$POST_LOG"
fi
end_section

start_section "postprocess_charpos_final"
bundle_pattern="$(select_bundle_pattern || true)"
[[ -n "$bundle_pattern" ]] || die "Charpos prerequisite failed: no bundle report pattern found in $BUNDLE_DIR"
timeline_bsvs=("$TIMELINES_DIR"/*.bsv.gz)
filter_selected_files_in_place timeline_bsvs
[[ "${#timeline_bsvs[@]}" -gt 0 ]] || die "Charpos prerequisite failed: no final timeline BSV files in $TIMELINES_DIR"
log "Post-processing final charpos with bundle pattern $bundle_pattern. Per-file progress is in $POST_LOG"

if selection_enabled && [[ "${EFFECTIVE_BUNDLE_DIR:-$BUNDLE_DIR}" == "$BUNDLE_DIR" ]]; then
    EFFECTIVE_BUNDLE_DIR="$SELECTION_WORK_DIR/post_bundle_${SLURM_JOB_ID:-manual}_$$"
    stage_selected_files_for_bases \
        "$EFFECTIVE_BUNDLE_DIR" \
        "$BUNDLE_DIR/{}.txt" \
        "$BUNDLE_DIR/{}.txt.txt"
fi

if run_logged "$POST_LOG" python "$GET_CHARPOS_SCRIPT" \
    --input-dir "${EFFECTIVE_BUNDLE_DIR:-$BUNDLE_DIR}/" \
    --pattern "$bundle_pattern" \
    --output-dir "$CHARPOS_DIR" \
    --join-mention \
    --timeline-dir "$TIMELINES_DIR"; then
    final_charpos=("$CHARPOS_DIR"/*.bsv)
    SECTION_SUCCESS="${#final_charpos[@]}"
else
    SECTION_FAILED=1
    record_failure "postprocess_charpos_final" "folder" "final_charpos_command_failed" "$POST_LOG"
fi
end_section

start_section "postprocess_charpos_unrevised"
temporal_tables=("$TEMPORAL_ANSWERS_DIR"/*_extracted.bsv.gz)
filter_selected_files_in_place temporal_tables
[[ "${#temporal_tables[@]}" -gt 0 ]] || die "Unrevised charpos prerequisite failed: no temporal answer tables in $TEMPORAL_ANSWERS_DIR"
log "Post-processing unrevised charpos with bundle pattern $bundle_pattern. Per-file progress is in $POST_LOG"
if run_logged "$POST_LOG" python "$GET_CHARPOS_SCRIPT" \
    --input-dir "${EFFECTIVE_BUNDLE_DIR:-$BUNDLE_DIR}/" \
    --pattern "$bundle_pattern" \
    --output-dir "$CHARPOS_UNREVISED_DIR" \
    --join-mention \
    --timeline-dir "$TEMPORAL_ANSWERS_DIR"; then
    rename_unrevised_charpos_outputs
    unrevised_charpos=("$CHARPOS_UNREVISED_DIR"/*.bsv)
    SECTION_SUCCESS="${#unrevised_charpos[@]}"
else
    SECTION_FAILED=1
    record_failure "postprocess_charpos_unrevised" "folder" "unrevised_charpos_command_failed" "$POST_LOG"
fi
end_section

###############################################################################
# Summary
###############################################################################

log "Done."
log "Summary TSV: $SUMMARY_TSV"
log "Failed cases TSV: $FAILED_CASES_TSV"
log "Main pipeline log: $PIPELINE_LOG"
log "Step logs:"
log "  Step 1: $STEP1_LOG"
log "  Step 2: $STEP2_LOG"
log "  Step 3: $STEP3_LOG"
log "  Step 4: $STEP4_LOG"
log "  Step 5A: $STEP5A_LOG"
log "  Step 5B: $STEP5B_LOG"
log "  Step 5C: $STEP5C_LOG"
log "  Step 5D: $STEP5D_LOG"
log "  Step 5E: $STEP5E_LOG"
log "  Step 6: $STEP6_LOG"
log "  Post-processing: $POST_LOG"

trap - EXIT
exit 0
