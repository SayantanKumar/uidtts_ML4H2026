#!/bin/bash
#SBATCH --job-name=uid_batch_pp
#SBATCH --time=7-00:00:00
#SBATCH --mem=192g
#SBATCH --partition=gpu
#SBATCH --qos=gpunlm2025.2
#SBATCH --gres=gpu:a100:2
#SBATCH --output=sbatch_output/slurm_%A_%a.out
#SBATCH --error=sbatch_error/slurm_%A_%a.err
#SBATCH --mail-type=BEGIN,TIME_LIMIT_90,END
#SBATCH --mail-user=sayantan.kumar@nih.gov

# Submit from the folder containing this script:
#   mkdir -p sbatch_output sbatch_error
#   sbatch batch_run_uid_pipeline.sh
#
# This is a batch-level controller around the parallel step scripts. It creates
# per-batch input/output folders, tracks progress, and symlinks final outputs
# into one consolidated folder. The step scripts still manage their own servers.

set -eo pipefail

###############################################################################
# User-editable defaults
###############################################################################

export PROJECT_DIR="${PROJECT_DIR:-/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/uid_anchor_detection}"
export RUN_SCRIPT_DIR="${RUN_SCRIPT_DIR:-$PROJECT_DIR/run_scripts/parallel}"

# Source folders. Note files may be .txt or .txt.gz. Tabular files must be
# per-subject <subject_id>.csv.gz.
export SOURCE_NOTE_DIR="${SOURCE_NOTE_DIR:-/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/i2m4_n30_notes}"
export SOURCE_TABULAR_DIR="${SOURCE_TABULAR_DIR:-/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/i2m4_n30_tabular}"
export NOTE_LINK_PATH="${NOTE_LINK_PATH:-/data/CHARM-MIMIC/kumars33/LLM_pratice/Textual_tabular_alignment-main/atp_tts/data/note_link_i2m4.csv}"

# Batch workspace. Each batch gets:
#   $BATCH_ROOT/batch_input_0001/
#   $BATCH_ROOT/batch_output_0001/
export BATCH_ROOT="${BATCH_ROOT:-$PROJECT_DIR/outputs/qwen35_parallel_batches}"
FINAL_OUTPUT_DIR_EXPLICIT="${FINAL_OUTPUT_DIR+x}"
export FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-$BATCH_ROOT/final_output}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export BATCH_GROUP_SIZE="${BATCH_GROUP_SIZE:-4}"

# Which phases to run: all, 1234, 5, 67, or comma-separated values such as
# 1234,5 or 5,67.
export BATCH_STEPS="${BATCH_STEPS:-all}"

# Optional batch range, 1-indexed. Examples: 1, 1-3, 0001-0003.
export BATCH_RANGE="${BATCH_RANGE:-}"

# Use txt for .txt inputs or txt.gz for .txt.gz inputs.
export BATCH_INPUT_TYPE="${BATCH_INPUT_TYPE:-txt}"

# Cleanup mode after a successful batch: none, intermediate, final.
# none is safest while validating.
export CLEANUP_MODE="${CLEANUP_MODE:-none}"

# Ports passed to the child scripts. These are sequential phases, so reuse would
# also be okay, but separate ports make logs easier to read.
export STEP1234_PORT="${STEP1234_PORT:-8085}"
export STEP5_PORT="${STEP5_PORT:-8086}"
export STEP67_PORT="${STEP67_PORT:-8087}"

# Default Qwen3.5 parallel model settings. Override from the sbatch command or
# by editing here for another model.
export CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
export INSTRUCT_MODEL_PATH="${INSTRUCT_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export INSTRUCT_MODEL_ALIAS="${INSTRUCT_MODEL_ALIAS:-qwen35}"
export REASONING_MODEL_PATH="${REASONING_MODEL_PATH:-/data/CHARM-MIMIC/.cache/huggingface/gguf/Qwen3.5-397B-A17B-UD-IQ2_M-00001-of-00004.gguf}"
export REASONING_MODEL_ALIAS="${REASONING_MODEL_ALIAS:-qwen35}"
export LLAMA_PARALLEL_SLOTS="${LLAMA_PARALLEL_SLOTS:-2}"
export STEP1_WORKERS="${STEP1_WORKERS:-2}"
export STEP3_WORKERS="${STEP3_WORKERS:-2}"
export STEP5B_WORKERS="${STEP5B_WORKERS:-2}"
export STEP6_WORKERS="${STEP6_WORKERS:-2}"

###############################################################################
# Derived paths
###############################################################################

STEP1234_SCRIPT="$RUN_SCRIPT_DIR/step_1234_parallel.sh"
STEP5_SCRIPT="$RUN_SCRIPT_DIR/step5_rag_parallel.sh"
STEP67_SCRIPT="$RUN_SCRIPT_DIR/step67_parallel.sh"

PATIENT_LIST="$BATCH_ROOT/patients_with_both.txt"
PATIENT_NOTE_MAP="$BATCH_ROOT/patient_note_map.tsv"
PROGRESS_FILE="$BATCH_ROOT/batch_progress.tsv"
LOG_DIR="$BATCH_ROOT/logs"

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

usage() {
    cat <<USAGE
Usage:
  sbatch batch_run_uid_pipeline.sh [options]

Options:
  --source-note-dir DIR       Folder with .txt/.txt.gz notes
  --source-tabular-dir DIR    Folder with <subject_id>.csv.gz tabular files
  --note-link FILE            note_link CSV used in Steps 5/67
  --batch-root DIR            Batch workspace/output root
  --final-output-dir DIR      Consolidated symlink output folder
  --batch-size N              Patients per batch (default: $BATCH_SIZE)
  --group-size N              Batches per logical group (default: $BATCH_GROUP_SIZE)
  --batch-range RANGE         1-indexed range, e.g. 1, 1-3, 0001-0003
  --steps STEPS               all, 1234, 5, 67, or comma list like 1234,5
  --input-type TYPE           txt or txt.gz (default: $BATCH_INPUT_TYPE)
  --cleanup-mode MODE         none, intermediate, final
  --refresh-patient-list      Rebuild patients_with_both.txt and note map
  --help                      Show this help

Environment variables with the same names can also be exported before sbatch.
USAGE
}

parse_args() {
    REFRESH_PATIENT_LIST="${REFRESH_PATIENT_LIST:-false}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --source-note-dir)
                SOURCE_NOTE_DIR="$2"; shift 2 ;;
            --source-tabular-dir)
                SOURCE_TABULAR_DIR="$2"; shift 2 ;;
            --note-link)
                NOTE_LINK_PATH="$2"; shift 2 ;;
            --batch-root)
                BATCH_ROOT="$2"; shift 2 ;;
            --final-output-dir)
                FINAL_OUTPUT_DIR="$2"; FINAL_OUTPUT_DIR_EXPLICIT=1; shift 2 ;;
            --batch-size)
                BATCH_SIZE="$2"; shift 2 ;;
            --group-size|--batch-group-size)
                BATCH_GROUP_SIZE="$2"; shift 2 ;;
            --batch-range|--batches)
                BATCH_RANGE="$2"; shift 2 ;;
            --steps)
                BATCH_STEPS="$2"; shift 2 ;;
            --input-type)
                BATCH_INPUT_TYPE="$2"; shift 2 ;;
            --cleanup-mode)
                CLEANUP_MODE="$2"; shift 2 ;;
            --refresh-patient-list)
                REFRESH_PATIENT_LIST=true; shift ;;
            --help|-h)
                usage; exit 0 ;;
            *)
                die "Unknown argument: $1" ;;
        esac
    done

    if [[ -z "$FINAL_OUTPUT_DIR_EXPLICIT" ]]; then
        FINAL_OUTPUT_DIR="$BATCH_ROOT/final_output"
    fi
    PATIENT_LIST="$BATCH_ROOT/patients_with_both.txt"
    PATIENT_NOTE_MAP="$BATCH_ROOT/patient_note_map.tsv"
    PROGRESS_FILE="$BATCH_ROOT/batch_progress.tsv"
    LOG_DIR="$BATCH_ROOT/logs"
}

validate_config() {
    [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || die "BATCH_SIZE must be an integer"
    [[ "$BATCH_GROUP_SIZE" =~ ^[0-9]+$ ]] || die "BATCH_GROUP_SIZE must be an integer"
    (( BATCH_SIZE >= 1 )) || die "BATCH_SIZE must be >= 1"
    (( BATCH_GROUP_SIZE >= 1 )) || die "BATCH_GROUP_SIZE must be >= 1"

    case "$BATCH_INPUT_TYPE" in
        txt|txt.gz) ;;
        *) die "BATCH_INPUT_TYPE must be txt or txt.gz" ;;
    esac

    case "$CLEANUP_MODE" in
        none|intermediate|final) ;;
        *) die "CLEANUP_MODE must be none, intermediate, or final" ;;
    esac

    check_path "$PROJECT_DIR" "project directory"
    check_path "$RUN_SCRIPT_DIR" "parallel run script directory"
    check_path "$STEP1234_SCRIPT" "step_1234_parallel.sh"
    check_path "$STEP5_SCRIPT" "step5_rag_parallel.sh"
    check_path "$STEP67_SCRIPT" "step67_parallel.sh"
    check_path "$SOURCE_NOTE_DIR" "source note directory"
    check_path "$SOURCE_TABULAR_DIR" "source tabular directory"
    check_path "$NOTE_LINK_PATH" "note_link file"
}

resolve_subject_from_note() {
    local note_file="$1"
    local name token

    name="$(basename "$note_file")"
    name="${name%.gz}"
    name="${name%.txt}"

    if [[ -f "$SOURCE_TABULAR_DIR/${name}.csv.gz" ]]; then
        echo "$name"
        return 0
    fi

    while IFS= read -r token; do
        [[ -n "$token" ]] || continue
        if [[ -f "$SOURCE_TABULAR_DIR/${token}.csv.gz" ]]; then
            echo "$token"
            return 0
        fi
    done < <(printf "%s\n" "$name" | grep -Eo '[0-9]{5,}' || true)

    return 1
}

generate_patient_list() {
    log "Generating patient list from notes and tabular files"
    mkdir -p "$BATCH_ROOT"
    : > "$PATIENT_LIST"
    : > "$PATIENT_NOTE_MAP"

    shopt -s nullglob
    local note_file subject_id
    declare -A seen_subjects=()
    local note_count=0
    local matched_note_count=0

    while IFS= read -r note_file; do
        note_count=$((note_count + 1))
        if subject_id="$(resolve_subject_from_note "$note_file")"; then
            printf "%s\t%s\n" "$subject_id" "$note_file" >> "$PATIENT_NOTE_MAP"
            matched_note_count=$((matched_note_count + 1))
            if [[ -z "${seen_subjects[$subject_id]:-}" ]]; then
                printf "%s\n" "$subject_id" >> "$PATIENT_LIST"
                seen_subjects[$subject_id]=1
            fi
        fi
    done < <(find -L "$SOURCE_NOTE_DIR" -maxdepth 1 -type f \( -name "*.txt" -o -name "*.txt.gz" \) | sort)

    sort -u "$PATIENT_LIST" -o "$PATIENT_LIST"
    sort -u "$PATIENT_NOTE_MAP" -o "$PATIENT_NOTE_MAP"

    local patient_count
    patient_count="$(wc -l < "$PATIENT_LIST")"
    log "Found $patient_count subjects with both notes and tabular files"
    log "Matched $matched_note_count/$note_count note files to tabular subject IDs"

    [[ "$patient_count" -gt 0 ]] || die "No subjects with both notes and tabular files found"
}

parse_batch_range() {
    local total_batches="$1"

    RANGE_START=1
    RANGE_END="$total_batches"

    if [[ -z "$BATCH_RANGE" ]]; then
        return 0
    fi

    if [[ "$BATCH_RANGE" =~ ^0*([0-9]+)-0*([0-9]+)$ ]]; then
        RANGE_START="${BASH_REMATCH[1]}"
        RANGE_END="${BASH_REMATCH[2]}"
    elif [[ "$BATCH_RANGE" =~ ^0*([0-9]+)$ ]]; then
        RANGE_START="${BASH_REMATCH[1]}"
        RANGE_END="${BASH_REMATCH[1]}"
    else
        die "Invalid BATCH_RANGE: $BATCH_RANGE"
    fi

    (( RANGE_START >= 1 )) || die "Batch range must start at >= 1"
    (( RANGE_END >= RANGE_START )) || die "Batch range end must be >= start"
    (( RANGE_START <= total_batches )) || die "Batch range starts after total batches"
    if (( RANGE_END > total_batches )); then
        RANGE_END="$total_batches"
    fi
}

step_requested() {
    local step="$1"
    local normalized
    normalized="$(echo "$BATCH_STEPS" | tr '[:upper:]' '[:lower:]' | tr ' ' ',')"

    case "$normalized" in
        all|full|1-7|1-67)
            return 0 ;;
    esac

    IFS=',' read -r -a step_tokens <<< "$normalized"
    local token
    for token in "${step_tokens[@]}"; do
        case "$step:$token" in
            1234:1234|1234:1-4|1234:step1234|1234:step_1234)
                return 0 ;;
            5:5|5:5_rag|5:rag|5:step5)
                return 0 ;;
            67:67|67:6|67:7|67:6-7|67:step67)
                return 0 ;;
        esac
    done

    return 1
}

batch_id_for_num() {
    printf "%04d" "$1"
}

batch_paths() {
    local batch_id="$1"
    BATCH_INPUT="$BATCH_ROOT/batch_input_$batch_id"
    BATCH_OUTPUT="$BATCH_ROOT/batch_output_$batch_id"
    BATCH_PATIENTS="$BATCH_ROOT/batch_patients_$batch_id.txt"
}

create_batch_input() {
    local batch_num="$1"
    local batch_id
    local start_line end_line
    local patient_id note_file
    local note_count=0
    local tabular_count=0

    batch_id="$(batch_id_for_num "$batch_num")"
    batch_paths "$batch_id"
    start_line=$(( (batch_num - 1) * BATCH_SIZE + 1 ))
    end_line=$(( start_line + BATCH_SIZE - 1 ))

    mkdir -p "$BATCH_INPUT/tabular" "$BATCH_OUTPUT" "$BATCH_OUTPUT/logs"
    sed -n "${start_line},${end_line}p" "$PATIENT_LIST" > "$BATCH_PATIENTS"

    while IFS= read -r patient_id; do
        [[ -n "$patient_id" ]] || continue

        if [[ -f "$SOURCE_TABULAR_DIR/${patient_id}.csv.gz" ]]; then
            ln -sfn "$SOURCE_TABULAR_DIR/${patient_id}.csv.gz" "$BATCH_INPUT/tabular/${patient_id}.csv.gz"
            tabular_count=$((tabular_count + 1))
        fi

        while IFS= read -r note_file; do
            [[ -n "$note_file" ]] || continue
            ln -sfn "$note_file" "$BATCH_INPUT/$(basename "$note_file")"
            note_count=$((note_count + 1))
        done < <(awk -F '\t' -v sid="$patient_id" '$1 == sid { print $2 }' "$PATIENT_NOTE_MAP")
    done < "$BATCH_PATIENTS"

    log "Batch $batch_id input ready: $note_count note files, $tabular_count tabular files"
}

count_files() {
    local dir="$1"
    local pattern="$2"
    [[ -d "$dir" ]] || {
        echo 0
        return
    }
    find -L "$dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l
}

has_step1234_outputs() {
    local batch_output="$1"
    [[ "$(count_files "$batch_output/temporal/answers" "*_extracted.bsv.gz")" -gt 0 ]]
}

has_step5_outputs() {
    local batch_output="$1"
    [[ "$(count_files "$batch_output/bundle/uidtts/comparison_results" "*_joined.csv")" -gt 0 ]]
}

has_step67_outputs() {
    local batch_output="$1"
    [[ "$(count_files "$batch_output/timelines" "*.bsv.gz")" -gt 0 ]] && \
        [[ "$(count_files "$batch_output/bundle/charpos" "*.bsv")" -gt 0 ]]
}

batch_done_for_requested_steps() {
    local batch_output="$1"

    if step_requested 67; then
        has_step67_outputs "$batch_output"
        return $?
    fi
    if step_requested 5; then
        has_step5_outputs "$batch_output"
        return $?
    fi
    if step_requested 1234; then
        has_step1234_outputs "$batch_output"
        return $?
    fi

    return 1
}

append_progress() {
    local batch_id="$1"
    local state="$2"
    local message="${3:-}"
    mkdir -p "$BATCH_ROOT"
    printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "$batch_id" "$state" "$message" >> "$PROGRESS_FILE"
}

run_step1234_for_batch() {
    local batch_id="$1"
    batch_paths "$batch_id"

    if has_step1234_outputs "$BATCH_OUTPUT"; then
        log "Batch $batch_id: Step 1-4 outputs already exist; skipping"
        append_progress "$batch_id" "SKIP:1234" "existing outputs"
        return 0
    fi

    log "Batch $batch_id: running Step 1-4"
    append_progress "$batch_id" "START:1234" "$BATCH_OUTPUT"

    if env \
        PROJECT_DIR="$PROJECT_DIR" \
        INPUT_DIR="$BATCH_INPUT" \
        OUTPUT_DIR="$BATCH_OUTPUT" \
        INPUT_TYPE="$BATCH_INPUT_TYPE" \
        FILE_SELECTION_MODE="all" \
        PORT="$STEP1234_PORT" \
        CUDA_DEVICES="$CUDA_DEVICES" \
        INSTRUCT_MODEL_PATH="$INSTRUCT_MODEL_PATH" \
        INSTRUCT_MODEL_ALIAS="$INSTRUCT_MODEL_ALIAS" \
        REASONING_MODEL_PATH="$REASONING_MODEL_PATH" \
        REASONING_MODEL_ALIAS="$REASONING_MODEL_ALIAS" \
        LLAMA_PARALLEL_SLOTS="$LLAMA_PARALLEL_SLOTS" \
        STEP1_WORKERS="$STEP1_WORKERS" \
        STEP3_WORKERS="$STEP3_WORKERS" \
        bash "$STEP1234_SCRIPT"; then
        append_progress "$batch_id" "DONE:1234" ""
        return 0
    fi

    append_progress "$batch_id" "FAILED:1234" "see $BATCH_OUTPUT/logs"
    return 1
}

run_step5_for_batch() {
    local batch_id="$1"
    batch_paths "$batch_id"

    if ! has_step1234_outputs "$BATCH_OUTPUT"; then
        log "Batch $batch_id: missing Step 1-4 outputs; cannot run Step 5"
        append_progress "$batch_id" "FAILED:5" "missing Step 1-4 outputs"
        return 1
    fi

    if has_step5_outputs "$BATCH_OUTPUT"; then
        log "Batch $batch_id: Step 5 joined outputs already exist; skipping"
        append_progress "$batch_id" "SKIP:5" "existing outputs"
        return 0
    fi

    log "Batch $batch_id: running Step 5"
    append_progress "$batch_id" "START:5" "$BATCH_OUTPUT"

    if env \
        PROJECT_DIR="$PROJECT_DIR" \
        OUTPUT_DIR="$BATCH_OUTPUT" \
        TABULAR_DIR="$BATCH_INPUT/tabular" \
        NOTE_LINK_PATH="$NOTE_LINK_PATH" \
        FILE_SELECTION_MODE="all" \
        PORT="$STEP5_PORT" \
        CUDA_DEVICES="$CUDA_DEVICES" \
        INSTRUCT_MODEL_PATH="$INSTRUCT_MODEL_PATH" \
        INSTRUCT_MODEL_ALIAS="$INSTRUCT_MODEL_ALIAS" \
        REASONING_MODEL_PATH="$REASONING_MODEL_PATH" \
        REASONING_MODEL_ALIAS="$REASONING_MODEL_ALIAS" \
        LLAMA_PARALLEL_SLOTS="$LLAMA_PARALLEL_SLOTS" \
        STEP5B_WORKERS="$STEP5B_WORKERS" \
        bash "$STEP5_SCRIPT"; then
        append_progress "$batch_id" "DONE:5" ""
        return 0
    fi

    append_progress "$batch_id" "FAILED:5" "see $BATCH_OUTPUT/logs"
    return 1
}

run_step67_for_batch() {
    local batch_id="$1"
    batch_paths "$batch_id"

    if ! has_step5_outputs "$BATCH_OUTPUT"; then
        log "Batch $batch_id: missing Step 5 joined outputs; cannot run Step 67"
        append_progress "$batch_id" "FAILED:67" "missing Step 5 outputs"
        return 1
    fi

    if has_step67_outputs "$BATCH_OUTPUT"; then
        log "Batch $batch_id: Step 67 outputs already exist; skipping"
        append_progress "$batch_id" "SKIP:67" "existing outputs"
        return 0
    fi

    log "Batch $batch_id: running Step 67"
    append_progress "$batch_id" "START:67" "$BATCH_OUTPUT"

    if env \
        PROJECT_DIR="$PROJECT_DIR" \
        OUTPUT_DIR="$BATCH_OUTPUT" \
        NOTE_LINK_PATH="$NOTE_LINK_PATH" \
        FILE_SELECTION_MODE="all" \
        PORT="$STEP67_PORT" \
        CUDA_DEVICES="$CUDA_DEVICES" \
        INSTRUCT_MODEL_PATH="$INSTRUCT_MODEL_PATH" \
        INSTRUCT_MODEL_ALIAS="$INSTRUCT_MODEL_ALIAS" \
        REASONING_MODEL_PATH="$REASONING_MODEL_PATH" \
        REASONING_MODEL_ALIAS="$REASONING_MODEL_ALIAS" \
        LLAMA_PARALLEL_SLOTS="$LLAMA_PARALLEL_SLOTS" \
        STEP6_WORKERS="$STEP6_WORKERS" \
        bash "$STEP67_SCRIPT"; then
        append_progress "$batch_id" "DONE:67" ""
        return 0
    fi

    append_progress "$batch_id" "FAILED:67" "see $BATCH_OUTPUT/logs"
    return 1
}

symlink_dir_files() {
    local src_dir="$1"
    local dst_dir="$2"
    local pattern="$3"

    [[ -d "$src_dir" ]] || return 0
    mkdir -p "$dst_dir"

    local file
    shopt -s nullglob
    for file in "$src_dir"/$pattern; do
        [[ -f "$file" ]] || continue
        ln -sfn "$file" "$dst_dir/$(basename "$file")"
    done
}

symlink_batch_to_final() {
    local batch_id="$1"
    batch_paths "$batch_id"

    mkdir -p "$FINAL_OUTPUT_DIR"
    symlink_dir_files "$BATCH_OUTPUT/timelines" "$FINAL_OUTPUT_DIR/timelines" "*"
    symlink_dir_files "$BATCH_OUTPUT/tags" "$FINAL_OUTPUT_DIR/tags" "*"
    symlink_dir_files "$BATCH_OUTPUT/temporal/answers" "$FINAL_OUTPUT_DIR/temporal" "*"
    symlink_dir_files "$BATCH_OUTPUT/bundle/charpos" "$FINAL_OUTPUT_DIR/charpos" "*.bsv"
    symlink_dir_files "$BATCH_OUTPUT/bundle/charpos_unrevised" "$FINAL_OUTPUT_DIR/charpos_unrevised" "*.bsv"
    symlink_dir_files "$BATCH_OUTPUT/bundle/uidtts/comparison_results" "$FINAL_OUTPUT_DIR/timeline_json" "*_timelines.json"
}

cleanup_batch_output() {
    local batch_id="$1"
    batch_paths "$batch_id"

    case "$CLEANUP_MODE" in
        none)
            return 0 ;;
        intermediate)
            log "Batch $batch_id: cleanup mode intermediate"
            find "$BATCH_OUTPUT/bundle/uidtts/comparison_results" -type f \
                \( -name "*_results.json" -o -name "*_summary.csv" \) -delete 2>/dev/null || true
            ;;
        final)
            log "Batch $batch_id: cleanup mode final"
            find "$BATCH_OUTPUT/bundle/uidtts/comparison_results" -type f \
                \( -name "*_results.json" -o -name "*_summary.csv" -o -name "*_joined.csv" \) -delete 2>/dev/null || true
            ;;
    esac
}

process_batch_group() {
    local batch_ids=("$@")
    [[ "${#batch_ids[@]}" -gt 0 ]] || return 0

    log "Processing batch group: ${batch_ids[*]}"
    local batch_id
    declare -A failed=()

    if step_requested 1234; then
        log "Group phase: Step 1-4"
        for batch_id in "${batch_ids[@]}"; do
            if ! run_step1234_for_batch "$batch_id"; then
                failed[$batch_id]=1
                log "WARNING: Batch $batch_id failed Step 1-4; continuing with other batches"
            fi
        done
    fi

    if step_requested 5; then
        log "Group phase: Step 5"
        for batch_id in "${batch_ids[@]}"; do
            [[ -n "${failed[$batch_id]:-}" ]] && continue
            if ! run_step5_for_batch "$batch_id"; then
                failed[$batch_id]=1
                log "WARNING: Batch $batch_id failed Step 5; continuing with other batches"
            fi
        done
    fi

    if step_requested 67; then
        log "Group phase: Step 67"
        for batch_id in "${batch_ids[@]}"; do
            [[ -n "${failed[$batch_id]:-}" ]] && continue
            if ! run_step67_for_batch "$batch_id"; then
                failed[$batch_id]=1
                log "WARNING: Batch $batch_id failed Step 67; continuing with other batches"
            fi
        done
    fi

    for batch_id in "${batch_ids[@]}"; do
        [[ -n "${failed[$batch_id]:-}" ]] && continue
        batch_paths "$batch_id"
        if batch_done_for_requested_steps "$BATCH_OUTPUT"; then
            append_progress "$batch_id" "COMPLETE" "$BATCH_OUTPUT"
            symlink_batch_to_final "$batch_id"
            cleanup_batch_output "$batch_id"
        fi
    done
}

###############################################################################
# Main
###############################################################################

main() {
    parse_args "$@"
    validate_config

    mkdir -p "$BATCH_ROOT" "$FINAL_OUTPUT_DIR" "$LOG_DIR"

    log "Batch UID pipeline"
    log "Project directory: $PROJECT_DIR"
    log "Source note directory: $SOURCE_NOTE_DIR"
    log "Source tabular directory: $SOURCE_TABULAR_DIR"
    log "Batch root: $BATCH_ROOT"
    log "Final output directory: $FINAL_OUTPUT_DIR"
    log "Batch size: $BATCH_SIZE"
    log "Batch group size: $BATCH_GROUP_SIZE"
    log "Steps: $BATCH_STEPS"
    log "Input type: $BATCH_INPUT_TYPE"
    log "Cleanup mode: $CLEANUP_MODE"
    log "CUDA devices: $CUDA_DEVICES"
    log "Model: $REASONING_MODEL_ALIAS"

    if [[ "$REFRESH_PATIENT_LIST" == "true" || ! -s "$PATIENT_LIST" || ! -s "$PATIENT_NOTE_MAP" ]]; then
        generate_patient_list
    else
        log "Using existing patient list: $PATIENT_LIST"
    fi

    local total_patients total_batches
    total_patients="$(wc -l < "$PATIENT_LIST")"
    total_batches=$(( (total_patients + BATCH_SIZE - 1) / BATCH_SIZE ))
    parse_batch_range "$total_batches"

    log "Total patients: $total_patients"
    log "Total batches: $total_batches"
    log "Processing batch range: $(batch_id_for_num "$RANGE_START")-$(batch_id_for_num "$RANGE_END")"
    log "Progress file: $PROGRESS_FILE"

    local batch_num batch_id
    local group=()

    for ((batch_num = RANGE_START; batch_num <= RANGE_END; batch_num++)); do
        batch_id="$(batch_id_for_num "$batch_num")"
        batch_paths "$batch_id"

        if batch_done_for_requested_steps "$BATCH_OUTPUT"; then
            log "Batch $batch_id already complete for requested steps; symlinking and skipping"
            append_progress "$batch_id" "SKIP:COMPLETE" "$BATCH_OUTPUT"
            symlink_batch_to_final "$batch_id"
            continue
        fi

        create_batch_input "$batch_num"
        group+=("$batch_id")

        if (( ${#group[@]} >= BATCH_GROUP_SIZE )); then
            process_batch_group "${group[@]}"
            group=()
        fi
    done

    if (( ${#group[@]} > 0 )); then
        process_batch_group "${group[@]}"
    fi

    log "Batch processing complete"
    log "Final outputs: $FINAL_OUTPUT_DIR"
    log "Progress file: $PROGRESS_FILE"
}

main "$@"
